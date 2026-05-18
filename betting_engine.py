"""
MATCH ANALYZER V7.1 — SHOTS & TACKLES ENGINE
Refined formula based on 28-match backtest (85.7% accuracy, 24/28)
- Over 2.5: Combined Goals ≥ 50 AND (Combined Shots ≥ 1.8 OR Combined Tackles < 2.0)
- Under 2.5: Combined Goals < 50
- Tackles < 2.0 signal: 6/6 on Overs (lock signal)
- Dead rubbers: Skip
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
import re
import json

# ============================================================================
# SUPABASE SETUP
# ============================================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Match Analyzer V7.1", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .verdict-skip { text-align: center; padding: 1.5rem; }
    .verdict-skip .big-text { font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .section-label { font-size: 0.9rem; font-weight: 700; color: #10b981; margin-top: 1rem; }
    .metric-card { background: #0f172a; border-radius: 10px; padding: 0.75rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; color: #94a3b8; }
    .accuracy-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .lock-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# V7.1 PARSER - Extracts shots, tackles, total goals from WhoScored data
# ============================================================================
def parse_match_data_v7(raw_text: str) -> dict:
    """
    V7.1 Parser - Extracts shots, tackles, total goals from WhoScored data
    """
    lines = raw_text.strip().split('\n')
    
    data = {
        "home_team": None,
        "away_team": None,
        "league": None,
        "home_goals_total": None,
        "away_goals_total": None,
        "home_shots_pg": None,
        "away_shots_pg": None,
        "home_tackles_pg": None,
        "away_tackles_pg": None,
        "home_aerial_pct": None,
        "away_aerial_pct": None,
        "home_dribbles_pg": None,
        "away_dribbles_pg": None,
        "home_standings_position": None,
        "away_standings_position": None,
        "home_standings_points": None,
        "away_standings_points": None,
        "is_dead_rubber": False,
        "score_matrix": [],
        "home_win": None,
        "draw": None,
        "away_win": None,
        "btts": None,
        "over_25": None,
        "under_25": None,
        "over_35": None,
    }
    
    # ================================================================
    # STEP 1: Extract Team Names
    # ================================================================
    team_names = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Predicted Lineups' and i > 0:
            for j in range(i-1, max(i-3, -1), -1):
                potential_team = lines[j].strip()
                if potential_team and not any(x in potential_team for x in ['Predicted', 'Lineups', 'Form', 'Standings']):
                    if potential_team not in team_names:
                        team_names.append(potential_team)
                        break
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # ================================================================
    # STEP 2: Extract Total Goals from Season Stats
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if data["home_team"] and data["home_team"] in stripped:
            for j in range(i, min(i+10, len(lines))):
                goal_line = lines[j].strip()
                match = re.search(r'\(?\d+\)?(\d+)\s*Goals', goal_line)
                if match:
                    data["home_goals_total"] = int(match.group(1))
                    break
        
        if data["away_team"] and data["away_team"] in stripped:
            for j in range(i, min(i+10, len(lines))):
                goal_line = lines[j].strip()
                match = re.search(r'\(?\d+\)?(\d+)\s*Goals', goal_line)
                if match:
                    data["away_goals_total"] = int(match.group(1))
                    break
    
    # ================================================================
    # STEP 3: Extract Shots, Tackles from Statistical Comparison
    # ================================================================
    in_stat_comp = False
    current_team = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Statistical Comparison' in stripped:
            in_stat_comp = True
            continue
        
        if in_stat_comp:
            if data["home_team"] and data["home_team"] in stripped:
                current_team = "home"
            elif data["away_team"] and data["away_team"] in stripped:
                current_team = "away"
            
            if 'Shots pg' in stripped:
                match = re.search(r'(\d+\.?\d*)\s*Shots pg', stripped)
                if match and current_team:
                    if current_team == "home":
                        data["home_shots_pg"] = float(match.group(1))
                    else:
                        data["away_shots_pg"] = float(match.group(1))
            
            if 'Tackles pg' in stripped:
                match = re.search(r'(\d+\.?\d*)\s*Tackles pg', stripped)
                if match and current_team:
                    if current_team == "home":
                        data["home_tackles_pg"] = float(match.group(1))
                    else:
                        data["away_tackles_pg"] = float(match.group(1))
            
            if 'Aerial Duel Success' in stripped:
                match = re.search(r'(\d+)%', stripped)
                if match and current_team:
                    if current_team == "home":
                        data["home_aerial_pct"] = int(match.group(1))
                    else:
                        data["away_aerial_pct"] = int(match.group(1))
            
            if 'Dribbles pg' in stripped:
                match = re.search(r'(\d+\.?\d*)\s*Dribbles pg', stripped)
                if match and current_team:
                    if current_team == "home":
                        data["home_dribbles_pg"] = float(match.group(1))
                    else:
                        data["away_dribbles_pg"] = float(match.group(1))
            
            if 'Head to Head' in stripped or 'Form Data' in stripped:
                break
    
    # ================================================================
    # STEP 4: Extract Standings
    # ================================================================
    in_standings = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Standings' in stripped:
            in_standings = True
            continue
        
        if in_standings and data["home_team"] and data["home_team"] in stripped:
            parts = stripped.split()
            for idx, part in enumerate(parts):
                if data["home_team"] in part and idx > 0:
                    try:
                        data["home_standings_position"] = int(parts[idx-1])
                    except:
                        pass
                    try:
                        data["home_standings_points"] = int(parts[-1])
                    except:
                        pass
                    break
        
        if in_standings and data["away_team"] and data["away_team"] in stripped:
            parts = stripped.split()
            for idx, part in enumerate(parts):
                if data["away_team"] in part and idx > 0:
                    try:
                        data["away_standings_position"] = int(parts[idx-1])
                    except:
                        pass
                    try:
                        data["away_standings_points"] = int(parts[-1])
                    except:
                        pass
                    break
        
        if in_standings and 'Offers' in stripped:
            break
    
    # ================================================================
    # STEP 5: Detect Dead Rubber
    # ================================================================
    dead_rubber_indicators = [
        "nothing at stake",
        "means very little",
        "already secured",
        "guaranteed to finish",
        "cannot move",
        "mathematically safe",
        "nothing to play for",
        "already won the title",
        "already relegated",
    ]
    
    for line in lines:
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in dead_rubber_indicators):
            data["is_dead_rubber"] = True
            break
    
    # ================================================================
    # STEP 6: Extract Score Matrix
    # ================================================================
    in_score_analysis = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped == 'Score analysis':
            in_score_analysis = True
            continue
        if in_score_analysis and stripped == 'Head to Head':
            break
        if in_score_analysis and 'Form Data' in stripped:
            break
        
        if in_score_analysis:
            m = re.match(r'(\d+)-(\d+)\s*@\s*(\d+\.?\d*)\s*%', stripped)
            if m:
                home = int(m.group(1))
                away = int(m.group(2))
                prob = float(m.group(3))
                data["score_matrix"].append({
                    "score": f"{home}-{away}",
                    "home_goals": home,
                    "away_goals": away,
                    "probability": prob
                })
    
    data["score_matrix"].sort(key=lambda x: x["probability"], reverse=True)
    data["score_matrix"] = data["score_matrix"][:10]
    
    # ================================================================
    # STEP 7: Extract Probabilities
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Result' in stripped:
            for j in range(i+1, min(i+15, len(lines))):
                prob_line = lines[j].strip()
                if data["home_team"] and data["home_team"] in prob_line:
                    m = re.search(r'(\d+\.?\d*)%', prob_line)
                    if m:
                        data["home_win"] = float(m.group(1))
                if 'Draw' in prob_line and not data["home_team"] in prob_line:
                    m = re.search(r'(\d+\.?\d*)%', prob_line)
                    if m:
                        data["draw"] = float(m.group(1))
                if data["away_team"] and data["away_team"] in prob_line:
                    m = re.search(r'(\d+\.?\d*)%', prob_line)
                    if m:
                        data["away_win"] = float(m.group(1))
                if 'Both Teams to Score' in prob_line:
                    m = re.search(r'(\d+\.?\d*)%', prob_line)
                    if m:
                        data["btts"] = float(m.group(1))
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Goals' in stripped:
            for j in range(i+1, min(i+20, len(lines))):
                goal_line = lines[j].strip()
                if 'Over 2.5' in goal_line:
                    m = re.search(r'(\d+\.?\d*)%', goal_line)
                    if m:
                        data["over_25"] = float(m.group(1))
                if 'Under 2.5' in goal_line:
                    m = re.search(r'(\d+\.?\d*)%', goal_line)
                    if m:
                        data["under_25"] = float(m.group(1))
                if 'Over 3.5' in goal_line:
                    m = re.search(r'(\d+\.?\d*)%', goal_line)
                    if m:
                        data["over_35"] = float(m.group(1))
    
    return data


# ============================================================================
# V7.1 ANALYSIS ENGINE - Shots & Tackles Formula
# ============================================================================
def analyze_match_v7(data: dict) -> dict:
    """
    V7.1 - Refined formula from 28-match backtest
    Over 2.5: Combined Goals >= 50 AND (Combined Shots >= 1.8 OR Combined Tackles < 2.0)
    Under 2.5: Combined Goals < 50
    Tackles < 2.0 on Over = LOCK signal (6/6 in backtest)
    Dead rubbers = SKIP
    """
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reasons": [],
        "is_lock": False,
    }
    
    # ================================================================
    # STEP 1: Check for dead rubber
    # ================================================================
    if data.get("is_dead_rubber"):
        result["skip_reasons"].append("Dead rubber - nothing at stake for either team")
        result["classification"] = "DEAD RUBBER"
        return result
    
    # ================================================================
    # STEP 2: Get required metrics
    # ================================================================
    home_goals = data.get("home_goals_total", 0) or 0
    away_goals = data.get("away_goals_total", 0) or 0
    home_shots = data.get("home_shots_pg", 0) or 0
    away_shots = data.get("away_shots_pg", 0) or 0
    home_tackles = data.get("home_tackles_pg", 0) or 0
    away_tackles = data.get("away_tackles_pg", 0) or 0
    
    combined_goals = home_goals + away_goals
    combined_shots = home_shots + away_shots
    combined_tackles = home_tackles + away_tackles
    
    result["metrics"] = {
        "home_goals": home_goals,
        "away_goals": away_goals,
        "combined_goals": combined_goals,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "combined_shots": combined_shots,
        "home_tackles": home_tackles,
        "away_tackles": away_tackles,
        "combined_tackles": combined_tackles,
    }
    
    # ================================================================
    # STEP 3: Check minimum required data
    # ================================================================
    missing = []
    if combined_goals == 0:
        missing.append("total goals")
    if combined_shots == 0:
        missing.append("shots per game")
    if combined_tackles == 0:
        missing.append("tackles per game")
    
    if missing:
        result["skip_reasons"].append(f"Missing data: {', '.join(missing)}")
        result["classification"] = "INSUFFICIENT DATA"
        return result
    
    # ================================================================
    # STEP 4: Apply the refined formula
    # ================================================================
    
    # Build signal breakdown for display
    signal_parts = []
    
    goals_signal = combined_goals >= 50
    shots_signal = combined_shots >= 1.8
    tackles_signal = combined_tackles < 2.0
    
    if goals_signal:
        signal_parts.append(f"Goals {combined_goals} ≥ 50 ✓")
    else:
        signal_parts.append(f"Goals {combined_goals} < 50")
    
    if shots_signal:
        signal_parts.append(f"Shots {combined_shots:.1f} ≥ 1.8 ✓")
    else:
        signal_parts.append(f"Shots {combined_shots:.1f} < 1.8")
    
    if tackles_signal:
        signal_parts.append(f"Tackles {combined_tackles:.1f} < 2.0 ✓")
    else:
        signal_parts.append(f"Tackles {combined_tackles:.1f} ≥ 2.0")
    
    result["signal_breakdown"] = signal_parts
    
    # OVER 2.5 RULE
    if goals_signal and (shots_signal or tackles_signal):
        # Determine if this is a lock signal
        is_lock = tackles_signal  # Tackles < 2.0 on Over = 6/6 lock
        
        reason_parts = []
        reason_parts.append(f"Combined goals {combined_goals} ≥ 50")
        
        if shots_signal and tackles_signal:
            reason_parts.append(f"shots {combined_shots:.1f} ≥ 1.8 AND tackles {combined_tackles:.1f} < 2.0 (lock signal)")
        elif shots_signal:
            reason_parts.append(f"shots {combined_shots:.1f} ≥ 1.8")
        elif tackles_signal:
            reason_parts.append(f"tackles {combined_tackles:.1f} < 2.0 (lock signal)")
        
        result["primary_bet"] = {
            "market": "Over 2.5 Goals",
            "reason": "; ".join(reason_parts),
            "historical_accuracy": "6/6 when tackles < 2.0" if is_lock else "85.7% overall (24/28 in backtest)",
        }
        result["classification"] = "OVER 2.5"
        result["verdict"] = "LOCK" if is_lock else "RECOMMENDED"
        result["is_lock"] = is_lock
        return result
    
    # UNDER 2.5 RULE
    if not goals_signal:
        result["primary_bet"] = {
            "market": "Under 2.5 Goals",
            "reason": f"Combined goals {combined_goals} < 50",
            "historical_accuracy": "85.7% overall (24/28 in backtest)",
        }
        result["classification"] = "UNDER 2.5"
        result["verdict"] = "RECOMMENDED"
        return result
    
    # NO CLEAR SIGNAL
    result["skip_reasons"].append(
        f"No clear signal: goals={combined_goals} (≥50), but shots={combined_shots:.1f} (<1.8) and tackles={combined_tackles:.1f} (≥2.0)"
    )
    result["classification"] = "SKIP"
    
    return result


# ============================================================================
# TRUTH-BASED EVALUATION ENGINE
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID DATA", "message": "Non-numeric score"}
    
    total = home + away
    btts = home > 0 and away > 0
    over25 = total > 2
    
    if home > away:
        winner = "HOME"
    elif away > home:
        winner = "AWAY"
    else:
        winner = "DRAW"
    
    pred = primary_pred.strip()
    is_correct = False
    
    if pred == "BTTS":
        is_correct = btts
    elif pred == "Over 2.5 Goals":
        is_correct = over25
    elif pred == "Under 2.5 Goals":
        is_correct = not over25
    elif pred == "Over 3.5 Goals":
        is_correct = total > 3
    elif pred == "Under 3.5 Goals":
        is_correct = total <= 3
    elif pred == "Home Win":
        is_correct = winner == "HOME"
    elif pred == "Away Win":
        is_correct = winner == "AWAY"
    elif pred == "Home Win to Nil":
        is_correct = (winner == "HOME" and away == 0)
    elif pred == "Away Win to Nil":
        is_correct = (winner == "AWAY" and home == 0)
    elif "Away Win or Draw" in pred:
        is_correct = winner in ["AWAY", "DRAW"]
    elif "Home Win or Draw" in pred:
        is_correct = winner in ["HOME", "DRAW"]
    elif pred == "Home Over 1.5 Goals":
        is_correct = home >= 2
    elif pred == "Away Over 1.5 Goals":
        is_correct = away >= 2
    else:
        return {"is_correct": False, "actual": f"{home}-{away}", "message": f"Unknown market: {pred}"}
    
    return {
        "is_correct": is_correct,
        "actual": f"{home}-{away}",
        "message": f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}: {pred} vs {home}-{away}"
    }


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    try:
        primary = analysis.get("primary_bet")
        metrics = analysis.get("metrics", {})
        
        record = {
            "home_team": data.get("home_team", "Unknown"),
            "away_team": data.get("away_team", "Unknown"),
            "match_date": str(date.today()),
            "league": data.get("league"),
            
            # Goals
            "home_goals_total": data.get("home_goals_total"),
            "away_goals_total": data.get("away_goals_total"),
            "combined_goals": metrics.get("combined_goals", 0),
            
            # Shots
            "home_shots_pg": data.get("home_shots_pg"),
            "away_shots_pg": data.get("away_shots_pg"),
            "combined_shots": metrics.get("combined_shots", 0),
            
            # Tackles
            "home_tackles_pg": data.get("home_tackles_pg"),
            "away_tackles_pg": data.get("away_tackles_pg"),
            "combined_tackles": metrics.get("combined_tackles", 0),
            
            # Flags
            "is_dead_rubber": data.get("is_dead_rubber", False),
            "is_lock": analysis.get("is_lock", False),
            
            # Prediction enums
            "prediction": primary["market"] if primary else "SKIP",
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "LOCK" if analysis.get("is_lock") else ("PRIMARY" if primary else "SKIP"),
            "verdict": analysis.get("verdict", "SKIP"),
            
            # Debug (text)
            "signal_breakdown": json.dumps(analysis.get("signal_breakdown", [])),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            
            # Probabilities
            "home_win_pct": data.get("home_win"),
            "draw_pct": data.get("draw"),
            "away_win_pct": data.get("away_win"),
            "btts_pct": data.get("btts"),
            "over25_pct": data.get("over_25"),
            "under25_pct": data.get("under_25"),
            "over35_pct": data.get("over_35"),
            
            # Tracking
            "result_entered": False,
        }
        
        response = supabase.table("match_analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def get_pending():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        over25 = total > 2
        actual_winner = "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW"
        btts_yes = home_goals > 0 and away_goals > 0
        
        supabase.table("match_analyses").update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total,
            "actual_over25": over25,
            "actual_winner": actual_winner,
            "actual_btts": btts_yes,
            "result_entered": True,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False


def get_results():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except:
        return []


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V7.1")
    st.caption("Shots & Tackles Engine | Backtested: 24/28 (85.7%) | Tackles < 2.0 = 6/6 Lock")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ========================================================================
    # TAB 1: ANALYZE
    # ========================================================================
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        st.markdown("*Paste the full match data block from WhoScored*")
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE V7.1", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                with st.spinner("Running shots & tackles analysis..."):
                    data = parse_match_data_v7(raw_text)
                
                if not data.get("home_team") or not data.get("away_team"):
                    st.error("Could not detect team names. Check the format.")
                else:
                    analysis = analyze_match_v7(data)
                    save_to_db(data, analysis)
                    
                    league_display = data.get('league') or 'Club Match'
                    
                    # Verdict header
                    if analysis["verdict"] == "LOCK":
                        st.success(f"🔒 LOCK: {data['home_team']} vs {data['away_team']} — {league_display}")
                    elif analysis["verdict"] == "RECOMMENDED":
                        st.success(f"✅ RECOMMENDED: {data['home_team']} vs {data['away_team']} — {league_display}")
                    else:
                        st.warning(f"⚠️ SKIP: {data['home_team']} vs {data['away_team']} — {league_display}")
                    
                    if analysis.get("classification"):
                        st.markdown(f"**Classification: {analysis['classification']}**")
                    
                    # Signal breakdown
                    if analysis.get("signal_breakdown"):
                        st.markdown("### 🔍 Signal Breakdown")
                        for signal in analysis["signal_breakdown"]:
                            st.markdown(f"- {signal}")
                    
                    # Key Metrics
                    if analysis.get("metrics"):
                        metrics = analysis["metrics"]
                        st.markdown("### 📊 Key Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{metrics['combined_goals']}</div>
                                <div class="metric-label">Combined Goals (Season)</div>
                                <div style="font-size:0.6rem; color:#64748b;">{metrics['home_goals']} + {metrics['away_goals']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{metrics['combined_shots']:.1f}</div>
                                <div class="metric-label">Combined Shots pg</div>
                                <div style="font-size:0.6rem; color:#64748b;">{metrics['home_shots']:.1f} + {metrics['away_shots']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{metrics['combined_tackles']:.1f}</div>
                                <div class="metric-label">Combined Tackles pg</div>
                                <div style="font-size:0.6rem; color:#64748b;">{metrics['home_tackles']:.1f} + {metrics['away_tackles']:.1f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Score Matrix (top 5)
                    if data.get("score_matrix"):
                        st.markdown("### 🎯 Score Matrix (Top 5)")
                        score_cols = st.columns(5)
                        for idx, s in enumerate(data["score_matrix"][:5]):
                            with score_cols[idx]:
                                bg = "#1e293b" if s["home_goals"] != s["away_goals"] else "#2a1a00"
                                st.markdown(f"""
                                <div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;">
                                    <div style="font-size:1.2rem; font-weight:800;">{s['score']}</div>
                                    <div style="font-size:0.7rem; color:#94a3b8;">{s['probability']:.1f}%</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # PRIMARY BET
                    if analysis.get("primary_bet"):
                        primary = analysis["primary_bet"]
                        is_lock = analysis.get("is_lock", False)
                        
                        if is_lock:
                            emoji = "🔒"
                            card_class = "lock-card"
                            badge_html = '<span class="lock-badge">🔒 LOCK SIGNAL — Tackles < 2.0 (6/6)</span>'
                        else:
                            emoji = "🔥"
                            card_class = "primary-card"
                            badge_html = '<span class="accuracy-badge">📊 Historical: 85.7% (24/28)</span>'
                        
                        st.markdown('<div class="section-label">🎯 PRIMARY BET</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="output-card {card_class}">
                            <div style="display:flex;align-items:center;gap:1rem;">
                                <div style="font-size:2.5rem;">{emoji}</div>
                                <div style="flex:1;">
                                    <div style="font-size:1.3rem;font-weight:800;">{primary['market']}</div>
                                    <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">{primary['reason']}</div>
                                    <div style="margin-top:0.5rem;">{badge_html}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SKIP
                    if analysis["verdict"] == "SKIP":
                        skip_text = "<br>".join(analysis.get("skip_reasons", ["No clear signal"]))
                        st.markdown(f"""
                        <div class="output-card skip-card">
                            <div class="verdict-skip">
                                <div class="big-text">⚠️ SKIP — NO BET</div>
                                <p style="color:#94a3b8;margin-top:0.5rem;">{skip_text}</p>
                                <p style="color:#64748b;font-size:0.8rem;">The formula requires a clear signal to fire. Forcing a bet without one reduces accuracy.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Dead rubber warning
                    if data.get("is_dead_rubber"):
                        st.warning("⚠️ Dead rubber detected — both teams have nothing meaningful at stake. The formula cannot predict motivation-less matches reliably.")
    
    # ========================================================================
    # TAB 2: POST-MATCH
    # ========================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for analysis in pending:
                ht = analysis.get('home_team', 'Home')
                at = analysis.get('away_team', 'Away')
                pred = analysis.get('prediction', 'No prediction')
                pattern = analysis.get('pattern', '')
                
                pattern_badge = "🔒 LOCK" if pattern == "LOCK" else ("📊 PRIMARY" if pattern == "PRIMARY" else "⚠️ SKIP")
                
                with st.expander(f"{pattern_badge} | {ht} vs {at} — Predicted: {pred}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2:
                        ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    
                    if st.button("✅ Submit Result", key=f"sub_{analysis['id']}"):
                        if submit_result(analysis['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses. Go to the Analyze tab to process matches.")
    
    # ========================================================================
    # TAB 3: RECORDS
    # ========================================================================
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        if not results:
            st.info("No results recorded yet. Submit results in the Post-Match tab.")
        else:
            total = len(results)
            skip_count = sum(1 for r in results if r.get('prediction') == 'SKIP')
            bet_count = total - skip_count
            
            correct = 0
            incorrect = 0
            lock_correct = 0
            lock_total = 0
            
            for r in results:
                pred = r.get('prediction', '')
                pattern = r.get('pattern', '')
                
                if pred == 'SKIP':
                    continue
                
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
                
                if evaluation["is_correct"]:
                    correct += 1
                    if pattern == "LOCK":
                        lock_correct += 1
                else:
                    incorrect += 1
                
                if pattern == "LOCK":
                    lock_total += 1
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Tracked</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{bet_count}</div><div class="stat-label">Bets Placed</div></div>', unsafe_allow_html=True)
            with col3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{skip_count}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
            
            # Lock signal stats
            if lock_total > 0:
                lock_rate = round(lock_correct / lock_total * 100) if lock_total > 0 else 0
                st.markdown(f"🔒 **Lock Signals (Tackles < 2.0):** {lock_correct}/{lock_total} correct ({lock_rate}%)")
            
            st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")
            
            # Results table
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                classification = r.get('classification', 'Unclassified')
                pattern = r.get('pattern', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                if pred == 'SKIP':
                    badge = "SKIP"
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = "✅ WIN" if evaluation["is_correct"] else "❌ LOSS"
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
                
                if pattern == "LOCK":
                    match_display = f"🔒 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                else:
                    match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
                
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": match_display,
                    "Class": classification,
                    "Primary Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
