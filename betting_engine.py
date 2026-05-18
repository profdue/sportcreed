"""
MATCH ANALYZER V7.1 — SHOTS & TACKLES ENGINE
Refined formula based on 28-match backtest (85.7% accuracy, 24/28)
- Over 2.5: Combined Goals ≥ 50 AND (Combined Shots ≥ 1.8 OR Combined Tackles < 2.0)
- Under 2.5: Combined Goals < 50
- Tackles < 2.0 signal: 6/6 on Overs (lock signal)
- Dead rubbers: Only when BOTH teams have nothing at stake
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
    .dead-rubber-warning { background: #7c2d12; color: #fed7aa; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.8rem; margin: 0.5rem 0; }
    .win-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .loss-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# V7.1 PARSER
# ============================================================================
def parse_match_data_v7(raw_text: str) -> dict:
    lines = raw_text.strip().split('\n')
    
    data = {
        "home_team": None, "away_team": None, "league": None,
        "home_goals_total": None, "away_goals_total": None,
        "home_shots_pg": None, "away_shots_pg": None,
        "home_tackles_pg": None, "away_tackles_pg": None,
        "home_aerial_pct": None, "away_aerial_pct": None,
        "home_dribbles_pg": None, "away_dribbles_pg": None,
        "home_standings_position": None, "away_standings_position": None,
        "home_standings_points": None, "away_standings_points": None,
        "is_dead_rubber": False, "dead_rubber_reason": None,
        "score_matrix": [],
        "home_win": None, "draw": None, "away_win": None,
        "btts": None, "over_25": None, "under_25": None, "over_35": None,
    }
    
    # ================================================================
    # STEP 1: Extract Team Names
    # ================================================================
    team_names = []
    
    # Method 1: Find "Predicted Lineups" and check the line right after it
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Predicted Lineups':
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                parts = re.split(r'\d-\d-\d-\d|\d-\d-\d', next_line)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1 and not any(x in part.lower() for x in ['formation', 'predicted', 'lineups']):
                        if part not in team_names:
                            team_names.append(part)
                if len(team_names) >= 2:
                    break
            for j in range(i-1, max(i-3, -1), -1):
                potential = lines[j].strip()
                if potential and len(potential) > 1 and potential != 'Predicted Lineups':
                    if potential not in team_names:
                        team_names.append(potential)
                    if len(team_names) >= 2:
                        break
            break
    
    # Method 2: Look for team names in Statistical Comparison logos
    if len(team_names) < 2:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if 'logo' in stripped.lower():
                name = stripped.replace('logo', '').strip()
                if name and len(name) > 1 and name not in team_names:
                    team_names.append(name)
    
    # Method 3: First meaningful lines
    if len(team_names) < 2:
        for line in lines[:5]:
            stripped = line.strip()
            if stripped and len(stripped) > 2 and stripped != 'Predicted Lineups':
                if not any(x in stripped.lower() for x in ['predicted', 'lineups', 'formation']):
                    if stripped not in team_names:
                        team_names.append(stripped)
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # ================================================================
    # STEP 2: Extract stats from Statistical Comparison
    # ================================================================
    in_stat_comp = False
    current_team = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Statistical Comparison' in stripped:
            in_stat_comp = True
            continue
        
        if in_stat_comp:
            if data["home_team"] and data["home_team"] in stripped and 'logo' not in stripped.lower():
                current_team = "home"
            elif data["away_team"] and data["away_team"] in stripped and 'logo' not in stripped.lower():
                current_team = "away"
            
            if current_team and ('Head to Head' in stripped or 'Form Data' in stripped):
                break
            
            if not current_team:
                continue
            
            # Goals
            if 'Goals' in stripped:
                match_before = re.search(r'(\d+)\s*Goals', stripped)
                match_after = re.search(r'Goals\s+(\d+)', stripped)
                if match_before:
                    val = int(match_before.group(1))
                    if current_team == "home":
                        if data["home_goals_total"] is None:
                            data["home_goals_total"] = val
                        if match_after and data["away_goals_total"] is None:
                            data["away_goals_total"] = int(match_after.group(1))
                    else:
                        if data["away_goals_total"] is None:
                            data["away_goals_total"] = val
                        if match_after and data["home_goals_total"] is None:
                            data["home_goals_total"] = int(match_after.group(1))
            
            # Shots pg
            if 'Shots pg' in stripped:
                match_before = re.search(r'(\d+\.?\d*)\s*Shots pg', stripped)
                match_after = re.search(r'Shots pg\s+(\d+\.?\d*)', stripped)
                if match_before:
                    val = float(match_before.group(1))
                    if current_team == "home":
                        if data["home_shots_pg"] is None:
                            data["home_shots_pg"] = val
                        if match_after and data["away_shots_pg"] is None:
                            data["away_shots_pg"] = float(match_after.group(1))
                    else:
                        if data["away_shots_pg"] is None:
                            data["away_shots_pg"] = val
                        if match_after and data["home_shots_pg"] is None:
                            data["home_shots_pg"] = float(match_after.group(1))
            
            # Tackles pg
            if 'Tackles pg' in stripped:
                match_before = re.search(r'(\d+\.?\d*)\s*Tackles pg', stripped)
                match_after = re.search(r'Tackles pg\s+(\d+\.?\d*)', stripped)
                if match_before:
                    val = float(match_before.group(1))
                    if current_team == "home":
                        if data["home_tackles_pg"] is None:
                            data["home_tackles_pg"] = val
                        if match_after and data["away_tackles_pg"] is None:
                            data["away_tackles_pg"] = float(match_after.group(1))
                    else:
                        if data["away_tackles_pg"] is None:
                            data["away_tackles_pg"] = val
                        if match_after and data["home_tackles_pg"] is None:
                            data["home_tackles_pg"] = float(match_after.group(1))
            
            # Aerial Duel Success
            if 'Aerial Duel Success' in stripped:
                matches = re.findall(r'(\d+)%', stripped)
                if len(matches) >= 1:
                    val = int(matches[0])
                    if current_team == "home" and data["home_aerial_pct"] is None:
                        data["home_aerial_pct"] = val
                    elif current_team == "away" and data["away_aerial_pct"] is None:
                        data["away_aerial_pct"] = val
            
            # Dribbles pg
            if 'Dribbles pg' in stripped:
                match_before = re.search(r'(\d+\.?\d*)\s*Dribbles pg', stripped)
                if match_before:
                    val = float(match_before.group(1))
                    if current_team == "home" and data["home_dribbles_pg"] is None:
                        data["home_dribbles_pg"] = val
                    elif current_team == "away" and data["away_dribbles_pg"] is None:
                        data["away_dribbles_pg"] = val
    
    # ================================================================
    # STEP 3: Extract Standings
    # ================================================================
    in_standings = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Standings' in stripped:
            in_standings = True
            continue
        if in_standings and 'Offers' in stripped:
            break
        if in_standings and data["home_team"] and data["home_team"] in stripped:
            parts = stripped.split()
            try:
                data["home_standings_position"] = int(parts[0])
                data["home_standings_points"] = int(parts[-1])
            except: pass
        if in_standings and data["away_team"] and data["away_team"] in stripped:
            parts = stripped.split()
            try:
                data["away_standings_position"] = int(parts[0])
                data["away_standings_points"] = int(parts[-1])
            except: pass
    
    # ================================================================
    # STEP 4: Dead Rubber Detection
    # ================================================================
    dead_rubber_indicators = [
        "nothing at stake", "nothing to play for", "cannot move",
        "means very little", "already secured", "guaranteed to finish",
        "mathematically safe", "already won the title", "already relegated",
    ]
    home_dead = False
    away_dead = False
    for line in lines:
        line_lower = line.lower()
        talks_home = data["home_team"] and data["home_team"].lower() in line_lower
        talks_away = data["away_team"] and data["away_team"].lower() in line_lower
        for indicator in dead_rubber_indicators:
            if indicator in line_lower:
                has_except = "except" in line_lower
                if talks_home and not talks_away and not has_except:
                    home_dead = True
                if talks_away and not talks_home and not has_except:
                    away_dead = True
    if home_dead and away_dead:
        data["is_dead_rubber"] = True
        data["dead_rubber_reason"] = "Both teams have nothing at stake"
    elif home_dead:
        data["dead_rubber_reason"] = f"Only {data['home_team']} has nothing at stake - match still live"
    elif away_dead:
        data["dead_rubber_reason"] = f"Only {data['away_team']} has nothing at stake - match still live"
    
    # ================================================================
    # STEP 5: Score Matrix
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Score analysis':
            for j in range(i+1, min(i+15, len(lines))):
                m = re.match(r'(\d+)-(\d+)\s*@\s*(\d+\.?\d*)\s*%', lines[j].strip())
                if m:
                    data["score_matrix"].append({
                        "score": f"{m.group(1)}-{m.group(2)}",
                        "home_goals": int(m.group(1)),
                        "away_goals": int(m.group(2)),
                        "probability": float(m.group(3))
                    })
            break
    data["score_matrix"].sort(key=lambda x: x["probability"], reverse=True)
    data["score_matrix"] = data["score_matrix"][:10]
    
    # ================================================================
    # STEP 6: Probabilities
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Result' in stripped:
            for j in range(i+1, min(i+15, len(lines))):
                pl = lines[j].strip()
                if data["home_team"] and data["home_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["home_win"] = float(m.group(1))
                if 'Draw' in pl and not data["home_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["draw"] = float(m.group(1))
                if data["away_team"] and data["away_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["away_win"] = float(m.group(1))
                if 'Both Teams to Score' in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["btts"] = float(m.group(1))
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Goals' in stripped and 'Shots' not in stripped and 'Tackles' not in stripped:
            for j in range(i+1, min(i+20, len(lines))):
                gl = lines[j].strip()
                if 'Over 2.5' in gl:
                    m = re.search(r'(\d+\.?\d*)%', gl)
                    if m: data["over_25"] = float(m.group(1))
                if 'Under 2.5' in gl:
                    m = re.search(r'(\d+\.?\d*)%', gl)
                    if m: data["under_25"] = float(m.group(1))
                if 'Over 3.5' in gl:
                    m = re.search(r'(\d+\.?\d*)%', gl)
                    if m: data["over_35"] = float(m.group(1))
    
    return data


# ============================================================================
# V7.1 ANALYSIS ENGINE
# ============================================================================
def analyze_match_v7(data: dict) -> dict:
    result = {
        "primary_bet": None, "classification": None, "verdict": "SKIP",
        "skip_reasons": [], "is_lock": False,
        "is_dead_rubber": data.get("is_dead_rubber", False),
        "dead_rubber_reason": data.get("dead_rubber_reason"),
    }
    
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
        "home_goals": home_goals, "away_goals": away_goals, "combined_goals": combined_goals,
        "home_shots": home_shots, "away_shots": away_shots, "combined_shots": combined_shots,
        "home_tackles": home_tackles, "away_tackles": away_tackles, "combined_tackles": combined_tackles,
    }
    
    if combined_goals == 0:
        result["skip_reasons"].append("Missing total goals data")
        result["classification"] = "INSUFFICIENT DATA"
        return result
    
    goals_signal = combined_goals >= 50
    shots_signal = combined_shots >= 1.8
    tackles_signal = combined_tackles < 2.0
    
    signal_parts = [
        f"Goals {combined_goals} {'≥ 50 ✓' if goals_signal else '< 50'}",
        f"Shots {combined_shots:.1f} {'≥ 1.8 ✓' if shots_signal else '< 1.8'}",
        f"Tackles {combined_tackles:.1f} {'< 2.0 ✓' if tackles_signal else '≥ 2.0'}",
    ]
    result["signal_breakdown"] = signal_parts
    
    # OVER 2.5
    if goals_signal and (shots_signal or tackles_signal):
        is_lock = tackles_signal
        reason = f"Combined goals {combined_goals} ≥ 50"
        if shots_signal and tackles_signal:
            reason += f"; shots {combined_shots:.1f} ≥ 1.8 AND tackles {combined_tackles:.1f} < 2.0 (lock)"
        elif shots_signal:
            reason += f"; shots {combined_shots:.1f} ≥ 1.8"
        elif tackles_signal:
            reason += f"; tackles {combined_tackles:.1f} < 2.0 (lock)"
        if data.get("is_dead_rubber") and is_lock:
            is_lock = False
            reason += " (dead rubber: lock downgraded)"
        result["primary_bet"] = {
            "market": "Over 2.5 Goals", "reason": reason,
            "historical_accuracy": "6/6 when tackles < 2.0" if is_lock else "85.7% overall (24/28)",
        }
        result["classification"] = "OVER 2.5"
        result["verdict"] = "LOCK" if is_lock else "RECOMMENDED"
        result["is_lock"] = is_lock
        return result
    
    # UNDER 2.5
    if not goals_signal:
        result["primary_bet"] = {
            "market": "Under 2.5 Goals",
            "reason": f"Combined goals {combined_goals} < 50",
            "historical_accuracy": "85.7% overall (24/28)",
        }
        result["classification"] = "UNDER 2.5"
        result["verdict"] = "RECOMMENDED"
        return result
    
    # NO SIGNAL
    result["skip_reasons"].append(f"No clear signal: goals={combined_goals}, shots={combined_shots:.1f}, tackles={combined_tackles:.1f}")
    result["classification"] = "DEAD RUBBER" if data.get("is_dead_rubber") else "SKIP"
    return result


# ============================================================================
# EVALUATION ENGINE
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID DATA", "message": "Non-numeric score"}
    total = home + away
    over25 = total > 2
    pred = primary_pred.strip()
    is_correct = False
    if pred == "Over 2.5 Goals": is_correct = over25
    elif pred == "Under 2.5 Goals": is_correct = not over25
    elif pred == "Over 3.5 Goals": is_correct = total > 3
    elif pred == "BTTS": is_correct = home > 0 and away > 0
    return {"is_correct": is_correct, "actual": f"{home}-{away}", "message": f"{'✅' if is_correct else '❌'} {pred} vs {home}-{away}"}


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
            "home_goals_total": data.get("home_goals_total"),
            "away_goals_total": data.get("away_goals_total"),
            "combined_goals": metrics.get("combined_goals", 0),
            "home_shots_pg": data.get("home_shots_pg"),
            "away_shots_pg": data.get("away_shots_pg"),
            "combined_shots": metrics.get("combined_shots", 0),
            "home_tackles_pg": data.get("home_tackles_pg"),
            "away_tackles_pg": data.get("away_tackles_pg"),
            "combined_tackles": metrics.get("combined_tackles", 0),
            "is_dead_rubber": data.get("is_dead_rubber", False),
            "is_lock": analysis.get("is_lock", False),
            "prediction": primary["market"] if primary else "SKIP",
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "LOCK" if analysis.get("is_lock") else ("PRIMARY" if primary else "SKIP"),
            "verdict": analysis.get("verdict", "SKIP"),
            "signal_breakdown": json.dumps(analysis.get("signal_breakdown", [])),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            "dead_rubber_reason": data.get("dead_rubber_reason"),
            "home_win_pct": data.get("home_win"),
            "draw_pct": data.get("draw"),
            "away_win_pct": data.get("away_win"),
            "btts_pct": data.get("btts"),
            "over25_pct": data.get("over_25"),
            "under25_pct": data.get("under_25"),
            "over35_pct": data.get("over_35"),
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
    except: return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        supabase.table("match_analyses").update({
            "actual_home_goals": home_goals, "actual_away_goals": away_goals,
            "actual_total_goals": total, "actual_over25": total > 2,
            "actual_winner": "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW",
            "actual_btts": home_goals > 0 and away_goals > 0,
            "result_entered": True,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False


def get_results():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except: return []


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
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE V7.1", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                with st.spinner("Analyzing..."):
                    data = parse_match_data_v7(raw_text)
                
                if not data.get("home_team") or not data.get("away_team"):
                    st.error("Could not detect team names.")
                else:
                    analysis = analyze_match_v7(data)
                    save_to_db(data, analysis)
                    
                    if data.get("is_dead_rubber"):
                        st.markdown(f'<div class="dead-rubber-warning">⚠️ {data["dead_rubber_reason"]}</div>', unsafe_allow_html=True)
                    elif data.get("dead_rubber_reason"):
                        st.info(f"ℹ️ {data['dead_rubber_reason']}")
                    
                    v = analysis["verdict"]
                    if v == "LOCK": st.success(f"🔒 LOCK: {data['home_team']} vs {data['away_team']}")
                    elif v == "RECOMMENDED": st.success(f"✅ RECOMMENDED: {data['home_team']} vs {data['away_team']}")
                    else: st.warning(f"⚠️ SKIP: {data['home_team']} vs {data['away_team']}")
                    
                    st.markdown(f"**Classification: {analysis['classification']}**")
                    
                    if analysis.get("signal_breakdown"):
                        st.markdown("### 🔍 Signal Breakdown")
                        for s in analysis["signal_breakdown"]:
                            st.markdown(f"- {s}")
                    
                    if analysis.get("metrics"):
                        m = analysis["metrics"]
                        st.markdown("### 📊 Key Metrics")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["combined_goals"]}</div><div class="metric-label">Combined Goals</div><div style="font-size:0.6rem;color:#64748b;">{m["home_goals"]} + {m["away_goals"]}</div></div>', unsafe_allow_html=True)
                        with c2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["combined_shots"]:.1f}</div><div class="metric-label">Combined Shots pg</div><div style="font-size:0.6rem;color:#64748b;">{m["home_shots"]:.1f} + {m["away_shots"]:.1f}</div></div>', unsafe_allow_html=True)
                        with c3:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{m["combined_tackles"]:.1f}</div><div class="metric-label">Combined Tackles pg</div><div style="font-size:0.6rem;color:#64748b;">{m["home_tackles"]:.1f} + {m["away_tackles"]:.1f}</div></div>', unsafe_allow_html=True)
                    
                    if data.get("score_matrix"):
                        st.markdown("### 🎯 Score Matrix (Top 5)")
                        score_cols = st.columns(5)
                        for idx, s in enumerate(data["score_matrix"][:5]):
                            with score_cols[idx]:
                                bg = "#1e293b" if s["home_goals"] != s["away_goals"] else "#2a1a00"
                                st.markdown(f'<div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;"><div style="font-size:1.2rem; font-weight:800;">{s["score"]}</div><div style="font-size:0.7rem; color:#94a3b8;">{s["probability"]:.1f}%</div></div>', unsafe_allow_html=True)
                    
                    if analysis.get("primary_bet"):
                        p = analysis["primary_bet"]
                        is_lock = analysis.get("is_lock", False)
                        emoji = "🔒" if is_lock else "🔥"
                        card_class = "lock-card" if is_lock else "primary-card"
                        badge = '<span class="lock-badge">🔒 LOCK — Tackles < 2.0 (6/6)</span>' if is_lock else '<span class="accuracy-badge">📊 85.7% (24/28)</span>'
                        st.markdown(f'<div class="section-label">🎯 PRIMARY BET</div><div class="output-card {card_class}"><div style="display:flex;align-items:center;gap:1rem;"><div style="font-size:2.5rem;">{emoji}</div><div style="flex:1;"><div style="font-size:1.3rem;font-weight:800;">{p["market"]}</div><div style="font-size:0.8rem;color:#64748b;">{p["reason"]}</div><div style="margin-top:0.5rem;">{badge}</div></div></div></div>', unsafe_allow_html=True)
                    
                    if analysis["verdict"] == "SKIP":
                        st.markdown(f'<div class="output-card skip-card"><div class="verdict-skip"><div class="big-text">⚠️ SKIP — NO BET</div><p style="color:#94a3b8;">{"<br>".join(analysis.get("skip_reasons", []))}</p></div></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: POST-MATCH
    # ========================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('prediction', 'No prediction')
                pat = a.get('pattern', '')
                is_dead = a.get('is_dead_rubber', False)
                
                if pat == "LOCK":
                    badge = "🔒 LOCK"
                elif pat == "PRIMARY" and is_dead:
                    badge = "⚠️ PRIMARY (Dead Rubber)"
                elif pat == "PRIMARY":
                    badge = "📊 PRIMARY"
                else:
                    badge = "⚠️ SKIP"
                
                with st.expander(f"{badge} | {ht} vs {at} — Predicted: {pred}"):
                    c1, c2 = st.columns(2)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
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
            dead_rubber_correct = 0
            dead_rubber_total = 0
            
            for r in results:
                pred = r.get('prediction', '')
                pattern = r.get('pattern', '')
                is_dead = r.get('is_dead_rubber', False)
                
                if pred == 'SKIP':
                    continue
                
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
                
                if evaluation["is_correct"]:
                    correct += 1
                    if pattern == "LOCK":
                        lock_correct += 1
                    if is_dead:
                        dead_rubber_correct += 1
                else:
                    incorrect += 1
                
                if pattern == "LOCK":
                    lock_total += 1
                if is_dead:
                    dead_rubber_total += 1
            
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
            
            # Dead rubber stats
            if dead_rubber_total > 0:
                dead_rate = round(dead_rubber_correct / dead_rubber_total * 100) if dead_rubber_total > 0 else 0
                st.markdown(f"⚠️ **Dead Rubber Bets:** {dead_rubber_correct}/{dead_rubber_total} correct ({dead_rate}%)")
            
            st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")
            
            # Results table
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                classification = r.get('classification', 'Unclassified')
                pattern = r.get('pattern', '')
                is_dead = r.get('is_dead_rubber', False)
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                if pred == 'SKIP':
                    badge = '<span class="skip-badge">⚪ SKIP</span>'
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
                
                if pattern == "LOCK":
                    match_display = f"🔒 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif is_dead:
                    match_display = f"⚠️ {r.get('home_team', '')} vs {r.get('away_team', '')}"
                else:
                    match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
                
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": match_display,
                    "Class": classification,
                    "Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
