python
"""
STREAK PREDICTOR V4 - Edge-Based Analysis
Paste raw active streaks → Auto-parse → Compare edges → Predict Over/Under, Winner, BTTS
"""

import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import json
import re

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
st.set_page_config(page_title="Streak Predictor V4", page_icon="⚽", layout="wide")

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .over-card { border-left: 5px solid #10b981; }
    .under-card { border-left: 5px solid #3b82f6; }
    .skip-card { border-left: 5px solid #fbbf24; }
    .home-card { border-left: 5px solid #10b981; }
    .away-card { border-left: 5px solid #ef4444; }
    .draw-card { border-left: 5px solid #94a3b8; }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .edge-home { border-left: 3px solid #10b981; }
    .edge-away { border-left: 3px solid #ef4444; }
    .edge-even { border-left: 3px solid #94a3b8; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .insight-box { background: #1a2a1a; border-left: 4px solid #10b981; padding: 0.8rem; border-radius: 8px; margin: 0.4rem 0; color: #fff; }
    .warning-box { background: #2a1a1a; border-left: 4px solid #ef4444; padding: 0.8rem; border-radius: 8px; margin: 0.4rem 0; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PARSER
# ============================================================================
def parse_raw_text(raw_text: str) -> dict:
    """Parse raw active streaks text into structured data"""
    
    # Split into home and away sections
    # Pattern: "Active streaks" followed by team data
    sections = re.split(r'\n(?=[A-Z])', raw_text)
    
    home_data = {}
    away_data = {}
    home_name = ""
    away_name = ""
    current_team = None
    current_section = None
    
    lines = raw_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip header lines
        if line.lower().startswith('active streaks'):
            continue
        
        # Check if this is a team name line (usually just text, no numbers)
        if not re.search(r'\d', line) and not any(kw in line.lower() for kw in ['landed', 'broken', 'under threat', 'needs', 'alive', 'extended', 'at risk', 'on track']):
            if not home_name:
                home_name = line
                current_team = 'home'
            elif not away_name:
                away_name = line
                current_team = 'away'
            continue
        
        # Parse streak lines
        streak_match = re.match(r'(.+?)\s+(\d+)', line)
        if streak_match:
            streak_name = streak_match.group(1).strip()
            streak_value = int(streak_match.group(2))
            
            # Clean streak name
            streak_name = streak_name.replace('Landed', '').replace('✓', '').replace('✕', '').strip()
            streak_name = streak_name.replace('Broken', '').strip()
            streak_name = streak_name.replace('Extended', '').strip()
            streak_name = streak_name.replace('Needs 1 more goal', '').strip()
            streak_name = streak_name.replace('Under threat', '').strip()
            streak_name = streak_name.replace('At risk', '').strip()
            streak_name = streak_name.replace('On track', '').strip()
            streak_name = streak_name.replace('Alive', '').strip()
            
            # Remove emojis
            streak_name = streak_name.replace('🏠', '').replace('✈️', '').strip()
            
            if current_team == 'home':
                home_data[streak_name] = streak_value
            elif current_team == 'away':
                away_data[streak_name] = streak_value
    
    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_data": home_data,
        "away_data": away_data
    }


def get_signal_value(data: dict, signal_keys: list) -> int:
    """Get the highest value from multiple possible keys"""
    for key in signal_keys:
        if key in data:
            return data[key]
    return 0


def extract_signals(team_data: dict) -> dict:
    """Extract all relevant signals from parsed team data"""
    return {
        "scoring": get_signal_value(team_data, ["Scoring"]),
        "over05": get_signal_value(team_data, ["Over 0.5"]),
        "over25_goals": get_signal_value(team_data, ["Over 2.5 Goals"]),
        "over25": get_signal_value(team_data, ["Over 2.5"]),
        "over15_hidden": get_signal_value(team_data, ["Over 1.5 (hidden)"]),
        "unbeaten": get_signal_value(team_data, ["Unbeaten"]),
        "win": get_signal_value(team_data, ["Win"]),
        "hot_form": get_signal_value(team_data, ["Hot Form"]),
        "goals2": get_signal_value(team_data, ["Goals 2+"]),
        "goals3": get_signal_value(team_data, ["Goals 3+"]),
        "without_win": get_signal_value(team_data, ["Without Win"]),
        "loss": get_signal_value(team_data, ["Loss"]),
        "cold_form": get_signal_value(team_data, ["Cold Form"]),
        "btts": get_signal_value(team_data, ["BTTS"]),
        "no_btts": get_signal_value(team_data, ["No BTTS"]),
        "clean_sheet": get_signal_value(team_data, ["Clean Sheet"]),
        "under25": get_signal_value(team_data, ["Under 2.5 Goals"]),
        "goal_drought": get_signal_value(team_data, ["Goal Drought"]),
        "first_to_score": get_signal_value(team_data, ["First to Score"]),
        "heavy_defeats": get_signal_value(team_data, ["Heavy Defeats"]),
    }


# ============================================================================
# EDGE CALCULATOR
# ============================================================================
def calculate_edges(home: dict, away: dict) -> dict:
    """Calculate edge comparison between home and away"""
    
    edges = []
    home_edge_count = 0
    away_edge_count = 0
    attacking_score = 0
    defensive_score = 0
    
    # Define signal comparisons
    comparisons = [
        # (display_name, home_key, away_key, type, home_weight, away_weight)
        ("Scoring", "scoring", "scoring", "attack", 2, 2),
        ("Over 2.5 Goals", "over25_goals", "over25_goals", "attack", 2, 2),
        ("Over 2.5", "over25", "over25", "attack", 2, 2),
        ("Over 1.5(hidden)", "over15_hidden", "over15_hidden", "attack", 2, 2),
        ("Goals 2+", "goals2", "goals2", "attack", 1, 1),
        ("Goals 3+", "goals3", "goals3", "attack", 1, 1),
        ("BTTS", "btts", "btts", "attack", 1, 1),
        ("First to Score", "first_to_score", "first_to_score", "attack", 1, 1),
        
        ("Unbeaten", "unbeaten", "unbeaten", "defense", -1, -1),
        ("Win", "win", "win", "attack", 2, 2),
        ("Hot Form", "hot_form", "hot_form", "attack", 2, 2),
        ("Clean Sheet", "clean_sheet", "clean_sheet", "defense", -2, -2),
        ("No BTTS", "no_btts", "no_btts", "defense", -1, -1),
        ("Under 2.5 Goals", "under25", "under25", "defense", -1, -1),
        
        ("Without Win", "without_win", "without_win", "defense", -1, -1),
        ("Loss", "loss", "loss", "defense", -2, -2),
        ("Cold Form", "cold_form", "cold_form", "defense", -2, -2),
        ("Goal Drought", "goal_drought", "goal_drought", "defense", -1, -1),
        ("Heavy Defeats", "heavy_defeats", "heavy_defeats", "defense", -2, -2),
    ]
    
    for display, h_key, a_key, sig_type, h_weight, a_weight in comparisons:
        h_val = home.get(h_key, 0)
        a_val = away.get(a_key, 0)
        
        if h_val == 0 and a_val == 0:
            continue
        
        if h_val > a_val:
            edge_to = "home"
            home_edge_count += 1
            if sig_type == "attack":
                attacking_score += h_weight
            else:
                defensive_score += abs(h_weight)
        elif a_val > h_val:
            edge_to = "away"
            away_edge_count += 1
            if sig_type == "attack":
                attacking_score += a_weight
            else:
                defensive_score += abs(a_weight)
        else:
            edge_to = "even"
        
        edges.append({
            "signal": display,
            "home_val": h_val,
            "away_val": a_val,
            "edge": edge_to,
            "type": sig_type
        })
    
    return {
        "edges": edges,
        "home_edge_count": home_edge_count,
        "away_edge_count": away_edge_count,
        "attacking_score": attacking_score,
        "defensive_score": defensive_score,
        "net_score": attacking_score - defensive_score
    }


# ============================================================================
# PREDICTION ENGINE
# ============================================================================
def predict(edge_data: dict, home: dict, away: dict) -> dict:
    """Generate predictions from edge data"""
    
    home_edges = edge_data["home_edge_count"]
    away_edges = edge_data["away_edge_count"]
    att_score = edge_data["attacking_score"]
    def_score = edge_data["defensive_score"]
    net = edge_data["net_score"]
    
    # Over/Under
    if net >= 4:
        over_under = "OVER 2.5"
        over_conf = min(95, 65 + net * 5)
    elif net >= 1:
        over_under = "OVER 2.5"
        over_conf = 55 + net * 5
    elif net <= -4:
        over_under = "UNDER 2.5"
        over_conf = min(95, 65 + abs(net) * 5)
    elif net <= -1:
        over_under = "UNDER 2.5"
        over_conf = 55 + abs(net) * 5
    else:
        over_under = "SKIP"
        over_conf = 50
    
    # Winner
    edge_diff = home_edges - away_edges
    if home_edges >= 6 and edge_diff >= 3:
        winner = "HOME"
        win_conf = 70 + edge_diff * 5
    elif away_edges >= 6 and edge_diff <= -3:
        winner = "AWAY"
        win_conf = 70 + abs(edge_diff) * 5
    elif abs(edge_diff) <= 2:
        winner = "DRAW"
        win_conf = 50 + (5 - abs(edge_diff)) * 5
    elif edge_diff > 0:
        winner = "HOME"
        win_conf = 55 + edge_diff * 3
    else:
        winner = "AWAY"
        win_conf = 55 + abs(edge_diff) * 3
    
    # BTTS
    home_scores = home.get("scoring", 0) > 0
    away_scores = away.get("scoring", 0) > 0
    has_btts_signal = home.get("btts", 0) >= 3 or away.get("btts", 0) >= 3
    home_no_btts = home.get("no_btts", 0) >= 3
    away_no_btts = away.get("no_btts", 0) >= 3
    
    if home_scores and away_scores and has_btts_signal:
        btts = "BTTS YES"
        btts_conf = 75
    elif home_scores and away_scores:
        btts = "BTTS YES"
        btts_conf = 60
    elif home_no_btts or away_no_btts:
        btts = "BTTS NO"
        btts_conf = 65
    else:
        btts = "BTTS NO"
        btts_conf = 55
    
    return {
        "over_under": over_under,
        "over_confidence": min(95, max(35, over_conf)),
        "winner": winner,
        "winner_confidence": min(95, max(35, win_conf)),
        "btts": btts,
        "btts_confidence": min(95, max(35, btts_conf)),
    }


# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_to_db(home_name: str, away_name: str, home_signals: dict, away_signals: dict,
               edge_data: dict, predictions: dict, raw_text: str):
    try:
        record = {
            "home_team": home_name,
            "away_team": away_name,
            "match_date": str(date.today()),
            "home_data": home_signals,
            "away_data": away_signals,
            "edge_data": edge_data,
            "prediction": predictions["over_under"],
            "confidence_score": predictions["over_confidence"] / 100,
            "winner": predictions["winner"],
            "winner_confidence": f"{predictions['winner_confidence']:.0f}%",
            "btts": predictions["btts"],
            "btts_confidence": predictions["btts_confidence"] / 100,
            "raw_text": raw_text,
            "result_entered": False,
        }
        response = supabase.table("analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def get_pending():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        over25 = total > 2
        if home_goals > away_goals:
            actual_winner = "HOME"
        elif away_goals > home_goals:
            actual_winner = "AWAY"
        else:
            actual_winner = "DRAW"
        
        btts_yes = home_goals > 0 and away_goals > 0
        
        record = supabase.table("analyses").select("prediction,winner,btts").eq("id", analysis_id).single().execute()
        if not record.data:
            return False
        
        pred = record.data.get("prediction", "SKIP")
        pred_winner = record.data.get("winner", "UNCLEAR")
        pred_btts = record.data.get("btts", "")
        
        over_correct = None if pred == "SKIP" else (("OVER" in pred) == over25)
        winner_correct = None if pred_winner in ["UNCLEAR", "DRAW"] else (pred_winner == actual_winner)
        btts_correct = ("YES" in pred_btts) == btts_yes if pred_btts else None
        
        update_data = {
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total,
            "actual_over25": over25,
            "actual_winner": actual_winner,
            "actual_btts": btts_yes,
            "result_entered": True,
            "correct": over_correct,
            "winner_correct": winner_correct,
            "btts_correct": btts_correct,
        }
        supabase.table("analyses").update(update_data).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False


def get_results():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except:
        return []


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V4")
    st.caption("Paste raw active streaks → Auto-parse → Compare edges → Predict Over/Under, Winner, BTTS")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ============================================================================
    # TAB 1: ANALYZE
    # ============================================================================
    with tab1:
        st.markdown("### 📋 Paste Raw Active Streaks")
        st.caption("Copy the full 'Active streaks' section from the source site and paste below")
        
        raw_text = st.text_area("Raw Data", height=300, key="raw_input",
                                placeholder="Active streaks\nChelsea\nOver 0.5                19\nScoring                 0\n...\n\nNottingham Forest\nOver 0.5 ✈️            17\nScoring ✈️             8\n...")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the raw active streaks data.")
            else:
                # Parse
                parsed = parse_raw_text(raw_text)
                
                if not parsed["home_name"] or not parsed["away_name"]:
                    st.error("Could not detect team names. Check the format.")
                else:
                    home_signals = extract_signals(parsed["home_data"])
                    away_signals = extract_signals(parsed["away_data"])
                    
                    # Calculate edges
                    edge_data = calculate_edges(home_signals, away_signals)
                    
                    # Predict
                    predictions = predict(edge_data, home_signals, away_signals)
                    
                    # Save
                    save_to_db(parsed["home_name"], parsed["away_name"],
                              home_signals, away_signals, edge_data, predictions, raw_text)
                    
                    # ============================================================
                    # DISPLAY RESULTS
                    # ============================================================
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="edge-box edge-home">
                            <strong>🏠 {parsed['home_name']}</strong><br>
                            Scoring: {home_signals['scoring']} | Over 0.5: {home_signals['over05']}<br>
                            Over 2.5 Goals: {home_signals['over25_goals']} | Over 2.5: {home_signals['over25']} | Over 1.5(h): {home_signals['over15_hidden']}<br>
                            Unbeaten: {home_signals['unbeaten']} | Win: {home_signals['win']} | Hot: {home_signals['hot_form']}<br>
                            Goals 2+: {home_signals['goals2']} | BTTS: {home_signals['btts']} | No BTTS: {home_signals['no_btts']}<br>
                            Without Win: {home_signals['without_win']} | Loss: {home_signals['loss']} | Cold: {home_signals['cold_form']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="edge-box edge-away">
                            <strong>✈️ {parsed['away_name']}</strong><br>
                            Scoring: {away_signals['scoring']} | Over 0.5: {away_signals['over05']}<br>
                            Over 2.5 Goals: {away_signals['over25_goals']} | Over 2.5: {away_signals['over25']} | Over 1.5(h): {away_signals['over15_hidden']}<br>
                            Unbeaten: {away_signals['unbeaten']} | Win: {away_signals['win']} | Hot: {away_signals['hot_form']}<br>
                            Goals 2+: {away_signals['goals2']} | BTTS: {away_signals['btts']} | No BTTS: {away_signals['no_btts']}<br>
                            Without Win: {away_signals['without_win']} | Loss: {away_signals['loss']} | Cold: {away_signals['cold_form']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Edge comparison
                    st.markdown("### ⚔️ Edge Comparison")
                    
                    edge_df = pd.DataFrame(edge_data["edges"])
                    if not edge_df.empty:
                        for _, row in edge_df.iterrows():
                            edge_class = "edge-home" if row["edge"] == "home" else "edge-away" if row["edge"] == "away" else "edge-even"
                            arrow = "←" if row["edge"] == "home" else "→" if row["edge"] == "away" else "↔"
                            st.markdown(f"""
                            <div class="edge-box {edge_class}">
                                <strong>{row['signal']}</strong> ({row['type']}) &nbsp; {arrow} &nbsp;
                                Home: {row['home_val']} | Away: {row['away_val']} | Edge: {row['edge'].upper()}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("### 📊 Edge Summary")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Home Edges", edge_data["home_edge_count"])
                    with c2:
                        st.metric("Away Edges", edge_data["away_edge_count"])
                    with c3:
                        st.metric("Attacking Score", edge_data["attacking_score"])
                    with c4:
                        st.metric("Net Score", edge_data["net_score"])
                    
                    # Predictions
                    st.markdown("### 🎯 Predictions")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        over_emoji = "🔥" if "OVER" in predictions["over_under"] else "🛡️" if "UNDER" in predictions["over_under"] else "⏭️"
                        ou_card = "over-card" if "OVER" in predictions["over_under"] else "under-card" if "UNDER" in predictions["over_under"] else "skip-card"
                        st.markdown(f"""
                        <div class="output-card {ou_card}">
                            <div style="text-align:center;">
                                <div style="font-size:2rem;">{over_emoji}</div>
                                <div style="font-size:1.3rem;font-weight:800;">{predictions['over_under']}</div>
                                <div style="font-size:0.85rem;color:#94a3b8;">{predictions['over_confidence']:.0f}% confidence</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        win_emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝"}.get(predictions["winner"], "❓")
                        win_card = "home-card" if predictions["winner"] == "HOME" else "away-card" if predictions["winner"] == "AWAY" else "draw-card"
                        st.markdown(f"""
                        <div class="output-card {win_card}">
                            <div style="text-align:center;">
                                <div style="font-size:2rem;">{win_emoji}</div>
                                <div style="font-size:1.3rem;font-weight:800;">{predictions['winner']}</div>
                                <div style="font-size:0.85rem;color:#94a3b8;">{predictions['winner_confidence']:.0f}% confidence</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        btts_emoji = "✅" if "YES" in predictions["btts"] else "❌"
                        btts_card = "over-card" if "YES" in predictions["btts"] else "under-card"
                        st.markdown(f"""
                        <div class="output-card {btts_card}">
                            <div style="text-align:center;">
                                <div style="font-size:2rem;">{btts_emoji}</div>
                                <div style="font-size:1.3rem;font-weight:800;">{predictions['btts']}</div>
                                <div style="font-size:0.85rem;color:#94a3b8;">{predictions['btts_confidence']:.0f}% confidence</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 2: POST-MATCH
    # ============================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"{len(pending)} pending")
            for analysis in pending:
                home_team = analysis.get('home_team', 'Home')
                away_team = analysis.get('away_team', 'Away')
                
                with st.expander(f"{home_team} vs {away_team} — {analysis.get('prediction', '?')} | {analysis.get('winner', '?')} | {analysis.get('btts', '?')}"):
                    st.write(f"**Date:** {analysis.get('match_date', '?')}")
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        home_goals = st.number_input(f"{home_team} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2:
                        away_goals = st.number_input(f"{away_team} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total = home_goals + away_goals
                        over25 = total > 2
                        if home_goals > away_goals:
                            actual_winner = "HOME"
                            winner_display = f"🏠 {home_team} WIN"
                        elif away_goals > home_goals:
                            actual_winner = "AWAY"
                            winner_display = f"✈️ {away_team} WIN"
                        else:
                            actual_winner = "DRAW"
                            winner_display = "🤝 DRAW"
                        
                        btts_yes = home_goals > 0 and away_goals > 0
                        
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{home_goals} - {away_goals}</div>
                            <div class="score-label">Total: {total} | {'Over 2.5' if over25 else 'Under 2.5'} | BTTS: {'Yes' if btts_yes else 'No'}</div>
                            <div class="score-label">{winner_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("✅ Submit", key=f"submit_{analysis['id']}"):
                        if submit_result(analysis['id'], home_goals, away_goals):
                            st.success("Submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
    # ============================================================================
    # TAB 3: RECORDS
    # ============================================================================
    with tab3:
        st.subheader("📊 Records")
        results = get_results()
        if not results:
            st.info("No results yet.")
        else:
            total = len([r for r in results if r.get("prediction") != "SKIP"])
            correct_ou = len([r for r in results if r.get("correct") == True])
            correct_winner = len([r for r in results if r.get("winner_correct") == True])
            correct_btts = len([r for r in results if r.get("btts_correct") == True])
            winner_total = len([r for r in results if r.get("winner_correct") is not None])
            btts_total = len([r for r in results if r.get("btts_correct") is not None])
            
            c1, c2, c3 = st.columns(3)
            with c1:
                rate = round(correct_ou / total * 100) if total > 0 else 0
                color = "#10b981" if rate >= 70 else "#ef4444"
                st.markdown(f"""
                <div class="output-card" style="text-align:center;">
                    <div style="font-size:0.9rem;">Over/Under</div>
                    <div style="font-size:2rem;font-weight:800;color:{color};">{correct_ou}/{total} ({rate}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                wrate = round(correct_winner / winner_total * 100) if winner_total > 0 else 0
                wcolor = "#10b981" if wrate >= 70 else "#ef4444"
                st.markdown(f"""
                <div class="output-card" style="text-align:center;">
                    <div style="font-size:0.9rem;">Winner</div>
                    <div style="font-size:2rem;font-weight:800;color:{wcolor};">{correct_winner}/{winner_total} ({wrate}%)</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                brate = round(correct_btts / btts_total * 100) if btts_total > 0 else 0
                bcolor = "#10b981" if brate >= 70 else "#ef4444"
                st.markdown(f"""
                <div class="output-card" style="text-align:center;">
                    <div style="font-size:0.9rem;">BTTS</div>
                    <div style="font-size:2rem;font-weight:800;color:{bcolor};">{correct_btts}/{btts_total} ({brate}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.checkbox("📋 Show all results"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"),
                    "home": r.get("home_team"),
                    "away": r.get("away_team"),
                    "prediction": r.get("prediction"),
                    "correct": r.get("correct"),
                    "winner": r.get("winner"),
                    "winner_correct": r.get("winner_correct"),
                    "btts": r.get("btts"),
                    "btts_correct": r.get("btts_correct"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
```
