"""
STREAK PREDICTOR V5 - Tiered Betting System
Tier 1: Lock Detector (exact thresholds, 90-95% confidence)
Tier 2: Edge Comparison (directional edges for gray zone matches)
Tier 3: No Bet (discipline to pass)
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
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
st.set_page_config(page_title="Streak Predictor V5", page_icon="⚽", layout="wide")

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
    .lock-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .edge-home { border-left: 3px solid #10b981; }
    .edge-away { border-left: 3px solid #ef4444; }
    .edge-even { border-left: 3px solid #94a3b8; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .tier-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .tier-lock { background: #f59e0b; color: #000; }
    .tier-edge { background: #3b82f6; color: #fff; }
    .tier-skip { background: #ef4444; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================
def reverse_replace_last(text, old, new):
    """Replace the last occurrence of old in text with new."""
    parts = text.rsplit(old, 1)
    return new.join(parts)

# ============================================================================
# PARSER
# ============================================================================
def parse_raw_text(raw_text: str) -> dict:
    """Parse raw active streaks text into structured data."""
    
    lines = raw_text.strip().split('\n')
    
    home_name = ""
    away_name = ""
    home_data = {}
    away_data = {}
    current_team = None
    found_home = False
    found_away = False
    pending_number = None
    
    streak_keywords = [
        'scoring', 'over', 'under', 'unbeaten', 'win', 'loss', 'cold', 'hot',
        'goals', 'btts', 'clean', 'without', 'first', 'goal', 'heavy', 'rampage',
        'frenzy', 'drought', 'defeats', 'sheet', 'form', 'nil', 'dominance'
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if stripped.lower().startswith('active streaks'):
            continue
        
        is_whole_number = re.match(r'^\d+$', stripped)
        
        if is_whole_number:
            pending_number = int(stripped)
            continue
        
        whole_numbers = re.findall(r'(?<!\d\.)\b(\d+)\b(?!\.\d)', stripped)
        line_lower = stripped.lower()
        has_streak_keyword = any(kw in line_lower for kw in streak_keywords)
        
        if not whole_numbers and not has_streak_keyword:
            if not found_home:
                home_name = stripped
                current_team = 'home'
                found_home = True
            elif not found_away:
                away_name = stripped
                current_team = 'away'
                found_away = True
            continue
        
        if current_team:
            if whole_numbers:
                streak_value = int(whole_numbers[-1])
            elif pending_number is not None:
                streak_value = pending_number
            else:
                pending_number = None
                continue
            
            pending_number = None
            
            streak_name = stripped
            if whole_numbers:
                streak_name = reverse_replace_last(streak_name, whole_numbers[-1], '')
            
            for sym in ['✓', '✕', '🏠', '✈️', '·']:
                streak_name = streak_name.replace(sym, '')
            
            streak_name = ' '.join(streak_name.split()).strip()
            
            if current_team == 'home' and streak_name:
                home_data[streak_name] = max(home_data.get(streak_name, 0), streak_value)
            elif current_team == 'away' and streak_name:
                away_data[streak_name] = max(away_data.get(streak_name, 0), streak_value)
    
    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_data": home_data,
        "away_data": away_data
    }


def get_signal(data: dict, keys: list) -> int:
    """Get highest value from multiple possible keys"""
    best = 0
    for key in keys:
        if key in data:
            best = max(best, data[key])
    return best


def extract_signals(team_data: dict) -> dict:
    """Extract all relevant signals from parsed team data"""
    return {
        "scoring": get_signal(team_data, ["Scoring"]),
        "over05": get_signal(team_data, ["Over 0.5"]),
        "over25_goals": get_signal(team_data, ["Over 2.5 Goals"]),
        "over25": get_signal(team_data, ["Over 2.5"]),
        "over15_hidden": get_signal(team_data, ["Over 1.5 (hidden)", "Over 1.5(hidden)"]),
        "unbeaten": get_signal(team_data, ["Unbeaten"]),
        "win": get_signal(team_data, ["Win"]),
        "hot_form": get_signal(team_data, ["Hot Form"]),
        "goals2": get_signal(team_data, ["Goals 2+"]),
        "goals3": get_signal(team_data, ["Goals 3+"]),
        "without_win": get_signal(team_data, ["Without Win"]),
        "loss": get_signal(team_data, ["Loss"]),
        "cold_form": get_signal(team_data, ["Cold Form"]),
        "btts": get_signal(team_data, ["BTTS"]),
        "no_btts": get_signal(team_data, ["No BTTS"]),
        "clean_sheet": get_signal(team_data, ["Clean Sheet"]),
        "under25": get_signal(team_data, ["Under 2.5 Goals"]),
        "goal_drought": get_signal(team_data, ["Goal Drought"]),
        "first_to_score": get_signal(team_data, ["First to Score"]),
        "heavy_defeats": get_signal(team_data, ["Heavy Defeats"]),
        "goal_frenzy": get_signal(team_data, ["Goal Frenzy"]),
        "rampage_attack": get_signal(team_data, ["Rampage Attack"]),
        "win_to_nil": get_signal(team_data, ["Win to Nil"]),
        "dominance": get_signal(team_data, ["Dominance Mode"]),
    }


# ============================================================================
# TIER 1: LOCK DETECTOR
# ============================================================================
def check_locks(home: dict, away: dict) -> dict:
    """Check all lock conditions. Locks always override edge predictions."""
    locks = {}
    
    # ========================================================================
    # OVER 2.5 LOCKS
    # ========================================================================
    
    # Lock 1: Scoring ≥ 10 (one team) + Opponent over25_goals ≥ 7
    if (home.get("scoring", 0) >= 10 and away.get("over25_goals", 0) >= 7):
        locks["over_under"] = {
            "prediction": "OVER 2.5", "confidence": 0.95, "tier": "LOCK",
            "reason": "Home scoring ≥ 10 + Away over25_goals ≥ 7"
        }
    elif (away.get("scoring", 0) >= 10 and home.get("over25_goals", 0) >= 7):
        locks["over_under"] = {
            "prediction": "OVER 2.5", "confidence": 0.95, "tier": "LOCK",
            "reason": "Away scoring ≥ 10 + Home over25_goals ≥ 7"
        }
    
    # Lock 2: Both teams scoring ≥ 10
    elif home.get("scoring", 0) >= 10 and away.get("scoring", 0) >= 10:
        locks["over_under"] = {
            "prediction": "OVER 2.5", "confidence": 0.90, "tier": "LOCK",
            "reason": "Both teams scoring ≥ 10"
        }
    
    # ========================================================================
    # BTTS LOCKS
    # ========================================================================
    
    # BTTS YES Lock: Both btts ≥ 5
    if home.get("btts", 0) >= 5 and away.get("btts", 0) >= 5:
        locks["btts"] = {
            "prediction": "BTTS YES", "confidence": 0.85, "tier": "LOCK",
            "reason": "Both btts ≥ 5"
        }
    
    # BTTS NO Lock: Either no_btts ≥ 4 + Opponent scoring ≤ 5
    if (home.get("no_btts", 0) >= 4 and away.get("scoring", 0) <= 5):
        locks["btts"] = {
            "prediction": "BTTS NO", "confidence": 0.75, "tier": "LOCK",
            "reason": "Home no_btts ≥ 4 + Away scoring ≤ 5"
        }
    elif (away.get("no_btts", 0) >= 4 and home.get("scoring", 0) <= 5):
        locks["btts"] = {
            "prediction": "BTTS NO", "confidence": 0.75, "tier": "LOCK",
            "reason": "Away no_btts ≥ 4 + Home scoring ≤ 5"
        }
    
    # ========================================================================
    # WINNER LOCKS
    # ========================================================================
    
    # HOME WIN Lock: Home first_to_score ≥ 5 + Away first_to_score = 0
    if home.get("first_to_score", 0) >= 5 and away.get("first_to_score", 0) == 0:
        locks["winner"] = {
            "prediction": "HOME", "confidence": 0.90, "tier": "LOCK",
            "reason": "Home first_to_score ≥ 5 + Away first_to_score = 0"
        }
    
    # AWAY WIN Lock: Home without_win ≥ 4 + Home over25_goals ≥ 7 + Away win_to_nil ≥ 3
    if home.get("without_win", 0) >= 4 and home.get("over25_goals", 0) >= 7 and away.get("win_to_nil", 0) >= 3:
        locks["winner"] = {
            "prediction": "AWAY", "confidence": 0.90, "tier": "LOCK",
            "reason": "Home collapse + Away win_to_nil ≥ 3"
        }
    
    # DRAW Lock: Both unbeaten ≥ 8 + Both first_to_score ≤ 3
    if home.get("unbeaten", 0) >= 8 and away.get("unbeaten", 0) >= 8 and \
       home.get("first_to_score", 0) <= 3 and away.get("first_to_score", 0) <= 3:
        locks["winner"] = {
            "prediction": "DRAW", "confidence": 0.70, "tier": "LOCK",
            "reason": "Both unbeaten ≥ 8 + low first_to_score"
        }
    
    return locks


# ============================================================================
# TIER 2: EDGE CALCULATOR
# ============================================================================
def calculate_edges(home: dict, away: dict) -> dict:
    """Calculate edge comparison between home and away"""
    
    edges = []
    home_edge_count = 0
    away_edge_count = 0
    attacking_score = 0
    defensive_score = 0
    
    comparisons = [
        ("Scoring", "scoring", "scoring", "attack", "attack", 2),
        ("Over 2.5 Goals", "over25_goals", "over25_goals", "attack", "attack", 2),
        ("Over 2.5", "over25", "over25", "attack", "attack", 1),
        ("Over 1.5(hidden)", "over15_hidden", "over15_hidden", "attack", "attack", 2),
        ("Goals 2+", "goals2", "goals2", "attack", "attack", 1),
        ("Goals 3+", "goals3", "goals3", "attack", "attack", 1),
        ("BTTS", "btts", "btts", "attack", "attack", 1),
        ("First to Score", "first_to_score", "first_to_score", "attack", "attack", 2),
        ("Win", "win", "win", "attack", "attack", 2),
        ("Hot Form", "hot_form", "hot_form", "attack", "attack", 2),
        ("Goal Frenzy", "goal_frenzy", "goal_frenzy", "attack", "attack", 2),
        ("Rampage Attack", "rampage_attack", "rampage_attack", "attack", "attack", 3),
        ("Win to Nil", "win_to_nil", "win_to_nil", "defense", "defense", 2),
        ("Unbeaten", "unbeaten", "unbeaten", "defense", "defense", 1),
        ("Clean Sheet", "clean_sheet", "clean_sheet", "defense", "defense", 2),
        ("No BTTS", "no_btts", "no_btts", "defense", "defense", 1),
        ("Under 2.5 Goals", "under25", "under25", "defense", "defense", 1),
        ("Dominance Mode", "dominance", "dominance", "attack", "attack", 3),
        ("Without Win", "without_win", "without_win", "attack", "defense", 1),
        ("Loss", "loss", "loss", "attack", "defense", 2),
        ("Cold Form", "cold_form", "cold_form", "attack", "defense", 2),
        ("Goal Drought", "goal_drought", "goal_drought", "attack", "defense", 1),
        ("Heavy Defeats", "heavy_defeats", "heavy_defeats", "attack", "defense", 2),
    ]
    
    for display, h_key, a_key, h_type, a_type, weight in comparisons:
        h_val = home.get(h_key, 0)
        a_val = away.get(a_key, 0)
        
        if h_val == 0 and a_val == 0:
            continue
        
        if h_val > a_val:
            edge_to = "home"
            home_edge_count += 1
            if h_type == "attack":
                attacking_score += weight
            else:
                defensive_score += weight
        elif a_val > h_val:
            edge_to = "away"
            away_edge_count += 1
            if a_type == "attack":
                attacking_score += weight
            else:
                defensive_score += weight
        else:
            edge_to = "even"
        
        edges.append({
            "signal": display,
            "home_val": h_val,
            "away_val": a_val,
            "edge": edge_to,
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
# TIER 2: EDGE-BASED PREDICTIONS
# ============================================================================
def predict_from_edges(edge_data: dict, home: dict, away: dict) -> dict:
    """Generate predictions from edge data (Tier 2)"""
    
    home_edges = edge_data["home_edge_count"]
    away_edges = edge_data["away_edge_count"]
    net = edge_data["net_score"]
    
    # Over/Under
    if net >= 4:
        over_under = "OVER 2.5"
        over_conf = min(85, 60 + net * 4)
    elif net >= 1:
        over_under = "OVER 2.5"
        over_conf = 50 + net * 4
    elif net <= -4:
        over_under = "UNDER 2.5"
        over_conf = min(85, 60 + abs(net) * 4)
    elif net <= -1:
        over_under = "UNDER 2.5"
        over_conf = 50 + abs(net) * 4
    else:
        over_under = "SKIP"
        over_conf = 50
    
    over_conf = min(85, max(35, over_conf))
    
    # Winner
    edge_diff = home_edges - away_edges
    if home_edges >= 6 and edge_diff >= 3:
        winner = "HOME"
        win_conf = 65 + edge_diff * 4
    elif away_edges >= 6 and edge_diff <= -3:
        winner = "AWAY"
        win_conf = 65 + abs(edge_diff) * 4
    elif abs(edge_diff) <= 2:
        winner = "DRAW"
        win_conf = 50 + (5 - abs(edge_diff)) * 3
    elif edge_diff > 0:
        winner = "HOME"
        win_conf = 50 + edge_diff * 3
    else:
        winner = "AWAY"
        win_conf = 50 + abs(edge_diff) * 3
    
    win_conf = min(85, max(35, win_conf))
    
    # BTTS
    home_scores = home.get("scoring", 0) > 0
    away_scores = away.get("scoring", 0) > 0
    has_btts = home.get("btts", 0) >= 3 or away.get("btts", 0) >= 3
    home_nobtts = home.get("no_btts", 0) >= 3
    away_nobtts = away.get("no_btts", 0) >= 3
    
    if home_scores and away_scores and has_btts:
        btts = "BTTS YES"
        btts_conf = 70
    elif home_scores and away_scores:
        btts = "BTTS YES"
        btts_conf = 55
    elif home_nobtts or away_nobtts:
        btts = "BTTS NO"
        btts_conf = 65
    else:
        btts = "BTTS NO"
        btts_conf = 55
    
    return {
        "over_under": over_under,
        "over_confidence": over_conf,
        "winner": winner,
        "winner_confidence": win_conf,
        "btts": btts,
        "btts_confidence": min(85, max(35, btts_conf)),
    }


# ============================================================================
# TIER 3: MERGE LOCKS + EDGES
# ============================================================================
def get_final_predictions(home: dict, away: dict, edge_data: dict) -> dict:
    """Tier 1 locks override Tier 2 edges."""
    
    locks = check_locks(home, away)
    edges = predict_from_edges(edge_data, home, away)
    
    final = {}
    
    # Over/Under
    if "over_under" in locks:
        final["over_under"] = locks["over_under"]
    else:
        final["over_under"] = {
            "prediction": edges["over_under"],
            "confidence": edges["over_confidence"] / 100,
            "tier": "EDGE",
            "reason": f"Edge comparison (net score: {edge_data['net_score']})"
        }
    
    # Winner
    if "winner" in locks:
        final["winner"] = locks["winner"]
    else:
        final["winner"] = {
            "prediction": edges["winner"],
            "confidence": edges["winner_confidence"] / 100,
            "tier": "EDGE",
            "reason": f"Edge comparison (diff: {edge_data['home_edge_count'] - edge_data['away_edge_count']})"
        }
    
    # BTTS
    if "btts" in locks:
        final["btts"] = locks["btts"]
    else:
        final["btts"] = {
            "prediction": edges["btts"],
            "confidence": edges["btts_confidence"] / 100,
            "tier": "EDGE",
            "reason": "Edge comparison"
        }
    
    return final


# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_to_db(home_name, away_name, home_signals, away_signals, predictions):
    try:
        record = {
            "home_team": home_name,
            "away_team": away_name,
            "match_date": str(date.today()),
            "home_data": home_signals,
            "away_data": away_signals,
            "prediction": predictions["over_under"]["prediction"],
            "confidence_score": predictions["over_under"]["confidence"],
            "winner": predictions["winner"]["prediction"],
            "winner_confidence": f"{predictions['winner']['confidence']:.0f}%",
            "btts": predictions["btts"]["prediction"],
            "btts_confidence": predictions["btts"]["confidence"],
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
    st.title("⚽ Streak Predictor V5")
    st.caption("Tier 1: Lock Detector → Tier 2: Edge Comparison → Tier 3: No Bet")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Paste Raw Active Streaks")
        
        raw_text = st.text_area("Raw Data", height=300, key="raw_input",
                                placeholder="Active streaks\nTeam A\nScoring              5\nOver 0.5             10\n\nTeam B\nScoring              3\nOver 0.5              8")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the raw active streaks data.")
            else:
                parsed = parse_raw_text(raw_text)
                
                if not parsed["home_name"] or not parsed["away_name"]:
                    st.error(f"Could not detect team names.")
                else:
                    home_signals = extract_signals(parsed["home_data"])
                    away_signals = extract_signals(parsed["away_data"])
                    edge_data = calculate_edges(home_signals, away_signals)
                    predictions = get_final_predictions(home_signals, away_signals, edge_data)
                    save_to_db(parsed["home_name"], parsed["away_name"],
                              home_signals, away_signals, predictions)
                    
                    st.success(f"✅ Parsed: {parsed['home_name']} vs {parsed['away_name']}")
                    
                    # Signal Summary
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="edge-box edge-home">
                            <strong>🏠 {parsed['home_name']}</strong><br>
                            Scoring: {home_signals['scoring']} | Over 2.5 Goals: {home_signals['over25_goals']}<br>
                            First to Score: {home_signals['first_to_score']} | BTTS: {home_signals['btts']} | No BTTS: {home_signals['no_btts']}<br>
                            Unbeaten: {home_signals['unbeaten']} | Win: {home_signals['win']} | Hot: {home_signals['hot_form']}<br>
                            Without Win: {home_signals['without_win']} | Win to Nil: {home_signals['win_to_nil']}<br>
                            Goal Frenzy: {home_signals['goal_frenzy']} | Rampage: {home_signals['rampage_attack']}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="edge-box edge-away">
                            <strong>✈️ {parsed['away_name']}</strong><br>
                            Scoring: {away_signals['scoring']} | Over 2.5 Goals: {away_signals['over25_goals']}<br>
                            First to Score: {away_signals['first_to_score']} | BTTS: {away_signals['btts']} | No BTTS: {away_signals['no_btts']}<br>
                            Unbeaten: {away_signals['unbeaten']} | Win: {away_signals['win']} | Hot: {away_signals['hot_form']}<br>
                            Without Win: {away_signals['without_win']} | Win to Nil: {away_signals['win_to_nil']}<br>
                            Goal Frenzy: {away_signals['goal_frenzy']} | Rampage: {away_signals['rampage_attack']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Edge Comparison
                    if edge_data["edges"]:
                        st.markdown("### ⚔️ Edge Comparison")
                        for edge in edge_data["edges"]:
                            edge_class = "edge-home" if edge["edge"] == "home" else "edge-away" if edge["edge"] == "away" else "edge-even"
                            arrow = "←" if edge["edge"] == "home" else "→" if edge["edge"] == "away" else "↔"
                            st.markdown(f"""
                            <div class="edge-box {edge_class}">
                                <strong>{edge['signal']}</strong> &nbsp; {arrow} &nbsp;
                                Home: {edge['home_val']} | Away: {edge['away_val']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Predictions
                    st.markdown("### 🎯 Predictions")
                    col1, col2, col3 = st.columns(3)
                    
                    for col, market, key in [
                        (col1, "Over/Under", "over_under"),
                        (col2, "Winner", "winner"),
                        (col3, "BTTS", "btts")
                    ]:
                        pred = predictions[key]
                        is_lock = pred.get("tier") == "LOCK"
                        
                        if key == "over_under":
                            emoji = "🔥" if "OVER" in pred["prediction"] else "🛡️" if "UNDER" in pred["prediction"] else "⏭️"
                            card_class = "over-card" if "OVER" in pred["prediction"] else "under-card" if "UNDER" in pred["prediction"] else "skip-card"
                        elif key == "winner":
                            emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝"}.get(pred["prediction"], "❓")
                            card_class = "home-card" if pred["prediction"] == "HOME" else "away-card" if pred["prediction"] == "AWAY" else "draw-card"
                        else:
                            emoji = "✅" if "YES" in pred["prediction"] else "❌"
                            card_class = "over-card" if "YES" in pred["prediction"] else "under-card"
                        
                        lock_badge = '<span class="tier-badge tier-lock">🔒 LOCK</span>' if is_lock else '<span class="tier-badge tier-edge">📊 EDGE</span>'
                        
                        with col:
                            st.markdown(f"""
                            <div class="output-card {card_class} {'lock-card' if is_lock else ''}">
                                <div style="text-align:center;">
                                    {lock_badge}
                                    <div style="font-size:2rem;">{emoji}</div>
                                    <div style="font-size:1.3rem;font-weight:800;">{pred['prediction']}</div>
                                    <div style="font-size:0.85rem;color:#94a3b8;">{pred['confidence']:.0%} confidence</div>
                                    <div style="font-size:0.75rem;color:#94a3b8;">{pred['reason']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
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
                        btts_yes = home_goals > 0 and away_goals > 0
                        
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{home_goals} - {away_goals}</div>
                            <div class="score-label">Total: {total} | {'Over 2.5' if over25 else 'Under 2.5'} | BTTS: {'Yes' if btts_yes else 'No'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("✅ Submit", key=f"submit_{analysis['id']}"):
                        if submit_result(analysis['id'], home_goals, away_goals):
                            st.success("Submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
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
            for col, label, correct, t in [
                (c1, "Over/Under", correct_ou, total),
                (c2, "Winner", correct_winner, winner_total),
                (c3, "BTTS", correct_btts, btts_total)
            ]:
                rate = round(correct / t * 100) if t > 0 else 0
                color = "#10b981" if rate >= 70 else "#ef4444"
                with col:
                    st.markdown(f"""
                    <div class="output-card" style="text-align:center;">
                        <div style="font-size:0.9rem;">{label}</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};">{correct}/{t} ({rate}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.checkbox("Show all results"):
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
