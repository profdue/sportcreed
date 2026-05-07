"""
STREAK PREDICTOR V5.1 — Complete Rewrite
Tier 1: Lock Detector (composite fragility, dual attack, form-based)
Tier 2: Scored Formulas (Over_Score, BTTS_Score, Winner scoring)
Tier 3: No Bet (confidence < 55%)
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
st.set_page_config(page_title="Streak Predictor V5.1", page_icon="⚽", layout="wide")

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
    .tier-nobet { background: #ef4444; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================
def reverse_replace_last(text, old, new):
    parts = text.rsplit(old, 1)
    return new.join(parts)

def clean_streak_name(raw_name: str) -> str:
    for sym in ['✓', '✕', '🏠', '✈️', '·']:
        raw_name = raw_name.replace(sym, '')
    return ' '.join(raw_name.split()).strip()

def cap(value, maximum):
    return min(value, maximum)

def is_defensively_fragile(team: dict) -> bool:
    score = 0
    if team.get("heavy_defeats", 0) >= 1:
        score += 3
    if team.get("over25_goals", 0) >= 7 and team.get("without_win", 0) >= 4:
        score += 3
    if team.get("over25_goals", 0) >= 10 and team.get("win", 0) == 0:
        score += 2
    if team.get("over25_goals", 0) >= 7 and team.get("hot_form", 0) == 0 and team.get("clean_sheet", 0) == 0:
        score += 1
    if team.get("loss", 0) >= 3:
        score += 1
    return score >= 3

def data_quality_check(home: dict, away: dict) -> int:
    total = home.get("scoring", 0) + away.get("scoring", 0) + \
            home.get("win", 0) + away.get("win", 0) + \
            home.get("btts", 0) + away.get("btts", 0) + \
            home.get("first_to_score", 0) + away.get("first_to_score", 0)
    return 55 if total == 0 else 85

# ============================================================================
# PARSER
# ============================================================================
def parse_raw_text(raw_text: str) -> dict:
    lines = raw_text.strip().split('\n')
    home_name, away_name = "", ""
    home_data, away_data = {}, {}
    current_team = None
    found_home, found_away = False, False
    pending_signal = None
    
    streak_keywords = [
        'scoring', 'over', 'under', 'unbeaten', 'win', 'loss', 'cold', 'hot',
        'goals', 'btts', 'clean', 'without', 'first', 'goal', 'heavy', 'rampage',
        'frenzy', 'drought', 'defeats', 'sheet', 'form', 'nil', 'dominance'
    ]
    
    def process_signal(signal_name: str, value: int):
        if current_team == 'home':
            home_data[signal_name] = max(home_data.get(signal_name, 0), value)
        elif current_team == 'away':
            away_data[signal_name] = max(away_data.get(signal_name, 0), value)
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if stripped.lower().startswith('active streaks'): continue
        
        is_whole_number = re.match(r'^\d+$', stripped)
        whole_numbers = re.findall(r'(?<!\d\.)\b(\d+)\b(?!\.\d)', stripped)
        has_keyword = any(kw in stripped.lower() for kw in streak_keywords)
        
        if is_whole_number:
            if pending_signal is not None and current_team is not None:
                process_signal(pending_signal, int(stripped))
                pending_signal = None
            continue
        
        if not whole_numbers and not has_keyword:
            pending_signal = None
            if not found_home:
                home_name = stripped; current_team = 'home'; found_home = True
            elif not found_away:
                away_name = stripped; current_team = 'away'; found_away = True
            continue
        
        if current_team:
            signal_name = clean_streak_name(stripped)
            if whole_numbers:
                value = int(whole_numbers[-1])
                signal_name = clean_streak_name(reverse_replace_last(stripped, whole_numbers[-1], ''))
                process_signal(signal_name, value)
                pending_signal = None
            else:
                pending_signal = signal_name
    
    return {"home_name": home_name, "away_name": away_name, "home_data": home_data, "away_data": away_data}

def get_signal(data: dict, keys: list) -> int:
    best = 0
    for key in keys:
        if key in data: best = max(best, data[key])
    return best

def extract_signals(team_data: dict) -> dict:
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
    locks = {}
    
    # LOCK 1: OVER 2.5
    over_lock = False
    if home.get("scoring", 0) >= 10 and is_defensively_fragile(away):
        over_lock = True
    elif away.get("scoring", 0) >= 10 and is_defensively_fragile(home):
        over_lock = True
    elif home.get("scoring", 0) >= 10 and away.get("scoring", 0) >= 10:
        over_lock = True
    
    if over_lock:
        locks["over_under"] = {"prediction": "OVER 2.5", "confidence": 95, "tier": "LOCK",
                               "reason": "Scoring ≥ 10 + opponent fragile OR both ≥ 10"}
    
    # LOCK 2: BTTS YES
    if home.get("btts", 0) >= 5 and away.get("btts", 0) >= 5:
        locks["btts"] = {"prediction": "BTTS YES", "confidence": 85, "tier": "LOCK",
                         "reason": "Both btts ≥ 5"}
    
    # LOCK 3: BTTS NO
    if (home.get("no_btts", 0) >= 4 and away.get("scoring", 0) <= 5):
        locks["btts"] = {"prediction": "BTTS NO", "confidence": 80, "tier": "LOCK",
                         "reason": "Home no_btts ≥ 4 + Away scoring ≤ 5"}
    elif (away.get("no_btts", 0) >= 4 and home.get("scoring", 0) <= 5):
        locks["btts"] = {"prediction": "BTTS NO", "confidence": 80, "tier": "LOCK",
                         "reason": "Away no_btts ≥ 4 + Home scoring ≤ 5"}
    
    # LOCK 4: HOME WIN
    home_win_lock = False
    if home.get("first_to_score", 0) >= 5 and away.get("first_to_score", 0) == 0 and away.get("unbeaten", 0) < 8:
        home_win_lock = True
    elif home.get("hot_form", 0) >= 5 and away.get("without_win", 0) >= 4 and away.get("heavy_defeats", 0) >= 1:
        home_win_lock = True
    
    if home_win_lock:
        locks["winner"] = {"prediction": "HOME", "confidence": 90, "tier": "LOCK",
                           "reason": "Home first_to_score domination OR form vs collapse"}
    
    # LOCK 5: AWAY WIN
    if home.get("without_win", 0) >= 4 and home.get("heavy_defeats", 0) >= 1 and \
       (away.get("win_to_nil", 0) >= 3 or away.get("first_to_score", 0) >= 5 or away.get("hot_form", 0) >= 5):
        locks["winner"] = {"prediction": "AWAY", "confidence": 90, "tier": "LOCK",
                           "reason": "Home collapse + Away win signal"}
    
    # LOCK 6: DRAW
    if home.get("win", 0) == 0 and away.get("win", 0) == 0 and \
       home.get("without_win", 0) >= 4 and away.get("without_win", 0) >= 4 and \
       home.get("unbeaten", 0) >= 4 and away.get("unbeaten", 0) >= 4 and \
       (home.get("btts", 0) >= 5 or away.get("btts", 0) >= 5):
        locks["winner"] = {"prediction": "DRAW", "confidence": 80, "tier": "LOCK",
                           "reason": "Both winless + unbeaten + BTTS"}
    
    return locks

# ============================================================================
# TIER 2: SCORED FORMULAS
# ============================================================================
def get_adj_over25_goals(team: dict) -> float:
    raw = team.get("over25_goals", 0)
    if team.get("without_win", 0) >= 4:
        return min(raw, 5)
    return min(raw, 10)

def predict_winner_edge(home: dict, away: dict, max_conf: int) -> dict:
    home_score = 0
    away_score = 0
    
    higher_better = [
        ("scoring", 3), ("first_to_score", 3), ("hot_form", 3), ("win", 2),
        ("over25_goals", 1), ("over15_hidden", 1), ("goal_frenzy", 2),
        ("rampage_attack", 3), ("clean_sheet", 2), ("win_to_nil", 2),
    ]
    lower_better = [
        ("without_win", 2), ("heavy_defeats", 3), ("goal_drought", 3),
        ("cold_form", 3), ("loss", 2),
    ]
    
    for metric, pts in higher_better:
        h = home.get(metric, 0); a = away.get(metric, 0)
        if h > a: home_score += pts
        elif a > h: away_score += pts
    
    for metric, pts in lower_better:
        h = home.get(metric, 0); a = away.get(metric, 0)
        if h > a: away_score += pts
        elif a > h: home_score += pts
    
    if home.get("unbeaten", 0) >= away.get("unbeaten", 0) + 5: home_score += 2
    if away.get("unbeaten", 0) >= home.get("unbeaten", 0) + 5: away_score += 2
    
    if home.get("win", 0) == 0 and home.get("without_win", 0) >= 4 and away.get("without_win", 0) < 4:
        away_score += 2
    if away.get("win", 0) == 0 and away.get("without_win", 0) >= 4 and home.get("without_win", 0) < 4:
        home_score += 2
    
    draw_bias = False
    if home.get("win", 0) == 0 and away.get("win", 0) == 0:
        if home.get("without_win", 0) >= 4 and away.get("without_win", 0) >= 4: draw_bias = True
        if home.get("unbeaten", 0) >= 4 and away.get("unbeaten", 0) >= 4: draw_bias = True
    
    diff = abs(home_score - away_score)
    if diff <= 3 and home.get("first_to_score", 0) <= 2 and away.get("first_to_score", 0) <= 2:
        draw_bias = True
    
    if draw_bias:
        draw_conf = min(60 + min(home.get("unbeaten", 0) + away.get("unbeaten", 0), 10), 78)
        if diff >= 10:
            winner = "HOME" if home_score > away_score else "AWAY"
            conf = min(85 + (diff - 10), max_conf)
        elif diff >= 5:
            edge_conf = min(65 + (diff - 5) * 2, max_conf)
            if draw_conf > edge_conf:
                winner, conf = "DRAW", draw_conf
            else:
                winner = "HOME" if home_score > away_score else "AWAY"
                conf = edge_conf
        else:
            winner, conf = "DRAW", draw_conf
    else:
        if diff >= 10:
            winner = "HOME" if home_score > away_score else "AWAY"
            conf = min(85 + (diff - 10), max_conf)
        elif diff >= 5:
            winner = "HOME" if home_score > away_score else "AWAY"
            conf = min(65 + (diff - 5) * 2, max_conf)
        elif diff >= 1:
            winner = "HOME" if home_score > away_score else "AWAY"
            conf = min(55 + diff, max_conf)
        else:
            winner, conf = "DRAW", 50
    
    return {"prediction": winner, "confidence": conf, "tier": "EDGE"}

def predict_over_edge(home: dict, away: dict, max_conf: int) -> dict:
    over_score = 0
    
    over_score += cap(home.get("scoring", 0), 15)
    over_score += cap(away.get("scoring", 0), 15)
    over_score += get_adj_over25_goals(home)
    over_score += get_adj_over25_goals(away)
    over_score += cap(home.get("over25", 0) * 2, 10)
    over_score += cap(away.get("over25", 0) * 2, 10)
    over_score += cap(home.get("btts", 0), 8)
    over_score += cap(away.get("btts", 0), 8)
    over_score += cap(home.get("over15_hidden", 0) // 2, 5)
    over_score += cap(away.get("over15_hidden", 0) // 2, 5)
    over_score += home.get("goal_frenzy", 0) * 5
    over_score += away.get("goal_frenzy", 0) * 5
    over_score += home.get("rampage_attack", 0) * 4
    over_score += away.get("rampage_attack", 0) * 4
    
    subtract = 0
    subtract += cap(home.get("under25", 0) * 3, 9)
    subtract += cap(away.get("under25", 0) * 3, 9)
    subtract += cap(home.get("no_btts", 0) * 2, 8)
    subtract += cap(away.get("no_btts", 0) * 2, 8)
    subtract += cap(home.get("win_to_nil", 0) * 2, 6)
    subtract += cap(away.get("win_to_nil", 0) * 2, 6)
    subtract += cap(home.get("clean_sheet", 0) * 2, 4)
    subtract += cap(away.get("clean_sheet", 0) * 2, 4)
    
    over_score -= subtract
    
    if over_score >= 40:
        return {"prediction": "OVER 2.5", "confidence": min(85, max_conf), "tier": "EDGE"}
    elif over_score >= 25:
        return {"prediction": "OVER 2.5", "confidence": 70, "tier": "EDGE"}
    elif over_score >= 15:
        return {"prediction": "OVER 2.5", "confidence": 60, "tier": "EDGE"}
    elif over_score >= 5:
        conf = 52
        if conf < 55: return {"prediction": "NO BET", "confidence": 0, "tier": "NO_BET"}
        return {"prediction": "OVER 2.5", "confidence": conf, "tier": "EDGE"}
    elif over_score <= -15:
        return {"prediction": "UNDER 2.5", "confidence": 75, "tier": "EDGE"}
    elif over_score <= -5:
        return {"prediction": "UNDER 2.5", "confidence": 65, "tier": "EDGE"}
    elif over_score <= 4:
        return {"prediction": "UNDER 2.5", "confidence": 55, "tier": "EDGE"}
    else:
        return {"prediction": "NO BET", "confidence": 0, "tier": "NO_BET"}

def predict_btts_edge(home: dict, away: dict, max_conf: int) -> dict:
    score = 0
    
    score += cap(home.get("btts", 0) * 3, 21)
    score += cap(away.get("btts", 0) * 3, 21)
    score += cap(home.get("scoring", 0) // 2, 7)
    score += cap(away.get("scoring", 0) // 2, 7)
    score += cap(int(get_adj_over25_goals(home)), 7)
    score += cap(int(get_adj_over25_goals(away)), 7)
    
    subtract = 0
    subtract += cap(home.get("no_btts", 0) * 4, 20)
    subtract += cap(away.get("no_btts", 0) * 4, 20)
    subtract += cap(home.get("win_to_nil", 0) * 3, 9)
    subtract += cap(away.get("win_to_nil", 0) * 3, 9)
    subtract += cap(home.get("clean_sheet", 0) * 3, 9)
    subtract += cap(away.get("clean_sheet", 0) * 3, 9)
    subtract += home.get("goal_drought", 0) * 5
    subtract += away.get("goal_drought", 0) * 5
    
    score -= subtract
    
    if score >= 25:
        return {"prediction": "BTTS YES", "confidence": min(80, max_conf), "tier": "EDGE"}
    elif score >= 15:
        return {"prediction": "BTTS YES", "confidence": 70, "tier": "EDGE"}
    elif score >= 8:
        return {"prediction": "BTTS YES", "confidence": 60, "tier": "EDGE"}
    elif score >= 0:
        conf = 52
        if conf < 55: return {"prediction": "NO BET", "confidence": 0, "tier": "NO_BET"}
        return {"prediction": "BTTS YES", "confidence": conf, "tier": "EDGE"}
    elif score <= -15:
        return {"prediction": "BTTS NO", "confidence": 75, "tier": "EDGE"}
    elif score <= -8:
        return {"prediction": "BTTS NO", "confidence": 65, "tier": "EDGE"}
    elif score <= -1:
        return {"prediction": "BTTS NO", "confidence": 55, "tier": "EDGE"}
    else:
        return {"prediction": "NO BET", "confidence": 0, "tier": "NO_BET"}

# ============================================================================
# MERGE
# ============================================================================
def get_final_predictions(home: dict, away: dict) -> dict:
    max_conf = data_quality_check(home, away)
    locks = check_locks(home, away)
    
    final = {}
    
    if "over_under" in locks:
        final["over_under"] = locks["over_under"]
    else:
        edge = predict_over_edge(home, away, max_conf)
        if edge["tier"] == "NO_BET":
            final["over_under"] = edge
        else:
            final["over_under"] = {**edge, "reason": f"Over_Score calculated", "confidence": min(edge["confidence"], max_conf)}
    
    if "winner" in locks:
        final["winner"] = locks["winner"]
    else:
        edge = predict_winner_edge(home, away, max_conf)
        final["winner"] = {**edge, "reason": f"Winner scoring", "confidence": min(edge["confidence"], max_conf)}
    
    if "btts" in locks:
        final["btts"] = locks["btts"]
    else:
        edge = predict_btts_edge(home, away, max_conf)
        if edge["tier"] == "NO_BET":
            final["btts"] = edge
        else:
            final["btts"] = {**edge, "reason": f"BTTS_Score calculated", "confidence": min(edge["confidence"], max_conf)}
    
    return final

# ============================================================================
# SUPABASE
# ============================================================================
def save_to_db(home_name, away_name, home_signals, away_signals, predictions):
    try:
        ou = predictions.get("over_under", {})
        win = predictions.get("winner", {})
        bt = predictions.get("btts", {})
        record = {
            "home_team": home_name, "away_team": away_name,
            "match_date": str(date.today()),
            "home_data": home_signals, "away_data": away_signals,
            "prediction": ou.get("prediction", "SKIP"),
            "confidence_score": ou.get("confidence", 0) / 100,
            "winner": win.get("prediction", "UNCLEAR"),
            "winner_confidence": f"{win.get('confidence', 0):.0f}%",
            "btts": bt.get("prediction", ""),
            "btts_confidence": bt.get("confidence", 0) / 100,
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
    except: return []

def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        over25 = total > 2
        actual_winner = "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW"
        btts_yes = home_goals > 0 and away_goals > 0
        
        record = supabase.table("analyses").select("prediction,winner,btts").eq("id", analysis_id).single().execute()
        if not record.data: return False
        
        pred = record.data.get("prediction", "SKIP")
        pred_winner = record.data.get("winner", "UNCLEAR")
        pred_btts = record.data.get("btts", "")
        
        over_correct = None if pred == "SKIP" else (("OVER" in pred) == over25)
        winner_correct = None if pred_winner in ["UNCLEAR", "DRAW"] else (pred_winner == actual_winner)
        btts_correct = ("YES" in pred_btts) == btts_yes if pred_btts else None
        
        supabase.table("analyses").update({
            "actual_home_goals": home_goals, "actual_away_goals": away_goals,
            "actual_total_goals": total, "actual_over25": over25,
            "actual_winner": actual_winner, "actual_btts": btts_yes,
            "result_entered": True, "correct": over_correct,
            "winner_correct": winner_correct, "btts_correct": btts_correct,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_results():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except: return []

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V5.1")
    st.caption("Tier 1: Lock Detector → Tier 2: Scored Formulas → Tier 3: No Bet")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Paste Raw Active Streaks")
        raw_text = st.text_area("Raw Data", height=300, key="raw_input")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the raw active streaks data.")
            else:
                parsed = parse_raw_text(raw_text)
                if not parsed["home_name"] or not parsed["away_name"]:
                    st.error("Could not detect team names.")
                else:
                    home_signals = extract_signals(parsed["home_data"])
                    away_signals = extract_signals(parsed["away_data"])
                    predictions = get_final_predictions(home_signals, away_signals)
                    save_to_db(parsed["home_name"], parsed["away_name"], home_signals, away_signals, predictions)
                    
                    st.success(f"✅ Parsed: {parsed['home_name']} vs {parsed['away_name']}")
                    
                    col1, col2 = st.columns(2)
                    for col, name, sigs in [(col1, f"🏠 {parsed['home_name']}", home_signals), (col2, f"✈️ {parsed['away_name']}", away_signals)]:
                        with col:
                            st.markdown(f"""
                            <div class="edge-box edge-home">
                                <strong>{name}</strong><br>
                                Scoring: {sigs['scoring']} | Over 2.5 Goals: {sigs['over25_goals']}<br>
                                First to Score: {sigs['first_to_score']} | BTTS: {sigs['btts']} | No BTTS: {sigs['no_btts']}<br>
                                Unbeaten: {sigs['unbeaten']} | Win: {sigs['win']} | Hot: {sigs['hot_form']}<br>
                                Without Win: {sigs['without_win']} | Loss: {sigs['loss']} | Heavy Defeats: {sigs['heavy_defeats']}<br>
                                Goal Frenzy: {sigs['goal_frenzy']} | Rampage: {sigs['rampage_attack']} | Clean Sheet: {sigs['clean_sheet']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Predictions")
                    c1, c2, c3 = st.columns(3)
                    
                    for col, market, key in [(c1, "Over/Under", "over_under"), (c2, "Winner", "winner"), (c3, "BTTS", "btts")]:
                        pred = predictions.get(key, {"prediction": "NO BET", "confidence": 0, "tier": "NO_BET", "reason": "Unknown"})
                        is_lock = pred.get("tier") == "LOCK"
                        is_nobet = pred.get("tier") == "NO_BET"
                        
                        if key == "over_under":
                            emoji = "🔥" if "OVER" in pred.get("prediction", "") else "🛡️" if "UNDER" in pred.get("prediction", "") else "⏭️"
                            card_class = "over-card" if "OVER" in pred.get("prediction", "") else "under-card" if "UNDER" in pred.get("prediction", "") else "skip-card"
                        elif key == "winner":
                            emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝"}.get(pred.get("prediction", ""), "❓")
                            card_class = "home-card" if pred.get("prediction") == "HOME" else "away-card" if pred.get("prediction") == "AWAY" else "draw-card"
                        else:
                            emoji = "✅" if "YES" in pred.get("prediction", "") else "❌" if "NO" in pred.get("prediction", "") else "⏭️"
                            card_class = "over-card" if "YES" in pred.get("prediction", "") else "under-card" if "NO" in pred.get("prediction", "") else "skip-card"
                        
                        if is_lock:
                            badge = '<span class="tier-badge tier-lock">🔒 LOCK</span>'
                        elif is_nobet:
                            badge = '<span class="tier-badge tier-nobet">❌ NO BET</span>'
                        else:
                            badge = '<span class="tier-badge tier-edge">📊 EDGE</span>'
                        
                        with col:
                            st.markdown(f"""
                            <div class="output-card {card_class} {'lock-card' if is_lock else ''}">
                                <div style="text-align:center;">
                                    {badge}
                                    <div style="font-size:2rem;">{emoji}</div>
                                    <div style="font-size:1.3rem;font-weight:800;">{pred.get('prediction', 'N/A')}</div>
                                    <div style="font-size:0.85rem;color:#94a3b8;">{pred.get('confidence', 0):.0f}% confidence</div>
                                    <div style="font-size:0.75rem;color:#94a3b8;">{pred.get('reason', '')}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"{len(pending)} pending")
            for analysis in pending:
                ht = analysis.get('home_team', 'Home'); at = analysis.get('away_team', 'Away')
                with st.expander(f"{ht} vs {at} — {analysis.get('prediction', '?')} | {analysis.get('winner', '?')} | {analysis.get('btts', '?')}"):
                    st.write(f"**Date:** {analysis.get('match_date', '?')}")
                    c1, c2, c3 = st.columns(3)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total = hg + ag
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{hg} - {ag}</div>
                            <div class="score-label">Total: {total} | {'Over 2.5' if total > 2 else 'Under 2.5'} | BTTS: {'Yes' if hg > 0 and ag > 0 else 'No'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    if st.button("✅ Submit", key=f"sub_{analysis['id']}"):
                        if submit_result(analysis['id'], hg, ag):
                            st.success("Submitted!"); st.rerun()
        else:
            st.info("No pending analyses.")
    
    with tab3:
        st.subheader("📊 Records")
        results = get_results()
        if not results:
            st.info("No results yet.")
        else:
            total = len([r for r in results if r.get("prediction") not in ["SKIP", "NO BET"]])
            c_ou = len([r for r in results if r.get("correct") == True])
            c_win = len([r for r in results if r.get("winner_correct") == True])
            c_btts = len([r for r in results if r.get("btts_correct") == True])
            w_total = len([r for r in results if r.get("winner_correct") is not None])
            b_total = len([r for r in results if r.get("btts_correct") is not None])
            
            c1, c2, c3 = st.columns(3)
            for col, label, c, t in [(c1, "Over/Under", c_ou, total), (c2, "Winner", c_win, w_total), (c3, "BTTS", c_btts, b_total)]:
                rate = round(c / t * 100) if t > 0 else 0
                color = "#10b981" if rate >= 70 else "#ef4444"
                with col:
                    st.markdown(f"""
                    <div class="output-card" style="text-align:center;">
                        <div style="font-size:0.9rem;">{label}</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};">{c}/{t} ({rate}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.checkbox("Show all results"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"), "home": r.get("home_team"), "away": r.get("away_team"),
                    "prediction": r.get("prediction"), "correct": r.get("correct"),
                    "winner": r.get("winner"), "winner_correct": r.get("winner_correct"),
                    "btts": r.get("btts"), "btts_correct": r.get("btts_correct"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
