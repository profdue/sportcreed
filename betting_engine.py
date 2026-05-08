"""
STREAK PREDICTOR V6 — Data-Driven Locks + Statistical Edge
16 matches backtested. Every threshold from actual outcomes.
Only 10 statistically significant fields retained.
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
st.set_page_config(page_title="Streak Predictor V6", page_icon="⚽", layout="wide")

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

def clamp(value, lo, hi):
    return max(lo, min(hi, value))

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
            if whole_numbers:
                value = int(whole_numbers[-1])
                signal_name = clean_streak_name(reverse_replace_last(stripped, whole_numbers[-1], ''))
                process_signal(signal_name, value)
                pending_signal = None
            else:
                pending_signal = clean_streak_name(stripped)
    
    return {"home_name": home_name, "away_name": away_name, "home_data": home_data, "away_data": away_data}

def get_signal(data: dict, keys: list) -> int:
    best = 0
    for key in keys:
        if key in data: best = max(best, data[key])
    return best

def extract_signals(team_data: dict) -> dict:
    """Extract only the 10 statistically significant fields."""
    return {
        "scoring": get_signal(team_data, ["Scoring"]),
        "hot_form": get_signal(team_data, ["Hot Form"]),
        "first_to_score": get_signal(team_data, ["First to Score"]),
        "win": get_signal(team_data, ["Win"]),
        "no_btts": get_signal(team_data, ["No BTTS"]),
        "under25": get_signal(team_data, ["Under 2.5 Goals"]),
        "without_win": get_signal(team_data, ["Without Win"]),
        "unbeaten": get_signal(team_data, ["Unbeaten"]),
        "btts": get_signal(team_data, ["BTTS"]),
        "over25": get_signal(team_data, ["Over 2.5"]),
        # Keep over25_goals ONLY for the U1 override
        "over25_goals": get_signal(team_data, ["Over 2.5 Goals"]),
    }

# ============================================================================
# TIER 1: LOCKS (DATA-DRIVEN THRESHOLDS)
# ============================================================================
def check_locks(home: dict, away: dict) -> dict:
    locks = {}
    
    h_score = home.get("scoring", 0)
    a_score = away.get("scoring", 0)
    h_hot = home.get("hot_form", 0)
    a_hot = away.get("hot_form", 0)
    h_fts = home.get("first_to_score", 0)
    a_fts = away.get("first_to_score", 0)
    h_win = home.get("win", 0)
    a_win = away.get("win", 0)
    h_nobtts = home.get("no_btts", 0)
    h_over25g = home.get("over25_goals", 0)
    
    # ========================================================================
    # OVER 2.5 LOCKS
    # ========================================================================
    
    # O2: Dual Attack (100% backtest)
    if h_score >= 10 and a_score >= 10:
        locks["over_under"] = {"prediction": "OVER 2.5", "confidence": 100, "tier": "LOCK",
                               "reason": "Both scoring ≥ 10 (3/3 backtest)"}
    
    # O1: Home Dominance (80% backtest)
    elif h_score >= 6 and h_hot >= 5:
        locks["over_under"] = {"prediction": "OVER 2.5", "confidence": 80, "tier": "LOCK",
                               "reason": f"Home scoring≥6 + Hot≥5 (4/5 backtest)"}
    
    # ========================================================================
    # UNDER 2.5 LOCKS
    # ========================================================================
    
    # U1: Home Can't Score (100% backtest, with override)
    if h_score <= 4:
        if a_score >= 10 and h_over25g >= 10:
            pass  # Override: super-club away, fall to edge
        else:
            locks["over_under"] = {"prediction": "UNDER 2.5", "confidence": 100, "tier": "LOCK",
                                   "reason": "Home scoring ≤ 4 (4/4 backtest)"}
    
    # U2: Home Offensive Struggle (100% backtest)
    elif h_score == 5:
        locks["over_under"] = {"prediction": "UNDER 2.5", "confidence": 100, "tier": "LOCK",
                               "reason": "Home scoring = 5 (3/3 backtest)"}
    
    # U3: Home No BTTS Signal (100% backtest)
    elif h_nobtts >= 3:
        locks["over_under"] = {"prediction": "UNDER 2.5", "confidence": 100, "tier": "LOCK",
                               "reason": "Home no_btts ≥ 3 (2/2 backtest)"}
    
    # U4: No Hot Form, Limited Scoring (80% backtest)
    elif h_hot == 0 and a_hot == 0 and h_score <= 8 and a_score <= 8:
        locks["over_under"] = {"prediction": "UNDER 2.5", "confidence": 80, "tier": "LOCK",
                               "reason": "No hot form + limited scoring (8/10 backtest)"}
    
    # ========================================================================
    # BTTS LOCKS
    # ========================================================================
    
    # B2: BTTS NO — Either team can't score (100% backtest)
    if h_score <= 4 or a_score <= 4:
        locks["btts"] = {"prediction": "BTTS NO", "confidence": 100, "tier": "LOCK",
                         "reason": "Either team scoring ≤ 4 (7/7 backtest)"}
    
    # B1: BTTS YES — Both scoring well (71% backtest)
    elif h_score >= 6 and a_score >= 6:
        locks["btts"] = {"prediction": "BTTS YES", "confidence": 71, "tier": "LOCK",
                         "reason": "Both scoring ≥ 6 (5/7 backtest)"}
    
    # ========================================================================
    # WINNER LOCKS
    # ========================================================================
    
    # W1: Home Win — First to Score Domination (80% backtest)
    if h_fts >= 5 and a_fts == 0:
        locks["winner"] = {"prediction": "HOME", "confidence": 80, "tier": "LOCK",
                           "reason": "Home fts≥5 + Away fts=0 (4/5 backtest)"}
    
    # W3: Away Win — Clear Win Signal (100% backtest)
    elif a_win >= 3 and h_win == 0 and a_fts >= 3:
        locks["winner"] = {"prediction": "AWAY", "confidence": 100, "tier": "LOCK",
                           "reason": "Away win≥3 + Home win=0 + Away fts≥3 (2/2 backtest)"}
    
    # W2: Home Win — Win Difference (75% backtest)
    elif h_win >= 6 and a_win == 0:
        locks["winner"] = {"prediction": "HOME", "confidence": 75, "tier": "LOCK",
                           "reason": f"Home win≥6 + Away win=0 (3/4 backtest)"}
    
    return locks


# ============================================================================
# TIER 2: EDGE FORMULAS (STATISTICAL PROBABILITY ADJUSTMENTS)
# ============================================================================
def predict_over_edge(home: dict, away: dict) -> dict:
    h_score = home.get("scoring", 0)
    h_hot = home.get("hot_form", 0)
    h_fts = home.get("first_to_score", 0)
    h_nobtts = home.get("no_btts", 0)
    h_under25 = home.get("under25", 0)
    a_score = away.get("scoring", 0)
    
    prob = 0.50
    
    if h_score >= 10:       prob += 0.17
    elif h_score >= 6:      prob += 0.07
    elif h_score <= 5:      prob -= 0.36
    
    if h_hot >= 5:          prob += 0.33
    elif h_hot == 0:        prob -= 0.30
    
    if h_fts >= 5:          prob += 0.17
    if h_nobtts >= 3:       prob -= 0.50
    if h_under25 >= 3:      prob -= 0.50
    if a_score >= 10:       prob += 0.10
    
    conf = clamp(int(prob * 100), 50, 95)
    
    if prob > 0.50:
        return {"prediction": "OVER 2.5", "confidence": conf, "tier": "EDGE",
                "reason": f"Statistical edge (prob={prob:.2f})"}
    else:
        return {"prediction": "UNDER 2.5", "confidence": conf, "tier": "EDGE",
                "reason": f"Statistical edge (prob={prob:.2f})"}


def predict_btts_edge(home: dict, away: dict) -> dict:
    h_score = home.get("scoring", 0)
    a_score = away.get("scoring", 0)
    h_nobtts = home.get("no_btts", 0)
    a_nobtts = away.get("no_btts", 0)
    
    prob = 0.50
    
    if h_score >= 5 and a_score >= 5: prob += 0.14
    if h_score >= 6 and a_score >= 6: prob += 0.07
    if h_score <= 4 or a_score <= 4:  prob -= 0.50
    if h_nobtts >= 3:                  prob -= 0.25
    if a_nobtts >= 3:                  prob -= 0.25
    
    conf = clamp(int(prob * 100), 50, 95)
    
    if prob > 0.50:
        return {"prediction": "BTTS YES", "confidence": conf, "tier": "EDGE",
                "reason": f"Statistical edge (prob={prob:.2f})"}
    else:
        return {"prediction": "BTTS NO", "confidence": conf, "tier": "EDGE",
                "reason": f"Statistical edge (prob={prob:.2f})"}


def predict_winner_edge(home: dict, away: dict) -> dict:
    h_fts = home.get("first_to_score", 0)
    a_fts = away.get("first_to_score", 0)
    h_win = home.get("win", 0)
    a_win = away.get("win", 0)
    h_hot = home.get("hot_form", 0)
    a_hot = away.get("hot_form", 0)
    h_wo = home.get("without_win", 0)
    a_wo = away.get("without_win", 0)
    h_unb = home.get("unbeaten", 0)
    a_unb = away.get("unbeaten", 0)
    
    home_adv = 0
    home_adv += h_fts * 3
    home_adv -= a_fts * 3
    home_adv += (h_win - a_win) * 1.5
    home_adv += (h_hot - a_hot) * 2
    
    if h_wo >= 4: home_adv -= 5
    if a_wo >= 4: home_adv += 5
    
    if home_adv >= 12:
        conf = clamp(55 + int(home_adv / 2), 55, 85)
        return {"prediction": "HOME", "confidence": conf, "tier": "EDGE",
                "reason": f"Home advantage +{home_adv}"}
    elif home_adv <= -8:
        conf = clamp(55 + int(abs(home_adv) / 2), 55, 80)
        return {"prediction": "AWAY", "confidence": conf, "tier": "EDGE",
                "reason": f"Away advantage {home_adv}"}
    else:
        conf = clamp(50 + min((h_unb + a_unb) / 2, 20), 50, 70)
        return {"prediction": "DRAW", "confidence": int(conf), "tier": "EDGE",
                "reason": f"Balanced (adv={home_adv})"}


# ============================================================================
# MERGE
# ============================================================================
def get_final_predictions(home: dict, away: dict) -> dict:
    locks = check_locks(home, away)
    final = {}
    
    if "over_under" in locks:
        final["over_under"] = locks["over_under"]
    else:
        final["over_under"] = predict_over_edge(home, away)
    
    if "winner" in locks:
        final["winner"] = locks["winner"]
    else:
        final["winner"] = predict_winner_edge(home, away)
    
    if "btts" in locks:
        final["btts"] = locks["btts"]
    else:
        final["btts"] = predict_btts_edge(home, away)
    
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
            "confidence_score": ou.get("confidence", 50) / 100,
            "winner": win.get("prediction", "UNCLEAR"),
            "winner_confidence": f"{win.get('confidence', 0):.0f}%",
            "btts": bt.get("prediction", ""),
            "btts_confidence": bt.get("confidence", 50) / 100,
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
    st.title("⚽ Streak Predictor V6")
    st.caption("Data-Driven Locks + Statistical Edge | 16 matches backtested | 10 fields retained")
    
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
                                Scoring: {sigs['scoring']} | Hot Form: {sigs['hot_form']} | First to Score: {sigs['first_to_score']}<br>
                                Win: {sigs['win']} | No BTTS: {sigs['no_btts']} | Under 2.5: {sigs['under25']}<br>
                                Without Win: {sigs['without_win']} | Unbeaten: {sigs['unbeaten']} | BTTS: {sigs['btts']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Predictions")
                    c1, c2, c3 = st.columns(3)
                    
                    for col, market, key in [(c1, "Over/Under", "over_under"), (c2, "Winner", "winner"), (c3, "BTTS", "btts")]:
                        pred = predictions.get(key, {"prediction": "N/A", "confidence": 50, "tier": "EDGE", "reason": ""})
                        is_lock = pred.get("tier") == "LOCK"
                        
                        if key == "over_under":
                            emoji = "🔥" if "OVER" in pred.get("prediction", "") else "🛡️" if "UNDER" in pred.get("prediction", "") else "⏭️"
                            card_class = "over-card" if "OVER" in pred.get("prediction", "") else "under-card"
                        elif key == "winner":
                            emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝"}.get(pred.get("prediction", ""), "❓")
                            card_class = "home-card" if pred.get("prediction") == "HOME" else "away-card" if pred.get("prediction") == "AWAY" else "draw-card"
                        else:
                            emoji = "✅" if "YES" in pred.get("prediction", "") else "❌" if "NO" in pred.get("prediction", "") else "⏭️"
                            card_class = "over-card" if "YES" in pred.get("prediction", "") else "under-card"
                        
                        badge = '<span class="tier-badge tier-lock">🔒 LOCK</span>' if is_lock else '<span class="tier-badge tier-edge">📊 EDGE</span>'
                        
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
            total = len([r for r in results if r.get("prediction") not in ["SKIP", "NO BET", None]])
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
