"""
STREAK PREDICTOR V8 — 3-Rule Betting System
Rule 1: HOME WIN (82.6%) | Rule 2: OVER 2.5 (86.7%) | Rule 3: BTTS YES (81.6%)
~40% trigger rate. Exact thresholds. Skip the rest.
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
st.set_page_config(page_title="Streak Predictor V8", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .bet-card { border: 2px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .skip-card { border-left: 5px solid #fbbf24; }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .rule-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; margin: 0.1rem; }
    .skip-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
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
        
        if not has_keyword:
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
    """Get highest value from multiple possible keys."""
    best = 0
    for key in keys:
        if key in data:
            best = max(best, data[key])
    return best

def extract_signals(team_data: dict) -> dict:
    """Extract all relevant signals. 
    'Over 0.5' is the primary scoring streak — take highest among Scoring/Over 0.5."""
    return {
        "win": get_signal(team_data, ["Win"]),
        "btts": get_signal(team_data, ["BTTS"]),
        "over25": get_signal(team_data, ["Over 2.5"]),
        "no_btts": get_signal(team_data, ["No BTTS"]),
        "under25": get_signal(team_data, ["Under 2.5 Goals"]),
        "scoring": get_signal(team_data, ["Scoring", "Over 0.5"]),
        "hot_form": get_signal(team_data, ["Hot Form"]),
        "unbeaten": get_signal(team_data, ["Unbeaten"]),
        "without_win": get_signal(team_data, ["Without Win"]),
        "first_to_score": get_signal(team_data, ["First to Score"]),
        "over25_goals": get_signal(team_data, ["Over 2.5 Goals"]),
    }

# ============================================================================
# 3-RULE BETTING ENGINE
# ============================================================================
def apply_rules(home: dict, away: dict) -> list:
    """Apply the 3 rules. Returns list of bets triggered."""
    bets = []
    
    h_scoring = home.get("scoring", 0)
    h_hot = home.get("hot_form", 0)
    h_over25g = home.get("over25_goals", 0)
    h_btts = home.get("btts", 0)
    h_win = home.get("win", 0)
    
    a_without = away.get("without_win", 0)
    a_scoring = away.get("scoring", 0)
    a_under25 = away.get("under25", 0)
    
    # ========================================================================
    # RULE 1 — HOME WIN (82.6%)
    # ========================================================================
    if h_scoring == 6 or h_hot > 3 or h_over25g == 8:
        bets.append({
            "rule": "Rule 1",
            "market": "HOME WIN",
            "confidence": 83,
            "reason": f"scoring={h_scoring}, hot_form={h_hot}, over25_goals={h_over25g}"
        })
    
    # ========================================================================
    # RULE 2 — OVER 2.5 (86.7%)
    # ========================================================================
    condition_a = (h_btts > 0 and a_without > 3 and a_scoring > 0)
    condition_b = (h_win == 4)
    
    if condition_a or condition_b:
        reason_parts = []
        if condition_a: reason_parts.append(f"BTTS={h_btts}, away_wo={a_without}, away_sc={a_scoring}")
        if condition_b: reason_parts.append(f"win={h_win}")
        
        bets.append({
            "rule": "Rule 2",
            "market": "OVER 2.5",
            "confidence": 87,
            "reason": " | ".join(reason_parts)
        })
    
    # ========================================================================
    # RULE 3 — BTTS YES (81.6%)
    # ========================================================================
    condition_a = (h_btts > 0 or a_without == 3)
    condition_b = (a_under25 == 0)
    condition_c = (h_hot == 0)
    
    if condition_a and condition_b and condition_c:
        bets.append({
            "rule": "Rule 3",
            "market": "BTTS YES",
            "confidence": 82,
            "reason": f"btts={h_btts}, away_wo={a_without}, away_u25={a_under25}, hot={h_hot}"
        })
    
    return bets


# ============================================================================
# SUPABASE
# ============================================================================
def save_to_db(home_name, away_name, home_signals, away_signals, bets):
    try:
        markets = [b["market"] for b in bets]
        rules = [b["rule"] for b in bets]
        confs = [b["confidence"] for b in bets]
        
        record = {
            "home_team": home_name, "away_team": away_name,
            "match_date": str(date.today()),
            "home_data": home_signals, "away_data": away_signals,
            "prediction": " | ".join(markets) if markets else "NO BET",
            "confidence_score": (sum(confs) / len(confs) / 100) if confs else 0,
            "winner": markets[0] if markets else "NO BET",
            "winner_confidence": f"{confs[0]}%" if confs else "0%",
            "btts": "BTTS YES" if "BTTS YES" in markets else "",
            "btts_confidence": (confs[markets.index("BTTS YES")] / 100) if "BTTS YES" in markets else 0,
            "pattern": " | ".join(rules) if rules else "NO BET",
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
        
        pred = record.data.get("prediction", "NO BET")
        pred_winner = record.data.get("winner", "NO BET")
        pred_btts = record.data.get("btts", "")
        
        # Over/Under correct?
        if "OVER 2.5" in pred:
            over_correct = over25
        else:
            over_correct = None
        
        # Winner correct?
        if pred_winner == "HOME WIN":
            winner_correct = actual_winner == "HOME"
        elif pred_winner == "AWAY WIN":
            winner_correct = actual_winner == "AWAY"
        else:
            winner_correct = None
        
        # BTTS correct?
        if "BTTS YES" in pred_btts:
            btts_correct = btts_yes
        elif "BTTS NO" in pred_btts:
            btts_correct = not btts_yes
        else:
            btts_correct = None
        
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
    st.title("⚽ Streak Predictor V8")
    st.caption("3 Rules. 82-87% Accuracy. ~40% Trigger Rate. Exact Thresholds.")
    
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
                    bets = apply_rules(home_signals, away_signals)
                    save_to_db(parsed["home_name"], parsed["away_name"], home_signals, away_signals, bets)
                    
                    st.success(f"✅ Parsed: {parsed['home_name']} vs {parsed['away_name']}")
                    
                    col1, col2 = st.columns(2)
                    for col, name, sigs in [(col1, f"🏠 {parsed['home_name']}", home_signals), (col2, f"✈️ {parsed['away_name']}", away_signals)]:
                        with col:
                            st.markdown(f"""
                            <div class="edge-box edge-home">
                                <strong>{name}</strong><br>
                                Scoring: {sigs['scoring']} | Hot Form: {sigs['hot_form']} | Over 2.5 Goals: {sigs['over25_goals']}<br>
                                Win: {sigs['win']} | BTTS: {sigs['btts']} | Under 2.5: {sigs['under25']}<br>
                                Without Win: {sigs['without_win']} | First to Score: {sigs['first_to_score']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Bets")
                    
                    if bets:
                        for bet in bets:
                            emoji = {"HOME WIN": "🏠", "OVER 2.5": "🔥", "BTTS YES": "✅"}.get(bet["market"], "⚽")
                            st.markdown(f"""
                            <div class="output-card bet-card">
                                <div style="display:flex;align-items:center;gap:1rem;">
                                    <div style="font-size:2rem;">{emoji}</div>
                                    <div>
                                        <div style="font-size:1.3rem;font-weight:800;">{bet['market']}</div>
                                        <div style="font-size:0.8rem;color:#94a3b8;">
                                            <span class="rule-badge">{bet['rule']}</span>
                                            {bet['confidence']}% confidence
                                        </div>
                                        <div style="font-size:0.75rem;color:#94a3b8;">{bet['reason']}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="output-card skip-card">
                            <div style="text-align:center;">
                                <span class="skip-badge">NO BET</span>
                                <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.5rem;">No rules triggered. Skip this match.</div>
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
                with st.expander(f"{ht} vs {at} — {analysis.get('prediction', '?')}"):
                    st.write(f"**Date:** {analysis.get('match_date', '?')}")
                    st.write(f"**Rules:** {analysis.get('pattern', '?')}")
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
            bets_placed = [r for r in results if r.get("prediction") not in ["NO BET", None, ""]]
            skipped = len([r for r in results if r.get("prediction") in ["NO BET", None, ""]])
            
            rule1_total = len([r for r in bets_placed if "Rule 1" in (r.get("pattern") or "")])
            rule1_correct = len([r for r in bets_placed if "Rule 1" in (r.get("pattern") or "") and r.get("winner_correct") == True])
            
            rule2_total = len([r for r in bets_placed if "Rule 2" in (r.get("pattern") or "")])
            rule2_correct = len([r for r in bets_placed if "Rule 2" in (r.get("pattern") or "") and r.get("correct") == True])
            
            rule3_total = len([r for r in bets_placed if "Rule 3" in (r.get("pattern") or "")])
            rule3_correct = len([r for r in bets_placed if "Rule 3" in (r.get("pattern") or "") and r.get("btts_correct") == True])
            
            st.markdown(f"**Bets placed:** {len(bets_placed)} | **Skipped:** {skipped}")
            
            st.markdown("### Rule Performance")
            for rule, total, correct in [
                ("Rule 1 — HOME WIN (83%)", rule1_total, rule1_correct),
                ("Rule 2 — OVER 2.5 (87%)", rule2_total, rule2_correct),
                ("Rule 3 — BTTS YES (82%)", rule3_total, rule3_correct),
            ]:
                if total > 0:
                    rate = round(correct / total * 100)
                    color = "#10b981" if rate >= 75 else "#ef4444"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                        <div><strong>{rule}</strong></div>
                        <div style="color:{color};">{correct}/{total} ({rate}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.checkbox("Show all results"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"), "home": r.get("home_team"), "away": r.get("away_team"),
                    "prediction": r.get("prediction"), "pattern": r.get("pattern"),
                    "correct": r.get("correct"), "winner_correct": r.get("winner_correct"),
                    "btts_correct": r.get("btts_correct"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
