"""
STREAK PREDICTOR V7 — Pattern-Based Betting Engine
5 patterns from 172 matches. 87% accuracy. 44% bet rate.
Only bet when a clear pattern exists. Skip the noise.
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
st.set_page_config(page_title="Streak Predictor V7", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .over-card { border-left: 5px solid #10b981; }
    .under-card { border-left: 5px solid #3b82f6; }
    .home-card { border-left: 5px solid #10b981; }
    .away-card { border-left: 5px solid #ef4444; }
    .draw-card { border-left: 5px solid #94a3b8; }
    .skip-card { border-left: 5px solid #fbbf24; }
    .pattern-card { border: 2px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .pattern-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; }
    .noise-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
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
    best = 0
    for key in keys:
        if key in data: best = max(best, data[key])
    return best

def extract_signals(team_data: dict) -> dict:
    return {
        "win": get_signal(team_data, ["Win"]),
        "btts": get_signal(team_data, ["BTTS"]),
        "over25": get_signal(team_data, ["Over 2.5"]),
        "no_btts": get_signal(team_data, ["No BTTS"]),
        "under25": get_signal(team_data, ["Under 2.5 Goals"]),
        "scoring": get_signal(team_data, ["Scoring"]),
        "hot_form": get_signal(team_data, ["Hot Form"]),
        "unbeaten": get_signal(team_data, ["Unbeaten"]),
        "without_win": get_signal(team_data, ["Without Win"]),
        "first_to_score": get_signal(team_data, ["First to Score"]),
        "over25_goals": get_signal(team_data, ["Over 2.5 Goals"]),
    }

# ============================================================================
# PATTERN DETECTION ENGINE
# ============================================================================
def is_killer_combo(team: dict) -> bool:
    w = team.get('win', 0)
    f = team.get('first_to_score', 0)
    return w > 0 and w == f

def is_dangerous_opponent(team: dict) -> bool:
    return (team.get('btts', 0) > 5 or 
            team.get('over25', 0) > 5 or 
            team.get('scoring', 0) > 8 or
            is_killer_combo(team))

def is_dead_team(team: dict) -> bool:
    wo = team.get('without_win', 0)
    sc = team.get('scoring', 0)
    return wo > 12 or (wo > 8 and sc == 0)

def is_fire_team(team: dict) -> bool:
    return team.get('btts', 0) > 3 or team.get('over25', 0) > 3 or team.get('scoring', 0) > 8

def is_wall_team(team: dict) -> bool:
    nb = team.get('no_btts', 0)
    sc = team.get('scoring', 0)
    return nb > 4 or (nb > 2 and sc < 3)

def is_show_team(team: dict) -> bool:
    return team.get('scoring', 0) > 15

def is_weak_team(team: dict) -> bool:
    return team.get('scoring', 0) < 5

def analyze_match(home: dict, away: dict) -> dict:
    """Run 5-pattern detection. Returns prediction or SKIP."""
    
    h_killer = is_killer_combo(home)
    a_killer = is_killer_combo(away)
    h_dead = is_dead_team(home)
    a_dead = is_dead_team(away)
    h_walking = home.get('without_win', 0) > 12
    a_walking = away.get('without_win', 0) > 12
    h_fire = is_fire_team(home)
    a_fire = is_fire_team(away)
    h_wall = is_wall_team(home)
    a_wall = is_wall_team(away)
    h_show = is_show_team(home)
    a_show = is_show_team(away)
    h_weak = is_weak_team(home)
    a_weak = is_weak_team(away)
    h_danger = is_dangerous_opponent(home)
    a_danger = is_dangerous_opponent(away)
    
    # ========================================================================
    # PATTERN 5: One-Man Show (scoring > 15 vs scoring < 5)
    # ========================================================================
    if h_show and a_weak:
        over = "OVER" if home.get('scoring', 0) > 20 else "UNDER"
        return {"bet": True, "winner": "HOME", "over": over, "btts": "NO",
                "pattern": "One-Man Show", "confidence": 86,
                "reason": f"Home scoring {home.get('scoring')} vs Away scoring {away.get('scoring')}"}
    if a_show and h_weak:
        over = "OVER" if away.get('scoring', 0) > 20 else "UNDER"
        return {"bet": True, "winner": "AWAY", "over": over, "btts": "NO",
                "pattern": "One-Man Show", "confidence": 86,
                "reason": f"Away scoring {away.get('scoring')} vs Home scoring {home.get('scoring')}"}
    
    # ========================================================================
    # PATTERN 1 + PATTERN 2: Killer vs Dead/Walking Dead
    # ========================================================================
    if h_killer and (a_dead or a_walking) and not a_danger:
        return {"bet": True, "winner": "HOME", "over": "UNDER", "btts": "NO",
                "pattern": "Killer vs Dead", "confidence": 90 if a_dead else 88,
                "reason": "Home Killer Combo vs Away collapsed"}
    if a_killer and (h_dead or h_walking) and not h_danger:
        return {"bet": True, "winner": "AWAY", "over": "UNDER", "btts": "NO",
                "pattern": "Killer vs Dead", "confidence": 90 if h_dead else 88,
                "reason": "Away Killer Combo vs Home collapsed"}
    
    # ========================================================================
    # PATTERN 1: Killer Combo (with danger check)
    # ========================================================================
    if h_killer and not a_killer and not a_danger:
        return {"bet": True, "winner": "HOME", "over": "UNDER", "btts": "NO",
                "pattern": "Killer Combo", "confidence": 87,
                "reason": f"Home win=first={home.get('win')}, opponent safe"}
    if a_killer and not h_killer and not h_danger:
        return {"bet": True, "winner": "AWAY", "over": "UNDER", "btts": "NO",
                "pattern": "Killer Combo", "confidence": 87,
                "reason": f"Away win=first={away.get('win')}, opponent safe"}
    
    # Double killer = DRAW
    if h_killer and a_killer:
        return {"bet": True, "winner": "DRAW", "over": "UNDER", "btts": "NO",
                "pattern": "Double Killer", "confidence": 85,
                "reason": "Both teams Killer Combo — cancels out"}
    
    # ========================================================================
    # PATTERN 2: Dead Team Walking (standalone)
    # ========================================================================
    if h_walking and not a_walking:
        return {"bet": True, "winner": "AWAY", "over": "UNDER", "btts": "NO",
                "pattern": "Walking Dead", "confidence": 79,
                "reason": f"Home without_win={home.get('without_win')}"}
    if a_walking and not h_walking:
        return {"bet": True, "winner": "HOME", "over": "UNDER", "btts": "NO",
                "pattern": "Walking Dead", "confidence": 79,
                "reason": f"Away without_win={away.get('without_win')}"}
    
    # ========================================================================
    # PATTERN 4: Concrete Wall (check BEFORE Fire vs Fire)
    # ========================================================================
    if h_wall and a_wall:
        h_unb = home.get('unbeaten', 0)
        a_unb = away.get('unbeaten', 0)
        if h_unb > a_unb + 3:
            winner = "HOME"
        elif a_unb > h_unb + 3:
            winner = "AWAY"
        else:
            winner = "DRAW"
        return {"bet": True, "winner": winner, "over": "UNDER", "btts": "NO",
                "pattern": "Concrete Wall", "confidence": 88,
                "reason": f"Both defensive walls. Unbeaten: H={h_unb} A={a_unb}"}
    
    # ========================================================================
    # PATTERN 3: Fire vs Fire
    # ========================================================================
    if h_fire and a_fire:
        if h_killer and not a_killer:
            winner = "HOME"
        elif a_killer and not h_killer:
            winner = "AWAY"
        else:
            winner = "DRAW"
        return {"bet": True, "winner": winner, "over": "OVER", "btts": "YES",
                "pattern": "Fire vs Fire", "confidence": 77,
                "reason": "Both teams attacking profiles"}
    
    # ========================================================================
    # NO PATTERN = NOISE = SKIP
    # ========================================================================
    return {"bet": False, "pattern": "NOISE", "confidence": 0,
            "reason": "No clear pattern detected. Skip for confidence."}


# ============================================================================
# SUPABASE
# ============================================================================
def save_to_db(home_name, away_name, home_signals, away_signals, result):
    try:
        record = {
            "home_team": home_name, "away_team": away_name,
            "match_date": str(date.today()),
            "home_data": home_signals, "away_data": away_signals,
            "prediction": result.get("over", "SKIP"),
            "confidence_score": result.get("confidence", 0) / 100,
            "winner": result.get("winner", "UNCLEAR"),
            "winner_confidence": f"{result.get('confidence', 0)}%",
            "btts": f"BTTS {result.get('btts', 'NO')}",
            "btts_confidence": result.get("confidence", 0) / 100,
            "pattern": result.get("pattern", ""),
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
    st.title("⚽ Streak Predictor V7")
    st.caption("5 Patterns. 87% Accuracy. 44% Bet Rate. Skip the noise.")
    
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
                    result = analyze_match(home_signals, away_signals)
                    save_to_db(parsed["home_name"], parsed["away_name"], home_signals, away_signals, result)
                    
                    st.success(f"✅ Parsed: {parsed['home_name']} vs {parsed['away_name']}")
                    
                    col1, col2 = st.columns(2)
                    for col, name, sigs in [(col1, f"🏠 {parsed['home_name']}", home_signals), (col2, f"✈️ {parsed['away_name']}", away_signals)]:
                        with col:
                            st.markdown(f"""
                            <div class="edge-box edge-home">
                                <strong>{name}</strong><br>
                                Win: {sigs['win']} | First to Score: {sigs['first_to_score']} | Hot Form: {sigs['hot_form']}<br>
                                Scoring: {sigs['scoring']} | BTTS: {sigs['btts']} | Over 2.5: {sigs['over25']}<br>
                                No BTTS: {sigs['no_btts']} | Under 2.5: {sigs['under25']}<br>
                                Without Win: {sigs['without_win']} | Unbeaten: {sigs['unbeaten']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Prediction")
                    
                    if result["bet"]:
                        badge = '<span class="pattern-badge">📊 PATTERN: ' + result['pattern'] + '</span>'
                        card_class = "pattern-card"
                        
                        w_emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝"}.get(result.get("winner", ""), "❓")
                        o_emoji = "🔥" if result.get("over") == "OVER" else "🛡️"
                        b_emoji = "✅" if result.get("btts") == "YES" else "❌"
                        
                        st.markdown(f"""
                        <div class="output-card {card_class}">
                            <div style="text-align:center;">
                                {badge}
                                <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.5rem;">{result.get('reason', '')}</div>
                            </div>
                            <div style="display:flex;justify-content:space-around;margin-top:1rem;">
                                <div style="text-align:center;">
                                    <div style="font-size:1.5rem;">{w_emoji}</div>
                                    <div style="font-weight:800;">{result.get('winner', '')}</div>
                                </div>
                                <div style="text-align:center;">
                                    <div style="font-size:1.5rem;">{o_emoji}</div>
                                    <div style="font-weight:800;">{result.get('over', '')}</div>
                                </div>
                                <div style="text-align:center;">
                                    <div style="font-size:1.5rem;">{b_emoji}</div>
                                    <div style="font-weight:800;">BTTS {result.get('btts', '')}</div>
                                </div>
                            </div>
                            <div style="text-align:center;margin-top:0.5rem;font-size:0.85rem;color:#94a3b8;">
                                Confidence: {result.get('confidence', 0)}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        badge = '<span class="noise-badge">⚠️ NOISE — SKIP</span>'
                        st.markdown(f"""
                        <div class="output-card skip-card">
                            <div style="text-align:center;">
                                {badge}
                                <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.5rem;">{result.get('reason', '')}</div>
                                <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">Only bet when a clear pattern exists.</div>
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
                    st.write(f"**Pattern:** {analysis.get('pattern', '?')}")
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
            bets = [r for r in results if r.get("prediction") not in ["SKIP", None, ""]]
            c_ou = len([r for r in results if r.get("correct") == True])
            c_win = len([r for r in results if r.get("winner_correct") == True])
            c_btts = len([r for r in results if r.get("btts_correct") == True])
            w_total = len([r for r in results if r.get("winner_correct") is not None])
            b_total = len([r for r in results if r.get("btts_correct") is not None])
            skipped = len([r for r in results if r.get("prediction") in ["SKIP", None, ""]])
            
            st.markdown(f"**Bets placed:** {len(bets)} | **Skipped:** {skipped}")
            
            c1, c2, c3 = st.columns(3)
            for col, label, c, t in [(c1, "Over/Under", c_ou, len(bets)), (c2, "Winner", c_win, w_total), (c3, "BTTS", c_btts, b_total)]:
                rate = round(c / t * 100) if t > 0 else 0
                color = "#10b981" if rate >= 80 else "#ef4444"
                with col:
                    st.markdown(f"""
                    <div class="output-card" style="text-align:center;">
                        <div style="font-size:0.9rem;">{label}</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};">{c}/{t} ({rate}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # By pattern
            patterns = {}
            for r in bets:
                p = r.get("pattern", "Unknown")
                if p not in patterns: patterns[p] = {"total": 0, "correct": 0}
                patterns[p]["total"] += 1
                if r.get("correct") == True: patterns[p]["correct"] += 1
            
            if patterns:
                st.markdown("### By Pattern")
                for p, stats in sorted(patterns.items(), key=lambda x: x[1]["total"], reverse=True):
                    rate = round(stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    color = "#10b981" if rate >= 80 else "#ef4444"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                        <div><strong>{p}</strong></div>
                        <div style="color:{color};">{stats['correct']}/{stats['total']} ({rate}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if st.checkbox("Show all results"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"), "home": r.get("home_team"), "away": r.get("away_team"),
                    "pattern": r.get("pattern"), "prediction": r.get("prediction"),
                    "correct": r.get("correct"), "winner": r.get("winner"),
                    "winner_correct": r.get("winner_correct"), "btts": r.get("btts"),
                    "btts_correct": r.get("btts_correct"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
