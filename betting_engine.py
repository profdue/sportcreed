"""
STREAK PREDICTOR V3 - Data-Discovered Rules
11 Rules | All Markets | Supabase Tracking | Post-Match Input
"""

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import date
from supabase import create_client, Client
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
st.set_page_config(page_title="Streak Predictor V3", page_icon="⚽", layout="centered")

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; border-left: 5px solid #10b981; }
    .avoid-card { background: #1e1e1e; border-radius: 12px; padding: 0.75rem; margin: 0.4rem 0; border-left: 4px solid #64748b; color: #94a3b8; font-size: 0.9rem; }
    .team-header { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0; color: #ffffff; }
    .team-name { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .metric-label { color: #0f172a; font-weight: 700; font-size: 0.85rem; margin-top: 0.5rem; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .record-badge { background: #0f172a; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.8rem; color: #10b981; font-weight: 700; }
    .info-note { background: #1a3a5f; border-left: 4px solid #3b82f6; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamData:
    name: str
    scored_05: float; scored_15: float; scored_25: float; scored_35: float
    conceded_05: float; conceded_15: float; conceded_25: float; conceded_35: float
    btts: float; over_15: float; over_25: float; over_35: float
    btts_over25: float; btts_no_over25: float
    fts_pct: float; cs_pct: float; xg: float; actual_scored: float

@dataclass
class Bet:
    market: str; bet: str; rule_name: str; record: str; reasoning: str

# ============================================================================
# ENGINE - 11 Data-Discovered Rules
# ============================================================================
def run_engine(home: TeamData, away: TeamData) -> Tuple[List[Bet], Dict]:
    # Derived
    home_concede_drop = home.conceded_05 - home.conceded_15
    away_concede_drop = away.conceded_05 - away.conceded_15
    home_scored_drop = home.scored_05 - home.scored_15
    away_scored_drop = away.scored_05 - away.scored_15
    combined_over_15 = (home.over_15 + away.over_15) / 2
    combined_over_25 = (home.over_25 + away.over_25) / 2
    combined_over_35 = (home.over_35 + away.over_35) / 2
    combined_btts = (home.btts + away.btts) / 2
    
    bets = []
    
    # Rule 1: Over 1.5 (13-0)
    if home.btts_no_over25 < 12 and combined_over_25 > 45.5:
        bets.append(Bet("Over 1.5 Goals", "Over 1.5", "Over 1.5", "13-0",
            f"Home BTTS No & Over 2.5 is {home.btts_no_over25:.0f}%. Combined Over 2.5 is {combined_over_25:.0f}%."))
    
    # Rule 2: Over 2.5 (5-0)
    if away.scored_05 > 84 and away_scored_drop > 50:
        bets.append(Bet("Over 2.5 Goals", "Over 2.5", "Over 2.5", "5-0",
            f"Away scores in {away.scored_05:.0f}% of games with {away_scored_drop:.0f}% drop. Explosive attack."))
    
    # Rule 3: Under 2.5 (5-0)
    if home.conceded_35 > 0 and home.btts_no_over25 > 11:
        bets.append(Bet("Under 2.5 Goals", "Under 2.5", "Under 2.5", "5-0",
            f"Home occasionally concedes 4+ ({home.conceded_35:.0f}%) but BTTS No & Over profile ({home.btts_no_over25:.0f}%) caps totals."))
    
    # Rule 4: Under 3.5 (11-0)
    if home.over_35 < 25 and home.btts_no_over25 > 3:
        bets.append(Bet("Under 3.5 Goals", "Under 3.5", "Under 3.5", "11-0",
            f"Home Over 3.5 is {home.over_35:.0f}%. BTTS No & Over is {home.btts_no_over25:.0f}%. Low ceiling."))
    
    # Rule 5: BTTS Yes (7-0)
    if home.btts_over25 > 45 and away_scored_drop > 43:
        bets.append(Bet("BTTS", "Yes", "BTTS Yes", "7-0",
            f"Home BTTS & Over is {home.btts_over25:.0f}%. Away scored drop is {away_scored_drop:.0f}%. Both score."))
    
    # Rule 6: Home Scores (11-0)
    if away.cs_pct < 32 and away_scored_drop > 24 and away.btts > 61:
        bets.append(Bet("Home Team Total O0.5", "Over 0.5", "Home Scores", "11-0",
            f"Away CS {away.cs_pct:.0f}%, scored drop {away_scored_drop:.0f}%, BTTS {away.btts:.0f}%. Home scores."))
    
    # Rule 7: Away Scores (10-0)
    if home.conceded_05 > 67 and home.cs_pct < 33 and home.btts_over25 > 45:
        bets.append(Bet("Away Team Total O0.5", "Over 0.5", "Away Scores", "10-0",
            f"Home concedes {home.conceded_05:.0f}%, CS {home.cs_pct:.0f}%. Away scores."))
    
    # Rule 8: Home CS No (12-0)
    if home.conceded_15 > 7 and away.scored_15 < 47 and away.scored_05 < 71:
        bets.append(Bet("Home Clean Sheet", "No", "Home CS No", "12-0",
            f"Home concedes 2+ in {home.conceded_15:.0f}%. Away can score."))
    
    # Rule 9: Away CS No (11-0)
    if away.cs_pct < 32 and away_scored_drop > 24 and away.btts > 61:
        bets.append(Bet("Away Clean Sheet", "No", "Away CS No", "11-0",
            f"Away CS {away.cs_pct:.0f}%. Home should score."))
    
    # Rule 10: Home Under 2.5 (11-0)
    if home.btts < 64 and away.scored_05 > 59 and away.scored_15 > 40:
        bets.append(Bet("Home Team Total Under 2.5", "Under 2.5", "Home Under 2.5", "11-0",
            f"Home BTTS {home.btts:.0f}%. Away scores freely but home attack capped."))
    
    # Rule 11: Away Under 2.5 (14-0)
    if away.scored_25 < 18 and home.conceded_25 > 9 and home.conceded_15 > 28:
        bets.append(Bet("Away Team Total Under 2.5", "Under 2.5", "Away Under 2.5", "14-0",
            f"Away rarely scores 3+ ({away.scored_25:.0f}%). Home defense limits."))
    
    profile = {
        "combined_o15": f"{combined_over_15:.0f}%",
        "combined_o25": f"{combined_over_25:.0f}%",
        "combined_o35": f"{combined_over_35:.0f}%",
        "combined_btts": f"{combined_btts:.0f}%",
        "home_scored_drop": f"{home_scored_drop:.0f}%",
        "away_scored_drop": f"{away_scored_drop:.0f}%",
        "home_concede_drop": f"{home_concede_drop:.0f}%",
        "away_concede_drop": f"{away_concede_drop:.0f}%",
    }
    
    return bets, profile


# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_match_to_db(home_data, away_data, league, match_date, bets):
    try:
        match_record = {
            "home_team": home_data.name, "away_team": away_data.name,
            "league": league, "match_date": str(match_date),
            "home_scored_05": home_data.scored_05, "home_scored_15": home_data.scored_15,
            "home_scored_25": home_data.scored_25, "home_scored_35": home_data.scored_35,
            "home_conceded_05": home_data.conceded_05, "home_conceded_15": home_data.conceded_15,
            "home_conceded_25": home_data.conceded_25, "home_conceded_35": home_data.conceded_35,
            "home_btts": home_data.btts, "home_over_15": home_data.over_15,
            "home_over_25": home_data.over_25, "home_over_35": home_data.over_35,
            "home_btts_over25": home_data.btts_over25, "home_btts_no_over25": home_data.btts_no_over25,
            "home_fts": home_data.fts_pct, "home_cs": home_data.cs_pct,
            "home_xg": home_data.xg, "home_actual_scored": home_data.actual_scored,
            "away_scored_05": away_data.scored_05, "away_scored_15": away_data.scored_15,
            "away_scored_25": away_data.scored_25, "away_scored_35": away_data.scored_35,
            "away_conceded_05": away_data.conceded_05, "away_conceded_15": away_data.conceded_15,
            "away_conceded_25": away_data.conceded_25, "away_conceded_35": away_data.conceded_35,
            "away_btts": away_data.btts, "away_over_15": away_data.over_15,
            "away_over_25": away_data.over_25, "away_over_35": away_data.over_35,
            "away_btts_over25": away_data.btts_over25, "away_btts_no_over25": away_data.btts_no_over25,
            "away_fts": away_data.fts_pct, "away_cs": away_data.cs_pct,
            "away_xg": away_data.xg, "away_actual_scored": away_data.actual_scored,
            "bets_fired": json.dumps([{"market": b.market, "bet": b.bet, "rule": b.rule_name, "record": b.record} for b in bets]),
        }
        response = supabase.table("matches").insert(match_record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending_matches():
    try:
        response = supabase.table("matches").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []

def submit_result(match_id, home_score, away_score):
    try:
        match = supabase.table("matches").select("*").eq("id", match_id).single().execute()
        match_data = match.data
        
        bets_fired = match_data.get("bets_fired")
        if isinstance(bets_fired, str):
            bets_fired = json.loads(bets_fired)
        
        if bets_fired:
            for bet in bets_fired:
                won = evaluate_bet(bet["market"], bet["bet"], home_score, away_score, match_data)
                
                supabase.table("prediction_results_v3").insert({
                    "match_id": match_id,
                    "prediction_id": bet["rule"].replace(" ", "_").replace(".", ""),
                    "bet_market": bet["market"],
                    "bet_selection": bet["bet"],
                    "tier": 1,
                    "won": won,
                    "actual_result": f"{home_score}-{away_score}"
                }).execute()
        
        supabase.table("matches").update({
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "result_entered": True
        }).eq("id", match_id).execute()
        
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False

def evaluate_bet(market, bet, home_score, away_score, match_data):
    total = home_score + away_score
    home_team = match_data.get("home_team", "")
    away_team = match_data.get("away_team", "")
    
    if "Over 1.5" in market and "Over 2.5" not in market and "Team Total" not in market:
        return total > 1
    if "Over 2.5" in market and "Team Total" not in market:
        return total > 2
    if "Under 2.5" in market and "Team Total" not in market:
        return total < 3
    if "Under 3.5" in market:
        return total < 4
    if "BTTS" in market and "Yes" in bet:
        return home_score > 0 and away_score > 0
    if "Clean Sheet" in market and "No" in bet:
        if home_team and home_team in market:
            return away_score > 0
        else:
            return home_score > 0
    if "Team Total O0.5" in market:
        if home_team and home_team in market:
            return home_score > 0
        else:
            return away_score > 0
    if "Team Total Under 2.5" in market:
        if home_team and home_team in market:
            return home_score <= 2
        else:
            return away_score <= 2
    return False


# ============================================================================
# UI INPUT
# ============================================================================
def team_input(team_name: str, prefix: str) -> TeamData:
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">⚽ Scored Per Game</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: s05 = st.number_input("Over 0.5 %", 0, 100, 80, 5, key=f"{prefix}_s05")
    with c2: s15 = st.number_input("Over 1.5 %", 0, 100, 38, 5, key=f"{prefix}_s15")
    with c3: s25 = st.number_input("Over 2.5 %", 0, 100, 15, 5, key=f"{prefix}_s25")
    with c4: s35 = st.number_input("Over 3.5 %", 0, 100, 0, 5, key=f"{prefix}_s35")
    
    st.markdown('<p class="metric-label">🛡️ Conceded / Game</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: c05 = st.number_input("Over 0.5 %", 0, 100, 77, 5, key=f"{prefix}_c05")
    with c2: c15 = st.number_input("Over 1.5 %", 0, 100, 40, 5, key=f"{prefix}_c15")
    with c3: c25 = st.number_input("Over 2.5 %", 0, 100, 10, 5, key=f"{prefix}_c25")
    with c4: c35 = st.number_input("Over 3.5 %", 0, 100, 0, 5, key=f"{prefix}_c35")
    
    st.markdown('<p class="metric-label">📊 Match Goals & Core</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: btts = st.number_input("BTTS %", 0, 100, 54, 5, key=f"{prefix}_btts")
    with c2: o15 = st.number_input("Over 1.5 %", 0, 100, 77, 5, key=f"{prefix}_o15")
    with c3: o25 = st.number_input("Over 2.5 %", 0, 100, 46, 5, key=f"{prefix}_o25")
    with c4: o35 = st.number_input("Over 3.5 %", 0, 100, 27, 5, key=f"{prefix}_o35")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: btts_o25 = st.number_input("BTTS & O2.5 %", 0, 100, 46, 5, key=f"{prefix}_btts_o25")
    with c2: btts_no = st.number_input("BTTS No & O2.5 %", 0, 100, 0, 5, key=f"{prefix}_btts_no")
    with c3: fts = st.number_input("Failed to Score %", 0, 100, 23, 5, key=f"{prefix}_fts")
    with c4: cs = st.number_input("Clean Sheet %", 0, 100, 23, 5, key=f"{prefix}_cs")
    
    st.markdown('<p class="metric-label">🎯 xG Context</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: xg = st.number_input("xG per game", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_xg")
    with c2: actual = st.number_input("Actual Scored", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_act")
    
    return TeamData(
        name=team_name, scored_05=float(s05), scored_15=float(s15),
        scored_25=float(s25), scored_35=float(s35), conceded_05=float(c05),
        conceded_15=float(c15), conceded_25=float(c25), conceded_35=float(c35),
        btts=float(btts), over_15=float(o15), over_25=float(o25), over_35=float(o35),
        btts_over25=float(btts_o25), btts_no_over25=float(btts_no),
        fts_pct=float(fts), cs_pct=float(cs), xg=float(xg), actual_scored=float(actual)
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("11 Data-Discovered Rules | Supabase Tracking | Live Records")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1: home_name = st.text_input("🏠 Home Team", key="home_name")
        with c2: away_name = st.text_input("✈️ Away Team", key="away_name")
        
        league = st.text_input("🏆 League", key="league")
        match_date = st.date_input("📅 Match Date", date.today())
        
        st.divider()
        st.subheader(f"🏠 {home_name}")
        home_data = team_input(home_name, "home")
        
        st.divider()
        st.subheader(f"✈️ {away_name}")
        away_data = team_input(away_name, "away")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            bets, profile = run_engine(home_data, away_data)
            
            match_id = save_match_to_db(home_data, away_data, league, match_date, bets)
            if match_id:
                st.success("✅ Analysis saved")
            
            if bets:
                st.markdown("### 🎯 PREDICTIONS")
                for bet in bets:
                    st.markdown(f"""
                    <div class="output-card">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <strong>{bet.market}</strong> → {bet.bet}
                                <div style="font-size:0.8rem;color:#94a3b8;">{bet.rule_name}</div>
                            </div>
                            <span class="record-badge">{bet.record}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No conditions met. PASS.")
            
            if profile:
                st.markdown(f"""
                <div class="info-note">
                <strong>Match Profile:</strong><br>
                Combined: O1.5={profile['combined_o15']} | O2.5={profile['combined_o25']} | O3.5={profile['combined_o35']} | BTTS={profile['combined_btts']}<br>
                Drops: Home Scored={profile['home_scored_drop']} | Away Scored={profile['away_scored_drop']} | Home Concede={profile['home_concede_drop']} | Away Concede={profile['away_concede_drop']}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_matches()
        if pending:
            match_options = {}
            for m in pending:
                label = f"{m['home_team']} vs {m['away_team']} ({m.get('match_date', '')})"
                match_options[label] = m['id']
            
            selected = st.selectbox("Select Match", list(match_options.keys()))
            
            c1, c2 = st.columns(2)
            with c1: home_score = st.number_input("Home Score", 0, 20, 0)
            with c2: away_score = st.number_input("Away Score", 0, 20, 0)
            
            if st.button("✅ Submit Result"):
                if submit_result(match_options[selected], home_score, away_score):
                    st.success("Result submitted!")
                    st.rerun()
        else:
            st.info("No pending matches.")
    
    with tab3:
        st.subheader("📊 Live Records")
        st.info("Records will appear after submitting match results through the Post-Match tab.")

if __name__ == "__main__":
    main()
