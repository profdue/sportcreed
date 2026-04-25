"""
STREAK PREDICTOR V3 - Complete 3-Layer System
Layer 1: Locks (100% record)
Layer 2: Volume Plays (77% record)
Layer 3: Universal Signals (79% record)
Supabase-Powered | No Contradictions | Built-in PASS Discipline
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
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .layer-1 { border-left: 5px solid #10b981; }
    .layer-2 { border-left: 5px solid #fbbf24; }
    .layer-3 { border-left: 5px solid #3b82f6; }
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
    market: str; bet: str; layer: int; tier: int; rule_name: str; reasoning: str

# ============================================================================
# ENGINE
# ============================================================================
def run_engine(home: TeamData, away: TeamData) -> Tuple[List[Bet], List[str], Dict]:
    # Derived metrics
    home_concede_drop = home.conceded_05 - home.conceded_15
    away_concede_drop = away.conceded_05 - away.conceded_15
    home_scored_drop = home.scored_05 - home.scored_15
    away_scored_drop = away.scored_05 - away.scored_15
    combined_over_15 = (home.over_15 + away.over_15) / 2
    combined_over_25 = (home.over_25 + away.over_25) / 2
    combined_over_35 = (home.over_35 + away.over_35) / 2
    combined_btts = (home.btts + away.btts) / 2
    leaky_defense = home.conceded_05 >= 70 or away.conceded_05 >= 70
    
    bets = []
    suppressed = set()
    
    # Suppression tracker
    l1_over15 = False; l1_btts = False; l1_home_cs = False; l1_away_u25 = False
    l1_over25 = False; l1_home_scores = False
    
    # ========================================================================
    # LAYER 1: LOCKS
    # ========================================================================
    
    # Lock 1: Over 1.5
    if away.conceded_05 >= 80 and away_concede_drop <= 30:
        l1_over15 = True
        bets.append(Bet("Over 1.5 Goals", "Over 1.5", 1, 1, 
                        "Over 1.5 - Collapse Defense",
                        f"Away defense leaks ({away.conceded_05:.0f}%) AND collapses (drop {away_concede_drop:.0f}%)."))
        suppressed.add("over15")
    
    # Lock 2: BTTS Yes
    if away_scored_drop >= 50 and home.conceded_05 >= 60:
        l1_btts = True
        bets.append(Bet("BTTS", "Yes", 1, 1,
                        "BTTS Yes - One-Goal Away Attack",
                        f"Away one-goal attack (drop {away_scored_drop:.0f}%). Home concedes {home.conceded_05:.0f}%."))
        suppressed.add("btts_yes")
        suppressed.add("btts_no")
    
    # Lock 3: Home CS No
    if home.conceded_05 >= 60 and away.scored_05 >= 70:
        l1_home_cs = True
        bets.append(Bet("Home Clean Sheet", "No", 1, 1,
                        "Home Clean Sheet No",
                        f"Home concedes {home.conceded_05:.0f}%. Away scores {away.scored_05:.0f}%."))
        suppressed.add("home_cs_no")
    
    # Lock 4: Away Under 2.5
    if home.conceded_25 >= 10 and away.scored_05 <= 70:
        l1_away_u25 = True
        bets.append(Bet("Away Team Total Under 2.5", "Under 2.5", 1, 1,
                        "Away Team Under 2.5",
                        f"Away scoring capped ({away.scored_05:.0f}%). Home rarely concedes 3+ ({home.conceded_25:.0f}%)."))
    
    # Lock 5: Over 2.5
    if home.btts_over25 >= 55 and combined_over_25 >= 50:
        l1_over25 = True
        bets.append(Bet("Over 2.5 Goals", "Over 2.5", 1, 2,
                        "Over 2.5 Goals",
                        f"Home BTTS & Over {home.btts_over25:.0f}%. Combined O2.5 {combined_over_25:.0f}%."))
        suppressed.add("over25")
        suppressed.add("under25")
    
    # Lock 6: Home Scores
    if home.scored_05 >= 70 and home.over_15 <= 60:
        l1_home_scores = True
        bets.append(Bet("Home Team Total O0.5", "Over 0.5", 1, 1,
                        "Home Team Scores",
                        f"Home scores {home.scored_05:.0f}%. Not explosive (O1.5 {home.over_15:.0f}%)."))
        suppressed.add("home_o05")
    
    # ========================================================================
    # LAYER 2: VOLUME PLAYS
    # ========================================================================
    
    # Play 7: Home Under 2.5
    if "home_u25" not in suppressed and home.scored_35 == 0 and away.conceded_25 < 25:
        bets.append(Bet("Home Team Total Under 2.5", "Under 2.5", 2, 2,
                        "Home Team Under 2.5",
                        f"Home never scores 4+. Away rarely concedes 3+ ({away.conceded_25:.0f}%)."))
    
    # Play 8: Away Under 2.5
    if "away_u25" not in suppressed and away.scored_35 == 0 and home.conceded_25 < 25:
        bets.append(Bet("Away Team Total Under 2.5", "Under 2.5", 2, 2,
                        "Away Team Under 2.5",
                        f"Away never scores 4+. Home rarely concedes 3+ ({home.conceded_25:.0f}%)."))
    
    # Play 9: Over 1.5
    if "over15" not in suppressed and combined_over_15 >= 75 and leaky_defense:
        bets.append(Bet("Over 1.5 Goals", "Over 1.5", 2, 2,
                        "Over 1.5 Goals",
                        f"Combined O1.5 {combined_over_15:.0f}%. At least one leaky defense."))
    
    # Play 10: Home O0.5
    if "home_o05" not in suppressed and home.fts_pct <= 25 and away.cs_pct < 35:
        bets.append(Bet("Home Team Total O0.5", "Over 0.5", 2, 2,
                        "Home Team O0.5",
                        f"Home FTS {home.fts_pct:.0f}%. Away CS {away.cs_pct:.0f}%."))
    
    # Play 11: Away O0.5
    if "away_o05" not in suppressed and away.fts_pct <= 25 and home.cs_pct < 35:
        bets.append(Bet("Away Team Total O0.5", "Over 0.5", 2, 2,
                        "Away Team O0.5",
                        f"Away FTS {away.fts_pct:.0f}%. Home CS {home.cs_pct:.0f}%."))
    
    # Play 12: Under 2.5
    if "under25" not in suppressed and combined_over_25 < 40 and home_concede_drop > 30 and away_concede_drop > 30 and home.conceded_05 <= 70 and away.conceded_05 <= 70:
        bets.append(Bet("Under 2.5 Goals", "Under 2.5", 2, 2,
                        "Under 2.5 Goals",
                        f"Low combined O2.5 ({combined_over_25:.0f}%). Both defenses bend, don't collapse."))
    
    # Play 13: Home CS No
    if "home_cs_no" not in suppressed and home.conceded_05 >= 75 and home_concede_drop <= 30:
        bets.append(Bet("Home Clean Sheet", "No", 2, 2,
                        "Home Clean Sheet No",
                        f"Home collapse defense. Concedes {home.conceded_05:.0f}%. Drop {home_concede_drop:.0f}%."))
    
    # Play 14: Away CS No
    if "away_cs_no" not in suppressed and away.conceded_05 >= 75 and away_concede_drop <= 30:
        bets.append(Bet("Away Clean Sheet", "No", 2, 2,
                        "Away Clean Sheet No",
                        f"Away collapse defense. Concedes {away.conceded_05:.0f}%. Drop {away_concede_drop:.0f}%."))
    
    # ========================================================================
    # LAYER 3: UNIVERSAL SIGNALS
    # ========================================================================
    
    # Signal 15: Home O0.5 Elite
    if "home_o05" not in suppressed and home.fts_pct <= 10 and away.cs_pct <= 40:
        bets.append(Bet("Home Team Total O0.5", "Over 0.5", 3, 2,
                        "Home Team O0.5 (Elite)",
                        f"Home almost never blanks (FTS {home.fts_pct:.0f}%). Away not elite defense."))
    
    # Signal 16: Away O0.5 Elite
    if "away_o05" not in suppressed and away.fts_pct <= 10 and home.cs_pct <= 40:
        bets.append(Bet("Away Team Total O0.5", "Over 0.5", 3, 2,
                        "Away Team O0.5 (Elite)",
                        f"Away almost never blanks (FTS {away.fts_pct:.0f}%). Home not elite defense."))
    
    # Signal 17: Home CS No Elite
    if "home_cs_no" not in suppressed and home.cs_pct <= 10 and away.scored_05 >= 50:
        bets.append(Bet("Home Clean Sheet", "No", 3, 2,
                        "Home Clean Sheet No (Elite)",
                        f"Home CS only {home.cs_pct:.0f}%. Away can score ({away.scored_05:.0f}%)."))
    
    # Signal 18: Away CS No Elite
    if "away_cs_no" not in suppressed and away.cs_pct <= 10 and home.scored_05 >= 50:
        bets.append(Bet("Away Clean Sheet", "No", 3, 2,
                        "Away Clean Sheet No (Elite)",
                        f"Away CS only {away.cs_pct:.0f}%. Home can score ({home.scored_05:.0f}%)."))
    
    # Signal 19: Under 3.5
    if "under35" not in suppressed and home.scored_35 == 0 and away.scored_35 == 0 and combined_over_35 < 30:
        bets.append(Bet("Under 3.5 Goals", "Under 3.5", 3, 2,
                        "Under 3.5 Goals",
                        f"Both never score 4+. Combined O3.5 only {combined_over_35:.0f}%."))
    
    # ========================================================================
    # AVOID BETS
    # ========================================================================
    avoid = []
    if l1_over25 or (combined_over_25 >= 55 and "under25" not in suppressed):
        avoid.append("Under 2.5 Goals (Over signals dominate)")
    if l1_over15:
        avoid.append("Under 1.5 Goals (Lock 1 fired)")
    if l1_btts:
        avoid.append("BTTS No (BTTS Yes lock fired)")
    
    # Profile
    profile = {
        "home_attack": f"{home.scored_05:.0f}% score, {home.scored_15:.0f}% O1.5",
        "home_defense": f"{home.conceded_05:.0f}% concede, drop {home_concede_drop:.0f}%",
        "away_attack": f"{away.scored_05:.0f}% score, {away.scored_15:.0f}% O1.5",
        "away_defense": f"{away.conceded_05:.0f}% concede, drop {away_concede_drop:.0f}%",
        "combined_o15": f"{combined_over_15:.0f}%",
        "combined_o25": f"{combined_over_25:.0f}%",
        "combined_o35": f"{combined_over_35:.0f}%",
        "combined_btts": f"{combined_btts:.0f}%",
    }
    
    return bets, avoid, profile


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
    st.caption("3-Layer System | 80.8% Win Rate | Zero Contradictions")
    
    tab1, tab2 = st.tabs(["🔮 Analyze", "📊 Records"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1: home_name = st.text_input("🏠 Home Team", "Monza", key="home_name")
        with c2: away_name = st.text_input("✈️ Away Team", "Modena", key="away_name")
        
        league = st.text_input("🏆 League", "Serie B", key="league")
        
        st.divider()
        st.subheader(f"🏠 {home_name}")
        home_data = team_input(home_name, "home")
        
        st.divider()
        st.subheader(f"✈️ {away_name}")
        away_data = team_input(away_name, "away")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            bets, avoid, profile = run_engine(home_data, away_data)
            
            # Layer 1
            l1_bets = [b for b in bets if b.layer == 1]
            if l1_bets:
                st.markdown("### 🔒 LAYER 1 - LOCKS (100% Record)")
                for bet in l1_bets:
                    st.markdown(f"""
                    <div class="output-card layer-1">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <strong>{bet.market}</strong> → {bet.bet}
                                <div style="font-size:0.8rem;color:#94a3b8;">{bet.rule_name}</div>
                            </div>
                            <span class="record-badge">TIER {bet.tier}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Layer 2
            l2_bets = [b for b in bets if b.layer == 2]
            if l2_bets:
                st.markdown("### 📊 LAYER 2 - VOLUME PLAYS (77% Record)")
                for bet in l2_bets:
                    st.markdown(f"""
                    <div class="output-card layer-2">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <strong>{bet.market}</strong> → {bet.bet}
                                <div style="font-size:0.8rem;color:#94a3b8;">{bet.rule_name}</div>
                            </div>
                            <span class="record-badge">TIER {bet.tier}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Layer 3
            l3_bets = [b for b in bets if b.layer == 3]
            if l3_bets:
                st.markdown("### 🌍 LAYER 3 - UNIVERSAL SIGNALS (79% Record)")
                for bet in l3_bets:
                    st.markdown(f"""
                    <div class="output-card layer-3">
                        <div style="display:flex;justify-content:space-between;">
                            <div>
                                <strong>{bet.market}</strong> → {bet.bet}
                                <div style="font-size:0.8rem;color:#94a3b8;">{bet.rule_name}</div>
                            </div>
                            <span class="record-badge">TIER {bet.tier}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Avoid
            if avoid:
                st.markdown("### ⛔ AVOID")
                for a in avoid:
                    st.markdown(f'<div class="avoid-card">🚫 {a}</div>', unsafe_allow_html=True)
            
            # Profile
            if profile:
                st.markdown(f"""
                <div class="info-note">
                <strong>Match Profile:</strong><br>
                Home: {profile['home_attack']} | {profile['home_defense']}<br>
                Away: {profile['away_attack']} | {profile['away_defense']}<br>
                Combined: O1.5={profile['combined_o15']} | O2.5={profile['combined_o25']} | O3.5={profile['combined_o35']} | BTTS={profile['combined_btts']}
                </div>
                """, unsafe_allow_html=True)
            
            if not bets:
                st.info("🎯 No conditions met. PASS on this match.")
    
    with tab2:
        st.subheader("📊 Live Prediction Records")
        try:
            records = supabase.table("predictions_v3").select("*").order("layer,tier").execute()
            if records.data:
                for r in records.data:
                    fired = r['total_fired']
                    won = r['total_won']
                    wr = (won/fired*100) if fired > 0 else 0
                    color = "#10b981" if wr >= 85 else "#fbbf24" if wr >= 70 else "#f97316"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                        <div><strong>L{r['layer']}: {r['name']}</strong></div>
                        <div style="color:{color};">{won}/{fired} ({wr:.0f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
        except:
            st.info("Run the Supabase SQL setup first to see records.")
    
    st.divider()
    st.markdown("""
    ### 📋 System Architecture
    
    | Layer | Name | Bets | Record | Purpose |
    |-------|------|------|--------|---------|
    | 1 | Locks | 14 | 100% | Maximum confidence, rare |
    | 2 | Volume Plays | 52 | 77% | Consistent volume |
    | 3 | Universal Signals | 38 | 79% | Catch extreme signals |
    | **Total** | | **104** | **80.8%** | |
    """)

if __name__ == "__main__":
    main()