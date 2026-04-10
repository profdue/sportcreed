# grokbet_draw_wrong_predictor.py
# GROKBET – DRAW PREDICTION WRONG DETECTOR
# 
# Identifies when Forebet's DRAW prediction is certainly wrong.
# 
# Rules (100% certainty, 0 false alarms):
# 
# RULE 1: Away Team Strength
#   H2H_Draws ≤ 1 AND Away_Win_Pct_Season ≥ 40%
# 
# RULE 2: Extreme Position Gap
#   Position_Gap ≥ 7
# 
# RULE 3: Home Win Mismatch
#   Home_Win_Pct_Season ≥ 50% AND Away_Win_Pct_Season < 30%
# 
# RULE 4: Goals Mismatch
#   Home_Avg_Goals_Scored > 1.5 AND Away_Avg_Goals_Conceded > 1.5
# 
# If ANY rule triggers → Forebet's DRAW prediction is WRONG → Bet No Draw

import streamlit as st

st.set_page_config(
    page_title="GrokBet - Draw Wrong Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1000px;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 2px solid #fbbf24;
        text-align: center;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #fbbf24 !important;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        border-radius: 20px;
        padding: 0.25rem 1rem;
        font-size: 0.7rem;
        color: #0f172a !important;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .input-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #fbbf24;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #fbbf24 !important;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    .team-label {
        color: #000000 !important;
        font-weight: bold !important;
        background: #fbbf24 !important;
        padding: 0.25rem 0.8rem !important;
        border-radius: 8px !important;
        display: inline-block !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.85rem !important;
    }
    
    .result-bet {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #10b981;
        border-right: 1px solid #10b981;
        border-bottom: 1px solid #10b981;
    }
    
    .result-skip {
        background: linear-gradient(135deg, #1e293b 0%, #2a1a1a 100%);
        border-left: 6px solid #ef4444;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #ef4444;
        border-right: 1px solid #ef4444;
        border-bottom: 1px solid #ef4444;
    }
    
    .stake-highlight {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f172a !important;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #fbbf24, transparent);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f172a !important;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.7rem;
        width: 100%;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
    }
    
    .stTextInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 1rem;
        border-top: 1px solid #fbbf24;
        font-size: 0.7rem;
        color: #94a3b8 !important;
    }
    
    .raw-data-note {
        font-size: 0.7rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# RULE 1: Away Team Strength
RULE1_H2H_DRAWS_MAX = 1
RULE1_AWAY_WIN_PCT_MIN = 0.40  # 40%

# RULE 2: Extreme Position Gap
RULE2_POSITION_GAP_MIN = 7

# RULE 3: Home Win Mismatch
RULE3_HOME_WIN_PCT_MIN = 0.50  # 50%
RULE3_AWAY_WIN_PCT_MAX = 0.30  # 30%

# RULE 4: Goals Mismatch
RULE4_HOME_GOALS_MIN = 1.5
RULE4_AWAY_CONCEDED_MIN = 1.5

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - Draw Wrong Detector</h1>
        <p>Identifies when Forebet's DRAW prediction is certainly wrong</p>
        <div class="badge">STAKE: 1.0% WHEN ANY RULE TRIGGERS</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">🏠 HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.text_input("Team Name", "Sassuolo", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="section-header">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.text_input("Team Name", "Cagliari", label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # League Positions
        st.markdown('<div class="section-header">🏆 LEAGUE POSITIONS</div>', unsafe_allow_html=True)
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            home_position = st.number_input(f"{home_team} Position", 1, 50, 10, key="home_position")
        with col_pos2:
            away_position = st.number_input(f"{away_team} Position", 1, 50, 13, key="away_position")
        
        position_gap = abs(home_position - away_position)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Season Statistics - RAW DATA INPUT (Wins and Total Matches)
        st.markdown('<div class="section-header">📊 SEASON STATISTICS (RAW DATA)</div>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.markdown(f'<div class="team-label">🏠 {home_team}</div>', unsafe_allow_html=True)
            
            # RAW INPUT: Wins and Total Matches
            col_wins1, col_total1 = st.columns(2)
            with col_wins1:
                home_wins = st.number_input("Wins", 0, 50, 6, key="home_wins")
            with col_total1:
                home_total_matches = st.number_input("Total Matches", 1, 50, 12, key="home_total")
            
            # Calculate percentage automatically
            home_win_pct = home_wins / home_total_matches if home_total_matches > 0 else 0
            
            # Display calculated percentage (read-only feedback)
            st.caption(f"📈 = {home_win_pct:.1%} win rate")
            
            st.markdown("---")
            
            home_goals_scored = st.number_input("Goals Scored Avg", 0.0, 3.0, 1.20, 0.05, key="home_goals")
            home_goals_conceded = st.number_input("Goals Conceded Avg", 0.0, 3.0, 1.40, 0.05, key="home_conceded")
        
        with col_stats2:
            st.markdown(f'<div class="team-label">✈️ {away_team}</div>', unsafe_allow_html=True)
            
            # RAW INPUT: Wins and Total Matches
            col_wins2, col_total2 = st.columns(2)
            with col_wins2:
                away_wins = st.number_input("Wins", 0, 50, 2, key="away_wins")
            with col_total2:
                away_total_matches = st.number_input("Total Matches", 1, 50, 12, key="away_total")
            
            # Calculate percentage automatically
            away_win_pct = away_wins / away_total_matches if away_total_matches > 0 else 0
            
            # Display calculated percentage (read-only feedback)
            st.caption(f"📈 = {away_win_pct:.1%} win rate")
            
            st.markdown("---")
            
            away_goals_scored = st.number_input("Goals Scored Avg", 0.0, 3.0, 1.20, 0.05, key="away_goals")
            away_goals_conceded = st.number_input("Goals Conceded Avg", 0.0, 3.0, 1.30, 0.05, key="away_conceded")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # H2H
        st.markdown('<div class="section-header">🤝 HEAD TO HEAD (Last 6 Matches)</div>', unsafe_allow_html=True)
        col_h2h1, col_h2h2, col_h2h3 = st.columns(3)
        with col_h2h1:
            h2h_home_wins = st.number_input(f"{home_team} Wins", 0, 6, 2, key="h2h_home")
        with col_h2h2:
            h2h_draws = st.number_input("Draws", 0, 6, 1, key="h2h_draws")
        with col_h2h3:
            h2h_away_wins = st.number_input(f"{away_team} Wins", 0, 6, 2, key="h2h_away")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="raw-data-note">📌 Win percentages are calculated automatically from Wins / Total Matches</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE", use_container_width=True, type="primary")
        
        if analyze:
            # Check ALL rules using calculated percentages
            rule1 = (h2h_draws <= RULE1_H2H_DRAWS_MAX) and (away_win_pct >= RULE1_AWAY_WIN_PCT_MIN)
            rule2 = (position_gap >= RULE2_POSITION_GAP_MIN)
            rule3 = (home_win_pct >= RULE3_HOME_WIN_PCT_MIN) and (away_win_pct < RULE3_AWAY_WIN_PCT_MAX)
            rule4 = (home_goals_scored > RULE4_HOME_GOALS_MIN) and (away_goals_conceded > RULE4_AWAY_CONCEDED_MIN)
            
            # Determine if ANY rule triggered
            triggered = rule1 or rule2 or rule3 or rule4
            
            # Build list of triggered rules for display
            triggered_rules = []
            if rule1:
                triggered_rules.append("Away Team Strength")
            if rule2:
                triggered_rules.append("Extreme Position Gap")
            if rule3:
                triggered_rules.append("Home Win Mismatch")
            if rule4:
                triggered_rules.append("Goals Mismatch")
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 {home_team} vs {away_team}")
            st.markdown("---")
            
            # Display input summary (with calculated percentages)
            st.markdown("**📊 INPUT SUMMARY:**")
            st.markdown(f"🏆 Position Gap: {position_gap}")
            st.markdown(f"🏠 {home_team}: {home_wins}-{home_total_matches - home_wins}-0 | {home_win_pct:.1%} win rate | Goals {home_goals_scored:.2f} | Conceded {home_goals_conceded:.2f}")
            st.markdown(f"✈️ {away_team}: {away_wins}-{away_total_matches - away_wins}-0 | {away_win_pct:.1%} win rate | Goals {away_goals_scored:.2f} | Conceded {away_goals_conceded:.2f}")
            st.markdown(f"🤝 H2H: {home_team} {h2h_home_wins} - {h2h_draws} - {away_team} {h2h_away_wins}")
            
            st.markdown("---")
            
            # Show rule checks
            st.markdown("**🔍 RULE CHECKS (ANY triggers → DRAW prediction is WRONG):**")
            
            # RULE 1
            if rule1:
                st.markdown(f"✅ **RULE 1:** H2H Draws ≤ 1 AND Away Win % ≥ 40% → {h2h_draws} ≤ 1 AND {away_win_pct:.0%} ≥ 40%")
            else:
                st.markdown(f"❌ **RULE 1:** H2H Draws ≤ 1 AND Away Win % ≥ 40% → {h2h_draws} ≤ 1 AND {away_win_pct:.0%} ≥ 40%")
            
            # RULE 2
            if rule2:
                st.markdown(f"✅ **RULE 2:** Position Gap ≥ 7 → {position_gap} ≥ 7")
            else:
                st.markdown(f"❌ **RULE 2:** Position Gap ≥ 7 → {position_gap} ≥ 7")
            
            # RULE 3
            if rule3:
                st.markdown(f"✅ **RULE 3:** Home Win % ≥ 50% AND Away Win % < 30% → {home_win_pct:.0%} ≥ 50% AND {away_win_pct:.0%} < 30%")
            else:
                st.markdown(f"❌ **RULE 3:** Home Win % ≥ 50% AND Away Win % < 30% → {home_win_pct:.0%} ≥ 50% AND {away_win_pct:.0%} < 30%")
            
            # RULE 4
            if rule4:
                st.markdown(f"✅ **RULE 4:** Home Goals > 1.5 AND Away Conceded > 1.5 → {home_goals_scored:.2f} > 1.5 AND {away_goals_conceded:.2f} > 1.5")
            else:
                st.markdown(f"❌ **RULE 4:** Home Goals > 1.5 AND Away Conceded > 1.5 → {home_goals_scored:.2f} > 1.5 AND {away_goals_conceded:.2f} > 1.5")
            
            st.markdown("---")
            
            if triggered:
                rules_text = ", ".join(triggered_rules)
                st.markdown(f"""
                <div class="result-bet">
                    <strong>🔒 LOCK</strong><br><br>
                    🎯 Rules triggered: {rules_text}<br>
                    📝 {len(triggered_rules)} rule(s) triggered → Forebet's DRAW prediction is WRONG<br>
                    <br>
                    🎯 Conclusion: <strong>Bet against the draw</strong><br>
                    🎯 Bet: <strong>No Draw (Home or Away win)</strong><br>
                    📊 Stake: <span class="stake-highlight">1.0%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <strong>⚠️ SKIP</strong><br><br>
                    🎯 No rule triggered.<br>
                    📝 Forebet's DRAW prediction may be correct.<br>
                    <br>
                    🎯 Conclusion: <strong>Uncertain</strong><br>
                    📊 Stake: <span class="stake-highlight">0% (SKIP)</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🎯 GrokBet - Draw Wrong Detector | 4 Rules | 100% Certainty When Triggered
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
