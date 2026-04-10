# grokbet_no_draw_final.py
# GROKBET – NO DRAW FILTER (FINAL)
# 
# Predicts when a match is unlikely to end in a draw.
# 
# Rules (if ANY is true → No Draw):
# RULE 1: Home team's recent home win % > 50%
# RULE 2: Away team's recent away win % > 33%
# RULE 3: H2H home wins in last 5 matches ≥ 3
# 
# OVERRIDE: If league position gap is 2 or 3 → DRAW (override No Draw)
# 
# Otherwise → Draw likely

import streamlit as st

st.set_page_config(
    page_title="GrokBet - No Draw Filter",
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
    
    .result-lock {
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

HOME_WIN_PCT_THRESHOLD = 50
AWAY_WIN_PCT_THRESHOLD = 33
H2H_HOME_WINS_THRESHOLD = 3

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - No Draw Filter</h1>
        <p>LOCK when match will have a winner | SKIP when draw likely</p>
        <div class="badge">STAKE: 1.0% ON LOCK</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">🏠 HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.text_input("Team Name", "Lorient", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="section-header">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.text_input("Team Name", "Paris FC", label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # League Positions
        st.markdown('<div class="section-header">🏆 LEAGUE POSITIONS</div>', unsafe_allow_html=True)
        col_pos1, col_pos2 = st.columns(2)
        with col_pos1:
            home_position = st.number_input(f"{home_team} Position", 1, 20, 10, key="home_position")
        with col_pos2:
            away_position = st.number_input(f"{away_team} Position", 1, 20, 13, key="away_position")
        
        position_gap = abs(home_position - away_position)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Recent Home Form
        st.markdown('<div class="section-header">📊 HOME FORM (Last 6 Home Matches)</div>', unsafe_allow_html=True)
        col3, col4, col5 = st.columns(3)
        with col3:
            home_wins = st.number_input("Wins", 0, 6, 4, key="home_wins")
        with col4:
            home_draws = st.number_input("Draws", 0, 6, 2, key="home_draws")
        with col5:
            home_losses = st.number_input("Losses", 0, 6, 0, key="home_losses")
        
        home_win_pct = (home_wins / 6) * 100
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Recent Away Form
        st.markdown('<div class="section-header">📊 AWAY FORM (Last 6 Away Matches)</div>', unsafe_allow_html=True)
        col6, col7, col8 = st.columns(3)
        with col6:
            away_wins = st.number_input("Wins", 0, 6, 1, key="away_wins")
        with col7:
            away_draws = st.number_input("Draws", 0, 6, 4, key="away_draws")
        with col8:
            away_losses = st.number_input("Losses", 0, 6, 1, key="away_losses")
        
        away_win_pct = (away_wins / 6) * 100
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # H2H
        st.markdown('<div class="section-header">🤝 HEAD TO HEAD (Last 5 Matches)</div>', unsafe_allow_html=True)
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home_wins = st.number_input(f"{home_team} Wins", 0, 5, 0, key="h2h_home_wins")
        with col10:
            h2h_draws = st.number_input("Draws", 0, 5, 0, key="h2h_draws")
        with col11:
            h2h_away_wins = st.number_input(f"{away_team} Wins", 0, 5, 1, key="h2h_away_wins")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            # Check rules
            rule1 = home_win_pct > HOME_WIN_PCT_THRESHOLD
            rule2 = away_win_pct > AWAY_WIN_PCT_THRESHOLD
            rule3 = h2h_home_wins >= H2H_HOME_WINS_THRESHOLD
            
            no_draw_original = rule1 or rule2 or rule3
            
            # Position gap override
            position_gap = abs(home_position - away_position)
            override_draw = (position_gap == 2 or position_gap == 3)
            
            # Determine reason
            if rule1:
                reason = f"{home_team} home win % {home_win_pct:.0f}% > 50%"
            elif rule2:
                reason = f"{away_team} away win % {away_win_pct:.0f}% > 33%"
            elif rule3:
                reason = f"H2H {home_team} wins {h2h_home_wins} ≥ 3"
            else:
                reason = "No strong signals for a winner"
            
            # Final prediction
            if override_draw and no_draw_original:
                # Override: No Draw signal but position gap says Draw
                st.markdown(f"""
                <div class="result-skip">
                    <strong>⚠️ SKIP</strong><br><br>
                    🎯 No bet<br>
                    📝 Reason: Position gap {position_gap} overrides No Draw signal
                </div>
                """, unsafe_allow_html=True)
            elif no_draw_original:
                # LOCK: No Draw
                st.markdown(f"""
                <div class="result-lock">
                    <strong>🔒 LOCK</strong><br><br>
                    🎯 Bet: No Draw (Home or Away win)<br>
                    📊 Stake: <span class="stake-highlight">1.0%</span><br>
                    📝 Reason: {reason}
                </div>
                """, unsafe_allow_html=True)
            else:
                # SKIP: Draw likely
                st.markdown(f"""
                <div class="result-skip">
                    <strong>⚠️ SKIP</strong><br><br>
                    🎯 No bet<br>
                    📝 Reason: {reason}
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🎯 GrokBet - No Draw Filter | LOCK when winner likely | SKIP when draw likely
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
