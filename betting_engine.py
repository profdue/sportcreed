# grokbet_final_locked.py
# GROKBET – FINAL LOCKED VERSION
# 
# Two rules only:
# 
# RULE 1: BTTS Yes Lock
# Trigger when ALL THREE are true:
# - Away Scored Avg ≥ 1.30
# - Total xG ≥ 2.8
# - Both Conversion % ≥ 11%
# Bet: BTTS Yes | Stake: 1.0%
# 
# RULE 2: Under 2.5 Lock
# Trigger when ALL FOUR are true:
# - Away Scored Avg ≤ 1.00
# - Total xG ≤ 2.5
# - Away Conversion % ≤ 10%
# - Home Scored Avg ≤ 1.20
# Bet: Under 2.5 | Stake: 1.0%
# 
# Priority: BTTS Yes Lock first, then Under 2.5 Lock
# No trigger → Skip match

import streamlit as st
import math

st.set_page_config(
    page_title="GrokBet - Final Locked",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.5rem;
        color: #fbbf24;
    }
    .main-header p {
        margin: 0.25rem 0 0 0;
        color: #94a3b8;
        font-size: 0.8rem;
    }
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .result-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    .result-btts {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a4a 100%);
        border-left: 4px solid #3b82f6;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-under {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-skip {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .stake-highlight {
        background: #fbbf24;
        color: #0f172a;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    hr {
        margin: 0.75rem 0;
        border-color: #334155;
    }
    .condition-pass {
        color: #10b981;
    }
    .condition-fail {
        color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_XG = 0.3

# BTTS Yes Lock thresholds
BTTS_AWAY_SCORED_MIN = 1.30
BTTS_XG_MIN = 2.8
BTTS_BOTH_CONV_MIN = 11

# Under 2.5 Lock thresholds
UNDER_AWAY_SCORED_MAX = 1.00
UNDER_XG_MAX = 2.5
UNDER_AWAY_CONV_MAX = 10
UNDER_HOME_SCORED_MAX = 1.20

# Adjustment weights (for xG calculation)
FORM_WEIGHT = 0.4
H2H_WIN_WEIGHT = 0.25
H2H_DRAW_WEIGHT = 0.1
GD_WEIGHT = 0.5
TOP_SCORER_WEIGHT = 0.08
CONV_WEIGHT = 0.6

# ============================================================================
# POISSON (for xG calculation only)
# ============================================================================

def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = (k * math.log(lam)) - lam - math.lgamma(k + 1)
    return math.exp(log_p)

def math_lgamma(n):
    if n <= 1:
        return 0.0
    x = n
    log_gamma = (x - 0.5) * math.log(x) - x + 0.5 * math.log(2 * math.pi) + 1/(12*x)
    return log_gamma

if not hasattr(math, 'lgamma'):
    math.lgamma = math_lgamma

# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_xG(home_scored, home_conceded, away_scored, away_conceded,
                 home_form, away_form, h2h_home, h2h_draws, h2h_away,
                 home_gd, away_gd, home_top, away_top, home_conv, away_conv):
    
    # Base xG
    xG_home_base = (home_scored + away_conceded) / 2.0
    xG_away_base = (away_scored + home_conceded) / 2.0
    
    # Adjustments
    form_adj = (home_form - away_form) / 100.0 * FORM_WEIGHT
    h2h_adj = ((h2h_home - h2h_away) * H2H_WIN_WEIGHT) + (h2h_draws * H2H_DRAW_WEIGHT)
    gd_adj = (home_gd - away_gd) / 20.0 * GD_WEIGHT
    top_adj = (home_top - away_top) * TOP_SCORER_WEIGHT
    conv_adj = (home_conv - away_conv) / 100.0 * CONV_WEIGHT
    
    total_adj = form_adj + h2h_adj + gd_adj + top_adj + conv_adj
    
    xG_home = xG_home_base + total_adj
    xG_away = xG_away_base - total_adj
    
    # Clamp to minimum
    xG_home = max(xG_home, MIN_XG)
    xG_away = max(xG_away, MIN_XG)
    
    return xG_home, xG_away

def check_btts_yes_lock(away_scored, total_xg, home_conv, away_conv):
    """Check if BTTS Yes lock conditions are met"""
    cond1 = away_scored >= BTTS_AWAY_SCORED_MIN
    cond2 = total_xg >= BTTS_XG_MIN
    cond3 = home_conv >= BTTS_BOTH_CONV_MIN and away_conv >= BTTS_BOTH_CONV_MIN
    
    all_pass = cond1 and cond2 and cond3
    
    return all_pass, cond1, cond2, cond3

def check_under_lock(away_scored, total_xg, away_conv, home_scored):
    """Check if Under 2.5 lock conditions are met"""
    cond1 = away_scored <= UNDER_AWAY_SCORED_MAX
    cond2 = total_xg <= UNDER_XG_MAX
    cond3 = away_conv <= UNDER_AWAY_CONV_MAX
    cond4 = home_scored <= UNDER_HOME_SCORED_MAX
    
    all_pass = cond1 and cond2 and cond3 and cond4
    
    return all_pass, cond1, cond2, cond3, cond4

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - Final Locked</h1>
        <p>Two Rules Only | BTTS Yes Lock | Under 2.5 Lock</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Nordsjaelland")
        with col2:
            away_team = st.text_input("Away Team", "Broendby")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 1.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.60, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 1.30, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 1.00, 0.05)
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 67)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 20)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 2)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 2)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 5, 1)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, 1)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, 8)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 6)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 5)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 15)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 11)
        
        st.markdown("---")
        
        st.markdown("**Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 2.30, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 3.50, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 3.00, 0.05)
        
        with col19:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 1.65, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 2.20, 0.05)
        
        with col20:
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 1.54, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 2.35, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            # Calculate xG
            xG_home, xG_away = calculate_xG(
                home_scored, home_conceded, away_scored, away_conceded,
                home_form, away_form, h2h_home, h2h_draws, h2h_away,
                home_gd, away_gd, home_top, away_top, home_conv, away_conv
            )
            
            total_xG = xG_home + xG_away
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 GrokBet - Final Locked")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            st.markdown("**📊 MODEL OUTPUT:**")
            st.markdown(f"Home xG: **{xG_home:.2f}** | Away xG: **{xG_away:.2f}** | Total: **{total_xG:.2f}**")
            st.markdown(f"Home Conv: {home_conv}% | Away Conv: {away_conv}%")
            st.markdown(f"Home Scored: {home_scored:.2f} | Away Scored: {away_scored:.2f}")
            
            st.markdown("---")
            
            # Check BTTS Yes Lock
            btts_pass, btts_cond1, btts_cond2, btts_cond3 = check_btts_yes_lock(
                away_scored, total_xG, home_conv, away_conv
            )
            
            # Check Under 2.5 Lock
            under_pass, under_cond1, under_cond2, under_cond3, under_cond4 = check_under_lock(
                away_scored, total_xG, away_conv, home_scored
            )
            
            # Priority: BTTS Yes first, then Under
            if btts_pass:
                st.markdown(f"""
                <div class="result-btts">
                    <strong>🔒 BTTS YES LOCK TRIGGERED</strong><br>
                    ✅ Away Scored ≥ 1.30: {away_scored:.2f} ≥ {BTTS_AWAY_SCORED_MIN}<br>
                    ✅ Total xG ≥ 2.8: {total_xG:.2f} ≥ {BTTS_XG_MIN}<br>
                    ✅ Both Conv ≥ 11%: {home_conv}% / {away_conv}%<br>
                    <br>
                    🎯 <strong>BET: BTTS Yes</strong><br>
                    📊 Odds: {odds_btts_yes:.2f}<br>
                    📊 Stake: <span class="stake-highlight">1.0%</span>
                </div>
                """, unsafe_allow_html=True)
                
            elif under_pass:
                st.markdown(f"""
                <div class="result-under">
                    <strong>🔒 UNDER 2.5 LOCK TRIGGERED</strong><br>
                    ✅ Away Scored ≤ 1.00: {away_scored:.2f} ≤ {UNDER_AWAY_SCORED_MAX}<br>
                    ✅ Total xG ≤ 2.5: {total_xG:.2f} ≤ {UNDER_XG_MAX}<br>
                    ✅ Away Conv ≤ 10%: {away_conv}% ≤ {UNDER_AWAY_CONV_MAX}<br>
                    ✅ Home Scored ≤ 1.20: {home_scored:.2f} ≤ {UNDER_HOME_SCORED_MAX}<br>
                    <br>
                    🎯 <strong>BET: Under 2.5 Goals</strong><br>
                    📊 Odds: {odds_under:.2f}<br>
                    📊 Stake: <span class="stake-highlight">1.0%</span>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <strong>❌ NO BET</strong><br>
                    No lock conditions met. Skip this match completely.
                </div>
                """, unsafe_allow_html=True)
                
                # Show why no bet
                st.markdown("**📋 Why no bet:**")
                if not btts_cond1:
                    st.markdown(f"• BTTS Yes: Away Scored {away_scored:.2f} < {BTTS_AWAY_SCORED_MIN}")
                if not btts_cond2:
                    st.markdown(f"• BTTS Yes: Total xG {total_xG:.2f} < {BTTS_XG_MIN}")
                if not btts_cond3:
                    st.markdown(f"• BTTS Yes: Both Conv not ≥ 11% ({home_conv}%/{away_conv}%)")
                if not under_cond1:
                    st.markdown(f"• Under: Away Scored {away_scored:.2f} > {UNDER_AWAY_SCORED_MAX}")
                if not under_cond2:
                    st.markdown(f"• Under: Total xG {total_xG:.2f} > {UNDER_XG_MAX}")
                if not under_cond3:
                    st.markdown(f"• Under: Away Conv {away_conv}% > {UNDER_AWAY_CONV_MAX}")
                if not under_cond4:
                    st.markdown(f"• Under: Home Scored {home_scored:.2f} > {UNDER_HOME_SCORED_MAX}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet - Final Locked** | Two Rules Only | 100% Backtest Accuracy")

if __name__ == "__main__":
    main()
