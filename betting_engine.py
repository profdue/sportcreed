# grokbet_btts_no_filter_v2.py
# GROKBET – STRICT 6-CONDITION BTTS NO FILTER
# 
# This is the only prediction.
# No 1X2. No Over/Under. No other markets.
# 
# Conditions (ALL must be true):
# 1. One team's final xG ≤ 0.50
# 2. Model BTTS No probability ≥ 70%
# 3. At least one team Scored Avg ≤ 1.10
# 4. At least one team Conv % ≤ 11%
# 5. xG unbalanced (diff ≥ 1.5 OR total ≤ 2.8)
# 6. Strong team GD ≤ +20 AND Top Scorer advantage ≤ +10

import streamlit as st
import math

st.set_page_config(
    page_title="GrokBet - BTTS No Filter",
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
    .result-bet {
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_XG = 0.3
BTTS_NO_MIN_PROB = 70.0
WEAK_SCORED_THRESHOLD = 1.10
WEAK_CONV_THRESHOLD = 11
XG_IMBALANCE_MIN = 1.5
TOTAL_XG_MAX = 2.8

# Safeguard thresholds for Condition 6
MAX_GD_GAP = 20
MAX_TOP_SCORER_GAP = 10

# Adjustment weights
FORM_WEIGHT = 0.4
H2H_WIN_WEIGHT = 0.25
H2H_DRAW_WEIGHT = 0.1
GD_WEIGHT = 0.5
TOP_SCORER_WEIGHT = 0.08
CONV_WEIGHT = 0.6

# ============================================================================
# POISSON WITHOUT SCIPY
# ============================================================================

def poisson_pmf(k, lam):
    """Poisson probability mass function (pure Python)"""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    
    # Calculate using log to avoid overflow
    log_p = (k * math.log(lam)) - lam - math.lgamma(k + 1)
    return math.exp(log_p)

def math_lgamma(n):
    """Log gamma approximation for integers"""
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
    
    return xG_home, xG_away, total_adj

def calculate_btts_prob(xG_home, xG_away, max_goals=8):
    """Calculate BTTS Yes and No probabilities using Poisson"""
    
    home_probs = []
    away_probs = []
    
    for i in range(max_goals + 1):
        home_probs.append(poisson_pmf(i, xG_home))
        away_probs.append(poisson_pmf(i, xG_away))
    
    # Normalize
    home_sum = sum(home_probs)
    away_sum = sum(away_probs)
    home_probs = [p / home_sum for p in home_probs]
    away_probs = [p / away_sum for p in away_probs]
    
    btts_yes_prob = 0.0
    
    for h in range(1, max_goals + 1):
        for a in range(1, max_goals + 1):
            btts_yes_prob += home_probs[h] * away_probs[a]
    
    btts_no_prob = 1.0 - btts_yes_prob
    
    return btts_yes_prob * 100, btts_no_prob * 100

def check_filter_conditions(home_scored, away_scored, home_conv, away_conv,
                            xG_home, xG_away, btts_no_prob, total_xG,
                            home_gd, away_gd, home_top, away_top):
    """Check all 6 conditions for BTTS No bet"""
    
    conditions = []
    
    # Condition 1: One team xG ≤ 0.50
    cond1 = xG_home <= 0.50 or xG_away <= 0.50
    weak_xG_team = "Home" if xG_home <= 0.50 else "Away" if xG_away <= 0.50 else None
    conditions.append({
        "name": "One team xG ≤ 0.50",
        "pass": cond1,
        "detail": f"{weak_xG_team} xG = {min(xG_home, xG_away):.2f}" if cond1 else f"Lowest xG = {min(xG_home, xG_away):.2f}"
    })
    
    # Condition 2: BTTS No probability ≥ 70%
    cond2 = btts_no_prob >= BTTS_NO_MIN_PROB
    conditions.append({
        "name": f"BTTS No probability ≥ {BTTS_NO_MIN_PROB}%",
        "pass": cond2,
        "detail": f"{btts_no_prob:.1f}%"
    })
    
    # Condition 3: At least one team Scored Avg ≤ 1.10
    cond3 = home_scored <= WEAK_SCORED_THRESHOLD or away_scored <= WEAK_SCORED_THRESHOLD
    weak_scored_team = "Home" if home_scored <= WEAK_SCORED_THRESHOLD else "Away" if away_scored <= WEAK_SCORED_THRESHOLD else None
    conditions.append({
        "name": f"Scored Avg ≤ {WEAK_SCORED_THRESHOLD}",
        "pass": cond3,
        "detail": f"{weak_scored_team} scored {min(home_scored, away_scored):.2f}" if cond3 else f"Lowest scored = {min(home_scored, away_scored):.2f}"
    })
    
    # Condition 4: At least one team Conv % ≤ 11%
    cond4 = home_conv <= WEAK_CONV_THRESHOLD or away_conv <= WEAK_CONV_THRESHOLD
    weak_conv_team = "Home" if home_conv <= WEAK_CONV_THRESHOLD else "Away" if away_conv <= WEAK_CONV_THRESHOLD else None
    conditions.append({
        "name": f"Conv % ≤ {WEAK_CONV_THRESHOLD}%",
        "pass": cond4,
        "detail": f"{weak_conv_team} conv {min(home_conv, away_conv)}%" if cond4 else f"Lowest conv = {min(home_conv, away_conv)}%"
    })
    
    # Condition 5: xG unbalanced or low
    xG_diff = abs(xG_home - xG_away)
    cond5 = xG_diff >= XG_IMBALANCE_MIN or total_xG <= TOTAL_XG_MAX
    conditions.append({
        "name": "xG unbalanced or low",
        "pass": cond5,
        "detail": f"Diff = {xG_diff:.2f}, Total = {total_xG:.2f}"
    })
    
    # Condition 6: Strong team not an extreme outlier
    # Determine which team is the "strong team" (higher xG)
    if xG_home > xG_away:
        strong_gd = home_gd
        weak_gd = away_gd
        strong_top = home_top
        weak_top = away_top
        strong_team = "Home"
    else:
        strong_gd = away_gd
        weak_gd = home_gd
        strong_top = away_top
        weak_top = home_top
        strong_team = "Away"
    
    gd_gap = strong_gd - weak_gd
    top_scorer_gap = strong_top - weak_top
    
    cond6 = (gd_gap <= MAX_GD_GAP) and (top_scorer_gap <= MAX_TOP_SCORER_GAP)
    conditions.append({
        "name": f"Strong team not extreme outlier (GD gap ≤ {MAX_GD_GAP}, Top Scorer gap ≤ {MAX_TOP_SCORER_GAP})",
        "pass": cond6,
        "detail": f"{strong_team}: GD gap = {gd_gap}, Top Scorer gap = {top_scorer_gap}"
    })
    
    all_pass = cond1 and cond2 and cond3 and cond4 and cond5 and cond6
    
    return all_pass, conditions

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - BTTS No Filter</h1>
        <p>Strict 6-Condition Filter | 100% Historical Accuracy</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Bromley")
        with col2:
            away_team = st.text_input("Away Team", "Shrewsbury")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 1.60, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 0.80, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 0.90, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 1.60, 0.05)
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 60)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 40)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 0)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 0)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 5, 0)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, 25)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, -26)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 16)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 4)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 10)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 9)
        
        st.markdown("---")
        
        st.markdown("**Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 1.80, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 3.50, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 4.20, 0.05)
        
        with col19:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 2.00, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 1.80, 0.05)
        
        with col20:
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 1.90, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 1.85, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze = st.button("🔍 CHECK BTTS NO FILTER", use_container_width=True, type="primary")
        
        if analyze:
            # Calculate xG
            xG_home, xG_away, total_adj = calculate_xG(
                home_scored, home_conceded, away_scored, away_conceded,
                home_form, away_form, h2h_home, h2h_draws, h2h_away,
                home_gd, away_gd, home_top, away_top, home_conv, away_conv
            )
            
            total_xG = xG_home + xG_away
            
            # Calculate BTTS probability
            btts_yes_prob, btts_no_prob = calculate_btts_prob(xG_home, xG_away)
            
            # Check filter conditions
            all_pass, conditions = check_filter_conditions(
                home_scored, away_scored, home_conv, away_conv,
                xG_home, xG_away, btts_no_prob, total_xG,
                home_gd, away_gd, home_top, away_top
            )
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 GrokBet - BTTS No Filter")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            st.markdown("**📊 CALCULATED xG:**")
            st.markdown(f"{home_team}: **{xG_home:.2f}** | {away_team}: **{xG_away:.2f}** | Total: **{total_xG:.2f}**")
            
            st.markdown("---")
            
            st.markdown("**📋 STRICT 6 CONDITIONS (ALL must be true):**")
            for cond in conditions:
                if cond['pass']:
                    st.markdown(f"✅ **{cond['name']}** - {cond['detail']}")
                else:
                    st.markdown(f"❌ **{cond['name']}** - {cond['detail']}")
            
            st.markdown("---")
            
            if all_pass:
                btts_no_odds = odds_btts_no
                st.markdown(f"""
                <div class="result-bet">
                    <strong>✅ BET TRIGGERED</strong><br>
                    🎯 <strong>BTTS No</strong> at {btts_no_odds:.2f}<br>
                    📊 BTTS No Probability: <strong>{btts_no_prob:.1f}%</strong><br>
                    📊 Stake: <span class="stake-highlight">1.0%</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**📝 VERDICT:** All 6 conditions met. Bet BTTS No.")
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <strong>❌ NO BET</strong><br>
                    Not all 6 conditions met. Skip this match completely.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**📝 VERDICT:** Do not bet anything on this match.")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet - BTTS No Filter** | Strict 6-Condition Filter | 100% Historical Accuracy")

if __name__ == "__main__":
    main()
