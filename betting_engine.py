# grokbet_ultra.py
# GROKBET ULTRA – ONE/TWO-THING-ONLY SYSTEM
# 
# Two rules only:
# 
# RULE 1: Strict BTTS No Filter
# ALL 5 conditions must be true:
# - One team pre-match xG ≤ 0.50
# - That same team scored avg ≤ 0.9
# - That same team conv % ≤ 10%
# - Total xG ≤ 2.6
# - At least one team defensive strength (conceded ≤ 1.0 OR form ≥ 60%)
# 
# RULE 2: Strict Under 2.5 Filter
# ALL 3 conditions must be true:
# - Total xG ≤ 2.4
# - Both teams conv % ≤ 12%
# - At least one team defensive strength (conceded ≤ 0.9 AND form ≥ 60%)
# 
# Priority: Rule 1 > Rule 2
# No BTTS Yes. No Over 2.5. No H2H. No Efficiency Gap. No Top Scorer.

import streamlit as st
import math

st.set_page_config(
    page_title="GrokBet Ultra",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* LOGIC DISPLAY CARD - FORCE EVERYTHING INSIDE TO BE WHITE */
    .logic-card, 
    .logic-card *,
    .logic-card .stat-card,
    .logic-card .stat-card *,
    .logic-card .stat-label,
    .logic-card .stat-value,
    .logic-card .stat-value-critical,
    .logic-card .logic-title {
        color: #ffffff !important;
    }
    
    /* Keep gold color for specific elements in display card */
    .logic-card .stat-value-critical {
        color: #fbbf24 !important;
    }
    
    .logic-card .logic-title {
        color: #fbbf24 !important;
    }
    
    .logic-card .stat-label {
        color: #fbbf24 !important;
    }
    
    /* Result boxes - ALL TEXT WHITE */
    .result-box, 
    .result-box *,
    .result-bet,
    .result-bet *,
    .result-skip,
    .result-skip *,
    .result-rule1,
    .result-rule1 *,
    .result-rule2,
    .result-rule2 *,
    .rule-check-status,
    .rule-check-status * {
        color: #ffffff !important;
    }
    
    /* Keep stake highlight with dark text */
    .stake-highlight {
        color: #0f172a !important;
    }
    
    /* Input labels - BLACK with gold background */
    .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #000000 !important;
        font-weight: bold !important;
        font-size: 0.85rem !important;
        background: #fbbf24 !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 6px !important;
        display: inline-block !important;
        margin-bottom: 0.3rem !important;
    }
    
    /* Input values stay gold */
    .stNumberInput input, .stTextInput input {
        color: #fbbf24 !important;
        font-weight: bold !important;
        background: #0f172a !important;
        border-color: #fbbf24 !important;
    }
    
    /* Team name labels - black with gold background */
    .team-label {
        color: #000000 !important;
        font-weight: bold !important;
        background: #fbbf24 !important;
        padding: 0.3rem 0.8rem !important;
        border-radius: 8px !important;
        display: inline-block !important;
        margin-bottom: 0.5rem !important;
        font-size: 0.9rem !important;
    }
    
    /* Animated gradient header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 2px solid #fbbf24;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
        box-shadow: 0 0 20px rgba(251, 191, 36, 0.2);
    }
    
    /* Header text */
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #fbbf24 !important;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.75rem;
        color: #0f172a !important;
        font-weight: bold;
        margin-top: 0.5rem;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.3);
    }
    
    /* Input cards */
    .input-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #fbbf24;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .input-card:hover {
        border-color: #f59e0b;
        box-shadow: 0 4px 20px rgba(251, 191, 36, 0.2);
        transform: translateY(-2px);
    }
    
    /* Section headers - gold text */
    .section-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #fbbf24 !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: bold;
    }
    
    .section-header::before {
        content: "▸";
        color: #fbbf24;
        font-size: 1.2rem;
    }
    
    /* Result boxes */
    .result-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 20px;
        padding: 1.5rem;
        border: 2px solid #fbbf24;
        margin-top: 1.5rem;
        animation: slideUp 0.4s ease-out;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-rule1 {
        background: linear-gradient(135deg, #1e293b 0%, #1a2a3a 100%);
        border-left: 6px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        transition: transform 0.2s ease;
        border-top: 1px solid #3b82f6;
        border-right: 1px solid #3b82f6;
        border-bottom: 1px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
    }
    
    .result-rule2 {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #10b981;
        border-right: 1px solid #10b981;
        border-bottom: 1px solid #10b981;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
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
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
    }
    
    /* Stake highlight */
    .stake-highlight {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f172a !important;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.4);
    }
    
    /* LOGIC DISPLAY CARD */
    .logic-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 2px solid #fbbf24;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.2);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid #fbbf24;
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: #f59e0b;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.2);
    }
    
    /* Rule indicator */
    .rule-indicator {
        font-size: 0.85rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        background: rgba(251, 191, 36, 0.2);
        display: inline-block;
        margin-right: 0.5rem;
        color: #fbbf24 !important;
        font-weight: bold;
    }
    
    /* Rule check status container */
    .rule-check-status {
        background: #0f172a !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
        border-left: 4px solid #fbbf24 !important;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #fbbf24, #f59e0b, transparent);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        color: #0f172a !important;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
        transition: all 0.3s ease;
        font-size: 1rem;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(251, 191, 36, 0.5);
    }
    
    /* Number inputs - keep gold values */
    .stNumberInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #f59e0b;
        box-shadow: 0 0 8px rgba(251, 191, 36, 0.3);
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #f59e0b;
        box-shadow: 0 0 8px rgba(251, 191, 36, 0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 1rem;
        border-top: 2px solid #fbbf24;
        font-size: 0.75rem;
        color: #000000 !important;
        font-weight: bold;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_XG = 0.3

# Rule 1: Strict BTTS No Filter thresholds
RULE1_WEAK_XG_MAX = 0.50
RULE1_WEAK_SCORED_MAX = 0.9
RULE1_WEAK_CONV_MAX = 10
RULE1_TOTAL_XG_MAX = 2.6
RULE1_DEF_CONCEDED_MAX = 1.0
RULE1_DEF_FORM_MIN = 60

# Rule 2: Strict Under 2.5 Filter thresholds
RULE2_TOTAL_XG_MAX = 2.4
RULE2_BOTH_CONV_MAX = 12
RULE2_DEF_CONCEDED_MAX = 0.9
RULE2_DEF_FORM_MIN = 60

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

def check_rule1(home_xg, away_xg, home_scored, away_scored, 
                home_conv, away_conv, total_xg, 
                home_conceded, away_conceded, home_form, away_form,
                home_team, away_team):
    """
    Rule 1: Strict BTTS No Filter
    ALL 5 conditions must be true
    """
    # Find the weak team (xG ≤ 0.50)
    weak_team = None
    weak_xg = None
    weak_scored = None
    weak_conv = None
    
    if home_xg <= RULE1_WEAK_XG_MAX:
        weak_team = home_team
        weak_xg = home_xg
        weak_scored = home_scored
        weak_conv = home_conv
    elif away_xg <= RULE1_WEAK_XG_MAX:
        weak_team = away_team
        weak_xg = away_xg
        weak_scored = away_scored
        weak_conv = away_conv
    
    if weak_team is None:
        return False, None, []
    
    # Condition 1 & 2 & 3 already captured by weak team selection
    cond1 = weak_xg <= RULE1_WEAK_XG_MAX
    cond2 = weak_scored <= RULE1_WEAK_SCORED_MAX
    cond3 = weak_conv <= RULE1_WEAK_CONV_MAX
    cond4 = total_xg <= RULE1_TOTAL_XG_MAX
    
    # Condition 5: At least one team defensive strength
    home_def = home_conceded <= RULE1_DEF_CONCEDED_MAX or home_form >= RULE1_DEF_FORM_MIN
    away_def = away_conceded <= RULE1_DEF_CONCEDED_MAX or away_form >= RULE1_DEF_FORM_MIN
    cond5 = home_def or away_def
    
    all_pass = cond1 and cond2 and cond3 and cond4 and cond5
    
    details = {
        "weak_team": weak_team,
        "weak_xg": weak_xg,
        "weak_scored": weak_scored,
        "weak_conv": weak_conv,
        "cond1": cond1,
        "cond2": cond2,
        "cond3": cond3,
        "cond4": cond4,
        "cond5": cond5,
        "home_def": home_def,
        "away_def": away_def
    }
    
    return all_pass, "BTTS No", details

def check_rule2(total_xg, home_conv, away_conv, 
                home_conceded, away_conceded, home_form, away_form):
    """
    Rule 2: Strict Under 2.5 Filter
    ALL 3 conditions must be true
    """
    cond1 = total_xg <= RULE2_TOTAL_XG_MAX
    cond2 = home_conv <= RULE2_BOTH_CONV_MAX and away_conv <= RULE2_BOTH_CONV_MAX
    
    # Condition 3: At least one team defensive strength (conceded ≤ 0.9 AND form ≥ 60%)
    home_def = home_conceded <= RULE2_DEF_CONCEDED_MAX and home_form >= RULE2_DEF_FORM_MIN
    away_def = away_conceded <= RULE2_DEF_CONCEDED_MAX and away_form >= RULE2_DEF_FORM_MIN
    cond3 = home_def or away_def
    
    all_pass = cond1 and cond2 and cond3
    
    details = {
        "cond1": cond1,
        "cond2": cond2,
        "cond3": cond3,
        "home_def": home_def,
        "away_def": away_def
    }
    
    return all_pass, "Under 2.5", details

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Ultra</h1>
        <p>One/Two-Thing-Only System | BTTS No + Under 2.5</p>
        <div class="badge">⚠️ STAKE: 1.0% WHEN TRIGGERED | NO BTTS YES | NO OVER 2.5</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names row
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="section-header">🏠 HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.text_input("Team Name", "Port Vale", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="section-header">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.text_input("Team Name", "Rotherham", label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Stats input - Organized in groups
        st.markdown('<div class="section-header">📊 TEAM STATISTICS</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f'<div class="team-label">🏠 {home_team}</div>', unsafe_allow_html=True)
            home_scored = st.number_input("⚽ Goals Scored", 0.0, 3.0, 0.80, 0.05, key="home_scored")
            home_conceded = st.number_input("🛡️ Goals Conceded", 0.0, 3.0, 1.40, 0.05, key="home_conceded")
            home_form = st.number_input("📈 Form %", 0, 100, 20, key="home_form")
            home_gd = st.number_input("➕ Goal Difference", -50, 50, -25, key="home_gd")
            home_top = st.number_input("🎯 Top Scorer Goals", 0, 30, 3, key="home_top")
            home_conv = st.number_input("🎯 Conversion %", 0, 100, 5, key="home_conv")
        with col4:
            st.markdown(f'<div class="team-label">✈️ {away_team}</div>', unsafe_allow_html=True)
            away_scored = st.number_input("⚽ Goals Scored", 0.0, 3.0, 0.90, 0.05, key="away_scored")
            away_conceded = st.number_input("🛡️ Goals Conceded", 0.0, 3.0, 1.50, 0.05, key="away_conceded")
            away_form = st.number_input("📈 Form %", 0, 100, 13, key="away_form")
            away_gd = st.number_input("➕ Goal Difference", -50, 50, -23, key="away_gd")
            away_top = st.number_input("🎯 Top Scorer Goals", 0, 30, 9, key="away_top")
            away_conv = st.number_input("🎯 Conversion %", 0, 100, 6, key="away_conv")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # H2H section (for xG calculation only)
        st.markdown('<div class="section-header">🤝 HEAD TO HEAD (Last 5 Matches)</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            h2h_home = st.number_input(f"{home_team} Wins", 0, 5, 1)
        with col6:
            h2h_draws = st.number_input("Draws", 0, 5, 0)
        with col7:
            h2h_away = st.number_input(f"{away_team} Wins", 0, 5, 4)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Collapsible odds section
        with st.expander("💰 ODDS (SportyBet) - Click to expand", expanded=False):
            st.markdown("### Betting Odds")
            
            col8, col9, col10 = st.columns(3)
            
            with col8:
                st.markdown("**1X2 Market**")
                odds_home = st.number_input("🏠 Home", 0.0, 10.0, 2.60, 0.05, key="odds_home")
                odds_draw = st.number_input("🤝 Draw", 0.0, 10.0, 3.30, 0.05, key="odds_draw")
                odds_away = st.number_input("✈️ Away", 0.0, 10.0, 2.80, 0.05, key="odds_away")
            
            with col9:
                st.markdown("**Over/Under 2.5**")
                odds_over = st.number_input("📈 Over 2.5", 0.0, 10.0, 2.15, 0.05, key="odds_over")
                odds_under = st.number_input("📉 Under 2.5", 0.0, 10.0, 1.69, 0.05, key="odds_under")
            
            with col10:
                st.markdown("**BTTS Market**")
                odds_btts_yes = st.number_input("✅ BTTS Yes", 0.0, 10.0, 1.87, 0.05, key="odds_btts_yes")
                odds_btts_no = st.number_input("❌ BTTS No", 0.0, 10.0, 1.87, 0.05, key="odds_btts_no")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            # Calculate xG
            xG_home, xG_away = calculate_xG(
                home_scored, home_conceded, away_scored, away_conceded,
                home_form, away_form, h2h_home, h2h_draws, h2h_away,
                home_gd, away_gd, home_top, away_top, home_conv, away_conv
            )
            
            total_xG = xG_home + xG_away
            
            # Set default odds values
            try:
                odds_btts_no_val = odds_btts_no
                odds_under_val = odds_under
            except NameError:
                odds_btts_no_val = 1.87
                odds_under_val = 1.69
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 {home_team} vs {away_team}")
            
            # LOGIC DISPLAY CARD
            st.markdown(f"""
            <div class="logic-card">
                <div class="logic-title">🔒 GROKBET ULTRA - INPUT DATA</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME xG</div>
                        <div class="stat-value-critical">{xG_home:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY xG</div>
                        <div class="stat-value-critical">{xG_away:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">📊 TOTAL xG</div>
                        <div class="stat-value-critical">{total_xG:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME CONV %</div>
                        <div class="stat-value">{home_conv}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY CONV %</div>
                        <div class="stat-value">{away_conv}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME FORM %</div>
                        <div class="stat-value">{home_form}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY FORM %</div>
                        <div class="stat-value">{away_form}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME CONCEDED</div>
                        <div class="stat-value">{home_conceded:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY CONCEDED</div>
                        <div class="stat-value">{away_conceded:.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Check Rule 1 (BTTS No)
            rule1_pass, rule1_bet, rule1_details = check_rule1(
                xG_home, xG_away, home_scored, away_scored,
                home_conv, away_conv, total_xG,
                home_conceded, away_conceded, home_form, away_form,
                home_team, away_team
            )
            
            # Check Rule 2 (Under 2.5)
            rule2_pass, rule2_bet, rule2_details = check_rule2(
                total_xG, home_conv, away_conv,
                home_conceded, away_conceded, home_form, away_form
            )
            
            # Show rule check status
            st.markdown("""
            <div class="rule-check-status">
                <strong>🔍 RULE CHECK STATUS:</strong><br><br>
            """, unsafe_allow_html=True)
            
            # Rule 1 status
            if rule1_pass:
                st.markdown(f'<span style="color: #10b981; font-weight: bold;">✅ RULE 1 (BTTS No): PASS</span>', unsafe_allow_html=True)
                st.markdown(f'<span style="color: #ffffff;">&nbsp;&nbsp;&nbsp;Weak team: {rule1_details["weak_team"]} (xG {rule1_details["weak_xg"]:.2f}, scored {rule1_details["weak_scored"]:.2f}, conv {rule1_details["weak_conv"]}%)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color: #ef4444; font-weight: bold;">❌ RULE 1 (BTTS No): FAIL</span>', unsafe_allow_html=True)
            
            # Rule 2 status
            if rule2_pass:
                st.markdown(f'<span style="color: #10b981; font-weight: bold;">✅ RULE 2 (Under 2.5): PASS</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color: #ef4444; font-weight: bold;">❌ RULE 2 (Under 2.5): FAIL</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            
            # Priority: Rule 1 > Rule 2
            if rule1_pass:
                st.markdown(f"""
                <div class="result-rule1">
                    <strong>🔒 RULE 1 TRIGGERED: BTTS No</strong><br><br>
                    <span class="rule-indicator">✅</span> Weak team xG ≤ 0.50: {rule1_details['weak_xg']:.2f} ≤ 0.50<br>
                    <span class="rule-indicator">✅</span> Weak team scored ≤ 0.9: {rule1_details['weak_scored']:.2f} ≤ 0.9<br>
                    <span class="rule-indicator">✅</span> Weak team conv ≤ 10%: {rule1_details['weak_conv']}% ≤ 10%<br>
                    <span class="rule-indicator">✅</span> Total xG ≤ 2.6: {total_xG:.2f} ≤ 2.6<br>
                    <span class="rule-indicator">✅</span> Defensive strength present: {rule1_details['home_def'] or rule1_details['away_def']}<br>
                    <br>
                    🎯 <strong>BET: BTTS No</strong><br>
                    📊 ODDS: {odds_btts_no_val:.2f}<br>
                    📊 STAKE: <span class="stake-highlight">1.0%</span>
                    <br><br>
                    <strong>📝 VERDICT:</strong> Rule 1 triggered. BTTS unlikely.
                </div>
                """, unsafe_allow_html=True)
                
            elif rule2_pass:
                st.markdown(f"""
                <div class="result-rule2">
                    <strong>🔒 RULE 2 TRIGGERED: Under 2.5 Goals</strong><br><br>
                    <span class="rule-indicator">✅</span> Total xG ≤ 2.4: {total_xG:.2f} ≤ 2.4<br>
                    <span class="rule-indicator">✅</span> Both conv ≤ 12%: {home_conv}% / {away_conv}% ≤ 12%<br>
                    <span class="rule-indicator">✅</span> Defensive strength present: {rule2_details['home_def'] or rule2_details['away_def']}<br>
                    <br>
                    🎯 <strong>BET: Under 2.5 Goals</strong><br>
                    📊 ODDS: {odds_under_val:.2f}<br>
                    📊 STAKE: <span class="stake-highlight">1.0%</span>
                    <br><br>
                    <strong>📝 VERDICT:</strong> Rule 2 triggered. Low-scoring match expected.
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <strong>❌ NO BET</strong><br><br>
                    No rules triggered. Skip this match completely.<br>
                    <br>
                    <strong>📝 VERDICT:</strong> No rule triggered. Skip match.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🎯 GrokBet Ultra | One/Two-Thing-Only System | BTTS No + Under 2.5 | No BTTS Yes | No Over 2.5
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
