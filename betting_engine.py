# grokbet_final.py
# GROKBET – FINAL LOCKED VERSION
# 
# Two rules only:
# 
# RULE A: Total xG ≥ 2.9 AND Home Top Scorer ≥ 2
# RULE B: Away Form ≤ 33% AND Away GD ≥ -13
# 
# ADDITIONAL FILTER (applies to both rules):
# Skip if either team has:
# - Scored Avg ≤ 1.0 OR
# - Conversion % ≤ 10%
# 
# This filter avoids weak attacks that kill BTTS.
# Trade-off: Skips 1 winning bet to avoid 4 losing bets.

import streamlit as st
import math

st.set_page_config(
    page_title="GrokBet - Final",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - BLACK INPUT TEXT FOR VISIBILITY
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Force ALL text to be bright and visible - NO GREY */
    .stMarkdown, .stMarkdown p, .stMarkdown div, p, span {
        color: #ffffff !important;
    }
    
    /* Rule check status - FORCE VISIBLE */
    .rule-check-status {
        background: #0f172a !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        margin: 0.5rem 0 !important;
        border-left: 4px solid #fbbf24 !important;
    }
    
    .rule-check-status strong {
        color: #fbbf24 !important;
    }
    
    .rule-check-status .pass-text {
        color: #10b981 !important;
        font-weight: bold !important;
    }
    
    .rule-check-status .fail-text {
        color: #ef4444 !important;
        font-weight: bold !important;
    }
    
    /* Input labels - BLACK for visibility */
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
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
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
    
    .result-bet {
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
    
    .result-bet:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .result-bet strong, .result-bet .rule-indicator {
        color: #ffffff !important;
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
    
    .result-filter {
        background: linear-gradient(135deg, #1e293b 0%, #2a1a1a 100%);
        border-left: 6px solid #f97316;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #f97316;
        border-right: 1px solid #f97316;
        border-bottom: 1px solid #f97316;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.2);
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
    
    .logic-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #fbbf24 !important;
        margin-bottom: 1rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 5px rgba(251, 191, 36, 0.5);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
    
    .stat-label {
        font-size: 0.7rem;
        color: #fbbf24 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ffffff !important;
        margin-top: 0.25rem;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.3);
    }
    
    .stat-value-critical {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fbbf24 !important;
        margin-top: 0.25rem;
        text-shadow: 0 0 8px rgba(251, 191, 36, 0.5);
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
    
    /* All headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fbbf24 !important;
        font-weight: bold !important;
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
    
    /* Override any white background text issues */
    .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
        background-color: #0f172a !important;
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

# Rule A thresholds
RULE_A_XG_MIN = 2.9
RULE_A_HOME_TOP_SCORER_MIN = 2

# Rule B thresholds
RULE_B_AWAY_FORM_MAX = 33
RULE_B_AWAY_GD_MIN = -13

# Weak attack filter thresholds
WEAK_SCORED_MAX = 1.0
WEAK_CONV_MAX = 10

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

def check_weak_attack_filter(home_scored, away_scored, home_conv, away_conv, home_team, away_team):
    """Skip if either team has weak attack"""
    home_weak = home_scored <= WEAK_SCORED_MAX or home_conv <= WEAK_CONV_MAX
    away_weak = away_scored <= WEAK_SCORED_MAX or away_conv <= WEAK_CONV_MAX
    
    is_weak = home_weak or away_weak
    reasons = []
    
    if home_scored <= WEAK_SCORED_MAX:
        reasons.append(f"{home_team} scored {home_scored:.2f} ≤ {WEAK_SCORED_MAX}")
    if home_conv <= WEAK_CONV_MAX:
        reasons.append(f"{home_team} conv {home_conv}% ≤ {WEAK_CONV_MAX}%")
    if away_scored <= WEAK_SCORED_MAX:
        reasons.append(f"{away_team} scored {away_scored:.2f} ≤ {WEAK_SCORED_MAX}")
    if away_conv <= WEAK_CONV_MAX:
        reasons.append(f"{away_team} conv {away_conv}% ≤ {WEAK_CONV_MAX}%")
    
    return is_weak, reasons

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - Final</h1>
        <p>2 Locked BTTS Yes Rules + Weak Attack Filter</p>
        <div class="badge">⚠️ STAKE: 1.0% WHEN TRIGGERED</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names row
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown('<div class="section-header">🏠 HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.text_input("Team Name", "Goztepe Izmir", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="section-header">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.text_input("Team Name", "Galatasaray", label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Stats input - Organized in groups
        st.markdown('<div class="section-header">📊 TEAM STATISTICS</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f'<div class="team-label">🏠 {home_team}</div>', unsafe_allow_html=True)
            home_scored = st.number_input("⚽ Goals Scored", 0.0, 3.0, 1.20, 0.05, key="home_scored")
            home_conceded = st.number_input("🛡️ Goals Conceded", 0.0, 3.0, 0.70, 0.05, key="home_conceded")
            home_form = st.number_input("📈 Form %", 0, 100, 33, key="home_form")
            home_gd = st.number_input("➕ Goal Difference", -50, 50, 12, key="home_gd")
            home_top = st.number_input("🎯 Top Scorer Goals", 0, 30, 7, key="home_top")
            home_conv = st.number_input("🎯 Conversion %", 0, 100, 11, key="home_conv")
        with col4:
            st.markdown(f'<div class="team-label">✈️ {away_team}</div>', unsafe_allow_html=True)
            away_scored = st.number_input("⚽ Goals Scored", 0.0, 3.0, 2.30, 0.05, key="away_scored")
            away_conceded = st.number_input("🛡️ Goals Conceded", 0.0, 3.0, 0.70, 0.05, key="away_conceded")
            away_form = st.number_input("📈 Form %", 0, 100, 60, key="away_form")
            away_gd = st.number_input("➕ Goal Difference", -50, 50, 43, key="away_gd")
            away_top = st.number_input("🎯 Top Scorer Goals", 0, 30, 13, key="away_top")
            away_conv = st.number_input("🎯 Conversion %", 0, 100, 15, key="away_conv")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # H2H section
        st.markdown('<div class="section-header">🤝 HEAD TO HEAD (Last 5 Matches)</div>', unsafe_allow_html=True)
        col5, col6, col7 = st.columns(3)
        with col5:
            h2h_home = st.number_input(f"{home_team} Wins", 0, 5, 0)
        with col6:
            h2h_draws = st.number_input("Draws", 0, 5, 0)
        with col7:
            h2h_away = st.number_input(f"{away_team} Wins", 0, 5, 5)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Odds section
        st.markdown('<div class="section-header">💰 ODDS (SportyBet)</div>', unsafe_allow_html=True)
        
        col8, col9, col10 = st.columns(3)
        
        with col8:
            st.markdown("**1X2 Market**")
            odds_home = st.number_input("🏠 Home", 0.0, 10.0, 4.28, 0.05, key="odds_home")
            odds_draw = st.number_input("🤝 Draw", 0.0, 10.0, 3.78, 0.05, key="odds_draw")
            odds_away = st.number_input("✈️ Away", 0.0, 10.0, 1.88, 0.05, key="odds_away")
        
        with col9:
            st.markdown("**Over/Under 2.5**")
            odds_over = st.number_input("📈 Over 2.5", 0.0, 10.0, 1.82, 0.05, key="odds_over")
            odds_under = st.number_input("📉 Under 2.5", 0.0, 10.0, 2.05, 0.05, key="odds_under")
        
        with col10:
            st.markdown("**BTTS Market**")
            odds_btts_yes = st.number_input("✅ BTTS Yes", 0.0, 10.0, 1.74, 0.05, key="odds_btts_yes")
            odds_btts_no = st.number_input("❌ BTTS No", 0.0, 10.0, 2.10, 0.05, key="odds_btts_no")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            # Calculate xG for Rule A
            xG_home, xG_away = calculate_xG(
                home_scored, home_conceded, away_scored, away_conceded,
                home_form, away_form, h2h_home, h2h_draws, h2h_away,
                home_gd, away_gd, home_top, away_top, home_conv, away_conv
            )
            
            total_xG = xG_home + xG_away
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 {home_team} vs {away_team}")
            
            # LOGIC DISPLAY CARD - Shows all important data
            st.markdown(f"""
            <div class="logic-card">
                <div class="logic-title">🔒 LOCK LOGIC DATA - 2 RULES + WEAK ATTACK FILTER</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">📊 TOTAL xG</div>
                        <div class="stat-value-critical">{total_xG:.2f}</div>
                        <div class="stat-label">Rule A: ≥ 2.9</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME TOP SCORER</div>
                        <div class="stat-value-critical">{home_top}</div>
                        <div class="stat-label">Rule A: ≥ 2</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">📉 AWAY FORM</div>
                        <div class="stat-value-critical">{away_form}%</div>
                        <div class="stat-label">Rule B: ≤ 33%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">➕ AWAY GD</div>
                        <div class="stat-value-critical">{away_gd}</div>
                        <div class="stat-label">Rule B: ≥ -13</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME SCORED</div>
                        <div class="stat-value">{home_scored:.2f}</div>
                        <div class="stat-label">Filter: > 1.0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME CONV %</div>
                        <div class="stat-value">{home_conv}%</div>
                        <div class="stat-label">Filter: > 10%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY SCORED</div>
                        <div class="stat-value">{away_scored:.2f}</div>
                        <div class="stat-label">Filter: > 1.0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY CONV %</div>
                        <div class="stat-value">{away_conv}%</div>
                        <div class="stat-label">Filter: > 10%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Check weak attack filter first
            is_weak, weak_reasons = check_weak_attack_filter(
                home_scored, away_scored, home_conv, away_conv, home_team, away_team
            )
            
            if is_weak:
                reasons_text = ' | '.join(weak_reasons)
                st.markdown(f"""
                <div class="result-filter">
                    <strong>⚠️ WEAK ATTACK FILTER TRIGGERED - NO BET</strong><br><br>
                    <span class="rule-indicator">🚫</span> {reasons_text}<br>
                    <br>
                    <strong>📝 VERDICT:</strong> Skipping this match to avoid potential BTTS No outcome.<br>
                    📊 STAKE: <span class="stake-highlight">0% (SKIPPED)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check rules
                rule_a = (total_xG >= RULE_A_XG_MIN) and (home_top >= RULE_A_HOME_TOP_SCORER_MIN)
                rule_b = (away_form <= RULE_B_AWAY_FORM_MAX) and (away_gd >= RULE_B_AWAY_GD_MIN)
                
                # Show rule check status with visible colors
                st.markdown("""
                <div class="rule-check-status">
                    <strong>🔍 RULE CHECK STATUS:</strong><br><br>
                """, unsafe_allow_html=True)
                
                # Rule A status
                if rule_a:
                    st.markdown(f'<span style="color: #10b981; font-weight: bold;">✅ Rule A: PASS</span> <span style="color: #ffffff;">(xG {total_xG:.2f} ≥ 2.9 AND Home Top Scorer {home_top} ≥ 2)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="color: #ef4444; font-weight: bold;">❌ Rule A: FAIL</span> <span style="color: #ffffff;">(xG {total_xG:.2f} ≥ 2.9 AND Home Top Scorer {home_top} ≥ 2)</span>', unsafe_allow_html=True)
                
                # Rule B status
                if rule_b:
                    st.markdown(f'<span style="color: #10b981; font-weight: bold;">✅ Rule B: PASS</span> <span style="color: #ffffff;">(Away Form {away_form}% ≤ 33% AND Away GD {away_gd} ≥ -13)</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="color: #ef4444; font-weight: bold;">❌ Rule B: FAIL</span> <span style="color: #ffffff;">(Away Form {away_form}% ≤ 33% AND Away GD {away_gd} ≥ -13)</span>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                
                if rule_a:
                    st.markdown(f"""
                    <div class="result-bet">
                        <strong>🔒 RULE A TRIGGERED: BTTS Yes</strong><br><br>
                        <span class="rule-indicator">✅</span> Total xG ≥ 2.9: {total_xG:.2f} ≥ 2.9<br>
                        <span class="rule-indicator">✅</span> Home Top Scorer ≥ 2: {home_top} ≥ 2<br>
                        <span class="rule-indicator">✅</span> Weak Attack Filter: PASSED (no weak attacks)<br>
                        <br>
                        🎯 <strong>BET: BTTS Yes</strong><br>
                        📊 ODDS: {odds_btts_yes:.2f}<br>
                        📊 STAKE: <span class="stake-highlight">1.0%</span>
                        <br><br>
                        <strong>📝 VERDICT:</strong> Rule A triggered. Bet BTTS Yes.
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif rule_b:
                    st.markdown(f"""
                    <div class="result-bet">
                        <strong>🔒 RULE B TRIGGERED: BTTS Yes</strong><br><br>
                        <span class="rule-indicator">✅</span> Away Form ≤ 33%: {away_form}% ≤ 33<br>
                        <span class="rule-indicator">✅</span> Away GD ≥ -13: {away_gd} ≥ -13<br>
                        <span class="rule-indicator">✅</span> Weak Attack Filter: PASSED (no weak attacks)<br>
                        <br>
                        🎯 <strong>BET: BTTS Yes</strong><br>
                        📊 ODDS: {odds_btts_yes:.2f}<br>
                        📊 STAKE: <span class="stake-highlight">1.0%</span>
                        <br><br>
                        <strong>📝 VERDICT:</strong> Rule B triggered. Bet BTTS Yes.
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="result-skip">
                        <strong>❌ NO BET</strong><br><br>
                        No rules triggered. Skip this match completely.<br>
                        <br>
                        <strong>📝 VERDICT:</strong> No rule triggered. Skip match.<br>
                        📊 STAKE: <span class="stake-highlight">0% (SKIPPED)</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🎯 GrokBet - Final | 2 Rules + Weak Attack Filter | Live Tested
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
