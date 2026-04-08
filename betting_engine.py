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
# CUSTOM CSS - ENHANCED WITH BETTER TEXT VISIBILITY
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Force all text to be visible */
    .stMarkdown, .stMarkdown p, .stMarkdown div, p, span, label {
        color: #e2e8f0 !important;
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
        color: #cbd5e1 !important;
        font-size: 0.9rem;
    }
    
    .badge {
        display: inline-block;
        background: rgba(251, 191, 36, 0.2);
        border: 1px solid #fbbf24;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.7rem;
        color: #fbbf24 !important;
        margin-top: 0.5rem;
    }
    
    /* Input cards */
    .input-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #fbbf24;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .input-card:hover {
        border-color: #f59e0b;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 0.85rem;
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
    }
    
    /* Result boxes */
    .result-box {
        background: #0f172a;
        border-radius: 20px;
        padding: 1.5rem;
        border: 2px solid #fbbf24;
        margin-top: 1.5rem;
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-bet {
        background: #1e293b;
        border-left: 6px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        transition: transform 0.2s ease;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    .result-bet:hover {
        transform: translateX(5px);
    }
    
    .result-skip {
        background: #1e293b;
        border-left: 6px solid #ef4444;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
    }
    
    .result-filter {
        background: #1e293b;
        border-left: 6px solid #f97316;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
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
        box-shadow: 0 2px 8px rgba(251, 191, 36, 0.3);
    }
    
    /* LOGIC DISPLAY CARD */
    .logic-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 2px solid #fbbf24;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    .logic-title {
        font-size: 1rem;
        font-weight: bold;
        color: #fbbf24 !important;
        margin-bottom: 1rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: #0f172a;
        border-radius: 10px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #fbbf24 !important;
        margin-top: 0.25rem;
    }
    
    .stat-value-critical {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fbbf24 !important;
        margin-top: 0.25rem;
    }
    
    .stat-pass {
        color: #10b981 !important;
        font-weight: bold;
    }
    
    .stat-fail {
        color: #ef4444 !important;
        font-weight: bold;
    }
    
    /* Filter status card */
    .filter-card {
        background: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #f97316;
    }
    
    .filter-title {
        font-size: 0.85rem;
        font-weight: bold;
        color: #f97316 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Rule indicator */
    .rule-indicator {
        font-size: 0.85rem;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        background: rgba(251, 191, 36, 0.15);
        display: inline-block;
        margin-right: 0.5rem;
        color: #fbbf24 !important;
    }
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #fbbf24, transparent);
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
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.4);
    }
    
    /* Number inputs */
    .stNumberInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
    }
    
    .stNumberInput label {
        color: #cbd5e1 !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
        font-size: 1rem;
    }
    
    .stTextInput label {
        color: #cbd5e1 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fbbf24 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 1rem;
        border-top: 1px solid #fbbf24;
        font-size: 0.7rem;
        color: #94a3b8 !important;
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
        <div class="badge">⚠️ Stake: 1.0% when triggered</div>
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
            st.markdown(f"**🏠 {home_team}**")
            home_scored = st.number_input("⚽ Goals Scored", 0.0, 3.0, 1.20, 0.05, key="home_scored")
            home_conceded = st.number_input("🛡️ Goals Conceded", 0.0, 3.0, 0.70, 0.05, key="home_conceded")
            home_form = st.number_input("📈 Form %", 0, 100, 33, key="home_form")
            home_gd = st.number_input("➕ Goal Difference", -50, 50, 12, key="home_gd")
            home_top = st.number_input("🎯 Top Scorer Goals", 0, 30, 7, key="home_top")
            home_conv = st.number_input("🎯 Conversion %", 0, 100, 11, key="home_conv")
        with col4:
            st.markdown(f"**✈️ {away_team}**")
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
            st.markdown("""
            <div class="logic-card">
                <div class="logic-title">🔒 LOCK LOGIC DATA - 2 RULES + WEAK ATTACK FILTER</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">📊 TOTAL xG</div>
                        <div class="stat-value-critical">{:.2f}</div>
                        <div class="stat-label">Rule A: ≥ 2.9</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME TOP SCORER</div>
                        <div class="stat-value-critical">{}</div>
                        <div class="stat-label">Rule A: ≥ 2</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">📉 AWAY FORM</div>
                        <div class="stat-value-critical">{}%</div>
                        <div class="stat-label">Rule B: ≤ 33%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">➕ AWAY GD</div>
                        <div class="stat-value-critical">{}</div>
                        <div class="stat-label">Rule B: ≥ -13</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME SCORED</div>
                        <div class="stat-value">{:.2f}</div>
                        <div class="stat-label">Filter: > 1.0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">🏠 HOME CONV %</div>
                        <div class="stat-value">{}%</div>
                        <div class="stat-label">Filter: > 10%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY SCORED</div>
                        <div class="stat-value">{:.2f}</div>
                        <div class="stat-label">Filter: > 1.0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">✈️ AWAY CONV %</div>
                        <div class="stat-value">{}%</div>
                        <div class="stat-label">Filter: > 10%</div>
                    </div>
                </div>
            </div>
            """.format(total_xG, home_top, away_form, away_gd, home_scored, home_conv, away_scored, away_conv), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Check weak attack filter first
            is_weak, weak_reasons = check_weak_attack_filter(
                home_scored, away_scored, home_conv, away_conv, home_team, away_team
            )
            
            if is_weak:
                st.markdown(f"""
                <div class="result-filter">
                    <strong>⚠️ WEAK ATTACK FILTER TRIGGERED - NO BET</strong><br><br>
                    <span class="rule-indicator">🚫</span> {' | '.join(weak_reasons)}<br>
                    <br>
                    <strong>📝 Verdict:</strong> Skipping this match to avoid potential BTTS No outcome.<br>
                    📊 Stake: <span class="stake-highlight">0% (Skipped)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check rules
                rule_a = (total_xG >= RULE_A_XG_MIN) and (home_top >= RULE_A_HOME_TOP_SCORER_MIN)
                rule_b = (away_form <= RULE_B_AWAY_FORM_MAX) and (away_gd >= RULE_B_AWAY_GD_MIN)
                
                # Show rule check status
                st.markdown("**🔍 Rule Check Status:**")
                rule_a_status = "✅ PASS" if rule_a else "❌ FAIL"
                rule_b_status = "✅ PASS" if rule_b else "❌ FAIL"
                st.markdown(f"- Rule A: {rule_a_status} (xG ≥ 2.9 AND Home Top Scorer ≥ 2)")
                st.markdown(f"- Rule B: {rule_b_status} (Away Form ≤ 33% AND Away GD ≥ -13)")
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
                        📊 Odds: {odds_btts_yes:.2f}<br>
                        📊 Stake: <span class="stake-highlight">1.0%</span>
                        <br><br>
                        <strong>📝 Verdict:</strong> Rule A triggered. Bet BTTS Yes.
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
                        📊 Odds: {odds_btts_yes:.2f}<br>
                        📊 Stake: <span class="stake-highlight">1.0%</span>
                        <br><br>
                        <strong>📝 Verdict:</strong> Rule B triggered. Bet BTTS Yes.
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="result-skip">
                        <strong>❌ NO BET</strong><br><br>
                        No rules triggered. Skip this match completely.<br>
                        <br>
                        <strong>📝 Verdict:</strong> No rule triggered. Skip match.<br>
                        📊 Stake: <span class="stake-highlight">0% (Skipped)</span>
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