# grokbet_poisson_zip.py
# GROKBET – NO DRAW PREDICTOR (Poisson + Zero-Inflation)
# 
# Based on the complete logic from the final production version.
# 
# Core features:
# - League-specific parameters (avg goals, thresholds, ZIP factors)
# - Zero-Inflation Adjustment for 0-0 draws
# - Value betting integration (edge calculation)
# - Kelly Criterion stake sizing
# - Clear decision matrix

import streamlit as st
import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple, Optional

st.set_page_config(
    page_title="GrokBet - No Draw Predictor",
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
        max-width: 1200px;
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
    
    .result-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
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
    
    .result-no-draw {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 6px solid #10b981;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #10b981;
        border-right: 1px solid #10b981;
        border-bottom: 1px solid #10b981;
    }
    
    .result-draw {
        background: linear-gradient(135deg, #1e293b 0%, #2a1a1a 100%);
        border-left: 6px solid #f97316;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-top: 1px solid #f97316;
        border-right: 1px solid #f97316;
        border-bottom: 1px solid #f97316;
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(251, 191, 36, 0.5);
    }
    
    .stNumberInput > div > div > input {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
        font-weight: bold;
    }
    
    .stSelectbox > div > div {
        background: #0f172a;
        border-color: #fbbf24;
        color: #fbbf24 !important;
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 1rem;
        border-top: 1px solid #fbbf24;
        font-size: 0.7rem;
        color: #94a3b8 !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid #fbbf24;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #fbbf24;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION 1: LEAGUE DATABASE (2025/26 Season)
# ============================================================================

LEAGUE_DATABASE = {
    # Top European Leagues
    "Premier League (England)": {"avg_goals": 2.84, "threshold": 0.22, "zip_factor": 0.05},
    "Serie A (Italy)": {"avg_goals": 2.65, "threshold": 0.24, "zip_factor": 0.06},
    "La Liga (Spain)": {"avg_goals": 2.62, "threshold": 0.24, "zip_factor": 0.06},
    "Bundesliga (Germany)": {"avg_goals": 3.10, "threshold": 0.19, "zip_factor": 0.04},
    "Ligue 1 (France)": {"avg_goals": 2.96, "threshold": 0.20, "zip_factor": 0.04},
    "Primeira Liga (Portugal)": {"avg_goals": 2.50, "threshold": 0.25, "zip_factor": 0.07},
    "Eredivisie (Netherlands)": {"avg_goals": 3.00, "threshold": 0.19, "zip_factor": 0.04},
    
    # Global High-Scoring Leagues
    "Singapore Premier League": {"avg_goals": 3.63, "threshold": 0.15, "zip_factor": 0.02},
    "Malaysia Super League": {"avg_goals": 3.42, "threshold": 0.16, "zip_factor": 0.02},
    "Qatar Stars League": {"avg_goals": 3.32, "threshold": 0.16, "zip_factor": 0.02},
    "Dutch Eerste Divisie": {"avg_goals": 3.31, "threshold": 0.16, "zip_factor": 0.02},
    "Iceland Besta deild": {"avg_goals": 3.30, "threshold": 0.17, "zip_factor": 0.02},
    "Swiss Super League": {"avg_goals": 3.28, "threshold": 0.17, "zip_factor": 0.03},
    "Denmark Superliga": {"avg_goals": 3.22, "threshold": 0.17, "zip_factor": 0.03},
    "USA Major League Soccer": {"avg_goals": 3.10, "threshold": 0.18, "zip_factor": 0.03},
}

DEFAULT_LEAGUE = {"avg_goals": 2.70, "threshold": 0.22, "zip_factor": 0.05}

# ============================================================================
# SECTION 2: CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def calculate_xg(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_avg: float
) -> Tuple[float, float]:
    """Calculate Expected Goals for home and away teams."""
    home_xg = (home_attack * away_defense) / league_avg
    away_xg = (away_attack * home_defense) / league_avg
    return home_xg, away_xg


def poisson_probability_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int = 7
) -> np.ndarray:
    """Generate the goal probability matrix using Poisson distribution."""
    prob_home = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    prob_away = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    # Normalize to ensure sum = 1.0
    prob_home = np.array(prob_home) / np.sum(prob_home)
    prob_away = np.array(prob_away) / np.sum(prob_away)
    
    return np.outer(prob_home, prob_away)


def apply_zero_inflation(
    matrix: np.ndarray,
    zip_factor: float
) -> np.ndarray:
    """Apply Zero-Inflated Poisson adjustment to increase 0-0 probability."""
    original_00 = matrix[0, 0]
    adjusted_00 = original_00 + zip_factor * (1 - original_00)
    
    if adjusted_00 <= original_00:
        return matrix
    
    scale = (1 - adjusted_00) / (1 - original_00)
    adjusted_matrix = matrix * scale
    adjusted_matrix[0, 0] = adjusted_00
    
    return adjusted_matrix


def calculate_draw_probability(
    home_xg: float,
    away_xg: float,
    zip_factor: float = 0.0,
    max_goals: int = 7
) -> float:
    """Calculate the probability of a draw using Poisson + optional ZIP."""
    matrix = poisson_probability_matrix(home_xg, away_xg, max_goals)
    
    if zip_factor > 0:
        matrix = apply_zero_inflation(matrix, zip_factor)
    
    draw_prob = np.trace(matrix)
    return min(draw_prob, 1.0)


def calculate_no_draw_odds(home_odds: float, away_odds: float) -> float:
    """Calculate implied 'No Draw' odds from 1X2 market."""
    return 1 / ((1/home_odds) + (1/away_odds))


def calculate_expected_value(model_prob: float, decimal_odds: float) -> float:
    """Calculate Expected Value for a bet."""
    implied_prob = 1 / decimal_odds
    return model_prob - implied_prob


def kelly_fraction(
    probability: float,
    decimal_odds: float,
    quarter_kelly: bool = True
) -> float:
    """Calculate Kelly Criterion stake fraction."""
    b = decimal_odds - 1
    p = probability
    q = 1 - p
    
    if p * b - q <= 0:
        return 0.0
    
    kelly = (p * b - q) / b
    
    if quarter_kelly:
        kelly = kelly * 0.25
    
    return min(max(kelly, 0.0), 0.25)


# ============================================================================
# SECTION 3: MAIN PREDICTION FUNCTION
# ============================================================================

def predict_no_draw(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_name: str = None,
    custom_league_avg: float = None,
    custom_threshold: float = None,
    use_zip: bool = True,
    home_odds: float = None,
    away_odds: float = None,
    bankroll: float = None
) -> Dict:
    """Main prediction function for 'No Draw' outcome."""
    
    # Get league parameters
    if league_name and league_name in LEAGUE_DATABASE:
        league = LEAGUE_DATABASE[league_name]
        league_avg = custom_league_avg or league["avg_goals"]
        threshold = custom_threshold or league["threshold"]
        zip_factor = league["zip_factor"] if use_zip else 0.0
    else:
        league_avg = custom_league_avg or DEFAULT_LEAGUE["avg_goals"]
        threshold = custom_threshold or DEFAULT_LEAGUE["threshold"]
        zip_factor = DEFAULT_LEAGUE["zip_factor"] if use_zip else 0.0
    
    # Calculate Expected Goals
    home_xg, away_xg = calculate_xg(
        home_attack, home_defense,
        away_attack, away_defense,
        league_avg
    )
    
    # Calculate Draw Probability
    draw_prob = calculate_draw_probability(home_xg, away_xg, zip_factor, max_goals=7)
    no_draw_prob = 1 - draw_prob
    
    # Decision Logic
    prediction = "NO DRAW" if draw_prob < threshold else "DRAW LIKELY"
    
    # Confidence Level
    if draw_prob < threshold * 0.7:
        confidence = "HIGH"
    elif draw_prob < threshold:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Value Betting (if odds provided)
    value_bet = None
    recommended_stake = None
    recommended_stake_pct = None
    edge = None
    
    if home_odds and away_odds and home_odds > 0 and away_odds > 0:
        no_draw_odds = calculate_no_draw_odds(home_odds, away_odds)
        edge = calculate_expected_value(no_draw_prob, no_draw_odds)
        
        if edge > 0.05 and prediction == "NO DRAW":
            value_bet = "YES"
            if bankroll and bankroll > 0:
                kelly = kelly_fraction(no_draw_prob, no_draw_odds)
                recommended_stake = round(bankroll * kelly, 2)
                recommended_stake_pct = round(kelly * 100, 1)
        else:
            value_bet = "NO"
    
    return {
        "league": league_name or "Custom",
        "league_avg_goals": round(league_avg, 2),
        "threshold_used": round(threshold * 100, 1),
        "zip_factor_used": zip_factor if use_zip else 0.0,
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "total_xg": round(home_xg + away_xg, 2),
        "draw_probability": round(draw_prob * 100, 1),
        "no_draw_probability": round(no_draw_prob * 100, 1),
        "prediction": prediction,
        "confidence": confidence,
        "value_bet": value_bet,
        "edge_percentage": round(edge * 100, 1) if edge is not None else None,
        "recommended_stake_usd": recommended_stake,
        "recommended_stake_pct": recommended_stake_pct,
    }


# ============================================================================
# SECTION 4: MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet - No Draw Predictor</h1>
        <p>Poisson Distribution + Zero-Inflation Adjustment</p>
        <div class="badge">BET WHEN DRAW PROBABILITY &lt; LEAGUE THRESHOLD</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header">🏠 HOME TEAM</div>', unsafe_allow_html=True)
            home_team = st.text_input("Team Name", "Bayern Munich", label_visibility="collapsed")
        with col2:
            st.markdown('<div class="section-header">✈️ AWAY TEAM</div>', unsafe_allow_html=True)
            away_team = st.text_input("Team Name", "Darmstadt", label_visibility="collapsed")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # League selection
        st.markdown('<div class="section-header">🏆 LEAGUE SETTINGS</div>', unsafe_allow_html=True)
        
        league_options = list(LEAGUE_DATABASE.keys()) + ["Custom"]
        selected_league = st.selectbox("Select League", league_options, index=3)  # Bundesliga default
        
        use_zip = st.checkbox("Apply Zero-Inflation Adjustment (ZIP)", value=True)
        
        if selected_league == "Custom":
            col_avg, col_thresh, col_zip = st.columns(3)
            with col_avg:
                custom_avg = st.number_input("League Avg Goals", 2.0, 4.0, 2.70, 0.05)
            with col_thresh:
                custom_thresh = st.number_input("Draw Threshold %", 10, 35, 22, 1) / 100.0
            with col_zip:
                custom_zip = st.number_input("ZIP Factor", 0.0, 0.10, 0.05, 0.01)
            league_name = None
            league_avg = custom_avg
            league_threshold = custom_thresh
            league_zip = custom_zip
        else:
            league_name = selected_league
            league_avg = None
            league_threshold = None
            league_zip = None
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Team statistics
        st.markdown('<div class="section-header">📊 TEAM STATISTICS (Last 5-10 Matches)</div>', unsafe_allow_html=True)
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.markdown(f'<div class="team-label">🏠 {home_team}</div>', unsafe_allow_html=True)
            home_attack = st.number_input("Goals Scored Per Game", 0.5, 4.0, 3.57, 0.05, key="home_attack")
            home_defense = st.number_input("Goals Conceded Per Game", 0.5, 4.0, 0.85, 0.05, key="home_defense")
        
        with col_stats2:
            st.markdown(f'<div class="team-label">✈️ {away_team}</div>', unsafe_allow_html=True)
            away_attack = st.number_input("Goals Scored Per Game", 0.5, 4.0, 0.75, 0.05, key="away_attack")
            away_defense = st.number_input("Goals Conceded Per Game", 0.5, 4.0, 2.10, 0.05, key="away_defense")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Odds and bankroll (optional)
        st.markdown('<div class="section-header">💰 BETTING SETTINGS (Optional)</div>', unsafe_allow_html=True)
        
        col_odds1, col_odds2, col_bank = st.columns(3)
        with col_odds1:
            home_odds = st.number_input(f"{home_team} Win Odds", 1.01, 15.0, 1.25, 0.05, key="home_odds")
        with col_odds2:
            away_odds = st.number_input(f"{away_team} Win Odds", 1.01, 15.0, 8.00, 0.05, key="away_odds")
        with col_bank:
            bankroll = st.number_input("Bankroll (USD)", 0, 10000, 1000, 100, key="bankroll")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            # Get league parameters
            if selected_league == "Custom":
                final_league_name = "Custom"
                final_league_avg = league_avg
                final_threshold = league_threshold
                final_zip = league_zip if use_zip else 0.0
            else:
                final_league_name = selected_league
                final_league_avg = None
                final_threshold = None
                final_zip = None
            
            # Make prediction
            result = predict_no_draw(
                home_attack=home_attack,
                home_defense=home_defense,
                away_attack=away_attack,
                away_defense=away_defense,
                league_name=final_league_name,
                custom_league_avg=final_league_avg,
                custom_threshold=final_threshold,
                use_zip=use_zip,
                home_odds=home_odds if home_odds > 0 else None,
                away_odds=away_odds if away_odds > 0 else None,
                bankroll=bankroll if bankroll > 0 else None
            )
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 {home_team} vs {away_team}")
            st.markdown("---")
            
            # Key metrics row
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['home_xg']}</div>
                    <div class="metric-label">🏠 HOME xG</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['away_xg']}</div>
                    <div class="metric-label">✈️ AWAY xG</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['total_xg']}</div>
                    <div class="metric-label">📊 TOTAL xG</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{result['draw_probability']}%</div>
                    <div class="metric-label">🎲 DRAW PROB</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # League info
            st.markdown(f"""
            **🏆 League:** {result['league']}  
            **📈 League Avg Goals:** {result['league_avg_goals']} | **🎯 Threshold:** {result['threshold_used']}% | **🔧 ZIP Factor:** {result['zip_factor_used']}
            """)
            
            st.markdown("---")
            
            # Decision
            if result['prediction'] == "NO DRAW":
                confidence_color = "#10b981" if result['confidence'] == "HIGH" else "#fbbf24" if result['confidence'] == "MEDIUM" else "#f97316"
                st.markdown(f"""
                <div class="result-no-draw">
                    <strong>🔒 NO DRAW PREDICTED</strong><br><br>
                    📊 Draw Probability: <strong>{result['draw_probability']}%</strong><br>
                    📊 No Draw Probability: <strong>{result['no_draw_probability']}%</strong><br>
                    🎯 League Threshold: <strong>{result['threshold_used']}%</strong><br>
                    📈 Confidence: <strong style="color:{confidence_color}">{result['confidence']}</strong><br>
                    <br>
                    🎯 <strong>CONCLUSION: Bet against the draw (Home or Away win)</strong>
                </div>
                """, unsafe_allow_html=True)
                
                # Value betting recommendation
                if result['value_bet'] == "YES":
                    stake_text = f"${result['recommended_stake_usd']} ({result['recommended_stake_pct']}% of bankroll)" if result['recommended_stake_usd'] else "Use quarter-Kelly formula"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a3a2a 0%, #1e293b 100%); padding: 1rem; border-radius: 12px; margin-top: 0.75rem;">
                        <strong>💰 VALUE BET DETECTED</strong><br>
                        📈 Edge: <strong>{result['edge_percentage']}%</strong><br>
                        🎯 Recommended Stake: <strong>{stake_text}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
            else:
                st.markdown(f"""
                <div class="result-draw">
                    <strong>⚠️ DRAW LIKELY</strong><br><br>
                    📊 Draw Probability: <strong>{result['draw_probability']}%</strong><br>
                    📊 No Draw Probability: <strong>{result['no_draw_probability']}%</strong><br>
                    🎯 League Threshold: <strong>{result['threshold_used']}%</strong><br>
                    <br>
                    🎯 <strong>CONCLUSION: Skip betting. Draw is likely.</strong>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="footer">
        🎯 GrokBet - No Draw Predictor | Poisson Distribution + Zero-Inflation Adjustment | Value Betting Integration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
