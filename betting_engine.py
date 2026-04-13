"""
Complete "No Draw" Prediction System
With Team-Specific Rolling Averages and Poisson Distribution
Production Version - Streamlit App
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import poisson
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="No Draw Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SECTION 1: LEAGUE DATABASE (2025/26 Season)
# ============================================================================

LEAGUE_DATABASE = {
    # Top European Leagues
    "Premier League": {"avg_goals": 2.84, "threshold": 0.22, "zip_factor": 0.05, "color": "#37003C"},
    "Serie A": {"avg_goals": 2.65, "threshold": 0.24, "zip_factor": 0.06, "color": "#0D2B42"},
    "La Liga": {"avg_goals": 2.62, "threshold": 0.24, "zip_factor": 0.06, "color": "#FFD700"},
    "Bundesliga": {"avg_goals": 3.10, "threshold": 0.19, "zip_factor": 0.04, "color": "#E1000F"},
    "Ligue 1": {"avg_goals": 2.96, "threshold": 0.20, "zip_factor": 0.04, "color": "#1A5B9C"},
    "Primeira Liga": {"avg_goals": 2.50, "threshold": 0.25, "zip_factor": 0.07, "color": "#008000"},
    "Eredivisie": {"avg_goals": 3.00, "threshold": 0.19, "zip_factor": 0.04, "color": "#FF6600"},
    
    # High-Scoring Leagues
    "Singapore Premier": {"avg_goals": 3.63, "threshold": 0.15, "zip_factor": 0.02, "color": "#E31A1A"},
    "Malaysia Super": {"avg_goals": 3.42, "threshold": 0.16, "zip_factor": 0.02, "color": "#1A4D2E"},
    "Qatar Stars": {"avg_goals": 3.32, "threshold": 0.16, "zip_factor": 0.02, "color": "#8A1F1F"},
    "Dutch Eerste Divisie": {"avg_goals": 3.31, "threshold": 0.16, "zip_factor": 0.02, "color": "#FF7F00"},
    "Iceland Besta deild": {"avg_goals": 3.30, "threshold": 0.17, "zip_factor": 0.02, "color": "#004C97"},
    "Swiss Super": {"avg_goals": 3.28, "threshold": 0.17, "zip_factor": 0.03, "color": "#E1000F"},
    "Denmark Superliga": {"avg_goals": 3.22, "threshold": 0.17, "zip_factor": 0.03, "color": "#1E5B3C"},
    "USA MLS": {"avg_goals": 3.10, "threshold": 0.18, "zip_factor": 0.03, "color": "#1A2B4C"},
}

DEFAULT_LEAGUE = {"avg_goals": 2.70, "threshold": 0.22, "zip_factor": 0.05, "color": "#333333"}


# ============================================================================
# SECTION 2: CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def calculate_expected_goals(
    home_attack_avg: float,
    home_defense_avg: float,
    away_attack_avg: float,
    away_defense_avg: float,
    league_avg: float
) -> Tuple[float, float]:
    """
    Calculate expected goals using the Dixon-Coles method.
    
    Formula: xG = (Attack_Avg × Defense_Avg) / League_Avg
    """
    home_xg = (home_attack_avg * away_defense_avg) / league_avg
    away_xg = (away_attack_avg * home_defense_avg) / league_avg
    
    # Clamp to realistic range
    home_xg = max(0.2, min(4.5, home_xg))
    away_xg = max(0.2, min(4.5, away_xg))
    
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
    
    # Sum of diagonal = draw probability
    draw_prob = np.trace(matrix)
    
    return min(draw_prob, 1.0)


def calculate_score_probabilities(
    home_xg: float,
    away_xg: float,
    max_goals: int = 4
) -> Dict[Tuple[int, int], float]:
    """Calculate probabilities for specific scorelines."""
    prob_home = [poisson.pmf(i, home_xg) for i in range(max_goals + 1)]
    prob_away = [poisson.pmf(i, away_xg) for i in range(max_goals + 1)]
    
    # Normalize
    prob_home = np.array(prob_home) / np.sum(prob_home)
    prob_away = np.array(prob_away) / np.sum(prob_away)
    
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores[(h, a)] = prob_home[h] * prob_away[a]
    
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:10])


# ============================================================================
# SECTION 3: VALUE BETTING FUNCTIONS
# ============================================================================

def calculate_no_draw_odds(home_odds: float, away_odds: float) -> float:
    """Calculate implied 'No Draw' odds from 1X2 market."""
    if home_odds <= 0 or away_odds <= 0:
        return 0
    return 1 / ((1/home_odds) + (1/away_odds))


def calculate_edge(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value edge."""
    if decimal_odds <= 0:
        return 0
    implied_prob = 1 / decimal_odds
    return model_prob - implied_prob


def kelly_fraction(probability: float, decimal_odds: float, quarter: bool = True) -> float:
    """Calculate Kelly Criterion stake fraction."""
    if decimal_odds <= 1:
        return 0.0
    
    b = decimal_odds - 1
    p = probability
    q = 1 - p
    
    if p * b - q <= 0:
        return 0.0
    
    kelly = (p * b - q) / b
    
    if quarter:
        kelly = kelly * 0.25
    
    return min(max(kelly, 0.0), 0.25)


# ============================================================================
# SECTION 4: MAIN PREDICTION FUNCTION
# ============================================================================

def predict_no_draw(
    home_goals_scored_avg: float,
    home_goals_conceded_avg: float,
    away_goals_scored_avg: float,
    away_goals_conceded_avg: float,
    league_name: str = None,
    league_avg_goals: float = None,
    home_xg_avg: float = None,
    away_xg_avg: float = None,
    use_zip: bool = True,
    custom_threshold: float = None,
    home_odds: float = None,
    away_odds: float = None,
    bankroll: float = None,
    use_xg_if_available: bool = True
) -> Dict:
    """
    Predict whether a match will NOT end in a draw.
    """
    
    # ===== Step 1: Get league parameters =====
    if league_name and league_name in LEAGUE_DATABASE:
        league = LEAGUE_DATABASE[league_name]
        league_avg = league_avg_goals or league["avg_goals"]
        threshold = custom_threshold or league["threshold"]
        zip_factor = league["zip_factor"] if use_zip else 0.0
    else:
        league_avg = league_avg_goals or DEFAULT_LEAGUE["avg_goals"]
        threshold = custom_threshold or DEFAULT_LEAGUE["threshold"]
        zip_factor = DEFAULT_LEAGUE["zip_factor"] if use_zip else 0.0
    
    # ===== Step 2: Choose best available stat =====
    if use_xg_if_available and home_xg_avg is not None and away_xg_avg is not None:
        home_attack = home_xg_avg
        away_attack = away_xg_avg
        home_defense = home_goals_conceded_avg
        away_defense = away_goals_conceded_avg
        stat_source = "xG (Advanced)"
    else:
        home_attack = home_goals_scored_avg
        away_attack = away_goals_scored_avg
        home_defense = home_goals_conceded_avg
        away_defense = away_goals_conceded_avg
        stat_source = "Goals (Basic)"
    
    # ===== Step 3: Calculate Expected Goals =====
    home_xg, away_xg = calculate_expected_goals(
        home_attack, home_defense,
        away_attack, away_defense,
        league_avg
    )
    
    # ===== Step 4: Calculate Draw Probability =====
    draw_prob = calculate_draw_probability(home_xg, away_xg, zip_factor)
    no_draw_prob = 1 - draw_prob
    
    # ===== Step 5: Decision Logic =====
    prediction = "NO DRAW" if draw_prob < threshold else "DRAW LIKELY"
    
    if draw_prob < threshold * 0.7:
        confidence = "HIGH"
    elif draw_prob < threshold:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # ===== Step 6: Value Betting =====
    value_bet = None
    recommended_stake = None
    edge = None
    
    if home_odds and away_odds and home_odds > 0 and away_odds > 0:
        no_draw_odds = calculate_no_draw_odds(home_odds, away_odds)
        edge = calculate_edge(no_draw_prob, no_draw_odds)
        
        if edge > 0.05 and prediction == "NO DRAW":
            value_bet = "YES (5%+ edge)"
            if bankroll and bankroll > 0:
                kelly = kelly_fraction(no_draw_prob, no_draw_odds)
                recommended_stake = round(bankroll * kelly, 2)
        elif edge > 0 and prediction == "NO DRAW":
            value_bet = "YES (small edge)"
        else:
            value_bet = "NO"
    
    # ===== Step 7: Calculate score probabilities =====
    top_scores = calculate_score_probabilities(home_xg, away_xg)
    
    return {
        "league": league_name or "Custom",
        "league_avg_goals": round(league_avg, 2),
        "stat_source": stat_source,
        "threshold": f"{threshold * 100:.1f}%",
        
        "home_goals_avg": round(home_goals_scored_avg, 2),
        "home_conceded_avg": round(home_goals_conceded_avg, 2),
        "away_goals_avg": round(away_goals_scored_avg, 2),
        "away_conceded_avg": round(away_goals_conceded_avg, 2),
        
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "total_xg": round(home_xg + away_xg, 2),
        
        "draw_probability": draw_prob,
        "no_draw_probability": no_draw_prob,
        
        "prediction": prediction,
        "confidence": confidence,
        
        "value_bet": value_bet,
        "edge": edge,
        "recommended_stake": recommended_stake,
        
        "top_scores": top_scores,
        "zip_factor_used": zip_factor,
    }


# ============================================================================
# SECTION 5: UI COMPONENTS
# ============================================================================

def render_prediction_card(result: Dict, home_team: str, away_team: str):
    """Render the prediction result in a beautiful card."""
    
    # Determine colors based on prediction
    if result["prediction"] == "NO DRAW":
        pred_color = "#10b981"  # Green
        pred_bg = "linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%)"
        border_color = "#10b981"
    else:
        pred_color = "#f97316"  # Orange
        pred_bg = "linear-gradient(135deg, #1e293b 0%, #3a2a1a 100%)"
        border_color = "#f97316"
    
    # Confidence badge
    if result["confidence"] == "HIGH":
        confidence_badge = "🟢 HIGH"
        confidence_color = "#10b981"
    elif result["confidence"] == "MEDIUM":
        confidence_badge = "🟡 MEDIUM"
        confidence_color = "#fbbf24"
    else:
        confidence_badge = "🔴 LOW"
        confidence_color = "#ef4444"
    
    # Draw probability gauge (0-100%)
    draw_pct = result["draw_probability"] * 100
    
    st.markdown(f"""
    <style>
    .prediction-card {{
        background: {pred_bg};
        border-left: 6px solid {border_color};
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    .prediction-title {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {pred_color};
        margin-bottom: 0.5rem;
    }}
    .prediction-subtitle {{
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }}
    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    .stat-card {{
        background: #0f172a;
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid #334155;
    }}
    .stat-label {{
        font-size: 0.7rem;
        text-transform: uppercase;
        color: #94a3b8;
    }}
    .stat-value {{
        font-size: 1.25rem;
        font-weight: bold;
        color: #fbbf24;
    }}
    .stat-value-critical {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {pred_color};
    }}
    .gauge-container {{
        background: #1e293b;
        border-radius: 12px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }}
    .gauge-bar {{
        background: {pred_color};
        height: 8px;
        border-radius: 4px;
        width: {draw_pct}%;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-title">{result['prediction']}</div>
        <div class="prediction-subtitle">Confidence: {confidence_badge}</div>
        
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">DRAW PROBABILITY</div>
                <div class="stat-value-critical">{draw_pct:.1f}%</div>
                <div class="gauge-container"><div class="gauge-bar"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">NO DRAW PROBABILITY</div>
                <div class="stat-value-critical">{(1-draw_pct/100)*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">TOTAL xG</div>
                <div class="stat-value">{result['total_xg']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Expected Goals display
    st.markdown("**🎯 EXPECTED GOALS**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{home_team} xG", f"{result['home_xg']:.2f}")
    with col2:
        st.metric(f"{away_team} xG", f"{result['away_xg']:.2f}")
    
    # Most likely scores
    st.markdown("**📊 MOST LIKELY SCORES**")
    score_cols = st.columns(5)
    for i, ((h, a), prob) in enumerate(list(result['top_scores'].items())[:5]):
        with score_cols[i]:
            st.metric(f"{h}-{a}", f"{prob*100:.1f}%")


def render_betting_card(result: Dict):
    """Render betting recommendation."""
    if result["value_bet"] and result["value_bet"].startswith("YES") and result["prediction"] == "NO DRAW":
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%); border-left: 6px solid #fbbf24; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
            <strong>💰 VALUE BET OPPORTUNITY</strong><br>
            Edge: {result['edge']*100:.1f}%<br>
            Recommended Stake: ${result['recommended_stake']:.2f} (if bankroll entered)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("⚠️ No value bet detected. Consider skipping or reducing stake.")


def render_team_inputs(home_team: str, away_team: str, is_home: bool):
    """Render input fields for a team."""
    if is_home:
        label = f"🏠 {home_team}"
        default_scored = 1.60
        default_conceded = 1.20
    else:
        label = f"✈️ {away_team}"
        default_scored = 1.30
        default_conceded = 1.40
    
    col1, col2 = st.columns(2)
    with col1:
        scored = st.number_input(f"{label} Goals Scored", 0.0, 4.0, default_scored, 0.05, key=f"{home_team if is_home else away_team}_scored")
    with col2:
        conceded = st.number_input(f"{label} Goals Conceded", 0.0, 4.0, default_conceded, 0.05, key=f"{home_team if is_home else away_team}_conceded")
    
    return scored, conceded


# ============================================================================
# SECTION 6: MAIN APP
# ============================================================================

def main():
    st.title("🎯 No Draw Predictor")
    st.caption("Predicts when a football match is unlikely to end in a draw | Poisson + ZIP Model")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        league_options = list(LEAGUE_DATABASE.keys()) + ["Custom League"]
        selected_league = st.selectbox("Select League", league_options)
        
        if selected_league == "Custom League":
            league_avg = st.number_input("League Avg Goals/Game", 2.0, 4.0, 2.70, 0.05)
            threshold = st.slider("Draw Threshold", 0.10, 0.35, 0.22, 0.01)
            zip_factor = st.slider("ZIP Factor", 0.00, 0.10, 0.05, 0.01)
            league_name = "Custom"
            league_avg_val = league_avg
            threshold_val = threshold
            zip_val = zip_factor
        else:
            league_data = LEAGUE_DATABASE[selected_league]
            league_name = selected_league
            league_avg_val = league_data["avg_goals"]
            threshold_val = league_data["threshold"]
            zip_val = league_data["zip_factor"]
        
        st.divider()
        
        st.subheader("💰 Bankroll Management")
        use_bankroll = st.checkbox("Use Bankroll Management", value=False)
        bankroll = None
        if use_bankroll:
            bankroll = st.number_input("Bankroll ($)", 100, 10000, 1000, 100)
        
        st.divider()
        
        st.subheader("📊 Advanced Options")
        use_zip = st.checkbox("Apply ZIP Adjustment", value=True)
        use_xg = st.checkbox("Use xG Data (if available)", value=False)
        
        st.caption("Thresholds are league-specific. ZIP factors adjust for 0-0 draws.")
    
    # Main content
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Home Team Name", "Bayern Munich")
    with col2:
        away_team = st.text_input("Away Team Name", "Darmstadt")
    
    st.divider()
    
    st.subheader("📊 Team Statistics (Last 5-6 Matches)")
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"**🏠 {home_team}**")
        home_scored, home_conceded = render_team_inputs(home_team, away_team, True)
        
        if use_xg:
            home_xg = st.number_input(f"{home_team} xG Avg", 0.5, 3.0, 1.80, 0.05, key="home_xg")
        else:
            home_xg = None
    with col_right:
        st.markdown(f"**✈️ {away_team}**")
        away_scored, away_conceded = render_team_inputs(home_team, away_team, False)
        
        if use_xg:
            away_xg = st.number_input(f"{away_team} xG Avg", 0.5, 3.0, 1.20, 0.05, key="away_xg")
        else:
            away_xg = None
    
    st.divider()
    
    st.subheader("💰 Bookmaker Odds (Optional)")
    col_odds1, col_odds2 = st.columns(2)
    with col_odds1:
        home_odds = st.number_input(f"{home_team} Win Odds", 1.01, 100.0, 1.40, 0.05)
    with col_odds2:
        away_odds = st.number_input(f"{away_team} Win Odds", 1.01, 100.0, 7.50, 0.05)
    
    # Predict button
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            result = predict_no_draw(
                home_goals_scored_avg=home_scored,
                home_goals_conceded_avg=home_conceded,
                away_goals_scored_avg=away_scored,
                away_goals_conceded_avg=away_conceded,
                league_name=league_name if league_name != "Custom" else None,
                league_avg_goals=league_avg_val if league_name == "Custom" else None,
                home_xg_avg=home_xg,
                away_xg_avg=away_xg,
                use_zip=use_zip,
                custom_threshold=threshold_val if league_name == "Custom" else None,
                home_odds=home_odds,
                away_odds=away_odds,
                bankroll=bankroll,
                use_xg_if_available=use_xg
            )
        
        # Display results
        render_prediction_card(result, home_team, away_team)
        
        st.divider()
        
        # Betting recommendation
        if result["value_bet"] and result["value_bet"].startswith("YES") and result["prediction"] == "NO DRAW":
            st.success(f"✅ VALUE BET: {result['value_bet']} (Edge: {result['edge']*100:.1f}%)")
            if result["recommended_stake"]:
                st.info(f"💰 Recommended Stake: ${result['recommended_stake']:.2f}")
        elif result["prediction"] == "NO DRAW":
            st.warning(f"⚠️ No value edge detected (Edge: {result['edge']*100:.1f}%). Consider smaller stake.")
        else:
            st.info("📝 No bet recommended. Draw likely.")
        
        # Show debug info in expander
        with st.expander("📊 Detailed Statistics"):
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("League Avg Goals", result['league_avg_goals'])
                st.metric("Threshold", result['threshold'])
                st.metric("ZIP Factor", result['zip_factor_used'])
            with col_d2:
                st.metric("Home Attack", result['home_goals_avg'])
                st.metric("Home Defense", result['home_conceded_avg'])
                st.metric("Home xG", result['home_xg'])
            with col_d3:
                st.metric("Away Attack", result['away_goals_avg'])
                st.metric("Away Defense", result['away_conceded_avg'])
                st.metric("Away xG", result['away_xg'])
            
            st.markdown("**Most Likely Scores**")
            score_df = pd.DataFrame([
                {"Score": f"{h}-{a}", "Probability": f"{p*100:.2f}%"}
                for (h, a), p in list(result['top_scores'].items())[:10]
            ])
            st.dataframe(score_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
