"""
Integrated xG Football Predictor
Logic: Base xG + Clean Sheets + Interceptions + Saves
No extra stats. No clearances. No added rules.
"""

import streamlit as st
import math

st.set_page_config(
    page_title="Integrated xG Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS STYLES
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1000px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #334155;
    }
    .adjustment-list {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CORE LOGIC (ONLY WHAT YOU SPECIFIED)
# ============================================================================

def calculate_base_xg(home_goals: float, home_conceded: float, away_goals: float, away_conceded: float) -> tuple:
    """Base xG = (Goals Scored + Opponent Conceded) / 2"""
    home_xg = (home_goals + away_conceded) / 2
    away_xg = (away_goals + home_conceded) / 2
    return home_xg, away_xg


def apply_modifiers(
    home_xg: float,
    away_xg: float,
    home_clean_sheets: int,
    away_clean_sheets: int,
    home_interceptions: float,
    away_interceptions: float,
    home_saves: float,
    away_saves: float,
    home_conceded: float,
    away_conceded: float
) -> dict:
    """
    Apply ONLY the three specified modifiers:
    1. Clean Sheet Factor (>10 clean sheets each → ×0.90)
    2. Interception Penalty (diff ≥5 → opponent -0.15)
    3. Save Penalty (>3 saves AND <1 conceded → opponent -0.10)
    """
    adjustments = []
    
    # Copy base values
    home_xg_adj = home_xg
    away_xg_adj = away_xg
    
    # Modifier 1: Interception Penalty
    interception_diff = home_interceptions - away_interceptions
    if interception_diff >= 5:
        away_xg_adj -= 0.15
        adjustments.append(f"Interceptions: Home ({home_interceptions:.1f}) > Away ({away_interceptions:.1f}) by {interception_diff:.1f} → Away xG -0.15")
    elif interception_diff <= -5:
        home_xg_adj -= 0.15
        adjustments.append(f"Interceptions: Away ({away_interceptions:.1f}) > Home ({home_interceptions:.1f}) by {abs(interception_diff):.1f} → Home xG -0.15")
    
    # Modifier 2: Save Reliability Penalty
    if home_saves > 3.0 and home_conceded < 1.0:
        away_xg_adj -= 0.10
        adjustments.append(f"Saves: Home ({home_saves:.1f}) >3.0 & conceded ({home_conceded:.1f}) <1.0 → Away xG -0.10")
    if away_saves > 3.0 and away_conceded < 1.0:
        home_xg_adj -= 0.10
        adjustments.append(f"Saves: Away ({away_saves:.1f}) >3.0 & conceded ({away_conceded:.1f}) <1.0 → Home xG -0.10")
    
    # Floor (safety net - xG never 0)
    home_xg_adj = max(0.2, home_xg_adj)
    away_xg_adj = max(0.2, away_xg_adj)
    
    # Modifier 3: Clean Sheet Factor
    if home_clean_sheets > 10 and away_clean_sheets > 10:
        cs_factor = 0.90
        adjustments.append(f"Clean sheets: Both >10 ({home_clean_sheets}, {away_clean_sheets}) → individual xG × 0.90")
    else:
        cs_factor = 1.0
    
    # Apply CS factor to individual xG
    home_xg_final = home_xg_adj * cs_factor
    away_xg_final = away_xg_adj * cs_factor
    total_xg_final = home_xg_final + away_xg_final
    
    return {
        "home_xg_final": round(home_xg_final, 2),
        "away_xg_final": round(away_xg_final, 2),
        "total_xg_final": round(total_xg_final, 2),
        "adjustments": adjustments,
    }


def poisson_probability(lmbda: float, k: int) -> float:
    """Poisson distribution formula."""
    if lmbda == 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)


def predict_match(
    home_goals: float,
    home_conceded: float,
    home_clean_sheets: int,
    home_interceptions: float,
    home_saves: float,
    away_goals: float,
    away_conceded: float,
    away_clean_sheets: int,
    away_interceptions: float,
    away_saves: float,
    max_goals: int = 5
) -> dict:
    """
    Complete prediction using only the specified logic.
    """
    
    # Step 1: Base xG
    base_home_xg, base_away_xg = calculate_base_xg(
        home_goals, home_conceded, away_goals, away_conceded
    )
    
    # Step 2: Apply modifiers
    modifiers = apply_modifiers(
        base_home_xg, base_away_xg,
        home_clean_sheets, away_clean_sheets,
        home_interceptions, away_interceptions,
        home_saves, away_saves,
        home_conceded, away_conceded
    )
    
    home_xg = modifiers["home_xg_final"]
    away_xg = modifiers["away_xg_final"]
    total_xg = modifiers["total_xg_final"]
    
    # Step 3: Over/Under decision
    if total_xg > 2.5:
        over_under = "Over 2.5 Goals"
    elif total_xg < 2.5:
        over_under = "Under 2.5 Goals"
    else:
        over_under = "Under 2.5 Goals (Lean)"
    
    # Step 4: Poisson probabilities
    home_probs = [poisson_probability(home_xg, k) for k in range(max_goals + 1)]
    away_probs = [poisson_probability(away_xg, k) for k in range(max_goals + 1)]
    
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    scorelines = {}
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = home_probs[h] * away_probs[a]
            scorelines[f"{h}-{a}"] = prob
            if h > a:
                home_win += prob
            elif h == a:
                draw += prob
            else:
                away_win += prob
    
    # Most likely scoreline
    most_likely = max(scorelines, key=scorelines.get)
    
    # Winner prediction
    if home_win > away_win and home_win > draw:
        winner = "Home Win"
    elif away_win > home_win and away_win > draw:
        winner = "Away Win"
    else:
        winner = "Draw"
    
    return {
        "base_home_xg": round(base_home_xg, 2),
        "base_away_xg": round(base_away_xg, 2),
        "home_xg": home_xg,
        "away_xg": away_xg,
        "total_xg": total_xg,
        "over_under": over_under,
        "winner": winner,
        "home_win_prob": round(home_win * 100, 1),
        "draw_prob": round(draw * 100, 1),
        "away_win_prob": round(away_win * 100, 1),
        "most_likely_score": most_likely,
        "most_likely_prob": round(scorelines[most_likely] * 100, 1),
        "adjustments": modifiers["adjustments"],
        "top_scorelines": dict(sorted(scorelines.items(), key=lambda x: -x[1])[:5]),
    }


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_team_inputs(team_name: str, is_home: bool, default_values: dict = None):
    """Render input fields for a team."""
    if default_values is None:
        default_values = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚽ Attacking**")
        goals_per_game = st.number_input(
            "Goals scored per game",
            min_value=0.0, max_value=5.0, step=0.1,
            value=default_values.get("goals_per_game", 1.5),
            key=f"{'home' if is_home else 'away'}_goals"
        )
        
        st.markdown("**🛡️ Defensive Activity**")
        interceptions_per_game = st.number_input(
            "Interceptions per game",
            min_value=0.0, max_value=50.0, step=0.1,
            value=default_values.get("interceptions", 25.0),
            key=f"{'home' if is_home else 'away'}_interceptions"
        )
        saves_per_game = st.number_input(
            "Saves per game",
            min_value=0.0, max_value=10.0, step=0.1,
            value=default_values.get("saves", 2.5),
            key=f"{'home' if is_home else 'away'}_saves"
        )
    
    with col2:
        st.markdown("**🛡️ Defending**")
        conceded_per_game = st.number_input(
            "Goals conceded per game",
            min_value=0.0, max_value=5.0, step=0.1,
            value=default_values.get("conceded_per_game", 1.2),
            key=f"{'home' if is_home else 'away'}_conceded"
        )
        clean_sheets = st.number_input(
            "Clean sheets (season total)",
            min_value=0, max_value=50, step=1,
            value=default_values.get("clean_sheets", 10),
            key=f"{'home' if is_home else 'away'}_clean_sheets"
        )
    
    return {
        "goals_per_game": goals_per_game,
        "conceded_per_game": conceded_per_game,
        "clean_sheets": clean_sheets,
        "interceptions": interceptions_per_game,
        "saves": saves_per_game,
    }


def render_prediction(result: dict, home_name: str, away_name: str):
    """Render prediction results."""
    
    st.markdown(f"### 🎯 {home_name} vs {away_name}")
    
    # xG display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{home_name} xG (base)", result['base_home_xg'])
        st.metric(f"{home_name} xG (final)", result['home_xg'])
    with col2:
        st.metric(f"{away_name} xG (base)", result['base_away_xg'])
        st.metric(f"{away_name} xG (final)", result['away_xg'])
    with col3:
        st.metric("Total xG", result['total_xg'])
    
    # Adjustments
    if result['adjustments']:
        st.markdown("**📝 Adjustments Applied:**")
        for adj in result['adjustments']:
            st.markdown(f"<div class='adjustment-list'>• {adj}</div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Over/Under
    st.markdown("### 📈 Over/Under 2.5 Prediction")
    st.markdown(f"""
    <div class="prediction-card">
        <strong>🔵 {result['over_under']}</strong><br>
        Total expected goals: {result['total_xg']}
    </div>
    """, unsafe_allow_html=True)
    
    # Winner probabilities
    st.markdown("### 🏆 Match Outcome Probabilities")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{home_name} Win", f"{result['home_win_prob']}%")
    with col2:
        st.metric("Draw", f"{result['draw_prob']}%")
    with col3:
        st.metric(f"{away_name} Win", f"{result['away_win_prob']}%")
    
    st.markdown(f"""
    <div class="prediction-card">
        <strong>✅ Most likely: {result['winner']}</strong><br>
        Most likely score: {result['most_likely_score']} ({result['most_likely_prob']}%)
    </div>
    """, unsafe_allow_html=True)
    
    # Top scorelines
    with st.expander("📊 Top 5 Most Likely Scorelines"):
        for score, prob in result['top_scorelines'].items():
            st.write(f"{score}: {prob * 100:.1f}%")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Integrated xG Football Predictor")
    st.caption("Base xG + Clean Sheets + Interceptions + Saves")
    
    st.divider()
    
    # Team inputs
    col_home, col_away = st.columns(2)
    
    with col_home:
        st.markdown("## 🏠 HOME TEAM")
        home_name = st.text_input("Team Name", "Home Team", key="home_name")
        
        home_defaults = {
            "goals_per_game": 1.4,
            "conceded_per_game": 0.8,
            "clean_sheets": 11,
            "interceptions": 32.7,
            "saves": 2.3,
        }
        
        home_stats = render_team_inputs(home_name, True, home_defaults)
    
    with col_away:
        st.markdown("## ✈️ AWAY TEAM")
        away_name = st.text_input("Team Name", "Away Team", key="away_name")
        
        away_defaults = {
            "goals_per_game": 2.2,
            "conceded_per_game": 0.7,
            "clean_sheets": 12,
            "interceptions": 27.8,
            "saves": 2.1,
        }
        
        away_stats = render_team_inputs(away_name, False, away_defaults)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            result = predict_match(
                home_goals=home_stats["goals_per_game"],
                home_conceded=home_stats["conceded_per_game"],
                home_clean_sheets=home_stats["clean_sheets"],
                home_interceptions=home_stats["interceptions"],
                home_saves=home_stats["saves"],
                away_goals=away_stats["goals_per_game"],
                away_conceded=away_stats["conceded_per_game"],
                away_clean_sheets=away_stats["clean_sheets"],
                away_interceptions=away_stats["interceptions"],
                away_saves=away_stats["saves"]
            )
            
            render_prediction(result, home_name, away_name)
    
    st.divider()
    st.caption("""
    **Exact Logic (no additions):**\n
    1. Base xG = (Goals Scored + Opponent Conceded) / 2\n
    2. If interceptions diff ≥5 → opponent xG -0.15\n
    3. If saves >3.0 AND conceded <1.0 → opponent xG -0.10\n
    4. If both teams have >10 clean sheets → individual xG × 0.90\n
    5. Poisson distribution for scoreline probabilities\n
    6. Over/Under: total xG >2.5 = Over, <2.5 = Under
    """)


if __name__ == "__main__":
    main()
