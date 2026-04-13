"""
Integrated xG Football Predictor
Base xG + Resistance Modifiers (Clean Sheets, Interceptions, Saves, Clearances)
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
# SECTION 1: CORE LOGIC
# ============================================================================

def calculate_base_xg(home_goals: float, home_conceded: float, away_goals: float, away_conceded: float) -> tuple:
    """Calculate base expected goals using simple average."""
    home_xg = (home_goals + away_conceded) / 2
    away_xg = (away_goals + home_conceded) / 2
    return home_xg, away_xg


def apply_resistance_modifiers(
    home_xg: float,
    away_xg: float,
    home_clean_sheets: int,
    away_clean_sheets: int,
    home_interceptions: float,
    away_interceptions: float,
    home_saves: float,
    away_saves: float,
    home_conceded: float,
    away_conceded: float,
    home_clearances: float,
    away_clearances: float,
    home_matches: int,
    away_matches: int
) -> dict:
    """
    Apply defensive resistance modifiers to base xG.
    
    Returns adjusted xG and explanation of adjustments.
    """
    adjustments = []
    
    # Calculate clean sheet rate (per game)
    home_cs_rate = home_clean_sheets / home_matches
    away_cs_rate = away_clean_sheets / away_matches
    
    home_xg_adj = home_xg
    away_xg_adj = away_xg
    
    # Modifier 1: Clean Sheet Factor (both teams have high clean sheets)
    if home_clean_sheets > 10 and away_clean_sheets > 10:
        cs_factor = 0.90
        adjustments.append(f"Both teams have >10 clean sheets → total xG reduced by 10%")
    elif home_cs_rate > 0.35 and away_cs_rate > 0.30:
        cs_factor = 0.92
        adjustments.append(f"Both teams have high clean sheet rates ({home_cs_rate:.0%}, {away_cs_rate:.0%}) → total xG reduced by 8%")
    else:
        cs_factor = 1.0
    
    # Modifier 2: Interception Intensity
    interception_diff = home_interceptions - away_interceptions
    if interception_diff >= 5:
        away_xg_adj -= 0.15
        adjustments.append(f"Home interceptions ({home_interceptions:.1f}) > Away ({away_interceptions:.1f}) by {interception_diff:.1f} → Away xG -0.15")
    elif interception_diff <= -5:
        home_xg_adj -= 0.15
        adjustments.append(f"Away interceptions ({away_interceptions:.1f}) > Home ({home_interceptions:.1f}) by {abs(interception_diff):.1f} → Home xG -0.15")
    
    # Modifier 3: Save Reliability (over-performing defence)
    if home_saves > 3.0 and home_conceded < 1.0:
        away_xg_adj -= 0.10
        adjustments.append(f"Home saves ({home_saves:.1f}) high with low conceded ({home_conceded:.1f}) → Away xG -0.10")
    if away_saves > 3.0 and away_conceded < 1.0:
        home_xg_adj -= 0.10
        adjustments.append(f"Away saves ({away_saves:.1f}) high with low conceded ({away_conceded:.1f}) → Home xG -0.10")
    
    # Modifier 4: Clearances Factor (deep defensive block)
    clearance_diff = home_clearances - away_clearances
    if clearance_diff >= 10:
        away_xg_adj -= 0.10
        adjustments.append(f"Home clearances ({home_clearances:.1f}) > Away ({away_clearances:.1f}) by {clearance_diff:.1f} → Away xG -0.10")
    elif clearance_diff <= -10:
        home_xg_adj -= 0.10
        adjustments.append(f"Away clearances ({away_clearances:.1f}) > Home ({home_clearances:.1f}) by {abs(clearance_diff):.1f} → Home xG -0.10")
    
    # Ensure no negative xG
    home_xg_adj = max(0.2, home_xg_adj)
    away_xg_adj = max(0.2, away_xg_adj)
    
    # Apply clean sheet factor to total
    total_xg_adj = (home_xg_adj + away_xg_adj) * cs_factor
    
    return {
        "home_xg_adj": round(home_xg_adj, 2),
        "away_xg_adj": round(away_xg_adj, 2),
        "total_xg_adj": round(total_xg_adj, 2),
        "cs_factor": cs_factor,
        "adjustments": adjustments,
    }


def poisson_probability(lmbda: float, k: int) -> float:
    """Calculate Poisson probability."""
    if lmbda == 0:
        return 1.0 if k == 0 else 0.0
    return (math.exp(-lmbda) * (lmbda ** k)) / math.factorial(k)


def predict_match(
    home_goals: float,
    home_conceded: float,
    home_clean_sheets: int,
    home_matches: int,
    home_interceptions: float,
    home_saves: float,
    home_clearances: float,
    away_goals: float,
    away_conceded: float,
    away_clean_sheets: int,
    away_matches: int,
    away_interceptions: float,
    away_saves: float,
    away_clearances: float,
    max_goals: int = 5
) -> dict:
    """
    Integrated prediction using base xG + resistance modifiers.
    """
    
    # Step 1: Base xG
    base_home_xg, base_away_xg = calculate_base_xg(
        home_goals, home_conceded, away_goals, away_conceded
    )
    
    # Step 2: Apply modifiers
    modifiers = apply_resistance_modifiers(
        base_home_xg, base_away_xg,
        home_clean_sheets, away_clean_sheets,
        home_interceptions, away_interceptions,
        home_saves, away_saves,
        home_conceded, away_conceded,
        home_clearances, away_clearances,
        home_matches, away_matches
    )
    
    home_xg = modifiers["home_xg_adj"]
    away_xg = modifiers["away_xg_adj"]
    total_xg = modifiers["total_xg_adj"]
    
    # Step 3: Over/Under decision
    if total_xg > 2.5:
        over_under = "Over 2.5 Goals"
        ou_confidence = "HIGH" if total_xg > 3.0 else "MEDIUM"
    elif total_xg < 2.5:
        over_under = "Under 2.5 Goals"
        ou_confidence = "HIGH" if total_xg < 2.0 else "MEDIUM"
    else:
        over_under = "Under 2.5 Goals (Lean)"
        ou_confidence = "LOW"
    
    # Step 4: Poisson probabilities
    home_probs = [poisson_probability(home_xg, k) for k in range(max_goals + 1)]
    away_probs = [poisson_probability(away_xg, k) for k in range(max_goals + 1)]
    
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    scorelines = {}
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = home_probs[h] * away_probs[a]
            scorelines[f"{h}-{a}"] = prob
            if h > a:
                home_win_prob += prob
            elif h == a:
                draw_prob += prob
            else:
                away_win_prob += prob
    
    # Most likely score
    most_likely = max(scorelines, key=scorelines.get)
    
    # Winner prediction
    if home_win_prob > away_win_prob and home_win_prob > draw_prob:
        winner = "Home Win"
    elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
        winner = "Away Win"
    else:
        winner = "Draw"
    
    return {
        "base_home_xg": round(base_home_xg, 2),
        "base_away_xg": round(base_away_xg, 2),
        "base_total": round(base_home_xg + base_away_xg, 2),
        "home_xg": home_xg,
        "away_xg": away_xg,
        "total_xg": total_xg,
        "cs_factor": modifiers["cs_factor"],
        "adjustments": modifiers["adjustments"],
        "over_under": over_under,
        "ou_confidence": ou_confidence,
        "winner": winner,
        "home_win_prob": round(home_win_prob * 100, 1),
        "draw_prob": round(draw_prob * 100, 1),
        "away_win_prob": round(away_win_prob * 100, 1),
        "most_likely_score": most_likely,
        "most_likely_prob": round(scorelines[most_likely] * 100, 1),
        "top_scorelines": dict(sorted(scorelines.items(), key=lambda x: -x[1])[:5]),
    }


# ============================================================================
# SECTION 2: UI COMPONENTS
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
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
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)


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
        matches = st.number_input(
            "Matches played",
            min_value=1, max_value=50, step=1,
            value=default_values.get("matches", 30),
            key=f"{'home' if is_home else 'away'}_matches"
        )
        clearances_per_game = st.number_input(
            "Clearances per game",
            min_value=0.0, max_value=60.0, step=0.1,
            value=default_values.get("clearances", 20.0),
            key=f"{'home' if is_home else 'away'}_clearances"
        )
    
    return {
        "goals_per_game": goals_per_game,
        "conceded_per_game": conceded_per_game,
        "clean_sheets": clean_sheets,
        "matches": matches,
        "interceptions": interceptions_per_game,
        "saves": saves_per_game,
        "clearances": clearances_per_game,
    }


def render_prediction(result: dict, home_name: str, away_name: str):
    """Render prediction results."""
    
    st.markdown(f"### 🎯 {home_name} vs {away_name}")
    
    # xG comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{home_name} xG (base)", result['base_home_xg'])
        st.metric(f"{home_name} xG (adj)", result['home_xg'])
    with col2:
        st.metric(f"{away_name} xG (base)", result['base_away_xg'])
        st.metric(f"{away_name} xG (adj)", result['away_xg'])
    with col3:
        st.metric("Total xG (base)", result['base_total'])
        st.metric("Total xG (adj)", result['total_xg'])
    
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
        Confidence: {result['ou_confidence']}<br>
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
# SECTION 3: MAIN APP
# ============================================================================

def main():
    st.title("⚽ Integrated xG Football Predictor")
    st.caption("Base xG + Resistance Modifiers (Clean Sheets, Interceptions, Saves, Clearances)")
    
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
            "matches": 31,
            "interceptions": 32.7,
            "saves": 2.3,
            "clearances": 25.0,
        }
        
        home_stats = render_team_inputs(home_name, True, home_defaults)
    
    with col_away:
        st.markdown("## ✈️ AWAY TEAM")
        away_name = st.text_input("Team Name", "Away Team", key="away_name")
        
        away_defaults = {
            "goals_per_game": 2.2,
            "conceded_per_game": 0.7,
            "clean_sheets": 12,
            "matches": 31,
            "interceptions": 27.8,
            "saves": 2.1,
            "clearances": 22.0,
        }
        
        away_stats = render_team_inputs(away_name, False, away_defaults)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True):
        with st.spinner("Calculating with integrated model..."):
            result = predict_match(
                home_goals=home_stats["goals_per_game"],
                home_conceded=home_stats["conceded_per_game"],
                home_clean_sheets=home_stats["clean_sheets"],
                home_matches=home_stats["matches"],
                home_interceptions=home_stats["interceptions"],
                home_saves=home_stats["saves"],
                home_clearances=home_stats["clearances"],
                away_goals=away_stats["goals_per_game"],
                away_conceded=away_stats["conceded_per_game"],
                away_clean_sheets=away_stats["clean_sheets"],
                away_matches=away_stats["matches"],
                away_interceptions=away_stats["interceptions"],
                away_saves=away_stats["saves"],
                away_clearances=away_stats["clearances"]
            )
            
            render_prediction(result, home_name, away_name)
    
    st.divider()
    st.caption("""
    **How it works:**\n
    1. Base xG = (Goals Scored + Opponent Conceded) / 2\n
    2. Modifiers: Clean Sheets (-8-10%), Interceptions (-0.15 if >5 diff), Saves (-0.10 if >3 saves & <1 conceded), Clearances (-0.10 if >10 diff)\n
    3. Poisson distribution calculates scoreline probabilities\n
    4. Over/Under based on adjusted total xG (>2.5 = Over, <2.5 = Under)
    """)


if __name__ == "__main__":
    main()
