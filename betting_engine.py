"""
Hybrid v2.2 Football Predictor
Based on: Defensive Score + Attacking Score + Draw Trigger + Clearances Override
INPUTS: All per-game averages (as provided in your Sofascore comparisons)
"""

import streamlit as st

st.set_page_config(
    page_title="Hybrid v2.2 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SECTION 1: LEAGUE DATABASE (League Average Conceded per Game)
# ============================================================================

LEAGUE_DATABASE = {
    "Premier League": {"avg_conceded": 1.40},
    "La Liga": {"avg_conceded": 1.25},
    "Bundesliga": {"avg_conceded": 1.50},
    "Serie A": {"avg_conceded": 1.25},
    "Ligue 1": {"avg_conceded": 1.30},
    "Brasileirão": {"avg_conceded": 1.25},
    "Eredivisie": {"avg_conceded": 1.55},
    "Primeira Liga": {"avg_conceded": 1.20},
    "MLS": {"avg_conceded": 1.60},
    "Championship": {"avg_conceded": 1.35},
    "League One": {"avg_conceded": 1.40},
}


# ============================================================================
# SECTION 2: HYBRID v2.2 CORE LOGIC (Using per-game averages directly)
# ============================================================================

def calculate_defensive_score(clean_sheets_per_game: float, conceded_per_game: float, league_avg_conceded: float) -> float:
    """
    Defensive Score = (Clean Sheets per game) × 10 + (League_Avg_Conceded - Conceded_per_Game) × 5
    
    Note: Clean sheets per game = clean_sheets / matches (already provided as float)
    """
    cs_part = clean_sheets_per_game * 10
    conceded_part = (league_avg_conceded - conceded_per_game) * 5
    return cs_part + conceded_part


def calculate_attacking_score(goals_per_game: float, big_chances_per_game: float, big_chances_missed_per_game: float) -> float:
    """
    Attacking Score = Goals_per_Game + (Big_Chances_per_Game - Big_Chances_Missed_per_Game) × 0.3
    """
    return goals_per_game + (big_chances_per_game - big_chances_missed_per_game) * 0.3


def predict_match(
    # Home team stats (all per-game averages, exactly as in your data)
    home_goals_per_game: float,
    home_conceded_per_game: float,
    home_clean_sheets_per_game: float,
    home_big_chances_per_game: float,
    home_big_chances_missed_per_game: float,
    home_possession: float,
    home_tackles_per_game: float,
    home_clearances_per_game: float,
    # Away team stats (all per-game averages)
    away_goals_per_game: float,
    away_conceded_per_game: float,
    away_clean_sheets_per_game: float,
    away_big_chances_per_game: float,
    away_big_chances_missed_per_game: float,
    away_possession: float,
    away_tackles_per_game: float,
    away_clearances_per_game: float,
    # League
    league_avg_conceded: float
) -> dict:
    """
    Hybrid v2.2 prediction based on final agreed rules.
    All inputs are per-game averages (floats) as provided.
    """
    
    # Defensive Scores
    def_score_home = calculate_defensive_score(
        home_clean_sheets_per_game, home_conceded_per_game, league_avg_conceded
    )
    def_score_away = calculate_defensive_score(
        away_clean_sheets_per_game, away_conceded_per_game, league_avg_conceded
    )
    
    # Attacking Scores
    att_score_home = calculate_attacking_score(
        home_goals_per_game, home_big_chances_per_game, home_big_chances_missed_per_game
    )
    att_score_away = calculate_attacking_score(
        away_goals_per_game, away_big_chances_per_game, away_big_chances_missed_per_game
    )
    
    # Total Scores
    home_total_score = def_score_home + att_score_home
    away_total_score = def_score_away + att_score_away
    
    # Net Edge (before home advantage)
    net_edge_raw = home_total_score - away_total_score
    
    # Home advantage +0.5
    net_edge = net_edge_raw + 0.5
    
    # Draw Trigger: possession diff ≤7% AND big chances diff ≤0.5 AND tackles diff ≤2
    possession_diff = abs(home_possession - away_possession)
    big_chances_diff = abs(home_big_chances_per_game - away_big_chances_per_game)
    tackles_diff = abs(home_tackles_per_game - away_tackles_per_game)
    
    draw_trigger = (possession_diff <= 7 and big_chances_diff <= 0.5 and tackles_diff <= 2)
    
    # Clearances Override: underdog has ≥8 more clearances/game AND |net_edge_raw| < 3.0
    if home_total_score > away_total_score:
        favourite_clearances = home_clearances_per_game
        underdog_clearances = away_clearances_per_game
    else:
        favourite_clearances = away_clearances_per_game
        underdog_clearances = home_clearances_per_game
    
    clearances_override = (underdog_clearances - favourite_clearances >= 8) and abs(net_edge_raw) < 3.0
    
    # Winner Prediction
    if draw_trigger:
        winner = {
            "prediction": "Draw",
            "confidence": "HIGH",
            "reason": f"Draw trigger: possession diff {possession_diff:.1f}% ≤7%, big chances diff {big_chances_diff:.1f} ≤0.5, tackles diff {tackles_diff:.1f} ≤2"
        }
    elif clearances_override:
        winner = {
            "prediction": "Draw",
            "confidence": "HIGH",
            "reason": f"Clearances override: underdog has {(underdog_clearances - favourite_clearances):.1f} more clearances/game (≥8) and |net_edge| {abs(net_edge_raw):.1f} < 3.0"
        }
    elif net_edge > 2.0:
        confidence = "HIGH" if net_edge > 3.0 else "MEDIUM"
        winner = {
            "prediction": "Home Win",
            "confidence": confidence,
            "reason": f"Net edge {net_edge:.2f} > 2.0 (with home advantage +0.5)"
        }
    elif net_edge < -2.0:
        confidence = "HIGH" if net_edge < -3.0 else "MEDIUM"
        winner = {
            "prediction": "Away Win",
            "confidence": confidence,
            "reason": f"Net edge {net_edge:.2f} < -2.0 (with home advantage +0.5)"
        }
    else:
        if net_edge > 0:
            winner = {
                "prediction": "Double Chance - Home or Draw",
                "confidence": "LOW",
                "reason": f"Net edge {net_edge:.2f} within ±2.0 → double chance to home"
            }
        elif net_edge < 0:
            winner = {
                "prediction": "Double Chance - Away or Draw",
                "confidence": "LOW",
                "reason": f"Net edge {net_edge:.2f} within ±2.0 → double chance to away"
            }
        else:
            winner = {
                "prediction": "Draw",
                "confidence": "LOW",
                "reason": "Net edge 0.0 within ±2.0"
            }
    
    # Over/Under 2.5 Goals
    combined_clean_sheets_per_game = home_clean_sheets_per_game + away_clean_sheets_per_game
    combined_big_chances = home_big_chances_per_game + away_big_chances_per_game
    
    # Note: For combined CS, we multiply by 10 to get approximate season total equivalent
    # because the rule "combined CS ≥10" assumes season totals (e.g., 29 games)
    # But since your CS are per-game (e.g., 11/31 = 0.355), we need to scale.
    # The rule from our 21-match analysis used actual clean sheet counts (e.g., 11 + 13 = 24)
    # To use per-game averages, we multiply by 30 (approx season length) for comparison.
    combined_clean_sheets_approx = combined_clean_sheets_per_game * 30
    
    under_conditions = (
        home_conceded_per_game <= 1.3 and
        away_conceded_per_game <= 1.3 and
        combined_clean_sheets_approx >= 10 and
        combined_big_chances < 5.5
    )
    
    if under_conditions:
        over_under = {
            "prediction": "Under 2.5 Goals",
            "confidence": "HIGH",
            "reason": f"Both concede ≤1.3, combined CS (~{combined_clean_sheets_approx:.0f}) ≥10, combined big chances {combined_big_chances:.1f} <5.5"
        }
    else:
        over_under = {
            "prediction": "Over 2.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Default: Under conditions not met (concede: {home_conceded_per_game:.1f}/{away_conceded_per_game:.1f}, CS: ~{combined_clean_sheets_approx:.0f}, big chances: {combined_big_chances:.1f})"
        }
    
    return {
        "home_goals_per_game": round(home_goals_per_game, 2),
        "home_conceded_per_game": round(home_conceded_per_game, 2),
        "away_goals_per_game": round(away_goals_per_game, 2),
        "away_conceded_per_game": round(away_conceded_per_game, 2),
        "home_def_score": round(def_score_home, 2),
        "home_att_score": round(att_score_home, 2),
        "away_def_score": round(def_score_away, 2),
        "away_att_score": round(att_score_away, 2),
        "home_total_score": round(home_total_score, 2),
        "away_total_score": round(away_total_score, 2),
        "net_edge_raw": round(net_edge_raw, 2),
        "net_edge": round(net_edge, 2),
        "possession_diff": round(possession_diff, 1),
        "big_chances_diff": round(big_chances_diff, 1),
        "tackles_diff": round(tackles_diff, 1),
        "draw_trigger": draw_trigger,
        "clearances_override": clearances_override,
        "winner": winner,
        "over_under": over_under,
    }


# ============================================================================
# SECTION 3: UI COMPONENTS
# ============================================================================

def render_team_inputs(team_name: str, is_home: bool, default_values: dict = None):
    """Render input fields for a team using per-game averages."""
    if default_values is None:
        default_values = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Attacking (per game)**")
        goals_per_game = st.number_input(
            "Goals per game", 0.0, 5.0, 0.1,
            value=default_values.get("goals_per_game", 2.1),
            key=f"{'home' if is_home else 'away'}_goals"
        )
        big_chances_per_game = st.number_input(
            "Big chances per game", 0.0, 10.0, 0.1,
            value=default_values.get("big_chances_per_game", 3.0),
            key=f"{'home' if is_home else 'away'}_big_chances"
        )
        big_chances_missed_per_game = st.number_input(
            "Big chances missed per game", 0.0, 10.0, 0.1,
            value=default_values.get("big_chances_missed_per_game", 1.5),
            key=f"{'home' if is_home else 'away'}_big_chances_missed"
        )
        possession = st.number_input(
            "Ball Possession %", 30, 70, 1,
            value=default_values.get("possession", 50),
            key=f"{'home' if is_home else 'away'}_possession"
        )
    
    with col2:
        st.markdown("**🛡️ Defending (per game)**")
        conceded_per_game = st.number_input(
            "Goals conceded per game", 0.0, 5.0, 0.1,
            value=default_values.get("conceded_per_game", 1.3),
            key=f"{'home' if is_home else 'away'}_conceded"
        )
        clean_sheets_per_game = st.number_input(
            "Clean sheets per game", 0.0, 1.0, 0.01,
            value=default_values.get("clean_sheets_per_game", 0.35),
            key=f"{'home' if is_home else 'away'}_clean_sheets"
        )
        tackles_per_game = st.number_input(
            "Tackles per game", 0.0, 50.0, 0.1,
            value=default_values.get("tackles_per_game", 15.0),
            key=f"{'home' if is_home else 'away'}_tackles"
        )
        clearances_per_game = st.number_input(
            "Clearances per game", 0.0, 60.0, 0.1,
            value=default_values.get("clearances_per_game", 25.0),
            key=f"{'home' if is_home else 'away'}_clearances"
        )
    
    return {
        "goals_per_game": goals_per_game,
        "conceded_per_game": conceded_per_game,
        "clean_sheets_per_game": clean_sheets_per_game,
        "big_chances_per_game": big_chances_per_game,
        "big_chances_missed_per_game": big_chances_missed_per_game,
        "possession": possession,
        "tackles_per_game": tackles_per_game,
        "clearances_per_game": clearances_per_game,
    }


def render_prediction(result: dict, home_name: str, away_name: str):
    """Render the prediction results."""
    
    st.markdown(f"### 🎯 {home_name} vs {away_name}")
    
    # Stats Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{home_name} Goals/g", result['home_goals_per_game'])
    with col2:
        st.metric(f"{home_name} Conceded/g", result['home_conceded_per_game'])
    with col3:
        st.metric(f"{away_name} Goals/g", result['away_goals_per_game'])
    with col4:
        st.metric(f"{away_name} Conceded/g", result['away_conceded_per_game'])
    
    st.divider()
    
    # Scores display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{home_name} Defensive Score", result['home_def_score'])
        st.metric(f"{home_name} Attacking Score", result['home_att_score'])
    with col2:
        st.metric(f"{away_name} Defensive Score", result['away_def_score'])
        st.metric(f"{away_name} Attacking Score", result['away_att_score'])
    with col3:
        st.metric("Net Edge (raw)", result['net_edge_raw'])
        st.metric("Net Edge (with home +0.5)", result['net_edge'])
    
    st.divider()
    
    # Draw trigger info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Possession Diff", f"{result['possession_diff']}%")
    with col2:
        st.metric("Big Chances Diff", result['big_chances_diff'])
    with col3:
        st.metric("Tackles Diff", result['tackles_diff'])
    
    st.caption(f"Draw Trigger: {'✅ ACTIVE' if result['draw_trigger'] else '❌ Not active'} | Clearances Override: {'✅ ACTIVE' if result['clearances_override'] else '❌ Not active'}")
    
    st.divider()
    
    # Winner Prediction
    st.markdown("### 🏆 Winner Prediction")
    
    winner = result['winner']
    
    if "Draw" in winner['prediction']:
        st.markdown(f"""
        <div class="prediction-draw">
            <strong>⚖️ {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            📝 {winner['reason']}
        </div>
        """, unsafe_allow_html=True)
    elif "Double Chance" in winner['prediction']:
        st.markdown(f"""
        <div class="prediction-draw">
            <strong>🔄 {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            📝 {winner['reason']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-win">
            <strong>✅ {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            📝 {winner['reason']}
        </div>
        """, unsafe_allow_html=True)
    
    # Over/Under Prediction
    st.markdown("### 📈 Over/Under 2.5 Prediction")
    
    ou = result['over_under']
    if "Over" in ou['prediction']:
        st.markdown(f"""
        <div class="prediction-over">
            <strong>🔵 {ou['prediction']}</strong><br>
            Confidence: {ou['confidence']}<br>
            📝 {ou['reason']}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-under">
            <strong>🟣 {ou['prediction']}</strong><br>
            Confidence: {ou['confidence']}<br>
            📝 {ou['reason']}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SECTION 4: CSS STYLES
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    .prediction-draw {
        background: linear-gradient(135deg, #1e293b 0%, #3a2a1a 100%);
        border-left: 6px solid #f97316;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .prediction-win {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 6px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .prediction-over {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a4a 100%);
        border-left: 6px solid #3b82f6;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .prediction-under {
        background: linear-gradient(135deg, #1e293b 0%, #3a2a4a 100%);
        border-left: 6px solid #a855f7;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 5: MAIN APP
# ============================================================================

def main():
    st.title("⚽ Hybrid v2.2 Football Predictor")
    st.caption("Inputs match your Sofascore data format (all per-game averages)")
    
    # League selection
    col_league1, col_league2 = st.columns([2, 1])
    with col_league1:
        league_options = list(LEAGUE_DATABASE.keys())
        selected_league = st.selectbox("Select League", league_options, index=2)
        league_avg_conceded = LEAGUE_DATABASE[selected_league]["avg_conceded"]
    with col_league2:
        st.metric("League Avg Conceded", f"{league_avg_conceded:.2f}")
    
    st.divider()
    
    # Team inputs
    col_home, col_away = st.columns(2)
    
    with col_home:
        st.markdown("## 🏠 HOME TEAM")
        home_name = st.text_input("Team Name", "Mallorca", key="home_name")
        
        home_defaults = {
            "goals_per_game": 1.3,
            "conceded_per_game": 1.5,
            "clean_sheets_per_game": 4/31,  # 0.129
            "big_chances_per_game": 2.1,
            "big_chances_missed_per_game": 1.3,
            "possession": 45,
            "tackles_per_game": 14.3,
            "clearances_per_game": 29.4,
        }
        
        home_stats = render_team_inputs(home_name, True, home_defaults)
    
    with col_away:
        st.markdown("## ✈️ AWAY TEAM")
        away_name = st.text_input("Team Name", "Real Madrid", key="away_name")
        
        away_defaults = {
            "goals_per_game": 2.1,
            "conceded_per_game": 0.9,
            "clean_sheets_per_game": 11/31,  # 0.355
            "big_chances_per_game": 3.4,
            "big_chances_missed_per_game": 2.1,
            "possession": 59,
            "tackles_per_game": 16.7,
            "clearances_per_game": 16.8,
        }
        
        away_stats = render_team_inputs(away_name, False, away_defaults)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True):
        with st.spinner("Calculating with Hybrid v2.2..."):
            result = predict_match(
                home_goals_per_game=home_stats["goals_per_game"],
                home_conceded_per_game=home_stats["conceded_per_game"],
                home_clean_sheets_per_game=home_stats["clean_sheets_per_game"],
                home_big_chances_per_game=home_stats["big_chances_per_game"],
                home_big_chances_missed_per_game=home_stats["big_chances_missed_per_game"],
                home_possession=home_stats["possession"],
                home_tackles_per_game=home_stats["tackles_per_game"],
                home_clearances_per_game=home_stats["clearances_per_game"],
                away_goals_per_game=away_stats["goals_per_game"],
                away_conceded_per_game=away_stats["conceded_per_game"],
                away_clean_sheets_per_game=away_stats["clean_sheets_per_game"],
                away_big_chances_per_game=away_stats["big_chances_per_game"],
                away_big_chances_missed_per_game=away_stats["big_chances_missed_per_game"],
                away_possession=away_stats["possession"],
                away_tackles_per_game=away_stats["tackles_per_game"],
                away_clearances_per_game=away_stats["clearances_per_game"],
                league_avg_conceded=league_avg_conceded
            )
            
            render_prediction(result, home_name, away_name)


if __name__ == "__main__":
    main()
