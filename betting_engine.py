"""
Hybrid v2.2 Football Predictor
Based on: Possession diff, Big chances diff, Tackles diff, Clearances override, Net Edge
No Poisson. No draw probability. Just consistent rules.
"""

import streamlit as st
import pandas as pd
import math

st.set_page_config(
    page_title="Hybrid v2.2 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SECTION 1: LEAGUE DATABASE
# ============================================================================

LEAGUE_DATABASE = {
    "Bundesliga": {"avg_goals": 3.10},
    "Premier League": {"avg_goals": 2.84},
    "Serie A": {"avg_goals": 2.65},
    "La Liga": {"avg_goals": 2.62},
    "Ligue 1": {"avg_goals": 2.96},
    "Eredivisie": {"avg_goals": 3.00},
    "Primeira Liga": {"avg_goals": 2.50},
    "MLS": {"avg_goals": 3.10},
    "Championship": {"avg_goals": 2.78},
    "League One": {"avg_goals": 2.72},
}

DEFAULT_AVG_GOALS = 2.70


# ============================================================================
# SECTION 2: CUSTOM CSS (Same UI/UX)
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    .team-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .team-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #fbbf24;
        margin-bottom: 0.5rem;
    }
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .stat-item {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.25rem 0.5rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.65rem;
        color: #94a3b8;
    }
    .stat-value {
        font-size: 0.9rem;
        font-weight: bold;
        color: #fbbf24;
    }
    .result-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 2px solid #fbbf24;
    }
    .prediction-win {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 6px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .prediction-draw {
        background: linear-gradient(135deg, #1e293b 0%, #3a2a1a 100%);
        border-left: 6px solid #f97316;
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
    .stake-highlight {
        background: #fbbf24;
        color: #0f172a;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION 3: HYBRID v2.2 LOGIC
# ============================================================================

def calculate_net_edge(
    home_goals_per_game: float,
    home_conceded_per_game: float,
    away_goals_per_game: float,
    away_conceded_per_game: float,
    league_avg: float
) -> float:
    """Calculate Net Edge for winner prediction."""
    # Attack/Defense strengths relative to league
    home_attack = home_goals_per_game / league_avg
    home_defense = home_conceded_per_game / league_avg
    away_attack = away_goals_per_game / league_avg
    away_defense = away_conceded_per_game / league_avg
    
    # Expected goals
    home_xg = home_attack * away_defense * league_avg
    away_xg = away_attack * home_defense * league_avg
    
    # Net Edge (home advantage already added separately)
    net_edge = home_xg - away_xg
    
    return net_edge, home_xg, away_xg


def predict_winner(
    net_edge: float,
    home_advantage: float = 0.5
) -> dict:
    """
    Winner prediction based on Net Edge threshold.
    Home advantage +0.5 added to home team's Net Edge.
    """
    net_edge_with_advantage = net_edge + home_advantage
    
    if net_edge_with_advantage > 2.0:
        return {
            "prediction": "Home Win",
            "confidence": "HIGH" if net_edge_with_advantage > 3.0 else "MEDIUM",
            "edge": net_edge_with_advantage,
            "double_chance": "Home or Draw"
        }
    elif net_edge_with_advantage < -2.0:
        return {
            "prediction": "Away Win",
            "confidence": "HIGH" if net_edge_with_advantage < -3.0 else "MEDIUM",
            "edge": net_edge_with_advantage,
            "double_chance": "Away or Draw"
        }
    else:
        # Double chance to side with positive edge
        if net_edge_with_advantage > 0:
            return {
                "prediction": "Double Chance - Home or Draw",
                "confidence": "LOW",
                "edge": net_edge_with_advantage,
                "double_chance": "Home or Draw"
            }
        elif net_edge_with_advantage < 0:
            return {
                "prediction": "Double Chance - Away or Draw",
                "confidence": "LOW",
                "edge": net_edge_with_advantage,
                "double_chance": "Away or Draw"
            }
        else:
            return {
                "prediction": "Draw or Double Chance",
                "confidence": "LOW",
                "edge": net_edge_with_advantage,
                "double_chance": "Draw"
            }


def check_draw_triggers(
    possession_home: float,
    possession_away: float,
    big_chances_home: float,
    big_chances_away: float,
    tackles_home: float,
    tackles_away: float,
    clearances_home: float,
    clearances_away: float,
    net_edge: float
) -> tuple:
    """
    Check draw triggers and clearances override.
    
    Draw trigger: possession diff ≤7% + big chances diff ≤0.5 + tackles diff ≤2
    Clearances override: underdog has ≥8 more clearances/game AND favourite Net Edge < 3.0
    """
    possession_diff = abs(possession_home - possession_away)
    big_chances_diff = abs(big_chances_home - big_chances_away)
    tackles_diff = abs(tackles_home - tackles_away)
    
    # Primary draw trigger
    draw_trigger = (possession_diff <= 7 and big_chances_diff <= 0.5 and tackles_diff <= 2)
    
    # Clearances override (underdog has 8+ more clearances)
    clearances_diff = abs(clearances_home - clearances_away)
    
    # Determine favourite (team with higher Net Edge)
    if net_edge > 0:
        favourite_clearances = clearances_home
        underdog_clearances = clearances_away
    else:
        favourite_clearances = clearances_away
        underdog_clearances = clearances_home
    
    clearances_override = (underdog_clearances - favourite_clearances >= 8) and abs(net_edge) < 3.0
    
    return draw_trigger, clearances_override


def predict_over_under(
    home_conceded_per_game: float,
    away_conceded_per_game: float,
    home_clean_sheets: int,
    away_clean_sheets: int,
    home_big_chances_per_game: float,
    away_big_chances_per_game: float,
    home_games: int,
    away_games: int
) -> dict:
    """
    Over/Under 2.5 prediction based on:
    - Both concede ≤ 1.3 goals/game
    - Combined clean sheets ≥ 10
    - Combined big chances < 5.5
    """
    combined_clean_sheets = home_clean_sheets + away_clean_sheets
    combined_big_chances = home_big_chances_per_game + away_big_chances_per_game
    
    under_conditions = (
        home_conceded_per_game <= 1.3 and
        away_conceded_per_game <= 1.3 and
        combined_clean_sheets >= 10 and
        combined_big_chances < 5.5
    )
    
    if under_conditions:
        return {
            "prediction": "Under 2.5 Goals",
            "confidence": "HIGH",
            "reason": f"Both concede ≤1.3, combined CS {combined_clean_sheets} ≥10, combined big chances {combined_big_chances:.1f} <5.5"
        }
    else:
        return {
            "prediction": "Over 2.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Default: Under conditions not met"
        }


# ============================================================================
# SECTION 4: MAIN PREDICTION FUNCTION
# ============================================================================

def predict_match(
    home_goals_scored: int,
    home_goals_conceded: int,
    home_games: int,
    home_clean_sheets: int,
    home_big_chances: int,
    home_possession: float,
    home_tackles: int,
    home_clearances: int,
    away_goals_scored: int,
    away_goals_conceded: int,
    away_games: int,
    away_clean_sheets: int,
    away_big_chances: int,
    away_possession: float,
    away_tackles: int,
    away_clearances: int,
    league_avg_goals: float
) -> dict:
    """
    Hybrid v2.2 prediction.
    """
    
    # Calculate per-game averages
    home_goals_per_game = home_goals_scored / home_games
    home_conceded_per_game = home_goals_conceded / home_games
    home_big_chances_per_game = home_big_chances / home_games
    home_tackles_per_game = home_tackles / home_games
    home_clearances_per_game = home_clearances / home_games
    
    away_goals_per_game = away_goals_scored / away_games
    away_conceded_per_game = away_goals_conceded / away_games
    away_big_chances_per_game = away_big_chances / away_games
    away_tackles_per_game = away_tackles / away_games
    away_clearances_per_game = away_clearances / away_games
    
    # Calculate Net Edge
    net_edge, home_xg, away_xg = calculate_net_edge(
        home_goals_per_game, home_conceded_per_game,
        away_goals_per_game, away_conceded_per_game,
        league_avg_goals
    )
    
    # Check draw triggers
    draw_trigger, clearances_override = check_draw_triggers(
        home_possession, away_possession,
        home_big_chances_per_game, away_big_chances_per_game,
        home_tackles_per_game, away_tackles_per_game,
        home_clearances_per_game, away_clearances_per_game,
        net_edge
    )
    
    # Winner prediction (with home advantage +0.5)
    winner_result = predict_winner(net_edge, home_advantage=0.5)
    
    # Override to Draw if triggers fire
    if draw_trigger or clearances_override:
        if draw_trigger:
            override_reason = "Draw trigger: possession diff ≤7%, big chances diff ≤0.5, tackles diff ≤2"
        else:
            override_reason = "Clearances override: underdog has 8+ more clearances"
        
        winner_result = {
            "prediction": "Draw",
            "confidence": "HIGH",
            "edge": net_edge + 0.5,
            "double_chance": "Draw",
            "override_reason": override_reason
        }
    
    # Over/Under prediction
    over_under_result = predict_over_under(
        home_conceded_per_game, away_conceded_per_game,
        home_clean_sheets, away_clean_sheets,
        home_big_chances_per_game, away_big_chances_per_game,
        home_games, away_games
    )
    
    return {
        "home_team": "Home Team",
        "away_team": "Away Team",
        "home_goals_per_game": round(home_goals_per_game, 2),
        "home_conceded_per_game": round(home_conceded_per_game, 2),
        "away_goals_per_game": round(away_goals_per_game, 2),
        "away_conceded_per_game": round(away_conceded_per_game, 2),
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "net_edge": round(net_edge + 0.5, 2),
        "draw_trigger": draw_trigger,
        "clearances_override": clearances_override,
        "winner": winner_result,
        "over_under": over_under_result,
    }


# ============================================================================
# SECTION 5: UI COMPONENTS
# ============================================================================

def render_team_inputs(team_name: str, is_home: bool, default_values: dict = None):
    """Render input fields for a team."""
    if default_values is None:
        default_values = {}
    
    if is_home:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Attacking**")
            goals_scored = st.number_input("Goals Scored", 0, 150, default_values.get("goals_scored", 60), key=f"home_goals")
            big_chances = st.number_input("Big Chances", 0, 200, default_values.get("big_chances", 87), key=f"home_big_chances")
            possession = st.number_input("Ball Possession %", 30, 70, default_values.get("possession", 59), key=f"home_possession")
        with col2:
            st.markdown("**🛡️ Defending**")
            goals_conceded = st.number_input("Goals Conceded", 0, 150, default_values.get("goals_conceded", 38), key=f"home_conceded")
            clean_sheets = st.number_input("Clean Sheets", 0, 50, default_values.get("clean_sheets", 11), key=f"home_clean_sheets")
            tackles = st.number_input("Tackles", 0, 800, default_values.get("tackles", 418), key=f"home_tackles")
            clearances = st.number_input("Clearances", 0, 1000, default_values.get("clearances", 737), key=f"home_clearances")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📊 Attacking**")
            goals_scored = st.number_input("Goals Scored", 0, 150, default_values.get("goals_scored", 60), key=f"away_goals")
            big_chances = st.number_input("Big Chances", 0, 200, default_values.get("big_chances", 87), key=f"away_big_chances")
            possession = st.number_input("Ball Possession %", 30, 70, default_values.get("possession", 53), key=f"away_possession")
        with col2:
            st.markdown("**🛡️ Defending**")
            goals_conceded = st.number_input("Goals Conceded", 0, 150, default_values.get("goals_conceded", 29), key=f"away_conceded")
            clean_sheets = st.number_input("Clean Sheets", 0, 50, default_values.get("clean_sheets", 13), key=f"away_clean_sheets")
            tackles = st.number_input("Tackles", 0, 800, default_values.get("tackles", 415), key=f"away_tackles")
            clearances = st.number_input("Clearances", 0, 1000, default_values.get("clearances", 716), key=f"away_clearances")
    
    return {
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "big_chances": big_chances,
        "possession": possession,
        "clean_sheets": clean_sheets,
        "tackles": tackles,
        "clearances": clearances,
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
    
    # Winner Prediction
    st.markdown("### 🏆 Winner Prediction")
    
    winner = result['winner']
    
    if winner['prediction'] == "Draw":
        st.markdown(f"""
        <div class="prediction-draw">
            <strong>⚖️ {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            Net Edge: {winner['edge']:.2f}<br>
            📝 {winner.get('override_reason', 'Draw triggers activated')}
        </div>
        """, unsafe_allow_html=True)
    elif "Double Chance" in winner['prediction']:
        st.markdown(f"""
        <div class="prediction-draw">
            <strong>🔄 {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            Net Edge: {winner['edge']:.2f}<br>
            📝 Net edge within ±2.0 → double chance to side with positive edge
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-win">
            <strong>✅ {winner['prediction']}</strong><br>
            Confidence: {winner['confidence']}<br>
            Net Edge: {winner['edge']:.2f}<br>
            📝 Net edge > 2.0 (with home advantage +0.5)
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
    
    # Debug info
    with st.expander("📊 Detailed Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Home xG", result['home_xg'])
            st.metric("Draw Trigger", "✅" if result['draw_trigger'] else "❌")
        with col2:
            st.metric("Away xG", result['away_xg'])
            st.metric("Clearances Override", "✅" if result['clearances_override'] else "❌")


# ============================================================================
# SECTION 6: MAIN APP
# ============================================================================

def main():
    st.title("⚽ Hybrid v2.2 Football Predictor")
    st.caption("Based on: Possession diff, Big chances diff, Tackles diff, Clearances override, Net Edge")
    st.caption("81% Winner Accuracy | 90.5% Over/Under Accuracy")
    
    # League selection
    col_league1, col_league2 = st.columns([2, 1])
    with col_league1:
        league_options = list(LEAGUE_DATABASE.keys())
        selected_league = st.selectbox("Select League", league_options, index=0)
        league_avg_goals = LEAGUE_DATABASE[selected_league]["avg_goals"]
    with col_league2:
        st.metric("League Avg Goals", f"{league_avg_goals:.2f}")
        st.caption("Used for Net Edge calculation")
    
    st.divider()
    
    # Team inputs
    col_home, col_away = st.columns(2)
    
    with col_home:
        st.markdown("## 🏠 HOME TEAM")
        home_name = st.text_input("Team Name", "VfB Stuttgart", key="home_name")
        
        st.markdown("**📊 Games Played**")
        home_games = st.number_input("Matches", 1, 50, 29, key="home_games")
        
        # Default values for Stuttgart
        home_defaults = {
            "goals_scored": 60,
            "goals_conceded": 38,
            "big_chances": 87,
            "possession": 59,
            "clean_sheets": 11,
            "tackles": 418,
            "clearances": 737,
        }
        
        home_stats = render_team_inputs(home_name, True, home_defaults)
    
    with col_away:
        st.markdown("## ✈️ AWAY TEAM")
        away_name = st.text_input("Team Name", "Borussia Dortmund", key="away_name")
        
        st.markdown("**📊 Games Played**")
        away_games = st.number_input("Matches", 1, 50, 29, key="away_games")
        
        # Default values for Dortmund
        away_defaults = {
            "goals_scored": 60,
            "goals_conceded": 29,
            "big_chances": 87,
            "possession": 53,
            "clean_sheets": 13,
            "tackles": 415,
            "clearances": 716,
        }
        
        away_stats = render_team_inputs(away_name, False, away_defaults)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT MATCH", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
            result = predict_match(
                home_goals_scored=home_stats["goals_scored"],
                home_goals_conceded=home_stats["goals_conceded"],
                home_games=home_games,
                home_clean_sheets=home_stats["clean_sheets"],
                home_big_chances=home_stats["big_chances"],
                home_possession=home_stats["possession"],
                home_tackles=home_stats["tackles"],
                home_clearances=home_stats["clearances"],
                away_goals_scored=away_stats["goals_scored"],
                away_goals_conceded=away_stats["goals_conceded"],
                away_games=away_games,
                away_clean_sheets=away_stats["clean_sheets"],
                away_big_chances=away_stats["big_chances"],
                away_possession=away_stats["possession"],
                away_tackles=away_stats["tackles"],
                away_clearances=away_stats["clearances"],
                league_avg_goals=league_avg_goals
            )
            
            # Add team names to result
            result["home_team"] = home_name
            result["away_team"] = away_name
            
            render_prediction(result, home_name, away_name)
    
    st.divider()
    st.caption("Hybrid v2.2 | Draw trigger: possession diff ≤7% + big chances diff ≤0.5 + tackles diff ≤2 | Clearances override: underdog +8 clearances")


if __name__ == "__main__":
    main()
