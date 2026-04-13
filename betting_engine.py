"""
Professional Accuracy-First Football Predictor
No Draw Prediction. Only: Strength Coefficients → xG → Game State → Clean Sheet Conflict → Bet/SKIP
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import Dict, Tuple, Optional, List

st.set_page_config(
    page_title="Accuracy-First Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SECTION 1: LEAGUE DATABASE (2025/26 Season)
# ============================================================================

LEAGUE_DATABASE = {
    "Premier League": {"avg_goals": 2.84, "color": "#37003C"},
    "Serie A": {"avg_goals": 2.65, "color": "#0D2B42"},
    "La Liga": {"avg_goals": 2.62, "color": "#FFD700"},
    "Bundesliga": {"avg_goals": 3.10, "color": "#E1000F"},
    "Ligue 1": {"avg_goals": 2.96, "color": "#1A5B9C"},
    "Primeira Liga": {"avg_goals": 2.50, "color": "#008000"},
    "Eredivisie": {"avg_goals": 3.00, "color": "#FF6600"},
    "Singapore Premier": {"avg_goals": 3.63, "color": "#E31A1A"},
    "Malaysia Super": {"avg_goals": 3.42, "color": "#1A4D2E"},
    "Qatar Stars": {"avg_goals": 3.32, "color": "#8A1F1F"},
    "Dutch Eerste Divisie": {"avg_goals": 3.31, "color": "#FF7F00"},
    "USA MLS": {"avg_goals": 3.10, "color": "#1A2B4C"},
}

DEFAULT_LEAGUE = {"avg_goals": 2.70, "color": "#333333"}


# ============================================================================
# SECTION 2: PURE PYTHON POISSON (No Scipy)
# ============================================================================

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson Probability Mass Function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = (k * math.log(lam)) - lam - math.lgamma(k + 1)
    return math.exp(log_p)


def math_lgamma(n: int) -> float:
    """Log gamma approximation for integers."""
    if n <= 1:
        return 0.0
    x = n
    log_gamma = (x - 0.5) * math.log(x) - x + 0.5 * math.log(2 * math.pi) + 1/(12 * x)
    return log_gamma


if not hasattr(math, 'lgamma'):
    math.lgamma = math_lgamma


def poisson_probabilities(lam: float, max_goals: int = 7) -> List[float]:
    """Calculate Poisson probabilities for 0 to max_goals."""
    probs = [poisson_pmf(i, lam) for i in range(max_goals + 1)]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    return probs


# ============================================================================
# SECTION 3: STRENGTH COEFFICIENTS (Professional Standard)
# ============================================================================

def calculate_strength_coefficients(
    team_goals_per_game: float,
    team_conceded_per_game: float,
    league_avg_goals: float
) -> Tuple[float, float]:
    """
    Calculate attack and defense strength coefficients.
    
    Attack Strength > 1.0 = better than average
    Defense Strength > 1.0 = worse than average (concedes more)
    """
    attack_strength = team_goals_per_game / league_avg_goals if league_avg_goals > 0 else 1.0
    defense_strength = team_conceded_per_game / league_avg_goals if league_avg_goals > 0 else 1.0
    return attack_strength, defense_strength


def calculate_expected_goals(
    home_attack: float,
    home_defense: float,
    away_attack: float,
    away_defense: float,
    league_avg: float
) -> Tuple[float, float]:
    """
    Calculate expected goals using Dixon-Coles method.
    
    Home_xG = Home_Attack × Away_Defense × League_Avg
    Away_xG = Away_Attack × Home_Defense × League_Avg
    """
    home_xg = home_attack * away_defense * league_avg
    away_xg = away_attack * home_defense * league_avg
    
    # Clamp to realistic range
    home_xg = max(0.2, min(4.5, home_xg))
    away_xg = max(0.2, min(4.5, away_xg))
    
    return home_xg, away_xg


# ============================================================================
# SECTION 4: GAME STATE ADJUSTMENT (Volatility Filter)
# ============================================================================

def classify_team(
    possession: float,
    assists_per_game: float,
    goals_per_game: float
) -> Tuple[str, float]:
    """
    Classify team as Surgical, Lucky, or Neutral based on possession and assists.
    
    Returns: (classification, adjustment_factor)
    """
    assist_ratio = assists_per_game / goals_per_game if goals_per_game > 0 else 0
    
    if possession >= 55 and assist_ratio >= 0.6:
        return "Surgical", 1.05  # Trust the xG
    elif possession <= 45 and assist_ratio <= 0.4:
        return "Lucky", 0.85    # Downgrade xG
    else:
        return "Neutral", 1.0   # Trust the model


def apply_game_state_adjustment(
    home_xg: float,
    away_xg: float,
    home_possession: float,
    home_assists_per_game: float,
    home_goals_per_game: float,
    away_possession: float,
    away_assists_per_game: float,
    away_goals_per_game: float
) -> Tuple[float, float, str, str]:
    """Apply game state adjustment to expected goals."""
    home_class, home_adj = classify_team(
        home_possession, home_assists_per_game, home_goals_per_game
    )
    away_class, away_adj = classify_team(
        away_possession, away_assists_per_game, away_goals_per_game
    )
    
    home_xg_adj = home_xg * home_adj
    away_xg_adj = away_xg * away_adj
    
    return home_xg_adj, away_xg_adj, home_class, away_class


# ============================================================================
# SECTION 5: CLEAN SHEET CONFLICT (Using REAL Clean Sheet Count)
# ============================================================================

def check_clean_sheet_conflict(
    underdog_xg: float,
    favorite_clean_sheet_pct: float,
    underdog_threshold: float = 0.8,
    favorite_cs_threshold: float = 0.20
) -> Tuple[bool, str]:
    """
    Detect conflict between Poisson math and REAL clean sheet data.
    
    Uses actual clean sheet count from season, not Poisson probability.
    """
    if underdog_xg > underdog_threshold and favorite_clean_sheet_pct < favorite_cs_threshold:
        return True, "BTTS Yes"
    return False, None


# ============================================================================
# SECTION 6: DOUBLE-SIDED FILTER
# ============================================================================

def check_coefficient_alignment(
    home_attack: float,
    away_defense: float,
    away_attack: float,
    home_defense: float,
    threshold: float = 1.0
) -> Tuple[bool, str, str]:
    """
    Check if coefficients align for goals.
    """
    home_alignment = "HIGH" if home_attack > threshold and away_defense > threshold else "LOW"
    away_alignment = "HIGH" if away_attack > threshold and home_defense > threshold else "LOW"
    
    # For total goals, check if both attacks are high OR both defenses are low
    both_attacks_high = home_attack > threshold and away_attack > threshold
    both_defenses_low = home_defense < threshold and away_defense < threshold
    
    aligned = both_attacks_high or both_defenses_low
    
    return aligned, home_alignment, away_alignment


# ============================================================================
# SECTION 7: ACCURACY-FIRST BET RECOMMENDATION (No Draw Prediction)
# ============================================================================

def recommend_bet(
    home_xg: float,
    away_xg: float,
    total_xg: float,
    home_attack: float,
    away_attack: float,
    home_defense: float,
    away_defense: float,
    home_clean_sheet_pct: float,
    away_clean_sheet_pct: float,
    home_class: str,
    away_class: str,
    coefficients_aligned: bool
) -> Dict:
    """
    Recommend the highest-accuracy bet based on the model.
    NO DRAW PREDICTION. Only goals, BTTS, team totals, or SKIP.
    """
    combined_cs_pct = home_clean_sheet_pct + away_clean_sheet_pct
    conflict, conflict_bet = check_clean_sheet_conflict(
        min(home_xg, away_xg),
        max(home_clean_sheet_pct, away_clean_sheet_pct)
    )
    
    recommendations = []
    
    # Rule 1: Very low total goals
    if total_xg < 1.8:
        recommendations.append({
            "bet": "Under 2.5 Goals",
            "confidence": "HIGH",
            "reason": f"Total xG ({total_xg:.2f}) < 1.8"
        })
    
    # Rule 2: Very high total goals
    elif total_xg > 3.0:
        recommendations.append({
            "bet": "Over 2.5 Goals (or Asian Over 2.0)",
            "confidence": "HIGH",
            "reason": f"Total xG ({total_xg:.2f}) > 3.0"
        })
    
    # Rule 3: Clean sheet conflict (most important)
    if conflict:
        recommendations.append({
            "bet": conflict_bet,
            "confidence": "VERY HIGH",
            "reason": f"Underdog xG ({min(home_xg, away_xg):.2f}) > 0.8 AND Favorite CS% ({max(home_clean_sheet_pct, away_clean_sheet_pct)*100:.1f}%) < 20%"
        })
    
    # Rule 4: Combined clean sheet low = BTTS
    elif combined_cs_pct < 0.30:
        recommendations.append({
            "bet": "BTTS Yes",
            "confidence": "HIGH",
            "reason": f"Combined Clean Sheet % ({combined_cs_pct*100:.1f}%) < 30%"
        })
    
    # Rule 5: Strong home attack vs weak away defense
    if home_attack > 1.15 and away_defense > 1.0:
        recommendations.append({
            "bet": f"Home Team Over 1.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Home Attack ({home_attack:.2f}) × Away Defense ({away_defense:.2f})"
        })
    
    # Rule 6: Strong away attack vs weak home defense
    if away_attack > 1.15 and home_defense > 1.0:
        recommendations.append({
            "bet": f"Away Team Over 1.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Away Attack ({away_attack:.2f}) × Home Defense ({home_defense:.2f})"
        })
    
    # Rule 7: Coefficient alignment for total goals
    if coefficients_aligned and total_xg > 2.5:
        recommendations.append({
            "bet": "Over 2.5 Goals",
            "confidence": "MEDIUM",
            "reason": "Both coefficients align for high scoring"
        })
    
    # Remove duplicates (keep highest confidence for each bet type)
    seen = set()
    unique_recs = []
    for rec in recommendations:
        if rec["bet"] not in seen:
            seen.add(rec["bet"])
            unique_recs.append(rec)
    
    # Sort by confidence
    confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    unique_recs.sort(key=lambda x: confidence_order.get(x["confidence"], 4))
    
    return {
        "recommendations": unique_recs,
        "combined_cs_pct": combined_cs_pct,
        "conflict_detected": conflict,
        "coefficients_aligned": coefficients_aligned
    }


# ============================================================================
# SECTION 8: SCORE PROBABILITIES (For Reference Only)
# ============================================================================

def calculate_score_probabilities(
    home_xg: float,
    away_xg: float,
    max_goals: int = 4
) -> Dict[Tuple[int, int], float]:
    """Calculate probabilities for specific scorelines (reference only)."""
    prob_home = poisson_probabilities(home_xg, max_goals)
    prob_away = poisson_probabilities(away_xg, max_goals)
    
    scores = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            scores[(h, a)] = prob_home[h] * prob_away[a]
    
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:10])


# ============================================================================
# SECTION 9: MAIN PREDICTION FUNCTION
# ============================================================================

def predict_match(
    # Core statistics
    home_goals_scored: float,
    home_goals_conceded: float,
    away_goals_scored: float,
    away_goals_conceded: float,
    home_games_played: int,
    away_games_played: int,
    
    # Game state statistics
    home_possession: float,
    away_possession: float,
    home_assists: int,
    away_assists: int,
    home_clean_sheets: int,
    away_clean_sheets: int,
    
    # League context
    league_name: str = None,
    league_avg_goals: float = None,
    
    # Team names
    home_team: str = "Home Team",
    away_team: str = "Away Team"
) -> Dict:
    """
    Professional match prediction with accuracy-first betting.
    NO DRAW PREDICTION. Only goals, BTTS, team totals, or SKIP.
    """
    
    # ===== Step 1: Get league average =====
    if league_name and league_name in LEAGUE_DATABASE:
        league_avg = LEAGUE_DATABASE[league_name]["avg_goals"]
    else:
        league_avg = league_avg_goals or DEFAULT_LEAGUE["avg_goals"]
    
    # ===== Step 2: Calculate per-game averages =====
    home_goals_per_game = home_goals_scored / home_games_played
    home_conceded_per_game = home_goals_conceded / home_games_played
    away_goals_per_game = away_goals_scored / away_games_played
    away_conceded_per_game = away_goals_conceded / away_games_played
    
    home_assists_per_game = home_assists / home_games_played
    away_assists_per_game = away_assists / away_games_played
    
    home_clean_sheet_pct = home_clean_sheets / home_games_played
    away_clean_sheet_pct = away_clean_sheets / away_games_played
    
    # ===== Step 3: Calculate strength coefficients =====
    home_attack, home_defense = calculate_strength_coefficients(
        home_goals_per_game, home_conceded_per_game, league_avg
    )
    away_attack, away_defense = calculate_strength_coefficients(
        away_goals_per_game, away_conceded_per_game, league_avg
    )
    
    # ===== Step 4: Calculate expected goals =====
    home_xg, away_xg = calculate_expected_goals(
        home_attack, home_defense, away_attack, away_defense, league_avg
    )
    
    # ===== Step 5: Apply game state adjustment =====
    home_xg_adj, away_xg_adj, home_class, away_class = apply_game_state_adjustment(
        home_xg, away_xg,
        home_possession, home_assists_per_game, home_goals_per_game,
        away_possession, away_assists_per_game, away_goals_per_game
    )
    
    total_xg = home_xg_adj + away_xg_adj
    
    # ===== Step 6: Check coefficient alignment =====
    coefficients_aligned, home_align, away_align = check_coefficient_alignment(
        home_attack, away_defense, away_attack, home_defense
    )
    
    # ===== Step 7: Get bet recommendations =====
    bet_recs = recommend_bet(
        home_xg_adj, away_xg_adj, total_xg,
        home_attack, away_attack, home_defense, away_defense,
        home_clean_sheet_pct, away_clean_sheet_pct,
        home_class, away_class,
        coefficients_aligned
    )
    
    # ===== Step 8: Calculate score probabilities (reference only) =====
    top_scores = calculate_score_probabilities(home_xg_adj, away_xg_adj)
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "league": league_name or "Custom",
        "league_avg_goals": round(league_avg, 2),
        
        # Strength coefficients
        "home_attack_strength": round(home_attack, 2),
        "home_defense_strength": round(home_defense, 2),
        "away_attack_strength": round(away_attack, 2),
        "away_defense_strength": round(away_defense, 2),
        
        # Game state
        "home_classification": home_class,
        "away_classification": away_class,
        "coefficients_aligned": coefficients_aligned,
        
        # Expected goals
        "home_xg": round(home_xg_adj, 2),
        "away_xg": round(away_xg_adj, 2),
        "total_xg": round(total_xg, 2),
        
        # Clean sheet data (REAL, not Poisson)
        "home_clean_sheet_pct": round(home_clean_sheet_pct * 100, 1),
        "away_clean_sheet_pct": round(away_clean_sheet_pct * 100, 1),
        
        # Bet recommendations
        "bet_recommendations": bet_recs,
        
        # Score probabilities (reference)
        "top_scores": top_scores,
    }


# ============================================================================
# SECTION 10: UI COMPONENTS
# ============================================================================

def render_prediction(result: Dict):
    """Render the prediction results."""
    
    st.markdown(f"## 🎯 {result['home_team']} vs {result['away_team']}")
    st.markdown(f"*League: {result['league']} (Avg {result['league_avg_goals']} goals/game)*")
    st.divider()
    
    # Strength Coefficients
    st.markdown("### 📊 Strength Coefficients (Relative to League Average)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{result['home_team']} Attack", f"{result['home_attack_strength']:.2f}",
                  delta="Above avg" if result['home_attack_strength'] > 1 else "Below avg")
    with col2:
        st.metric(f"{result['home_team']} Defense", f"{result['home_defense_strength']:.2f}",
                  delta="Weak" if result['home_defense_strength'] > 1 else "Strong")
    with col3:
        st.metric(f"{result['away_team']} Attack", f"{result['away_attack_strength']:.2f}",
                  delta="Above avg" if result['away_attack_strength'] > 1 else "Below avg")
    with col4:
        st.metric(f"{result['away_team']} Defense", f"{result['away_defense_strength']:.2f}",
                  delta="Weak" if result['away_defense_strength'] > 1 else "Strong")
    
    # Game State Classification
    st.markdown("### 🎮 Game State Classification")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**{result['home_team']}**: {result['home_classification']}")
    with col2:
        st.info(f"**{result['away_team']}**: {result['away_classification']}")
    
    # Expected Goals
    st.markdown("### 🎯 Expected Goals")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Home xG", result['home_xg'])
    with col2:
        st.metric("Away xG", result['away_xg'])
    with col3:
        st.metric("Total xG", result['total_xg'])
    
    # Clean Sheet Data (REAL)
    st.markdown("### 🛡️ Clean Sheet Data (Actual Season Stats)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"{result['home_team']} Clean Sheet %", f"{result['home_clean_sheet_pct']}%")
    with col2:
        st.metric(f"{result['away_team']} Clean Sheet %", f"{result['away_clean_sheet_pct']}%")
    
    st.divider()
    
    # Bet Recommendations
    st.markdown("## 🎯 ACCURACY-FIRST BET RECOMMENDATIONS")
    st.caption("No draw prediction. Only goals, BTTS, team totals, or SKIP.")
    
    if result["bet_recommendations"]["recommendations"]:
        for rec in result["bet_recommendations"]["recommendations"]:
            if rec["confidence"] == "VERY HIGH":
                st.success(f"✅ **{rec['bet']}** - {rec['confidence']} Confidence")
                st.caption(f"📝 {rec['reason']}")
            elif rec["confidence"] == "HIGH":
                st.info(f"📌 **{rec['bet']}** - {rec['confidence']} Confidence")
                st.caption(f"📝 {rec['reason']}")
            else:
                st.warning(f"⚠️ **{rec['bet']}** - {rec['confidence']} Confidence")
                st.caption(f"📝 {rec['reason']}")
    else:
        st.warning("⚠️ **SKIP** - No clear signal. No bet recommended.")
    
    # Most likely scores (reference only)
    with st.expander("📊 Most Likely Scorelines (Reference Only)"):
        score_cols = st.columns(5)
        for i, ((h, a), prob) in enumerate(list(result['top_scores'].items())[:5]):
            with score_cols[i]:
                st.metric(f"{h}-{a}", f"{prob*100:.1f}%")


# ============================================================================
# SECTION 11: MAIN APP
# ============================================================================

def main():
    st.title("🎯 Accuracy-First Football Predictor")
    st.caption("Strength Coefficients → xG → Game State → Clean Sheet Conflict → Bet/SKIP")
    st.caption("No draw prediction. Only goals, BTTS, team totals.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ League Settings")
        
        league_options = list(LEAGUE_DATABASE.keys()) + ["Custom League"]
        selected_league = st.selectbox("Select League", league_options)
        
        if selected_league == "Custom League":
            league_avg = st.number_input("League Avg Goals/Game", 2.0, 4.0, 2.70, 0.05)
            league_name = "Custom"
            league_avg_val = league_avg
        else:
            league_name = selected_league
            league_avg_val = LEAGUE_DATABASE[selected_league]["avg_goals"]
    
    # Main content
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("Home Team Name", "Liverpool")
    with col2:
        away_team = st.text_input("Away Team Name", "Everton")
    
    st.divider()
    
    # Team statistics
    st.subheader("📊 Team Statistics (Season Totals)")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"**🏠 {home_team}**")
        home_games = st.number_input("Games Played", 1, 50, 31, key="home_games")
        home_scored = st.number_input("Goals Scored", 0, 150, 65, key="home_scored")
        home_conceded = st.number_input("Goals Conceded", 0, 150, 45, key="home_conceded")
        home_clean_sheets = st.number_input("Clean Sheets", 0, 50, 7, key="home_cs")
        home_possession = st.slider("Possession %", 30, 70, 55, key="home_pos")
        home_assists = st.number_input("Assists", 0, 150, 48, key="home_assists")
    
    with col_right:
        st.markdown(f"**✈️ {away_team}**")
        away_games = st.number_input("Games Played", 1, 50, 31, key="away_games")
        away_scored = st.number_input("Goals Scored", 0, 150, 40, key="away_scored")
        away_conceded = st.number_input("Goals Conceded", 0, 150, 50, key="away_conceded")
        away_clean_sheets = st.number_input("Clean Sheets", 0, 50, 5, key="away_cs")
        away_possession = st.slider("Possession %", 30, 70, 48, key="away_pos")
        away_assists = st.number_input("Assists", 0, 150, 30, key="away_assists")
    
    # Predict button
    if st.button("🔮 ANALYZE MATCH", type="primary", use_container_width=True):
        with st.spinner("Calculating strength coefficients and game state..."):
            result = predict_match(
                home_goals_scored=home_scored,
                home_goals_conceded=home_conceded,
                away_goals_scored=away_scored,
                away_goals_conceded=away_conceded,
                home_games_played=home_games,
                away_games_played=away_games,
                home_possession=home_possession,
                away_possession=away_possession,
                home_assists=home_assists,
                away_assists=away_assists,
                home_clean_sheets=home_clean_sheets,
                away_clean_sheets=away_clean_sheets,
                league_name=league_name if league_name != "Custom" else None,
                league_avg_goals=league_avg_val if league_name == "Custom" else None,
                home_team=home_team,
                away_team=away_team
            )
        
        render_prediction(result)


if __name__ == "__main__":
    main()
