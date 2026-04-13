"""
Professional No Draw Prediction System
With Game State Adjustment, Clean Sheet Conflict Resolution, and Accuracy-First Betting
"""

import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import Dict, Tuple, Optional, List

st.set_page_config(
    page_title="Professional No Draw Predictor",
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
# SECTION 2: PURE PYTHON POISSON (No Scipy)
# ============================================================================

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson Probability Mass Function - Pure Python implementation."""
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
    team_goals_avg: float,
    team_conceded_avg: float,
    league_avg_goals: float
) -> Tuple[float, float]:
    """
    Calculate attack and defense strength coefficients relative to league average.
    
    Attack Strength = Team Goals / League Avg Goals
    Defense Strength = Team Conceded / League Avg Goals
    
    > 1.0 = better than average, < 1.0 = worse than average
    """
    attack_strength = team_goals_avg / league_avg_goals if league_avg_goals > 0 else 1.0
    defense_strength = team_conceded_avg / league_avg_goals if league_avg_goals > 0 else 1.0
    
    return attack_strength, defense_strength


def calculate_expected_goals_pro(
    home_attack_strength: float,
    home_defense_strength: float,
    away_attack_strength: float,
    away_defense_strength: float,
    league_avg_goals: float
) -> Tuple[float, float]:
    """
    Calculate expected goals using professional strength coefficients.
    
    Home_xG = Home_Attack × Away_Defense × League_Avg
    Away_xG = Away_Attack × Home_Defense × League_Avg
    """
    home_xg = home_attack_strength * away_defense_strength * league_avg_goals
    away_xg = away_attack_strength * home_defense_strength * league_avg_goals
    
    # Clamp to realistic range
    home_xg = max(0.2, min(4.5, home_xg))
    away_xg = max(0.2, min(4.5, away_xg))
    
    return home_xg, away_xg


# ============================================================================
# SECTION 4: GAME STATE ADJUSTMENT (Volatility Filter)
# ============================================================================

def calculate_volatility_adjustment(
    possession: float,
    assists: int,
    games_played: int,
    goals_scored: float
) -> float:
    """
    Determine if a team is "Surgical" (repeatable) or "Lucky" (volatile).
    
    Returns adjustment factor (0.85 to 1.15)
    """
    assists_per_game = assists / games_played if games_played > 0 else 0
    goals_per_game = goals_scored
    
    # Assist-to-goal ratio (how many goals are assisted)
    assist_ratio = assists_per_game / goals_per_game if goals_per_game > 0 else 0
    
    # Possession-based adjustment
    if possession >= 55 and assist_ratio >= 0.6:
        # Surgical team - trust the xG
        return 1.05
    elif possession <= 45 and assist_ratio <= 0.4:
        # Lucky team - downgrade xG
        return 0.85
    else:
        # Neutral - trust the model
        return 1.0


def apply_volatility_filter(
    home_xg: float,
    away_xg: float,
    home_possession: float,
    home_assists: int,
    home_games: int,
    home_goals: float,
    away_possession: float,
    away_assists: int,
    away_games: int,
    away_goals: float
) -> Tuple[float, float, str, str]:
    """
    Apply game state adjustment to expected goals.
    Returns adjusted xG and team classifications.
    """
    home_adj = calculate_volatility_adjustment(
        home_possession, home_assists, home_games, home_goals
    )
    away_adj = calculate_volatility_adjustment(
        away_possession, away_assists, away_games, away_goals
    )
    
    home_classification = "Surgical" if home_adj > 1.02 else "Lucky" if home_adj < 0.98 else "Neutral"
    away_classification = "Surgical" if away_adj > 1.02 else "Lucky" if away_adj < 0.98 else "Neutral"
    
    home_xg_adj = home_xg * home_adj
    away_xg_adj = away_xg * away_adj
    
    return home_xg_adj, away_xg_adj, home_classification, away_classification


# ============================================================================
# SECTION 5: CLEAN SHEET CONFLICT RESOLUTION
# ============================================================================

def calculate_clean_sheet_probability(
    xg_conceded: float,
    max_goals: int = 7
) -> float:
    """
    Calculate probability of a clean sheet from Poisson distribution.
    P(clean sheet) = P(opponent scores 0 goals) = Poisson(0, xg_conceded)
    """
    return poisson_pmf(0, xg_conceded)


def detect_clean_sheet_conflict(
    underdog_xg: float,
    favorite_clean_sheet_pct: float,
    underdog_threshold: float = 0.8,
    favorite_cs_threshold: float = 0.20
) -> Tuple[bool, str]:
    """
    Detect conflict between Poisson math and raw clean sheet data.
    
    Returns: (conflict_detected, recommended_bet)
    """
    if underdog_xg > underdog_threshold and favorite_clean_sheet_pct < favorite_cs_threshold:
        return True, "BTTS Yes"
    return False, None


# ============================================================================
# SECTION 6: DOUBLE-SIDED FILTER (Coefficient Alignment)
# ============================================================================

def check_coefficient_alignment(
    home_attack_strength: float,
    away_defense_strength: float,
    away_attack_strength: float,
    home_defense_strength: float,
    threshold: float = 1.0
) -> Tuple[bool, str, str]:
    """
    Check if coefficients align for the same outcome.
    
    Returns: (aligned, home_alignment, away_alignment)
    """
    home_alignment = "HIGH" if home_attack_strength > threshold and away_defense_strength > threshold else "LOW"
    away_alignment = "HIGH" if away_attack_strength > threshold and home_defense_strength > threshold else "LOW"
    
    # Both align for the same outcome? No - they are opposing.
    # For total goals, we check if both attack strengths are high
    both_attacks_high = home_attack_strength > threshold and away_attack_strength > threshold
    both_defenses_low = home_defense_strength < threshold and away_defense_strength < threshold
    
    aligned = both_attacks_high or both_defenses_low
    
    return aligned, home_alignment, away_alignment


# ============================================================================
# SECTION 7: ACCURACY-FIRST BET RECOMMENDATION
# ============================================================================

def recommend_accuracy_bet(
    home_xg: float,
    away_xg: float,
    total_xg: float,
    home_attack_strength: float,
    away_attack_strength: float,
    home_defense_strength: float,
    away_defense_strength: float,
    home_clean_sheet_pct: float,
    away_clean_sheet_pct: float,
    home_classification: str,
    away_classification: str,
    coefficients_aligned: bool
) -> Dict:
    """
    Recommend the highest-accuracy bet based on model outputs.
    """
    combined_cs_pct = home_clean_sheet_pct + away_clean_sheet_pct
    conflict, conflict_bet = detect_clean_sheet_conflict(
        min(home_xg, away_xg),
        max(home_clean_sheet_pct, away_clean_sheet_pct)
    )
    
    recommendations = []
    
    # Rule 1: Low total goals
    if total_xg < 1.8:
        recommendations.append({
            "bet": "Under 2.5 Goals",
            "confidence": "HIGH",
            "reason": f"Total xG ({total_xg:.2f}) < 1.8"
        })
    
    # Rule 2: High total goals
    if total_xg > 3.0:
        recommendations.append({
            "bet": "Over 2.5 Goals or Asian Over 2.0",
            "confidence": "HIGH",
            "reason": f"Total xG ({total_xg:.2f}) > 3.0"
        })
    
    # Rule 3: Clean sheet conflict
    if conflict:
        recommendations.append({
            "bet": conflict_bet,
            "confidence": "VERY HIGH",
            "reason": f"Underdog xG ({min(home_xg, away_xg):.2f}) > 0.8 AND Favorite CS% ({max(home_clean_sheet_pct, away_clean_sheet_pct)*100:.1f}%) < 20%"
        })
    
    # Rule 4: Combined clean sheet low = BTTS
    if combined_cs_pct < 0.30:
        recommendations.append({
            "bet": "BTTS Yes",
            "confidence": "HIGH",
            "reason": f"Combined Clean Sheet % ({combined_cs_pct*100:.1f}%) < 30%"
        })
    
    # Rule 5: Strong attack vs weak defense
    if home_attack_strength > 1.15 and away_defense_strength > 1.0:
        recommendations.append({
            "bet": f"Home Team Over 1.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Home Attack ({home_attack_strength:.2f}) × Away Defense ({away_defense_strength:.2f})"
        })
    
    if away_attack_strength > 1.15 and home_defense_strength > 1.0:
        recommendations.append({
            "bet": f"Away Team Over 1.5 Goals",
            "confidence": "MEDIUM",
            "reason": f"Away Attack ({away_attack_strength:.2f}) × Home Defense ({home_defense_strength:.2f})"
        })
    
    # Rule 6: Coefficient alignment for total goals
    if coefficients_aligned and total_xg > 2.5:
        recommendations.append({
            "bet": "Over 2.5 Goals",
            "confidence": "HIGH",
            "reason": "Both coefficients align for high scoring"
        })
    
    # Sort by confidence
    confidence_order = {"VERY HIGH": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recommendations.sort(key=lambda x: confidence_order.get(x["confidence"], 4))
    
    return {
        "recommendations": recommendations,
        "combined_cs_pct": combined_cs_pct,
        "conflict_detected": conflict,
        "conflict_bet": conflict_bet,
        "coefficients_aligned": coefficients_aligned
    }


# ============================================================================
# SECTION 8: POISSON MATRIX AND DRAW PROBABILITY
# ============================================================================

def poisson_probability_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int = 7
) -> np.ndarray:
    """Generate the goal probability matrix using Poisson distribution."""
    prob_home = poisson_probabilities(home_xg, max_goals)
    prob_away = poisson_probabilities(away_xg, max_goals)
    
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = prob_home[i] * prob_away[j]
    
    return matrix


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
    
    draw_prob = sum(matrix[i, i] for i in range(max_goals + 1))
    
    return min(draw_prob, 1.0)


def calculate_score_probabilities(
    home_xg: float,
    away_xg: float,
    max_goals: int = 4
) -> Dict[Tuple[int, int], float]:
    """Calculate probabilities for specific scorelines."""
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
    home_games_played: int = 30,
    away_games_played: int = 30,
    
    # Advanced statistics (for game state adjustment)
    home_possession: float = 50.0,
    away_possession: float = 50.0,
    home_assists: int = 30,
    away_assists: int = 30,
    home_clean_sheets: int = 5,
    away_clean_sheets: int = 5,
    
    # League context
    league_name: str = None,
    league_avg_goals: float = None,
    
    # Model parameters
    use_zip: bool = True,
    use_volatility_filter: bool = True,
    
    # Team names
    home_team: str = "Home Team",
    away_team: str = "Away Team"
) -> Dict:
    """
    Professional match prediction with all expert overlays.
    """
    
    # ===== Step 1: Get league parameters =====
    if league_name and league_name in LEAGUE_DATABASE:
        league = LEAGUE_DATABASE[league_name]
        league_avg = league_avg_goals or league["avg_goals"]
        threshold = league["threshold"]
        zip_factor = league["zip_factor"] if use_zip else 0.0
    else:
        league_avg = league_avg_goals or DEFAULT_LEAGUE["avg_goals"]
        threshold = DEFAULT_LEAGUE["threshold"]
        zip_factor = DEFAULT_LEAGUE["zip_factor"] if use_zip else 0.0
    
    # Calculate per-game averages
    home_goals_avg = home_goals_scored / home_games_played if home_games_played > 0 else 1.0
    home_conceded_avg = home_goals_conceded / home_games_played if home_games_played > 0 else 1.0
    away_goals_avg = away_goals_scored / away_games_played if away_games_played > 0 else 1.0
    away_conceded_avg = away_goals_conceded / away_games_played if away_games_played > 0 else 1.0
    
    # ===== Step 2: Calculate strength coefficients =====
    home_attack_strength, home_defense_strength = calculate_strength_coefficients(
        home_goals_avg, home_conceded_avg, league_avg
    )
    away_attack_strength, away_defense_strength = calculate_strength_coefficients(
        away_goals_avg, away_conceded_avg, league_avg
    )
    
    # ===== Step 3: Calculate base expected goals =====
    home_xg, away_xg = calculate_expected_goals_pro(
        home_attack_strength, home_defense_strength,
        away_attack_strength, away_defense_strength,
        league_avg
    )
    
    # ===== Step 4: Apply volatility filter (game state adjustment) =====
    home_classification = "Neutral"
    away_classification = "Neutral"
    
    if use_volatility_filter:
        home_xg, away_xg, home_classification, away_classification = apply_volatility_filter(
            home_xg, away_xg,
            home_possession, home_assists, home_games_played, home_goals_avg,
            away_possession, away_assists, away_games_played, away_goals_avg
        )
    
    total_xg = home_xg + away_xg
    
    # ===== Step 5: Calculate clean sheet probabilities =====
    home_clean_sheet_prob = calculate_clean_sheet_probability(away_xg)
    away_clean_sheet_prob = calculate_clean_sheet_probability(home_xg)
    home_clean_sheet_pct_raw = home_clean_sheets / home_games_played if home_games_played > 0 else 0.1
    away_clean_sheet_pct_raw = away_clean_sheets / away_games_played if away_games_played > 0 else 0.1
    
    # ===== Step 6: Check coefficient alignment =====
    coefficients_aligned, home_alignment, away_alignment = check_coefficient_alignment(
        home_attack_strength, away_defense_strength,
        away_attack_strength, home_defense_strength
    )
    
    # ===== Step 7: Calculate draw probability =====
    draw_prob = calculate_draw_probability(home_xg, away_xg, zip_factor)
    no_draw_prob = 1 - draw_prob
    
    # ===== Step 8: Get accuracy-first bet recommendations =====
    bet_recommendations = recommend_accuracy_bet(
        home_xg, away_xg, total_xg,
        home_attack_strength, away_attack_strength,
        home_defense_strength, away_defense_strength,
        home_clean_sheet_pct_raw, away_clean_sheet_pct_raw,
        home_classification, away_classification,
        coefficients_aligned
    )
    
    # ===== Step 9: Calculate score probabilities =====
    top_scores = calculate_score_probabilities(home_xg, away_xg)
    
    return {
        # Input summary
        "home_team": home_team,
        "away_team": away_team,
        "league": league_name or "Custom",
        "league_avg_goals": round(league_avg, 2),
        
        # Strength coefficients
        "home_attack_strength": round(home_attack_strength, 2),
        "home_defense_strength": round(home_defense_strength, 2),
        "away_attack_strength": round(away_attack_strength, 2),
        "away_defense_strength": round(away_defense_strength, 2),
        "home_classification": home_classification,
        "away_classification": away_classification,
        "coefficients_aligned": coefficients_aligned,
        "home_alignment": home_alignment,
        "away_alignment": away_alignment,
        
        # Expected goals
        "home_xg": round(home_xg, 2),
        "away_xg": round(away_xg, 2),
        "total_xg": round(total_xg, 2),
        
        # Probabilities
        "draw_probability": draw_prob,
        "no_draw_probability": no_draw_prob,
        "home_clean_sheet_prob": home_clean_sheet_prob,
        "away_clean_sheet_prob": away_clean_sheet_prob,
        "home_clean_sheet_pct_raw": home_clean_sheet_pct_raw,
        "away_clean_sheet_pct_raw": away_clean_sheet_pct_raw,
        
        # Bet recommendations
        "bet_recommendations": bet_recommendations,
        
        # Score probabilities
        "top_scores": top_scores,
        
        # Decision
        "draw_prediction": "NO DRAW" if draw_prob < threshold else "DRAW LIKELY",
    }


# ============================================================================
# SECTION 10: UI COMPONENTS
# ============================================================================

def render_prediction_card(result: Dict):
    """Render the prediction result."""
    
    # Determine colors based on prediction
    if result["draw_prediction"] == "NO DRAW":
        pred_color = "#10b981"
        pred_bg = "linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%)"
        border_color = "#10b981"
    else:
        pred_color = "#f97316"
        pred_bg = "linear-gradient(135deg, #1e293b 0%, #3a2a1a 100%)"
        border_color = "#f97316"
    
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
        <div class="prediction-title">{result['draw_prediction']}</div>
        
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">DRAW PROBABILITY</div>
                <div class="stat-value-critical">{draw_pct:.1f}%</div>
                <div style="background:#1e293b; border-radius:12px; padding:0.5rem; margin-top:0.5rem;"><div class="gauge-bar"></div></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">TOTAL xG</div>
                <div class="stat-value">{result['total_xg']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">HOME xG</div>
                <div class="stat-value">{result['home_xg']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">AWAY xG</div>
                <div class="stat-value">{result['away_xg']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Strength coefficients
    st.markdown("**📊 STRENGTH COEFFICIENTS**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{result['home_team']} Attack", f"{result['home_attack_strength']:.2f}", 
                  delta="Above avg" if result['home_attack_strength'] > 1 else "Below avg")
    with col2:
        st.metric(f"{result['home_team']} Defense", f"{result['home_defense_strength']:.2f}",
                  delta="Above avg" if result['home_defense_strength'] < 1 else "Below avg")
    with col3:
        st.metric(f"{result['away_team']} Attack", f"{result['away_attack_strength']:.2f}",
                  delta="Above avg" if result['away_attack_strength'] > 1 else "Below avg")
    with col4:
        st.metric(f"{result['away_team']} Defense", f"{result['away_defense_strength']:.2f}",
                  delta="Above avg" if result['away_defense_strength'] < 1 else "Below avg")
    
    # Game state classification
    st.markdown("**🎮 GAME STATE CLASSIFICATION**")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"{result['home_team']}: **{result['home_classification']}**")
    with col2:
        st.info(f"{result['away_team']}: **{result['away_classification']}**")
    
    # Bet recommendations
    st.markdown("---")
    st.markdown("## 🎯 ACCURACY-FIRST BET RECOMMENDATIONS")
    
    if result["bet_recommendations"]["recommendations"]:
        for rec in result["bet_recommendations"]["recommendations"]:
            if rec["confidence"] == "VERY HIGH":
                st.success(f"✅ **{rec['bet']}** - {rec['confidence']} confidence")
                st.caption(f"📝 {rec['reason']}")
            elif rec["confidence"] == "HIGH":
                st.info(f"📌 **{rec['bet']}** - {rec['confidence']} confidence")
                st.caption(f"📝 {rec['reason']}")
            else:
                st.warning(f"⚠️ **{rec['bet']}** - {rec['confidence']} confidence")
                st.caption(f"📝 {rec['reason']}")
    else:
        st.warning("No strong recommendations. Consider skipping this match.")
    
    # Most likely scores
    with st.expander("📊 Most Likely Scorelines"):
        score_cols = st.columns(5)
        for i, ((h, a), prob) in enumerate(list(result['top_scores'].items())[:5]):
            with score_cols[i]:
                st.metric(f"{h}-{a}", f"{prob*100:.1f}%")


# ============================================================================
# SECTION 11: MAIN APP
# ============================================================================

def main():
    st.title("🎯 Professional No Draw Predictor")
    st.caption("Accuracy-First Betting | Poisson + Strength Coefficients + Game State Adjustment")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        league_options = list(LEAGUE_DATABASE.keys()) + ["Custom League"]
        selected_league = st.selectbox("Select League", league_options)
        
        if selected_league == "Custom League":
            league_avg = st.number_input("League Avg Goals/Game", 2.0, 4.0, 2.70, 0.05)
            league_name = "Custom"
            league_avg_val = league_avg
        else:
            league_data = LEAGUE_DATABASE[selected_league]
            league_name = selected_league
            league_avg_val = league_data["avg_goals"]
        
        st.divider()
        
        st.subheader("📊 Advanced Options")
        use_zip = st.checkbox("Apply ZIP Adjustment", value=True)
        use_volatility = st.checkbox("Apply Game State Adjustment", value=True)
    
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
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        with st.spinner("Calculating..."):
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
                use_zip=use_zip,
                use_volatility_filter=use_volatility,
                home_team=home_team,
                away_team=away_team
            )
        
        render_prediction_card(result)


if __name__ == "__main__":
    main()
