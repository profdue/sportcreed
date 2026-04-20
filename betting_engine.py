"""
Expected Goals Predictor - Over 2.5 / Under 2.5 Strict System with Checkmate Safeguards
Based on statistical analysis of home/away averages, H2H history, and league context.

Core Formula:
Expected Goals = (Home_Scored_Home_Avg + Away_Conceded_Away_Avg + Away_Scored_Away_Avg + Home_Conceded_Home_Avg) / 2

Input Rules:
- Home Team: Use HOME row only (Scored + Conceded)
- Away Team: Use AWAY row only (Scored + Conceded)
- Ignore the other numbers (the "double" stats are for reference only)

Checkmate Safeguards (Red Flags):
1. If Home team scores < 0.90 at home → Strong Under bias
2. If Away team scores < 0.70 away → Strong Under bias
3. If H2H has < 8 matches → Reduce confidence by one level
4. If Expected Goals and H2H disagree strongly → Trust the lower one
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Expected Goals Predictor - Over/Under 2.5",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS STYLES
# ============================================================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        border: 1px solid #334155;
    }
    .prediction-over {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10b981;
    }
    .prediction-under {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ef4444;
    }
    .prediction-lean-over {
        font-size: 2rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .expected-goals {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0.5rem 0;
    }
    .confidence-high {
        font-size: 1rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 0.5rem;
    }
    .confidence-medium-high {
        font-size: 1rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 0.5rem;
    }
    .confidence-medium {
        font-size: 1rem;
        font-weight: 600;
        color: #fbbf24;
        margin-top: 0.5rem;
    }
    .confidence-low {
        font-size: 1rem;
        font-weight: 600;
        color: #f97316;
        margin-top: 0.5rem;
    }
    .secondary {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .checkmate-warning {
        background: #7f1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #fca5a5;
        text-align: left;
    }
    .team-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .h2h-table {
        background: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .league-note {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    .input-note {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: #fbbf24;
        text-align: center;
    }
    h1 {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stButton button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        border: none;
        width: 100%;
    }
    hr {
        margin: 1rem 0;
    }
    .step-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.8rem;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamStats:
    """Team statistical data - ONLY the relevant numbers for calculation"""
    name: str
    avg_scored_home: float = 0.0      # Home team's home scoring
    avg_conceded_home: float = 0.0    # Home team's home conceding
    avg_scored_away: float = 0.0      # Away team's away scoring
    avg_conceded_away: float = 0.0    # Away team's away conceding
    btts_percent: float = 0.0


@dataclass
class H2HStats:
    """Head-to-head statistics"""
    matches_played: int = 0
    over15_percent: float = 0.0
    over25_percent: float = 0.0
    btts_percent: float = 0.0


@dataclass
class LeagueContext:
    """League baseline information"""
    name: str
    avg_goals_per_game: float
    is_low_scoring: bool = False


@dataclass
class CheckmateFlags:
    """Checkmate safeguard flags"""
    home_scoring_low: bool = False      # Home scores < 0.90 at home
    away_scoring_low: bool = False      # Away scores < 0.70 away
    h2h_small_sample: bool = False      # H2H < 8 matches
    exp_h2h_disagreement: bool = False  # Expected Goals and H2H disagree strongly
    red_flags: List[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    main_bet: str  # "Over 2.5", "Lean Over 2.5", or "Under 2.5"
    confidence: str  # "High", "Medium-High", "Medium", "Low"
    secondary: Optional[str]  # "Over 1.5" or None
    expected_goals: float
    checkmate: CheckmateFlags
    reasoning: List[str]
    decision_path: List[str]


# ============================================================================
# LEAGUE DATA
# ============================================================================
LEAGUE_CONTEXT = {
    "Ligue 2": LeagueContext("Ligue 2", 2.45, is_low_scoring=False),
    "Ukraine Premier League": LeagueContext("Ukraine Premier League", 2.30, is_low_scoring=False),
    "Ethiopian Premier League": LeagueContext("Ethiopian Premier League", 1.80, is_low_scoring=True),
    "Argentine Liga Profesional": LeagueContext("Argentine Liga Profesional", 1.90, is_low_scoring=True),
    "Bundesliga": LeagueContext("Bundesliga", 3.10, is_low_scoring=False),
    "Premier League": LeagueContext("Premier League", 2.80, is_low_scoring=False),
    "Serie A": LeagueContext("Serie A", 2.60, is_low_scoring=False),
    "La Liga": LeagueContext("La Liga", 2.50, is_low_scoring=False),
    "Ligue 1": LeagueContext("Ligue 1", 2.70, is_low_scoring=False),
    "Eredivisie": LeagueContext("Eredivisie", 3.20, is_low_scoring=False),
    "Serie B": LeagueContext("Serie B", 2.40, is_low_scoring=False),
    "Championship": LeagueContext("Championship", 2.60, is_low_scoring=False),
    "Default": LeagueContext("Default", 2.50, is_low_scoring=False),
}


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_expected_goals(home: TeamStats, away: TeamStats) -> float:
    """Calculate expected total goals using the correct 4 numbers"""
    expected = (home.avg_scored_home + away.avg_conceded_away + away.avg_scored_away + home.avg_conceded_home) / 2
    return round(expected, 2)


def check_checkmate_flags(
    home: TeamStats,
    away: TeamStats,
    h2h: H2HStats,
    expected_goals: float
) -> CheckmateFlags:
    """Evaluate all Checkmate safeguard flags"""
    red_flags = []
    
    # Flag 1: Home team scores < 0.90 at home
    home_scoring_low = home.avg_scored_home < 0.90
    if home_scoring_low:
        red_flags.append(f"⚠️ Home team scores only {home.avg_scored_home:.2f} at home (< 0.90) → Strong Under bias")
    
    # Flag 2: Away team scores < 0.70 away
    away_scoring_low = away.avg_scored_away < 0.70
    if away_scoring_low:
        red_flags.append(f"⚠️ Away team scores only {away.avg_scored_away:.2f} away (< 0.70) → Strong Under bias")
    
    # Flag 3: H2H sample size < 8 matches
    h2h_small_sample = h2h.matches_played < 8
    if h2h_small_sample:
        red_flags.append(f"⚠️ H2H sample size only {h2h.matches_played} matches (< 8) → Reduce confidence")
    
    # Flag 4: Expected Goals and H2H disagree strongly
    exp_h2h_disagreement = False
    if expected_goals >= 2.40 and h2h.over25_percent < 40:
        exp_h2h_disagreement = True
        red_flags.append(f"⚠️ Expected Goals {expected_goals} suggests Over but H2H Over 2.5% = {h2h.over25_percent}% → Trust the lower (Under bias)")
    elif expected_goals <= 2.30 and h2h.over25_percent > 55:
        exp_h2h_disagreement = True
        red_flags.append(f"⚠️ Expected Goals {expected_goals} suggests Under but H2H Over 2.5% = {h2h.over25_percent}% → Trust the lower (Under bias)")
    
    return CheckmateFlags(
        home_scoring_low=home_scoring_low,
        away_scoring_low=away_scoring_low,
        h2h_small_sample=h2h_small_sample,
        exp_h2h_disagreement=exp_h2h_disagreement,
        red_flags=red_flags
    )


def apply_checkmate_to_prediction(
    main_bet: str,
    confidence: str,
    checkmate: CheckmateFlags
) -> Tuple[str, str]:
    """Apply Checkmate safeguards to adjust prediction and confidence"""
    
    # If any red flags exist, adjust
    if checkmate.red_flags:
        # Strong Under bias triggers
        if checkmate.home_scoring_low or checkmate.away_scoring_low:
            if "Over" in main_bet:
                main_bet = "Lean Over 2.5"
                confidence = "Medium"
            # Under bets remain but may have confidence adjusted
        
        # Small sample size reduces confidence
        if checkmate.h2h_small_sample:
            if confidence == "High":
                confidence = "Medium-High"
            elif confidence == "Medium-High":
                confidence = "Medium"
            elif confidence == "Medium":
                confidence = "Low"
        
        # Disagreement: trust the lower (Under bias)
        if checkmate.exp_h2h_disagreement:
            if "Over" in main_bet:
                main_bet = "Lean Over 2.5"
                confidence = "Low"
    
    return main_bet, confidence


def make_prediction(
    expected_goals: float,
    h2h: H2HStats,
    league: LeagueContext,
    home: TeamStats,
    away: TeamStats
) -> PredictionResult:
    """
    Make Over 2.5 / Under 2.5 prediction based on strict decision rules with Checkmate.
    """
    reasoning = []
    decision_path = []
    
    avg_btts = (home.btts_percent + away.btts_percent) / 2
    
    reasoning.append(f"📊 **Input Summary:**")
    reasoning.append(f"   • Expected Goals: {expected_goals}")
    reasoning.append(f"   • H2H Over 2.5%: {h2h.over25_percent}%")
    reasoning.append(f"   • H2H Over 1.5%: {h2h.over15_percent}%")
    reasoning.append(f"   • H2H BTTS%: {h2h.btts_percent}%")
    reasoning.append(f"   • {home.name} Home Scored: {home.avg_scored_home}")
    reasoning.append(f"   • {away.name} Away Scored: {away.avg_scored_away}")
    reasoning.append(f"   • League avg: {league.avg_goals_per_game} goals/game")
    
    # Calculate Checkmate flags
    checkmate = check_checkmate_flags(home, away, h2h, expected_goals)
    
    if checkmate.red_flags:
        reasoning.append(f"\n⚠️ **CHECKMATE SAFEGUARDS TRIGGERED:**")
        for flag in checkmate.red_flags:
            reasoning.append(f"   {flag}")
    
    # ========================================================================
    # STEP 1: Expected Goals < 2.20 → Under 2.5
    # ========================================================================
    if expected_goals < 2.20:
        decision_path.append("Step 1: Expected Goals < 2.20")
        main_bet = "Under 2.5"
        confidence = "High"
        reasoning.append(f"\n✅ **STEP 1:** Expected Goals {expected_goals} < 2.20")
        reasoning.append(f"   → **Under 2.5** (High confidence)")
        
        # Check for secondary Over 1.5 suggestion
        secondary = None
        if expected_goals >= 2.20 and h2h.over15_percent >= 65:
            secondary = "Over 1.5"
            reasoning.append(f"\n📌 **Secondary:** Over 1.5 (H2H Over 1.5% = {h2h.over15_percent}% ≥ 65%)")
        
        # Apply Checkmate
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 2: Expected Goals > 2.80 → Over 2.5
    # ========================================================================
    if expected_goals > 2.80:
        decision_path.append("Step 2: Expected Goals > 2.80")
        main_bet = "Over 2.5"
        confidence = "High"
        reasoning.append(f"\n✅ **STEP 2:** Expected Goals {expected_goals} > 2.80")
        reasoning.append(f"   → **Over 2.5** (High confidence)")
        
        secondary = None
        
        # Apply Checkmate
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 3: Expected Goals 2.20 – 2.80 → Go to Step 4
    # ========================================================================
    decision_path.append(f"Step 3: Expected Goals {expected_goals} in range 2.20-2.80")
    reasoning.append(f"\n✅ **STEP 3:** Expected Goals {expected_goals} in range 2.20-2.80")
    reasoning.append(f"   → Checking H2H Over 2.5%...")
    
    # ========================================================================
    # STEP 4: H2H Over 2.5% > 55% → Over 2.5
    # ========================================================================
    if h2h.over25_percent > 55:
        decision_path.append(f"Step 4: H2H Over 2.5% = {h2h.over25_percent}% > 55%")
        main_bet = "Over 2.5"
        confidence = "Medium"
        reasoning.append(f"\n✅ **STEP 4:** H2H Over 2.5% = {h2h.over25_percent}% > 55%")
        reasoning.append(f"   → **Over 2.5** (Medium confidence)")
        
        secondary = None
        
        # Apply Checkmate (may downgrade to Lean Over)
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 5: H2H Over 2.5% < 40% → Under 2.5
    # ========================================================================
    if h2h.over25_percent < 40:
        decision_path.append(f"Step 5: H2H Over 2.5% = {h2h.over25_percent}% < 40%")
        main_bet = "Under 2.5"
        confidence = "Medium-High"
        reasoning.append(f"\n✅ **STEP 5:** H2H Over 2.5% = {h2h.over25_percent}% < 40%")
        reasoning.append(f"   → **Under 2.5** (Medium-High confidence)")
        
        secondary = None
        if expected_goals >= 2.20 and h2h.over15_percent >= 65:
            secondary = "Over 1.5"
            reasoning.append(f"\n📌 **Secondary:** Over 1.5 (H2H Over 1.5% = {h2h.over15_percent}% ≥ 65%)")
        
        # Apply Checkmate
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 6: H2H Over 2.5% between 40-55% → Go to Step 7
    # ========================================================================
    decision_path.append(f"Step 6: H2H Over 2.5% = {h2h.over25_percent}% in range 40-55%")
    reasoning.append(f"\n✅ **STEP 6:** H2H Over 2.5% = {h2h.over25_percent}% in range 40-55%")
    reasoning.append(f"   → Checking BTTS%...")
    
    # ========================================================================
    # STEP 7: BTTS% > 55% + Expected Goals > 2.40 → Over 2.5
    # ========================================================================
    if avg_btts > 55 and expected_goals > 2.40:
        decision_path.append(f"Step 7: BTTS% = {avg_btts}% > 55% AND Expected Goals {expected_goals} > 2.40")
        main_bet = "Over 2.5"
        confidence = "Medium"
        reasoning.append(f"\n✅ **STEP 7:** BTTS% = {avg_btts}% > 55% AND Expected Goals {expected_goals} > 2.40")
        reasoning.append(f"   → **Over 2.5** (Medium confidence)")
        
        secondary = None
        
        # Apply Checkmate
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 8: BTTS% < 40% → Under 2.5
    # ========================================================================
    if avg_btts < 40:
        decision_path.append(f"Step 8: BTTS% = {avg_btts}% < 40%")
        main_bet = "Under 2.5"
        confidence = "Medium"
        reasoning.append(f"\n✅ **STEP 8:** BTTS% = {avg_btts}% < 40%")
        reasoning.append(f"   → **Under 2.5** (Medium confidence)")
        
        secondary = None
        if expected_goals >= 2.20 and h2h.over15_percent >= 65:
            secondary = "Over 1.5"
            reasoning.append(f"\n📌 **Secondary:** Over 1.5 (H2H Over 1.5% = {h2h.over15_percent}% ≥ 65%)")
        
        # Apply Checkmate
        main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            checkmate=checkmate,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 9: All else (borderline) → Under 2.5 (default)
    # ========================================================================
    decision_path.append("Step 9: Borderline case - applying default")
    main_bet = "Under 2.5"
    confidence = "Low"
    reasoning.append(f"\n✅ **STEP 9:** Borderline case (no clear signals)")
    reasoning.append(f"   → **Under 2.5** (Default - Low confidence)")
    
    secondary = None
    if expected_goals >= 2.20 and h2h.over15_percent >= 65:
        secondary = "Over 1.5"
        reasoning.append(f"\n📌 **Secondary:** Over 1.5 (H2H Over 1.5% = {h2h.over15_percent}% ≥ 65%)")
    
    # Apply Checkmate
    main_bet, confidence = apply_checkmate_to_prediction(main_bet, confidence, checkmate)
    
    return PredictionResult(
        main_bet=main_bet,
        confidence=confidence,
        secondary=secondary,
        expected_goals=expected_goals,
        checkmate=checkmate,
        reasoning=reasoning,
        decision_path=decision_path
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def team_stats_input(team_name: str, key_prefix: str, is_home: bool = True) -> TeamStats:
    """Create input fields for team statistics - ONLY the relevant numbers"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    if is_home:
        st.markdown("<div class='input-note'>📌 Enter HOME team's HOME stats only</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            avg_scored_home = st.number_input(
                "🏠 Avg Goals Scored at HOME",
                min_value=0.0, max_value=5.0, value=1.2, step=0.05,
                key=f"{key_prefix}_scored_home"
            )
        with col2:
            avg_conceded_home = st.number_input(
                "🏠 Avg Goals Conceded at HOME",
                min_value=0.0, max_value=5.0, value=1.0, step=0.05,
                key=f"{key_prefix}_conceded_home"
            )
        btts_percent = st.number_input(
            "BTTS % (Overall)",
            min_value=0, max_value=100, value=45, step=5,
            key=f"{key_prefix}_btts"
        )
        return TeamStats(
            name=team_name,
            avg_scored_home=avg_scored_home,
            avg_conceded_home=avg_conceded_home,
            avg_scored_away=0.0,
            avg_conceded_away=0.0,
            btts_percent=float(btts_percent)
        )
    else:
        st.markdown("<div class='input-note'>📌 Enter AWAY team's AWAY stats only</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            avg_scored_away = st.number_input(
                "✈️ Avg Goals Scored AWAY",
                min_value=0.0, max_value=5.0, value=0.8, step=0.05,
                key=f"{key_prefix}_scored_away"
            )
        with col2:
            avg_conceded_away = st.number_input(
                "✈️ Avg Goals Conceded AWAY",
                min_value=0.0, max_value=5.0, value=1.4, step=0.05,
                key=f"{key_prefix}_conceded_away"
            )
        btts_percent = st.number_input(
            "BTTS % (Overall)",
            min_value=0, max_value=100, value=45, step=5,
            key=f"{key_prefix}_btts"
        )
        return TeamStats(
            name=team_name,
            avg_scored_home=0.0,
            avg_conceded_home=0.0,
            avg_scored_away=avg_scored_away,
            avg_conceded_away=avg_conceded_away,
            btts_percent=float(btts_percent)
        )


def h2h_stats_input() -> H2HStats:
    """Create input fields for head-to-head statistics"""
    st.markdown("<div class='h2h-table'><p style='font-weight:700; text-align:center;'>📊 HEAD-TO-HEAD HISTORY</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        matches_played = st.number_input(
            "Matches Played",
            min_value=0, max_value=20, value=6, step=1,
            key="h2h_matches"
        )
        over15_percent = st.number_input(
            "Over 1.5 %",
            min_value=0, max_value=100, value=55, step=5,
            key="h2h_over15"
        )
    with col2:
        over25_percent = st.number_input(
            "Over 2.5 %",
            min_value=0, max_value=100, value=35, step=5,
            key="h2h_over25"
        )
        btts_percent = st.number_input(
            "BTTS %",
            min_value=0, max_value=100, value=40, step=5,
            key="h2h_btts"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return H2HStats(
        matches_played=matches_played,
        over15_percent=float(over15_percent),
        over25_percent=float(over25_percent),
        btts_percent=float(btts_percent)
    )


def league_context_input() -> LeagueContext:
    """Create dropdown for league selection"""
    league_names = list(LEAGUE_CONTEXT.keys())
    selected_league = st.selectbox(
        "League / Competition",
        options=league_names,
        index=0,
        key="league_select"
    )
    return LEAGUE_CONTEXT[selected_league]


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Expected Goals Predictor")
    st.caption("Over 2.5 / Under 2.5 Strict System with Checkmate Safeguards")
    
    st.markdown("""
    <div class="league-note">
        📊 <strong>Formula:</strong> Expected Goals = (Home_Scored_Home + Away_Conceded_Away + Away_Scored_Away + Home_Conceded_Home) / 2<br>
        🛡️ <strong>Checkmate Safeguards:</strong><br>
        &nbsp;&nbsp;&nbsp;• Home scores &lt; 0.90 at home → Strong Under bias<br>
        &nbsp;&nbsp;&nbsp;• Away scores &lt; 0.70 away → Strong Under bias<br>
        &nbsp;&nbsp;&nbsp;• H2H &lt; 8 matches → Reduce confidence<br>
        &nbsp;&nbsp;&nbsp;• Expected Goals vs H2H disagree → Trust the lower
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # TEAM INPUTS
    # ========================================================================
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team", "Home Team", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team", "Away Team", key="away_name")
    
    st.divider()
    
    # Home Team Stats
    st.subheader(f"🏠 {home_name} (HOME Team)")
    home_stats = team_stats_input(home_name, "home", is_home=True)
    
    st.divider()
    
    # Away Team Stats
    st.subheader(f"✈️ {away_name} (AWAY Team)")
    away_stats = team_stats_input(away_name, "away", is_home=False)
    
    st.divider()
    
    # Head-to-Head Stats
    st.subheader("📊 Head-to-Head History")
    h2h_stats = h2h_stats_input()
    
    st.divider()
    
    # League Context
    st.subheader("🏆 League Context")
    league_context = league_context_input()
    
    st.divider()
    
    # ========================================================================
    # PREDICT BUTTON
    # ========================================================================
    if st.button("🔮 PREDICT", type="primary"):
        # Calculate expected goals
        expected_goals = calculate_expected_goals(home_stats, away_stats)
        
        # Make prediction
        result = make_prediction(
            expected_goals=expected_goals,
            h2h=h2h_stats,
            league=league_context,
            home=home_stats,
            away=away_stats
        )
        
        # Display prediction card
        if "Over" in result.main_bet:
            if "Lean" in result.main_bet:
                pred_class = "prediction-lean-over"
                pred_icon = "📈"
            else:
                pred_class = "prediction-over"
                pred_icon = "🔥"
        else:
            pred_class = "prediction-under"
            pred_icon = "❄️"
        
        confidence_class = f"confidence-{result.confidence.lower().replace('-', '')}"
        
        secondary_html = ""
        if result.secondary:
            secondary_html = f'<div class="secondary">📌 Secondary: {result.secondary}</div>'
        
        checkmate_html = ""
        if result.checkmate.red_flags:
            checkmate_html = '<div class="checkmate-warning">' + '<br>'.join(result.checkmate.red_flags) + '</div>'
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="expected-goals">⚽ Expected Goals: {result.expected_goals}</div>
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div class="{confidence_class}">Confidence: {result.confidence}</div>
            {secondary_html}
            {checkmate_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Display the 4 numbers used
        with st.expander("📐 Calculation Details", expanded=True):
            st.markdown(f"""
            <div class="step-box">
            <strong>Formula:</strong> (Home_Scored_Home + Away_Conceded_Away + Away_Scored_Away + Home_Conceded_Home) / 2
            </div>
            <div class="step-box">
            <strong>Numbers used:</strong><br>
            • {home_name} Avg Scored at HOME: {home_stats.avg_scored_home}<br>
            • {away_name} Avg Conceded AWAY: {away_stats.avg_conceded_away}<br>
            • {away_name} Avg Scored AWAY: {away_stats.avg_scored_away}<br>
            • {home_name} Avg Conceded at HOME: {home_stats.avg_conceded_home}<br>
            <strong>= ({home_stats.avg_scored_home} + {away_stats.avg_conceded_away} + {away_stats.avg_scored_away} + {home_stats.avg_conceded_home}) / 2 = {result.expected_goals}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Display decision path
        with st.expander("📋 Decision Path", expanded=True):
            for i, step in enumerate(result.decision_path, 1):
                st.markdown(f'<div class="step-box">{i}. {step}</div>', unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📊 Detailed Analysis", expanded=False):
            for line in result.reasoning:
                if "✅" in line:
                    st.success(line)
                elif "📌" in line:
                    st.info(line)
                elif "⚠️" in line:
                    st.warning(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📈 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_name} Avg Scored (HOME) - Used",
                    f"{home_name} Avg Conceded (HOME) - Used",
                    f"{away_name} Avg Scored (AWAY) - Used",
                    f"{away_name} Avg Conceded (AWAY) - Used",
                    "Expected Goals",
                    "H2H Over 1.5%",
                    "H2H Over 2.5%",
                    "H2H BTTS%",
                    "H2H Matches",
                    "Avg BTTS%",
                    "League Avg Goals/Game"
                ],
                "Value": [
                    f"{home_stats.avg_scored_home}",
                    f"{home_stats.avg_conceded_home}",
                    f"{away_stats.avg_scored_away}",
                    f"{away_stats.avg_conceded_away}",
                    f"{expected_goals}",
                    f"{h2h_stats.over15_percent}%",
                    f"{h2h_stats.over25_percent}%",
                    f"{h2h_stats.btts_percent}%",
                    f"{h2h_stats.matches_played}",
                    f"{(home_stats.btts_percent + away_stats.btts_percent) / 2:.1f}%",
                    f"{league_context.avg_goals_per_game}"
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Strict Decision Rules with Checkmate Safeguards
    
    | Step | Condition | Decision | Confidence | Checkmate Effect |
    |------|-----------|----------|------------|------------------|
    | 1 | Expected Goals **< 2.20** | **Under 2.5** | High | - |
    | 2 | Expected Goals **> 2.80** | **Over 2.5** | High | - |
    | 3 | Expected Goals 2.20 – 2.80 | Go to Step 4 | - | - |
    | 4 | H2H Over 2.5% **> 55%** | **Over 2.5** | Medium | May downgrade to Lean |
    | 5 | H2H Over 2.5% **< 40%** | **Under 2.5** | Medium-High | - |
    | 6 | H2H Over 2.5% 40–55% | Go to Step 7 | - | - |
    | 7 | BTTS% > 55% + Exp Goals > 2.40 | **Over 2.5** | Medium | May downgrade |
    | 8 | BTTS% < 40% | **Under 2.5** | Medium | - |
    | 9 | Borderline | **Under 2.5** (default) | Low | - |
    
    ### 🛡️ Checkmate Safeguards (Red Flags)
    
    | Condition | Effect |
    |-----------|--------|
    | Home team scores **< 0.90** at home | Strong Under bias → Downgrade Over bets |
    | Away team scores **< 0.70** away | Strong Under bias → Downgrade Over bets |
    | H2H sample size **< 8 matches** | Reduce confidence by one level |
    | Expected Goals and H2H disagree strongly | Trust the lower (Under bias) |
    
    ### 🎯 Correct Input Rules
    
    | Team | Use These Stats Only |
    |------|---------------------|
    | **Home Team** | HOME goals scored + HOME goals conceded |
    | **Away Team** | AWAY goals scored + AWAY goals conceded |
    
    **Ignore all other numbers** (the "double" stats are for reference only)
    """)

if __name__ == "__main__":
    main()