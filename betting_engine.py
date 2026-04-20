"""
Expected Goals Predictor - Over 2.5 / Under 2.5 Strict System
Based on statistical analysis of home/away averages, H2H history, and league context.

Core Formula:
Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2

Strict Decision Rules (Follow in Order):
1. Expected Goals < 2.20 → Under 2.5 (High)
2. Expected Goals > 2.80 → Over 2.5 (High)
3. Expected Goals 2.20 – 2.80 → Check H2H Over 2.5%
4. H2H Over 2.5% > 50% → Over 2.5 (Medium-High)
5. H2H Over 2.5% < 40% → Under 2.5 (Medium-High)
6. H2H Over 2.5% 40-50% → Check BTTS%
7. BTTS% > 55% + Expected Goals > 2.40 → Over 2.5 (Medium)
8. BTTS% < 40% → Under 2.5 (Medium)
9. All else (borderline) → Under 2.5 (Low - default)

Secondary Suggestion (Over 1.5 only):
- Only suggest when Expected Goals ≥ 2.20 AND H2H Over 1.5 ≥ 65%
- Never make Over 1.5 main recommendation
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

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
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fbbf24;
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
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamStats:
    name: str
    avg_scored_home: float = 0.0
    avg_scored_away: float = 0.0
    avg_conceded_home: float = 0.0
    avg_conceded_away: float = 0.0
    btts_percent: float = 0.0


@dataclass
class H2HStats:
    matches_played: int = 0
    over15_percent: float = 0.0
    over25_percent: float = 0.0
    btts_percent: float = 0.0
    avg_goals: float = 0.0


@dataclass
class LeagueContext:
    name: str
    avg_goals_per_game: float
    is_low_scoring: bool = False


@dataclass
class PredictionResult:
    main_bet: str  # "Over 2.5" or "Under 2.5"
    confidence: str  # "High", "Medium-High", "Medium", "Low"
    secondary: Optional[str]  # "Over 1.5" or None
    expected_goals: float
    reasoning: list
    decision_path: list


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
    "Default": LeagueContext("Default", 2.50, is_low_scoring=False),
}


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_expected_goals(home: TeamStats, away: TeamStats) -> float:
    """
    Calculate expected total goals using the formula:
    Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2
    """
    expected = (home.avg_scored_home + away.avg_conceded_away + away.avg_scored_away + home.avg_conceded_home) / 2
    return round(expected, 2)


def make_prediction(
    expected_goals: float,
    h2h: H2HStats,
    league: LeagueContext,
    home_btts: float = 0.0,
    away_btts: float = 0.0
) -> PredictionResult:
    """
    Make Over 2.5 / Under 2.5 prediction based on strict decision rules.
    """
    reasoning = []
    decision_path = []
    
    avg_btts = (home_btts + away_btts) / 2 if home_btts > 0 and away_btts > 0 else h2h.btts_percent
    
    reasoning.append(f"📊 **Input Summary:**")
    reasoning.append(f"   • Expected Goals: {expected_goals}")
    reasoning.append(f"   • H2H Over 2.5%: {h2h.over25_percent}%")
    reasoning.append(f"   • H2H Over 1.5%: {h2h.over15_percent}%")
    reasoning.append(f"   • H2H BTTS%: {h2h.btts_percent}%")
    reasoning.append(f"   • League avg: {league.avg_goals_per_game} goals/game")
    reasoning.append(f"   • League low-scoring: {league.is_low_scoring}")
    reasoning.append(f"   • Avg BTTS%: {avg_btts}%")
    
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
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
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
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
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
    # STEP 4: H2H Over 2.5% > 50% → Over 2.5
    # ========================================================================
    if h2h.over25_percent > 50:
        decision_path.append(f"Step 4: H2H Over 2.5% = {h2h.over25_percent}% > 50%")
        main_bet = "Over 2.5"
        confidence = "Medium-High"
        reasoning.append(f"\n✅ **STEP 4:** H2H Over 2.5% = {h2h.over25_percent}% > 50%")
        reasoning.append(f"   → **Over 2.5** (Medium-High confidence)")
        
        secondary = None
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
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
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # ========================================================================
    # STEP 6: H2H Over 2.5% between 40-50% → Go to Step 7
    # ========================================================================
    decision_path.append(f"Step 6: H2H Over 2.5% = {h2h.over25_percent}% in range 40-50%")
    reasoning.append(f"\n✅ **STEP 6:** H2H Over 2.5% = {h2h.over25_percent}% in range 40-50%")
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
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
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
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            secondary=secondary,
            expected_goals=expected_goals,
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
    
    return PredictionResult(
        main_bet=main_bet,
        confidence=confidence,
        secondary=secondary,
        expected_goals=expected_goals,
        reasoning=reasoning,
        decision_path=decision_path
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def team_stats_input(team_name: str, key_prefix: str) -> TeamStats:
    """Create input fields for team statistics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p style='text-align:center; font-weight:700;'>🏠 HOME</p>", unsafe_allow_html=True)
        avg_scored_home = st.number_input(
            "Avg Goals Scored",
            min_value=0.0, max_value=5.0, value=1.2, step=0.05,
            key=f"{key_prefix}_scored_home"
        )
        avg_conceded_home = st.number_input(
            "Avg Goals Conceded",
            min_value=0.0, max_value=5.0, value=1.0, step=0.05,
            key=f"{key_prefix}_conceded_home"
        )
    
    with col2:
        st.markdown("<p style='text-align:center; font-weight:700;'>✈️ AWAY</p>", unsafe_allow_html=True)
        avg_scored_away = st.number_input(
            "Avg Goals Scored",
            min_value=0.0, max_value=5.0, value=0.8, step=0.05,
            key=f"{key_prefix}_scored_away"
        )
        avg_conceded_away = st.number_input(
            "Avg Goals Conceded",
            min_value=0.0, max_value=5.0, value=1.4, step=0.05,
            key=f"{key_prefix}_conceded_away"
        )
    
    btts_percent = st.number_input(
        "BTTS %",
        min_value=0, max_value=100, value=45, step=5,
        key=f"{key_prefix}_btts",
        help="Percentage of matches where both teams scored"
    )
    
    return TeamStats(
        name=team_name,
        avg_scored_home=avg_scored_home,
        avg_scored_away=avg_scored_away,
        avg_conceded_home=avg_conceded_home,
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
        btts_percent=float(btts_percent),
        avg_goals=0.0
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
    st.caption("Over 2.5 / Under 2.5 Strict System | Statistical Model")
    
    st.markdown("""
    <div class="league-note">
        📊 <strong>Formula:</strong> Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2<br>
        🎯 <strong>Strict Rules:</strong> &lt;2.20 = Under | &gt;2.80 = Over | Otherwise check H2H Over 2.5% | Borderline defaults to Under
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
    st.subheader(f"🏠 {home_name} Statistics")
    home_stats = team_stats_input(home_name, "home")
    
    st.divider()
    
    # Away Team Stats
    st.subheader(f"✈️ {away_name} Statistics")
    away_stats = team_stats_input(away_name, "away")
    
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
            home_btts=home_stats.btts_percent,
            away_btts=away_stats.btts_percent
        )
        
        # Display prediction card
        if "Over" in result.main_bet:
            pred_class = "prediction-over"
            pred_icon = "🔥"
        else:
            pred_class = "prediction-under"
            pred_icon = "❄️"
        
        confidence_class = f"confidence-{result.confidence.lower().replace('-', '')}"
        
        secondary_html = ""
        if result.secondary:
            secondary_html = f'<div class="secondary">📌 Secondary: {result.secondary}</div>'
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="expected-goals">⚽ Expected Goals: {result.expected_goals}</div>
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div class="{confidence_class}">Confidence: {result.confidence}</div>
            {secondary_html}
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
                elif "📊" in line:
                    st.info(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📈 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_name} Avg Scored (Home)",
                    f"{home_name} Avg Conceded (Home)",
                    f"{away_name} Avg Scored (Away)",
                    f"{away_name} Avg Conceded (Away)",
                    "Expected Goals",
                    "H2H Over 1.5%",
                    "H2H Over 2.5%",
                    "H2H BTTS%",
                    "Avg BTTS%",
                    "League Avg Goals/Game",
                    "Low Scoring League?"
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
                    f"{(home_stats.btts_percent + away_stats.btts_percent) / 2:.1f}%",
                    f"{league_context.avg_goals_per_game}",
                    "Yes" if league_context.is_low_scoring else "No"
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Strict Decision Rules (Follow in Order)
    
    | Step | Condition | Decision | Confidence |
    |------|-----------|----------|------------|
    | 1 | Expected Goals **< 2.20** | **Under 2.5** | High |
    | 2 | Expected Goals **> 2.80** | **Over 2.5** | High |
    | 3 | Expected Goals 2.20 – 2.80 | Go to Step 4 | - |
    | 4 | H2H Over 2.5% **> 50%** | **Over 2.5** | Medium-High |
    | 5 | H2H Over 2.5% **< 40%** | **Under 2.5** | Medium-High |
    | 6 | H2H Over 2.5% 40–50% | Go to Step 7 | - |
    | 7 | BTTS% > 55% + Exp Goals > 2.40 | **Over 2.5** | Medium |
    | 8 | BTTS% < 40% | **Under 2.5** | Medium |
    | 9 | All else (borderline) | **Under 2.5** (default) | Low |
    
    ### 📌 Secondary Suggestion (Over 1.5)
    
    Only suggested when:
    - Expected Goals ≥ 2.20 **AND**
    - H2H Over 1.5% ≥ 65%
    
    ### 🎯 How to Use
    
    1. Enter **Home Team** and **Away Team** names
    2. Enter each team's **home/away averages**
    3. Enter **Head-to-Head percentages**
    4. Select the **League**
    5. Click **PREDICT**
    """)

if __name__ == "__main__":
    main()