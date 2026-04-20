"""
Streak Predictor - Expected Goals System (Over/Under Only)
Based on statistical analysis of home/away averages, H2H history, and league context.

Core Formula:
Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2

Decision Thresholds:
- < 2.10 → Strong Under 2.5
- 2.10 – 2.50 → Lean Under 2.5 or Over 1.5 (check H2H)
- 2.51 – 2.90 → Lean Over 1.5
- > 2.90 → Strong Over 2.5

Supporting Filters:
- H2H Over 1.5% (>65% = Over 1.5 safe)
- H2H Over 2.5% (>50% = Over 2.5 likely)
- H2H BTTS% (>50% = goals expected)
- League baseline (goals/game)
- BTTS% (<35% = Under lean)
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - Expected Goals",
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
    .prediction-strong-over {
        font-size: 2rem;
        font-weight: 800;
        color: #10b981;
    }
    .prediction-lean-over {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .prediction-strong-under {
        font-size: 2rem;
        font-weight: 800;
        color: #ef4444;
    }
    .prediction-lean-under {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f97316;
    }
    .expected-goals {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0.5rem 0;
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
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8;
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
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamStats:
    """Team statistical data"""
    name: str
    avg_scored_home: float = 0.0
    avg_scored_away: float = 0.0
    avg_conceded_home: float = 0.0
    avg_conceded_away: float = 0.0
    btts_percent: float = 0.0
    over15_percent: float = 0.0
    over25_percent: float = 0.0


@dataclass
class H2HStats:
    """Head-to-head statistics"""
    matches_played: int = 0
    over15_percent: float = 0.0
    over25_percent: float = 0.0
    btts_percent: float = 0.0
    avg_goals: float = 0.0


@dataclass
class LeagueContext:
    """League baseline information"""
    name: str
    avg_goals_per_game: float
    is_low_scoring: bool = False
    is_high_scoring: bool = False


@dataclass
class PredictionResult:
    prediction: str  # "Strong Over 2.5", "Lean Over 1.5", "Lean Under 2.5", "Strong Under 2.5"
    confidence: str  # "High", "Medium", "Low"
    expected_goals: float
    reasoning: list
    details: dict


# ============================================================================
# LEAGUE DATA
# ============================================================================
LEAGUE_CONTEXT = {
    "Ligue 2": LeagueContext("Ligue 2", 2.45, is_low_scoring=False, is_high_scoring=False),
    "Ukraine": LeagueContext("Ukraine Premier League", 2.30, is_low_scoring=False, is_high_scoring=False),
    "Ethiopia": LeagueContext("Ethiopian Premier League", 1.80, is_low_scoring=True, is_high_scoring=False),
    "Argentina": LeagueContext("Argentine Liga Profesional", 1.90, is_low_scoring=True, is_high_scoring=False),
    "Bundesliga": LeagueContext("Bundesliga", 3.10, is_low_scoring=False, is_high_scoring=True),
    "Premier League": LeagueContext("Premier League", 2.80, is_low_scoring=False, is_high_scoring=False),
    "Serie A": LeagueContext("Serie A", 2.60, is_low_scoring=False, is_high_scoring=False),
    "La Liga": LeagueContext("La Liga", 2.50, is_low_scoring=False, is_high_scoring=False),
    "Ligue 1": LeagueContext("Ligue 1", 2.70, is_low_scoring=False, is_high_scoring=False),
    "Eredivisie": LeagueContext("Eredivisie", 3.20, is_low_scoring=False, is_high_scoring=True),
    "Default": LeagueContext("Default", 2.50, is_low_scoring=False, is_high_scoring=False),
}


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_expected_goals(home: TeamStats, away: TeamStats, is_home_match: bool = True) -> float:
    """
    Calculate expected total goals using the formula:
    Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2
    """
    if is_home_match:
        home_score = home.avg_scored_home
        home_concede = home.avg_conceded_home
        away_score = away.avg_scored_away
        away_concede = away.avg_conceded_away
    else:
        # If teams are swapped (away team playing at home)
        home_score = away.avg_scored_home
        home_concede = away.avg_conceded_home
        away_score = home.avg_scored_away
        away_concede = home.avg_conceded_away
    
    expected = (home_score + away_concede + away_score + home_concede) / 2
    return round(expected, 2)


def make_prediction(
    expected_goals: float,
    h2h: H2HStats,
    league: LeagueContext,
    home_btts: float = 0.0,
    away_btts: float = 0.0
) -> PredictionResult:
    """
    Make Over/Under prediction based on expected goals and supporting filters.
    """
    reasoning = []
    details = {
        "expected_goals": expected_goals,
        "h2h_over15": h2h.over15_percent,
        "h2h_over25": h2h.over25_percent,
        "h2h_btts": h2h.btts_percent,
        "league_avg": league.avg_goals_per_game,
    }
    
    avg_btts = (home_btts + away_btts) / 2 if home_btts > 0 and away_btts > 0 else h2h.btts_percent
    
    reasoning.append(f"📊 **Expected Goals Calculation:** {expected_goals}")
    reasoning.append(f"   • H2H Over 1.5: {h2h.over15_percent}%")
    reasoning.append(f"   • H2H Over 2.5: {h2h.over25_percent}%")
    reasoning.append(f"   • H2H BTTS: {h2h.btts_percent}%")
    reasoning.append(f"   • League average: {league.avg_goals_per_game} goals/game")
    
    # Decision logic
    if expected_goals < 2.10:
        if league.is_low_scoring or avg_btts < 35:
            prediction = "Strong Under 2.5"
            confidence = "High"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} < 2.10 + low scoring context → Strong Under 2.5**")
        else:
            prediction = "Lean Under 2.5"
            confidence = "Medium"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} < 2.10 → Lean Under 2.5**")
    
    elif expected_goals <= 2.50:
        # 2.10 - 2.50 range
        if h2h.over25_percent > 50:
            prediction = "Over 2.5"
            confidence = "Medium"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} + H2H Over 2.5 {h2h.over25_percent}% > 50% → Over 2.5**")
        elif h2h.over15_percent > 65:
            prediction = "Over 1.5"
            confidence = "High"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} + H2H Over 1.5 {h2h.over15_percent}% > 65% → Over 1.5**")
        elif league.is_low_scoring or avg_btts < 35:
            prediction = "Under 2.5"
            confidence = "Medium"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} + low scoring context → Under 2.5**")
        else:
            prediction = "Lean Over 1.5"
            confidence = "Low"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} → Lean Over 1.5 (safest)**")
    
    elif expected_goals <= 2.90:
        # 2.51 - 2.90 range
        if h2h.over25_percent > 50:
            prediction = "Strong Over 2.5"
            confidence = "High"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} + H2H Over 2.5 {h2h.over25_percent}% > 50% → Strong Over 2.5**")
        else:
            prediction = "Lean Over 1.5"
            confidence = "Medium"
            reasoning.append(f"\n✅ **Expected Goals {expected_goals} → Lean Over 1.5**")
    
    else:  # > 2.90
        prediction = "Strong Over 2.5"
        confidence = "High"
        reasoning.append(f"\n✅ **Expected Goals {expected_goals} > 2.90 → Strong Over 2.5**")
    
    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        expected_goals=expected_goals,
        reasoning=reasoning,
        details=details
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def team_stats_input(team_name: str, key_prefix: str, is_home: bool = True) -> TeamStats:
    """Create input fields for team statistics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p style='text-align:center; font-weight:700;'>🏠 HOME</p>", unsafe_allow_html=True)
        avg_scored_home = st.number_input(
            "Avg Goals Scored",
            min_value=0.0, max_value=5.0, value=1.2, step=0.05,
            key=f"{key_prefix}_scored_home",
            help="Average goals scored when playing at home"
        )
        avg_conceded_home = st.number_input(
            "Avg Goals Conceded",
            min_value=0.0, max_value=5.0, value=1.0, step=0.05,
            key=f"{key_prefix}_conceded_home",
            help="Average goals conceded when playing at home"
        )
    
    with col2:
        st.markdown("<p style='text-align:center; font-weight:700;'>✈️ AWAY</p>", unsafe_allow_html=True)
        avg_scored_away = st.number_input(
            "Avg Goals Scored",
            min_value=0.0, max_value=5.0, value=0.8, step=0.05,
            key=f"{key_prefix}_scored_away",
            help="Average goals scored when playing away"
        )
        avg_conceded_away = st.number_input(
            "Avg Goals Conceded",
            min_value=0.0, max_value=5.0, value=1.4, step=0.05,
            key=f"{key_prefix}_conceded_away",
            help="Average goals conceded when playing away"
        )
    
    col3, col4 = st.columns(2)
    with col3:
        btts_percent = st.number_input(
            "BTTS %",
            min_value=0, max_value=100, value=45, step=5,
            key=f"{key_prefix}_btts",
            help="Percentage of matches where both teams scored"
        )
    with col4:
        st.markdown(" ")  # placeholder
    
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        matches_played = st.number_input(
            "Matches Played",
            min_value=0, max_value=20, value=6, step=1,
            key="h2h_matches"
        )
    with col2:
        over15_percent = st.number_input(
            "Over 1.5 %",
            min_value=0, max_value=100, value=55, step=5,
            key="h2h_over15",
            help="Percentage of H2H matches with 2+ goals"
        )
    with col3:
        over25_percent = st.number_input(
            "Over 2.5 %",
            min_value=0, max_value=100, value=35, step=5,
            key="h2h_over25",
            help="Percentage of H2H matches with 3+ goals"
        )
    
    col4, col5 = st.columns(2)
    with col4:
        btts_percent = st.number_input(
            "BTTS %",
            min_value=0, max_value=100, value=40, step=5,
            key="h2h_btts",
            help="Percentage of H2H matches where both teams scored"
        )
    with col5:
        avg_goals = st.number_input(
            "Avg Goals",
            min_value=0.0, max_value=6.0, value=2.2, step=0.1,
            key="h2h_avg_goals",
            help="Average total goals in H2H matches"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return H2HStats(
        matches_played=matches_played,
        over15_percent=float(over15_percent),
        over25_percent=float(over25_percent),
        btts_percent=float(btts_percent),
        avg_goals=avg_goals
    )


def league_context_input() -> LeagueContext:
    """Create dropdown for league selection"""
    league_names = list(LEAGUE_CONTEXT.keys())
    selected_league = st.selectbox(
        "League / Competition",
        options=league_names,
        index=0,
        key="league_select",
        help="Select the league to apply baseline averages"
    )
    return LEAGUE_CONTEXT[selected_league]


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Expected Goals Predictor")
    st.caption("Over/Under System | Statistical Model | 100% Data-Driven")
    
    st.markdown("""
    <div class="league-note">
        📊 <strong>Formula:</strong> Expected Goals = (Home_Scored_Avg + Away_Conceded_Avg + Away_Scored_Avg + Home_Conceded_Avg) / 2<br>
        🎯 <strong>Thresholds:</strong> &lt;2.10 = Under | 2.10-2.50 = Lean | 2.51-2.90 = Over 1.5 | &gt;2.90 = Over 2.5
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
    home_stats = team_stats_input(home_name, "home", is_home=True)
    
    st.divider()
    
    # Away Team Stats
    st.subheader(f"✈️ {away_name} Statistics")
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
        expected_goals = calculate_expected_goals(home_stats, away_stats, is_home_match=True)
        
        # Make prediction
        result = make_prediction(
            expected_goals=expected_goals,
            h2h=h2h_stats,
            league=league_context,
            home_btts=home_stats.btts_percent,
            away_btts=away_stats.btts_percent
        )
        
        # Display prediction card
        if "Strong Over" in result.prediction:
            pred_class = "prediction-strong-over"
            pred_icon = "🔥"
        elif "Lean Over" in result.prediction or "Over 1.5" in result.prediction or "Over 2.5" in result.prediction:
            pred_class = "prediction-lean-over"
            pred_icon = "📈"
        elif "Strong Under" in result.prediction:
            pred_class = "prediction-strong-under"
            pred_icon = "❄️"
        else:
            pred_class = "prediction-lean-under"
            pred_icon = "📉"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="expected-goals">⚽ Expected Goals: {result.expected_goals}</div>
            <div class="{pred_class}">{pred_icon} {result.prediction}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                Confidence: {result.confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📋 Detailed Analysis", expanded=True):
            for line in result.reasoning:
                if "✅" in line:
                    st.success(line)
                elif "📊" in line:
                    st.info(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📊 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_name} Avg Scored (Home)",
                    f"{home_name} Avg Conceded (Home)",
                    f"{away_name} Avg Scored (Away)",
                    f"{away_name} Avg Conceded (Away)",
                    "H2H Over 1.5%",
                    "H2H Over 2.5%",
                    "H2H BTTS%",
                    "League Avg Goals/Game"
                ],
                "Value": [
                    f"{home_stats.avg_scored_home}",
                    f"{home_stats.avg_conceded_home}",
                    f"{away_stats.avg_scored_away}",
                    f"{away_stats.avg_conceded_away}",
                    f"{h2h_stats.over15_percent}%",
                    f"{h2h_stats.over25_percent}%",
                    f"{h2h_stats.btts_percent}%",
                    f"{league_context.avg_goals_per_game}"
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 How to Use
    
    1. Enter **Home Team** and **Away Team** names
    2. Enter each team's **statistics**:
       - Average goals scored at home / away
       - Average goals conceded at home / away
       - BTTS percentage (optional)
    3. Enter **Head-to-Head** statistics:
       - Over 1.5%, Over 2.5%, BTTS%, Avg Goals
    4. Select the **League** for baseline context
    5. Click **PREDICT**
    
    ### 📊 Decision Thresholds
    
    | Expected Goals | Prediction | Confidence Trigger |
    |----------------|------------|---------------------|
    | < 2.10 | Strong Under 2.5 | + low BTTS or low-scoring league |
    | 2.10 – 2.50 | Lean Under 2.5 / Over 1.5 | Check H2H Over 1.5% > 65% |
    | 2.51 – 2.90 | Lean Over 1.5 | Safe play |
    | > 2.90 | Strong Over 2.5 | + H2H Over 2.5% > 50% |
    
    ### 🎯 Supporting Filters
    
    - H2H Over 1.5% > 65% → Over 1.5 is very safe
    - H2H Over 2.5% > 50% → Over 2.5 likely
    - League low-scoring (<2.0) → Under 2.5 lean
    - BTTS% < 35% → Under 2.5 lean
    """)

if __name__ == "__main__":
    main()