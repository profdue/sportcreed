"""
Expected Goals Predictor - Over 2.5 / Under 2.5 System
Based on the philosophy that ALL signals must align for a confident bet.

Core Principle:
- If 3 or more signals point the same way → that is the bet
- If signals are mixed → NO BET (wait for clearer opportunity)

The 5 Signals (in priority order):
1. Expected Goals (<2.20 = Under, >2.80 = Over)
2. Home team scoring avg (<0.90 = Under bias)
3. Away team scoring avg (<0.70 = Under bias)
4. H2H Over 2.5% (<40% = Under, >55% = Over)
5. League avg goals (<2.00 = Under bias)

Decision Rule:
- Count how many signals point to Under
- Count how many signals point to Over
- If Under count >= 3 → Bet Under 2.5
- If Over count >= 3 → Bet Over 2.5
- Else → NO BET (signals conflict)
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
    .prediction-nobet {
        font-size: 2rem;
        font-weight: 700;
        color: #f59e0b;
    }
    .expected-goals {
        font-size: 2rem;
        font-weight: 800;
        color: #60a5fa;
        margin: 0.5rem 0;
    }
    .signal-count {
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .signal-under {
        color: #ef4444;
    }
    .signal-over {
        color: #10b981;
    }
    .signal-neutral {
        color: #94a3b8;
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
    .signal-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
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
    avg_conceded_home: float = 0.0
    avg_scored_away: float = 0.0
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


@dataclass
class SignalResult:
    name: str
    value: float
    direction: str  # "Under", "Over", or "Neutral"
    explanation: str


@dataclass
class PredictionResult:
    main_bet: str  # "Over 2.5", "Under 2.5", or "No Bet"
    expected_goals: float
    under_count: int
    over_count: int
    signals: List[SignalResult]
    reasoning: List[str]


# ============================================================================
# LEAGUE DATA
# ============================================================================
LEAGUE_CONTEXT = {
    "Ethiopian Premier League": LeagueContext("Ethiopian Premier League", 1.80),
    "Argentine Liga Profesional": LeagueContext("Argentine Liga Profesional", 1.90),
    "Ukraine Premier League": LeagueContext("Ukraine Premier League", 2.30),
    "Ligue 2": LeagueContext("Ligue 2", 2.45),
    "Serie B": LeagueContext("Serie B", 2.40),
    "Championship": LeagueContext("Championship", 2.60),
    "Serie A": LeagueContext("Serie A", 2.60),
    "La Liga": LeagueContext("La Liga", 2.50),
    "Ligue 1": LeagueContext("Ligue 1", 2.70),
    "Premier League": LeagueContext("Premier League", 2.80),
    "Bundesliga": LeagueContext("Bundesliga", 3.10),
    "Eredivisie": LeagueContext("Eredivisie", 3.20),
    "Default": LeagueContext("Default", 2.50),
}


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_expected_goals(home: TeamStats, away: TeamStats) -> float:
    """Calculate expected total goals using the 4 key numbers"""
    expected = (home.avg_scored_home + away.avg_conceded_away + away.avg_scored_away + home.avg_conceded_home) / 2
    return round(expected, 2)


def evaluate_signals(
    expected_goals: float,
    home: TeamStats,
    away: TeamStats,
    h2h: H2HStats,
    league: LeagueContext
) -> Tuple[List[SignalResult], int, int]:
    """Evaluate all 5 signals and count Under/Over directions"""
    signals = []
    under_count = 0
    over_count = 0
    
    # Signal 1: Expected Goals
    if expected_goals < 2.20:
        direction = "Under"
        under_count += 1
        explanation = f"Expected Goals {expected_goals} < 2.20"
    elif expected_goals > 2.80:
        direction = "Over"
        over_count += 1
        explanation = f"Expected Goals {expected_goals} > 2.80"
    else:
        direction = "Neutral"
        explanation = f"Expected Goals {expected_goals} in 2.20-2.80 range (neutral)"
    
    signals.append(SignalResult("Expected Goals", expected_goals, direction, explanation))
    
    # Signal 2: Home team scoring avg
    if home.avg_scored_home < 0.90:
        direction = "Under"
        under_count += 1
        explanation = f"Home scores {home.avg_scored_home:.2f} at home (< 0.90)"
    elif home.avg_scored_home > 1.30:
        direction = "Over"
        over_count += 1
        explanation = f"Home scores {home.avg_scored_home:.2f} at home (> 1.30)"
    else:
        direction = "Neutral"
        explanation = f"Home scores {home.avg_scored_home:.2f} at home (neutral range 0.90-1.30)"
    
    signals.append(SignalResult("Home Scoring", home.avg_scored_home, direction, explanation))
    
    # Signal 3: Away team scoring avg
    if away.avg_scored_away < 0.70:
        direction = "Under"
        under_count += 1
        explanation = f"Away scores {away.avg_scored_away:.2f} away (< 0.70)"
    elif away.avg_scored_away > 1.10:
        direction = "Over"
        over_count += 1
        explanation = f"Away scores {away.avg_scored_away:.2f} away (> 1.10)"
    else:
        direction = "Neutral"
        explanation = f"Away scores {away.avg_scored_away:.2f} away (neutral range 0.70-1.10)"
    
    signals.append(SignalResult("Away Scoring", away.avg_scored_away, direction, explanation))
    
    # Signal 4: H2H Over 2.5%
    if h2h.over25_percent > 0:  # Only if data is provided
        if h2h.over25_percent < 40:
            direction = "Under"
            under_count += 1
            explanation = f"H2H Over 2.5% = {h2h.over25_percent}% (< 40%)"
        elif h2h.over25_percent > 55:
            direction = "Over"
            over_count += 1
            explanation = f"H2H Over 2.5% = {h2h.over25_percent}% (> 55%)"
        else:
            direction = "Neutral"
            explanation = f"H2H Over 2.5% = {h2h.over25_percent}% (neutral range 40-55%)"
        
        signals.append(SignalResult("H2H Over 2.5%", h2h.over25_percent, direction, explanation))
    else:
        signals.append(SignalResult("H2H Over 2.5%", 0, "Neutral", "No H2H data provided"))
    
    # Signal 5: League avg goals
    if league.avg_goals_per_game < 2.00:
        direction = "Under"
        under_count += 1
        explanation = f"League avg {league.avg_goals_per_game} goals/game (< 2.00)"
    elif league.avg_goals_per_game > 2.80:
        direction = "Over"
        over_count += 1
        explanation = f"League avg {league.avg_goals_per_game} goals/game (> 2.80)"
    else:
        direction = "Neutral"
        explanation = f"League avg {league.avg_goals_per_game} goals/game (neutral range 2.00-2.80)"
    
    signals.append(SignalResult("League Average", league.avg_goals_per_game, direction, explanation))
    
    return signals, under_count, over_count


def make_prediction(
    expected_goals: float,
    home: TeamStats,
    away: TeamStats,
    h2h: H2HStats,
    league: LeagueContext
) -> PredictionResult:
    """Make prediction based on signal alignment"""
    reasoning = []
    
    signals, under_count, over_count = evaluate_signals(expected_goals, home, away, h2h, league)
    
    reasoning.append(f"📊 **Expected Goals:** {expected_goals}")
    reasoning.append(f"")
    reasoning.append(f"📋 **5 Signals Analysis:**")
    
    for s in signals:
        if s.direction == "Under":
            reasoning.append(f"   • {s.name}: {s.explanation} → 🔽 UNDER")
        elif s.direction == "Over":
            reasoning.append(f"   • {s.name}: {s.explanation} → 🔼 OVER")
        else:
            reasoning.append(f"   • {s.name}: {s.explanation} → ⚪ NEUTRAL")
    
    reasoning.append(f"")
    reasoning.append(f"📊 **Signal Count:** UNDER = {under_count}, OVER = {over_count}")
    
    # Decision rule: Need 3 or more signals pointing same direction
    if under_count >= 3:
        main_bet = "Under 2.5"
        reasoning.append(f"")
        reasoning.append(f"✅ **DECISION:** {under_count} signals point to UNDER (≥ 3) → Bet UNDER 2.5")
    elif over_count >= 3:
        main_bet = "Over 2.5"
        reasoning.append(f"")
        reasoning.append(f"✅ **DECISION:** {over_count} signals point to OVER (≥ 3) → Bet OVER 2.5")
    else:
        main_bet = "No Bet"
        reasoning.append(f"")
        reasoning.append(f"⚠️ **DECISION:** Only {max(under_count, over_count)} signals aligned (< 3) → NO BET")
        reasoning.append(f"   • Signals are conflicting. Wait for clearer opportunity.")
    
    return PredictionResult(
        main_bet=main_bet,
        expected_goals=expected_goals,
        under_count=under_count,
        over_count=over_count,
        signals=signals,
        reasoning=reasoning
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def team_stats_input(team_name: str, key_prefix: str, is_home: bool = True) -> TeamStats:
    """Create input fields for team statistics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    if is_home:
        st.markdown("<div class='input-note'>📌 Enter HOME team's HOME stats only</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            avg_scored_home = st.number_input(
                "🏠 Avg Goals Scored at HOME",
                min_value=0.0, max_value=5.0, value=0.85, step=0.05,
                key=f"{key_prefix}_scored_home"
            )
        with col2:
            avg_conceded_home = st.number_input(
                "🏠 Avg Goals Conceded at HOME",
                min_value=0.0, max_value=5.0, value=0.69, step=0.05,
                key=f"{key_prefix}_conceded_home"
            )
        return TeamStats(
            name=team_name,
            avg_scored_home=avg_scored_home,
            avg_conceded_home=avg_conceded_home,
            avg_scored_away=0.0,
            avg_conceded_away=0.0,
            btts_percent=0.0
        )
    else:
        st.markdown("<div class='input-note'>📌 Enter AWAY team's AWAY stats only</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            avg_scored_away = st.number_input(
                "✈️ Avg Goals Scored AWAY",
                min_value=0.0, max_value=5.0, value=1.00, step=0.05,
                key=f"{key_prefix}_scored_away"
            )
        with col2:
            avg_conceded_away = st.number_input(
                "✈️ Avg Goals Conceded AWAY",
                min_value=0.0, max_value=5.0, value=1.08, step=0.05,
                key=f"{key_prefix}_conceded_away"
            )
        return TeamStats(
            name=team_name,
            avg_scored_home=0.0,
            avg_conceded_home=0.0,
            avg_scored_away=avg_scored_away,
            avg_conceded_away=avg_conceded_away,
            btts_percent=0.0
        )


def h2h_stats_input() -> H2HStats:
    """Create input fields for head-to-head statistics (optional)"""
    st.markdown("<div class='h2h-table'><p style='font-weight:700; text-align:center;'>📊 HEAD-TO-HEAD HISTORY (Optional)</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        matches_played = st.number_input(
            "Matches Played",
            min_value=0, max_value=20, value=6, step=1,
            key="h2h_matches"
        )
        over25_percent = st.number_input(
            "Over 2.5 %",
            min_value=0, max_value=100, value=24, step=5,
            key="h2h_over25"
        )
    with col2:
        st.markdown(" ")  # placeholder
        st.markdown(" ")  # placeholder
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return H2HStats(
        matches_played=matches_played,
        over15_percent=0.0,
        over25_percent=float(over25_percent),
        btts_percent=0.0,
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
    st.caption("Over 2.5 / Under 2.5 | Signal Alignment System")
    
    st.markdown("""
    <div class="league-note">
        📊 <strong>Core Principle:</strong> Bet only when 3 or more of the 5 signals point the same direction.<br>
        🎯 <strong>The 5 Signals:</strong><br>
        &nbsp;&nbsp;&nbsp;1. Expected Goals (&lt;2.20 = Under, &gt;2.80 = Over)<br>
        &nbsp;&nbsp;&nbsp;2. Home scoring avg (&lt;0.90 = Under, &gt;1.30 = Over)<br>
        &nbsp;&nbsp;&nbsp;3. Away scoring avg (&lt;0.70 = Under, &gt;1.10 = Over)<br>
        &nbsp;&nbsp;&nbsp;4. H2H Over 2.5% (&lt;40% = Under, &gt;55% = Over)<br>
        &nbsp;&nbsp;&nbsp;5. League avg goals (&lt;2.00 = Under, &gt;2.80 = Over)
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # TEAM INPUTS
    # ========================================================================
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team", "Bahir Dar Kenema FC", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team", "Sheger Ketema", key="away_name")
    
    st.divider()
    
    # Home Team Stats
    st.subheader(f"🏠 {home_name} (HOME Team)")
    home_stats = team_stats_input(home_name, "home", is_home=True)
    
    st.divider()
    
    # Away Team Stats
    st.subheader(f"✈️ {away_name} (AWAY Team)")
    away_stats = team_stats_input(away_name, "away", is_home=False)
    
    st.divider()
    
    # Head-to-Head Stats (Optional)
    st.subheader("📊 Head-to-Head History (Optional)")
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
            home=home_stats,
            away=away_stats,
            h2h=h2h_stats,
            league=league_context
        )
        
        # Display prediction card
        if result.main_bet == "Under 2.5":
            pred_class = "prediction-under"
            pred_icon = "❄️"
        elif result.main_bet == "Over 2.5":
            pred_class = "prediction-over"
            pred_icon = "🔥"
        else:
            pred_class = "prediction-nobet"
            pred_icon = "⏸️"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="expected-goals">⚽ Expected Goals: {result.expected_goals}</div>
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div class="signal-count">
                📊 Signal Count: <span class="signal-under">UNDER {result.under_count}</span> | 
                <span class="signal-over">OVER {result.over_count}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📋 Detailed Analysis", expanded=True):
            for line in result.reasoning:
                if "✅" in line:
                    st.success(line)
                elif "⚠️" in line:
                    st.warning(line)
                elif "🔽" in line:
                    st.error(line)
                elif "🔼" in line:
                    st.success(line)
                elif "⚪" in line:
                    st.info(line)
                else:
                    st.write(line)
        
        # Display signals table
        with st.expander("📊 Signal Details", expanded=False):
            data = {
                "Signal": [],
                "Value": [],
                "Direction": [],
                "Explanation": []
            }
            for s in result.signals:
                data["Signal"].append(s.name)
                data["Value"].append(s.value)
                data["Direction"].append(s.direction)
                data["Explanation"].append(s.explanation)
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 How to Use
    
    1. Enter **Home Team** and **Away Team** names
    2. Enter **Home Team's HOME stats** (goals scored + conceded)
    3. Enter **Away Team's AWAY stats** (goals scored + conceded)
    4. Enter **H2H Over 2.5%** (optional, improves accuracy)
    5. Select the **League**
    6. Click **PREDICT**
    
    ### 🎯 Decision Rule
    
    | Signal Count | Action |
    |--------------|--------|
    | UNDER ≥ 3 | Bet **Under 2.5** |
    | OVER ≥ 3 | Bet **Over 2.5** |
    | Neither ≥ 3 | **NO BET** (signals conflict) |
    
    ### 📊 The 5 Signals
    
    | Signal | Under Threshold | Over Threshold |
    |--------|----------------|----------------|
    | Expected Goals | < 2.20 | > 2.80 |
    | Home Scoring Avg | < 0.90 | > 1.30 |
    | Away Scoring Avg | < 0.70 | > 1.10 |
    | H2H Over 2.5% | < 40% | > 55% |
    | League Avg Goals | < 2.00 | > 2.80 |
    """)

if __name__ == "__main__":
    main()