"""
Streak Predictor - New Logic
Based on 100%/0% metrics scanning and market-based betting.

Core Principles:
1. NEVER predict a winner (no "Team A will win")
2. Only predict goal markets: Over 2.5, Under 2.5, Over 3.5, Under 3.5, BTTS Yes, BTTS No
3. Find 100% or 0% metrics first (anchor bet)
4. Then use tiered thresholds (75%+ / 25%-)

Rules Priority:
1. Scan for 100% or 0% metrics → Anchor bet
2. Check Team Total Over 2.5 for BOTH teams
3. Check Conceded Over 0.5 vs Failed to Score conflict
4. Check BTTS & Over 2.5 correlation
5. Output goal-market bets only
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - Goal Markets",
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
    .prediction-market {
        font-size: 1.8rem;
        font-weight: 800;
        color: #10b981;
        margin: 0.5rem 0;
    }
    .prediction-anchor {
        font-size: 1.2rem;
        font-weight: 600;
        color: #fbbf24;
        margin-bottom: 1rem;
    }
    .confidence-high {
        color: #10b981;
    }
    .confidence-medium {
        color: #fbbf24;
    }
    .confidence-low {
        color: #f97316;
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
    .metric-highlight {
        background: #064e3b;
        border-left: 3px solid #10b981;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
    }
    .metric-warning {
        background: #7f1a1a;
        border-left: 3px solid #ef4444;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamMetrics:
    """Team performance metrics (percentages)"""
    name: str
    over15_percent: float = 0.0      # % of matches with 2+ goals
    over25_percent: float = 0.0      # % of matches with 3+ goals
    over35_percent: float = 0.0      # % of matches with 4+ goals
    btts_percent: float = 0.0        # % of matches where both teams scored
    scored_over15_percent: float = 0.0   # % of matches where team scored 2+ goals
    scored_over25_percent: float = 0.0   # % of matches where team scored 3+ goals
    conceded_over05_percent: float = 0.0 # % of matches where team conceded
    failed_to_score_percent: float = 0.0 # % of matches where team failed to score
    clean_sheet_percent: float = 0.0     # % of matches where team kept clean sheet


@dataclass
class MatchPrediction:
    market: str          # "Over 2.5", "Under 2.5", "Over 3.5", "Under 3.5", "BTTS Yes", "BTTS No"
    confidence: str      # "High", "Medium", "Low"
    anchor_metric: str   # What triggered this bet
    reasoning: List[str]


# ============================================================================
# SCANNING FUNCTIONS
# ============================================================================
def scan_100_percent(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Find 100% metrics → anchor bet"""
    predictions = []
    
    # Home team 100% metrics
    if home.over25_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="High",
            anchor_metric=f"{home.name} has 100% Over 2.5 rate",
            reasoning=[f"{home.name} has Over 2.5 in 100% of matches → Anchor bet Over 2.5"]
        ))
    
    if home.over35_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 3.5",
            confidence="High",
            anchor_metric=f"{home.name} has 100% Over 3.5 rate",
            reasoning=[f"{home.name} has Over 3.5 in 100% of matches → Anchor bet Over 3.5"]
        ))
    
    if home.scored_over25_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="High",
            anchor_metric=f"{home.name} scores 3+ goals in 100% of matches",
            reasoning=[f"{home.name} always scores 3+ goals → Anchor bet Over 2.5"]
        ))
    
    if home.clean_sheet_percent == 100:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{home.name} has 100% clean sheets",
            reasoning=[f"{home.name} never concedes → Anchor bet Under 2.5"]
        ))
    
    if home.failed_to_score_percent == 100:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{home.name} fails to score in 100% of matches",
            reasoning=[f"{home.name} never scores → Anchor bet Under 2.5"]
        ))
    
    # Away team 100% metrics
    if away.over25_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="High",
            anchor_metric=f"{away.name} has 100% Over 2.5 rate",
            reasoning=[f"{away.name} has Over 2.5 in 100% of matches → Anchor bet Over 2.5"]
        ))
    
    if away.over35_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 3.5",
            confidence="High",
            anchor_metric=f"{away.name} has 100% Over 3.5 rate",
            reasoning=[f"{away.name} has Over 3.5 in 100% of matches → Anchor bet Over 3.5"]
        ))
    
    if away.scored_over25_percent == 100:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="High",
            anchor_metric=f"{away.name} scores 3+ goals in 100% of matches",
            reasoning=[f"{away.name} always scores 3+ goals → Anchor bet Over 2.5"]
        ))
    
    if away.clean_sheet_percent == 100:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{away.name} has 100% clean sheets",
            reasoning=[f"{away.name} never concedes → Anchor bet Under 2.5"]
        ))
    
    if away.failed_to_score_percent == 100:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{away.name} fails to score in 100% of matches",
            reasoning=[f"{away.name} never scores → Anchor bet Under 2.5"]
        ))
    
    return predictions


def scan_0_percent(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Find 0% metrics → anchor bet (reverse)"""
    predictions = []
    
    # Home team 0% metrics
    if home.over25_percent == 0:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{home.name} has 0% Over 2.5 rate",
            reasoning=[f"{home.name} never goes Over 2.5 → Anchor bet Under 2.5"]
        ))
    
    if home.over35_percent == 0:
        predictions.append(MatchPrediction(
            market="Under 3.5",
            confidence="High",
            anchor_metric=f"{home.name} has 0% Over 3.5 rate",
            reasoning=[f"{home.name} never goes Over 3.5 → Anchor bet Under 3.5"]
        ))
    
    if home.btts_percent == 0:
        predictions.append(MatchPrediction(
            market="BTTS No",
            confidence="High",
            anchor_metric=f"{home.name} has 0% BTTS rate",
            reasoning=[f"{home.name} never has both teams score → Anchor bet BTTS No"]
        ))
    
    if home.clean_sheet_percent == 0:
        predictions.append(MatchPrediction(
            market="Over 0.5",  # They always concede
            confidence="Medium",
            anchor_metric=f"{home.name} has 0% clean sheets",
            reasoning=[f"{home.name} always concedes → Supports goals"]
        ))
    
    # Away team 0% metrics
    if away.over25_percent == 0:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="High",
            anchor_metric=f"{away.name} has 0% Over 2.5 rate",
            reasoning=[f"{away.name} never goes Over 2.5 → Anchor bet Under 2.5"]
        ))
    
    if away.over35_percent == 0:
        predictions.append(MatchPrediction(
            market="Under 3.5",
            confidence="High",
            anchor_metric=f"{away.name} has 0% Over 3.5 rate",
            reasoning=[f"{away.name} never goes Over 3.5 → Anchor bet Under 3.5"]
        ))
    
    if away.btts_percent == 0:
        predictions.append(MatchPrediction(
            market="BTTS No",
            confidence="High",
            anchor_metric=f"{away.name} has 0% BTTS rate",
            reasoning=[f"{away.name} never has both teams score → Anchor bet BTTS No"]
        ))
    
    if away.clean_sheet_percent == 0:
        predictions.append(MatchPrediction(
            market="Over 0.5",
            confidence="Medium",
            anchor_metric=f"{away.name} has 0% clean sheets",
            reasoning=[f"{away.name} always concedes → Supports goals"]
        ))
    
    return predictions


def check_team_total_over25(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Check Team Total Over 2.5 for BOTH teams"""
    predictions = []
    
    home_over25_rate = home.scored_over25_percent
    away_over25_rate = away.scored_over25_percent
    
    if home_over25_rate == 0 and away_over25_rate == 0:
        predictions.append(MatchPrediction(
            market="Under 3.5",
            confidence="High",
            anchor_metric="Both teams have 0% chance of scoring 3+ goals",
            reasoning=[
                f"{home.name} scores 3+ goals in {home_over25_rate}% of matches",
                f"{away.name} scores 3+ goals in {away_over25_rate}% of matches",
                "→ Bet Under 3.5 or Under 2.5"
            ]
        ))
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="Medium",
            anchor_metric="Both teams have 0% chance of scoring 3+ goals",
            reasoning=["Both teams unlikely to score multiple goals → Under 2.5"]
        ))
    
    elif home_over25_rate >= 50 or away_over25_rate >= 50:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="Medium",
            anchor_metric=f"One team scores 3+ in {max(home_over25_rate, away_over25_rate)}% of matches",
            reasoning=[
                f"{home.name} scores 3+ goals in {home_over25_rate}%",
                f"{away.name} scores 3+ goals in {away_over25_rate}%",
                "→ Bet Over 2.5"
            ]
        ))
    
    return predictions


def check_btts_conflict(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Check Conceded Over 0.5 vs Failed to Score conflict"""
    predictions = []
    
    home_concede_rate = home.conceded_over05_percent
    away_concede_rate = away.conceded_over05_percent
    home_fail_rate = home.failed_to_score_percent
    away_fail_rate = away.failed_to_score_percent
    
    avg_concede = (home_concede_rate + away_concede_rate) / 2
    avg_fail = (home_fail_rate + away_fail_rate) / 2
    
    if avg_concede > 75 and avg_fail < 25:
        predictions.append(MatchPrediction(
            market="BTTS Yes",
            confidence="High",
            anchor_metric=f"Concede rate {avg_concede:.0f}% > Fail rate {avg_fail:.0f}%",
            reasoning=[
                "Both teams likely to concede and score",
                "→ BTTS Yes"
            ]
        ))
    
    elif avg_fail > 75 and avg_concede < 50:
        predictions.append(MatchPrediction(
            market="BTTS No",
            confidence="High",
            anchor_metric=f"Fail rate {avg_fail:.0f}% > Concede rate {avg_concede:.0f}%",
            reasoning=[
                "One or both teams likely to fail to score",
                "→ BTTS No"
            ]
        ))
    
    return predictions


def check_btts_over25_correlation(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Check BTTS & Over 2.5 correlation"""
    predictions = []
    
    avg_btts = (home.btts_percent + away.btts_percent) / 2
    avg_over25 = (home.over25_percent + away.over25_percent) / 2
    
    # 100% linked: BTTS always means Over 2.5
    if avg_btts > 75 and avg_over25 > 75:
        predictions.append(MatchPrediction(
            market="Over 2.5",
            confidence="Medium",
            anchor_metric="BTTS and Over 2.5 strongly correlated (75%+ each)",
            reasoning=[
                f"BTTS rate: {avg_btts:.0f}%",
                f"Over 2.5 rate: {avg_over25:.0f}%",
                "→ Over 2.5 requires BTTS. Adjust stake accordingly."
            ]
        ))
    
    # 0% linked: Over 2.5 never happens with BTTS
    elif avg_btts > 50 and avg_over25 < 25:
        predictions.append(MatchPrediction(
            market="Under 2.5",
            confidence="Medium",
            anchor_metric="BTTS high but Over 2.5 low → Shutout likely",
            reasoning=[
                f"BTTS rate: {avg_btts:.0f}%",
                f"Over 2.5 rate: {avg_over25:.0f}%",
                "→ Over 2.5 in shutout never happens. Fade Over."
            ]
        ))
    
    return predictions


def get_all_predictions(home: TeamMetrics, away: TeamMetrics) -> List[MatchPrediction]:
    """Run all rules and return unique predictions"""
    all_predictions = []
    
    # Tier 1: 100% / 0% metrics (anchor bets)
    all_predictions.extend(scan_100_percent(home, away))
    all_predictions.extend(scan_0_percent(home, away))
    
    # Tier 2: 75%+ / 25%- thresholds
    if not all_predictions:
        all_predictions.extend(check_team_total_over25(home, away))
        all_predictions.extend(check_btts_conflict(home, away))
        all_predictions.extend(check_btts_over25_correlation(home, away))
    
    # Remove duplicates (same market)
    seen = set()
    unique_predictions = []
    for p in all_predictions:
        if p.market not in seen:
            seen.add(p.market)
            unique_predictions.append(p)
    
    return unique_predictions[:3]  # Max 3 recommendations


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def team_metrics_input(team_name: str, key_prefix: str) -> TeamMetrics:
    """Create input fields for team metrics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        over15_percent = st.number_input(
            "Over 1.5 %",
            min_value=0, max_value=100, value=50, step=5,
            key=f"{key_prefix}_over15",
            help="% of matches with 2+ goals"
        )
        over25_percent = st.number_input(
            "Over 2.5 %",
            min_value=0, max_value=100, value=35, step=5,
            key=f"{key_prefix}_over25",
            help="% of matches with 3+ goals"
        )
        over35_percent = st.number_input(
            "Over 3.5 %",
            min_value=0, max_value=100, value=20, step=5,
            key=f"{key_prefix}_over35",
            help="% of matches with 4+ goals"
        )
        btts_percent = st.number_input(
            "BTTS %",
            min_value=0, max_value=100, value=40, step=5,
            key=f"{key_prefix}_btts",
            help="% of matches where both teams scored"
        )
    
    with col2:
        scored_over15_percent = st.number_input(
            "Scored 2+ %",
            min_value=0, max_value=100, value=30, step=5,
            key=f"{key_prefix}_scored15",
            help="% of matches where team scored 2+ goals"
        )
        scored_over25_percent = st.number_input(
            "Scored 3+ %",
            min_value=0, max_value=100, value=15, step=5,
            key=f"{key_prefix}_scored25",
            help="% of matches where team scored 3+ goals"
        )
        conceded_over05_percent = st.number_input(
            "Conceded %",
            min_value=0, max_value=100, value=70, step=5,
            key=f"{key_prefix}_conceded",
            help="% of matches where team conceded"
        )
        failed_to_score_percent = st.number_input(
            "Failed to Score %",
            min_value=0, max_value=100, value=30, step=5,
            key=f"{key_prefix}_failed",
            help="% of matches where team failed to score"
        )
        clean_sheet_percent = st.number_input(
            "Clean Sheet %",
            min_value=0, max_value=100, value=30, step=5,
            key=f"{key_prefix}_clean",
            help="% of matches where team kept clean sheet"
        )
    
    return TeamMetrics(
        name=team_name,
        over15_percent=float(over15_percent),
        over25_percent=float(over25_percent),
        over35_percent=float(over35_percent),
        btts_percent=float(btts_percent),
        scored_over15_percent=float(scored_over15_percent),
        scored_over25_percent=float(scored_over25_percent),
        conceded_over05_percent=float(conceded_over05_percent),
        failed_to_score_percent=float(failed_to_score_percent),
        clean_sheet_percent=float(clean_sheet_percent)
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("Goal Markets Only | 100%/0% Anchor Bets | No Winner Picks")
    
    st.markdown("""
    <div class="league-note">
        📊 <strong>Core Principles:</strong><br>
        • <strong>NEVER predict a winner</strong> (no "Team A will win")<br>
        • Only predict goal markets: Over 2.5, Under 2.5, Over 3.5, Under 3.5, BTTS Yes, BTTS No<br>
        • Find 100% or 0% metrics first → Anchor bet<br>
        • Then use tiered thresholds (75%+ / 25%-)
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
    
    # Home Team Metrics
    st.subheader(f"🏠 {home_name} - Team Metrics")
    home_metrics = team_metrics_input(home_name, "home")
    
    st.divider()
    
    # Away Team Metrics
    st.subheader(f"✈️ {away_name} - Team Metrics")
    away_metrics = team_metrics_input(away_name, "away")
    
    st.divider()
    
    # ========================================================================
    # PREDICT BUTTON
    # ========================================================================
    if st.button("🔮 PREDICT", type="primary"):
        predictions = get_all_predictions(home_metrics, away_metrics)
        
        if not predictions:
            st.markdown("""
            <div class="prediction-card">
                <div class="prediction-market">⏸️ NO CLEAR BET</div>
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #94a3b8;">
                    No 100%/0% metrics found. No strong thresholds triggered.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for pred in predictions:
                confidence_class = "confidence-high" if pred.confidence == "High" else "confidence-medium" if pred.confidence == "Medium" else "confidence-low"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-anchor">🎯 {pred.anchor_metric}</div>
                    <div class="prediction-market">📊 {pred.market}</div>
                    <div class="{confidence_class}">Confidence: {pred.confidence}</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Reasoning", expanded=True):
                    for line in pred.reasoning:
                        st.write(f"• {line}")
        
        # Display data summary
        with st.expander("📈 Metrics Summary", expanded=False):
            data = {
                "Metric": [
                    "Over 1.5 %",
                    "Over 2.5 %",
                    "Over 3.5 %",
                    "BTTS %",
                    "Scored 2+ %",
                    "Scored 3+ %",
                    "Conceded %",
                    "Failed to Score %",
                    "Clean Sheet %"
                ],
                home_name: [
                    f"{home_metrics.over15_percent}%",
                    f"{home_metrics.over25_percent}%",
                    f"{home_metrics.over35_percent}%",
                    f"{home_metrics.btts_percent}%",
                    f"{home_metrics.scored_over15_percent}%",
                    f"{home_metrics.scored_over25_percent}%",
                    f"{home_metrics.conceded_over05_percent}%",
                    f"{home_metrics.failed_to_score_percent}%",
                    f"{home_metrics.clean_sheet_percent}%"
                ],
                away_name: [
                    f"{away_metrics.over15_percent}%",
                    f"{away_metrics.over25_percent}%",
                    f"{away_metrics.over35_percent}%",
                    f"{away_metrics.btts_percent}%",
                    f"{away_metrics.scored_over15_percent}%",
                    f"{away_metrics.scored_over25_percent}%",
                    f"{away_metrics.conceded_over05_percent}%",
                    f"{away_metrics.failed_to_score_percent}%",
                    f"{away_metrics.clean_sheet_percent}%"
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Rules Summary
    
    | Priority | Rule | Trigger | Bet |
    |----------|------|---------|-----|
    | **1** | Scan 100% metrics | Any metric = 100% | Anchor bet (Over/Under/BTTS) |
    | **2** | Scan 0% metrics | Any metric = 0% | Anchor bet (reverse) |
    | **3** | Team Total Over 2.5 | Both teams 0% → Under 3.5/2.5 | Goal market |
    | **4** | BTTS Conflict | Concede 100% > Fail 75% → BTTS Yes | BTTS market |
    | **5** | BTTS & Over 2.5 Correlation | 100% linked → Over 2.5 requires BTTS | Stake adjustment |
    
    ### 🎯 Key Principles
    
    - **NO WINNER PICKS** - Never predict which team will win
    - **Goal markets only** - Over/Under/BTTS
    - **Anchor bets first** - 100% or 0% metrics are most reliable
    - **Maximum 3 recommendations** per match
    
    ### 🎯 How to Use
    
    1. Enter **Home Team** and **Away Team** names
    2. Enter each team's **percentage metrics** (from your data source)
    3. Click **PREDICT**
    
    ### ✅ What This System Does
    
    - Finds statistical certainties (100%/0%)
    - Identifies conflicts and correlations
    - Outputs only goal-market bets
    - Never predicts a winner
    """)

if __name__ == "__main__":
    main()
