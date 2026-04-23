"""
Streak Predictor - Core Logic Framework
Based on hierarchical statistical analysis of team metrics.

TIER SYSTEM:
- Tier 1: 100% or 0% rates (Heavy weight - structural truths)
- Tier 2: 75%+ or 25%- rates (Strong - clear tendency)
- Tier 3: 50-70% or 30-50% (Directional - slight lean)
- Tier 4: Averages (Context only - masks extremes)

CONFLICT RESOLUTION: Conceded > Scored > Match Totals > BTTS

TEAM TOTAL FIRST PRINCIPLE: Check Team Total Over/Under before Match Total.

CLEAN SHEET / FAILED TO SCORE MATRIX: Treat as inverse probabilities.

MARKET EXPECTATION GAP: Compare averages with individual extremes.

BTTS & OVER 2.5 LINKAGE: Check correlation for both teams.
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - Core Logic Framework",
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
        max-width: 1000px;
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
        font-size: 2rem;
        font-weight: 800;
        color: #10b981;
    }
    .prediction-under {
        font-size: 2rem;
        font-weight: 800;
        color: #ef4444;
    }
    .prediction-btts-yes {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10b981;
    }
    .prediction-btts-no {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ef4444;
    }
    .prediction-nobet {
        font-size: 2rem;
        font-weight: 700;
        color: #f59e0b;
    }
    .tier-badge {
        display: inline-block;
        background: #1e293b;
        border-radius: 12px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    .team-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .team-name {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .section-header {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 700;
        text-align: center;
    }
    hr {
        margin: 1rem 0;
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
    .metric-highlight {
        font-size: 1.2rem;
        font-weight: 700;
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamMetrics:
    """Complete team metrics for analysis"""
    name: str
    # Goal-based streaks
    scored_0_5_pct: float = 0.0      # Over 0.5 Team Goals %
    scored_1_5_pct: float = 0.0      # Over 1.5 Team Goals %
    scored_2_5_pct: float = 0.0      # Over 2.5 Team Goals %
    failed_to_score_pct: float = 0.0  # Failed to Score %
    # Defensive streaks
    conceded_0_5_pct: float = 0.0     # Conceded Over 0.5 %
    conceded_1_5_pct: float = 0.0     # Conceded Over 1.5 %
    conceded_2_5_pct: float = 0.0     # Conceded Over 2.5 %
    clean_sheet_pct: float = 0.0      # Clean Sheet %
    # Match-based streaks
    btts_pct: float = 0.0             # Both Teams to Score %
    over_1_5_pct: float = 0.0         # Match Over 1.5 %
    over_2_5_pct: float = 0.0         # Match Over 2.5 %
    # Averages (Tier 4 - Context Only)
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    avg_match_goals: float = 0.0


@dataclass
class PredictionResult:
    main_bet: str
    confidence: str
    reasoning: List[str]
    tier_signals: dict


# ============================================================================
# TIER CLASSIFICATION
# ============================================================================
def get_tier(percentage: float) -> Tuple[int, str]:
    """Classify a metric into its tier based on percentage"""
    if percentage >= 100 or percentage <= 0:
        return 1, "TIER 1"
    elif percentage >= 75 or percentage <= 25:
        return 2, "TIER 2"
    elif percentage >= 50 or percentage <= 50:
        return 3, "TIER 3"
    else:
        return 4, "TIER 4"


# ============================================================================
# CORE ANALYSIS FUNCTIONS
# ============================================================================
def analyze_team_totals(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 1: Check Team Total Over 2.5 for BOTH teams.
    - Both 0% → Bet Under 3.5 / Under 2.5
    - One or both high → Bet Over 2.5
    """
    reasoning = []
    
    home_high = home.scored_2_5_pct >= 50
    away_high = away.scored_2_5_pct >= 50
    both_zero = home.scored_2_5_pct == 0 and away.scored_2_5_pct == 0
    
    if both_zero:
        reasoning.append(f"✓ Both teams have 0% Over 2.5 Team Goals → Under 2.5")
        return "Under 2.5", reasoning
    elif home_high or away_high:
        reasoning.append(f"✓ Team Total Over 2.5: Home {home.scored_2_5_pct}% | Away {away.scored_2_5_pct}% → Lean Over")
        return "Over 2.5", reasoning
    
    return None, reasoning


def analyze_clean_sheet_matrix(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 2: Clean Sheet / Failed to Score Matrix
    Treat as inverse probabilities.
    """
    reasoning = []
    
    # Home Failed to Score vs Away Clean Sheet
    home_fail = home.failed_to_score_pct
    away_cs = away.clean_sheet_pct
    
    if home_fail >= 75 and away_cs >= 25:
        reasoning.append(f"✓ Home Fail to Score {home_fail}% + Away Clean Sheet {away_cs}% → BTTS No")
        return "BTTS No", reasoning
    elif home_fail >= 54 and away_cs >= 31:
        reasoning.append(f"✓ Home Fail to Score {home_fail}% + Away Clean Sheet {away_cs}% → BTTS No lean")
        return "BTTS No", reasoning
    
    # Away Failed to Score vs Home Clean Sheet
    away_fail = away.failed_to_score_pct
    home_cs = home.clean_sheet_pct
    
    if away_fail >= 75 and home_cs >= 25:
        reasoning.append(f"✓ Away Fail to Score {away_fail}% + Home Clean Sheet {home_cs}% → BTTS No")
        return "BTTS No", reasoning
    elif away_fail >= 54 and home_cs >= 31:
        reasoning.append(f"✓ Away Fail to Score {away_fail}% + Home Clean Sheet {home_cs}% → BTTS No lean")
        return "BTTS No", reasoning
    
    return None, reasoning


def analyze_conflict_resolution(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 3: Conflict Resolution Rule
    Order of precedence: Conceded > Scored > Match Totals > BTTS
    """
    reasoning = []
    
    # Check Conceded vs Failed to Score conflict
    home_concede = home.conceded_0_5_pct
    away_fail = away.failed_to_score_pct
    home_fail = home.failed_to_score_pct
    away_concede = away.conceded_0_5_pct
    
    # Home Concede 100% vs Away Fail 75%+ → BTTS Yes
    if home_concede >= 90 and away_fail >= 75:
        reasoning.append(f"✓ Home Concede {home_concede}% > Away Fail {away_fail}% → BTTS Yes")
        return "BTTS Yes", reasoning
    
    # Away Concede 100% vs Home Fail 75%+ → BTTS Yes
    if away_concede >= 90 and home_fail >= 75:
        reasoning.append(f"✓ Away Concede {away_concede}% > Home Fail {home_fail}% → BTTS Yes")
        return "BTTS Yes", reasoning
    
    # Home Fail 75%+ vs Away Concede <50% → BTTS No
    if home_fail >= 75 and away_concede < 50:
        reasoning.append(f"✓ Home Fail {home_fail}% > Away Concede {away_concede}% → BTTS No")
        return "BTTS No", reasoning
    
    # Away Fail 75%+ vs Home Concede <50% → BTTS No
    if away_fail >= 75 and home_concede < 50:
        reasoning.append(f"✓ Away Fail {away_fail}% > Home Concede {home_concede}% → BTTS No")
        return "BTTS No", reasoning
    
    return None, reasoning


def analyze_btts_over25_linkage(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 4: BTTS & Over 2.5 Linkage Check
    If BTTS & Over 2.5 = 100% for a team, Over 2.5 requires BTTS.
    If BTTS No & Over 2.5 = 0%, Over 2.5 in shutout never happens.
    """
    reasoning = []
    
    # Check if both teams have perfect correlation
    home_correlation = home.btts_pct == 100 and home.over_2_5_pct == 100
    away_correlation = away.btts_pct == 100 and away.over_2_5_pct == 100
    
    if home_correlation or away_correlation:
        reasoning.append(f"✓ BTTS & Over 2.5 are 100% correlated for one or both teams")
        reasoning.append(f"  → Over 2.5 requires BTTS Yes")
        return "Over 2.5 requires BTTS", reasoning
    
    # Check if either team never covers Over 2.5 in a shutout
    home_no_btts_over = away.btts_pct == 0 and away.over_2_5_pct == 0
    away_no_btts_over = home.btts_pct == 0 and home.over_2_5_pct == 0
    
    if home_no_btts_over or away_no_btts_over:
        reasoning.append(f"✓ One team has 0% BTTS and 0% Over 2.5 → Over 2.5 requires BTTS")
        return "Over 2.5 requires BTTS", reasoning
    
    return None, reasoning


def analyze_market_gap(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 5: Market Expectation Gap
    Compare averages with individual extremes.
    """
    reasoning = []
    
    # Check Team Total Over 2.5 extremes vs Match Over 2.5 average
    if home.scored_2_5_pct == 0 and away.scored_2_5_pct == 0:
        match_avg = home.over_2_5_pct
        reasoning.append(f"✓ Both teams 0% Over 2.5 Team Goals vs Match Avg {match_avg}% → Gap detected")
        return "Under 2.5", reasoning
    
    # Check high individual extremes
    if home.scored_2_5_pct >= 80 or away.scored_2_5_pct >= 80:
        reasoning.append(f"✓ High Team Total Over 2.5: Home {home.scored_2_5_pct}% | Away {away.scored_2_5_pct}%")
        return "Over 2.5", reasoning
    
    return None, reasoning


def analyze_tier1_signals(home: TeamMetrics, away: TeamMetrics) -> Tuple[Optional[str], List[str]]:
    """
    STEP 0: Scan for 100% or 0% metrics (Tier 1)
    These are structural truths and anchor bets.
    """
    reasoning = []
    
    # Check for 0% Over 2.5 Team Goals on both teams
    if home.scored_2_5_pct == 0 and away.scored_2_5_pct == 0:
        reasoning.append(f"✓ TIER 1: Both teams have 0% Over 2.5 Team Goals")
        return "Under 2.5 / Under 3.5", reasoning
    
    # Check for 100% BTTS on one team
    if home.btts_pct == 100:
        reasoning.append(f"✓ TIER 1: Home BTTS = 100% → They always concede")
        return "BTTS Yes (Home side)", reasoning
    if away.btts_pct == 100:
        reasoning.append(f"✓ TIER 1: Away BTTS = 100% → They always concede")
        return "BTTS Yes (Away side)", reasoning
    
    # Check for 0% BTTS on one team
    if home.btts_pct == 0:
        reasoning.append(f"✓ TIER 1: Home BTTS = 0% → They never concede")
        return "BTTS No", reasoning
    if away.btts_pct == 0:
        reasoning.append(f"✓ TIER 1: Away BTTS = 0% → They never concede")
        return "BTTS No", reasoning
    
    # Check for 100% Conceded Over 0.5
    if home.conceded_0_5_pct == 100:
        reasoning.append(f"✓ TIER 1: Home concedes in 100% of matches")
    if away.conceded_0_5_pct == 100:
        reasoning.append(f"✓ TIER 1: Away concedes in 100% of matches")
    
    return None, reasoning


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================
def predict_match(home: TeamMetrics, away: TeamMetrics) -> PredictionResult:
    """
    Main prediction function implementing the Core Logic Framework.
    """
    reasoning = []
    signals = {}
    
    reasoning.append("📊 **CORE LOGIC FRAMEWORK ANALYSIS**")
    reasoning.append("")
    reasoning.append("**TIER CLASSIFICATION:**")
    reasoning.append(f"  Home Over 2.5 TG: {home.scored_2_5_pct}% → {get_tier(home.scored_2_5_pct)[1]}")
    reasoning.append(f"  Away Over 2.5 TG: {away.scored_2_5_pct}% → {get_tier(away.scored_2_5_pct)[1]}")
    reasoning.append(f"  Home BTTS: {home.btts_pct}% → {get_tier(home.btts_pct)[1]}")
    reasoning.append(f"  Away BTTS: {away.btts_pct}% → {get_tier(away.btts_pct)[1]}")
    reasoning.append(f"  Home Concede 0.5: {home.conceded_0_5_pct}% → {get_tier(home.conceded_0_5_pct)[1]}")
    reasoning.append(f"  Away Concede 0.5: {away.conceded_0_5_pct}% → {get_tier(away.conceded_0_5_pct)[1]}")
    
    # ========================================================================
    # STEP 0: Tier 1 Signals (100% or 0%)
    # ========================================================================
    tier1_result, tier1_reasoning = analyze_tier1_signals(home, away)
    if tier1_result:
        reasoning.extend(tier1_reasoning)
        return PredictionResult(
            main_bet=tier1_result,
            confidence="High (Tier 1)",
            reasoning=reasoning,
            tier_signals=signals
        )
    
    # ========================================================================
    # STEP 1: Team Total First Principle
    # ========================================================================
    reasoning.append("\n**STEP 1: Team Total First Principle**")
    team_total_result, team_total_reasoning = analyze_team_totals(home, away)
    if team_total_result:
        reasoning.extend(team_total_reasoning)
        signals["team_total"] = team_total_result
    
    # ========================================================================
    # STEP 2: Clean Sheet / Failed to Score Matrix
    # ========================================================================
    reasoning.append("\n**STEP 2: Clean Sheet / Failed to Score Matrix**")
    cs_result, cs_reasoning = analyze_clean_sheet_matrix(home, away)
    if cs_result:
        reasoning.extend(cs_reasoning)
        signals["clean_sheet"] = cs_result
    
    # ========================================================================
    # STEP 3: Conflict Resolution
    # ========================================================================
    reasoning.append("\n**STEP 3: Conflict Resolution (Conceded > Scored > Totals > BTTS)**")
    conflict_result, conflict_reasoning = analyze_conflict_resolution(home, away)
    if conflict_result:
        reasoning.extend(conflict_reasoning)
        signals["conflict"] = conflict_result
    
    # ========================================================================
    # STEP 4: BTTS & Over 2.5 Linkage
    # ========================================================================
    reasoning.append("\n**STEP 4: BTTS & Over 2.5 Linkage Check**")
    linkage_result, linkage_reasoning = analyze_btts_over25_linkage(home, away)
    if linkage_result:
        reasoning.extend(linkage_reasoning)
        signals["linkage"] = linkage_result
    
    # ========================================================================
    # STEP 5: Market Expectation Gap
    # ========================================================================
    reasoning.append("\n**STEP 5: Market Expectation Gap**")
    gap_result, gap_reasoning = analyze_market_gap(home, away)
    if gap_result:
        reasoning.extend(gap_reasoning)
        signals["gap"] = gap_result
    
    # ========================================================================
    # FINAL DECISION: Synthesize signals
    # ========================================================================
    reasoning.append("\n**FINAL SYNTHESIS:**")
    
    # Count signal directions
    over_signals = 0
    under_signals = 0
    btts_yes = 0
    btts_no = 0
    
    for key, value in signals.items():
        if "Over" in str(value):
            over_signals += 1
        if "Under" in str(value):
            under_signals += 1
        if "BTTS Yes" in str(value):
            btts_yes += 1
        if "BTTS No" in str(value):
            btts_no += 1
    
    # Determine primary bet
    if over_signals > under_signals:
        main_bet = "Over 2.5"
        confidence = "Medium"
        reasoning.append(f"✓ {over_signals} signal(s) point to Over, {under_signals} to Under")
    elif under_signals > over_signals:
        main_bet = "Under 2.5"
        confidence = "Medium"
        reasoning.append(f"✓ {under_signals} signal(s) point to Under, {over_signals} to Over")
    else:
        # Use BTTS as tiebreaker
        if btts_yes > btts_no:
            main_bet = "BTTS Yes"
            confidence = "Medium"
            reasoning.append(f"✓ Tie in Over/Under signals, BTTS Yes leads ({btts_yes} vs {btts_no})")
        elif btts_no > btts_yes:
            main_bet = "BTTS No"
            confidence = "Medium"
            reasoning.append(f"✓ Tie in Over/Under signals, BTTS No leads ({btts_no} vs {btts_yes})")
        else:
            main_bet = "No Bet"
            confidence = "Low"
            reasoning.append(f"✓ Conflicting signals → No Bet")
    
    reasoning.append(f"\n**FINAL RECOMMENDATION:** {main_bet}")
    
    return PredictionResult(
        main_bet=main_bet,
        confidence=confidence,
        reasoning=reasoning,
        tier_signals=signals
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def metric_input(team_name: str, prefix: str) -> TeamMetrics:
    """Create input fields for team metrics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚽ Goal-Based Streaks**")
        scored_0_5_pct = st.number_input("Over 0.5 Team Goals %", min_value=0, max_value=100, value=80, step=5, key=f"{prefix}_scored_05")
        scored_1_5_pct = st.number_input("Over 1.5 Team Goals %", min_value=0, max_value=100, value=40, step=5, key=f"{prefix}_scored_15")
        scored_2_5_pct = st.number_input("Over 2.5 Team Goals %", min_value=0, max_value=100, value=10, step=5, key=f"{prefix}_scored_25")
        failed_to_score_pct = st.number_input("Failed to Score %", min_value=0, max_value=100, value=30, step=5, key=f"{prefix}_failed_score")
    
    with col2:
        st.markdown("**🛡️ Defensive Streaks**")
        conceded_0_5_pct = st.number_input("Conceded Over 0.5 %", min_value=0, max_value=100, value=70, step=5, key=f"{prefix}_conceded_05")
        conceded_1_5_pct = st.number_input("Conceded Over 1.5 %", min_value=0, max_value=100, value=40, step=5, key=f"{prefix}_conceded_15")
        conceded_2_5_pct = st.number_input("Conceded Over 2.5 %", min_value=0, max_value=100, value=15, step=5, key=f"{prefix}_conceded_25")
        clean_sheet_pct = st.number_input("Clean Sheet %", min_value=0, max_value=100, value=30, step=5, key=f"{prefix}_clean_sheet")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**📊 Match-Based Streaks**")
        btts_pct = st.number_input("BTTS %", min_value=0, max_value=100, value=45, step=5, key=f"{prefix}_btts")
        over_1_5_pct = st.number_input("Over 1.5 %", min_value=0, max_value=100, value=60, step=5, key=f"{prefix}_over15")
        over_2_5_pct = st.number_input("Over 2.5 %", min_value=0, max_value=100, value=35, step=5, key=f"{prefix}_over25")
    
    with col4:
        st.markdown("**📈 Averages (Context Only)**")
        avg_scored = st.number_input("Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.2, step=0.1, key=f"{prefix}_avg_scored")
        avg_conceded = st.number_input("Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.2, step=0.1, key=f"{prefix}_avg_conceded")
        avg_match = st.number_input("Avg Match Goals", min_value=0.0, max_value=6.0, value=2.4, step=0.1, key=f"{prefix}_avg_match")
    
    return TeamMetrics(
        name=team_name,
        scored_0_5_pct=float(scored_0_5_pct),
        scored_1_5_pct=float(scored_1_5_pct),
        scored_2_5_pct=float(scored_2_5_pct),
        failed_to_score_pct=float(failed_to_score_pct),
        conceded_0_5_pct=float(conceded_0_5_pct),
        conceded_1_5_pct=float(conceded_1_5_pct),
        conceded_2_5_pct=float(conceded_2_5_pct),
        clean_sheet_pct=float(clean_sheet_pct),
        btts_pct=float(btts_pct),
        over_1_5_pct=float(over_1_5_pct),
        over_2_5_pct=float(over_2_5_pct),
        avg_goals_scored=avg_scored,
        avg_goals_conceded=avg_conceded,
        avg_match_goals=avg_match
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("Core Logic Framework | Hierarchical Statistical Analysis")
    
    st.markdown("""
    <div class="section-header">
    🎯 TIER SYSTEM
    </div>
    <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
        <div style="flex: 1; background: #1e293b; padding: 0.5rem; border-radius: 8px;">
            <strong>Tier 1 (Heavy)</strong><br>
            100% or 0% rates<br>
            Structural truths
        </div>
        <div style="flex: 1; background: #1e293b; padding: 0.5rem; border-radius: 8px;">
            <strong>Tier 2 (Strong)</strong><br>
            75%+ or 25%- rates<br>
            Clear tendency
        </div>
        <div style="flex: 1; background: #1e293b; padding: 0.5rem; border-radius: 8px;">
            <strong>Tier 3 (Directional)</strong><br>
            50-70% or 30-50%<br>
            Slight lean
        </div>
        <div style="flex: 1; background: #1e293b; padding: 0.5rem; border-radius: 8px;">
            <strong>Tier 4 (Context)</strong><br>
            Averages only<br>
            Masks extremes
        </div>
    </div>
    
    <div class="section-header">
    ⚖️ CONFLICT RESOLUTION ORDER
    </div>
    <div style="text-align: center; margin-bottom: 1rem;">
        <code>Conceded > Scored > Match Totals > BTTS</code>
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
    home_metrics = metric_input(home_name, "home")
    
    st.divider()
    
    # Away Team Metrics
    st.subheader(f"✈️ {away_name} - Team Metrics")
    away_metrics = metric_input(away_name, "away")
    
    st.divider()
    
    # ========================================================================
    # PREDICT BUTTON
    # ========================================================================
    if st.button("🔮 PREDICT", type="primary"):
        result = predict_match(home_metrics, away_metrics)
        
        # Display prediction card
        if "Over" in result.main_bet:
            pred_class = "prediction-over"
            pred_icon = "🔥"
        elif "Under" in result.main_bet:
            pred_class = "prediction-under"
            pred_icon = "❄️"
        elif "BTTS Yes" in result.main_bet:
            pred_class = "prediction-btts-yes"
            pred_icon = "✅"
        elif "BTTS No" in result.main_bet:
            pred_class = "prediction-btts-no"
            pred_icon = "🚫"
        else:
            pred_class = "prediction-nobet"
            pred_icon = "⏸️"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                Confidence: {result.confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📋 Detailed Analysis", expanded=True):
            for line in result.reasoning:
                if "✓" in line:
                    st.success(line)
                elif "**" in line:
                    st.markdown(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📊 Data Summary", expanded=False):
            data = {
                "Metric": [
                    "Over 0.5 TG",
                    "Over 1.5 TG",
                    "Over 2.5 TG",
                    "Failed to Score",
                    "Conceded 0.5",
                    "Conceded 1.5",
                    "Conceded 2.5",
                    "Clean Sheet",
                    "BTTS",
                    "Over 1.5",
                    "Over 2.5",
                    "Avg Scored",
                    "Avg Conceded",
                    "Avg Match"
                ],
                home_name: [
                    f"{home_metrics.scored_0_5_pct}%",
                    f"{home_metrics.scored_1_5_pct}%",
                    f"{home_metrics.scored_2_5_pct}%",
                    f"{home_metrics.failed_to_score_pct}%",
                    f"{home_metrics.conceded_0_5_pct}%",
                    f"{home_metrics.conceded_1_5_pct}%",
                    f"{home_metrics.conceded_2_5_pct}%",
                    f"{home_metrics.clean_sheet_pct}%",
                    f"{home_metrics.btts_pct}%",
                    f"{home_metrics.over_1_5_pct}%",
                    f"{home_metrics.over_2_5_pct}%",
                    f"{home_metrics.avg_goals_scored}",
                    f"{home_metrics.avg_goals_conceded}",
                    f"{home_metrics.avg_match_goals}"
                ],
                away_name: [
                    f"{away_metrics.scored_0_5_pct}%",
                    f"{away_metrics.scored_1_5_pct}%",
                    f"{away_metrics.scored_2_5_pct}%",
                    f"{away_metrics.failed_to_score_pct}%",
                    f"{away_metrics.conceded_0_5_pct}%",
                    f"{away_metrics.conceded_1_5_pct}%",
                    f"{away_metrics.conceded_2_5_pct}%",
                    f"{away_metrics.clean_sheet_pct}%",
                    f"{away_metrics.btts_pct}%",
                    f"{away_metrics.over_1_5_pct}%",
                    f"{away_metrics.over_2_5_pct}%",
                    f"{away_metrics.avg_goals_scored}",
                    f"{away_metrics.avg_goals_conceded}",
                    f"{away_metrics.avg_match_goals}"
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Core Logic Framework Summary
    
    | Step | Action | Output |
    |------|--------|--------|
    | 1 | Scan for 100% or 0% metrics | Anchor bet (Tier 1) |
    | 2 | Check Team Total Over 2.5 for BOTH teams | Under/Over 2.5 |
    | 3 | Clean Sheet / Failed to Score Matrix | BTTS Yes/No |
    | 4 | Conflict Resolution (Conceded > Scored > Totals > BTTS) | Tie-breaker |
    | 5 | BTTS & Over 2.5 Linkage Check | Correlation insight |
    | 6 | Market Expectation Gap | Under/Over confirmation |
    
    ### 🎯 How to Use
    
    1. Enter **Home Team** and **Away Team** names
    2. Enter all percentage metrics for each team
    3. Click **PREDICT**
    
    ### ✅ Key Principles
    
    - **Tier 1 (100%/0%)** → Heavy weight, structural truths
    - **Tier 2 (75%+/25%-)** → Strong, clear tendency
    - **Tier 3 (50-70%/30-50%)** → Directional, slight lean
    - **Tier 4 (Averages)** → Context only, masks extremes
    
    **Conflict Resolution Order:** Conceded > Scored > Match Totals > BTTS
    """)

if __name__ == "__main__":
    main()
