"""
Streak Predictor - Complete Decision Framework
REVISED VERSION with all 8 fixes from the systematic review.

FIXES IMPLEMENTED:
1. Team Total O0.5: Failed to Score ≤20% as primary trigger
2. Under 3.5: Allow one team 0% + other <20% conceded O2.5
3. BTTS: Add FTS ≤20% for both teams + concede ≥65%
4. O2.5 with 0% TG: Downgrade from Tier 2 to Tier 3
5. 2-2 probability: Change wording to "Both teams score 2+"
6. Under 3.5 + BTTS correlation: Added
7. Clean Sheet No redundancy: Filter when BTTS Yes present
8. Scoreline range: Dynamic based on recommendations
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - Complete Decision Framework",
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
        max-width: 1100px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #334155;
    }
    .tier-1 { color: #10b981; font-weight: 800; }
    .tier-2 { color: #fbbf24; font-weight: 700; }
    .tier-3 { color: #f97316; font-weight: 600; }
    .tier-4 { color: #94a3b8; font-weight: 500; }
    .team-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .team-name { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .section-header {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 700;
        text-align: center;
    }
    hr { margin: 1rem 0; }
    .stButton button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        border: none;
        width: 100%;
    }
    .risk-note {
        background: #7f1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamMetrics:
    name: str
    scored_0_5_pct: float = 0.0
    scored_1_5_pct: float = 0.0
    scored_2_5_pct: float = 0.0
    scored_3_5_pct: float = 0.0
    failed_to_score_pct: float = 0.0
    conceded_0_5_pct: float = 0.0
    conceded_1_5_pct: float = 0.0
    conceded_2_5_pct: float = 0.0
    conceded_3_5_pct: float = 0.0
    clean_sheet_pct: float = 0.0
    btts_pct: float = 0.0
    btts_and_over25_pct: float = 0.0
    btts_no_and_over25_pct: float = 0.0
    over_1_5_pct: float = 0.0
    over_2_5_pct: float = 0.0
    over_3_5_pct: float = 0.0


@dataclass
class MarketRecommendation:
    market: str
    bet: str
    confidence: str
    tier: int
    reasoning: List[str]


@dataclass
class AnalysisResult:
    anchor_signals: List[str]
    conflict_resolution: List[str]
    market_recommendations: List[MarketRecommendation]
    correlations: List[str]
    risk_notes: List[str]
    scoreline_range: List[str]


# ============================================================================
# PHASE 1: INITIAL SCAN
# ============================================================================
def initial_scan(home: TeamMetrics, away: TeamMetrics) -> List[str]:
    anchors = []
    
    if home.scored_0_5_pct == 100:
        anchors.append(f"✓ {home.name} Scored Over 0.5 = 100% (never blanks)")
    if home.scored_2_5_pct == 0:
        anchors.append(f"✓ {home.name} Scored Over 2.5 = 0% (never scores 3+)")
    if away.scored_2_5_pct == 0:
        anchors.append(f"✓ {away.name} Scored Over 2.5 = 0% (never scores 3+)")
    
    if home.conceded_0_5_pct == 100:
        anchors.append(f"✓ {home.name} Conceded Over 0.5 = 100% (never keeps clean sheet)")
    if home.conceded_2_5_pct == 0:
        anchors.append(f"✓ {home.name} Conceded Over 2.5 = 0% (never concedes 3+)")
    if away.conceded_2_5_pct == 0:
        anchors.append(f"✓ {away.name} Conceded Over 2.5 = 0% (never concedes 3+)")
    if home.conceded_3_5_pct == 0:
        anchors.append(f"✓ {home.name} Conceded Over 3.5 = 0% (defense has a ceiling)")
    if away.conceded_3_5_pct == 0:
        anchors.append(f"✓ {away.name} Conceded Over 3.5 = 0% (defense has a ceiling)")
    
    if home.btts_no_and_over25_pct == 0:
        anchors.append(f"✓ {home.name} BTTS No & Over 2.5 = 0% (Over 2.5 requires BTTS)")
    if away.btts_no_and_over25_pct == 0:
        anchors.append(f"✓ {away.name} BTTS No & Over 2.5 = 0% (Over 2.5 requires BTTS)")
    
    if home.failed_to_score_pct == 0:
        anchors.append(f"✓ {home.name} Failed to Score = 0% (always scores)")
    if away.failed_to_score_pct == 0:
        anchors.append(f"✓ {away.name} Failed to Score = 0% (always scores)")
    
    return anchors


# ============================================================================
# PHASE 2: CONFLICT RESOLUTION
# ============================================================================
def resolve_conflicts(home: TeamMetrics, away: TeamMetrics) -> List[str]:
    resolutions = []
    
    if home.conceded_0_5_pct >= 90 and away.failed_to_score_pct >= 70:
        resolutions.append(f"✓ {home.name} Concede {home.conceded_0_5_pct}% > {away.name} Fail {away.failed_to_score_pct}% → BTTS Yes")
    elif away.conceded_0_5_pct >= 90 and home.failed_to_score_pct >= 70:
        resolutions.append(f"✓ {away.name} Concede {away.conceded_0_5_pct}% > {home.name} Fail {home.failed_to_score_pct}% → BTTS Yes")
    elif home.failed_to_score_pct >= 70 and away.conceded_0_5_pct < 50:
        resolutions.append(f"✓ {home.name} Fail {home.failed_to_score_pct}% > {away.name} Concede {away.conceded_0_5_pct}% → BTTS No")
    elif away.failed_to_score_pct >= 70 and home.conceded_0_5_pct < 50:
        resolutions.append(f"✓ {away.name} Fail {away.failed_to_score_pct}% > {home.name} Concede {home.conceded_0_5_pct}% → BTTS No")
    else:
        resolutions.append("✓ No primary conflict detected")
    
    return resolutions


# ============================================================================
# PHASE 3: MARKET SELECTION
# ============================================================================
def evaluate_btts(home: TeamMetrics, away: TeamMetrics) -> Optional[MarketRecommendation]:
    reasoning = []
    
    # FIX #3: HIGH CONFIDENCE - Both Failed to Score ≤ 20% + concede ≥ 65%
    if home.failed_to_score_pct <= 20 and away.failed_to_score_pct <= 20:
        if home.conceded_0_5_pct >= 65 or away.conceded_0_5_pct >= 65:
            reasoning.append(f"✓ Both teams have low FTS ({home.failed_to_score_pct}% / {away.failed_to_score_pct}%)")
            reasoning.append(f"✓ {home.name if home.conceded_0_5_pct >= 65 else away.name} concedes {max(home.conceded_0_5_pct, away.conceded_0_5_pct)}% of games")
            return MarketRecommendation(
                market="BTTS", bet="Yes",
                confidence="High", tier=2, reasoning=reasoning
            )
        else:
            reasoning.append(f"✓ Both teams have low FTS ({home.failed_to_score_pct}% / {away.failed_to_score_pct}%)")
            return MarketRecommendation(
                market="BTTS", bet="Yes",
                confidence="Medium", tier=3, reasoning=reasoning
            )
    
    # HIGH CONFIDENCE: Concede 80%+ AND Scored 80%+
    if (home.conceded_0_5_pct >= 80 and away.scored_0_5_pct >= 80) or \
       (away.conceded_0_5_pct >= 80 and home.scored_0_5_pct >= 80):
        reasoning.append(f"✓ High concede and score rates")
        return MarketRecommendation(
            market="BTTS", bet="Yes",
            confidence="High", tier=2, reasoning=reasoning
        )
    
    # MEDIUM CONFIDENCE: Both concede 70%+
    if home.conceded_0_5_pct >= 70 and away.conceded_0_5_pct >= 70:
        reasoning.append(f"✓ Both teams concede regularly ({home.conceded_0_5_pct}% / {away.conceded_0_5_pct}%)")
        return MarketRecommendation(
            market="BTTS", bet="Yes",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    # BTTS No: Both fail to score 40%+
    if home.failed_to_score_pct >= 40 and away.failed_to_score_pct >= 40:
        reasoning.append(f"✓ Both teams blank regularly ({home.failed_to_score_pct}% / {away.failed_to_score_pct}%)")
        return MarketRecommendation(
            market="BTTS", bet="No",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    return None


def evaluate_over25(home: TeamMetrics, away: TeamMetrics, btts_rec: Optional[MarketRecommendation]) -> Optional[MarketRecommendation]:
    reasoning = []
    combined_avg = (home.over_2_5_pct + away.over_2_5_pct) / 2
    
    # FIX #4: Both teams 0% Over 2.5 TG → Under 2.5 (Tier 3, not Tier 2)
    if home.scored_2_5_pct == 0 and away.scored_2_5_pct == 0:
        reasoning.append(f"✓ Both teams have 0% Over 2.5 Team Goals")
        return MarketRecommendation(
            market="Over/Under 2.5", bet="Under 2.5",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    # Combined Over 2.5 ≥ 60% → Over 2.5
    if combined_avg >= 60:
        reasoning.append(f"✓ Combined Over 2.5 average = {combined_avg:.1f}% (≥ 60%)")
        return MarketRecommendation(
            market="Over/Under 2.5", bet="Over 2.5",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    # Combined Over 2.5 ≤ 40% → Under 2.5
    if combined_avg <= 40:
        reasoning.append(f"✓ Combined Over 2.5 average = {combined_avg:.1f}% (≤ 40%)")
        return MarketRecommendation(
            market="Over/Under 2.5", bet="Under 2.5",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    # Correlated play: BTTS Yes + BTTS No & Over 2.5 = 0% → Over 2.5
    if btts_rec and btts_rec.bet == "Yes":
        if home.btts_no_and_over25_pct == 0 and away.btts_no_and_over25_pct == 0:
            reasoning.append(f"✓ Both teams have 0% BTTS No & Over 2.5 → Over 2.5 requires BTTS")
            reasoning.append(f"✓ Since BTTS Yes is recommended, Over 2.5 is correlated")
            return MarketRecommendation(
                market="Over/Under 2.5", bet="Over 2.5",
                confidence="Medium", tier=3, reasoning=reasoning
            )
    
    return None


def evaluate_over35(home: TeamMetrics, away: TeamMetrics) -> Optional[MarketRecommendation]:
    reasoning = []
    
    # FIX #2: One team 0% + other <20% conceded O2.5 → Under 3.5 (Tier 2)
    if home.conceded_2_5_pct == 0 and away.conceded_2_5_pct < 20:
        reasoning.append(f"✓ {home.name} 0% Conceded Over 2.5, {away.name} {away.conceded_2_5_pct}%")
        return MarketRecommendation(
            market="Over/Under 3.5", bet="Under 3.5",
            confidence="High", tier=2, reasoning=reasoning
        )
    if away.conceded_2_5_pct == 0 and home.conceded_2_5_pct < 20:
        reasoning.append(f"✓ {away.name} 0% Conceded Over 2.5, {home.name} {home.conceded_2_5_pct}%")
        return MarketRecommendation(
            market="Over/Under 3.5", bet="Under 3.5",
            confidence="High", tier=2, reasoning=reasoning
        )
    
    # Both teams Over 2.5 TG < 25% → Under 3.5 (Tier 3)
    if home.scored_2_5_pct < 25 and away.scored_2_5_pct < 25:
        reasoning.append(f"✓ Both teams have low Over 2.5 TG ({home.scored_2_5_pct}% / {away.scored_2_5_pct}%)")
        
        # Check Over 1.5 TG for 2-2 risk (downgrade confidence)
        if home.scored_1_5_pct >= 35 and away.scored_1_5_pct >= 35:
            reasoning.append(f"⚠️ Both teams have {home.scored_1_5_pct}% / {away.scored_1_5_pct}% Over 1.5 TG → 2-2 risk")
            return MarketRecommendation(
                market="Over/Under 3.5", bet="Under 3.5",
                confidence="Medium", tier=3, reasoning=reasoning
            )
        else:
            return MarketRecommendation(
                market="Over/Under 3.5", bet="Under 3.5",
                confidence="Medium", tier=3, reasoning=reasoning
            )
    
    return None


def evaluate_team_total(team: TeamMetrics, opponent: TeamMetrics) -> Optional[MarketRecommendation]:
    reasoning = []
    
    # FIX #1: VERY HIGH - Scored Over 0.5 ≥ 90%
    if team.scored_0_5_pct >= 90:
        reasoning.append(f"✓ {team.name} scores in {team.scored_0_5_pct}% of games")
        return MarketRecommendation(
            market=f"{team.name} Team Total", bet="Over 0.5 Goals",
            confidence="Very High", tier=1, reasoning=reasoning
        )
    
    # HIGH - Failed to Score ≤ 20%
    if team.failed_to_score_pct <= 20:
        reasoning.append(f"✓ {team.name} fails to score only {team.failed_to_score_pct}% of games")
        return MarketRecommendation(
            market=f"{team.name} Team Total", bet="Over 0.5 Goals",
            confidence="High", tier=2, reasoning=reasoning
        )
    
    # MEDIUM - Scored ≥ 75% AND opponent conceded ≥ 60%
    if team.scored_0_5_pct >= 75 and opponent.conceded_0_5_pct >= 60:
        reasoning.append(f"✓ {team.name} scores {team.scored_0_5_pct}% of games")
        reasoning.append(f"✓ {opponent.name} concedes {opponent.conceded_0_5_pct}% of games")
        return MarketRecommendation(
            market=f"{team.name} Team Total", bet="Over 0.5 Goals",
            confidence="Medium", tier=3, reasoning=reasoning
        )
    
    return None


def evaluate_clean_sheet_no(team: TeamMetrics, opponent: TeamMetrics) -> Optional[MarketRecommendation]:
    reasoning = []
    
    if team.clean_sheet_pct == 0:
        reasoning.append(f"✓ {team.name} keeps clean sheet in 0% of games")
        return MarketRecommendation(
            market=f"{team.name} Clean Sheet", bet="No",
            confidence="Very High", tier=1, reasoning=reasoning
        )
    
    if team.conceded_0_5_pct >= 80:
        reasoning.append(f"✓ {team.name} concedes in {team.conceded_0_5_pct}% of games")
        return MarketRecommendation(
            market=f"{team.name} Clean Sheet", bet="No",
            confidence="High", tier=2, reasoning=reasoning
        )
    
    if opponent.scored_0_5_pct == 100:
        reasoning.append(f"✓ {opponent.name} scores in 100% of games")
        return MarketRecommendation(
            market=f"{team.name} Clean Sheet", bet="No",
            confidence="High", tier=2, reasoning=reasoning
        )
    
    return None


# ============================================================================
# PHASE 4: CORRELATION CHECK
# ============================================================================
def check_correlations(home: TeamMetrics, away: TeamMetrics, recommendations: List[MarketRecommendation]) -> List[str]:
    correlations = []
    
    if home.btts_no_and_over25_pct == 0 and away.btts_no_and_over25_pct == 0:
        correlations.append("Over 2.5 requires BTTS Yes (both teams 0% BTTS No & Over 2.5)")
    
    # FIX #6: Under 3.5 + BTTS Yes coexistence
    has_under35 = any(rec.market == "Over/Under 3.5" and rec.bet == "Under 3.5" for rec in recommendations)
    has_btts_yes = any(rec.market == "BTTS" and rec.bet == "Yes" for rec in recommendations)
    if has_under35 and has_btts_yes:
        correlations.append("Under 3.5 and BTTS Yes can coexist → 1-1, 2-1 scorelines favored")
    
    return correlations


# ============================================================================
# PHASE 5: FINAL OUTPUT
# ============================================================================
def filter_redundant_recommendations(recommendations: List[MarketRecommendation]) -> List[MarketRecommendation]:
    """FIX #7: Remove Clean Sheet No when BTTS Yes is present with same or higher tier"""
    btts_rec = next((r for r in recommendations if r.market == "BTTS" and r.bet == "Yes"), None)
    
    if btts_rec:
        # Filter out Clean Sheet No recommendations with same or higher tier
        filtered = []
        for r in recommendations:
            if "Clean Sheet" in r.market and r.tier >= btts_rec.tier:
                continue
            filtered.append(r)
        return filtered
    return recommendations


def generate_scoreline_range(recommendations: List[MarketRecommendation]) -> List[str]:
    """FIX #8: Dynamic scoreline range based on recommendations"""
    range_list = []
    
    btts_rec = next((r for r in recommendations if r.market == "BTTS"), None)
    over35_rec = next((r for r in recommendations if r.market == "Over/Under 3.5"), None)
    
    if btts_rec and btts_rec.bet == "Yes":
        range_list.extend(["1-1", "2-1", "1-2"])
        if over35_rec and over35_rec.bet == "Under 3.5":
            pass  # Already includes those
        else:
            range_list.append("2-2")
    elif btts_rec and btts_rec.bet == "No":
        range_list.extend(["1-0", "0-1", "2-0", "0-2", "0-0"])
    else:
        range_list.extend(["1-1", "2-1", "1-2"])
    
    return list(dict.fromkeys(range_list))  # Remove duplicates


def predict_match(home: TeamMetrics, away: TeamMetrics) -> AnalysisResult:
    anchors = initial_scan(home, away)
    conflicts = resolve_conflicts(home, away)
    
    recommendations = []
    
    btts_rec = evaluate_btts(home, away)
    if btts_rec:
        recommendations.append(btts_rec)
    
    over25_rec = evaluate_over25(home, away, btts_rec)
    if over25_rec:
        recommendations.append(over25_rec)
    
    over35_rec = evaluate_over35(home, away)
    if over35_rec:
        recommendations.append(over35_rec)
    
    home_tt_rec = evaluate_team_total(home, away)
    if home_tt_rec:
        recommendations.append(home_tt_rec)
    away_tt_rec = evaluate_team_total(away, home)
    if away_tt_rec:
        recommendations.append(away_tt_rec)
    
    home_cs_rec = evaluate_clean_sheet_no(home, away)
    if home_cs_rec:
        recommendations.append(home_cs_rec)
    away_cs_rec = evaluate_clean_sheet_no(away, home)
    if away_cs_rec:
        recommendations.append(away_cs_rec)
    
    # FIX #7: Filter redundant Clean Sheet recommendations
    recommendations = filter_redundant_recommendations(recommendations)
    
    correlations = check_correlations(home, away, recommendations)
    
    risk_notes = []
    
    # FIX #5: Correct 2-2 wording
    both_score_2_plus = (home.scored_1_5_pct * away.scored_1_5_pct) / 100
    if both_score_2_plus >= 10:
        risk_notes.append(f"Both teams score 2+ in {both_score_2_plus:.1f}% of games → 2-2, 3-2, 2-3 are live risks")
    
    if home.scored_0_5_pct >= 80 and away.scored_0_5_pct >= 80:
        risk_notes.append("Both teams score regularly → BTTS likely")
    
    # FIX #8: Dynamic scoreline range
    scoreline_range = generate_scoreline_range(recommendations)
    
    recommendations.sort(key=lambda x: x.tier)
    
    return AnalysisResult(
        anchor_signals=anchors,
        conflict_resolution=conflicts,
        market_recommendations=recommendations,
        correlations=correlations,
        risk_notes=risk_notes,
        scoreline_range=scoreline_range
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def metric_input(team_name: str, prefix: str) -> TeamMetrics:
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚽ Goal-Based Streaks**")
        scored_0_5_pct = st.number_input("Over 0.5 Team Goals %", min_value=0, max_value=100, value=80, step=5, key=f"{prefix}_scored_05")
        scored_1_5_pct = st.number_input("Over 1.5 Team Goals %", min_value=0, max_value=100, value=38, step=5, key=f"{prefix}_scored_15")
        scored_2_5_pct = st.number_input("Over 2.5 Team Goals %", min_value=0, max_value=100, value=15, step=5, key=f"{prefix}_scored_25")
        scored_3_5_pct = st.number_input("Over 3.5 Team Goals %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_scored_35")
        failed_to_score_pct = st.number_input("Failed to Score %", min_value=0, max_value=100, value=20, step=5, key=f"{prefix}_failed_score")
    
    with col2:
        st.markdown("**🛡️ Defensive Streaks**")
        conceded_0_5_pct = st.number_input("Conceded Over 0.5 %", min_value=0, max_value=100, value=77, step=5, key=f"{prefix}_conceded_05")
        conceded_1_5_pct = st.number_input("Conceded Over 1.5 %", min_value=0, max_value=100, value=40, step=5, key=f"{prefix}_conceded_15")
        conceded_2_5_pct = st.number_input("Conceded Over 2.5 %", min_value=0, max_value=100, value=10, step=5, key=f"{prefix}_conceded_25")
        conceded_3_5_pct = st.number_input("Conceded Over 3.5 %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_conceded_35")
        clean_sheet_pct = st.number_input("Clean Sheet %", min_value=0, max_value=100, value=23, step=5, key=f"{prefix}_clean_sheet")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("**📊 Match-Based Streaks**")
        btts_pct = st.number_input("BTTS %", min_value=0, max_value=100, value=54, step=5, key=f"{prefix}_btts")
        btts_and_over25_pct = st.number_input("BTTS & Over 2.5 %", min_value=0, max_value=100, value=46, step=5, key=f"{prefix}_btts_over25")
        btts_no_and_over25_pct = st.number_input("BTTS No & Over 2.5 %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_btts_no_over25")
        over_1_5_pct = st.number_input("Over 1.5 %", min_value=0, max_value=100, value=77, step=5, key=f"{prefix}_over15")
        over_2_5_pct = st.number_input("Over 2.5 %", min_value=0, max_value=100, value=46, step=5, key=f"{prefix}_over25")
        over_3_5_pct = st.number_input("Over 3.5 %", min_value=0, max_value=100, value=27, step=5, key=f"{prefix}_over35")
    
    with col4:
        st.markdown("**📈 Averages (Context Only)**")
        st.caption("Not used in calculations - for reference only")
        st.number_input("Avg Goals Scored", min_value=0.0, max_value=5.0, value=1.54, step=0.1, key=f"{prefix}_avg_scored", disabled=True)
        st.number_input("Avg Goals Conceded", min_value=0.0, max_value=5.0, value=1.23, step=0.1, key=f"{prefix}_avg_conceded", disabled=True)
        st.number_input("Avg Match Goals", min_value=0.0, max_value=6.0, value=2.77, step=0.1, key=f"{prefix}_avg_match", disabled=True)
    
    return TeamMetrics(
        name=team_name,
        scored_0_5_pct=float(scored_0_5_pct),
        scored_1_5_pct=float(scored_1_5_pct),
        scored_2_5_pct=float(scored_2_5_pct),
        scored_3_5_pct=float(scored_3_5_pct),
        failed_to_score_pct=float(failed_to_score_pct),
        conceded_0_5_pct=float(conceded_0_5_pct),
        conceded_1_5_pct=float(conceded_1_5_pct),
        conceded_2_5_pct=float(conceded_2_5_pct),
        conceded_3_5_pct=float(conceded_3_5_pct),
        clean_sheet_pct=float(clean_sheet_pct),
        btts_pct=float(btts_pct),
        btts_and_over25_pct=float(btts_and_over25_pct),
        btts_no_and_over25_pct=float(btts_no_and_over25_pct),
        over_1_5_pct=float(over_1_5_pct),
        over_2_5_pct=float(over_2_5_pct),
        over_3_5_pct=float(over_3_5_pct)
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("Complete Decision Framework | Phase 1-5 Execution | ALL FIXES APPLIED")
    
    st.markdown("""
    <div class="section-header">
    📋 EXECUTION PROCESS
    </div>
    <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Phase 1</strong><br>Initial Scan<br>Circle 100%/0%
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Phase 2</strong><br>Conflict Resolution<br>Conceded > Scored > Totals > BTTS
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Phase 3</strong><br>Market Selection<br>5 Markets
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Phase 4</strong><br>Correlation Check<br>Linked Markets
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Phase 5</strong><br>Final Output<br>Ranked Plays
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team", "Home Team", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team", "Away Team", key="away_name")
    
    st.divider()
    
    st.subheader(f"🏠 {home_name} - Team Metrics")
    home_metrics = metric_input(home_name, "home")
    
    st.divider()
    
    st.subheader(f"✈️ {away_name} - Team Metrics")
    away_metrics = metric_input(away_name, "away")
    
    st.divider()
    
    if st.button("🔮 EXECUTE FRAMEWORK", type="primary"):
        result = predict_match(home_metrics, away_metrics)
        
        st.markdown("---")
        st.markdown("## 📊 ANALYSIS RESULTS")
        
        with st.expander("🔍 PHASE 1: Anchor Signals (100%/0% Metrics)", expanded=True):
            if result.anchor_signals:
                for signal in result.anchor_signals:
                    st.success(signal)
            else:
                st.info("No 100% or 0% metrics detected")
        
        with st.expander("⚖️ PHASE 2: Conflict Resolution", expanded=True):
            for resolution in result.conflict_resolution:
                st.info(resolution)
        
        st.markdown("### 🎯 MARKET RECOMMENDATIONS")
        
        for rec in result.market_recommendations:
            tier_color = {
                1: "🔴 TIER 1 (Lock)",
                2: "🟡 TIER 2 (Strong)",
                3: "🟢 TIER 3 (Value)",
                4: "⚪ TIER 4 (Lean)",
                5: "🔵 TIER 5 (Pass)"
            }.get(rec.tier, f"TIER {rec.tier}")
            
            confidence_color = {
                "Very High": "🟢",
                "High": "🟡",
                "Medium": "🟠",
                "Low": "🔴"
            }.get(rec.confidence, "⚪")
            
            st.markdown(f"""
            <div style="background: #1e293b; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{rec.market}</strong> → <span style="font-size: 1.2rem; font-weight: bold;">{rec.bet}</span>
                    </div>
                    <div>
                        <span style="background: #0f172a; padding: 0.2rem 0.6rem; border-radius: 12px;">{tier_color}</span>
                        <span style="margin-left: 0.5rem;">{confidence_color} {rec.confidence} Confidence</span>
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #94a3b8;">
                    {' • '.join(rec.reasoning)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if result.correlations:
            with st.expander("🔗 PHASE 4: Correlated Markets", expanded=True):
                for corr in result.correlations:
                    st.info(corr)
        
        st.markdown("### ⚠️ RISK NOTES & SCORELINE")
        
        if result.risk_notes:
            for note in result.risk_notes:
                st.markdown(f'<div class="risk-note">⚠️ {note}</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="background: #0f172a; border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
            <strong>🎯 Most Likely Scorelines:</strong><br>
            {' | '.join(result.scoreline_range)}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 FINAL VERDICT")
        
        primary_bets = [rec for rec in result.market_recommendations if rec.tier <= 2]
        if primary_bets:
            st.markdown("**Primary Plays (Tier 1-2):**")
            for rec in primary_bets:
                st.markdown(f"- **{rec.market}**: {rec.bet} ({rec.confidence} confidence)")
        else:
            st.markdown("**No Tier 1-2 plays identified. Consider passing this match.**")
        
        if any(rec.tier == 1 for rec in result.market_recommendations):
            st.success("🔴 TIER 1 LOCK DETECTED - High confidence play available")
    
    st.divider()
    st.markdown("""
    ### 📋 Decision Framework Summary
    
    | Phase | Action | Output |
    |-------|--------|--------|
    | 1 | Scan for 100%/0% metrics | Anchor signals |
    | 2 | Apply hierarchy (Conceded > Scored > Totals > BTTS) | Conflict resolution |
    | 3 | Evaluate 5 markets (BTTS, O/U 2.5, O/U 3.5, Team Total, Clean Sheet) | Market recommendations |
    | 4 | Check correlations | Linked markets |
    | 5 | Rank plays by confidence | Final verdict with scoreline range |
    
    ### 🎯 Confidence Tiers
    
    | Tier | Label | Criteria |
    |------|-------|----------|
    | 1 | Lock | Multiple 100%/0% metrics aligned |
    | 2 | Strong | One 100%/0% metric + supporting data |
    | 3 | Value | 70%+ trend, no conflict |
    | 4 | Lean | 55-60% edge, small sample |
    | 5 | Pass | Coin flip or conflicting signals |
    
    ### ✅ FIXES APPLIED IN THIS VERSION
    
    | # | Issue | Fix |
    |---|-------|-----|
    | 1 | Team Total thresholds too strict | Failed to Score ≤20% as primary trigger |
    | 2 | Under 3.5 requires both 0% | Allow one team 0% + other <20% |
    | 3 | BTTS missing FTS trigger | Add FTS ≤20% for both + concede ≥65% |
    | 4 | O2.5 with 0% TG overconfident | Downgrade to Tier 3 |
    | 5 | 2-2 probability wording | Changed to "Both teams score 2+" |
    | 6 | Missing Under 3.5 + BTTS correlation | Added to correlation check |
    | 7 | Clean Sheet No redundancy | Filter when BTTS Yes present |
    | 8 | Static scoreline range | Made dynamic based on recommendations |
    """)

if __name__ == "__main__":
    main()
