"""
Over 1.5 Goals Predictor - Kill Switch System
Based on statistical analysis of home/away scoring frequency, SOT, and clean sheet rates.

Kill Switch Logic (MANDATORY):
- If either team has Scoring Frequency < 80% → IMMEDIATE AVOID

Then Filter Count (only if Kill Switch passes):
- Filter 1: Combined SOT ≥ 8.5
- Filter 3: Both teams clean sheet rate ≤ 20%

Confidence Levels:
- Kill Switch fails → 🔴 AVOID / Under 1.5
- Kill Switch passes + 2/2 filters → 🟢 HIGH Over 1.5
- Kill Switch passes + 1/2 filters → 🟡 MODERATE Over 1.5
- Kill Switch passes + 0/2 filters → 🔴 AVOID / Under 1.5

xG Tiebreaker (for MODERATE only):
- xG ≥ 2.4 → Upgrade to HIGH
- xG < 2.0 → Downgrade to AVOID
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Over 1.5 Goals Predictor - Kill Switch System",
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
        max-width: 950px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        border: 1px solid #334155;
    }
    .prediction-high {
        font-size: 2.5rem;
        font-weight: 800;
        color: #10b981;
    }
    .prediction-moderate {
        font-size: 2rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .prediction-avoid {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ef4444;
    }
    .confidence-high {
        font-size: 1rem;
        font-weight: 600;
        color: #10b981;
        margin-top: 0.5rem;
    }
    .confidence-moderate {
        font-size: 1rem;
        font-weight: 600;
        color: #fbbf24;
        margin-top: 0.5rem;
    }
    .confidence-avoid {
        font-size: 1rem;
        font-weight: 600;
        color: #ef4444;
        margin-top: 0.5rem;
    }
    .kill-switch-active {
        background: #7f1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #fca5a5;
        text-align: left;
        font-weight: bold;
    }
    .kill-switch-pass {
        background: #064e3b;
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #a7f3d0;
        text-align: left;
        font-weight: bold;
    }
    .filter-pass {
        background: #064e3b;
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #a7f3d0;
        text-align: left;
    }
    .filter-fail {
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
    .input-note {
        background: #1e293b;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.8rem;
        color: #fbbf24;
        text-align: center;
    }
    .results-table {
        background: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
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
    .warning-text {
        color: #fbbf24;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamStats:
    """Team statistical data - using home/away splits only"""
    name: str
    
    # For Home Team (use HOME stats only)
    home_total_shots: float = 0.0      # Total shots per game at home
    home_on_target_pct: float = 0.0    # % of shots on target at home
    home_scoring_freq: float = 0.0     # % of home matches they scored in
    home_cs_rate: float = 0.0          # % of home matches they kept a clean sheet
    home_xg: float = 0.0               # Expected goals per game at home (optional)
    
    # For Away Team (use AWAY stats only)
    away_total_shots: float = 0.0      # Total shots per game away
    away_on_target_pct: float = 0.0    # % of shots on target away
    away_scoring_freq: float = 0.0     # % of away matches they scored in
    away_cs_rate: float = 0.0          # % of away matches they kept a clean sheet
    away_xg: float = 0.0               # Expected goals per game away (optional)
    
    # Flags
    is_home: bool = True


@dataclass
class FilterResults:
    """Results of all filters"""
    kill_switch_pass: bool              # Both teams scoring frequency ≥ 80%
    kill_switch_failed_team: Optional[str] = None
    
    filter1_pass: bool = False          # Combined SOT ≥ 8.5
    filter3_pass: bool = False          # Both teams clean sheet rate ≤ 20%
    
    combined_sot: float = 0.0
    home_sot: float = 0.0
    away_sot: float = 0.0
    
    home_scoring_freq: float = 0.0
    away_scoring_freq: float = 0.0
    
    home_cs_rate: float = 0.0
    away_cs_rate: float = 0.0
    
    combined_xg: Optional[float] = None


@dataclass
class PredictionResult:
    main_bet: str      # "Over 1.5" or "AVOID / Under 1.5"
    confidence: str    # "High", "Moderate", "Avoid"
    filters_passed: int
    filter_results: FilterResults
    reasoning: List[str]
    decision_path: List[str]


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_sot(total_shots: float, on_target_pct: float) -> float:
    """Calculate shots on target from total shots and on-target percentage"""
    return round(total_shots * (on_target_pct / 100), 2)


def calculate_combined_sot(home: TeamStats, away: TeamStats) -> Tuple[float, float, float]:
    """Calculate home SOT, away SOT, and combined SOT"""
    home_sot = calculate_sot(home.home_total_shots, home.home_on_target_pct)
    away_sot = calculate_sot(away.away_total_shots, away.away_on_target_pct)
    combined_sot = round(home_sot + away_sot, 2)
    return home_sot, away_sot, combined_sot


def evaluate_filters(home: TeamStats, away: TeamStats) -> FilterResults:
    """Evaluate Kill Switch and all filters"""
    
    # Get scoring frequencies
    home_scoring_freq = home.home_scoring_freq
    away_scoring_freq = away.away_scoring_freq
    
    # KILL SWITCH: Both teams scoring frequency ≥ 80%
    kill_switch_pass = home_scoring_freq >= 80.0 and away_scoring_freq >= 80.0
    kill_switch_failed_team = None
    if not kill_switch_pass:
        if home_scoring_freq < 80.0:
            kill_switch_failed_team = home.name
        else:
            kill_switch_failed_team = away.name
    
    # Calculate SOT
    home_sot, away_sot, combined_sot = calculate_combined_sot(home, away)
    
    # Filter 1: Combined SOT ≥ 8.5 (only matters if kill switch passes)
    filter1_pass = combined_sot >= 8.5
    
    # Get clean sheet rates
    home_cs_rate = home.home_cs_rate
    away_cs_rate = away.away_cs_rate
    
    # Filter 3: Both teams clean sheet rate ≤ 20% (only matters if kill switch passes)
    filter3_pass = home_cs_rate <= 20.0 and away_cs_rate <= 20.0
    
    # Combined xG (if available)
    combined_xg = None
    if home.home_xg > 0 and away.away_xg > 0:
        combined_xg = round(home.home_xg + away.away_xg, 2)
    
    return FilterResults(
        kill_switch_pass=kill_switch_pass,
        kill_switch_failed_team=kill_switch_failed_team,
        filter1_pass=filter1_pass,
        filter3_pass=filter3_pass,
        combined_sot=combined_sot,
        home_sot=home_sot,
        away_sot=away_sot,
        home_scoring_freq=home_scoring_freq,
        away_scoring_freq=away_scoring_freq,
        home_cs_rate=home_cs_rate,
        away_cs_rate=away_cs_rate,
        combined_xg=combined_xg
    )


def make_prediction(home: TeamStats, away: TeamStats) -> PredictionResult:
    """
    Make Over 1.5 Goals prediction based on Kill Switch logic.
    """
    reasoning = []
    decision_path = []
    
    # Evaluate all filters
    filters = evaluate_filters(home, away)
    
    # Display input summary
    reasoning.append(f"📊 **INPUT SUMMARY:**")
    reasoning.append(f"   • {home.name} (HOME):")
    reasoning.append(f"     - Scoring Frequency: {filters.home_scoring_freq}% (Threshold: ≥80%)")
    reasoning.append(f"     - Clean Sheet Rate: {filters.home_cs_rate}% (Threshold: ≤20%)")
    reasoning.append(f"     - Total Shots: {home.home_total_shots} | ON Target: {home.home_on_target_pct}% → SOT: {filters.home_sot}")
    reasoning.append(f"")
    reasoning.append(f"   • {away.name} (AWAY):")
    reasoning.append(f"     - Scoring Frequency: {filters.away_scoring_freq}% (Threshold: ≥80%)")
    reasoning.append(f"     - Clean Sheet Rate: {filters.away_cs_rate}% (Threshold: ≤20%)")
    reasoning.append(f"     - Total Shots: {away.away_total_shots} | ON Target: {away.away_on_target_pct}% → SOT: {filters.away_sot}")
    reasoning.append(f"")
    reasoning.append(f"   • Combined SOT: {filters.combined_sot} (Threshold: ≥8.5)")
    
    if filters.combined_xg:
        reasoning.append(f"   • Combined xG: {filters.combined_xg} (Tiebreaker: ≥2.4 upgrades, <2.0 downgrades)")
    
    # ========================================================================
    # STEP 1: KILL SWITCH
    # ========================================================================
    decision_path.append("STEP 1: Kill Switch Check")
    
    if not filters.kill_switch_pass:
        reasoning.append(f"\n🔴 **KILL SWITCH ACTIVATED**")
        reasoning.append(f"   → {filters.kill_switch_failed_team} has Scoring Frequency {filters.home_scoring_freq if filters.kill_switch_failed_team == home.name else filters.away_scoring_freq}% (< 80%)")
        reasoning.append(f"   → This team fails to score in too many matches")
        reasoning.append(f"   → IMMEDIATE AVOID / UNDER 1.5")
        
        return PredictionResult(
            main_bet="AVOID / Under 1.5",
            confidence="Avoid",
            filters_passed=0,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # Kill switch passed
    reasoning.append(f"\n✅ **KILL SWITCH PASSED**")
    reasoning.append(f"   → Both teams have Scoring Frequency ≥ 80%")
    reasoning.append(f"   → {home.name}: {filters.home_scoring_freq}% | {away.name}: {filters.away_scoring_freq}%")
    
    # ========================================================================
    # STEP 2: COUNT FILTERS 1 & 3
    # ========================================================================
    decision_path.append("STEP 2: Count Filters 1 & 3")
    
    filters_passed = 0
    if filters.filter1_pass:
        filters_passed += 1
        reasoning.append(f"\n✅ **FILTER 1 PASS**: Combined SOT {filters.combined_sot} ≥ 8.5")
    else:
        reasoning.append(f"\n❌ **FILTER 1 FAIL**: Combined SOT {filters.combined_sot} < 8.5")
    
    if filters.filter3_pass:
        filters_passed += 1
        reasoning.append(f"✅ **FILTER 3 PASS**: Both CS Rates ≤ 20% ({filters.home_cs_rate}% / {filters.away_cs_rate}%)")
    else:
        reasoning.append(f"❌ **FILTER 3 FAIL**: CS Rates {filters.home_cs_rate}% / {filters.away_cs_rate}% (Need both ≤20%)")
    
    reasoning.append(f"\n📊 **Filters Passed (out of 2): {filters_passed}/2**")
    
    # ========================================================================
    # STEP 3: BASE VERDICT FROM FILTER COUNT
    # ========================================================================
    decision_path.append(f"STEP 3: Base Verdict from {filters_passed}/2 filters")
    
    if filters_passed == 2:
        main_bet = "Over 1.5"
        confidence = "High"
        reasoning.append(f"\n🟢 **BASE VERDICT:** {filters_passed}/2 filters passed → OVER 1.5 (HIGH CONFIDENCE)")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filters_passed=filters_passed,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    elif filters_passed == 1:
        main_bet = "Over 1.5"
        confidence = "Moderate"
        reasoning.append(f"\n🟡 **BASE VERDICT:** {filters_passed}/2 filters passed → OVER 1.5 (MODERATE CONFIDENCE)")
        
        # ====================================================================
        # STEP 4: xG TIEBREAKER FOR MODERATE
        # ====================================================================
        if filters.combined_xg:
            decision_path.append(f"STEP 4: xG Tiebreaker (combined xG = {filters.combined_xg})")
            
            if filters.combined_xg >= 2.4:
                main_bet = "Over 1.5"
                confidence = "High"
                reasoning.append(f"\n⬆️ **xG TIEBREAKER:** Combined xG {filters.combined_xg} ≥ 2.4 → UPGRADE to HIGH CONFIDENCE")
            elif filters.combined_xg < 2.0:
                main_bet = "AVOID / Under 1.5"
                confidence = "Avoid"
                reasoning.append(f"\n⬇️ **xG TIEBREAKER:** Combined xG {filters.combined_xg} < 2.0 → DOWNGRADE to AVOID")
            else:
                reasoning.append(f"\n➡️ **xG TIEBREAKER:** Combined xG {filters.combined_xg} in 2.0-2.4 range → No change (MODERATE remains)")
        else:
            reasoning.append(f"\n➡️ **No xG data available** → No tiebreaker applied")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filters_passed=filters_passed,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    else:  # 0 filters passed
        main_bet = "AVOID / Under 1.5"
        confidence = "Avoid"
        reasoning.append(f"\n🔴 **VERDICT:** {filters_passed}/2 filters passed → AVOID / UNDER 1.5")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filters_passed=filters_passed,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def home_team_stats_input(team_name: str) -> TeamStats:
    """Create input fields for HOME team (using HOME stats only)"""
    st.markdown(f"<div class='team-header'><span class='team-name'>🏠 {team_name} (HOME Team)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-note'>📌 Enter HOME team's HOME stats only (last 5-10 matches)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        home_scoring_freq = st.number_input(
            "✅ Scoring Frequency % (HOME) - Scored in % of matches",
            min_value=0, max_value=100, value=80, step=5,
            key=f"{team_name}_home_scoring_freq"
        )
        home_total_shots = st.number_input(
            "🎯 Total Shots per game (HOME)",
            min_value=0.0, max_value=25.0, value=12.0, step=0.5,
            key=f"{team_name}_home_total_shots"
        )
        home_cs_rate = st.number_input(
            "🧤 Clean Sheet Rate % (HOME) - Conceded 0 in % of matches",
            min_value=0, max_value=100, value=15, step=5,
            key=f"{team_name}_home_cs_rate"
        )
    with col2:
        home_on_target_pct = st.number_input(
            "🎯 Shots ON Target % (HOME)",
            min_value=0, max_value=100, value=30, step=5,
            key=f"{team_name}_home_on_target_pct"
        )
        home_xg = st.number_input(
            "📊 xG per game (HOME) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_home_xg"
        )
    
    return TeamStats(
        name=team_name,
        home_total_shots=home_total_shots,
        home_on_target_pct=float(home_on_target_pct),
        home_scoring_freq=float(home_scoring_freq),
        home_cs_rate=float(home_cs_rate),
        home_xg=home_xg,
        is_home=True
    )


def away_team_stats_input(team_name: str) -> TeamStats:
    """Create input fields for AWAY team (using AWAY stats only)"""
    st.markdown(f"<div class='team-header'><span class='team-name'>✈️ {team_name} (AWAY Team)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-note'>📌 Enter AWAY team's AWAY stats only (last 5-10 matches)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        away_scoring_freq = st.number_input(
            "✅ Scoring Frequency % (AWAY) - Scored in % of matches",
            min_value=0, max_value=100, value=75, step=5,
            key=f"{team_name}_away_scoring_freq"
        )
        away_total_shots = st.number_input(
            "🎯 Total Shots per game (AWAY)",
            min_value=0.0, max_value=25.0, value=10.0, step=0.5,
            key=f"{team_name}_away_total_shots"
        )
        away_cs_rate = st.number_input(
            "🧤 Clean Sheet Rate % (AWAY) - Conceded 0 in % of matches",
            min_value=0, max_value=100, value=20, step=5,
            key=f"{team_name}_away_cs_rate"
        )
    with col2:
        away_on_target_pct = st.number_input(
            "🎯 Shots ON Target % (AWAY)",
            min_value=0, max_value=100, value=30, step=5,
            key=f"{team_name}_away_on_target_pct"
        )
        away_xg = st.number_input(
            "📊 xG per game (AWAY) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_away_xg"
        )
    
    return TeamStats(
        name=team_name,
        away_total_shots=away_total_shots,
        away_on_target_pct=float(away_on_target_pct),
        away_scoring_freq=float(away_scoring_freq),
        away_cs_rate=float(away_cs_rate),
        away_xg=away_xg,
        is_home=False
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Over 1.5 Goals Predictor")
    st.caption("Kill Switch System: Scoring Frequency ≥ 80% is MANDATORY")
    
    st.markdown("""
    <div class="step-box">
    <strong>🔴 KILL SWITCH RULE (MANDATORY):</strong><br>
    If either team has <strong>Scoring Frequency &lt; 80%</strong> → IMMEDIATE AVOID / UNDER 1.5<br><br>
    <strong>✅ If Kill Switch passes, then check:</strong><br>
    • <strong>Filter 1:</strong> Combined SOT ≥ 8.5<br>
    • <strong>Filter 3:</strong> Both teams Clean Sheet Rate ≤ 20%<br><br>
    <strong>Confidence Levels:</strong><br>
    • 2/2 filters pass → 🟢 HIGH CONFIDENCE Over 1.5<br>
    • 1/2 filters pass → 🟡 MODERATE CONFIDENCE Over 1.5<br>
    • 0/2 filters pass → 🔴 AVOID / Under 1.5<br><br>
    <strong>xG Tiebreaker (MODERATE only):</strong><br>
    • xG ≥ 2.4 → Upgrade to HIGH<br>
    • xG &lt; 2.0 → Downgrade to AVOID
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # TEAM INPUTS
    # ========================================================================
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team Name", "Home Team", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team Name", "Away Team", key="away_name")
    
    st.divider()
    
    # Home Team Stats
    st.subheader(f"🏠 {home_name} - HOME STATS ONLY")
    home_stats = home_team_stats_input(home_name)
    
    st.divider()
    
    # Away Team Stats
    st.subheader(f"✈️ {away_name} - AWAY STATS ONLY")
    away_stats = away_team_stats_input(away_name)
    
    st.divider()
    
    # ========================================================================
    # PREDICT BUTTON
    # ========================================================================
    if st.button("🔮 PREDICT Over 1.5 Goals", type="primary"):
        # Make prediction
        result = make_prediction(home_stats, away_stats)
        
        # Display prediction card
        if result.main_bet == "Over 1.5" and result.confidence == "High":
            pred_class = "prediction-high"
            pred_icon = "🔥"
        elif result.main_bet == "Over 1.5" and result.confidence == "Moderate":
            pred_class = "prediction-moderate"
            pred_icon = "🟡"
        else:
            pred_class = "prediction-avoid"
            pred_icon = "🔴"
        
        confidence_class = f"confidence-{result.confidence.lower()}"
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div class="{confidence_class}">Confidence: {result.confidence}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display filter details
        with st.expander("📋 Filter Results", expanded=True):
            f = result.filter_results
            
            if f.kill_switch_pass:
                st.markdown(f'<div class="kill-switch-pass">✅ KILL SWITCH PASSED: Both teams Scoring Frequency ≥ 80% ({f.home_scoring_freq}% / {f.away_scoring_freq}%)</div>', unsafe_allow_html=True)
                
                # Show filter results only if kill switch passed
                if f.filter1_pass:
                    st.markdown(f'<div class="filter-pass">✅ FILTER 1 PASS: Combined SOT = {f.combined_sot} (Home: {f.home_sot} + Away: {f.away_sot}) ≥ 8.5</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="filter-fail">❌ FILTER 1 FAIL: Combined SOT = {f.combined_sot} (Home: {f.home_sot} + Away: {f.away_sot}) < 8.5</div>', unsafe_allow_html=True)
                
                if f.filter3_pass:
                    st.markdown(f'<div class="filter-pass">✅ FILTER 3 PASS: Both CS Rates ≤ 20% ({f.home_cs_rate}% / {f.away_cs_rate}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="filter-fail">❌ FILTER 3 FAIL: CS Rates {f.home_cs_rate}% / {f.away_cs_rate}% (Need both ≤20%)</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="step-box">📊 Filters Passed (out of 2): {result.filters_passed}/2</div>', unsafe_allow_html=True)
                
                if f.combined_xg:
                    st.markdown(f'<div class="step-box">📊 Combined xG (Tiebreaker): {f.combined_xg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="kill-switch-active">🔴 KILL SWITCH ACTIVATED: {f.kill_switch_failed_team} has Scoring Frequency {f.home_scoring_freq if f.kill_switch_failed_team == home_stats.name else f.away_scoring_freq}% (< 80%)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="step-box">⚠️ Match automatically flagged as AVOID / Under 1.5</div>', unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📊 Detailed Analysis", expanded=False):
            for line in result.reasoning:
                if "🔴" in line or "❌" in line:
                    st.error(line)
                elif "🟢" in line or "✅" in line:
                    st.success(line)
                elif "🟡" in line or "⬆️" in line or "⬇️" in line:
                    st.warning(line)
                else:
                    st.write(line)
        
        # Display data summary table
        with st.expander("📈 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_stats.name} (HOME) - Scoring Frequency",
                    f"{away_stats.name} (AWAY) - Scoring Frequency",
                    f"{home_stats.name} (HOME) - Total Shots",
                    f"{home_stats.name} (HOME) - ON Target %",
                    f"{home_stats.name} (HOME) - SOT (calculated)",
                    f"{away_stats.name} (AWAY) - Total Shots",
                    f"{away_stats.name} (AWAY) - ON Target %",
                    f"{away_stats.name} (AWAY) - SOT (calculated)",
                    "Combined SOT",
                    f"{home_stats.name} (HOME) - Clean Sheet Rate",
                    f"{away_stats.name} (AWAY) - Clean Sheet Rate",
                ],
                "Value": [
                    f"{result.filter_results.home_scoring_freq}%",
                    f"{result.filter_results.away_scoring_freq}%",
                    f"{home_stats.home_total_shots}",
                    f"{home_stats.home_on_target_pct}%",
                    f"{result.filter_results.home_sot}",
                    f"{away_stats.away_total_shots}",
                    f"{away_stats.away_on_target_pct}%",
                    f"{result.filter_results.away_sot}",
                    f"{result.filter_results.combined_sot}",
                    f"{result.filter_results.home_cs_rate}%",
                    f"{result.filter_results.away_cs_rate}%",
                ],
                "Threshold": [
                    "≥ 80%",
                    "≥ 80%",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "≥ 8.5",
                    "≤ 20%",
                    "≤ 20%",
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Complete Decision Flowchart
    
