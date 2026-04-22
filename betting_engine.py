"""
Over 1.5 Goals Predictor - Kill Switch System
Based on 3-Filter logic with mandatory Scoring Frequency kill switch.

FINAL LOGIC:

STEP 1: THE KILL SWITCH (MANDATORY)
- If either team has Scoring Frequency < 80% → IMMEDIATE AVOID / Under 1.5

STEP 2: FILTER COUNT (Only if Kill Switch passes)
- Filter 1: Combined SOT ≥ 8.5
- Filter 3: Both teams clean sheet rate ≤ 20%

STEP 3: BASE VERDICT
- 2 of 2 filters pass → HIGH CONFIDENCE Over 1.5
- 1 of 2 filters pass → MODERATE CONFIDENCE Over 1.5
- 0 of 2 filters pass → AVOID / Under 1.5

STEP 4: xG TIEBREAKER (For MODERATE only)
- Combined xG ≥ 2.4 → Upgrade to HIGH
- Combined xG < 2.0 → Downgrade to AVOID
- Combined xG 2.0-2.4 → Leave as MODERATE
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
        font-size: 2.5rem;
        font-weight: 800;
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
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        color: #fca5a5;
        text-align: center;
    }
    .kill-switch-inactive {
        background: #064e3b;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        color: #a7f3d0;
        text-align: center;
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
    .results-note {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.75rem;
        color: #94a3b8;
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
    """Team statistical data - using home/away splits only"""
    name: str
    
    # For Home Team (use HOME stats only)
    home_total_shots: float = 0.0      # Total shots per game at home
    home_on_target_percent: float = 0.0  # % of shots on target at home
    home_scoring_freq: float = 0.0     # % of home matches they scored in
    home_cs_rate: float = 0.0          # % of home matches they kept a clean sheet
    home_xg: float = 0.0               # Expected goals per game at home (optional)
    
    # For Away Team (use AWAY stats only)
    away_total_shots: float = 0.0      # Total shots per game away
    away_on_target_percent: float = 0.0  # % of shots on target away
    away_scoring_freq: float = 0.0     # % of away matches they scored in
    away_cs_rate: float = 0.0          # % of away matches they kept a clean sheet
    away_xg: float = 0.0               # Expected goals per game away (optional)
    
    # Flags
    is_home: bool = True


@dataclass
class FilterResults:
    """Results of all filters"""
    kill_switch_passed: bool           # Both scoring frequencies ≥ 80%
    kill_switch_reason: str            # Which team failed if any
    
    filter1_pass: bool                 # Combined SOT ≥ 8.5
    filter3_pass: bool                 # Both clean sheet rates ≤ 20%
    
    combined_sot: float
    home_sot: float
    away_sot: float
    
    home_scoring_freq: float
    away_scoring_freq: float
    
    home_cs_rate: float
    away_cs_rate: float
    
    combined_xg: Optional[float] = None
    
    @property
    def filters_passed_count(self) -> int:
        """Count how many of Filter 1 and Filter 3 passed (only if kill switch passed)"""
        return sum([self.filter1_pass, self.filter3_pass])


@dataclass
class PredictionResult:
    main_bet: str      # "Over 1.5" or "AVOID / Under 1.5"
    confidence: str    # "High", "Moderate", or "Avoid"
    filter_results: FilterResults
    reasoning: List[str]
    decision_path: List[str]


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_sot(total_shots: float, on_target_percent: float) -> float:
    """Calculate Shots on Target per game from total shots and on-target %"""
    return round(total_shots * (on_target_percent / 100), 2)


def calculate_combined_sot(home: TeamStats, away: TeamStats) -> Tuple[float, float, float]:
    """Calculate home SOT, away SOT, and combined SOT"""
    home_sot = calculate_sot(home.home_total_shots, home.home_on_target_percent)
    away_sot = calculate_sot(away.away_total_shots, away.away_on_target_percent)
    combined = home_sot + away_sot
    return home_sot, away_sot, combined


def calculate_combined_xg(home: TeamStats, away: TeamStats) -> Optional[float]:
    """Calculate combined xG if available"""
    if home.home_xg > 0 and away.away_xg > 0:
        return round(home.home_xg + away.away_xg, 2)
    return None


def evaluate_filters(home: TeamStats, away: TeamStats) -> FilterResults:
    """Evaluate all filters including the Kill Switch"""
    
    reasoning = []
    
    # Calculate SOT
    home_sot, away_sot, combined_sot = calculate_combined_sot(home, away)
    
    # Calculate combined xG if available
    combined_xg = calculate_combined_xg(home, away)
    
    # KILL SWITCH: Check scoring frequencies
    home_scoring_freq = home.home_scoring_freq
    away_scoring_freq = away.away_scoring_freq
    
    kill_switch_passed = home_scoring_freq >= 80.0 and away_scoring_freq >= 80.0
    
    kill_switch_reason = ""
    if not kill_switch_passed:
        if home_scoring_freq < 80.0 and away_scoring_freq < 80.0:
            kill_switch_reason = f"Both teams failed: {home.name} {home_scoring_freq}% / {away.name} {away_scoring_freq}%"
        elif home_scoring_freq < 80.0:
            kill_switch_reason = f"{home.name} failed: {home_scoring_freq}% (<80%)"
        else:
            kill_switch_reason = f"{away.name} failed: {away_scoring_freq}% (<80%)"
    
    # Filter 1: Combined SOT ≥ 8.5
    filter1_pass = combined_sot >= 8.5
    
    # Filter 3: Both clean sheet rates ≤ 20%
    filter3_pass = home.home_cs_rate <= 20.0 and away.away_cs_rate <= 20.0
    
    return FilterResults(
        kill_switch_passed=kill_switch_passed,
        kill_switch_reason=kill_switch_reason,
        filter1_pass=filter1_pass,
        filter3_pass=filter3_pass,
        combined_sot=combined_sot,
        home_sot=home_sot,
        away_sot=away_sot,
        home_scoring_freq=home_scoring_freq,
        away_scoring_freq=away_scoring_freq,
        home_cs_rate=home.home_cs_rate,
        away_cs_rate=away.away_cs_rate,
        combined_xg=combined_xg
    )


def make_prediction(home: TeamStats, away: TeamStats) -> PredictionResult:
    """
    Make Over 1.5 Goals prediction based on final refined logic.
    """
    reasoning = []
    decision_path = []
    
    # Evaluate all filters
    filters = evaluate_filters(home, away)
    
    # ========================================================================
    # STEP 1: KILL SWITCH
    # ========================================================================
    decision_path.append("STEP 1: Kill Switch (Scoring Frequency ≥ 80% for both teams)")
    
    if not filters.kill_switch_passed:
        reasoning.append(f"🔴 **KILL SWITCH ACTIVATED**")
        reasoning.append(f"   {filters.kill_switch_reason}")
        reasoning.append(f"   → Immediate AVOID / Under 1.5")
        
        return PredictionResult(
            main_bet="AVOID / Under 1.5",
            confidence="Avoid",
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    # Kill switch passed
    reasoning.append(f"🟢 **KILL SWITCH PASSED**")
    reasoning.append(f"   {home.name} Scoring Frequency: {filters.home_scoring_freq}% (≥80%)")
    reasoning.append(f"   {away.name} Scoring Frequency: {filters.away_scoring_freq}% (≥80%)")
    
    # ========================================================================
    # STEP 2: FILTER COUNT (Filter 1 and Filter 3 only)
    # ========================================================================
    decision_path.append("STEP 2: Count remaining filters (SOT ≥ 8.5, Both CS ≤ 20%)")
    
    reasoning.append(f"\n📊 **REMAINING FILTERS:**")
    
    # Filter 1
    if filters.filter1_pass:
        reasoning.append(f"   ✅ Filter 1 PASS: Combined SOT = {filters.combined_sot} ≥ 8.5")
    else:
        reasoning.append(f"   ❌ Filter 1 FAIL: Combined SOT = {filters.combined_sot} < 8.5")
    
    # Filter 3
    if filters.filter3_pass:
        reasoning.append(f"   ✅ Filter 3 PASS: Both CS rates ≤ 20% ({filters.home_cs_rate}% / {filters.away_cs_rate}%)")
    else:
        reasoning.append(f"   ❌ Filter 3 FAIL: {home.name} {filters.home_cs_rate}% / {away.name} {filters.away_cs_rate}% (Need both ≤20%)")
    
    filters_passed = filters.filters_passed_count
    reasoning.append(f"\n📊 **Filters Passed: {filters_passed}/2**")
    
    # ========================================================================
    # STEP 3: BASE VERDICT
    # ========================================================================
    if filters_passed == 2:
        decision_path.append("STEP 3: 2/2 filters passed → HIGH CONFIDENCE")
        main_bet = "Over 1.5"
        confidence = "High"
        reasoning.append(f"\n✅ **BASE VERDICT:** {filters_passed}/2 filters passed → **OVER 1.5** (HIGH CONFIDENCE)")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    elif filters_passed == 1:
        decision_path.append("STEP 3: 1/2 filters passed → MODERATE (check xG tiebreaker)")
        main_bet = "Over 1.5"
        confidence = "Moderate"
        reasoning.append(f"\n🟡 **BASE VERDICT:** {filters_passed}/2 filters passed → **OVER 1.5** (MODERATE CONFIDENCE)")
        
        # ====================================================================
        # STEP 4: xG TIEBREAKER
        # ====================================================================
        if filters.combined_xg is not None:
            decision_path.append(f"STEP 4: xG Tiebreaker - Combined xG = {filters.combined_xg}")
            reasoning.append(f"\n📊 **xG TIEBREAKER:** Combined xG = {filters.combined_xg}")
            
            if filters.combined_xg >= 2.4:
                main_bet = "Over 1.5"
                confidence = "High"
                reasoning.append(f"   → xG ≥ 2.4 → **UPGRADE to HIGH CONFIDENCE**")
            elif filters.combined_xg < 2.0:
                main_bet = "AVOID / Under 1.5"
                confidence = "Avoid"
                reasoning.append(f"   → xG < 2.0 → **DOWNGRADE to AVOID / Under 1.5**")
            else:
                reasoning.append(f"   → xG {filters.combined_xg} in 2.0-2.4 range → **No change (MODERATE)**")
        else:
            decision_path.append("STEP 4: xG data not available - no tiebreaker applied")
            reasoning.append(f"\n📊 **xG TIEBREAKER:** No xG data available → Leave as MODERATE")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )
    
    else:  # 0 filters passed
        decision_path.append("STEP 3: 0/2 filters passed → AVOID")
        main_bet = "AVOID / Under 1.5"
        confidence = "Avoid"
        reasoning.append(f"\n🔴 **VERDICT:** {filters_passed}/2 filters passed → **AVOID / UNDER 1.5**")
        
        return PredictionResult(
            main_bet=main_bet,
            confidence=confidence,
            filter_results=filters,
            reasoning=reasoning,
            decision_path=decision_path
        )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def home_team_stats_input(team_name: str) -> TeamStats:
    """Create input fields for HOME team (using HOME stats only)"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name} (HOME Team)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-note'>📌 Enter HOME team's HOME stats only (last 5-10 matches)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_total_shots = st.number_input(
            "🎯 Total Shots per game (HOME)",
            min_value=0.0, max_value=30.0, value=12.0, step=0.5,
            key=f"{team_name}_home_total_shots"
        )
        home_on_target_percent = st.number_input(
            "📊 Shots ON TARGET % (HOME)",
            min_value=0, max_value=100, value=30, step=1,
            key=f"{team_name}_home_on_target_percent"
        )
        home_scoring_freq = st.number_input(
            "✅ Scoring Frequency % (HOME) - Scored in % of matches",
            min_value=0, max_value=100, value=80, step=5,
            key=f"{team_name}_home_scoring_freq"
        )
    
    with col2:
        home_cs_rate = st.number_input(
            "🧤 Clean Sheet Rate % (HOME) - Conceded 0 in % of matches",
            min_value=0, max_value=100, value=15, step=5,
            key=f"{team_name}_home_cs_rate"
        )
        home_xg = st.number_input(
            "📈 xG per game (HOME) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_home_xg"
        )
    
    # Display calculated SOT
    calculated_sot = calculate_sot(home_total_shots, home_on_target_percent)
    st.markdown(f"<div class='results-note'>🎯 Calculated Shots ON TARGET per game: {calculated_sot}</div>", unsafe_allow_html=True)
    
    return TeamStats(
        name=team_name,
        home_total_shots=home_total_shots,
        home_on_target_percent=float(home_on_target_percent),
        home_scoring_freq=float(home_scoring_freq),
        home_cs_rate=float(home_cs_rate),
        home_xg=home_xg,
        is_home=True
    )


def away_team_stats_input(team_name: str) -> TeamStats:
    """Create input fields for AWAY team (using AWAY stats only)"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name} (AWAY Team)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-note'>📌 Enter AWAY team's AWAY stats only (last 5-10 matches)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        away_total_shots = st.number_input(
            "🎯 Total Shots per game (AWAY)",
            min_value=0.0, max_value=30.0, value=10.0, step=0.5,
            key=f"{team_name}_away_total_shots"
        )
        away_on_target_percent = st.number_input(
            "📊 Shots ON TARGET % (AWAY)",
            min_value=0, max_value=100, value=30, step=1,
            key=f"{team_name}_away_on_target_percent"
        )
        away_scoring_freq = st.number_input(
            "✅ Scoring Frequency % (AWAY) - Scored in % of matches",
            min_value=0, max_value=100, value=75, step=5,
            key=f"{team_name}_away_scoring_freq"
        )
    
    with col2:
        away_cs_rate = st.number_input(
            "🧤 Clean Sheet Rate % (AWAY) - Conceded 0 in % of matches",
            min_value=0, max_value=100, value=20, step=5,
            key=f"{team_name}_away_cs_rate"
        )
        away_xg = st.number_input(
            "📈 xG per game (AWAY) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_away_xg"
        )
    
    # Display calculated SOT
    calculated_sot = calculate_sot(away_total_shots, away_on_target_percent)
    st.markdown(f"<div class='results-note'>🎯 Calculated Shots ON TARGET per game: {calculated_sot}</div>", unsafe_allow_html=True)
    
    return TeamStats(
        name=team_name,
        away_total_shots=away_total_shots,
        away_on_target_percent=float(away_on_target_percent),
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
    <strong>🎯 FINAL LOGIC:</strong><br><br>
    <strong>STEP 1 - KILL SWITCH:</strong> If either team has Scoring Frequency &lt; 80% → 🔴 IMMEDIATE AVOID<br>
    <strong>STEP 2 - FILTER COUNT:</strong> (Only if Kill Switch passes)<br>
    &nbsp;&nbsp;&nbsp;• Filter 1: Combined SOT ≥ 8.5<br>
    &nbsp;&nbsp;&nbsp;• Filter 3: Both teams Clean Sheet Rate ≤ 20%<br>
    <strong>STEP 3 - BASE VERDICT:</strong><br>
    &nbsp;&nbsp;&nbsp;• 2/2 pass → 🟢 HIGH CONFIDENCE Over 1.5<br>
    &nbsp;&nbsp;&nbsp;• 1/2 pass → 🟡 MODERATE CONFIDENCE Over 1.5<br>
    &nbsp;&nbsp;&nbsp;• 0/2 pass → 🔴 AVOID<br>
    <strong>STEP 4 - xG TIEBREAKER:</strong> (For MODERATE only)<br>
    &nbsp;&nbsp;&nbsp;• xG ≥ 2.4 → Upgrade to HIGH<br>
    &nbsp;&nbsp;&nbsp;• xG &lt; 2.0 → Downgrade to AVOID
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
        
        # Determine display class
        if "AVOID" in result.main_bet or result.confidence == "Avoid":
            pred_class = "prediction-avoid"
            pred_icon = "🔴"
        elif result.confidence == "High":
            pred_class = "prediction-high"
            pred_icon = "🔥"
        else:
            pred_class = "prediction-moderate"
            pred_icon = "🟡"
        
        confidence_class = f"confidence-{result.confidence.lower()}"
        
        # Display prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <div class="{pred_class}">{pred_icon} {result.main_bet}</div>
            <div class="{confidence_class}">Confidence: {result.confidence}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Kill Switch status
        f = result.filter_results
        if not f.kill_switch_passed:
            st.markdown(f"""
            <div class="kill-switch-active">
            🛑 KILL SWITCH ACTIVATED<br>
            {f.kill_switch_reason}<br>
            → Immediate AVOID / Under 1.5
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kill-switch-inactive">
            ✅ KILL SWITCH PASSED<br>
            {home_stats.name} {f.home_scoring_freq}% / {away_stats.name} {f.away_scoring_freq}% (Both ≥80%)
            </div>
            """, unsafe_allow_html=True)
        
        # Display filter details (only if kill switch passed)
        if f.kill_switch_passed:
            with st.expander("📋 Filter Details", expanded=True):
                
                if f.filter1_pass:
                    st.markdown(f'<div class="filter-pass">✅ FILTER 1 PASS: Combined SOT = {f.combined_sot} ≥ 8.5<br>(Home SOT: {f.home_sot} | Away SOT: {f.away_sot})</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="filter-fail">❌ FILTER 1 FAIL: Combined SOT = {f.combined_sot} < 8.5<br>(Home SOT: {f.home_sot} | Away SOT: {f.away_sot})</div>', unsafe_allow_html=True)
                
                if f.filter3_pass:
                    st.markdown(f'<div class="filter-pass">✅ FILTER 3 PASS: Both CS rates ≤ 20%<br>({home_stats.name} {f.home_cs_rate}% | {away_stats.name} {f.away_cs_rate}%)</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="filter-fail">❌ FILTER 3 FAIL: Need both ≤20%<br>({home_stats.name} {f.home_cs_rate}% | {away_stats.name} {f.away_cs_rate}%)</div>', unsafe_allow_html=True)
                
                if f.combined_xg:
                    st.markdown(f'<div class="step-box">📊 Combined xG (Tiebreaker): {f.combined_xg}</div>', unsafe_allow_html=True)
        
        # Display decision path
        with st.expander("📋 Decision Path", expanded=False):
            for i, step in enumerate(result.decision_path, 1):
                st.markdown(f'<div class="step-box">{i}. {step}</div>', unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📊 Detailed Analysis", expanded=False):
            for line in result.reasoning:
                if "🔴" in line or "❌" in line:
                    st.error(line)
                elif "🟢" in line or "✅" in line:
                    st.success(line)
                elif "🟡" in line:
                    st.warning(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📈 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_stats.name} - Total Shots (HOME)",
                    f"{home_stats.name} - ON Target % (HOME)",
                    f"{home_stats.name} - SOT (HOME) - Calculated",
                    f"{home_stats.name} - Scoring Frequency (HOME)",
                    f"{home_stats.name} - Clean Sheet Rate (HOME)",
                    f"{away_stats.name} - Total Shots (AWAY)",
                    f"{away_stats.name} - ON Target % (AWAY)",
                    f"{away_stats.name} - SOT (AWAY) - Calculated",
                    f"{away_stats.name} - Scoring Frequency (AWAY)",
                    f"{away_stats.name} - Clean Sheet Rate (AWAY)",
                    "Combined SOT",
                ],
                "Value": [
                    f"{home_stats.home_total_shots}",
                    f"{home_stats.home_on_target_percent}%",
                    f"{f.home_sot}",
                    f"{f.home_scoring_freq}%",
                    f"{f.home_cs_rate}%",
                    f"{away_stats.away_total_shots}",
                    f"{away_stats.away_on_target_percent}%",
                    f"{f.away_sot}",
                    f"{f.away_scoring_freq}%",
                    f"{f.away_cs_rate}%",
                    f"{f.combined_sot}",
                ],
                "Threshold": [
                    "-",
                    "-",
                    "-",
                    "≥ 80%",
                    "≤ 20%",
                    "-",
                    "-",
                    "-",
                    "≥ 80%",
                    "≤ 20%",
                    "≥ 8.5",
                ],
                "Pass?": [
                    "-",
                    "-",
                    "-",
                    "✅" if f.home_scoring_freq >= 80 else "❌",
                    "✅" if f.home_cs_rate <= 20 else "❌",
                    "-",
                    "-",
                    "-",
                    "✅" if f.away_scoring_freq >= 80 else "❌",
                    "✅" if f.away_cs_rate <= 20 else "❌",
                    "✅" if f.filter1_pass else "❌",
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 Complete Decision Rules
    
    | Step | Condition | Action |
    |------|-----------|--------|
    | **KILL SWITCH** | Either team Scoring Frequency < 80% | 🔴 **IMMEDIATE AVOID** |
    | **Filter Count** | Both teams ≥80% scoring frequency | Proceed to check SOT + CS Rate |
    | **2/2 Pass** | SOT ≥ 8.5 AND Both CS ≤ 20% | 🟢 **HIGH CONFIDENCE Over 1.5** |
    | **1/2 Pass** | Only one of SOT or CS passes | 🟡 **MODERATE CONFIDENCE** |
    | **0/2 Pass** | Neither SOT nor CS passes | 🔴 **AVOID / Under 1.5** |
    
    ### xG Tiebreaker (For MODERATE only)
    
    | Combined xG | Action |
    |-------------|--------|
    | **≥ 2.4** | ⬆️ Upgrade to HIGH |
    | **2.0 - 2.4** | 🟡 Leave as MODERATE |
    | **< 2.0** | ⬇️ Downgrade to AVOID |
    
    ### 🎯 Critical Input Rules
    
    | Team | Use These Stats Only |
    |------|---------------------|
    | **Home Team** | HOME Total Shots + HOME ON Target % + HOME Scoring Frequency + HOME Clean Sheet Rate |
    | **Away Team** | AWAY Total Shots + AWAY ON Target % + AWAY Scoring Frequency + AWAY Clean Sheet Rate |
    
    ### 📊 Data Source
    
    All stats should be based on **last 5-10 matches** (minimum 5 matches for statistical significance).
    
    ### 🛡️ The Kill Switch
        
    **If either team fails to score in ≥20% of their matches → DO NOT BET Over 1.5.**
    
    This single rule would have saved every failed prediction in our test cases.
    """)

if __name__ == "__main__":
    main()
