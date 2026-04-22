"""
Over 1.5 Goals Predictor - 3-Filter Strict System
Based on statistical analysis of home/away SOT, scoring frequency, and clean sheet rates.

3-Filter Logic:
- Filter 1: Combined SOT (Home SOT + Away SOT) ≥ 8.5
- Filter 2: Both teams scoring frequency ≥ 80%
- Filter 3: Both teams clean sheet rate ≤ 20%

Confidence Levels:
- 3/3 Filters Pass → HIGH CONFIDENCE Over 1.5
- 2/3 Filters Pass → MODERATE CONFIDENCE Over 1.5
- 0-1/3 Filters Pass → AVOID / Under 1.5

Tiebreaker (xG):
- If 2/3 pass AND combined xG ≥ 2.4 → Upgrade to HIGH
- If 2/3 pass AND combined xG < 2.0 → Downgrade to AVOID
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Over 1.5 Goals Predictor - 3-Filter System",
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
    .prediction-avoid {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ef4444;
    }
    .prediction-moderate {
        font-size: 2rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .expected-goals {
        font-size: 1.5rem;
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
    .secondary {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
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
    home_sot: float = 0.0           # Shots on target per game at home
    home_scoring_freq: float = 0.0  # % of home matches they scored in
    home_cs_rate: float = 0.0       # % of home matches they kept a clean sheet
    home_xg: float = 0.0            # Expected goals per game at home (optional)
    
    # For Away Team (use AWAY stats only)
    away_sot: float = 0.0           # Shots on target per game away
    away_scoring_freq: float = 0.0  # % of away matches they scored in
    away_cs_rate: float = 0.0       # % of away matches they kept a clean sheet
    away_xg: float = 0.0            # Expected goals per game away (optional)
    
    # Flags to indicate which stats are being used
    is_home: bool = True


@dataclass
class FilterResults:
    """Results of the 3 filters"""
    filter1_pass: bool  # Combined SOT ≥ 8.5
    filter2_pass: bool  # Both teams scoring frequency ≥ 80%
    filter3_pass: bool  # Both teams clean sheet rate ≤ 20%
    combined_sot: float
    home_scoring_freq: float
    away_scoring_freq: float
    home_cs_rate: float
    away_cs_rate: float
    combined_xg: Optional[float] = None


@dataclass
class PredictionResult:
    main_bet: str  # "Over 1.5", "MODERATE Over 1.5", or "AVOID / Under 1.5"
    confidence: str  # "High", "Moderate", "Avoid"
    filters_passed: int
    filter_results: FilterResults
    reasoning: List[str]
    decision_path: List[str]


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================
def calculate_combined_sot(home: TeamStats, away: TeamStats) -> float:
    """Calculate combined shots on target"""
    home_sot = home.home_sot if home.is_home else home.away_sot
    away_sot = away.away_sot if not away.is_home else away.home_sot
    return round(home_sot + away_sot, 2)


def calculate_combined_xg(home: TeamStats, away: TeamStats) -> Optional[float]:
    """Calculate combined xG if available"""
    home_xg = home.home_xg if home.is_home else home.away_xg
    away_xg = away.away_xg if not away.is_home else away.home_xg
    
    if home_xg > 0 and away_xg > 0:
        return round(home_xg + away_xg, 2)
    return None


def evaluate_filters(
    home: TeamStats,
    away: TeamStats
) -> FilterResults:
    """Evaluate all 3 filters"""
    
    # Get the correct stats based on home/away
    home_sot = home.home_sot if home.is_home else home.away_sot
    away_sot = away.away_sot if not away.is_home else away.home_sot
    combined_sot = home_sot + away_sot
    
    home_scoring_freq = home.home_scoring_freq if home.is_home else home.away_scoring_freq
    away_scoring_freq = away.away_scoring_freq if not away.is_home else away.home_scoring_freq
    
    home_cs_rate = home.home_cs_rate if home.is_home else home.away_cs_rate
    away_cs_rate = away.away_cs_rate if not away.is_home else away.home_cs_rate
    
    # Filter 1: Combined SOT ≥ 8.5
    filter1_pass = combined_sot >= 8.5
    
    # Filter 2: Both teams scoring frequency ≥ 80%
    filter2_pass = home_scoring_freq >= 80.0 and away_scoring_freq >= 80.0
    
    # Filter 3: Both teams clean sheet rate ≤ 20%
    filter3_pass = home_cs_rate <= 20.0 and away_cs_rate <= 20.0
    
    # Combined xG (if available)
    combined_xg = None
    if home.home_xg > 0 and away.away_xg > 0:
        home_xg = home.home_xg if home.is_home else home.away_xg
        away_xg = away.away_xg if not away.is_home else away.home_xg
        combined_xg = home_xg + away_xg
    
    return FilterResults(
        filter1_pass=filter1_pass,
        filter2_pass=filter2_pass,
        filter3_pass=filter3_pass,
        combined_sot=combined_sot,
        home_scoring_freq=home_scoring_freq,
        away_scoring_freq=away_scoring_freq,
        home_cs_rate=home_cs_rate,
        away_cs_rate=away_cs_rate,
        combined_xg=combined_xg
    )


def make_prediction(
    home: TeamStats,
    away: TeamStats
) -> PredictionResult:
    """
    Make Over 1.5 Goals prediction based on 3-Filter logic.
    """
    reasoning = []
    decision_path = []
    
    # Evaluate filters
    filters = evaluate_filters(home, away)
    
    filters_passed = sum([
        filters.filter1_pass,
        filters.filter2_pass,
        filters.filter3_pass
    ])
    
    reasoning.append(f"📊 **Input Summary:**")
    reasoning.append(f"   • Combined SOT: {filters.combined_sot} (Threshold: ≥8.5)")
    reasoning.append(f"   • {home.name} Scoring Frequency: {filters.home_scoring_freq}% (Threshold: ≥80%)")
    reasoning.append(f"   • {away.name} Scoring Frequency: {filters.away_scoring_freq}% (Threshold: ≥80%)")
    reasoning.append(f"   • {home.name} Clean Sheet Rate: {filters.home_cs_rate}% (Threshold: ≤20%)")
    reasoning.append(f"   • {away.name} Clean Sheet Rate: {filters.away_cs_rate}% (Threshold: ≤20%)")
    
    if filters.combined_xg:
        reasoning.append(f"   • Combined xG: {filters.combined_xg} (Tiebreaker threshold: ≥2.4)")
    
    # Display filter results
    reasoning.append(f"\n📋 **3-Filter Results:**")
    
    if filters.filter1_pass:
        reasoning.append(f"   ✅ Filter 1 PASS: Combined SOT {filters.combined_sot} ≥ 8.5")
    else:
        reasoning.append(f"   ❌ Filter 1 FAIL: Combined SOT {filters.combined_sot} < 8.5")
    
    if filters.filter2_pass:
        reasoning.append(f"   ✅ Filter 2 PASS: Both scoring frequencies ≥ 80%")
    else:
        reasoning.append(f"   ❌ Filter 2 FAIL: {home.name} {filters.home_scoring_freq}% / {away.name} {filters.away_scoring_freq}%")
    
    if filters.filter3_pass:
        reasoning.append(f"   ✅ Filter 3 PASS: Both clean sheet rates ≤ 20%")
    else:
        reasoning.append(f"   ❌ Filter 3 FAIL: {home.name} {filters.home_cs_rate}% / {away.name} {filters.away_cs_rate}%")
    
    reasoning.append(f"\n📊 **Filters Passed: {filters_passed}/3**")
    
    # Determine base prediction
    if filters_passed == 3:
        decision_path.append("3 of 3 filters passed")
        main_bet = "Over 1.5"
        confidence = "High"
        reasoning.append(f"\n✅ **VERDICT:** {filters_passed}/3 filters passed → **OVER 1.5** (HIGH CONFIDENCE)")
        
    elif filters_passed == 2:
        decision_path.append("2 of 3 filters passed - checking tiebreaker")
        
        # Apply xG tiebreaker
        if filters.combined_xg:
            if filters.combined_xg >= 2.4:
                main_bet = "Over 1.5"
                confidence = "High"
                reasoning.append(f"\n✅ **TIEBREAKER:** Combined xG {filters.combined_xg} ≥ 2.4 → Upgrade to HIGH CONFIDENCE")
                reasoning.append(f"   → **OVER 1.5** (HIGH CONFIDENCE)")
            elif filters.combined_xg < 2.0:
                main_bet = "AVOID / Under 1.5"
                confidence = "Avoid"
                reasoning.append(f"\n⚠️ **TIEBREAKER:** Combined xG {filters.combined_xg} < 2.0 → Downgrade to AVOID")
                reasoning.append(f"   → **AVOID / UNDER 1.5**")
            else:
                main_bet = "Over 1.5"
                confidence = "Moderate"
                reasoning.append(f"\n🟡 **VERDICT:** {filters_passed}/3 filters passed → **OVER 1.5** (MODERATE CONFIDENCE)")
                reasoning.append(f"   (xG {filters.combined_xg} in 2.0-2.4 range - no upgrade/downgrade)")
        else:
            # No xG data available
            main_bet = "Over 1.5"
            confidence = "Moderate"
            reasoning.append(f"\n🟡 **VERDICT:** {filters_passed}/3 filters passed → **OVER 1.5** (MODERATE CONFIDENCE)")
            reasoning.append(f"   (xG data not available for tiebreaker)")
        
    else:  # 0 or 1 filter passed
        decision_path.append(f"{filters_passed} of 3 filters passed - insufficient")
        main_bet = "AVOID / Under 1.5"
        confidence = "Avoid"
        reasoning.append(f"\n🔴 **VERDICT:** Only {filters_passed}/3 filters passed → **AVOID / UNDER 1.5**")
        reasoning.append(f"   (Statistical risk of low-scoring match)")
    
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
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name} (HOME Team)</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='input-note'>📌 Enter HOME team's HOME stats only (last 5-10 matches)</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        home_sot = st.number_input(
            "🎯 Shots on Target per game (HOME)",
            min_value=0.0, max_value=15.0, value=4.5, step=0.1,
            key=f"{team_name}_home_sot"
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
            "📊 xG per game (HOME) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_home_xg"
        )
    
    return TeamStats(
        name=team_name,
        home_sot=home_sot,
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
        away_sot = st.number_input(
            "🎯 Shots on Target per game (AWAY)",
            min_value=0.0, max_value=15.0, value=4.0, step=0.1,
            key=f"{team_name}_away_sot"
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
            "📊 xG per game (AWAY) - Optional (0 if unknown)",
            min_value=0.0, max_value=5.0, value=0.0, step=0.05,
            key=f"{team_name}_away_xg"
        )
    
    return TeamStats(
        name=team_name,
        away_sot=away_sot,
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
    st.caption("3-Filter Strict System: SOT + Scoring Frequency + Clean Sheet Rate")
    
    st.markdown("""
    <div class="step-box">
    <strong>🎯 3-Filter Logic:</strong><br>
    • <strong>Filter 1:</strong> Combined SOT (Home + Away) ≥ 8.5<br>
    • <strong>Filter 2:</strong> Both teams scoring frequency ≥ 80%<br>
    • <strong>Filter 3:</strong> Both teams clean sheet rate ≤ 20%<br><br>
    <strong>Confidence Levels:</strong><br>
    • 3/3 PASS → 🔥 HIGH CONFIDENCE Over 1.5<br>
    • 2/3 PASS → 🟡 MODERATE CONFIDENCE Over 1.5 (xG tiebreaker may upgrade/downgrade)<br>
    • 0-1/3 PASS → 🔴 AVOID / Under 1.5
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
        if "Over 1.5" in result.main_bet and "MODERATE" not in result.main_bet and "AVOID" not in result.main_bet:
            pred_class = "prediction-over"
            pred_icon = "🔥"
        elif "MODERATE" in result.main_bet or (result.main_bet == "Over 1.5" and result.confidence == "Moderate"):
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
            <div class="secondary">Filters Passed: {result.filters_passed}/3</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display filter details
        with st.expander("📋 Filter Details", expanded=True):
            f = result.filter_results
            
            if f.filter1_pass:
                st.markdown(f'<div class="filter-pass">✅ FILTER 1 PASS: Combined SOT = {f.combined_sot} ≥ 8.5</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="filter-fail">❌ FILTER 1 FAIL: Combined SOT = {f.combined_sot} < 8.5</div>', unsafe_allow_html=True)
            
            if f.filter2_pass:
                st.markdown(f'<div class="filter-pass">✅ FILTER 2 PASS: {home_stats.name} {f.home_scoring_freq}% / {away_stats.name} {f.away_scoring_freq}% (Both ≥80%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="filter-fail">❌ FILTER 2 FAIL: {home_stats.name} {f.home_scoring_freq}% / {away_stats.name} {f.away_scoring_freq}% (Need both ≥80%)</div>', unsafe_allow_html=True)
            
            if f.filter3_pass:
                st.markdown(f'<div class="filter-pass">✅ FILTER 3 PASS: {home_stats.name} {f.home_cs_rate}% / {away_stats.name} {f.away_cs_rate}% (Both ≤20%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="filter-fail">❌ FILTER 3 FAIL: {home_stats.name} {f.home_cs_rate}% / {away_stats.name} {f.away_cs_rate}% (Need both ≤20%)</div>', unsafe_allow_html=True)
            
            if f.combined_xg:
                st.markdown(f'<div class="step-box">📊 Combined xG (Tiebreaker): {f.combined_xg}</div>', unsafe_allow_html=True)
        
        # Display detailed reasoning
        with st.expander("📊 Detailed Analysis", expanded=False):
            for line in result.reasoning:
                if "✅" in line:
                    st.success(line)
                elif "🟡" in line:
                    st.warning(line)
                elif "🔴" in line or "❌" in line:
                    st.error(line)
                elif "⚠️" in line:
                    st.warning(line)
                else:
                    st.write(line)
        
        # Display data table
        with st.expander("📈 Data Summary", expanded=False):
            data = {
                "Metric": [
                    f"{home_stats.name} - SOT (HOME)",
                    f"{away_stats.name} - SOT (AWAY)",
                    "Combined SOT",
                    f"{home_stats.name} - Scoring Frequency (HOME)",
                    f"{away_stats.name} - Scoring Frequency (AWAY)",
                    f"{home_stats.name} - Clean Sheet Rate (HOME)",
                    f"{away_stats.name} - Clean Sheet Rate (AWAY)",
                ],
                "Value": [
                    f"{home_stats.home_sot}",
                    f"{away_stats.away_sot}",
                    f"{result.filter_results.combined_sot}",
                    f"{result.filter_results.home_scoring_freq}%",
                    f"{result.filter_results.away_scoring_freq}%",
                    f"{result.filter_results.home_cs_rate}%",
                    f"{result.filter_results.away_cs_rate}%",
                ],
                "Threshold": [
                    "-",
                    "-",
                    "≥ 8.5",
                    "≥ 80%",
                    "≥ 80%",
                    "≤ 20%",
                    "≤ 20%",
                ],
                "Pass?": [
                    "-",
                    "-",
                    "✅" if result.filter_results.filter1_pass else "❌",
                    "✅" if result.filter_results.filter2_pass else "❌",
                    "✅" if result.filter_results.filter2_pass else "❌",
                    "✅" if result.filter_results.filter3_pass else "❌",
                    "✅" if result.filter_results.filter3_pass else "❌",
                ]
            }
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Footer
    st.divider()
    st.markdown("""
    ### 📋 3-Filter Decision Rules
    
    | Filters Passed | Verdict | Confidence |
    |----------------|---------|------------|
    | **3 of 3** | **Over 1.5** | HIGH 🔥 |
    | **2 of 3** | **Over 1.5** | MODERATE 🟡 |
    | **2 of 3 + xG ≥ 2.4** | **Over 1.5** | HIGH (Upgraded) 🔥 |
    | **2 of 3 + xG < 2.0** | **AVOID / Under 1.5** | AVOID (Downgraded) 🔴 |
    | **0-1 of 3** | **AVOID / Under 1.5** | AVOID 🔴 |
    
    ### 🎯 Critical Input Rules
    
    | Team | Use These Stats Only |
    |------|---------------------|
    | **Home Team** | HOME SOT + HOME Scoring Frequency + HOME Clean Sheet Rate |
    | **Away Team** | AWAY SOT + AWAY Scoring Frequency + AWAY Clean Sheet Rate |
    
    ### 📊 Data Source
    
    All stats should be based on **last 5-10 matches** (minimum 5 matches for statistical significance).
    
    ### 🛡️ When to AVOID
    
    - Combined SOT < 8.5 (most important filter)
    - Either team scores in < 80% of matches
    - Either team keeps clean sheets in > 20% of matches
    - 2 filters pass but xG < 2.0 (downgrade to AVOID)
    """)

if __name__ == "__main__":
    main()
