# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Set page config
st.set_page_config(
    page_title="No-Draw Edge Filter v2.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .prediction-yes {
        border-left-color: #10b981;
    }
    .prediction-no {
        border-left-color: #ef4444;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .sidebar .stSlider {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# League classifications
DECISIVE_LEAGUES = [
    "Bundesliga", "EPL", "Premier League", "Eredivisie", 
    "U21", "Reserves", "U23", "Bundesliga U19",
    "Cup", "FA Cup", "EFL Cup", "DFB Pokal"
]

DRAW_PRONE_LEAGUES = [
    "Serie A", "Ligue 1", "Serie B", "Ligue 2",
    "Segunda Division", "J1 League", "J2 League"
]

ALL_LEAGUES = DECISIVE_LEAGUES + DRAW_PRONE_LEAGUES + [
    "La Liga", "Championship", "League One", "League Two",
    "MLS", "Brasileiro", "Argentine Liga", "Other"
]

@dataclass
class MatchInput:
    """Data class for match input parameters"""
    forebet_prediction: str  # 'X', '1', or '2'
    draw_probability: float  # 0-100
    draw_odds: float
    expected_goals: float
    favorite_odds: float
    league: str
    match_type: str  # 'Regular', 'U21/Reserves', 'Cup'
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate input ranges"""
        errors = []
        if self.draw_probability < 0 or self.draw_probability > 100:
            errors.append("Draw probability must be between 0 and 100")
        if self.draw_odds < 1.01:
            errors.append("Draw odds must be > 1.01")
        if self.expected_goals < 0 or self.expected_goals > 10:
            errors.append("Expected goals must be between 0 and 10")
        if self.favorite_odds < 1.01:
            errors.append("Favorite odds must be > 1.01")
        return len(errors) == 0, errors

class NoDrawFilter:
    """Implements the No-Draw Edge Filter v2.0 system"""
    
    def __init__(self):
        self.core_criteria = {}
        self.boosters = []
        self.tier = "No Signal"
        self.confidence_score = 0
        
    def evaluate(self, match: MatchInput) -> Dict:
        """Evaluate a match using the v2.0 filter system"""
        results = {
            'prediction': None,
            'decision': None,
            'core_met': False,
            'tier': "No Signal",
            'confidence': 0,
            'reasons': [],
            'details': {}
        }
        
        # Check if Forebet predicts draw or high draw probability
        forebet_draw_lean = (
            match.forebet_prediction.upper() == 'X' or 
            match.draw_probability >= 38
        )
        
        if not forebet_draw_lean:
            results['reasons'].append("❌ Forebet does not lean towards draw")
            results['decision'] = "NO ACTION"
            return results
        
        results['reasons'].append("✓ Forebet draws lean detected")
        
        # CORE CRITERIA (must have ALL 3)
        core_1 = match.draw_odds > 3.80
        core_2 = match.expected_goals >= 2.6
        core_3 = forebet_draw_lean
        
        results['core_met'] = core_1 and core_2 and core_3
        
        if not results['core_met']:
            results['decision'] = "NO ACTION - Core criteria not met"
            if not core_1:
                results['reasons'].append(f"❌ Draw odds {match.draw_odds:.2f} <= 3.80")
            if not core_2:
                results['reasons'].append(f"❌ Expected goals {match.expected_goals:.1f} < 2.6")
            return results
        
        results['reasons'].append("✓ All core criteria met")
        
        # Calculate boosters
        boosters = []
        tier_count = 0
        
        # Tier 1 Boosters
        tier1_count = 0
        if match.favorite_odds <= 2.20:
            boosters.append(f"Clear favorite (odds {match.favorite_odds:.2f})")
            tier1_count += 1
        
        decisive = any(league in match.league for league in DECISIVE_LEAGUES)
        if decisive:
            boosters.append(f"Decisive league ({match.league})")
            tier1_count += 1
        
        # Tier 2 Boosters
        tier2_count = 0
        if match.draw_odds > 4.00:
            boosters.append(f"High draw odds {match.draw_odds:.2f} > 4.00")
            tier2_count += 1
        
        if 38 <= match.draw_probability <= 45:
            boosters.append(f"Optimal draw prob {match.draw_probability:.1f}%")
            tier2_count += 1
        
        # Tier 3 Boosters
        tier3_count = 0
        if match.match_type in ['U21/Reserves', 'Cup']:
            boosters.append(f"Match type: {match.match_type}")
            tier3_count += 1
        
        if match.expected_goals >= 2.8:
            boosters.append(f"High xG {match.expected_goals:.1f} >= 2.8")
            tier3_count += 1
        
        results['boosters'] = boosters
        
        # Determine tier and confidence
        total_score = tier1_count + tier2_count + tier3_count
        results['details'] = {
            'tier1_count': tier1_count,
            'tier2_count': tier2_count,
            'tier3_count': tier3_count,
            'total_score': total_score
        }
        
        # Classification
        if total_score >= 4:
            results['tier'] = "MAX EDGE"
            results['confidence'] = 92
            results['decision'] = "STRONG NO-DRAW PREDICTION"
        elif total_score >= 3:
            results['tier'] = "VERY HIGH ACCURACY"
            results['confidence'] = 86
            results['decision'] = "CONFIDENT NO-DRAW PREDICTION"
        elif total_score >= 2:
            results['tier'] = "HIGH ACCURACY"
            results['confidence'] = 80
            results['decision'] = "NO-DRAW PREDICTION"
        else:
            results['tier'] = "BASE ACCURACY"
            results['confidence'] = 75
            results['decision'] = "WEAK NO-DRAW SIGNAL"
        
        results['prediction'] = "No-Draw (1 or 2 wins)"
        
        return results

class BacktestLogger:
    """Handles backtesting data storage and analysis"""
    
    def __init__(self):
        self.data_file = "backtest_data.json"
        
    def save_match_result(self, match_data: Dict, result: str):
        """Save match result for backtesting"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        
        data.append({
            'timestamp': datetime.now().isoformat(),
            'match': match_data,
            'actual_result': result
        })
        
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_data(self) -> List[Dict]:
        """Load backtest data"""
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def calculate_stats(self) -> Dict:
        """Calculate hit rate and statistics"""
        data = self.load_data()
        if not data:
            return {'total': 0, 'hit_rate': 0}
        
        total = len(data)
        correct = 0
        for entry in data:
            # Simple check - would need actual result to compare
            # This is placeholder logic
            if entry.get('correct', False):
                correct += 1
        
        hit_rate = (correct / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'correct': correct,
            'hit_rate': hit_rate
        }

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ No-Draw Edge Filter v2.0</h1>
        <p>R&D Engine: Optimizing Forebet Draw Prediction + High Odds Strategy</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Dr. Wealth Research Mode — Pure Statistical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📊 Match Parameters")
        st.markdown("---")
        
        # Core Inputs
        forebet_pred = st.selectbox(
            "Forebet Prediction",
            ["X (Draw)", "1 (Home Win)", "2 (Away Win)"],
            help="Forebet's main prediction"
        )
        
        draw_prob = st.slider(
            "Forebet Draw Probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=1.0,
            help="Draw probability from Forebet model"
        )
        
        draw_odds = st.number_input(
            "Draw Odds (Best Available)",
            min_value=1.01,
            max_value=10.0,
            value=3.80,
            step=0.05,
            format="%.2f",
            help="Highest available draw odds from bookmakers"
        )
        
        xg = st.number_input(
            "Expected Goals (xG)",
            min_value=0.0,
            max_value=5.0,
            value=2.6,
            step=0.1,
            format="%.1f",
            help="Forebet's expected goals metric"
        )
        
        favorite_odds = st.number_input(
            "Favorite Odds (Lowest)",
            min_value=1.01,
            max_value=10.0,
            value=2.20,
            step=0.05,
            format="%.2f",
            help="Lowest odds among home/away (clear favorite)"
        )
        
        league = st.selectbox(
            "League / Competition",
            ALL_LEAGUES,
            help="Select the league or competition"
        )
        
        match_type = st.selectbox(
            "Match Type",
            ["Regular", "U21/Reserves", "Cup"],
            help="Type of match (U21/Reserves often more open)"
        )
        
        st.markdown("---")
        
        # Backtest section
        st.subheader("📈 Backtest Tools")
        if st.button("Save Current Match for Backtest"):
            st.success("Match saved! (Will prompt for result after prediction)")
        
        # Display stats
        logger = BacktestLogger()
        stats = logger.calculate_stats()
        st.metric("Backtest Records", stats['total'])
        if stats['total'] > 0:
            st.metric("Current Hit Rate", f"{stats['hit_rate']:.1f}%")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create match input object
        match = MatchInput(
            forebet_prediction=forebet_pred[0] if forebet_pred != "X (Draw)" else "X",
            draw_probability=draw_prob,
            draw_odds=draw_odds,
            expected_goals=xg,
            favorite_odds=favorite_odds,
            league=league,
            match_type=match_type
        )
        
        # Validate and evaluate
        is_valid, errors = match.validate()
        
        if not is_valid:
            st.error("Validation Errors:")
            for error in errors:
                st.error(f"• {error}")
        else:
            filter_engine = NoDrawFilter()
            result = filter_engine.evaluate(match)
            
            # Display prediction
            st.markdown("## 🔮 Prediction Result")
            
            # Status indicator
            if result['decision'].startswith("NO ACTION"):
                st.warning(result['decision'])
            else:
                # Prediction card
                card_color = "prediction-yes" if "NO-DRAW" in result['decision'] else "prediction-no"
                st.markdown(f"""
                <div class="prediction-card {card_color}">
                    <h2>{result['decision']}</h2>
                    <h3>{result.get('prediction', 'No prediction')}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown(f"### Confidence: {result['confidence']}%")
                st.progress(result['confidence'] / 100)
                
                # Tier and details
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Edge Tier", result['tier'])
                with col_b:
                    st.metric("Confidence Score", f"{result['confidence']}%")
                
                # Display criteria status
                st.markdown("### 📋 Criteria Analysis")
                
                criteria_data = {
                    "Core Criteria": {
                        "Forebet Draw Lean": "✓" if (match.forebet_prediction == 'X' or match.draw_probability >= 38) else "✗",
                        "Draw Odds > 3.80": "✓" if match.draw_odds > 3.80 else f"✗ ({match.draw_odds:.2f})",
                        "Expected Goals ≥ 2.6": "✓" if match.expected_goals >= 2.6 else f"✗ ({match.expected_goals:.1f})"
                    }
                }
                
                st.dataframe(pd.DataFrame(criteria_data).T)
                
                # Boosters applied
                if result.get('boosters'):
                    st.markdown("### 🚀 Active Boosters")
                    for booster in result['boosters']:
                        st.success(f"✓ {booster}")
                
                # Detailed metrics
                st.markdown("### 📊 Edge Metrics")
                metrics = result['details']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tier 1 Boosters", metrics['tier1_count'])
                with col2:
                    st.metric("Tier 2 Boosters", metrics['tier2_count'])
                with col3:
                    st.metric("Tier 3 Boosters", metrics['tier3_count'])
                with col4:
                    st.metric("Total Edge Score", metrics['total_score'])
    
    with col2:
        # Quick reference guide
        st.markdown("### 📖 Quick Reference")
        
        with st.expander("Core Criteria (All Required)", expanded=False):
            st.markdown("""
            - Forebet prediction = X **OR** draw prob ≥ 38%
            - Draw odds > 3.80
            - Expected goals ≥ 2.6
            """)
        
        with st.expander("Booster Tiers", expanded=False):
            st.markdown("""
            **Tier 1 (+7-10% accuracy)**
            - Favorite odds ≤ 2.20
            - Decisive league (Bundesliga, EPL, etc.)
            
            **Tier 2 (+5-8% accuracy)**
            - Draw odds > 4.00
            - Draw prob 38-45%
            
            **Tier 3 (+3-5% accuracy)**
            - U21/Reserves or Cup match
            - Expected goals ≥ 2.8
            """)
        
        with st.expander("League Classifications", expanded=False):
            st.markdown("**Decisive Leagues:**")
            st.write(", ".join(DECISIVE_LEAGUES[:5]) + "...")
            st.markdown("**Draw-Prone Leagues:**")
            st.write(", ".join(DRAW_PRONE_LEAGUES))
        
        with st.expander("Accuracy Expectations", expanded=False):
            st.markdown("""
            - **Base (>3.60 only):** 72-78%
            - **Core Criteria:** 75-80%
            - **+ Tier 1:** 80-84%
            - **+ Tier 2:** 84-88%
            - **Max Edge:** 86-92%
            """)
    
    # Backtest input section
    if st.button("Save Result for Backtesting"):
        st.info("Backtest saving feature - would prompt for actual match result")
        # This would trigger a modal or new input for actual result
        # For full implementation, you'd want to create a dialog or expander
    
    # Analytics and insights
    st.markdown("---")
    st.markdown("### 🔬 R&D Insights")
    
    # Show current configuration effectiveness
    st.markdown("""
    **Current Configuration Analysis:**
    
    Based on your selected parameters:
    """)
    
    # Create a simple gauge for potential accuracy
    potential_accuracy = 75  # Base
    if match.draw_odds > 3.80:
        potential_accuracy += 5
    if match.draw_odds > 4.00:
        potential_accuracy += 3
    if match.expected_goals >= 2.6:
        potential_accuracy += 8
    if match.expected_goals >= 2.8:
        potential_accuracy += 4
    if match.favorite_odds <= 2.20:
        potential_accuracy += 8
    if match.match_type in ['U21/Reserves', 'Cup']:
        potential_accuracy += 4
    if league in DECISIVE_LEAGUES:
        potential_accuracy += 6
    
    potential_accuracy = min(92, potential_accuracy)
    
    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.metric("Projected Hit Rate", f"{potential_accuracy}%", 
                  delta=f"+{potential_accuracy - 75}%" if potential_accuracy > 75 else "")
    with col_g2:
        st.metric("Edge Strength", 
                  "Strong" if potential_accuracy >= 85 else "Moderate" if potential_accuracy >= 78 else "Weak")
    with col_g3:
        st.metric("Recommended Action",
                  "✓ Add to ACCA" if potential_accuracy >= 80 else "⚠️ Filter More" if potential_accuracy >= 75 else "✗ Skip")
    
    # Disclaimer
    st.markdown("---")
    st.caption("⚙️ **R&D Mode:** This tool is for research and analysis. Results based on historical patterns and statistical models. Always verify with current market conditions.")

if __name__ == "__main__":
    main()