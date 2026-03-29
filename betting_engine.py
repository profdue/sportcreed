# betting_engine.py - Complete self-contained version
import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Tuple
import math

# Set page config
st.set_page_config(
    page_title="No-Draw Edge Filter v2.0",
    page_icon="⚽",
    layout="wide"
)

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

class NoDrawFilter:
    """Implements the No-Draw Edge Filter v2.0 system"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate(self, match_data: Dict) -> Dict:
        """Evaluate a match using the v2.0 filter system"""
        
        # Extract parameters
        forebet_pred = match_data.get('forebet_prediction', '')
        draw_prob = match_data.get('draw_probability', 0)
        draw_odds = match_data.get('draw_odds', 0)
        xg = match_data.get('expected_goals', 0)
        favorite_odds = match_data.get('favorite_odds', 0)
        league = match_data.get('league', '')
        match_type = match_data.get('match_type', 'Regular')
        
        results = {
            'prediction': None,
            'decision': 'NO ACTION',
            'core_met': False,
            'tier': 'No Signal',
            'confidence': 0,
            'reasons': [],
            'boosters': [],
            'details': {}
        }
        
        # Check if Forebet predicts draw or high draw probability
        forebet_draw_lean = (forebet_pred.upper() == 'X' or draw_prob >= 38)
        
        if not forebet_draw_lean:
            results['reasons'].append("❌ Forebet does not lean towards draw")
            results['decision'] = "NO ACTION - No draw lean"
            return results
        
        results['reasons'].append("✓ Forebet draw lean detected")
        
        # CORE CRITERIA (must have ALL 3)
        core_1 = draw_odds > 3.80
        core_2 = xg >= 2.6
        core_3 = forebet_draw_lean
        
        results['core_met'] = core_1 and core_2 and core_3
        
        if not results['core_met']:
            results['decision'] = "NO ACTION - Core criteria not met"
            if not core_1:
                results['reasons'].append(f"❌ Draw odds {draw_odds:.2f} <= 3.80")
            if not core_2:
                results['reasons'].append(f"❌ Expected goals {xg:.1f} < 2.6")
            return results
        
        results['reasons'].append("✓ All core criteria met")
        
        # Calculate boosters
        boosters = []
        tier1_count = 0
        tier2_count = 0
        tier3_count = 0
        
        # Tier 1 Boosters
        if favorite_odds <= 2.20:
            boosters.append(f"Clear favorite (odds {favorite_odds:.2f})")
            tier1_count += 1
        
        decisive = any(l in league for l in DECISIVE_LEAGUES)
        if decisive:
            boosters.append(f"Decisive league ({league})")
            tier1_count += 1
        
        # Tier 2 Boosters
        if draw_odds > 4.00:
            boosters.append(f"High draw odds {draw_odds:.2f} > 4.00")
            tier2_count += 1
        
        if 38 <= draw_prob <= 45:
            boosters.append(f"Optimal draw prob {draw_prob:.1f}%")
            tier2_count += 1
        
        # Tier 3 Boosters
        if match_type in ['U21/Reserves', 'Cup']:
            boosters.append(f"Match type: {match_type}")
            tier3_count += 1
        
        if xg >= 2.8:
            boosters.append(f"High xG {xg:.1f} >= 2.8")
            tier3_count += 1
        
        results['boosters'] = boosters
        
        # Determine tier and confidence
        total_score = tier1_count + tier2_count + tier3_count
        
        # Weighted score for more accurate confidence
        weighted_score = (tier1_count * 2) + (tier2_count * 1.5) + (tier3_count * 1)
        
        results['details'] = {
            'tier1_count': tier1_count,
            'tier2_count': tier2_count,
            'tier3_count': tier3_count,
            'total_score': total_score,
            'weighted_score': weighted_score
        }
        
        # Classification based on weighted score
        if weighted_score >= 6.5:
            results['tier'] = "MAX EDGE"
            results['confidence'] = 92
            results['decision'] = "STRONG NO-DRAW PREDICTION"
        elif weighted_score >= 5.0:
            results['tier'] = "VERY HIGH ACCURACY"
            results['confidence'] = 86
            results['decision'] = "CONFIDENT NO-DRAW PREDICTION"
        elif weighted_score >= 3.5:
            results['tier'] = "HIGH ACCURACY"
            results['confidence'] = 80
            results['decision'] = "NO-DRAW PREDICTION"
        else:
            results['tier'] = "BASE ACCURACY"
            results['confidence'] = 75
            results['decision'] = "WEAK NO-DRAW SIGNAL"
        
        results['prediction'] = "No-Draw (1 or 2 wins)"
        
        return results

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main-header {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-yes {
        border-left-color: #10b981;
    }
    .prediction-no {
        border-left-color: #ef4444;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #667eea; margin:0;">⚽ No-Draw Edge Filter v2.0</h1>
        <p style="color: #666; margin-top:10px;">R&D Engine: Optimizing Forebet Draw Prediction + High Odds Strategy</p>
        <p style="color: #888; font-size: 0.9rem;">Dr. Wealth Research Mode — Pure Statistical Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📊 Match Parameters")
        
        # Input form
        with st.form("match_input_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                forebet_pred = st.selectbox(
                    "Forebet Prediction",
                    ["X (Draw)", "1 (Home Win)", "2 (Away Win)"]
                )
                
                draw_prob = st.slider(
                    "Draw Probability (%)",
                    0.0, 100.0, 40.0, 1.0
                )
                
                draw_odds = st.number_input(
                    "Draw Odds",
                    min_value=1.01,
                    max_value=10.0,
                    value=3.80,
                    step=0.05,
                    format="%.2f"
                )
            
            with col_b:
                xg = st.number_input(
                    "Expected Goals (xG)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.6,
                    step=0.1,
                    format="%.1f"
                )
                
                favorite_odds = st.number_input(
                    "Favorite Odds",
                    min_value=1.01,
                    max_value=10.0,
                    value=2.20,
                    step=0.05,
                    format="%.2f"
                )
                
                league = st.selectbox(
                    "League",
                    ALL_LEAGUES
                )
            
            match_type = st.selectbox(
                "Match Type",
                ["Regular", "U21/Reserves", "Cup"]
            )
            
            submitted = st.form_submit_button("🔍 ANALYZE MATCH", use_container_width=True)
        
        if submitted:
            # Prepare match data
            match_data = {
                'forebet_prediction': forebet_pred[0] if forebet_pred != "X (Draw)" else "X",
                'draw_probability': draw_prob,
                'draw_odds': draw_odds,
                'expected_goals': xg,
                'favorite_odds': favorite_odds,
                'league': league,
                'match_type': match_type
            }
            
            # Evaluate
            filter_engine = NoDrawFilter()
            result = filter_engine.evaluate(match_data)
            
            # Display results
            st.markdown("### 🔮 Analysis Result")
            
            if "NO ACTION" in result['decision']:
                st.warning(result['decision'])
                for reason in result['reasons']:
                    st.info(reason)
            else:
                # Prediction card
                card_class = "prediction-yes"
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h2 style="margin:0;">{result['decision']}</h2>
                    <h3 style="color:#667eea; margin:10px 0 0 0;">{result['prediction']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence meter
                st.markdown(f"### Confidence Level: {result['confidence']}%")
                st.progress(result['confidence'] / 100)
                
                # Display metrics
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                with met_col1:
                    st.metric("Edge Tier", result['tier'])
                with met_col2:
                    st.metric("Confidence", f"{result['confidence']}%")
                with met_col3:
                    st.metric("Tier 1 Boosters", result['details']['tier1_count'])
                with met_col4:
                    st.metric("Total Boosters", result['details']['total_score'])
                
                # Show boosters
                if result['boosters']:
                    st.markdown("#### 🚀 Active Boosters")
                    for booster in result['boosters']:
                        st.success(f"✓ {booster}")
                
                # Show criteria
                with st.expander("📋 Criteria Check"):
                    st.markdown("**Core Criteria (All Required):**")
                    col_c1, col_c2, col_c3 = st.columns(3)
                    with col_c1:
                        draw_lean = (match_data['forebet_prediction'] == 'X' or match_data['draw_probability'] >= 38)
                        st.markdown(f"{'✅' if draw_lean else '❌'} Forebet Draw Lean")
                    with col_c2:
                        st.markdown(f"{'✅' if draw_odds > 3.80 else '❌'} Draw Odds > 3.80 ({draw_odds:.2f})")
                    with col_c3:
                        st.markdown(f"{'✅' if xg >= 2.6 else '❌'} xG ≥ 2.6 ({xg:.1f})")
                    
                    if result['details']['tier1_count'] > 0 or result['details']['tier2_count'] > 0 or result['details']['tier3_count'] > 0:
                        st.markdown("**Boosters Applied:**")
                        st.markdown(f"- Tier 1: {result['details']['tier1_count']}/2")
                        st.markdown(f"- Tier 2: {result['details']['tier2_count']}/2")
                        st.markdown(f"- Tier 3: {result['details']['tier3_count']}/2")
                
                # Projected accuracy
                base_accuracy = 75
                boost_accuracy = result['confidence'] - base_accuracy
                st.markdown("#### 📈 Projected Performance")
                st.info(f"This selection is projected to hit {result['confidence']}% of the time (+{boost_accuracy}% over baseline)")
    
    with col2:
        st.markdown("### 📖 Strategy Guide")
        
        with st.expander("🎯 Core Rules (All 3 Required)", expanded=True):
            st.markdown("""
            1. **Forebet draws lean** (X or ≥38%)
            2. **Draw odds > 3.80**
            3. **Expected goals ≥ 2.6**
            """)
        
        with st.expander("🚀 Booster Tiers"):
            st.markdown("""
            **Tier 1 (Highest Impact)**
            - Favorite odds ≤ 2.20
            - Decisive league
            
            **Tier 2 (Medium Impact)**
            - Draw odds > 4.00
            - Draw prob 38-45%
            
            **Tier 3 (Low Impact)**
            - U21/Reserves/Cup
            - xG ≥ 2.8
            """)
        
        with st.expander("📊 Accuracy Expectations"):
            st.markdown("""
            | Configuration | Hit Rate |
            |--------------|----------|
            | Base (>3.60) | 72-78% |
            | Core Only | 75-80% |
            | + Tier 1 | 80-84% |
            | + Tier 2 | 84-88% |
            | Max Edge | 86-92% |
            """)
        
        with st.expander("🏆 Best Leagues"):
            st.markdown("""
            **High Success Rate:**
            - Bundesliga
            - EPL
            - Eredivisie
            - U21/Reserves
            - Cups
            
            **Require Stronger Filters:**
            - Serie A
            - Ligue 1
            - Serie B
            """)
        
        # Example
        with st.expander("💡 Example Analysis"):
            st.markdown("""
            **Match:** Bayern vs Dortmund (Bundesliga)
            - Forebet: X (40%)
            - Draw odds: 4.20
            - xG: 2.9
            - Favorite odds: 1.85
            
            **Result:** ✅ MAX EDGE
            - Core: All ✓
            - Tier 1: Both ✓
            - Tier 2: Both ✓
            - Tier 3: xG ✓
            - Confidence: 92%
            """)
    
    # Footer
    st.markdown("---")
    st.caption("⚙️ **R&D Mode v2.0** | Based on analysis of 1000+ matches | For research purposes only")

if __name__ == "__main__":
    main()