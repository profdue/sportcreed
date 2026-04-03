# grokbet_consensus_v2_complete.py
# GROKBET CONSENSUS LOGIC v2.0 (FINAL LOCKED)
# Forebet + SoccerVista | 280+ matches | 76-80% win rate
# Complete working application with all features

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GrokBet Consensus v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        color: #fbbf24;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        background: #fbbf24;
        color: #0f172a;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
        margin-right: 0.5rem;
    }
    
    /* Card styles */
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .result-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    
    /* Verdict styles */
    .verdict-strong {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-skip {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-ou {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a4a 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Utility styles */
    .stat-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    .section-title {
        color: #fbbf24;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .rule-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .stake-highlight {
        background: #fbbf24;
        color: #0f172a;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .filter-pass {
        color: #10b981;
        font-family: monospace;
    }
    .filter-fail {
        color: #ef4444;
        font-family: monospace;
    }
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    
    /* Form input styling */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #0f172a !important;
        color: white !important;
        border-color: #334155 !important;
    }
    label {
        color: #94a3b8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PERFORMANCE DATA (280+ Matches)
# ============================================================================

PERFORMANCE = {
    "total_matches": 280,
    "strong_consensus_rate": 76.8,
    "medium_consensus_rate": 75.0,
    "forebet_only_rate": 68.7,
    "soccervista_only_rate": 72.5
}

# ============================================================================
# LEAGUE REFERENCE (Optional - for context)
# ============================================================================

LEAGUE_TYPES = {
    "Top Tier": ["EPL", "Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1", "Eredivisie", "Primeira Liga", "MLS", "A-League"],
    "Lower Divisions": ["Championship", "League One", "League Two", "Serie B", "Ligue 2", "2. Bundesliga", "Scottish Championship"],
    "Youth/Other": ["U19", "U21", "U23", "Youth", "Academy", "Reserves", "Women"]
}

# ============================================================================
# GROKBET CONSENSUS V2.0 CLASS
# ============================================================================

class GrokBetConsensusV2:
    """GrokBet Consensus v2.0 - Final Locked Version"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        """Load match history from JSON file"""
        try:
            if os.path.exists("grokbet_consensus_v2.json"):
                with open("grokbet_consensus_v2.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data: Dict, result: str):
        """Save match result for tracking"""
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_consensus_v2.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def analyze_form(self, form_sequence: str) -> Dict:
        """Analyze form sequence (last 6 matches)"""
        if not form_sequence:
            return {"wins": 0, "losses": 0, "draws": 0, "strength": "unknown"}
        
        form_upper = form_sequence.upper()
        wins = form_upper.count('W')
        losses = form_upper.count('L')
        draws = form_upper.count('D')
        
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "strength": "good" if wins >= 3 else "mixed" if wins == 2 else "poor"
        }
    
    def evaluate_1x2(self, match_data: Dict) -> Dict:
        """Evaluate 1X2/12 with v2.0 rules"""
        
        # Forebet inputs
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        
        # SoccerVista inputs
        sv_pred = match_data.get('sv_pred', '')
        sv_odds = match_data.get('sv_odds', 0)
        sv_draw_odds = match_data.get('sv_draw_odds', 0)
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        
        # ================================================================
        # RULE: Home Win (1)
        # ================================================================
        if fb_pred == '1':
            if not (fb_prob >= 48 and fb_coef >= 1.45):
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": f"Forebet filters failed: Prob={fb_prob}% (need ≥48), Coef={fb_coef} (need ≥1.45)",
                    "consensus": "none"
                }
            
            sv_passes = (sv_pred == '1' and sv_odds >= 1.45)
            form_passes = home_form['wins'] >= 3
            
            if sv_passes and form_passes:
                return {
                    "valid": True,
                    "type": "1X2",
                    "bet": "HOME WIN (1)",
                    "consensus": "STRONG",
                    "stake": "0.75-1%",
                    "expected": "76-80%",
                    "reasons": [
                        f"Forebet: Pred 1, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"Form: Home team {home_form['wins']} wins in last 6 (≥3)"
                    ]
                }
            else:
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": "SoccerVista consensus failed",
                    "consensus": "none"
                }
        
        # ================================================================
        # RULE: Away Win (2) with High Goals Stake Reduction
        # ================================================================
        elif fb_pred == '2':
            if not (fb_prob >= 48 and fb_coef >= 1.45):
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": f"Forebet filters failed: Prob={fb_prob}% (need ≥48), Coef={fb_coef} (need ≥1.45)",
                    "consensus": "none"
                }
            
            sv_passes = (sv_pred == '2' and sv_odds >= 1.45)
            form_passes = away_form['wins'] >= 3
            
            if sv_passes and form_passes:
                # v2.0 refinement: High goals reduce stake for away wins
                if fb_avg_goals >= 3.0:
                    stake = "0.5% max"
                    stake_note = "⚠️ Stake capped at 0.5% due to high goals (≥3.0)"
                else:
                    stake = "0.75-1%"
                    stake_note = ""
                
                return {
                    "valid": True,
                    "type": "1X2",
                    "bet": "AWAY WIN (2)",
                    "consensus": "STRONG",
                    "stake": stake,
                    "expected": "76-80%",
                    "reasons": [
                        f"Forebet: Pred 2, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"Form: Away team {away_form['wins']} wins in last 6 (≥3)",
                        stake_note
                    ] if stake_note else [
                        f"Forebet: Pred 2, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"Form: Away team {away_form['wins']} wins in last 6 (≥3)"
                    ],
                    "stake_capped": fb_avg_goals >= 3.0
                }
            else:
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": "SoccerVista consensus failed",
                    "consensus": "none"
                }
        
        # ================================================================
        # RULE: X (Draw) → Flip to 12
        # ================================================================
        elif fb_pred == 'X':
            if not (fb_prob <= 42 and fb_coef >= 2.80):
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": f"Forebet filters failed: Prob={fb_prob}% (need ≤42), Coef={fb_coef} (need ≥2.80)",
                    "consensus": "none"
                }
            
            sv_passes = (sv_draw_odds >= 2.80)
            form_passes = (home_form['wins'] <= 2 or away_form['wins'] <= 2)
            
            if sv_passes and form_passes:
                return {
                    "valid": True,
                    "type": "1X2",
                    "bet": "DOUBLE CHANCE 12 (No Draw)",
                    "consensus": "STRONG",
                    "stake": "0.75-1%",
                    "expected": "75-79%",
                    "reasons": [
                        f"Forebet: Pred X, Prob {fb_prob}% ≤42%, Coef {fb_coef} ≥2.80",
                        f"SoccerVista: Draw odds {sv_draw_odds} ≥2.80",
                        f"Form: Mixed form (Home {home_form['wins']} wins, Away {away_form['wins']} wins)"
                    ]
                }
            else:
                return {
                    "valid": False,
                    "type": "1X2",
                    "reason": "SoccerVista consensus failed for X flip",
                    "consensus": "none"
                }
        
        else:
            return {
                "valid": False,
                "type": "1X2",
                "reason": f"Invalid Forebet prediction: {fb_pred}",
                "consensus": "none"
            }
    
    def evaluate_ou(self, match_data: Dict) -> Dict:
        """Evaluate Over/Under 2.5 with v2.0 rules"""
        
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        fb_correct_score_total = match_data.get('fb_correct_score_total', 0)
        sv_ou_pred = match_data.get('sv_ou_pred', '')
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        total_wins = home_form['wins'] + away_form['wins']
        
        # Grey zone check
        if 2.3 <= fb_avg_goals <= 2.7:
            return {
                "valid": False,
                "type": "O/U",
                "reason": f"Grey zone: Avg Goals = {fb_avg_goals} (2.3-2.7 range)",
                "consensus": "none"
            }
        
        # OVER 2.5
        if fb_avg_goals >= 2.80 and fb_correct_score_total >= 3:
            if sv_ou_pred == 'O' and total_wins >= 7:
                return {
                    "valid": True,
                    "type": "O/U",
                    "bet": "OVER 2.5 GOALS",
                    "consensus": "STRONG",
                    "stake": "0.5-0.75%",
                    "expected": "72-76%",
                    "reasons": [
                        f"Forebet: Avg Goals {fb_avg_goals} ≥2.80, implied {fb_correct_score_total}+ goals",
                        f"SoccerVista: O prediction",
                        f"Form: Combined {total_wins} wins in last 6 (≥7)"
                    ]
                }
            elif sv_ou_pred == 'O':
                return {
                    "valid": True,
                    "type": "O/U",
                    "bet": "OVER 2.5 GOALS",
                    "consensus": "MEDIUM",
                    "stake": "0.25-0.5%",
                    "expected": "68-72%",
                    "reasons": [
                        f"Forebet: Avg Goals {fb_avg_goals} ≥2.80, implied {fb_correct_score_total}+ goals",
                        f"SoccerVista: O prediction",
                        f"⚠️ Form condition not fully met (combined {total_wins} wins, need ≥7)"
                    ]
                }
        
        # UNDER 2.5
        elif fb_avg_goals <= 2.20 and fb_correct_score_total <= 2:
            if sv_ou_pred == 'U' and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                return {
                    "valid": True,
                    "type": "O/U",
                    "bet": "UNDER 2.5 GOALS",
                    "consensus": "STRONG",
                    "stake": "0.5-0.75%",
                    "expected": "68-72%",
                    "reasons": [
                        f"Forebet: Avg Goals {fb_avg_goals} ≤2.20, implied {fb_correct_score_total} or fewer goals",
                        f"SoccerVista: U prediction",
                        f"Form: At least one team has ≤2 wins in last 6"
                    ]
                }
            elif sv_ou_pred == 'U':
                return {
                    "valid": True,
                    "type": "O/U",
                    "bet": "UNDER 2.5 GOALS",
                    "consensus": "MEDIUM",
                    "stake": "0.25-0.5%",
                    "expected": "64-68%",
                    "reasons": [
                        f"Forebet: Avg Goals {fb_avg_goals} ≤2.20, implied {fb_correct_score_total} or fewer goals",
                        f"SoccerVista: U prediction",
                        f"⚠️ Form condition not fully met"
                    ]
                }
        
        return {
            "valid": False,
            "type": "O/U",
            "reason": f"No clear consensus: Avg Goals={fb_avg_goals}, SV Pred={sv_ou_pred}",
            "consensus": "none"
        }
    
    def get_stats(self) -> Optional[Dict]:
        """Return statistics from match history"""
        if not self.match_history:
            return None
        
        total = len(self.match_history)
        correct = sum(1 for m in self.match_history if m.get('actual_result') == 'Win')
        
        return {
            "total": total,
            "correct": correct,
            "win_rate": (correct / total * 100) if total > 0 else 0
        }
    
    def export_history(self) -> pd.DataFrame:
        """Export match history as DataFrame"""
        if not self.match_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.match_history)
        return df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Consensus Logic v2.0</h1>
        <p>Final Locked Version | Forebet + SoccerVista | 280+ Matches Analyzed</p>
        <div>
            <span class="badge">🏆 Strong Consensus: 76.8%</span>
            <span class="badge">⚡ Away Win + High Goals: 0.5% max stake</span>
            <span class="badge">📊 Max 4 bets per day</span>
            <span class="badge">💰 0.5-1% bankroll per bet</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize
    consensus = GrokBetConsensusV2()
    stats = consensus.get_stats()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 1X2 / 12 SYSTEM", "⚽ OVER/UNDER 2.5", "📊 RULES & REFERENCE", "📜 HISTORY & STATS"])
    
    # ============================================================================
    # TAB 1: 1X2 / 12 SYSTEM
    # ============================================================================
    with tab1:
        st.markdown('<div class="section-title">📊 FOREBET + SOCCERVISTA DATA</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            # Team names
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.text_input("Home Team Name", "Macarthur FC", key="home_team")
            with col2:
                away_team = st.text_input("Away Team Name", "Newcastle Jets", key="away_team")
            
            st.markdown("---")
            st.markdown("##### Forebet Inputs")
            
            # Forebet inputs row
            col3, col4, col5, col6 = st.columns(4)
            with col3:
                fb_pred = st.selectbox("Pred", ["1", "X", "2"], key="fb_pred_1x2")
            with col4:
                fb_prob = st.number_input("Prob %", 0.0, 100.0, 51.0, 1.0, key="fb_prob_1x2")
            with col5:
                fb_coef = st.number_input("Coef.", 0.0, 10.0, 2.50, 0.05, format="%.2f", key="fb_coef_1x2")
            with col6:
                fb_avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 3.20, 0.05, format="%.2f", key="fb_avg_1x2")
            
            st.markdown("##### SoccerVista Inputs")
            
            # SoccerVista inputs row
            col7, col8, col9 = st.columns(3)
            with col7:
                sv_pred = st.selectbox("SV Pred", ["1", "X", "2"], key="sv_pred_1x2")
            with col8:
                sv_odds = st.number_input("Odds", 0.0, 10.0, 2.15, 0.05, format="%.2f", key="sv_odds_1x2")
            with col9:
                sv_draw_odds = st.number_input("Draw Odds", 0.0, 10.0, 3.80, 0.05, format="%.2f", key="sv_draw_1x2")
            
            st.markdown("##### Form (last 6 matches - use W/L/D)")
            
            col10, col11 = st.columns(2)
            with col10:
                form_home = st.text_input("Home Team Form", "WLLLL", key="form_home_1x2", help="Example: WWDLWW (W=Win, L=Loss, D=Draw)")
            with col11:
                form_away = st.text_input("Away Team Form", "WLWDW", key="form_away_1x2", help="Example: LLDWLL")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button
            analyze_1x2 = st.button("🔍 ANALYZE 1X2 CONSENSUS", use_container_width=True, type="primary")
            
            if analyze_1x2:
                match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'fb_pred': fb_pred,
                    'fb_prob': fb_prob,
                    'fb_coef': fb_coef,
                    'fb_avg_goals': fb_avg_goals,
                    'sv_pred': sv_pred,
                    'sv_odds': sv_odds,
                    'sv_draw_odds': sv_draw_odds,
                    'sv_form_home': form_home.upper(),
                    'sv_form_away': form_away.upper()
                }
                
                result = consensus.evaluate_1x2(match_data)
                
                st.markdown("---")
                st.markdown("### 🔮 ANALYSIS RESULT")
                
                if result['valid']:
                    # Display match info
                    st.markdown(f"""
                    <div style="background: #0f172a; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                        <strong>🏟️ {home_team} vs {away_team}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="verdict-strong">
                        <h2 style="margin: 0; color: #fbbf24;">🏆 STRONG CONSENSUS</h2>
                        <p style="margin: 0.5rem 0; font-size: 1.2rem;">🎯 {result['bet']}</p>
                        <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result['stake']}</span></p>
                        <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for reason in result['reasons']:
                        if reason and ("⚠️" in reason or "capped" in reason.lower()):
                            st.warning(reason)
                        elif reason:
                            st.success(f"✓ {reason}")
                    
                    # Save buttons
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        if st.button("✅ Save as WIN", use_container_width=True, key="save_win_1x2"):
                            consensus.save_match(match_data, "Win")
                            st.success("Saved as WIN!")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS", use_container_width=True, key="save_loss_1x2"):
                            consensus.save_match(match_data, "Loss")
                            st.warning("Saved as LOSS!")
                            st.rerun()
                    with col_s3:
                        if st.button("📝 Save as DRAW (No Bet)", use_container_width=True, key="save_draw_1x2"):
                            consensus.save_match(match_data, "Draw")
                            st.info("Saved as DRAW!")
                            st.rerun()
                else:
                    st.markdown(f"""
                    <div class="verdict-skip">
                        <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                        <p style="margin: 0.5rem 0;">{result.get('reason', 'No consensus')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 2: OVER/UNDER 2.5 SYSTEM
    # ============================================================================
    with tab2:
        st.markdown('<div class="section-title">📊 FOREBET + SOCCERVISTA DATA</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            # Team names
            col1, col2 = st.columns(2)
            with col1:
                home_team_ou = st.text_input("Home Team Name", "Macarthur FC", key="home_team_ou")
            with col2:
                away_team_ou = st.text_input("Away Team Name", "Newcastle Jets", key="away_team_ou")
            
            st.markdown("---")
            st.markdown("##### Forebet Inputs")
            
            col3, col4 = st.columns(2)
            with col3:
                fb_avg_goals_ou = st.number_input("Avg Goals", 0.0, 5.0, 3.20, 0.05, format="%.2f", key="fb_avg_ou")
            with col4:
                fb_correct_score_total = st.number_input("Implied Goals from Correct Score", 0, 10, 4, key="fb_implied_ou")
            
            st.markdown("##### SoccerVista Inputs")
            
            col5, col6 = st.columns(2)
            with col5:
                sv_ou_pred = st.selectbox("O/U Prediction", ["O", "U", "-"], key="sv_ou_pred")
            with col6:
                st.markdown(" ")
            
            st.markdown("##### Form (last 6 matches - use W/L/D)")
            
            col7, col8 = st.columns(2)
            with col7:
                form_home_ou = st.text_input("Home Team Form", "WLLLL", key="form_home_ou")
            with col8:
                form_away_ou = st.text_input("Away Team Form", "WLWDW", key="form_away_ou")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Analyze button
            analyze_ou = st.button("🔍 ANALYZE O/U CONSENSUS", use_container_width=True, type="primary")
            
            if analyze_ou:
                match_data = {
                    'home_team': home_team_ou,
                    'away_team': away_team_ou,
                    'fb_avg_goals': fb_avg_goals_ou,
                    'fb_correct_score_total': fb_correct_score_total,
                    'sv_ou_pred': sv_ou_pred,
                    'sv_form_home': form_home_ou.upper(),
                    'sv_form_away': form_away_ou.upper()
                }
                
                result = consensus.evaluate_ou(match_data)
                
                st.markdown("---")
                st.markdown("### 🔮 ANALYSIS RESULT")
                
                if result['valid']:
                    st.markdown(f"""
                    <div class="verdict-ou">
                        <h2 style="margin: 0; color: #10b981;">{'🏆 STRONG' if result['consensus'] == 'STRONG' else '⚠️ MEDIUM'} CONSENSUS</h2>
                        <p style="margin: 0.5rem 0; font-size: 1.2rem;">⚽ {result['bet']}</p>
                        <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result['stake']}</span></p>
                        <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for reason in result['reasons']:
                        if "⚠️" in reason:
                            st.warning(reason)
                        else:
                            st.success(f"✓ {reason}")
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    with col_s1:
                        if st.button("✅ Save as WIN", use_container_width=True, key="save_win_ou"):
                            consensus.save_match(match_data, "Win")
                            st.success("Saved as WIN!")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS", use_container_width=True, key="save_loss_ou"):
                            consensus.save_match(match_data, "Loss")
                            st.warning("Saved as LOSS!")
                            st.rerun()
                    with col_s3:
                        if st.button("📝 Save as DRAW (No Bet)", use_container_width=True, key="save_draw_ou"):
                            consensus.save_match(match_data, "Draw")
                            st.info("Saved as DRAW!")
                            st.rerun()
                else:
                    st.markdown(f"""
                    <div class="verdict-skip">
                        <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                        <p style="margin: 0.5rem 0;">{result.get('reason', 'No clear consensus')}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 3: RULES & REFERENCE
    # ============================================================================
    with tab3:
        st.markdown('<div class="section-title">📋 GROKBET CONSENSUS v2.0 - COMPLETE RULES</div>', unsafe_allow_html=True)
        
        # 1X2 Rules
        st.markdown("""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">🎯 1X2 / 12 SYSTEM (Primary)</h3>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #10b981;">HOME WIN (1)</strong><br>
                <span class="filter-pass">✓ Forebet: Prob1 ≥48%, Coef ≥1.45</span><br>
                <span class="filter-pass">✓ SoccerVista: Odds1 ≥1.45, Home ≥3 wins in last 6</span><br>
                <span style="color: #94a3b8;">→ Stake: 0.75-1%</span>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #10b981;">AWAY WIN (2)</strong><br>
                <span class="filter-pass">✓ Forebet: Prob2 ≥48%, Coef ≥1.45</span><br>
                <span class="filter-pass">✓ SoccerVista: Odds2 ≥1.45, Away ≥3 wins in last 6</span><br>
                <span class="filter-pass">⚠️ If Avg Goals ≥3.0 → Stake capped at 0.5%</span>
                <span style="color: #94a3b8;">→ Stake: 0.75-1% (0.5% if high goals)</span>
            </div>
            
            <div>
                <strong style="color: #10b981;">X → 12 (NO DRAW)</strong><br>
                <span class="filter-pass">✓ Forebet: ProbX ≤42%, Coef ≥2.80</span><br>
                <span class="filter-pass">✓ SoccerVista: Draw odds ≥2.80, Mixed form (≤2 wins)</span><br>
                <span style="color: #94a3b8;">→ Stake: 0.75-1%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # O/U Rules
        st.markdown("""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">⚽ OVER/UNDER 2.5 SYSTEM (Secondary)</h3>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #10b981;">OVER 2.5 GOALS</strong><br>
                <span class="filter-pass">✓ Forebet: Avg Goals ≥2.80, implied ≥3 goals</span><br>
                <span class="filter-pass">✓ SoccerVista: "O" prediction</span><br>
                <span class="filter-pass">✓ Form: Combined wins ≥7 in last 6</span>
                <span style="color: #94a3b8;">→ Stake: 0.5-0.75%</span>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <strong style="color: #10b981;">UNDER 2.5 GOALS</strong><br>
                <span class="filter-pass">✓ Forebet: Avg Goals ≤2.20, implied ≤2 goals</span><br>
                <span class="filter-pass">✓ SoccerVista: "U" prediction</span><br>
                <span class="filter-pass">✓ Form: One team has ≤2 wins in last 6</span>
                <span style="color: #94a3b8;">→ Stake: 0.5-0.75%</span>
            </div>
            
            <div>
                <strong style="color: #f59e0b;">GREY ZONE (SKIP)</strong><br>
                <span>⚠️ Avg Goals 2.3-2.7 → Always skip</span><br>
                <span>⚠️ Weak alignment between avg goals and correct score → Skip</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Bankroll Rules
        st.markdown("""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">💰 BANKROLL RULES (v2.0 Locked)</h3>
            
            <div style="margin-bottom: 0.5rem;">🏆 <strong>Strong Consensus (1X2/12):</strong> 0.75-1% of bankroll</div>
            <div style="margin-bottom: 0.5rem;">⚠️ <strong>Away Win + High Goals (≥3.0):</strong> 0.5% max</div>
            <div style="margin-bottom: 0.5rem;">📊 <strong>Over/Under 2.5:</strong> 0.5-0.75%</div>
            <div style="margin-bottom: 0.5rem;">🎯 <strong>Maximum bets per day:</strong> 4</div>
            <div>⚡ <strong>Maximum daily exposure:</strong> 2% of bankroll</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Form Guide
        st.markdown("""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">📝 FORM GUIDE (last 6 matches)</h3>
            
            <div style="margin-bottom: 0.5rem;"><strong>W</strong> = Win (3 points)</div>
            <div style="margin-bottom: 0.5rem;"><strong>D</strong> = Draw (1 point)</div>
            <div style="margin-bottom: 0.5rem;"><strong>L</strong> = Loss (0 points)</div>
            <div style="margin-bottom: 0.5rem;">Example: <strong>WWDLWW</strong> = 4 wins, 1 draw, 1 loss</div>
            <div><strong>Required:</strong> ≥3 wins for 1/2 bets, ≤2 wins for mixed form check</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance Stats
        st.markdown(f"""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">📊 PERFORMANCE (280+ MATCHES)</h3>
            
            <div style="margin-bottom: 0.5rem;">🏆 <strong>Strong Consensus:</strong> {PERFORMANCE['strong_consensus_rate']}%</div>
            <div style="margin-bottom: 0.5rem;">📈 <strong>Medium Consensus:</strong> {PERFORMANCE['medium_consensus_rate']}%</div>
            <div style="margin-bottom: 0.5rem;">📊 <strong>Forebet Only:</strong> {PERFORMANCE['forebet_only_rate']}%</div>
            <div>⭐ <strong>SoccerVista Only:</strong> {PERFORMANCE['soccervista_only_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Checklist
        st.markdown("""
        <div class="rule-card">
            <h3 style="color: #fbbf24; margin: 0 0 0.5rem 0;">⚡ 3-SECOND CHECKLIST</h3>
            
            <div style="margin-bottom: 0.5rem;">☐ <strong>Forebet = X/1/2</strong></div>
            <div style="margin-bottom: 0.5rem;">☐ <strong>Forebet: Prob ≥48% (1/2) OR ≤42% (X)</strong></div>
            <div style="margin-bottom: 0.5rem;">☐ <strong>Forebet: Coef ≥1.45 (1/2) OR ≥2.80 (X)</strong></div>
            <div style="margin-bottom: 0.5rem;">☐ <strong>SoccerVista: Same pred OR draw odds ≥2.80</strong></div>
            <div style="margin-bottom: 0.5rem;">☐ <strong>Form: ≥3 wins (1/2) OR mixed form (X)</strong></div>
            <div style="margin-bottom: 0.5rem;">☐ <strong>If Away Win + Avg Goals ≥3.0 → Stake 0.5% max</strong></div>
            <div style="color: #fbbf24; font-weight: bold; margin-top: 0.5rem;">→ ALL PASS = BET | ANY FAIL = SKIP</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 4: HISTORY & STATS
    # ============================================================================
    with tab4:
        if stats:
            st.markdown('<div class="section-title">📊 YOUR PERFORMANCE SUMMARY</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bets", stats['total'])
            with col2:
                st.metric("Wins", stats['correct'])
            with col3:
                st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
            
            st.markdown("---")
            
            # Export button
            df = consensus.export_history()
            if not df.empty:
                st.markdown('<div class="section-title">📜 BETTING HISTORY</div>', unsafe_allow_html=True)
                
                # Display recent matches
                st.dataframe(
                    df[['timestamp', 'home_team', 'away_team', 'fb_pred', 'actual_result']].sort_values('timestamp', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download History (CSV)",
                    data=csv,
                    file_name=f"grokbet_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No betting history yet. Save some bets to see your stats here!")
        else:
            st.info("No betting history yet. Start by analyzing and saving bets from the 1X2 or O/U tabs!")
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet Consensus v2.0** | Final Locked Version | 280+ matches | 76-80% win rate | Built from your data")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
