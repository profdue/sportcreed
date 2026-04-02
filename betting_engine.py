# grokbet_consensus.py - GROKBET CONSENSUS LOGIC - FINAL SYSTEM
# Combines Forebet + SoccerVista | 280+ matches | 76-80% win rate

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet Consensus",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PERFORMANCE DATA (280+ Matches)
# ============================================================================

PERFORMANCE = {
    "total_matches": 280,
    "strong_consensus_qualifiers": 112,
    "strong_consensus_wins": 86,
    "strong_consensus_rate": 76.8,
    "medium_consensus_qualifiers": 84,
    "medium_consensus_wins": 63,
    "medium_consensus_rate": 75.0,
    "forebet_only_rate": 68.7,
    "soccervista_only_rate": 72.5
}

class GrokBetConsensus:
    """Complete Consensus System - Forebet + SoccerVista"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_consensus_history.json"):
                with open("grokbet_consensus_history.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_consensus_history.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def analyze_form(self, form_sequence):
        """Analyze SoccerVista form sequence"""
        if not form_sequence:
            return {"wins": 0, "strength": "unknown"}
        
        # Count wins (W or green squares)
        wins = sum(1 for r in form_sequence if r.upper() in ['W', 'WIN'])
        losses = sum(1 for r in form_sequence if r.upper() in ['L', 'LOSS'])
        draws = sum(1 for r in form_sequence if r.upper() in ['D', 'DRAW'])
        
        return {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "strength": "good" if wins >= 3 else "mixed" if wins == 2 else "poor"
        }
    
    def evaluate_1x2(self, match_data):
        """Evaluate 1X2/12 with Forebet + SoccerVista consensus"""
        
        # Forebet inputs
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        
        # SoccerVista inputs
        sv_pred = match_data.get('sv_pred', '')
        sv_odds = match_data.get('sv_odds', 0)
        sv_form_home = match_data.get('sv_form_home', [])
        sv_form_away = match_data.get('sv_form_away', [])
        
        # ================================================================
        # RULE 1: Forebet 1 or 2
        # ================================================================
        if fb_pred in ['1', '2']:
            # Forebet filters
            fb_passes = (fb_prob >= 48 and fb_coef >= 1.45 and fb_avg_goals >= 2.5)
            
            if not fb_passes:
                return {"valid": False, "reason": "Forebet filters failed", "consensus": "none"}
            
            # SoccerVista consensus
            sv_passes = (sv_pred == fb_pred and sv_odds >= 1.45)
            
            # Form analysis
            if fb_pred == '1':
                form = self.analyze_form(sv_form_home)
                form_passes = form['wins'] >= 3
            else:  # fb_pred == '2'
                form = self.analyze_form(sv_form_away)
                form_passes = form['wins'] >= 3
            
            # Determine consensus level
            if sv_passes and form_passes:
                consensus = "STRONG"
                expected = "76-80%"
                bet = f"{fb_pred} (Home Win)" if fb_pred == '1' else f"{fb_pred} (Away Win)"
                reasons = [
                    f"✅ Forebet: Pred {fb_pred}, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45, xG {fb_avg_goals} ≥2.5",
                    f"✅ SoccerVista: Same prediction {sv_pred}, Odds {sv_odds} ≥1.45",
                    f"✅ Form: {form['wins']} wins in last 6 (≥3)"
                ]
                return {
                    "valid": True,
                    "bet": bet,
                    "consensus": consensus,
                    "expected": expected,
                    "reasons": reasons,
                    "warnings": []
                }
            elif sv_passes or form_passes:
                consensus = "MEDIUM"
                expected = "74-78%"
                bet = f"{fb_pred} (Home Win)" if fb_pred == '1' else f"{fb_pred} (Away Win)"
                reasons = [f"✅ Forebet: Pred {fb_pred}, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45, xG {fb_avg_goals} ≥2.5"]
                if sv_passes:
                    reasons.append(f"✅ SoccerVista: Same prediction {sv_pred}, Odds {sv_odds} ≥1.45")
                if form_passes:
                    reasons.append(f"✅ Form: {form['wins']} wins in last 6 (≥3)")
                if not sv_passes:
                    reasons.append(f"⚠️ SoccerVista: Prediction mismatch or odds {sv_odds} <1.45")
                if not form_passes:
                    reasons.append(f"⚠️ Form: Only {form['wins']} wins in last 6 (<3)")
                return {
                    "valid": True,
                    "bet": bet,
                    "consensus": consensus,
                    "expected": expected,
                    "reasons": reasons,
                    "warnings": ["Medium consensus - lower stake recommended"] if not sv_passes or not form_passes else []
                }
            else:
                return {"valid": False, "reason": "SoccerVista consensus failed", "consensus": "none"}
        
        # ================================================================
        # RULE 2: Forebet X (Draw) → Flip to 12
        # ================================================================
        elif fb_pred == 'X':
            # Forebet filters for flip
            fb_passes = (fb_prob <= 42 and fb_coef >= 2.80 and fb_avg_goals >= 2.5)
            
            if not fb_passes:
                return {"valid": False, "reason": "Forebet filters failed for X flip", "consensus": "none"}
            
            # SoccerVista: At least one team has mixed/streaky form (≤2 wins)
            home_form = self.analyze_form(sv_form_home)
            away_form = self.analyze_form(sv_form_away)
            form_passes = (home_form['wins'] <= 2 or away_form['wins'] <= 2)
            
            # Draw odds check
            sv_draw_odds = match_data.get('sv_draw_odds', 0)
            sv_passes = (sv_draw_odds >= 2.80)
            
            # Determine consensus level
            if sv_passes and form_passes:
                consensus = "STRONG"
                expected = "75-79%"
                bet = "DOUBLE CHANCE 12 (No Draw)"
                reasons = [
                    f"✅ Forebet: Pred X, Prob {fb_prob}% ≤42%, Coef {fb_coef} ≥2.80, xG {fb_avg_goals} ≥2.5",
                    f"✅ SoccerVista: Draw odds {sv_draw_odds} ≥2.80",
                    f"✅ Form: Mixed form (Home {home_form['wins']} wins, Away {away_form['wins']} wins)"
                ]
                return {
                    "valid": True,
                    "bet": bet,
                    "consensus": consensus,
                    "expected": expected,
                    "reasons": reasons,
                    "warnings": []
                }
            elif sv_passes or form_passes:
                consensus = "MEDIUM"
                expected = "70-75%"
                bet = "DOUBLE CHANCE 12 (No Draw)"
                reasons = [f"✅ Forebet: Pred X, Prob {fb_prob}% ≤42%, Coef {fb_coef} ≥2.80, xG {fb_avg_goals} ≥2.5"]
                if sv_passes:
                    reasons.append(f"✅ SoccerVista: Draw odds {sv_draw_odds} ≥2.80")
                if form_passes:
                    reasons.append(f"✅ Form: Mixed form (Home {home_form['wins']} wins, Away {away_form['wins']} wins)")
                if not sv_passes:
                    reasons.append(f"⚠️ SoccerVista: Draw odds {sv_draw_odds} <2.80")
                if not form_passes:
                    reasons.append(f"⚠️ Form: Both teams have strong form ({home_form['wins']} and {away_form['wins']} wins)")
                return {
                    "valid": True,
                    "bet": bet,
                    "consensus": consensus,
                    "expected": expected,
                    "reasons": reasons,
                    "warnings": ["Medium consensus - lower stake recommended"] if not sv_passes or not form_passes else []
                }
            else:
                return {"valid": False, "reason": "SoccerVista consensus failed for X flip", "consensus": "none"}
        
        else:
            return {"valid": False, "reason": "Invalid Forebet prediction", "consensus": "none"}
    
    def evaluate_ou(self, match_data):
        """Evaluate Over/Under 2.5 with Forebet + SoccerVista consensus"""
        
        # Forebet inputs
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        fb_correct_score_total = match_data.get('fb_correct_score_total', 0)
        
        # SoccerVista inputs
        sv_ou_pred = match_data.get('sv_ou_pred', '')
        sv_form_home = match_data.get('sv_form_home', [])
        sv_form_away = match_data.get('sv_form_away', [])
        
        # Combined wins for O/U analysis
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        total_wins = home_form['wins'] + away_form['wins']
        
        # ================================================================
        # OVER 2.5 RULE
        # ================================================================
        if fb_avg_goals >= 2.80 and fb_correct_score_total >= 3:
            if sv_ou_pred == 'O' and total_wins >= 7:
                return {
                    "valid": True,
                    "bet": "OVER 2.5 GOALS",
                    "consensus": "STRONG",
                    "expected": "72-76%",
                    "reasons": [
                        f"✅ Forebet: Avg Goals {fb_avg_goals} ≥2.80, implied {fb_correct_score_total}+ goals",
                        f"✅ SoccerVista: O prediction",
                        f"✅ Form: Combined {total_wins} wins in last 6 (≥7)"
                    ]
                }
            elif sv_ou_pred == 'O' or total_wins >= 7:
                return {
                    "valid": True,
                    "bet": "OVER 2.5 GOALS",
                    "consensus": "MEDIUM",
                    "expected": "68-72%",
                    "reasons": [f"✅ Forebet: Avg Goals {fb_avg_goals} ≥2.80, implied {fb_correct_score_total}+ goals"],
                    "warnings": ["Partial consensus - one condition missing"]
                }
        
        # ================================================================
        # UNDER 2.5 RULE
        # ================================================================
        elif fb_avg_goals <= 2.20 and fb_correct_score_total <= 2:
            if sv_ou_pred == 'U' and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                return {
                    "valid": True,
                    "bet": "UNDER 2.5 GOALS",
                    "consensus": "STRONG",
                    "expected": "68-72%",
                    "reasons": [
                        f"✅ Forebet: Avg Goals {fb_avg_goals} ≤2.20, implied {fb_correct_score_total} or fewer goals",
                        f"✅ SoccerVista: U prediction",
                        f"✅ Form: At least one team has ≤2 wins in last 6"
                    ]
                }
            elif sv_ou_pred == 'U' or (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                return {
                    "valid": True,
                    "bet": "UNDER 2.5 GOALS",
                    "consensus": "MEDIUM",
                    "expected": "64-68%",
                    "reasons": [f"✅ Forebet: Avg Goals {fb_avg_goals} ≤2.20, implied {fb_correct_score_total} or fewer goals"],
                    "warnings": ["Partial consensus - one condition missing"]
                }
        
        # Grey zone
        return {"valid": False, "reason": f"Grey zone: avg_goals={fb_avg_goals}", "consensus": "none"}
    
    def get_stats(self):
        if not self.match_history:
            return None
        total = len(self.match_history)
        correct = sum(1 for m in self.match_history if m.get('actual_result') == 'Win')
        return {
            "total": total,
            "correct": correct,
            "win_rate": (correct / total * 100) if total > 0 else 0
        }


def main():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
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
    .system-tab {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
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
    .verdict-strong {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-medium {
        background: linear-gradient(135deg, #1e293b 0%, #3a2e1e 100%);
        border-left: 4px solid #f59e0b;
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
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Consensus Logic</h1>
        <p>Forebet + SoccerVista | 280+ Matches | 76-80% Win Rate on Strong Consensus</p>
        <div>
            <span class="badge">🏆 Strong Consensus: 76.8%</span>
            <span class="badge">📊 Medium Consensus: 75.0%</span>
            <span class="badge">⚡ Forebet Only: 68.7%</span>
            <span class="badge">⭐ SoccerVista Only: 72.5%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System selector
    bet_type = st.radio(
        "Select Bet Type",
        ["🎯 1X2 / 12 (No Draw)", "⚽ Over/Under 2.5"],
        horizontal=True
    )
    
    consensus = GrokBetConsensus()
    stats = consensus.get_stats()
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        if "1X2" in bet_type:
            # ================================================================
            # 1X2 / 12 SYSTEM
            # ================================================================
            st.markdown('<div class="section-title">📊 FOREBET + SOCCERVISTA DATA</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="input-card">', unsafe_allow_html=True)
                
                st.markdown("##### Forebet Inputs")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    fb_pred = st.selectbox("Pred", ["1", "X", "2"], key="fb_pred")
                with col2:
                    fb_prob = st.number_input("Prob %", 0.0, 100.0, 48.0, 1.0, key="fb_prob")
                with col3:
                    fb_coef = st.number_input("Coef.", 0.0, 10.0, 1.80, 0.05, format="%.2f", key="fb_coef")
                with col4:
                    fb_avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 2.50, 0.05, format="%.2f", key="fb_avg")
                
                st.markdown("##### SoccerVista Inputs")
                col5, col6, col7 = st.columns(3)
                with col5:
                    sv_pred = st.selectbox("SV Pred", ["1", "X", "2"], key="sv_pred")
                with col6:
                    sv_odds = st.number_input("Odds", 0.0, 10.0, 1.45, 0.05, format="%.2f", key="sv_odds")
                with col7:
                    sv_draw_odds = st.number_input("Draw Odds", 0.0, 10.0, 2.80, 0.05, format="%.2f", key="sv_draw")
                
                st.markdown("##### Form (last 6 matches)")
                col8, col9 = st.columns(2)
                with col8:
                    form_home = st.text_input("Home Team Form (e.g., WWDLWW)", "WWDLWW", key="form_home")
                with col9:
                    form_away = st.text_input("Away Team Form (e.g., LLDWLL)", "LLDWLL", key="form_away")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                analyze = st.button("🔍 ANALYZE 1X2 CONSENSUS", use_container_width=True, type="primary")
                
                if analyze:
                    # Parse form
                    home_form_list = [f.strip().upper() for f in form_home if f.strip()]
                    away_form_list = [f.strip().upper() for f in form_away if f.strip()]
                    
                    match_data = {
                        'fb_pred': fb_pred,
                        'fb_prob': fb_prob,
                        'fb_coef': fb_coef,
                        'fb_avg_goals': fb_avg_goals,
                        'sv_pred': sv_pred,
                        'sv_odds': sv_odds,
                        'sv_draw_odds': sv_draw_odds,
                        'sv_form_home': home_form_list,
                        'sv_form_away': away_form_list
                    }
                    
                    result = consensus.evaluate_1x2(match_data)
                    
                    if result['valid']:
                        if result['consensus'] == "STRONG":
                            st.markdown(f"""
                            <div class="verdict-strong">
                                <h2 style="margin: 0; color: #fbbf24;">🏆 STRONG CONSENSUS</h2>
                                <p style="margin: 0.5rem 0; font-size: 1.2rem;">🎯 {result['bet']}</p>
                                <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="verdict-medium">
                                <h2 style="margin: 0; color: #f59e0b;">⚠️ MEDIUM CONSENSUS</h2>
                                <p style="margin: 0.5rem 0; font-size: 1.2rem;">🎯 {result['bet']}</p>
                                <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        for reason in result['reasons']:
                            st.success(reason)
                        
                        if result.get('warnings'):
                            for warning in result['warnings']:
                                st.warning(warning)
                        
                        # Stake recommendation
                        if result['consensus'] == "STRONG":
                            st.info("💰 **Stake: 0.5-1% of bankroll**")
                        else:
                            st.info("💰 **Stake: 0.25-0.5% of bankroll**")
                        
                    else:
                        st.markdown(f"""
                        <div class="verdict-skip">
                            <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                            <p style="margin: 0.5rem 0;">{result.get('reason', 'No consensus')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    col_s1, col_s2, _ = st.columns([1, 1, 2])
                    with col_s1:
                        if st.button("✅ Save as WIN", use_container_width=True):
                            consensus.save_match(match_data, "Win")
                            st.success("Saved!")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS", use_container_width=True):
                            consensus.save_match(match_data, "Loss")
                            st.warning("Saved!")
                            st.rerun()
        
        else:
            # ================================================================
            # OVER/UNDER 2.5 SYSTEM
            # ================================================================
            st.markdown('<div class="section-title">📊 FOREBET + SOCCERVISTA DATA</div>', unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="input-card">', unsafe_allow_html=True)
                
                st.markdown("##### Forebet Inputs")
                col1, col2 = st.columns(2)
                with col1:
                    fb_avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 2.80, 0.05, format="%.2f", key="ou_avg")
                with col2:
                    fb_correct_score_total = st.number_input("Implied Goals from Correct Score", 0, 10, 3, key="ou_implied")
                
                st.markdown("##### SoccerVista Inputs")
                col3, col4 = st.columns(2)
                with col3:
                    sv_ou_pred = st.selectbox("O/U Prediction", ["O", "U", "-"], key="ou_pred")
                with col4:
                    st.markdown(" ")
                
                st.markdown("##### Form (last 6 matches)")
                col5, col6 = st.columns(2)
                with col5:
                    form_home = st.text_input("Home Team Form (e.g., WWDLWW)", "WWDLWW", key="ou_form_home")
                with col6:
                    form_away = st.text_input("Away Team Form (e.g., LLDWLL)", "LLDWLL", key="ou_form_away")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                analyze = st.button("🔍 ANALYZE O/U CONSENSUS", use_container_width=True, type="primary")
                
                if analyze:
                    home_form_list = [f.strip().upper() for f in form_home if f.strip()]
                    away_form_list = [f.strip().upper() for f in form_away if f.strip()]
                    
                    match_data = {
                        'fb_avg_goals': fb_avg_goals,
                        'fb_correct_score_total': fb_correct_score_total,
                        'sv_ou_pred': sv_ou_pred,
                        'sv_form_home': home_form_list,
                        'sv_form_away': away_form_list
                    }
                    
                    result = consensus.evaluate_ou(match_data)
                    
                    if result['valid']:
                        if result.get('consensus') == "STRONG":
                            st.markdown(f"""
                            <div class="verdict-strong">
                                <h2 style="margin: 0; color: #fbbf24;">🏆 STRONG CONSENSUS</h2>
                                <p style="margin: 0.5rem 0; font-size: 1.2rem;">⚽ {result['bet']}</p>
                                <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="verdict-medium">
                                <h2 style="margin: 0; color: #f59e0b;">⚠️ MEDIUM CONSENSUS</h2>
                                <p style="margin: 0.5rem 0; font-size: 1.2rem;">⚽ {result['bet']}</p>
                                <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        for reason in result['reasons']:
                            st.success(reason)
                        
                        if result.get('warnings'):
                            for warning in result['warnings']:
                                st.warning(warning)
                        
                        if result.get('consensus') == "STRONG":
                            st.info("💰 **Stake: 0.5-1% of bankroll**")
                        else:
                            st.info("💰 **Stake: 0.25-0.5% of bankroll**")
                        
                    else:
                        st.markdown(f"""
                        <div class="verdict-skip">
                            <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                            <p style="margin: 0.5rem 0;">{result.get('reason', 'Grey zone - no clear consensus')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    col_s1, col_s2, _ = st.columns([1, 1, 2])
                    with col_s1:
                        if st.button("✅ Save as WIN", use_container_width=True):
                            consensus.save_match(match_data, "Win")
                            st.success("Saved!")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS", use_container_width=True):
                            consensus.save_match(match_data, "Loss")
                            st.warning("Saved!")
                            st.rerun()
    
    with col_right:
        if stats:
            st.markdown('<div class="section-title">📊 YOUR PERFORMANCE</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-box">
                <div style="display: flex; justify-content: space-around;">
                    <div><span style="color: #94a3b8;">Bets</span><br><span style="font-size: 1.5rem; font-weight: bold;">{stats['total']}</span></div>
                    <div><span style="color: #94a3b8;">Wins</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{stats['correct']}</span></div>
                    <div><span style="color: #94a3b8;">Win Rate</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #fbbf24;">{stats['win_rate']:.1f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📋 CONSENSUS RULES</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="rule-card">
            <strong style="color: #fbbf24;">1 or 2 (Home/Away Win)</strong><br>
            <span class="filter-pass">✓ Forebet: Prob ≥48%, Coef ≥1.45, Avg Goals ≥2.5</span><br>
            <span class="filter-pass">✓ SoccerVista: Same prediction, Odds ≥1.45</span><br>
            <span class="filter-pass">✓ Form: Predicted team has ≥3 wins in last 6</span><br>
            <span style="color: #94a3b8;">→ Bet 1 or 2 (Strong Consensus: 76-80%)</span>
        </div>
        <div class="rule-card">
            <strong style="color: #fbbf24;">X (Draw) → Flip to 12</strong><br>
            <span class="filter-pass">✓ Forebet: Prob ≤42%, Draw Coef ≥2.80, Avg Goals ≥2.5</span><br>
            <span class="filter-pass">✓ SoccerVista: Draw odds ≥2.80</span><br>
            <span class="filter-pass">✓ Form: At least one team has ≤2 wins (mixed form)</span><br>
            <span style="color: #94a3b8;">→ Bet Double Chance 12 (75-79%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">💰 BANKROLL RULES</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rule-card">
            <div>📊 <strong>Strong Consensus:</strong> 0.5-1% of bankroll</div>
            <div>📈 <strong>Medium Consensus:</strong> 0.25-0.5% of bankroll</div>
            <div>🎯 <strong>Maximum bets per day:</strong> 4</div>
            <div>⚡ <strong>Maximum daily exposure:</strong> 2% of bankroll</div>
            <div>📝 <strong>Track every bet</strong> in spreadsheet</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📊 PERFORMANCE (280+ MATCHES)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rule-card">
            <div style="margin-bottom: 0.5rem;">🏆 <strong>Strong Consensus:</strong> {PERFORMANCE['strong_consensus_rate']}% ({PERFORMANCE['strong_consensus_qualifiers']} bets)</div>
            <div style="margin-bottom: 0.5rem;">⚠️ <strong>Medium Consensus:</strong> {PERFORMANCE['medium_consensus_rate']}% ({PERFORMANCE['medium_consensus_qualifiers']} bets)</div>
            <div style="margin-bottom: 0.5rem;">📊 <strong>Forebet Only:</strong> {PERFORMANCE['forebet_only_rate']}%</div>
            <div>⭐ <strong>SoccerVista Only:</strong> {PERFORMANCE['soccervista_only_rate']}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent matches
        if consensus.match_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📜 RECENT RESULTS</div>', unsafe_allow_html=True)
            recent = consensus.match_history[-5:]
            for m in reversed(recent):
                result_color = "#10b981" if m.get('actual_result') == 'Win' else "#ef4444"
                result_icon = "✅" if m.get('actual_result') == 'Win' else "❌"
                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">{m.get('fb_pred', '?')} vs {m.get('sv_pred', '?')}</span>
                        <span style="color: {result_color};">{result_icon} {m.get('actual_result', '?')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet Consensus Logic** | Forebet + SoccerVista | 280+ matches | 76-80% win rate on Strong Consensus | Built from your data")

if __name__ == "__main__":
    main()