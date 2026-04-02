# grokbet_consensus_v2.py - GROKBET CONSENSUS LOGIC v2.0
# Final Locked Version | Clean, Mechanical, Data-Driven

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet Consensus v2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

class GrokBetConsensusV2:
    """GrokBet Consensus v2.0 - Final Locked Version"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_consensus_v2.json"):
                with open("grokbet_consensus_v2.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_consensus_v2.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def analyze_form(self, form_sequence):
        """Analyze form sequence (last 6 matches)"""
        if not form_sequence:
            return {"wins": 0, "strength": "unknown"}
        
        wins = sum(1 for r in form_sequence if r.upper() in ['W', 'WIN'])
        return {
            "wins": wins,
            "strength": "good" if wins >= 3 else "mixed" if wins == 2 else "poor"
        }
    
    def evaluate_1x2(self, match_data):
        """v2.0 1X2/12 Evaluation with refined rules"""
        
        # Forebet inputs
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        
        # SoccerVista inputs
        sv_pred = match_data.get('sv_pred', '')
        sv_odds = match_data.get('sv_odds', 0)
        sv_draw_odds = match_data.get('sv_draw_odds', 0)
        sv_form_home = match_data.get('sv_form_home', [])
        sv_form_away = match_data.get('sv_form_away', [])
        
        # ================================================================
        # RULE: Home Win (1)
        # ================================================================
        if fb_pred == '1':
            # Forebet filters
            if not (fb_prob >= 48 and fb_coef >= 1.45):
                return {"valid": False, "reason": "Forebet filters failed", "consensus": "none"}
            
            # SoccerVista consensus
            sv_passes = (sv_pred == '1' and sv_odds >= 1.45)
            
            # Form analysis
            home_form = self.analyze_form(sv_form_home)
            form_passes = home_form['wins'] >= 3
            
            if sv_passes and form_passes:
                # No stake reduction for home wins
                return {
                    "valid": True,
                    "bet": "HOME WIN (1)",
                    "consensus": "STRONG",
                    "stake": "0.75-1%",
                    "expected": "76-80%",
                    "reasons": [
                        f"✅ Forebet: Pred 1, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"✅ SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"✅ Form: Home team {home_form['wins']} wins in last 6 (≥3)"
                    ]
                }
            else:
                return {"valid": False, "reason": "SoccerVista consensus failed", "consensus": "none"}
        
        # ================================================================
        # RULE: Away Win (2) with High Goals Stake Reduction
        # ================================================================
        elif fb_pred == '2':
            # Forebet filters
            if not (fb_prob >= 48 and fb_coef >= 1.45):
                return {"valid": False, "reason": "Forebet filters failed", "consensus": "none"}
            
            # SoccerVista consensus
            sv_passes = (sv_pred == '2' and sv_odds >= 1.45)
            
            # Form analysis
            away_form = self.analyze_form(sv_form_away)
            form_passes = away_form['wins'] >= 3
            
            if sv_passes and form_passes:
                # v2.0 refinement: High goals reduce stake for away wins
                if fb_avg_goals >= 3.0:
                    stake = "0.5% max (high goals volatility)"
                    note = "⚠️ High goals (>3.0) - stake reduced to 0.5%"
                else:
                    stake = "0.75-1%"
                    note = ""
                
                return {
                    "valid": True,
                    "bet": "AWAY WIN (2)",
                    "consensus": "STRONG",
                    "stake": stake,
                    "expected": "76-80%",
                    "reasons": [
                        f"✅ Forebet: Pred 2, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"✅ SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"✅ Form: Away team {away_form['wins']} wins in last 6 (≥3)",
                        note
                    ] if note else [
                        f"✅ Forebet: Pred 2, Prob {fb_prob}% ≥48%, Coef {fb_coef} ≥1.45",
                        f"✅ SoccerVista: Same prediction, Odds {sv_odds} ≥1.45",
                        f"✅ Form: Away team {away_form['wins']} wins in last 6 (≥3)"
                    ]
                }
            else:
                return {"valid": False, "reason": "SoccerVista consensus failed", "consensus": "none"}
        
        # ================================================================
        # RULE: X (Draw) → Flip to 12
        # ================================================================
        elif fb_pred == 'X':
            # Forebet filters for flip
            if not (fb_prob <= 42 and fb_coef >= 2.80):
                return {"valid": False, "reason": "Forebet filters failed for X flip", "consensus": "none"}
            
            # SoccerVista: Draw odds check
            sv_passes = (sv_draw_odds >= 2.80)
            
            # Form: At least one team has mixed form (≤2 wins)
            home_form = self.analyze_form(sv_form_home)
            away_form = self.analyze_form(sv_form_away)
            form_passes = (home_form['wins'] <= 2 or away_form['wins'] <= 2)
            
            if sv_passes and form_passes:
                # High goals actually help 12 bets - no reduction
                return {
                    "valid": True,
                    "bet": "DOUBLE CHANCE 12 (No Draw)",
                    "consensus": "STRONG",
                    "stake": "0.75-1%",
                    "expected": "75-79%",
                    "reasons": [
                        f"✅ Forebet: Pred X, Prob {fb_prob}% ≤42%, Coef {fb_coef} ≥2.80",
                        f"✅ SoccerVista: Draw odds {sv_draw_odds} ≥2.80",
                        f"✅ Form: Mixed form (Home {home_form['wins']} wins, Away {away_form['wins']} wins)"
                    ]
                }
            else:
                return {"valid": False, "reason": "SoccerVista consensus failed for X flip", "consensus": "none"}
        
        else:
            return {"valid": False, "reason": "Invalid Forebet prediction", "consensus": "none"}
    
    def evaluate_ou(self, match_data):
        """v2.0 Over/Under 2.5 Evaluation (unchanged)"""
        
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        fb_correct_score_total = match_data.get('fb_correct_score_total', 0)
        sv_ou_pred = match_data.get('sv_ou_pred', '')
        sv_form_home = match_data.get('sv_form_home', [])
        sv_form_away = match_data.get('sv_form_away', [])
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        total_wins = home_form['wins'] + away_form['wins']
        
        # OVER 2.5
        if fb_avg_goals >= 2.80 and fb_correct_score_total >= 3:
            if sv_ou_pred == 'O' and total_wins >= 7:
                return {
                    "valid": True,
                    "bet": "OVER 2.5 GOALS",
                    "consensus": "STRONG",
                    "stake": "0.5-0.75%",
                    "expected": "72-76%",
                    "reasons": [
                        f"✅ Forebet: Avg Goals {fb_avg_goals} ≥2.80, implied {fb_correct_score_total}+ goals",
                        f"✅ SoccerVista: O prediction",
                        f"✅ Form: Combined {total_wins} wins in last 6 (≥7)"
                    ]
                }
        
        # UNDER 2.5
        elif fb_avg_goals <= 2.20 and fb_correct_score_total <= 2:
            if sv_ou_pred == 'U' and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                return {
                    "valid": True,
                    "bet": "UNDER 2.5 GOALS",
                    "consensus": "STRONG",
                    "stake": "0.5-0.75%",
                    "expected": "68-72%",
                    "reasons": [
                        f"✅ Forebet: Avg Goals {fb_avg_goals} ≤2.20, implied {fb_correct_score_total} or fewer goals",
                        f"✅ SoccerVista: U prediction",
                        f"✅ Form: At least one team has ≤2 wins in last 6"
                    ]
                }
        
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
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
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
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Consensus Logic v2.0</h1>
        <p>Final Locked Version | Forebet + SoccerVista | 280+ Matches</p>
        <div>
            <span class="badge">🏆 Strong Consensus: 76.8%</span>
            <span class="badge">⚡ Away Win High Goals: 0.5% max stake</span>
            <span class="badge">📊 Max 4 bets per day</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # System selector
    bet_type = st.radio(
        "Select System",
        ["🎯 1X2 / 12 (Primary)", "⚽ Over/Under 2.5 (Secondary)"],
        horizontal=True
    )
    
    consensus = GrokBetConsensusV2()
    stats = consensus.get_stats()
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        if "1X2" in bet_type:
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
                
                analyze = st.button("🔍 ANALYZE v2.0 CONSENSUS", use_container_width=True, type="primary")
                
                if analyze:
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
                        st.markdown(f"""
                        <div class="verdict-strong">
                            <h2 style="margin: 0; color: #fbbf24;">🏆 STRONG CONSENSUS</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.2rem;">🎯 {result['bet']}</p>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result['stake']}</span></p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for reason in result['reasons']:
                            if reason and "⚠️" in reason:
                                st.warning(reason)
                            elif reason:
                                st.success(reason)
                        
                        if fb_pred == '2' and fb_avg_goals >= 3.0:
                            st.info("📌 **Note:** Stake reduced to 0.5% due to high goals (>3.0) volatility")
                        
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
            # Over/Under 2.5 System
            st.markdown('<div class="section-title">📊 OVER/UNDER 2.5 DATA</div>', unsafe_allow_html=True)
            
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
                    form_home = st.text_input("Home Team Form", "WWDLWW", key="ou_form_home")
                with col6:
                    form_away = st.text_input("Away Team Form", "LLDWLL", key="ou_form_away")
                
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
                        st.markdown(f"""
                        <div class="verdict-strong">
                            <h2 style="margin: 0; color: #fbbf24;">🏆 STRONG CONSENSUS</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.2rem;">⚽ {result['bet']}</p>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result['stake']}</span></p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for reason in result['reasons']:
                            st.success(reason)
                    else:
                        st.markdown(f"""
                        <div class="verdict-skip">
                            <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                            <p style="margin: 0.5rem 0;">{result.get('reason', 'Grey zone')}</p>
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
        
        st.markdown('<div class="section-title">📋 v2.0 FINAL RULES</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="rule-card">
            <strong style="color: #fbbf24;">HOME WIN (1)</strong><br>
            <span class="filter-pass">✓ Forebet: Prob1 ≥48%, Coef ≥1.45</span><br>
            <span class="filter-pass">✓ SoccerVista: Odds1 ≥1.45, Home ≥3 wins in last 6</span><br>
            <span style="color: #94a3b8;">→ Stake: 0.75-1%</span>
        </div>
        <div class="rule-card">
            <strong style="color: #fbbf24;">AWAY WIN (2)</strong><br>
            <span class="filter-pass">✓ Forebet: Prob2 ≥48%, Coef ≥1.45</span><br>
            <span class="filter-pass">✓ SoccerVista: Odds2 ≥1.45, Away ≥3 wins in last 6</span><br>
            <span class="filter-pass">⚠️ If Avg Goals ≥3.0 → Stake capped at 0.5%</span>
            <span style="color: #94a3b8;">→ Stake: 0.75-1% (0.5% if high goals)</span>
        </div>
        <div class="rule-card">
            <strong style="color: #fbbf24;">X → 12 (NO DRAW)</strong><br>
            <span class="filter-pass">✓ Forebet: ProbX ≤42%, Coef ≥2.80</span><br>
            <span class="filter-pass">✓ SoccerVista: Draw odds ≥2.80, Mixed form (≤2 wins)</span><br>
            <span style="color: #94a3b8;">→ Stake: 0.75-1%</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">💰 BANKROLL RULES</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rule-card">
            <div>🏆 <strong>Strong Consensus:</strong> 0.75-1% of bankroll</div>
            <div>⚠️ <strong>Away Win + High Goals (≥3.0):</strong> 0.5% max</div>
            <div>📊 <strong>Over/Under:</strong> 0.5-0.75%</div>
            <div>🎯 <strong>Maximum bets per day:</strong> 4</div>
            <div>⚡ <strong>Maximum daily exposure:</strong> 2% of bankroll</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📊 PERFORMANCE (280+ MATCHES)</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rule-card">
            <div>🏆 <strong>Strong Consensus:</strong> {PERFORMANCE['strong_consensus_rate']}%</div>
            <div>📊 <strong>Forebet Only:</strong> {PERFORMANCE['forebet_only_rate']}%</div>
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
                        <span style="color: #94a3b8;">Pred {m.get('fb_pred', '?')}</span>
                        <span style="color: {result_color};">{result_icon} {m.get('actual_result', '?')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet Consensus v2.0** | Final Locked Version | 280+ matches | 76-80% win rate | Stake reduced for high-goal away wins")

if __name__ == "__main__":
    main()