# grokbet_consensus_v2.1_final.py
# GROKBET CONSENSUS v2.1 (FINAL LOCKED)
# - Strong Consensus 1/2 (Both sites agree)
# - Draw Contrarian 12 (Either site shows X + 4 conditions)
# - Over/Under 2.5 (Secondary)

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="GrokBet Consensus v2.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
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
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .result-strong {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-contrarian {
        background: linear-gradient(135deg, #1e293b 0%, #3a2e1e 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-ou {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a4a 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-skip {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stake-highlight {
        background: #fbbf24;
        color: #0f172a;
        padding: 0.25rem 0.5rem;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
    }
    .section-title {
        color: #fbbf24;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .rule-note {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GROKBET CONSENSUS V2.1 CLASS
# ============================================================================

class GrokBetConsensusV21:
    """GrokBet Consensus v2.1 - Final Locked Version"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v21.json"):
                with open("grokbet_v21.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_v21.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def analyze_form(self, form_sequence):
        """Count wins in form sequence (W = Win)"""
        if not form_sequence:
            return {"wins": 0}
        form_upper = form_sequence.upper()
        return {"wins": form_upper.count('W')}
    
    def evaluate_draw_contrarian(self, match_data):
        """
        Draw Contrarian v2.1 - Triggered by EITHER site showing X
        Conditions: 
        1. Draw odds ≥2.80 (from either site)
        2. Low confidence (Forebet Prob ≤42% OR weak form)
        3. Avg Goals ≥2.5
        4. Mixed form (one team ≤2 wins in last 6)
        """
        
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        
        sv_pred = match_data.get('sv_pred', '')
        sv_draw_odds = match_data.get('sv_draw_odds', 0)
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        
        # Check if EITHER site shows X
        fb_shows_x = (fb_pred == 'X')
        sv_shows_x = (sv_pred == 'X')
        
        if not (fb_shows_x or sv_shows_x):
            return {"valid": False, "reason": "No draw prediction from either site"}
        
        # Condition 1: Draw odds ≥2.80
        fb_odds_ok = (fb_coef >= 2.80) if fb_shows_x else False
        sv_odds_ok = (sv_draw_odds >= 2.80) if sv_shows_x else False
        
        if not (fb_odds_ok or sv_odds_ok):
            return {"valid": False, "reason": f"Draw odds too low (FB Coef={fb_coef}, SV Draw Odds={sv_draw_odds})"}
        
        # Condition 2: Low confidence
        fb_confidence_ok = (fb_prob <= 42) if fb_shows_x else True  # If FB doesn't show X, ignore
        sv_confidence_ok = (home_form['wins'] <= 2 or away_form['wins'] <= 2)  # Mixed form
        
        if fb_shows_x and not fb_confidence_ok:
            return {"valid": False, "reason": f"Forebet confidence too high (Prob={fb_prob}% >42%)"}
        
        if not sv_confidence_ok:
            return {"valid": False, "reason": f"Both teams in strong form (Home {home_form['wins']} wins, Away {away_form['wins']} wins)"}
        
        # Condition 3: Avg Goals ≥2.5
        if fb_avg_goals < 2.5:
            return {"valid": False, "reason": f"Avg Goals {fb_avg_goals} <2.5 (low scoring = draw risk)"}
        
        # Determine stake based on conditions
        both_show_x = (fb_shows_x and sv_shows_x)
        high_goals = (fb_avg_goals >= 3.5)
        
        if both_show_x:
            stake = "1.0%"
            stake_reason = "Both sites show X + all conditions"
        elif high_goals:
            stake = "0.5%"
            stake_reason = "High volatility (Avg Goals ≥3.5)"
        else:
            stake = "0.75%"
            stake_reason = "Standard Draw Contrarian"
        
        # Build reasons list
        reasons = []
        if fb_shows_x:
            reasons.append(f"Forebet shows X (Prob {fb_prob}%, Coef {fb_coef})")
        if sv_shows_x:
            reasons.append(f"SoccerVista shows X (Draw odds {sv_draw_odds})")
        reasons.append(f"Draw odds ≥2.80 ✓")
        reasons.append(f"Mixed form (Home {home_form['wins']} wins, Away {away_form['wins']} wins) ✓")
        reasons.append(f"Avg Goals {fb_avg_goals} ≥2.5 ✓")
        if both_show_x:
            reasons.append("🏆 BOTH SITES AGREE ON X - Extra strong signal")
        if high_goals:
            reasons.append("⚠️ Very high goals - stake reduced to 0.5%")
        
        return {
            "valid": True,
            "bet": "DOUBLE CHANCE 12 (No Draw)",
            "stake": stake,
            "stake_reason": stake_reason,
            "expected": "77-82%",
            "reasons": reasons,
            "both_show_x": both_show_x,
            "high_goals": high_goals
        }
    
    def evaluate_strong_consensus(self, match_data):
        """
        Strong Consensus 1 or 2 - Both sites agree on winner
        """
        
        home_team = match_data.get('home_team', 'Home')
        away_team = match_data.get('away_team', 'Away')
        
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        
        sv_pred = match_data.get('sv_pred', '')
        sv_odds = match_data.get('sv_odds', 0)
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        
        # Must be 1 or 2 (not X)
        if fb_pred not in ['1', '2']:
            return {"valid": False, "reason": f"Forebet predicts {fb_pred}, not 1 or 2"}
        
        if sv_pred not in ['1', '2']:
            return {"valid": False, "reason": f"SoccerVista predicts {sv_pred}, not 1 or 2"}
        
        # Must agree
        if fb_pred != sv_pred:
            return {"valid": False, "reason": f"Split prediction (Forebet {fb_pred}, SV {sv_pred})"}
        
        # Forebet filters
        if fb_prob < 48:
            return {"valid": False, "reason": f"Forebet Prob {fb_prob}% <48%"}
        
        if fb_coef < 1.45:
            return {"valid": False, "reason": f"Forebet Coef {fb_coef} <1.45"}
        
        # SoccerVista filters
        if sv_odds < 1.45:
            return {"valid": False, "reason": f"SoccerVista Odds {sv_odds} <1.45"}
        
        # Form check
        if fb_pred == '1' and home_form['wins'] < 3:
            return {"valid": False, "reason": f"Home team only {home_form['wins']} wins in last 6 (need ≥3)"}
        
        if fb_pred == '2' and away_form['wins'] < 3:
            return {"valid": False, "reason": f"Away team only {away_form['wins']} wins in last 6 (need ≥3)"}
        
        # Determine stake
        if fb_pred == '2' and fb_avg_goals >= 3.0:
            stake = "0.5% max"
            stake_reason = "Away win + high goals (≥3.0) - stake capped"
        else:
            stake = "0.75-1%"
            stake_reason = "Standard Strong Consensus"
        
        bet = f"{'HOME' if fb_pred == '1' else 'AWAY'} WIN ({fb_pred}) - {home_team if fb_pred == '1' else away_team}"
        
        reasons = [
            f"Forebet: Pred {fb_pred}, Prob {fb_prob}% ≥48, Coef {fb_coef} ≥1.45",
            f"SoccerVista: Pred {sv_pred}, Odds {sv_odds} ≥1.45",
            f"Form: {'Home' if fb_pred == '1' else 'Away'} has {home_form['wins'] if fb_pred == '1' else away_form['wins']} wins in last 6 (≥3)"
        ]
        
        if fb_pred == '2' and fb_avg_goals >= 3.0:
            reasons.append(f"⚠️ High goals ({fb_avg_goals}) - stake capped at 0.5%")
        
        return {
            "valid": True,
            "bet": bet,
            "stake": stake,
            "stake_reason": stake_reason,
            "expected": "76-80%",
            "reasons": reasons
        }
    
    def evaluate_ou(self, match_data):
        """
        Over/Under 2.5 - Secondary system
        """
        
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        fb_correct_total = match_data.get('fb_correct_total', 0)
        sv_ou_pred = match_data.get('sv_ou_pred', '')
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        total_wins = home_form['wins'] + away_form['wins']
        
        # Grey zone
        if 2.3 <= fb_avg_goals <= 2.7:
            return {"valid": False, "reason": f"Grey zone: Avg Goals = {fb_avg_goals}"}
        
        # OVER 2.5
        if fb_avg_goals >= 2.80 and fb_correct_total >= 3:
            if sv_ou_pred == 'O' and total_wins >= 7:
                return {
                    "valid": True,
                    "bet": "OVER 2.5 GOALS",
                    "stake": "0.5-0.75%",
                    "expected": "72-76%",
                    "reasons": [
                        f"Avg Goals {fb_avg_goals} ≥2.80",
                        f"Implied {fb_correct_total}+ goals",
                        f"SV says O",
                        f"Combined {total_wins} wins (≥7)"
                    ]
                }
            elif sv_ou_pred == 'O':
                return {
                    "valid": True,
                    "bet": "OVER 2.5 GOALS",
                    "stake": "0.25-0.5%",
                    "expected": "68-72%",
                    "reasons": [
                        f"Avg Goals {fb_avg_goals} ≥2.80",
                        f"SV says O",
                        f"⚠️ Form condition not met"
                    ]
                }
        
        # UNDER 2.5
        if fb_avg_goals <= 2.20 and fb_correct_total <= 2:
            if sv_ou_pred == 'U' and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                return {
                    "valid": True,
                    "bet": "UNDER 2.5 GOALS",
                    "stake": "0.5-0.75%",
                    "expected": "68-72%",
                    "reasons": [
                        f"Avg Goals {fb_avg_goals} ≤2.20",
                        f"Implied {fb_correct_total} goals",
                        f"SV says U",
                        f"Mixed form present"
                    ]
                }
        
        return {"valid": False, "reason": "No clear O/U consensus"}
    
    def get_stats(self):
        if not self.match_history:
            return None
        total = len(self.match_history)
        correct = sum(1 for m in self.match_history if m.get('actual_result') == 'Win')
        return {"total": total, "correct": correct, "win_rate": (correct / total * 100) if total > 0 else 0}


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Consensus v2.1</h1>
        <p>Final Locked System | Strong Consensus 1/2 + Draw Contrarian 12 + Over/Under 2.5</p>
        <div>
            <span class="badge">🏆 Strong Consensus: 76-80%</span>
            <span class="badge">🃏 Draw Contrarian: 77-82%</span>
            <span class="badge">⚽ Over/Under: 68-72%</span>
            <span class="badge">💰 One form. All results.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    consensus = GrokBetConsensusV21()
    stats = consensus.get_stats()
    
    # ================================================================
    # SINGLE INPUT FORM
    # ================================================================
    
    st.markdown('<div class="section-title">📊 ONE FORM - ALL DATA</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Row 1: Team Names
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Bihor Oradea")
        with col2:
            away_team = st.text_input("Away Team", "FC Voluntari")
        
        st.markdown("---")
        
        # Row 2: Forebet Inputs
        st.markdown("**Forebet Inputs**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            fb_pred = st.selectbox("Pred", ["1", "X", "2"])
        with col4:
            fb_prob = st.number_input("Prob %", 0.0, 100.0, 48.0, 1.0)
        with col5:
            fb_coef = st.number_input("Coef.", 0.0, 10.0, 2.87, 0.05, format="%.2f")
        with col6:
            fb_avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 3.83, 0.05, format="%.2f")
        
        col7, col8 = st.columns(2)
        with col7:
            fb_correct_total = st.number_input("Correct Score Implied Goals", 0, 10, 5)
        with col8:
            st.markdown(" ")
        
        st.markdown("---")
        
        # Row 3: SoccerVista Inputs
        st.markdown("**SoccerVista Inputs**")
        col9, col10, col11 = st.columns(3)
        with col9:
            sv_pred = st.selectbox("1X2 Prediction", ["1", "X", "2"])
        with col10:
            sv_odds = st.number_input("Odds", 0.0, 10.0, 3.00, 0.05, format="%.2f")
        with col11:
            sv_draw_odds = st.number_input("Draw Odds", 0.0, 10.0, 3.10, 0.05, format="%.2f")
        
        col12, col13 = st.columns(2)
        with col12:
            sv_ou_pred = st.selectbox("O/U Prediction", ["O", "U", "-"])
        with col13:
            st.markdown(" ")
        
        st.markdown("---")
        
        # Row 4: Form
        st.markdown("**Form (last 6 matches - use W/L/D)**")
        col14, col15 = st.columns(2)
        with col14:
            form_home = st.text_input(f"{home_team} Form", "LLDLWW")
        with col15:
            form_away = st.text_input(f"{away_team} Form", "WLLWWW")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze Button
        analyze = st.button("🔍 ANALYZE ALL SYSTEMS", use_container_width=True, type="primary")
        
        if analyze:
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'fb_pred': fb_pred,
                'fb_prob': fb_prob,
                'fb_coef': fb_coef,
                'fb_avg_goals': fb_avg_goals,
                'fb_correct_total': fb_correct_total,
                'sv_pred': sv_pred,
                'sv_odds': sv_odds,
                'sv_draw_odds': sv_draw_odds,
                'sv_ou_pred': sv_ou_pred,
                'sv_form_home': form_home.upper(),
                'sv_form_away': form_away.upper()
            }
            
            st.markdown("---")
            st.markdown("### 🔮 RESULTS")
            
            # ============================================================
            # SYSTEM 1: STRONG CONSENSUS 1 or 2
            # ============================================================
            
            st.markdown("#### 🎯 SYSTEM 1: Strong Consensus 1 or 2")
            result_strong = consensus.evaluate_strong_consensus(match_data)
            
            if result_strong['valid']:
                st.markdown(f"""
                <div class="result-strong">
                    <h3 style="margin: 0; color: #fbbf24;">✅ {result_strong['bet']}</h3>
                    <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result_strong['stake']}</span></p>
                    <p style="margin: 0; color: #94a3b8;">Projected: {result_strong['expected']}</p>
                    <div class="rule-note">
                        {' | '.join(result_strong['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO QUALIFYING 1/2 BET</h3>
                    <p style="margin: 0.5rem 0;">{result_strong.get('reason', 'Filters not met')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # SYSTEM 2: DRAW CONTRARIAN 12
            # ============================================================
            
            st.markdown("#### 🃏 SYSTEM 2: Draw Contrarian 12 (No Draw)")
            result_dc = consensus.evaluate_draw_contrarian(match_data)
            
            if result_dc['valid']:
                st.markdown(f"""
                <div class="result-contrarian">
                    <h3 style="margin: 0; color: #f59e0b;">✅ {result_dc['bet']}</h3>
                    <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result_dc['stake']}</span></p>
                    <p style="margin: 0; color: #94a3b8;">Projected: {result_dc['expected']}</p>
                    <div class="rule-note">
                        {' | '.join(result_dc['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO DRAW CONTRARIAN BET</h3>
                    <p style="margin: 0.5rem 0;">{result_dc.get('reason', 'Conditions not met')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # SYSTEM 3: OVER/UNDER 2.5
            # ============================================================
            
            st.markdown("#### ⚽ SYSTEM 3: Over/Under 2.5")
            result_ou = consensus.evaluate_ou(match_data)
            
            if result_ou['valid']:
                st.markdown(f"""
                <div class="result-ou">
                    <h3 style="margin: 0; color: #10b981;">✅ {result_ou['bet']}</h3>
                    <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {result_ou['stake']}</span></p>
                    <p style="margin: 0; color: #94a3b8;">Projected: {result_ou['expected']}</p>
                    <div class="rule-note">
                        {' | '.join(result_ou['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ SKIP OVER/UNDER</h3>
                    <p style="margin: 0.5rem 0;">{result_ou.get('reason', 'No clear consensus')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # COMBINED SUMMARY
            # ============================================================
            
            bets_found = []
            if result_strong['valid']:
                bets_found.append("Strong Consensus 1/2")
            if result_dc['valid']:
                bets_found.append("Draw Contrarian 12")
            if result_ou['valid']:
                bets_found.append("Over/Under")
            
            if len(bets_found) > 1:
                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.75rem; margin-top: 1rem; text-align: center;">
                    <span style="color: #fbbf24;">🎯 MULTIPLE BETS QUALIFY: {', '.join(bets_found)}</span><br>
                    <span style="color: #94a3b8; font-size: 0.8rem;">Max 4 bets per day | Max 2% daily exposure</span>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # SAVE BUTTONS
            # ============================================================
            
            st.markdown("---")
            st.markdown("#### 📝 Record Actual Result")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("✅ WIN (No Draw / Over Hit)", use_container_width=True):
                    consensus.save_match(match_data, "Win")
                    st.success("Saved as WIN!")
                    st.rerun()
            with col_s2:
                if st.button("❌ LOSS (Draw / Under Hit)", use_container_width=True):
                    consensus.save_match(match_data, "Loss")
                    st.warning("Saved as LOSS!")
                    st.rerun()
            with col_s3:
                if st.button("📝 SAVE WITHOUT RESULT", use_container_width=True):
                    consensus.save_match(match_data, "Pending")
                    st.info("Saved for later!")
                    st.rerun()
    
    # ================================================================
    # SIDEBAR / RIGHT COLUMN - STATS AND REFERENCE
    # ================================================================
    
    st.markdown("---")
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        if stats:
            st.markdown('<div class="section-title">📊 YOUR STATS</div>', unsafe_allow_html=True)
            st.metric("Total Bets", stats['total'])
            st.metric("Wins", stats['correct'])
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    
    with col_right:
        st.markdown('<div class="section-title">⚡ QUICK REFERENCE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 8px; padding: 0.75rem;">
            <div><strong>🎯 STRONG CONSENSUS 1/2:</strong> Both sites agree | Prob ≥48% | Coef ≥1.45 | Form ≥3 wins</div>
            <div style="margin-top: 0.5rem;"><strong>🃏 DRAW CONTRARIAN 12:</strong> Either site shows X | Draw odds ≥2.80 | Mixed form | xG ≥2.5</div>
            <div style="margin-top: 0.5rem;"><strong>⚽ OVER/UNDER:</strong> Over: xG ≥2.80 + SV "O" | Under: xG ≤2.20 + SV "U" | Grey 2.3-2.7 = SKIP</div>
            <div style="margin-top: 0.5rem;"><strong>💰 STAKE RULES:</strong> 0.75-1% (normal) | 0.5% (away + high goals OR xG≥3.5) | 1.0% (both sites show X)</div>
            <div style="margin-top: 0.5rem;"><strong>📊 MAX BETS:</strong> 4 per day | 2% daily exposure</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet Consensus v2.1** | Final Locked | Strong Consensus + Draw Contrarian + Over/Under | Built from 280+ matches")

if __name__ == "__main__":
    main()
