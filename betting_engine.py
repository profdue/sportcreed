# grokbet_consensus_v2_merged.py
# SINGLE FORM - Both 1X2 and O/U from same inputs

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

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
    .result-primary {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .result-secondary {
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
    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GROKBET CONSENSUS V2.0 CLASS
# ============================================================================

class GrokBetConsensusV2:
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v2_merged.json"):
                with open("grokbet_v2_merged.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_v2_merged.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def analyze_form(self, form_sequence):
        if not form_sequence:
            return {"wins": 0}
        form_upper = form_sequence.upper()
        return {"wins": form_upper.count('W')}
    
    def evaluate(self, match_data):
        # Extract all inputs
        home_team = match_data.get('home_team', 'Home')
        away_team = match_data.get('away_team', 'Away')
        
        # Forebet
        fb_pred = match_data.get('fb_pred', '')
        fb_prob = match_data.get('fb_prob', 0)
        fb_coef = match_data.get('fb_coef', 0)
        fb_avg_goals = match_data.get('fb_avg_goals', 0)
        fb_correct_total = match_data.get('fb_correct_total', 0)
        
        # SoccerVista
        sv_pred = match_data.get('sv_pred', '')
        sv_odds = match_data.get('sv_odds', 0)
        sv_draw_odds = match_data.get('sv_draw_odds', 0)
        sv_ou_pred = match_data.get('sv_ou_pred', '')
        sv_form_home = match_data.get('sv_form_home', '')
        sv_form_away = match_data.get('sv_form_away', '')
        
        home_form = self.analyze_form(sv_form_home)
        away_form = self.analyze_form(sv_form_away)
        total_wins = home_form['wins'] + away_form['wins']
        
        results = {
            "home_team": home_team,
            "away_team": away_team,
            "1x2": {"valid": False},
            "ou": {"valid": False}
        }
        
        # ================================================================
        # 1X2 EVALUATION
        # ================================================================
        
        if fb_pred == '1':
            if fb_prob >= 48 and fb_coef >= 1.45 and sv_pred == '1' and sv_odds >= 1.45 and home_form['wins'] >= 3:
                results["1x2"] = {
                    "valid": True,
                    "bet": f"HOME WIN (1) - {home_team}",
                    "stake": "0.75-1%",
                    "expected": "76-80%",
                    "reasons": [f"Prob {fb_prob}% ≥48", f"Coef {fb_coef} ≥1.45", f"SV odds {sv_odds} ≥1.45", f"Home form {home_form['wins']} wins"]
                }
        
        elif fb_pred == '2':
            if fb_prob >= 48 and fb_coef >= 1.45 and sv_pred == '2' and sv_odds >= 1.45 and away_form['wins'] >= 3:
                stake = "0.5% max" if fb_avg_goals >= 3.0 else "0.75-1%"
                results["1x2"] = {
                    "valid": True,
                    "bet": f"AWAY WIN (2) - {away_team}",
                    "stake": stake,
                    "expected": "76-80%",
                    "reasons": [f"Prob {fb_prob}% ≥48", f"Coef {fb_coef} ≥1.45", f"SV odds {sv_odds} ≥1.45", f"Away form {away_form['wins']} wins"],
                    "capped": fb_avg_goals >= 3.0
                }
        
        elif fb_pred == 'X':
            if fb_prob <= 42 and fb_coef >= 2.80 and sv_draw_odds >= 2.80 and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
                results["1x2"] = {
                    "valid": True,
                    "bet": "DOUBLE CHANCE 12 (No Draw)",
                    "stake": "0.75-1%",
                    "expected": "75-79%",
                    "reasons": [f"Draw Prob {fb_prob}% ≤42", f"Draw Coef {fb_coef} ≥2.80", f"SV Draw odds {sv_draw_odds} ≥2.80", f"Mixed form"]
                }
        
        # ================================================================
        # OVER/UNDER EVALUATION (from same data - no re-entry)
        # ================================================================
        
        # Grey zone check
        if 2.3 <= fb_avg_goals <= 2.7:
            results["ou"] = {"valid": False, "reason": f"Grey zone: Avg Goals = {fb_avg_goals}"}
        elif fb_avg_goals >= 2.80 and fb_correct_total >= 3 and sv_ou_pred == 'O' and total_wins >= 7:
            results["ou"] = {
                "valid": True,
                "bet": "OVER 2.5 GOALS",
                "stake": "0.5-0.75%",
                "expected": "72-76%",
                "reasons": [f"Avg Goals {fb_avg_goals} ≥2.80", f"Implied {fb_correct_total}+ goals", f"SV says O", f"Combined {total_wins} wins"]
            }
        elif fb_avg_goals <= 2.20 and fb_correct_total <= 2 and sv_ou_pred == 'U' and (home_form['wins'] <= 2 or away_form['wins'] <= 2):
            results["ou"] = {
                "valid": True,
                "bet": "UNDER 2.5 GOALS",
                "stake": "0.5-0.75%",
                "expected": "68-72%",
                "reasons": [f"Avg Goals {fb_avg_goals} ≤2.20", f"Implied {fb_correct_total} goals", f"SV says U"]
            }
        elif fb_avg_goals >= 2.80 and fb_correct_total >= 3 and sv_ou_pred == 'O':
            results["ou"] = {
                "valid": True,
                "bet": "OVER 2.5 GOALS",
                "consensus": "MEDIUM",
                "stake": "0.25-0.5%",
                "expected": "68-72%",
                "reasons": [f"Avg Goals {fb_avg_goals} ≥2.80", f"SV says O", "⚠️ Form condition not met"]
            }
        else:
            results["ou"] = {"valid": False, "reason": "No clear O/U consensus"}
        
        return results
    
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
        <h1>🎯 GrokBet Consensus v2.0</h1>
        <p>One Form. Both Systems. Instant Results.</p>
        <div>
            <span class="badge">🏆 1X2 System: 76-80%</span>
            <span class="badge">⚽ O/U System: 68-72%</span>
            <span class="badge">💰 One input. Zero duplication.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    consensus = GrokBetConsensusV2()
    stats = consensus.get_stats()
    
    # ================================================================
    # SINGLE INPUT FORM - EVERYTHING IN ONE PLACE
    # ================================================================
    
    st.markdown('<div class="section-title">📊 ONE FORM - ALL DATA</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Row 1: Team Names
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Macarthur FC")
        with col2:
            away_team = st.text_input("Away Team", "Newcastle Jets")
        
        st.markdown("---")
        
        # Row 2: Forebet Core
        st.markdown("**Forebet Inputs**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            fb_pred = st.selectbox("Pred", ["1", "X", "2"])
        with col4:
            fb_prob = st.number_input("Prob %", 0.0, 100.0, 51.0, 1.0)
        with col5:
            fb_coef = st.number_input("Coef.", 0.0, 10.0, 2.50, 0.05, format="%.2f")
        with col6:
            fb_avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 3.20, 0.05, format="%.2f")
        
        # Row 3: Forebet O/U specific
        col7, col8 = st.columns(2)
        with col7:
            fb_correct_total = st.number_input("Correct Score Implied Goals", 0, 10, 4)
        with col8:
            st.markdown(" ")
        
        st.markdown("---")
        
        # Row 4: SoccerVista
        st.markdown("**SoccerVista Inputs**")
        col9, col10, col11 = st.columns(3)
        with col9:
            sv_pred = st.selectbox("1X2 Prediction", ["1", "X", "2"])
        with col10:
            sv_odds = st.number_input("Odds", 0.0, 10.0, 2.15, 0.05, format="%.2f")
        with col11:
            sv_draw_odds = st.number_input("Draw Odds", 0.0, 10.0, 3.80, 0.05, format="%.2f")
        
        col12, col13 = st.columns(2)
        with col12:
            sv_ou_pred = st.selectbox("O/U Prediction", ["O", "U", "-"])
        with col13:
            st.markdown(" ")
        
        st.markdown("---")
        
        # Row 5: Form
        st.markdown("**Form (last 6 matches - use W/L/D)**")
        col14, col15 = st.columns(2)
        with col14:
            form_home = st.text_input(f"{home_team} Form", "WLLLL")
        with col15:
            form_away = st.text_input(f"{away_team} Form", "WLWDW")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze Button
        analyze = st.button("🔍 ANALYZE BOTH SYSTEMS", use_container_width=True, type="primary")
        
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
            
            results = consensus.evaluate(match_data)
            
            st.markdown("---")
            st.markdown("### 🔮 RESULTS")
            
            # ============================================================
            # PRIMARY SYSTEM: 1X2 / 12
            # ============================================================
            
            st.markdown("#### 🎯 PRIMARY: 1X2 / 12")
            
            if results["1x2"]["valid"]:
                capped_note = ""
                if results["1x2"].get("capped"):
                    capped_note = " (capped due to high goals ≥3.0)"
                st.markdown(f"""
                <div class="result-primary">
                    <h3 style="margin: 0; color: #fbbf24;">✅ {results['1x2']['bet']}</h3>
                    <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {results['1x2']['stake']}{capped_note}</span></p>
                    <p style="margin: 0; color: #94a3b8;">Projected: {results['1x2']['expected']}</p>
                    <div class="rule-note">
                        {' | '.join(results['1x2']['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO QUALIFYING 1X2 BET</h3>
                    <p style="margin: 0.5rem 0;">Filters not met. Skip this market.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # SECONDARY SYSTEM: OVER/UNDER 2.5
            # ============================================================
            
            st.markdown("#### ⚽ SECONDARY: OVER/UNDER 2.5")
            
            if results["ou"]["valid"]:
                st.markdown(f"""
                <div class="result-secondary">
                    <h3 style="margin: 0; color: #10b981;">✅ {results['ou']['bet']}</h3>
                    <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {results['ou']['stake']}</span></p>
                    <p style="margin: 0; color: #94a3b8;">Projected: {results['ou']['expected']}</p>
                    <div class="rule-note">
                        {' | '.join(results['ou']['reasons'])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                reason = results["ou"].get("reason", "No clear O/U consensus")
                st.markdown(f"""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ SKIP OVER/UNDER</h3>
                    <p style="margin: 0.5rem 0;">{reason}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # COMBINED RECOMMENDATION
            # ============================================================
            
            if results["1x2"]["valid"] and results["ou"]["valid"]:
                st.markdown("""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.75rem; margin-top: 1rem; text-align: center;">
                    <span style="color: #fbbf24;">🎯 BOTH SYSTEMS QUALIFY</span><br>
                    <span style="color: #94a3b8; font-size: 0.8rem;">Consider correlation: High goals + Away Win often correlated</span>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # SAVE BUTTONS
            # ============================================================
            
            st.markdown("---")
            st.markdown("#### 📝 Record Actual Result")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("✅ WIN (No-Draw / Over/Under Hit)", use_container_width=True):
                    consensus.save_match(match_data, "Win")
                    st.success("Saved as WIN!")
                    st.rerun()
            with col_s2:
                if st.button("❌ LOSS (Draw / Over/Under Missed)", use_container_width=True):
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
            <div><strong>1X2 RULES:</strong> Prob ≥48% (1/2) OR ≤42% (X) | Coef ≥1.45 (1/2) OR ≥2.80 (X) | Form ≥3 wins (1/2) OR mixed (X)</div>
            <div style="margin-top: 0.5rem;"><strong>O/U RULES:</strong> Over: xG ≥2.80 + SV "O" | Under: xG ≤2.20 + SV "U" | Grey: 2.3-2.7 = SKIP</div>
            <div style="margin-top: 0.5rem;"><strong>STAKE:</strong> 0.75-1% (1X2) | 0.5% max (away + high goals) | 0.5-0.75% (O/U)</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet Consensus v2.0** | One form. Both systems. No duplicate entry. | Built from 280+ matches")

if __name__ == "__main__":
    main()
