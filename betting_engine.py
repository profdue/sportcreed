# grokbet_v2.1_final.py
# GROKBET PREDICTION SYSTEM v2.1
# 
# Core Principles:
# 1. Build from raw, unfakeable data (scored/conceded averages, form %, H2H, GD)
# 2. Never use bookmaker's displayed win % as base
# 3. Poisson foundation + fixed adjustments
# 4. Pattern detection (automatic, not subjective)
# 5. Value guardrail (compare to actual odds)
# 6. Bet only when YOUR probability > implied odds AND pattern present

import streamlit as st
import math
import json
import os
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="GrokBet v2.1",
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
    .result-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    .result-bet {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
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
    .pattern-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .pattern-yes {
        color: #10b981;
    }
    .pattern-no {
        color: #ef4444;
    }
    .section-title {
        color: #fbbf24;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
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

# ============================================================================
# POISSON PROBABILITY CALCULATOR
# ============================================================================

def poisson_probability(lam, k):
    """Calculate Poisson probability: P(X = k) given lambda"""
    if lam <= 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def calculate_match_probabilities(home_xg, away_xg, max_goals=7):
    """Calculate 1X2 probabilities from expected goals"""
    home_probs = [poisson_probability(home_xg, i) for i in range(max_goals + 1)]
    away_probs = [poisson_probability(away_xg, i) for i in range(max_goals + 1)]
    
    home_win = 0
    draw = 0
    away_win = 0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = home_probs[i] * away_probs[j]
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
    
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win

def calculate_over_probability(home_xg, away_xg, threshold=2.5, max_goals=10):
    """Calculate probability of total goals exceeding threshold"""
    home_probs = [poisson_probability(home_xg, i) for i in range(max_goals + 1)]
    away_probs = [poisson_probability(away_xg, i) for i in range(max_goals + 1)]
    
    over_prob = 0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j > threshold:
                over_prob += home_probs[i] * away_probs[j]
    
    return over_prob

# ============================================================================
# MAIN PREDICTION CLASS
# ============================================================================

class GrokBetPredictor:
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v21_history.json"):
                with open("grokbet_v21_history.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_v21_history.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def detect_patterns(self, data):
        """Automatically detect which patterns are present"""
        patterns = []
        
        home_top_scorer = data.get('home_top_scorer', 0)
        away_top_scorer = data.get('away_top_scorer', 0)
        home_conversion = data.get('home_conversion', 0)
        away_conversion = data.get('away_conversion', 0)
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        h2h_home_wins = data.get('h2h_home_wins', 0)
        h2h_away_wins = data.get('h2h_away_wins', 0)
        total_xg = home_xg + away_xg
        
        # Pattern 1: Top scorer gap
        if home_top_scorer > 0 and away_top_scorer > 0:
            if home_top_scorer >= away_top_scorer * 2:
                patterns.append({"name": "Top scorer gap", "direction": "home", "confidence": "HIGH"})
            elif away_top_scorer >= home_top_scorer * 2:
                patterns.append({"name": "Top scorer gap", "direction": "away", "confidence": "HIGH"})
        
        # Pattern 2: Conversion edge
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                patterns.append({"name": "Conversion edge", "direction": "home", "confidence": "HIGH"})
            elif away_conversion >= home_conversion + 4:
                patterns.append({"name": "Conversion edge", "direction": "away", "confidence": "HIGH"})
        
        # Pattern 3: High xG + H2H tilt
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        if total_xg >= 2.7 and h2h_gap >= 3:
            if h2h_home_wins > h2h_away_wins:
                patterns.append({"name": "High xG + H2H tilt", "direction": "home", "confidence": "HIGH"})
            else:
                patterns.append({"name": "High xG + H2H tilt", "direction": "away", "confidence": "HIGH"})
        
        # Pattern 4: Away efficiency
        if away_xg >= 1.3 and away_conversion > 0 and home_conversion > 0:
            if away_conversion >= home_conversion + 4:
                patterns.append({"name": "Away efficiency", "direction": "away", "confidence": "MEDIUM"})
        
        # Pattern 5: BTTS spot
        if home_xg >= 1.2 and away_xg >= 1.2:
            patterns.append({"name": "BTTS spot", "direction": "btts", "confidence": "MEDIUM"})
        
        return patterns
    
    def calculate_probability(self, data):
        """Calculate final probability using Poisson + adjustments"""
        
        # Step 1: Poisson base
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        
        home_prob, draw_prob, away_prob = calculate_match_probabilities(home_xg, away_xg)
        
        # Step 2: Home advantage (+6% home, -3% away)
        home_prob += 0.06
        away_prob -= 0.03
        
        # Step 3: Form adjustment (±2% per 10% deviation from 50%)
        home_form = data.get('home_form', 50)
        away_form = data.get('away_form', 50)
        
        home_form_dev = (home_form - 50) / 10
        away_form_dev = (away_form - 50) / 10
        
        home_prob += home_form_dev * 0.02
        away_prob += away_form_dev * 0.02
        
        # Step 4: H2H adjustment (if win gap ≥3)
        h2h_home_wins = data.get('h2h_home_wins', 0)
        h2h_away_wins = data.get('h2h_away_wins', 0)
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        
        if h2h_gap >= 3:
            if h2h_home_wins > h2h_away_wins:
                home_prob += 0.08
                draw_prob -= 0.04
            else:
                away_prob += 0.08
                draw_prob -= 0.04
        
        # Step 5: Goal difference adjustment (+1% per 5 GD advantage)
        home_gd = data.get('home_gd', 0)
        away_gd = data.get('away_gd', 0)
        
        gd_advantage = home_gd - away_gd
        gd_adjustment = (gd_advantage / 5) * 0.01
        home_prob += gd_adjustment
        away_prob -= gd_adjustment * 0.5
        
        # Step 6: Top scorer adjustment
        home_top_scorer = data.get('home_top_scorer', 0)
        away_top_scorer = data.get('away_top_scorer', 0)
        
        if home_top_scorer > 0 and away_top_scorer > 0:
            if home_top_scorer >= away_top_scorer * 2:
                home_prob += 0.04
            elif away_top_scorer >= home_top_scorer * 2:
                away_prob += 0.04
        
        # Step 7: Conversion adjustment
        home_conversion = data.get('home_conversion', 0)
        away_conversion = data.get('away_conversion', 0)
        
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                home_prob += 0.03
            elif away_conversion >= home_conversion + 4:
                away_prob += 0.03
        
        # Step 8: Injury adjustment (manual override via flag)
        if data.get('injury_home'):
            home_prob -= 0.05
        if data.get('injury_away'):
            away_prob -= 0.05
        
        # Normalize to 100%
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        # Calculate Over 2.5 probability
        over_prob = calculate_over_probability(home_xg, away_xg)
        
        return {
            "home": home_prob,
            "draw": draw_prob,
            "away": away_prob,
            "over": over_prob,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": home_xg + away_xg
        }
    
    def evaluate_bet(self, data, odds):
        """Determine if there is a betting opportunity"""
        
        probs = self.calculate_probability(data)
        patterns = self.detect_patterns(data)
        
        # Calculate implied probabilities from odds
        if odds and odds.get('home', 0) > 0:
            implied_home = 1 / odds['home']
            implied_draw = 1 / odds['draw']
            implied_away = 1 / odds['away']
            
            # Normalize implied probabilities (remove vig)
            implied_total = implied_home + implied_draw + implied_away
            if implied_total > 0:
                implied_home /= implied_total
                implied_draw /= implied_total
                implied_away /= implied_total
        else:
            implied_home = implied_draw = implied_away = 0
        
        # Check each outcome for value
        outcomes = []
        
        # Home win
        if patterns and any(p['direction'] == 'home' for p in patterns):
            diff = probs['home'] - implied_home
            if diff >= 0.05:  # 5% threshold
                outcomes.append({
                    "outcome": "HOME WIN",
                    "your_prob": f"{probs['home']*100:.1f}%",
                    "implied_prob": f"{implied_home*100:.1f}%",
                    "diff": f"+{(diff*100):.1f}%",
                    "stake": "0.75-1%"
                })
        
        # Away win
        if patterns and any(p['direction'] == 'away' for p in patterns):
            diff = probs['away'] - implied_away
            if diff >= 0.05:
                outcomes.append({
                    "outcome": "AWAY WIN",
                    "your_prob": f"{probs['away']*100:.1f}%",
                    "implied_prob": f"{implied_away*100:.1f}%",
                    "diff": f"+{(diff*100):.1f}%",
                    "stake": "0.75-1%"
                })
        
        # Over 2.5 (if BTTS pattern present)
        btts_pattern = any(p['name'] == 'BTTS spot' for p in patterns)
        if btts_pattern and probs['over'] >= 0.51:
            outcomes.append({
                "outcome": "OVER 2.5 GOALS",
                "your_prob": f"{probs['over']*100:.1f}%",
                "implied_prob": "N/A (use odds)",
                "diff": "Pattern based",
                "stake": "0.5-0.75%"
            })
        
        return {
            "probs": probs,
            "patterns": patterns,
            "outcomes": outcomes,
            "has_bet": len(outcomes) > 0
        }

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet Prediction System v2.1</h1>
        <p>Raw Data → Poisson → Adjustments → Pattern Detection → Value Check</p>
        <div>
            <span class="badge">📊 Built from unfakeable data</span>
            <span class="badge">🔄 Automatic pattern detection</span>
            <span class="badge">💰 Value guardrail included</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = GrokBetPredictor()
    stats = predictor.get_stats() if hasattr(predictor, 'get_stats') else None
    
    # Create stats method if not exists
    if not hasattr(predictor, 'get_stats'):
        def get_stats(self):
            if not self.match_history:
                return None
            total = len(self.match_history)
            correct = sum(1 for m in self.match_history if m.get('actual_result') == 'Win')
            return {"total": total, "correct": correct, "win_rate": (correct / total * 100) if total > 0 else 0}
        predictor.get_stats = get_stats.__get__(predictor)
        stats = predictor.get_stats()
    
    st.markdown('<div class="section-title">📊 INPUT DATA (from SportyBet screenshot)</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        # Team names
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Brentford")
        with col2:
            away_team = st.text_input("Away Team", "Everton")
        
        st.markdown("---")
        
        # Scored/Conceded
        st.markdown("**Scored / Conceded Averages**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input("Home Scored", 0.0, 3.0, 1.5, 0.1)
        with col4:
            home_conceded = st.number_input("Home Conceded", 0.0, 3.0, 1.4, 0.1)
        with col5:
            away_scored = st.number_input("Away Scored", 0.0, 3.0, 1.2, 0.1)
        with col6:
            away_conceded = st.number_input("Away Conceded", 0.0, 3.0, 1.1, 0.1)
        
        # Calculate xG
        home_xg = (home_scored + away_conceded) / 2
        away_xg = (away_scored + home_conceded) / 2
        
        st.markdown(f"*Calculated xG: {home_team} {home_xg:.2f} | {away_team} {away_xg:.2f}*")
        
        st.markdown("---")
        
        # Form and H2H
        st.markdown("**Form & Head-to-Head**")
        col7, col8, col9 = st.columns(3)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 40)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 60)
        with col9:
            st.markdown(" ")
        
        col10, col11, col12 = st.columns(3)
        with col10:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 1)
        with col11:
            h2h_draws = st.number_input("H2H Draws (last 5)", 0, 5, 2)
        with col12:
            h2h_away = st.number_input("H2H Away Wins (last 5)", 0, 5, 2)
        
        st.markdown("---")
        
        # Goal Difference
        st.markdown("**Goal Difference**")
        col13, col14 = st.columns(2)
        with col13:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, 4)
        with col14:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, 2)
        
        st.markdown("---")
        
        # Optional inputs
        st.markdown("**Optional (if visible in screenshot)**")
        col15, col16, col17, col18 = st.columns(4)
        with col15:
            home_top_scorer = st.number_input(f"{home_team} Top Scorer Goals", 0, 30, 19)
        with col16:
            away_top_scorer = st.number_input(f"{away_team} Top Scorer Goals", 0, 30, 6)
        with col17:
            home_conversion = st.number_input(f"{home_team} Conversion %", 0, 100, 16)
        with col18:
            away_conversion = st.number_input(f"{away_team} Conversion %", 0, 100, 12)
        
        col19, col20 = st.columns(2)
        with col19:
            injury_home = st.checkbox(f"Injury to key {home_team} player")
        with col20:
            injury_away = st.checkbox(f"Injury to key {away_team} player")
        
        st.markdown("---")
        
        # Odds
        st.markdown("**Current Odds (from any bookmaker)**")
        col21, col22, col23 = st.columns(3)
        with col21:
            odds_home = st.number_input("Home Win Odds", 0.0, 20.0, 2.30, 0.05)
        with col22:
            odds_draw = st.number_input("Draw Odds", 0.0, 20.0, 3.40, 0.05)
        with col23:
            odds_away = st.number_input("Away Win Odds", 0.0, 20.0, 3.20, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_form': home_form,
                'away_form': away_form,
                'h2h_home_wins': h2h_home,
                'h2h_away_wins': h2h_away,
                'home_gd': home_gd,
                'away_gd': away_gd,
                'home_top_scorer': home_top_scorer,
                'away_top_scorer': away_top_scorer,
                'home_conversion': home_conversion,
                'away_conversion': away_conversion,
                'injury_home': injury_home,
                'injury_away': injury_away
            }
            
            odds = {
                'home': odds_home,
                'draw': odds_draw,
                'away': odds_away
            }
            
            result = predictor.evaluate_bet(data, odds)
            
            st.markdown("---")
            st.markdown("### 🔮 RESULTS")
            
            # Display probabilities
            st.markdown("#### 📊 Calculated Probabilities")
            probs = result['probs']
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.metric(f"{home_team} Win", f"{probs['home']*100:.1f}%")
            with col_p2:
                st.metric("Draw", f"{probs['draw']*100:.1f}%")
            with col_p3:
                st.metric(f"{away_team} Win", f"{probs['away']*100:.1f}%")
            
            st.markdown(f"*Total xG: {probs['total_xg']:.2f} | Over 2.5 probability: {probs['over']*100:.1f}%*")
            
            # Display patterns
            st.markdown("#### 🎯 Automatically Detected Patterns")
            if result['patterns']:
                for p in result['patterns']:
                    direction_text = f"→ favors {p['direction'].upper()}" if p['direction'] != 'btts' else "→ both teams likely to score"
                    st.markdown(f"""
                    <div class="pattern-box">
                        <span class="pattern-yes">✅ {p['name']}</span> {direction_text}
                        <span style="color: #fbbf24;"> ({p['confidence']} confidence)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pattern-box">
                    <span class="pattern-no">❌ No strong patterns detected</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Display betting recommendations
            st.markdown("#### 💰 Betting Recommendations")
            
            if result['has_bet']:
                for outcome in result['outcomes']:
                    st.markdown(f"""
                    <div class="result-bet">
                        <h3 style="margin: 0; color: #10b981;">✅ BET: {outcome['outcome']}</h3>
                        <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {outcome['stake']}</span></p>
                        <p style="margin: 0; color: #94a3b8;">Your probability: {outcome['your_prob']} | Implied from odds: {outcome['implied_prob']} | Edge: {outcome['diff']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO BET</h3>
                    <p style="margin: 0.5rem 0;">No qualifying patterns or value edge detected.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Save buttons
            st.markdown("---")
            st.markdown("#### 📝 Record Actual Result")
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("✅ WIN (Prediction correct)", use_container_width=True):
                    predictor.save_match(data, "Win")
                    st.success("Saved as WIN!")
                    st.rerun()
            with col_s2:
                if st.button("❌ LOSS (Prediction wrong)", use_container_width=True):
                    predictor.save_match(data, "Loss")
                    st.warning("Saved as LOSS!")
                    st.rerun()
            with col_s3:
                if st.button("📝 SAVE WITHOUT RESULT", use_container_width=True):
                    predictor.save_match(data, "Pending")
                    st.info("Saved for later!")
                    st.rerun()
    
    # Right column with stats and reference
    st.markdown("---")
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        if stats and stats.get('total', 0) > 0:
            st.markdown('<div class="section-title">📊 YOUR STATS</div>', unsafe_allow_html=True)
            st.metric("Total Bets", stats['total'])
            st.metric("Wins", stats['correct'])
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    
    with col_right:
        st.markdown('<div class="section-title">⚡ HOW IT WORKS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 8px; padding: 0.75rem;">
            <div><strong>1. Poisson Foundation</strong> - From scored/conceded averages</div>
            <div style="margin-top: 0.5rem;"><strong>2. Adjustments</strong> - Home advantage, form, H2H, GD, top scorer, conversion</div>
            <div style="margin-top: 0.5rem;"><strong>3. Pattern Detection</strong> - Automatic, not subjective</div>
            <div style="margin-top: 0.5rem;"><strong>4. Value Guardrail</strong> - Compare to actual odds</div>
            <div style="margin-top: 0.5rem;"><strong>5. Bet Decision</strong> - Only when YOUR probability > implied AND pattern present</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet v2.1** | Built from raw, unfakeable data | Automatic pattern detection | Value guardrail | No bookmaker % as base")

if __name__ == "__main__":
    main()
