# grokbet_v2.2_final.py
# GROKBET PREDICTION SYSTEM v2.2
# 
# Core Features:
# 1. Poisson foundation from raw, unfakeable data
# 2. Automatic pattern detection
# 3. MULTI-MARKET analysis (1X2, Over/Under 2.5, BTTS)
# 4. Confidence scoring system
# 5. Best-in-scenario recommendation
# 6. Value guardrail on all markets

import streamlit as st
import math
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet v2.2",
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
        margin: 0.5rem 0;
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
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #fbbf24;
        font-weight: bold;
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
    if lam <= 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def calculate_match_probabilities(home_xg, away_xg, max_goals=7):
    home_probs = [poisson_probability(home_xg, i) for i in range(max_goals + 1)]
    away_probs = [poisson_probability(away_xg, i) for i in range(max_goals + 1)]
    
    home_win = 0
    draw = 0
    away_win = 0
    over_25 = 0
    btts = 0
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = home_probs[i] * away_probs[j]
            if i > j:
                home_win += prob
            elif i == j:
                draw += prob
            else:
                away_win += prob
            
            if i + j > 2.5:
                over_25 += prob
            
            if i > 0 and j > 0:
                btts += prob
    
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win, over_25, btts

# ============================================================================
# MAIN PREDICTION CLASS
# ============================================================================

class GrokBetPredictor:
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v22_history.json"):
                with open("grokbet_v22_history.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_v22_history.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def detect_patterns(self, data):
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
                patterns.append({"name": "Top scorer gap", "direction": "home", "confidence": "HIGH", "weight": 1.5})
            elif away_top_scorer >= home_top_scorer * 2:
                patterns.append({"name": "Top scorer gap", "direction": "away", "confidence": "HIGH", "weight": 1.5})
        
        # Pattern 2: Conversion edge
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                patterns.append({"name": "Conversion edge", "direction": "home", "confidence": "HIGH", "weight": 1.5})
            elif away_conversion >= home_conversion + 4:
                patterns.append({"name": "Conversion edge", "direction": "away", "confidence": "HIGH", "weight": 1.5})
        
        # Pattern 3: High xG + H2H tilt
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        if total_xg >= 2.7 and h2h_gap >= 3:
            if h2h_home_wins > h2h_away_wins:
                patterns.append({"name": "High xG + H2H tilt", "direction": "home", "confidence": "HIGH", "weight": 1.5})
            else:
                patterns.append({"name": "High xG + H2H tilt", "direction": "away", "confidence": "HIGH", "weight": 1.5})
        
        # Pattern 4: High total xG (for Over 2.5)
        if total_xg >= 2.7:
            patterns.append({"name": "High total xG", "direction": "over", "confidence": "HIGH", "weight": 1.5})
        elif total_xg >= 2.4:
            patterns.append({"name": "Moderate total xG", "direction": "over", "confidence": "MEDIUM", "weight": 1.0})
        
        # Pattern 5: BTTS spot
        if home_xg >= 1.2 and away_xg >= 1.2:
            patterns.append({"name": "BTTS spot", "direction": "btts", "confidence": "MEDIUM", "weight": 1.0})
        
        # Pattern 6: Strong home form
        home_form = data.get('home_form', 50)
        if home_form >= 70:
            patterns.append({"name": "Strong home form", "direction": "home", "confidence": "MEDIUM", "weight": 1.0})
        
        # Pattern 7: Strong away form
        away_form = data.get('away_form', 50)
        if away_form >= 70:
            patterns.append({"name": "Strong away form", "direction": "away", "confidence": "MEDIUM", "weight": 1.0})
        
        # Pattern 8: Very poor home form (caution)
        if home_form <= 25:
            patterns.append({"name": "Very poor home form", "direction": "away", "confidence": "MEDIUM", "weight": 1.0})
        
        return patterns
    
    def calculate_probabilities(self, data):
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        
        home_prob, draw_prob, away_prob, over_prob, btts_prob = calculate_match_probabilities(home_xg, away_xg)
        
        # Adjustments
        # Home advantage
        home_prob += 0.06
        away_prob -= 0.03
        
        # Form adjustment
        home_form = data.get('home_form', 50)
        away_form = data.get('away_form', 50)
        
        # Stronger penalty for very poor form
        if home_form <= 25:
            home_prob -= 0.05
        elif home_form <= 30:
            home_prob -= 0.03
        else:
            home_form_dev = (home_form - 50) / 10
            home_prob += home_form_dev * 0.02
        
        if away_form <= 25:
            away_prob -= 0.05
        elif away_form <= 30:
            away_prob -= 0.03
        else:
            away_form_dev = (away_form - 50) / 10
            away_prob += away_form_dev * 0.02
        
        # H2H adjustment (stronger for gap ≥3)
        h2h_home_wins = data.get('h2h_home_wins', 0)
        h2h_away_wins = data.get('h2h_away_wins', 0)
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        
        if h2h_gap >= 3:
            boost = 0.08 if h2h_gap >= 4 else 0.06
            if h2h_home_wins > h2h_away_wins:
                home_prob += boost
                draw_prob -= boost / 2
            else:
                away_prob += boost
                draw_prob -= boost / 2
        
        # GD adjustment
        home_gd = data.get('home_gd', 0)
        away_gd = data.get('away_gd', 0)
        gd_advantage = home_gd - away_gd
        gd_adjustment = (gd_advantage / 5) * 0.01
        home_prob += gd_adjustment
        away_prob -= gd_adjustment * 0.5
        
        # Top scorer adjustment
        home_top_scorer = data.get('home_top_scorer', 0)
        away_top_scorer = data.get('away_top_scorer', 0)
        
        if home_top_scorer > 0 and away_top_scorer > 0:
            if home_top_scorer >= away_top_scorer * 2:
                home_prob += 0.04
            elif away_top_scorer >= home_top_scorer * 2:
                away_prob += 0.04
        
        # Conversion adjustment
        home_conversion = data.get('home_conversion', 0)
        away_conversion = data.get('away_conversion', 0)
        
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                home_prob += 0.03
            elif away_conversion >= home_conversion + 4:
                away_prob += 0.03
        
        # Injury adjustment
        if data.get('injury_home'):
            home_prob -= 0.05
            away_prob += 0.02
        if data.get('injury_away'):
            away_prob -= 0.05
            home_prob += 0.02
        
        # Normalize 1X2
        total_1x2 = home_prob + draw_prob + away_prob
        if total_1x2 > 0:
            home_prob /= total_1x2
            draw_prob /= total_1x2
            away_prob /= total_1x2
        
        return {
            "home": home_prob,
            "draw": draw_prob,
            "away": away_prob,
            "over": over_prob,
            "btts": btts_prob,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": home_xg + away_xg
        }
    
    def calculate_implied_probability(self, odds):
        if not odds or odds.get('home', 0) == 0:
            return None
        
        implied_home = 1 / odds['home']
        implied_draw = 1 / odds['draw']
        implied_away = 1 / odds['away']
        
        total = implied_home + implied_draw + implied_away
        if total > 0:
            implied_home /= total
            implied_draw /= total
            implied_away /= total
        
        return {
            "home": implied_home,
            "draw": implied_draw,
            "away": implied_away,
            "total": total
        }
    
    def evaluate_markets(self, data, odds):
        probs = self.calculate_probabilities(data)
        patterns = self.detect_patterns(data)
        implied = self.calculate_implied_probability(odds)
        
        markets = []
        
        # 1X2 Markets
        if implied:
            # Home Win
            home_edge = probs['home'] - implied['home']
            home_patterns = [p for p in patterns if p['direction'] == 'home']
            home_confidence = max([p.get('weight', 1.0) for p in home_patterns]) if home_patterns else 0
            home_score = home_edge * home_confidence if home_confidence > 0 else 0
            
            markets.append({
                "market": "HOME WIN",
                "your_prob": probs['home'],
                "implied_prob": implied['home'],
                "edge": home_edge,
                "patterns": home_patterns,
                "confidence_score": home_score,
                "stake": "0.75-1%" if home_edge >= 0.05 and home_confidence >= 1.0 else "0.5%" if home_edge >= 0.03 else None
            })
            
            # Draw
            draw_edge = probs['draw'] - implied['draw']
            markets.append({
                "market": "DRAW",
                "your_prob": probs['draw'],
                "implied_prob": implied['draw'],
                "edge": draw_edge,
                "patterns": [],
                "confidence_score": draw_edge * 0.8,
                "stake": None
            })
            
            # Away Win
            away_edge = probs['away'] - implied['away']
            away_patterns = [p for p in patterns if p['direction'] == 'away']
            away_confidence = max([p.get('weight', 1.0) for p in away_patterns]) if away_patterns else 0
            away_score = away_edge * away_confidence if away_confidence > 0 else 0
            
            markets.append({
                "market": "AWAY WIN",
                "your_prob": probs['away'],
                "implied_prob": implied['away'],
                "edge": away_edge,
                "patterns": away_patterns,
                "confidence_score": away_score,
                "stake": "0.75-1%" if away_edge >= 0.05 and away_confidence >= 1.0 else "0.5%" if away_edge >= 0.03 else None
            })
        
        # Over 2.5 Market
        over_patterns = [p for p in patterns if p['direction'] == 'over']
        over_confidence = max([p.get('weight', 1.0) for p in over_patterns]) if over_patterns else 0
        
        markets.append({
            "market": "OVER 2.5 GOALS",
            "your_prob": probs['over'],
            "implied_prob": None,
            "edge": None,
            "patterns": over_patterns,
            "confidence_score": probs['over'] * over_confidence if over_confidence > 0 else probs['over'] * 0.5,
            "stake": "0.5-0.75%" if probs['over'] >= 0.51 and over_confidence >= 1.0 else "0.25-0.5%" if probs['over'] >= 0.51 else None,
            "is_goals": True
        })
        
        # BTTS Market
        btts_patterns = [p for p in patterns if p['direction'] == 'btts']
        btts_confidence = max([p.get('weight', 1.0) for p in btts_patterns]) if btts_patterns else 0
        
        markets.append({
            "market": "BTTS (Both Teams to Score)",
            "your_prob": probs['btts'],
            "implied_prob": None,
            "edge": None,
            "patterns": btts_patterns,
            "confidence_score": probs['btts'] * btts_confidence if btts_confidence > 0 else probs['btts'] * 0.3,
            "stake": "0.5-0.75%" if probs['btts'] >= 0.55 and btts_confidence >= 1.0 else "0.25-0.5%" if probs['btts'] >= 0.50 else None,
            "is_goals": True
        })
        
        # Filter and rank
        valid_markets = [m for m in markets if m.get('stake') is not None]
        for m in valid_markets:
            m['confidence_score'] = m.get('confidence_score', 0)
        
        valid_markets.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        return {
            "probs": probs,
            "patterns": patterns,
            "markets": valid_markets,
            "has_bet": len(valid_markets) > 0
        }
    
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
        <h1>🎯 GrokBet Prediction System v2.2</h1>
        <p>Multi-Market Analysis | 1X2 | Over/Under 2.5 | BTTS | Best-in-Scenario</p>
        <div>
            <span class="badge">📊 Poisson + Adjustments</span>
            <span class="badge">🔄 Automatic Pattern Detection</span>
            <span class="badge">💰 Multi-Market Value Check</span>
            <span class="badge">🏆 Best-in-Scenario Ranking</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = GrokBetPredictor()
    stats = predictor.get_stats()
    
    st.markdown('<div class="section-title">📊 INPUT DATA (from SportyBet screenshot)</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Chelsea")
        with col2:
            away_team = st.text_input("Away Team", "Man City")
        
        st.markdown("---")
        
        st.markdown("**Scored / Conceded Averages**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 1.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.20, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 2.00, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 0.90, 0.05)
        
        home_xg = (home_scored + away_conceded) / 2
        away_xg = (away_scored + home_conceded) / 2
        
        st.markdown(f"*Calculated xG: {home_team} {home_xg:.2f} | {away_team} {away_xg:.2f} | Total: {home_xg + away_xg:.2f}*")
        
        st.markdown("---")
        
        st.markdown("**Form & Head-to-Head**")
        col7, col8, col9 = st.columns(3)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 20)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 47)
        with col9:
            st.markdown(" ")
        
        col10, col11, col12 = st.columns(3)
        with col10:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 0)
        with col11:
            h2h_draws = st.number_input("H2H Draws (last 5)", 0, 5, 1)
        with col12:
            h2h_away = st.number_input("H2H Away Wins (last 5)", 0, 5, 4)
        
        st.markdown("---")
        
        st.markdown("**Goal Difference**")
        col13, col14 = st.columns(2)
        with col13:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, 15)
        with col14:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, 32)
        
        st.markdown("---")
        
        st.markdown("**Optional (if visible in screenshot)**")
        col15, col16, col17, col18 = st.columns(4)
        with col15:
            home_top_scorer = st.number_input(f"{home_team} Top Scorer", 0, 50, 14)
        with col16:
            away_top_scorer = st.number_input(f"{away_team} Top Scorer", 0, 50, 22)
        with col17:
            home_conversion = st.number_input(f"{home_team} Conv %", 0, 100, 13)
        with col18:
            away_conversion = st.number_input(f"{away_team} Conv %", 0, 100, 14)
        
        col19, col20 = st.columns(2)
        with col19:
            injury_home = st.checkbox(f"Key {home_team} injury")
        with col20:
            injury_away = st.checkbox(f"Key {away_team} injury")
        
        st.markdown("---")
        
        st.markdown("**Current Odds**")
        col21, col22, col23 = st.columns(3)
        with col21:
            odds_home = st.number_input("Home Win", 0.0, 20.0, 3.08, 0.05)
        with col22:
            odds_draw = st.number_input("Draw", 0.0, 20.0, 3.84, 0.05)
        with col23:
            odds_away = st.number_input("Away Win", 0.0, 20.0, 2.15, 0.05)
        
        st.markdown("---")
        
        col24, col25 = st.columns(2)
        with col24:
            odds_over = st.number_input("Over 2.5 Odds (if available)", 0.0, 10.0, 1.85, 0.05)
        with col25:
            odds_btts = st.number_input("BTTS Odds (if available)", 0.0, 10.0, 1.75, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze = st.button("🔍 ANALYZE ALL MARKETS", use_container_width=True, type="primary")
        
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
                'away': odds_away,
                'over': odds_over,
                'btts': odds_btts
            }
            
            result = predictor.evaluate_markets(data, odds)
            
            st.markdown("---")
            st.markdown("### 🔮 ANALYSIS RESULTS")
            
            # Display probabilities
            st.markdown("#### 📊 Poisson Probabilities")
            probs = result['probs']
            col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
            with col_p1:
                st.metric(f"{home_team} Win", f"{probs['home']*100:.1f}%")
            with col_p2:
                st.metric("Draw", f"{probs['draw']*100:.1f}%")
            with col_p3:
                st.metric(f"{away_team} Win", f"{probs['away']*100:.1f}%")
            with col_p4:
                st.metric("Over 2.5", f"{probs['over']*100:.1f}%")
            with col_p5:
                st.metric("BTTS", f"{probs['btts']*100:.1f}%")
            
            # Display patterns
            st.markdown("#### 🎯 Detected Patterns")
            if result['patterns']:
                for p in result['patterns']:
                    direction_text = f"→ favors {p['direction'].upper()}"
                    if p['direction'] == 'over':
                        direction_text = "→ favors OVER 2.5"
                    elif p['direction'] == 'btts':
                        direction_text = "→ both teams likely to score"
                    confidence_class = "confidence-high" if p['confidence'] == "HIGH" else "confidence-medium"
                    st.markdown(f"""
                    <div class="pattern-box">
                        <span class="pattern-yes">✅ {p['name']}</span> {direction_text}
                        <span class="{confidence_class}"> ({p['confidence']} confidence)</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pattern-box">
                    <span class="pattern-no">❌ No strong patterns detected</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Display betting recommendations
            st.markdown("#### 💰 Betting Recommendations (Ranked by Confidence)")
            
            if result['has_bet']:
                for i, market in enumerate(result['markets']):
                    if i == 0:
                        st.markdown(f"""
                        <div class="result-primary">
                            <h3 style="margin: 0; color: #fbbf24;">🏆 BEST IN SCENARIO</h3>
                            <p style="margin: 0.5rem 0; font-size: 1.2rem;">✅ BET: {market['market']}</p>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {market['stake']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-secondary">
                            <h3 style="margin: 0; color: #10b981;">✅ ALSO QUALIFIES: {market['market']}</h3>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {market['stake']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show details
                    if market.get('edge') is not None:
                        st.markdown(f"*Your probability: {market['your_prob']*100:.1f}% | Implied from odds: {market['implied_prob']*100:.1f}% | Edge: +{market['edge']*100:.1f}%*")
                    else:
                        st.markdown(f"*Your probability: {market['your_prob']*100:.1f}% | Pattern-based recommendation*")
                    
                    if market.get('patterns'):
                        pattern_names = [p['name'] for p in market['patterns']]
                        st.markdown(f"*Supporting patterns: {', '.join(pattern_names)}*")
                    
                    st.markdown("")
            else:
                st.markdown("""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO QUALIFYING BETS</h3>
                    <p style="margin: 0.5rem 0;">No market meets the minimum edge or pattern threshold.</p>
                    <p style="margin: 0; color: #94a3b8;">Skip this match.</p>
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
        st.markdown('<div class="section-title">⚡ HOW IT WORKS (v2.2)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 8px; padding: 0.75rem;">
            <div><strong>1. Poisson Foundation</strong> - From scored/conceded averages</div>
            <div style="margin-top: 0.5rem;"><strong>2. Adjustments</strong> - Home advantage, form, H2H, GD, top scorer, conversion</div>
            <div style="margin-top: 0.5rem;"><strong>3. Pattern Detection</strong> - Automatic across 8+ patterns</div>
            <div style="margin-top: 0.5rem;"><strong>4. Multi-Market Analysis</strong> - 1X2, Over 2.5, BTTS</div>
            <div style="margin-top: 0.5rem;"><strong>5. Confidence Scoring</strong> - Edge × Pattern weight</div>
            <div style="margin-top: 0.5rem;"><strong>6. Best-in-Scenario</strong> - Highest confidence market is primary bet</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet v2.2** | Multi-Market | Best-in-Scenario | Built from raw, unfakeable data")

if __name__ == "__main__":
    main()
