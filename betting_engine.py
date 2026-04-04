# grokbet_v2.4_final.py
# GROKBET PREDICTION SYSTEM v2.4
# 
# FULLY TRANSPARENT | ALL MARKETS RANKED | STAKE SCALING APPLIED
# 
# Fixes from QA:
# 1. All 9 patterns explicitly listed in output
# 2. Confidence scores calculated for ALL markets (1X2, Over, BTTS)
# 3. Visible stake scaling applied (not flat)
# 4. Form adjustment formula documented
# 5. H2H threshold documented (gap≥3=+6%, gap≥4=+8%)
# 6. Precise 1/odds for implied probability

import streamlit as st
import math
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet v2.4",
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
    .adjustment-table {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8rem;
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
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    .system-health {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & FORMULAS
# ============================================================================

# Form adjustment: ±2% per 10% deviation from 50%
def calculate_form_adjustment(form_percentage):
    deviation = (form_percentage - 50) / 10
    return deviation * 0.02

# H2H adjustment: gap ≥3 = +6%, gap ≥4 = +8%
def calculate_h2h_adjustment(home_wins, away_wins):
    gap = abs(home_wins - away_wins)
    if gap >= 4:
        return 0.08
    elif gap >= 3:
        return 0.06
    return 0

# Stake scaling by edge size
def get_stake_by_edge(edge):
    if edge < 0.03:
        return "0.25% (or skip)"
    elif edge < 0.05:
        return "0.5%"
    elif edge < 0.08:
        return "0.75%"
    else:
        return "1.0%"

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
# COMPLETE PATTERN LIST (9 FIXED PATTERNS)
# ============================================================================

ALL_PATTERNS = [
    {"name": "Top scorer gap", "weight": 1.5, "condition": "top_scorer_gap"},
    {"name": "Conversion edge", "weight": 1.5, "condition": "conversion_edge"},
    {"name": "High xG + H2H tilt", "weight": 1.5, "condition": "high_xg_h2h_tilt"},
    {"name": "High total xG", "weight": 1.5, "condition": "high_total_xg"},
    {"name": "Moderate total xG", "weight": 1.0, "condition": "moderate_total_xg"},
    {"name": "BTTS spot", "weight": 1.0, "condition": "btts_spot"},
    {"name": "Strong home form", "weight": 1.0, "condition": "strong_home_form"},
    {"name": "Strong away form", "weight": 1.0, "condition": "strong_away_form"},
    {"name": "Very poor home form", "weight": 1.0, "condition": "very_poor_home_form"}
]

# ============================================================================
# MAIN PREDICTION CLASS
# ============================================================================

class GrokBetPredictor:
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v24_history.json"):
                with open("grokbet_v24_history.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_v24_history.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def detect_patterns(self, data):
        """Detect which patterns are present from the 9 fixed patterns"""
        patterns_found = []
        
        home_top_scorer = data.get('home_top_scorer', 0)
        away_top_scorer = data.get('away_top_scorer', 0)
        home_conversion = data.get('home_conversion', 0)
        away_conversion = data.get('away_conversion', 0)
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        h2h_home_wins = data.get('h2h_home_wins', 0)
        h2h_away_wins = data.get('h2h_away_wins', 0)
        total_xg = home_xg + away_xg
        home_form = data.get('home_form', 50)
        away_form = data.get('away_form', 50)
        
        # Pattern 1: Top scorer gap
        if home_top_scorer > 0 and away_top_scorer > 0:
            if home_top_scorer >= away_top_scorer * 2:
                patterns_found.append({"name": "Top scorer gap", "direction": "home", "weight": 1.5})
            elif away_top_scorer >= home_top_scorer * 2:
                patterns_found.append({"name": "Top scorer gap", "direction": "away", "weight": 1.5})
        
        # Pattern 2: Conversion edge
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                patterns_found.append({"name": "Conversion edge", "direction": "home", "weight": 1.5})
            elif away_conversion >= home_conversion + 4:
                patterns_found.append({"name": "Conversion edge", "direction": "away", "weight": 1.5})
        
        # Pattern 3: High xG + H2H tilt
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        if total_xg >= 2.7 and h2h_gap >= 3:
            if h2h_home_wins > h2h_away_wins:
                patterns_found.append({"name": "High xG + H2H tilt", "direction": "home", "weight": 1.5})
            else:
                patterns_found.append({"name": "High xG + H2H tilt", "direction": "away", "weight": 1.5})
        
        # Pattern 4: High total xG
        if total_xg >= 2.7:
            patterns_found.append({"name": "High total xG", "direction": "over", "weight": 1.5})
        
        # Pattern 5: Moderate total xG
        elif total_xg >= 2.4:
            patterns_found.append({"name": "Moderate total xG", "direction": "over", "weight": 1.0})
        
        # Pattern 6: BTTS spot
        if home_xg >= 1.2 and away_xg >= 1.2:
            patterns_found.append({"name": "BTTS spot", "direction": "btts", "weight": 1.0})
        
        # Pattern 7: Strong home form
        if home_form >= 70:
            patterns_found.append({"name": "Strong home form", "direction": "home", "weight": 1.0})
        
        # Pattern 8: Strong away form
        if away_form >= 70:
            patterns_found.append({"name": "Strong away form", "direction": "away", "weight": 1.0})
        
        # Pattern 9: Very poor home form
        if home_form <= 25:
            patterns_found.append({"name": "Very poor home form", "direction": "away", "weight": 1.0})
        
        return patterns_found
    
    def calculate_adjustments(self, data, raw_home, raw_draw, raw_away):
        """Calculate all adjustments and return detailed breakdown"""
        adjustments = []
        
        home_prob = raw_home
        draw_prob = raw_draw
        away_prob = raw_away
        
        home_form = data.get('home_form', 50)
        away_form = data.get('away_form', 50)
        h2h_home_wins = data.get('h2h_home_wins', 0)
        h2h_away_wins = data.get('h2h_away_wins', 0)
        home_gd = data.get('home_gd', 0)
        away_gd = data.get('away_gd', 0)
        home_top_scorer = data.get('home_top_scorer', 0)
        away_top_scorer = data.get('away_top_scorer', 0)
        home_conversion = data.get('home_conversion', 0)
        away_conversion = data.get('away_conversion', 0)
        
        # 1. Home advantage
        home_prob += 0.06
        away_prob -= 0.03
        adjustments.append("Home advantage: +6.0% home, -3.0% away")
        
        # 2. Form adjustment (using documented formula: ±2% per 10% deviation from 50)
        home_form_adj = calculate_form_adjustment(home_form)
        if home_form <= 25:
            # Extra penalty for very poor form
            home_prob -= 0.05
            adjustments.append(f"Very poor home form ({home_form}%): -5.0% home (extra penalty)")
        elif home_form_adj != 0:
            home_prob += home_form_adj
            adjustments.append(f"Home form ({home_form}%): {home_form_adj*100:+.1f}% home (formula: ±2% per 10% from 50)")
        
        away_form_adj = calculate_form_adjustment(away_form)
        if away_form <= 25:
            away_prob -= 0.05
            adjustments.append(f"Very poor away form ({away_form}%): -5.0% away (extra penalty)")
        elif away_form_adj != 0:
            away_prob += away_form_adj
            adjustments.append(f"Away form ({away_form}%): {away_form_adj*100:+.1f}% away (formula: ±2% per 10% from 50)")
        
        # 3. H2H adjustment (documented: gap≥3=+6%, gap≥4=+8%)
        h2h_gap = abs(h2h_home_wins - h2h_away_wins)
        h2h_boost = calculate_h2h_adjustment(h2h_home_wins, h2h_away_wins)
        if h2h_boost > 0:
            if h2h_home_wins > h2h_away_wins:
                home_prob += h2h_boost
                draw_prob -= h2h_boost / 2
                adjustments.append(f"H2H ({h2h_home_wins}-?-{h2h_away_wins}): +{h2h_boost*100:.0f}% home, -{h2h_boost/2*100:.0f}% draw (gap≥{h2h_gap})")
            else:
                away_prob += h2h_boost
                draw_prob -= h2h_boost / 2
                adjustments.append(f"H2H ({h2h_home_wins}-?-{h2h_away_wins}): +{h2h_boost*100:.0f}% away, -{h2h_boost/2*100:.0f}% draw (gap≥{h2h_gap})")
        
        # 4. Goal difference
        gd_advantage = home_gd - away_gd
        gd_adjustment = (gd_advantage / 5) * 0.01
        if gd_adjustment != 0:
            home_prob += gd_adjustment
            away_prob -= gd_adjustment * 0.5
            adjustments.append(f"GD ({home_gd} vs {away_gd}): {gd_adjustment*100:+.1f}% home")
        
        # 5. Top scorer
        if home_top_scorer > 0 and away_top_scorer > 0:
            if home_top_scorer >= away_top_scorer * 2:
                home_prob += 0.04
                adjustments.append(f"Top scorer ({home_top_scorer} vs {away_top_scorer}): +4.0% home")
            elif away_top_scorer >= home_top_scorer * 2:
                away_prob += 0.04
                adjustments.append(f"Top scorer ({home_top_scorer} vs {away_top_scorer}): +4.0% away")
        
        # 6. Conversion
        if home_conversion > 0 and away_conversion > 0:
            if home_conversion >= away_conversion + 4:
                home_prob += 0.03
                adjustments.append(f"Conversion ({home_conversion}% vs {away_conversion}%): +3.0% home")
            elif away_conversion >= home_conversion + 4:
                away_prob += 0.03
                adjustments.append(f"Conversion ({home_conversion}% vs {away_conversion}%): +3.0% away")
        
        # 7. Injuries
        if data.get('injury_home'):
            home_prob -= 0.05
            away_prob += 0.02
            adjustments.append("Key home injury: -5.0% home, +2.0% away")
        if data.get('injury_away'):
            away_prob -= 0.05
            home_prob += 0.02
            adjustments.append("Key away injury: -5.0% away, +2.0% home")
        
        return home_prob, draw_prob, away_prob, adjustments
    
    def calculate_probabilities(self, data):
        """Calculate final probabilities with full transparency"""
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        
        # Raw Poisson
        raw_home, raw_draw, raw_away, raw_over, raw_btts = calculate_match_probabilities(home_xg, away_xg)
        
        # Apply adjustments
        final_home, final_draw, final_away, adjustments = self.calculate_adjustments(data, raw_home, raw_draw, raw_away)
        
        # Normalize 1X2
        total_1x2 = final_home + final_draw + final_away
        if total_1x2 > 0:
            final_home /= total_1x2
            final_draw /= total_1x2
            final_away /= total_1x2
        
        return {
            "raw": {"home": raw_home, "draw": raw_draw, "away": raw_away},
            "final": {"home": final_home, "draw": final_draw, "away": final_away},
            "over": raw_over,
            "btts": raw_btts,
            "adjustments": adjustments,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": home_xg + away_xg
        }
    
    def calculate_implied_probability(self, odds):
        """Calculate precise implied probability from odds using 1/odds"""
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
            "total": total,
            "raw_home": 1 / odds['home'],
            "raw_draw": 1 / odds['draw'],
            "raw_away": 1 / odds['away']
        }
    
    def evaluate_all_markets(self, data, odds):
        """Evaluate ALL markets with confidence scores"""
        probs = self.calculate_probabilities(data)
        patterns = self.detect_patterns(data)
        implied_1x2 = self.calculate_implied_probability(odds)
        
        markets = []
        
        # Helper to get pattern weight for a direction
        def get_pattern_weight(direction):
            weights = [p['weight'] for p in patterns if p.get('direction') == direction]
            return max(weights) if weights else 1.0
        
        # 1X2 Markets
        if implied_1x2:
            # Home Win
            home_edge = probs['final']['home'] - implied_1x2['home']
            home_weight = get_pattern_weight('home')
            home_score = home_edge * home_weight * 100
            
            markets.append({
                "market": "HOME WIN",
                "your_prob": probs['final']['home'],
                "implied_prob": implied_1x2['home'],
                "implied_raw": implied_1x2['raw_home'],
                "edge": home_edge,
                "pattern_weight": home_weight,
                "confidence_score": home_score,
                "stake": get_stake_by_edge(home_edge) if home_edge > 0 else None
            })
            
            # Draw
            draw_edge = probs['final']['draw'] - implied_1x2['draw']
            markets.append({
                "market": "DRAW",
                "your_prob": probs['final']['draw'],
                "implied_prob": implied_1x2['draw'],
                "implied_raw": implied_1x2['raw_draw'],
                "edge": draw_edge,
                "pattern_weight": 1.0,
                "confidence_score": draw_edge * 100,
                "stake": get_stake_by_edge(draw_edge) if draw_edge > 0 else None
            })
            
            # Away Win
            away_edge = probs['final']['away'] - implied_1x2['away']
            away_weight = get_pattern_weight('away')
            away_score = away_edge * away_weight * 100
            
            markets.append({
                "market": "AWAY WIN",
                "your_prob": probs['final']['away'],
                "implied_prob": implied_1x2['away'],
                "implied_raw": implied_1x2['raw_away'],
                "edge": away_edge,
                "pattern_weight": away_weight,
                "confidence_score": away_score,
                "stake": get_stake_by_edge(away_edge) if away_edge > 0 else None
            })
        
        # Over 2.5 Market
        if odds.get('over', 0) > 0:
            implied_over = 1 / odds['over']
            over_edge = probs['over'] - implied_over
            over_weight = get_pattern_weight('over')
            over_score = over_edge * over_weight * 100
            
            markets.append({
                "market": "OVER 2.5 GOALS",
                "your_prob": probs['over'],
                "implied_prob": implied_over,
                "implied_raw": implied_over,
                "edge": over_edge,
                "pattern_weight": over_weight,
                "confidence_score": over_score,
                "stake": get_stake_by_edge(over_edge) if over_edge > 0.02 else None,
                "is_goals": True
            })
        else:
            over_weight = get_pattern_weight('over')
            markets.append({
                "market": "OVER 2.5 GOALS",
                "your_prob": probs['over'],
                "implied_prob": None,
                "implied_raw": None,
                "edge": None,
                "pattern_weight": over_weight,
                "confidence_score": probs['over'] * over_weight,
                "stake": "0.5-0.75%" if probs['over'] >= 0.51 and over_weight >= 1.0 else None,
                "is_goals": True
            })
        
        # BTTS Market
        if odds.get('btts', 0) > 0:
            implied_btts = 1 / odds['btts']
            btts_edge = probs['btts'] - implied_btts
            btts_weight = get_pattern_weight('btts')
            btts_score = btts_edge * btts_weight * 100
            
            markets.append({
                "market": "BTTS (Both Teams to Score)",
                "your_prob": probs['btts'],
                "implied_prob": implied_btts,
                "implied_raw": implied_btts,
                "edge": btts_edge,
                "pattern_weight": btts_weight,
                "confidence_score": btts_score,
                "stake": get_stake_by_edge(btts_edge) if btts_edge > 0.02 else None,
                "is_goals": True
            })
        else:
            btts_weight = get_pattern_weight('btts')
            markets.append({
                "market": "BTTS (Both Teams to Score)",
                "your_prob": probs['btts'],
                "implied_prob": None,
                "implied_raw": None,
                "edge": None,
                "pattern_weight": btts_weight,
                "confidence_score": probs['btts'] * btts_weight,
                "stake": "0.5-0.75%" if probs['btts'] >= 0.55 and btts_weight >= 1.0 else None,
                "is_goals": True
            })
        
        # Filter and rank by confidence score (only positive edge or pattern-based)
        valid_markets = [m for m in markets if m.get('stake') is not None]
        valid_markets.sort(key=lambda x: x['confidence_score'], reverse=True)
        
        # Calculate total edge across qualified bets (for system health)
        total_edge = sum([m.get('edge', 0) for m in valid_markets if m.get('edge') is not None]) / len(valid_markets) if valid_markets else 0
        
        return {
            "probs": probs,
            "patterns": patterns,
            "markets": valid_markets,
            "has_bet": len(valid_markets) > 0,
            "total_edge": total_edge
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
        <h1>🎯 GrokBet Prediction System v2.4</h1>
        <p>ALL MARKETS RANKED | FULLY TRANSPARENT | STAKE SCALING APPLIED</p>
        <div>
            <span class="badge">📊 Raw Poisson → Adjustments → Final %</span>
            <span class="badge">🔄 9 Fixed Patterns with Weights (Listed Below)</span>
            <span class="badge">💰 Confidence Scores for ALL Markets</span>
            <span class="badge">📈 Stake Scales by Edge (0.25% to 1.0%)</span>
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
        
        col24, col25 = st.columns(2)
        with col24:
            odds_over = st.number_input("Over 2.5 Odds", 0.0, 10.0, 1.54, 0.05)
        with col25:
            odds_btts = st.number_input("BTTS Odds", 0.0, 10.0, 1.51, 0.05)
        
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
            
            result = predictor.evaluate_all_markets(data, odds)
            probs = result['probs']
            
            st.markdown("---")
            st.markdown("### 🔮 ANALYSIS RESULTS")
            
            # ================================================================
            # SECTION 1: ADJUSTMENT SUMMARY (FULL TRANSPARENCY)
            # ================================================================
            
            st.markdown("#### 📊 Adjustment Summary (Raw Poisson → Final %)")
            
            adj_html = f"""
            <div class="adjustment-table">
                <strong>RAW POISSON:</strong>         {home_team} {probs['raw']['home']*100:.1f}% | Draw {probs['raw']['draw']*100:.1f}% | {away_team} {probs['raw']['away']*100:.1f}%<br>
            """
            
            for adj in probs['adjustments']:
                adj_html += f"<strong>• {adj}</strong><br>"
            
            adj_html += f"""
                <strong>FINAL (normalized):</strong>   {home_team} {probs['final']['home']*100:.1f}% | Draw {probs['final']['draw']*100:.1f}% | {away_team} {probs['final']['away']*100:.1f}%
            </div>
            """
            
            st.markdown(adj_html, unsafe_allow_html=True)
            
            # ================================================================
            # SECTION 2: DETECTED PATTERNS (FROM 9 FIXED PATTERNS)
            # ================================================================
            
            st.markdown("#### 🎯 Detected Patterns (from 9 Fixed Patterns)")
            
            if result['patterns']:
                for p in result['patterns']:
                    direction_text = f"→ favors {p['direction'].upper()}"
                    if p['direction'] == 'over':
                        direction_text = "→ favors OVER 2.5"
                    elif p['direction'] == 'btts':
                        direction_text = "→ both teams likely to score"
                    st.markdown(f"""
                    <div class="pattern-box">
                        <span class="pattern-yes">✅ {p['name']}</span> {direction_text}
                        <span style="color: #fbbf24;"> (weight: {p['weight']})</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="pattern-box">
                    <span class="pattern-no">❌ No patterns detected</span>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # SECTION 3: ALL MARKETS RANKED (NUMERICAL CONFIDENCE SCORES)
            # ================================================================
            
            st.markdown("#### 💰 All Markets Ranked by Confidence Score")
            st.markdown("*Confidence Score = (Your Prob - Implied Prob) × Pattern Weight × 100*")
            
            if result['has_bet']:
                for i, market in enumerate(result['markets']):
                    if i == 0:
                        st.markdown(f"""
                        <div class="result-primary">
                            <h3 style="margin: 0; color: #fbbf24;">🏆 BEST IN SCENARIO (Score: {market['confidence_score']:.1f})</h3>
                            <p style="margin: 0.5rem 0; font-size: 1.2rem;">✅ BET: {market['market']}</p>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {market['stake']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-secondary">
                            <h3 style="margin: 0; color: #10b981;">✅ ALSO QUALIFIES (Score: {market['confidence_score']:.1f})</h3>
                            <p style="margin: 0.5rem 0;">🎯 {market['market']}</p>
                            <p style="margin: 0.5rem 0;"><span class="stake-highlight">💰 Stake: {market['stake']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed metrics
                    if market.get('edge') is not None:
                        st.markdown(f"""
                        *Your probability: {market['your_prob']*100:.1f}% | Implied: {market['implied_prob']*100:.1f}% (from {1/market['implied_raw']:.2f} odds) | Edge: +{market['edge']*100:.1f}% | Pattern weight: {market['pattern_weight']}*
                        """)
                    else:
                        st.markdown(f"*Your probability: {market['your_prob']*100:.1f}% | Pattern-based recommendation (no odds provided)*")
                    
                    st.markdown("")
            else:
                st.markdown("""
                <div class="result-skip">
                    <h3 style="margin: 0; color: #ef4444;">❌ NO QUALIFYING BETS</h3>
                    <p style="margin: 0.5rem 0;">No market meets the minimum edge or pattern threshold.</p>
                    <p style="margin: 0; color: #94a3b8;">Skip this match.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # SECTION 4: SYSTEM HEALTH
            # ================================================================
            
            if result['has_bet']:
                st.markdown(f"""
                <div class="system-health">
                    📈 System Health: Total edge across qualified bets = +{result['total_edge']*100:.1f}% | {len(result['markets'])} bet(s) qualify
                </div>
                """, unsafe_allow_html=True)
            
            # ================================================================
            # SECTION 5: STAKE GUIDE
            # ================================================================
            
            with st.expander("📖 Stake Guide (Edge → Stake)"):
                st.markdown("""
                | Edge Size | Stake |
                |-----------|-------|
                | <3% | 0.25% (or skip) |
                | 3-5% | 0.5% |
                | 5-8% | 0.75% |
                | >8% | 1.0% |
                """)
            
            # ================================================================
            # SAVE BUTTONS
            # ================================================================
            
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
    
    # ================================================================
    # RIGHT COLUMN - STATS AND REFERENCE
    # ================================================================
    
    st.markdown("---")
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        if stats and stats.get('total', 0) > 0:
            st.markdown('<div class="section-title">📊 YOUR STATS</div>', unsafe_allow_html=True)
            st.metric("Total Bets", stats['total'])
            st.metric("Wins", stats['correct'])
            st.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    
    with col_right:
        st.markdown('<div class="section-title">⚡ HOW IT WORKS (v2.4)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 8px; padding: 0.75rem;">
            <div><strong>1. Poisson Foundation</strong> - From scored/conceded averages</div>
            <div style="margin-top: 0.5rem;"><strong>2. Adjustments Shown</strong> - Full transparency table</div>
            <div style="margin-top: 0.5rem;"><strong>3. 9 Fixed Patterns</strong> - Listed below with weights</div>
            <div style="margin-top: 0.5rem;"><strong>4. ALL Markets Ranked</strong> - 1X2, Over 2.5, BTTS</div>
            <div style="margin-top: 0.5rem;"><strong>5. Numerical Confidence Score</strong> = Edge × Weight × 100</div>
            <div style="margin-top: 0.5rem;"><strong>6. Stake Scaling</strong> - 0.25% to 1.0% by edge size</div>
            <div style="margin-top: 0.5rem;"><strong>7. System Health</strong> - Total edge across qualified bets</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================================
    # FOOTER WITH COMPLETE PATTERN REFERENCE (ALL 9)
    # ================================================================
    
    with st.expander("📋 Complete Pattern Reference (9 Fixed Patterns with Weights)"):
        st.markdown("""
        | # | Pattern Name | Direction | Trigger | Weight |
        |---|--------------|-----------|---------|--------|
        | 1 | Top scorer gap | home/away | Top scorer ≥2× opponent | 1.5 |
        | 2 | Conversion edge | home/away | Conversion % ≥4% higher | 1.5 |
        | 3 | High xG + H2H tilt | home/away | Total xG ≥2.7 AND H2H win gap ≥3 | 1.5 |
        | 4 | High total xG | over | Total xG ≥2.7 | 1.5 |
        | 5 | Moderate total xG | over | Total xG 2.4-2.7 | 1.0 |
        | 6 | BTTS spot | btts | Both xG ≥1.2 | 1.0 |
        | 7 | Strong home form | home | Home form ≥70% | 1.0 |
        | 8 | Strong away form | away | Away form ≥70% | 1.0 |
        | 9 | Very poor home form | away | Home form ≤25% | 1.0 |
        """)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet v2.4** | ALL MARKETS RANKED | 9 Fixed Patterns | Numerical Confidence Scores | Stake Scaling by Edge | System Health Monitor | Built from raw, unfakeable data")

if __name__ == "__main__":
    main()
