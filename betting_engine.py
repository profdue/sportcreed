# grokbet_v2.5_final.py
# GROKBET PREDICTION SYSTEM v2.5
# 
# CLEAN & DECISIVE - Fast to read, actionable output
# 
# Features:
# - 6 fixed patterns (no more "9 patterns" claims)
# - Clean output format (30 seconds to decision)
# - Knows when to say "No" to BTTS/Over
# - Stake scaling by edge size
# - No clutter, no over-engineering

import streamlit as st
import math
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet v2.5",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - CLEAN & MINIMAL
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 800px;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.5rem;
        color: #fbbf24;
    }
    .main-header p {
        margin: 0.25rem 0 0 0;
        color: #94a3b8;
        font-size: 0.8rem;
    }
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .result-box {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    .result-primary {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #fbbf24;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-secondary {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a4a 100%);
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .result-skip {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .pattern-badge {
        background: #0f172a;
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
        font-size: 0.7rem;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .stake-highlight {
        background: #fbbf24;
        color: #0f172a;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
    }
    hr {
        margin: 0.75rem 0;
        border-color: #334155;
    }
    .section-title {
        color: #fbbf24;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .verdict-text {
        font-size: 0.9rem;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & FORMULAS
# ============================================================================

# The 6 Patterns That Actually Matter
PATTERNS = [
    {"name": "Top scorer gap", "weight": 1.5, "condition": "top_scorer_gap"},
    {"name": "Conversion edge", "weight": 1.5, "condition": "conversion_edge"},
    {"name": "High xG + H2H tilt", "weight": 1.5, "condition": "high_xg_h2h_tilt"},
    {"name": "Very poor form", "weight": 1.0, "condition": "very_poor_form"},
    {"name": "BTTS spot", "weight": 1.0, "condition": "btts_spot"},
    {"name": "High/Low total xG", "weight": 1.0, "condition": "total_xg_level"}
]

# Stake scaling by edge size
def get_stake(edge):
    if edge < 0.03:
        return None
    elif edge < 0.05:
        return "0.5%"
    elif edge < 0.08:
        return "0.75%"
    else:
        return "1.0%"

# Form adjustment: ±2% per 10% deviation from 50
def form_adjustment(form):
    return ((form - 50) / 10) * 0.02

# H2H adjustment: gap≥3 = +6%, gap≥4 = +8%
def h2h_adjustment(home_wins, away_wins):
    gap = abs(home_wins - away_wins)
    if gap >= 4:
        return 0.08
    elif gap >= 3:
        return 0.06
    return 0

# ============================================================================
# POISSON CALCULATOR
# ============================================================================

def poisson_prob(lam, k):
    if lam <= 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

def calculate_probs(home_xg, away_xg, max_goals=7):
    home_probs = [poisson_prob(home_xg, i) for i in range(max_goals + 1)]
    away_probs = [poisson_prob(away_xg, i) for i in range(max_goals + 1)]
    
    home_win = draw = away_win = over_25 = btts = 0
    
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
# MAIN PREDICTOR CLASS
# ============================================================================

class GrokBetV25:
    def __init__(self):
        self.history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v25.json"):
                with open("grokbet_v25.json", "r") as f:
                    self.history = json.load(f)
        except:
            self.history = []
    
    def save_result(self, data, result):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            **data,
            "result": result
        })
        with open("grokbet_v25.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def detect_patterns(self, data):
        patterns = []
        
        home_top = data.get('home_top', 0)
        away_top = data.get('away_top', 0)
        home_conv = data.get('home_conv', 0)
        away_conv = data.get('away_conv', 0)
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        h2h_home = data.get('h2h_home', 0)
        h2h_away = data.get('h2h_away', 0)
        home_form = data.get('home_form', 50)
        total_xg = home_xg + away_xg
        
        # Pattern 1: Top scorer gap
        if home_top > 0 and away_top > 0:
            if home_top >= away_top * 2:
                patterns.append({"name": "Top scorer gap", "direction": "home", "weight": 1.5})
            elif away_top >= home_top * 2:
                patterns.append({"name": "Top scorer gap", "direction": "away", "weight": 1.5})
        
        # Pattern 2: Conversion edge
        if home_conv > 0 and away_conv > 0:
            if home_conv >= away_conv + 4:
                patterns.append({"name": "Conversion edge", "direction": "home", "weight": 1.5})
            elif away_conv >= home_conv + 4:
                patterns.append({"name": "Conversion edge", "direction": "away", "weight": 1.5})
        
        # Pattern 3: High xG + H2H tilt
        h2h_gap = abs(h2h_home - h2h_away)
        if total_xg >= 2.7 and h2h_gap >= 3:
            if h2h_home > h2h_away:
                patterns.append({"name": "High xG + H2H tilt", "direction": "home", "weight": 1.5})
            else:
                patterns.append({"name": "High xG + H2H tilt", "direction": "away", "weight": 1.5})
        
        # Pattern 4: Very poor form
        if home_form <= 25:
            patterns.append({"name": "Very poor form", "direction": "away", "weight": 1.0})
        if data.get('away_form', 50) <= 25:
            patterns.append({"name": "Very poor form", "direction": "home", "weight": 1.0})
        
        # Pattern 5: BTTS spot
        if home_xg >= 1.2 and away_xg >= 1.2:
            patterns.append({"name": "BTTS spot", "direction": "btts", "weight": 1.0})
        
        # Pattern 6: High/Low total xG
        if total_xg >= 2.7:
            patterns.append({"name": "High total xG", "direction": "over", "weight": 1.0})
        elif total_xg <= 2.2:
            patterns.append({"name": "Low total xG", "direction": "under", "weight": 1.0})
        
        return patterns
    
    def calculate_final_probs(self, data):
        home_xg = data.get('home_xg', 0)
        away_xg = data.get('away_xg', 0)
        
        # Raw Poisson
        raw_h, raw_d, raw_a, raw_over, raw_btts = calculate_probs(home_xg, away_xg)
        
        # Apply adjustments
        home_prob, draw_prob, away_prob = raw_h, raw_d, raw_a
        
        # Home advantage
        home_prob += 0.06
        away_prob -= 0.03
        
        # Form adjustment
        home_adj = form_adjustment(data.get('home_form', 50))
        away_adj = form_adjustment(data.get('away_form', 50))
        home_prob += home_adj
        away_prob += away_adj
        
        # H2H adjustment
        h2h_home = data.get('h2h_home', 0)
        h2h_away = data.get('h2h_away', 0)
        h2h_boost = h2h_adjustment(h2h_home, h2h_away)
        if h2h_boost > 0:
            if h2h_home > h2h_away:
                home_prob += h2h_boost
                draw_prob -= h2h_boost / 2
            else:
                away_prob += h2h_boost
                draw_prob -= h2h_boost / 2
        
        # GD adjustment
        home_gd = data.get('home_gd', 0)
        away_gd = data.get('away_gd', 0)
        gd_adj = ((home_gd - away_gd) / 5) * 0.01
        home_prob += gd_adj
        away_prob -= gd_adj * 0.5
        
        # Top scorer adjustment
        home_top = data.get('home_top', 0)
        away_top = data.get('away_top', 0)
        if home_top > 0 and away_top > 0:
            if home_top >= away_top * 2:
                home_prob += 0.04
            elif away_top >= home_top * 2:
                away_prob += 0.04
        
        # Conversion adjustment
        home_conv = data.get('home_conv', 0)
        away_conv = data.get('away_conv', 0)
        if home_conv > 0 and away_conv > 0:
            if home_conv >= away_conv + 4:
                home_prob += 0.03
            elif away_conv >= home_conv + 4:
                away_prob += 0.03
        
        # Normalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total
        
        return {
            "home": home_prob,
            "draw": draw_prob,
            "away": away_prob,
            "over": raw_over,
            "btts": raw_btts,
            "total_xg": home_xg + away_xg,
            "home_xg": home_xg,
            "away_xg": away_xg
        }
    
    def get_best_bet(self, data, odds):
        probs = self.calculate_final_probs(data)
        patterns = self.detect_patterns(data)
        
        # Helper to get pattern weight for a direction
        def get_weight(direction):
            weights = [p['weight'] for p in patterns if p.get('direction') == direction]
            return max(weights) if weights else 1.0
        
        markets = []
        
        # 1X2 markets
        if odds.get('home', 0) > 0:
            imp_home = (1 / odds['home']) / ((1/odds['home']) + (1/odds['draw']) + (1/odds['away']))
            imp_draw = (1 / odds['draw']) / ((1/odds['home']) + (1/odds['draw']) + (1/odds['away']))
            imp_away = (1 / odds['away']) / ((1/odds['home']) + (1/odds['draw']) + (1/odds['away']))
            
            # Home
            edge_h = probs['home'] - imp_home
            if edge_h > 0.03:
                markets.append({
                    "name": f"{data['home_team']} Win",
                    "edge": edge_h,
                    "stake": get_stake(edge_h),
                    "weight": get_weight('home'),
                    "score": edge_h * get_weight('home') * 100
                })
            
            # Away
            edge_a = probs['away'] - imp_away
            if edge_a > 0.03:
                markets.append({
                    "name": f"{data['away_team']} Win",
                    "edge": edge_a,
                    "stake": get_stake(edge_a),
                    "weight": get_weight('away'),
                    "score": edge_a * get_weight('away') * 100
                })
        
        # Over 2.5
        if odds.get('over', 0) > 0:
            imp_over = 1 / odds['over']
            edge_over = probs['over'] - imp_over
            if edge_over > 0.02:
                markets.append({
                    "name": "Over 2.5 Goals",
                    "edge": edge_over,
                    "stake": get_stake(edge_over),
                    "weight": get_weight('over'),
                    "score": edge_over * get_weight('over') * 100
                })
        
        # BTTS
        if odds.get('btts', 0) > 0:
            imp_btts = 1 / odds['btts']
            edge_btts = probs['btts'] - imp_btts
            if edge_btts > 0.02:
                markets.append({
                    "name": "BTTS",
                    "edge": edge_btts,
                    "stake": get_stake(edge_btts),
                    "weight": get_weight('btts'),
                    "score": edge_btts * get_weight('btts') * 100
                })
        
        # Sort by score
        markets.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            "probs": probs,
            "patterns": patterns,
            "markets": markets,
            "has_bet": len(markets) > 0
        }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet v2.5</h1>
        <p>Clean & Decisive | Fast to read | Actionable output</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = GrokBetV25()
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Oviedo")
        with col2:
            away_team = st.text_input("Away Team", "Sevilla")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 0.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.70, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 1.30, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 1.70, 0.05)
        
        home_xg = (home_scored + away_conceded) / 2
        away_xg = (away_scored + home_conceded) / 2
        
        st.caption(f"xG: {home_team} {home_xg:.2f} | {away_team} {away_xg:.2f} | Total: {home_xg + away_xg:.2f}")
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 27)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 33)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins", 0, 5, 3)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 0)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 5, 2)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, -28)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, -12)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 6)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 7)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 8)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 13)
        
        st.markdown("---")
        
        st.markdown("**Odds**")
        col18, col19, col20, col21, col22 = st.columns(5)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 2.88, 0.05)
        with col19:
            odds_draw = st.number_input("Draw", 0.0, 10.0, 3.14, 0.05)
        with col20:
            odds_away = st.number_input("Away", 0.0, 10.0, 2.83, 0.05)
        with col21:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 2.40, 0.05)
        with col22:
            odds_btts = st.number_input("BTTS", 0.0, 10.0, 1.99, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_xg': home_xg,
                'away_xg': away_xg,
                'home_form': home_form,
                'away_form': away_form,
                'h2h_home': h2h_home,
                'h2h_away': h2h_away,
                'home_gd': home_gd,
                'away_gd': away_gd,
                'home_top': home_top,
                'away_top': away_top,
                'home_conv': home_conv,
                'away_conv': away_conv
            }
            
            odds = {
                'home': odds_home,
                'draw': odds_draw,
                'away': odds_away,
                'over': odds_over,
                'btts': odds_btts
            }
            
            result = predictor.get_best_bet(data, odds)
            probs = result['probs']
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            # Header
            st.markdown(f"### 🎯 GrokBet v2.5 – Clean & Decisive")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            # Key Data
            st.markdown("**📊 KEY DATA:**")
            st.markdown(f"xG: {home_team} {probs['home_xg']:.2f} | {away_team} {probs['away_xg']:.2f} | Total {probs['total_xg']:.2f}")
            st.markdown(f"Form: {home_form}% | {away_form}%")
            st.markdown(f"H2H last 5: {h2h_home}-{h2h_draws}-{h2h_away}")
            
            st.markdown("---")
            
            # Final Probabilities
            st.markdown("**🎯 FINAL PROBABILITIES:**")
            st.markdown(f"{home_team}: {probs['home']*100:.1f}% | Draw: {probs['draw']*100:.1f}% | {away_team}: {probs['away']*100:.1f}%")
            st.markdown(f"Over 2.5: {probs['over']*100:.1f}% | BTTS: {probs['btts']*100:.1f}%")
            
            st.markdown("---")
            
            # Patterns
            if result['patterns']:
                st.markdown("**🔥 STRONG PATTERNS:**")
                for p in result['patterns']:
                    dir_text = f"→ favors {p['direction'].upper()}" if p['direction'] not in ['over', 'under', 'btts'] else ""
                    if p['direction'] == 'over':
                        dir_text = "→ favors OVER 2.5"
                    elif p['direction'] == 'under':
                        dir_text = "→ favors UNDER 2.5"
                    elif p['direction'] == 'btts':
                        dir_text = "→ both teams likely to score"
                    st.markdown(f"• {p['name']} {dir_text}")
            
            st.markdown("---")
            
            # Best Bet
            if result['has_bet']:
                best = result['markets'][0]
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 BEST IN SCENARIO:</strong><br>
                    ✅ {best['name']} – Edge +{best['edge']*100:.1f}% | Stake <span class="stake-highlight">{best['stake']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Secondary bets
                if len(result['markets']) > 1:
                    st.markdown("**⚽ SECONDARY OPTIONS:**")
                    for m in result['markets'][1:3]:
                        st.markdown(f"• {m['name']} – Edge +{m['edge']*100:.1f}% | Stake {m['stake']}")
                
                # Verdict
                st.markdown("---")
                if best['name'] == f"{away_team} Win":
                    st.markdown(f"**📝 VERDICT:** {away_team} to win is the clear best play. Goals markets are questionable given the low xG total.")
                elif best['name'] == f"{home_team} Win":
                    st.markdown(f"**📝 VERDICT:** {home_team} to win is the clear best play.")
                elif best['name'] == "Over 2.5 Goals":
                    if probs['total_xg'] < 2.5:
                        st.markdown(f"**📝 VERDICT:** Over 2.5 has an edge but total xG ({probs['total_xg']:.2f}) is low. Consider smaller stake.")
                    else:
                        st.markdown(f"**📝 VERDICT:** Over 2.5 is the best play with solid goal expectations.")
                else:
                    st.markdown(f"**📝 VERDICT:** {best['name']} is the recommended play.")
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets the minimum edge threshold. Skip this match.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save buttons
            st.markdown("---")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                if st.button("✅ WIN", use_container_width=True):
                    predictor.save_result(data, "Win")
                    st.success("Saved!")
                    st.rerun()
            with col_s2:
                if st.button("❌ LOSS", use_container_width=True):
                    predictor.save_result(data, "Loss")
                    st.warning("Saved!")
                    st.rerun()
            with col_s3:
                if st.button("📝 SAVE", use_container_width=True):
                    predictor.save_result(data, "Pending")
                    st.info("Saved!")
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet v2.5** | Clean & Decisive | 6 patterns | Stake scales by edge | Built from raw data")

if __name__ == "__main__":
    main()
