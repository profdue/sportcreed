# grokbet_v3.0.py
# GROKBET v3.0 – SCREENSHOT VALUE FILTER
# 
# Unique core: SportyBet Efficiency Gap
# Efficiency = (Scored avg × Conversion %) − (Conceded avg × ((100 − Form %)/100))
# Gap = Home Efficiency − Away Efficiency
# 
# Features:
# - No standard Poisson (everyone copies that)
# - Built only from SportyBet screenshot data
# - Clear "NO" rules for BTTS/Over
# - BTTS Yes AND BTTS No both evaluated
# - Under 2.5 odds input included
# - Stake scales by edge (3%/5%/8% thresholds)

import streamlit as st
import math
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet v3.0",
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
    .efficiency-badge {
        font-family: monospace;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & FORMULAS - v3.0 UNIQUE LOGIC
# ============================================================================

DEFAULT_CONV = 0.10  # 10% default if missing

def calculate_efficiency(scored_avg, conceded_avg, form_pct, conv_pct):
    """
    Efficiency = (Scored avg × Conversion %) − (Conceded avg × ((100 − Form %)/100))
    
    Higher number = stronger team relative to what the numbers "should" allow.
    """
    conv_decimal = conv_pct / 100.0
    form_decimal = form_pct / 100.0
    weakness_multiplier = 1.0 - form_decimal  # (100 - Form)/100
    
    attack_score = scored_avg * conv_decimal
    defense_penalty = conceded_avg * weakness_multiplier
    
    return attack_score - defense_penalty

def get_stake_by_edge(edge):
    """Stake scaling: edge > 8% → 1.0%, 5-8% → 0.75%, 3-5% → 0.5%, <3% → skip"""
    if edge < 0.03:
        return None, False
    elif edge < 0.05:
        return "0.5%", True
    elif edge < 0.08:
        return "0.75%", True
    else:
        return "1.0%", True

def check_btts_trigger(home_xg, away_xg, home_conv, away_conv):
    """
    BTTS Yes only triggered if:
    - both xG ≥ 1.25
    - both conversion ≥ 11%
    
    BTTS No is the opposite structural play
    """
    btts_yes_trigger = (home_xg >= 1.25 and away_xg >= 1.25 and home_conv >= 11 and away_conv >= 11)
    btts_no_trigger = (home_xg < 1.0 or away_xg < 1.0 or home_conv < 9 or away_conv < 9)
    
    return btts_yes_trigger, btts_no_trigger

def check_over_trigger(total_xg, home_conv, away_conv):
    """
    Over 2.5 triggered if:
    - Total xG ≥ 2.8 AND both conversion ≥ 11%
    """
    if total_xg >= 2.8 and home_conv >= 11 and away_conv >= 11:
        return True, "Over 2.5"
    return False, None

def check_under_trigger(total_xg, home_scored, away_scored):
    """
    Under 2.5 triggered if:
    - Total xG ≤ 2.7 OR one team scored avg ≤ 0.8
    """
    if total_xg <= 2.7 or home_scored <= 0.8 or away_scored <= 0.8:
        return True, "Under 2.5 likely"
    return False, None

def calculate_simple_btts_prob(home_xg, away_xg):
    """Simple BTTS probability from xG (capped)"""
    home_factor = min(0.9, home_xg / 1.5)
    away_factor = min(0.9, away_xg / 1.5)
    return home_factor * away_factor

def calculate_simple_over_prob(total_xg):
    """Simple Over 2.5 probability from total xG (capped)"""
    return min(0.65, total_xg / 4.5)

def normalize_probs(home_prob, draw_prob, away_prob):
    """Normalize probabilities to sum to 1.0"""
    total = home_prob + draw_prob + away_prob
    if total > 0:
        return home_prob/total, draw_prob/total, away_prob/total
    return 0.33, 0.34, 0.33

# ============================================================================
# MAIN PREDICTOR CLASS - v3.0
# ============================================================================

class GrokBetV30:
    def __init__(self):
        self.history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_v30.json"):
                with open("grokbet_v30.json", "r") as f:
                    self.history = json.load(f)
        except:
            self.history = []
    
    def save_result(self, data, result):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            **data,
            "result": result
        })
        with open("grokbet_v30.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_bet(self, data, odds):
        # Unpack data
        home_team = data['home_team']
        away_team = data['away_team']
        home_scored = data['home_scored']
        home_conceded = data['home_conceded']
        away_scored = data['away_scored']
        away_conceded = data['away_conceded']
        home_form = data['home_form']
        away_form = data['away_form']
        home_conv = data.get('home_conv', 10)
        away_conv = data.get('away_conv', 10)
        h2h_home = data.get('h2h_home', 0)
        h2h_away = data.get('h2h_away', 0)
        home_gd = data.get('home_gd', 0)
        away_gd = data.get('away_gd', 0)
        
        # Calculate xG from scored/conceded
        home_xg = (home_scored + away_conceded) / 2
        away_xg = (away_scored + home_conceded) / 2
        total_xg = home_xg + away_xg
        
        # ========== STEP 1: EFFICIENCY SCORES ==========
        home_efficiency = calculate_efficiency(home_scored, home_conceded, home_form, home_conv)
        away_efficiency = calculate_efficiency(away_scored, away_conceded, away_form, away_conv)
        efficiency_gap = home_efficiency - away_efficiency
        
        # ========== STEP 2: 1X2 PROBABILITIES FROM EFFICIENCY GAP ==========
        base_home = 0.34
        base_away = 0.33
        base_draw = 0.33
        
        # Adjust by efficiency gap (capped at ±0.25 swing)
        gap_adj = efficiency_gap * 0.4
        gap_adj = max(-0.15, min(0.15, gap_adj))
        
        raw_home = base_home + gap_adj
        raw_away = base_away - gap_adj
        raw_draw = base_draw
        
        # H2H adjustment
        h2h_gap = abs(h2h_home - h2h_away)
        if h2h_gap >= 4:
            h2h_adj = 0.08
        elif h2h_gap >= 3:
            h2h_adj = 0.06
        else:
            h2h_adj = 0
        
        if h2h_home > h2h_away:
            raw_home += h2h_adj
            raw_draw -= h2h_adj / 2
        elif h2h_away > h2h_home:
            raw_away += h2h_adj
            raw_draw -= h2h_adj / 2
        
        # GD adjustment
        gd_adj = ((home_gd - away_gd) / 50) * 0.05
        raw_home += gd_adj
        raw_away -= gd_adj
        
        # Normalize
        home_prob, draw_prob, away_prob = normalize_probs(raw_home, raw_draw, raw_away)
        
        # ========== STEP 3: MARKET VALUE COMPARISON ==========
        markets = []
        
        # 1X2 markets
        if odds.get('home', 0) > 0 and odds.get('draw', 0) > 0 and odds.get('away', 0) > 0:
            total_implied = (1/odds['home']) + (1/odds['draw']) + (1/odds['away'])
            imp_home = (1/odds['home']) / total_implied
            imp_draw = (1/odds['draw']) / total_implied
            imp_away = (1/odds['away']) / total_implied
            
            home_edge = home_prob - imp_home
            stake_pct, is_valid = get_stake_by_edge(home_edge)
            if is_valid:
                markets.append({
                    "name": f"{home_team} Win",
                    "odds": odds['home'],
                    "edge": home_edge,
                    "stake": stake_pct,
                    "type": "1X2"
                })
            
            away_edge = away_prob - imp_away
            stake_pct, is_valid = get_stake_by_edge(away_edge)
            if is_valid:
                markets.append({
                    "name": f"{away_team} Win",
                    "odds": odds['away'],
                    "edge": away_edge,
                    "stake": stake_pct,
                    "type": "1X2"
                })
        
        # Over 2.5 check
        over_triggered, over_msg = check_over_trigger(total_xg, home_conv, away_conv)
        
        if odds.get('over', 0) > 0 and over_triggered:
            imp_over = 1 / odds['over']
            over_prob = calculate_simple_over_prob(total_xg)
            over_edge = over_prob - imp_over
            stake_pct, is_valid = get_stake_by_edge(over_edge)
            if is_valid:
                markets.append({
                    "name": "Over 2.5 Goals",
                    "odds": odds['over'],
                    "edge": over_edge,
                    "stake": stake_pct,
                    "type": "O/U"
                })
        
        # Under 2.5 check (uses actual entered odds)
        under_triggered, under_msg = check_under_trigger(total_xg, home_scored, away_scored)
        
        if odds.get('under', 0) > 0 and under_triggered:
            imp_under = 1 / odds['under']
            under_prob = 1 - calculate_simple_over_prob(total_xg)
            under_edge = under_prob - imp_under
            stake_pct, is_valid = get_stake_by_edge(under_edge)
            if is_valid:
                markets.append({
                    "name": "Under 2.5 Goals",
                    "odds": odds['under'],
                    "edge": under_edge,
                    "stake": stake_pct,
                    "type": "O/U"
                })
        
        # BTTS Yes and BTTS No checks
        btts_yes_trigger, btts_no_trigger = check_btts_trigger(home_xg, away_xg, home_conv, away_conv)
        
        # BTTS Yes
        if odds.get('btts_yes', 0) > 0 and btts_yes_trigger:
            imp_btts_yes = 1 / odds['btts_yes']
            btts_yes_prob = calculate_simple_btts_prob(home_xg, away_xg)
            btts_yes_edge = btts_yes_prob - imp_btts_yes
            stake_pct, is_valid = get_stake_by_edge(btts_yes_edge)
            if is_valid:
                markets.append({
                    "name": "BTTS Yes",
                    "odds": odds['btts_yes'],
                    "edge": btts_yes_edge,
                    "stake": stake_pct,
                    "type": "BTTS"
                })
        
        # BTTS No
        if odds.get('btts_no', 0) > 0 and btts_no_trigger:
            imp_btts_no = 1 / odds['btts_no']
            btts_no_prob = 1 - calculate_simple_btts_prob(home_xg, away_xg)
            btts_no_edge = btts_no_prob - imp_btts_no
            stake_pct, is_valid = get_stake_by_edge(btts_no_edge)
            if is_valid:
                markets.append({
                    "name": "BTTS No",
                    "odds": odds['btts_no'],
                    "edge": btts_no_edge,
                    "stake": stake_pct,
                    "type": "BTTS"
                })
        
        # Sort by edge size
        markets.sort(key=lambda x: x['edge'], reverse=True)
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": total_xg,
            "home_form": home_form,
            "away_form": away_form,
            "h2h_home": h2h_home,
            "h2h_draws": data.get('h2h_draws', 0),
            "h2h_away": h2h_away,
            "home_efficiency": home_efficiency,
            "away_efficiency": away_efficiency,
            "efficiency_gap": efficiency_gap,
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "btts_yes_trigger": btts_yes_trigger,
            "btts_no_trigger": btts_no_trigger,
            "over_triggered": over_triggered,
            "under_triggered": under_triggered,
            "markets": markets,
            "has_bet": len(markets) > 0
        }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet v3.0</h1>
        <p>Screenshot Value Filter | Efficiency Gap | BTTS Yes + No | Clean & Decisive</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = GrokBetV30()
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Kovalivka")
        with col2:
            away_team = st.text_input("Away Team", "Metalist 1925 Kharkiv")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 0.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.70, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 1.10, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 0.60, 0.05)
        
        # xG display
        home_xg_display = (home_scored + away_conceded) / 2
        away_xg_display = (away_scored + home_conceded) / 2
        st.caption(f"xG: {home_team} {home_xg_display:.2f} | {away_team} {away_xg_display:.2f} | Total: {home_xg_display + away_xg_display:.2f}")
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 47)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 80)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins", 0, 10, 1)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 10, 2)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 10, 2)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, 11)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, 0)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 7)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 4)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 10)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 11)
        
        st.markdown("---")
        
        st.markdown("**Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 3.10, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 2.90, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 2.45, 0.05)
        
        with col19:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 2.80, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 1.43, 0.05)
        
        with col20:
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 2.20, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 1.60, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
        
        if analyze:
            data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_scored': home_scored,
                'home_conceded': home_conceded,
                'away_scored': away_scored,
                'away_conceded': away_conceded,
                'home_form': home_form,
                'away_form': away_form,
                'h2h_home': h2h_home,
                'h2h_draws': h2h_draws,
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
                'under': odds_under,
                'btts_yes': odds_btts_yes,
                'btts_no': odds_btts_no
            }
            
            result = predictor.get_best_bet(data, odds)
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            # Header
            st.markdown(f"### 🎯 GrokBet v3.0")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            # KEY DATA
            st.markdown("**📊 KEY DATA:**")
            st.markdown(f"xG: {result['home_xg']:.2f} | {result['away_xg']:.2f} | Total {result['total_xg']:.2f}")
            st.markdown(f"Form: {home_form}% | {away_form}%")
            st.markdown(f"H2H: {h2h_home}-{h2h_draws}-{h2h_away}")
            
            st.markdown("---")
            
            # EFFICIENCY GAP
            st.markdown("**⚡ EFFICIENCY GAP:**")
            gap_color = "🟢" if result['efficiency_gap'] > 0 else "🔴"
            st.markdown(f"{gap_color} **{result['efficiency_gap']:+.3f}** (favors **{home_team if result['efficiency_gap'] > 0 else away_team}**)")
            st.caption(f"Home EFF: {result['home_efficiency']:.3f} | Away EFF: {result['away_efficiency']:.3f}")
            
            st.markdown("---")
            
            # TRIGGER CHECKS
            st.markdown("**📋 TRIGGER CHECKS:**")
            
            # BTTS
            if result['btts_yes_trigger']:
                st.markdown("✅ BTTS Yes: TRIGGERED (both xG ≥1.25, both conv ≥11%)")
            else:
                st.markdown("❌ BTTS Yes: NOT TRIGGERED")
            
            if result['btts_no_trigger']:
                st.markdown("✅ BTTS No: TRIGGERED (low xG or low conversion)")
            else:
                st.markdown("❌ BTTS No: NOT TRIGGERED")
            
            # Over/Under
            if result['over_triggered']:
                st.markdown("✅ Over 2.5: TRIGGERED (total xG ≥2.8, both conv ≥11%)")
            else:
                st.markdown("❌ Over 2.5: NOT TRIGGERED")
            
            if result['under_triggered']:
                st.markdown("📉 Under 2.5: TRIGGERED (total xG ≤2.7 or weak attack)")
            else:
                st.markdown("⚠️ Under 2.5: NOT TRIGGERED")
            
            st.markdown("---")
            
            # BEST BET
            if result['has_bet']:
                best = result['markets'][0]
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 BEST BET:</strong><br>
                    ✅ {best['name']} at {best['odds']:.2f}<br>
                    📊 Edge: +{best['edge']*100:.1f}% | Stake: <span class="stake-highlight">{best['stake']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Reason
                if best['type'] == "1X2":
                    fav = home_team if home_team in best['name'] else away_team
                    st.markdown(f"**Reason:** Efficiency gap ({result['efficiency_gap']:+.3f}) favors {fav}, market mispriced.")
                elif "Over" in best['name']:
                    st.markdown(f"**Reason:** Total xG {result['total_xg']:.2f} ≥ 2.8 and both conversions ≥ 11%.")
                elif "Under" in best['name']:
                    st.markdown(f"**Reason:** Total xG {result['total_xg']:.2f} ≤ 2.7 or weak attack.")
                elif "BTTS Yes" in best['name']:
                    st.markdown(f"**Reason:** Both xG ≥ 1.25 ({result['home_xg']:.2f}/{result['away_xg']:.2f}) and both conv ≥ 11%.")
                elif "BTTS No" in best['name']:
                    st.markdown(f"**Reason:** Low xG ({result['home_xg']:.2f}/{result['away_xg']:.2f}) or poor conversion rates.")
                
                # Secondary bets
                if len(result['markets']) > 1:
                    st.markdown("---")
                    st.markdown("**⚽ SECONDARY OPTIONS:**")
                    for m in result['markets'][1:3]:
                        st.markdown(f"• {m['name']} at {m['odds']:.2f} – Edge +{m['edge']*100:.1f}% | Stake {m['stake']}")
                
                # VERDICT
                st.markdown("---")
                if abs(result['efficiency_gap']) > 0.05:
                    fav = home_team if result['efficiency_gap'] > 0 else away_team
                    verdict = f"{fav} stronger than numbers suggest. Value on {fav.lower()}."
                elif result['under_triggered'] and not result['over_triggered']:
                    verdict = "Low-scoring game expected. Under 2.5 is the structural play."
                elif result['btts_no_trigger'] and not result['btts_yes_trigger']:
                    verdict = "BTTS unlikely given xG or conversion fails. Focus on 1X2 or Under."
                else:
                    verdict = "Tight game. Small stake or skip if edge < 3%."
                
                st.markdown(f"**📝 VERDICT:** {verdict}")
                
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets minimum edge threshold (3%). Skip this match.
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
    st.caption("🎯 **GrokBet v3.0** | Screenshot Value Filter | Efficiency Gap | BTTS Yes + No | Stake scales by edge")

if __name__ == "__main__":
    main()
