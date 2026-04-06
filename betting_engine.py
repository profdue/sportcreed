# grokbet_vfinal.py
# GROKBET vFINAL – STRONG SIGNAL + SAFETY FILTER
# 
# Core Philosophy: Bet only when data gives a clear, strong signal. Skip everything else.
# 
# Bet Type 1: "One Team Unlikely to Score" (BTTS No / Under 2.5)
#   Trigger: 2+ of (scored ≤1.0, conv ≤10%, total xG ≤2.3, form ≤30%)
# 
# Bet Type 2: "Clear Stronger Team" (1X2)
#   Trigger: |Gap| > 0.6 AND (H2H wins ≥3 OR form diff ≥30%)
# 
# Safety Rules:
#   - Small Gap Rule: |Gap| ≤ 0.4 → no 1X2 side bets (Draw only if odds ≥3.20)
#   - No Bet Zone: No triggers → skip match
#   - Over 2.5: Only if xG ≥3.0 AND both conv ≥12% (very rare)
# 
# NO MORE CHANGES. This is the final version.

import streamlit as st
import math
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet vFinal",
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
    .result-trigger {
        background: linear-gradient(135deg, #1e293b 0%, #2a3e4a 100%);
        border-left: 4px solid #3b82f6;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
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
    .trigger-badge {
        background: #1e3a2e;
        border-radius: 6px;
        padding: 0.25rem 0.5rem;
        font-size: 0.7rem;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS & FORMULAS - vFINAL LOCKED
# ============================================================================

# Thresholds
SMALL_GAP_THRESHOLD = 0.4
STRONG_GAP_THRESHOLD = 0.6
VERY_STRONG_GAP_THRESHOLD = 0.8
FORM_DIFF_THRESHOLD = 30  # percentage points
H2H_WIN_THRESHOLD = 3  # wins in last 5
BTTS_NO_MIN_ODDS = 1.70
UNDER_MIN_ODDS = 1.80
DRAW_MIN_ODDS = 3.20
OVER_XG_THRESHOLD = 3.0
OVER_CONV_THRESHOLD = 12

# Low xG / weak attack thresholds
SCORED_WEAK_THRESHOLD = 1.0
TOTAL_XG_LOW_THRESHOLD = 2.3
CONV_WEAK_THRESHOLD = 10
FORM_WEAK_THRESHOLD = 30

def calculate_efficiency(scored_avg, conceded_avg, form_pct, conv_pct):
    """
    Efficiency = (Scored avg × Conversion %) − (Conceded avg × ((100 − Form %)/100))
    """
    conv_decimal = conv_pct / 100.0
    form_decimal = form_pct / 100.0
    weakness_multiplier = 1.0 - form_decimal
    
    attack_score = scored_avg * conv_decimal
    defense_penalty = conceded_avg * weakness_multiplier
    
    return attack_score - defense_penalty

def check_btts_triggers(home_scored, away_scored, total_xg, home_conv, away_conv, home_form, away_form):
    """
    Bet Type 1: "One Team Unlikely to Score"
    Returns: (trigger_count, triggers_list, should_bet, recommended_bet, stake)
    """
    triggers = []
    trigger_count = 0
    
    # Condition 1: One team scored avg ≤ 1.0
    if home_scored <= SCORED_WEAK_THRESHOLD:
        triggers.append(f"{home_team} scored {home_scored:.2f} ≤ 1.0")
        trigger_count += 1
    elif away_scored <= SCORED_WEAK_THRESHOLD:
        triggers.append(f"{away_team} scored {away_scored:.2f} ≤ 1.0")
        trigger_count += 1
    
    # Condition 2: One team conversion % ≤ 10%
    if home_conv <= CONV_WEAK_THRESHOLD:
        triggers.append(f"{home_team} conv {home_conv}% ≤ 10%")
        trigger_count += 1
    elif away_conv <= CONV_WEAK_THRESHOLD:
        triggers.append(f"{away_team} conv {away_conv}% ≤ 10%")
        trigger_count += 1
    
    # Condition 3: Total xG ≤ 2.3
    if total_xg <= TOTAL_XG_LOW_THRESHOLD:
        triggers.append(f"Total xG {total_xg:.2f} ≤ 2.3")
        trigger_count += 1
    
    # Condition 4: One team form % ≤ 30%
    if home_form <= FORM_WEAK_THRESHOLD:
        triggers.append(f"{home_team} form {home_form}% ≤ 30%")
        trigger_count += 1
    elif away_form <= FORM_WEAK_THRESHOLD:
        triggers.append(f"{away_team} form {away_form}% ≤ 30%")
        trigger_count += 1
    
    should_bet = trigger_count >= 2
    
    # Determine stake based on trigger count
    if trigger_count >= 3:
        stake = "1.0%"
    elif trigger_count >= 2:
        stake = "0.75%"
    else:
        stake = None
    
    return trigger_count, triggers, should_bet, stake

def check_strong_team_triggers(efficiency_gap, h2h_home, h2h_away, home_form, away_form):
    """
    Bet Type 2: "Clear Stronger Team"
    Returns: (direction, gap_strength, should_bet, stake, reason)
    direction: "home", "away", or None
    """
    gap_abs = abs(efficiency_gap)
    
    # Gap must be > 0.6
    if gap_abs <= STRONG_GAP_THRESHOLD:
        return None, None, False, None, f"Gap {efficiency_gap:+.3f} ≤ {STRONG_GAP_THRESHOLD}"
    
    # Determine favored team
    if efficiency_gap > 0:
        favored = "home"
        h2h_wins = h2h_home
        form = home_form
        other_form = away_form
    else:
        favored = "away"
        h2h_wins = h2h_away
        form = away_form
        other_form = home_form
    
    # Check H2H or form difference
    form_diff = abs(form - other_form)
    h2h_strong = h2h_wins >= H2H_WIN_THRESHOLD
    form_strong = form_diff >= FORM_DIFF_THRESHOLD
    
    if not (h2h_strong or form_strong):
        return None, None, False, None, f"No H2H (wins={h2h_wins}) or form diff ({form_diff}%)"
    
    # Determine stake based on gap strength
    if gap_abs >= VERY_STRONG_GAP_THRESHOLD:
        stake = "1.0%"
        gap_strength = "very strong"
    else:
        stake = "0.75%"
        gap_strength = "strong"
    
    reason = f"Gap {efficiency_gap:+.3f} (>0.6) + "
    if h2h_strong:
        reason += f"H2H wins={h2h_wins}"
    if h2h_strong and form_strong:
        reason += " and "
    if form_strong:
        reason += f"form diff={form_diff}%"
    
    return favored, gap_strength, True, stake, reason

def check_over_trigger(total_xg, home_conv, away_conv):
    """
    Over 2.5: Only if total xG ≥ 3.0 AND both conv ≥ 12%
    """
    if total_xg >= OVER_XG_THRESHOLD and home_conv >= OVER_CONV_THRESHOLD and away_conv >= OVER_CONV_THRESHOLD:
        return True, f"xG {total_xg:.2f} ≥ 3.0, both conv ≥ 12%"
    return False, None

def calculate_simple_btts_prob(home_xg, away_xg):
    """Simple BTTS probability from xG (capped)"""
    home_factor = min(0.9, home_xg / 1.5)
    away_factor = min(0.9, away_xg / 1.5)
    return home_factor * away_factor

def calculate_simple_over_prob(total_xg):
    """Simple Over 2.5 probability from total xG (capped)"""
    return min(0.65, total_xg / 4.5)

def get_edge_from_odds(odds, prob):
    """Calculate edge from odds and probability"""
    if odds <= 0:
        return 0
    implied = 1 / odds
    return prob - implied

# ============================================================================
# MAIN PREDICTOR CLASS - vFINAL LOCKED
# ============================================================================

class GrokBetVFinal:
    def __init__(self):
        self.history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_vfinal.json"):
                with open("grokbet_vfinal.json", "r") as f:
                    self.history = json.load(f)
        except:
            self.history = []
    
    def save_result(self, data, result):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            **data,
            "result": result
        })
        with open("grokbet_vfinal.json", "w") as f:
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
        
        # Calculate Efficiency Gap
        home_efficiency = calculate_efficiency(home_scored, home_conceded, home_form, home_conv)
        away_efficiency = calculate_efficiency(away_scored, away_conceded, away_form, away_conv)
        efficiency_gap = home_efficiency - away_efficiency
        gap_abs = abs(efficiency_gap)
        
        # ========== BET TYPE 1: BTTS No / Under 2.5 ==========
        btts_trigger_count, btts_triggers, btts_should_bet, btts_stake = check_btts_triggers(
            home_scored, away_scored, total_xg, home_conv, away_conv, home_form, away_form
        )
        
        # ========== BET TYPE 2: Clear Stronger Team ==========
        favored_direction, gap_strength, strong_should_bet, strong_stake, strong_reason = check_strong_team_triggers(
            efficiency_gap, h2h_home, h2h_away, home_form, away_form
        )
        
        # ========== SAFETY RULES ==========
        # Small Gap Rule: |Gap| ≤ 0.4 → no 1X2 side bets
        small_gap = gap_abs <= SMALL_GAP_THRESHOLD
        if small_gap:
            strong_should_bet = False  # Override - no side bets when gap is small
        
        # No Bet Zone: If no triggers, skip entirely
        no_bet_zone = not btts_should_bet and not strong_should_bet
        
        # Over 2.5 check (very rare)
        over_triggered, over_reason = check_over_trigger(total_xg, home_conv, away_conv)
        
        # ========== BUILD RECOMMENDATIONS ==========
        recommendations = []
        
        # BTTS No recommendation
        if btts_should_bet and odds.get('btts_no', 0) >= BTTS_NO_MIN_ODDS:
            btts_no_prob = 1 - calculate_simple_btts_prob(home_xg, away_xg)
            edge = get_edge_from_odds(odds['btts_no'], btts_no_prob)
            if edge > 0.02:
                recommendations.append({
                    "name": "BTTS No",
                    "odds": odds['btts_no'],
                    "stake": btts_stake,
                    "edge": edge,
                    "type": "BTTS No",
                    "triggers": btts_trigger_count
                })
        
        # Under 2.5 recommendation
        if btts_should_bet and odds.get('under', 0) >= UNDER_MIN_ODDS:
            under_prob = 1 - calculate_simple_over_prob(total_xg)
            edge = get_edge_from_odds(odds['under'], under_prob)
            if edge > 0.02:
                recommendations.append({
                    "name": "Under 2.5 Goals",
                    "odds": odds['under'],
                    "stake": btts_stake,
                    "edge": edge,
                    "type": "Under",
                    "triggers": btts_trigger_count
                })
        
        # Strong Team 1X2 recommendation
        if strong_should_bet:
            if favored_direction == "home" and odds.get('home', 0) > 0:
                # Simple probability estimate based on gap
                home_prob = 0.4 + min(0.3, gap_abs * 0.3)
                edge = get_edge_from_odds(odds['home'], home_prob)
                recommendations.append({
                    "name": f"{home_team} Win",
                    "odds": odds['home'],
                    "stake": strong_stake,
                    "edge": edge,
                    "type": "1X2",
                    "gap": efficiency_gap,
                    "gap_strength": gap_strength,
                    "reason": strong_reason
                })
            elif favored_direction == "away" and odds.get('away', 0) > 0:
                away_prob = 0.4 + min(0.3, gap_abs * 0.3)
                edge = get_edge_from_odds(odds['away'], away_prob)
                recommendations.append({
                    "name": f"{away_team} Win",
                    "odds": odds['away'],
                    "stake": strong_stake,
                    "edge": edge,
                    "type": "1X2",
                    "gap": efficiency_gap,
                    "gap_strength": gap_strength,
                    "reason": strong_reason
                })
        
        # Draw recommendation (only when small gap AND odds ≥ 3.20)
        if small_gap and odds.get('draw', 0) >= DRAW_MIN_ODDS:
            draw_prob = 0.35
            edge = get_edge_from_odds(odds['draw'], draw_prob)
            if edge > 0.02:
                recommendations.append({
                    "name": "Draw",
                    "odds": odds['draw'],
                    "stake": "0.5%",
                    "edge": edge,
                    "type": "Draw",
                    "small_gap": True
                })
        
        # Over 2.5 recommendation (very rare)
        if over_triggered and odds.get('over', 0) > 0:
            over_prob = calculate_simple_over_prob(total_xg)
            edge = get_edge_from_odds(odds['over'], over_prob)
            if edge > 0.03:
                recommendations.append({
                    "name": "Over 2.5 Goals",
                    "odds": odds['over'],
                    "stake": "0.5%",
                    "edge": edge,
                    "type": "Over",
                    "reason": over_reason
                })
        
        # Sort by edge
        recommendations.sort(key=lambda x: x['edge'], reverse=True)
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "total_xg": total_xg,
            "home_form": home_form,
            "away_form": away_form,
            "h2h_home": h2h_home,
            "h2h_away": h2h_away,
            "home_gd": home_gd,
            "away_gd": away_gd,
            "home_conv": home_conv,
            "away_conv": away_conv,
            "efficiency_gap": efficiency_gap,
            "gap_abs": gap_abs,
            "small_gap": small_gap,
            "btts_trigger_count": btts_trigger_count,
            "btts_triggers": btts_triggers,
            "btts_should_bet": btts_should_bet,
            "strong_should_bet": strong_should_bet,
            "no_bet_zone": no_bet_zone,
            "recommendations": recommendations,
            "has_bet": len(recommendations) > 0
        }

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet vFinal</h1>
        <p>Strong Signal + Safety Filter | Locked & Final</p>
    </div>
    """, unsafe_allow_html=True)
    
    predictor = GrokBetVFinal()
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Atletico Madrid")
        with col2:
            away_team = st.text_input("Away Team", "Barcelona")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 1.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.00, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 2.70, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 1.00, 0.05)
        
        # xG display
        home_xg_display = (home_scored + away_conceded) / 2
        away_xg_display = (away_scored + home_conceded) / 2
        st.caption(f"xG: {home_team} {home_xg_display:.2f} | {away_team} {away_xg_display:.2f} | Total: {home_xg_display + away_xg_display:.2f}")
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 60)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 87)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 1)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 0)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 5, 4)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, -2)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, -8)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 10)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 14)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 13)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 14)
        
        st.markdown("---")
        
        st.markdown("**Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 3.12, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 4.09, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 2.20, 0.05)
        
        with col19:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 1.42, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 3.00, 0.05)
        
        with col20:
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 1.39, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 3.00, 0.05)
        
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
            st.markdown(f"### 🎯 GrokBet vFinal")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            # KEY DATA
            st.markdown("**📊 KEY DATA:**")
            st.markdown(f"xG: {result['home_xg']:.2f} | {result['away_xg']:.2f} | Total {result['total_xg']:.2f}")
            st.markdown(f"Form: {home_form}% | {away_form}%")
            st.markdown(f"H2H: {h2h_home}-{h2h_draws}-{h2h_away}")
            st.markdown(f"GD: {home_gd} | {away_gd}")
            st.markdown(f"Conv: {home_conv}% | {away_conv}%")
            
            st.markdown("---")
            
            # EFFICIENCY GAP
            st.markdown("**⚡ EFFICIENCY GAP:**")
            gap_color = "🟢" if result['efficiency_gap'] > 0 else "🔴"
            fav_team = home_team if result['efficiency_gap'] > 0 else away_team
            st.markdown(f"{gap_color} **{result['efficiency_gap']:+.3f}** (favors **{fav_team}**)")
            
            if result['small_gap']:
                st.markdown(f"⚠️ **SMALL GAP:** |{result['efficiency_gap']:+.3f}| ≤ {SMALL_GAP_THRESHOLD} → No 1X2 side bets. Draw only if odds ≥ {DRAW_MIN_ODDS}.")
            
            st.markdown("---")
            
            # BET TYPE 1 TRIGGERS
            st.markdown("**📋 BET TYPE 1: 'One Team Unlikely to Score'**")
            if result['btts_trigger_count'] >= 2:
                st.markdown(f"✅ **TRIGGERED** ({result['btts_trigger_count']} triggers)")
                for t in result['btts_triggers']:
                    st.markdown(f"  • {t}")
                st.markdown(f"→ Recommended stake: {result['btts_trigger_count'] * 0.25 + 0.5 if result['btts_trigger_count'] == 2 else '1.0'}%")
            else:
                st.markdown(f"❌ **NOT TRIGGERED** ({result['btts_trigger_count']} triggers - need 2+)")
                if result['btts_triggers']:
                    for t in result['btts_triggers']:
                        st.markdown(f"  • {t}")
            
            st.markdown("---")
            
            # BET TYPE 2 TRIGGERS
            st.markdown("**📋 BET TYPE 2: 'Clear Stronger Team'**")
            if result['strong_should_bet'] and not result['small_gap']:
                st.markdown(f"✅ **TRIGGERED**")
                st.markdown(f"→ Gap {result['efficiency_gap']:+.3f} > {STRONG_GAP_THRESHOLD}")
            elif result['small_gap']:
                st.markdown(f"⚠️ **OVERRIDDEN** — Small gap rule prevents 1X2 side bets")
            else:
                st.markdown(f"❌ **NOT TRIGGERED**")
            
            st.markdown("---")
            
            # FINAL RECOMMENDATIONS
            if result['no_bet_zone']:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO BET ZONE</strong><br>
                    No triggers met. Skip this match entirely.
                </div>
                """, unsafe_allow_html=True)
            elif result['has_bet']:
                best = result['recommendations'][0]
                
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 BEST BET:</strong><br>
                    ✅ <strong>{best['name']}</strong> at {best['odds']:.2f}<br>
                    📊 Edge: +{best['edge']*100:.1f}% | Stake: <span class="stake-highlight">{best['stake']}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Reason
                if best['type'] == "BTTS No":
                    st.markdown(f"**Reason:** {result['btts_trigger_count']} triggers indicate one team unlikely to score.")
                elif best['type'] == "Under":
                    st.markdown(f"**Reason:** {result['btts_trigger_count']} triggers indicate low-scoring likely.")
                elif best['type'] == "1X2":
                    st.markdown(f"**Reason:** {best.get('reason', 'Strong gap + H2H/form difference')}")
                elif best['type'] == "Draw":
                    st.markdown(f"**Reason:** Small gap ({result['efficiency_gap']:+.3f}) + Draw odds ≥ {DRAW_MIN_ODDS}")
                elif best['type'] == "Over":
                    st.markdown(f"**Reason:** {best.get('reason', 'Rare Over trigger')}")
                
                # Secondary bets
                if len(result['recommendations']) > 1:
                    st.markdown("---")
                    st.markdown("**⚽ SECONDARY OPTIONS:**")
                    for m in result['recommendations'][1:3]:
                        st.markdown(f"• {m['name']} at {m['odds']:.2f} – Edge +{m['edge']*100:.1f}% | Stake {m['stake']}")
                
                # VERDICT
                st.markdown("---")
                if result['btts_trigger_count'] >= 2 and result['strong_should_bet'] and not result['small_gap']:
                    verdict = f"Both triggers active. Priority on {best['name']}."
                elif result['btts_trigger_count'] >= 2:
                    verdict = "BTTS/Under trigger active. Goals unlikely."
                elif result['strong_should_bet'] and not result['small_gap']:
                    verdict = f"Strong team trigger active. Back the favorite."
                else:
                    verdict = "Follow the recommendation above. Skip if uncertain."
                
                st.markdown(f"**📝 VERDICT:** {verdict}")
                
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets minimum edge threshold or odds requirements. Skip this match.
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
    st.caption("🎯 **GrokBet vFinal** | Strong Signal + Safety Filter | Locked & Final | No more changes")

if __name__ == "__main__":
    main()
