# grokbet_vfinal_universal.py
# GROKBET vFINAL – UNIVERSAL LOGIC
# 
# For every match, the same logic applies:
# 1. Check LOCK conditions (100% proven patterns)
# 2. If ANY LOCK true → LOCK → Stake 1.0%
# 3. If NO LOCK true → NO LOCK → Odds favorite → Stake 0.5%
# 4. Skip if odds favorite > 2.00 for goals markets

import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="GrokBet vFinal",
    page_icon="🎯",
    layout="centered",
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
    .result-lock {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-nolock {
        background: linear-gradient(135deg, #1e293b 0%, #3a2a1a 100%);
        border-left: 4px solid #f97316;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-primary {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a2a 100%);
        border-left: 4px solid #fbbf24;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-secondary {
        background: linear-gradient(135deg, #1e293b 0%, #1a3a4a 100%);
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
    .lock-badge {
        background: #10b981;
        color: #0f172a;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.7rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .nolock-badge {
        background: #f97316;
        color: #0f172a;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.7rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    hr {
        margin: 0.75rem 0;
        border-color: #334155;
    }
    .warning {
        background: #3a2a1a;
        border-left: 3px solid #fbbf24;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        font-size: 0.75rem;
        color: #fcd34d;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

ODDS_THRESHOLD_GOALS = 2.00
ODDS_THRESHOLD_1X2 = 2.00
DRAW_MIN_ODDS = 3.00
SMALL_GAP_THRESHOLD = 0.4

# Lock thresholds
WEAK_CONV_THRESHOLD = 10
WEAK_SCORED_THRESHOLD = 1.2
ELITE_DEFENSE_CONCEDED_THRESHOLD = 1.0
ELITE_DEFENSE_FORM_THRESHOLD = 60
HIGH_XG_THRESHOLD = 3.0
HIGH_CONV_THRESHOLD = 15
H2H_WIN_THRESHOLD = 3
ELITE_DEFENSE_CONCEDED_STRICT = 0.8
HIGH_FORM_THRESHOLD = 80
HIGH_FORM_CONV_THRESHOLD = 10

# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_efficiency(scored_avg, conceded_avg, form_pct, conv_pct):
    """Efficiency = (Scored × Conv%) − (Conceded × ((100 − Form%)/100))"""
    conv_decimal = conv_pct / 100.0
    form_decimal = form_pct / 100.0
    weakness_multiplier = 1.0 - form_decimal
    attack_score = scored_avg * conv_decimal
    defense_penalty = conceded_avg * weakness_multiplier
    return attack_score - defense_penalty

def get_odds_favorite(odds_home, odds_draw, odds_away):
    """Return the 1X2 favorite and their odds"""
    min_odds = min(odds_home, odds_draw, odds_away)
    if min_odds == odds_home:
        return "Home", odds_home
    elif min_odds == odds_draw:
        return "Draw", odds_draw
    else:
        return "Away", odds_away

def check_lock_conditions(home_team, away_team, home_conv, away_conv, home_scored, away_scored, 
                         home_conceded, away_conceded, home_form, away_form, total_xg, h2h_wins_for_favorite):
    """
    Check all lock conditions (100% proven patterns)
    Priority order as defined in the logic breakdown:
    LOCK #1: Weak attack → BTTS No
    LOCK #2: Elite defense → BTTS No
    LOCK #3: High xG → BTTS Yes + Over 2.5
    LOCK #4: H2H dominance → BTTS Yes + Over 2.5
    LOCK #5: High xG + High Conv → BTTS Yes + Over 2.5
    
    Returns: (lock_type, lock_reason, bet_over, bet_btts_yes, bet_btts_no, bet_under)
    """
    
    # LOCK #1: Weak attack → BTTS No
    # Condition: One team conv ≤ 10% AND that same team scored ≤ 1.2
    weak_attack_home = (home_conv <= WEAK_CONV_THRESHOLD and home_scored <= WEAK_SCORED_THRESHOLD)
    weak_attack_away = (away_conv <= WEAK_CONV_THRESHOLD and away_scored <= WEAK_SCORED_THRESHOLD)
    
    if weak_attack_home or weak_attack_away:
        weak_team = home_team if weak_attack_home else away_team
        return ("btts_no", f"🔒 LOCK #1: Weak attack — {weak_team} (conv {home_conv if weak_attack_home else away_conv}%, scored {home_scored if weak_attack_home else away_scored:.2f}) → BTTS No", 
                False, False, True, False)
    
    # LOCK #2: Elite defense → BTTS No
    # Condition: One team conceded ≤ 1.0 AND that team form ≥ 60%
    elite_defense_home = (home_conceded <= ELITE_DEFENSE_CONCEDED_THRESHOLD and home_form >= ELITE_DEFENSE_FORM_THRESHOLD)
    elite_defense_away = (away_conceded <= ELITE_DEFENSE_CONCEDED_THRESHOLD and away_form >= ELITE_DEFENSE_FORM_THRESHOLD)
    
    if elite_defense_home or elite_defense_away:
        elite_team = home_team if elite_defense_home else away_team
        return ("btts_no", f"🔒 LOCK #2: Elite defense — {elite_team} (conceded {home_conceded if elite_defense_home else away_conceded:.2f}, form {home_form if elite_defense_home else away_form}%) → BTTS No",
                False, False, True, False)
    
    # LOCK #3: High xG → BTTS Yes + Over 2.5
    # Condition: Total xG ≥ 3.0
    if total_xg >= HIGH_XG_THRESHOLD:
        return ("both", f"🔒 LOCK #3: High xG — Total xG {total_xg:.2f} ≥ 3.0 → BTTS Yes + Over 2.5",
                True, True, False, False)
    
    # LOCK #4: H2H dominance → BTTS Yes + Over 2.5
    # Condition: H2H wins ≥ 3 (last 5 matches)
    if h2h_wins_for_favorite >= H2H_WIN_THRESHOLD:
        return ("both", f"🔒 LOCK #4: H2H dominance — {h2h_wins_for_favorite} wins in last 5 H2H → BTTS Yes + Over 2.5",
                True, True, False, False)
    
    # LOCK #5: High xG + High Conv → BTTS Yes + Over 2.5
    # Condition: Total xG ≥ 3.0 AND (home conv ≥ 15% OR away conv ≥ 15%)
    if total_xg >= HIGH_XG_THRESHOLD and (home_conv >= HIGH_CONV_THRESHOLD or away_conv >= HIGH_CONV_THRESHOLD):
        high_conv_team = "Home" if home_conv >= HIGH_CONV_THRESHOLD else "Away"
        return ("both", f"🔒 LOCK #5: High xG + High conv — xG {total_xg:.2f} ≥ 3.0 and {high_conv_team} conv {max(home_conv, away_conv)}% ≥ 15% → BTTS Yes + Over 2.5",
                True, True, False, False)
    
    return None, None, False, False, False, False

def get_informational_warnings(home_team, away_team, home_conv, away_conv, home_scored, away_scored,
                               home_conceded, away_conceded, home_form, away_form):
    """
    Get informational warnings (no bet impact)
    Warning P1: Weak team cannot score 2+
    Warning P2: Elite defense may suppress goals
    Warning P3: High form team may overperform
    """
    warnings = []
    
    # Warning P1: Weak team cannot score 2+
    if home_conv <= WEAK_CONV_THRESHOLD and home_scored <= WEAK_SCORED_THRESHOLD:
        warnings.append(f"⚠️ {home_team} (conv {home_conv}%, scored {home_scored:.2f}) cannot score 2+ goals")
    if away_conv <= WEAK_CONV_THRESHOLD and away_scored <= WEAK_SCORED_THRESHOLD:
        warnings.append(f"⚠️ {away_team} (conv {away_conv}%, scored {away_scored:.2f}) cannot score 2+ goals")
    
    # Warning P2: Elite defense may suppress goals
    if home_conceded <= ELITE_DEFENSE_CONCEDED_STRICT:
        warnings.append(f"⚠️ {home_team} has elite defense (conceded {home_conceded:.2f}) — may suppress goals")
    if away_conceded <= ELITE_DEFENSE_CONCEDED_STRICT:
        warnings.append(f"⚠️ {away_team} has elite defense (conceded {away_conceded:.2f}) — may suppress goals")
    
    # Warning P3: High form team may overperform
    if home_form >= HIGH_FORM_THRESHOLD and home_conv >= HIGH_FORM_CONV_THRESHOLD:
        warnings.append(f"⚠️ {home_team} has high form ({home_form}%) — may overperform expectations")
    if away_form >= HIGH_FORM_THRESHOLD and away_conv >= HIGH_FORM_CONV_THRESHOLD:
        warnings.append(f"⚠️ {away_team} has high form ({away_form}%) — may overperform expectations")
    
    return warnings

# ============================================================================
# MAIN PREDICTOR
# ============================================================================

def get_best_bet(data, odds):
    """
    STEP 0: Input Data Required
    STEP 1: Calculate Derived Metrics (xG + Efficiency Gap)
    STEP 2: Check LOCK Conditions
    STEP 3: If NO LOCK → Fallback to Odds Favorite
    STEP 4: 1X2 Markets
    STEP 5: Add Informational Warnings
    STEP 6: Output Priority Order
    """
    
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
    
    # STEP 1: Calculate Derived Metrics
    home_xg = (home_scored + away_conceded) / 2
    away_xg = (away_scored + home_conceded) / 2
    total_xg = home_xg + away_xg
    
    # Calculate Efficiency Gap
    home_efficiency = calculate_efficiency(home_scored, home_conceded, home_form, home_conv)
    away_efficiency = calculate_efficiency(away_scored, away_conceded, away_form, away_conv)
    efficiency_gap = home_efficiency - away_efficiency
    gap_abs = abs(efficiency_gap)
    
    # Get 1X2 favorite for H2H check
    favorite_direction, favorite_odds = get_odds_favorite(odds['home'], odds['draw'], odds['away'])
    
    # Determine H2H wins for favorite (for LOCK #4)
    if favorite_direction == "Home":
        h2h_wins_for_favorite = h2h_home
        favored_team_name = home_team
    elif favorite_direction == "Away":
        h2h_wins_for_favorite = h2h_away
        favored_team_name = away_team
    else:
        h2h_wins_for_favorite = 0
        favored_team_name = "Draw"
    
    # STEP 2: Check LOCK Conditions (in priority order)
    lock_type, lock_reason, bet_over, bet_btts_yes, bet_btts_no, bet_under = check_lock_conditions(
        home_team, away_team, home_conv, away_conv, home_scored, away_scored,
        home_conceded, away_conceded, home_form, away_form, total_xg, h2h_wins_for_favorite
    )
    
    is_lock = lock_type is not None
    stake_lock = "1.0%" if is_lock else "0.5%"
    
    # Build recommendations
    recommendations = []
    
    # ========== GOALS MARKETS BASED ON LOCK ==========
    if is_lock:
        # LOCK bets (Priority 1)
        if lock_type == "btts_no":
            if odds['btts_no'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "BTTS No",
                    "odds": odds['btts_no'],
                    "stake": stake_lock,
                    "type": "BTTS",
                    "is_lock": True,
                    "priority": 1,
                    "reason": lock_reason
                })
        
        elif lock_type == "both":
            if odds['over'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "Over 2.5 Goals",
                    "odds": odds['over'],
                    "stake": stake_lock,
                    "type": "O/U",
                    "is_lock": True,
                    "priority": 1,
                    "reason": lock_reason
                })
            if odds['btts_yes'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "BTTS Yes",
                    "odds": odds['btts_yes'],
                    "stake": stake_lock,
                    "type": "BTTS",
                    "is_lock": True,
                    "priority": 1,
                    "reason": lock_reason
                })
    
    else:
        # STEP 3: NO LOCK → Fallback to Odds Favorite (Priority 2)
        # Over / Under 2.5
        if odds['over'] < odds['under'] and odds['over'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Over 2.5 Goals",
                "odds": odds['over'],
                "stake": stake_lock,
                "type": "O/U",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['over']}"
            })
        elif odds['under'] < odds['over'] and odds['under'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Under 2.5 Goals",
                "odds": odds['under'],
                "stake": stake_lock,
                "type": "O/U",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['under']}"
            })
        
        # BTTS Yes / No
        if odds['btts_yes'] < odds['btts_no'] and odds['btts_yes'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS Yes",
                "odds": odds['btts_yes'],
                "stake": stake_lock,
                "type": "BTTS",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['btts_yes']}"
            })
        elif odds['btts_no'] < odds['btts_yes'] and odds['btts_no'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS No",
                "odds": odds['btts_no'],
                "stake": stake_lock,
                "type": "BTTS",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['btts_no']}"
            })
    
    # STEP 4: 1X2 Markets (Separate, Lower Priority - Priority 3)
    # Winner Bet
    if favorite_direction != "Draw" and favorite_odds <= ODDS_THRESHOLD_1X2:
        recommendations.append({
            "name": f"{favored_team_name} Win",
            "odds": favorite_odds,
            "stake": "0.5%",
            "type": "1X2",
            "is_lock": False,
            "priority": 3,
            "reason": f"Odds favorite at {favorite_odds}"
        })
    
    # Draw Bet (Special Case: small gap + odds ≥ 3.00)
    if gap_abs <= SMALL_GAP_THRESHOLD and odds['draw'] >= DRAW_MIN_ODDS:
        recommendations.append({
            "name": "Draw",
            "odds": odds['draw'],
            "stake": "0.5%",
            "type": "Draw",
            "is_lock": False,
            "priority": 3,
            "reason": f"Small gap ({efficiency_gap:+.3f}) + Draw odds ≥ {DRAW_MIN_ODDS}"
        })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x.get('priority', 3))
    
    # STEP 5: Add Informational Warnings
    warnings = get_informational_warnings(
        home_team, away_team, home_conv, away_conv, home_scored, away_scored,
        home_conceded, away_conceded, home_form, away_form
    )
    
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
        "home_conv": home_conv,
        "away_conv": away_conv,
        "home_scored": home_scored,
        "away_scored": away_scored,
        "home_conceded": home_conceded,
        "away_conceded": away_conceded,
        "efficiency_gap": efficiency_gap,
        "gap_abs": gap_abs,
        "is_lock": is_lock,
        "lock_type": lock_type,
        "lock_reason": lock_reason,
        "warnings": warnings,
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
        <p>Universal Logic | LOCK Indicator | 100% Proven Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Kolos")
        with col2:
            away_team = st.text_input("Away Team", "Metalist 1925")
        
        st.markdown("---")
        
        st.markdown("**📊 Scoring Averages**")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored Avg", 0.0, 3.0, 0.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded Avg", 0.0, 3.0, 1.70, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored Avg", 0.0, 3.0, 1.10, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded Avg", 0.0, 3.0, 0.60, 0.05)
        
        home_xg_display = (home_scored + away_conceded) / 2
        away_xg_display = (away_scored + home_conceded) / 2
        st.caption(f"xG: {home_team} {home_xg_display:.2f} | {away_team} {away_xg_display:.2f} | Total: {home_xg_display + away_xg_display:.2f}")
        
        st.markdown("---")
        
        st.markdown("**📈 Team Form**")
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 47)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 80)
        
        st.markdown("---")
        
        st.markdown("**🆚 Head-to-Head (Last 5 Matches)**")
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("Home Wins", 0, 5, 1)
        with col10:
            h2h_draws = st.number_input("Draws", 0, 5, 2)
        with col11:
            h2h_away = st.number_input("Away Wins", 0, 5, 2)
        
        st.markdown("---")
        
        st.markdown("**🎯 Conversion Rates**")
        col16, col17 = st.columns(2)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 10)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 11)
        
        st.markdown("---")
        
        st.markdown("**💰 Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            st.markdown("**1X2**")
            odds_home = st.number_input("Home", 0.0, 10.0, 3.10, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 2.90, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 2.45, 0.05)
        
        with col19:
            st.markdown("**Over/Under 2.5**")
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 2.80, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 1.37, 0.05)
        
        with col20:
            st.markdown("**BTTS**")
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 2.20, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 1.53, 0.05)
        
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
                'h2h_away': h2h_away,
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
            
            result = get_best_bet(data, odds)
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 GrokBet vFinal")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            # STEP 1: Display Derived Metrics
            st.markdown("**📊 STEP 1: DERIVED METRICS**")
            st.markdown(f"xG: {result['home_xg']:.2f} | {result['away_xg']:.2f} | **Total {result['total_xg']:.2f}**")
            st.markdown(f"Form: {home_form}% | {away_form}%")
            st.markdown(f"H2H: {h2h_home}-{h2h_draws}-{h2h_away}")
            st.markdown(f"Conv Rate: {home_conv}% | {away_conv}%")
            st.markdown(f"Scored Avg: {home_scored:.2f} | {away_scored:.2f}")
            st.markdown(f"Conceded Avg: {home_conceded:.2f} | {away_conceded:.2f}")
            
            st.markdown("---")
            
            st.markdown("**⚡ EFFICIENCY GAP:**")
            gap_color = "🟢" if result['efficiency_gap'] > 0 else "🔴"
            fav_team = home_team if result['efficiency_gap'] > 0 else away_team
            st.markdown(f"{gap_color} **{result['efficiency_gap']:+.3f}** (favors **{fav_team}**)")
            
            st.markdown("---")
            
            # STEP 2: LOCK / NO LOCK Indicator
            if result['is_lock']:
                st.markdown(f"""
                <div class="result-lock">
                    <strong>🔒 STEP 2: LOCK TRIGGERED</strong> <span class="lock-badge">100% PROVEN</span><br>
                    {result['lock_reason']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nolock">
                    <strong>⚠️ STEP 2: NO LOCK</strong> <span class="nolock-badge">FALLBACK TO ODDS FAVORITE</span><br>
                    No lock condition met. Falling back to odds favorite (STEP 3).
                </div>
                """, unsafe_allow_html=True)
            
            # STEP 5: Informational Warnings
            if result['warnings']:
                st.markdown("---")
                st.markdown("**📢 STEP 5: INFORMATIONAL WARNINGS**")
                for warning in result['warnings']:
                    st.markdown(f"<div class='warning'>{warning}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # STEP 6: Output Priority Order
            if result['has_bet']:
                primary = result['recommendations'][0]
                lock_badge = " 🔒 LOCK (Priority 1)" if primary.get('is_lock', False) else " ⚠️ FALLBACK (Priority 2)" if primary.get('priority') == 2 else " (Priority 3)"
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 STEP 6: BEST BET</strong><br>
                    ✅ <strong>{primary['name']}{lock_badge}</strong> at {primary['odds']:.2f}<br>
                    📊 Stake: <span class="stake-highlight">{primary['stake']}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Reason:** {primary['reason']}")
                
                if len(result['recommendations']) > 1:
                    st.markdown("---")
                    st.markdown("**⚽ SECONDARY OPTIONS:**")
                    for bet in result['recommendations'][1:4]:
                        lock_badge = " 🔒 LOCK" if bet.get('is_lock', False) else ""
                        st.markdown(f"• {bet['name']}{lock_badge} at {bet['odds']:.2f} – Stake {bet['stake']}")
                
                st.markdown("---")
                if result['is_lock']:
                    st.markdown("**📝 VERDICT:** 🔒 LOCK bet — 100% proven pattern. High confidence. (Stake: 1.0%)")
                else:
                    st.markdown("**📝 VERDICT:** ⚠️ NO LOCK — Betting odds favorite. Standard confidence (87% historical). (Stake: 0.5%)")
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets the odds threshold (≤ 2.00 for goals markets, ≤ 2.00 for 1X2, or draw with gap ≤ 0.4 and odds ≥ 3.00). Skip this match.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet vFinal** | Universal Logic | LOCK (1.0% stake) | NO LOCK → Odds Favorite (0.5% stake)")

if __name__ == "__main__":
    main()
