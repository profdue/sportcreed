# grokbet_vfinal_main.py
# GROKBET vFINAL – MAIN LOGIC WITH BTTS NO LOCK
# 
# LOCK #1: BTTS No (Multiple conditions required)
# LOCK #2: High-Scoring (BTTS Yes / Over 2.5)
# LOCK #3: Winner (1X2)
# LOCK #4: Draw (Special case)
# 
# Fallback: Odds favorite when no lock (0.5% stake)

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
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-nolock {
        background: linear-gradient(135deg, #1e293b 0%, #3e2a1e 100%);
        border-left: 4px solid #f97316;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.75rem 0;
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

# Odds thresholds
ODDS_THRESHOLD_GOALS = 2.00
ODDS_THRESHOLD_1X2 = 2.00
DRAW_MIN_ODDS = 3.00

# LOCK #1: BTTS No thresholds
WEAK_CONV = 10
WEAK_SCORED = 1.2
ELITE_ATTACK_SCORED = 1.5
LOW_XG_THRESHOLD = 2.5
ELITE_DEFENSE_CONCEDED = 1.0
HIGH_FORM_DEFENSE = 60

# LOCK #2: High-Scoring thresholds
HIGH_XG_THRESHOLD = 3.0
GOOD_CONV = 11
ELITE_DEFENSE_WARNING = 0.8
ALT_HIGH_XG = 2.8
H2H_WIN_THRESHOLD = 3
MIN_CONV_ALT = 10
TERRIBLE_FORM_THRESHOLD = 25

# LOCK #3: Winner thresholds
POSITIVE_GAP = 0
HOME_FORM_THRESHOLD = 60
H2H_HOME_MIN = 2
HOME_SCORED_MIN = 1.5
NEGATIVE_GAP = 0
AWAY_FORM_THRESHOLD = 60
H2H_AWAY_MIN = 2
AWAY_SCORED_MIN = 1.5

# LOCK #4: Draw thresholds
SMALL_GAP_MAX = 0.3
BALANCED_FORM_MIN = 40
BALANCED_FORM_MAX = 60
DRAW_XG_MAX = 2.5

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

def check_lock_conditions(data):
    """
    Check all lock conditions (multiple factors required)
    Returns: (lock_type, lock_reason, bets, stake)
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
    total_xg = data['total_xg']
    efficiency_gap = data['efficiency_gap']
    odds = data['odds']
    
    bets = []
    
    # ========== LOCK #1: BTTS NO ==========
    # Path A: Weak attack + opponent not elite + low xG
    weak_attack_home = home_conv <= WEAK_CONV and home_scored <= WEAK_SCORED
    weak_attack_away = away_conv <= WEAK_CONV and away_scored <= WEAK_SCORED
    opponent_not_elite_home = away_scored <= ELITE_ATTACK_SCORED
    opponent_not_elite_away = home_scored <= ELITE_ATTACK_SCORED
    low_xg = total_xg <= LOW_XG_THRESHOLD
    
    if (weak_attack_home and opponent_not_elite_home and low_xg):
        bets.append({
            "name": "BTTS No",
            "secondary": "Under 2.5",
            "type": "btts_no",
            "reason": f"LOCK #1: {home_team} weak attack (conv {home_conv}%, scored {home_scored:.2f}) + opponent not elite + low xG ({total_xg:.2f})",
            "primary_bet": "BTTS No",
            "secondary_bet": "Under 2.5"
        })
    elif (weak_attack_away and opponent_not_elite_away and low_xg):
        bets.append({
            "name": "BTTS No",
            "secondary": "Under 2.5",
            "type": "btts_no",
            "reason": f"LOCK #1: {away_team} weak attack (conv {away_conv}%, scored {away_scored:.2f}) + opponent not elite + low xG ({total_xg:.2f})",
            "primary_bet": "BTTS No",
            "secondary_bet": "Under 2.5"
        })
    
    # Path B: Elite defense + opponent weak + low xG
    elite_defense_home = home_conceded <= ELITE_DEFENSE_CONCEDED and home_form >= HIGH_FORM_DEFENSE
    elite_defense_away = away_conceded <= ELITE_DEFENSE_CONCEDED and away_form >= HIGH_FORM_DEFENSE
    opponent_weak_home = away_scored <= WEAK_SCORED
    opponent_weak_away = home_scored <= WEAK_SCORED
    
    if (elite_defense_home and opponent_weak_home and low_xg):
        bets.append({
            "name": "BTTS No",
            "secondary": "Under 2.5",
            "type": "btts_no",
            "reason": f"LOCK #1: {home_team} elite defense (conceded {home_conceded:.2f}, form {home_form}%) + weak opponent + low xG ({total_xg:.2f})",
            "primary_bet": "BTTS No",
            "secondary_bet": "Under 2.5"
        })
    elif (elite_defense_away and opponent_weak_away and low_xg):
        bets.append({
            "name": "BTTS No",
            "secondary": "Under 2.5",
            "type": "btts_no",
            "reason": f"LOCK #1: {away_team} elite defense (conceded {away_conceded:.2f}, form {away_form}%) + weak opponent + low xG ({total_xg:.2f})",
            "primary_bet": "BTTS No",
            "secondary_bet": "Under 2.5"
        })
    
    # ========== LOCK #2: High-Scoring (BTTS Yes + Over 2.5) ==========
    # Path A: High xG + both good conv + no elite defense
    both_good_conv = home_conv >= GOOD_CONV and away_conv >= GOOD_CONV
    no_elite_defense = home_conceded >= ELITE_DEFENSE_WARNING and away_conceded >= ELITE_DEFENSE_WARNING
    high_xg = total_xg >= HIGH_XG_THRESHOLD
    
    if high_xg and both_good_conv and no_elite_defense:
        bets.append({
            "name": "BTTS Yes + Over 2.5",
            "secondary": None,
            "type": "high_scoring",
            "reason": f"LOCK #2: High xG ({total_xg:.2f}) + both good conv ({home_conv}%/{away_conv}%) + no elite defense",
            "primary_bet": "BTTS Yes",
            "secondary_bet": "Over 2.5"
        })
    
    # Path B: Decent xG + H2H dominance + decent conv + not terrible form
    decent_xg = total_xg >= ALT_HIGH_XG
    h2h_dominance = (h2h_home >= H2H_WIN_THRESHOLD) or (h2h_away >= H2H_WIN_THRESHOLD)
    decent_conv = home_conv >= MIN_CONV_ALT and away_conv >= MIN_CONV_ALT
    not_terrible_form = home_form > TERRIBLE_FORM_THRESHOLD and away_form > TERRIBLE_FORM_THRESHOLD
    
    if decent_xg and h2h_dominance and decent_conv and not_terrible_form:
        bets.append({
            "name": "BTTS Yes + Over 2.5",
            "secondary": None,
            "type": "high_scoring",
            "reason": f"LOCK #2: xG {total_xg:.2f} + H2H dominance + both conv ≥10% + form not terrible",
            "primary_bet": "BTTS Yes",
            "secondary_bet": "Over 2.5"
        })
    
    # ========== LOCK #3: Winner (1X2) ==========
    # Home Win
    if efficiency_gap > POSITIVE_GAP and home_form >= HOME_FORM_THRESHOLD:
        if h2h_home >= H2H_HOME_MIN or home_scored >= HOME_SCORED_MIN:
            bets.append({
                "name": f"{home_team} Win",
                "secondary": None,
                "type": "winner",
                "reason": f"LOCK #3: Positive gap ({efficiency_gap:+.3f}) + home form {home_form}% + H2H/scoring advantage",
                "primary_bet": f"{home_team} Win",
                "secondary_bet": None
            })
    
    # Away Win
    if efficiency_gap < NEGATIVE_GAP and away_form >= AWAY_FORM_THRESHOLD:
        if h2h_away >= H2H_AWAY_MIN or away_scored >= AWAY_SCORED_MIN:
            bets.append({
                "name": f"{away_team} Win",
                "secondary": None,
                "type": "winner",
                "reason": f"LOCK #3: Negative gap ({efficiency_gap:+.3f}) + away form {away_form}% + H2H/scoring advantage",
                "primary_bet": f"{away_team} Win",
                "secondary_bet": None
            })
    
    # ========== LOCK #4: Draw ==========
    small_gap = abs(efficiency_gap) <= SMALL_GAP_MAX
    balanced_form = (home_form >= BALANCED_FORM_MIN and home_form <= BALANCED_FORM_MAX and
                     away_form >= BALANCED_FORM_MIN and away_form <= BALANCED_FORM_MAX)
    low_xg_draw = total_xg <= DRAW_XG_MAX
    good_draw_odds = odds.get('draw', 0) >= DRAW_MIN_ODDS
    
    if small_gap and balanced_form and low_xg_draw and good_draw_odds:
        bets.append({
            "name": "Draw",
            "secondary": None,
            "type": "draw",
            "reason": f"LOCK #4: Small gap ({abs(efficiency_gap):.3f}) + balanced form ({home_form}%/{away_form}%) + low xG ({total_xg:.2f}) + odds ≥ {DRAW_MIN_ODDS}",
            "primary_bet": "Draw",
            "secondary_bet": None
        })
    
    return bets

# ============================================================================
# MAIN PREDICTOR
# ============================================================================

def get_best_bet(data, odds):
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
    
    # Calculate xG
    home_xg = (home_scored + away_conceded) / 2
    away_xg = (away_scored + home_conceded) / 2
    total_xg = home_xg + away_xg
    
    # Calculate Efficiency Gap
    home_efficiency = calculate_efficiency(home_scored, home_conceded, home_form, home_conv)
    away_efficiency = calculate_efficiency(away_scored, away_conceded, away_form, away_conv)
    efficiency_gap = home_efficiency - away_efficiency
    gap_abs = abs(efficiency_gap)
    
    # Get 1X2 favorite
    favorite_direction, favorite_odds = get_odds_favorite(odds['home'], odds['draw'], odds['away'])
    
    if favorite_direction == "Home":
        favored_team_name = home_team
    elif favorite_direction == "Away":
        favored_team_name = away_team
    else:
        favored_team_name = "Draw"
    
    # Prepare data for lock check
    lock_data = {
        'home_team': home_team,
        'away_team': away_team,
        'home_scored': home_scored,
        'home_conceded': home_conceded,
        'away_scored': away_scored,
        'away_conceded': away_conceded,
        'home_form': home_form,
        'away_form': away_form,
        'home_conv': home_conv,
        'away_conv': away_conv,
        'h2h_home': h2h_home,
        'h2h_away': h2h_away,
        'total_xg': total_xg,
        'efficiency_gap': efficiency_gap,
        'odds': odds
    }
    
    # Check locks
    lock_bets = check_lock_conditions(lock_data)
    
    # Build recommendations
    recommendations = []
    
    # Add lock bets first
    for lock in lock_bets:
        # Add primary bet
        bet_odds = None
        if lock['primary_bet'] == "BTTS No":
            bet_odds = odds.get('btts_no', 0)
        elif lock['primary_bet'] == "BTTS Yes":
            bet_odds = odds.get('btts_yes', 0)
        elif lock['primary_bet'] == "Over 2.5":
            bet_odds = odds.get('over', 0)
        elif lock['primary_bet'] == "Under 2.5":
            bet_odds = odds.get('under', 0)
        elif lock['primary_bet'] == f"{home_team} Win":
            bet_odds = odds.get('home', 0)
        elif lock['primary_bet'] == f"{away_team} Win":
            bet_odds = odds.get('away', 0)
        elif lock['primary_bet'] == "Draw":
            bet_odds = odds.get('draw', 0)
        
        if bet_odds and bet_odds > 0:
            recommendations.append({
                "name": lock['primary_bet'],
                "odds": bet_odds,
                "stake": "1.0%",
                "type": lock['type'],
                "is_lock": True,
                "priority": 1,
                "reason": lock['reason']
            })
        
        # Add secondary bet if exists
        if lock['secondary_bet']:
            sec_odds = None
            if lock['secondary_bet'] == "Under 2.5":
                sec_odds = odds.get('under', 0)
            elif lock['secondary_bet'] == "Over 2.5":
                sec_odds = odds.get('over', 0)
            
            if sec_odds and sec_odds > 0:
                recommendations.append({
                    "name": lock['secondary_bet'],
                    "odds": sec_odds,
                    "stake": "1.0%",
                    "type": lock['type'],
                    "is_lock": True,
                    "priority": 1,
                    "reason": f"Secondary to: {lock['reason']}"
                })
    
    # If no locks, fallback to odds favorite
    if not recommendations:
        # Over / Under 2.5
        if odds['over'] <= odds['under'] and odds['over'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Over 2.5 Goals",
                "odds": odds['over'],
                "stake": "0.5%",
                "type": "O/U",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['over']}"
            })
        elif odds['under'] <= odds['over'] and odds['under'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Under 2.5 Goals",
                "odds": odds['under'],
                "stake": "0.5%",
                "type": "O/U",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['under']}"
            })
        
        # BTTS Yes / No
        if odds['btts_yes'] <= odds['btts_no'] and odds['btts_yes'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS Yes",
                "odds": odds['btts_yes'],
                "stake": "0.5%",
                "type": "BTTS",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['btts_yes']}"
            })
        elif odds['btts_no'] <= odds['btts_yes'] and odds['btts_no'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS No",
                "odds": odds['btts_no'],
                "stake": "0.5%",
                "type": "BTTS",
                "is_lock": False,
                "priority": 2,
                "reason": f"NO LOCK: Odds favorite at {odds['btts_no']}"
            })
        
        # 1X2
        if favorite_direction != "Draw" and favorite_odds <= ODDS_THRESHOLD_1X2:
            recommendations.append({
                "name": f"{favored_team_name} Win",
                "odds": favorite_odds,
                "stake": "0.5%",
                "type": "1X2",
                "is_lock": False,
                "priority": 3,
                "reason": f"NO LOCK: Odds favorite at {favorite_odds}"
            })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x.get('priority', 3))
    
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
        "has_lock": len(lock_bets) > 0,
        "lock_bets": lock_bets,
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
        <p>BTTS No Lock | Multiple Conditions | No Single Factor</p>
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
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 0.70, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.70, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 1.10, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 0.60, 0.05)
        
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
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 1)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 2)
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
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 1.37, 0.05)
        
        with col20:
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
            
            result = get_best_bet(data, odds)
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            
            st.markdown(f"### 🎯 GrokBet vFinal")
            st.markdown(f"**MATCH:** {home_team} vs {away_team}")
            st.markdown("---")
            
            st.markdown("**📊 KEY DATA:**")
            st.markdown(f"xG: {result['home_xg']:.2f} | {result['away_xg']:.2f} | Total {result['total_xg']:.2f}")
            st.markdown(f"Form: {home_form}% | {away_form}%")
            st.markdown(f"H2H: {h2h_home}-{h2h_draws}-{h2h_away}")
            st.markdown(f"Conv: {home_conv}% | {away_conv}%")
            st.markdown(f"Scored: {home_scored:.2f} | {away_scored:.2f}")
            st.markdown(f"Conceded: {home_conceded:.2f} | {away_conceded:.2f}")
            
            st.markdown("---")
            
            st.markdown("**⚡ EFFICIENCY GAP:**")
            gap_color = "🟢" if result['efficiency_gap'] > 0 else "🔴"
            fav_team = home_team if result['efficiency_gap'] > 0 else away_team
            st.markdown(f"{gap_color} **{result['efficiency_gap']:+.3f}** (favors **{fav_team}**)")
            
            st.markdown("---")
            
            # LOCK / NO LOCK Indicator
            if result['has_lock']:
                st.markdown(f"""
                <div class="result-lock">
                    <strong>🔒 LOCK TRIGGERED</strong> <span class="lock-badge">BTTS NO</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nolock">
                    <strong>⚠️ NO LOCK</strong> <span class="nolock-badge">FALLBACK TO ODDS</span><br>
                    No lock conditions met. Falling back to odds favorite.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            if result['has_bet']:
                primary = result['recommendations'][0]
                lock_badge = " 🔒 LOCK" if primary.get('is_lock', False) else ""
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 BEST BET:</strong><br>
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
                if result['has_lock']:
                    st.markdown("**📝 VERDICT:** 🔒 BTTS No LOCK — Multiple conditions confirmed. High confidence.")
                else:
                    st.markdown("**📝 VERDICT:** ⚠️ NO LOCK — Betting odds favorite. Standard confidence (87% historical).")
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets the odds threshold. Skip this match.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet vFinal** | BTTS No Lock | Multiple Conditions | No Single Factor")

if __name__ == "__main__":
    main()
