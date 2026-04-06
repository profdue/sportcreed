# grokbet_vfinal_simplified.py
# GROKBET vFINAL – SIMPLIFIED LOGIC
# 
# Core Principle: Bet odds favorite for goals markets, unless a proven override applies.
# 
# Proven Overrides (100% on 15+ matches):
# 1. Both conv ≤ 10% AND both scored ≤ 1.2 → BTTS No / Under 2.5
# 2. Total xG ≥ 3.0 AND (home conv ≥ 15% OR away conv ≥ 15%) → Over 2.5 + BTTS Yes
# 
# Otherwise: Bet odds favorite for Over/Under and BTTS (if odds ≤ 2.00)
# 1X2: Bet odds favorite if odds ≤ 2.00, or Draw if small gap + odds ≥ 3.00
# 
# Stake: 0.75% if override active, 0.5% if no override

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
    .result-override {
        background: linear-gradient(135deg, #1e293b 0%, #3e2a1e 100%);
        border-left: 4px solid #f97316;
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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

ODDS_THRESHOLD_GOALS = 2.00
ODDS_THRESHOLD_1X2 = 2.00
DRAW_MIN_ODDS = 3.00
SMALL_GAP_THRESHOLD = 0.4

# Override thresholds
WEAK_CONV_THRESHOLD = 10
WEAK_SCORED_THRESHOLD = 1.2
HIGH_XG_THRESHOLD = 3.0
HIGH_CONV_THRESHOLD = 15

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

def check_overrides(home_conv, away_conv, home_scored, away_scored, total_xg):
    """
    Check proven overrides for goals markets.
    Returns: (override_type, override_reason, bet_over, bet_btts_yes)
    """
    # PRIORITY 1: Both weak attacks → BTTS No / Under
    if (home_conv <= WEAK_CONV_THRESHOLD and away_conv <= WEAK_CONV_THRESHOLD and
        home_scored <= WEAK_SCORED_THRESHOLD and away_scored <= WEAK_SCORED_THRESHOLD):
        return "btts_no", f"Both conv ≤ 10% ({home_conv}%/{away_conv}%) AND both scored ≤ 1.2 ({home_scored:.2f}/{away_scored:.2f}) → goals unlikely", False, False
    
    # PRIORITY 2: High xG + high conversion → Over + BTTS Yes
    if total_xg >= HIGH_XG_THRESHOLD and (home_conv >= HIGH_CONV_THRESHOLD or away_conv >= HIGH_CONV_THRESHOLD):
        high_conv_team = "Home" if home_conv >= HIGH_CONV_THRESHOLD else "Away"
        return "both", f"xG {total_xg:.2f} ≥ 3.0 AND {high_conv_team} conv {max(home_conv, away_conv)}% ≥ 15%", True, True
    
    return None, None, False, False

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
    
    # Calculate Efficiency Gap (for display and small gap check only)
    home_efficiency = calculate_efficiency(home_scored, home_conceded, home_form, home_conv)
    away_efficiency = calculate_efficiency(away_scored, away_conceded, away_form, away_conv)
    efficiency_gap = home_efficiency - away_efficiency
    gap_abs = abs(efficiency_gap)
    
    # Check overrides
    override_type, override_reason, bet_over, bet_btts_yes = check_overrides(
        home_conv, away_conv, home_scored, away_scored, total_xg
    )
    
    override_active = override_type is not None
    
    # Build recommendations
    recommendations = []
    
    # ========== GOALS MARKETS ==========
    if override_active:
        stake = "0.75%"
        
        if override_type == "btts_no":
            # Bet BTTS No
            if odds['btts_no'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "BTTS No",
                    "odds": odds['btts_no'],
                    "stake": stake,
                    "type": "BTTS",
                    "override": True,
                    "priority": 1,
                    "reason": f"Override: {override_reason} → betting BTTS No"
                })
            # Also consider Under 2.5 if odds are reasonable
            if odds['under'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "Under 2.5 Goals",
                    "odds": odds['under'],
                    "stake": stake,
                    "type": "O/U",
                    "override": True,
                    "priority": 1,
                    "reason": f"Override: {override_reason} → betting Under 2.5"
                })
        
        elif override_type == "both":
            # Bet Over 2.5
            if odds['over'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "Over 2.5 Goals",
                    "odds": odds['over'],
                    "stake": stake,
                    "type": "O/U",
                    "override": True,
                    "priority": 1,
                    "reason": f"Override: {override_reason} → betting Over 2.5"
                })
            # Bet BTTS Yes
            if odds['btts_yes'] <= ODDS_THRESHOLD_GOALS:
                recommendations.append({
                    "name": "BTTS Yes",
                    "odds": odds['btts_yes'],
                    "stake": stake,
                    "type": "BTTS",
                    "override": True,
                    "priority": 1,
                    "reason": f"Override: {override_reason} → betting BTTS Yes"
                })
    
    else:
        # No override: bet odds favorite for goals markets
        stake = "0.5%"
        
        # Over / Under 2.5
        if odds['over'] <= odds['under'] and odds['over'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Over 2.5 Goals",
                "odds": odds['over'],
                "stake": stake,
                "type": "O/U",
                "override": False,
                "priority": 2,
                "reason": f"Odds favorite at {odds['over']}"
            })
        elif odds['under'] <= odds['over'] and odds['under'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "Under 2.5 Goals",
                "odds": odds['under'],
                "stake": stake,
                "type": "O/U",
                "override": False,
                "priority": 2,
                "reason": f"Odds favorite at {odds['under']}"
            })
        
        # BTTS Yes / No
        if odds['btts_yes'] <= odds['btts_no'] and odds['btts_yes'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS Yes",
                "odds": odds['btts_yes'],
                "stake": stake,
                "type": "BTTS",
                "override": False,
                "priority": 2,
                "reason": f"Odds favorite at {odds['btts_yes']}"
            })
        elif odds['btts_no'] <= odds['btts_yes'] and odds['btts_no'] <= ODDS_THRESHOLD_GOALS:
            recommendations.append({
                "name": "BTTS No",
                "odds": odds['btts_no'],
                "stake": stake,
                "type": "BTTS",
                "override": False,
                "priority": 2,
                "reason": f"Odds favorite at {odds['btts_no']}"
            })
    
    # ========== 1X2 MARKET ==========
    favorite_direction, favorite_odds = get_odds_favorite(odds['home'], odds['draw'], odds['away'])
    
    if favorite_direction != "Draw" and favorite_odds <= ODDS_THRESHOLD_1X2:
        stake = "0.5%"
        recommendations.append({
            "name": f"{home_team if favorite_direction == 'Home' else away_team} Win",
            "odds": favorite_odds,
            "stake": stake,
            "type": "1X2",
            "override": False,
            "priority": 3,
            "reason": f"Odds favorite at {favorite_odds}"
        })
    
    # ========== DRAW (small gap only) ==========
    if gap_abs <= SMALL_GAP_THRESHOLD and odds['draw'] >= DRAW_MIN_ODDS:
        recommendations.append({
            "name": "Draw",
            "odds": odds['draw'],
            "stake": "0.5%",
            "type": "Draw",
            "override": False,
            "priority": 3,
            "reason": f"Small gap ({efficiency_gap:+.3f}) + Draw odds ≥ {DRAW_MIN_ODDS}"
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
        "efficiency_gap": efficiency_gap,
        "gap_abs": gap_abs,
        "override_active": override_active,
        "override_type": override_type,
        "override_reason": override_reason,
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
        <p>Simplified | Odds Favorite + Proven Overrides | Locked</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.text_input("Home Team", "Girona")
        with col2:
            away_team = st.text_input("Away Team", "Villarreal")
        
        st.markdown("---")
        
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            home_scored = st.number_input(f"{home_team} Scored", 0.0, 3.0, 1.10, 0.05)
        with col4:
            home_conceded = st.number_input(f"{home_team} Conceded", 0.0, 3.0, 1.50, 0.05)
        with col5:
            away_scored = st.number_input(f"{away_team} Scored", 0.0, 3.0, 1.90, 0.05)
        with col6:
            away_conceded = st.number_input(f"{away_team} Conceded", 0.0, 3.0, 1.20, 0.05)
        
        home_xg_display = (home_scored + away_conceded) / 2
        away_xg_display = (away_scored + home_conceded) / 2
        st.caption(f"xG: {home_team} {home_xg_display:.2f} | {away_team} {away_xg_display:.2f} | Total: {home_xg_display + away_xg_display:.2f}")
        
        st.markdown("---")
        
        col7, col8 = st.columns(2)
        with col7:
            home_form = st.number_input(f"{home_team} Form %", 0, 100, 33)
        with col8:
            away_form = st.number_input(f"{away_team} Form %", 0, 100, 67)
        
        col9, col10, col11 = st.columns(3)
        with col9:
            h2h_home = st.number_input("H2H Home Wins (last 5)", 0, 5, 1)
        with col10:
            h2h_draws = st.number_input("H2H Draws", 0, 5, 1)
        with col11:
            h2h_away = st.number_input("H2H Away Wins", 0, 5, 3)
        
        st.markdown("---")
        
        col12, col13 = st.columns(2)
        with col12:
            home_gd = st.number_input(f"{home_team} GD", -50, 50, -13)
        with col13:
            away_gd = st.number_input(f"{away_team} GD", -50, 50, 20)
        
        st.markdown("---")
        
        col14, col15, col16, col17 = st.columns(4)
        with col14:
            home_top = st.number_input(f"{home_team} Top Scorer", 0, 30, 9)
        with col15:
            away_top = st.number_input(f"{away_team} Top Scorer", 0, 30, 9)
        with col16:
            home_conv = st.number_input(f"{home_team} Conv %", 0, 100, 11)
        with col17:
            away_conv = st.number_input(f"{away_team} Conv %", 0, 100, 17)
        
        st.markdown("---")
        
        st.markdown("**Odds (from SportyBet screenshot)**")
        col18, col19, col20 = st.columns(3)
        with col18:
            odds_home = st.number_input("Home", 0.0, 10.0, 3.19, 0.05)
            odds_draw = st.number_input("Draw", 0.0, 10.0, 3.56, 0.05)
            odds_away = st.number_input("Away", 0.0, 10.0, 2.35, 0.05)
        
        with col19:
            odds_over = st.number_input("Over 2.5", 0.0, 10.0, 1.70, 0.05)
            odds_under = st.number_input("Under 2.5", 0.0, 10.0, 2.20, 0.05)
        
        with col20:
            odds_btts_yes = st.number_input("BTTS Yes", 0.0, 10.0, 1.57, 0.05)
            odds_btts_no = st.number_input("BTTS No", 0.0, 10.0, 2.40, 0.05)
        
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
            
            st.markdown("---")
            
            st.markdown("**⚡ EFFICIENCY GAP:**")
            gap_color = "🟢" if result['efficiency_gap'] > 0 else "🔴"
            fav_team = home_team if result['efficiency_gap'] > 0 else away_team
            st.markdown(f"{gap_color} **{result['efficiency_gap']:+.3f}** (favors **{fav_team}**)")
            
            st.markdown("---")
            
            if result['override_active']:
                st.markdown(f"""
                <div class="result-override">
                    <strong>⚠️ OVERRIDE ACTIVE</strong><br>
                    {result['override_reason']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("✅ **No override** — betting odds favorite")
            
            st.markdown("---")
            
            if result['has_bet']:
                primary = result['recommendations'][0]
                override_badge = " (OVERRIDE)" if primary['override'] else ""
                st.markdown(f"""
                <div class="result-primary">
                    <strong>🏆 BEST BET:</strong><br>
                    ✅ <strong>{primary['name']}{override_badge}</strong> at {primary['odds']:.2f}<br>
                    📊 Stake: <span class="stake-highlight">{primary['stake']}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Reason:** {primary['reason']}")
                
                if len(result['recommendations']) > 1:
                    st.markdown("---")
                    st.markdown("**⚽ SECONDARY OPTIONS:**")
                    for bet in result['recommendations'][1:4]:
                        override_badge = " (OVERRIDE)" if bet['override'] else ""
                        st.markdown(f"• {bet['name']}{override_badge} at {bet['odds']:.2f} – Stake {bet['stake']}")
                
                st.markdown("---")
                if result['override_active']:
                    st.markdown(f"**📝 VERDICT:** Override active: {result['override_reason']}")
                else:
                    st.markdown("**📝 VERDICT:** No override. Betting odds favorite in all markets.")
            else:
                st.markdown("""
                <div class="result-skip">
                    <strong>❌ NO QUALIFYING BETS</strong><br>
                    No market meets the odds threshold. Skip this match.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **GrokBet vFinal** | Simplified | Odds Favorite + Proven Overrides | Locked")

if __name__ == "__main__":
    main()