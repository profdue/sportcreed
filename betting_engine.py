"""
MATCH ANALYZER V1.4 — H2H Parser Rewrite + Trend Display Fix
Production Ready. All parsers working. All signals firing.
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
import re

# ============================================================================
# SUPABASE SETUP
# ============================================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Match Analyzer V1.4", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .tier1-card { border: 2px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .tier2-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .skip-card { border-left: 5px solid #fbbf24; }
    .avoid-card { border-left: 5px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .badge-upgrade { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; margin: 0.1rem; }
    .badge-caution { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #ef4444; color: #fff; margin: 0.1rem; }
    .badge-conflict { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #f97316; color: #000; margin: 0.1rem; }
    .badge-skip { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PARSER
# ============================================================================
def parse_match_data(raw_text: str) -> dict:
    lines = raw_text.strip().split('\n')
    
    data = {
        "home_team": None, "away_team": None, "league": None,
        "home_win": None, "draw": None, "away_win": None,
        "btts": None,
        "over_15": None, "over_25": None, "under_25": None, "over_35": None,
        "home_over_15_goals": None, "away_over_15_goals": None,
        "away_over_05_goals": None, "home_over_05_goals": None,
        "home_win_trend": 0, "btts_trend": None, "over_25_trend": 0,
        "home_form_all": [], "away_form_all": [],
        "h2h_scores": [], "h2h_btts_count": 0, "h2h_total": 0,
    }
    
    # ========================================================================
    # TEAM NAMES
    # ========================================================================
    team_names = []
    for i, line in enumerate(lines):
        if line.strip() == 'All competitions' and i > 0:
            name = lines[i-1].strip()
            if name and name not in team_names:
                team_names.append(name)
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # ========================================================================
    # LEAGUE
    # ========================================================================
    for line in lines:
        m = re.search(r'(Premier League|La Liga|Bundesliga|Serie A|Ligue 1|Championship|Süper Lig|Pro League|Primeira Liga|EFL Cup)', line, re.IGNORECASE)
        if m and 'Gameweek' not in line:
            data["league"] = m.group(1)
            break
    
    # ========================================================================
    # HELPER
    # ========================================================================
    def find_pct(start_idx, max_lookahead=3):
        for j in range(start_idx, min(start_idx + max_lookahead, len(lines))):
            sub = lines[j].strip()
            m = re.search(r'(\d+\.?\d*)\s*%', sub)
            if m:
                prob = float(m.group(1))
                trend_m = re.search(r'([+-]\d+\.?\d*)', sub)
                trend = float(trend_m.group(1)) if trend_m else None
                return prob, trend
        return None, None
    
    # ========================================================================
    # STATE MACHINE
    # ========================================================================
    current_section = None
    current_subsection = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped in ['Result', 'Goals', 'First Half Winner', 'Team To Score First', 
                        'Corners', 'Score analysis', 'Head to Head', 'Form Data']:
            current_section = stripped.lower().replace(' ', '_')
            if current_section == 'head_to_head': current_section = 'h2h'
            if current_section == 'form_data': current_section = 'form'
            current_subsection = None
            continue
        
        if current_section == 'result':
            if data["home_team"] and data["home_team"] in stripped:
                prob, trend = find_pct(i)
                if prob: data["home_win"] = prob; data["home_win_trend"] = trend
            elif stripped.startswith('Draw'):
                prob, trend = find_pct(i)
                if prob: data["draw"] = prob
            elif data["away_team"] and data["away_team"] in stripped:
                prob, trend = find_pct(i)
                if prob: data["away_win"] = prob
            elif 'Both Teams to Score' in stripped:
                prob, trend = find_pct(i)
                if prob: data["btts"] = prob; data["btts_trend"] = trend
        
        if current_section == 'goals':
            if 'Over 1.5' in stripped and 'Goals' not in stripped:
                prob, _ = find_pct(i)
                if prob: data["over_15"] = prob
            elif 'Over 2.5' in stripped and 'Goals' not in stripped:
                prob, trend = find_pct(i)
                if prob: data["over_25"] = prob; data["over_25_trend"] = trend if trend else 0
            elif 'Under 2.5' in stripped and 'Goals' not in stripped:
                prob, _ = find_pct(i)
                if prob: data["under_25"] = prob
            elif 'Over 3.5' in stripped and 'Goals' not in stripped:
                prob, _ = find_pct(i)
                if prob: data["over_35"] = prob
        
        if data["home_team"] and f'{data["home_team"]} Goals' in stripped:
            current_subsection = 'home_goals'
        if data["away_team"] and f'{data["away_team"]} Goals' in stripped:
            current_subsection = 'away_goals'
        
        if current_subsection == 'home_goals':
            if 'Over 0.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["home_over_05_goals"] = prob
            if 'Over 1.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["home_over_15_goals"] = prob
        
        if current_subsection == 'away_goals':
            if 'Over 0.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["away_over_05_goals"] = prob
            elif 'Over 1.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["away_over_15_goals"] = prob
    
    # ========================================================================
    # FORM STRINGS
    # ========================================================================
    form_start = -1
    for i, line in enumerate(lines):
        if 'Form, Standings, Stats' in line:
            form_start = i
            break
    
    if form_start >= 0:
        form_groups = []
        current_group = []
        current_team_for_group = None
        
        for i in range(form_start, len(lines)):
            stripped = lines[i].strip()
            
            if stripped == 'Data analysis':
                if len(current_group) >= 4:
                    form_groups.append({"team": current_team_for_group, "form": current_group})
                break
            
            if i+1 < len(lines) and lines[i+1].strip() == 'All competitions':
                if len(current_group) >= 4:
                    form_groups.append({"team": current_team_for_group, "form": current_group})
                current_group = []
                current_team_for_group = stripped
                continue
            
            if stripped in ['W', 'D', 'L']:
                current_group.append(stripped)
        
        if len(current_group) >= 4:
            form_groups.append({"team": current_team_for_group, "form": current_group})
        
        for g in form_groups:
            if g["team"] == data["home_team"]:
                data["home_form_all"] = g["form"]
            elif g["team"] == data["away_team"]:
                data["away_form_all"] = g["form"]
    
    # ========================================================================
    # H2H SCORES — REWRITTEN: Find FT then preceding standalone numbers
    # ========================================================================
    h2h_section = False
    h2h_scores = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Head to Head' in stripped:
            h2h_section = True
            continue
        if h2h_section and 'Form Data' in stripped:
            break
        
        if h2h_section and stripped == 'FT':
            # Look backward for standalone numbers (the scores)
            # Structure: ... score1 \n score2 \n HT : \n ...
            back_nums = []
            for j in range(i-1, max(i-8, 0), -1):
                sub = lines[j].strip()
                if sub == 'HT' or sub == 'HT :':
                    continue
                m = re.match(r'^(\d+)$', sub)
                if m:
                    back_nums.insert(0, int(m.group(1)))
                elif re.search(r'[a-zA-Z]{3,}', sub):
                    # Hit text — stop
                    break
                if len(back_nums) >= 2:
                    break
            
            if len(back_nums) >= 2:
                # Take the last two numbers found
                h = back_nums[-2]
                a = back_nums[-1]
                if h < 20 and a < 20:
                    h2h_scores.append((h, a))
    
    data["h2h_scores"] = h2h_scores
    data["h2h_total"] = len(h2h_scores)
    data["h2h_btts_count"] = sum(1 for h, a in h2h_scores if h > 0 and a > 0)
    
    return data


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_match(data: dict) -> dict:
    result = {"bets": [], "avoid": [], "badges": [], "warnings": []}
    seen_markets = set()
    
    def add_bet(market, tier, confidence, probability, reason):
        if market not in seen_markets:
            seen_markets.add(market)
            result["bets"].append({
                "market": market, "tier": tier,
                "confidence": confidence, "probability": probability,
                "reason": reason
            })
    
    # STEP 1: Strongest Signal
    signals = {}
    if data["btts"]: signals["BTTS"] = data["btts"]
    if data["over_25"]: signals["Over 2.5"] = data["over_25"]
    if data["under_25"]: signals["Under 2.5"] = data["under_25"]
    if data["home_over_15_goals"]: signals["Home Over 1.5 Goals"] = data["home_over_15_goals"]
    if data["away_over_15_goals"]: signals["Away Over 1.5 Goals"] = data["away_over_15_goals"]
    
    if signals:
        strongest = max(signals, key=signals.get)
        strongest_pct = signals[strongest]
        
        trend_bonus = 0
        threshold = 52
        
        btts_trend = data.get("btts_trend") or 0
        over25_trend = data.get("over_25_trend") or 0
        
        if strongest == "BTTS" and btts_trend >= 0.50:
            threshold = 48; trend_bonus = 1.5
            result["badges"].append(f"▲ Strong BTTS Trend +{btts_trend:.2f}")
        elif strongest == "BTTS" and btts_trend >= 0.30:
            trend_bonus = 1
            result["badges"].append(f"▲ BTTS Trend +{btts_trend:.2f}")
        elif strongest == "Over 2.5" and over25_trend >= 0.50:
            threshold = 48; trend_bonus = 1.5
        elif strongest == "Over 2.5" and over25_trend >= 0.30:
            trend_bonus = 1
        
        if strongest_pct >= threshold:
            confidence = min(8.5, 5 + (strongest_pct - threshold) / 10 + trend_bonus)
            add_bet(strongest, "TIER 1", round(confidence, 1), strongest_pct,
                    f"Strongest signal at {strongest_pct:.1f}%")
    
    # STEP 2: Draw Streak → BTTS
    home_form = data.get("home_form_all") or []
    away_form = data.get("away_form_all") or []
    
    if home_form and away_form:
        home_draws = sum(1 for r in home_form[:6] if r == 'D')
        away_draws = sum(1 for r in away_form[:6] if r == 'D')
        
        if (home_draws >= 4 or away_draws >= 4) and data["btts"] and data["btts"] >= 45:
            add_bet("BTTS", "TIER 1", 7.5, data["btts"],
                    f"Draw streak ({max(home_draws, away_draws)} draws)")
    
    # STEP 3: Two In-Form Collision → Draw
    if home_form and away_form:
        home_wins = sum(1 for r in home_form[:3] if r == 'W')
        away_wins = sum(1 for r in away_form[:3] if r == 'W')
        home_unb = sum(1 for r in home_form[:5] if r in ['W', 'D'])
        away_unb = sum(1 for r in away_form[:5] if r in ['W', 'D'])
        
        if home_wins >= 3 and away_wins >= 3 and data["draw"] and data["draw"] >= 22:
            add_bet("Draw", "TIER 2", 6.5, data["draw"], "Both on win streaks (3+)")
            result["badges"].append("Win Streak Collision")
        elif home_unb >= 4 and away_unb >= 4 and data["draw"] and data["draw"] >= 22:
            add_bet("Draw", "TIER 2", 6.0, data["draw"],
                    f"Both unbeaten (H:{home_unb}/5 A:{away_unb}/5)")
            result["badges"].append("Unbeaten Collision")
    
    # STEP 4: H2H BTTS Pattern
    h2h_total = data.get("h2h_total", 0)
    h2h_btts = data.get("h2h_btts_count", 0)
    
    if h2h_btts >= 4 and h2h_total >= 5:
        away_scoring = data.get("away_over_05_goals", 100) or 100
        home_scoring = data.get("home_over_05_goals", 100) or 100
        
        if data["btts"] and data["btts"] >= 45:
            if data["home_win"] and data["away_win"] and data["home_win"] > data["away_win"]:
                underdog_scoring = away_scoring
            else:
                underdog_scoring = home_scoring
            
            if underdog_scoring >= 52:
                add_bet("BTTS", "TIER 1", 8.0, data["btts"],
                        f"H2H BTTS in {h2h_btts}/{h2h_total} meetings")
                result["badges"].append(f"H2H BTTS: {h2h_btts}/{h2h_total}")
            else:
                result["avoid"].append({
                    "market": "BTTS",
                    "reason": f"H2H says BTTS ({h2h_btts}/{h2h_total}) but underdog only {underdog_scoring:.0f}% to score. Trap."
                })
                result["badges"].append(f"⚠️ H2H BTTS trap: underdog {underdog_scoring:.0f}%")
    elif h2h_total > 0:
        result["badges"].append(f"H2H BTTS: {h2h_btts}/{h2h_total}")
    
    # STEP 5: Dominant Favorite
    if data["home_win"] and data["home_win"] >= 60:
        away_scoring = data.get("away_over_05_goals", 100) or 100
        home_over15 = data.get("home_over_15_goals", 0) or 0
        
        if away_scoring < 50:
            add_bet("Home Win to Nil", "TIER 1", 8.5, data["home_win"],
                    f"Home {data['home_win']:.0f}% + Away only {away_scoring:.0f}% to score")
            result["badges"].append("Dominant Home — Win to Nil")
        elif away_scoring < 55:
            add_bet("Home Win to Nil", "TIER 1", 7.5, data["home_win"],
                    f"Home {data['home_win']:.0f}% + Away {away_scoring:.0f}% to score")
        
        if home_over15 >= 50:
            add_bet("Home Over 1.5 Goals", "TIER 1", 7.5, home_over15,
                    f"Home scores 2+ regularly ({home_over15:.0f}%)")
    
    if data["away_win"] and data["away_win"] >= 60:
        home_scoring = data.get("home_over_05_goals", 100) or 100
        away_over15 = data.get("away_over_15_goals", 0) or 0
        
        if home_scoring < 50:
            add_bet("Away Win to Nil", "TIER 1", 8.5, data["away_win"],
                    f"Away {data['away_win']:.0f}% + Home only {home_scoring:.0f}% to score")
            result["badges"].append("Dominant Away — Win to Nil")
        
        if away_over15 >= 50:
            add_bet("Away Over 1.5 Goals", "TIER 1", 7.5, away_over15,
                    f"Away scores 2+ regularly ({away_over15:.0f}%)")
    
    # Under 2.5 check
    if data["under_25"] and data["under_25"] >= 48 and data["under_25"] > (data.get("over_25") or 0):
        add_bet("Under 2.5", "TIER 2", 6.5, data["under_25"],
                "Under 2.5 edges Over 2.5 in probabilities")
    
    # FINALIZE
    tier_order = {"TIER 1": 0, "TIER 2": 1}
    result["bets"].sort(key=lambda b: (tier_order.get(b["tier"], 3), -b["confidence"]))
    
    if data["draw"] and data["draw"] >= 25:
        result["warnings"].append(f"High draw ({data['draw']:.1f}%) — avoid match result")
    if data["away_win"] and data["home_win"] and abs(data["away_win"] - data["home_win"]) < 10:
        result["warnings"].append("Close match — no clear favorite")
    
    return result


# ============================================================================
# SUPABASE
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    try:
        bets_str = " | ".join([b["market"] for b in analysis["bets"]]) if analysis["bets"] else "NO BET"
        top = analysis["bets"][0] if analysis["bets"] else None
        
        record = {
            "home_team": data.get("home_team", "Unknown"),
            "away_team": data.get("away_team", "Unknown"),
            "match_date": str(date.today()),
            "home_data": {
                "league": data.get("league"),
                "home_win_pct": data.get("home_win"),
                "draw_pct": data.get("draw"),
                "away_win_pct": data.get("away_win"),
                "btts_pct": data.get("btts"),
                "over25_pct": data.get("over_25"),
                "home_form": '-'.join(data.get("home_form_all", [])),
                "away_form": '-'.join(data.get("away_form_all", [])),
            },
            "away_data": {},
            "prediction": bets_str,
            "confidence_score": top["confidence"] / 10 if top else 0,
            "winner": top["market"] if top else "NO BET",
            "winner_confidence": f"{top['confidence']}/10" if top else "0",
            "btts": "BTTS YES" if any("BTTS" in b["market"] for b in analysis["bets"]) else "",
            "btts_confidence": max([b["confidence"]/10 for b in analysis["bets"] if "BTTS" in b["market"]]) if any("BTTS" in b["market"] for b in analysis["bets"]) else 0,
            "pattern": " | ".join([b["tier"] for b in analysis["bets"]]) if analysis["bets"] else "NO BET",
            "result_entered": False,
        }
        response = supabase.table("analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except: return []

def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        over25 = total > 2
        actual_winner = "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW"
        btts_yes = home_goals > 0 and away_goals > 0
        
        supabase.table("analyses").update({
            "actual_home_goals": home_goals, "actual_away_goals": away_goals,
            "actual_total_goals": total, "actual_over25": over25,
            "actual_winner": actual_winner, "actual_btts": btts_yes,
            "result_entered": True,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_results():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except: return []

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V1.4")
    st.caption("H2H Parser Rewrite + Trend N/A Fix | All Parsers Working")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                data = parse_match_data(raw_text)
                
                if not data.get("home_team"):
                    st.error("Could not detect team names.")
                else:
                    analysis = analyze_match(data)
                    save_to_db(data, analysis)
                    
                    st.success(f"✅ {data['home_team']} vs {data['away_team']} — {data.get('league', 'Unknown')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="edge-box edge-home">
                            <strong>📊 Probabilities</strong><br>
                            Home: {data.get('home_win', '?')}% | Draw: {data.get('draw', '?')}% | Away: {data.get('away_win', '?')}%<br>
                            BTTS: {data.get('btts', '?')}% | Over 2.5: {data.get('over_25', '?')}% | Under 2.5: {data.get('under_25', '?')}%<br>
                            Home O1.5: {data.get('home_over_15_goals', '?')}% | Away O0.5: {data.get('away_over_05_goals', '?')}%
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        btts_trend_display = f"{data.get('btts_trend', 0):+.2f}" if data.get('btts_trend') is not None else "N/A"
                        st.markdown(f"""
                        <div class="edge-box edge-away">
                            <strong>📈 Form & H2H</strong><br>
                            BTTS Trend: {btts_trend_display} | O2.5 Trend: {data.get('over_25_trend', 0):+.2f}<br>
                            Home: {'-'.join(data.get('home_form_all', [])[:6])}<br>
                            Away: {'-'.join(data.get('away_form_all', [])[:6])}<br>
                            H2H BTTS: {data.get('h2h_btts_count', 0)}/{data.get('h2h_total', 0)}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Recommendations")
                    
                    if analysis["bets"]:
                        for bet in analysis["bets"]:
                            tier_class = "tier1-card" if bet["tier"] == "TIER 1" else "tier2-card"
                            emoji = {
                                "BTTS": "⚽⚽", "Over 2.5": "🔥", "Under 2.5": "🛡️", "Draw": "🤝",
                                "Home Over 1.5 Goals": "🏠⚽", "Away Over 1.5 Goals": "✈️⚽",
                                "Home Win to Nil": "🏠🧤", "Away Win to Nil": "✈️🧤",
                            }.get(bet["market"], "📊")
                            st.markdown(f"""
                            <div class="output-card {tier_class}">
                                <div style="display:flex;align-items:center;gap:1rem;">
                                    <div style="font-size:2rem;">{emoji}</div>
                                    <div>
                                        <div style="font-size:1.2rem;font-weight:800;">{bet['market']} — {bet['tier']}</div>
                                        <div style="font-size:0.85rem;color:#94a3b8;">Confidence: {bet['confidence']}/10 | Prob: {bet['probability']:.1f}%</div>
                                        <div style="font-size:0.8rem;color:#94a3b8;">{bet['reason']}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""<div class="output-card skip-card"><div style="text-align:center;"><span class="badge-skip">NO STRONG SIGNAL</span></div></div>""", unsafe_allow_html=True)
                    
                    if analysis.get("avoid"):
                        st.markdown("### ⚠️ Avoid")
                        for item in analysis["avoid"]:
                            st.markdown(f"""
                            <div class="output-card avoid-card">
                                <strong>❌ AVOID: {item['market']}</strong><br>
                                <small>{item['reason']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if analysis["badges"]:
                        badges_html = " ".join([
                            f'<span class="badge-conflict">{b}</span>' if '⚠️' in b or 'trap' in b
                            else f'<span class="badge-upgrade">{b}</span>'
                            for b in analysis["badges"]
                        ])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    if analysis["warnings"]:
                        for w in analysis["warnings"]:
                            st.markdown(f'<span class="badge-caution">{w}</span>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"{len(pending)} pending")
            for analysis in pending:
                ht = analysis.get('home_team', 'Home'); at = analysis.get('away_team', 'Away')
                with st.expander(f"{ht} vs {at}"):
                    c1, c2, c3 = st.columns(3)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total = hg + ag
                        st.markdown(f"""<div class="score-box"><div class="score-number">{hg} - {ag}</div><div class="score-label">Total: {total}</div></div>""", unsafe_allow_html=True)
                    if st.button("✅ Submit", key=f"sub_{analysis['id']}"):
                        if submit_result(analysis['id'], hg, ag):
                            st.success("Submitted!"); st.rerun()
        else:
            st.info("No pending analyses.")
    
    with tab3:
        st.subheader("📊 Records")
        results = get_results()
        if not results:
            st.info("No results yet.")
        else:
            st.write(f"**Total tracked:** {len(results)}")
            if st.checkbox("Show all"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"), "home": r.get("home_team"), "away": r.get("away_team"),
                    "prediction": r.get("prediction"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
