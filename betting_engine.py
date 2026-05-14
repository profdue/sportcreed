"""
MATCH ANALYZER V1 — Multi-Source Probability + Trend + Form + H2H Analysis
Companion to Streak Predictor V8. Uses model probabilities, trends, form, H2H.
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
st.set_page_config(page_title="Match Analyzer V1", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .tier1-card { border: 2px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .tier2-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .tier3-card { border-left: 5px solid #94a3b8; }
    .skip-card { border-left: 5px solid #fbbf24; }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .badge-upgrade { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; }
    .badge-caution { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #ef4444; color: #fff; }
    .badge-info { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #3b82f6; color: #fff; }
    .badge-skip { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPERS
# ============================================================================
def parse_probability(text: str) -> dict:
    """Parse a probability line like '52.96% ▲ (+0.58)' or '52.96%'"""
    prob = None
    trend_val = 0
    
    pct_match = re.search(r'(\d+\.?\d*)\s*%', text)
    if pct_match:
        prob = float(pct_match.group(1))
    
    trend_match = re.search(r'([+-]\d+\.?\d*)', text)
    if trend_match:
        trend_val = float(trend_match.group(1))
    
    return {"probability": prob, "trend_value": trend_val}


def parse_h2h_scores(text: str) -> list:
    """Extract scores from H2H section like '1-2', '1-4', etc."""
    scores = re.findall(r'(\d+)\s*[-–]\s*(\d+)', text)
    # Filter out year dates (like 2023) and other non-score numbers
    filtered = []
    for h, a in scores:
        h_int, a_int = int(h), int(a)
        if h_int < 20 and a_int < 20 and h_int + a_int > 0:
            filtered.append((h_int, a_int))
    return filtered


def parse_match_data(raw_text: str) -> dict:
    """Parse the full match data text block using section headers."""
    
    data = {
        "home_team": None, "away_team": None,
        "league": None,
        "home_win": None, "draw": None, "away_win": None,
        "btts": None,
        "over_15": None, "over_25": None, "under_25": None, "over_35": None,
        "home_over_15_goals": None, "away_over_15_goals": None,
        "away_over_05_goals": None,
        "home_win_trend": 0, "draw_trend": 0, "away_win_trend": 0,
        "btts_trend": 0, "over_25_trend": 0,
        "home_form_all": [], "away_form_all": [],
        "home_form_league": [], "away_form_league": [],
        "h2h_scores": [],
        "h2h_btts_count": 0,
        "home_scoring_first_pct": None,
    }
    
    lines = raw_text.strip().split('\n')
    
    # ========================================================================
    # EXTRACT TEAM NAMES
    # ========================================================================
    team_names = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line and not re.search(r'[▲▼%]', line) and not re.search(r'\d{4}', line):
            if i+1 < len(lines):
                next_line = lines[i+1].strip()
                if 'All competitions' in next_line or re.match(r'^[WDL]', next_line):
                    if line not in team_names and line not in [
                        'Form', 'Standings', 'Stats', 'Data analysis',
                        'Head to Head', 'Form Data', 'Result', 'Goals',
                        'First Half Winner', 'Team To Score First', 'Corners',
                        'Score analysis', 'Premier League', 'La Liga', 'Bundesliga',
                        'Serie A', 'Ligue 1', 'Championship', 'Süper Lig',
                        'Pro League', 'Primeira Liga', 'EFL Cup'
                    ]:
                        team_names.append(line)
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # ========================================================================
    # EXTRACT LEAGUE
    # ========================================================================
    for line in lines:
        league_match = re.search(
            r'(Premier League|La Liga|Bundesliga|Serie A|Ligue 1|Championship|Süper Lig|Pro League|Primeira Liga|EFL Cup)',
            line, re.IGNORECASE
        )
        if league_match:
            data["league"] = league_match.group(1)
            break
    
    # ========================================================================
    # EXTRACT PROBABILITIES FROM "Result" SECTION
    # ========================================================================
    result_section = False
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line == 'Result':
            result_section = True
            continue
        if result_section and line == 'Goals':
            break
        
        if result_section and '%' in line:
            parsed = parse_probability(line)
            prob = parsed["probability"]
            trend = parsed["trend_value"]
            
            if data["home_team"] and data["home_team"].split()[-1] in line:
                data["home_win"] = prob
                data["home_win_trend"] = trend
            elif line.startswith('Draw'):
                data["draw"] = prob
                data["draw_trend"] = trend
            elif data["away_team"] and data["away_team"].split()[-1] in line:
                data["away_win"] = prob
                data["away_win_trend"] = trend
            elif 'Both Teams to Score' in line:
                data["btts"] = prob
                data["btts_trend"] = trend
    
    # ========================================================================
    # EXTRACT GOALS PROBABILITIES
    # ========================================================================
    goals_section = False
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line == 'Goals':
            goals_section = True
            continue
        if goals_section and 'First Half Winner' in line:
            break
        
        if goals_section and '%' in line:
            parsed = parse_probability(line)
            prob = parsed["probability"]
            trend = parsed["trend_value"]
            
            if 'Over 1.5' in line and 'Goals' not in line:
                data["over_15"] = prob
            elif 'Over 2.5' in line and 'Goals' not in line:
                data["over_25"] = prob
                data["over_25_trend"] = trend
            elif 'Under 2.5' in line and 'Goals' not in line:
                data["under_25"] = prob
            elif 'Over 3.5' in line and 'Goals' not in line:
                data["over_35"] = prob
    
    # ========================================================================
    # EXTRACT TEAM-SPECIFIC GOALS
    # ========================================================================
    for i, line in enumerate(lines):
        line = line.strip()
        
        if data["home_team"] and f'{data["home_team"]} Goals' in line:
            for j in range(i+1, min(i+10, len(lines))):
                sub = lines[j].strip()
                if 'Over 1.5' in sub and '%' in sub:
                    parsed = parse_probability(sub)
                    if parsed["probability"]:
                        data["home_over_15_goals"] = parsed["probability"]
                    break
        
        if data["away_team"] and f'{data["away_team"]} Goals' in line:
            for j in range(i+1, min(i+10, len(lines))):
                sub = lines[j].strip()
                if 'Over 0.5' in sub and '%' in sub and not data["away_over_05_goals"]:
                    parsed = parse_probability(sub)
                    if parsed["probability"]:
                        data["away_over_05_goals"] = parsed["probability"]
                if 'Over 1.5' in sub and '%' in sub and not data["away_over_15_goals"]:
                    parsed = parse_probability(sub)
                    if parsed["probability"]:
                        data["away_over_15_goals"] = parsed["probability"]
                    break
    
    # ========================================================================
    # EXTRACT FORM STRINGS
    # ========================================================================
    form_section = False
    form_count = 0
    for i, line in enumerate(lines):
        if 'Form, Standings, Stats' in line:
            form_section = True
            continue
        if form_section and re.match(r'^[WDL]', line.strip()):
            results = re.findall(r'[WDL]', line.strip())
            if len(results) >= 4:
                if form_count == 0:
                    data["home_form_all"] = results
                elif form_count == 1:
                    data["away_form_all"] = results
                elif form_count == 2:
                    data["home_form_league"] = results
                elif form_count == 3:
                    data["away_form_league"] = results
                form_count += 1
    
    # ========================================================================
    # EXTRACT H2H
    # ========================================================================
    h2h_section = False
    h2h_text = ""
    for line in lines:
        if 'Head to Head' in line:
            h2h_section = True
            continue
        if h2h_section and 'Form Data' in line:
            break
        if h2h_section:
            h2h_text += line + " "
    
    data["h2h_scores"] = parse_h2h_scores(h2h_text)
    
    btts_count = 0
    for h, a in data["h2h_scores"]:
        if h > 0 and a > 0:
            btts_count += 1
    data["h2h_btts_count"] = btts_count
    
    # ========================================================================
    # EXTRACT TEAM TO SCORE FIRST
    # ========================================================================
    for i, line in enumerate(lines):
        if 'Team To Score First' in line:
            for j in range(i+1, min(i+6, len(lines))):
                sub = lines[j].strip()
                if data["home_team"] and data["home_team"].split()[-1] in sub and '%' in sub:
                    parsed = parse_probability(sub)
                    if parsed["probability"]:
                        data["home_scoring_first_pct"] = parsed["probability"]
                        break
            break
    
    return data


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_match(data: dict) -> dict:
    """Run the multi-step analysis framework."""
    
    result = {
        "bets": [],
        "badges": [],
        "warnings": [],
    }
    
    # ========================================================================
    # STEP 1: Find Strongest Single Signal
    # ========================================================================
    signals = {}
    if data["btts"]: signals["BTTS"] = data["btts"]
    if data["over_25"]: signals["Over 2.5"] = data["over_25"]
    if data["under_25"]: signals["Under 2.5"] = data["under_25"]
    if data["home_over_15_goals"]: signals["Home Over 1.5 Goals"] = data["home_over_15_goals"]
    if data["away_over_15_goals"]: signals["Away Over 1.5 Goals"] = data["away_over_15_goals"]
    
    if signals:
        strongest = max(signals, key=signals.get)
        strongest_pct = signals[strongest]
        
        # Apply trend weight
        trend_bonus = 0
        if strongest == "BTTS" and data["btts_trend"] >= 0.30:
            trend_bonus = 1
            result["badges"].append(f"▲ BTTS Trend +{data['btts_trend']:.2f}")
        elif strongest == "Over 2.5" and data["over_25_trend"] >= 0.30:
            trend_bonus = 1
            result["badges"].append(f"▲ Over 2.5 Trend +{data['over_25_trend']:.2f}")
        
        if strongest_pct >= 52:
            confidence = min(8.5, 6 + (strongest_pct - 50) / 10 + trend_bonus)
            result["bets"].append({
                "market": strongest,
                "tier": "TIER 1",
                "confidence": round(confidence, 1),
                "probability": strongest_pct,
                "reason": f"Strongest signal at {strongest_pct:.1f}%"
            })
    
    # ========================================================================
    # STEP 2: Check Streak Rules
    # ========================================================================
    home_form = data.get("home_form_league") or data.get("home_form_all") or []
    away_form = data.get("away_form_league") or data.get("away_form_all") or []
    
    if home_form and away_form:
        home_draws = sum(1 for r in home_form[:6] if r == 'D')
        away_draws = sum(1 for r in away_form[:6] if r == 'D')
        
        if (home_draws >= 4 or away_draws >= 4) and data["btts"] and data["btts"] >= 45:
            already_has_btts = any(b["market"] == "BTTS" for b in result["bets"])
            if not already_has_btts:
                result["bets"].append({
                    "market": "BTTS",
                    "tier": "TIER 1",
                    "confidence": 7.5,
                    "probability": data["btts"],
                    "reason": f"Draw streak detected ({max(home_draws, away_draws)} draws)"
                })
    
    # ========================================================================
    # STEP 3: Two In-Form Collision → Draw
    # ========================================================================
    if home_form and away_form:
        home_wins = sum(1 for r in home_form[:3] if r == 'W')
        away_wins = sum(1 for r in away_form[:3] if r == 'W')
        
        if home_wins >= 3 and away_wins >= 3 and data["draw"] and data["draw"] >= 22:
            result["bets"].append({
                "market": "Draw",
                "tier": "TIER 2",
                "confidence": 6.5,
                "probability": data["draw"],
                "reason": "Both on win streaks — draw value"
            })
            result["badges"].append("In-Form Collision")
    
    # ========================================================================
    # STEP 4: H2H BTTS Pattern
    # ========================================================================
    if data["h2h_btts_count"] >= 4 and len(data.get("h2h_scores", [])) >= 5:
        if data["btts"] and data["btts"] >= 45:
            already_has_btts = any(b["market"] == "BTTS" for b in result["bets"])
            if not already_has_btts:
                result["bets"].append({
                    "market": "BTTS",
                    "tier": "TIER 1",
                    "confidence": 8.0,
                    "probability": data["btts"],
                    "reason": f"H2H BTTS in {data['h2h_btts_count']}/5 meetings"
                })
            result["badges"].append(f"H2H BTTS: {data['h2h_btts_count']}/5")
    
    # ========================================================================
    # STEP 5: Finalize
    # ========================================================================
    tier_order = {"TIER 1": 0, "TIER 2": 1, "TIER 3": 2}
    result["bets"].sort(key=lambda b: (tier_order.get(b["tier"], 3), -b["confidence"]))
    
    if data["draw"] and data["draw"] >= 25:
        result["warnings"].append(f"High draw probability ({data['draw']:.1f}%) — avoid match result bets")
    
    if data["away_win"] and data["home_win"]:
        if abs(data["away_win"] - data["home_win"]) < 10:
            result["warnings"].append("Close match — no clear favorite")
    
    return result


# ============================================================================
# SUPABASE
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    try:
        bets_str = " | ".join([b["market"] for b in analysis["bets"]]) if analysis["bets"] else "NO BET"
        top_bet = analysis["bets"][0] if analysis["bets"] else None
        
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
            },
            "away_data": {},
            "prediction": bets_str,
            "confidence_score": top_bet["confidence"] / 10 if top_bet else 0,
            "winner": top_bet["market"] if top_bet else "NO BET",
            "winner_confidence": f"{top_bet['confidence']}/10" if top_bet else "0",
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
    st.title("📊 Match Analyzer V1")
    st.caption("Multi-Source: Probabilities + Trends + Form + H2H | Section-based parsing")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        st.caption("Paste the full match data block including Data Analysis, Result, Form, and H2H sections.")
        
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                data = parse_match_data(raw_text)
                
                if not data.get("home_team"):
                    st.error("Could not detect team names. Check the data format.")
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
                            Home Over 1.5: {data.get('home_over_15_goals', '?')}% | Away Over 1.5: {data.get('away_over_15_goals', '?')}%
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="edge-box edge-away">
                            <strong>📈 Form & H2H</strong><br>
                            BTTS Trend: {data.get('btts_trend', 0):+.2f} | Over 2.5 Trend: {data.get('over_25_trend', 0):+.2f}<br>
                            Home Form: {'-'.join(data.get('home_form_all', [])[:6])}<br>
                            Away Form: {'-'.join(data.get('away_form_all', [])[:6])}<br>
                            H2H BTTS: {data.get('h2h_btts_count', 0)}/{len(data.get('h2h_scores', []))}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Recommendations")
                    
                    if analysis["bets"]:
                        for bet in analysis["bets"]:
                            tier_class = "tier1-card" if bet["tier"] == "TIER 1" else "tier2-card" if bet["tier"] == "TIER 2" else "tier3-card"
                            emoji = {"BTTS": "⚽⚽", "Over 2.5": "🔥", "Under 2.5": "🛡️", "Draw": "🤝", "Home Win": "🏠", "Away Win": "✈️"}.get(bet["market"], "📊")
                            
                            st.markdown(f"""
                            <div class="output-card {tier_class}">
                                <div style="display:flex;align-items:center;gap:1rem;">
                                    <div style="font-size:2rem;">{emoji}</div>
                                    <div style="flex:1;">
                                        <div style="font-size:1.2rem;font-weight:800;">{bet['market']} — {bet['tier']}</div>
                                        <div style="font-size:0.85rem;color:#94a3b8;">
                                            Confidence: {bet['confidence']}/10 | Probability: {bet['probability']:.1f}%
                                        </div>
                                        <div style="font-size:0.8rem;color:#94a3b8;">{bet['reason']}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="output-card skip-card">
                            <div style="text-align:center;">
                                <span class="badge-skip">NO STRONG SIGNAL</span>
                                <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.5rem;">No clear betting opportunity detected.</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if analysis["badges"]:
                        st.markdown("### 🏷️ Signals")
                        badges_html = " ".join([
                            f'<span class="badge-upgrade">{b}</span>' if '▲' in b or 'H2H' in b
                            else f'<span class="badge-info">{b}</span>'
                            for b in analysis["badges"]
                        ])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    
                    if analysis["warnings"]:
                        st.markdown("### ⚠️")
                        for w in analysis["warnings"]:
                            st.markdown(f'<span class="badge-caution">{w}</span>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"{len(pending)} pending")
            for analysis in pending:
                ht = analysis.get('home_team', 'Home'); at = analysis.get('away_team', 'Away')
                with st.expander(f"{ht} vs {at} — {analysis.get('prediction', '?')}"):
                    c1, c2, c3 = st.columns(3)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total = hg + ag
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{hg} - {ag}</div>
                            <div class="score-label">Total: {total} | Over 2.5: {'Yes' if total > 2 else 'No'} | BTTS: {'Yes' if hg > 0 and ag > 0 else 'No'}</div>
                        </div>
                        """, unsafe_allow_html=True)
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
            if st.checkbox("Show all results"):
                st.dataframe(pd.DataFrame([{
                    "date": r.get("match_date"), "home": r.get("home_team"), "away": r.get("away_team"),
                    "prediction": r.get("prediction"),
                    "score": f"{r.get('actual_home_goals', '-')}-{r.get('actual_away_goals', '-')}",
                } for r in results]))

if __name__ == "__main__":
    main()
