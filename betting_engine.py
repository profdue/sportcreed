import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import traceback

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
# TABLE NAME
# ============================================================================
TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="DC12-VS Double Chance 12 V2.0", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .bet-card { border-left: 5px solid #10b981; background: linear-gradient(135deg, #0a2a1a 0%, #0a1a0a 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .metric-card { background: #0f172a; border-radius: 10px; padding: 0.75rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; color: #94a3b8; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-bet { color: #10b981; }
    .prediction-skip { color: #f59e0b; }
    .final-badge { background: #10b981; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 2px solid #10b981; }
    .bet-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .skip-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .factor-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }
    .factor-name { color: #94a3b8; }
    .factor-value { font-weight: 600; }
    .upload-container { border: 2px dashed #10b981; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .upload-container:hover { border-color: #34d399; background: rgba(16, 185, 129, 0.05); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_match_date(date_val) -> datetime:
    if not date_val:
        return datetime(1900, 1, 1)
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    date_str = str(date_val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return datetime(1900, 1, 1)

def format_date_display(date_val) -> str:
    dt = parse_match_date(date_val)
    if dt.year == 1900:
        return str(date_val)
    return dt.strftime("%Y-%m-%d")

def check_match_exists(home_team: str, away_team: str, match_date: str) -> bool:
    try:
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        response = supabase.table(TABLE_NAME).select("id").eq("home_team", home_team).eq("away_team", away_team).eq("match_date", date_part).execute()
        return len(response.data) > 0
    except:
        return False

# ============================================================================
# DC12-VS CALCULATION - V2.0
# ============================================================================
def calculate_dc12_vs(match_data: dict) -> dict:
    """
    Calculate Double Chance 12 Value Score (DC12-VS) - V2.0
    ND (No Draw) is the CORE metric with 10x weight.
    """
    nd_home = match_data.get('nd_home', 0)
    nd_away = match_data.get('nd_away', 0)
    w_home = match_data.get('w_home', 0)
    w_away = match_data.get('w_away', 0)
    l_home = match_data.get('l_home', 0)
    l_away = match_data.get('l_away', 0)
    best_team_home = match_data.get('best_team_home', 0)
    best_team_away = match_data.get('best_team_away', 0)
    worst_team_home = match_data.get('worst_team_home', 0)
    worst_team_away = match_data.get('worst_team_away', 0)
    best_off_home = match_data.get('best_off_home', 0)
    best_off_away = match_data.get('best_off_away', 0)
    worst_def_home = match_data.get('worst_def_home', 0)
    worst_def_away = match_data.get('worst_def_away', 0)
    now_home = match_data.get('now_home', 0)
    now_away = match_data.get('now_away', 0)
    nol_home = match_data.get('nol_home', 0)
    nol_away = match_data.get('nol_away', 0)

    # DC12-VS V2.0 - ND has 10x weight
    dc12_vs = (
        (nd_home + nd_away) * 10 +
        (w_home + w_away) * 2 +
        (l_home + l_away) * -2 +
        (best_team_home + best_team_away) * 1 +
        (worst_team_home + worst_team_away) * -1 +
        (best_off_home + best_off_away) * 1 +
        (worst_def_home + worst_def_away) * 1 -
        (now_home + now_away) * 3 -
        (nol_home + nol_away) * 3
    )

    home_odds = match_data.get('home_odds', 0)
    away_odds = match_data.get('away_odds', 0)
    dc12_odds = 1 / ((1 / home_odds) + (1 / away_odds)) if home_odds > 0 and away_odds > 0 else 0
    draw_odds = match_data.get('draw_odds', 0)

    if dc12_vs > 80 and draw_odds > 3.00:
        decision, action, confidence = "BET", "✅ BET on Double Chance 12", "HIGH"
    elif dc12_vs > 50 and draw_odds > 3.00:
        decision, action, confidence = "CONSIDER", "⚠️ CONSIDER betting", "MEDIUM"
    else:
        decision, action, confidence = "SKIP", "❌ SKIP - No value", "LOW"

    return {
        "dc12_vs": dc12_vs,
        "dc12_odds": dc12_odds,
        "decision": decision,
        "action": action,
        "confidence": confidence,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "nd_home": nd_home,
        "nd_away": nd_away,
        "w_home": w_home,
        "w_away": w_away,
        "l_home": l_home,
        "l_away": l_away,
        "best_team_home": best_team_home,
        "best_team_away": best_team_away,
        "worst_team_home": worst_team_home,
        "worst_team_away": worst_team_away,
        "best_off_home": best_off_home,
        "best_off_away": best_off_away,
        "worst_def_home": worst_def_home,
        "worst_def_away": worst_def_away,
    }

# ============================================================================
# PARSER
# ============================================================================
def parse_betexplorer_data(text: str) -> list:
    matches = []
    lines = text.split('\n')
    match_cache = {}
    current_page_type = None
    current_country = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect page type
        if 'Team\tW\tNext match' in line or 'Team, W, Next match' in line:
            current_page_type = 'wins'
            continue
        elif 'Team\tD\tNext match' in line or 'Team, D, Next match' in line:
            current_page_type = 'draws'
            continue
        elif 'Team\tL\tNext match' in line or 'Team, L, Next match' in line:
            current_page_type = 'losses'
            continue
        elif 'Team\tNW\tNext match' in line or 'Team, NW, Next match' in line:
            current_page_type = 'no_wins'
            continue
        elif 'Team\tNL\tNext match' in line or 'Team, NL, Next match' in line:
            current_page_type = 'no_losses'
            continue
        elif 'Team\tND\tNext match' in line or 'Team, ND, Next match' in line:
            current_page_type = 'no_draws'
            continue
        elif 'Best teams' in line or 'Worst teams' in line or 'Best offensive' in line or 'Best defensive' in line or 'Worst defensive' in line:
            current_page_type = 'quality'
            continue
        
        if current_page_type == 'quality':
            continue
        
        # Country/league name
        if re.match(r'^[A-Za-z\s]+$', line):
            current_country = line
            continue
        
        # Parse page-specific data
        if current_page_type in ['wins', 'draws', 'losses', 'no_wins', 'no_losses', 'no_draws']:
            parts = re.split(r'\t+', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 4:
                try:
                    if re.search(r'[A-Za-z]', parts[0]) and parts[1].isdigit():
                        team = parts[0]
                        streak_value = int(parts[1])
                        next_match = parts[2]
                        
                        match_parts = re.split(r'\s*[-–]\s*', next_match)
                        if len(match_parts) == 2:
                            home_team = match_parts[0].strip()
                            away_team = match_parts[1].strip()
                            
                            odds_text = ' '.join(parts[3:])
                            odds = re.findall(r'[\d.]+', odds_text)
                            
                            if len(odds) >= 3:
                                home_odds = float(odds[0]) if odds[0] else 0
                                draw_odds = float(odds[1]) if odds[1] else 0
                                away_odds = float(odds[2]) if odds[2] else 0
                                
                                if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                                    match_key = f"{home_team}|{away_team}"
                                    
                                    if match_key not in match_cache:
                                        match_cache[match_key] = {
                                            "home_team": home_team,
                                            "away_team": away_team,
                                            "home_odds": home_odds,
                                            "draw_odds": draw_odds,
                                            "away_odds": away_odds,
                                            "nd_home": 0, "nd_away": 0,
                                            "w_home": 0, "w_away": 0,
                                            "l_home": 0, "l_away": 0,
                                            "now_home": 0, "now_away": 0,
                                            "nol_home": 0, "nol_away": 0,
                                            "best_team_home": 0, "best_team_away": 0,
                                            "worst_team_home": 0, "worst_team_away": 0,
                                            "best_off_home": 0, "best_off_away": 0,
                                            "worst_def_home": 0, "worst_def_away": 0,
                                            "league": current_country or "Unknown",
                                            "date": datetime.now().strftime("%Y-%m-%d"),
                                            "is_finished": False,
                                        }
                                    
                                    if team == home_team:
                                        if current_page_type == 'wins':
                                            match_cache[match_key]['w_home'] = streak_value
                                        elif current_page_type == 'losses':
                                            match_cache[match_key]['l_home'] = streak_value
                                        elif current_page_type == 'no_wins':
                                            match_cache[match_key]['now_home'] = streak_value
                                        elif current_page_type == 'no_losses':
                                            match_cache[match_key]['nol_home'] = streak_value
                                        elif current_page_type == 'no_draws':
                                            match_cache[match_key]['nd_home'] = streak_value
                                    elif team == away_team:
                                        if current_page_type == 'wins':
                                            match_cache[match_key]['w_away'] = streak_value
                                        elif current_page_type == 'losses':
                                            match_cache[match_key]['l_away'] = streak_value
                                        elif current_page_type == 'no_wins':
                                            match_cache[match_key]['now_away'] = streak_value
                                        elif current_page_type == 'no_losses':
                                            match_cache[match_key]['nol_away'] = streak_value
                                        elif current_page_type == 'no_draws':
                                            match_cache[match_key]['nd_away'] = streak_value
                                    
                                    current_odds = match_cache[match_key].get('home_odds', 0)
                                    if home_odds > current_odds and home_odds > 0:
                                        match_cache[match_key]['home_odds'] = home_odds
                                        match_cache[match_key]['draw_odds'] = draw_odds
                                        match_cache[match_key]['away_odds'] = away_odds
                                    continue
                except (ValueError, IndexError):
                    pass
    
    # Parse combined CSV format
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if any(key in line for key in ['Team\tW\t', 'Team\tD\t', 'Team\tL\t', 'Team\tNW\t', 'Team\tNL\t', 'Team\tND\t']):
            continue
        
        parts = re.split(r'[\t,;|]+', line)
        parts = [p.strip() for p in parts if p.strip()]
        
        if len(parts) >= 5:
            try:
                if re.search(r'[A-Za-z]', parts[0]) and re.search(r'[A-Za-z]', parts[1]):
                    home_odds = float(parts[2]) if parts[2] and parts[2].replace('.', '').replace('-', '').isdigit() else 0
                    draw_odds = float(parts[3]) if parts[3] and parts[3].replace('.', '').replace('-', '').isdigit() else 0
                    away_odds = float(parts[4]) if parts[4] and parts[4].replace('.', '').replace('-', '').isdigit() else 0
                    
                    if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                        match_key = f"{parts[0]}|{parts[1]}"
                        
                        if match_key not in match_cache:
                            match_data = {
                                "home_team": parts[0],
                                "away_team": parts[1],
                                "home_odds": home_odds,
                                "draw_odds": draw_odds,
                                "away_odds": away_odds,
                                "nd_home": int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 0,
                                "nd_away": int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else 0,
                                "w_home": int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else 0,
                                "w_away": int(parts[8]) if len(parts) > 8 and parts[8].isdigit() else 0,
                                "l_home": int(parts[9]) if len(parts) > 9 and parts[9].isdigit() else 0,
                                "l_away": int(parts[10]) if len(parts) > 10 and parts[10].isdigit() else 0,
                                "best_team_home": 1 if len(parts) > 11 and parts[11] == '1' else 0,
                                "best_team_away": 1 if len(parts) > 12 and parts[12] == '1' else 0,
                                "worst_team_home": 1 if len(parts) > 13 and parts[13] == '1' else 0,
                                "worst_team_away": 1 if len(parts) > 14 and parts[14] == '1' else 0,
                                "best_off_home": 1 if len(parts) > 15 and parts[15] == '1' else 0,
                                "best_off_away": 1 if len(parts) > 16 and parts[16] == '1' else 0,
                                "worst_def_home": 1 if len(parts) > 17 and parts[17] == '1' else 0,
                                "worst_def_away": 1 if len(parts) > 18 and parts[18] == '1' else 0,
                                "now_home": int(parts[19]) if len(parts) > 19 and parts[19].isdigit() else 0,
                                "now_away": int(parts[20]) if len(parts) > 20 and parts[20].isdigit() else 0,
                                "nol_home": int(parts[21]) if len(parts) > 21 and parts[21].isdigit() else 0,
                                "nol_away": int(parts[22]) if len(parts) > 22 and parts[22].isdigit() else 0,
                                "league": "Unknown",
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "is_finished": False,
                            }
                            match_cache[match_key] = match_data
            except (ValueError, IndexError):
                pass
    
    for match_data in match_cache.values():
        matches.append(match_data)
    
    return matches

# ============================================================================
# DISPLAY
# ============================================================================
def display_dc12_analysis(match: dict, analysis: dict, already_stored: bool = False):
    decision = analysis.get("decision", "SKIP")
    dc12_vs = analysis.get("dc12_vs", 0)
    dc12_odds = analysis.get("dc12_odds", 0)
    draw_odds = analysis.get("draw_odds", 0)
    action = analysis.get("action", "")
    confidence = analysis.get("confidence", "LOW")
    home_team = match.get("home_team", "Home")
    away_team = match.get("away_team", "Away")

    if decision == "BET":
        card_class, pred_class, pred_emoji, pred_text, badge = "bet-card", "prediction-bet", "🎯", "BET DOUBLE CHANCE 12", f'<span class="bet-badge">✅ BET</span>'
    elif decision == "CONSIDER":
        card_class, pred_class, pred_emoji, pred_text, badge = "skip-card", "prediction-skip", "⚠️", "CONSIDER BETTING", f'<span class="skip-badge">⚠️ CONSIDER</span>'
    else:
        card_class, pred_class, pred_emoji, pred_text, badge = "skip-card", "prediction-skip", "❌", "SKIP - NO VALUE", f'<span class="skip-badge">❌ SKIP</span>'

    st.markdown(f"""
    <div class="output-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">DC12-VS - DOUBLE CHANCE 12</div>
                <div class="prediction-display {pred_class}">
                    {pred_emoji} {pred_text}
                </div>
                <div>
                    {badge}
                    <span class="final-badge" style="margin-left:0.5rem;">DC12-VS V2.0</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">DC12-VS Score</div>
                <div style="font-size: 2.5rem; font-weight: 800;">{dc12_vs}</div>
                <div>
                    <span style="font-size:0.7rem; color:#94a3b8;">{confidence} Confidence</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            {action}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 DC12-VS Breakdown")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{dc12_odds:.2f}</div><div class="metric-label">DC12 Odds</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{draw_odds:.2f}</div><div class="metric-label">Draw Odds</div></div>', unsafe_allow_html=True)
    with col3:
        implied_prob = 1 / dc12_odds if dc12_odds > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{implied_prob:.1%}</div><div class="metric-label">Implied No-Draw %</div></div>', unsafe_allow_html=True)
    with col4:
        nd_total = analysis.get("nd_home", 0) + analysis.get("nd_away", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{nd_total}</div><div class="metric-label">Total ND Streak</div></div>', unsafe_allow_html=True)

    st.markdown("### 📈 Component Breakdown")
    with st.expander("Show DC12-VS Components"):
        st.markdown(f"""
        <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.25rem 0;">
            <div style="color: #10b981; font-weight: 700; margin-bottom: 0.5rem;">ND (No Draw) - CORE METRIC (×10)</div>
            <div class="factor-row"><span class="factor-name">🏠 {home_team} ND</span><span class="factor-value">{analysis.get('nd_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">✈️ {away_team} ND</span><span class="factor-value">{analysis.get('nd_away', 0)}</span></div>
            <div style="color: #fbbf24; font-weight: 700; margin: 0.5rem 0;">Momentum (×2)</div>
            <div class="factor-row"><span class="factor-name">🏆 {home_team} Wins</span><span class="factor-value">{analysis.get('w_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🏆 {away_team} Wins</span><span class="factor-value">{analysis.get('w_away', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">📉 {home_team} Losses</span><span class="factor-value">-{analysis.get('l_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">📉 {away_team} Losses</span><span class="factor-value">-{analysis.get('l_away', 0)}</span></div>
            <div style="color: #ef4444; font-weight: 700; margin: 0.5rem 0;">Draw Risk Penalties (×3)</div>
            <div class="factor-row"><span class="factor-name">🚫 No Win ({home_team})</span><span class="factor-value">-{analysis.get('now_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🚫 No Win ({away_team})</span><span class="factor-value">-{analysis.get('now_away', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🚫 No Loss ({home_team})</span><span class="factor-value">-{analysis.get('nol_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🚫 No Loss ({away_team})</span><span class="factor-value">-{analysis.get('nol_away', 0)}</span></div>
            <div style="color: #64748b; font-weight: 700; margin: 0.5rem 0;">Quality Indicators (×1)</div>
            <div class="factor-row"><span class="factor-name">⭐ Best Team ({home_team})</span><span class="factor-value">{analysis.get('best_team_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">⭐ Best Team ({away_team})</span><span class="factor-value">{analysis.get('best_team_away', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">⚠️ Worst Team ({home_team})</span><span class="factor-value">-{analysis.get('worst_team_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">⚠️ Worst Team ({away_team})</span><span class="factor-value">-{analysis.get('worst_team_away', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">⚽ Best Offense ({home_team})</span><span class="factor-value">{analysis.get('best_off_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">⚽ Best Offense ({away_team})</span><span class="factor-value">{analysis.get('best_off_away', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🛡️ Worst Defense ({home_team})</span><span class="factor-value">{analysis.get('worst_def_home', 0)}</span></div>
            <div class="factor-row"><span class="factor-name">🛡️ Worst Defense ({away_team})</span><span class="factor-value">{analysis.get('worst_def_away', 0)}</span></div>
        </div>
        """, unsafe_allow_html=True)
    st.caption("📐 DC12-VS = (ND×10) + (W×2) - (L×2) + (Best×1) - (Worst×1) + (BestOff×1) + (WorstDef×1) - (NoW×3) - (NoL×3)")

# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, analysis: dict):
    """Only save matches that are BET or CONSIDER"""
    try:
        decision = analysis.get("decision", "SKIP")
        
        if decision == "SKIP":
            return "SKIPPED"
            
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", datetime.now().strftime("%Y-%m-%d"))
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else datetime.now().strftime("%Y-%m-%d")
        
        if check_match_exists(home_team, away_team, match_date):
            return "ALREADY_EXISTS"
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            "nd_home": match.get("nd_home", 0),
            "nd_away": match.get("nd_away", 0),
            "w_home": match.get("w_home", 0),
            "w_away": match.get("w_away", 0),
            "l_home": match.get("l_home", 0),
            "l_away": match.get("l_away", 0),
            "best_team_home": match.get("best_team_home", 0),
            "best_team_away": match.get("best_team_away", 0),
            "worst_team_home": match.get("worst_team_home", 0),
            "worst_team_away": match.get("worst_team_away", 0),
            "best_off_home": match.get("best_off_home", 0),
            "best_off_away": match.get("best_off_away", 0),
            "worst_def_home": match.get("worst_def_home", 0),
            "worst_def_away": match.get("worst_def_away", 0),
            "now_home": match.get("now_home", 0),
            "now_away": match.get("now_away", 0),
            "nol_home": match.get("nol_home", 0),
            "nol_away": match.get("nol_away", 0),
            "dc12_vs": analysis.get("dc12_vs", 0),
            "dc12_odds": analysis.get("dc12_odds", 0),
            "predicted": decision,
            "confidence": analysis.get("confidence", "LOW"),
        }
        
        response = supabase.table(TABLE_NAME).insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending():
    try:
        response = supabase.table(TABLE_NAME).select("*").is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except:
        return []

def submit_result(analysis_id, home_goals, away_goals):
    try:
        actual_result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        response = supabase.table(TABLE_NAME).select("predicted").eq("id", analysis_id).execute()
        if response.data:
            predicted = response.data[0].get("predicted")
            is_correct = (predicted == "BET" and actual_result != "X") or (predicted == "NO_BET" and actual_result == "X")
        else:
            is_correct = False
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_result": actual_result,
            "is_correct": is_correct
        }).eq("id", analysis_id).execute()
        return True
    except:
        return False

def get_results():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except:
        return []

def display_records_table(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct'))
    incorrect = total - correct
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
    st.markdown(f"**Overall: {correct} wins | {incorrect} losses**")
    rows = []
    for r in results:
        pred = r.get('predicted', '?')
        actual = r.get('actual_result', '?')
        is_correct = r.get('is_correct', False)
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        pred_display = "🎯 BET" if pred == "BET" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ NO BET"
        actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
        rows.append({
            "Date": r.get("match_date", ""),
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": pred_display,
            "Actual": actual_display,
            "DC12-VS": r.get('dc12_vs', 0),
            "Result": result_badge,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("🎯 DC12-VS: Double Chance 12 Value Score V2.0")
    st.caption(f"Bet on Home Win OR Away Win (No Draw) | ND is CORE (×10) | Table: {TABLE_NAME}")

    with st.expander("📖 HOW DC12-VS V2.0 WORKS", expanded=False):
        st.markdown("""
        **Double Chance 12** = Betting that the match will **NOT end in a draw** (Home Win OR Away Win)

        ### The DC12-VS Formula (V2.0):
        DC12-VS = (ND_Home + ND_Away) × 10 + (W_Home + W_Away) × 2 - (L_Home + L_Away) × 2 + (BestTeam_Home + BestTeam_Away) × 1 - (WorstTeam_Home + WorstTeam_Away) × 1 + (BestOff_Home + BestOff_Away) × 1 + (WorstDef_Home + WorstDef_Away) × 1 - (NoW_Home + NoW_Away) × 3 - (NoL_Home + NoL_Away) × 3

        ### Decision Rules:
        | Score | Draw Odds | Decision |
        |-------|-----------|----------|
        | **> 80** | **> 3.00** | ✅ **BET** |
        | **50-80** | **> 3.00** | ⚠️ **CONSIDER** |
        | **< 50** | Any | ❌ **SKIP** |
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Analyze", "📝 Pending Matches", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("DC12-VS V2.0: ND is CORE (×10). Only BET/CONSIDER matches are stored.")

        text_data = st.text_area(
            "Paste Betexplorer data here",
            height=300,
            key="text_paste",
            placeholder="Paste all Betexplorer page data here..."
        )

        if st.button("🎯 CALCULATE DC12-VS V2.0", type="primary"):
            if not text_data or len(text_data.strip()) < 10:
                st.error("❌ Please paste valid data (minimum 10 characters).")
            else:
                try:
                    with st.spinner("Calculating DC12-VS V2.0 scores..."):
                        matches = parse_betexplorer_data(text_data)
                    if matches:
                        st.success(f"✅ Found {len(matches)} unique matches")
                        analyzed_results = []
                        stored_count = already_stored_count = bet_count = consider_count = 0
                        
                        for match in matches:
                            analysis = calculate_dc12_vs(match)
                            exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                            
                            if exists:
                                already_stored_count += 1
                                analyzed_results.append((match, analysis, True))
                            else:
                                saved_id = save_to_db(match, analysis)
                                if saved_id == "ALREADY_EXISTS":
                                    already_stored_count += 1
                                    analyzed_results.append((match, analysis, True))
                                elif saved_id == "SKIPPED":
                                    analyzed_results.append((match, analysis, False))
                                elif saved_id:
                                    stored_count += 1
                                    analyzed_results.append((match, analysis, False))
                                    if analysis.get("decision") == "BET":
                                        bet_count += 1
                                    elif analysis.get("decision") == "CONSIDER":
                                        consider_count += 1
                                else:
                                    analyzed_results.append((match, analysis, False))
                        
                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed | 🎯 {bet_count} bets | ⚠️ {consider_count} considered")
                        
                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🎯 DC12-VS V2.0 RESULTS")
                            bets = [(m, a, s) for m, a, s in analyzed_results if a.get("decision") == "BET"]
                            considers = [(m, a, s) for m, a, s in analyzed_results if a.get("decision") == "CONSIDER"]
                            skips = [(m, a, s) for m, a, s in analyzed_results if a.get("decision") == "SKIP"]

                            if bets:
                                st.markdown("#### ✅ BETS TO PLACE")
                                for idx, (match, analysis, already_stored) in enumerate(bets, 1):
                                    st.markdown(f"##### Bet #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    st.caption(f"🎯 DC12-VS: {analysis.get('dc12_vs', 0)} | DC12 Odds: {analysis.get('dc12_odds', 0):.2f} | Draw Odds: {analysis.get('draw_odds', 0):.2f}")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("DC12-VS Score", analysis.get('dc12_vs', 0))
                                    with col2:
                                        st.metric("DC12 Odds", f"{analysis.get('dc12_odds', 0):.2f}")
                                    with col3:
                                        st.metric("Draw Odds", f"{analysis.get('draw_odds', 0):.2f}")
                                    display_dc12_analysis(match, analysis, already_stored)
                                    if idx < len(bets):
                                        st.markdown("---")

                            if considers:
                                st.markdown("#### ⚠️ CONSIDER BETS")
                                for idx, (match, analysis, already_stored) in enumerate(considers, 1):
                                    with st.expander(f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')} - DC12-VS: {analysis.get('dc12_vs', 0)}"):
                                        display_dc12_analysis(match, analysis, already_stored)

                            if skips:
                                st.markdown("#### ❌ SKIPPED (No Value)")
                                st.caption(f"Total skipped: {len(skips)} matches (DC12-VS < 50 or Draw Odds ≤ 3.00)")

                            st.markdown("---")
                            st.markdown("### 📊 Summary")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Total Matches", len(matches))
                            with col2:
                                st.metric("🎯 Bets Found", bet_count)
                            with col3:
                                st.metric("⚠️ Considered", consider_count)
                            with col4:
                                st.metric("💾 Stored", stored_count)
                            with col5:
                                st.metric("📌 Already Stored", already_stored_count)
                    else:
                        st.error("No matches found in the data. Please check the format.")
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Pending Matches")
        st.caption("Enter actual scores once matches are played.")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('predicted', '?')
                confidence = a.get('confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                dc12_vs = a.get('dc12_vs', 0)
                pred_display = "🎯 BET" if pred == "BET" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ NO BET"
                badge = f"{pred_display} ({confidence}) — DC12-VS: {dc12_vs}"
                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred_display} — DC12-VS: {dc12_vs}")
                    st.caption(f"📅 Match Date: {match_date}")
                    c1, c2 = st.columns(2)
                    with c1: 
                        hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: 
                        ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending matches.")

    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        display_records_table(results)

    with tab4:
        st.subheader("📊 Live Dashboard")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
            return
        total = len(results)
        correct = sum(1 for r in results if r.get('is_correct'))
        incorrect = total - correct
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
        with col2:
            win_rate = round(correct / total * 100) if total > 0 else 0
            st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
        st.markdown(f"**Overall: {correct} wins | {incorrect} losses**")
        rows = []
        for r in results:
            pred = r.get('predicted', '?')
            actual = r.get('actual_result', '?')
            is_correct = r.get('is_correct', False)
            result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
            pred_display = "🎯 BET" if pred == "BET" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ NO BET"
            actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
            rows.append({
                "Date": r.get("match_date", ""),
                "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
                "Prediction": pred_display,
                "Actual": actual_display,
                "DC12-VS": r.get('dc12_vs', 0),
                "Result": result_badge,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()
