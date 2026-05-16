"""
MATCH ANALYZER V4.0 — PROFIT-DRIVEN ENGINE
Based on performance analysis: LOW-SCORING (75%), HIGH-SCORING (100%), AWAY THREAT (100%)
Removed: DRAW PRESSURE, COMPRESSION (50% win rate = random)
Tightened: BTTS (65%+ thresholds)
Prioritized: Low-scoring patterns first
Skip rate target: 40-50% (only clear edges)
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
import re
import json

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
st.set_page_config(page_title="Match Analyzer V4.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .secondary-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); opacity: 0.9; }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .badge-upgrade { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; margin: 0.1rem; }
    .badge-caution { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #ef4444; color: #fff; margin: 0.1rem; }
    .badge-info { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #3b82f6; color: #fff; margin: 0.1rem; }
    .badge-skip { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .correct-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .incorrect-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .verdict-skip { text-align: center; padding: 1.5rem; }
    .verdict-skip .big-text { font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .section-label { font-size: 0.9rem; font-weight: 700; color: #10b981; margin-top: 1rem; }
    .section-label-secondary { font-size: 0.9rem; font-weight: 700; color: #f59e0b; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TOP LEAGUES
# ============================================================================
TOP_LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga", "Eredivisie", "Saudi Pro League", "Major League Soccer",
    "Championship", "Argentine Primera Division"
]

# ============================================================================
# TEAM ABBREVIATIONS (abbreviated for space - add your full dict)
# ============================================================================
TEAM_ABBREVIATIONS = {
    "Manchester United": ["man utd", "manchester utd", "man united"],
    "Manchester City": ["man city", "manchester city"],
    "Tottenham Hotspur": ["spurs", "tottenham"],
    "West Ham United": ["west ham", "west ham utd"],
    "Nottingham Forest": ["nott'm forest", "nottm forest"],
    "Newcastle United": ["newcastle", "newcastle utd"],
    "Wolverhampton Wanderers": ["wolves", "wolverhampton"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Brighton & Hove Albion": ["brighton"],
    "Aston Villa": ["aston villa", "villa"],
    "Bayern Munich": ["bayern", "fc bayern", "bayern munich"],
    "Borussia Dortmund": ["dortmund", "bvb"],
    "Real Madrid": ["real madrid", "real"],
    "Barcelona": ["barcelona", "barca"],
    "Atletico Madrid": ["atletico", "atlético"],
    "Paris Saint-Germain": ["psg", "paris sg"],
    "Liverpool": ["liverpool"],
    "Arsenal": ["arsenal"],
    "Chelsea": ["chelsea"],
    "Burnley": ["burnley"],
}

def fuzzy_team_match(team_name, text):
    if not team_name or not text:
        return False
    text_lower = text.lower().strip()
    team_lower = team_name.lower().strip()
    if team_lower in text_lower or text_lower in team_lower:
        return True
    for abbr in TEAM_ABBREVIATIONS.get(team_name, []):
        if abbr in text_lower:
            return True
    return False

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
        "home_over_05_goals": None, "home_over_15_goals": None,
        "away_over_05_goals": None, "away_over_15_goals": None,
        "home_win_trend": None, "draw_trend": None, "away_win_trend": None,
        "btts_trend": None, "over_25_trend": None,
        "home_form_all": [], "away_form_all": [],
        "h2h_scores": [], "h2h_btts_count": 0, "h2h_total": 0,
        "score_matrix": [],
    }
    
    # Team names
    team_names = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'All competitions' and i > 0:
            name = lines[i-1].strip()
            if name and not name.startswith('Last game') and not name.startswith('Goals scored') and not name.startswith('Top scorer') and name not in team_names:
                team_names.append(name)
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # League detection
    leagues_found = []
    in_form_data = False
    for line in lines:
        if 'Form Data' in line:
            in_form_data = True
        if in_form_data:
            continue
        m = re.search(r'(Premier League|La Liga|Bundesliga|Serie A|Serie B|Ligue 1|Championship|'
                      r'Süper Lig|Pro League|Primeira Liga|EFL Cup|Swiss Super League|'
                      r'Saudi Pro League|Ukrainian Premier League|Belarusian Premier League|'
                      r'Liga MX|League One|League Two|Argentine Primera Division|'
                      r'Major League Soccer|Segunda Division|Segunda División|'
                      r'Croatian 1\. HNL|HNL|Prva HNL|Scottish Premiership|Scottish Premiership Playoffs|'
                      r'Eredivisie|A-League|Ekstraklasa|Polish Ekstraklasa|'
                      r'Turkish Super Lig|Süper Lig|Israeli Premier League|Ligat HaAl|'
                      r'Hungarian NB I|OTP Bank Liga|Romanian SuperLiga|SuperLiga|'
                      r'Chinese Super League|CSL|Australian A-League|'
                      r'Scottish Championship|League Two|League of Ireland Premier Division|'
                      r'Belgian Pro League|Jupiler Pro League)', 
                      line, re.IGNORECASE)
        if m and 'Gameweek' not in line and 'Head to Head' not in line:
            league_name = m.group(1)
            if league_name not in leagues_found:
                leagues_found.append(league_name)
    
    if leagues_found:
        data["league"] = leagues_found[0]
    
    def find_pct(start_idx, max_lookahead=3):
        for j in range(start_idx, min(start_idx + max_lookahead, len(lines))):
            sub = lines[j].strip()
            m = re.search(r'(\d+\.?\d*)\s*%', sub)
            if m:
                prob = float(m.group(1))
                trend_m = re.search(r'([+-]\d+\.?\d*)', sub)
                trend = float(trend_m.group(1)) if trend_m else 0
                return prob, trend
        return None, 0

    def has_arrow(start_idx, max_lookahead=3):
        for j in range(start_idx, min(start_idx + max_lookahead, len(lines))):
            sub = lines[j].strip()
            if '▲' in sub or '▼' in sub:
                return True
        return False
    
    current_section = None
    current_subsection = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped in ['Result', 'Goals', 'First Half Winner', 'Team To Score First', 
                        'Corners', 'Score analysis']:
            current_section = stripped.lower().replace(' ', '_')
            current_subsection = None
            continue
        if stripped == 'Head to Head':
            current_section = 'h2h'
            current_subsection = None
            continue
        if stripped == 'Form Data':
            current_section = 'form'
            current_subsection = None
            continue
        
        if current_section == 'result':
            if data["home_team"] and data["home_team"] in stripped:
                prob, trend = find_pct(i)
                if prob: 
                    data["home_win"] = prob
                    data["home_win_trend"] = trend if has_arrow(i) else None
            elif stripped.startswith('Draw'):
                prob, trend = find_pct(i)
                if prob: 
                    data["draw"] = prob
                    data["draw_trend"] = trend if has_arrow(i) else None
            elif data["away_team"] and data["away_team"] in stripped:
                prob, trend = find_pct(i)
                if prob: 
                    data["away_win"] = prob
                    data["away_win_trend"] = trend if has_arrow(i) else None
            elif 'Both Teams to Score' in stripped:
                prob, trend = find_pct(i)
                if prob: 
                    data["btts"] = prob
                    data["btts_trend"] = trend if has_arrow(i) else None
        
        if current_section == 'goals':
            if 'Over 1.5' in stripped and 'Goals' not in stripped:
                prob, _ = find_pct(i)
                if prob: data["over_15"] = prob
            elif 'Over 2.5' in stripped and 'Goals' not in stripped:
                prob, trend = find_pct(i)
                if prob: 
                    data["over_25"] = prob
                    data["over_25_trend"] = trend if has_arrow(i) else None
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
            elif 'Over 1.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["home_over_15_goals"] = prob
        
        if current_subsection == 'away_goals':
            if 'Over 0.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["away_over_05_goals"] = prob
            elif 'Over 1.5' in stripped:
                prob, _ = find_pct(i)
                if prob: data["away_over_15_goals"] = prob
    
    # SCORE MATRIX
    in_score_analysis = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped == 'Score analysis':
            in_score_analysis = True
            continue
        if in_score_analysis and stripped == 'Head to Head':
            break
        if in_score_analysis and 'Form Data' in stripped:
            break
        
        if in_score_analysis:
            m = re.match(r'(\d+)-(\d+)\s*@\s*(\d+\.?\d*)\s*%', stripped)
            if m:
                home = int(m.group(1))
                away = int(m.group(2))
                prob = float(m.group(3))
                data["score_matrix"].append({
                    "score": "{}-{}".format(home, away),
                    "home_goals": home,
                    "away_goals": away,
                    "probability": prob
                })
    
    data["score_matrix"].sort(key=lambda x: x["probability"], reverse=True)
    data["score_matrix"] = data["score_matrix"][:10]
    
    # Form strings
    form_blocks = []
    current_block = []
    collecting = False
    
    league_headers = [
        'All competitions', 'Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Serie B',
        'Ligue 1', 'Championship', 'Süper Lig', 'Pro League', 'Primeira Liga',
        'Swiss Super League', 'Saudi Pro League', 'Ukrainian Premier League',
        'Belarusian Premier League', 'Liga MX', 'League One', 'League Two',
        'Argentine Primera Division', 'Major League Soccer', 'Segunda Division',
        'Segunda División', 'Scottish Premiership', 'Scottish Premiership Playoffs',
        'Eredivisie', 'A-League', 'Croatian 1. HNL', 'HNL', 'Prva HNL', 'Ekstraklasa',
        'Turkish Super Lig', 'Israeli Premier League', 'Hungarian NB I',
        'Romanian SuperLiga', 'Chinese Super League', 'Australian A-League',
        'Scottish Championship', 'League of Ireland Premier Division',
        'Belgian Pro League', 'Jupiler Pro League'
    ]
    
    for line in lines:
        stripped = line.strip()
        
        if stripped in league_headers:
            if current_block and len(current_block) >= 4:
                form_blocks.append(current_block)
            current_block = []
            collecting = True
            continue
        
        if stripped.startswith('Last game') or stripped.startswith('Goals scored') or stripped.startswith('Top scorer'):
            if current_block and len(current_block) >= 4:
                form_blocks.append(current_block)
            current_block = []
            collecting = False
            continue
        
        if stripped == 'Data analysis':
            if current_block and len(current_block) >= 4:
                form_blocks.append(current_block)
            current_block = []
            collecting = False
            continue
        
        if collecting and stripped in ['W', 'D', 'L']:
            current_block.append(stripped)
    
    if current_block and len(current_block) >= 4:
        form_blocks.append(current_block)
    
    if len(form_blocks) >= 1:
        data["home_form_all"] = form_blocks[0][:6]
    if len(form_blocks) >= 3:
        data["away_form_all"] = form_blocks[2][:6]
    if not data["away_form_all"] and len(form_blocks) >= 2:
        data["away_form_all"] = form_blocks[1][:6]
    
    # Form Data fallback
    form_data_started = False
    current_team_form = None
    team_forms = {}
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Form Data' in stripped:
            form_data_started = True
            continue
        if not form_data_started:
            continue
        
        is_team = False
        for team in team_names:
            if team and fuzzy_team_match(team, stripped):
                is_team = True
                current_team_form = team
                if team not in team_forms:
                    team_forms[team] = []
                break
        
        if is_team:
            continue
        
        if stripped in ['W', 'D', 'L'] and current_team_form:
            team_forms[current_team_form].append(stripped)
    
    if not data["home_form_all"] and data["home_team"] in team_forms:
        data["home_form_all"] = team_forms[data["home_team"]][:6]
    if not data["away_form_all"] and data["away_team"] in team_forms:
        data["away_form_all"] = team_forms[data["away_team"]][:6]
    
    # H2H
    h2h_section = False
    h2h_scores = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if stripped == 'Head to Head':
            h2h_section = True
            continue
        if h2h_section and 'Form Data' in stripped:
            break
        
        if h2h_section and stripped == 'FT':
            numbers_found = []
            for j in range(i-1, max(i-15, 0), -1):
                prev_line = lines[j].strip()
                m = re.match(r'^(\d+)$', prev_line)
                if m:
                    num = int(m.group(1))
                    if num < 20:
                        numbers_found.append(num)
                        if len(numbers_found) == 2:
                            break
                if prev_line in ['FT', 'HT'] or re.match(r'\d{1,2}:\d{2}', prev_line):
                    break
            
            if len(numbers_found) == 2:
                away_score = numbers_found[0]
                home_score = numbers_found[1]
                h2h_scores.append((home_score, away_score))
    
    data["h2h_scores"] = h2h_scores
    data["h2h_total"] = len(h2h_scores)
    data["h2h_btts_count"] = sum(1 for h, a in h2h_scores if h > 0 and a > 0)
    
    return data


# ============================================================================
# OVERHAULED STRUCTURAL ENGINE V4.0
# ============================================================================
def analyze_match(data: dict) -> dict:
    result = {
        "primary_bet": None,
        "secondary_bet": None,
        "badges": [],
        "warnings": [],
        "verdict": "PENDING",
        "classification": None,
        "skip_reasons": []
    }
    
    # Extract measurements
    home_win = data.get("home_win") or 0
    away_win = data.get("away_win") or 0
    draw_pct = data.get("draw") or 0
    btts = data.get("btts") or 0
    over_25 = data.get("over_25") or 0
    over_35 = data.get("over_35") or 0
    under_25 = data.get("under_25") or 0
    
    home_o15 = data.get("home_over_15_goals") or 0
    away_o15 = data.get("away_over_15_goals") or 0
    home_o05 = data.get("home_over_05_goals")
    away_o05 = data.get("away_over_05_goals")
    
    if home_o05 is None and btts:
        home_o05 = btts
    if away_o05 is None and btts:
        away_o05 = btts
    
    home_win_trend = data.get("home_win_trend")
    away_win_trend = data.get("away_win_trend")
    btts_trend = data.get("btts_trend")
    
    home_form = data.get("home_form_all") or []
    away_form = data.get("away_form_all") or []
    home_losses = sum(1 for r in home_form[:6] if r == 'L')
    away_losses = sum(1 for r in away_form[:6] if r == 'L')
    
    score_matrix = data.get("score_matrix", [])
    
    league = data.get("league")
    is_top_league = league in TOP_LEAGUES if league else False
    is_saudi = league == "Saudi Pro League" if league else False
    
    # ========================================================================
    # SCORE MATRIX STRUCTURE ANALYSIS
    # ========================================================================
    tight_cluster = False
    btts_dominant = False
    low_scoring_cluster = False
    goals_expected = False
    modal_outcome = None
    home_cluster = 0
    away_cluster = 0
    draw_cluster = 0
    
    if len(score_matrix) >= 5:
        top5_spread = score_matrix[0]["probability"] - score_matrix[4]["probability"]
        tight_cluster = top5_spread < 5.0
        
        btts_count = sum(1 for s in score_matrix[:5] if s["home_goals"] > 0 and s["away_goals"] > 0)
        btts_dominant = btts_count >= 4  # Stricter: 4 of top 5
        
        low_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] <= 2)
        low_scoring_cluster = low_count >= 4
        
        goals_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] >= 3)
        goals_expected = goals_count >= 4  # Stricter: 4 of top 5
        
        modal_outcome = score_matrix[0]["score"]
        
        for s in score_matrix[:5]:
            if s["home_goals"] > s["away_goals"]:
                home_cluster += s["probability"]
            elif s["away_goals"] > s["home_goals"]:
                away_cluster += s["probability"]
            else:
                draw_cluster += s["probability"]
    
    under_35_pct = (100 - over_35) if over_35 else 0
    
    # ========================================================================
    # CONDITION DEFINITIONS (STRICT)
    # ========================================================================
    
    # Trend Reversal (stricter: ±2.0 instead of ±1.0)
    trend_reversal = False
    if home_win_trend is not None and away_win_trend is not None:
        if home_win_trend <= -2.0 and away_win_trend >= 2.0:
            trend_reversal = True
    
    # True Strong Favorite (increased thresholds)
    is_true_strong = (home_win >= 65 and home_o15 >= 60 and not tight_cluster and home_losses <= 2)
    is_true_strong_away = (away_win >= 65 and away_o15 >= 60 and not tight_cluster and away_losses <= 2)
    
    # LOW-SCORING (Priority #1 - 75% win rate historically)
    is_low_scoring = (
        low_scoring_cluster and 
        under_25 >= 55 and 
        btts < 50 and
        not (home_losses >= 4 and away_losses >= 4)
    )
    
    # HIGH-SCORING (Priority #2 - 100% win rate historically)
    is_high_scoring = (
        goals_expected and 
        over_25 >= 60 and 
        btts >= 55 and
        home_o15 >= 50 and 
        away_o15 >= 50
    )
    
    # AWAY THREAT (Priority #3 - 100% win rate historically)
    is_away_threat = (home_o15 < 35 and away_win >= 35 and away_o15 >= 30)
    
    # BTTS (DEMOTED, STRICT THRESHOLDS)
    btts_contradiction = (btts >= 60 and (home_o15 < 40 or away_o15 < 40))
    is_btts_play = (
        btts >= 65 and
        btts_dominant and
        home_o05 is not None and home_o05 >= 65 and
        away_o05 is not None and away_o05 >= 65 and
        home_o15 >= 40 and
        away_o15 >= 40 and
        not btts_contradiction and
        (btts_trend is None or btts_trend >= 0)
    )
    
    # Fragile Favorite (higher bar)
    is_fragile = (home_win >= 60 and home_o15 < 55 and home_o15 >= 30)
    is_fragile_away = (away_win >= 60 and away_o15 < 55 and away_o15 >= 30)
    
    # Saudi League Override
    is_saudi_dominant = (is_saudi and home_win >= 65)
    
    # ========================================================================
    # BETTING DECISIONS
    # ========================================================================
    
    if trend_reversal:
        result["classification"] = "TREND REVERSAL"
        result["badges"].append(f"Trend Reversal: Home {home_win_trend:.1f} / Away {away_win_trend:.1f}")
        result["primary_bet"] = {
            "market": "Away Win or Draw",
            "confidence": 8.0,
            "probability": away_win + draw_pct,
            "reason": f"Market reversal. Fade home team. Home trend {home_win_trend:.1f}, Away trend {away_win_trend:.1f}"
        }
    
    elif is_true_strong:
        result["classification"] = "TRUE STRONG FAVORITE"
        if away_o05 is not None and away_o05 < 55:
            result["primary_bet"] = {
                "market": "Home Win to Nil",
                "confidence": 8.5,
                "probability": home_win,
                "reason": f"Dominant favorite ({home_win:.0f}% win, O1.5 {home_o15:.0f}%) + underdog unlikely to score ({away_o05:.0f}%)"
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win",
                "confidence": 7.5,
                "probability": home_win,
                "reason": f"Dominant favorite ({home_win:.0f}% win, O1.5 {home_o15:.0f}%)"
            }
    
    elif is_true_strong_away:
        result["classification"] = "TRUE STRONG FAVORITE (AWAY)"
        if home_o05 is not None and home_o05 < 55:
            result["primary_bet"] = {
                "market": "Away Win to Nil",
                "confidence": 8.5,
                "probability": away_win,
                "reason": f"Dominant away favorite ({away_win:.0f}% win, O1.5 {away_o15:.0f}%) + home unlikely to score ({home_o05:.0f}%)"
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win",
                "confidence": 7.5,
                "probability": away_win,
                "reason": f"Dominant away favorite ({away_win:.0f}% win, O1.5 {away_o15:.0f}%)"
            }
    
    elif is_low_scoring:
        result["classification"] = "LOW-SCORING"
        if under_25 >= 60:
            result["primary_bet"] = {
                "market": "Under 2.5 Goals",
                "confidence": 7.5,
                "probability": under_25,
                "reason": f"Low-scoring cluster. Under 2.5 at {under_25:.1f}%. BTTS only {btts:.1f}%."
            }
        else:
            result["primary_bet"] = {
                "market": "Under 3.5 Goals",
                "confidence": 7.0,
                "probability": under_35_pct,
                "reason": f"Low-scoring profile. Under 3.5 at {under_35_pct:.0f}%."
            }
    
    elif is_high_scoring:
        result["classification"] = "HIGH-SCORING"
        result["primary_bet"] = {
            "market": "Over 2.5 Goals",
            "confidence": 7.5,
            "probability": over_25,
            "reason": f"High-scoring matrix. Over 2.5 at {over_25:.1f}%. Both teams can score."
        }
        if btts >= 60:
            result["secondary_bet"] = {
                "market": "BTTS",
                "confidence": 6.5,
                "probability": btts,
                "reason": "Secondary: BTTS in high-scoring setup"
            }
    
    elif is_away_threat:
        result["classification"] = "AWAY THREAT"
        result["primary_bet"] = {
            "market": "Away Win or Draw (Double Chance)",
            "confidence": 7.0,
            "probability": away_win + draw_pct,
            "reason": f"Home O1.5 only {home_o15:.1f}% — home can't score. Away won't lose."
        }
    
    elif is_btts_play:
        result["classification"] = "BTTS"
        confidence = 7.0 if btts_trend and btts_trend >= 0.5 else 6.5
        result["primary_bet"] = {
            "market": "BTTS",
            "confidence": confidence,
            "probability": btts,
            "reason": f"Strong BTTS setup: {btts:.1f}% probability, both teams reliable scorers."
        }
    
    elif is_fragile:
        result["classification"] = "FRAGILE FAVORITE"
        result["badges"].append(f"Fragile favorite — O1.5 only {home_o15:.0f}%")
        if is_away_threat:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": away_win + draw_pct,
                "reason": f"Fragile favorite ({home_win:.0f}% win) + away threat. Home vulnerable."
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": home_win + draw_pct,
                "reason": f"Fragile favorite ({home_win:.0f}% win) but underdog can't score."
            }
    
    elif is_fragile_away:
        result["classification"] = "FRAGILE FAVORITE (AWAY)"
        result["badges"].append(f"Fragile away favorite — O1.5 only {away_o15:.0f}%")
        if home_o15 >= 35:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": home_win + draw_pct,
                "reason": f"Fragile away favorite ({away_win:.0f}% win) + home scoring threat."
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": away_win + draw_pct,
                "reason": f"Fragile away favorite but home can't score."
            }
    
    elif is_saudi_dominant:
        result["classification"] = "SAUDI DOMINANT"
        result["primary_bet"] = {
            "market": "Home Win to Nil",
            "confidence": 7.5,
            "probability": home_win,
            "reason": "Saudi league structural dominance"
        }
    
    # ========================================================================
    # FINALIZE - SKIP if no bet or confidence too low
    # ========================================================================
    if result["primary_bet"]:
        # Skip if confidence < 7.0 (except LOW-SCORING which can be 7.0)
        if result["primary_bet"]["confidence"] < 7.0:
            result["verdict"] = "SKIP"
            result["skip_reasons"].append(f"Confidence too low ({result['primary_bet']['confidence']}/10) - skipping marginal edge")
            result["primary_bet"] = None
            result["classification"] = "SKIP"
        else:
            result["verdict"] = "RECOMMENDED"
    else:
        result["verdict"] = "SKIP"
        if not result.get("skip_reasons"):
            result["skip_reasons"].append("No structural pattern matched with sufficient confidence")
        result["classification"] = "SKIP"
    
    # Warnings
    if draw_pct >= 28:
        result["warnings"].append(f"High draw probability ({draw_pct:.1f}%) — avoid straight match result bets")
    if not is_top_league and league:
        result["warnings"].append(f"'{league}' is not a top league — lower reliability")
    
    return result


# ============================================================================
# TRUTH-BASED EVALUATION ENGINE
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID DATA", "message": "Non-numeric score"}
    
    total = home + away
    btts = home > 0 and away > 0
    over25 = total > 2
    over35 = total > 3
    
    if home > away:
        winner = "HOME"
    elif away > home:
        winner = "AWAY"
    else:
        winner = "DRAW"
    
    pred = primary_pred.strip()
    is_correct = False
    
    if pred == "BTTS":
        is_correct = btts
    elif pred == "Over 2.5 Goals":
        is_correct = over25
    elif pred == "Under 2.5 Goals":
        is_correct = not over25
    elif pred == "Over 3.5 Goals":
        is_correct = over35
    elif pred == "Under 3.5 Goals":
        is_correct = total <= 3
    elif pred == "Home Win":
        is_correct = winner == "HOME"
    elif pred == "Away Win":
        is_correct = winner == "AWAY"
    elif pred == "Home Win to Nil":
        is_correct = (winner == "HOME" and away == 0)
    elif pred == "Away Win to Nil":
        is_correct = (winner == "AWAY" and home == 0)
    elif "Away Win or Draw" in pred:
        is_correct = winner in ["AWAY", "DRAW"]
    elif "Home Win or Draw" in pred:
        is_correct = winner in ["HOME", "DRAW"]
    elif pred == "Home Over 1.5 Goals":
        is_correct = home >= 2
    elif pred == "Away Over 1.5 Goals":
        is_correct = away >= 2
    else:
        return {"is_correct": False, "actual": f"{home}-{away}", "message": f"Unknown market: {pred}"}
    
    return {
        "is_correct": is_correct,
        "actual": f"{home}-{away}",
        "message": f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}: {pred} vs {home}-{away}"
    }


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    try:
        primary = analysis.get("primary_bet")
        secondary = analysis.get("secondary_bet")
        
        if primary:
            bets_str = primary["market"]
            if secondary:
                bets_str += " | " + secondary["market"]
        else:
            bets_str = "SKIP"
        
        top = primary if primary else secondary
        
        home_form_str = '-'.join(data.get("home_form_all", [])) if data.get("home_form_all") else ""
        away_form_str = '-'.join(data.get("away_form_all", [])) if data.get("away_form_all") else ""
        
        over_35 = data.get("over_35") or 0
        under_35 = (100 - over_35) if over_35 else 0
        
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
                "over15_pct": data.get("over_15"),
                "over25_pct": data.get("over_25"),
                "under25_pct": data.get("under_25"),
                "over35_pct": data.get("over_35"),
                "under35_pct": under_35,
                "home_over05_pct": data.get("home_over_05_goals"),
                "home_over15_pct": data.get("home_over_15_goals"),
                "away_over05_pct": data.get("away_over_05_goals"),
                "away_over15_pct": data.get("away_over_15_goals"),
                "home_win_trend": data.get("home_win_trend"),
                "draw_trend": data.get("draw_trend"),
                "away_win_trend": data.get("away_win_trend"),
                "btts_trend": data.get("btts_trend"),
                "over25_trend": data.get("over_25_trend"),
                "home_form": home_form_str,
                "away_form": away_form_str,
                "h2h_scores": json.dumps(data.get("h2h_scores", [])),
                "h2h_total": data.get("h2h_total"),
                "h2h_btts_count": data.get("h2h_btts_count"),
            },
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            "prediction": bets_str,
            "confidence_score": round(top["confidence"] / 10, 2) if top else 0,
            "winner": top["market"] if top else "SKIP",
            "winner_confidence": f"{top['confidence']}/10" if top else "0",
            "classification": analysis.get("classification"),
            "btts": "BTTS YES" if primary and "BTTS" in primary["market"] else "",
            "btts_confidence": round(primary["confidence"] / 10, 2) if primary and "BTTS" in primary["market"] else 0,
            "pattern": "PRIMARY" if primary else "SKIP",
            "result_entered": False,
        }
        response = supabase.table("match_analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except: 
        return []

def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        over25 = total > 2
        actual_winner = "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW"
        btts_yes = home_goals > 0 and away_goals > 0
        
        supabase.table("match_analyses").update({
            "actual_home_goals": home_goals, 
            "actual_away_goals": away_goals,
            "actual_total_goals": total, 
            "actual_over25": over25,
            "actual_winner": actual_winner, 
            "actual_btts": btts_yes,
            "result_entered": True,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_results():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except: 
        return []


# ============================================================================
# MAIN APP INTERFACE
# ============================================================================
def main():
    st.title("📊 Match Analyzer V4.0")
    st.caption("Profit-Driven Engine | LOW-SCORING Priority | High Thresholds | No Garbage Categories")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ========================================================================
    # TAB 1: ANALYZE
    # ========================================================================
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        st.markdown("*Paste the full match data block from 'Form, Standings, Stats' through 'Form Data'*")
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                with st.spinner("Running structural analysis..."):
                    data = parse_match_data(raw_text)
                
                if not data.get("home_team") or not data.get("away_team"):
                    st.error("Could not detect team names. Please check the format.")
                else:
                    analysis = analyze_match(data)
                    save_to_db(data, analysis)
                    
                    league_display = data.get('league') or 'Club Match'
                    
                    if analysis["verdict"] == "SKIP":
                        st.warning(f"⚠️ SKIP: {data['home_team']} vs {data['away_team']} — {league_display}")
                    else:
                        st.success(f"✅ RECOMMENDED: {data['home_team']} vs {data['away_team']} — {league_display}")
                    
                    if analysis.get("classification"):
                        st.markdown(f"**Classification: {analysis['classification']}**")
                    
                    # Probability display
                    col1, col2 = st.columns(2)
                    with col1:
                        home_o05_str = f"{data['home_over_05_goals']:.1f}%" if data.get('home_over_05_goals') is not None else "N/A"
                        home_o15_str = f"{data['home_over_15_goals']:.1f}%" if data.get('home_over_15_goals') is not None else "N/A"
                        away_o05_str = f"{data['away_over_05_goals']:.1f}%" if data.get('away_over_05_goals') is not None else "N/A"
                        away_o15_str = f"{data['away_over_15_goals']:.1f}%" if data.get('away_over_15_goals') is not None else "N/A"
                        
                        st.markdown(f"""
                        <div class="edge-box">
                            <strong>📊 Probabilities</strong><br>
                            Home: {data.get('home_win', 0):.1f}% | Draw: {data.get('draw', 0):.1f}% | Away: {data.get('away_win', 0):.1f}%<br>
                            BTTS: {data.get('btts', 0):.1f}% | O2.5: {data.get('over_25', 0):.1f}% | U2.5: {data.get('under_25', 0):.1f}%<br>
                            Home O0.5: {home_o05_str} | Home O1.5: {home_o15_str}<br>
                            Away O0.5: {away_o05_str} | Away O1.5: {away_o15_str}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        home_form_str = '-'.join(data.get('home_form_all', [])[:6]) if data.get('home_form_all') else 'N/A'
                        away_form_str = '-'.join(data.get('away_form_all', [])[:6]) if data.get('away_form_all') else 'N/A'
                        
                        st.markdown(f"""
                        <div class="edge-box">
                            <strong>📈 Form & H2H</strong><br>
                            Home: {home_form_str}<br>Away: {away_form_str}<br>H2H BTTS: {data.get('h2h_btts_count', 0)}/{data.get('h2h_total', 0)}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Score Matrix
                    if data.get("score_matrix"):
                        st.markdown("### 🎯 Score Matrix (Top 6)")
                        score_cols = st.columns(6)
                        for idx, s in enumerate(data["score_matrix"][:6]):
                            with score_cols[idx]:
                                bg = "#1e293b" if s["home_goals"] != s["away_goals"] else "#2a1a00"
                                st.markdown(f"""
                                <div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;">
                                    <div style="font-size:1.2rem; font-weight:800;">{s['score']}</div>
                                    <div style="font-size:0.7rem; color:#94a3b8;">{s['probability']:.1f}%{' 👑' if idx == 0 else ''}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # PRIMARY BET
                    if analysis.get("primary_bet"):
                        primary = analysis["primary_bet"]
                        emoji_map = {
                            "BTTS": "⚽⚽", "Over 2.5 Goals": "🔥", "Under 2.5 Goals": "🛡️", 
                            "Under 3.5 Goals": "🛡️", "Home Win to Nil": "🏠🧤", "Away Win to Nil": "✈️🧤",
                            "Home Win": "🏠", "Away Win": "✈️", "Away Win or Draw": "✈️🤝",
                            "Home Win or Draw": "🏠🤝"
                        }
                        emoji = emoji_map.get(primary["market"], "📊")
                        
                        st.markdown('<div class="section-label">🎯 PRIMARY BET</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="output-card primary-card">
                            <div style="display:flex;align-items:center;gap:1rem;">
                                <div style="font-size:2.5rem;">{emoji}</div>
                                <div style="flex:1;">
                                    <div style="font-size:1.3rem;font-weight:800;">{primary['market']}</div>
                                    <div style="font-size:0.9rem;color:#94a3b8;">Confidence: {primary['confidence']}/10 | Probability: {primary['probability']:.1f}%</div>
                                    <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">{primary['reason']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SECONDARY BET
                    if analysis.get("secondary_bet"):
                        secondary = analysis["secondary_bet"]
                        st.markdown('<div class="section-label-secondary">📌 SECONDARY BET</div>', unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="output-card secondary-card">
                            <div style="display:flex;align-items:center;gap:1rem;">
                                <div style="font-size:1.8rem;">📊</div>
                                <div style="flex:1;">
                                    <div style="font-size:1.1rem;font-weight:800;">{secondary['market']}</div>
                                    <div style="font-size:0.85rem;color:#94a3b8;">Confidence: {secondary['confidence']}/10 | Probability: {secondary['probability']:.1f}%</div>
                                    <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">{secondary['reason']}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # SKIP
                    if analysis["verdict"] == "SKIP":
                        skip_text = "<br>".join(analysis.get("skip_reasons", ["No structural pattern matched"]))
                        st.markdown(f"""
                        <div class="output-card skip-card">
                            <div class="verdict-skip">
                                <div class="big-text">⚠️ SKIP — NO BET</div>
                                <p style="color:#94a3b8;margin-top:0.5rem;">{skip_text}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Badges
                    if analysis["badges"]:
                        st.markdown(" ")
                        badges_html = " ".join([f'<span class="badge-upgrade">{b}</span>' for b in analysis["badges"]])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    
                    # Warnings
                    if analysis["warnings"]:
                        st.markdown(" ")
                        for w in analysis["warnings"]:
                            st.markdown(f'<span class="badge-caution">⚠️ {w}</span>', unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 2: POST-MATCH
    # ========================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for analysis in pending:
                ht = analysis.get('home_team', 'Home')
                at = analysis.get('away_team', 'Away')
                pred = analysis.get('prediction', 'No prediction')
                
                with st.expander(f"{ht} vs {at} — Predicted: {pred}"):
                    c1, c2, c3 = st.columns(3)
                    with c1: 
                        hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2: 
                        ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total = hg + ag
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{hg} - {ag}</div>
                            <div class="score-label">Total: {total} | {'Over 2.5' if total > 2 else 'Under 2.5'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("✅ Submit Result", key=f"sub_{analysis['id']}"):
                        if submit_result(analysis['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
    # ========================================================================
    # TAB 3: RECORDS
    # ========================================================================
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
        else:
            # Overall stats
            total = len(results)
            skip_count = sum(1 for r in results if r.get('prediction') == 'SKIP')
            bet_count = total - skip_count
            
            correct = 0
            incorrect = 0
            category_stats = {}
            
            for r in results:
                pred = r.get('prediction', '')
                classification = r.get('classification', 'Unclassified')
                
                if classification not in category_stats:
                    category_stats[classification] = {"wins": 0, "losses": 0, "total": 0}
                
                if pred == 'SKIP':
                    category_stats[classification]["total"] += 1
                    continue
                
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
                
                if evaluation["is_correct"]:
                    correct += 1
                    category_stats[classification]["wins"] += 1
                else:
                    incorrect += 1
                    category_stats[classification]["losses"] += 1
                
                category_stats[classification]["total"] += 1
            
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Tracked</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{bet_count}</div><div class="stat-label">Bets Placed</div></div>', unsafe_allow_html=True)
            with col3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{skip_count}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
            
            st.write(f"**Primary Bet: Correct: {correct} | Incorrect: {incorrect}**")
            
            # Category breakdown
            st.markdown("---")
            st.markdown("### 🏷️ Performance by Classification")
            
            category_data = []
            for cat, stats in category_stats.items():
                if cat == "SKIP":
                    continue
                total_bets = stats["total"]
                wins = stats["wins"]
                losses = stats["losses"]
                win_rate_cat = round(wins / total_bets * 100) if total_bets > 0 else 0
                
                if win_rate_cat >= 70:
                    rating = "🔥 Excellent"
                    color = "#10b981"
                elif win_rate_cat >= 60:
                    rating = "✅ Good"
                    color = "#3b82f6"
                elif win_rate_cat >= 50:
                    rating = "⚠️ Marginal"
                    color = "#f59e0b"
                else:
                    rating = "❌ Poor"
                    color = "#ef4444"
                
                category_data.append({
                    "Classification": cat, "Wins": wins, "Losses": losses, 
                    "Total": total_bets, "Win Rate": f"{win_rate_cat}%", "Rating": rating, "Color": color
                })
            
            category_data.sort(key=lambda x: int(x["Win Rate"].replace("%", "")), reverse=True)
            
            # Display category table
            st.markdown("""
            <style>
            .cat-header { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1.5fr; gap: 10px; padding: 10px; background: #1e293b; border-radius: 8px; margin-bottom: 5px; font-weight: 700; }
            .cat-row { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr 1.5fr; gap: 10px; padding: 8px 10px; background: #0f172a; border-radius: 8px; margin-bottom: 3px; }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="cat-header"><div>Classification</div><div>Wins</div><div>Losses</div><div>Total</div><div>Rating</div></div>', unsafe_allow_html=True)
            for cat in category_data:
                st.markdown(f'''
                <div class="cat-row">
                    <div><strong>{cat["Classification"]}</strong></div>
                    <div style="color:#10b981;">{cat["Wins"]}</div>
                    <div style="color:#ef4444;">{cat["Losses"]}</div>
                    <div>{cat["Total"]}</div>
                    <div style="color:{cat['Color']}; font-weight:700;">{cat["Rating"]} ({cat["Win Rate"]})</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Full results table
            st.markdown("---")
            st.markdown("### 📋 Complete Match Records")
            
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                classification = r.get('classification', 'Unclassified')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                if pred == 'SKIP':
                    badge = "SKIP"
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = "WIN" if evaluation["is_correct"] else "LOSS"
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
                
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
                    "Class": classification,
                    "Primary Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
