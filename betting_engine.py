"""
MATCH ANALYZER V3.1 — Structural Framework Engine
Score Matrix is King | Separation Power | Draw Cluster | Double Chance
FIX: Draw replaced with Double Chance | Under 3.5 remains primary for compression
Supabase table: match_analyses
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
st.set_page_config(page_title="Match Analyzer V3.1", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .tier1-card { border: 2px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .tier2-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
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
# TEAM ABBREVIATIONS
# ============================================================================
TEAM_ABBREVIATIONS = {
    "Nottingham Forest": ["nott'm forest", "nottm forest", "notts forest"],
    "Manchester United": ["man utd", "manchester utd", "man united"],
    "Manchester City": ["man city", "manchester city"],
    "Wolverhampton Wanderers": ["wolves", "wolverhampton"],
    "Newcastle United": ["newcastle", "newcastle utd"],
    "Tottenham Hotspur": ["spurs", "tottenham"],
    "West Ham United": ["west ham", "west ham utd"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Leeds United": ["leeds", "leeds utd"],
    "Brighton & Hove Albion": ["brighton"],
    "Brentford": ["brentford"],
    "Everton": ["everton"],
    "Chelsea": ["chelsea"],
    "Arsenal": ["arsenal"],
    "Liverpool": ["liverpool"],
    "Aston Villa": ["aston villa", "villa"],
    "Fulham": ["fulham"],
    "Bournemouth": ["bournemouth"],
    "Southampton": ["southampton"],
    "Burnley": ["burnley"],
    "Sunderland": ["sunderland"],
    "Bayern Munich": ["bayern", "fc bayern", "bayern munich"],
    "FC Koln": ["koln", "fc koln", "köln", "cologne"],
    "Porto": ["porto", "fc porto"],
    "Santa Clara": ["santa clara"],
    "Famalicao": ["famalicao", "famalicão"],
    "Alverca": ["alverca"],
    "Cordoba": ["cordoba", "córdoba"],
    "Albacete": ["albacete"],
    "Castellon": ["castellon", "castellón"],
    "Cadiz": ["cadiz", "cádiz"],
    "Vukovar": ["vukovar"],
    "NK Varazdin": ["varazdin", "varaždin"],
    "Neman": ["neman", "fc neman"],
    "Isloch": ["isloch", "fc isloch"],
    "FC Minsk": ["fc minsk", "minsk"],
    "Belshina": ["belshina"],
    "Caykur Rizespor": ["rizespor", "çaykur rizespor"],
    "Besiktas": ["besiktas", "beşiktaş"],
    "Zaglebie Lubin": ["zaglebie", "zagłębie"],
    "Pogon Szczecin": ["pogon", "szczecin", "pogoń"],
    "Nyiregyhaza Spartacus": ["nyiregyhaza", "nyíregyháza"],
    "Kazincbarcika": ["kazincbarcika"],
    "Arges": ["arges", "argeș"],
    "Rapid Bucuresti": ["rapid", "rapid bucuresti"],
    "Hapoel Tel Aviv": ["hapoel tel aviv", "h. tel aviv"],
    "Hapoel Be'er Sheva": ["hapoel be'er sheva", "h. be'er sheva"],
    "Damac": ["damac"],
    "Al Fayha": ["al fayha", "fayha"],
    "Adelaide United": ["adelaide", "adelaide united"],
    "Auckland FC": ["auckland", "auckland fc"],
    "Shanghai Port": ["shanghai port", "shanghai sipg"],
    "Zhejiang Professional": ["zhejiang"],
    "Henan": ["henan"],
    "Sichuan Jiuniu": ["sichuan", "sichuan jiuniu"],
    "Tianjin Jinmen Tiger": ["tianjin", "jinmen tiger"],
    "Chengdu Rongcheng": ["chengdu", "chengdu rongcheng"],
    "Beijing Guoan": ["beijing", "beijing guoan"],
    "Qingdao Hainiu": ["qingdao hainiu", "hainiu"],
    "Partick Thistle": ["partick", "partick thistle"],
    "Dunfermline Athletic": ["dunfermline"],
    "Salford City": ["salford", "salford city"],
    "Grimsby Town": ["grimsby", "grimsby town"],
    "St Louis City": ["st louis", "st. louis"],
    "Los Angeles FC": ["los angeles", "lafc"],
    "Los Angeles Galaxy": ["la galaxy"],
    "Sporting Kansas City": ["sporting kc", "kansas city"],
    "San Jose Earthquakes": ["san jose", "earthquakes"],
    "Seattle Sounders": ["seattle", "sounders"],
    "Austin FC": ["austin"],
    "San Diego": ["san diego"],
    "Real Salt Lake": ["real salt lake", "salt lake", "rsl"],
    "Houston Dynamo": ["houston", "dynamo"],
    "Minnesota United": ["minnesota", "minnesota utd"],
    "Colorado Rapids": ["colorado", "rapids"],
    "Dallas": ["dallas", "fc dallas"],
    "Vancouver Whitecaps": ["vancouver", "whitecaps"],
    "CD Guadalajara": ["guadalajara", "chivas"],
    "Cruz Azul": ["cruz azul"],
    "River Plate": ["river plate", "river"],
    "Gimnasia": ["gimnasia"],
    "Thun": ["thun", "fc thun"],
    "Young Boys": ["young boys", "yb", "bern"],
    "Basel": ["basel", "fc basel"],
    "St Gallen": ["st gallen", "st. gallen"],
    "Sion": ["sion", "fc sion"],
    "Lugano": ["lugano", "fc lugano"],
    "Al Fateh": ["al fateh", "fateh"],
    "Al Najma": ["al najma", "najma"],
    "Al Ettifaq": ["al ettifaq", "ettifaq"],
    "Al Ittihad": ["al ittihad", "ittihad"],
    "Al Quadisiya": ["al quadisiya", "quadisiya"],
    "Al Hazem": ["al hazem", "hazem"],
    "Gomel": ["gomel"],
    "Baranovichi": ["baranovichi"],
    "Kudrivka": ["kudrivka"],
    "Rukh Lviv": ["rukh lviv", "rukh"],
    "Bradford City": ["bradford"],
    "Bolton Wanderers": ["bolton"],
    "Girona": ["girona"],
    "Real Sociedad": ["real sociedad", "sociedad"],
    "Valencia": ["valencia"],
    "Rayo Vallecano": ["rayo", "rayo vallecano"],
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
    
    # League
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
                      r'Croatian 1\. HNL|HNL|Prva HNL|Scottish Premiership|Scottish Premiership Playoffs|Eredivisie|A-League|'
                      r'Ekstraklasa|Polish Ekstraklasa|'
                      r'Turkish Super Lig|Süper Lig|'
                      r'Israeli Premier League|Ligat HaAl|'
                      r'Hungarian NB I|OTP Bank Liga|'
                      r'Romanian SuperLiga|SuperLiga|'
                      r'Chinese Super League|CSL|'
                      r'Australian A-League|'
                      r'Scottish Championship|League Two)', 
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
        'Scottish Championship'
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
# STRUCTURAL FRAMEWORK ENGINE V3.1 — Draw replaced with Double Chance
# ============================================================================
def analyze_match(data: dict) -> dict:
    result = {
        "bets": [],
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
    
    home_wins = sum(1 for r in home_form[:6] if r == 'W')
    home_draws = sum(1 for r in home_form[:6] if r == 'D')
    away_wins = sum(1 for r in away_form[:6] if r == 'W')
    away_draws = sum(1 for r in away_form[:6] if r == 'D')
    
    h2h_total = data.get("h2h_total", 0)
    h2h_btts = data.get("h2h_btts_count", 0)
    
    score_matrix = data.get("score_matrix", [])
    
    league = data.get("league")
    is_top_league = league in TOP_LEAGUES if league else False
    is_saudi = league == "Saudi Pro League" if league else False
    
    # Score matrix analysis
    draw_cluster = 0
    modal_outcome = None
    modal_is_draw = False
    
    if score_matrix:
        modal_outcome = score_matrix[0]["score"]
        modal_is_draw = score_matrix[0]["home_goals"] == score_matrix[0]["away_goals"]
        
        for s in score_matrix:
            if s["home_goals"] == s["away_goals"]:
                draw_cluster += s["probability"]
    
    # Under 3.5 calculation
    under_35_pct = (100 - over_35) if over_35 else 0
    
    # Classifications
    is_true_strong_favorite = home_win >= 60 and home_o15 >= 60
    is_fragile_favorite = home_win >= 60 and home_o15 < 60
    is_moderate_favorite = 45 <= home_win < 60
    is_balanced = abs(home_win - away_win) < 12
    
    underdog_scoring_threat = False
    if away_o05 is not None and away_o05 >= 60:
        underdog_scoring_threat = True
    if away_o15 is not None and away_o15 >= 30:
        underdog_scoring_threat = True
    
    home_scoring_threat = False
    if home_o05 is not None and home_o05 >= 60:
        home_scoring_threat = True
    if home_o15 is not None and home_o15 >= 30:
        home_scoring_threat = True
    
    trend_reversal = False
    if home_win_trend is not None and away_win_trend is not None:
        if home_win_trend <= -1.0 and away_win_trend >= 1.0:
            trend_reversal = True
    
    btts_contradiction = False
    if btts >= 55 and home_o15 < 35:
        btts_contradiction = True
        result["warnings"].append("BTTS {:.1f}% but Home O1.5 only {:.1f}% — home team unlikely to contribute".format(btts, home_o15))
    if btts >= 55 and away_o15 < 30:
        btts_contradiction = True
        result["warnings"].append("BTTS {:.1f}% but Away O1.5 only {:.1f}% — away team unlikely to contribute".format(btts, away_o15))
    
    is_compression = False
    if draw_pct >= 24 and under_35_pct >= 65 and home_o15 < 45 and away_o15 < 45:
        is_compression = True
    
    # ========================================================================
    # BETTING DECISIONS
    # ========================================================================
    
    # RULE 1: Trend Reversal
    if trend_reversal:
        result["classification"] = "TREND REVERSAL"
        result["badges"].append("Trend Reversal: Home {:.1f} / Away {:.1f}".format(home_win_trend, away_win_trend))
        result["bets"].append({
            "market": "Away Win or Draw",
            "tier": "TIER 1",
            "confidence": 8.0 if home_win_trend <= -3.0 else 6.5,
            "probability": away_win + draw_pct,
            "reason": "Market reversal — FADE home team"
        })
    
    # RULE 2: TRUE Strong Favorite
    elif is_true_strong_favorite and not trend_reversal:
        result["classification"] = "TRUE STRONG FAVORITE"
        if away_o05 is not None and away_o05 < 50:
            result["bets"].append({
                "market": "Home Win to Nil",
                "tier": "TIER 1",
                "confidence": 8.5,
                "probability": home_win,
                "reason": "True dominant favorite ({:.0f}% win, O1.5 {:.0f}%) + underdog unlikely to score ({:.0f}%)".format(home_win, home_o15, away_o05)
            })
        else:
            result["bets"].append({
                "market": "Home Win",
                "tier": "TIER 1",
                "confidence": 7.5,
                "probability": home_win,
                "reason": "True dominant favorite but underdog can score"
            })
            result["bets"].append({
                "market": "Home Over 1.5 Goals",
                "tier": "TIER 2",
                "confidence": 7.0,
                "probability": home_o15,
                "reason": "Home scores 2+ regularly ({:.0f}%)".format(home_o15)
            })
    
    # RULE 3: FRAGILE Favorite
    elif is_fragile_favorite and not trend_reversal:
        result["classification"] = "FRAGILE FAVORITE"
        result["badges"].append("Fragile favorite — O1.5 only {:.0f}%".format(home_o15))
        
        if underdog_scoring_threat:
            # Away can score → Double Chance: Away or Draw
            result["bets"].append({
                "market": "Away Win or Draw (Double Chance)",
                "tier": "TIER 1",
                "confidence": 7.0,
                "probability": away_win + draw_pct,
                "reason": "Fragile favorite + away scoring threat (O0.5 {:.0f}%). Home vulnerable.".format(away_o05 or 0)
            })
        else:
            # Away can't score → Home still vulnerable but away unlikely to win
            result["bets"].append({
                "market": "Home Win or Draw (Double Chance)",
                "tier": "TIER 1",
                "confidence": 6.5,
                "probability": home_win + draw_pct,
                "reason": "Fragile favorite but underdog can't score. Draw protection needed."
            })
    
    # RULE 4: COMPRESSION
    elif is_compression:
        result["classification"] = "COMPRESSION"
        result["badges"].append("Compression Match — tight score cluster, low margins")
        
        result["bets"].append({
            "market": "Under 3.5 Goals",
            "tier": "TIER 1",
            "confidence": 7.5,
            "probability": under_35_pct,
            "reason": "Compression match — Under 3.5 at {:.0f}%".format(under_35_pct)
        })
        
        # Determine which side the score matrix favors for Double Chance
        home_cluster = sum(s["probability"] for s in score_matrix if s["home_goals"] > s["away_goals"])
        away_cluster = sum(s["probability"] for s in score_matrix if s["away_goals"] > s["home_goals"])
        
        if home_cluster > away_cluster:
            result["bets"].append({
                "market": "Home Win or Draw (Double Chance)",
                "tier": "TIER 2",
                "confidence": 6.0,
                "probability": home_win + draw_pct,
                "reason": "Compression — home side slightly favored in score matrix ({:.1f}% vs {:.1f}%)".format(home_cluster, away_cluster)
            })
        else:
            result["bets"].append({
                "market": "Away Win or Draw (Double Chance)",
                "tier": "TIER 2",
                "confidence": 6.0,
                "probability": away_win + draw_pct,
                "reason": "Compression — away side slightly favored in score matrix ({:.1f}% vs {:.1f}%)".format(away_cluster, home_cluster)
            })
    
    # RULE 5: BTTS Contradiction
    elif btts_contradiction:
        result["classification"] = "BTTS CONTRADICTION"
        if home_o15 < 35 and away_win >= 40:
            result["bets"].append({
                "market": "Away Win",
                "tier": "TIER 1",
                "confidence": 6.5,
                "probability": away_win,
                "reason": "BTTS contradiction — home can't score, away can win"
            })
        elif away_o15 < 30 and home_win >= 40:
            result["bets"].append({
                "market": "Home Win",
                "tier": "TIER 1",
                "confidence": 6.5,
                "probability": home_win,
                "reason": "BTTS contradiction — away can't score, home can win"
            })
    
    # RULE 6: BALANCED
    elif is_balanced:
        result["classification"] = "BALANCED"
        if btts >= 55 and not btts_contradiction:
            result["bets"].append({
                "market": "BTTS",
                "tier": "TIER 1",
                "confidence": 6.5,
                "probability": btts,
                "reason": "Balanced match — both teams can score"
            })
        # If modal is draw, add Double Chance
        if modal_is_draw and draw_pct >= 20:
            if home_win > away_win:
                result["bets"].append({
                    "market": "Home Win or Draw (Double Chance)",
                    "tier": "TIER 2",
                    "confidence": 5.5,
                    "probability": home_win + draw_pct,
                    "reason": "Modal outcome is {} — draw protection value".format(modal_outcome)
                })
            else:
                result["bets"].append({
                    "market": "Away Win or Draw (Double Chance)",
                    "tier": "TIER 2",
                    "confidence": 5.5,
                    "probability": away_win + draw_pct,
                    "reason": "Modal outcome is {} — draw protection value".format(modal_outcome)
                })
    
    # RULE 7: MODERATE Favorite
    elif is_moderate_favorite:
        result["classification"] = "MODERATE FAVORITE"
        if home_wins >= 5:
            result["bets"].append({
                "market": "Home Win",
                "tier": "TIER 2",
                "confidence": 6.5,
                "probability": home_win,
                "reason": "Home on {}-match win streak".format(home_wins)
            })
            result["badges"].append("Win Streak: Home {}W".format(home_wins))
        if btts >= 55 and not btts_contradiction:
            result["bets"].append({
                "market": "BTTS",
                "tier": "TIER 1",
                "confidence": 6.0,
                "probability": btts,
                "reason": "Both teams likely to score"
            })
    
    # RULE 8: Unbeaten Collision → Double Chance (not Draw)
    home_unbeaten_5 = sum(1 for r in home_form[:5] if r in ['W', 'D'])
    away_unbeaten_5 = sum(1 for r in away_form[:5] if r in ['W', 'D'])
    if home_unbeaten_5 >= 4 and away_unbeaten_5 >= 4 and draw_pct >= 20:
        result["badges"].append("Unbeaten Collision: Home {}/5, Away {}/5".format(home_unbeaten_5, away_unbeaten_5))
        
        if not any("Double Chance" in b["market"] for b in result["bets"]):
            # Pick the side the score matrix favors
            if score_matrix:
                home_cluster = sum(s["probability"] for s in score_matrix if s["home_goals"] > s["away_goals"])
                away_cluster = sum(s["probability"] for s in score_matrix if s["away_goals"] > s["home_goals"])
                
                if home_cluster > away_cluster:
                    result["bets"].append({
                        "market": "Home Win or Draw (Double Chance)",
                        "tier": "TIER 2",
                        "confidence": 6.0,
                        "probability": home_win + draw_pct,
                        "reason": "Unbeaten collision — home side favored in score matrix"
                    })
                else:
                    result["bets"].append({
                        "market": "Away Win or Draw (Double Chance)",
                        "tier": "TIER 2",
                        "confidence": 6.0,
                        "probability": away_win + draw_pct,
                        "reason": "Unbeaten collision — away side favored in score matrix"
                    })
            else:
                # Fallback: pick the side with higher win probability
                if home_win >= away_win:
                    result["bets"].append({
                        "market": "Home Win or Draw (Double Chance)",
                        "tier": "TIER 2",
                        "confidence": 6.0,
                        "probability": home_win + draw_pct,
                        "reason": "Unbeaten collision — home side has edge"
                    })
                else:
                    result["bets"].append({
                        "market": "Away Win or Draw (Double Chance)",
                        "tier": "TIER 2",
                        "confidence": 6.0,
                        "probability": away_win + draw_pct,
                        "reason": "Unbeaten collision — away side has edge"
                    })
    
    # ========================================================================
    # SAUDI LEAGUE OVERRIDE
    # ========================================================================
    if is_saudi and home_win >= 60:
        if not any(b["market"] == "Home Win to Nil" for b in result["bets"]):
            result["bets"].append({
                "market": "Home Win to Nil",
                "tier": "TIER 1",
                "confidence": 7.5,
                "probability": home_win,
                "reason": "Saudi league structural dominance"
            })
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    tier_order = {"TIER 1": 0, "TIER 2": 1}
    result["bets"].sort(key=lambda b: (tier_order.get(b["tier"], 3), -b["confidence"]))
    
    seen_markets = set()
    unique_bets = []
    for bet in result["bets"]:
        if bet["market"] not in seen_markets:
            seen_markets.add(bet["market"])
            unique_bets.append(bet)
    result["bets"] = unique_bets
    
    if result["bets"]:
        result["verdict"] = "RECOMMENDED"
    else:
        result["verdict"] = "SKIP"
        result["skip_reasons"].append("No structural pattern matched with sufficient confidence")
    
    # Warnings
    if draw_pct >= 25:
        result["warnings"].append("High draw probability ({:.1f}%) — avoid straight match result bets".format(draw_pct))
    if not is_top_league and league:
        result["warnings"].append("'{}' is not a top league — lower reliability".format(league))
    
    return result


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    try:
        bets_str = " | ".join([b["market"] for b in analysis["bets"]]) if analysis["bets"] else "SKIP"
        top = analysis["bets"][0] if analysis["bets"] else None
        
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
            "btts": "BTTS YES" if any("BTTS" in b["market"] for b in analysis["bets"]) else "",
            "btts_confidence": round(max([b["confidence"]/10 for b in analysis["bets"] if "BTTS" in b["market"]]), 2) if any("BTTS" in b["market"] for b in analysis["bets"]) else 0,
            "pattern": " | ".join([b["tier"] for b in analysis["bets"]]) if analysis["bets"] else "SKIP",
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
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V3.1")
    st.caption("Structural Framework Engine | Double Chance replaces Draw | Under 3.5 for Compression")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
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
                    st.error("Could not detect team names.")
                else:
                    analysis = analyze_match(data)
                    save_to_db(data, analysis)
                    
                    league_display = data.get('league') or 'Club Match'
                    
                    if analysis["verdict"] == "SKIP":
                        st.warning("{} vs {} — {}".format(data['home_team'], data['away_team'], league_display))
                    else:
                        st.success("{} vs {} — {}".format(data['home_team'], data['away_team'], league_display))
                    
                    if analysis.get("classification"):
                        st.markdown("**Classification: {}**".format(analysis["classification"]))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        home_o05_str = "{:.1f}%".format(data['home_over_05_goals']) if data.get('home_over_05_goals') is not None else "N/A"
                        home_o15_str = "{:.1f}%".format(data['home_over_15_goals']) if data.get('home_over_15_goals') is not None else "N/A"
                        away_o05_str = "{:.1f}%".format(data['away_over_05_goals']) if data.get('away_over_05_goals') is not None else "N/A"
                        away_o15_str = "{:.1f}%".format(data['away_over_15_goals']) if data.get('away_over_15_goals') is not None else "N/A"
                        
                        st.markdown("""
                        <div class="edge-box">
                            <strong>📊 Probabilities</strong><br>
                            Home: {:.1f}% | Draw: {:.1f}% | Away: {:.1f}%<br>
                            BTTS: {:.1f}% | O2.5: {:.1f}% | U2.5: {:.1f}%<br>
                            Home O0.5: {} | Home O1.5: {}<br>
                            Away O0.5: {} | Away O1.5: {}
                        </div>
                        """.format(
                            data.get('home_win', 0), data.get('draw', 0), data.get('away_win', 0),
                            data.get('btts', 0), data.get('over_25', 0), data.get('under_25', 0),
                            home_o05_str, home_o15_str, away_o05_str, away_o15_str
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        home_form_str = '-'.join(data.get('home_form_all', [])[:6]) if data.get('home_form_all') else 'N/A'
                        away_form_str = '-'.join(data.get('away_form_all', [])[:6]) if data.get('away_form_all') else 'N/A'
                        
                        st.markdown("""
                        <div class="edge-box">
                            <strong>📈 Form & H2H</strong><br>
                            Home: {}<br>Away: {}<br>H2H BTTS: {}/{}
                        </div>
                        """.format(home_form_str, away_form_str, data.get('h2h_btts_count', 0), data.get('h2h_total', 0)),
                        unsafe_allow_html=True)
                    
                    # Score Matrix
                    if data.get("score_matrix"):
                        st.markdown("### 🎯 Score Matrix (Top 6)")
                        score_cols = st.columns(6)
                        for idx, s in enumerate(data["score_matrix"][:6]):
                            with score_cols[idx]:
                                is_draw = s["home_goals"] == s["away_goals"]
                                bg = "#1e293b" if not is_draw else "#2a1a00"
                                st.markdown("""
                                <div style="background:{}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;">
                                    <div style="font-size:1.2rem; font-weight:800;">{}</div>
                                    <div style="font-size:0.7rem; color:#94a3b8;">{:.1f}%{}</div>
                                </div>
                                """.format(bg, s["score"], s["probability"], " 👑" if idx == 0 else ""), unsafe_allow_html=True)
                    
                    st.markdown("### 🎯 Recommendations")
                    
                    if analysis["bets"]:
                        emoji_map = {
                            "BTTS": "⚽⚽", "Over 2.5": "🔥", "Under 2.5": "🛡️", 
                            "Under 3.5 Goals": "🛡️",
                            "Home Over 1.5 Goals": "🏠⚽", "Away Over 1.5 Goals": "✈️⚽",
                            "Home Win to Nil": "🏠🧤", "Away Win to Nil": "✈️🧤",
                            "Home Win": "🏠", "Away Win": "✈️",
                            "Away Win or Draw": "✈️🤝",
                            "Away Win or Draw (Double Chance)": "✈️🤝",
                            "Home Win or Draw (Double Chance)": "🏠🤝"
                        }
                        
                        for bet in analysis["bets"]:
                            tier_class = "tier1-card" if bet["tier"] == "TIER 1" else "tier2-card"
                            emoji = emoji_map.get(bet["market"], "📊")
                            
                            st.markdown("""
                            <div class="output-card {}">
                                <div style="display:flex;align-items:center;gap:1rem;">
                                    <div style="font-size:2rem;">{}</div>
                                    <div style="flex:1;">
                                        <div style="font-size:1.2rem;font-weight:800;">{} — {}</div>
                                        <div style="font-size:0.85rem;color:#94a3b8;">Confidence: {}/10 | Probability: {:.1f}%</div>
                                        <div style="font-size:0.8rem;color:#64748b;">{}</div>
                                    </div>
                                </div>
                            </div>
                            """.format(tier_class, emoji, bet['market'], bet['tier'],
                                       bet['confidence'], bet['probability'], bet['reason']),
                            unsafe_allow_html=True)
                    else:
                        skip_text = "<br>".join(analysis.get("skip_reasons", ["No structural pattern matched"]))
                        st.markdown("""
                        <div class="output-card skip-card">
                            <div class="verdict-skip">
                                <div class="big-text">⚠️ SKIP — NO BET</div>
                                <p style="color:#94a3b8;margin-top:0.5rem;">{}</p>
                            </div>
                        </div>
                        """.format(skip_text), unsafe_allow_html=True)
                    
                    if analysis["badges"]:
                        st.markdown(" ")
                        badges_html = " ".join(['<span class="badge-upgrade">{}</span>'.format(b) for b in analysis["badges"]])
                        st.markdown(badges_html, unsafe_allow_html=True)
                    
                    if analysis["warnings"]:
                        st.markdown(" ")
                        for w in analysis["warnings"]:
                            st.markdown('<span class="badge-caution">⚠️ {}</span>'.format(w), unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write("**{} pending result(s)**".format(len(pending)))
            for analysis in pending:
                ht = analysis.get('home_team', 'Home')
                at = analysis.get('away_team', 'Away')
                pred = analysis.get('prediction', 'No prediction')
                
                with st.expander("{} vs {} — Predicted: {}".format(ht, at, pred)):
                    c1, c2, c3 = st.columns(3)
                    with c1: 
                        hg = st.number_input("{} Goals".format(ht), 0, 15, 0, key="hg_{}".format(analysis['id']))
                    with c2: 
                        ag = st.number_input("{} Goals".format(at), 0, 15, 0, key="ag_{}".format(analysis['id']))
                    with c3:
                        total = hg + ag
                        st.markdown("""
                        <div class="score-box">
                            <div class="score-number">{} - {}</div>
                            <div class="score-label">Total: {} | {}</div>
                        </div>
                        """.format(hg, ag, total, 'Over 2.5' if total > 2 else 'Under 2.5'), unsafe_allow_html=True)
                    
                    if st.button("✅ Submit Result", key="sub_{}".format(analysis['id'])):
                        if submit_result(analysis['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
        else:
            total = len(results)
            skip_count = sum(1 for r in results if r.get('prediction') == 'SKIP')
            bet_count = total - skip_count
            
            correct = 0
            incorrect = 0
            
            for r in results:
                pred = r.get('prediction', '')
                if pred == 'SKIP':
                    continue
                
                actual_btts = r.get('actual_btts')
                actual_over25 = r.get('actual_over25')
                actual_winner = r.get('actual_winner')
                actual_home = r.get('actual_home_goals', 0) or 0
                actual_away = r.get('actual_away_goals', 0) or 0
                actual_total = actual_home + actual_away
                
                is_correct = False
                markets = pred.split(' | ')
                for market in markets:
                    market = market.strip()
                    if market == 'BTTS' and actual_btts:
                        is_correct = True; break
                    if market == 'Over 2.5' and actual_over25:
                        is_correct = True; break
                    if market == 'Under 2.5' and not actual_over25 and actual_total > 0:
                        is_correct = True; break
                    if market == 'Under 3.5 Goals' and actual_total <= 3:
                        is_correct = True; break
                    if market == 'Home Win' and actual_winner == 'HOME':
                        is_correct = True; break
                    if market == 'Away Win' and actual_winner == 'AWAY':
                        is_correct = True; break
                    if market == 'Home Win to Nil' and actual_winner == 'HOME' and actual_away == 0:
                        is_correct = True; break
                    if market == 'Away Win to Nil' and actual_winner == 'AWAY' and actual_home == 0:
                        is_correct = True; break
                    if market == 'Home Over 1.5 Goals' and actual_home >= 2:
                        is_correct = True; break
                    if market == 'Away Over 1.5 Goals' and actual_away >= 2:
                        is_correct = True; break
                    if 'Away Win or Draw' in market and actual_winner in ['AWAY', 'DRAW']:
                        is_correct = True; break
                    if 'Home Win or Draw' in market and actual_winner in ['HOME', 'DRAW']:
                        is_correct = True; break
                    if market == 'Draw' and actual_winner == 'DRAW':
                        is_correct = True; break
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Total</div></div>'.format(total), unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Bets</div></div>'.format(bet_count), unsafe_allow_html=True)
            with c3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown('<div class="stat-box"><div class="stat-number">{}%</div><div class="stat-label">Win Rate</div></div>'.format(win_rate), unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Skipped</div></div>'.format(skip_count), unsafe_allow_html=True)
            
            st.write("**Correct: {} | Incorrect: {}**".format(correct, incorrect))
            
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                actual_total = (actual_home or 0) + (actual_away or 0)
                
                row_correct = None
                if pred == 'SKIP':
                    row_correct = 'skip'
                else:
                    markets = pred.split(' | ')
                    for market in markets:
                        market = market.strip()
                        if market == 'BTTS' and r.get('actual_btts'):
                            row_correct = 'correct'; break
                        if market == 'Over 2.5' and r.get('actual_over25'):
                            row_correct = 'correct'; break
                        if market == 'Under 2.5' and not r.get('actual_over25') and actual_total > 0:
                            row_correct = 'correct'; break
                        if market == 'Under 3.5 Goals' and actual_total <= 3:
                            row_correct = 'correct'; break
                        if market == 'Home Win' and r.get('actual_winner') == 'HOME':
                            row_correct = 'correct'; break
                        if market == 'Away Win' and r.get('actual_winner') == 'AWAY':
                            row_correct = 'correct'; break
                        if market == 'Home Win to Nil' and r.get('actual_winner') == 'HOME' and actual_away == 0:
                            row_correct = 'correct'; break
                        if 'Away Win or Draw' in market and r.get('actual_winner') in ['AWAY', 'DRAW']:
                            row_correct = 'correct'; break
                        if 'Home Win or Draw' in market and r.get('actual_winner') in ['HOME', 'DRAW']:
                            row_correct = 'correct'; break
                        if market == 'Draw' and r.get('actual_winner') == 'DRAW':
                            row_correct = 'correct'; break
                        if market == 'Home Over 1.5 Goals' and (actual_home or 0) >= 2:
                            row_correct = 'correct'; break
                        if market == 'Away Over 1.5 Goals' and (actual_away or 0) >= 2:
                            row_correct = 'correct'; break
                    if row_correct is None:
                        row_correct = 'incorrect'
                
                if row_correct == 'correct':
                    badge = '<span class="correct-badge">WIN</span>'
                elif row_correct == 'incorrect':
                    badge = '<span class="incorrect-badge">LOSS</span>'
                else:
                    badge = '<span class="skip-badge">SKIP</span>'
                
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": "{} vs {}".format(r.get('home_team', ''), r.get('away_team', '')),
                    "Classification": r.get("classification", ""),
                    "Prediction": pred,
                    "Score": "{}-{}".format(actual_home if actual_home is not None else '-', actual_away if actual_away is not None else '-'),
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
