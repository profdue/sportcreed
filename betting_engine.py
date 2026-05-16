"""
MATCH ANALYZER V3.2 — Structural Framework Engine
Score Matrix is King | Separation Power | Draw Cluster | Double Chance
FIX: Score matrix drives classification | Single primary bet | Secondary clearly labeled
FIX: Truth-based evaluation engine (no reliance on corrupted derived fields)
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
st.set_page_config(page_title="Match Analyzer V3.2", page_icon="📊", layout="wide")

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
    "Hamilton Academical": ["hamilton", "hamilton academical"],
    "Clyde": ["clyde"],
    "Leuven": ["leuven", "oh leuven"],
    "Royal Antwerp": ["antwerp", "royal antwerp"],
    "Korona Kielce": ["korona", "korona kielce"],
    "Widzew Lodz": ["widzew", "widzew lodz", "widzew łódź"],
    "Bohemians": ["bohemians", "bohemian"],
    "Drogheda United": ["drogheda", "drogheda united"],
    "Waterford United": ["waterford", "waterford united"],
    "Derry City": ["derry", "derry city"],
    "Puskas Academy": ["puskas", "puskas academy"],
    "MTK": ["mtk", "mtk budapest"],
    "St Patrick's Athletic": ["st patrick's", "st pat's", "st patricks"],
    "Shelbourne": ["shelbourne", "shels"],
    "Dundalk": ["dundalk"],
    "Shamrock Rovers": ["shamrock", "shamrock rovers"],
    "Notts County": ["notts county", "notts co"],
    "Chesterfield": ["chesterfield"],
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
# STRUCTURAL FRAMEWORK ENGINE V3.2
# Score Matrix drives classification. Single primary bet. Secondary clearly labeled.
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
    
    home_wins = sum(1 for r in home_form[:6] if r == 'W')
    home_losses = sum(1 for r in home_form[:6] if r == 'L')
    away_wins = sum(1 for r in away_form[:6] if r == 'W')
    
    h2h_total = data.get("h2h_total", 0)
    h2h_btts = data.get("h2h_btts_count", 0)
    
    score_matrix = data.get("score_matrix", [])
    
    league = data.get("league")
    is_top_league = league in TOP_LEAGUES if league else False
    is_saudi = league == "Saudi Pro League" if league else False
    
    # ========================================================================
    # SCORE MATRIX STRUCTURE ANALYSIS (KING — do this FIRST)
    # ========================================================================
    tight_cluster = False
    btts_dominant = False
    low_scoring_cluster = False
    goals_expected = False
    modal_is_draw = False
    modal_outcome = None
    home_cluster = 0
    away_cluster = 0
    draw_cluster = 0
    
    if len(score_matrix) >= 5:
        top5_spread = score_matrix[0]["probability"] - score_matrix[4]["probability"]
        tight_cluster = top5_spread < 5.0
        
        btts_count = sum(1 for s in score_matrix[:5] if s["home_goals"] > 0 and s["away_goals"] > 0)
        btts_dominant = btts_count >= 3
        
        low_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] <= 2)
        low_scoring_cluster = low_count >= 4
        
        goals_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] >= 3)
        goals_expected = goals_count >= 3
        
        modal_outcome = score_matrix[0]["score"]
        modal_is_draw = score_matrix[0]["home_goals"] == score_matrix[0]["away_goals"]
        
        for s in score_matrix[:5]:
            if s["home_goals"] > s["away_goals"]:
                home_cluster += s["probability"]
            elif s["away_goals"] > s["home_goals"]:
                away_cluster += s["probability"]
            else:
                draw_cluster += s["probability"]
    
    # Under 3.5
    under_35_pct = (100 - over_35) if over_35 else 0
    
    # ========================================================================
    # CLASSIFICATION — Score Matrix First, Percentages Confirm
    # ========================================================================
    
    # Trend reversal
    trend_reversal = False
    if home_win_trend is not None and away_win_trend is not None:
        if home_win_trend <= -1.0 and away_win_trend >= 1.0:
            trend_reversal = True
    
    # Strong favorite
    is_true_strong = (home_win >= 60 and home_o15 >= 55 and not tight_cluster)
    is_true_strong_away = (away_win >= 60 and away_o15 >= 55 and not tight_cluster)
    
    # Fragile favorite
    is_fragile = (home_win >= 60 and home_o15 < 60)
    is_fragile_away = (away_win >= 60 and away_o15 < 60)
    
    # BTTS
    btts_contradiction = (btts >= 55 and (home_o15 < 35 or away_o15 < 25))
    is_btts_play = (btts >= 55 and not btts_contradiction and 
                    (away_o05 is None or away_o05 >= 60 or home_o05 is None or home_o05 >= 60) and
                    btts_dominant)
    
    # High-scoring
    is_high_scoring = (goals_expected and over_25 >= 55 and btts >= 55)
    
    # Low-scoring / Under
    is_low_scoring = (low_scoring_cluster and under_25 >= 50 and btts < 52 and
                      not (home_losses >= 4 and sum(1 for r in away_form[:6] if r == 'L') >= 4))
    
    # Draw pressure → Double Chance
    is_draw_pressure = (modal_is_draw and draw_pct >= 20 and 
                        abs(home_win - away_win) <= 15)
    
    # Away threat
    is_away_threat = (home_o15 < 35 and away_win >= 30)
    
    # Compression
    is_compression = (tight_cluster and low_scoring_cluster and 
                      draw_pct >= 20 and under_35_pct >= 65 and
                      home_o15 < 45 and away_o15 < 45)
    
    # ========================================================================
    # BETTING DECISIONS — Single Primary Bet
    # ========================================================================
    
    # RULE 1: Trend Reversal
    if trend_reversal:
        result["classification"] = "TREND REVERSAL"
        result["badges"].append("Trend Reversal: Home {:.1f} / Away {:.1f}".format(home_win_trend, away_win_trend))
        result["primary_bet"] = {
            "market": "Away Win or Draw",
            "confidence": 8.0 if home_win_trend <= -3.0 else 6.5,
            "probability": away_win + draw_pct,
            "reason": "Market reversal — FADE home team. Home trend {:.1f}, Away trend {:.1f}".format(home_win_trend, away_win_trend)
        }
    
    # RULE 2: TRUE Strong Favorite (Home)
    elif is_true_strong:
        result["classification"] = "TRUE STRONG FAVORITE"
        if away_o05 is not None and away_o05 < 52:
            result["primary_bet"] = {
                "market": "Home Win to Nil",
                "confidence": 8.5,
                "probability": home_win,
                "reason": "Dominant favorite ({:.0f}% win, O1.5 {:.0f}%) + underdog unlikely to score ({:.0f}%)".format(home_win, home_o15, away_o05)
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win",
                "confidence": 7.5,
                "probability": home_win,
                "reason": "Dominant favorite ({:.0f}% win, O1.5 {:.0f}%) but underdog can score".format(home_win, home_o15)
            }
    
    # RULE 3: TRUE Strong Favorite (Away)
    elif is_true_strong_away:
        result["classification"] = "TRUE STRONG FAVORITE (AWAY)"
        if home_o05 is not None and home_o05 < 52:
            result["primary_bet"] = {
                "market": "Away Win to Nil",
                "confidence": 8.5,
                "probability": away_win,
                "reason": "Dominant away favorite ({:.0f}% win, O1.5 {:.0f}%) + home unlikely to score".format(away_win, away_o15)
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win",
                "confidence": 7.5,
                "probability": away_win,
                "reason": "Dominant away favorite ({:.0f}% win, O1.5 {:.0f}%) but home can score".format(away_win, away_o15)
            }
    
    # RULE 4: High-Scoring
    elif is_high_scoring:
        result["classification"] = "HIGH-SCORING"
        result["primary_bet"] = {
            "market": "Over 2.5 Goals",
            "confidence": 7.0,
            "probability": over_25,
            "reason": "Score matrix shows goals expected. Over 2.5 at {:.1f}%.".format(over_25)
        }
        if btts >= 55:
            result["secondary_bet"] = {
                "market": "BTTS",
                "confidence": 6.5,
                "probability": btts,
                "reason": "Both teams scoring in high-scoring setup"
            }
    
    # RULE 5: BTTS
    elif is_btts_play:
        result["classification"] = "BTTS"
        result["primary_bet"] = {
            "market": "BTTS",
            "confidence": 7.0 if btts_trend and btts_trend >= 0.30 else 6.5,
            "probability": btts,
            "reason": "BTTS-dominant score matrix. BTTS at {:.1f}%. Both teams expected to score.".format(btts)
        }
        # Add secondary if applicable
        if is_away_threat:
            result["secondary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 5.5,
                "probability": away_win + draw_pct,
                "reason": "Home O1.5 only {:.1f}% — away side won't lose".format(home_o15)
            }
    
    # RULE 6: Low-Scoring / Under
    elif is_low_scoring:
        result["classification"] = "LOW-SCORING"
        if under_25 >= 55:
            result["primary_bet"] = {
                "market": "Under 2.5 Goals",
                "confidence": 7.0,
                "probability": under_25,
                "reason": "Low-scoring score cluster. Under 2.5 at {:.1f}%. BTTS only {:.1f}%.".format(under_25, btts)
            }
        else:
            result["primary_bet"] = {
                "market": "Under 3.5 Goals",
                "confidence": 6.5,
                "probability": under_35_pct,
                "reason": "Low-scoring profile. Under 3.5 at {:.0f}%.".format(under_35_pct)
            }
    
    # RULE 7: Compression
    elif is_compression:
        result["classification"] = "COMPRESSION"
        result["badges"].append("Tight score cluster — low variance match")
        result["primary_bet"] = {
            "market": "Under 3.5 Goals",
            "confidence": 7.5,
            "probability": under_35_pct,
            "reason": "Compression match — tight score cluster, Under 3.5 at {:.0f}%".format(under_35_pct)
        }
        # Secondary: Double Chance based on score matrix
        if home_cluster > away_cluster:
            result["secondary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.0,
                "probability": home_win + draw_pct,
                "reason": "Score matrix favors home side ({:.1f}% vs {:.1f}%)".format(home_cluster, away_cluster)
            }
        else:
            result["secondary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.0,
                "probability": away_win + draw_pct,
                "reason": "Score matrix favors away side ({:.1f}% vs {:.1f}%)".format(away_cluster, home_cluster)
            }
    
    # RULE 8: Fragile Favorite
    elif is_fragile:
        result["classification"] = "FRAGILE FAVORITE"
        result["badges"].append("Fragile favorite — O1.5 only {:.0f}%".format(home_o15))
        if is_away_threat:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": away_win + draw_pct,
                "reason": "Fragile favorite ({:.0f}% win, O1.5 {:.0f}%) + away scoring threat. Home vulnerable.".format(home_win, home_o15)
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": home_win + draw_pct,
                "reason": "Fragile favorite ({:.0f}% win, O1.5 {:.0f}%) but underdog can't score. Draw protection.".format(home_win, home_o15)
            }
    
    # RULE 9: Fragile Favorite (Away)
    elif is_fragile_away:
        result["classification"] = "FRAGILE FAVORITE (AWAY)"
        result["badges"].append("Fragile away favorite — O1.5 only {:.0f}%".format(away_o15))
        if home_o15 >= 30:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": home_win + draw_pct,
                "reason": "Fragile away favorite ({:.0f}% win) + home scoring threat. Away vulnerable.".format(away_win)
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": away_win + draw_pct,
                "reason": "Fragile away favorite but home can't score. Away won't lose.".format(away_win)
            }
    
    # RULE 10: Draw Pressure → Double Chance
    elif is_draw_pressure:
        result["classification"] = "DRAW PRESSURE"
        if home_cluster > away_cluster:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": home_win + draw_pct,
                "reason": "Modal outcome is {}. Draw at {:.1f}%. Home side slightly favored.".format(modal_outcome, draw_pct)
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": away_win + draw_pct,
                "reason": "Modal outcome is {}. Draw at {:.1f}%. Away side slightly favored.".format(modal_outcome, draw_pct)
            }
    
    # RULE 11: Away Threat (standalone)
    elif is_away_threat:
        result["classification"] = "AWAY THREAT"
        result["primary_bet"] = {
            "market": "Away Win or Draw (Double Chance)",
            "confidence": 6.5,
            "probability": away_win + draw_pct,
            "reason": "Home O1.5 only {:.1f}% — home can't score. Away won't lose.".format(home_o15)
        }
    
    # ========================================================================
    # UNBEATEN COLLISION — only if nothing else classified
    # ========================================================================
    home_unbeaten_5 = sum(1 for r in home_form[:5] if r in ['W', 'D'])
    away_unbeaten_5 = sum(1 for r in away_form[:5] if r in ['W', 'D'])
    
    if result["primary_bet"] is None and home_unbeaten_5 >= 4 and away_unbeaten_5 >= 4 and draw_pct >= 20:
        result["classification"] = "UNBEATEN COLLISION"
        result["badges"].append("Both unbeaten: Home {}/5, Away {}/5".format(home_unbeaten_5, away_unbeaten_5))
        if home_cluster > away_cluster:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.0,
                "probability": home_win + draw_pct,
                "reason": "Both teams unbeaten. Score matrix favors home side."
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.0,
                "probability": away_win + draw_pct,
                "reason": "Both teams unbeaten. Score matrix favors away side."
            }
    
    # ========================================================================
    # SAUDI LEAGUE OVERRIDE
    # ========================================================================
    if is_saudi and home_win >= 60 and result["primary_bet"] is None:
        result["classification"] = "SAUDI DOMINANT"
        result["primary_bet"] = {
            "market": "Home Win to Nil",
            "confidence": 7.5,
            "probability": home_win,
            "reason": "Saudi league structural dominance"
        }
    
    # ========================================================================
    # FINALIZE
    # ========================================================================
    if result["primary_bet"]:
        result["verdict"] = "RECOMMENDED"
    else:
        result["verdict"] = "SKIP"
        result["skip_reasons"].append("No structural pattern matched with sufficient confidence")
    
    # Warnings
    if draw_pct >= 25:
        result["warnings"].append("High draw probability ({:.1f}%) — avoid straight match result bets".format(draw_pct))
    if not is_top_league and league:
        result["warnings"].append("'{}' is not a top league — lower reliability".format(league))
    if btts_contradiction and result["primary_bet"] and result["primary_bet"]["market"] == "BTTS":
        result["warnings"].append("BTTS contradiction — one team may not contribute")
    
    return result


# ============================================================================
# TRUTH-BASED EVALUATION ENGINE (FIXED)
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    """
    Evaluate any bet from raw scores only.
    No reliance on corrupted derived fields (actual_winner, actual_btts, actual_over25).
    
    Returns:
        {
            "is_correct": bool,
            "actual": str,
            "message": str
        }
    """
    # Normalize inputs to integers
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {
            "is_correct": False, 
            "actual": "INVALID DATA", 
            "message": f"Non-numeric score: home={home_goals}, away={away_goals}"
        }
    
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
    
    # Actual outcomes (human readable)
    actual_outcome = {
        "score": f"{home}-{away}",
        "btts": "BTTS YES" if btts else "BTTS NO",
        "over25": "Over 2.5 YES" if over25 else "Over 2.5 NO",
        "over35": "Over 3.5 YES" if over35 else "Over 3.5 NO",
        "winner": winner,
        "total": total
    }
    
    # Evaluate against prediction
    pred = primary_pred.strip()
    is_correct = False
    
    # Goal line markets
    if pred == "BTTS":
        is_correct = btts
    elif pred == "Over 2.5 Goals":
        is_correct = over25
    elif pred == "Under 2.5 Goals":
        is_correct = not over25 and total > 0
    elif pred == "Over 3.5 Goals":
        is_correct = over35
    elif pred == "Under 3.5 Goals":
        is_correct = total <= 3
    
    # Match result markets
    elif pred == "Home Win":
        is_correct = winner == "HOME"
    elif pred == "Away Win":
        is_correct = winner == "AWAY"
    elif pred == "Home Win to Nil":
        is_correct = (winner == "HOME" and away == 0)
    elif pred == "Away Win to Nil":
        is_correct = (winner == "AWAY" and home == 0)
    
    # Double chance markets
    elif "Away Win or Draw" in pred:
        is_correct = winner in ["AWAY", "DRAW"]
    elif "Home Win or Draw" in pred:
        is_correct = winner in ["HOME", "DRAW"]
    
    # Team goal markets
    elif pred == "Home Over 1.5 Goals":
        is_correct = home >= 2
    elif pred == "Away Over 1.5 Goals":
        is_correct = away >= 2
    elif pred == "Home Over 0.5 Goals":
        is_correct = home >= 1
    elif pred == "Away Over 0.5 Goals":
        is_correct = away >= 1
    
    # Fallback (unknown market)
    else:
        return {
            "is_correct": False,
            "actual": f"{home}-{away}",
            "message": f"Unknown market: {pred}"
        }
    
    return {
        "is_correct": is_correct,
        "actual": f"{home}-{away} | {actual_outcome['btts']} | {actual_outcome['over25']}",
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
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V3.2")
    st.caption("Structural Framework Engine | Score Matrix is King | Single Primary Bet | Secondary Clearly Labeled")
    
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
                    
                    # PRIMARY BET
                    if analysis.get("primary_bet"):
                        primary = analysis["primary_bet"]
                        emoji_map = {
                            "BTTS": "⚽⚽", "Over 2.5 Goals": "🔥", "Under 2.5 Goals": "🛡️", 
                            "Under 3.5 Goals": "🛡️",
                            "Home Over 1.5 Goals": "🏠⚽", "Away Over 1.5 Goals": "✈️⚽",
                            "Home Win to Nil": "🏠🧤", "Away Win to Nil": "✈️🧤",
                            "Home Win": "🏠", "Away Win": "✈️",
                            "Away Win or Draw": "✈️🤝",
                            "Away Win or Draw (Double Chance)": "✈️🤝",
                            "Home Win or Draw (Double Chance)": "🏠🤝"
                        }
                        emoji = emoji_map.get(primary["market"], "📊")
                        
                        st.markdown('<div class="section-label">🎯 PRIMARY BET</div>', unsafe_allow_html=True)
                        st.markdown("""
                        <div class="output-card primary-card">
                            <div style="display:flex;align-items:center;gap:1rem;">
                                <div style="font-size:2.5rem;">{}</div>
                                <div style="flex:1;">
                                    <div style="font-size:1.3rem;font-weight:800;">{}</div>
                                    <div style="font-size:0.9rem;color:#94a3b8;">Confidence: {}/10 | Probability: {:.1f}%</div>
                                    <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">{}</div>
                                </div>
                            </div>
                        </div>
                        """.format(emoji, primary['market'], primary['confidence'], primary['probability'], primary['reason']),
                        unsafe_allow_html=True)
                    
                    # SECONDARY BET
                    if analysis.get("secondary_bet"):
                        secondary = analysis["secondary_bet"]
                        emoji2 = emoji_map.get(secondary["market"], "📊")
                        
                        st.markdown('<div class="section-label-secondary">📌 SECONDARY BET</div>', unsafe_allow_html=True)
                        st.markdown("""
                        <div class="output-card secondary-card">
                            <div style="display:flex;align-items:center;gap:1rem;">
                                <div style="font-size:1.8rem;">{}</div>
                                <div style="flex:1;">
                                    <div style="font-size:1.1rem;font-weight:800;">{}</div>
                                    <div style="font-size:0.85rem;color:#94a3b8;">Confidence: {}/10 | Probability: {:.1f}%</div>
                                    <div style="font-size:0.8rem;color:#64748b;margin-top:0.3rem;">{}</div>
                                </div>
                            </div>
                        </div>
                        """.format(emoji2, secondary['market'], secondary['confidence'], secondary['probability'], secondary['reason']),
                        unsafe_allow_html=True)
                    
                    # SKIP
                    if analysis["verdict"] == "SKIP":
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
            # ========================================================================
            # TRUTH-BASED EVALUATION (FIXED)
            # ========================================================================
            total = len(results)
            skip_count = sum(1 for r in results if r.get('prediction') == 'SKIP')
            bet_count = total - skip_count
            
            correct = 0
            incorrect = 0
            
            for r in results:
                pred = r.get('prediction', '')
                if pred == 'SKIP':
                    continue
                
                # Primary bet is the first one before " | "
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                # Use the truth-based evaluator
                evaluation = evaluate_bet(
                    primary_pred, 
                    r.get('actual_home_goals'), 
                    r.get('actual_away_goals')
                )
                
                if evaluation["is_correct"]:
                    correct += 1
                else:
                    incorrect += 1
            
            # Display stats
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Total Tracked</div></div>'.format(total), unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Bets Placed</div></div>'.format(bet_count), unsafe_allow_html=True)
            with c3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown('<div class="stat-box"><div class="stat-number">{}%</div><div class="stat-label">Win Rate (Primary)</div></div>'.format(win_rate), unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="stat-box"><div class="stat-number">{}</div><div class="stat-label">Skipped</div></div>'.format(skip_count), unsafe_allow_html=True)
            
            st.write("**Primary Bet: Correct: {} | Incorrect: {}**".format(correct, incorrect))
            
            # Build results table
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                if pred == 'SKIP':
                    badge = '<span class="skip-badge">SKIP</span>'
                    score_display = "—"
                else:
                    # Use evaluator to determine result
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    if evaluation["is_correct"]:
                        badge = '<span class="correct-badge">WIN</span>'
                    else:
                        badge = '<span class="incorrect-badge">LOSS</span>'
                    
                    # Handle None values gracefully
                    if actual_home is not None and actual_away is not None:
                        score_display = f"{actual_home}-{actual_away}"
                    else:
                        score_display = "—"
                
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": "{} vs {}".format(r.get('home_team', ''), r.get('away_team', '')),
                    "Class": r.get("classification", ""),
                    "Primary Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
