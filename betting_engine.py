"""
MATCH ANALYZER V2.0.1 — Evidence-Based Decision Engine
Built from 37-match backtest analysis
No more max() — every recommendation requires supporting evidence
Fixes: Syntax error + Historical Records performance tracking
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
st.set_page_config(page_title="Match Analyzer V2.0.1", page_icon="📊", layout="wide")

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
    .badge-warning { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #f97316; color: #000; margin: 0.1rem; }
    .verdict-skip { text-align: center; padding: 1.5rem; }
    .verdict-skip .big-text { font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .correct-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .incorrect-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
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
    "Nottingham Forest": ["nott'm forest", "nottingham for", "nottm forest", "notts forest"],
    "Manchester United": ["man utd", "manchester utd", "man united"],
    "Manchester City": ["man city", "manchester city"],
    "Wolverhampton Wanderers": ["wolves", "wolverhampton"],
    "Newcastle United": ["newcastle", "newcastle utd"],
    "Tottenham Hotspur": ["spurs", "tottenham"],
    "West Ham United": ["west ham", "west ham utd"],
    "Crystal Palace": ["crystal palace", "palace"],
    "Leeds United": ["leeds", "leeds utd"],
    "Brighton & Hove Albion": ["brighton", "brighton & hove"],
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
    "St Louis City": ["st louis", "st. louis"],
    "Los Angeles FC": ["los angeles", "lafc"],
    "Los Angeles Galaxy": ["la galaxy", "los angeles galaxy"],
    "Sporting Kansas City": ["sporting kc", "sporting kansas", "kansas city"],
    "San Jose Earthquakes": ["san jose", "earthquakes"],
    "Seattle Sounders": ["seattle", "sounders"],
    "Austin FC": ["austin"],
    "San Diego": ["san diego"],
    "Real Salt Lake": ["real salt lake", "salt lake", "rsl"],
    "Houston Dynamo": ["houston", "dynamo"],
    "Minnesota United": ["minnesota", "minnesota utd", "minnesota united"],
    "Colorado Rapids": ["colorado", "rapids"],
    "Dallas": ["dallas", "fc dallas"],
    "Vancouver Whitecaps": ["vancouver", "whitecaps"],
    "CD Guadalajara": ["guadalajara", "chivas"],
    "Cruz Azul": ["cruz azul"],
    "River Plate": ["river plate", "river"],
    "Gimnasia": ["gimnasia", "gimnasia la plata"],
    "Thun": ["thun", "fc thun"],
    "Young Boys": ["young boys", "yb", "bern"],
    "Basel": ["basel", "fc basel"],
    "St Gallen": ["st gallen", "st. gallen", "sankt gallen"],
    "Sion": ["sion", "fc sion"],
    "Lugano": ["lugano", "fc lugano"],
    "Al Fateh": ["al fateh", "fateh"],
    "Al Najma": ["al najma", "najma"],
    "Al Ettifaq": ["al ettifaq", "ettifaq"],
    "Al Ittihad": ["al ittihad", "ittihad"],
    "Al Quadisiya": ["al quadisiya", "quadisiya", "al qadsiah"],
    "Al Hazem": ["al hazem", "hazem"],
    "Gomel": ["gomel", "fc gomel"],
    "Baranovichi": ["baranovichi", "fc baranovichi"],
    "Kudrivka": ["kudrivka"],
    "Rukh Lviv": ["rukh lviv", "rukh"],
    "Bradford City": ["bradford", "bradford city"],
    "Bolton Wanderers": ["bolton", "bolton wanderers"],
    "Girona": ["girona", "girona fc"],
    "Real Sociedad": ["real sociedad", "sociedad", "la real"],
    "Valencia": ["valencia", "valencia cf"],
    "Rayo Vallecano": ["rayo", "rayo vallecano"],
    "Bayern Munich": ["bayern", "bayern munich", "fc bayern", "bayern münchen"],
    "FC Koln": ["koln", "fc koln", "köln", "fc köln", "cologne"],
    "Porto": ["porto", "fc porto"],
    "Santa Clara": ["santa clara", "cd santa clara"],
    "Famalicao": ["famalicao", "famalicão", "fc famalicao"],
    "Alverca": ["alverca", "fc alverca"],
    "Cordoba": ["cordoba", "córdoba", "cordoba cf"],
    "Albacete": ["albacete", "albacete bp"],
    "Vukovar": ["vukovar", "hsk vukovar", "vukovar 1991"],
    "NK Varazdin": ["varazdin", "nk varazdin", "varaždin", "nk varaždin"],
    "Bari": ["bari", "ssc bari", "as bari"],
    "Sudtirol": ["sudtirol", "südtirol", "fc sudtirol", "fc südtirol"],
    "Neman": ["neman", "fc neman", "neman grodno"],
    "Isloch": ["isloch", "fc isloch", "isloch minsk"],
    "Korona Kielce": ["korona", "korona kielce", "mks korona"],
    "Widzew Lodz": ["widzew", "widzew lodz", "widzew łódź", "rts widzew"],
    "Caykur Rizespor": ["rizespor", "caykur rizespor", "çaykur rizespor"],
    "Besiktas": ["besiktas", "beşiktaş", "besiktas jk"],
    "Zaglebie Lubin": ["zaglebie", "zaglebie lubin", "zagłębie lubin"],
    "Pogon Szczecin": ["pogon", "pogon szczecin", "szczecin", "pogoń szczecin"],
    "Nyiregyhaza Spartacus": ["nyiregyhaza", "nyíregyháza", "spartacus"],
    "Kazincbarcika": ["kazincbarcika", "kazincbarcikai"],
    "Arges": ["arges", "arges pitesti", "argeș"],
    "Rapid Bucuresti": ["rapid", "rapid bucuresti", "rapid bucurești"],
    "FC Minsk": ["fc minsk", "minsk"],
    "Belshina": ["belshina", "belshina bobruisk"],
    "Hapoel Tel Aviv": ["hapoel tel aviv", "h. tel aviv"],
    "Hapoel Be'er Sheva": ["hapoel be'er sheva", "h. be'er sheva", "hapoel beer sheva"],
    "Damac": ["damac", "damac fc"],
    "Al Fayha": ["al fayha", "fayha", "al feiha"],
    "Adelaide United": ["adelaide", "adelaide united"],
    "Auckland FC": ["auckland", "auckland fc"],
    "Shanghai Port": ["shanghai port", "shanghai sipg"],
    "Zhejiang Professional": ["zhejiang", "zhejiang professional"],
    "Henan": ["henan", "henan fc", "henan jianye"],
    "Sichuan Jiuniu": ["sichuan", "sichuan jiuniu"],
    "Tianjin Jinmen Tiger": ["tianjin", "tianjin jinmen", "jinmen tiger"],
    "Chengdu Rongcheng": ["chengdu", "chengdu rongcheng"],
    "Beijing Guoan": ["beijing", "beijing guoan"],
    "Qingdao Hainiu": ["qingdao hainiu", "hainiu"],
    "Partick Thistle": ["partick", "partick thistle"],
    "Dunfermline Athletic": ["dunfermline", "dunfermline athletic"],
    "Salford City": ["salford", "salford city"],
    "Grimsby Town": ["grimsby", "grimsby town"],
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
                      r'Croatian 1\. HNL|HNL|Prva HNL|Scottish Premiership|Eredivisie|A-League|'
                      r'Ekstraklasa|PKO Ekstraklasa|Polish Ekstraklasa|'
                      r'Turkish Super Lig|Süper Lig|'
                      r'Israeli Premier League|Ligat HaAl|'
                      r'Hungarian NB I|OTP Bank Liga|'
                      r'Romanian SuperLiga|SuperLiga|'
                      r'Chinese Super League|CSL|'
                      r'Australian A-League|'
                      r'Scottish Championship|'
                      r'League Two)', 
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
        'Segunda División', 'Scottish Premiership', 'Eredivisie', 'A-League',
        'Croatian 1. HNL', 'HNL', 'Prva HNL', 'Ekstraklasa', 'PKO Ekstraklasa',
        'Turkish Super Lig', 'Israeli Premier League', 'Hungarian NB I',
        'Romanian SuperLiga', 'Chinese Super League', 'Australian A-League',
        'Scottish Championship', 'Scottish Premiership Playoffs'
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
# ANALYSIS ENGINE — Evidence-Based Decision Tree
# ============================================================================
def analyze_match(data: dict) -> dict:
    result = {
        "bets": [],
        "badges": [],
        "warnings": [],
        "verdict": "PENDING",
        "skip_reasons": []
    }
    
    # Extract measurements
    home_win = data.get("home_win") or 0
    away_win = data.get("away_win") or 0
    draw_pct = data.get("draw") or 0
    btts = data.get("btts") or 0
    over_25 = data.get("over_25") or 0
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
    over25_trend = data.get("over_25_trend")
    
    home_form = data.get("home_form_all") or []
    away_form = data.get("away_form_all") or []
    
    home_wins = sum(1 for r in home_form[:6] if r == 'W')
    home_draws = sum(1 for r in home_form[:6] if r == 'D')
    home_losses = sum(1 for r in home_form[:6] if r == 'L')
    away_wins = sum(1 for r in away_form[:6] if r == 'W')
    away_draws = sum(1 for r in away_form[:6] if r == 'D')
    away_losses = sum(1 for r in away_form[:6] if r == 'L')
    
    home_unbeaten_5 = sum(1 for r in home_form[:5] if r in ['W', 'D'])
    away_unbeaten_5 = sum(1 for r in away_form[:5] if r in ['W', 'D'])
    
    h2h_total = data.get("h2h_total", 0)
    h2h_btts = data.get("h2h_btts_count", 0)
    h2h_scores = data.get("h2h_scores", [])
    
    league = data.get("league")
    is_top_league = league in TOP_LEAGUES if league else False
    
    # ========================================================================
    # STEP 1: TREND REVERSAL
    # ========================================================================
    trend_reversal = False
    if home_win_trend is not None and away_win_trend is not None:
        if home_win_trend <= -1.0 and away_win_trend >= 1.0:
            trend_reversal = True
            result["badges"].append("Trend Reversal: Home {:.1f} / Away {:.1f}".format(home_win_trend, away_win_trend))
            
            if home_win_trend <= -3.0 and away_win_trend >= 3.0:
                result["bets"].append({
                    "market": "Away Win or Draw",
                    "tier": "TIER 1",
                    "confidence": 8.0,
                    "probability": away_win + draw_pct,
                    "reason": "MAJOR market reversal: Home {:.1f}, Away {:.1f}. FADE home.".format(home_win_trend, away_win_trend)
                })
            else:
                result["bets"].append({
                    "market": "Away Win or Draw",
                    "tier": "TIER 2",
                    "confidence": 6.5,
                    "probability": away_win + draw_pct,
                    "reason": "Market reversal: Home {:.1f}, Away {:.1f}".format(home_win_trend, away_win_trend)
                })
    
    # ========================================================================
    # STEP 2: DOMINANT FAVORITE
    # ========================================================================
    if home_win >= 60 and not trend_reversal:
        underdog_scoring = away_o05 if away_o05 is not None else btts
        
        if underdog_scoring is not None and underdog_scoring < 52:
            if away_draws >= 3:
                result["bets"].append({
                    "market": "Home Win",
                    "tier": "TIER 2",
                    "confidence": 5.5,
                    "probability": home_win,
                    "reason": "Home favorite ({:.0f}%) but away team draws often ({}/6). Win to Nil too risky.".format(home_win, away_draws)
                })
                result["badges"].append("Resilient Away: {} draws in 6".format(away_draws))
            else:
                evidence = 1
                if home_o15 >= 60: evidence += 1
                if under_25 >= 50: evidence += 1
                if h2h_total >= 3 and h2h_btts <= 2: evidence += 1
                if is_top_league: evidence += 1
                
                confidence = 8.5 if evidence >= 3 else 7.0
                result["bets"].append({
                    "market": "Home Win to Nil",
                    "tier": "TIER 1",
                    "confidence": confidence,
                    "probability": home_win,
                    "reason": "Dominant favorite ({:.0f}%) + underdog scoring {:.0f}% | Evidence: {}/5".format(home_win, underdog_scoring, evidence)
                })
    
    if away_win >= 60 and not trend_reversal:
        underdog_scoring = home_o05 if home_o05 is not None else btts
        
        if underdog_scoring is not None and underdog_scoring < 52:
            if home_draws >= 3:
                result["bets"].append({
                    "market": "Away Win",
                    "tier": "TIER 2",
                    "confidence": 5.5,
                    "probability": away_win,
                    "reason": "Away favorite ({:.0f}%) but home team draws often ({}/6).".format(away_win, home_draws)
                })
            else:
                evidence = 1
                if away_o15 >= 60: evidence += 1
                if under_25 >= 50: evidence += 1
                if is_top_league: evidence += 1
                
                confidence = 8.5 if evidence >= 3 else 7.0
                result["bets"].append({
                    "market": "Away Win to Nil",
                    "tier": "TIER 1",
                    "confidence": confidence,
                    "probability": away_win,
                    "reason": "Dominant favorite ({:.0f}%) + underdog scoring {:.0f}% | Evidence: {}/4".format(away_win, underdog_scoring, evidence)
                })
    
    # ========================================================================
    # STEP 3: BTTS — Requires 2+ evidence
    # ========================================================================
    if btts >= 55:
        btts_evidence = 0
        evidence_details = []
        
        if btts_trend is not None and btts_trend >= 0.30:
            btts_evidence += 1
            evidence_details.append("BTTS trend +{:.2f}".format(btts_trend))
        
        if h2h_btts >= 4 and h2h_total >= 5:
            btts_evidence += 2
            evidence_details.append("H2H BTTS {}/{}".format(h2h_btts, h2h_total))
        
        home_scores_often = sum(1 for r in home_form[:6] if r in ['W', 'D']) >= 3
        away_scores_often = sum(1 for r in away_form[:6] if r in ['W', 'D']) >= 3
        if home_scores_often and away_scores_often:
            btts_evidence += 1
            evidence_details.append("Both teams scoring regularly")
        
        if home_draws >= 4 or away_draws >= 4:
            btts_evidence += 2
            evidence_details.append("Draw streak ({} draws)".format(max(home_draws, away_draws)))
        
        if (home_wins >= 5 and away_losses >= 4) or (away_wins >= 5 and home_losses >= 4):
            btts_evidence += 1
            evidence_details.append("Opposing form extremes")
        
        if abs(home_win - away_win) < 12:
            btts_evidence += 1
            evidence_details.append("Close match")
        
        if home_o05 is not None and away_o05 is not None and home_o05 >= 65 and away_o05 >= 60:
            btts_evidence += 1
            evidence_details.append("Both likely to score (H:{:.0f}% A:{:.0f}%)".format(home_o05, away_o05))
        
        # Contradictions
        if btts_trend is not None and btts_trend < -0.05:
            btts_evidence -= 1
            result["warnings"].append("BTTS trending down ({:+.2f})".format(btts_trend))
        
        if h2h_total >= 3 and h2h_btts <= 1:
            btts_evidence -= 2
            result["warnings"].append("H2H contradicts BTTS ({}/{})".format(h2h_btts, h2h_total))
        
        if btts_evidence >= 2:
            confidence = 7.5 if btts_evidence >= 4 else 6.5 if btts_evidence >= 3 else 5.5
            result["bets"].append({
                "market": "BTTS",
                "tier": "TIER 1" if btts_evidence >= 3 else "TIER 2",
                "confidence": confidence,
                "probability": btts,
                "reason": "BTTS {:.1f}% | Evidence: {}/7 ({})".format(btts, btts_evidence, ', '.join(evidence_details[:3]))
            })
        else:
            result["skip_reasons"].append("BTTS {:.1f}% — insufficient evidence ({}/7)".format(btts, btts_evidence))
    
    # ========================================================================
    # STEP 4: OVER 2.5
    # ========================================================================
    if over_25 >= 58:
        over_evidence = 0
        over_details = []
        
        if over25_trend is not None and over25_trend >= 0.30:
            over_evidence += 1
            over_details.append("O2.5 trend +{:.2f}".format(over25_trend))
        
        if btts >= 55:
            over_evidence += 1
            over_details.append("BTTS supports goals")
        
        if home_o15 >= 40 and away_o15 >= 25:
            over_evidence += 1
            over_details.append("Both can score 2+")
        
        if h2h_total >= 3:
            h2h_over = sum(1 for h, a in h2h_scores if h + a > 2)
            if h2h_over >= h2h_total * 0.6:
                over_evidence += 2
                over_details.append("H2H Over 2.5: {}/{}".format(h2h_over, h2h_total))
        
        if is_top_league:
            over_evidence += 1
            over_details.append("Top league")
        
        if over_evidence >= 2:
            confidence = 7.5 if over_evidence >= 4 else 6.5 if over_evidence >= 3 else 5.5
            result["bets"].append({
                "market": "Over 2.5",
                "tier": "TIER 1" if over_evidence >= 3 else "TIER 2",
                "confidence": confidence,
                "probability": over_25,
                "reason": "Over 2.5 {:.1f}% | Evidence: {}/5 ({})".format(over_25, over_evidence, ', '.join(over_details[:3]))
            })
    
    # ========================================================================
    # STEP 5: UNDER 2.5
    # ========================================================================
    if under_25 >= 55:
        under_evidence = 0
        under_details = []
        
        if btts < 50:
            under_evidence += 1
            under_details.append("BTTS below 50%")
        
        if home_o15 < 45 and away_o15 < 35:
            under_evidence += 1
            under_details.append("Both low scoring")
        
        if h2h_total >= 3:
            h2h_under = sum(1 for h, a in h2h_scores if h + a <= 2)
            if h2h_under >= h2h_total * 0.6:
                under_evidence += 2
                under_details.append("H2H Under 2.5: {}/{}".format(h2h_under, h2h_total))
        
        if home_losses >= 4 and away_losses >= 4:
            under_evidence -= 2
            result["warnings"].append("Both on losing streaks — Under 2.5 risky")
        
        if under_evidence >= 2:
            confidence = 7.5 if under_evidence >= 4 else 6.5 if under_evidence >= 3 else 5.5
            result["bets"].append({
                "market": "Under 2.5",
                "tier": "TIER 1" if under_evidence >= 3 else "TIER 2",
                "confidence": confidence,
                "probability": under_25,
                "reason": "Under 2.5 {:.1f}% | Evidence: {}/4 ({})".format(under_25, under_evidence, ', '.join(under_details[:3]))
            })
    
    # ========================================================================
    # STEP 6: DRAW — Unbeaten collision
    # ========================================================================
    if home_unbeaten_5 >= 4 and away_unbeaten_5 >= 4 and draw_pct >= 20:
        result["bets"].append({
            "market": "Draw",
            "tier": "TIER 2",
            "confidence": 6.0,
            "probability": draw_pct,
            "reason": "Unbeaten collision: Home {}/5, Away {}/5".format(home_unbeaten_5, away_unbeaten_5)
        })
        result["badges"].append("Unbeaten Collision")
    
    # ========================================================================
    # STEP 7: TEAM GOALS
    # ========================================================================
    if home_win >= 55 and home_o15 >= 55:
        result["bets"].append({
            "market": "Home Over 1.5 Goals",
            "tier": "TIER 2",
            "confidence": 6.0,
            "probability": home_o15,
            "reason": "Home scores 2+ regularly ({:.0f}%)".format(home_o15)
        })
    
    if away_win >= 55 and away_o15 >= 55:
        result["bets"].append({
            "market": "Away Over 1.5 Goals",
            "tier": "TIER 2",
            "confidence": 6.0,
            "probability": away_o15,
            "reason": "Away scores 2+ regularly ({:.0f}%)".format(away_o15)
        })
    
    # ========================================================================
    # STEP 8: WIN STREAK
    # ========================================================================
    if home_wins >= 5 and home_win >= 45:
        if not any(b["market"] == "Home Win" for b in result["bets"]):
            result["bets"].append({
                "market": "Home Win",
                "tier": "TIER 2",
                "confidence": 6.5,
                "probability": home_win,
                "reason": "Home on {}-match win streak".format(home_wins)
            })
        result["badges"].append("Win Streak: Home {}W".format(home_wins))
    
    if away_wins >= 5 and away_win >= 45:
        if not any(b["market"] == "Away Win" for b in result["bets"]):
            result["bets"].append({
                "market": "Away Win",
                "tier": "TIER 2",
                "confidence": 6.5,
                "probability": away_win,
                "reason": "Away on {}-match win streak".format(away_wins)
            })
        result["badges"].append("Win Streak: Away {}W".format(away_wins))
    
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
    
    if draw_pct >= 25:
        result["warnings"].append("High draw probability ({:.1f}%) — avoid match result bets".format(draw_pct))
    if abs(home_win - away_win) < 10 and home_win > 0 and away_win > 0:
        result["warnings"].append("Close match — no clear favorite")
    if btts < 50 and not any(b["market"] == "BTTS" for b in result["bets"]) and btts > 0:
        result["warnings"].append("BTTS below 50% — BTTS is likely a trap here")
    if not is_top_league and league:
        result["warnings"].append("'{}' is not a top-5 league — lower reliability".format(league))
    
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
                "home_form": home_form_str,
                "away_form": away_form_str,
            },
            "away_data": {},
            "prediction": bets_str,
            "confidence_score": round(top["confidence"] / 10, 2) if top else 0,
            "winner": top["market"] if top else "SKIP",
            "winner_confidence": f"{top['confidence']}/10" if top else "0",
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
        
        # Determine if prediction was correct
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
    st.title("📊 Match Analyzer V2.0.1")
    st.caption("Evidence-Based Decision Engine | 37-Match Backtest | Multi-Factor Scoring")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        st.markdown("*Paste the full match data block from 'Form, Standings, Stats' through 'Form Data'*")
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                with st.spinner("Running evidence-based analysis..."):
                    data = parse_match_data(raw_text)
                
                if not data.get("home_team") or not data.get("away_team"):
                    st.error("Could not detect team names. Check the data format.")
                else:
                    analysis = analyze_match(data)
                    save_to_db(data, analysis)
                    
                    league_display = data.get('league') or 'Club Match'
                    
                    if analysis["verdict"] == "SKIP":
                        st.warning("{} vs {} — {}".format(data['home_team'], data['away_team'], league_display))
                    else:
                        st.success("{} vs {} — {}".format(data['home_team'], data['away_team'], league_display))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        btts_trend_str = "{:+.2f}".format(data.get('btts_trend', 0)) if data.get('btts_trend') is not None else "N/A"
                        o25_trend_str = "{:+.2f}".format(data.get('over_25_trend', 0)) if data.get('over_25_trend') is not None else "N/A"
                        
                        home_o05 = "{:.1f}%".format(data['home_over_05_goals']) if data.get('home_over_05_goals') is not None else "N/A"
                        home_o15 = "{:.1f}%".format(data['home_over_15_goals']) if data.get('home_over_15_goals') is not None else "N/A"
                        away_o05 = "{:.1f}%".format(data['away_over_05_goals']) if data.get('away_over_05_goals') is not None else "N/A"
                        away_o15 = "{:.1f}%".format(data['away_over_15_goals']) if data.get('away_over_15_goals') is not None else "N/A"
                        
                        st.markdown("""
                        <div class="edge-box">
                            <strong>Probabilities</strong><br>
                            Home: {:.1f}% | Draw: {:.1f}% | Away: {:.1f}%<br>
                            BTTS: {:.1f}% | O2.5: {:.1f}% | U2.5: {:.1f}%<br>
                            Home O0.5: {} | Home O1.5: {}<br>
                            Away O0.5: {} | Away O1.5: {}
                        </div>
                        """.format(
                            data.get('home_win', 0), data.get('draw', 0), data.get('away_win', 0),
                            data.get('btts', 0), data.get('over_25', 0), data.get('under_25', 0),
                            home_o05, home_o15, away_o05, away_o15
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        home_form_str = '-'.join(data.get('home_form_all', [])[:6]) if data.get('home_form_all') else 'N/A'
                        away_form_str = '-'.join(data.get('away_form_all', [])[:6]) if data.get('away_form_all') else 'N/A'
                        
                        st.markdown("""
                        <div class="edge-box">
                            <strong>Form & H2H</strong><br>
                            BTTS Trend: {} | O2.5 Trend: {}<br>
                            Home: {}<br>
                            Away: {}<br>
                            H2H BTTS: {}/{}
                        </div>
                        """.format(btts_trend_str, o25_trend_str, home_form_str, away_form_str,
                                   data.get('h2h_btts_count', 0), data.get('h2h_total', 0)),
                        unsafe_allow_html=True)
                    
                    st.markdown("### Recommendations")
                    
                    if analysis["bets"]:
                        emoji_map = {
                            "BTTS": "⚽⚽", "Over 2.5": "🔥", "Under 2.5": "🛡️", 
                            "Draw": "🤝", "Home Over 1.5 Goals": "🏠⚽", 
                            "Away Over 1.5 Goals": "✈️⚽",
                            "Home Win to Nil": "🏠🧤", "Away Win to Nil": "✈️🧤",
                            "Home Win": "🏠", "Away Win": "✈️",
                            "Away Win or Draw": "✈️🤝"
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
                        skip_reasons = analysis.get("skip_reasons", [])
                        skip_text = "<br>".join(skip_reasons) if skip_reasons else "No strong signal with sufficient supporting evidence detected."
                        st.markdown("""
                        <div class="output-card skip-card">
                            <div class="verdict-skip">
                                <div class="big-text">SKIP — NO BET</div>
                                <p style="color:#94a3b8;margin-top:0.5rem;">{}</p>
                                <p style="color:#64748b;font-size:0.8rem;">Evidence-based engine requires 2+ supporting signals for any recommendation.</p>
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
                            st.markdown('<span class="badge-caution"> {}</span>'.format(w), unsafe_allow_html=True)
    
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
            st.info("No pending analyses. Go to the Analyze tab to paste match data.")
    
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        if not results:
            st.info("No results recorded yet. Submit results in the Post-Match tab.")
        else:
            # Calculate performance metrics
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
                actual_home = r.get('actual_home_goals', 0)
                actual_away = r.get('actual_away_goals', 0)
                actual_total = (actual_home or 0) + (actual_away or 0)
                
                is_correct = False
                
                # Check each predicted market
                markets = pred.split(' | ')
                for market in markets:
                    market = market.strip()
                    if market == 'BTTS' and actual_btts:
                        is_correct = True
                        break
                    if market == 'Over 2.5' and actual_over25:
                        is_correct = True
                        break
                    if market == 'Under 2.5' and not actual_over25 and actual_total > 0:
                        is_correct = True
                        break
                    if market == 'Home Win' and actual_winner == 'HOME':
                        is_correct = True
                        break
                    if market == 'Away Win' and actual_winner == 'AWAY':
                        is_correct = True
                        break
                    if market == 'Draw' and actual_winner == 'DRAW':
                        is_correct = True
                        break
                    if market == 'Home Win to Nil' and actual_winner == 'HOME' and actual_away == 0:
                        is_correct = True
                        break
                    if market == 'Away Win to Nil' and actual_winner == 'AWAY' and actual_home == 0:
                        is_correct = True
                        break
                    if market == 'Home Over 1.5 Goals' and (actual_home or 0) >= 2:
                        is_correct = True
                        break
                    if market == 'Away Over 1.5 Goals' and (actual_away or 0) >= 2:
                        is_correct = True
                        break
                    if market == 'Away Win or Draw' and actual_winner in ['AWAY', 'DRAW']:
                        is_correct = True
                        break
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
            
            # Display stats
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Total Tracked</div>
                </div>
                """.format(total), unsafe_allow_html=True)
            with c2:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Bets Placed</div>
                </div>
                """.format(bet_count), unsafe_allow_html=True)
            with c3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-number">{}%</div>
                    <div class="stat-label">Win Rate</div>
                </div>
                """.format(win_rate), unsafe_allow_html=True)
            with c4:
                st.markdown("""
                <div class="stat-box">
                    <div class="stat-number">{}</div>
                    <div class="stat-label">Skipped</div>
                </div>
                """.format(skip_count), unsafe_allow_html=True)
            
            st.markdown("---")
            st.write("**Correct: {} | Incorrect: {}**".format(correct, incorrect))
            
            # Detailed table
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                
                # Determine if this row was correct
                row_correct = None
                if pred == 'SKIP':
                    row_correct = 'skip'
                else:
                    markets = pred.split(' | ')
                    for market in markets:
                        market = market.strip()
                        if market == 'BTTS' and r.get('actual_btts'):
                            row_correct = 'correct'
                            break
                        if market == 'Over 2.5' and r.get('actual_over25'):
                            row_correct = 'correct'
                            break
                        if market == 'Under 2.5' and not r.get('actual_over25') and (actual_home or 0) + (actual_away or 0) > 0:
                            row_correct = 'correct'
                            break
                        if market == 'Home Win' and r.get('actual_winner') == 'HOME':
                            row_correct = 'correct'
                            break
                        if market == 'Away Win' and r.get('actual_winner') == 'AWAY':
                            row_correct = 'correct'
                            break
                        if market == 'Draw' and r.get('actual_winner') == 'DRAW':
                            row_correct = 'correct'
                            break
                        if market == 'Home Win to Nil' and r.get('actual_winner') == 'HOME' and actual_away == 0:
                            row_correct = 'correct'
                            break
                        if market == 'Away Win to Nil' and r.get('actual_winner') == 'AWAY' and actual_home == 0:
                            row_correct = 'correct'
                            break
                        if market == 'Home Over 1.5 Goals' and (actual_home or 0) >= 2:
                            row_correct = 'correct'
                            break
                        if market == 'Away Over 1.5 Goals' and (actual_away or 0) >= 2:
                            row_correct = 'correct'
                            break
                        if market == 'Away Win or Draw' and r.get('actual_winner') in ['AWAY', 'DRAW']:
                            row_correct = 'correct'
                            break
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
                    "Prediction": pred,
                    "Score": "{}-{}".format(actual_home if actual_home is not None else '-', actual_away if actual_away is not None else '-'),
                    "Result": badge,
                })
            
            df = pd.DataFrame(rows)
            st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
