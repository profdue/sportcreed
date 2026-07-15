"""
MATCH ANALYZER V13.1 — FORENSIC LOGIC ENGINE
Direct implementation of ALL forensic discoveries from 21+ completed matches
"""

import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import json
import time
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
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Match Analyzer V13.1", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .cautious-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #1a2a00 0%, #0a1a00 100%); }
    .danger-card { border: 3px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .ft-card { border-left: 5px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .verdict-skip { text-align: center; padding: 1.5rem; }
    .verdict-skip .big-text { font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .section-label { font-size: 0.9rem; font-weight: 700; color: #10b981; margin-top: 1rem; }
    .metric-card { background: #0f172a; border-radius: 10px; padding: 0.75rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; color: #94a3b8; }
    .accuracy-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .lock-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .dead-rubber-warning { background: #7c2d12; color: #fed7aa; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.8rem; margin: 0.5rem 0; border: 2px solid #ef4444; }
    .win-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .loss-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .condition-true { color: #10b981; font-weight: 700; }
    .condition-false { color: #ef4444; font-weight: 700; }
    .condition-box { background: #0f172a; border-radius: 8px; padding: 0.75rem; margin: 0.25rem 0; }
    .priority-rule { background: #1a2a1a; border-left: 4px solid #f59e0b; padding: 0.75rem; margin: 0.5rem 0; border-radius: 4px; }
    .upload-container { border: 2px dashed #3b82f6; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .upload-container:hover { border-color: #60a5fa; background: rgba(59, 130, 246, 0.05); }
    .league-badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }
    .league-badge.no { background: #ef4444; color: #fff; }
    .league-badge.br { background: #10b981; color: #fff; }
    .league-badge.uk { background: #3b82f6; color: #fff; }
    .league-badge.es { background: #f59e0b; color: #000; }
    .league-badge.it { background: #8b5cf6; color: #fff; }
    .league-badge.de { background: #ec4899; color: #fff; }
    .league-badge.au { background: #f59e0b; color: #000; }
    .league-badge.unknown { background: #64748b; color: #fff; }
    .score-container { background: #0f172a; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .score-number { font-size: 3rem; font-weight: 800; text-align: center; }
    .score-label { font-size: 0.8rem; color: #94a3b8; text-align: center; }
    .factor-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }
    .factor-name { color: #94a3b8; }
    .factor-value { font-weight: 600; }
    .factor-points { color: #fbbf24; font-weight: 700; }
    .stake-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .stake-full { background: #10b981; color: #000; }
    .stake-half { background: #f59e0b; color: #000; }
    .stake-small { background: #ef4444; color: #fff; }
    .stake-zero { background: #64748b; color: #fff; }
    .quality-box { background: #0a2a0a; border-radius: 8px; padding: 0.5rem 1rem; margin: 0.25rem 0; border: 1px solid #2a4a2a; }
    .streak-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .streak-loss { background: #ef4444; color: #fff; }
    .streak-win { background: #10b981; color: #000; }
    .dead-rubber-badge { background: #7c2d12; color: #fed7aa; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; border: 1px solid #ef4444; }
    .goal-badge { background: #3b82f6; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .bet-separator { border: none; border-top: 1px dashed #475569; margin: 0.5rem 0; }
    .primary-bet-card { border-left: 4px solid #f59e0b; background: linear-gradient(135deg, #1a2a1a 0%, #0a1a0a 100%); }
    .goal-bet-card { border-left: 4px solid #3b82f6; background: linear-gradient(135deg, #0a1a2a 0%, #0a0a1a 100%); }
    .already-stored { background: #1a2a2a; border: 1px solid #f59e0b; border-radius: 4px; padding: 0.2rem 0.6rem; color: #fbbf24; font-size: 0.7rem; font-weight: 700; display: inline-block; }
    .forensic-finding { background: #1a2a1a; border: 1px solid #10b981; border-radius: 4px; padding: 0.2rem 0.6rem; color: #10b981; font-size: 0.7rem; font-weight: 700; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_stake_display(stake_value: str) -> tuple:
    stake_mapping = {
        "Full": ("2 units", "stake-full"),
        "Half": ("1 unit", "stake-half"),
        "Small": ("0.25 unit", "stake-small"),
        "Small or Skip": ("0.25 unit", "stake-small"),
        "2 units": ("2 units", "stake-full"),
        "1 unit": ("1 unit", "stake-half"),
        "0.25 unit": ("0.25 unit", "stake-small"),
        "0 units": ("0 units", "stake-zero"),
        "": ("0 units", "stake-zero")
    }
    
    if stake_value in stake_mapping:
        return stake_mapping[stake_value]
    else:
        return (stake_value, "stake-zero")

def parse_match_date(date_str: str) -> datetime:
    if not date_str:
        return datetime(1900, 1, 1)
    
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M")
    except:
        pass
    
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d")
    except:
        pass
    
    try:
        return datetime.strptime(date_str.strip(), "%d/%m/%Y %H:%M")
    except:
        pass
    
    return datetime(1900, 1, 1)

def format_date_display(date_str: str) -> str:
    dt = parse_match_date(date_str)
    if dt.year == 1900:
        return date_str
    return dt.strftime("%Y-%m-%d %H:%M")

def check_match_exists(home_team: str, away_team: str, match_date: str) -> bool:
    try:
        response = supabase.table("match_analyses").select("id").eq("home_team", home_team).eq("away_team", away_team).eq("match_date", match_date).execute()
        return len(response.data) > 0
    except:
        return False

def get_league_config(league: str) -> dict:
    config = {
        "relegation_threshold": 15,
        "league_size": 20,
        "europe_threshold": 4,
        "goals_fallback": 2.50
    }
    
    if "Norway" in league or "Eliteserien" in league:
        config["relegation_threshold"] = 15
        config["league_size"] = 16
        config["goals_fallback"] = 2.75
    elif "Brazil" in league or "Serie A" in league or "Br1" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["europe_threshold"] = 4
        config["goals_fallback"] = 2.66
    elif "Premier" in league or "EPL" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.75
    elif "AuV" in league or "NPL" in league or "Australia" in league:
        config["relegation_threshold"] = 11
        config["league_size"] = 14
        config["europe_threshold"] = 3
        config["goals_fallback"] = 2.80
    else:
        config["relegation_threshold"] = 15
        config["league_size"] = 20
        config["goals_fallback"] = 2.50
    
    return config

def detect_league(text: str) -> str:
    if "Brasileiro Serie A" in text or "Brazil" in text or "Br1" in text:
        return "Brazil Serie A"
    elif "Premier League" in text or "England" in text or "EPL" in text:
        return "Premier League"
    elif "Eliteserien" in text or "Norway" in text:
        return "Norway Eliteserien"
    elif "LaLiga" in text or "Spain" in text:
        return "La Liga"
    elif "Serie A" in text and "Italy" in text:
        return "Serie A"
    elif "Bundesliga" in text or "Germany" in text:
        return "Bundesliga"
    elif "AuV" in text or "NPL Victoria" in text:
        return "Australia NPL Victoria"
    else:
        return "Unknown League"

def get_league_badge(league: str) -> str:
    if "Norway" in league or "Eliteserien" in league:
        return "no"
    elif "Brazil" in league or "Serie A" in league:
        return "br"
    elif "Premier" in league or "EPL" in league:
        return "uk"
    elif "La Liga" in league:
        return "es"
    elif "Serie A" in league and "Italy" in league:
        return "it"
    elif "Bundesliga" in league:
        return "de"
    elif "Australia" in league or "NPL" in league:
        return "au"
    else:
        return "unknown"

# ============================================================================
# TEXT PARSER
# ============================================================================
def parse_text_data(text: str) -> dict:
    league = detect_league(text)
    league_config = get_league_config(league)
    
    result = {
        "league": league,
        "league_config": league_config,
        "matches": [],
        "home_table": {},
        "away_table": {},
        "form_data": {}
    }
    
    sections = split_into_sections(text)
    
    matches = parse_predictions(sections.get("predictions", ""))
    result["matches"] = matches
    
    home_table = parse_table(sections.get("home_table", ""), "HOME")
    result["home_table"] = home_table
    
    away_table = parse_table(sections.get("away_table", ""), "AWAY")
    result["away_table"] = away_table
    
    form_data = parse_form(sections.get("form", ""))
    result["form_data"] = form_data
    
    return result

def split_into_sections(text: str) -> dict:
    result = {
        "predictions": "",
        "home_table": "",
        "away_table": "",
        "form": ""
    }
    lines = text.split('\n')
    
    predictions_start = None
    home_table_start = None
    away_table_start = None
    form_start = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if re.match(r'^Round\s*\d+', line_stripped):
            if predictions_start is None:
                predictions_start = i
        
        if "HOME TABLE" in line_stripped:
            if home_table_start is None:
                home_table_start = i
        
        if "AWAY TABLE" in line_stripped:
            if away_table_start is None:
                away_table_start = i
        
        if "LAST 6 MATCHES TABLE" in line_stripped:
            if form_start is None:
                form_start = i
    
    if predictions_start is not None:
        end = home_table_start if home_table_start is not None else len(lines)
        result["predictions"] = '\n'.join(lines[predictions_start:end])
    
    if home_table_start is not None:
        end = away_table_start if away_table_start is not None else len(lines)
        result["home_table"] = '\n'.join(lines[home_table_start:end])
    
    if away_table_start is not None:
        end = form_start if form_start is not None else len(lines)
        result["away_table"] = '\n'.join(lines[away_table_start:end])
    
    if form_start is not None:
        result["form"] = '\n'.join(lines[form_start:])
    
    return result

def parse_predictions(text: str) -> list:
    matches = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        is_finished = False
        actual_home = None
        actual_away = None
        
        for j in range(i, min(i + 3, len(lines))):
            check_line = lines[j].strip()
            if "FT" in check_line:
                is_finished = True
                ft_match = re.search(r'FT\s+(\d+)\s*-\s*(\d+)', check_line)
                if ft_match:
                    actual_home = int(ft_match.group(1))
                    actual_away = int(ft_match.group(2))
                break
        
        if not re.search(r'\d{6,}.*°', line):
            i += 1
            continue
        
        cleaned = line.replace(' ', '')
        
        pct_match = re.search(r'^(\d{2})(\d{2})(\d{2})', cleaned)
        if not pct_match:
            i += 1
            continue
        
        home_pct = int(pct_match.group(1))
        draw_pct = int(pct_match.group(2))
        away_pct = int(pct_match.group(3))
        
        rest = cleaned[6:]
        
        prediction_char = rest[0]
        rest = rest[1:]
        
        if prediction_char == 'X':
            prediction = 'X'
        elif prediction_char in ['1', '2']:
            prediction = prediction_char
        else:
            i += 1
            continue
        
        dash_pos = rest.find('-')
        if dash_pos == -1:
            i += 1
            continue
        
        score_part = rest[:dash_pos].strip()
        avg_part = rest[dash_pos+1:].strip()
        
        if prediction == 'X':
            if score_part and score_part[0].isdigit():
                draw_score = int(score_part[0])
            else:
                draw_score = 1
            home_goals = draw_score
            away_goals = draw_score
        else:
            score_dash_pos = score_part.find('-')
            if score_dash_pos != -1:
                home_str = score_part[:score_dash_pos].strip()
                away_str = score_part[score_dash_pos+1:].strip()
                
                if home_str and home_str[0].isdigit():
                    home_goals = int(home_str[0])
                else:
                    home_goals = 1
                
                if away_str and away_str[0].isdigit():
                    away_goals = int(away_str[0])
                else:
                    away_goals = 0
            else:
                if score_part and score_part[0].isdigit():
                    home_goals = int(score_part[0])
                else:
                    home_goals = 1
                
                if avg_part and avg_part[0].isdigit():
                    away_goals = int(avg_part[0])
                else:
                    away_goals = 0
        
        avg_match = re.search(r'(\d+\.\d+)°', avg_part)
        if not avg_match:
            i += 1
            continue
        
        raw = avg_match.group(1)
        raw = raw[1:]
        
        match = re.search(r'(\d+)\.(\d{2})(\d*)', raw)
        if match:
            int_part = int(match.group(1))
            dec_part = int(match.group(2))
            temp_str = match.group(3)
            
            avg_goals = float(f"{int_part}.{dec_part:02d}")
            temperature = int(temp_str) if temp_str else 0
        else:
            i += 1
            continue
        
        coeff_match = re.search(r'°(\d+\.\d+)', avg_part)
        coefficient = float(coeff_match.group(1)) if coeff_match else None
        
        date_index = None
        for j in range(i-1, max(0, i-10), -1):
            prev_line = lines[j].strip()
            if re.match(r'^\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}', prev_line):
                date_index = j
                break
        
        if date_index is None:
            i += 1
            continue
        
        date_line = lines[date_index].strip()
        
        away_team = None
        for j in range(date_index - 1, max(0, date_index - 4), -1):
            candidate = lines[j].strip()
            if candidate and candidate not in ["", "AuV", "EPL", "Br1", "PRE", "VIEW"]:
                if re.search(r'[A-Za-zÀ-ÿ]', candidate):
                    away_team = candidate
                    break
        
        home_team = None
        if away_team:
            away_index = None
            for j in range(date_index - 1, max(0, date_index - 4), -1):
                if lines[j].strip() == away_team:
                    away_index = j
                    break
            
            if away_index:
                for j in range(away_index - 1, max(0, away_index - 4), -1):
                    candidate = lines[j].strip()
                    if candidate and candidate not in ["", "AuV", "EPL", "Br1", "PRE", "VIEW"]:
                        if re.search(r'[A-Za-zÀ-ÿ]', candidate):
                            home_team = candidate
                            break
        
        if not home_team or not away_team:
            i += 1
            continue
        
        is_duplicate = False
        for existing in matches:
            if (existing["home_team"] == home_team and 
                existing["away_team"] == away_team and 
                existing["date"] == date_line):
                is_duplicate = True
                break
        
        if is_duplicate:
            i += 1
            continue
        
        matches.append({
            "home_team": home_team,
            "away_team": away_team,
            "date": date_line,
            "home_pct": home_pct,
            "draw_pct": draw_pct,
            "away_pct": away_pct,
            "prediction": prediction,
            "correct_score_home": home_goals,
            "correct_score_away": away_goals,
            "avg_goals": avg_goals,
            "temperature": temperature,
            "coefficient": coefficient,
            "is_finished": is_finished,
            "actual_home": actual_home,
            "actual_away": actual_away
        })
        
        i += 1
    
    return matches

def parse_table(text: str, table_type: str) -> dict:
    table_data = {}
    lines = text.split('\n')
    
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if "PTS" in line and "GP" in line and "W" in line and "D" in line and "L" in line:
            in_table = True
            continue
        
        if not in_table:
            continue
        
        line = line.replace('\t', ' ')
        line = ' '.join(line.split())
        
        match = re.search(r'^(\d+)\s+([^\d]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)$', line)
        if not match:
            match = re.search(r'^(\d+)\s+([^\d]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)', line)
        
        if match:
            position = int(match.group(1))
            team_name = match.group(2).strip()
            points = int(match.group(3))
            gp = int(match.group(4))
            wins = int(match.group(5))
            draws = int(match.group(6))
            losses = int(match.group(7))
            gf = int(match.group(8))
            ga = int(match.group(9))
            gd = int(match.group(10))
            
            table_data[team_name] = {
                "position": position,
                "points": points,
                "gp": gp,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "gf": gf,
                "ga": ga,
                "gd": gd
            }
    
    return table_data

def parse_form(text: str) -> dict:
    form_data = {}
    lines = text.split('\n')
    
    in_form = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if "PTS" in line and "GP" in line and "W" in line and "D" in line and "L" in line:
            in_form = True
            continue
        
        if not in_form:
            continue
        
        line = line.replace('\t', ' ')
        line = ' '.join(line.split())
        
        match = re.search(r'^(\d+)\s+([^\d]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)$', line)
        if not match:
            match = re.search(r'^(\d+)\s+([^\d]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)', line)
        
        if match:
            position = int(match.group(1))
            team_name = match.group(2).strip()
            points = int(match.group(3))
            gp = int(match.group(4))
            wins = int(match.group(5))
            draws = int(match.group(6))
            losses = int(match.group(7))
            gf = int(match.group(8))
            ga = int(match.group(9))
            gd = int(match.group(10))
            
            losing_streak = 0
            if losses >= 3:
                losing_streak = min(losses, 6)
            
            form_data[team_name] = {
                "position": position,
                "points": points,
                "gp": gp,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "gf": gf,
                "ga": ga,
                "gd": gd,
                "form_points": points,
                "losing_streak": losing_streak,
                "winning_streak": wins if wins >= 3 else 0
            }
    
    return form_data

def convert_match_to_data(match: dict, home_table: dict, away_table: dict, form_data: dict, 
                          league: str = "Unknown") -> dict:
    league_config = get_league_config(league)
    home_team = match.get('home_team', 'Unknown')
    away_team = match.get('away_team', 'Unknown')
    
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "date": match.get('date'),
        "home_pct": match.get('home_pct'),
        "draw_pct": match.get('draw_pct'),
        "away_pct": match.get('away_pct'),
        "prediction": match.get('prediction'),
        "correct_score_home": match.get('correct_score_home'),
        "correct_score_away": match.get('correct_score_away'),
        "avg_goals": match.get('avg_goals', league_config["goals_fallback"]),
        "temperature": match.get('temperature'),
        "coefficient": match.get('coefficient'),
        "score_matrix": [],
        "is_finished": match.get('is_finished', False),
        "actual_home": match.get('actual_home'),
        "actual_away": match.get('actual_away'),
        "league_config": league_config
    }
    
    if home_team in home_table:
        data["home_position"] = home_table[home_team]["position"]
        data["home_points"] = home_table[home_team]["points"]
        data["home_gp"] = home_table[home_team]["gp"]
        data["home_wins"] = home_table[home_team]["wins"]
        data["home_draws"] = home_table[home_team]["draws"]
        data["home_losses"] = home_table[home_team]["losses"]
        data["home_gf"] = home_table[home_team]["gf"]
        data["home_ga"] = home_table[home_team]["ga"]
        data["home_gd"] = home_table[home_team]["gd"]
    
    if away_team in away_table:
        data["away_position"] = away_table[away_team]["position"]
        data["away_points"] = away_table[away_team]["points"]
        data["away_gp"] = away_table[away_team]["gp"]
        data["away_wins"] = away_table[away_team]["wins"]
        data["away_draws"] = away_table[away_team]["draws"]
        data["away_losses"] = away_table[away_team]["losses"]
        data["away_gf"] = away_table[away_team]["gf"]
        data["away_ga"] = away_table[away_team]["ga"]
        data["away_gd"] = away_table[away_team]["gd"]
    
    if home_team in form_data:
        data["home_form_points"] = form_data[home_team]["form_points"]
        data["home_form_wins"] = form_data[home_team]["wins"]
        data["home_form_draws"] = form_data[home_team]["draws"]
        data["home_form_losses"] = form_data[home_team]["losses"]
        data["home_losing_streak"] = form_data[home_team]["losing_streak"]
        data["home_winning_streak"] = form_data[home_team].get("winning_streak", 0)
    if away_team in form_data:
        data["away_form_points"] = form_data[away_team]["form_points"]
        data["away_form_wins"] = form_data[away_team]["wins"]
        data["away_form_draws"] = form_data[away_team]["draws"]
        data["away_form_losses"] = form_data[away_team]["losses"]
        data["away_losing_streak"] = form_data[away_team]["losing_streak"]
        data["away_winning_streak"] = form_data[away_team].get("winning_streak", 0)
    
    def get_block(position):
        if position is None:
            return None
        try:
            pos = int(position)
            league_size = league_config["league_size"]
            relegation_threshold = league_config["relegation_threshold"]
            europe_threshold = league_config["europe_threshold"]
            
            if pos <= europe_threshold:
                return "europe"
            elif pos >= relegation_threshold:
                return "relegation"
            else:
                return "mid"
        except:
            return None
    
    data["home_block"] = get_block(data.get("home_position"))
    data["away_block"] = get_block(data.get("away_position"))
    data["is_relegation_fight"] = (
        data["home_block"] == "relegation" or 
        data["away_block"] == "relegation"
    )
    data["home_desperate"] = data.get("home_losing_streak", 0) >= 3 or data["home_block"] == "relegation"
    data["away_desperate"] = data.get("away_losing_streak", 0) >= 3 or data["away_block"] == "relegation"
    
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    
    return data

# ============================================================================
# V13.1 FORENSIC LOGIC ENGINE - DIRECT IMPLEMENTATION OF ALL DISCOVERIES
# ============================================================================

def calculate_factor_breakdown(data: dict) -> dict:
    """Calculate the factor breakdown - same as before"""
    factors = {}
    total_score = 0
    
    # FACTOR 1: DRAW RATE (0-4 points)
    home_draw_rate = data.get("home_draws", 0) / max(data.get("home_gp", 1), 1)
    away_draw_rate = data.get("away_draws", 0) / max(data.get("away_gp", 1), 1)
    combined_draw_rate = (home_draw_rate + away_draw_rate) / 2
    
    if combined_draw_rate > 0.40:
        f1_points = 4
    elif combined_draw_rate >= 0.35:
        f1_points = 3
    elif combined_draw_rate >= 0.30:
        f1_points = 2
    elif combined_draw_rate >= 0.25:
        f1_points = 1
    else:
        f1_points = 0
    
    factors["draw_rate"] = {
        "value": f"{combined_draw_rate:.1%}",
        "points": f1_points,
        "weight": "HIGH"
    }
    total_score += f1_points
    
    # FACTOR 2: POINTS DIFFERENCE (0-3 points)
    home_points = data.get("home_points", 0)
    away_points = data.get("away_points", 0)
    points_diff = abs(home_points - away_points)
    
    if points_diff <= 2:
        f2_points = 3
    elif points_diff <= 5:
        f2_points = 2
    elif points_diff <= 8:
        f2_points = 1
    elif points_diff <= 11:
        f2_points = 0
    else:
        f2_points = -1
    
    factors["points_difference"] = {
        "value": f"{points_diff} pts",
        "points": f2_points,
        "weight": "HIGH"
    }
    total_score += f2_points
    
    # FACTOR 3: GOAL EXPECTANCY (0-3 points)
    avg_goals = data.get("avg_goals", 2.0)
    
    if avg_goals < 2.00:
        f3_points = 3
    elif avg_goals <= 2.20:
        f3_points = 3
    elif avg_goals <= 2.40:
        f3_points = 2
    elif avg_goals <= 2.60:
        f3_points = 1
    elif avg_goals <= 2.80:
        f3_points = 0
    elif avg_goals <= 3.00:
        f3_points = -1
    else:
        f3_points = -2
    
    factors["goal_expectancy"] = {
        "value": f"{avg_goals:.2f}",
        "points": f3_points,
        "weight": "HIGH"
    }
    total_score += f3_points
    
    # FACTOR 4: GD GAP (0-3 points)
    home_gd = data.get("home_gd", 0)
    away_gd = data.get("away_gd", 0)
    gd_gap = abs(home_gd - away_gd)
    
    if gd_gap <= 5:
        f4_points = 3
    elif gd_gap <= 10:
        f4_points = 2
    elif gd_gap <= 15:
        f4_points = 1
    elif gd_gap <= 20:
        f4_points = 0
    else:
        f4_points = -1
    
    factors["gd_gap"] = {
        "value": f"{gd_gap}",
        "points": f4_points,
        "weight": "MEDIUM"
    }
    total_score += f4_points
    
    # FACTOR 5: SCORING RATE (0-3 points)
    home_scoring = float(data.get("home_gf", 0)) / max(data.get("home_gp", 1), 1)
    away_scoring = float(data.get("away_gf", 0)) / max(data.get("away_gp", 1), 1)
    combined_scoring = (home_scoring + away_scoring) / 2
    
    if combined_scoring < 1.0:
        f5_points = 3
    elif combined_scoring < 1.20:
        f5_points = 2
    elif combined_scoring < 1.40:
        f5_points = 1
    elif combined_scoring < 1.60:
        f5_points = 0
    elif combined_scoring < 1.80:
        f5_points = -1
    else:
        f5_points = -2
    
    factors["scoring_rate"] = {
        "value": f"{combined_scoring:.2f}",
        "points": f5_points,
        "weight": "MEDIUM"
    }
    total_score += f5_points
    
    # FACTOR 6: PROBABILITY BALANCE (0-2 points)
    home_pct = data.get("home_pct", 33)
    draw_pct = data.get("draw_pct", 34)
    away_pct = data.get("away_pct", 33)
    
    probs = [home_pct, draw_pct, away_pct]
    variance = max(probs) - min(probs)
    
    if variance <= 5:
        f6_points = 2
    elif variance <= 10:
        f6_points = 1
    elif variance <= 15:
        f6_points = 0
    elif variance <= 20:
        f6_points = -1
    else:
        f6_points = -2
    
    factors["probability_balance"] = {
        "value": f"{variance}% range",
        "points": f6_points,
        "weight": "LOW"
    }
    total_score += f6_points
    
    # FACTOR 7: DOMINANCE GAP (0 to -4 points)
    home_win_rate = data.get("home_wins", 0) / max(data.get("home_gp", 1), 1)
    away_win_rate = data.get("away_wins", 0) / max(data.get("away_gp", 1), 1)
    dominance_gap = abs(home_win_rate - away_win_rate) * 100
    
    if dominance_gap <= 15:
        f7_points = 0
    elif dominance_gap <= 25:
        f7_points = -1
    elif dominance_gap <= 35:
        f7_points = -2
    elif dominance_gap <= 45:
        f7_points = -3
    else:
        f7_points = -4
    
    factors["dominance_gap"] = {
        "value": f"{dominance_gap:.0f}%",
        "points": f7_points,
        "weight": "LOW"
    }
    total_score += f7_points
    
    # FACTOR 8: STREAK (0 to +3 or -2 points)
    streak_points = 0
    home_losing = data.get("home_losing_streak", 0)
    away_losing = data.get("away_losing_streak", 0)
    home_winning = data.get("home_winning_streak", 0)
    away_winning = data.get("away_winning_streak", 0)
    
    if home_losing >= 5 or away_losing >= 5:
        streak_points += 3
    elif home_losing >= 3 or away_losing >= 3:
        streak_points += 2
    
    if home_winning >= 3 or away_winning >= 3:
        streak_points -= 2
    
    factors["streak"] = {
        "value": f"L{home_losing}/{away_losing} W{home_winning}/{away_winning}",
        "points": streak_points,
        "weight": "NEW"
    }
    total_score += streak_points
    
    # FACTOR 9: TEAM QUALITY (0 to +2 or -1 points)
    home_quality = (float(data.get("home_gf", 0)) / max(data.get("home_gp", 1), 1) + 
                   float(data.get("home_ga", 0)) / max(data.get("home_gp", 1), 1))
    away_quality = (float(data.get("away_gf", 0)) / max(data.get("away_gp", 1), 1) + 
                   float(data.get("away_ga", 0)) / max(data.get("away_gp", 1), 1))
    quality_diff = abs(home_quality - away_quality)
    
    if quality_diff <= 0.5:
        f9_points = 2
    elif quality_diff <= 1.0:
        f9_points = 1
    elif quality_diff <= 1.5:
        f9_points = 0
    else:
        f9_points = -1
    
    factors["team_quality"] = {
        "value": f"{quality_diff:.2f}",
        "points": f9_points,
        "weight": "NEW"
    }
    total_score += f9_points
    
    # FACTOR 10: DEAD RUBBER (0 or +2 points)
    home_block = data.get("home_block")
    away_block = data.get("away_block")
    is_relegation_fight = data.get("is_relegation_fight", False)
    
    is_dead_rubber = False
    if (home_block == "mid" and away_block == "mid" and not is_relegation_fight):
        is_dead_rubber = True
        f10_points = 2
        factors["dead_rubber"] = {
            "value": "YES ⚠️",
            "points": f10_points,
            "weight": "NEW"
        }
        total_score += f10_points
    else:
        factors["dead_rubber"] = {
            "value": "No",
            "points": 0,
            "weight": "NEW"
        }
    
    # NORMALIZE: Add offset to ensure no negatives
    MIN_OFFSET = 15
    normalized_score = total_score + MIN_OFFSET
    if normalized_score < 0:
        normalized_score = 0
    
    return {
        "factors": factors,
        "total_score": normalized_score,
        "raw_score": total_score,
        "is_dead_rubber": is_dead_rubber
    }

def calculate_total_factor_score(factors: dict) -> int:
    """Calculate total factor score by summing all factor points"""
    total = 0
    for name, factor in factors.items():
        points = factor.get("points", 0)
        total += points
    return total

# ============================================================================
# FORENSIC DISCOVERY 1: Goal Accuracy 95% = PERFECT
# ============================================================================
def analyze_goal_accuracy(goal_accuracy: str) -> dict:
    """
    FORENSIC DISCOVERY #6:
    - 95% accuracy = 100% correct (4/4) → AUTO FULL STAKE
    - 100% accuracy = 83.3% correct → STRONG
    - 83% accuracy = STRONG
    - 80% accuracy = 50% correct (coin flip) → SKIP
    """
    if not goal_accuracy:
        return {"level": "UNKNOWN", "score": 0, "action": "SKIP", "reason": "No goal accuracy data"}
    
    accuracy_map = {
        "95%": {"level": "PERFECT", "score": 100, "action": "FULL", "reason": "95% accuracy = PERFECT (4/4 in data)"},
        "100%": {"level": "STRONG", "score": 85, "action": "FULL", "reason": "100% accuracy = 83.3% correct"},
        "83%": {"level": "STRONG", "score": 70, "action": "FULL", "reason": "83% accuracy = STRONG"},
        "80%": {"level": "WEAK", "score": 0, "action": "SKIP", "reason": "80% accuracy = coin flip (50%)"}
    }
    
    for key, value in accuracy_map.items():
        if key in goal_accuracy:
            return value
    
    return {"level": "UNKNOWN", "score": 0, "action": "SKIP", "reason": f"Unknown accuracy: {goal_accuracy}"}

# ============================================================================
# FORENSIC DISCOVERY 2: Total Factor Score ≥ 12 = STRONG
# ============================================================================
def analyze_factor_score(total_factor_score: int) -> dict:
    """
    FORENSIC DISCOVERY #11:
    - Factor Score ≥ 12 = 83.3% correct (5/6) → FULL STAKE
    - Factor Score 8-11 = 50% correct → HALF STAKE
    - Factor Score < 8 = 28.6% correct → SMALL/SKIP
    """
    if total_factor_score >= 12:
        return {"level": "STRONG", "action": "FULL", "reason": f"Factor Score {total_factor_score} ≥ 12 (83.3% correct)"}
    elif total_factor_score >= 8:
        return {"level": "MEDIUM", "action": "HALF", "reason": f"Factor Score {total_factor_score} in 8-11 range (50% correct)"}
    else:
        return {"level": "WEAK", "action": "SMALL", "reason": f"Factor Score {total_factor_score} < 8 (28.6% correct)"}

# ============================================================================
# FORENSIC DISCOVERY 3: UNDER 2.5 = STRONGEST SIGNAL
# ============================================================================
def analyze_goal_bet_forensic(data: dict, factor_result: dict) -> dict:
    """
    FORENSIC DISCOVERY #5:
    - UNDER 2.5 = 90.9% correct → STRONGEST signal
    - OVER 2.5 = 77.8% correct → STRONG signal
    - Use combined scoring and conceding from database
    """
    
    # Calculate from database
    home_scoring = float(data.get("home_gf", 0)) / max(data.get("home_gp", 1), 1)
    away_scoring = float(data.get("away_gf", 0)) / max(data.get("away_gp", 1), 1)
    combined_scoring = home_scoring + away_scoring
    
    home_conceding = float(data.get("home_ga", 0)) / max(data.get("home_gp", 1), 1)
    away_conceding = float(data.get("away_ga", 0)) / max(data.get("away_gp", 1), 1)
    combined_conceding = home_conceding + away_conceding
    
    # Form adjustment
    home_form = data.get("home_form_points", 0) / 18
    away_form = data.get("away_form_points", 0) / 18
    form_adjustment = ((home_form + away_form) / 2) * 0.15
    
    # Expected goals
    expected_goals = (combined_scoring * 0.35) + (combined_conceding * 0.25) + form_adjustment
    
    # FORENSIC DISCOVERY: UNDER 2.5 is 90.9% correct
    if expected_goals <= 2.00:
        return {
            "goal_bet": "UNDER 2.5",
            "confidence": "HIGH (90.9% forensic accuracy)",
            "reason": f"Expected goals {expected_goals:.2f} ≤ 2.00 (UNDER 2.5 is 90.9% correct in data)",
            "expected_goals": round(expected_goals, 2)
        }
    elif expected_goals >= 2.50:
        return {
            "goal_bet": "OVER 2.5",
            "confidence": "HIGH (77.8% forensic accuracy)",
            "reason": f"Expected goals {expected_goals:.2f} ≥ 2.50 (OVER 2.5 is 77.8% correct in data)",
            "expected_goals": round(expected_goals, 2)
        }
    else:
        return {
            "goal_bet": None,
            "confidence": "LOW",
            "reason": f"Expected goals {expected_goals:.2f} in uncertain range (2.00-2.50)",
            "expected_goals": round(expected_goals, 2)
        }

# ============================================================================
# FORENSIC DISCOVERY 4: Combined Decision Engine
# ============================================================================
def forensic_decision_engine(data: dict, factor_result: dict) -> dict:
    """
    FORENSIC DECISION ENGINE - Direct implementation of ALL discoveries
    """
    
    # Get goal accuracy
    goal_accuracy = factor_result.get("goal_accuracy", "")
    accuracy_result = analyze_goal_accuracy(goal_accuracy)
    
    # Get factor score
    total_factor_score = calculate_total_factor_score(factor_result["factors"])
    factor_result_score = analyze_factor_score(total_factor_score)
    
    # Dead rubber (DISCOVERY 4: +3 not +2)
    is_dead_rubber = factor_result.get("is_dead_rubber", False)
    dead_rubber_bonus = 3 if is_dead_rubber else 0
    
    # Relegation fight (DISCOVERY 1: -2 penalty)
    is_relegation_fight = data.get("is_relegation_fight", False)
    relegation_penalty = -2 if is_relegation_fight else 0
    
    # Point gap (DISCOVERY 10: +2 if ≤ 3)
    home_points = data.get("home_points", 0)
    away_points = data.get("away_points", 0)
    point_gap = abs(home_points - away_points)
    point_gap_bonus = 2 if point_gap <= 3 else 0
    
    # ========================================================================
    # STEP 1: Goal Accuracy 95% = AUTO FULL STAKE (DISCOVERY 6)
    # ========================================================================
    if accuracy_result["action"] == "FULL" and accuracy_result["level"] == "PERFECT":
        return {
            "action": "SAFE",
            "stake": "2 units",
            "stake_display": "FULL",
            "confidence": "HIGHEST",
            "reason": f"Goal Accuracy 95% = PERFECT (4/4 in data)",
            "bet_strength": 100,
            "forensic_findings": [
                f"🔬 Goal Accuracy 95% = PERFECT (4/4)",
                f"🔬 Total Factor Score: {total_factor_score}",
                f"🔬 Dead Rubber: {is_dead_rubber} (+{dead_rubber_bonus})",
                f"🔬 Relegation Fight: {is_relegation_fight} ({relegation_penalty})",
                f"🔬 Point Gap: {point_gap} (+{point_gap_bonus})"
            ]
        }
    
    # ========================================================================
    # STEP 2: Goal Accuracy 80% = AUTO SKIP (DISCOVERY 6)
    # ========================================================================
    if accuracy_result["action"] == "SKIP" and accuracy_result["level"] == "WEAK":
        return {
            "action": "SKIP",
            "stake": "0 units",
            "stake_display": "SKIP",
            "confidence": "VERY LOW",
            "reason": f"Goal Accuracy 80% = coin flip (50%)",
            "bet_strength": 0,
            "forensic_findings": [
                f"🔬 Goal Accuracy 80% = coin flip (50%)",
                f"🔬 Total Factor Score: {total_factor_score}",
                f"🔬 Dead Rubber: {is_dead_rubber} (+{dead_rubber_bonus})",
                f"🔬 Relegation Fight: {is_relegation_fight} ({relegation_penalty})",
                f"🔬 Point Gap: {point_gap} (+{point_gap_bonus})"
            ]
        }
    
    # ========================================================================
    # STEP 3: Total Factor Score ≥ 12 = FULL STAKE (DISCOVERY 11)
    # ========================================================================
    if factor_result_score["action"] == "FULL":
        return {
            "action": "SAFE",
            "stake": "2 units",
            "stake_display": "FULL",
            "confidence": "HIGH",
            "reason": f"Total Factor Score {total_factor_score} ≥ 12 (83.3% correct)",
            "bet_strength": 85,
            "forensic_findings": [
                f"🔬 Total Factor Score {total_factor_score} ≥ 12 (83.3%)",
                f"🔬 Goal Accuracy: {accuracy_result['level']}",
                f"🔬 Dead Rubber: {is_dead_rubber} (+{dead_rubber_bonus})",
                f"🔬 Relegation Fight: {is_relegation_fight} ({relegation_penalty})",
                f"🔬 Point Gap: {point_gap} (+{point_gap_bonus})"
            ]
        }
    
    # ========================================================================
    # STEP 4: Calculate combined score with forensic adjustments
    # ========================================================================
    combined_score = (
        accuracy_result["score"] * 0.35 +
        total_factor_score * 0.25 +
        dead_rubber_bonus * 0.20 +
        point_gap_bonus * 0.10 +
        relegation_penalty * 0.05
    )
    
    # Normalize
    combined_score = min(100, max(0, combined_score * 1.2))
    
    # ========================================================================
    # STEP 5: Determine action based on combined score
    # ========================================================================
    if combined_score >= 65:
        action = "SAFE"
        stake = "2 units"
        stake_display = "FULL"
        confidence = "HIGH"
        bet_strength = combined_score
    elif combined_score >= 45:
        action = "CAUTIOUS"
        stake = "1 unit"
        stake_display = "HALF"
        confidence = "MEDIUM"
        bet_strength = combined_score
    elif combined_score >= 25:
        action = "DANGEROUS"
        stake = "0.25 unit"
        stake_display = "SMALL"
        confidence = "LOW"
        bet_strength = combined_score
    else:
        action = "SKIP"
        stake = "0 units"
        stake_display = "SKIP"
        confidence = "VERY LOW"
        bet_strength = combined_score
    
    # Build forensic findings
    forensic_findings = [
        f"🔬 Goal Accuracy: {accuracy_result['level']} ({accuracy_result['reason']})",
        f"🔬 Total Factor Score: {total_factor_score} ({factor_result_score['level']})",
        f"🔬 Dead Rubber: {is_dead_rubber} (+{dead_rubber_bonus})",
        f"🔬 Relegation Fight: {is_relegation_fight} ({relegation_penalty})",
        f"🔬 Point Gap: {point_gap} (+{point_gap_bonus})"
    ]
    
    return {
        "action": action,
        "stake": stake,
        "stake_display": stake_display,
        "confidence": confidence,
        "reason": f"Combined Score: {combined_score:.1f} → {action}",
        "bet_strength": round(bet_strength, 1),
        "forensic_findings": forensic_findings
    }

# ============================================================================
# V13.1 MAIN ANALYSIS ENGINE
# ============================================================================

def analyze_draw_match_v131(data: dict) -> dict:
    """
    V13.1 FORENSIC ANALYSIS ENGINE
    Direct implementation of ALL forensic discoveries
    """
    
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reason": None,
        "is_lock": False,
        "lock_reason": None,
        "winner_selection": None,
        "winner_reason": None,
        "goal_bet": None,
        "goal_confidence": None,
        "goal_reason": None,
        "warning": None,
        "warning_type": None,
        "draw_survival_score": None,
        "factor_breakdown": None,
        "action": None,
        "stake": None,
        "recommended_bet": None,
        "bet_strength": None,
        "forensic_findings": []
    }
    
    # ========================================================================
    # FILTER 1: Skip FT matches
    # ========================================================================
    if data.get("is_finished"):
        result["verdict"] = "SKIP"
        result["skip_reason"] = "Already played (FT)"
        result["classification"] = "⏭️ SKIPPED — Already Played"
        return result
    
    # ========================================================================
    # FILTER 2: Only analyze draw predictions
    # ========================================================================
    if data.get("prediction") != 'X':
        result["verdict"] = "SKIP"
        result["skip_reason"] = "Not a draw prediction (X)"
        result["classification"] = "⏭️ SKIPPED — Not a Draw Prediction"
        return result
    
    # ========================================================================
    # STEP 1: Calculate factor breakdown
    # ========================================================================
    factor_result = calculate_factor_breakdown(data)
    result["factor_breakdown"] = factor_result["factors"]
    result["draw_survival_score"] = factor_result["total_score"]
    
    # ========================================================================
    # STEP 2: Goal bet using forensic logic (DISCOVERY 5)
    # ========================================================================
    goal_result = analyze_goal_bet_forensic(data, factor_result)
    result["goal_bet"] = goal_result["goal_bet"]
    result["goal_confidence"] = goal_result["confidence"]
    result["goal_reason"] = goal_result["reason"]
    
    # Store goal accuracy for forensic analysis
    factor_result["goal_accuracy"] = goal_result.get("confidence", "")
    
    # ========================================================================
    # STEP 3: Forensic decision engine (DISCOVERIES 1-11)
    # ========================================================================
    decision = forensic_decision_engine(data, factor_result)
    
    result["action"] = decision["action"]
    result["stake"] = decision["stake"]
    result["bet_strength"] = decision["bet_strength"]
    result["forensic_findings"] = decision["forensic_findings"]
    
    # ========================================================================
    # STEP 4: Determine classification
    # ========================================================================
    if decision["action"] == "SAFE":
        result["verdict"] = "LOCK"
        result["is_lock"] = True
        result["lock_reason"] = f"Forensic Score: {decision['bet_strength']} → SAFE (2 units)"
        result["classification"] = "🔒 LOCK"
    elif decision["action"] == "CAUTIOUS":
        result["verdict"] = "RECOMMENDED"
        result["is_lock"] = False
        result["classification"] = "⚠️ CAUTIOUS"
    elif decision["action"] == "DANGEROUS":
        result["verdict"] = "RECOMMENDED"
        result["is_lock"] = False
        result["classification"] = "❗ DANGEROUS"
    else:
        result["verdict"] = "SKIP"
        result["skip_reason"] = f"Forensic Score: {decision['bet_strength']} < 25"
        result["classification"] = "⏭️ SKIP"
        result["winner_selection"] = "SKIP"
        result["winner_reason"] = "Forensic analysis indicates SKIP"
        return result
    
    # ========================================================================
    # STEP 5: Dead rubber warning
    # ========================================================================
    if factor_result.get("is_dead_rubber", False):
        result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for (+3 forensic bonus)"
        result["warning_type"] = "dead_rubber"
    
    # ========================================================================
    # STEP 6: Build primary bet
    # ========================================================================
    outcome_bet = "DOUBLE CHANCE: HOME or AWAY"
    outcome_reason = f"Forensic Score: {decision['bet_strength']} → {decision['action']}"
    
    result["winner_selection"] = outcome_bet
    result["winner_reason"] = outcome_reason
    result["recommended_bet"] = outcome_bet
    
    # ========================================================================
    # STEP 7: Build final result
    # ========================================================================
    result["primary_bet"] = {
        "outcome_bet": outcome_bet,
        "outcome_accuracy": "83%",
        "outcome_reason": outcome_reason,
        "goal_bet": goal_result["goal_bet"],
        "goal_confidence": goal_result["confidence"],
        "goal_reason": goal_result["reason"],
        "is_lock": result["is_lock"],
        "lock_reason": result["lock_reason"],
        "score": factor_result["total_score"],
        "raw_score": factor_result["raw_score"],
        "factors": factor_result["factors"],
        "action": decision["action"],
        "stake": decision["stake"],
        "bet_strength": decision["bet_strength"],
        "forensic_findings": decision["forensic_findings"]
    }
    
    # Classification
    classification = outcome_bet
    if result["goal_bet"]:
        classification += f" | {result['goal_bet']}"
    result["classification"] = classification
    
    return result

# ============================================================================
# EVALUATION ENGINE
# ============================================================================
def evaluate_bet(prediction: str, home_goals, away_goals) -> dict:
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {
            "results": [],
            "all_won": False,
            "is_correct": False,
            "actual": "INVALID",
            "summary": "INVALID DATA"
        }
    
    total = home + away
    pred = prediction.strip().upper()
    
    if ' | ' in pred:
        bets = pred.split(' | ')
    else:
        bets = [pred]
    
    results = []
    
    for bet in bets:
        bet = bet.strip()
        is_win = False
        bet_type = "UNKNOWN"
        
        if 'DOUBLE CHANCE' in bet:
            bet_type = "DOUBLE CHANCE"
            if 'HOME' in bet:
                is_win = home != away
            elif 'AWAY' in bet:
                is_win = away != home
            else:
                is_win = home != away
        
        elif 'OVER 2.5' in bet:
            bet_type = "OVER 2.5"
            is_win = total > 2
        
        elif 'UNDER 2.5' in bet:
            bet_type = "UNDER 2.5"
            is_win = total <= 2
        
        elif 'REDUCE STAKE' in bet or 'SKIP' in bet:
            bet_type = "DOUBLE CHANCE"
            is_win = home != away
        
        results.append({
            "bet": bet,
            "type": bet_type,
            "win": is_win,
            "result": "WIN" if is_win else "LOSS"
        })
    
    if not results:
        return {
            "results": [],
            "all_won": False,
            "is_correct": False,
            "actual": f"{home}-{away}",
            "summary": "NO BETS EVALUATED"
        }
    
    all_won = all(r["win"] for r in results)
    
    return {
        "results": results,
        "all_won": all_won,
        "is_correct": all_won,
        "actual": f"{home}-{away}",
        "summary": "✅ ALL WON" if all_won else "⚠️ ONE OR MORE LOST"
    }

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================
def backtest(results: list) -> list:
    backtest_results = []
    for r in results:
        pred = r.get('bet_market', '')
        if pred == 'SKIP':
            continue
        
        score = r.get('draw_survival_score', 0)
        display_score = max(0, score) if isinstance(score, (int, float)) else score
        
        if display_score <= 4:
            action = "SAFE"
            stake = "2 units"
        elif display_score <= 7:
            action = "CAUTIOUS"
            stake = "1 unit"
        elif display_score <= 9:
            action = "DANGEROUS"
            stake = "0.25 unit"
        else:
            action = "SKIP"
            stake = "0 units"
        
        actual_home = r.get('actual_home_goals')
        actual_away = r.get('actual_away_goals')
        
        if actual_home is not None and actual_away is not None:
            evaluation = evaluate_bet(pred, actual_home, actual_away)
            result_text = "WIN" if evaluation["all_won"] else "LOSS"
            details = " | ".join([f"{b['bet']}: {b['result']}" for b in evaluation["results"]])
        else:
            result_text = "PENDING"
            details = ""
        
        backtest_results.append({
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Date": r.get('match_date', ''),
            "Score": display_score,
            "Action": action,
            "Stake": stake,
            "Bet": pred,
            "Actual": f"{actual_home}-{actual_away}" if actual_home is not None else "—",
            "Result": result_text,
            "Details": details
        })
    
    return backtest_results

# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(data: dict, analysis: dict, league: str = "Unknown"):
    is_draw_prediction = data.get("prediction") == 'X'
    is_ft = data.get("is_finished", False)
    
    if is_ft or not is_draw_prediction:
        return None
    
    try:
        # Check if match already exists
        home_team = data.get("home_team", "Unknown")
        away_team = data.get("away_team", "Unknown")
        match_date = data.get("date", str(date.today()))
        
        exists = check_match_exists(home_team, away_team, match_date)
        if exists:
            return "ALREADY_EXISTS"
        
        primary = analysis.get("primary_bet", {})
        score_result = analysis.get("draw_survival_score", 0)
        
        # Calculate desperate flags
        home_losing_streak = data.get("home_losing_streak", 0)
        away_losing_streak = data.get("away_losing_streak", 0)
        home_block = data.get("home_block")
        away_block = data.get("away_block")
        
        home_desperate = home_losing_streak >= 3 or home_block == "relegation"
        away_desperate = away_losing_streak >= 3 or away_block == "relegation"
        
        # Build bet_market WITHOUT combining with "|"
        bet_market = "DOUBLE CHANCE: HOME or AWAY"
        goal_bet = analysis.get("goal_bet")
        if goal_bet:
            bet_market = f"DOUBLE CHANCE: HOME or AWAY + {goal_bet}"
        
        record = {
            # ===== BASIC INFO =====
            "home_team": home_team,
            "away_team": away_team,
            "match_date": match_date,
            "league": league,
            
            # ===== FOREBET PREDICTIONS =====
            "home_pct": data.get("home_pct"),
            "draw_pct": data.get("draw_pct"),
            "away_pct": data.get("away_pct"),
            "prediction": data.get("prediction"),
            "avg_goals": data.get("avg_goals"),
            "temperature": data.get("temperature"),
            "coefficient": data.get("coefficient"),
            
            # ===== CORRECT SCORE (PREDICTED) =====
            "correct_score_home": data.get("correct_score_home"),
            "correct_score_away": data.get("correct_score_away"),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            
            # ===== TABLE DATA =====
            "home_position": data.get("home_position"),
            "away_position": data.get("away_position"),
            "home_points": data.get("home_points"),
            "away_points": data.get("away_points"),
            "home_block": data.get("home_block"),
            "away_block": data.get("away_block"),
            "is_relegation_fight": data.get("is_relegation_fight", False),
            
            # ===== FORM DATA =====
            "home_form_points": data.get("home_form_points"),
            "away_form_points": data.get("away_form_points"),
            
            # ===== STREAKS =====
            "home_losing_streak": data.get("home_losing_streak", 0),
            "away_losing_streak": data.get("away_losing_streak", 0),
            "home_winning_streak": data.get("home_winning_streak", 0),
            "away_winning_streak": data.get("away_winning_streak", 0),
            
            # ===== DESPERATE FLAGS =====
            "home_desperate": home_desperate,
            "away_desperate": away_desperate,
            
            # ===== DRAW SURVIVAL SCORE =====
            "draw_survival_score": score_result,
            "action": analysis.get("action"),
            "stake": analysis.get("stake"),
            "recommended_bet": analysis.get("recommended_bet"),
            "factor_breakdown": json.dumps(analysis.get("factor_breakdown", {})),
            
            # ===== BET DISPLAY =====
            "bet_market": bet_market,
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "DOUBLE_CHANCE",
            
            # ===== VERDICT =====
            "verdict": analysis.get("verdict", "SKIP"),
            "is_lock": analysis.get("is_lock", False),
            "lock_reason": analysis.get("lock_reason"),
            
            # ===== WINNER SELECTION =====
            "winner_selection": analysis.get("winner_selection"),
            "winner_reason": analysis.get("winner_reason"),
            
            # ===== GOAL BET =====
            "goal_bet": analysis.get("goal_bet"),
            "goal_confidence": analysis.get("goal_confidence"),
            
            # ===== WARNINGS =====
            "warning": analysis.get("warning"),
            "warning_type": analysis.get("warning_type"),
            
            # ===== DRAW CONDITIONS =====
            "draw_conditions": json.dumps(analysis.get("draw_conditions", {})),
            
            # ===== STATUS FLAGS =====
            "is_finished": data.get("is_finished", False),
            "skip_reason": analysis.get("skip_reason"),
            
            # ===== ACTUAL RESULTS (will be filled later) =====
            "result_entered": False,
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_total_goals": None,
            "actual_over25": False,
            "actual_winner": None,
            "actual_btts": False,
        }
        
        response = supabase.table("match_analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", False).execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date", "")))
    except Exception as e:
        st.error(f"Error fetching pending: {e}")
        return []

def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        supabase.table("match_analyses").update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total,
            "actual_over25": total > 2,
            "actual_winner": "HOME" if home_goals > away_goals else "AWAY" if away_goals > home_goals else "DRAW",
            "actual_btts": home_goals > 0 and away_goals > 0,
            "result_entered": True,
            "result_entered_at": datetime.now().isoformat()
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False

def get_results():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", True).execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date", "")), reverse=True)
    except: return []

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_score_breakdown(factors: dict, total_score: int, raw_score: int = None):
    display_score = max(0, total_score)
    
    if display_score <= 4:
        color = "#10b981"
        status = "SAFE"
        emoji = "✅"
        stake_label = "2 units"
        stake_class = "stake-full"
    elif display_score <= 7:
        color = "#f59e0b"
        status = "CAUTIOUS"
        emoji = "⚠️"
        stake_label = "1 unit"
        stake_class = "stake-half"
    elif display_score <= 9:
        color = "#ef4444"
        status = "DANGEROUS"
        emoji = "❗"
        stake_label = "0.25 unit"
        stake_class = "stake-small"
    else:
        color = "#64748b"
        status = "SKIP"
        emoji = "⏭️"
        stake_label = "0 units"
        stake_class = "stake-zero"
    
    raw_display = ""
    if raw_score is not None and raw_score != display_score:
        raw_display = f"<div style='font-size:0.7rem; color:#64748b;'>Raw: {raw_score} → Normalized: {display_score}</div>"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0; border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">DRAW SURVIVAL SCORE (Normalized)</div>
                <div style="font-size: 3rem; font-weight: 800; color: {color};">{display_score}</div>
                {raw_display}
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{emoji} {status}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    <span class="stake-badge {stake_class}">Stake: {stake_label}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Factor Breakdown")
    
    for name, factor in factors.items():
        weight = factor.get("weight", "")
        points = factor.get("points", 0)
        value = factor.get("value", "")
        
        if points >= 0:
            points_display = f"+{points}"
        else:
            points_display = str(points)
        
        is_new = weight == "NEW"
        new_badge = " 🆕" if is_new else ""
        
        st.markdown(f"""
        <div class="factor-row" style="{'background: #1a2a1a; border-radius: 4px;' if is_new else ''}">
            <span class="factor-name">{name.replace('_', ' ').title()} <span style="color:#64748b; font-size:0.7rem;">({weight})</span>{new_badge}</span>
            <span class="factor-value">{value}</span>
            <span class="factor-points">{points_display}</span>
        </div>
        """, unsafe_allow_html=True)


def display_analysis(data: dict, analysis: dict, league: str = "Unknown", already_stored: bool = False):
    if analysis.get("verdict") == "SKIP":
        skip_reason = analysis.get("skip_reason") or "Not a draw prediction"
        is_ft = "Already played" in skip_reason or "FT" in skip_reason
        
        if is_ft:
            st.markdown(f"""
            <div class="output-card ft-card">
                <div class="verdict-skip">
                    <div class="big-text">⏭️ SKIPPED — Already Played</div>
                    <p style="color:#94a3b8; font-size:1.1rem; margin:0.5rem 0;">
                        {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}
                    </p>
                    <p style="color:#ef4444; font-weight:600;">{skip_reason}</p>
                    <p style="color:#94a3b8;">Prediction: {data.get('prediction', '?')}</p>
                    <p style="color:#64748b; font-size:0.8rem;">
                        Expected Score: {data.get('correct_score_home', '?')}-{data.get('correct_score_away', '?')} 
                        | Avg Goals: {data.get('avg_goals', 0):.2f}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="output-card skip-card">
                <div class="verdict-skip">
                    <div class="big-text">⏭️ SKIPPED</div>
                    <p style="color:#94a3b8; font-size:1.1rem; margin:0.5rem 0;">
                        {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}
                    </p>
                    <p style="color:#94a3b8;">Prediction: {data.get('prediction', '?')} — {skip_reason}</p>
                    <p style="color:#64748b; font-size:0.8rem;">
                        Expected Score: {data.get('correct_score_home', '?')}-{data.get('correct_score_away', '?')} 
                        | Avg Goals: {data.get('avg_goals', 0):.2f}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return
    
    badge_class = get_league_badge(league)
    st.markdown(f'<span class="league-badge {badge_class}">{league}</span>', unsafe_allow_html=True)
    
    if already_stored:
        st.markdown('<span class="already-stored">📌 ALREADY STORED — Displaying prediction only</span>', unsafe_allow_html=True)
    
    warning = analysis.get("warning") or ''
    if warning:
        st.markdown(f'<div class="dead-rubber-warning">{warning}</div>', unsafe_allow_html=True)
    
    primary = analysis.get("primary_bet", {})
    
    display_score_breakdown(
        analysis.get("factor_breakdown", {}),
        primary.get("score", 0),
        primary.get("raw_score")
    )
    
    st.markdown("---")
    
    # ========================================================================
    # V13.1: Forensic Findings Display
    # ========================================================================
    forensic_findings = primary.get("forensic_findings", [])
    if forensic_findings:
        st.markdown("### 🔬 FORENSIC FINDINGS")
        for finding in forensic_findings:
            st.markdown(f'<div style="background: #0f172a; border-radius: 4px; padding: 0.25rem 0.5rem; margin: 0.15rem 0; color: #94a3b8; font-size: 0.85rem;">{finding}</div>', unsafe_allow_html=True)
        st.markdown("---")
    
    # ========================================================================
    # Bet Strength Display
    # ========================================================================
    bet_strength = primary.get("bet_strength", 0)
    if bet_strength:
        strength_color = "#10b981" if bet_strength >= 65 else "#f59e0b" if bet_strength >= 45 else "#ef4444" if bet_strength >= 25 else "#64748b"
        st.markdown(f"""
        <div style="background: #0f172a; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0; border-left: 4px solid {strength_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.8rem; color: #94a3b8;">BET STRENGTH (Forensic)</span>
                    <div style="font-size: 1.5rem; font-weight: 800; color: {strength_color};">{bet_strength}</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.8rem; color: #94a3b8;">Confidence: {primary.get('action', 'UNKNOWN')}</span>
                    <div style="font-size: 0.8rem; color: #64748b;">
                        Goal Acc: {primary.get('goal_confidence', 'N/A')} | Factor: {primary.get('score', 0)}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    action = primary.get("action", "UNKNOWN")
    stake = primary.get("stake", "?")
    stake_display, stake_class = get_stake_display(stake)
    
    if action == "SAFE":
        card_class = "lock-card"
        lock_icon = "🔒"
        lock_text = f"SAFE — {stake_display}"
        border_color = "#f59e0b"
    elif action == "CAUTIOUS":
        card_class = "cautious-card"
        lock_icon = "⚠️"
        lock_text = f"CAUTIOUS — {stake_display}"
        border_color = "#f59e0b"
    elif action == "DANGEROUS":
        card_class = "danger-card"
        lock_icon = "❗"
        lock_text = f"DANGEROUS — {stake_display}"
        border_color = "#ef4444"
    else:
        card_class = "skip-card"
        lock_icon = "⏭️"
        lock_text = f"SKIP — {stake_display}"
        border_color = "#64748b"
    
    st.markdown(f"""
    <div class="output-card {card_class} primary-bet-card" style="border-left: 4px solid {border_color};">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{lock_icon}</span>
            <span style="font-size: 1.2rem; font-weight: 700;">PRIMARY BET — {lock_text}</span>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.25rem; color: #10b981;">DOUBLE CHANCE: HOME or AWAY</div>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem;">
            <span>📊 Accuracy: {primary.get('outcome_accuracy', 'N/A')}</span>
            <span>📝 Reason: {primary.get('outcome_reason', 'N/A')}</span>
            <span style="background: {border_color}; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">Stake: {stake_display}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<hr class="bet-separator">', unsafe_allow_html=True)
    
    # ========================================================================
    # Goal Bet Display
    # ========================================================================
    goal_bet = primary.get('goal_bet')
    goal_confidence = primary.get('goal_confidence')
    goal_reason = primary.get('goal_reason')
    
    if goal_bet:
        confidence_color = "#10b981" if "HIGH" in goal_confidence else "#f59e0b" if "MEDIUM" in goal_confidence else "#ef4444"
        st.markdown(f"""
        <div class="output-card goal-bet-card" style="border-left: 4px solid {confidence_color};">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">⚽</span>
                <span style="font-size: 1.2rem; font-weight: 700;">GOAL BET — SEPARATE BET</span>
                <span style="background: {confidence_color}; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">{goal_confidence or 'LOW'}</span>
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; margin-bottom: 0.25rem; color: {confidence_color};">{goal_bet}</div>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem;">
                <span>📝 Reason: {goal_reason}</span>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #fbbf24;">
                ⚠️ This is a SEPARATE bet from the PRIMARY BET above. Place it independently.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="output-card" style="border-left: 4px solid #64748b; background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">⚽</span>
                <span style="font-size: 1.2rem; font-weight: 700;">GOAL BET</span>
            </div>
            <div style="font-size: 1rem; color: #94a3b8;">None available — {goal_reason or 'No clear pattern'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # Key Metrics
    # ========================================================================
    st.markdown("### 📊 Key Metrics")
    
    avg_goals = data.get("avg_goals", 2.0)
    home_form = data.get("home_form_points", "?")
    away_form = data.get("away_form_points", "?")
    home_pos = data.get("home_position", "?")
    away_pos = data.get("away_position", "?")
    home_block = data.get("home_block", "?")
    away_block = data.get("away_block", "?")
    home_losing = data.get("home_losing_streak", 0)
    away_losing = data.get("away_losing_streak", 0)
    home_winning = data.get("home_winning_streak", 0)
    away_winning = data.get("away_winning_streak", 0)
    home_scoring = float(data.get("home_gf", 0)) / max(data.get("home_gp", 1), 1)
    away_scoring = float(data.get("away_gf", 0)) / max(data.get("away_gp", 1), 1)
    combined_scoring = home_scoring + away_scoring
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_form} vs {away_form}</div><div class="metric-label">Form Points</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{combined_scoring:.2f}</div><div class="metric-label">Combined Scoring</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_block} vs {away_block}</div><div class="metric-label">Competitive Block</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_pos} vs {away_pos}</div><div class="metric-label">Standings</div></div>', unsafe_allow_html=True)
    with m5:
        desperate = "Yes" if (home_losing >= 3 or away_losing >= 3 or data.get("is_relegation_fight", False)) else "No"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{desperate}</div><div class="metric-label">Desperation</div></div>', unsafe_allow_html=True)
    
    if home_losing >= 3 or away_losing >= 3 or home_winning >= 3 or away_winning >= 3:
        streak_msg = []
        if home_losing >= 3:
            streak_msg.append(f'<span class="streak-badge streak-loss">🔴 {data.get("home_team", "Home")}: {home_losing}L</span>')
        if away_losing >= 3:
            streak_msg.append(f'<span class="streak-badge streak-loss">🔴 {data.get("away_team", "Away")}: {away_losing}L</span>')
        if home_winning >= 3:
            streak_msg.append(f'<span class="streak-badge streak-win">🟢 {data.get("home_team", "Home")}: {home_winning}W</span>')
        if away_winning >= 3:
            streak_msg.append(f'<span class="streak-badge streak-win">🟢 {data.get("away_team", "Away")}: {away_winning}W</span>')
        st.markdown(f'<div style="margin: 0.5rem 0;">{" ".join(streak_msg)}</div>', unsafe_allow_html=True)
    
    if data.get("score_matrix"):
        st.markdown("### 🎯 Score Matrix")
        sorted_scores = sorted(data["score_matrix"], key=lambda x: x.get("probability", 0), reverse=True)[:5]
        score_cols = st.columns(min(5, len(sorted_scores)))
        for idx, s in enumerate(sorted_scores):
            with score_cols[idx]:
                if s.get("home_goals") is not None and s.get("away_goals") is not None:
                    bg = "#1e293b" if s.get("home_goals", 0) != s.get("away_goals", 0) else "#2a1a00"
                    prob = s.get("probability", 0)
                    st.markdown(f'<div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;"><div style="font-size:1.2rem; font-weight:800;">{s.get("score", "?-?")}</div><div style="font-size:0.7rem; color:#94a3b8;">{prob:.1f}%</div></div>', unsafe_allow_html=True)

# ============================================================================
# LIVE DASHBOARD
# ============================================================================
def display_live_dashboard():
    st.markdown("### 📊 Live Dashboard")
    
    results = get_results()
    if not results:
        st.info("No results recorded yet.")
        return
    
    total = len(results)
    correct = 0
    incorrect = 0
    
    safe_total = 0
    safe_correct = 0
    cautious_total = 0
    cautious_correct = 0
    dangerous_total = 0
    dangerous_correct = 0
    
    goal_total = 0
    goal_correct = 0
    
    for r in results:
        pred = r.get('bet_market', '')
        if pred == 'SKIP':
            continue
        
        primary_pred = "DOUBLE CHANCE: HOME or AWAY"
        evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
        
        if evaluation["all_won"]:
            correct += 1
        else:
            incorrect += 1
        
        stake = r.get('stake', '')
        if stake == 'Full' or stake == '2 units':
            safe_total += 1
            if evaluation["all_won"]:
                safe_correct += 1
        elif stake == 'Half' or stake == '1 unit':
            cautious_total += 1
            if evaluation["all_won"]:
                cautious_correct += 1
        elif stake == 'Small or Skip' or stake == '0.25 unit':
            dangerous_total += 1
            if evaluation["all_won"]:
                dangerous_correct += 1
        
        goal_bet = r.get('goal_bet')
        if goal_bet:
            goal_total += 1
            goal_eval = evaluate_bet(goal_bet, r.get('actual_home_goals'), r.get('actual_away_goals'))
            if goal_eval["all_won"]:
                goal_correct += 1
    
    overall_rate = round(correct / total * 100) if total > 0 else 0
    safe_rate = round(safe_correct / safe_total * 100) if safe_total > 0 else 0
    cautious_rate = round(cautious_correct / cautious_total * 100) if cautious_total > 0 else 0
    dangerous_rate = round(dangerous_correct / dangerous_total * 100) if dangerous_total > 0 else 0
    goal_rate = round(goal_correct / goal_total * 100) if goal_total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{total}</div>
            <div class="stat-label">Total Bets</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{overall_rate}%</div>
            <div class="stat-label">PRIMARY BET Win Rate ({correct}/{total})</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{goal_rate}%</div>
            <div class="stat-label">GOAL BET Win Rate ({goal_correct}/{goal_total})</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### By Stake Level (PRIMARY BET)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-box" style="background: #0a2a0a; border: 1px solid #10b981;">
            <div class="stat-number">{safe_rate}%</div>
            <div class="stat-label">SAFE (2 units) — {safe_correct}/{safe_total}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box" style="background: #1a2a00; border: 1px solid #f59e0b;">
            <div class="stat-number">{cautious_rate}%</div>
            <div class="stat-label">CAUTIOUS (1 unit) — {cautious_correct}/{cautious_total}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-box" style="background: #2a0a0a; border: 1px solid #ef4444;">
            <div class="stat-number">{dangerous_rate}%</div>
            <div class="stat-label">DANGEROUS (0.25 unit) — {dangerous_correct}/{dangerous_total}</div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# DISPLAY RECORDS TABLE
# ============================================================================
def display_records_table(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    
    total = len(results)
    correct = 0
    incorrect = 0
    lock_correct = 0
    lock_total = 0

    scores = []
    safe_scores = []
    cautious_scores = []
    dangerous_scores = []
    
    for r in results:
        score = r.get('draw_survival_score', 0)
        display_score = max(0, score)
        scores.append(display_score)
        if display_score <= 4:
            safe_scores.append(display_score)
        elif display_score <= 7:
            cautious_scores.append(display_score)
        elif display_score <= 9:
            dangerous_scores.append(display_score)

    for r in results:
        pred = r.get('bet_market', '')
        if pred == 'SKIP':
            continue
        
        primary_pred = "DOUBLE CHANCE: HOME or AWAY"
        evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
        
        if evaluation["all_won"]:
            correct += 1
            if r.get('is_lock', False):
                lock_correct += 1
        else:
            incorrect += 1
        if r.get('is_lock', False):
            lock_total += 1

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Draws Tracked</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">PRIMARY BET Win Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Incorrect</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{len(safe_scores)}</div><div class="stat-label">SAFE Bets (2u)</div></div>', unsafe_allow_html=True)

    st.markdown("### 📊 Draw Survival Score Distribution")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ SAFE (≤4)", len(safe_scores))
    with col2:
        st.metric("⚠️ CAUTIOUS (5-7)", len(cautious_scores))
    with col3:
        st.metric("❗ DANGEROUS (8-9)", len(dangerous_scores))
    with col4:
        st.metric("⏭️ SKIP (≥10)", total - len(safe_scores) - len(cautious_scores) - len(dangerous_scores))

    if lock_total > 0:
        lock_rate = round(lock_correct / lock_total * 100) if lock_total > 0 else 0
        st.markdown(f"🔒 **Lock Signals:** {lock_correct}/{lock_total} correct ({lock_rate}%)")

    st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")

    rows = []
    for r in results:
        pred = r.get('bet_market', '')
        actual_home = r.get('actual_home_goals')
        actual_away = r.get('actual_away_goals')
        league = r.get('league', '')
        badge_class = get_league_badge(league)
        score = r.get('draw_survival_score', '?')
        display_score = max(0, score) if isinstance(score, (int, float)) else score
        action = r.get('action', '?')
        stake = r.get('stake', '?')
        stake_display, _ = get_stake_display(stake)
        
        warning = r.get('warning') or ''
        dead_rubber_badge = ' ⚠️DR' if warning and 'DEAD RUBBER' in warning else ''
        
        primary_result = "🟢 WIN" if (actual_home is not None and actual_away is not None and actual_home != actual_away) else "🔴 LOSS"
        
        goal_bet = r.get('goal_bet')
        if goal_bet and actual_home is not None and actual_away is not None:
            goal_eval = evaluate_bet(goal_bet, actual_home, actual_away)
            goal_result = "🟢 WIN" if goal_eval["all_won"] else "🔴 LOSS"
            goal_display = f"{goal_bet} → {goal_result}"
        else:
            goal_display = "—"
        
        score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
        match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
        score_label = f"{display_score} ({action}) — {stake_display}{dead_rubber_badge}" if score != '?' else '?'

        rows.append({
            "Date": r.get("match_date", ""),
            "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
            "Match": match_display,
            "Score": score_label,
            "PRIMARY BET": "DOUBLE CHANCE: HOME or AWAY",
            "PRIMARY Result": primary_result,
            "GOAL BET": goal_display,
            "Actual": score_display,
        })

    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.markdown("### 🔬 Backtest Results")
    backtest_results = backtest(results)
    if backtest_results:
        backtest_df = pd.DataFrame(backtest_results)
        st.dataframe(backtest_df, use_container_width=True)
        
        total_backtest = len(backtest_results)
        wins = sum(1 for r in backtest_results if r["Result"] == "WIN")
        losses = sum(1 for r in backtest_results if r["Result"] == "LOSS")
        pending = sum(1 for r in backtest_results if r["Result"] == "PENDING")
        
        if total_backtest > 0:
            win_rate_backtest = round(wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", total_backtest)
            with col2:
                st.metric("Wins", wins)
            with col3:
                st.metric("Losses", losses)
            with col4:
                st.metric("Win Rate", f"{win_rate_backtest}%")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🎯 Match Analyzer V13.1")
    st.caption("FORENSIC ENGINE: Direct Implementation of ALL Discoveries from 21+ Completed Matches")

    with st.expander("📖 V13.1 — FORENSIC LOGIC ENGINE", expanded=False):
        st.markdown("""
        **V13.1 DIRECT IMPLEMENTATION OF ALL FORENSIC DISCOVERIES**
        
        ### Discovery 1: Relegation Fight
        - Non-relegation fights: 81.3% winners
        - Relegation fights: -2 penalty
        
        ### Discovery 2: Home Block vs Away Block
        - Mid/Mid: 66.7% HOME wins
        - Europe involved: Unpredictable
        
        ### Discovery 3: Desperate Teams
        - Away Desperate: 40% not lose
        - Home Desperate: Unpredictable
        
        ### Discovery 4: Dead Rubber
        - **+3 bonus** (was +2)
        - 56.3% → 83.3% with other factors
        
        ### Discovery 5: UNDER 2.5 = 90.9% CORRECT
        - STRONGEST signal in dataset
        - Use combined scoring + conceding
        
        ### Discovery 6: Goal Accuracy
        - **95% = PERFECT (4/4) → AUTO FULL STAKE**
        - 100% = STRONG
        - 80% = coin flip → SKIP
        
        ### Discovery 7: Pattern
        - DOUBLE_CHANCE: 77.8% correct
        - SKIP: Useless
        
        ### Discovery 8: Action Labels are Backwards
        - DANGEROUS (62.5%) > SAFE (50%)
        
        ### Discovery 9: Winner Selection
        - DOUBLE CHANCE: 76.2% correct
        
        ### Discovery 10: Failure Patterns
        - Both teams HIGH form → Draw
        - No streaks, no warnings → Draw
        - High point gap + losing streak → Draw
        
        ### Discovery 11: Total Factor Score
        - ≥ 12 = 83.3% correct → FULL STAKE
        - 8-11 = 50% correct → HALF STAKE
        - < 8 = 28.6% correct → SMALL/SKIP
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info("🎯 V13.1: Forensic logic from 21+ completed matches applied directly.")

        st.markdown("""
        <div class="upload-container">
            <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">📋 Paste All Data</p>
            <p style="color: #94a3b8; margin-bottom: 1rem;">Paste the Predictions, Tables, and Form data together</p>
        </div>
        """, unsafe_allow_html=True)

        text_data = st.text_area(
            "Paste all data here", 
            height=400, 
            key="text_paste",
            placeholder="Paste the complete text data (Predictions + HOME TABLE + AWAY TABLE + LAST 6 MATCHES TABLE)..."
        )

        if st.button("🎯 ANALYZE V13.1", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Applying Forensic Logic..."):
                        parsed = parse_text_data(text_data)

                    league = parsed.get("league", "Unknown League")
                    matches = parsed.get("matches", [])
                    home_table = parsed.get("home_table", {})
                    away_table = parsed.get("away_table", {})
                    form_data = parsed.get("form_data", {})
                    league_config = parsed.get("league_config", {})

                    if matches:
                        ft_matches = [m for m in matches if m.get("is_finished")]
                        draw_matches = [m for m in matches if m.get("prediction") == 'X' and not m.get("is_finished")]
                        non_draw_matches = [m for m in matches if m.get("prediction") != 'X' and not m.get("is_finished")]
                        total_matches = len(matches)
                        
                        st.success(f"✅ Found {total_matches} matches in {league}")
                        
                        if ft_matches:
                            st.info(f"⏭️ {len(ft_matches)} matches already played (FT) — skipped")
                        
                        draw_matches_sorted = sorted(draw_matches, key=lambda x: parse_match_date(x.get("date", "")))
                        
                        analyzed_results = []
                        skipped_results = []
                        stored_count = 0
                        already_stored_count = 0
                        
                        for match in draw_matches_sorted:
                            match_with_config = dict(match)
                            match_with_config["league_config"] = league_config
                            data = convert_match_to_data(match_with_config, home_table, away_table, form_data, league)
                            analysis = analyze_draw_match_v131(data)
                            
                            if analysis.get("verdict") != "SKIP":
                                exists = check_match_exists(data.get("home_team"), data.get("away_team"), data.get("date"))
                                
                                if exists:
                                    already_stored_count += 1
                                    analyzed_results.append((match, data, analysis, True))
                                else:
                                    saved_id = save_to_db(data, analysis, league)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, data, analysis, True))
                                    elif saved_id:
                                        stored_count += 1
                                        analyzed_results.append((match, data, analysis, False))
                                    else:
                                        analyzed_results.append((match, data, analysis, False))
                            else:
                                skipped_results.append((match, data, analysis))

                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed (shown below)")

                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🎯 DRAW PREDICTIONS (V13.1 Forensic)")
                            st.caption(f"{len(analyzed_results)} draw predictions analyzed with Forensic Logic (sorted by date)")
                            
                            for idx, (match, data, analysis, already_stored) in enumerate(analyzed_results, 1):
                                action = analysis.get("action", "UNKNOWN")
                                stake = analysis.get("stake", "?")
                                stake_display, _ = get_stake_display(stake)
                                bet_strength = analysis.get("bet_strength", 0)
                                
                                stored_badge = " 📌 ALREADY STORED" if already_stored else " ✅ NEW"
                                
                                if action == "SAFE":
                                    emoji = "✅"
                                    label = "SAFE"
                                elif action == "CAUTIOUS":
                                    emoji = "⚠️"
                                    label = "CAUTIOUS"
                                elif action == "DANGEROUS":
                                    emoji = "❗"
                                    label = "DANGEROUS"
                                else:
                                    emoji = "⏭️"
                                    label = "SKIP"
                                
                                date_display = format_date_display(match.get('date', ''))
                                st.markdown(f"#### {emoji} Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} ({label} — {stake_display}) | Forensic Score: {bet_strength} {stored_badge}")
                                st.caption(f"📅 {date_display}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Prediction", match.get('prediction', '?'))
                                with col2:
                                    st.metric("Correct Score", f"{match.get('correct_score_home', '?')}-{match.get('correct_score_away', '?')}")
                                with col3:
                                    st.metric("Avg Goals", f"{match.get('avg_goals', 0):.2f}")
                                
                                display_analysis(data, analysis, league, already_stored)
                                
                                if idx < len(analyzed_results):
                                    st.markdown("---")
                        
                        if skipped_results:
                            st.markdown("---")
                            st.markdown("### ⏭️ SKIPPED MATCHES (Not Stored)")
                            st.caption(f"{len(skipped_results)} matches skipped (FT or non-draw or Forensic Score < 25)")
                            
                            with st.expander(f"Click to expand {len(skipped_results)} skipped matches"):
                                for idx, (match, data, analysis) in enumerate(skipped_results, 1):
                                    st.markdown(f"#### Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Prediction", match.get('prediction', '?'))
                                    with col2:
                                        st.metric("Correct Score", f"{match.get('correct_score_home', '?')}-{match.get('correct_score_away', '?')}")
                                    with col3:
                                        st.metric("Avg Goals", f"{match.get('avg_goals', 0):.2f}")
                                    
                                    if match.get("is_finished"):
                                        st.info(f"📅 Already played — FT Score: {match.get('actual_home', '?')}-{match.get('actual_away', '?')}")
                                    
                                    display_analysis(data, analysis, league)
                                    
                                    if idx < len(skipped_results):
                                        st.markdown("---")
                        
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Matches", total_matches)
                        with col2:
                            st.metric("🎯 Draws Found", len(draw_matches))
                        with col3:
                            st.metric("💾 New Stored", stored_count)
                        with col4:
                            st.metric("📌 Already Stored", already_stored_count)
                        with col5:
                            st.metric("⏭️ Skipped", len(skipped_results) + len(ft_matches) + len(non_draw_matches))
                            
                    else:
                        st.error("No matches found in the data. Please make sure you're pasting valid data.")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Enter Match Results")
        st.caption("📅 Matches sorted by date (earliest first)")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('bet_market', 'No prediction')
                score = a.get('draw_survival_score', '?')
                action = a.get('action', '?')
                stake = a.get('stake', '?')
                match_date = a.get('match_date', 'Date unknown')
                stake_display, _ = get_stake_display(stake)
                date_display = format_date_display(match_date)

                if action == "SAFE":
                    badge = f"✅ SAFE (Score: {score}) — {stake_display}"
                elif action == "CAUTIOUS":
                    badge = f"⚠️ CAUTIOUS (Score: {score}) — {stake_display}"
                elif action == "DANGEROUS":
                    badge = f"❗ DANGEROUS (Score: {score}) — {stake_display}"
                else:
                    badge = f"⏭️ SKIP (Score: {score}) — {stake_display}"

                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Draw Survival Score: {score} ({action}) — Stake: {stake_display}")
                    st.caption(f"📅 Match Date: {match_date}")
                    c1, c2 = st.columns(2)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    expected_home = a.get('correct_score_home')
                    expected_away = a.get('correct_score_away')
                    if expected_home is not None and expected_away is not None:
                        st.caption(f"📊 Expected Score: {expected_home}-{expected_away}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")

    with tab3:
        st.subheader("📊 Performance Records")
        st.caption("📅 Matches sorted by date (latest first)")
        results = get_results()
        display_records_table(results)

    with tab4:
        display_live_dashboard()


if __name__ == "__main__":
    main()
