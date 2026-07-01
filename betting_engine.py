"""
MATCH ANALYZER V11.3 — STAKE-ADJUSTED DRAW SYSTEM
Always bet DOUBLE CHANCE (12) on every draw prediction.
Stake adjusts based on Draw Survival Score:
- SAFE (≤4): Full stake
- CAUTIOUS (5-7): Half stake
- DANGEROUS (≥8): Small stake or skip
"""

import streamlit as st
from datetime import date
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
st.set_page_config(page_title="Match Analyzer V11.3", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .cautious-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #1a2a00 0%, #0a1a00 100%); }
    .danger-card { border: 3px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .dead-rubber-card { border: 3px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LEAGUE CONFIGURATION
# ============================================================================
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
        
        # Check for FT flag
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
                "losing_streak": losing_streak
            }
    
    return form_data


# ============================================================================
# CONVERT DATA TO ANALYSIS FORMAT
# ============================================================================
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
    if away_team in form_data:
        data["away_form_points"] = form_data[away_team]["form_points"]
        data["away_form_wins"] = form_data[away_team]["wins"]
        data["away_form_draws"] = form_data[away_team]["draws"]
        data["away_form_losses"] = form_data[away_team]["losses"]
        data["away_losing_streak"] = form_data[away_team]["losing_streak"]
    
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
    
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    
    return data


# ============================================================================
# DRAW SURVIVAL SCORE SYSTEM
# ============================================================================
def calculate_draw_survival_score(data: dict) -> dict:
    """
    Calculate the Draw Survival Score using 7 factors.
    Score range: 0-14+
    """
    
    factors = {}
    total_score = 0
    
    # FACTOR 1: TEAM DRAW RATE (HIGH weight)
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
    
    factors["team_draw_rate"] = {
        "value": f"{combined_draw_rate:.1%}",
        "points": f1_points,
        "weight": "HIGH"
    }
    total_score += f1_points
    
    # FACTOR 2: POINTS DIFFERENCE (HIGH weight)
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
    
    # FACTOR 3: GOAL EXPECTANCY (HIGH weight)
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
    
    # FACTOR 4: GD GAP (MEDIUM weight)
    home_gd = data.get("home_gd", 0)
    away_gd = data.get("away_gd", 0)
    gd_gap = abs(home_gd - away_gd)
    
    if gd_gap <= 5:
        f4_points = 2
    elif gd_gap <= 10:
        f4_points = 1
    elif gd_gap <= 15:
        f4_points = 0
    elif gd_gap <= 20:
        f4_points = -1
    else:
        f4_points = -2
    
    factors["gd_gap"] = {
        "value": f"{gd_gap}",
        "points": f4_points,
        "weight": "MEDIUM"
    }
    total_score += f4_points
    
    # FACTOR 5: SCORING RATE (MEDIUM weight)
    home_scoring = data.get("home_gf", 0) / max(data.get("home_gp", 1), 1)
    away_scoring = data.get("away_gf", 0) / max(data.get("away_gp", 1), 1)
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
    
    # FACTOR 6: PROBABILITY BALANCE (LOW weight)
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
    
    # FACTOR 7: DOMINANCE GAP (LOW weight - negative)
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
    
    # DETERMINE DECISION
    if total_score <= 4:
        action = "SAFE"
        stake = "Full"
        color = "#10b981"
        emoji = "✅"
        explanation = f"Draw unlikely to survive (Score: {total_score}) → Full stake"
        recommended_bet = "DOUBLE CHANCE: HOME or AWAY"
        accuracy = "83%"
    elif total_score <= 7:
        action = "CAUTIOUS"
        stake = "Half"
        color = "#f59e0b"
        emoji = "⚠️"
        explanation = f"Draw might survive (Score: {total_score}) → Half stake"
        recommended_bet = "DOUBLE CHANCE: HOME or AWAY"
        accuracy = "70-80%"
    else:
        action = "DANGEROUS"
        stake = "Small or Skip"
        color = "#ef4444"
        emoji = "❗"
        explanation = f"Draw likely to survive (Score: {total_score}) → Small stake or skip"
        recommended_bet = "DOUBLE CHANCE: HOME or AWAY"
        accuracy = "N/A"
    
    # GOAL BET
    goal_bet = None
    goal_accuracy = None
    goal_reason = None
    
    if avg_goals < 2.00:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "95%"
        goal_reason = f"Avg goals {avg_goals:.2f} < 2.00 → UNDER 2.5"
    elif avg_goals > 3.00:
        goal_bet = "OVER 2.5"
        goal_accuracy = "100%"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Draw → OVER 2.5"
    elif 2.00 <= avg_goals <= 2.40:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "80%"
        goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot → UNDER 2.5"
    
    return {
        "total_score": total_score,
        "factors": factors,
        "action": action,
        "stake": stake,
        "color": color,
        "emoji": emoji,
        "explanation": explanation,
        "recommended_bet": recommended_bet,
        "accuracy": accuracy,
        "goal_bet": goal_bet,
        "goal_accuracy": goal_accuracy,
        "goal_reason": goal_reason,
        "draw_pct": draw_pct,
        "avg_goals": avg_goals,
        "points_diff": points_diff,
        "gd_gap": gd_gap
    }


# ============================================================================
# DRAW-FOCUSED ANALYSIS ENGINE
# ============================================================================
def analyze_draw_match(data: dict) -> dict:
    """
    Analyze draw predictions using the Draw Survival Score system.
    ALWAYS bet DOUBLE CHANCE, but adjust stake based on score.
    """
    
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reason": None,
        "is_lock": False,
        "lock_reason": None,
        "draw_conditions": {},
        "winner_selection": None,
        "winner_reason": None,
        "used_priority": None,
        "goal_bet": None,
        "goal_reason": None,
        "goal_accuracy": None,
        "goal_is_lock": False,
        "warning": None,
        "warning_type": None,
        "draw_survival_score": None,
        "factor_breakdown": None,
        "action": None,
        "stake": None,
        "recommended_bet": None
    }
    
    # FILTER 1: Skip FT matches
    if data.get("is_finished"):
        result["verdict"] = "SKIP"
        result["skip_reason"] = f"Already played (FT)"
        result["classification"] = "⏭️ SKIPPED — Already Played"
        return result
    
    # FILTER 2: Only analyze draw predictions
    if data.get("prediction") != 'X':
        result["verdict"] = "SKIP"
        result["skip_reason"] = "Not a draw prediction (X)"
        result["classification"] = "⏭️ SKIPPED — Not a Draw Prediction"
        return result
    
    # Calculate Draw Survival Score
    score_result = calculate_draw_survival_score(data)
    
    result["draw_survival_score"] = score_result["total_score"]
    result["factor_breakdown"] = score_result["factors"]
    result["action"] = score_result["action"]
    result["stake"] = score_result["stake"]
    result["recommended_bet"] = score_result["recommended_bet"]
    
    # Dead rubber detection
    home_block = data.get("home_block")
    away_block = data.get("away_block")
    is_relegation_fight = data.get("is_relegation_fight", False)
    
    is_dead_rubber = False
    if (home_block == "mid" and away_block == "mid" and not is_relegation_fight):
        is_dead_rubber = True
        result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for"
        result["warning_type"] = "dead_rubber"
    
    # BUILD OUTCOME BET — ALWAYS DOUBLE CHANCE (12)
    outcome_bet = "DOUBLE CHANCE: HOME or AWAY"
    outcome_accuracy = score_result["accuracy"]
    outcome_reason = score_result["explanation"]
    
    if score_result["action"] == "SAFE":
        is_lock = True
        lock_reason = f"Score: {score_result['total_score']} (≤4) → SAFE — Full stake"
        result["used_priority"] = "safe"
        result["verdict"] = "LOCK"
    elif score_result["action"] == "CAUTIOUS":
        is_lock = False
        lock_reason = None
        result["used_priority"] = "cautious"
        result["verdict"] = "RECOMMENDED"
    else:  # DANGEROUS
        is_lock = False
        lock_reason = None
        result["used_priority"] = "dangerous"
        result["verdict"] = "RECOMMENDED"
    
    result["winner_selection"] = outcome_bet
    result["winner_reason"] = outcome_reason
    result["is_lock"] = is_lock
    result["lock_reason"] = lock_reason
    
    # GOAL BET
    goal_bet = score_result.get("goal_bet")
    goal_accuracy = score_result.get("goal_accuracy")
    goal_reason = score_result.get("goal_reason")
    
    result["goal_bet"] = goal_bet
    result["goal_reason"] = goal_reason
    result["goal_accuracy"] = goal_accuracy
    result["goal_is_lock"] = goal_bet is not None
    
    # BUILD FINAL RESULT
    result["primary_bet"] = {
        "outcome_bet": outcome_bet,
        "outcome_accuracy": outcome_accuracy,
        "outcome_reason": outcome_reason,
        "goal_bet": goal_bet,
        "goal_accuracy": goal_accuracy,
        "goal_reason": goal_reason,
        "is_lock": is_lock,
        "lock_reason": lock_reason,
        "score": score_result["total_score"],
        "factors": score_result["factors"],
        "action": score_result["action"],
        "stake": score_result["stake"]
    }
    
    classification = outcome_bet
    if goal_bet:
        classification += f" | {goal_bet}"
    result["classification"] = classification
    
    return result


# ============================================================================
# EVALUATION ENGINE — FIXED
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    """
    Evaluate if a bet was correct.
    FIXED: DOUBLE CHANCE: HOME or AWAY wins if NOT a draw.
    """
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID DATA", "message": "Non-numeric score"}
    
    total = home + away
    pred = primary_pred.strip().upper()
    
    if ' | ' in pred:
        bets = pred.split(' | ')
    else:
        bets = [pred]
    
    correct_count = 0
    total_bets = 0
    
    for bet in bets:
        bet = bet.strip()
        total_bets += 1
        
        if 'DRAW' in bet and not 'DOUBLE' in bet:
            # Exact Draw bet
            if home == away:
                correct_count += 1
                
        elif 'DOUBLE CHANCE' in bet or 'DOUBLE CHANCE:' in bet:
            # Double Chance bet
            if 'HOME' in bet and 'AWAY' in bet:
                # DOUBLE CHANCE: HOME or AWAY → wins if NOT a draw
                if home != away:
                    correct_count += 1
            elif 'HOME' in bet or '1' in bet:
                # DOUBLE CHANCE: HOME or DRAW → wins if home wins or draw
                if home >= away:
                    correct_count += 1
            elif 'AWAY' in bet or '2' in bet:
                # DOUBLE CHANCE: AWAY or DRAW → wins if away wins or draw
                if away >= home:
                    correct_count += 1
            else:
                # Fallback
                if home >= away or away >= home:
                    correct_count += 1
                    
        elif 'OVER 2.5' in bet:
            if total > 2:
                correct_count += 1
                
        elif 'UNDER 2.5' in bet:
            if total <= 2:
                correct_count += 1
                
        elif 'HOME WIN' in bet or ('HOME' in bet and not 'DOUBLE' in bet):
            if home > away:
                correct_count += 1
                
        elif 'AWAY WIN' in bet or ('AWAY' in bet and not 'DOUBLE' in bet):
            if away > home:
                correct_count += 1
    
    is_correct = correct_count == total_bets if total_bets > 0 else False
    
    return {
        "is_correct": is_correct,
        "actual": f"{home}-{away}",
        "message": f"{'✅' if is_correct else '❌'} {pred} vs {home}-{away}",
        "correct_count": correct_count,
        "total_bets": total_bets
    }


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(data: dict, analysis: dict, league: str = "Unknown"):
    """
    Save ALL draw predictions (SAFE, CAUTIOUS, DANGEROUS) to Supabase.
    Only FT and non-draw matches are skipped.
    """
    
    is_draw_prediction = data.get("prediction") == 'X'
    is_ft = data.get("is_finished", False)
    
    # Skip FT and non-draw matches ONLY
    if is_ft or not is_draw_prediction:
        return None
    
    try:
        primary = analysis.get("primary_bet", {})
        score_result = analysis.get("draw_survival_score", 0)
        
        record = {
            "home_team": data.get("home_team", "Unknown"),
            "away_team": data.get("away_team", "Unknown"),
            "match_date": data.get("date", str(date.today())),
            "league": league,
            "home_pct": data.get("home_pct"),
            "draw_pct": data.get("draw_pct"),
            "away_pct": data.get("away_pct"),
            "prediction": data.get("prediction"),
            "avg_goals": data.get("avg_goals"),
            "temperature": data.get("temperature"),
            "coefficient": data.get("coefficient"),
            "home_position": data.get("home_position"),
            "away_position": data.get("away_position"),
            "home_points": data.get("home_points"),
            "away_points": data.get("away_points"),
            "home_block": data.get("home_block"),
            "away_block": data.get("away_block"),
            "is_relegation_fight": data.get("is_relegation_fight", False),
            "home_form_points": data.get("home_form_points"),
            "away_form_points": data.get("away_form_points"),
            "home_losing_streak": data.get("home_losing_streak", 0),
            "away_losing_streak": data.get("away_losing_streak", 0),
            "draw_survival_score": score_result,
            "action": analysis.get("action"),
            "stake": analysis.get("stake"),
            "recommended_bet": analysis.get("recommended_bet"),
            "factor_breakdown": json.dumps(analysis.get("factor_breakdown", {})),
            "bet_market": analysis.get("classification", "SKIP"),
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "DRAW" if "DRAW" in analysis.get("classification", "") else "DOUBLE_CHANCE" if "DOUBLE" in analysis.get("classification", "") else "WIN" if "WIN" in analysis.get("classification", "") else "SKIP",
            "verdict": analysis.get("verdict", "SKIP"),
            "is_lock": analysis.get("is_lock", False),
            "lock_reason": analysis.get("lock_reason"),
            "winner_selection": analysis.get("winner_selection"),
            "winner_reason": analysis.get("winner_reason"),
            "goal_bet": analysis.get("goal_bet"),
            "goal_accuracy": analysis.get("goal_accuracy"),
            "warning": analysis.get("warning"),
            "warning_type": analysis.get("warning_type"),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            "draw_conditions": json.dumps(analysis.get("draw_conditions", {})),
            "result_entered": False,
            "actual_home": None,
            "actual_away": None,
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
    except: return []


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
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False


def get_results():
    try:
        response = supabase.table("match_analyses").select("*").eq("result_entered", True).order("match_date", desc=True).execute()
        return response.data if response.data else []
    except: return []


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
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


def display_score_breakdown(factors: dict, total_score: int):
    """Display the Draw Survival Score breakdown."""
    
    if total_score <= 4:
        color = "#10b981"
        status = "SAFE"
        emoji = "✅"
    elif total_score <= 7:
        color = "#f59e0b"
        status = "CAUTIOUS"
        emoji = "⚠️"
    else:
        color = "#ef4444"
        status = "DANGEROUS"
        emoji = "❗"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 1.5rem; margin: 0.5rem 0; border-left: 4px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">DRAW SURVIVAL SCORE</div>
                <div style="font-size: 3rem; font-weight: 800; color: {color};">{total_score}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 1.2rem; font-weight: 700; color: {color};">{emoji} {status}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">Score ≤ 4 = SAFE<br>Score 5-7 = CAUTIOUS<br>Score ≥ 8 = DANGEROUS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display factor breakdown
    st.markdown("#### Factor Breakdown")
    
    for name, factor in factors.items():
        weight = factor.get("weight", "")
        points = factor.get("points", 0)
        value = factor.get("value", "")
        
        if points >= 0:
            points_display = f"+{points}"
        else:
            points_display = str(points)
        
        st.markdown(f"""
        <div class="factor-row">
            <span class="factor-name">{name.replace('_', ' ').title()} <span style="color:#64748b; font-size:0.7rem;">({weight})</span></span>
            <span class="factor-value">{value}</span>
            <span class="factor-points">{points_display}</span>
        </div>
        """, unsafe_allow_html=True)


def display_analysis(data: dict, analysis: dict, league: str = "Unknown"):
    """Display analysis results for a single match."""
    
    # Check if skipped (FT or non-draw)
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
    
    # ================================================================
    # DISPLAY ANALYZED MATCH
    # ================================================================
    
    badge_class = get_league_badge(league)
    st.markdown(f'<span class="league-badge {badge_class}">{league}</span>', unsafe_allow_html=True)
    
    if analysis.get("warning"):
        st.markdown(f'<div class="dead-rubber-warning">{analysis["warning"]}</div>', unsafe_allow_html=True)
    
    # Draw Survival Score
    display_score_breakdown(
        analysis.get("factor_breakdown", {}),
        analysis.get("draw_survival_score", 0)
    )
    
    st.markdown("---")
    
    # OUTCOME BET
    primary = analysis.get("primary_bet", {})
    action = primary.get("action", "UNKNOWN")
    
    if action == "SAFE":
        card_class = "lock-card"
        lock_icon = "🔒"
        lock_text = "SAFE — Full Stake"
        border_color = "#f59e0b"
    elif action == "CAUTIOUS":
        card_class = "cautious-card"
        lock_icon = "⚠️"
        lock_text = "CAUTIOUS — Half Stake"
        border_color = "#f59e0b"
    else:  # DANGEROUS
        card_class = "danger-card"
        lock_icon = "❗"
        lock_text = "DANGEROUS — Small Stake or Skip"
        border_color = "#ef4444"
    
    st.markdown(f"""
    <div class="output-card {card_class}" style="border-left: 4px solid {border_color};">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{lock_icon}</span>
            <span style="font-size: 1.2rem; font-weight: 700;">OUTCOME BET — {lock_text}</span>
        </div>
        <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 0.25rem;">{primary.get('outcome_bet', 'Unknown')}</div>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem;">
            <span>📊 Accuracy: {primary.get('outcome_accuracy', 'N/A')}</span>
            <span>📝 Reason: {primary.get('outcome_reason', 'N/A')}</span>
            <span style="background: {border_color}; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">Stake: {primary.get('stake', '?')}</span>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.8rem; color: #64748b;">
            Score: {primary.get('score', '?')} → {action}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # GOAL BET
    goal_bet = primary.get('goal_bet')
    
    if goal_bet:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #10b981;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">⚽</span>
                <span style="font-size: 1.2rem; font-weight: 700;">GOAL BET</span>
            </div>
            <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 0.25rem;">{goal_bet}</div>
            <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem;">
                <span>📊 Accuracy: {primary.get('goal_accuracy', 'N/A')}</span>
                <span>📝 Reason: {primary.get('goal_reason', 'N/A')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <hr style="border: none; border-top: 1px dashed #475569; margin: 0.5rem 0;">
        <div style="background: #1a2a1a; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #2a4a2a; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.9rem;">📌 These are </span>
            <span style="color: #fbbf24; font-weight: 700; font-size: 0.9rem;">SEPARATE</span>
            <span style="color: #94a3b8; font-size: 0.9rem;"> bets. Place them individually.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #64748b;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
                <span style="font-size: 1.5rem;">⚽</span>
                <span style="font-size: 1.2rem; font-weight: 700;">GOAL BET</span>
            </div>
            <div style="font-size: 1rem; color: #94a3b8;">None available</div>
        </div>
        """, unsafe_allow_html=True)
    
    # KEY METRICS
    st.markdown("### 📊 Key Metrics")
    
    avg_goals = data.get("avg_goals", 2.0)
    home_form = data.get("home_form_points", "?")
    away_form = data.get("away_form_points", "?")
    home_pos = data.get("home_position", "?")
    away_pos = data.get("away_position", "?")
    home_block = data.get("home_block", "?")
    away_block = data.get("away_block", "?")
    home_streak = data.get("home_losing_streak", 0)
    away_streak = data.get("away_losing_streak", 0)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_form} vs {away_form}</div><div class="metric-label">Form Points</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_goals:.2f}</div><div class="metric-label">Avg Goals</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_block} vs {away_block}</div><div class="metric-label">Competitive Block</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_pos} vs {away_pos}</div><div class="metric-label">Standings</div></div>', unsafe_allow_html=True)
    with m5:
        desperate = "Yes" if (home_streak >= 3 or away_streak >= 3 or data.get("is_relegation_fight", False)) else "No"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{desperate}</div><div class="metric-label">Desperation</div></div>', unsafe_allow_html=True)
    
    if home_streak >= 3 or away_streak >= 3:
        streak_msg = []
        if home_streak >= 3:
            streak_msg.append(f"🔴 {data.get('home_team', 'Home')}: {home_streak} game losing streak")
        if away_streak >= 3:
            streak_msg.append(f"🔴 {data.get('away_team', 'Away')}: {away_streak} game losing streak")
        st.warning(" | ".join(streak_msg))
    
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
# MAIN APP
# ============================================================================
def main():
    st.title("🎯 Match Analyzer V11.3")
    st.caption("Stake-Adjusted Draw System | Always Bet DOUBLE CHANCE")

    with st.expander("📖 The Stake-Adjusted Draw System", expanded=False):
        st.markdown("""
        **ONLY analyzes matches where Forebet predicts DRAW (X).**
        **ALWAYS bet DOUBLE CHANCE (12), but adjust stake based on risk.**
        
        | Score | Action | Stake | Accuracy |
        |-------|--------|-------|----------|
        | **≤ 4** | ✅ SAFE | **Full** | 83% |
        | **5-7** | ⚠️ CAUTIOUS | **Half** | 70-80% |
        | **≥ 8** | ❗ DANGEROUS | **Small or Skip** | N/A |
        
        **7 Factors Analyzed:**
        
        | Factor | Weight | What It Measures |
        |--------|--------|------------------|
        | **1. Team Draw Rate** | HIGH | How often each team draws |
        | **2. Points Difference** | HIGH | How evenly matched teams are |
        | **3. Goal Expectancy** | HIGH | Low-scoring games = more draws |
        | **4. GD Gap** | MEDIUM | Balanced attack/defense = draws |
        | **5. Scoring Rate** | MEDIUM | Low scoring = more draws |
        | **6. Probability Balance** | LOW | Balanced probabilities = draws |
        | **7. Dominance Gap** | LOW (negative) | Dominance kills draws |
        
        **Goal Bets (Secondary):**
        - Avg Goals < 2.00 → UNDER 2.5 (95%)
        - Avg Goals > 3.00 → OVER 2.5 (100%)
        - Avg Goals 2.00-2.40 → UNDER 2.5 (80%)
        """)

    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info("🎯 ALL draw predictions are analyzed. Stake adjusts based on Draw Survival Score.")

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

        if st.button("🎯 ANALYZE V11.3", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Calculating Draw Survival Scores..."):
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
                        
                        analyzed_results = []
                        skipped_results = []
                        stored_count = 0
                        
                        for match in matches:
                            match_with_config = dict(match)
                            match_with_config["league_config"] = league_config
                            data = convert_match_to_data(match_with_config, home_table, away_table, form_data, league)
                            analysis = analyze_draw_match(data)
                            
                            if analysis.get("verdict") != "SKIP":
                                saved_id = save_to_db(data, analysis, league)
                                if saved_id:
                                    stored_count += 1
                                analyzed_results.append((match, data, analysis))
                            else:
                                skipped_results.append((match, data, analysis))

                        st.info(f"💾 {stored_count} draw predictions stored in Supabase")

                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🎯 DRAW PREDICTIONS (Stored)")
                            st.caption(f"{len(analyzed_results)} draw predictions analyzed with stake adjustment")
                            
                            for idx, (match, data, analysis) in enumerate(analyzed_results, 1):
                                action = analysis.get("action", "UNKNOWN")
                                stake = analysis.get("stake", "?")
                                
                                if action == "SAFE":
                                    emoji = "✅"
                                    label = "SAFE"
                                elif action == "CAUTIOUS":
                                    emoji = "⚠️"
                                    label = "CAUTIOUS"
                                else:
                                    emoji = "❗"
                                    label = "DANGEROUS"
                                
                                st.markdown(f"#### {emoji} Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} ({label} — {stake} Stake)")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Prediction", match.get('prediction', '?'))
                                with col2:
                                    st.metric("Correct Score", f"{match.get('correct_score_home', '?')}-{match.get('correct_score_away', '?')}")
                                with col3:
                                    st.metric("Avg Goals", f"{match.get('avg_goals', 0):.2f}")
                                
                                display_analysis(data, analysis, league)
                                
                                if idx < len(analyzed_results):
                                    st.markdown("---")
                        
                        if skipped_results:
                            st.markdown("---")
                            st.markdown("### ⏭️ SKIPPED MATCHES (Not Stored)")
                            st.caption(f"{len(skipped_results)} matches skipped (FT or non-draw)")
                            
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
                            st.metric("🎯 Draws Stored", stored_count)
                        with col3:
                            st.metric("⏭️ FT (Played)", len(ft_matches))
                        with col4:
                            st.metric("⏭️ Non-Draws", len(non_draw_matches))
                        with col5:
                            st.metric("🎯 Draws Found", len(draw_matches))
                            
                    else:
                        st.error("No matches found in the data. Please make sure you're pasting valid data.")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Enter Match Results")
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

                if action == "SAFE":
                    badge = f"✅ SAFE (Score: {score}) — {stake} Stake"
                elif action == "CAUTIOUS":
                    badge = f"⚠️ CAUTIOUS (Score: {score}) — {stake} Stake"
                else:
                    badge = f"❗ DANGEROUS (Score: {score}) — {stake} Stake"

                with st.expander(f"{badge} | {ht} vs {at} — Predicted: {pred}"):
                    st.info(f"📊 Draw Survival Score: {score} ({action}) — Stake: {stake}")
                    c1, c2 = st.columns(2)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
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
                scores.append(score)
                if score <= 4:
                    safe_scores.append(score)
                elif score <= 7:
                    cautious_scores.append(score)
                else:
                    dangerous_scores.append(score)

            for r in results:
                pred = r.get('bet_market', '')
                if pred == 'SKIP':
                    continue

                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))

                if evaluation["is_correct"]:
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
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Incorrect</div></div>', unsafe_allow_html=True)
            with col5:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{len(safe_scores)}</div><div class="stat-label">SAFE Bets</div></div>', unsafe_allow_html=True)

            st.markdown("### 📊 Draw Survival Score Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ SAFE (≤4)", len(safe_scores))
            with col2:
                st.metric("⚠️ CAUTIOUS (5-7)", len(cautious_scores))
            with col3:
                st.metric("❗ DANGEROUS (≥8)", len(dangerous_scores))

            if lock_total > 0:
                lock_rate = round(lock_correct / lock_total * 100) if lock_total > 0 else 0
                st.markdown(f"🔒 **Lock Signals:** {lock_correct}/{lock_total} correct ({lock_rate}%)")

            st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")

            rows = []
            for r in results:
                pred = r.get('bet_market', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                league = r.get('league', '')
                badge_class = get_league_badge(league)
                score = r.get('draw_survival_score', '?')
                action = r.get('action', '?')
                stake = r.get('stake', '?')

                evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"

                match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
                score_label = f"{score} ({action}) — {stake}" if score != '?' else '?'

                rows.append({
                    "Date": r.get("match_date", ""),
                    "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
                    "Match": match_display,
                    "Score": score_label,
                    "Bet": primary_pred,
                    "Actual": score_display,
                    "Result": badge,
                })

            df = pd.DataFrame(rows)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
