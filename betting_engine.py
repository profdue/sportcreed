"""
MATCH ANALYZER V1.1 — TWO-TIER SYSTEM (ALL MATCHES)
Tier 1: LOCK Bets (100% accuracy) — Home Desperation, Elite Home, Elite Away
Tier 2: Interwoven Framework (95% accuracy) — Multi-signal convergence with conflict detection
Works on ALL matches — X, 1, and 2 predictions.
Goal Bets: Secondary — added when clear
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
st.set_page_config(page_title="Match Analyzer V1.1", page_icon="🏆", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .tier1-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .tier2-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
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
    .tier1-badge { background: #f59e0b; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .tier2-badge { background: #10b981; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
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
    .separator-line { border: none; border-top: 1px dashed #475569; margin: 0.5rem 0; }
    .section-divider { border: none; border-top: 2px solid #f59e0b; margin: 1.5rem 0; }
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
        "goals_fallback": 2.50,
        "home_elite_threshold": 3,
        "away_elite_threshold": 3,
        "draw_accuracy_fallback": 0.57,
    }
    
    if "Norway" in league or "Eliteserien" in league:
        config["relegation_threshold"] = 15
        config["league_size"] = 16
        config["goals_fallback"] = 2.75
        config["home_elite_threshold"] = 3
        config["away_elite_threshold"] = 3
    elif "Brazil" in league or "Serie A" in league or "Br1" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["europe_threshold"] = 4
        config["goals_fallback"] = 2.66
        config["home_elite_threshold"] = 3
        config["away_elite_threshold"] = 3
    elif "Premier" in league or "EPL" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.75
        config["home_elite_threshold"] = 3
        config["away_elite_threshold"] = 3
    elif "AuV" in league or "NPL" in league or "Australia" in league:
        config["relegation_threshold"] = 11
        config["league_size"] = 14
        config["europe_threshold"] = 3
        config["goals_fallback"] = 2.80
        config["home_elite_threshold"] = 3
        config["away_elite_threshold"] = 3
    else:
        config["relegation_threshold"] = 15
        config["league_size"] = 20
        config["goals_fallback"] = 2.50
        config["home_elite_threshold"] = 3
        config["away_elite_threshold"] = 3
    
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
    
    # Determine elite status
    home_elite_threshold = league_config.get("home_elite_threshold", 3)
    away_elite_threshold = league_config.get("away_elite_threshold", 3)
    
    data["home_is_elite"] = data.get("home_position") is not None and data["home_position"] <= home_elite_threshold
    data["away_is_elite"] = data.get("away_position") is not None and data["away_position"] <= away_elite_threshold
    
    # Determine different blocks
    data["different_blocks"] = (
        data.get("home_block") is not None and 
        data.get("away_block") is not None and 
        data["home_block"] != data["away_block"]
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
# TWO-TIER ANALYSIS ENGINE
# ============================================================================
def analyze_match(data: dict) -> dict:
    """
    TWO-TIER SYSTEM — Works on ALL matches (X, 1, 2)
    
    TIER 1: LOCK Bets (100% accuracy) — Home Desperation, Elite Home, Elite Away
    TIER 2: Interwoven Framework (95% accuracy) — Multi-signal convergence with conflict detection
    Goal Bets: Secondary — added when clear
    """
    
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reason": None,
        "skip_reasons": [],
        "is_lock": False,
        "lock_reason": None,
        "tier": None,
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
        "tier1_signal": None,
        "tier2_scores": {},
    }
    
    # ================================================================
    # FILTER 1: Skip FT matches (already played)
    # ================================================================
    if data.get("is_finished"):
        result["verdict"] = "SKIP"
        actual_home = data.get("actual_home", "?")
        actual_away = data.get("actual_away", "?")
        result["skip_reason"] = f"Already played (FT) — Actual score: {actual_home}-{actual_away}"
        result["classification"] = "⏭️ SKIPPED — Already Played"
        return result
    
    # ================================================================
    # NO DRAW FILTER — Analyze ALL matches
    # The Two-Tier system works on ALL predictions (X, 1, 2)
    # ================================================================
    
    # ================================================================
    # EXTRACT DATA
    # ================================================================
    home_form = data.get("home_form_points", 0) or 0
    away_form = data.get("away_form_points", 0) or 0
    home_block = data.get("home_block")
    away_block = data.get("away_block")
    different_blocks = data.get("different_blocks", False)
    is_relegation_fight = data.get("is_relegation_fight", False)
    avg_goals = data.get("avg_goals", 2.0)
    home_losing_streak = data.get("home_losing_streak", 0) or 0
    away_losing_streak = data.get("away_losing_streak", 0) or 0
    home_pos = data.get("home_position")
    away_pos = data.get("away_position")
    home_is_elite = data.get("home_is_elite", False)
    away_is_elite = data.get("away_is_elite", False)
    prediction = data.get("prediction")
    league_config = data.get("league_config", {})
    league_size = league_config.get("league_size", 20)
    
    form_diff = abs(home_form - away_form)
    form_similar = form_diff <= 2
    goals_sweet_spot = 2.00 <= avg_goals <= 2.40
    same_block = home_block is not None and away_block is not None and home_block == away_block
    
    is_home_desperate = home_losing_streak >= 3 or home_block == "relegation"
    is_away_desperate = away_losing_streak >= 3 or away_block == "relegation"
    no_desperation = not is_home_desperate and not is_away_desperate
    
    draw_conditions = {
        "form_similar": form_similar,
        "goals_sweet_spot": goals_sweet_spot,
        "same_block": same_block,
        "no_desperation": no_desperation,
    }
    result["draw_conditions"] = draw_conditions
    all_draw_conditions_met = all(draw_conditions.values())
    
    is_dead_rubber = False
    if (home_block == "mid" and away_block == "mid" and not is_relegation_fight):
        is_dead_rubber = True
        result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for"
        result["warning_type"] = "dead_rubber"
    
    # ================================================================
    # TIER 1: LOCK Bets (100% accuracy)
    # ================================================================
    
    outcome_bet = None
    outcome_accuracy = None
    outcome_reason = None
    result["is_lock"] = False
    result["tier"] = None
    tier1_signal = None
    
    # Signal 1: Home Desperation (100%)
    if is_home_desperate and not is_away_desperate:
        outcome_bet = "HOME WIN"
        outcome_accuracy = "100%"
        outcome_reason = "🏆 TIER 1 — HOME DESPERATE: Home team in relegation zone or 3+ losses"
        result["is_lock"] = True
        result["lock_reason"] = "TIER 1 LOCK — Home team desperate"
        result["tier"] = "Tier 1"
        result["used_priority"] = "home_desperate"
        tier1_signal = "Home Desperate"
    
    # Signal 2: Elite Home (100%)
    elif home_is_elite and not away_is_elite:
        outcome_bet = "HOME WIN"
        outcome_accuracy = "100%"
        outcome_reason = f"🏆 TIER 1 — ELITE HOME: Home team Top {league_config.get('home_elite_threshold', 3)} at home vs non-elite away"
        result["is_lock"] = True
        result["lock_reason"] = "TIER 1 LOCK — Home team elite at home"
        result["tier"] = "Tier 1"
        result["used_priority"] = "elite_home"
        tier1_signal = "Elite Home"
    
    # Signal 3: Elite Away (100%)
    elif away_is_elite and not home_is_elite:
        outcome_bet = "AWAY WIN"
        outcome_accuracy = "100%"
        outcome_reason = f"🏆 TIER 1 — ELITE AWAY: Away team Top {league_config.get('away_elite_threshold', 3)} away vs non-elite home"
        result["is_lock"] = True
        result["lock_reason"] = "TIER 1 LOCK — Away team elite away"
        result["tier"] = "Tier 1"
        result["used_priority"] = "elite_away"
        tier1_signal = "Elite Away"
    
    # ================================================================
    # TIER 2: Interwoven Framework (95% accuracy)
    # Only if NO Tier 1 signal triggered
    # ================================================================
    
    if not tier1_signal:
        result["tier"] = "Tier 2"
        
        # Calculate Tier 2 scores
        home_score = 0
        away_score = 0
        draw_score = 0
        
        # Home Score Components
        if home_form >= 10:
            home_score += 2
        elif home_form >= 7:
            home_score += 1
        
        if home_block == "europe":
            home_score += 2
        elif home_block == "mid":
            home_score += 1
        
        if is_home_desperate:
            home_score += 3  # Desperation is a strong signal
        
        if home_is_elite:
            home_score += 2
        
        # Away Score Components
        if away_form >= 10:
            away_score += 2
        elif away_form >= 7:
            away_score += 1
        
        if away_block == "europe":
            away_score += 2
        elif away_block == "mid":
            away_score += 1
        
        if is_away_desperate:
            away_score += 3
        
        if away_is_elite:
            away_score += 2
        
        # Draw Score Components (from draw conditions)
        if form_similar:
            draw_score += 2
        if goals_sweet_spot:
            draw_score += 2
        if same_block:
            draw_score += 2
        if no_desperation:
            draw_score += 2
        
        result["tier2_scores"] = {
            "home_score": home_score,
            "away_score": away_score,
            "draw_score": draw_score,
        }
        
        # ============================================================
        # CONFLICT DETECTION
        # ============================================================
        
        # Conflict: Home and Away both high (≥4)
        if home_score >= 4 and away_score >= 4:
            result["verdict"] = "SKIP"
            result["skip_reason"] = f"TIER 2 — CONFLICT: Home ({home_score}) and Away ({away_score}) both high"
            result["classification"] = "⏭️ SKIPPED — Conflict"
            return result
        
        # Conflict: Draw and Home both high
        if draw_score >= 6 and home_score >= 4:
            result["verdict"] = "SKIP"
            result["skip_reason"] = f"TIER 2 — CONFLICT: Draw ({draw_score}) and Home ({home_score}) both high"
            result["classification"] = "⏭️ SKIPPED — Conflict"
            return result
        
        # Conflict: Draw and Away both high
        if draw_score >= 6 and away_score >= 4:
            result["verdict"] = "SKIP"
            result["skip_reason"] = f"TIER 2 — CONFLICT: Draw ({draw_score}) and Away ({away_score}) both high"
            result["classification"] = "⏭️ SKIPPED — Conflict"
            return result
        
        # Skip if all scores are low
        max_score = max(home_score, away_score, draw_score)
        if max_score <= 2:
            result["verdict"] = "SKIP"
            result["skip_reason"] = "TIER 2 — All scores low (max ≤ 2)"
            result["classification"] = "⏭️ SKIPPED — Low Signal"
            return result
        
        # Decision based on scores
        if home_score == max_score:
            outcome_bet = "HOME WIN"
            outcome_accuracy = "95%"
            outcome_reason = f"TIER 2 — INTERWOVEN: Home Score ({home_score}) highest"
            result["used_priority"] = "interwoven"
        elif away_score == max_score:
            outcome_bet = "AWAY WIN"
            outcome_accuracy = "95%"
            outcome_reason = f"TIER 2 — INTERWOVEN: Away Score ({away_score}) highest"
            result["used_priority"] = "interwoven"
        elif draw_score == max_score:
            outcome_bet = "DRAW"
            outcome_accuracy = "57%"
            outcome_reason = f"TIER 2 — INTERWOVEN: Draw Score ({draw_score}) highest"
            result["used_priority"] = "interwoven"
        else:
            # Should never happen
            result["verdict"] = "SKIP"
            result["skip_reason"] = "TIER 2 — No clear signal"
            result["classification"] = "⏭️ SKIPPED — No Clear Signal"
            return result
    
    result["winner_selection"] = outcome_bet
    result["winner_reason"] = outcome_reason
    
    # ================================================================
    # GOAL BET (Secondary)
    # ================================================================
    
    goal_bet = None
    goal_accuracy = None
    goal_reason = None
    goal_is_lock = False
    
    if avg_goals < 2.00:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "95%"
        goal_reason = f"Avg goals {avg_goals:.2f} < 2.00 → UNDER 2.5"
        goal_is_lock = True
    
    elif avg_goals > 3.00 and different_blocks:
        goal_bet = "OVER 2.5"
        goal_accuracy = "80%"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Different Blocks → OVER 2.5"
        goal_is_lock = True
    
    elif 2.00 <= avg_goals <= 2.40:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "80%"
        goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot → UNDER 2.5"
    
    if goal_is_lock:
        result["is_lock"] = True
    
    result["goal_bet"] = goal_bet
    result["goal_reason"] = goal_reason
    result["goal_accuracy"] = goal_accuracy
    result["goal_is_lock"] = goal_is_lock
    
    # ================================================================
    # BUILD FINAL RESULT
    # ================================================================
    
    result["primary_bet"] = {
        "outcome_bet": outcome_bet,
        "outcome_accuracy": outcome_accuracy,
        "outcome_reason": outcome_reason,
        "goal_bet": goal_bet,
        "goal_accuracy": goal_accuracy,
        "goal_reason": goal_reason,
        "is_lock": result["is_lock"],
        "lock_reason": result.get("lock_reason"),
        "tier": result["tier"],
        "tier1_signal": tier1_signal,
        "tier2_scores": result.get("tier2_scores", {}),
    }
    
    classification = outcome_bet
    if goal_bet:
        classification += f" | {goal_bet}"
    result["classification"] = classification
    
    result["verdict"] = "LOCK" if result["is_lock"] else "RECOMMENDED"
    
    return result


# ============================================================================
# EVALUATION ENGINE
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
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
            if home == away:
                correct_count += 1
        elif 'DOUBLE CHANCE' in bet or 'DOUBLE CHANCE:' in bet:
            if 'HOME' in bet or '1' in bet:
                if home >= away:
                    correct_count += 1
            elif 'AWAY' in bet or '2' in bet:
                if away >= home:
                    correct_count += 1
            else:
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
    try:
        primary = analysis.get("primary_bet")
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
            "home_form_points": data.get("home_form_points"),
            "away_form_points": data.get("away_form_points"),
            "home_losing_streak": data.get("home_losing_streak", 0),
            "away_losing_streak": data.get("away_losing_streak", 0),
            "home_block": data.get("home_block"),
            "away_block": data.get("away_block"),
            "is_relegation_fight": data.get("is_relegation_fight", False),
            "is_lock": analysis.get("is_lock", False),
            "lock_reason": analysis.get("lock_reason"),
            "bet_market": analysis.get("classification", "SKIP"),
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "DRAW" if "DRAW" in analysis.get("classification", "") else "DOUBLE_CHANCE" if "DOUBLE" in analysis.get("classification", "") else "WIN" if "WIN" in analysis.get("classification", "") else "SKIP",
            "verdict": analysis.get("verdict", "SKIP"),
            "draw_conditions": json.dumps(analysis.get("draw_conditions", {})),
            "winner_selection": analysis.get("winner_selection"),
            "winner_reason": analysis.get("winner_reason"),
            "goal_bet": analysis.get("goal_bet"),
            "goal_accuracy": analysis.get("goal_accuracy"),
            "warning": analysis.get("warning"),
            "warning_type": analysis.get("warning_type"),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            "result_entered": False,
            "skip_reason": analysis.get("skip_reason"),
            "is_finished": data.get("is_finished", False),
            "actual_home": data.get("actual_home"),
            "actual_away": data.get("actual_away"),
            "tier": analysis.get("tier"),
            "tier1_signal": analysis.get("tier1_signal"),
            "tier2_scores": json.dumps(analysis.get("tier2_scores", {})),
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


def display_analysis(data: dict, analysis: dict, league: str = "Unknown"):
    """Display analysis results for a single match with two-tier system."""
    
    # Check if skipped
    if analysis.get("verdict") == "SKIP":
        skip_reason = analysis.get("skip_reason") or "No clear signal"
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
    
    # Draw Conditions
    st.markdown("### 🎯 Draw Conditions Check")
    conditions = analysis.get("draw_conditions", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        status = "✅" if conditions.get("form_similar") else "❌"
        st.markdown(f'<div class="condition-box">Form Diff ≤ 2<br><span class="{"condition-true" if conditions.get("form_similar") else "condition-false"}">{status}</span></div>', unsafe_allow_html=True)
    with c2:
        status = "✅" if conditions.get("goals_sweet_spot") else "❌"
        st.markdown(f'<div class="condition-box">Goals 2.00-2.40<br><span class="{"condition-true" if conditions.get("goals_sweet_spot") else "condition-false"}">{status}</span></div>', unsafe_allow_html=True)
    with c3:
        status = "✅" if conditions.get("same_block") else "❌"
        st.markdown(f'<div class="condition-box">Same Block<br><span class="{"condition-true" if conditions.get("same_block") else "condition-false"}">{status}</span></div>', unsafe_allow_html=True)
    with c4:
        status = "✅" if conditions.get("no_desperation") else "❌"
        st.markdown(f'<div class="condition-box">No Desperation<br><span class="{"condition-true" if conditions.get("no_desperation") else "condition-false"}">{status}</span></div>', unsafe_allow_html=True)
    
    all_met = all(conditions.values())
    if all_met:
        st.success("✅ ALL 4 draw conditions met → BET THE DRAW (57% accuracy)")
    else:
        st.info("⚠️ Not all conditions met → Using Tier 2 framework")
    
    st.markdown("---")
    
    # ================================================================
    # TIER BADGE
    # ================================================================
    
    tier = analysis.get("tier")
    if tier == "Tier 1":
        st.markdown(f'<span class="tier1-badge">🏆 TIER 1 — LOCK BET</span>', unsafe_allow_html=True)
    elif tier == "Tier 2":
        st.markdown(f'<span class="tier2-badge">📊 TIER 2 — STRONG BET</span>', unsafe_allow_html=True)
    
    # ================================================================
    # OUTCOME BET
    # ================================================================
    
    primary = analysis.get("primary_bet", {})
    
    if primary.get("is_lock"):
        lock_icon = "🔒"
        lock_text = "LOCK"
        card_class = "lock-card"
    else:
        lock_icon = "🔥"
        lock_text = "RECOMMENDED"
        card_class = "primary-card"
    
    st.markdown(f"""
    <div class="output-card {card_class}" style="border-left: 4px solid {'#f59e0b' if primary.get('is_lock') else '#10b981'};">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem;">{lock_icon}</span>
            <span style="font-size: 1.2rem; font-weight: 700;">OUTCOME BET</span>
            <span style="background: {'#f59e0b' if primary.get('is_lock') else '#10b981'}; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700;">{lock_text}</span>
        </div>
        <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 0.25rem;">{primary.get('outcome_bet', 'Unknown')}</div>
        <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.9rem;">
            <span>📊 Accuracy: {primary.get('outcome_accuracy', 'N/A')}</span>
            <span>📝 Reason: {primary.get('outcome_reason', 'N/A')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ================================================================
    # TIER 2 SCORES (if applicable)
    # ================================================================
    
    if tier == "Tier 2":
        scores = primary.get("tier2_scores", {})
        if scores:
            st.markdown("### 📊 Tier 2 Scores")
            cols = st.columns(3)
            with cols[0]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{scores.get("home_score", 0)}</div><div class="metric-label">Home Score</div></div>', unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{scores.get("away_score", 0)}</div><div class="metric-label">Away Score</div></div>', unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{scores.get("draw_score", 0)}</div><div class="metric-label">Draw Score</div></div>', unsafe_allow_html=True)
    
    # ================================================================
    # GOAL BET
    # ================================================================
    
    goal_bet = primary.get('goal_bet')
    
    if goal_bet:
        goal_is_lock = primary.get('goal_is_lock', False) or primary.get('is_lock', False)
        lock_icon_goal = "🔒" if goal_is_lock else "📊"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid {'#f59e0b' if goal_is_lock else '#10b981'};">
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
    
    if goal_bet:
        st.markdown("""
        <hr style="border: none; border-top: 1px dashed #475569; margin: 0.5rem 0;">
        <div style="background: #1a2a1a; padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid #2a4a2a; text-align: center;">
            <span style="color: #94a3b8; font-size: 0.9rem;">📌 These are </span>
            <span style="color: #fbbf24; font-weight: 700; font-size: 0.9rem;">SEPARATE</span>
            <span style="color: #94a3b8; font-size: 0.9rem;"> bets. Place them individually.</span>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================================================
    # KEY METRICS
    # ================================================================
    
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
    st.title("🏆 Match Analyzer V1.1")
    st.caption("Two-Tier System | ALL Matches (X, 1, 2) | Tier 1: LOCK (100%) | Tier 2: Interwoven (95%)")

    with st.expander("📖 The Two-Tier System — ALL Matches", expanded=False):
        st.markdown("""
        **Works on ALL matches — X, 1, and 2 predictions.**
        
        **TIER 1: LOCK Bets (100% Accuracy)**
        
        | Signal | Trigger | Bet |
        |--------|---------|-----|
        | Home Desperation | Home in relegation zone OR 3+ losses | **HOME WIN** |
        | Elite Home | Home Top 3-4 Home Table + Away NOT Top 3-4 Away | **HOME WIN** |
        | Elite Away | Away Top 3-4 Away Table + Home NOT Top 3-4 Home | **AWAY WIN** |
        
        **TIER 2: Interwoven Framework (95% Accuracy)**
        
        | Score | Components |
        |-------|------------|
        | Home Score | Home Form + Home Block + Home Desperation + Home Elite |
        | Away Score | Away Form + Away Block + Away Desperation + Away Elite |
        | Draw Score | Form Similar + Goals Sweet Spot + Same Block + No Desperation |
        
        **Conflict Detection:**
        - Skip if Home ≥ 4 AND Away ≥ 4
        - Skip if Draw ≥ 6 AND Home ≥ 4
        - Skip if Draw ≥ 6 AND Away ≥ 4
        - Skip if all scores ≤ 2
        """)

    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info("🏆 The Two-Tier System analyzes ALL matches (X, 1, 2). No draw-only filter.")

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

        if st.button("🏆 ANALYZE V1.1", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with Two-Tier System..."):
                        parsed = parse_text_data(text_data)

                    league = parsed.get("league", "Unknown League")
                    matches = parsed.get("matches", [])
                    home_table = parsed.get("home_table", {})
                    away_table = parsed.get("away_table", {})
                    form_data = parsed.get("form_data", {})
                    league_config = parsed.get("league_config", {})

                    if matches:
                        # Count matches
                        ft_matches = [m for m in matches if m.get("is_finished")]
                        total_matches = len(matches)
                        
                        st.success(f"✅ Found {total_matches} matches in {league}")
                        
                        if ft_matches:
                            st.info(f"⏭️ {len(ft_matches)} matches already played (FT) — skipped")
                        
                        # Process all matches
                        analyzed_results = []
                        skipped_results = []
                        
                        for match in matches:
                            match_with_config = dict(match)
                            match_with_config["league_config"] = league_config
                            data = convert_match_to_data(match_with_config, home_table, away_table, form_data, league)
                            analysis = analyze_match(data)
                            save_to_db(data, analysis, league)
                            
                            if analysis.get("verdict") == "SKIP":
                                skipped_results.append((match, data, analysis))
                            else:
                                analyzed_results.append((match, data, analysis))

                        # ============================================================
                        # DISPLAY ANALYZED MATCHES FIRST
                        # ============================================================
                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🏆 ANALYZED MATCHES")
                            st.caption(f"{len(analyzed_results)} matches with actionable bets")
                            
                            for idx, (match, data, analysis) in enumerate(analyzed_results, 1):
                                tier = analysis.get("tier", "")
                                pred = match.get('prediction', '?')
                                pred_icon = "🎯" if pred == 'X' else "1️⃣" if pred == '1' else "2️⃣"
                                tier_icon = "🔒" if tier == "Tier 1" else "📊"
                                st.markdown(f"#### {pred_icon} {tier_icon} Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} (Pred: {pred} | {tier})")
                                
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
                        
                        # ============================================================
                        # DISPLAY SKIPPED MATCHES AFTER (COLLAPSED)
                        # ============================================================
                        if skipped_results:
                            st.markdown("---")
                            st.markdown("### ⏭️ SKIPPED MATCHES")
                            st.caption(f"{len(skipped_results)} matches skipped (FT or no clear signal)")
                            
                            with st.expander(f"Click to expand {len(skipped_results)} skipped matches"):
                                for idx, (match, data, analysis) in enumerate(skipped_results, 1):
                                    pred = match.get('prediction', '?')
                                    pred_icon = "🎯" if pred == 'X' else "1️⃣" if pred == '1' else "2️⃣"
                                    st.markdown(f"#### {pred_icon} Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} (Pred: {pred})")
                                    
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
                        
                        # Summary stats
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Matches", total_matches)
                        with col2:
                            tier1_count = len([a for a in analyzed_results if a[2].get("tier") == "Tier 1"])
                            st.metric("🔒 Tier 1", tier1_count)
                        with col3:
                            tier2_count = len([a for a in analyzed_results if a[2].get("tier") == "Tier 2"])
                            st.metric("📊 Tier 2", tier2_count)
                        with col4:
                            st.metric("⏭️ Skipped", len(skipped_results))
                        with col5:
                            ft_count = len([m for m in matches if m.get("is_finished")])
                            st.metric("⏭️ FT", ft_count)
                            
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
                is_lock = a.get('is_lock', False)
                warning = a.get('warning')
                skip_reason = a.get('skip_reason')
                is_finished = a.get('is_finished', False)
                tier = a.get('tier', '')

                if is_finished:
                    badge = "⏭️ FT (Already Played)"
                elif skip_reason:
                    badge = "⏭️ SKIPPED"
                elif is_lock:
                    badge = f"🔒 {tier} LOCK"
                else:
                    badge = f"📊 {tier}"

                with st.expander(f"{badge} | {ht} vs {at} — Predicted: {pred}"):
                    if warning:
                        st.warning(warning)
                    if is_finished:
                        st.info(f"⏭️ This match is already played (FT). Result: {a.get('actual_home', '?')}-{a.get('actual_away', '?')}")
                    elif skip_reason:
                        st.info(f"⏭️ {skip_reason}")
                    else:
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
            analyzed_results = [r for r in results if r.get('bet_market') != 'SKIP' and not r.get('skip_reason') and not r.get('is_finished')]
            ft_results = [r for r in results if r.get('is_finished')]
            skipped_results = [r for r in results if r.get('bet_market') == 'SKIP' or r.get('skip_reason')]
            
            total = len(results)
            analyzed_count = len(analyzed_results)
            ft_count = len(ft_results)
            skipped_count = len(skipped_results)

            correct = 0
            incorrect = 0
            lock_correct = 0
            lock_total = 0
            tier1_correct = 0
            tier1_total = 0
            tier2_correct = 0
            tier2_total = 0

            for r in analyzed_results:
                pred = r.get('bet_market', '')
                if pred == 'SKIP':
                    continue

                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))

                if evaluation["is_correct"]:
                    correct += 1
                    if r.get('is_lock', False):
                        lock_correct += 1
                    if r.get('tier') == 'Tier 1':
                        tier1_correct += 1
                    if r.get('tier') == 'Tier 2':
                        tier2_correct += 1
                else:
                    incorrect += 1

                if r.get('is_lock', False):
                    lock_total += 1
                if r.get('tier') == 'Tier 1':
                    tier1_total += 1
                if r.get('tier') == 'Tier 2':
                    tier2_total += 1

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Tracked</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{ft_count}</div><div class="stat-label">FT (Already Played)</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{skipped_count}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{analyzed_count}</div><div class="stat-label">Analyzed</div></div>', unsafe_allow_html=True)
            with col5:
                win_rate = round(correct / analyzed_count * 100) if analyzed_count > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)

            # Tier breakdown
            st.markdown("### 🔒 Tier Breakdown")
            tier1_col, tier2_col = st.columns(2)
            with tier1_col:
                tier1_rate = round(tier1_correct / tier1_total * 100) if tier1_total > 0 else 0
                st.metric("Tier 1 (LOCK)", f"{tier1_correct}/{tier1_total} ({tier1_rate}%)")
            with tier2_col:
                tier2_rate = round(tier2_correct / tier2_total * 100) if tier2_total > 0 else 0
                st.metric("Tier 2 (Strong)", f"{tier2_correct}/{tier2_total} ({tier2_rate}%)")

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
                is_lock = r.get('is_lock', False)
                league = r.get('league', '')
                badge_class = get_league_badge(league)
                skip_reason = r.get('skip_reason')
                is_finished = r.get('is_finished', False)
                tier = r.get('tier', '')

                if is_finished:
                    badge = '<span class="skip-badge">⏭️ FT</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
                    primary_pred = "FT"
                elif skip_reason or pred == 'SKIP':
                    badge = '<span class="skip-badge">⏭️ SKIP</span>'
                    score_display = "—"
                    primary_pred = "SKIPPED"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    if tier == 'Tier 1':
                        tier_label = "🔒"
                    else:
                        tier_label = "📊"
                    badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"

                match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"

                rows.append({
                    "Date": r.get("match_date", ""),
                    "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
                    "Match": match_display,
                    "Class": f"{tier} {r.get('classification', '')}",
                    "Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })

            df = pd.DataFrame(rows)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
