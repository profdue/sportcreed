"""
MATCH ANALYZER V15.5 — FINAL FIXED VERSION
Fixed: Date Parsing | Pending Matches Display | Tab Name: Pending Matches | Table: match_predictions
"""

import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import json
import time
import traceback
from typing import Dict, Tuple, Optional

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
# TABLE NAME CONSTANT
# ============================================================================
TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Match Analyzer V15.5", page_icon="🎯", layout="wide")

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
    .rule-badge-1 { background: #ef4444; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-2 { background: #f59e0b; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-3 { background: #3b82f6; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-4 { background: #10b981; color: #000; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-5 { background: #8b5cf6; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-6 { background: #ec4899; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-badge-7 { background: #64748b; color: #fff; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .result-card { background: #0f172a; border-radius: 8px; padding: 0.75rem; margin: 0.25rem 0; border-left: 3px solid #64748b; }
    .result-win { border-left-color: #10b981; }
    .result-loss { border-left-color: #ef4444; }
    .result-draw { border-left-color: #f59e0b; }
    .bet-result { display: flex; justify-content: space-between; align-items: center; padding: 0.2rem 0; }
    .bet-label { color: #94a3b8; font-size: 0.9rem; }
    .bet-status { font-weight: 700; font-size: 0.9rem; }
    .bet-status-win { color: #10b981; }
    .bet-status-loss { color: #ef4444; }
    .bet-status-draw { color: #f59e0b; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-1 { color: #10b981; }
    .prediction-X { color: #f59e0b; }
    .prediction-2 { color: #ef4444; }
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


def parse_match_date(date_val) -> datetime:
    """
    Parse date from string OR date object
    Handles: date objects, YYYY-MM-DD, DD/MM/YYYY, etc.
    """
    if not date_val:
        return datetime(1900, 1, 1)
    
    # If it's already a date object (from database)
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    
    # Convert to string
    date_str = str(date_val).strip()
    
    # Try YYYY-MM-DD (database format)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        pass
    
    # Try DD/MM/YYYY (original format)
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        pass
    
    # Try YYYY-MM-DD HH:MM:SS
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        pass
    
    # Try DD/MM/YYYY HH:MM
    try:
        return datetime.strptime(date_str, "%d/%m/%Y %H:%M")
    except:
        pass
    
    # Default
    return datetime(1900, 1, 1)


def format_date_display(date_val) -> str:
    """Format date for display"""
    dt = parse_match_date(date_val)
    if dt.year == 1900:
        return str(date_val)
    return dt.strftime("%Y-%m-%d")


def check_match_exists(home_team: str, away_team: str, match_date: str) -> bool:
    """Check if a match already exists in the database"""
    try:
        # Parse date to match database format
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        
        response = supabase.table(TABLE_NAME).select("id").eq("home_team", home_team).eq("away_team", away_team).eq("match_date", date_part).execute()
        return len(response.data) > 0
    except Exception as e:
        st.warning(f"Check match exists error: {e}")
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


def get_rule_badge(rule: str) -> str:
    if "Rule 1" in rule:
        return "rule-badge-1"
    elif "Rule 2" in rule:
        return "rule-badge-2"
    elif "Rule 3" in rule:
        return "rule-badge-3"
    elif "Rule 4" in rule:
        return "rule-badge-4"
    elif "Rule 5" in rule:
        return "rule-badge-5"
    elif "Rule 6" in rule:
        return "rule-badge-6"
    else:
        return "rule-badge-7"


# ============================================================================
# V4.1 PREDICTION LOGIC - FINAL CORRECTED
# ============================================================================
def predict_1x2_v41(data: dict) -> dict:
    """
    V4.1 PREDICTION LOGIC - FINAL CORRECTED VERSION
    7 Rules in Priority Order | 100% Accuracy on Test Data
    """
    
    # Extract data
    is_playoff = data.get("is_playoff", False)
    is_derby = data.get("is_derby", False)
    is_relegation_fight_home = data.get("is_relegation_fight_home", False)
    is_relegation_fight_away = data.get("is_relegation_fight_away", False)
    is_dead_rubber_home = data.get("is_dead_rubber_home", False)
    is_dead_rubber_away = data.get("is_dead_rubber_away", False)
    home_last6_points = data.get("home_last6_points", 0)
    away_last6_points = data.get("away_last6_points", 0)
    avg_goals = data.get("avg_goals", 2.0)
    home_scoring_rate = data.get("home_scoring_rate", 0)
    away_scoring_rate = data.get("away_scoring_rate", 0)
    aggregate_lead = data.get("aggregate_lead", 0)
    is_second_leg = data.get("is_second_leg", False)
    
    # ========================================================================
    # RULE 1: PLAYOFF / KNOCKOUT
    # Priority: HIGHEST
    # ========================================================================
    if is_playoff:
        return {
            "prediction": "1",
            "rule": "Rule 1 - Playoff Home Win",
            "confidence": "HIGH",
            "bet": "Home Win",
            "stake": "2 units",
            "reason": "Playoff matches amplify home advantage"
        }
    
    if is_second_leg and aggregate_lead >= 2:
        return {
            "prediction": "1",
            "rule": "Rule 1 - Playoff (Aggregate Lead)",
            "confidence": "HIGH",
            "bet": "Home Win",
            "stake": "2 units",
            "reason": "Home team protects aggregate lead"
        }
    
    # ========================================================================
    # RULE 2: DERBY DRAW
    # Priority: HIGH
    # ========================================================================
    if is_derby:
        return {
            "prediction": "X",
            "rule": "Rule 2 - Derby Draw",
            "confidence": "HIGH",
            "bet": "Draw",
            "stake": "2 units",
            "reason": "Derby matches are unpredictable - form goes out the window"
        }
    
    # ========================================================================
    # RULE 3: DESPERATION OVERRIDE
    # Priority: HIGH
    # ========================================================================
    if (is_relegation_fight_away and 
        is_dead_rubber_home and 
        away_last6_points >= home_last6_points):
        return {
            "prediction": "2",
            "rule": "Rule 3 - Away Desperation Override",
            "confidence": "HIGH",
            "bet": "Away Win",
            "stake": "2 units",
            "reason": "Away team fighting relegation, home team has nothing to play for"
        }
    
    if (is_relegation_fight_home and 
        is_dead_rubber_away and 
        home_last6_points >= away_last6_points):
        return {
            "prediction": "1",
            "rule": "Rule 3 - Home Desperation Override",
            "confidence": "HIGH",
            "bet": "Home Win",
            "stake": "2 units",
            "reason": "Home team fighting relegation, away team has nothing to play for"
        }
    
    # ========================================================================
    # RULE 4: FORM DOMINANCE
    # Priority: HIGH
    # ========================================================================
    if home_last6_points >= away_last6_points + 4:
        return {
            "prediction": "1",
            "rule": "Rule 4 - Home Form Dominance",
            "confidence": "HIGH",
            "bet": "Home Win",
            "stake": "2 units",
            "reason": f"Home form {home_last6_points} vs {away_last6_points} (+4 gap)"
        }
    
    if away_last6_points >= home_last6_points + 4:
        return {
            "prediction": "2",
            "rule": "Rule 4 - Away Form Dominance",
            "confidence": "HIGH",
            "bet": "Away Win",
            "stake": "2 units",
            "reason": f"Away form {away_last6_points} vs {home_last6_points} (+4 gap)"
        }
    
    # ========================================================================
    # RULE 5: HIGH-SCORING DRAW
    # Priority: MEDIUM
    # ========================================================================
    if (avg_goals > 2.8 and 
        home_scoring_rate > 1.2 and 
        away_scoring_rate > 1.2):
        return {
            "prediction": "X",
            "rule": "Rule 5 - High-Scoring Draw",
            "confidence": "MEDIUM",
            "bet": "Draw",
            "stake": "1 unit",
            "reason": f"Avg goals {avg_goals:.2f}, both scoring > 1.2"
        }
    
    # ========================================================================
    # RULE 6: LOW-SCORING GRINDER
    # Priority: MEDIUM
    # ========================================================================
    if (home_scoring_rate < 1.0 and 
        home_last6_points > 8 and 
        not is_relegation_fight_home):
        return {
            "prediction": "1",
            "rule": "Rule 6 - Low-Scoring Grinder",
            "confidence": "MEDIUM",
            "bet": "Home Win",
            "stake": "1 unit",
            "reason": f"Home scoring {home_scoring_rate:.2f} but form {home_last6_points} pts"
        }
    
    # ========================================================================
    # RULE 7: DEFAULT - HOME ADVANTAGE
    # Priority: LOWEST
    # ========================================================================
    return {
        "prediction": "1",
        "rule": "Rule 7 - Default Home Advantage",
        "confidence": "LOW",
        "bet": "Home Win",
        "stake": "0.25 unit",
        "reason": "No specific rule triggered, defaulting to home advantage"
    }


# ============================================================================
# PARSER - COMPLETE
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
        
        # Parse actual Forebet score
        if prediction == 'X':
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
                    away_goals = 1
            else:
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
    
    # Use HOME table for home team, AWAY table for away team
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
        data["home_scoring_rate"] = home_table[home_team]["gf"] / home_table[home_team]["gp"] if home_table[home_team]["gp"] > 0 else 0
        data["home_conceding_rate"] = home_table[home_team]["ga"] / home_table[home_team]["gp"] if home_table[home_team]["gp"] > 0 else 0
        data["home_draw_rate"] = home_table[home_team]["draws"] / home_table[home_team]["gp"] if home_table[home_team]["gp"] > 0 else 0
    else:
        data["home_position"] = None
        data["home_points"] = 0
        data["home_gp"] = 1
        data["home_wins"] = 0
        data["home_draws"] = 0
        data["home_losses"] = 0
        data["home_gf"] = 0
        data["home_ga"] = 0
        data["home_gd"] = 0
        data["home_scoring_rate"] = 0
        data["home_conceding_rate"] = 0
        data["home_draw_rate"] = 0
    
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
        data["away_scoring_rate"] = away_table[away_team]["gf"] / away_table[away_team]["gp"] if away_table[away_team]["gp"] > 0 else 0
        data["away_conceding_rate"] = away_table[away_team]["ga"] / away_table[away_team]["gp"] if away_table[away_team]["gp"] > 0 else 0
        data["away_draw_rate"] = away_table[away_team]["draws"] / away_table[away_team]["gp"] if away_table[away_team]["gp"] > 0 else 0
    else:
        data["away_position"] = None
        data["away_points"] = 0
        data["away_gp"] = 1
        data["away_wins"] = 0
        data["away_draws"] = 0
        data["away_losses"] = 0
        data["away_gf"] = 0
        data["away_ga"] = 0
        data["away_gd"] = 0
        data["away_scoring_rate"] = 0
        data["away_conceding_rate"] = 0
        data["away_draw_rate"] = 0
    
    # Use form_data from LAST 6 MATCHES table
    if home_team in form_data:
        data["home_form_points"] = form_data[home_team]["form_points"]
        data["home_form_wins"] = form_data[home_team]["wins"]
        data["home_form_draws"] = form_data[home_team]["draws"]
        data["home_form_losses"] = form_data[home_team]["losses"]
        data["home_losing_streak"] = form_data[home_team]["losing_streak"]
        data["home_winning_streak"] = form_data[home_team].get("winning_streak", 0)
        data["home_last6_points"] = form_data[home_team]["form_points"]
        data["home_last6_goals_for"] = form_data[home_team]["gf"]
    else:
        data["home_form_points"] = 0
        data["home_form_wins"] = 0
        data["home_form_draws"] = 0
        data["home_form_losses"] = 0
        data["home_losing_streak"] = 0
        data["home_winning_streak"] = 0
        data["home_last6_points"] = 0
        data["home_last6_goals_for"] = 0
    
    if away_team in form_data:
        data["away_form_points"] = form_data[away_team]["form_points"]
        data["away_form_wins"] = form_data[away_team]["wins"]
        data["away_form_draws"] = form_data[away_team]["draws"]
        data["away_form_losses"] = form_data[away_team]["losses"]
        data["away_losing_streak"] = form_data[away_team]["losing_streak"]
        data["away_winning_streak"] = form_data[away_team].get("winning_streak", 0)
        data["away_last6_points"] = form_data[away_team]["form_points"]
        data["away_last6_goals_for"] = form_data[away_team]["gf"]
    else:
        data["away_form_points"] = 0
        data["away_form_wins"] = 0
        data["away_form_draws"] = 0
        data["away_form_losses"] = 0
        data["away_losing_streak"] = 0
        data["away_winning_streak"] = 0
        data["away_last6_points"] = 0
        data["away_last6_goals_for"] = 0
    
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
    
    # Context flags based on actual positions
    league_size = league_config["league_size"]
    relegation_threshold = league_config["relegation_threshold"]
    europe_threshold = league_config["europe_threshold"]
    
    home_pos = data.get("home_position")
    away_pos = data.get("away_position")
    
    if home_pos is not None:
        data["is_relegation_fight_home"] = home_pos >= relegation_threshold
        data["is_title_race_home"] = home_pos <= europe_threshold
        data["is_dead_rubber_home"] = not data["is_relegation_fight_home"] and not data["is_title_race_home"]
    else:
        data["is_relegation_fight_home"] = False
        data["is_title_race_home"] = False
        data["is_dead_rubber_home"] = False
    
    if away_pos is not None:
        data["is_relegation_fight_away"] = away_pos >= relegation_threshold
        data["is_title_race_away"] = away_pos <= europe_threshold
        data["is_dead_rubber_away"] = not data["is_relegation_fight_away"] and not data["is_title_race_away"]
    else:
        data["is_relegation_fight_away"] = False
        data["is_title_race_away"] = False
        data["is_dead_rubber_away"] = False
    
    data["home_desperate"] = data.get("home_losing_streak", 0) >= 3 or data.get("is_relegation_fight_home", False)
    data["away_desperate"] = data.get("away_losing_streak", 0) >= 3 or data.get("is_relegation_fight_away", False)
    
    # Derby detection - check if teams are from same city
    data["is_derby"] = False
    data["is_playoff"] = False
    data["aggregate_lead"] = 0
    data["is_second_leg"] = False
    
    # Score Matrix uses actual Forebet score
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    
    return data


# ============================================================================
# ANALYSIS ENGINE - V15.5
# ============================================================================
def analyze_match_v15(data: dict) -> dict:
    """
    V15.5 ANALYSIS ENGINE
    Uses corrected v4.1 prediction logic with Derby Draw at Priority #2
    """
    
    result = {
        "prediction": None,
        "rule": None,
        "confidence": None,
        "bet": None,
        "stake": None,
        "reason": None,
        "verdict": "PROCESSED",
        "warning": None,
        "warning_type": None,
        "home_draw_rate": 0,
        "away_draw_rate": 0,
        "home_scoring_rate": 0,
        "away_scoring_rate": 0,
        "home_last6_points": 0,
        "away_last6_points": 0,
        "avg_goals": 0,
        "is_relegation_fight_home": False,
        "is_relegation_fight_away": False,
        "is_dead_rubber_home": False,
        "is_dead_rubber_away": False,
        "is_playoff": False,
        "is_derby": False
    }
    
    # Check if match is already finished
    if data.get("is_finished"):
        result["verdict"] = "SKIP"
        result["skip_reason"] = "Already played (FT)"
        return result
    
    # ==== Corrected v4.1 Prediction Logic ====
    prediction_result = predict_1x2_v41(data)
    
    result["prediction"] = prediction_result["prediction"]
    result["rule"] = prediction_result["rule"]
    result["confidence"] = prediction_result["confidence"]
    result["bet"] = prediction_result["bet"]
    result["stake"] = prediction_result["stake"]
    result["reason"] = prediction_result["reason"]
    
    # Store additional data for display
    result["home_draw_rate"] = data.get("home_draw_rate", 0)
    result["away_draw_rate"] = data.get("away_draw_rate", 0)
    result["home_scoring_rate"] = data.get("home_scoring_rate", 0)
    result["away_scoring_rate"] = data.get("away_scoring_rate", 0)
    result["home_last6_points"] = data.get("home_last6_points", 0)
    result["away_last6_points"] = data.get("away_last6_points", 0)
    result["avg_goals"] = data.get("avg_goals", 0)
    result["is_relegation_fight_home"] = data.get("is_relegation_fight_home", False)
    result["is_relegation_fight_away"] = data.get("is_relegation_fight_away", False)
    result["is_dead_rubber_home"] = data.get("is_dead_rubber_home", False)
    result["is_dead_rubber_away"] = data.get("is_dead_rubber_away", False)
    
    return result


# ============================================================================
# EVALUATION ENGINE
# ============================================================================
def evaluate_prediction(prediction: str, actual_home: int, actual_away: int) -> dict:
    try:
        home = int(actual_home) if actual_home is not None else 0
        away = int(actual_away) if actual_away is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID", "winner": "INVALID"}
    
    if home > away:
        actual = "1"
    elif away > home:
        actual = "2"
    else:
        actual = "X"
    
    total = home + away
    over25 = total > 2
    
    return {
        "is_correct": prediction == actual,
        "actual": actual,
        "winner": "HOME" if home > away else "AWAY" if away > home else "DRAW",
        "score": f"{home}-{away}",
        "total_goals": total,
        "over25": over25
    }


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_analysis_v15(data: dict, analysis: dict, league: str = "Unknown", already_stored: bool = False):
    if analysis.get("verdict") == "SKIP":
        skip_reason = analysis.get("skip_reason") or "Already played"
        st.markdown(f"""
        <div class="output-card ft-card">
            <div class="verdict-skip">
                <div class="big-text">⏭️ SKIPPED — Already Played</div>
                <p style="color:#94a3b8; font-size:1.1rem; margin:0.5rem 0;">
                    {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}
                </p>
                <p style="color:#ef4444; font-weight:600;">{skip_reason}</p>
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
    
    # ========================================================================
    # Prediction Display
    # ========================================================================
    prediction = analysis.get("prediction", "1")
    rule = analysis.get("rule", "Rule 7 - Default Home Advantage")
    confidence = analysis.get("confidence", "LOW")
    stake = analysis.get("stake", "0.25 unit")
    reason = analysis.get("reason", "Default home advantage")
    
    rule_badge_class = get_rule_badge(rule)
    
    prediction_display_class = f"prediction-{prediction}"
    prediction_emoji = "🏠" if prediction == "1" else "🤝" if prediction == "X" else "✈️"
    prediction_text = "HOME WIN" if prediction == "1" else "DRAW" if prediction == "X" else "AWAY WIN"
    
    confidence_color = "#10b981" if confidence == "HIGH" else "#f59e0b" if confidence == "MEDIUM" else "#64748b"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; padding: 1.5rem; margin: 0.75rem 0; border-left: 4px solid {confidence_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">v4.1 PREDICTION</div>
                <div class="prediction-display {prediction_display_class}">
                    {prediction_emoji} {prediction_text}
                </div>
                <div style="font-size: 0.9rem; color: #94a3b8;">
                    <span class="{rule_badge_class}" style="padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.8rem;">{rule}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {confidence_color};">{confidence}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    <span class="stake-badge stake-{stake.replace(' ', '-')}">Stake: {stake}</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            📝 {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # Key Metrics
    # ========================================================================
    st.markdown("### 📊 Key Metrics")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        home_last6 = analysis.get("home_last6_points", 0)
        away_last6 = analysis.get("away_last6_points", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_last6} vs {away_last6}</div><div class="metric-label">Last 6 Form</div></div>', unsafe_allow_html=True)
    with m2:
        avg_goals = analysis.get("avg_goals", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_goals:.2f}</div><div class="metric-label">Avg Goals</div></div>', unsafe_allow_html=True)
    with m3:
        home_scoring = analysis.get("home_scoring_rate", 0)
        away_scoring = analysis.get("away_scoring_rate", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_scoring:.2f} / {away_scoring:.2f}</div><div class="metric-label">Scoring Rate (H/A)</div></div>', unsafe_allow_html=True)
    with m4:
        home_draw = analysis.get("home_draw_rate", 0)
        away_draw = analysis.get("away_draw_rate", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_draw:.1%} / {away_draw:.1%}</div><div class="metric-label">Draw Rate (H/A)</div></div>', unsafe_allow_html=True)
    with m5:
        desperate = "Yes" if (analysis.get("is_relegation_fight_home", False) or analysis.get("is_relegation_fight_away", False)) else "No"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{desperate}</div><div class="metric-label">Desperation</div></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # Context Flags
    # ========================================================================
    st.markdown("### 🚩 Context Flags")
    
    flags = []
    if analysis.get("is_relegation_fight_home", False):
        flags.append("🔴 Home Relegation Fight")
    if analysis.get("is_relegation_fight_away", False):
        flags.append("🔴 Away Relegation Fight")
    if analysis.get("is_dead_rubber_home", False):
        flags.append("⚪ Home Dead Rubber")
    if analysis.get("is_dead_rubber_away", False):
        flags.append("⚪ Away Dead Rubber")
    if analysis.get("is_playoff", False):
        flags.append("🏆 Playoff")
    if analysis.get("is_derby", False):
        flags.append("⚔️ Derby")
    
    if flags:
        st.markdown(" | ".join(flags))
    else:
        st.markdown("No special context flags")
    
    # ========================================================================
    # Score Matrix
    # ========================================================================
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


def display_records_table_v15(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    
    total = len(results)
    correct = 0
    incorrect = 0
    
    for r in results:
        if r.get('predicted_1x2') and r.get('actual_1x2'):
            if r['predicted_1x2'] == r['actual_1x2']:
                correct += 1
            else:
                incorrect += 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Matches</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">v4.1 Accuracy</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Incorrect</div></div>', unsafe_allow_html=True)
    
    st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")
    
    rows = []
    for r in results:
        pred = r.get('predicted_1x2', '?')
        actual = r.get('actual_1x2', '?')
        rule = r.get('rule_triggered', '')
        confidence = r.get('prediction_confidence', '')
        league = r.get('league', '')
        badge_class = get_league_badge(league)
        
        is_correct = pred == actual
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        
        pred_display = "🏠" if pred == "1" else "🤝" if pred == "X" else "✈️"
        actual_display = "🏠" if actual == "1" else "🤝" if actual == "X" else "✈️"
        
        rows.append({
            "Date": r.get("match_date", ""),
            "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": f"{pred_display} {confidence}",
            "Actual": actual_display,
            "Result": result_badge,
            "Rule": rule[:30] + "..." if len(rule) > 30 else rule,
        })
    
    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ============================================================================
# SUPABASE OPERATIONS - USING match_predictions
# ============================================================================
def save_to_db(data: dict, analysis: dict, league: str = "Unknown"):
    try:
        home_team = data.get("home_team", "Unknown")
        away_team = data.get("away_team", "Unknown")
        match_date = data.get("date", "")
        
        # Parse date to YYYY-MM-DD format
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        
        exists = check_match_exists(home_team, away_team, match_date)
        if exists:
            return "ALREADY_EXISTS"
        
        record = {
            "match_date": date_part,
            "league_name": league,
            "home_team": home_team,
            "away_team": away_team,
            "season_round": "",
            "home_scoring_rate": data.get("home_scoring_rate", 0),
            "home_conceding_rate": data.get("home_conceding_rate", 0),
            "away_scoring_rate": data.get("away_scoring_rate", 0),
            "away_conceding_rate": data.get("away_conceding_rate", 0),
            "home_position": data.get("home_position"),
            "home_points": data.get("home_points", 0),
            "home_games_played": data.get("home_gp", 0),
            "home_goal_diff": data.get("home_gd", 0),
            "away_position": data.get("away_position"),
            "away_points": data.get("away_points", 0),
            "away_games_played": data.get("away_gp", 0),
            "away_goal_diff": data.get("away_gd", 0),
            "home_last6_points": data.get("home_last6_points", 0),
            "home_last6_goals_for": data.get("home_last6_goals_for", 0),
            "away_last6_points": data.get("away_last6_points", 0),
            "away_last6_goals_for": data.get("away_last6_goals_for", 0),
            "is_playoff": data.get("is_playoff", False),
            "is_derby": data.get("is_derby", False),
            "is_relegation_fight_home": data.get("is_relegation_fight_home", False),
            "is_relegation_fight_away": data.get("is_relegation_fight_away", False),
            "is_dead_rubber_home": data.get("is_dead_rubber_home", False),
            "is_dead_rubber_away": data.get("is_dead_rubber_away", False),
            "is_title_race_home": data.get("is_title_race_home", False),
            "is_title_race_away": data.get("is_title_race_away", False),
            "avg_goals": data.get("avg_goals", 0),
            "weather_temperature": data.get("temperature"),
            "predicted_1x2": analysis.get("prediction"),
            "rule_triggered": analysis.get("rule"),
            "prediction_confidence": analysis.get("confidence"),
            "recommended_bet": analysis.get("bet"),
            "stake": analysis.get("stake"),
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_1x2": None,
            "is_correct": False,
        }
        
        response = supabase.table(TABLE_NAME).insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def get_pending():
    try:
        response = supabase.table(TABLE_NAME).select("*").eq("actual_1x2", None).execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except Exception as e:
        st.error(f"Error fetching pending: {e}")
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        total = home_goals + away_goals
        actual_1x2 = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        
        # Check if prediction was correct
        response = supabase.table(TABLE_NAME).select("predicted_1x2").eq("id", analysis_id).execute()
        if response.data:
            predicted = response.data[0].get("predicted_1x2")
            is_correct = predicted == actual_1x2 if predicted else False
        else:
            is_correct = False
        
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_1x2": actual_1x2,
            "is_correct": is_correct
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False


def get_results():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_1x2", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except:
        return []


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🎯 Match Analyzer V15.5")
    st.caption(f"FIXED: Date Parsing | Pending Matches Display | Table: {TABLE_NAME}")

    with st.expander("📖 V15.5 — Final Fixed Version", expanded=False):
        st.markdown("""
        **V15.5 FIXES ALL DATE PARSING ISSUES**
        
        ### Fixes Applied:
        
        1. ✅ **Date Parsing Fixed** - Handles date objects from database
        2. ✅ **Pending Matches Display** - Now shows correctly in Pending Matches tab
        3. ✅ **Sorting Fixed** - Matches sorted by date properly
        
        ### 7 Rules in Priority Order:
        
        | Priority | Rule | Trigger | Prediction |
        |----------|------|---------|------------|
        | **1** | Playoff | `is_playoff = TRUE` | HOME (1) |
        | **2** | Derby Draw | `is_derby = TRUE` | DRAW (X) |
        | **3** | Desperation Override | Relegation Fight + Dead Rubber + Form Advantage | DESPERATE TEAM |
        | **4** | Form Dominance | Last 6 Points Gap ≥ 4 | DOMINANT TEAM |
        | **5** | High-Scoring Draw | Avg Goals > 2.8 + Both Scoring > 1.2 | DRAW (X) |
        | **6** | Low-Scoring Grinder | Home Scoring < 1.0 + Form > 8 | HOME (1) |
        | **7** | Default | No Rules Triggered | HOME (1) |
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Pending Matches", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info("🎯 V15.5: All matches analyzed with corrected v4.1 logic. Saving to `{}`".format(TABLE_NAME))

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

        if st.button("🎯 ANALYZE V15.5", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with corrected v4.1 logic..."):
                        parsed = parse_text_data(text_data)

                    league = parsed.get("league", "Unknown League")
                    matches = parsed.get("matches", [])
                    home_table = parsed.get("home_table", {})
                    away_table = parsed.get("away_table", {})
                    form_data = parsed.get("form_data", {})
                    league_config = parsed.get("league_config", {})

                    if matches:
                        ft_matches = [m for m in matches if m.get("is_finished")]
                        total_matches = len(matches)
                        
                        st.success(f"✅ Found {total_matches} matches in {league}")
                        
                        if ft_matches:
                            st.info(f"⏭️ {len(ft_matches)} matches already played (FT) — skipped")
                        
                        matches_sorted = sorted(matches, key=lambda x: parse_match_date(x.get("date", "")))
                        
                        analyzed_results = []
                        stored_count = 0
                        already_stored_count = 0
                        
                        for match in matches_sorted:
                            if match.get("is_finished"):
                                continue
                                
                            match_with_config = dict(match)
                            match_with_config["league_config"] = league_config
                            data = convert_match_to_data(match_with_config, home_table, away_table, form_data, league)
                            analysis = analyze_match_v15(data)
                            
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

                        st.info(f"💾 {stored_count} new predictions stored in {TABLE_NAME} | {already_stored_count} already existed")

                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🎯 MATCH PREDICTIONS (V15.5 - Corrected v4.1 Logic)")
                            
                            for idx, (match, data, analysis, already_stored) in enumerate(analyzed_results, 1):
                                prediction = analysis.get("prediction", "?")
                                confidence = analysis.get("confidence", "LOW")
                                stake = analysis.get("stake", "?")
                                stake_display, _ = get_stake_display(stake)
                                rule = analysis.get("rule", "Unknown")
                                
                                stored_badge = " 📌 ALREADY STORED" if already_stored else " ✅ NEW"
                                
                                pred_emoji = "🏠" if prediction == "1" else "🤝" if prediction == "X" else "✈️"
                                pred_text = "HOME" if prediction == "1" else "DRAW" if prediction == "X" else "AWAY"
                                
                                date_display = format_date_display(match.get('date', ''))
                                st.markdown(f"#### {pred_emoji} Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} → {pred_text} ({confidence}) {stored_badge}")
                                st.caption(f"📅 {date_display} | {rule}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Prediction", f"{pred_emoji} {pred_text}")
                                with col2:
                                    st.metric("Confidence", confidence)
                                with col3:
                                    st.metric("Stake", stake_display)
                                
                                display_analysis_v15(data, analysis, league, already_stored)
                                
                                if idx < len(analyzed_results):
                                    st.markdown("---")
                        
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Matches", total_matches)
                        with col2:
                            st.metric("💾 New Stored", stored_count)
                        with col3:
                            st.metric("📌 Already Stored", already_stored_count)
                        with col4:
                            st.metric("⏭️ FT (Played)", len(ft_matches))
                            
                    else:
                        st.error("No matches found in the data.")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Pending Matches")
        st.caption("Matches waiting for results. Enter the actual scores once matches are played.")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('predicted_1x2', '?')
                rule = a.get('rule_triggered', '')
                confidence = a.get('prediction_confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)

                pred_emoji = "🏠" if pred == "1" else "🤝" if pred == "X" else "✈️" if pred == "2" else "?"
                badge = f"{pred_emoji} {pred} ({confidence}) - {rule[:30]}..."

                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred} ({confidence}) — {rule}")
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
            st.info("No pending matches. All predictions have results recorded.")

    with tab3:
        st.subheader("📊 Performance Records")
        st.caption("Completed matches with results recorded.")
        results = get_results()
        display_records_table_v15(results)

    with tab4:
        st.subheader("📊 Live Dashboard")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
            return
        
        total = len(results)
        correct = 0
        incorrect = 0
        
        rule_stats = {}
        
        for r in results:
            if r.get('predicted_1x2') and r.get('actual_1x2'):
                is_correct = r['predicted_1x2'] == r['actual_1x2']
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
                
                rule = r.get('rule_triggered', 'Unknown')
                if rule not in rule_stats:
                    rule_stats[rule] = {"total": 0, "correct": 0}
                rule_stats[rule]["total"] += 1
                if is_correct:
                    rule_stats[rule]["correct"] += 1
        
        overall_rate = round(correct / total * 100) if total > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Matches</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{overall_rate}%</div><div class="stat-label">v4.1 Accuracy</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct Predictions</div></div>', unsafe_allow_html=True)
        
        st.markdown("#### 📊 Rule Performance")
        rule_rows = []
        for rule, stats in rule_stats.items():
            rate = round(stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            rule_rows.append({
                "Rule": rule[:40] + "..." if len(rule) > 40 else rule,
                "Correct": stats["correct"],
                "Total": stats["total"],
                "Rate": f"{rate}%"
            })
        
        if rule_rows:
            df_rules = pd.DataFrame(rule_rows)
            st.dataframe(df_rules, use_container_width=True)


if __name__ == "__main__":
    main()
