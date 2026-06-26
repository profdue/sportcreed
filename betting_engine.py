"""
MATCH ANALYZER V9.0 — COMPLETE REWRITE
Fully understands Forebet data structure: Percentages + Score Code + Avg Goals + Temperature + Coefficient
Handles X predictions (1-1 correct score), numeric score codes, and all table data
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
st.set_page_config(page_title="Match Analyzer V9.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
    .dead-rubber-card { border: 3px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
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
    .debug-box { background: #1a1a2e; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.75rem; margin: 0.25rem 0; font-family: monospace; font-size: 0.8rem; max-height: 500px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
    .debug-box .success { color: #10b981; }
    .debug-box .error { color: #ef4444; }
    .debug-box .info { color: #3b82f6; }
    .debug-box .warning { color: #f59e0b; }
    .debug-box .highlight { color: #f472b6; }
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
        config["goals_fallback"] = 2.66
    elif "Premier" in league or "EPL" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.75
    elif "AuV" in league or "NPL" in league:
        config["relegation_threshold"] = 11  # Bottom 4 of 14
        config["league_size"] = 14
        config["europe_threshold"] = 3  # Top 3
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
    """Parse the complete text data."""
    start_time = time.time()
    
    debug_log = []
    
    def debug(msg, type="info"):
        debug_log.append({"msg": msg, "type": type})
    
    debug(f"📊 Starting parser...", "info")
    debug(f"📄 Text length: {len(text):,} characters", "info")
    
    league = detect_league(text)
    league_config = get_league_config(league)
    debug(f"🏆 Detected league: {league}", "info")
    
    result = {
        "league": league,
        "league_config": league_config,
        "matches": [],
        "home_table": {},
        "away_table": {},
        "form_data": {},
        "statistics": {}
    }
    
    # Split into sections
    sections = split_into_sections(text, debug)
    debug(f"📂 Sections: Predictions={len(sections.get('predictions', '')):,} chars, "
          f"Home Table={len(sections.get('home_table', '')):,} chars, "
          f"Away Table={len(sections.get('away_table', '')):,} chars, "
          f"Form={len(sections.get('form', '')):,} chars", "info")
    
    # Parse predictions
    matches = parse_predictions(sections.get("predictions", ""), debug)
    result["matches"] = matches
    debug(f"✅ Found {len(matches)} matches", "success")
    
    # Parse home table
    home_table = parse_table(sections.get("home_table", ""), "HOME", debug)
    result["home_table"] = home_table
    debug(f"✅ Found {len(home_table)} teams in home table", "success")
    
    # Parse away table
    away_table = parse_table(sections.get("away_table", ""), "AWAY", debug)
    result["away_table"] = away_table
    debug(f"✅ Found {len(away_table)} teams in away table", "success")
    
    # Parse form (LAST 6 MATCHES TABLE)
    form_data = parse_form(sections.get("form", ""), debug)
    result["form_data"] = form_data
    debug(f"✅ Found {len(form_data)} teams in form table", "success")
    
    elapsed = time.time() - start_time
    debug(f"⏱️ Parser completed in {elapsed:.2f} seconds", "success")
    
    # Show debug output
    st.markdown("### 🐛 Debug Output")
    debug_html = ""
    for entry in debug_log:
        icon = "ℹ️"
        if entry["type"] == "success":
            icon = "✅"
        elif entry["type"] == "error":
            icon = "❌"
        elif entry["type"] == "warning":
            icon = "⚠️"
        debug_html += f'<div class="debug-box"><span class="{entry["type"]}">{icon} {entry["msg"]}</span></div>'
    st.markdown(debug_html, unsafe_allow_html=True)
    
    return result


def split_into_sections(text: str, debug=None) -> dict:
    """Split text into sections."""
    result = {
        "predictions": "",
        "home_table": "",
        "away_table": "",
        "form": ""
    }
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    predictions_start = None
    home_table_start = None
    away_table_start = None
    form_start = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if re.match(r'^Round\s*\d+', line_stripped):
            if predictions_start is None:
                predictions_start = i
                log(f"  → Found 'Round' at line {i+1}", "highlight")
        
        if "HOME TABLE" in line_stripped:
            if home_table_start is None:
                home_table_start = i
                log(f"  → Found 'HOME TABLE' at line {i+1}", "highlight")
        
        if "AWAY TABLE" in line_stripped:
            if away_table_start is None:
                away_table_start = i
                log(f"  → Found 'AWAY TABLE' at line {i+1}", "highlight")
        
        if "LAST 6 MATCHES TABLE" in line_stripped:
            if form_start is None:
                form_start = i
                log(f"  → Found 'LAST 6 MATCHES TABLE' at line {i+1}", "highlight")
    
    # Extract sections
    if predictions_start is not None:
        end = home_table_start if home_table_start is not None else len(lines)
        result["predictions"] = '\n'.join(lines[predictions_start:end])
        log(f"  Predictions: lines {predictions_start+1} to {end}", "info")
    
    if home_table_start is not None:
        end = away_table_start if away_table_start is not None else len(lines)
        result["home_table"] = '\n'.join(lines[home_table_start:end])
        log(f"  Home Table: lines {home_table_start+1} to {end}", "info")
    
    if away_table_start is not None:
        end = form_start if form_start is not None else len(lines)
        result["away_table"] = '\n'.join(lines[away_table_start:end])
        log(f"  Away Table: lines {away_table_start+1} to {end}", "info")
    
    if form_start is not None:
        result["form"] = '\n'.join(lines[form_start:])
        log(f"  Form: lines {form_start+1} to {len(lines)}", "info")
    
    return result


def parse_predictions(text: str, debug=None) -> list:
    """Parse match predictions from the text."""
    matches = []
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    log(f"Processing {len(lines)} lines in Predictions section", "info")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this is a data line (contains 6+ digits and a degree symbol)
        if not re.search(r'\d{6,}.*°', line):
            i += 1
            continue
        
        # Remove all spaces
        cleaned = line.replace(' ', '')
        
        # Extract percentages (first 6 digits)
        pct_match = re.search(r'^(\d{2})(\d{2})(\d{2})', cleaned)
        if not pct_match:
            i += 1
            continue
        
        home_pct = int(pct_match.group(1))
        draw_pct = int(pct_match.group(2))
        away_pct = int(pct_match.group(3))
        
        # Get the rest after percentages
        rest = cleaned[6:]
        
        # ========== EXTRACT SCORE CODE ==========
        # Score code can be: 12, 21, X1, X2, etc.
        # X means Draw, followed by the first digit of the score
        score_match = re.search(r'^([X\d]{2})', rest)
        if not score_match:
            log(f"  ⚠️ Could not extract score code from: {line[:50]}", "warning")
            i += 1
            continue
        
        score_code = score_match.group(1)
        rest = rest[len(score_code):]  # Remove score code
        
        # Determine prediction and correct score
        if score_code[0] == 'X':
            # X means Draw, correct score is always 1-1
            prediction = 'X'
            correct_score_home = 1
            correct_score_away = 1
            # The second char might be part of the avg goals
            # For X1: rest starts with " - 1.889°..."
            # For X2: rest starts with " - 3.4116°..."
        elif score_code[0].isdigit() and score_code[1].isdigit():
            prediction = score_code[0]
            correct_score_home = int(score_code[0])
            correct_score_away = int(score_code[1])
        else:
            log(f"  ⚠️ Unknown score code: {score_code}", "warning")
            i += 1
            continue
        
        # ========== EXTRACT AVG GOALS AND TEMPERATURE ==========
        # Pattern: "-3.2325°" or "-1.889°" or "-02.1528°"
        avg_match = re.search(r'-?\s*(\d+\.\d+)°', rest)
        if not avg_match:
            log(f"  ⚠️ Could not extract avg_goals from: {line[:50]}", "warning")
            i += 1
            continue
        
        raw = avg_match.group(1)  # "13.2325", "1.889", "02.1528", etc.
        
        # Extract avg goals and temperature
        # Pattern: "3.2325" -> avg=3.23, temp=25
        #          "1.889" -> avg=1.88, temp=9
        #          "02.1528" -> avg=2.15, temp=28
        match = re.search(r'(\d+)\.(\d{2})(\d*)', raw)
        if match:
            int_part = int(match.group(1))
            dec_part = int(match.group(2))
            temp_str = match.group(3)
            
            # If int_part is > 10 and the score code is numeric, 
            # the first digit might be from the score code
            if int_part >= 10 and score_code[0].isdigit():
                # Remove the first digit (which is the score code's first digit)
                int_part = int_part % 10
            
            avg_goals = float(f"{int_part}.{dec_part:02d}")
            temperature = int(temp_str) if temp_str else 0
        else:
            log(f"  ⚠️ Could not parse avg/temp from: {raw}", "warning")
            i += 1
            continue
        
        # ========== EXTRACT COEFFICIENT ==========
        # Look for coefficient after the degree symbol
        coeff_match = re.search(r'°(\d+\.\d+)', rest)
        coefficient = float(coeff_match.group(1)) if coeff_match else None
        
        log(f"✅ Found: {home_pct}/{draw_pct}/{away_pct}, Code: {score_code}, "
            f"Pred: {prediction}, Score: {correct_score_home}-{correct_score_away}, "
            f"Avg: {avg_goals:.2f}, Temp: {temperature}, Coef: {coefficient}", "success")
        
        # ========== FIND TEAMS ==========
        home_team = None
        away_team = None
        date_line = None
        
        for j in range(i-1, max(0, i-15), -1):
            prev_line = lines[j].strip()
            
            # Check for date line
            if re.match(r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}', prev_line):
                date_line = prev_line
                # The team names are 2 lines before the date line
                if j - 2 >= 0:
                    home_candidate = lines[j-2].strip()
                    away_candidate = lines[j-1].strip()
                    
                    # Skip if "PRE" or "VIEW"
                    if home_candidate not in ["PRE", "VIEW"] and away_candidate not in ["PRE", "VIEW"]:
                        home_team = home_candidate
                        away_team = away_candidate
                        break
                    # If one is PRE/VIEW, try going further back
                    elif j - 4 >= 0:
                        home_candidate = lines[j-4].strip()
                        away_candidate = lines[j-3].strip()
                        if home_candidate not in ["PRE", "VIEW"] and away_candidate not in ["PRE", "VIEW"]:
                            home_team = home_candidate
                            away_team = away_candidate
                            break
        
        if not home_team or not away_team or not date_line:
            log(f"  ⚠️ Could not find teams or date", "warning")
            i += 1
            continue
        
        # Skip if home or away is "PRE" or "VIEW"
        if home_team in ["PRE", "VIEW"] or away_team in ["PRE", "VIEW"]:
            log(f"  ⚠️ Skipping preview lines", "warning")
            i += 1
            continue
        
        # Check for duplicate
        is_duplicate = False
        for existing in matches:
            if (existing["home_team"] == home_team and 
                existing["away_team"] == away_team and 
                existing["date"] == date_line):
                is_duplicate = True
                log(f"  ⚠️ Skipping duplicate: {home_team} vs {away_team}", "warning")
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
            "score_code": score_code,
            "prediction": prediction,
            "correct_score_home": correct_score_home,
            "correct_score_away": correct_score_away,
            "avg_goals": avg_goals,
            "temperature": temperature,
            "coefficient": coefficient
        })
        
        log(f"  ✅ Match added: {home_team} vs {away_team}", "success")
        i += 1
    
    log(f"Total matches extracted: {len(matches)}", "info")
    return matches


def parse_table(text: str, table_type: str, debug=None) -> dict:
    """Parse HOME TABLE or AWAY TABLE."""
    table_data = {}
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    in_table = False
    header_found = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a header line
        if "PTS" in line and "GP" in line and "W" in line and "D" in line and "L" in line:
            header_found = True
            in_table = True
            continue
        
        if not in_table:
            continue
        
        # Replace tabs and collapse spaces
        line = line.replace('\t', ' ')
        line = ' '.join(line.split())
        
        # Pattern: "1 Arsenal 47 19 15 2 2 41 11 30"
        match = re.search(r'^(\d+)\s+([A-Za-z\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)$', line)
        if not match:
            # Try simpler pattern: position, team, points, then numbers
            match = re.search(r'^(\d+)\s+([A-Za-z\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)', line)
        
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
            log(f"  Found {table_type} table: {position}. {team_name} ({points} pts)", "info")
    
    return table_data


def parse_form(text: str, debug=None) -> dict:
    """Parse LAST 6 MATCHES TABLE (form data)."""
    form_data = {}
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    in_form = False
    header_found = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a header line
        if "PTS" in line and "GP" in line and "W" in line and "D" in line and "L" in line:
            header_found = True
            in_form = True
            continue
        
        if not in_form:
            continue
        
        # Replace tabs and collapse spaces
        line = line.replace('\t', ' ')
        line = ' '.join(line.split())
        
        # Pattern: "1 Manchester United 16 6 5 1 0 12 5 7"
        match = re.search(r'^(\d+)\s+([A-Za-z\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)$', line)
        if not match:
            match = re.search(r'^(\d+)\s+([A-Za-z\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)', line)
        
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
            
            # Determine losing streak from losses
            # If losses >= 3, they might have a losing streak
            losing_streak = 0
            if losses >= 3:
                # Approximate: if they have 3+ losses in 6 games
                # They could be on a losing streak
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
                "form_points": points,  # Points from last 6 = form points
                "losing_streak": losing_streak
            }
            log(f"  Found form: {position}. {team_name} ({points} pts)", "info")
    
    return form_data


# ============================================================================
# CONVERT DATA TO ANALYSIS FORMAT
# ============================================================================
def convert_match_to_data(match: dict, home_table: dict, away_table: dict, form_data: dict, 
                          league: str = "Unknown") -> dict:
    """Convert extracted match data to analysis format."""
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
        "score_code": match.get('score_code'),
        "prediction": match.get('prediction'),
        "correct_score_home": match.get('correct_score_home'),
        "correct_score_away": match.get('correct_score_away'),
        "avg_goals": match.get('avg_goals', league_config["goals_fallback"]),
        "temperature": match.get('temperature'),
        "coefficient": match.get('coefficient'),
        "score_matrix": [],
        "is_finished": False,
        "actual_home": None,
        "actual_away": None
    }
    
    # Add home table data if available
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
    
    # Add away table data if available
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
    
    # Add form data if available
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
    
    # Set competitive blocks
    def get_block(position, table_type="home"):
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
    
    # Check relegation fight
    data["is_relegation_fight"] = (
        data["home_block"] == "relegation" or 
        data["away_block"] == "relegation"
    )
    
    # Create score matrix
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    elif data.get('score_code'):
        data['score_matrix'].append({
            "score": data['score_code'],
            "home_goals": None,
            "away_goals": None,
            "probability": 100.0
        })
    
    return data


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_match(data: dict) -> dict:
    """Universal logic analysis engine."""
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reasons": [],
        "is_lock": False,
        "lock_reason": None,
        "draw_conditions": {},
        "winner_selection": None,
        "winner_reason": None,
        "goal_bet": None,
        "goal_reason": None,
        "goal_accuracy": None,
        "warning": None,
        "warning_type": None,
    }
    
    # Extract values
    home_form = data.get("home_form_points", 0) or 0
    away_form = data.get("away_form_points", 0) or 0
    home_block = data.get("home_block")
    away_block = data.get("away_block")
    is_relegation_fight = data.get("is_relegation_fight", False)
    draw_pct = data.get("draw_pct", 0) or 0
    avg_goals = data.get("avg_goals", 2.0)
    prediction = data.get("prediction")
    home_losing_streak = data.get("home_losing_streak", 0) or 0
    away_losing_streak = data.get("away_losing_streak", 0) or 0
    home_position = data.get("home_position")
    away_position = data.get("away_position")
    
    form_diff = abs(home_form - away_form)
    
    # Check if draw is predicted
    is_draw_predicted = prediction == 'X' or draw_pct > 0
    
    same_block = home_block is not None and away_block is not None and home_block == away_block
    
    # Desperation check
    is_home_desperate = home_losing_streak >= 3 or home_block == "relegation"
    is_away_desperate = away_losing_streak >= 3 or away_block == "relegation"
    is_any_desperate = is_home_desperate or is_away_desperate
    
    # Form check
    form_similar = form_diff <= 2
    
    # Goals sweet spot
    goals_in_sweet_spot = 2.00 <= avg_goals <= 2.40
    
    # Dead rubber detection
    is_dead_rubber = False
    if (home_block == "mid" and away_block == "mid" and 
        not is_relegation_fight):
        is_dead_rubber = True
        result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for"
        result["warning_type"] = "dead_rubber"
    
    # Draw conditions
    draw_conditions = {
        "form_similar": form_similar,
        "goals_sweet_spot": goals_in_sweet_spot,
        "same_block": same_block,
        "no_desperation": not is_any_desperate,
    }
    result["draw_conditions"] = draw_conditions
    all_draw_conditions_met = all(draw_conditions.values())
    
    # Winner selection
    winner_selection = "HOME"
    
    # PRIORITY 1: Home Desperation Lock
    if is_home_desperate and not is_away_desperate:
        winner_selection = "HOME"
        result["winner_reason"] = "🏆 HOME TEAM DESPERATE → 100% accuracy"
        result["is_lock"] = True
        result["lock_reason"] = "Home team desperate (losing streak 3+ or relegation fight)"
    
    # PRIORITY 2: 65% Rule when draw fails
    elif is_draw_predicted and not all_draw_conditions_met:
        winner_selection = "HOME"
        result["winner_reason"] = "65% RULE: When draw fails, home wins 65% of the time"
    
    # PRIORITY 3: Form advantage
    elif home_form - away_form >= 3:
        winner_selection = "HOME"
        result["winner_reason"] = f"Home team better form: {home_form} vs {away_form} points → 89% accuracy"
    elif away_form - home_form >= 3 and is_away_desperate:
        winner_selection = "AWAY"
        result["winner_reason"] = f"Away team better form ({away_form} vs {home_form}) and desperate → 83% accuracy"
    
    # Default
    if "winner_selection" not in result or result["winner_selection"] is None:
        winner_selection = "HOME"
        result["winner_reason"] = "Default: Home team wins 65% of non-draws"
    
    result["winner_selection"] = winner_selection
    
    # Goal bets
    goal_bet = None
    goal_accuracy = None
    goal_reason = None
    
    if avg_goals < 2.00:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "95%"
        goal_reason = f"Avg goals {avg_goals:.2f} < 2.00 → UNDER 2.5 is a LOCK"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals < 2.00 → UNDER 2.5 (95% accuracy)"
    
    elif avg_goals > 3.00 and is_draw_predicted:
        goal_bet = "OVER 2.5"
        goal_accuracy = "100%"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Draw Prediction → OVER 2.5 is a LOCK"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals > 3.00 + Draw → OVER 2.5 (100% accuracy)"
    
    elif avg_goals > 3.00 and winner_selection == "AWAY":
        goal_bet = "OVER 2.5"
        goal_accuracy = "80%"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Away Win → OVER 2.5"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals > 3.00 + Away Win → OVER 2.5 (80% accuracy)"
    
    elif avg_goals > 3.00 and winner_selection == "HOME":
        goal_bet = "OVER 2.5"
        goal_accuracy = "62%"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Home Win → OVER 2.5 (62%)"
    
    elif 2.00 <= avg_goals <= 2.40:
        if is_draw_predicted:
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80%"
            goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot + Draw → UNDER 2.5"
        elif winner_selection == "HOME":
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80%"
            goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot + Home Win → UNDER 2.5"
        else:
            goal_bet = "SKIP"
            goal_accuracy = "50%"
            goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot + Away Win → SKIP (50%)"
    
    result["goal_bet"] = goal_bet
    result["goal_reason"] = goal_reason
    result["goal_accuracy"] = goal_accuracy
    
    # Final decision
    if is_draw_predicted and all_draw_conditions_met:
        result["primary_bet"] = {
            "market": "DRAW" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else ""),
            "reason": "ALL 4 draw conditions met: Form similar, goals in sweet spot, same block, no desperation",
            "historical_accuracy": "57%",
        }
        result["classification"] = "DRAW" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else "")
        result["verdict"] = "LOCK" if result["is_lock"] else "RECOMMENDED"
    
    elif is_draw_predicted and not all_draw_conditions_met:
        if is_dead_rubber:
            result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for - proceed with caution"
        
        result["primary_bet"] = {
            "market": f"DOUBLE CHANCE: {winner_selection} or Draw" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else ""),
            "reason": f"Draw conditions not fully met → Double Chance wins 83% when draw predicted. {result.get('winner_reason', '')}",
            "historical_accuracy": "83%",
        }
        result["classification"] = f"DOUBLE CHANCE: {winner_selection}" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else "")
        result["verdict"] = "LOCK" if result["is_lock"] else "RECOMMENDED"
    
    elif not is_draw_predicted:
        if is_dead_rubber:
            result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for - results may be unpredictable"
        
        result["primary_bet"] = {
            "market": f"{winner_selection} WIN" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else ""),
            "reason": result.get("winner_reason", "Default selection"),
            "historical_accuracy": "89%" if "better form" in result.get("winner_reason", "") else "100%" if "DESPERATE" in result.get("winner_reason", "") else "65%",
        }
        result["classification"] = f"{winner_selection} WIN" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else "")
        result["verdict"] = "LOCK" if result["is_lock"] else "RECOMMENDED"
    
    if not result.get("primary_bet"):
        result["skip_reasons"].append("No clear betting signal from universal logic")
        result["verdict"] = "SKIP"
    
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
        elif 'OVER 3.5' in bet:
            if total > 3:
                correct_count += 1
        elif 'BTTS' in bet:
            if home > 0 and away > 0:
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
            "score_code": data.get("score_code"),
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
            "bet_market": primary["market"] if primary else "SKIP",
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
    """Display analysis results for a single match."""
    
    badge_class = get_league_badge(league)
    st.markdown(f'<span class="league-badge {badge_class}">{league}</span>', unsafe_allow_html=True)
    
    if analysis.get("warning"):
        st.markdown(f'<div class="dead-rubber-warning">{analysis["warning"]}</div>', unsafe_allow_html=True)
    
    if analysis.get("is_lock"):
        st.success(f"🔒 LOCK SIGNAL: {analysis.get('lock_reason', '')}")
    
    is_draw_predicted = data.get("prediction") == 'X' or (data.get("draw_pct", 0) or 0) > 0
    
    if is_draw_predicted:
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
            st.success("✅ ALL 4 conditions met → BET THE DRAW (57% accuracy)")
        else:
            st.info("⚠️ Not all conditions met → Double Chance or Exact Winner recommended")
    
    v = analysis["verdict"]
    if v == "LOCK": 
        st.success(f"🔒 LOCK: {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}")
    elif v == "RECOMMENDED": 
        st.success(f"✅ RECOMMENDED: {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}")
    else: 
        st.warning(f"⚠️ SKIP: {data.get('home_team', 'Unknown')} vs {data.get('away_team', 'Unknown')}")
    
    st.markdown(f"**Classification: {analysis.get('classification', 'Unknown')}**")
    
    if analysis.get("winner_reason"):
        st.markdown(f"**Winner Selection:** {analysis.get('winner_selection', 'Unknown')} — {analysis.get('winner_reason', '')}")
    
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
    
    if analysis.get("goal_reason"):
        st.info(f"⚽ Goal Bet: {analysis.get('goal_bet')} — {analysis.get('goal_reason')} (Accuracy: {analysis.get('goal_accuracy', 'N/A')})")
    
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
                else:
                    bg = "#1a2a1a"
                    st.markdown(f'<div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;"><div style="font-size:1.2rem; font-weight:800;">{s.get("score", "?")}</div><div style="font-size:0.7rem; color:#94a3b8;">Double Chance</div></div>', unsafe_allow_html=True)
    
    if analysis.get("primary_bet"):
        p = analysis["primary_bet"]
        if analysis.get("is_lock"):
            card_class = "lock-card"
            badge = f'<span class="lock-badge">🔒 LOCK — {analysis.get("lock_reason", "")}</span>'
        elif analysis.get("warning_type") == "dead_rubber":
            card_class = "dead-rubber-card"
            badge = f'<span class="accuracy-badge">⚠️ DEAD RUBBER — {p.get("historical_accuracy", "")}</span>'
        else:
            card_class = "primary-card"
            badge = f'<span class="accuracy-badge">📊 {p.get("historical_accuracy", "")}</span>'
        
        st.markdown(f'<div class="section-label">🎯 PRIMARY BET</div><div class="output-card {card_class}"><div style="display:flex;align-items:center;gap:1rem;"><div style="font-size:2.5rem;">{"🔒" if analysis.get("is_lock") else "🔥"}</div><div style="flex:1;"><div style="font-size:1.3rem;font-weight:800;">{p.get("market", "Unknown")}</div><div style="font-size:0.8rem;color:#64748b;">{p.get("reason", "")}</div><div style="margin-top:0.5rem;">{badge}</div></div></div></div>', unsafe_allow_html=True)
    
    if analysis["verdict"] == "SKIP":
        st.markdown(f'<div class="output-card skip-card"><div class="verdict-skip"><div class="big-text">⚠️ SKIP — NO BET</div><p style="color:#94a3b8;">{"<br>".join(analysis.get("skip_reasons", []))}</p></div></div>', unsafe_allow_html=True)
    
    if data.get("home_losing_streak", 0) >= 3:
        st.markdown(f'<div class="priority-rule">🏆 PRIORITY RULE: {data.get("home_team", "Home")} has {data.get("home_losing_streak", 0)} game losing streak → BET HOME WIN (100% accuracy)</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V9.0")
    st.caption("Universal Logic Engine | Complete Rewrite")

    with st.expander("📖 The Universal Truths", expanded=False):
        st.markdown("""
        **TRUTH #1: Draw Predictions Are Wrong 83% of the Time**  
        When Forebet predicts a draw, it happens only 17% of the time.

        **TRUTH #2: When Draw Prediction Fails — Home Wins 65% of the Time**

        **TRUTH #3: Form Difference is the #1 Indicator**  
        If form difference is >2 points, someone ALMOST ALWAYS wins.

        **TRUTH #4: Average Goals has a "Sweet Spot"**  
        Draws only happen when average goals are between 2.00 and 2.40.

        **TRUTH #5: Same Competitive Block Matters**  
        Draws happen when both teams have the SAME motivation.

        **TRUTH #6: Desperation Kills Draws**  
        Desperate teams don't play for draws.

        **TRUTH #7: The Goal Locks**  
        • Avg Goals < 2.00 → UNDER 2.5 (95% accuracy)  
        • Avg Goals > 3.00 + Draw → OVER 2.5 (100% accuracy)  
        • Avg Goals > 3.00 + Away Win → OVER 2.5 (80% accuracy)

        **PRIORITY RULE: Home Desperation = 100% Accuracy**  
        When the home team is desperate (losing streak 3+ or relegation fight), BET HOME WIN.
        """)

    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")

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

        if st.button("🔮 ANALYZE V9.0", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with Universal Logic..."):
                        parsed = parse_text_data(text_data)

                    league = parsed.get("league", "Unknown League")
                    matches = parsed.get("matches", [])
                    home_table = parsed.get("home_table", {})
                    away_table = parsed.get("away_table", {})
                    form_data = parsed.get("form_data", {})

                    if matches:
                        st.success(f"✅ Found {len(matches)} matches in {league}")

                        if home_table:
                            st.markdown("### 🏆 Home Table (Top 5)")
                            home_df = pd.DataFrame([
                                {"Pos": data["position"], "Team": team, "Pts": data["points"], 
                                 "W": data["wins"], "D": data["draws"], "L": data["losses"], 
                                 "GF": data["gf"], "GA": data["ga"], "GD": data["gd"]}
                                for team, data in list(home_table.items())[:5]
                            ])
                            st.dataframe(home_df, use_container_width=True, hide_index=True)

                        if away_table:
                            st.markdown("### 🏆 Away Table (Top 5)")
                            away_df = pd.DataFrame([
                                {"Pos": data["position"], "Team": team, "Pts": data["points"], 
                                 "W": data["wins"], "D": data["draws"], "L": data["losses"], 
                                 "GF": data["gf"], "GA": data["ga"], "GD": data["gd"]}
                                for team, data in list(away_table.items())[:5]
                            ])
                            st.dataframe(away_df, use_container_width=True, hide_index=True)

                        for idx, match in enumerate(matches):
                            st.markdown(f"### Match {idx + 1}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")

                            home = match.get('home_team')
                            away = match.get('away_team')
                            
                            # Show prediction details
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prediction", match.get('prediction', '?'))
                            with col2:
                                st.metric("Correct Score", f"{match.get('correct_score_home', '?')}-{match.get('correct_score_away', '?')}")
                            with col3:
                                st.metric("Avg Goals", f"{match.get('avg_goals', 0):.2f}")

                            data = convert_match_to_data(match, home_table, away_table, form_data, league)
                            analysis = analyze_match(data)
                            save_to_db(data, analysis, league)

                            display_analysis(data, analysis, league)

                            if idx < len(matches) - 1:
                                st.markdown("---")
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

                badge = "🔒 LOCK" if is_lock else "📊"

                with st.expander(f"{badge} | {ht} vs {at} — Predicted: {pred}"):
                    if warning:
                        st.warning(warning)
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
            skip_count = sum(1 for r in results if r.get('bet_market') == 'SKIP')
            bet_count = total - skip_count

            correct = 0
            incorrect = 0
            lock_correct = 0
            lock_total = 0

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
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Tracked</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{bet_count}</div><div class="stat-label">Bets Placed</div></div>', unsafe_allow_html=True)
            with col3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{skip_count}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
            with col5:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct Bets</div></div>', unsafe_allow_html=True)

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

                if pred == 'SKIP':
                    badge = '<span class="skip-badge">⚪ SKIP</span>'
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"

                match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"

                rows.append({
                    "Date": r.get("match_date", ""),
                    "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
                    "Match": match_display,
                    "Class": r.get("classification", ""),
                    "Bet": primary_pred if pred != 'SKIP' else "SKIP",
                    "Score": score_display,
                    "Result": badge,
                })

            df = pd.DataFrame(rows)
            st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
