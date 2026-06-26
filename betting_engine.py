"""
MATCH ANALYZER V8.2 — COMPLETE FIX
Fixed: Avg goals parsing, X1/X2 score codes, Form parsing, Team detection, Duplicate matches
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
st.set_page_config(page_title="Match Analyzer V8.2", page_icon="📊", layout="wide")

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
    elif "Brazil" in league or "Serie A" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.66
    elif "Premier" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.75
    else:
        config["relegation_threshold"] = 15
        config["league_size"] = 20
        config["goals_fallback"] = 2.50
    
    return config


def detect_league(text: str) -> str:
    if "Brasileiro Serie A" in text or "Brazil" in text or "Br1" in text:
        return "Brazil Serie A"
    elif "Premier League" in text or "England" in text:
        return "Premier League"
    elif "Eliteserien" in text or "Norway" in text:
        return "Norway Eliteserien"
    elif "LaLiga" in text or "Spain" in text:
        return "La Liga"
    elif "Serie A" in text and "Italy" in text:
        return "Serie A"
    elif "Bundesliga" in text or "Germany" in text:
        return "Bundesliga"
    else:
        return "Unknown League"


# ============================================================================
# TEXT PARSER WITH DETAILED DEBUG
# ============================================================================
def parse_text_data(text: str) -> dict:
    """Parse the clean text format containing Predictions, Form, and Statistics."""
    start_time = time.time()
    
    # Create debug log
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
        "form_data": {},
        "statistics": {},
        "standings": {}
    }
    
    # ============================================================
    # SECTION 1: Split text into sections
    # ============================================================
    sections = split_into_sections(text, debug)
    debug(f"📂 Sections found: Predictions={len(sections.get('predictions', '')):,} chars, Form={len(sections.get('form', '')):,} chars, Statistics={len(sections.get('statistics', '')):,} chars", "info")
    
    # ============================================================
    # SECTION 2: Parse Predictions
    # ============================================================
    matches = parse_predictions(sections.get("predictions", ""), debug)
    result["matches"] = matches
    debug(f"✅ Found {len(matches)} matches", "success")
    
    # ============================================================
    # SECTION 3: Parse Form
    # ============================================================
    form_data = parse_form(sections.get("form", ""), debug)
    result["form_data"] = form_data
    debug(f"✅ Found {len(form_data)} teams with form data", "success")
    
    # ============================================================
    # SECTION 4: Parse Statistics
    # ============================================================
    stats_data = parse_statistics(sections.get("statistics", ""), debug)
    result["statistics"] = stats_data
    debug(f"✅ Statistics parsed", "success")
    
    # Extract standings from statistics
    standings = parse_standings(sections.get("statistics", ""), debug)
    result["standings"] = standings
    debug(f"📊 Found {len(standings)} teams in standings", "success")
    
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
    """Split the text into Predictions, Form, and Statistics sections."""
    result = {"predictions": "", "form": "", "statistics": ""}
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    predictions_start = None
    form_start = None
    stats_start = None
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        if re.match(r'^Round\s*\d+', line_stripped):
            if predictions_start is None:
                predictions_start = i
                log(f"  → Found 'Round' at line {i+1}", "highlight")
        
        if re.match(r'^\d+\.\s*[A-Za-z]', line_stripped):
            if form_start is None and predictions_start is not None and i > predictions_start + 5:
                form_start = i
                log(f"  → Found form start at line {i+1}", "highlight")
        
        if "Home wins / Draws / Away wins" in line_stripped or "Best attack" in line_stripped:
            if stats_start is None:
                stats_start = i
                log(f"  → Found stats start at line {i+1}", "highlight")
    
    if predictions_start is not None:
        end = form_start if form_start is not None else stats_start if stats_start is not None else len(lines)
        result["predictions"] = '\n'.join(lines[predictions_start:end])
        log(f"  Predictions: lines {predictions_start+1} to {end}", "info")
    else:
        log(f"  ⚠️ Predictions section NOT FOUND!", "error")
    
    if form_start is not None:
        end = stats_start if stats_start is not None else len(lines)
        result["form"] = '\n'.join(lines[form_start:end])
        log(f"  Form: lines {form_start+1} to {end}", "info")
    else:
        log(f"  ⚠️ Form section NOT FOUND!", "warning")
    
    if stats_start is not None:
        result["statistics"] = '\n'.join(lines[stats_start:])
        log(f"  Statistics: lines {stats_start+1} to {len(lines)}", "info")
    else:
        log(f"  ⚠️ Statistics section NOT FOUND!", "warning")
    
    return result


def parse_predictions(text: str, debug=None) -> list:
    """Parse predictions from the text format - COMPLETE FIX."""
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
        
        home_win_pct = int(pct_match.group(1))
        draw_pct = int(pct_match.group(2))
        away_win_pct = int(pct_match.group(3))
        
        # Get the rest after percentages
        rest = cleaned[6:]
        
        # ========== EXTRACT SCORE CODE ==========
        # Score code can be: 12, X2, X1, 21, etc.
        score_match = re.search(r'^([X\d]{2})', rest)
        if not score_match:
            log(f"  ⚠️ Could not extract score code from: {line[:50]}", "warning")
            i += 1
            continue
        
        score_code = score_match.group(1)
        rest = rest[len(score_code):]  # Remove score code
        
        # Determine prediction
        if score_code == 'X1':
            prediction = 'X'
            correct_score_home = 1
            correct_score_away = 0
        elif score_code == 'X2':
            prediction = 'X'
            correct_score_home = 0
            correct_score_away = 1
        elif score_code[0].isdigit() and score_code[1].isdigit():
            prediction = score_code[0]
            correct_score_home = int(score_code[0])
            correct_score_away = int(score_code[1])
        else:
            log(f"  ⚠️ Unknown score code: {score_code}", "warning")
            i += 1
            continue
        
        # ========== EXTRACT AVG GOALS AND TEMPERATURE ==========
        # Pattern: "-3.2325°-" or "-02.1528°" or " - 13.2325°-"
        avg_match = re.search(r'-?\s*(\d+\.\d+)°', rest)
        if not avg_match:
            log(f"  ⚠️ Could not extract avg_goals from: {line[:50]}", "warning")
            i += 1
            continue
        
        raw = avg_match.group(1)  # "13.2325", "23.4116", "02.1528", etc.
        
        # Remove the score code's first digit if it's attached
        if raw[0].isdigit() and score_code[0].isdigit() and raw[0] == score_code[0]:
            raw = raw[1:]  # "13.2325" -> "3.2325"
        
        # Extract avg and temperature
        # Pattern: "3.2325" -> avg=3.23, temp=25
        #          "02.1528" -> avg=2.15, temp=28
        #          "2.1528" -> avg=2.15, temp=28
        match = re.search(r'(\d+)\.(\d{2})(\d*)', raw)
        if match:
            int_part = int(match.group(1))
            dec_part = int(match.group(2))
            temp_str = match.group(3)
            avg_goals = float(f"{int_part}.{dec_part:02d}")
            temperature = int(temp_str) if temp_str else 0
            
            # Double-check: if avg_goals is > 10, we probably missed removing a digit
            if avg_goals > 10:
                # Try removing first digit
                raw2 = raw[1:]
                match2 = re.search(r'(\d+)\.(\d{2})(\d*)', raw2)
                if match2:
                    int_part2 = int(match2.group(1))
                    dec_part2 = int(match2.group(2))
                    avg_goals = float(f"{int_part2}.{dec_part2:02d}")
        else:
            log(f"  ⚠️ Could not parse avg/temp from: {raw}", "warning")
            i += 1
            continue
        
        log(f"✅ Found: {home_win_pct}/{draw_pct}/{away_win_pct}, Code: {score_code}, Avg: {avg_goals:.2f}, Temp: {temperature}", "success")
        
        # ========== FIND TEAMS ==========
        # Look backwards for teams
        # Format: Home team, Away team, Date line
        home_team = None
        away_team = None
        date_line = None
        
        # Skip the current line and look backwards
        # We need to skip "PRE" and "VIEW" lines
        for j in range(i-1, max(0, i-15), -1):
            prev_line = lines[j].strip()
            
            # Check for date line
            if re.match(r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}', prev_line):
                date_line = prev_line
                # The team names are 2 lines before the date line
                # But skip "PRE" and "VIEW" lines
                if j - 2 >= 0:
                    home_candidate = lines[j-2].strip()
                    away_candidate = lines[j-1].strip()
                    
                    # Skip if "PRE" or "VIEW"
                    if home_candidate not in ["PRE", "VIEW"] and away_candidate not in ["PRE", "VIEW"]:
                        home_team = home_candidate
                        away_team = away_candidate
                        break
                    # If one of them is PRE/VIEW, try going further back
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
        
        # Skip if home or away is "PRE" or "VIEW" (redundant check)
        if home_team in ["PRE", "VIEW"] or away_team in ["PRE", "VIEW"]:
            log(f"  ⚠️ Skipping preview lines", "warning")
            i += 1
            continue
        
        # Check for duplicate match (same home, away, date)
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
            "home_win_pct": home_win_pct,
            "draw_pct": draw_pct,
            "away_win_pct": away_win_pct,
            "prediction": prediction,
            "score_code": score_code,
            "correct_score_home": correct_score_home,
            "correct_score_away": correct_score_away,
            "avg_goals": avg_goals,
            "temperature": temperature
        })
        
        log(f"  ✅ Match added: {home_team} vs {away_team}", "success")
        i += 1
    
    log(f"Total matches extracted: {len(matches)}", "info")
    return matches


def parse_form(text: str, debug=None) -> dict:
    """Parse form data from the text format - COMPLETE FIX."""
    form_data = {}
    lines = text.split('\n')
    
    def log(msg, type="info"):
        if debug:
            debug(f"  {msg}", type)
    
    log(f"Processing {len(lines)} lines in Form section", "info")
    
    in_top_form = False
    in_bottom_form = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if "Top 5 in form teams" in line or "Top form teams" in line:
            in_top_form = True
            in_bottom_form = False
            continue
        
        if "Worst form teams" in line or "Bottom form teams" in line:
            in_bottom_form = True
            in_top_form = False
            continue
        
        if "Team" in line and "Form" in line and "Points" in line:
            continue
        
        # Parse form data line
        # Pattern: "1.  RB Bragantino   WWWLWL  12"
        match = re.search(r'^(\d+)\.\s+([A-Za-z\s]+?)\s+([DWL]+)\s+(\d+)$', line)
        if not match:
            # Try with tabs
            line = line.replace('\t', ' ')
            match = re.search(r'^(\d+)\.\s+([A-Za-z\s]+?)\s+([DWL]+)\s+(\d+)$', line)
        
        if match:
            position = int(match.group(1))
            team_name = match.group(2).strip()
            form_sequence = match.group(3).strip()
            points = int(match.group(4))
            
            results = list(form_sequence)
            wins = results.count('W')
            draws = results.count('D')
            losses = results.count('L')
            total = wins + draws + losses
            
            if total == 0:
                continue
            
            losing_streak = 0
            for r in reversed(results):
                if r == 'L':
                    losing_streak += 1
                else:
                    break
            
            last_5 = results[-5:] if len(results) >= 5 else results
            form_points = sum(3 if x == 'W' else 1 if x == 'D' else 0 for x in last_5)
            
            form_data[team_name] = {
                "position": position,
                "points": points,
                "form_points": form_points,
                "losing_streak": losing_streak,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "games_played": total,
                "form_sequence": form_sequence
            }
            log(f"  Found form data for {team_name}: {form_sequence} ({points} pts)", "info")
    
    if debug:
        debug(f"Found {len(form_data)} teams with form data", "info")
    
    return form_data


def parse_statistics(text: str, debug=None) -> dict:
    """Parse the Statistics section."""
    stats = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if "Home wins:" in line:
            match = re.search(r'Home wins:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["home_wins"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
        
        if "Draws:" in line and "Home wins" not in line:
            match = re.search(r'Draws:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["draws"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
        
        if "Away wins:" in line:
            match = re.search(r'Away wins:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["away_wins"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
        
        if "Under 2.5 goals:" in line:
            match = re.search(r'Under 2.5 goals:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["under_25"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
        
        if "Over 2.5 goals:" in line:
            match = re.search(r'Over 2.5 goals:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["over_25"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
        
        if "Goals per game:" in line:
            match = re.search(r'Goals per game:\s*(\d+\.\d+)', line)
            if match:
                stats["goals_per_game"] = float(match.group(1))
        
        if "Both teams scored games:" in line:
            match = re.search(r'Both teams scored games:\s*(\d+).*?(\d+)%', line)
            if match:
                stats["btts"] = {"count": int(match.group(1)), "percentage": int(match.group(2))}
    
    return stats


def parse_standings(text: str, debug=None) -> dict:
    """Extract standings from Statistics section."""
    standings = {}
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        line = line.replace('\t', ' ')
        
        match = re.search(r'^(\d+)\.\s+([A-Za-z\s]+?)\s+(\d+)', line)
        if match:
            position = int(match.group(1))
            team_name = match.group(2).strip()
            
            points_match = re.search(r'(\d+)$', line)
            if points_match:
                points = int(points_match.group(1))
                standings[team_name] = {
                    "position": position,
                    "points": points
                }
    
    return standings


# ============================================================================
# CONVERT DATA TO ANALYSIS FORMAT
# ============================================================================
def convert_match_to_data(match: dict, standings: dict, form_data: dict, league: str = "Unknown") -> dict:
    """Convert extracted match data to analysis format"""
    league_config = get_league_config(league)
    home_team = match.get('home_team', 'Unknown')
    away_team = match.get('away_team', 'Unknown')
    
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals_total": match.get('home_goals', 0),
        "away_goals_total": match.get('away_goals', 0),
        "prediction": match.get('prediction'),
        "score_code": match.get('score_code'),
        "correct_score_home": match.get('correct_score_home'),
        "correct_score_away": match.get('correct_score_away'),
        "avg_goals": match.get('avg_goals', league_config["goals_fallback"]),
        "score_matrix": [],
        "home_win_pct": match.get('home_win_pct'),
        "draw_pct": match.get('draw_pct'),
        "away_win_pct": match.get('away_win_pct'),
        "match_url": match.get('match_url'),
        "actual_home": match.get('actual_home'),
        "actual_away": match.get('actual_away'),
        "is_finished": match.get('is_finished', False),
    }
    
    # Add standings data if available
    if home_team in standings:
        data["home_standings_position"] = standings[home_team]["position"]
        data["home_standings_points"] = standings[home_team]["points"]
    if away_team in standings:
        data["away_standings_position"] = standings[away_team]["position"]
        data["away_standings_points"] = standings[away_team]["points"]
    
    # Add form data if available
    if home_team in form_data:
        data["home_form_points"] = form_data[home_team]["form_points"]
        data["home_losing_streak"] = form_data[home_team]["losing_streak"]
        data["home_games_played"] = form_data[home_team]["games_played"]
        data["home_form_sequence"] = form_data[home_team].get("form_sequence", "")
    if away_team in form_data:
        data["away_form_points"] = form_data[away_team]["form_points"]
        data["away_losing_streak"] = form_data[away_team]["losing_streak"]
        data["away_games_played"] = form_data[away_team]["games_played"]
        data["away_form_sequence"] = form_data[away_team].get("form_sequence", "")
    
    # Set competitive blocks
    def get_block(position):
        if position is None:
            return None
        try:
            pos = int(position)
            league_size = league_config["league_size"]
            relegation_threshold = league_config["relegation_threshold"]
            
            if pos <= 4:
                return "europe"
            elif pos >= relegation_threshold:
                return "relegation"
            else:
                return "mid"
        except:
            return None
    
    data["competitive_block_home"] = get_block(data.get("home_standings_position"))
    data["competitive_block_away"] = get_block(data.get("away_standings_position"))
    
    # Check relegation fight
    data["is_relegation_fight"] = (
        data["competitive_block_home"] == "relegation" or 
        data["competitive_block_away"] == "relegation"
    )
    
    # Create score matrix from correct score
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    
    return data


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_match_v8(data: dict) -> dict:
    """Universal logic analysis engine"""
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
    
    # Extract all needed values safely
    home_form = data.get("home_form_points", 0) or 0
    away_form = data.get("away_form_points", 0) or 0
    home_goals = data.get("home_goals_total", 0) or 0
    away_goals = data.get("away_goals_total", 0) or 0
    home_losing_streak = data.get("home_losing_streak", 0) or 0
    away_losing_streak = data.get("away_losing_streak", 0) or 0
    home_block = data.get("competitive_block_home")
    away_block = data.get("competitive_block_away")
    is_relegation_fight = data.get("is_relegation_fight", False)
    
    # Get draw_pct safely
    draw_pct = data.get("draw_pct", 0)
    if draw_pct is None:
        draw_pct = 0
    
    # Get avg goals
    avg_goals = data.get("avg_goals", 2.0)
    if avg_goals is None or avg_goals == 0:
        avg_goals = 2.0
    
    form_diff = abs(home_form - away_form)
    
    # Check if draw is predicted
    pred = data.get("prediction")
    score_code = data.get("score_code", "")
    is_draw_predicted = (draw_pct is not None and draw_pct > 0) or pred == 'X' or 'X' in score_code
    
    same_block = home_block is not None and away_block is not None and home_block == away_block
    
    # Desperation check
    is_home_desperate = home_losing_streak >= 3 or is_relegation_fight
    is_away_desperate = away_losing_streak >= 3 or is_relegation_fight
    is_any_desperate = is_home_desperate or is_away_desperate
    
    # Form check
    form_similar = form_diff <= 2
    
    # Goals sweet spot
    goals_in_sweet_spot = 2.00 <= avg_goals <= 2.40
    
    # Dead rubber detection
    is_dead_rubber = False
    if (home_block == "mid" and away_block == "mid" and 
        not is_relegation_fight and
        not data.get("is_title_race", False)):
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
        result["winner_reason"] = "🏆 HOME TEAM DESPERATE → 100% accuracy (7/7 in dataset)"
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
        goal_accuracy = "95% (20/21 in dataset)"
        goal_reason = f"Avg goals {avg_goals:.2f} < 2.00 → UNDER 2.5 is a LOCK"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals < 2.00 → UNDER 2.5 (95% accuracy)"
    
    elif avg_goals > 3.00 and is_draw_predicted:
        goal_bet = "OVER 2.5"
        goal_accuracy = "100% (2/2 in dataset)"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Draw Prediction → OVER 2.5 is a LOCK"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals > 3.00 + Draw → OVER 2.5 (100% accuracy)"
    
    elif avg_goals > 3.00 and winner_selection == "AWAY":
        goal_bet = "OVER 2.5"
        goal_accuracy = "80% (4/5 in dataset)"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Away Win → OVER 2.5"
        result["is_lock"] = True
        result["lock_reason"] = "Avg goals > 3.00 + Away Win → OVER 2.5 (80% accuracy)"
    
    elif avg_goals > 3.00 and winner_selection == "HOME":
        goal_bet = "OVER 2.5"
        goal_accuracy = "62% (in dataset)"
        goal_reason = f"Avg goals {avg_goals:.2f} > 3.00 + Home Win → OVER 2.5 (62%)"
    
    elif 2.00 <= avg_goals <= 2.40:
        if is_draw_predicted:
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80% (in dataset)"
            goal_reason = f"Avg goals {avg_goals:.2f} in sweet spot + Draw → UNDER 2.5"
        elif winner_selection == "HOME":
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80% (in dataset)"
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
            "historical_accuracy": "57% (4/7 in dataset)",
        }
        result["classification"] = "DRAW" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else "")
        result["verdict"] = "LOCK" if result["is_lock"] else "RECOMMENDED"
    
    elif is_draw_predicted and not all_draw_conditions_met:
        if is_dead_rubber:
            result["warning"] = "⚠️ DEAD RUBBER: Both teams have nothing to play for - proceed with caution"
        
        result["primary_bet"] = {
            "market": f"DOUBLE CHANCE: {winner_selection} or Draw" + (f" | {goal_bet}" if goal_bet and goal_bet != "SKIP" else ""),
            "reason": f"Draw conditions not fully met → Double Chance wins 83% when draw predicted. {result.get('winner_reason', '')}",
            "historical_accuracy": "83% (24/29 in dataset)",
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
            "match_date": str(date.today()),
            "league": league,
            "home_goals_total": data.get("home_goals_total"),
            "away_goals_total": data.get("away_goals_total"),
            "combined_goals": (data.get("home_goals_total") or 0) + (data.get("away_goals_total") or 0),
            "home_form_points": data.get("home_form_points"),
            "away_form_points": data.get("away_form_points"),
            "home_form_sequence": data.get("home_form_sequence"),
            "away_form_sequence": data.get("away_form_sequence"),
            "form_difference": abs((data.get("home_form_points") or 0) - (data.get("away_form_points") or 0)),
            "same_block": data.get("competitive_block_home") == data.get("competitive_block_away"),
            "is_relegation_fight": data.get("is_relegation_fight", False),
            "home_losing_streak": data.get("home_losing_streak", 0),
            "away_losing_streak": data.get("away_losing_streak", 0),
            "is_lock": analysis.get("is_lock", False),
            "lock_reason": analysis.get("lock_reason"),
            "prediction": primary["market"] if primary else "SKIP",
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
            "home_win_pct": data.get("home_win_pct"),
            "draw_pct": data.get("draw_pct"),
            "away_win_pct": data.get("away_win_pct"),
            "score_code": data.get("score_code"),
            "avg_goals": data.get("avg_goals"),
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
            "actual_home_goals": home_goals, "actual_away_goals": away_goals,
            "actual_total_goals": total, "actual_over25": total > 2,
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
    elif "Premier" in league:
        return "uk"
    elif "La Liga" in league:
        return "es"
    elif "Serie A" in league and "Italy" in league:
        return "it"
    elif "Bundesliga" in league:
        return "de"
    else:
        return "unknown"


def display_analysis(data: dict, analysis: dict, league: str = "Unknown"):
    """Display analysis results for a single match"""
    
    # League badge
    badge_class = get_league_badge(league)
    st.markdown(f'<span class="league-badge {badge_class}">{league}</span>', unsafe_allow_html=True)
    
    # Display warnings first
    if analysis.get("warning"):
        st.markdown(f'<div class="dead-rubber-warning">{analysis["warning"]}</div>', unsafe_allow_html=True)
    
    # Display Lock status
    if analysis.get("is_lock"):
        st.success(f"🔒 LOCK SIGNAL: {analysis.get('lock_reason', '')}")
    
    # Check if draw is predicted
    draw_value = data.get("draw_pct", 0)
    prediction = data.get("prediction")
    score_code = data.get("score_code", "")
    is_draw_predicted = (draw_value is not None and draw_value > 0) or prediction == 'X' or 'X' in score_code
    
    # Display Draw Conditions if draw was predicted
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
    
    # Key Metrics
    st.markdown("### 📊 Key Metrics")
    
    # Calculate metrics safely
    home_goals = data.get("home_goals_total", 0) or 0
    away_goals = data.get("away_goals_total", 0) or 0
    avg_goals = data.get("avg_goals", home_goals + away_goals)
    if avg_goals == 0:
        avg_goals = 2.0
    
    home_form = data.get("home_form_points", "?")
    away_form = data.get("away_form_points", "?")
    home_pos = data.get("home_standings_position", "?")
    away_pos = data.get("away_standings_position", "?")
    home_streak = data.get("home_losing_streak", 0)
    away_streak = data.get("away_losing_streak", 0)
    home_seq = data.get("home_form_sequence", "")
    away_seq = data.get("away_form_sequence", "")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        form_display = f"{home_form} vs {away_form}"
        if home_seq and away_seq:
            form_display += f"\n({home_seq} vs {away_seq})"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_form} vs {away_form}</div><div class="metric-label">Form Points</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_goals:.2f}</div><div class="metric-label">Avg Goals</div></div>', unsafe_allow_html=True)
    with m3:
        block_home = data.get("competitive_block_home", "?")
        block_away = data.get("competitive_block_away", "?")
        st.markdown(f'<div class="metric-card"><div class="metric-value">{block_home} vs {block_away}</div><div class="metric-label">Competitive Block</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_pos} vs {away_pos}</div><div class="metric-label">Standings</div></div>', unsafe_allow_html=True)
    with m5:
        desperate = "Yes" if (home_streak >= 3 or away_streak >= 3 or data.get("is_relegation_fight", False)) else "No"
        st.markdown(f'<div class="metric-card"><div class="metric-value">{desperate}</div><div class="metric-label">Desperation</div></div>', unsafe_allow_html=True)
    
    # Show streaks if present
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
        st.markdown("### 🎯 Score Matrix (Top 5)")
        sorted_scores = sorted(data["score_matrix"], key=lambda x: x.get("probability", 0), reverse=True)[:5]
        score_cols = st.columns(min(5, len(sorted_scores)))
        for idx, s in enumerate(sorted_scores):
            with score_cols[idx]:
                bg = "#1e293b" if s.get("home_goals", 0) != s.get("away_goals", 0) else "#2a1a00"
                prob = s.get("probability", 0)
                st.markdown(f'<div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;"><div style="font-size:1.2rem; font-weight:800;">{s.get("score", "?-?")}</div><div style="font-size:0.7rem; color:#94a3b8;">{prob:.1f}%</div></div>', unsafe_allow_html=True)
    
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
    
    # Show priority rules
    if data.get("home_losing_streak", 0) >= 3:
        st.markdown(f'<div class="priority-rule">🏆 PRIORITY RULE: {data.get("home_team", "Home")} has {data.get("home_losing_streak", 0)} game losing streak → BET HOME WIN (100% accuracy)</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V8.2")
    st.caption("Universal Logic Engine | Complete Fix (Avg Goals, X1/X2, Form, Teams)")

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

    # ========================================================================
    # TAB 1: ANALYZE
    # ========================================================================
    with tab1:
        st.markdown("### 📝 Paste Match Data")

        st.markdown("""
        <div class="upload-container">
            <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">📋 Paste All Data</p>
            <p style="color: #94a3b8; margin-bottom: 1rem;">Paste the Predictions, Form, and Statistics pages together in one text area</p>
        </div>
        """, unsafe_allow_html=True)

        text_data = st.text_area(
            "Paste all data here", 
            height=400, 
            key="text_paste",
            placeholder="Paste the complete text data (Predictions + Form + Statistics) here..."
        )

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        with col_btn1:
            analyze_clicked = st.button("🔮 ANALYZE V8.2", type="primary")

        if analyze_clicked:
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with Universal Logic..."):
                        parsed = parse_text_data(text_data)

                    league = parsed.get("league", "Unknown League")
                    matches = parsed.get("matches", [])
                    standings = parsed.get("standings", {})
                    form_data = parsed.get("form_data", {})
                    stats = parsed.get("statistics", {})

                    if matches:
                        st.success(f"✅ Found {len(matches)} matches in {league}")

                        if stats:
                            st.markdown("### 📊 League Statistics")
                            stat_cols = st.columns(4)
                            with stat_cols[0]:
                                if "home_wins" in stats:
                                    st.metric("Home Wins", f"{stats['home_wins']['count']} ({stats['home_wins']['percentage']}%)")
                            with stat_cols[1]:
                                if "draws" in stats:
                                    st.metric("Draws", f"{stats['draws']['count']} ({stats['draws']['percentage']}%)")
                            with stat_cols[2]:
                                if "away_wins" in stats:
                                    st.metric("Away Wins", f"{stats['away_wins']['count']} ({stats['away_wins']['percentage']}%)")
                            with stat_cols[3]:
                                if "goals_per_game" in stats:
                                    st.metric("Goals/Game", f"{stats['goals_per_game']:.2f}")

                        if standings:
                            st.markdown("### 🏆 Standings (Top 5)")
                            standings_df = pd.DataFrame([
                                {"Pos": data["position"], "Team": team, "Pts": data["points"]}
                                for team, data in list(standings.items())[:5]
                            ])
                            st.dataframe(standings_df, use_container_width=True, hide_index=True)

                        for idx, match in enumerate(matches):
                            st.markdown(f"### Match {idx + 1}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")

                            home = match.get('home_team')
                            away = match.get('away_team')
                            if home in form_data and away in form_data:
                                f1, f2, f3 = st.columns(3)
                                with f1:
                                    st.metric(f"{home} Form", f"{form_data[home]['form_points']} pts",
                                             f"Last 5: {form_data[home]['wins']}W {form_data[home]['draws']}D {form_data[home]['losses']}L")
                                with f2:
                                    st.metric(f"{away} Form", f"{form_data[away]['form_points']} pts",
                                             f"Last 5: {form_data[away]['wins']}W {form_data[away]['draws']}D {form_data[away]['losses']}L")
                                with f3:
                                    diff = abs(form_data[home]['form_points'] - form_data[away]['form_points'])
                                    st.metric("Form Difference", diff,
                                             "Similar" if diff <= 2 else "Significant")

                            data = convert_match_to_data(match, standings, form_data, league)
                            analysis = analyze_match_v8(data)
                            save_to_db(data, analysis, league)

                            display_analysis(data, analysis, league)

                            if idx < len(matches) - 1:
                                st.markdown("---")
                    else:
                        st.error("No matches found in the data. Please make sure you're pasting valid data.")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    # ========================================================================
    # TAB 2: POST-MATCH
    # ========================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('prediction', 'No prediction')
                pat = a.get('pattern', '')
                is_lock = a.get('is_lock', False)
                warning = a.get('warning')

                if is_lock:
                    badge = "🔒 LOCK"
                elif pat == "DRAW":
                    badge = "🎯 DRAW"
                elif pat == "DOUBLE_CHANCE":
                    badge = "🔄 DOUBLE CHANCE"
                elif pat == "WIN":
                    badge = "🏆 WIN"
                else:
                    badge = "⚠️ SKIP"

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
            st.info("No pending analyses. Go to the Analyze tab to process matches.")

    # ========================================================================
    # TAB 3: RECORDS
    # ========================================================================
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        if not results:
            st.info("No results recorded yet. Submit results in the Post-Match tab.")
        else:
            total = len(results)
            skip_count = sum(1 for r in results if r.get('prediction') == 'SKIP')
            bet_count = total - skip_count

            correct = 0
            incorrect = 0
            lock_correct = 0
            lock_total = 0

            for r in results:
                pred = r.get('prediction', '')
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
                pred = r.get('prediction', '')
                pattern = r.get('pattern', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                is_lock = r.get('is_lock', False)
                warning = r.get('warning')
                league = r.get('league', '')
                badge_class = get_league_badge(league)

                if pred == 'SKIP':
                    badge = '<span class="skip-badge">⚪ SKIP</span>'
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"

                if is_lock:
                    match_display = f"🔒 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif warning:
                    match_display = f"⚠️ {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif pattern == "DRAW":
                    match_display = f"🎯 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif pattern == "DOUBLE_CHANCE":
                    match_display = f"🔄 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif pattern == "WIN":
                    match_display = f"🏆 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                else:
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
