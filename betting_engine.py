"""
REFINED FORMULA - YOUR 5 RULES ONLY
Rules:
1. Home Fortress
2. Away Form Killer
3. H2H Dominance
4. H2H Draw Rate
5. Midweek Fatigue
"""

import streamlit as st
from datetime import date, datetime, timedelta
from supabase import create_client, Client
import pandas as pd
import re
import traceback
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

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
TABLE_NAME = "match_predictions_v2"
H2H_TABLE = "h2h_history"
FORM_TABLE = "team_form_history"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Refined Formula V1.1", page_icon="🎯", layout="wide")

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def parse_match_date(date_val):
    if not date_val:
        return None
    if isinstance(date_val, (date, datetime)):
        return date_val
    if isinstance(date_val, str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%Y %H:%M"):
            try:
                return datetime.strptime(date_val, fmt).date()
            except:
                continue
    return None

def get_h2h_history(home_team: str, away_team: str, limit: int = 6) -> List[dict]:
    three_years_ago = (datetime.now() - timedelta(days=3*365)).date()
    try:
        response = supabase.table(H2H_TABLE).select("*")\
            .or_(f"and(home_team.eq.{home_team},away_team.eq.{away_team}),and(home_team.eq.{away_team},away_team.eq.{home_team})")\
            .gte("match_date", three_years_ago.isoformat())\
            .order("match_date", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if response.data else []
    except:
        return []

def get_team_form(team: str, limit: int = 6, is_home: bool = None) -> List[dict]:
    try:
        query = supabase.table(FORM_TABLE).select("*").eq("team_name", team).order("match_date", desc=True).limit(limit)
        if is_home is not None:
            query = query.eq("is_home", is_home)
        response = query.execute()
        return response.data if response.data else []
    except:
        return []

def save_prediction_to_db(data: dict) -> str:
    try:
        existing = supabase.table(TABLE_NAME).select("id")\
            .eq("home_team", data['home_team'])\
            .eq("away_team", data['away_team'])\
            .eq("match_date", data['match_date'])\
            .execute()
        if existing.data:
            return "ALREADY_EXISTS"
        response = supabase.table(TABLE_NAME).insert(data).execute()
        return str(response.data[0]['id']) if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def submit_result(match_id: int, home_goals: int, away_goals: int):
    try:
        actual_1x2 = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        response = supabase.table(TABLE_NAME).select("refined_prediction").eq("id", match_id).execute()
        is_correct = False
        if response.data:
            predicted = response.data[0].get("refined_prediction")
            is_correct = predicted == actual_1x2 if predicted else False
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_1x2": actual_1x2,
            "is_correct": is_correct,
            "updated_at": datetime.now().isoformat()
        }).eq("id", match_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False

# ============================================================================
# PARSER
# ============================================================================

def clean_team_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'- Logo$|Logo$|\([^)]*\)', '', name)
    name = re.sub(r'[^\w\s\-\.]', '', name)
    name = re.sub(r'\d+$', '', name)
    name = re.sub(r'^\d+\s+', '', name)
    for suffix in [' Rs1', ' Rs2', ' Rs3', ' (H)', ' (A)', ' Logo']:
        name = name.replace(suffix, '')
    name = name.strip()
    team_pattern = re.search(r'([A-Za-z\s]+?)(?:\s+\d+%|\s+[A-Z]|$)', name)
    if team_pattern:
        potential = team_pattern.group(1).strip()
        if len(potential) > 2:
            return potential
    return name if len(name) > 2 else "Unknown"

def parse_text_data(text: str) -> dict:
    result = {
        'matches': [],
        'league': 'Unknown'
    }
    lines = text.split('\n')
    
    # Detect league
    league_keywords = ['Superliga', 'Premier League', 'Serie A', 'La Liga', 'Bundesliga', 'Ligue 1', 'Serie B', 'Championship', 'Russia Premier League']
    for line in lines:
        for kw in league_keywords:
            if kw in line:
                result['league'] = line.strip()
                break
        if result['league'] != 'Unknown':
            break

    current_match = {}
    match_found = False
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if ' VS ' in line:
            parts = line.split(' VS ')
            if len(parts) == 2:
                home = clean_team_name(parts[0])
                away = clean_team_name(parts[1])
                if home and away and len(home) > 2 and len(away) > 2:
                    current_match = {
                        'home_team': home,
                        'away_team': away,
                        'date': datetime.now().strftime("%Y-%m-%d"),
                        'is_finished': False,
                        'home_pct': 33,
                        'draw_pct': 33,
                        'away_pct': 34,
                        'forebet_prediction': 'X',
                        'avg_goals': 2.5,
                        'h2h_data': [],
                        'home_scoring_rate': 1.0,
                        'away_scoring_rate': 1.0
                    }
                    match_found = True
                    # find date nearby
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        dt_line = lines[j].strip()
                        dt_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})', dt_line)
                        if dt_match:
                            try:
                                dt = datetime.strptime(dt_match.group(1), "%d/%m/%Y")
                                current_match['date'] = dt.strftime("%Y-%m-%d")
                            except:
                                pass

        # Look for encoded data: e.g., "305613X1 - 11.5323°-"
        if match_found and re.search(r'^\d{6}[1X2]', line):
            cleaned = line.replace(' ', '')
            pct_match = re.search(r'^(\d{2})(\d{2})(\d{2})([1X2])', cleaned)
            if pct_match:
                current_match['home_pct'] = int(pct_match.group(1))
                current_match['draw_pct'] = int(pct_match.group(2))
                current_match['away_pct'] = int(pct_match.group(3))
                current_match['forebet_prediction'] = pct_match.group(4)
                current_match['prediction'] = pct_match.group(4)

                # Score
                score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
                if score_match:
                    current_match['correct_score_home'] = int(score_match.group(1))
                    current_match['correct_score_away'] = int(score_match.group(2))
                # Avg goals
                avg_match = re.search(r'(\d+\.\d{2})\s*°', line)
                if avg_match:
                    current_match['avg_goals'] = float(avg_match.group(1))
                # Double chance
                dc_match = re.search(r'([1X2]{2})', line)
                if dc_match:
                    current_match['double_chance'] = dc_match.group(1)
                
                # Extract scoring rates from statistics section
                for j in range(i, min(len(lines), i + 30)):
                    stat_line = lines[j].strip()
                    if 'Scored' in stat_line and 'Avg.' in stat_line:
                        score_match = re.search(r'Scored\s+(\d+)\s+Avg\.\s+per\s+game\s+([\d.]+)', stat_line)
                        if score_match:
                            current_match['home_scoring_rate'] = float(score_match.group(2))
                        # Look for away scoring rate
                        for k in range(j+1, min(len(lines), j+10)):
                            away_line = lines[k].strip()
                            if 'Scored' in away_line and 'Avg.' in away_line:
                                away_score_match = re.search(r'Scored\s+(\d+)\s+Avg\.\s+per\s+game\s+([\d.]+)', away_line)
                                if away_score_match:
                                    current_match['away_scoring_rate'] = float(away_score_match.group(2))
                                    break
                        break

        # Check if match is finished (FT)
        if match_found and 'FT' in line:
            current_match['is_finished'] = True
            ft_score = re.search(r'FT\s+(\d+)\s*-\s*(\d+)', line)
            if ft_score:
                current_match['actual_home'] = int(ft_score.group(1))
                current_match['actual_away'] = int(ft_score.group(2))

        # H2H section
        if 'Head to head' in line or 'H2H' in line:
            j = i + 1
            while j < len(lines) and j < i + 20:
                h2h_line = lines[j].strip()
                if re.search(r'\d{2}/\d{2}/\d{4}', h2h_line):
                    h2h = parse_h2h_line(h2h_line)
                    if h2h and match_found:
                        if 'h2h_data' not in current_match:
                            current_match['h2h_data'] = []
                        current_match['h2h_data'].append(h2h)
                    j += 1
                else:
                    break

        # If we have a complete match, save it
        if match_found and current_match.get('home_team') and current_match.get('away_team'):
            has_data = (current_match.get('home_pct') != 33 or
                       current_match.get('draw_pct') != 33 or
                       current_match.get('away_pct') != 34)
            if has_data:
                already_added = False
                for m in result['matches']:
                    if (m.get('home_team') == current_match.get('home_team') and
                        m.get('away_team') == current_match.get('away_team')):
                        already_added = True
                        break
                if not already_added:
                    result['matches'].append(current_match.copy())
                    current_match = {}
                    match_found = False

        i += 1

    if not result['matches']:
        result = fallback_parse(text)

    if result['league'] == 'Unknown':
        for kw in league_keywords:
            if kw in text:
                result['league'] = kw
                break

    return result

def parse_h2h_line(line: str) -> dict:
    date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
    if not date_match:
        return None
    date_str = date_match.group(1)
    score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
    if not score_match:
        return None
    home_goals = int(score_match.group(1))
    away_goals = int(score_match.group(2))
    winner = 'home' if home_goals > away_goals else 'away' if away_goals > home_goals else 'draw'
    parts = re.split(r'\d+\s*-\s*\d+', line)
    if len(parts) >= 2:
        left = re.sub(r'\d{2}/\d{2}/\d{4}', '', parts[0]).strip()
        right = parts[1].strip()
        home_team = clean_team_name(left)
        away_team = clean_team_name(right)
    else:
        teams = re.findall(r'([A-Za-z\s]+?)\s+\d+\s*-\s*\d+\s+([A-Za-z\s]+)', line)
        if teams:
            home_team = clean_team_name(teams[0][0])
            away_team = clean_team_name(teams[0][1])
        else:
            home_team = "Unknown"
            away_team = "Unknown"
    return {
        'home_team': home_team,
        'away_team': away_team,
        'match_date': date_str,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'winner': winner
    }

def fallback_parse(text: str) -> dict:
    result = {'matches': [], 'league': 'Unknown'}
    vs_pattern = r'([A-Za-z\s]+?)\s+VS\s+([A-Za-z\s]+?)(?:\s+\d+|$)'
    for home, away in re.findall(vs_pattern, text):
        home_team = clean_team_name(home)
        away_team = clean_team_name(away)
        if home_team and away_team and len(home_team) > 2 and len(away_team) > 2:
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'is_finished': False,
                'home_pct': 33,
                'draw_pct': 33,
                'away_pct': 34,
                'forebet_prediction': 'X',
                'avg_goals': 2.5,
                'h2h_data': [],
                'home_scoring_rate': 1.0,
                'away_scoring_rate': 1.0
            }
            enc = re.search(r'(\d{6}[1X2])', text)
            if enc:
                code = enc.group(1)
                if len(code) >= 7:
                    match_data['home_pct'] = int(code[0:2])
                    match_data['draw_pct'] = int(code[2:4])
                    match_data['away_pct'] = int(code[4:6])
                    match_data['forebet_prediction'] = code[6]
                    match_data['prediction'] = code[6]
            avg = re.search(r'(\d+\.\d{2})\s*°', text)
            if avg:
                match_data['avg_goals'] = float(avg.group(1))
            dt_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})', text)
            if dt_match:
                try:
                    dt = datetime.strptime(dt_match.group(1), "%d/%m/%Y")
                    match_data['date'] = dt.strftime("%Y-%m-%d")
                except:
                    pass
            result['matches'].append(match_data)
    return result

# ============================================================================
# YOUR 5 RULES - REFINED FORMULA
# ============================================================================

def check_home_fortress(home_form: List[dict]) -> Tuple[bool, int, str]:
    """
    RULE 1: Home Fortress
    If home team is unbeaten in last 5 home games → Back Home Win
    """
    if len(home_form) < 5:
        return False, 0, "Not enough data (need 5+ home games)"
    
    home_form = home_form[:5]
    unbeaten = 0
    for m in home_form:
        if m.get('result') != 'L':
            unbeaten += 1
        else:
            break
    
    if unbeaten >= 5:
        return True, unbeaten, f"Unbeaten in last {unbeaten} home games"
    return False, unbeaten, f"Only {unbeaten}/5 unbeaten"

def check_away_form_killer(away_form: List[dict]) -> Tuple[bool, int, str]:
    """
    RULE 2: Away Form Killer
    If away team has lost 4 of last 6 away games → Back Home Win
    """
    if len(away_form) < 6:
        return False, 0, "Not enough data (need 6 away games)"
    
    away_form = away_form[:6]
    losses = sum(1 for m in away_form if m.get('result') == 'L')
    
    if losses >= 4:
        return True, losses, f"Lost {losses}/6 away games"
    return False, losses, f"Only {losses}/6 losses"

def check_h2h_dominance(h2h_data: List[dict], home_fatigued: bool, away_fatigued: bool) -> Tuple[Optional[str], int, int, str, Optional[str]]:
    """
    RULE 3: H2H Dominance (Refined)
    If one team has won 3 of last 4 H2Hs (within 3 years) → Draw is a trap.
    Back the dominant side only if no fatigue.
    """
    if not h2h_data or len(h2h_data) < 4:
        return None, 0, 0, "Not enough H2H data (need 4+ matches)", None
    
    h2h_data = h2h_data[:4]  # Last 4 H2Hs
    home_wins = sum(1 for m in h2h_data if m.get('winner') == 'home')
    away_wins = sum(1 for m in h2h_data if m.get('winner') == 'away')
    draws = sum(1 for m in h2h_data if m.get('winner') == 'draw')
    
    if home_wins >= 3:
        if home_fatigued:
            return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs but fatigued → Draw", 'fatigue'
        else:
            return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs → Draw is a trap", 'dominant'
    elif away_wins >= 3:
        if away_fatigued:
            return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs but fatigued → Draw", 'fatigue'
        else:
            return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs → Draw is a trap", 'dominant'
    else:
        return None, max(home_wins, away_wins), draws, f"No dominance (H:{home_wins}, A:{away_wins}, D:{draws})", None

def check_h2h_draw_rate(h2h_data: List[dict]) -> Tuple[bool, int, str]:
    """
    RULE 4: H2H Draw Rate (Refined)
    Only trust the draw if 4+ draws in last 6 H2Hs.
    If fewer than 4 total H2Hs, ignore this rule completely.
    """
    if not h2h_data or len(h2h_data) < 6:
        return False, 0, "Not enough H2H data (need 6+ matches)"
    
    h2h_data = h2h_data[:6]
    draws = sum(1 for m in h2h_data if m.get('winner') == 'draw')
    
    if draws >= 4:
        return True, draws, f"{draws}/6 H2Hs were draws → Trust the Draw"
    return False, draws, f"Only {draws}/6 draws"

def check_midweek_fatigue(team: str, match_date, fixtures: List[dict]) -> Tuple[bool, str]:
    """
    RULE 5: Midweek Fatigue
    If away team played a competitive match 3-4 days ago → Downgrade away team by 30%.
    Back Home Win or Draw.
    """
    if not match_date or not fixtures:
        return False, "No fixtures data"
    
    if isinstance(match_date, str):
        try:
            match_date = datetime.strptime(match_date, "%Y-%m-%d").date()
        except:
            return False, "Invalid match date"
    
    for fixture in fixtures:
        if fixture.get('team') == team:
            fixture_date = fixture.get('date')
            if fixture_date and isinstance(fixture_date, (date, datetime)):
                if isinstance(fixture_date, datetime):
                    fixture_date = fixture_date.date()
                days_diff = (match_date - fixture_date).days
                if 3 <= days_diff <= 4:
                    return True, f"Played {days_diff} days ago"
    return False, "No recent midweek fixture"

def get_stake_display(stake: str) -> Tuple[str, str]:
    stake_map = {
        "2 units": ("2 units", "HIGH"),
        "1.5 units": ("1.5 units", "MEDIUM"),
        "1 unit": ("1 unit", "MEDIUM"),
        "0.5 units": ("0.5 units", "LOW"),
        "0.25 units": ("0.25 units", "LOW"),
        "0.1 units": ("0.1 units", "LOW"),
    }
    return stake_map.get(stake, (stake, "LOW"))

# ============================================================================
# YOUR REFINED FORMULA DECISION LOGIC - 5 RULES ONLY
# ============================================================================

def refined_formula_decision(data: dict) -> dict:
    """
    YOUR REFINED FORMULA - 5 Rules Only
    No Goal Discrepancy, no Early/Late goals, no Clean Sheets
    """
    
    home_team = data.get('home_team', 'Unknown')
    away_team = data.get('away_team', 'Unknown')
    match_date = data.get('date')
    forebet_pred = data.get('forebet_prediction', 'X')
    fixtures = data.get('midweek_fixtures', [])
    
    # Get data needed for your rules
    home_form = get_team_form(home_team, limit=6, is_home=True)
    away_form = get_team_form(away_team, limit=6, is_home=False)
    h2h_data = data.get('h2h_data', [])
    
    # Check fatigue for H2H dominance rule
    home_fatigued, home_fatigue_msg = check_midweek_fatigue(home_team, match_date, fixtures)
    away_fatigued, away_fatigue_msg = check_midweek_fatigue(away_team, match_date, fixtures)
    
    # === RULE 1: HOME FORTRESS ===
    fortress, streak, msg = check_home_fortress(home_form)
    if fortress:
        return {
            'prediction': '1',
            'rule': 'Home Fortress',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Home Fortress']
        }
    
    # === RULE 2: AWAY FORM KILLER ===
    killer, losses, msg = check_away_form_killer(away_form)
    if killer:
        return {
            'prediction': '1',
            'rule': 'Away Form Killer',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Away Form Killer']
        }
    
    # === RULE 3: H2H DOMINANCE ===
    dominant, wins, draws, msg, status = check_h2h_dominance(h2h_data, home_fatigued, away_fatigued)
    if dominant:
        if status == 'fatigue':
            return {
                'prediction': 'X',
                'rule': 'H2H Dominance + Fatigue',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Draw',
                'reason': msg,
                'rules_passed': ['H2H Dominance', 'Midweek Fatigue']
            }
        else:
            pred = '1' if dominant == 'home' else '2'
            winner = 'Home' if dominant == 'home' else 'Away'
            return {
                'prediction': pred,
                'rule': 'H2H Dominance',
                'confidence': 'HIGH',
                'stake': '2 units',
                'bet': f"{winner} Win",
                'reason': msg,
                'rules_passed': ['H2H Dominance']
            }
    
    # === RULE 4: H2H DRAW RATE ===
    draw_rate, draws2, msg = check_h2h_draw_rate(h2h_data)
    if draw_rate:
        return {
            'prediction': 'X',
            'rule': 'H2H Draw Rate',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Draw',
            'reason': msg,
            'rules_passed': ['H2H Draw Rate']
        }
    
    # === RULE 5: MIDWEEK FATIGUE ===
    if away_fatigued and not home_fatigued:
        return {
            'prediction': '1',
            'rule': 'Midweek Fatigue (Away)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Home Win',
            'reason': away_fatigue_msg,
            'rules_passed': ['Midweek Fatigue']
        }
    elif home_fatigued and not away_fatigued:
        return {
            'prediction': 'X',
            'rule': 'Midweek Fatigue (Home)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Draw',
            'reason': home_fatigue_msg,
            'rules_passed': ['Midweek Fatigue']
        }
    
    # === DEFAULT: TRUST FOREBET ===
    bet_text = 'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'
    return {
        'prediction': forebet_pred,
        'rule': 'Forebet Default',
        'confidence': 'LOW',
        'stake': '0.25 units',
        'bet': bet_text,
        'reason': 'No rules triggered → Trust Forebet',
        'rules_passed': ['Forebet Default']
    }

# ============================================================================
# DISPLAY FUNCTION - NATIVE STREAMLIT COMPONENTS
# ============================================================================

def display_refined_analysis_native(match_data: dict, decision: dict, league: str = "Unknown"):
    """Display analysis using native Streamlit components"""
    
    home_team = match_data.get('home_team', 'Unknown')
    away_team = match_data.get('away_team', 'Unknown')
    home_pct = match_data.get('home_pct', '?')
    draw_pct = match_data.get('draw_pct', '?')
    away_pct = match_data.get('away_pct', '?')
    
    # Get stake display
    stake_display, confidence_level = get_stake_display(decision.get('stake', '0.25 units'))
    
    # Confidence color mapping
    conf_color = {
        'HIGH': 'green',
        'MEDIUM': 'orange',
        'LOW': 'gray'
    }.get(decision.get('confidence', 'LOW'), 'gray')
    
    # Prediction emoji
    pred_emoji = {
        '1': '🏠',
        'X': '🤝',
        '2': '✈️',
        'Under 2.5': '⬇️',
        'Over 2.5': '⬆️'
    }.get(decision.get('prediction', 'X'), '🎯')
    
    # ----- HEADER -----
    st.subheader(f"{pred_emoji} {home_team} vs {away_team}")
    st.caption(f"📅 {match_data.get('date', '')} | 🏆 {league}")
    
    # ----- PERCENTAGES ROW -----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🏠 Home", f"{home_pct}%")
    with c2:
        st.metric("🤝 Draw", f"{draw_pct}%")
    with c3:
        st.metric("✈️ Away", f"{away_pct}%")
    
    # ----- PREDICTION CARD -----
    st.markdown("---")
    
    # Main prediction display
    st.markdown(f"### 🎯 {decision.get('bet', 'Unknown')}")
    st.markdown(f"**Rule:** {decision.get('rule', 'Unknown')}")
    st.markdown(f"**Confidence:** :{conf_color}[{decision.get('confidence', 'LOW')}]")
    st.markdown(f"**Stake:** {stake_display}")
    st.markdown(f"**Reason:** {decision.get('reason', '')}")
    
    # ----- RULES PASSED -----
    rules_passed = decision.get('rules_passed', ['Forebet Default'])
    st.caption(f"📋 Rules Passed: {', '.join(rules_passed)}")
    
    # ----- FOREBET ORIGINAL -----
    st.caption(f"📊 Forebet Original: {match_data.get('forebet_prediction', '?')} | Avg Goals: {match_data.get('avg_goals', '?')}")
    
    st.markdown("---")
    
    # ----- H2H DATA -----
    if 'h2h_data' in match_data and match_data['h2h_data']:
        with st.expander("📊 Head-to-Head History"):
            h2h_df = pd.DataFrame(match_data['h2h_data'])
            st.dataframe(h2h_df, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎯 Refined Formula V1.1")
    st.caption("Your 5 Rules: Home Fortress | Away Form Killer | H2H Dominance | H2H Draw Rate | Midweek Fatigue")

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Pending", "📊 Records", "📈 Dashboard"])
    
    with tab1:
        st.markdown("### 📝 Paste Match Data")
        text_data = st.text_area(
            "Paste Forebet data here",
            height=400,
            placeholder="Paste the complete text data from Forebet..."
        )
        col1, col2 = st.columns(2)
        with col1:
            midweek_fixtures = st.text_area(
                "Midweek Fixtures (optional)",
                height=100,
                placeholder="Team: Date (e.g., Mjällby: 2026-07-21)"
            )
        with col2:
            st.info("""
            **Your 5 Refined Formula Rules:**
            1. 🏰 Home Fortress - Unbeaten in last 5 home games → Back Home Win
            2. 💀 Away Form Killer - Lost 4 of last 6 away games → Back Home Win
            3. 🏆 H2H Dominance - 3 of last 4 H2Hs won → Draw is a trap
            4. 🤝 H2H Draw Rate - 4+ draws in last 6 H2Hs → Trust the Draw
            5. 😴 Midweek Fatigue - Away played 3-4 days ago → Downgrade away
            """)
        
        if st.button("🎯 Analyze & Auto-Save", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with YOUR 5 rules..."):
                        parsed = parse_text_data(text_data)
                        matches = parsed.get('matches', [])
                        league = parsed.get('league', 'Unknown')
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} matches in {league}")
                            
                            # Parse midweek fixtures
                            fixtures = []
                            if midweek_fixtures:
                                for line in midweek_fixtures.strip().split('\n'):
                                    if ':' in line:
                                        team, date_str = line.split(':', 1)
                                        try:
                                            fixture_date = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
                                            fixtures.append({'team': team.strip(), 'date': fixture_date})
                                        except:
                                            pass
                            
                            saved_count = 0
                            duplicate_count = 0
                            
                            for i, match in enumerate(matches, 1):
                                match['midweek_fixtures'] = fixtures
                                
                                # Run YOUR refined formula (5 rules only)
                                decision = refined_formula_decision(match)
                                
                                # Display using native Streamlit components
                                display_refined_analysis_native(match, decision, league)
                                
                                # Auto-save to database
                                db_data = {
                                    'match_date': match.get('date', datetime.now().date()),
                                    'league_name': league,
                                    'home_team': match.get('home_team', 'Unknown'),
                                    'away_team': match.get('away_team', 'Unknown'),
                                    'forebet_home_pct': match.get('home_pct', 0),
                                    'forebet_draw_pct': match.get('draw_pct', 0),
                                    'forebet_away_pct': match.get('away_pct', 0),
                                    'forebet_prediction': match.get('forebet_prediction', 'X'),
                                    'forebet_avg_goals': match.get('avg_goals', 2.5),
                                    'forebet_double_chance': match.get('double_chance', ''),
                                    'refined_prediction': decision['prediction'],
                                    'refined_rule_triggered': decision['rule'],
                                    'refined_confidence': decision['confidence'],
                                    'refined_bet': decision['bet'],
                                    'refined_stake': decision['stake'],
                                    'refined_reason': decision['reason'],
                                    'rules_passed': decision.get('rules_passed', [])
                                }
                                
                                result = save_prediction_to_db(db_data)
                                if result == "ALREADY_EXISTS":
                                    duplicate_count += 1
                                    st.warning(f"⚠️ Match {i} already exists in database (skipped)")
                                elif result:
                                    saved_count += 1
                                    st.success(f"✅ Match {i} saved with ID: {result}")
                                else:
                                    st.error(f"❌ Failed to save match {i}")
                                
                                st.markdown("---")
                            
                            st.info(f"📊 Summary: {saved_count} new matches saved, {duplicate_count} duplicates skipped.")
                        else:
                            st.error("No matches found in the data. Please check the format.")
                            st.info("Expected format: 'Team VS Team' with encoded data like '255421X1 - 12.1526°3.80'")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())
    
    # PENDING MATCHES TAB
    with tab2:
        st.subheader("📝 Pending Matches")
        st.caption("Enter actual results for completed matches")
        try:
            response = supabase.table(TABLE_NAME).select("*").is_("actual_1x2", "null").execute()
            pending = response.data if response.data else []
            if pending:
                st.write(f"**{len(pending)} pending result(s)**")
                for match in pending:
                    with st.expander(f"{match.get('home_team', '')} vs {match.get('away_team', '')} ({match.get('refined_prediction', '?')})"):
                        st.info(f"Prediction: {match.get('refined_bet', '?')} | Rule: {match.get('refined_rule_triggered', '?')}")
                        c1, c2 = st.columns(2)
                        with c1:
                            hg = st.number_input(f"{match.get('home_team', 'Home')} Goals", 0, 15, 0, key=f"hg_{match['id']}")
                        with c2:
                            ag = st.number_input(f"{match.get('away_team', 'Away')} Goals", 0, 15, 0, key=f"ag_{match['id']}")
                        if st.button("✅ Submit Result", key=f"sub_{match['id']}"):
                            if submit_result(match['id'], hg, ag):
                                st.success("✅ Result submitted successfully!")
                                st.rerun()
            else:
                st.info("No pending matches.")
        except Exception as e:
            st.error(f"Error fetching pending matches: {e}")

    # RECORDS TAB
    with tab3:
        st.subheader("📊 Performance Records")
        try:
            response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_1x2", "null").execute()
            results = response.data if response.data else []
            if results:
                total = len(results)
                correct = sum(1 for r in results if r.get('is_correct', False))
                rate = round(correct / total * 100) if total > 0 else 0
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Matches", total)
                with col2:
                    st.metric("Correct", correct)
                with col3:
                    st.metric("Accuracy", f"{rate}%")
                rule_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
                for r in results:
                    rule = r.get('refined_rule_triggered', 'Unknown')
                    rule_stats[rule]['total'] += 1
                    if r.get('is_correct', False):
                        rule_stats[rule]['correct'] += 1
                if rule_stats:
                    st.subheader("Rule Performance")
                    rule_data = []
                    for rule, stats in rule_stats.items():
                        rule_rate = round(stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        rule_data.append({
                            'Rule': rule[:40],
                            'Correct': stats['correct'],
                            'Total': stats['total'],
                            'Rate': f"{rule_rate}%"
                        })
                    df = pd.DataFrame(rule_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No results recorded yet.")
        except Exception as e:
            st.error(f"Error fetching results: {e}")

    # DASHBOARD TAB
    with tab4:
        st.subheader("📈 Dashboard")
        try:
            response = supabase.table(TABLE_NAME).select("*").execute()
            all_matches = response.data if response.data else []
            if all_matches:
                total = len(all_matches)
                pending = sum(1 for m in all_matches if m.get('actual_1x2') is None)
                completed = total - pending
                correct = sum(1 for m in all_matches if m.get('is_correct', False))
                rate = round(correct / completed * 100) if completed > 0 else 0
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Matches", total)
                with col2:
                    st.metric("Completed", completed)
                with col3:
                    st.metric("Correct", correct)
                with col4:
                    st.metric("Accuracy", f"{rate}%")
                confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
                for m in all_matches:
                    if m.get('actual_1x2') is not None:
                        conf = m.get('refined_confidence', 'LOW')
                        confidence_stats[conf]['total'] += 1
                        if m.get('is_correct', False):
                            confidence_stats[conf]['correct'] += 1
                if confidence_stats:
                    st.subheader("Confidence Performance")
                    conf_data = []
                    for conf, stats in confidence_stats.items():
                        conf_rate = round(stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        conf_data.append({
                            'Confidence': conf,
                            'Correct': stats['correct'],
                            'Total': stats['total'],
                            'Rate': f"{conf_rate}%"
                        })
                    df = pd.DataFrame(conf_data)
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("No matches in database yet.")
        except Exception as e:
            st.error(f"Error fetching dashboard data: {e}")

if __name__ == "__main__":
    main()
