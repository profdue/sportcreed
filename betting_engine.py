"""
REFINED FORMULA V1.1 - AUTO-SAVE VERSION
- Automatically saves predictions to Supabase after analysis
- Fixed HTML rendering
- Removed manual save button
"""

import streamlit as st
from datetime import date, datetime, timedelta
from supabase import create_client, Client
import pandas as pd
import re
import json
import time
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

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .rule-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-high { background: #10b981; color: #000; }
    .rule-medium { background: #f59e0b; color: #000; }
    .rule-low { background: #64748b; color: #fff; }
    .stake-badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .stake-2-units { background: #10b981; color: #000; }
    .stake-1-5-units { background: #f59e0b; color: #000; }
    .stake-1-unit { background: #f59e0b; color: #000; }
    .stake-0-5-units { background: #64748b; color: #fff; }
    .stake-0-25-units { background: #64748b; color: #fff; }
    .stake-0-1-units { background: #64748b; color: #fff; }
</style>
""", unsafe_allow_html=True)

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
    league_keywords = ['Superliga', 'Premier League', 'Serie A', 'La Liga', 'Bundesliga', 'Ligue 1', 'Serie B', 'Championship']
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
                        'h2h_data': []
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

        # Look for encoded data: e.g., "255421X1 - 12.1526°3.80"
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
            # Check if we have actual percentages (not defaults)
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
                    # Don't reset match_found, we keep going for possible next matches
                    # but we need to clear current_match to avoid duplicate
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
                'h2h_data': []
            }
            # try to get encoded percentages
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
# REFINED FORMULA - RULE CHECKERS (unchanged)
# ============================================================================

def check_home_fortress(home_team, home_form):
    if len(home_form) < 5:
        return False, 0, "Not enough data"
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

def check_away_form_killer(away_team, away_form):
    if len(away_form) < 6:
        return False, 0, "Not enough data"
    away_form = away_form[:6]
    losses = sum(1 for m in away_form if m.get('result') == 'L')
    if losses >= 4:
        return True, losses, f"Lost {losses}/6 away games"
    return False, losses, f"Only {losses}/6 losses"

def check_clean_sheet_streak(team, form, is_home):
    clean = 0
    for m in form:
        if m.get('clean_sheet', False):
            clean += 1
        else:
            break
    context = "home" if is_home else "away"
    return clean, f"{clean} consecutive clean sheets ({context})"

def check_early_goal_tendency(team, form):
    total_goals = 0
    early_goals = 0
    for m in form:
        goals = m.get('goals_for', 0)
        early = m.get('goals_0_15', 0)
        total_goals += goals
        early_goals += early
    if total_goals == 0:
        return False, 0, "No goals"
    ratio = early_goals / total_goals
    if ratio >= 0.3:
        return True, ratio, f"{ratio*100:.1f}% goals in 0-15 min"
    return False, ratio, f"Only {ratio*100:.1f}% early goals"

def check_late_goal_tendency(team, form):
    total_goals = 0
    late_goals = 0
    for m in form:
        goals = m.get('goals_for', 0)
        late = m.get('goals_75_90', 0)
        total_goals += goals
        late_goals += late
    if total_goals == 0:
        return False, 0, "No goals"
    ratio = late_goals / total_goals
    if ratio >= 0.4:
        return True, ratio, f"{ratio*100:.1f}% goals in 75-90+ min"
    return False, ratio, f"Only {ratio*100:.1f}% late goals"

def check_h2h_dominance(h2h_data):
    if not h2h_data or len(h2h_data) < 4:
        return None, 0, 0, "Not enough H2H"
    h2h_data = h2h_data[:6]
    home_wins = sum(1 for m in h2h_data if m.get('winner') == 'home')
    away_wins = sum(1 for m in h2h_data if m.get('winner') == 'away')
    draws = sum(1 for m in h2h_data if m.get('winner') == 'draw')
    if home_wins >= 3:
        return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs"
    elif away_wins >= 3:
        return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs"
    else:
        return None, max(home_wins, away_wins), draws, f"No dominance (H:{home_wins}, A:{away_wins}, D:{draws})"

def check_h2h_draw_rate(h2h_data):
    if not h2h_data or len(h2h_data) < 6:
        return False, 0, "Not enough H2H"
    h2h_data = h2h_data[:6]
    draws = sum(1 for m in h2h_data if m.get('winner') == 'draw')
    if draws >= 4:
        return True, draws, f"{draws}/6 H2Hs were draws"
    return False, draws, f"Only {draws}/6 draws"

def check_midweek_fatigue(team, match_date, fixtures):
    if not match_date or not fixtures:
        return False, "No fixtures"
    if isinstance(match_date, str):
        try:
            match_date = datetime.strptime(match_date, "%Y-%m-%d").date()
        except:
            return False, "Invalid date"
    for f in fixtures:
        if f.get('team') == team:
            f_date = f.get('date')
            if f_date and isinstance(f_date, (date, datetime)):
                if isinstance(f_date, datetime):
                    f_date = f_date.date()
                days = (match_date - f_date).days
                if 3 <= days <= 4:
                    return True, f"Played {days} days ago"
    return False, "No recent midweek"

def check_goal_discrepancy(forebet_avg, home_scoring, away_scoring):
    actual_avg = (home_scoring + away_scoring) / 2
    diff = forebet_avg - actual_avg
    if abs(diff) < 0.3:
        return 'MATCH', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (match)"
    elif diff > 0.3:
        return 'OVER_INFLATED', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (over by {diff:.2f})"
    else:
        return 'UNDER_INFLATED', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (under by {abs(diff):.2f})"

def check_double_chance_validation(forebet_pred, double_chance):
    if not double_chance:
        return False, "No double chance"
    if forebet_pred == '1' and '1' in double_chance:
        return True, "1X supports Home Win"
    elif forebet_pred == '2' and '2' in double_chance:
        return True, "X2 supports Away Win"
    elif forebet_pred == 'X' and 'X' in double_chance:
        return True, "1X or X2 supports Draw"
    else:
        return False, f"Double chance {double_chance} contradicts {forebet_pred}"

def get_stake_display(stake: str) -> Tuple[str, str]:
    stake_map = {
        "2 units": ("2 units", "stake-2-units"),
        "1.5 units": ("1.5 units", "stake-1-5-units"),
        "1 unit": ("1 unit", "stake-1-unit"),
        "0.5 units": ("0.5 units", "stake-0-5-units"),
        "0.25 units": ("0.25 units", "stake-0-25-units"),
        "0.1 units": ("0.1 units", "stake-0-1-units"),
    }
    return stake_map.get(stake, (stake, "stake-0-25-units"))

# ============================================================================
# DECISION LOGIC
# ============================================================================

def refined_formula_decision(data: dict) -> dict:
    home_team = data.get('home_team', 'Unknown')
    away_team = data.get('away_team', 'Unknown')
    match_date = data.get('date')
    forebet_pred = data.get('forebet_prediction', 'X')
    forebet_avg = data.get('avg_goals', 2.5)
    home_scoring = data.get('home_scoring_rate', 1.0)
    away_scoring = data.get('away_scoring_rate', 1.0)

    home_form = get_team_form(home_team, limit=6, is_home=True)
    away_form = get_team_form(away_team, limit=6, is_home=False)

    # Rule 1: Home Fortress
    fortress, _, msg = check_home_fortress(home_team, home_form)
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

    # Rule 2: Away Form Killer
    killer, _, msg = check_away_form_killer(away_team, away_form)
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

    # Rule 3: Clean Sheet Streak
    home_cs, home_cs_msg = check_clean_sheet_streak(home_team, home_form, True)
    away_cs, away_cs_msg = check_clean_sheet_streak(away_team, away_form, False)
    if home_cs >= 3 and away_cs >= 1:
        return {
            'prediction': 'Under 2.5',
            'rule': 'Clean Sheet Streak',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Under 2.5 Goals',
            'reason': f"Home: {home_cs_msg}, Away: {away_cs_msg}",
            'rules_passed': ['Clean Sheet Streak']
        }

    # Rule 4: Early Goal Tendency
    home_early, _, msg = check_early_goal_tendency(home_team, home_form)
    away_early, _, msg2 = check_early_goal_tendency(away_team, away_form)
    if home_early and not away_early:
        return {
            'prediction': '1',
            'rule': 'Early Goal Tendency',
            'confidence': 'MEDIUM',
            'stake': '1.5 units',
            'bet': 'Home Win',
            'reason': f"Home: {msg}, Away: {msg2}",
            'rules_passed': ['Early Goal Tendency']
        }
    elif away_early and not home_early:
        return {
            'prediction': '2',
            'rule': 'Early Goal Tendency',
            'confidence': 'MEDIUM',
            'stake': '1.5 units',
            'bet': 'Away Win',
            'reason': f"Away: {msg2}, Home: {msg}",
            'rules_passed': ['Early Goal Tendency']
        }

    # Rule 5: Late Goal Tendency
    home_late, ratio3, msg3 = check_late_goal_tendency(home_team, home_form)
    away_late, ratio4, msg4 = check_late_goal_tendency(away_team, away_form)
    if (home_late or away_late) and forebet_pred == 'X':
        if home_late and away_late:
            winner = 'Home' if ratio3 > ratio4 else 'Away'
            pred = '1' if ratio3 > ratio4 else '2'
            return {
                'prediction': pred,
                'rule': 'Late Goal Tendency (Both)',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': f"{winner} Win",
                'reason': f"Home: {msg3}, Away: {msg4}",
                'rules_passed': ['Late Goal Tendency']
            }
        elif home_late:
            return {
                'prediction': '1',
                'rule': 'Late Goal Tendency (Home)',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Home Win',
                'reason': msg3,
                'rules_passed': ['Late Goal Tendency']
            }
        else:
            return {
                'prediction': '2',
                'rule': 'Late Goal Tendency (Away)',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Away Win',
                'reason': msg4,
                'rules_passed': ['Late Goal Tendency']
            }

    # Rule 6: H2H Dominance
    h2h_data = data.get('h2h_data', [])
    if not h2h_data:
        h2h_data = get_h2h_history(home_team, away_team, limit=6)
    dominant, _, _, msg5 = check_h2h_dominance(h2h_data)
    if dominant:
        fixtures = data.get('midweek_fixtures', [])
        home_fatigued, _ = check_midweek_fatigue(home_team, match_date, fixtures)
        away_fatigued, _ = check_midweek_fatigue(away_team, match_date, fixtures)
        if dominant == 'home' and home_fatigued:
            return {
                'prediction': 'X',
                'rule': 'H2H Dominance + Home Fatigue',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Draw',
                'reason': f"{msg5} but home team fatigued",
                'rules_passed': ['H2H Dominance', 'Midweek Fatigue']
            }
        elif dominant == 'away' and away_fatigued:
            return {
                'prediction': 'X',
                'rule': 'H2H Dominance + Away Fatigue',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Draw',
                'reason': f"{msg5} but away team fatigued",
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
                'reason': msg5,
                'rules_passed': ['H2H Dominance']
            }

    # Rule 7: H2H Draw Rate
    draw_rate, _, msg6 = check_h2h_draw_rate(h2h_data)
    if draw_rate:
        return {
            'prediction': 'X',
            'rule': 'H2H Draw Rate',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Draw',
            'reason': msg6,
            'rules_passed': ['H2H Draw Rate']
        }

    # Rule 8: Midweek Fatigue
    fixtures = data.get('midweek_fixtures', [])
    home_fatigued, home_fatigue_msg = check_midweek_fatigue(home_team, match_date, fixtures)
    away_fatigued, away_fatigue_msg = check_midweek_fatigue(away_team, match_date, fixtures)
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
            'prediction': '2',
            'rule': 'Midweek Fatigue (Home)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Away Win',
            'reason': home_fatigue_msg,
            'rules_passed': ['Midweek Fatigue']
        }

    # Rule 9: Goal Discrepancy
    discrepancy, _, msg7 = check_goal_discrepancy(forebet_avg, home_scoring, away_scoring)
    if discrepancy == 'OVER_INFLATED' and forebet_pred == 'X':
        return {
            'prediction': 'X',
            'rule': 'Goal Discrepancy (Over-inflated)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Draw',
            'reason': msg7,
            'rules_passed': ['Goal Discrepancy']
        }

    # Rule 10: Double Chance Validation
    double_chance = data.get('double_chance', '')
    validated, msg8 = check_double_chance_validation(forebet_pred, double_chance)
    if validated:
        bet_text = 'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'
        return {
            'prediction': forebet_pred,
            'rule': 'Double Chance Validated',
            'confidence': 'LOW',
            'stake': '0.5 units',
            'bet': bet_text,
            'reason': msg8,
            'rules_passed': ['Double Chance']
        }

    # Default: Forebet
    bet_text = 'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'
    return {
        'prediction': forebet_pred,
        'rule': 'Forebet Default',
        'confidence': 'LOW',
        'stake': '0.25 units',
        'bet': bet_text,
        'reason': 'No rules triggered, using Forebet prediction',
        'rules_passed': ['Forebet Default']
    }

# ============================================================================
# DISPLAY FUNCTION - RENDERS HTML PROPERLY
# ============================================================================

def display_refined_analysis_with_context(match_data: dict, decision: dict, league: str = "Unknown"):
    stake_display, stake_class = get_stake_display(decision.get('stake', '0.25 units'))
    confidence_color = {
        'HIGH': '#10b981',
        'MEDIUM': '#f59e0b',
        'LOW': '#64748b'
    }.get(decision.get('confidence', 'LOW'), '#64748b')
    pred_color = {
        '1': '#10b981',
        'X': '#f59e0b',
        '2': '#ef4444'
    }.get(decision.get('prediction', 'X'), '#3b82f6')

    home_team = match_data.get('home_team', 'Unknown')
    away_team = match_data.get('away_team', 'Unknown')
    home_pct = match_data.get('home_pct', '?')
    draw_pct = match_data.get('draw_pct', '?')
    away_pct = match_data.get('away_pct', '?')

    html = f"""
    <div class="output-card" style="border-left: 4px solid {confidence_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700;">
                    🏠 {home_team} vs ✈️ {away_team}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    📅 {match_data.get('date', '')} | 🏆 {league}
                </div>
            </div>
            <div style="text-align: right;">
                <span class="rule-badge rule-{decision.get('confidence', 'LOW').lower()}">
                    {decision.get('confidence', 'LOW')}
                </span>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.2rem;">
                    <span class="stake-badge {stake_class}">Stake: {stake_display}</span>
                </div>
            </div>
        </div>
        
        <div style="display: flex; gap: 1rem; margin: 0.75rem 0; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 80px; background: #0f172a; border-radius: 8px; padding: 0.5rem; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8;">Home</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #10b981;">{home_pct}%</div>
            </div>
            <div style="flex: 1; min-width: 80px; background: #0f172a; border-radius: 8px; padding: 0.5rem; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8;">Draw</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">{draw_pct}%</div>
            </div>
            <div style="flex: 1; min-width: 80px; background: #0f172a; border-radius: 8px; padding: 0.5rem; text-align: center;">
                <div style="font-size: 0.7rem; color: #94a3b8;">Away</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #ef4444;">{away_pct}%</div>
            </div>
        </div>
        
        <div style="margin-top: 0.75rem; padding: 0.75rem; background: #0f172a; border-radius: 8px;">
            <div style="font-size: 1.5rem; font-weight: 800; text-align: center; color: {pred_color};">
                🎯 {decision.get('bet', 'Unknown')}
            </div>
            <div style="text-align: center; font-size: 0.9rem; color: #94a3b8; margin-top: 0.25rem;">
                📋 {decision.get('rule', 'Unknown')}
            </div>
            <div style="text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">
                {decision.get('reason', '')}
            </div>
        </div>
        
        <div style="margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
            <div style="font-size: 0.7rem; color: #94a3b8;">Rules Passed:</div>
            {''.join([f'<span style="font-size: 0.7rem; background: #1e293b; padding: 0.1rem 0.4rem; border-radius: 4px; margin-right: 0.3rem;">{rule}</span>' for rule in decision.get('rules_passed', ['Forebet Default'])])}
        </div>
        
        <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            📊 Forebet Original: {match_data.get('forebet_prediction', '?')} | Avg Goals: {match_data.get('avg_goals', '?')}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Show H2H data if available
    if 'h2h_data' in match_data and match_data['h2h_data']:
        with st.expander("📊 Head-to-Head History"):
            h2h_df = pd.DataFrame(match_data['h2h_data'])
            st.dataframe(h2h_df, use_container_width=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎯 Refined Formula V1.1")
    st.caption("Complete implementation with goal timing, clean sheets, and self-learning - Auto-save enabled")

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
            **Refined Formula Rules:**
            1. 🏰 Home Fortress
            2. 💀 Away Form Killer
            3. 🧹 Clean Sheet Streak
            4. ⏰ Early Goal Tendency
            5. ⏰ Late Goal Tendency
            6. 🏆 H2H Dominance
            7. 🤝 H2H Draw Rate
            8. 😴 Midweek Fatigue
            9. 📊 Goal Discrepancy
            10. ✅ Double Chance
            """)
        
        if st.button("🎯 Analyze & Auto-Save", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing and saving to database..."):
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
                                if 'home_scoring_rate' not in match:
                                    match['home_scoring_rate'] = 0.83
                                if 'away_scoring_rate' not in match:
                                    match['away_scoring_rate'] = 1.5
                                
                                decision = refined_formula_decision(match)
                                
                                # Display the result
                                display_refined_analysis_with_context(match, decision, league)
                                
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
    
    # Tabs 2,3,4 remain unchanged (pending, records, dashboard)
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
