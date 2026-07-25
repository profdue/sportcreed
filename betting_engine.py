"""
REFINED FORMULA V1.1 - Complete Implementation
- Goal timing data (0-15, 75-90 minutes)
- Clean sheet streaks
- Double chance validation
- Goal expectation discrepancy
- Self-learning failure tracking
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
    .rule-card { border-left: 4px solid #f59e0b; background: #0f172a; border-radius: 8px; padding: 0.75rem; margin: 0.25rem 0; }
    .rule-triggered { border-left-color: #10b981; }
    .rule-not-triggered { border-left-color: #64748b; }
    .rule-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .rule-high { background: #10b981; color: #000; }
    .rule-medium { background: #f59e0b; color: #000; }
    .rule-low { background: #64748b; color: #fff; }
    .rule-name { font-size: 0.9rem; font-weight: 700; }
    .rule-reason { font-size: 0.8rem; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATABASE HELPERS
# ============================================================================
def get_h2h_history(home_team: str, away_team: str, limit: int = 6) -> List[dict]:
    """Fetch last N H2H matches within 3 years"""
    three_years_ago = (datetime.now() - timedelta(days=3*365)).date()
    
    try:
        response = supabase.table(H2H_TABLE).select("*")\
            .or_(f"and(home_team.eq.{home_team},away_team.eq.{away_team}),and(home_team.eq.{away_team},away_team.eq.{home_team})")\
            .gte("match_date", three_years_ago.isoformat())\
            .order("match_date", desc=True)\
            .limit(limit)\
            .execute()
        return response.data if response.data else []
    except Exception as e:
        return []

def get_team_form(team: str, limit: int = 6, is_home: bool = None) -> List[dict]:
    """Fetch last N form results for a team"""
    try:
        query = supabase.table(FORM_TABLE).select("*").eq("team_name", team).order("match_date", desc=True).limit(limit)
        if is_home is not None:
            query = query.eq("is_home", is_home)
        response = query.execute()
        return response.data if response.data else []
    except Exception as e:
        return []

def save_prediction_to_db(data: dict) -> str:
    """Save refined prediction to database"""
    try:
        # Check if exists
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
    """Submit actual result and update correctness"""
    try:
        total = home_goals + away_goals
        actual_1x2 = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        
        # Get prediction
        response = supabase.table(TABLE_NAME).select("refined_prediction").eq("id", match_id).execute()
        if response.data:
            predicted = response.data[0].get("refined_prediction")
            is_correct = predicted == actual_1x2 if predicted else False
        else:
            is_correct = False
        
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
# REFINED FORMULA - RULE CHECKERS
# ============================================================================

def check_home_fortress(home_team: str, home_form: List[dict]) -> Tuple[bool, int, str]:
    """
    Rule 1: Home Fortress
    Home team unbeaten in last 5+ home games
    """
    if len(home_form) < 5:
        return False, 0, "Not enough data (need 5+ home games)"
    
    home_form = home_form[:5]
    unbeaten_streak = 0
    for match in home_form:
        if match.get('result') != 'L':
            unbeaten_streak += 1
        else:
            break
    
    if unbeaten_streak >= 5:
        return True, unbeaten_streak, f"Unbeaten in last {unbeaten_streak} home games"
    return False, unbeaten_streak, f"Only {unbeaten_streak}/5 unbeaten"

def check_away_form_killer(away_team: str, away_form: List[dict]) -> Tuple[bool, int, str]:
    """
    Rule 2: Away Form Killer
    Away team lost 4+ of last 6 away games
    """
    if len(away_form) < 6:
        return False, 0, "Not enough data (need 6 away games)"
    
    away_form = away_form[:6]
    losses = sum(1 for m in away_form if m.get('result') == 'L')
    
    if losses >= 4:
        return True, losses, f"Lost {losses}/6 away games"
    return False, losses, f"Only {losses}/6 losses"

def check_clean_sheet_streak(team: str, form: List[dict], is_home: bool) -> Tuple[int, str]:
    """
    Rule 3: Clean Sheet Streak
    Count consecutive clean sheets
    """
    clean_sheets = 0
    for match in form:
        if match.get('clean_sheet', False):
            clean_sheets += 1
        else:
            break
    
    context = "home" if is_home else "away"
    return clean_sheets, f"{clean_sheets} consecutive clean sheets ({context})"

def check_early_goal_tendency(team: str, form: List[dict]) -> Tuple[bool, float, str]:
    """
    Rule 4: Early Goal Tendency
    30%+ of goals scored in first 15 minutes
    """
    total_goals = 0
    early_goals = 0
    
    for match in form:
        goals = match.get('goals_for', 0)
        early = match.get('goals_0_15', 0)
        total_goals += goals
        early_goals += early
    
    if total_goals == 0:
        return False, 0, "No goals scored in recent matches"
    
    ratio = early_goals / total_goals
    if ratio >= 0.3:
        return True, ratio, f"{ratio*100:.1f}% of goals in 0-15 min"
    return False, ratio, f"Only {ratio*100:.1f}% early goals"

def check_late_goal_tendency(team: str, form: List[dict]) -> Tuple[bool, float, str]:
    """
    Rule 5: Late Goal Tendency
    40%+ of goals scored in 75-90+ minutes
    """
    total_goals = 0
    late_goals = 0
    
    for match in form:
        goals = match.get('goals_for', 0)
        late = match.get('goals_75_90', 0)
        total_goals += goals
        late_goals += late
    
    if total_goals == 0:
        return False, 0, "No goals scored in recent matches"
    
    ratio = late_goals / total_goals
    if ratio >= 0.4:
        return True, ratio, f"{ratio*100:.1f}% of goals in 75-90+ min"
    return False, ratio, f"Only {ratio*100:.1f}% late goals"

def check_h2h_dominance(h2h_data: List[dict]) -> Tuple[Optional[str], int, int, str]:
    """
    Rule 6: H2H Dominance
    One team won 3+ of last 4 H2Hs (within 3 years)
    """
    if len(h2h_data) < 4:
        return None, 0, 0, "Not enough H2H data (need 4+ matches)"
    
    h2h_data = h2h_data[:6]  # Last 6
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for match in h2h_data:
        if match.get('winner') == 'home':
            home_wins += 1
        elif match.get('winner') == 'away':
            away_wins += 1
        else:
            draws += 1
    
    if home_wins >= 3:
        return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs"
    elif away_wins >= 3:
        return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs"
    else:
        return None, max(home_wins, away_wins), draws, f"No dominance (H:{home_wins}, A:{away_wins}, D:{draws})"

def check_h2h_draw_rate(h2h_data: List[dict]) -> Tuple[bool, int, str]:
    """
    Rule 7: H2H Draw Rate
    4+ draws in last 6 H2Hs
    """
    if len(h2h_data) < 6:
        return False, 0, "Not enough H2H data (need 6+ matches)"
    
    h2h_data = h2h_data[:6]
    draws = sum(1 for m in h2h_data if m.get('winner') == 'draw')
    
    if draws >= 4:
        return True, draws, f"{draws}/6 H2Hs were draws"
    return False, draws, f"Only {draws}/6 draws"

def check_midweek_fatigue(team: str, match_date: date, fixtures: List[dict]) -> Tuple[bool, str]:
    """
    Rule 8: Midweek Fatigue
    Team played competitive match 3-4 days ago
    """
    for fixture in fixtures:
        if fixture.get('team') == team:
            fixture_date = parse_match_date(fixture.get('date'))
            if fixture_date and match_date:
                days_diff = (match_date - fixture_date).days
                if 3 <= days_diff <= 4:
                    return True, f"Played {days_diff} days ago"
    return False, "No recent midweek fixture"

def check_goal_discrepancy(forebet_avg: float, home_scoring: float, away_scoring: float) -> Tuple[str, float, str]:
    """
    Rule 9: Goal Expectation Discrepancy
    Compare Forebet's avg goals against recent form
    """
    actual_avg = (home_scoring + away_scoring) / 2
    diff = forebet_avg - actual_avg
    
    if abs(diff) < 0.3:
        return 'MATCH', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (match)"
    elif diff > 0.3:
        return 'OVER_INFLATED', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (over by {diff:.2f})"
    else:
        return 'UNDER_INFLATED', diff, f"Forebet {forebet_avg:.2f} vs actual {actual_avg:.2f} (under by {abs(diff):.2f})"

def check_double_chance_validation(forebet_pred: str, double_chance: str) -> Tuple[bool, str]:
    """
    Rule 10: Double Chance Validation
    Check if double chance supports prediction
    """
    if not double_chance:
        return False, "No double chance data"
    
    if forebet_pred == '1' and '1' in double_chance:
        return True, "1X supports Home Win"
    elif forebet_pred == '2' and '2' in double_chance:
        return True, "X2 supports Away Win"
    elif forebet_pred == 'X' and 'X' in double_chance:
        return True, "1X or X2 supports Draw"
    else:
        return False, f"Double chance {double_chance} contradicts {forebet_pred}"

# ============================================================================
# REFINED FORMULA - DECISION LOGIC
# ============================================================================

def refined_formula_decision(data: dict) -> dict:
    """
    V1.1 - Complete decision logic with all 10 rules
    """
    
    # Extract data
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    match_date = data.get('match_date')
    forebet_pred = data.get('forebet_prediction', 'X')
    forebet_avg = data.get('forebet_avg_goals', 2.5)
    home_scoring = data.get('home_scoring_rate', 1.0)
    away_scoring = data.get('away_scoring_rate', 1.0)
    
    # Rule results storage
    triggered_rules = []
    results = {}
    
    # --- Rule 1: Home Fortress ---
    home_form = get_team_form(home_team, limit=6, is_home=True)
    home_fortress, streak, msg = check_home_fortress(home_team, home_form)
    if home_fortress:
        return {
            'prediction': '1',
            'rule': 'Home Fortress',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Home Fortress']
        }
    
    # --- Rule 2: Away Form Killer ---
    away_form = get_team_form(away_team, limit=6, is_home=False)
    away_killer, losses, msg = check_away_form_killer(away_team, away_form)
    if away_killer:
        return {
            'prediction': '1',
            'rule': 'Away Form Killer',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Away Form Killer']
        }
    
    # --- Rule 3: Clean Sheet Streak (Under 2.5) ---
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
    
    # --- Rule 4: Early Goal Tendency ---
    home_early, ratio, msg = check_early_goal_tendency(home_team, home_form)
    away_early, ratio2, msg2 = check_early_goal_tendency(away_team, away_form)
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
    
    # --- Rule 5: Late Goal Tendency (kills draws) ---
    home_late, ratio3, msg3 = check_late_goal_tendency(home_team, home_form)
    away_late, ratio4, msg4 = check_late_goal_tendency(away_team, away_form)
    if (home_late or away_late) and forebet_pred == 'X':
        if home_late and away_late:
            return {
                'prediction': '1' if ratio3 > ratio4 else '2',
                'rule': 'Late Goal Tendency (Both)',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': f"{'Home' if ratio3 > ratio4 else 'Away'} Win',
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
    
    # --- Rule 6: H2H Dominance ---
    h2h_data = get_h2h_history(home_team, away_team, limit=6)
    dominant, wins, draws, msg5 = check_h2h_dominance(h2h_data)
    if dominant:
        # Check fatigue
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
            return {
                'prediction': '1' if dominant == 'home' else '2',
                'rule': 'H2H Dominance',
                'confidence': 'HIGH',
                'stake': '2 units',
                'bet': f"{'Home' if dominant == 'home' else 'Away'} Win",
                'reason': msg5,
                'rules_passed': ['H2H Dominance']
            }
    
    # --- Rule 7: H2H Draw Rate ---
    h2h_draw_rate, draws2, msg6 = check_h2h_draw_rate(h2h_data)
    if h2h_draw_rate:
        return {
            'prediction': 'X',
            'rule': 'H2H Draw Rate',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Draw',
            'reason': msg6,
            'rules_passed': ['H2H Draw Rate']
        }
    
    # --- Rule 8: Midweek Fatigue ---
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
    
    # --- Rule 9: Goal Discrepancy ---
    discrepancy, diff, msg7 = check_goal_discrepancy(forebet_avg, home_scoring, away_scoring)
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
    
    # --- Rule 10: Double Chance Validation ---
    double_chance = data.get('forebet_double_chance', '')
    validated, msg8 = check_double_chance_validation(forebet_pred, double_chance)
    if validated:
        return {
            'prediction': forebet_pred,
            'rule': 'Double Chance Validated',
            'confidence': 'LOW',
            'stake': '0.5 units',
            'bet': f"{'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'}",
            'reason': msg8,
            'rules_passed': ['Double Chance']
        }
    
    # --- Default: Use Forebet ---
    return {
        'prediction': forebet_pred,
        'rule': 'Forebet Default',
        'confidence': 'LOW',
        'stake': '0.25 units',
        'bet': f"{'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'}",
        'reason': 'No rules triggered, using Forebet prediction',
        'rules_passed': ['Forebet Default']
    }

# ============================================================================
# PARSER - MATCH DATA EXTRACTION
# ============================================================================

def parse_match_date(date_val) -> datetime:
    """Parse date from various formats"""
    if not date_val:
        return datetime(1900, 1, 1)
    
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    
    date_str = str(date_val).strip()
    
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        pass
    
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        pass
    
    try:
        return datetime.strptime(date_str, "%d/%m/%Y %H:%M")
    except:
        pass
    
    return datetime(1900, 1, 1)

def parse_text_data(text: str) -> dict:
    """Parse the complete text data from Forebet"""
    # This is a simplified parser - in production you'd expand this
    result = {
        'matches': [],
        'home_table': {},
        'away_table': {},
        'form_data': {}
    }
    
    # Basic parsing logic (you'll expand this based on your data format)
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Look for match data pattern
        if re.search(r'\d{6,}.*°', line):
            # Parse match data (simplified)
            match = parse_match_line(line)
            if match:
                result['matches'].append(match)
    
    return result

def parse_match_line(line: str) -> Optional[dict]:
    """Parse a single match line from Forebet data"""
    try:
        # Basic parsing (you'll expand this based on your actual data format)
        cleaned = line.replace(' ', '')
        
        # Extract percentages
        pct_match = re.search(r'^(\d{2})(\d{2})(\d{2})', cleaned)
        if not pct_match:
            return None
        
        home_pct = int(pct_match.group(1))
        draw_pct = int(pct_match.group(2))
        away_pct = int(pct_match.group(3))
        
        rest = cleaned[6:]
        
        # Extract prediction
        prediction_char = rest[0]
        if prediction_char == 'X':
            prediction = 'X'
        elif prediction_char in ['1', '2']:
            prediction = prediction_char
        else:
            return None
        
        rest = rest[1:]
        
        # Extract score
        dash_pos = rest.find('-')
        if dash_pos == -1:
            return None
        
        score_part = rest[:dash_pos].strip()
        avg_part = rest[dash_pos+1:].strip()
        
        # Extract avg goals
        avg_goals = 2.5
        avg_match = re.search(r'(\d)\.(\d{2})', avg_part)
        if avg_match:
            goals = int(avg_match.group(1))
            dec = int(avg_match.group(2))
            avg_goals = float(f"{goals}.{dec:02d}")
        
        # Extract double chance
        double_chance = ''
        dc_match = re.search(r'([1X2]{2})', avg_part)
        if dc_match:
            double_chance = dc_match.group(1)
        
        return {
            'home_pct': home_pct,
            'draw_pct': draw_pct,
            'away_pct': away_pct,
            'prediction': prediction,
            'avg_goals': avg_goals,
            'double_chance': double_chance,
            'home_team': 'Unknown',  # You'll extract this from surrounding context
            'away_team': 'Unknown',
            'date': datetime.now().strftime("%Y-%m-%d"),
            'is_finished': 'FT' in line
        }
    except Exception:
        return None

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_refined_analysis(match_data: dict, decision: dict):
    """Display the refined analysis results"""
    
    st.markdown(f"""
    <div class="output-card" style="border-left: 4px solid {'#10b981' if decision['confidence'] == 'HIGH' else '#f59e0b' if decision['confidence'] == 'MEDIUM' else '#64748b'};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700;">
                    {match_data.get('home_team', 'Home')} vs {match_data.get('away_team', 'Away')}
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8;">
                    {match_data.get('date', '')}
                </div>
            </div>
            <div style="text-align: right;">
                <span class="rule-badge {'rule-high' if decision['confidence'] == 'HIGH' else 'rule-medium' if decision['confidence'] == 'MEDIUM' else 'rule-low'}">
                    {decision['confidence']}
                </span>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-top: 0.2rem;">
                    Stake: {decision['stake']}
                </div>
            </div>
        </div>
        
        <div style="margin-top: 0.75rem; padding: 0.75rem; background: #0f172a; border-radius: 8px;">
            <div style="font-size: 1.5rem; font-weight: 800; text-align: center; color: {'#10b981' if decision['prediction'] == '1' else '#f59e0b' if decision['prediction'] == 'X' else '#ef4444' if decision['prediction'] == '2' else '#3b82f6'};">
                {decision['bet']}
            </div>
            <div style="text-align: center; font-size: 0.9rem; color: #94a3b8; margin-top: 0.25rem;">
                📋 {decision['rule']}
            </div>
            <div style="text-align: center; font-size: 0.8rem; color: #64748b; margin-top: 0.25rem;">
                {decision['reason']}
            </div>
        </div>
        
        <div style="margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem;">
            <div style="font-size: 0.7rem; color: #94a3b8;">Rules Passed:</div>
            {''.join([f'<span style="font-size: 0.7rem; background: #1e293b; padding: 0.1rem 0.4rem; border-radius: 4px; margin-right: 0.3rem;">{rule}</span>' for rule in decision.get('rules_passed', ['Forebet Default'])])}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎯 Refined Formula V1.1")
    st.caption("Complete implementation with goal timing, clean sheets, and self-learning")
    
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
        
        if st.button("🎯 Analyze with Refined Formula", type="primary"):
            if not text_data or len(text_data.strip()) < 100:
                st.error("❌ Please paste valid data (minimum 100 characters).")
            else:
                try:
                    with st.spinner("Analyzing with Refined Formula V1.1..."):
                        # Parse data
                        parsed = parse_text_data(text_data)
                        matches = parsed.get('matches', [])
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} matches")
                            
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
                            
                            for i, match in enumerate(matches, 1):
                                # Add midweek fixtures
                                match['midweek_fixtures'] = fixtures
                                
                                # Run refined formula
                                decision = refined_formula_decision(match)
                                
                                # Display result
                                display_refined_analysis(match, decision)
                                
                                # Option to save to database
                                if st.button(f"💾 Save Match {i}", key=f"save_{i}"):
                                    # Prepare data for database
                                    db_data = {
                                        'match_date': match.get('date', datetime.now().date()),
                                        'league_name': match.get('league', 'Unknown'),
                                        'home_team': match.get('home_team', 'Unknown'),
                                        'away_team': match.get('away_team', 'Unknown'),
                                        'forebet_home_pct': match.get('home_pct', 0),
                                        'forebet_draw_pct': match.get('draw_pct', 0),
                                        'forebet_away_pct': match.get('away_pct', 0),
                                        'forebet_prediction': match.get('prediction', 'X'),
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
                                        st.warning("⚠️ This match already exists in the database")
                                    elif result:
                                        st.success(f"✅ Saved! ID: {result}")
                                    else:
                                        st.error("❌ Failed to save")
                                
                                st.markdown("---")
                        else:
                            st.error("No matches found in the data.")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())
    
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
                
                # Rule performance breakdown
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
            # Get all matches
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
                
                # Confidence breakdown
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
