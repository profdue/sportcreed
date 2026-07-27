"""
REFINED FORMULA - YOUR 5 RULES
Single table: match_predictions
NO FALLBACK DATA - extracts ONLY real data from text
"""

import streamlit as st
from datetime import date, datetime, timedelta
from supabase import create_client, Client
import pandas as pd
import re
import json
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

TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Refined Formula V1.1", page_icon="🎯", layout="wide")

# ============================================================================
# DATABASE HELPERS
# ============================================================================

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

def get_pending_matches():
    try:
        response = supabase.table(TABLE_NAME).select("*").is_("actual_1x2", "null").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching pending: {e}")
        return []

def get_completed_matches():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_1x2", "null").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching results: {e}")
        return []

def get_all_matches():
    try:
        response = supabase.table(TABLE_NAME).select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching matches: {e}")
        return []

# ============================================================================
# PARSER - EXTRACTS ONLY REAL DATA, NO FALLBACKS
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
    return name if len(name) > 2 else None

def parse_encoded_line(line: str) -> dict:
    """
    Parse the encoded line: 255421X1 - 12.1526°3.80
    Returns: {home_pct, draw_pct, away_pct, prediction, score_home, score_away, avg_goals}
    Returns None if data cannot be extracted
    """
    # Remove spaces
    cleaned = line.replace(' ', '')
    
    # Extract 6 digits + prediction
    pct_match = re.search(r'(\d{2})(\d{2})(\d{2})([1X2])', cleaned)
    if not pct_match:
        return None
    
    home_pct = int(pct_match.group(1))
    draw_pct = int(pct_match.group(2))
    away_pct = int(pct_match.group(3))
    prediction = pct_match.group(4)
    
    # Extract score - look for pattern like "1 - 1" or "1-1"
    score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
    if not score_match:
        return None
    
    score_home = int(score_match.group(1))
    score_away = int(score_match.group(2))
    
    # Extract avg goals - look for pattern like "2.15°"
    avg_match = re.search(r'(\d+\.\d{2})\s*°', line)
    if not avg_match:
        return None
    
    avg_goals = float(avg_match.group(1))
    
    # Extract double chance - look for 1X, X2, 12
    dc_match = re.search(r'([1X2]{2})', line)
    double_chance = dc_match.group(1) if dc_match else None
    
    return {
        'home_pct': home_pct,
        'draw_pct': draw_pct,
        'away_pct': away_pct,
        'prediction': prediction,
        'score_home': score_home,
        'score_away': score_away,
        'avg_goals': avg_goals,
        'double_chance': double_chance
    }

def parse_form_data(lines: List[str], start_idx: int, team_name: str, match_type: str) -> Tuple[List[str], int]:
    """
    Parse form data from "home matches" or "away matches" section
    Returns: (form_results, next_index)
    """
    form_results = []
    i = start_idx
    
    # Find the Win/Draw/Lost line
    while i < len(lines) and i < start_idx + 20:
        line = lines[i].strip()
        
        win_match = re.search(r'Win\s+(\d+)\s+(\d+)%', line)
        draw_match = re.search(r'Draw\s+(\d+)\s+(\d+)%', line)
        loss_match = re.search(r'Lost\s+(\d+)\s+(\d+)%', line)
        
        if win_match or draw_match or loss_match:
            wins = int(win_match.group(1)) if win_match else 0
            draws = int(draw_match.group(1)) if draw_match else 0
            losses = int(loss_match.group(1)) if loss_match else 0
            
            # Build form string: W repeated wins, D repeated draws, L repeated losses
            form_results = ['W'] * wins + ['D'] * draws + ['L'] * losses
            return form_results, i + 1
        
        i += 1
    
    return [], i

def parse_h2h_data(lines: List[str], start_idx: int) -> Tuple[List[dict], int]:
    """
    Parse H2H data from "Head to head" section
    Returns: (h2h_matches, next_index)
    """
    h2h_matches = []
    i = start_idx + 1
    h2h_count = 0
    
    while i < len(lines) and i < start_idx + 25 and h2h_count < 6:
        line = lines[i].strip()
        
        # Look for date pattern
        if re.search(r'\d{2}/\d{2}/\d{4}', line):
            # Parse the H2H line
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            score_match = re.search(r'(\d+)\s*-\s*(\d+)', line)
            
            if date_match and score_match:
                date_str = date_match.group(1)
                home_goals = int(score_match.group(1))
                away_goals = int(score_match.group(2))
                
                # Determine winner
                if home_goals > away_goals:
                    winner = 'home'
                elif away_goals > home_goals:
                    winner = 'away'
                else:
                    winner = 'draw'
                
                # Extract team names
                parts = re.split(r'\d+\s*-\s*\d+', line)
                if len(parts) >= 2:
                    left = re.sub(r'\d{2}/\d{2}/\d{4}', '', parts[0]).strip()
                    right = parts[1].strip()
                    home_team = clean_team_name(left)
                    away_team = clean_team_name(right)
                else:
                    home_team = "Unknown"
                    away_team = "Unknown"
                
                if home_team and away_team and home_team != "Unknown" and away_team != "Unknown":
                    h2h_matches.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'match_date': date_str,
                        'home_goals': home_goals,
                        'away_goals': away_goals,
                        'winner': winner
                    })
                    h2h_count += 1
        
        i += 1
    
    return h2h_matches, i

def parse_text_data(text: str) -> dict:
    """
    Parse the complete text data from Forebet
    NO FALLBACKS - returns only real data found
    """
    result = {
        'matches': [],
        'league': None
    }
    
    if not text or len(text.strip()) < 100:
        return result
    
    lines = text.split('\n')
    
    # Detect league
    league_keywords = ['Superliga', 'Premier League', 'Serie A', 'La Liga', 'Bundesliga', 
                       'Ligue 1', 'Serie B', 'Championship', 'Russia Premier League', 'EPL']
    league = None
    for line in lines:
        for kw in league_keywords:
            if kw in line:
                league = line.strip()
                break
        if league:
            break
    
    # Find match
    current_match = {}
    match_found = False
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # ----- Find match header (Team VS Team) -----
        if ' VS ' in line:
            parts = line.split(' VS ')
            if len(parts) == 2:
                home = clean_team_name(parts[0])
                away = clean_team_name(parts[1])
                if home and away:
                    current_match = {
                        'home_team': home,
                        'away_team': away,
                        'date': None,
                        'league': league,
                        'home_pct': None,
                        'draw_pct': None,
                        'away_pct': None,
                        'forebet_prediction': None,
                        'avg_goals': None,
                        'h2h_data': [],
                        'home_form': [],
                        'away_form': [],
                        'correct_score_home': None,
                        'correct_score_away': None,
                        'double_chance': None,
                        'is_finished': False,
                        'actual_home': None,
                        'actual_away': None
                    }
                    match_found = True
                    
                    # Find date nearby
                    for j in range(max(0, i-5), min(len(lines), i+5)):
                        dt_line = lines[j].strip()
                        dt_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2})', dt_line)
                        if dt_match:
                            try:
                                dt = datetime.strptime(dt_match.group(1), "%d/%m/%Y")
                                current_match['date'] = dt.strftime("%Y-%m-%d")
                            except:
                                pass

        # ----- Parse encoded data line -----
        if match_found and re.search(r'\d{6}[1X2]', line):
            encoded = parse_encoded_line(line)
            if encoded:
                current_match['home_pct'] = encoded['home_pct']
                current_match['draw_pct'] = encoded['draw_pct']
                current_match['away_pct'] = encoded['away_pct']
                current_match['forebet_prediction'] = encoded['prediction']
                current_match['correct_score_home'] = encoded['score_home']
                current_match['correct_score_away'] = encoded['score_away']
                current_match['avg_goals'] = encoded['avg_goals']
                current_match['double_chance'] = encoded['double_chance']

        # ----- Check if match is finished -----
        if match_found and 'FT' in line:
            current_match['is_finished'] = True
            ft_score = re.search(r'(\d+)\s*-\s*(\d+)', line)
            if ft_score:
                current_match['actual_home'] = int(ft_score.group(1))
                current_match['actual_away'] = int(ft_score.group(2))

        # ----- Parse H2H section -----
        if match_found and ('Head to head' in line or 'H2H' in line):
            h2h_data, next_idx = parse_h2h_data(lines, i)
            if h2h_data:
                current_match['h2h_data'] = h2h_data
            i = next_idx
            continue

        # ----- Parse Home Form -----
        if match_found and ('home matches' in line.lower() or 'Home matches' in line):
            # Look for the team name in previous lines
            team_name = None
            for k in range(max(0, i-3), i):
                prev = lines[k].strip()
                if current_match['home_team'] in prev:
                    team_name = current_match['home_team']
                    break
            
            if team_name:
                form_data, next_idx = parse_form_data(lines, i + 1, team_name, 'home')
                if form_data:
                    current_match['home_form'] = form_data
                i = next_idx
                continue

        # ----- Parse Away Form -----
        if match_found and ('away matches' in line.lower() or 'Away matches' in line):
            team_name = None
            for k in range(max(0, i-3), i):
                prev = lines[k].strip()
                if current_match['away_team'] in prev:
                    team_name = current_match['away_team']
                    break
            
            if team_name:
                form_data, next_idx = parse_form_data(lines, i + 1, team_name, 'away')
                if form_data:
                    current_match['away_form'] = form_data
                i = next_idx
                continue

        # ----- Save complete match -----
        if match_found and current_match.get('home_team') and current_match.get('away_team'):
            # Only save if we have the essential data
            has_essential = (
                current_match.get('home_pct') is not None and
                current_match.get('draw_pct') is not None and
                current_match.get('away_pct') is not None and
                current_match.get('forebet_prediction') is not None and
                current_match.get('correct_score_home') is not None and
                current_match.get('correct_score_away') is not None and
                current_match.get('avg_goals') is not None
            )
            
            if has_essential:
                # Check if already added
                already_added = False
                for m in result['matches']:
                    if (m.get('home_team') == current_match.get('home_team') and
                        m.get('away_team') == current_match.get('away_team')):
                        already_added = True
                        break
                
                if not already_added:
                    result['matches'].append(current_match.copy())
                    
                    # Reset for next match
                    current_match = {}
                    match_found = False

        i += 1

    if league:
        result['league'] = league

    return result

# ============================================================================
# YOUR 5 RULES
# ============================================================================

def check_home_fortress(home_form: List[str]) -> Tuple[bool, int, str]:
    """Rule 1: Home Fortress - Unbeaten in last 5 home games"""
    if len(home_form) < 5:
        return False, 0, f"Only {len(home_form)} home matches available (need 5)"
    
    recent = home_form[:5]
    unbeaten = sum(1 for r in recent if r != 'L')
    if unbeaten >= 5:
        return True, unbeaten, f"Unbeaten in last {unbeaten} home games"
    return False, unbeaten, f"Only {unbeaten}/5 unbeaten"

def check_away_form_killer(away_form: List[str]) -> Tuple[bool, int, str]:
    """Rule 2: Away Form Killer - Lost 4 of last 6 away games"""
    if len(away_form) < 6:
        return False, 0, f"Only {len(away_form)} away matches available (need 6)"
    
    recent = away_form[:6]
    losses = sum(1 for r in recent if r == 'L')
    if losses >= 4:
        return True, losses, f"Lost {losses}/6 away games"
    return False, losses, f"Only {losses}/6 losses"

def check_h2h_dominance(h2h_data: List[dict]) -> Tuple[Optional[str], int, int, str]:
    """Rule 3: H2H Dominance - One team won 3 of last 4 H2Hs"""
    if len(h2h_data) < 4:
        return None, 0, 0, f"Only {len(h2h_data)} H2H matches available (need 4)"
    
    recent = h2h_data[:4]
    home_wins = sum(1 for m in recent if m.get('winner') == 'home')
    away_wins = sum(1 for m in recent if m.get('winner') == 'away')
    draws = sum(1 for m in recent if m.get('winner') == 'draw')
    
    if home_wins >= 3:
        return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs → Draw is a trap"
    elif away_wins >= 3:
        return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs → Draw is a trap"
    return None, max(home_wins, away_wins), draws, f"No dominance (H:{home_wins}, A:{away_wins}, D:{draws})"

def check_h2h_draw_rate(h2h_data: List[dict]) -> Tuple[bool, int, str]:
    """Rule 4: H2H Draw Rate - 4+ draws in last 6 H2Hs"""
    if len(h2h_data) < 6:
        return False, 0, f"Only {len(h2h_data)} H2H matches available (need 6)"
    
    recent = h2h_data[:6]
    draws = sum(1 for m in recent if m.get('winner') == 'draw')
    if draws >= 4:
        return True, draws, f"{draws}/6 H2Hs were draws → Trust the Draw"
    return False, draws, f"Only {draws}/6 draws"

def check_midweek_fatigue(team: str, match_date, fixtures: List[dict]) -> Tuple[bool, str]:
    """Rule 5: Midweek Fatigue - Away played 3-4 days ago"""
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
    }
    return stake_map.get(stake, (stake, "LOW"))

# ============================================================================
# REFINED FORMULA DECISION - YOUR 5 RULES
# ============================================================================

def refined_formula_decision(data: dict) -> dict:
    """Your 5 Rules - uses parsed data only"""
    home_team = data.get('home_team', 'Unknown')
    away_team = data.get('away_team', 'Unknown')
    match_date = data.get('date')
    forebet_pred = data.get('forebet_prediction', 'X')
    fixtures = data.get('midweek_fixtures', [])
    
    home_form = data.get('home_form', [])
    away_form = data.get('away_form', [])
    h2h_data = data.get('h2h_data', [])
    
    home_fatigued, home_fatigue_msg = check_midweek_fatigue(home_team, match_date, fixtures)
    away_fatigued, away_fatigue_msg = check_midweek_fatigue(away_team, match_date, fixtures)
    
    # Rule 1: Home Fortress
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
    
    # Rule 2: Away Form Killer
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
    
    # Rule 3: H2H Dominance
    dominant, wins, draws, msg = check_h2h_dominance(h2h_data)
    if dominant:
        pred = '1' if dominant == 'home' else '2'
        winner = 'Home' if dominant == 'home' else 'Away'
        if (dominant == 'home' and home_fatigued) or (dominant == 'away' and away_fatigued):
            return {
                'prediction': 'X',
                'rule': 'H2H Dominance + Fatigue',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Draw',
                'reason': f"{msg} but {winner} team fatigued → Draw",
                'rules_passed': ['H2H Dominance', 'Midweek Fatigue']
            }
        return {
            'prediction': pred,
            'rule': 'H2H Dominance',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': f"{winner} Win",
            'reason': msg,
            'rules_passed': ['H2H Dominance']
        }
    
    # Rule 4: H2H Draw Rate
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
    
    # Rule 5: Midweek Fatigue
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
    
    # Default: Trust Forebet
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
# DISPLAY FUNCTION
# ============================================================================

def display_refined_analysis(match_data: dict, decision: dict, league: str = "Unknown"):
    home_team = match_data.get('home_team', 'Unknown')
    away_team = match_data.get('away_team', 'Unknown')
    home_pct = match_data.get('home_pct', '?')
    draw_pct = match_data.get('draw_pct', '?')
    away_pct = match_data.get('away_pct', '?')
    
    stake_display, confidence_level = get_stake_display(decision.get('stake', '0.25 units'))
    conf_color = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'gray'}.get(decision.get('confidence', 'LOW'), 'gray')
    pred_emoji = {'1': '🏠', 'X': '🤝', '2': '✈️'}.get(decision.get('prediction', 'X'), '🎯')
    
    st.subheader(f"{pred_emoji} {home_team} vs {away_team}")
    st.caption(f"📅 {match_data.get('date', '')} | 🏆 {league}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🏠 Home", f"{home_pct}%")
    c2.metric("🤝 Draw", f"{draw_pct}%")
    c3.metric("✈️ Away", f"{away_pct}%")
    
    st.markdown("---")
    st.markdown(f"### 🎯 {decision.get('bet', 'Unknown')}")
    st.markdown(f"**Rule:** {decision.get('rule', 'Unknown')}")
    st.markdown(f"**Confidence:** :{conf_color}[{decision.get('confidence', 'LOW')}]")
    st.markdown(f"**Stake:** {stake_display}")
    st.markdown(f"**Reason:** {decision.get('reason', '')}")
    
    rules_passed = decision.get('rules_passed', ['Forebet Default'])
    st.caption(f"📋 Rules Passed: {', '.join(rules_passed)}")
    st.caption(f"📊 Forebet Original: {match_data.get('forebet_prediction', '?')} | Avg Goals: {match_data.get('avg_goals', '?')}")
    st.markdown("---")
    
    # Show H2H data
    if match_data.get('h2h_data'):
        with st.expander("📊 Head-to-Head History"):
            h2h_df = pd.DataFrame(match_data['h2h_data'])
            st.dataframe(h2h_df, use_container_width=True)
    
    # Show form data
    if match_data.get('home_form') or match_data.get('away_form'):
        with st.expander("📈 Recent Form"):
            if match_data.get('home_form'):
                st.write(f"**{home_team} (home):** {' '.join(match_data['home_form'])} ({len(match_data['home_form'])} matches)")
            if match_data.get('away_form'):
                st.write(f"**{away_team} (away):** {' '.join(match_data['away_form'])} ({len(match_data['away_form'])} matches)")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🎯 Refined Formula V1.1")
    st.caption("Your 5 Rules: Home Fortress | Away Form Killer | H2H Dominance | H2H Draw Rate | Midweek Fatigue")
    st.info(f"📊 Using table: `{TABLE_NAME}`")

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Pending", "📊 Records", "📈 Dashboard"])
    
    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.warning("⚠️ No fallback data - only real data extracted from your text will be used")
        
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
                                
                                # Run YOUR refined formula
                                decision = refined_formula_decision(match)
                                
                                # Display
                                display_refined_analysis(match, decision, league)
                                
                                # Prepare data for database
                                h2h_json = json.dumps(match.get('h2h_data', []))
                                home_form_str = ','.join(match.get('home_form', []))
                                away_form_str = ','.join(match.get('away_form', []))
                                
                                # Calculate H2H stats
                                h2h_data = match.get('h2h_data', [])
                                h2h_dominance = None
                                h2h_dominance_count = 0
                                h2h_draw_count = 0
                                
                                if len(h2h_data) >= 4:
                                    home_wins = sum(1 for m in h2h_data[:4] if m.get('winner') == 'home')
                                    away_wins = sum(1 for m in h2h_data[:4] if m.get('winner') == 'away')
                                    if home_wins >= 3:
                                        h2h_dominance = 'home'
                                        h2h_dominance_count = home_wins
                                    elif away_wins >= 3:
                                        h2h_dominance = 'away'
                                        h2h_dominance_count = away_wins
                                
                                if len(h2h_data) >= 6:
                                    h2h_draw_count = sum(1 for m in h2h_data[:6] if m.get('winner') == 'draw')
                                
                                db_data = {
                                    'match_date': match.get('date', datetime.now().date()),
                                    'league_name': league if league else 'Unknown',
                                    'home_team': match.get('home_team', 'Unknown'),
                                    'away_team': match.get('away_team', 'Unknown'),
                                    'season_round': None,
                                    'forebet_home_pct': match.get('home_pct', 0),
                                    'forebet_draw_pct': match.get('draw_pct', 0),
                                    'forebet_away_pct': match.get('away_pct', 0),
                                    'forebet_prediction': match.get('forebet_prediction', 'X'),
                                    'forebet_correct_score_home': match.get('correct_score_home'),
                                    'forebet_correct_score_away': match.get('correct_score_away'),
                                    'forebet_avg_goals': match.get('avg_goals', 2.5),
                                    'forebet_double_chance': match.get('double_chance'),
                                    'home_form': home_form_str,
                                    'away_form': away_form_str,
                                    'h2h_data': h2h_json,
                                    'h2h_dominance': h2h_dominance,
                                    'h2h_dominance_count': h2h_dominance_count,
                                    'h2h_draw_count': h2h_draw_count,
                                    'midweek_fatigue_home': False,
                                    'midweek_fatigue_away': False,
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
                                    st.warning(f"⚠️ Match {i} already exists (skipped)")
                                elif result:
                                    saved_count += 1
                                    st.success(f"✅ Match {i} saved with ID: {result}")
                                else:
                                    st.error(f"❌ Failed to save match {i}")
                                
                                st.markdown("---")
                            
                            st.info(f"📊 Summary: {saved_count} saved, {duplicate_count} duplicates skipped.")
                        else:
                            st.error("❌ No matches found in the data. Please check the format.")
                            st.info("The parser needs:\n- 'Team VS Team' line\n- Encoded data like '255421X1 - 12.1526°3.80'\n- Form data (Win X Y%, Draw X Y%, Lost X Y%)\n- H2H data (Head to head section)")
                            
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("📝 Pending Matches")
        st.caption("Enter actual results for completed matches")
        pending = get_pending_matches()
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
                            st.success("✅ Result submitted!")
                            st.rerun()
        else:
            st.info("No pending matches.")
    
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_completed_matches()
        if results:
            total = len(results)
            correct = sum(1 for r in results if r.get('is_correct', False))
            rate = round(correct / total * 100) if total > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Matches", total)
            c2.metric("Correct", correct)
            c3.metric("Accuracy", f"{rate}%")
            
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
    
    with tab4:
        st.subheader("📈 Dashboard")
        all_matches = get_all_matches()
        if all_matches:
            total = len(all_matches)
            pending = sum(1 for m in all_matches if m.get('actual_1x2') is None)
            completed = total - pending
            correct = sum(1 for m in all_matches if m.get('is_correct', False))
            rate = round(correct / completed * 100) if completed > 0 else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Matches", total)
            c2.metric("Completed", completed)
            c3.metric("Correct", correct)
            c4.metric("Accuracy", f"{rate}%")
            
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

if __name__ == "__main__":
    main()
