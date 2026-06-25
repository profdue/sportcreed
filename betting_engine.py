"""
MATCH ANALYZER V8.0 — UNIVERSAL LOGIC ENGINE (FILE UPLOAD + MULTI-LEAGUE SUPPORT)
Based on analysis of 81 matches across 7 leagues
- Draw predictions are wrong 83% of the time
- Form difference > 2 points → someone wins
- Goals sweet spot: 2.00-2.40 for draws
- Desperation kills draws
- Home team wins 65% of non-draws
- Home desperation = 100% accuracy lock
- UNIVERSAL: Works with ANY league HTML (Norway, Brazil, Premier League, etc.)
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
import re
import json
from io import StringIO

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
st.set_page_config(page_title="Match Analyzer V8.0", page_icon="📊", layout="wide")

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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LEAGUE CONFIGURATION — DYNAMIC FOR ALL LEAGUES
# ============================================================================
def get_league_config(league: str) -> dict:
    """Get league-specific configuration"""
    config = {
        "relegation_threshold": 15,  # Default
        "league_size": 20,           # Default
        "europe_threshold": 4,       # Default
        "goals_fallback": 2.50       # Default
    }
    
    if "Norway" in league or "Eliteserien" in league:
        config["relegation_threshold"] = 15
        config["league_size"] = 16
        config["goals_fallback"] = 2.75
    elif "Brazil" in league or "Serie A" in league and "Brazil" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.66
    elif "Premier" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.75
    elif "La Liga" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.50
    elif "Serie A" in league and "Italy" in league:
        config["relegation_threshold"] = 18
        config["league_size"] = 20
        config["goals_fallback"] = 2.60
    elif "Bundesliga" in league:
        config["relegation_threshold"] = 16
        config["league_size"] = 18
        config["goals_fallback"] = 2.80
    
    return config


def detect_league(raw_text: str) -> str:
    """Detect which league the HTML is from"""
    if 'football-tips-and-predictions-for-norway' in raw_text or 'Eliteserien' in raw_text:
        return "Norway Eliteserien"
    elif 'football-tips-and-predictions-for-brazil' in raw_text or 'Brasileiro Serie A' in raw_text:
        return "Brazil Serie A"
    elif 'football-tips-and-predictions-for-england' in raw_text or 'Premier League' in raw_text:
        return "Premier League"
    elif 'football-tips-and-predictions-for-spain' in raw_text or 'LaLiga' in raw_text:
        return "La Liga"
    elif 'football-tips-and-predictions-for-italy' in raw_text or 'Serie A' in raw_text:
        return "Serie A"
    elif 'football-tips-and-predictions-for-germany' in raw_text or 'Bundesliga' in raw_text:
        return "Bundesliga"
    else:
        return "Unknown League"


# ============================================================================
# UNIVERSAL HTML PARSER — WORKS WITH ANY LEAGUE
# ============================================================================
def extract_all_data_universal(raw_text: str) -> dict:
    """
    UNIVERSAL PARSER — Extracts from ANY league HTML
    Works with Norway, Brazil, Premier League, etc.
    """
    league = detect_league(raw_text)
    league_config = get_league_config(league)
    
    result = {
        "league": league,
        "league_config": league_config,
        "matches": [],
        "standings": {},
        "form_data": {},
        "statistics": {}
    }
    
    # ================================================================
    # SECTION 1: Extract MATCH DATA (UNIVERSAL)
    # ================================================================
    match_pattern = r'<div class="rcnt tr_[01]">(.*?)</div>'
    match_blocks = re.findall(match_pattern, raw_text, re.DOTALL)
    
    for block in match_blocks:
        match = {}
        
        # Team names
        team_pattern = r'<span itemprop="name">([^<]+)</span>'
        teams = re.findall(team_pattern, block)
        if len(teams) >= 2:
            match['home_team'] = teams[0].strip()
            match['away_team'] = teams[1].strip()
        
        # Match URL
        url_pattern = r'<a class="tnmscn" itemprop="url" href="([^"]+)"'
        url_match = re.search(url_pattern, block)
        if url_match:
            match['match_url'] = url_match.group(1)
        
        # Prediction (1, X, or 2)
        pred_pattern = r'<span class="forepr"><span>([1X2])</span></span>'
        pred_match = re.search(pred_pattern, block)
        if pred_match:
            match['prediction'] = pred_match.group(1)
        
        # Correct score
        score_pattern = r'<div class="ex_sc tabonly">(\d+)\s*-\s*(\d+)</div>'
        score_match = re.search(score_pattern, block)
        if score_match:
            match['correct_score_home'] = int(score_match.group(1))
            match['correct_score_away'] = int(score_match.group(2))
        
        # Avg goals
        avg_pattern = r'<div class="avg_sc tabonly">(\d+\.\d+)</div>'
        avg_match = re.search(avg_pattern, block)
        if avg_match:
            match['avg_goals'] = float(avg_match.group(1))
        
        # Percentages — works no matter which span has class="fpr"
        fprc_pattern = r'<div class="fprc">(.*?)</div>'
        fprc_match = re.search(fprc_pattern, block)
        if fprc_match:
            numbers = re.findall(r'>(\d+)<', fprc_match.group(1))
            if len(numbers) >= 3:
                match['home_win_pct'] = int(numbers[0])
                match['draw_pct'] = int(numbers[1])
                match['away_win_pct'] = int(numbers[2])
        
        # Check if finished (has a score)
        finished_pattern = r'<b class="l_scr">(\d+)\s*-\s*(\d+)</b>'
        finished_match = re.search(finished_pattern, block)
        if finished_match:
            match['actual_home'] = int(finished_match.group(1))
            match['actual_away'] = int(finished_match.group(2))
            match['is_finished'] = True
        else:
            match['is_finished'] = False
        
        # Extract score matrix if available
        score_matrix_pattern = r'<td>(?:<b>)?(\d+)\s*-\s*(\d+)(?:</b>)?</td>\s*<td>(\d+)</td>'
        score_matches = re.findall(score_matrix_pattern, block, re.DOTALL)
        match['score_matrix_raw'] = []
        for s in score_matches:
            if len(s) >= 3:
                match['score_matrix_raw'].append({
                    "home_goals": int(s[0]),
                    "away_goals": int(s[1]),
                    "probability": float(s[2]) if s[2] else 0
                })
        
        # Fallback: If avg_goals not found, use league default
        if match.get('avg_goals') is None or match['avg_goals'] == 0:
            match['avg_goals'] = league_config["goals_fallback"]
        
        # Fallback: If prediction not found, use highest percentage
        if not match.get('prediction'):
            if match.get('home_win_pct') and match.get('draw_pct') and match.get('away_win_pct'):
                pcts = {
                    '1': match['home_win_pct'], 
                    'X': match['draw_pct'], 
                    '2': match['away_win_pct']
                }
                match['prediction'] = max(pcts, key=pcts.get)
        
        if match.get('home_team') and match.get('away_team'):
            result["matches"].append(match)
    
    # ================================================================
    # SECTION 2: Extract STANDINGS DATA (UNIVERSAL)
    # ================================================================
    standings_pattern = r'<table class="standings mod_std".*?>(.*?)</table>'
    standings_match = re.search(standings_pattern, raw_text, re.DOTALL)
    
    if standings_match:
        # Team rows
        row_pattern = r'<tr class="color[01]">.*?<td class="std_pos">.*?<span class="std_zn">(\d+)</span>.*?</td>.*?<td class="standing-second-td"><a href="[^"]+">([^<]+)</a></td>.*?<td align="center"><b>(\d+)</b></td>'
        rows = re.findall(row_pattern, standings_match.group(1), re.DOTALL)
        
        for position, team_name, points in rows:
            result["standings"][team_name.strip()] = {
                "position": int(position),
                "points": int(points)
            }
    
    # ================================================================
    # SECTION 3: Extract FORM DATA (UNIVERSAL) — USES ALL GAMES
    # ================================================================
    form_pattern = r'<tr class="tr_[01]">.*?<td width="10".*?>\d+\.</td>.*?<td width="110".*?><a href="[^"]+">([^<]+)</a></td>.*?<ul class="form">(.*?)</ul>'
    form_rows = re.findall(form_pattern, raw_text, re.DOTALL)
    
    for team_name, form_html in form_rows:
        team_name = team_name.strip()
        
        # Count results
        win_count = form_html.count('li-win')
        draw_count = form_html.count('li-draw')
        loss_count = form_html.count('li-lose')
        total_games = win_count + draw_count + loss_count
        
        # Calculate losing streak
        losing_streak = 0
        items = re.findall(r'<li class="li-(win|draw|lose)"', form_html)
        for item in reversed(items):
            if item == 'lose':
                losing_streak += 1
            else:
                break
        
        # Calculate form points from ALL games
        total_points = (win_count * 3) + draw_count
        
        # Normalize form points to "per 5 games" for consistency
        form_points_normalized = (total_points / total_games * 5) if total_games > 0 else 0
        
        # Also calculate last 5 games form points
        last_5 = items[-5:] if len(items) >= 5 else items
        form_points_last_5 = sum(3 if x == 'win' else 1 if x == 'draw' else 0 for x in last_5)
        
        result["form_data"][team_name] = {
            "points": total_points,
            "form_points": form_points_last_5,  # For compatibility with existing logic
            "form_points_normalized": form_points_normalized,
            "losing_streak": losing_streak,
            "wins": win_count,
            "draws": draw_count,
            "losses": loss_count,
            "games_played": total_games
        }
    
    # ================================================================
    # SECTION 4: Extract STATISTICS (UNIVERSAL)
    # ================================================================
    stats_patterns = {
        "home_wins": r'<td>Home wins:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>',
        "draws": r'<td>Draws:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>',
        "away_wins": r'<td>Away wins:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>',
        "under_25": r'<td>Under 2.5 goals:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>',
        "over_25": r'<td>Over 2.5 goals:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>',
        "goals_per_game": r'<td>Goals per game:</td>.*?<td align="center"><b>(\d+\.\d+)</b>',
        "btts": r'<td>Both teams scored games:</td>.*?<td align="center"><b>(\d+)</b>.*?<td align="center"><b>(\d+)%</b>'
    }
    
    for key, pattern in stats_patterns.items():
        match = re.search(pattern, raw_text, re.DOTALL)
        if match:
            if key in ["goals_per_game"]:
                result["statistics"][key] = float(match.group(1))
            else:
                result["statistics"][key] = {
                    "count": int(match.group(1)),
                    "percentage": int(match.group(2))
                }
    
    # If avg_goals from statistics is available, use it as fallback for missing matches
    if "goals_per_game" in result["statistics"]:
        league_avg_goals = result["statistics"]["goals_per_game"]
        for match in result["matches"]:
            if match.get('avg_goals') is None or match['avg_goals'] == 0:
                match['avg_goals'] = league_avg_goals
    
    return result


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
    if away_team in form_data:
        data["away_form_points"] = form_data[away_team]["form_points"]
        data["away_losing_streak"] = form_data[away_team]["losing_streak"]
        data["away_games_played"] = form_data[away_team]["games_played"]
    
    # Set competitive blocks — DYNAMIC based on league
    def get_block(position):
        if position is None:
            return None
        try:
            pos = int(position)
            league_size = league_config["league_size"]
            relegation_threshold = league_config["relegation_threshold"]
            
            if pos <= 4:  # Top 4 = Europe
                return "europe"
            elif pos >= relegation_threshold:  # Relegation zone
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
    
    # Create score matrix from correct score and raw data
    if data.get('correct_score_home') is not None and data.get('correct_score_away') is not None:
        data['score_matrix'].append({
            "score": f"{data['correct_score_home']}-{data['correct_score_away']}",
            "home_goals": data['correct_score_home'],
            "away_goals": data['correct_score_away'],
            "probability": 100.0
        })
    
    # Add any additional score matrix from raw extraction
    if match.get('score_matrix_raw'):
        for s in match['score_matrix_raw']:
            data['score_matrix'].append({
                "score": f"{s['home_goals']}-{s['away_goals']}",
                "home_goals": s['home_goals'],
                "away_goals": s['away_goals'],
                "probability": s['probability']
            })
    
    return data


# ============================================================================
# V8.0 ANALYSIS ENGINE — UNIVERSAL LOGIC
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
    is_draw_predicted = (draw_pct is not None and draw_pct > 0) or pred == 'X'
    
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
            "btts_pct": data.get("btts"),
            "over25_pct": data.get("over_25"),
            "under25_pct": data.get("under_25"),
            "over35_pct": data.get("over_35"),
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
    """Get CSS class for league badge"""
    if "Norway" in league or "Eliteserien" in league:
        return "no"
    elif "Brazil" in league or "Serie A" in league and "Brazil" in league:
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
    is_draw_predicted = (draw_value is not None and draw_value > 0) or prediction == 'X'
    
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
    
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
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
        # Show top 5 by probability (highest first)
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
    st.title("📊 Match Analyzer V8.0")
    st.caption("Universal Logic Engine | File Upload + Multi-League Support | Works with ANY league HTML")
    
    # Show the universal truths
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
        st.markdown("### 📂 Upload Match Data File")
        
        # File upload section
        st.markdown("""
        <div class="upload-container">
            <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">📄 Upload HTML File</p>
            <p style="color: #94a3b8; margin-bottom: 1rem;">Upload the HTML file from Forebet (Norway, Brazil, Premier League, etc.)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'html', 'htm'], label_visibility="collapsed")
        
        # Also keep text input for manual paste
        st.markdown("### ✏️ Or Paste Data Manually")
        raw_text = st.text_area("Match Data", height=200, key="raw_input", 
                               placeholder="Paste HTML data here, or upload a file above...")
        
        # Process uploaded file or text input
        data_to_process = None
        
        if uploaded_file is not None:
            # Read the uploaded file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            data_to_process = stringio.read()
            st.success(f"✅ File loaded: {uploaded_file.name} ({len(data_to_process):,} characters)")
        
        elif raw_text.strip():
            data_to_process = raw_text
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            analyze_clicked = st.button("🔮 ANALYZE V8.0", type="primary")
        
        if analyze_clicked:
            if not data_to_process:
                st.error("Please upload a file or paste data to analyze.")
            else:
                with st.spinner("Analyzing with Universal Logic..."):
                    parsed = extract_all_data_universal(data_to_process)
                
                league = parsed.get("league", "Unknown League")
                matches = parsed.get("matches", [])
                standings = parsed.get("standings", {})
                form_data = parsed.get("form_data", {})
                stats = parsed.get("statistics", {})
                
                if matches:
                    st.success(f"✅ Found {len(matches)} matches in {league}")
                    
                    # Show statistics summary
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
                    
                    # Show standings summary
                    if standings:
                        st.markdown("### 🏆 Standings (Top 5)")
                        standings_df = pd.DataFrame([
                            {"Pos": data["position"], "Team": team, "Pts": data["points"]}
                            for team, data in list(standings.items())[:5]
                        ])
                        st.dataframe(standings_df, use_container_width=True, hide_index=True)
                    
                    # Process each match
                    for idx, match in enumerate(matches):
                        st.markdown(f"### Match {idx + 1}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}")
                        
                        # Show form data if available
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
                        
                        # Convert and analyze
                        data = convert_match_to_data(match, standings, form_data, league)
                        analysis = analyze_match_v8(data)
                        save_to_db(data, analysis, league)
                        
                        display_analysis(data, analysis, league)
                        
                        if idx < len(matches) - 1:
                            st.markdown("---")
                else:
                    st.error("No matches found in the data. Please make sure you're uploading valid Forebet HTML data.")
    
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
            
            # Summary stats
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
            
            # Results table
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
