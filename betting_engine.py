import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import traceback
import numpy as np

# Try to import sklearn, fall back to simple rules if not available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import pickle
import os

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
# TABLE NAME
# ============================================================================
TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Advanced No-Draw Predictor", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .no-draw-card { border-left: 5px solid #10b981; background: linear-gradient(135deg, #0a2a1a 0%, #0a1a0a 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .draw-card { border-left: 5px solid #3b82f6; background: linear-gradient(135deg, #0a1a2a 0%, #0a0a1a 100%); }
    .over-card { border-left: 5px solid #8b5cf6; background: linear-gradient(135deg, #1a0a2a 0%, #0a0a1a 100%); }
    .under-card { border-left: 5px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0a00 100%); }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .metric-card { background: #0f172a; border-radius: 10px; padding: 0.75rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; color: #94a3b8; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-no-draw { color: #10b981; }
    .prediction-skip { color: #f59e0b; }
    .prediction-draw { color: #3b82f6; }
    .prediction-over { color: #8b5cf6; }
    .prediction-under { color: #f59e0b; }
    .final-badge { background: #10b981; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 2px solid #10b981; }
    .no-draw-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .skip-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .draw-badge { background: #3b82f6; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .over-badge { background: #8b5cf6; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .under-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .factor-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }
    .factor-name { color: #94a3b8; }
    .factor-value { font-weight: 600; }
    .feature-box { background: #0f172a; border-radius: 6px; padding: 0.5rem; margin: 0.25rem 0; }
    .feature-label { color: #94a3b8; font-size: 0.7rem; }
    .feature-value { font-weight: 700; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_match_date(date_val) -> datetime:
    if not date_val:
        return datetime(1900, 1, 1)
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    date_str = str(date_val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return datetime(1900, 1, 1)


def format_date_display(date_val) -> str:
    dt = parse_match_date(date_val)
    if dt.year == 1900:
        return str(date_val)
    return dt.strftime("%Y-%m-%d")


def check_match_exists(home_team: str, away_team: str, match_date: str) -> bool:
    try:
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        response = supabase.table(TABLE_NAME).select("id").eq("home_team", home_team).eq("away_team", away_team).eq("match_date", date_part).execute()
        return len(response.data) > 0
    except:
        return False


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def calculate_team_profile(team_data: dict) -> dict:
    """Calculate team profile metrics for the new betting logic"""
    
    # Count appearances by category
    positive_sections = ["best_team", "best_off", "best_def"]
    negative_sections = ["worst_team", "worst_off", "worst_def", "l_team"]
    
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_appearances = 0
    
    for section in positive_sections:
        if team_data.get(section, 0) > 0:
            positive_count += 1
            total_appearances += 1
    
    for section in negative_sections:
        if team_data.get(section, 0) > 0:
            negative_count += 1
            total_appearances += 1
    
    # Neutral sections (W, D, NW, ND)
    neutral_sections = ["w_team", "d_team", "nw_team", "nd_team"]
    for section in neutral_sections:
        if team_data.get(section, 0) > 0:
            neutral_count += 1
            total_appearances += 1
    
    # Calculate ratios (handle division by zero)
    offensive_ratio = positive_count / total_appearances if total_appearances > 0 else 0.5
    defensive_ratio = negative_count / total_appearances if total_appearances > 0 else 0.5
    
    # Determine profile
    if total_appearances == 0:
        profile = "WEAK_PROFILE"
    elif positive_count >= 2 and negative_count == 0:
        profile = "POSITIVE"
    elif negative_count >= 2 and positive_count == 0:
        profile = "NEGATIVE"
    elif positive_count >= 2 and negative_count >= 2:
        profile = "MIXED"
    elif total_appearances >= 3:
        profile = "ESTABLISHED"
    else:
        profile = "WEAK_PROFILE"
    
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_appearances": total_appearances,
        "offensive_ratio": offensive_ratio,
        "defensive_ratio": defensive_ratio,
        "profile": profile
    }


def engineer_features(match_data: dict) -> dict:
    """Engineer features for the predictive model"""
    
    home_profile = calculate_team_profile(match_data.get("home_team_data", {}))
    away_profile = calculate_team_profile(match_data.get("away_team_data", {}))
    
    draw_odds = match_data.get("draw_odds", 0)
    home_odds = match_data.get("home_odds", 0)
    away_odds = match_data.get("away_odds", 0)
    
    # Calculate implied probabilities
    draw_implied = 1 / draw_odds if draw_odds > 0 else 0
    dc12_odds = 1 / ((1 / home_odds) + (1 / away_odds)) if home_odds > 0 and away_odds > 0 else 0
    dc_implied_no_draw = 1 / dc12_odds if dc12_odds > 0 else 0
    
    # Profile difference
    profile_difference = home_profile["offensive_ratio"] - away_profile["defensive_ratio"]
    
    # Both weak indicator
    both_weak = 1 if (home_profile["profile"] == "WEAK_PROFILE" and away_profile["profile"] == "WEAK_PROFILE") else 0
    
    # Calculate total offensive/defensive ratio
    home_off_ratio = home_profile["offensive_ratio"]
    away_def_ratio = away_profile["defensive_ratio"]
    
    # Over/under indicators
    total_off_ratio = home_profile["offensive_ratio"] + away_profile["offensive_ratio"]
    
    features = {
        "draw_odds_implied": draw_implied,
        "home_off_ratio": home_off_ratio,
        "away_def_ratio": away_def_ratio,
        "profile_difference": profile_difference,
        "both_weak": both_weak,
        "best_team_home": match_data.get("home_team_data", {}).get("best_team", 0),
        "worst_def_away": match_data.get("away_team_data", {}).get("worst_def", 0),
        "dc_implied_no_draw": dc_implied_no_draw,
        "total_off_ratio": total_off_ratio,
        "draw_odds": draw_odds,
    }
    
    return features


# ============================================================================
# DECISION TREE RULES (Simplified from Logistic Regression)
# ============================================================================
def apply_decision_rules(features: dict) -> dict:
    """
    Apply the simplified decision tree rules derived from the logistic regression.
    Returns prediction, confidence, and bet type.
    """
    
    draw_odds = features.get('draw_odds', 0)
    home_off_ratio = features.get('home_off_ratio', 0)
    away_def_ratio = features.get('away_def_ratio', 0)
    profile_difference = features.get('profile_difference', 0)
    both_weak = features.get('both_weak', 0)
    total_off_ratio = features.get('total_off_ratio', 0)
    best_team_home = features.get('best_team_home', 0)
    worst_def_away = features.get('worst_def_away', 0)
    
    # Primary Rule: Strong No-Draw signal
    if draw_odds > 4.50 and home_off_ratio > 0.30 and away_def_ratio < 0.25:
        return {
            'prediction': 'NO_DRAW',
            'confidence': 'HIGH',
            'action': '✅ BET - No Draw expected (Double Chance 12)',
            'reason': 'High draw odds + strong home offense + weak away defense',
            'bet_type': 'NO_DRAW'
        }
    
    # Secondary Rule: Medium No-Draw signal
    if draw_odds > 3.80 and profile_difference > 0.15 and both_weak == 0:
        return {
            'prediction': 'NO_DRAW',
            'confidence': 'MEDIUM',
            'action': '✅ BET - No Draw expected (Double Chance 12)',
            'reason': 'Good draw odds + significant profile difference',
            'bet_type': 'NO_DRAW'
        }
    
    # Draw Rule: Avoid betting on No-Draw
    if draw_odds <= 4.00 and both_weak == 1 and home_off_ratio < 0.20 and away_def_ratio > 0.30:
        return {
            'prediction': 'DRAW',
            'confidence': 'MEDIUM',
            'action': '❌ AVOID - Draw likely',
            'reason': 'Low draw odds + both teams weak + defensive mismatch',
            'bet_type': 'DRAW'
        }
    
    # Over 2.5 Goals Rule
    if draw_odds > 5.0 and total_off_ratio > 0.80:
        return {
            'prediction': 'OVER_2.5',
            'confidence': 'MEDIUM',
            'action': '⚽ BET - Over 2.5 goals expected',
            'reason': 'High draw odds + high offensive output',
            'bet_type': 'OVER'
        }
    
    # Under 2.5 Goals Rule
    if draw_odds < 3.5 and total_off_ratio < 0.50:
        return {
            'prediction': 'UNDER_2.5',
            'confidence': 'MEDIUM',
            'action': '⚽ BET - Under 2.5 goals expected',
            'reason': 'Low draw odds + low offensive output',
            'bet_type': 'UNDER'
        }
    
    # Weak No-Draw signal (consider but low confidence)
    if draw_odds > 4.0 and (best_team_home == 1 or worst_def_away == 1):
        return {
            'prediction': 'NO_DRAW',
            'confidence': 'LOW',
            'action': '⚠️ CONSIDER - Weak no-draw signal',
            'reason': 'Moderate signal - consider as value bet',
            'bet_type': 'NO_DRAW'
        }
    
    # Default: Skip
    return {
        'prediction': 'SKIP',
        'confidence': 'LOW',
        'action': '❌ SKIP - Insufficient evidence',
        'reason': 'No clear signal from decision rules',
        'bet_type': 'SKIP'
    }


# ============================================================================
# COMPLETE PARSER - Extracts ALL data from ALL pages
# ============================================================================
def parse_betexplorer_data(text: str) -> list:
    """Parse Betexplorer data - extracts matches from ALL pages."""
    matches = []
    lines = text.split('\n')
    
    # Store data by team name
    team_cache = {}
    match_cache = {}
    current_page_type = None
    current_country = None
    
    def get_or_create_team(team_name):
        if team_name not in team_cache:
            team_cache[team_name] = {
                "team_name": team_name,
                "w_team": 0, "d_team": 0, "l_team": 0,
                "nw_team": 0, "nd_team": 0,
                "best_team": 0, "worst_team": 0,
                "best_off": 0, "best_def": 0,
                "worst_off": 0, "worst_def": 0,
                "appearances": []
            }
        return team_cache[team_name]
    
    def get_or_create_match(home_team, away_team, home_odds=0, draw_odds=0, away_odds=0):
        match_key = f"{home_team}|{away_team}"
        if match_key not in match_cache:
            match_cache[match_key] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "home_team_data": {},
                "away_team_data": {},
                "date": datetime.now().strftime("%Y-%m-%d"),
                "league": current_country or "Unknown",
            }
        if home_odds > 0 and draw_odds > 0 and away_odds > 0:
            current_odds = match_cache[match_key].get('home_odds', 0)
            if home_odds > current_odds:
                match_cache[match_key]['home_odds'] = home_odds
                match_cache[match_key]['draw_odds'] = draw_odds
                match_cache[match_key]['away_odds'] = away_odds
        return match_cache[match_key]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect page type
        if 'Team\tW\tNext match' in line or 'Team, W, Next match' in line:
            current_page_type = 'wins'
            continue
        elif 'Team\tD\tNext match' in line or 'Team, D, Next match' in line:
            current_page_type = 'draws'
            continue
        elif 'Team\tL\tNext match' in line or 'Team, L, Next match' in line:
            current_page_type = 'losses'
            continue
        elif 'Team\tNW\tNext match' in line or 'Team, NW, Next match' in line:
            current_page_type = 'no_wins'
            continue
        elif 'ND' in line and ('Next match' in line or '1\tX\t2' in line) or 'Team\tND\tNext match' in line or 'Team, ND, Next match' in line:
            current_page_type = 'no_draws'
            continue
        elif 'Team\tNL\tNext match' in line or 'Team, NL, Next match' in line:
            current_page_type = 'no_losses'
            continue
        elif 'Best teams' in line or 'Less streaks' in line:
            current_page_type = 'best_teams'
            continue
        elif 'Worst teams' in line:
            current_page_type = 'worst_teams'
            continue
        elif 'Best offensive' in line:
            current_page_type = 'best_offensive'
            continue
        elif 'Best defensive' in line:
            current_page_type = 'best_defensive'
            continue
        elif 'Worst offensive' in line:
            current_page_type = 'worst_offensive'
            continue
        elif 'Worst defensive' in line:
            current_page_type = 'worst_defensive'
            continue
        
        # Detect country/league
        if re.match(r'^[A-Za-z\s]+$', line) and not re.search(r'[0-9.]', line) and len(line) < 30:
            if line not in ['Team', 'W', 'D', 'L', 'NW', 'ND', 'NL', 'Best teams', 'Worst teams', 'Best offensive', 'Best defensive', 'Worst offensive', 'Worst defensive', 'Next match']:
                current_country = line
                continue
        
        # Parse page-specific data
        if current_page_type in ['wins', 'draws', 'losses', 'no_wins', 'no_draws', 'no_losses']:
            parts = re.split(r'\t+', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 4:
                try:
                    if re.search(r'[A-Za-z]', parts[0]) and re.search(r'\d', parts[1]):
                        team = parts[0]
                        streak_value = int(re.search(r'\d+', parts[1]).group()) if re.search(r'\d+', parts[1]) else 0
                        next_match = parts[2]
                        
                        match_parts = re.split(r'\s*[-–]\s*', next_match)
                        if len(match_parts) == 2:
                            home_team = match_parts[0].strip()
                            away_team = match_parts[1].strip()
                            
                            odds_text = ' '.join(parts[3:])
                            odds = re.findall(r'[\d.]+', odds_text)
                            
                            if len(odds) >= 3:
                                home_odds = float(odds[0]) if odds[0] else 0
                                draw_odds = float(odds[1]) if odds[1] else 0
                                away_odds = float(odds[2]) if odds[2] else 0
                                
                                if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                                    match_data = get_or_create_match(home_team, away_team, home_odds, draw_odds, away_odds)
                                    team_obj = get_or_create_team(team)
                                    team_obj["appearances"].append(current_page_type)
                                    
                                    # Update team data based on page type
                                    if current_page_type == 'wins':
                                        team_obj["w_team"] = max(team_obj.get("w_team", 0), streak_value)
                                    elif current_page_type == 'draws':
                                        team_obj["d_team"] = max(team_obj.get("d_team", 0), streak_value)
                                    elif current_page_type == 'losses':
                                        team_obj["l_team"] = max(team_obj.get("l_team", 0), streak_value)
                                    elif current_page_type == 'no_wins':
                                        team_obj["nw_team"] = max(team_obj.get("nw_team", 0), streak_value)
                                    elif current_page_type == 'no_draws':
                                        team_obj["nd_team"] = max(team_obj.get("nd_team", 0), streak_value)
                                    
                                    continue
                except (ValueError, IndexError, AttributeError):
                    pass
        
        # Parse Best/Worst teams pages
        if current_page_type in ['best_teams', 'worst_teams', 'best_offensive', 'best_defensive', 'worst_offensive', 'worst_defensive']:
            parts = re.split(r'\t+', line)
            parts = [p.strip() for p in parts if p.strip()]
            
            if len(parts) >= 4:
                try:
                    if re.search(r'[A-Za-z]', parts[0]) and not re.search(r'^[A-Za-z\s]+$', parts[0]) or len(parts) > 3:
                        team = parts[0]
                        
                        next_match = None
                        for part in parts:
                            if '-' in part or 'vs' in part:
                                next_match = part
                                break
                        
                        if next_match:
                            match_parts = re.split(r'\s*[-–]\s*', next_match)
                            if len(match_parts) == 2:
                                home_team = match_parts[0].strip()
                                away_team = match_parts[1].strip()
                                
                                odds = re.findall(r'[\d.]+', line)
                                home_odds = float(odds[-3]) if len(odds) >= 3 else 0
                                draw_odds = float(odds[-2]) if len(odds) >= 3 else 0
                                away_odds = float(odds[-1]) if len(odds) >= 3 else 0
                                
                                match_data = get_or_create_match(home_team, away_team, home_odds, draw_odds, away_odds)
                                team_obj = get_or_create_team(team)
                                team_obj["appearances"].append(current_page_type)
                                
                                # Update team data based on page type
                                if current_page_type == 'best_teams':
                                    team_obj["best_team"] = 1
                                elif current_page_type == 'worst_teams':
                                    team_obj["worst_team"] = 1
                                elif current_page_type == 'best_offensive':
                                    team_obj["best_off"] = 1
                                elif current_page_type == 'best_defensive':
                                    team_obj["best_def"] = 1
                                elif current_page_type == 'worst_offensive':
                                    team_obj["worst_off"] = 1
                                elif current_page_type == 'worst_defensive':
                                    team_obj["worst_def"] = 1
                                
                                continue
                except (ValueError, IndexError):
                    pass
    
    # Build matches with team data
    for match_key, match_data in match_cache.items():
        home_team_name = match_data["home_team"]
        away_team_name = match_data["away_team"]
        
        # Get team data
        home_data = team_cache.get(home_team_name, {})
        away_data = team_cache.get(away_team_name, {})
        
        match_data["home_team_data"] = home_data
        match_data["away_team_data"] = away_data
        matches.append(match_data)
    
    return matches


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_features(features: dict):
    """Display engineered features for a match"""
    st.markdown("### 📊 Feature Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Draw Odds</div>
            <div class="feature-value">{features.get('draw_odds', 0):.2f}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">Draw Implied Probability</div>
            <div class="feature-value">{1 / features.get('draw_odds', 0) if features.get('draw_odds', 0) > 0 else 0:.1%}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">DC12 Odds</div>
            <div class="feature-value">{features.get('dc_implied_no_draw', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Home Offensive Ratio</div>
            <div class="feature-value">{features.get('home_off_ratio', 0):.2f}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">Away Defensive Ratio</div>
            <div class="feature-value">{features.get('away_def_ratio', 0):.2f}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">Profile Difference</div>
            <div class="feature-value">{features.get('profile_difference', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Both Weak</div>
            <div class="feature-value">{"✅" if features.get('both_weak', 0) == 1 else "❌"}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">Best Team Home</div>
            <div class="feature-value">{"✅" if features.get('best_team_home', 0) == 1 else "❌"}</div>
        </div>
        <div class="feature-box">
            <div class="feature-label">Worst Def Away</div>
            <div class="feature-value">{"✅" if features.get('worst_def_away', 0) == 1 else "❌"}</div>
        </div>
        """, unsafe_allow_html=True)


def display_prediction(result: dict, features: dict = None):
    """Display the prediction result"""
    prediction = result.get('prediction', 'SKIP')
    confidence = result.get('confidence', 'LOW')
    
    if prediction == 'NO_DRAW':
        if confidence == 'HIGH':
            card_class = "no-draw-card"
            pred_class = "prediction-no-draw"
            pred_emoji = "⚔️"
            pred_text = "NO DRAW EXPECTED"
            badge = f'<span class="no-draw-badge">✅ BET (Double Chance 12)</span>'
        else:
            card_class = "consider-card"
            pred_class = "prediction-consider"
            pred_emoji = "⚠️"
            pred_text = "CONSIDER NO DRAW"
            badge = f'<span class="consider-badge">⚠️ CONSIDER</span>'
    elif prediction == 'DRAW':
        card_class = "draw-card"
        pred_class = "prediction-draw"
        pred_emoji = "🤝"
        pred_text = "DRAW LIKELY"
        badge = f'<span class="draw-badge">🤝 AVOID NO-DRAW</span>'
    elif prediction == 'OVER_2.5':
        card_class = "over-card"
        pred_class = "prediction-over"
        pred_emoji = "⚽"
        pred_text = "OVER 2.5 GOALS"
        badge = f'<span class="over-badge">⚽ BET OVER 2.5</span>'
    elif prediction == 'UNDER_2.5':
        card_class = "under-card"
        pred_class = "prediction-under"
        pred_emoji = "⚽"
        pred_text = "UNDER 2.5 GOALS"
        badge = f'<span class="under-badge">⚽ BET UNDER 2.5</span>'
    else:
        card_class = "skip-card"
        pred_class = "prediction-skip"
        pred_emoji = "❌"
        pred_text = "SKIP"
        badge = f'<span class="skip-badge">❌ SKIP</span>'
    
    st.markdown(f"""
    <div class="output-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">ADVANCED NO-DRAW PREDICTOR</div>
                <div class="prediction-display {pred_class}">
                    {pred_emoji} {pred_text}
                </div>
                <div>
                    {badge}
                    <span class="final-badge" style="margin-left:0.5rem;">Confidence: {confidence}</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            {result.get('action', '')}
            <br><span style="color:#94a3b8;">{result.get('reason', '')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if features:
        display_features(features)


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, result: dict, features: dict):
    """Save matches that have a betting recommendation"""
    try:
        prediction = result.get('prediction', 'SKIP')
        
        # Only save if there's a betting recommendation
        if prediction in ['SKIP']:
            return "SKIPPED"
            
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", datetime.now().strftime("%Y-%m-%d"))
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else datetime.now().strftime("%Y-%m-%d")
        
        if check_match_exists(home_team, away_team, match_date):
            return "ALREADY_EXISTS"
        
        # Get team data
        home_data = match.get("home_team_data", {})
        away_data = match.get("away_team_data", {})
        
        # Map prediction to valid values for the constraint
        pred_map = {
            'NO_DRAW': 'NO_DRAW',
            'DRAW': 'DRAW_POSSIBLE',  # Map to valid constraint value
            'OVER_2.5': 'NO_DRAW',  # Map to valid constraint value
            'UNDER_2.5': 'NO_DRAW',  # Map to valid constraint value
        }
        db_prediction = pred_map.get(prediction, 'SKIP')
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            
            # Home team appearances
            "w_home": home_data.get("w_team", 0),
            "d_home": home_data.get("d_team", 0),
            "l_home": home_data.get("l_team", 0),
            "nw_home": home_data.get("nw_team", 0),
            "nd_home": home_data.get("nd_team", 0),
            "best_team_home": home_data.get("best_team", 0),
            "worst_team_home": home_data.get("worst_team", 0),
            "best_off_home": home_data.get("best_off", 0),
            "best_def_home": home_data.get("best_def", 0),
            "worst_off_home": home_data.get("worst_off", 0),
            "worst_def_home": home_data.get("worst_def", 0),
            
            # Away team appearances
            "w_away": away_data.get("w_team", 0),
            "d_away": away_data.get("d_team", 0),
            "l_away": away_data.get("l_team", 0),
            "nw_away": away_data.get("nw_team", 0),
            "nd_away": away_data.get("nd_team", 0),
            "best_team_away": away_data.get("best_team", 0),
            "worst_team_away": away_data.get("worst_team", 0),
            "best_off_away": away_data.get("best_off", 0),
            "best_def_away": away_data.get("best_def", 0),
            "worst_off_away": away_data.get("worst_off", 0),
            "worst_def_away": away_data.get("worst_def", 0),
            
            # Feature scores
            "draw_odds_implied": features.get("draw_odds_implied", 0),
            "home_off_ratio": features.get("home_off_ratio", 0),
            "away_def_ratio": features.get("away_def_ratio", 0),
            "profile_difference": features.get("profile_difference", 0),
            "both_weak": features.get("both_weak", 0),
            "total_off_ratio": features.get("total_off_ratio", 0),
            
            "dc12_odds": 1 / ((1 / match.get("home_odds", 0)) + (1 / match.get("away_odds", 0))) if match.get("home_odds", 0) > 0 and match.get("away_odds", 0) > 0 else 0,
            "predicted": db_prediction,
            "confidence": result.get("confidence", "LOW"),
            "multi_score": features.get("profile_difference", 0) * 10,  # Legacy field
            
            # Store actual bet type separately
            "bet_type": prediction,
        }
        
        response = supabase.table(TABLE_NAME).insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def get_pending():
    try:
        response = supabase.table(TABLE_NAME).select("*").is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except:
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        actual_result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        response = supabase.table(TABLE_NAME).select("predicted", "bet_type").eq("id", analysis_id).execute()
        if response.data:
            predicted = response.data[0].get("predicted")
            bet_type = response.data[0].get("bet_type", predicted)
            # Define correct predictions based on bet type
            if bet_type in ["NO_DRAW"]:
                is_correct = actual_result != "X"
            elif bet_type == "DRAW":
                is_correct = actual_result == "X"
            elif bet_type in ["OVER_2.5"]:
                is_correct = (home_goals + away_goals) > 2.5
            elif bet_type in ["UNDER_2.5"]:
                is_correct = (home_goals + away_goals) < 2.5
            else:
                is_correct = False
        else:
            is_correct = False
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_result": actual_result,
            "is_correct": is_correct
        }).eq("id", analysis_id).execute()
        return True
    except:
        return False


def get_results():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except:
        return []


def display_records_table(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct'))
    incorrect = total - correct
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
    st.markdown(f"**Overall: {correct} wins | {incorrect} losses**")
    rows = []
    for r in results:
        pred = r.get('bet_type', r.get('predicted', '?'))
        actual = r.get('actual_result', '?')
        is_correct = r.get('is_correct', False)
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW" if pred == "DRAW" else "⚽ OVER 2.5" if pred == "OVER_2.5" else "⚽ UNDER 2.5" if pred == "UNDER_2.5" else "❌ SKIP"
        actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
        rows.append({
            "Date": r.get("match_date", ""),
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": pred_display,
            "Actual": actual_display,
            "Result": result_badge,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("⚔️ Advanced No-Draw Predictor")
    st.caption("Multi-feature prediction using draw odds and team profile ratios")

    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ scikit-learn not available. Using simplified decision rules.")

    with st.expander("📖 HOW IT WORKS", expanded=False):
        st.markdown("""
        ### The New Approach
        
        This system uses **8 engineered features** derived from the raw data.
        
        ### Key Features
        
        | Feature | Description | Importance |
        |---------|-------------|------------|
        | Draw Odds Implied | 1 / draw_odds | Highest predictive power |
        | Home Offensive Ratio | home positive appearances / total | Strength of home attack |
        | Away Defensive Ratio | away negative appearances / total | Weakness of away defense |
        | Profile Difference | home_off - away_def | Imbalance indicator |
        | Both Weak | both teams have weak profiles | Draw indicator |
        | Best Team Home | binary flag | Strong home team |
        | Worst Def Away | binary flag | Weak away defense |
        | DC Implied No-Draw | 1 / double chance odds | Market expectation |
        
        ### Decision Rules
        
        1. **Primary No-Draw**: draw_odds > 4.50 AND home_off > 0.30 AND away_def < 0.25
        2. **Secondary No-Draw**: draw_odds > 3.80 AND profile_diff > 0.15 AND NOT both_weak
        3. **Draw Likely**: draw_odds ≤ 4.00 AND both_weak AND home_off < 0.20 AND away_def > 0.30
        4. **Over 2.5 Goals**: draw_odds > 5.0 AND total_off_ratio > 0.80
        5. **Under 2.5 Goals**: draw_odds < 3.5 AND total_off_ratio < 0.50
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("Predicts no-draw using advanced multi-feature analysis")

        text_data = st.text_area(
            "Paste Betexplorer data here",
            height=300,
            key="text_paste",
            placeholder="Paste all Betexplorer page data here..."
        )

        if st.button("⚔️ PREDICT", type="primary"):
            if not text_data or len(text_data.strip()) < 10:
                st.error("❌ Please paste valid data.")
            else:
                try:
                    with st.spinner("Analyzing data with advanced features..."):
                        matches = parse_betexplorer_data(text_data)
                    if matches:
                        st.success(f"✅ Found {len(matches)} unique matches")
                        analyzed_results = []
                        stored_count = already_stored_count = 0
                        predictions_count = {
                            'NO_DRAW': 0,
                            'DRAW': 0,
                            'OVER_2.5': 0,
                            'UNDER_2.5': 0,
                            'SKIP': 0
                        }
                        
                        for match in matches:
                            # Engineer features
                            features = engineer_features(match)
                            
                            # Apply decision rules
                            result = apply_decision_rules(features)
                            
                            exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                            
                            if exists:
                                already_stored_count += 1
                                analyzed_results.append((match, result, features, True))
                            else:
                                if result["prediction"] != "SKIP":
                                    saved_id = save_to_db(match, result, features)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, result, features, True))
                                    elif saved_id:
                                        stored_count += 1
                                        predictions_count[result["prediction"]] += 1
                                        analyzed_results.append((match, result, features, False))
                                    else:
                                        analyzed_results.append((match, result, features, False))
                                else:
                                    analyzed_results.append((match, result, features, False))
                        
                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed")
                        
                        # Show prediction counts
                        st.markdown("### 📊 Prediction Summary")
                        cols = st.columns(5)
                        predictions_labels = {
                            'NO_DRAW': '⚔️ No Draw',
                            'DRAW': '🤝 Draw',
                            'OVER_2.5': '⚽ Over 2.5',
                            'UNDER_2.5': '⚽ Under 2.5',
                            'SKIP': '❌ Skip'
                        }
                        for idx, (key, label) in enumerate(predictions_labels.items()):
                            with cols[idx]:
                                st.metric(label, predictions_count[key])
                        
                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### ⚔️ PREDICTION RESULTS")
                            
                            # Filter out skips for detailed view
                            active_predictions = [(m, r, f, s) for m, r, f, s in analyzed_results if r["prediction"] != "SKIP"]
                            skips = [(m, r, f, s) for m, r, f, s in analyzed_results if r["prediction"] == "SKIP"]

                            if active_predictions:
                                for idx, (match, result, features, already_stored) in enumerate(active_predictions, 1):
                                    st.markdown(f"##### Match #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    display_prediction(result, features)
                                    if idx < len(active_predictions):
                                        st.markdown("---")

                            if skips:
                                st.markdown("#### ❌ SKIPPED")
                                st.caption(f"Total skipped: {len(skips)} matches")
                                for idx, (match, result, features, already_stored) in enumerate(skips[:5], 1):
                                    with st.expander(f"SKIP: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}"):
                                        display_prediction(result, features)
                                if len(skips) > 5:
                                    st.caption(f"... and {len(skips) - 5} more skipped matches")

                            st.markdown("---")
                            st.markdown("### 📊 Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Matches", len(matches))
                            with col2:
                                active = len(active_predictions)
                                st.metric("📈 Active Bets", active)
                            with col3:
                                st.metric("💾 New Stored", stored_count)
                            with col4:
                                st.metric("📌 Already Stored", already_stored_count)
                    else:
                        st.error("No matches found in the data.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Pending Matches")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('bet_type', a.get('predicted', '?'))
                confidence = a.get('confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW" if pred == "DRAW" else "⚽ OVER 2.5" if pred == "OVER_2.5" else "⚽ UNDER 2.5" if pred == "UNDER_2.5" else "❌ SKIP"
                badge = f"{pred_display} ({confidence})"
                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred_display}")
                    c1, c2 = st.columns(2)
                    with c1: hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending matches.")

    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        display_records_table(results)

    with tab4:
        st.subheader("📊 Live Dashboard")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
        else:
            total = len(results)
            correct = sum(1 for r in results if r.get('is_correct'))
            incorrect = total - correct
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
            with col2:
                win_rate = round(correct / total * 100) if total > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
            st.markdown(f"**Overall: {correct} wins | {incorrect} losses**")
            rows = []
            for r in results:
                pred = r.get('bet_type', r.get('predicted', '?'))
                actual = r.get('actual_result', '?')
                is_correct = r.get('is_correct', False)
                result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW" if pred == "DRAW" else "⚽ OVER 2.5" if pred == "OVER_2.5" else "⚽ UNDER 2.5" if pred == "UNDER_2.5" else "❌ SKIP"
                actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
                    "Prediction": pred_display,
                    "Actual": actual_display,
                    "Result": result_badge,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
