import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import traceback
import numpy as np

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
    .high-card { border-left: 5px solid #10b981; background: linear-gradient(135deg, #0a2a1a 0%, #0a1a0a 100%); }
    .consider-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .skip-card { border-left: 5px solid #64748b; background: linear-gradient(135deg, #1a1a2a 0%, #0a0a1a 100%); }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-high { color: #10b981; }
    .prediction-consider { color: #fbbf24; }
    .prediction-skip { color: #64748b; }
    .badge { padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .badge-high { background: #10b981; color: #000; }
    .badge-consider { background: #fbbf24; color: #000; }
    .badge-skip { background: #64748b; color: #fff; }
    .feature-box { background: #0f172a; border-radius: 6px; padding: 0.5rem; margin: 0.25rem 0; }
    .feature-label { color: #94a3b8; font-size: 0.7rem; }
    .feature-value { font-weight: 700; font-size: 1rem; }
    .profile-badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 600; }
    .profile-positive { background: #10b981; color: #000; }
    .profile-negative { background: #ef4444; color: #fff; }
    .profile-mixed { background: #fbbf24; color: #000; }
    .profile-established { background: #3b82f6; color: #fff; }
    .profile-weak { background: #64748b; color: #fff; }
    .tier-header { padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .tier-high { background: rgba(16, 185, 129, 0.2); border-left: 4px solid #10b981; }
    .tier-consider { background: rgba(251, 191, 36, 0.2); border-left: 4px solid #fbbf24; }
    .tier-skip { background: rgba(100, 116, 139, 0.2); border-left: 4px solid #64748b; }
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
    
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "total_appearances": total_appearances,
        "offensive_ratio": offensive_ratio,
        "defensive_ratio": defensive_ratio,
    }


def get_profile_string(profile_data: dict) -> str:
    """Convert profile data to a string label"""
    total = profile_data["total_appearances"]
    pos = profile_data["positive_count"]
    neg = profile_data["negative_count"]
    
    if total == 0:
        return "WEAK_PROFILE"
    elif pos >= 2 and neg == 0:
        return "POSITIVE"
    elif neg >= 2 and pos == 0:
        return "NEGATIVE"
    elif pos >= 2 and neg >= 2:
        return "MIXED"
    elif total >= 3:
        return "ESTABLISHED"
    else:
        return "WEAK_PROFILE"


def get_profile_color(profile: str) -> str:
    """Get color for profile display"""
    colors = {
        "POSITIVE": "profile-positive",
        "NEGATIVE": "profile-negative",
        "MIXED": "profile-mixed",
        "ESTABLISHED": "profile-established",
        "WEAK_PROFILE": "profile-weak"
    }
    return colors.get(profile, "profile-weak")


def engineer_features(match_data: dict) -> dict:
    """Engineer features for the predictive model"""
    
    home_profile_data = calculate_team_profile(match_data.get("home_team_data", {}))
    away_profile_data = calculate_team_profile(match_data.get("away_team_data", {}))
    
    # Get profile strings
    home_profile = get_profile_string(home_profile_data)
    away_profile = get_profile_string(away_profile_data)
    
    draw_odds = match_data.get("draw_odds", 0)
    home_odds = match_data.get("home_odds", 0)
    away_odds = match_data.get("away_odds", 0)
    
    # Calculate implied probabilities
    draw_implied = 1 / draw_odds if draw_odds > 0 else 0
    dc12_odds = 1 / ((1 / home_odds) + (1 / away_odds)) if home_odds > 0 and away_odds > 0 else 0
    dc_implied_no_draw = 1 / dc12_odds if dc12_odds > 0 else 0
    
    # Profile difference
    profile_difference = home_profile_data["offensive_ratio"] - away_profile_data["defensive_ratio"]
    
    # Both weak indicator - teams with 0-1 appearances are weak
    both_weak = 1 if (home_profile_data["total_appearances"] <= 1) and \
                       (away_profile_data["total_appearances"] <= 1) else 0
    
    # Calculate total offensive ratio
    total_off_ratio = home_profile_data["offensive_ratio"] + away_profile_data["offensive_ratio"]
    
    # Calculate score (for tier classification)
    score = profile_difference * 10
    
    features = {
        "draw_odds_implied": draw_implied,
        "home_off_ratio": home_profile_data["offensive_ratio"],
        "away_def_ratio": away_profile_data["defensive_ratio"],
        "profile_difference": profile_difference,
        "both_weak": both_weak,
        "best_team_home": match_data.get("home_team_data", {}).get("best_team", 0),
        "worst_def_away": match_data.get("away_team_data", {}).get("worst_def", 0),
        "dc_implied_no_draw": dc_implied_no_draw,
        "total_off_ratio": total_off_ratio,
        "draw_odds": draw_odds,
        "home_positive": home_profile_data["positive_count"],
        "home_negative": home_profile_data["negative_count"],
        "home_total": home_profile_data["total_appearances"],
        "away_positive": away_profile_data["positive_count"],
        "away_negative": away_profile_data["negative_count"],
        "away_total": away_profile_data["total_appearances"],
        "home_profile": home_profile,
        "away_profile": away_profile,
        "home_profile_data": home_profile_data,
        "away_profile_data": away_profile_data,
        "score": score,
    }
    
    return features


# ============================================================================
# THREE-TIER CLASSIFICATION SYSTEM
# ============================================================================
def classify_match(features: dict) -> dict:
    """
    Classify match into three tiers:
    - HIGH: Strong no-draw signal (score ≥ 50)
    - CONSIDER: Moderate no-draw signal (score 30-49)
    - DRAW/SKIP: Draw likely or insufficient evidence (score < 30)
    """
    
    score = features.get('score', 0)
    draw_odds = features.get('draw_odds', 0)
    home_off_ratio = features.get('home_off_ratio', 0)
    away_def_ratio = features.get('away_def_ratio', 0)
    profile_difference = features.get('profile_difference', 0)
    both_weak = features.get('both_weak', 0)
    total_off_ratio = features.get('total_off_ratio', 0)
    best_team_home = features.get('best_team_home', 0)
    worst_def_away = features.get('worst_def_away', 0)
    home_profile = features.get('home_profile', 'WEAK_PROFILE')
    away_profile = features.get('away_profile', 'WEAK_PROFILE')
    home_total = features.get('home_total', 0)
    away_total = features.get('away_total', 0)
    
    # Check if both teams have weak profiles
    if both_weak == 1 and draw_odds <= 4.0:
        return {
            'tier': 'SKIP',
            'prediction': 'DRAW',
            'confidence': 'LOW',
            'action': '❌ SKIP - Draw likely',
            'reason': f'Both teams weak (Home: {home_total} apps, Away: {away_total} apps) + low draw odds ({draw_odds:.2f})',
            'stake': '0 units',
            'score': score
        }
    
    # Check if both teams have very low offensive output
    if total_off_ratio < 0.50 and draw_odds < 3.5:
        return {
            'tier': 'SKIP',
            'prediction': 'DRAW',
            'confidence': 'LOW',
            'action': '❌ SKIP - Draw likely',
            'reason': f'Low offensive output ({total_off_ratio:.2f}) + low draw odds ({draw_odds:.2f})',
            'stake': '0 units',
            'score': score
        }
    
    # HIGH TIER: Strong no-draw signal
    if score >= 50:
        return {
            'tier': 'HIGH',
            'prediction': 'NO_DRAW',
            'confidence': 'HIGH',
            'action': '✅ BET - Strong no-draw signal',
            'reason': f'Score: {score:.0f} (≥50) - Strong profile difference',
            'stake': '2-3 units',
            'score': score,
            'winrate_expected': '85-90%'
        }
    
    # HIGH TIER: Secondary check for strong profile
    if draw_odds > 4.5 and home_off_ratio > 0.40 and away_def_ratio < 0.25 and home_profile in ['POSITIVE', 'ESTABLISHED']:
        return {
            'tier': 'HIGH',
            'prediction': 'NO_DRAW',
            'confidence': 'HIGH',
            'action': '✅ BET - Strong no-draw signal',
            'reason': f'High draw odds ({draw_odds:.2f}) + Strong home profile ({home_profile}) + Weak away defense',
            'stake': '2-3 units',
            'score': score,
            'winrate_expected': '85-90%'
        }
    
    # CONSIDER TIER: Moderate no-draw signal
    if score >= 30:
        return {
            'tier': 'CONSIDER',
            'prediction': 'CONSIDER',
            'confidence': 'MEDIUM',
            'action': '⚠️ BET - Moderate no-draw signal',
            'reason': f'Score: {score:.0f} (30-49) - Moderate profile difference',
            'stake': '1 unit',
            'score': score,
            'winrate_expected': '70-75%'
        }
    
    # CONSIDER TIER: Secondary check
    if draw_odds > 3.8 and profile_difference > 0.15 and both_weak == 0:
        return {
            'tier': 'CONSIDER',
            'prediction': 'CONSIDER',
            'confidence': 'MEDIUM',
            'action': '⚠️ BET - Moderate no-draw signal',
            'reason': f'Good draw odds ({draw_odds:.2f}) + Profile difference ({profile_difference:.2f})',
            'stake': '1 unit',
            'score': score,
            'winrate_expected': '70-75%'
        }
    
    # Check for weak no-draw signal
    if draw_odds > 4.0 and (best_team_home == 1 or worst_def_away == 1):
        return {
            'tier': 'CONSIDER',
            'prediction': 'CONSIDER',
            'confidence': 'LOW',
            'action': '⚠️ CONSIDER - Weak no-draw signal',
            'reason': 'Moderate signal - consider as value bet',
            'stake': '0.5 units',
            'score': score,
            'winrate_expected': '65-70%'
        }
    
    # Default: Skip
    return {
        'tier': 'SKIP',
        'prediction': 'SKIP',
        'confidence': 'LOW',
        'action': '❌ SKIP - Insufficient evidence',
        'reason': f'No clear signal (Home: {home_profile}, {home_total} apps | Away: {away_profile}, {away_total} apps)',
        'stake': '0 units',
        'score': score
    }


# ============================================================================
# COMPLETE PARSER
# ============================================================================
def parse_betexplorer_data(text: str) -> list:
    """Parse Betexplorer data - extracts matches from ALL pages."""
    matches = []
    lines = text.split('\n')
    
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
    
    home_profile = features.get('home_profile', 'WEAK_PROFILE')
    away_profile = features.get('away_profile', 'WEAK_PROFILE')
    home_color = get_profile_color(home_profile)
    away_color = get_profile_color(away_profile)
    
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
            <div class="feature-label">Score</div>
            <div class="feature-value">{features.get('score', 0):.0f}</div>
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
        <div class="feature-box">
            <div class="feature-label">Total Offensive Ratio</div>
            <div class="feature-value">{features.get('total_off_ratio', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Show profiles
    st.markdown("### 🏷️ Team Profiles")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:#0f172a; border-radius:6px; padding:0.5rem;">
            <span style="font-weight:700;">🏠 Home</span>
            <span class="profile-badge {home_color}">{home_profile}</span>
            <span style="font-size:0.7rem; color:#94a3b8; margin-left:0.5rem;">{features.get('home_total', 0)} appearances</span>
            <div style="font-size:0.7rem; color:#94a3b8;">
                ✅ {features.get('home_positive', 0)} positive | ❌ {features.get('home_negative', 0)} negative
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:#0f172a; border-radius:6px; padding:0.5rem;">
            <span style="font-weight:700;">✈️ Away</span>
            <span class="profile-badge {away_color}">{away_profile}</span>
            <span style="font-size:0.7rem; color:#94a3b8; margin-left:0.5rem;">{features.get('away_total', 0)} appearances</span>
            <div style="font-size:0.7rem; color:#94a3b8;">
                ✅ {features.get('away_positive', 0)} positive | ❌ {features.get('away_negative', 0)} negative
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_prediction(result: dict, features: dict = None):
    """Display the prediction result"""
    tier = result.get('tier', 'SKIP')
    confidence = result.get('confidence', 'LOW')
    stake = result.get('stake', '0 units')
    
    badge_class = {
        'HIGH': 'badge-high',
        'CONSIDER': 'badge-consider',
        'SKIP': 'badge-skip'
    }
    
    pred_display = {
        'HIGH': ('⭐⭐⭐ HIGH CONFIDENCE', 'prediction-high', 'high-card'),
        'CONSIDER': ('⭐⭐ CONSIDER', 'prediction-consider', 'consider-card'),
        'SKIP': ('❌ SKIP - Draw Likely', 'prediction-skip', 'skip-card')
    }
    
    text, pred_class, card_class = pred_display.get(tier, ('❌ SKIP', 'prediction-skip', 'skip-card'))
    badge = f'<span class="badge {badge_class.get(tier, "badge-skip")}">{result.get("action", "")}</span>'
    
    winrate_info = ""
    if tier in ['HIGH', 'CONSIDER']:
        winrate_info = f'<span style="margin-left:0.5rem; padding:0.3rem 0.75rem; border-radius:8px; font-size:0.8rem; font-weight:700; background:#1e293b; color:#10b981;">Winrate: {result.get("winrate_expected", "N/A")}</span>'
    
    st.markdown(f"""
    <div class="output-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">ADVANCED NO-DRAW PREDICTOR</div>
                <div class="prediction-display {pred_class}">
                    {text}
                </div>
                <div>
                    {badge}
                    <span style="margin-left:0.5rem; padding:0.3rem 0.75rem; border-radius:8px; font-size:0.8rem; font-weight:700; background:#1e293b; color:#94a3b8;">Confidence: {confidence}</span>
                    <span style="margin-left:0.5rem; padding:0.3rem 0.75rem; border-radius:8px; font-size:0.8rem; font-weight:700; background:#1e293b; color:#fbbf24;">Stake: {stake}</span>
                    {winrate_info}
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            {result.get('reason', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if features:
        display_features(features)


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, result: dict, features: dict):
    """Save only HIGH and CONSIDER predictions (actionable bets)"""
    try:
        tier = result.get('tier', 'SKIP')
        
        # Only save HIGH and CONSIDER - skip DRAW/SKIP
        if tier in ['SKIP']:
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
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "league": match.get("league", "Unknown"),
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
            
            # Profile counts
            "home_positive_count": features.get("home_positive", 0),
            "home_negative_count": features.get("home_negative", 0),
            "home_total_appearances": features.get("home_total", 0),
            "away_positive_count": features.get("away_positive", 0),
            "away_negative_count": features.get("away_negative", 0),
            "away_total_appearances": features.get("away_total", 0),
            "home_profile": features.get("home_profile", "WEAK_PROFILE"),
            "away_profile": features.get("away_profile", "WEAK_PROFILE"),
            
            "dc12_odds": 1 / ((1 / match.get("home_odds", 0)) + (1 / match.get("away_odds", 0))) if match.get("home_odds", 0) > 0 and match.get("away_odds", 0) > 0 else 0,
            "predicted": result.get("prediction", "NO_DRAW"),
            "confidence": result.get("confidence", "LOW"),
            "multi_score": features.get("score", 0),
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
        response = supabase.table(TABLE_NAME).select("predicted").eq("id", analysis_id).execute()
        if response.data:
            predicted = response.data[0].get("predicted")
            if predicted in ["NO_DRAW", "CONSIDER"]:
                is_correct = actual_result != "X"
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
    except Exception as e:
        st.error(f"Failed to submit result: {e}")
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
        pred = r.get('predicted', '?')
        actual = r.get('actual_result', '?')
        is_correct = r.get('is_correct', False)
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ SKIP"
        actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
        rows.append({
            "Date": r.get("match_date", ""),
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": pred_display,
            "Actual": actual_display,
            "Score": r.get('multi_score', 0),
            "Result": result_badge,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("⚔️ Advanced No-Draw Predictor")
    st.caption("Three-tier prediction system: HIGH → CONSIDER → SKIP")

    with st.expander("📖 HOW IT WORKS", expanded=False):
        st.markdown("""
        ### Three-Tier Prediction System
        
        | Tier | Rating | Winrate | Stake | Action |
        |------|--------|---------|-------|--------|
        | **⭐⭐⭐ HIGH** | Best Bets | 85-90% | 2-3 units | ✅ BET |
        | **⭐⭐ CONSIDER** | Value Bets | 70-75% | 1 unit | ⚠️ BET |
        | **❌ SKIP** | Avoid | N/A | 0 units | ❌ NO BET |
        
        ### Why Skip DRAW Predictions?
        - Only betting on HIGH and CONSIDER maintains a high winrate
        - SKIP matches are shown to users but NOT stored in the database
        - Cleaner analytics - all stored records are actionable bets
        - Better ROI by avoiding low-confidence matches
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("Predicts matches using three-tier classification system")

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
                    with st.spinner("Analyzing data with three-tier system..."):
                        matches = parse_betexplorer_data(text_data)
                    if matches:
                        st.success(f"✅ Found {len(matches)} unique matches")
                        analyzed_results = []
                        stored_count = already_stored_count = 0
                        predictions_count = {
                            'HIGH': 0,
                            'CONSIDER': 0,
                            'SKIP': 0
                        }
                        
                        for match in matches:
                            # Engineer features
                            features = engineer_features(match)
                            
                            # Classify match into tiers
                            result = classify_match(features)
                            
                            exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                            
                            if exists:
                                already_stored_count += 1
                                analyzed_results.append((match, result, features, True))
                            else:
                                # Only save HIGH and CONSIDER (not SKIP)
                                if result["tier"] in ["HIGH", "CONSIDER"]:
                                    saved_id = save_to_db(match, result, features)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, result, features, True))
                                    elif saved_id:
                                        stored_count += 1
                                        predictions_count[result["tier"]] += 1
                                        analyzed_results.append((match, result, features, False))
                                    else:
                                        analyzed_results.append((match, result, features, False))
                                else:
                                    predictions_count['SKIP'] += 1
                                    analyzed_results.append((match, result, features, False))
                        
                        # Show prediction counts
                        st.markdown("### 📊 Prediction Summary")
                        cols = st.columns(3)
                        predictions_labels = {
                            'HIGH': '⭐⭐⭐ HIGH',
                            'CONSIDER': '⭐⭐ CONSIDER',
                            'SKIP': '❌ SKIP'
                        }
                        colors = {
                            'HIGH': '#10b981',
                            'CONSIDER': '#fbbf24',
                            'SKIP': '#64748b'
                        }
                        for idx, (key, label) in enumerate(predictions_labels.items()):
                            with cols[idx]:
                                st.markdown(f"""
                                <div style="text-align:center; padding:0.5rem; background:#0f172a; border-radius:8px; border-left:3px solid {colors[key]};">
                                    <div style="font-size:2rem; font-weight:800; color:{colors[key]};">{predictions_count[key]}</div>
                                    <div style="font-size:0.8rem; color:#94a3b8;">{label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed | {predictions_count['SKIP']} skipped")
                        
                        if analyzed_results:
                            st.markdown("---")
                            
                            # Filter by tier and display in order: HIGH → CONSIDER → SKIP
                            high_predictions = [(m, r, f, s) for m, r, f, s in analyzed_results if r["tier"] == "HIGH"]
                            consider_predictions = [(m, r, f, s) for m, r, f, s in analyzed_results if r["tier"] == "CONSIDER"]
                            skip_predictions = [(m, r, f, s) for m, r, f, s in analyzed_results if r["tier"] == "SKIP"]

                            if high_predictions:
                                st.markdown("""
                                <div class="tier-header tier-high">
                                    <span style="font-size:1.5rem; font-weight:700; color:#10b981;">⭐⭐⭐ HIGH CONFIDENCE</span>
                                    <span style="margin-left:1rem; font-size:0.9rem; color:#94a3b8;">(Winrate: 85-90% | Stake: 2-3 units)</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for idx, (match, result, features, already_stored) in enumerate(high_predictions, 1):
                                    st.markdown(f"##### Match #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    display_prediction(result, features)
                                    if idx < len(high_predictions):
                                        st.markdown("---")

                            if consider_predictions:
                                st.markdown("""
                                <div class="tier-header tier-consider">
                                    <span style="font-size:1.5rem; font-weight:700; color:#fbbf24;">⭐⭐ CONSIDER</span>
                                    <span style="margin-left:1rem; font-size:0.9rem; color:#94a3b8;">(Winrate: 70-75% | Stake: 1 unit)</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for idx, (match, result, features, already_stored) in enumerate(consider_predictions, 1):
                                    st.markdown(f"##### Match #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    display_prediction(result, features)
                                    if idx < len(consider_predictions):
                                        st.markdown("---")

                            if skip_predictions:
                                st.markdown("""
                                <div class="tier-header tier-skip">
                                    <span style="font-size:1.5rem; font-weight:700; color:#64748b;">❌ SKIP</span>
                                    <span style="margin-left:1rem; font-size:0.9rem; color:#94a3b8;">(No bet recommended)</span>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                for idx, (match, result, features, already_stored) in enumerate(skip_predictions[:5], 1):
                                    with st.expander(f"SKIP: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}"):
                                        display_prediction(result, features)
                                if len(skip_predictions) > 5:
                                    st.caption(f"... and {len(skip_predictions) - 5} more skipped matches")

                            st.markdown("---")
                            st.markdown("### 📊 Summary")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Matches", len(matches))
                            with col2:
                                st.metric("⭐⭐⭐ HIGH", len(high_predictions))
                            with col3:
                                st.metric("⭐⭐ CONSIDER", len(consider_predictions))
                            with col4:
                                st.metric("💾 Stored", stored_count)
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
                pred = a.get('predicted', '?')
                confidence = a.get('confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ SKIP"
                with st.expander(f"📅 {date_display} | {pred_display} ({confidence}) | {ht} vs {at}"):
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
                pred = r.get('predicted', '?')
                actual = r.get('actual_result', '?')
                is_correct = r.get('is_correct', False)
                result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ SKIP"
                actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
                    "Prediction": pred_display,
                    "Actual": actual_display,
                    "Score": r.get('multi_score', 0),
                    "Result": result_badge,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
