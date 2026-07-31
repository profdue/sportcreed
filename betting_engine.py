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
    .actual-score { font-size: 1.2rem; font-weight: 700; padding: 0.3rem 0.75rem; border-radius: 8px; display: inline-block; }
    .score-win { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .score-loss { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .score-pending { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
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


def check_match_exists(home_team: str, away_team: str) -> bool:
    """
    Check if a match already exists for this team pair (regardless of date)
    This prevents duplicates when the same match appears in daily updates
    """
    try:
        response = supabase.table(TABLE_NAME).select("id")\
            .eq("home_team", home_team)\
            .eq("away_team", away_team)\
            .is_("actual_result", "null")\
            .execute()
        return len(response.data) > 0
    except:
        return False


def update_match(match: dict, result: dict, features: dict) -> bool:
    """
    Update an existing match with new prediction data
    """
    try:
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        
        # Find the existing match ID
        response = supabase.table(TABLE_NAME).select("id")\
            .eq("home_team", home_team)\
            .eq("away_team", away_team)\
            .is_("actual_result", "null")\
            .execute()
        
        if not response.data:
            return False
        
        match_id = response.data[0]["id"]
        
        # Get team data
        home_data = match.get("home_team_data", {})
        away_data = match.get("away_team_data", {})
        
        record = {
            "match_date": datetime.now().strftime("%Y-%m-%d"),
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
            "updated_at": datetime.now().isoformat(),
        }
        
        response = supabase.table(TABLE_NAME).update(record).eq("id", match_id).execute()
        return True
        
    except Exception as e:
        st.error(f"Failed to update: {e}")
        return False


def save_to_db(match: dict, result: dict, features: dict):
    """Save new match or update existing one"""
    try:
        tier = result.get('tier', 'SKIP')
        
        # Only save HIGH and CONSIDER - skip DRAW/SKIP
        if tier in ['SKIP']:
            return "SKIPPED"
            
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        
        # Check if match already exists (regardless of date)
        exists = check_match_exists(home_team, away_team)
        
        if exists:
            # Update existing match
            updated = update_match(match, result, features)
            if updated:
                return "UPDATED"
            else:
                return "UPDATE_FAILED"
        
        # Get team data
        home_data = match.get("home_team_data", {})
        away_data = match.get("away_team_data", {})
        
        record = {
            "match_date": datetime.now().strftime("%Y-%m-%d"),
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


# ============================================================================
# FEATURE ENGINEERING (unchanged from previous version)
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
def classify_match(features: dict, match_type: str = "men") -> dict:
    """
    Classify match into three tiers:
    - HIGH: Strong no-draw signal (score ≥ 40 with strict filters)
    - CONSIDER: Moderate no-draw signal (score 30-49 with filters)
    - DRAW/SKIP: Draw likely or insufficient evidence
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
    
    # NEW FILTER: Skip Weak vs Weak matches
    if both_weak == 1 and draw_odds <= 4.0:
        return {
            'tier': 'SKIP',
            'prediction': 'DRAW',
            'confidence': 'LOW',
            'action': '❌ SKIP - Both teams weak',
            'reason': f'Both teams weak (Home: {home_total} apps, Away: {away_total} apps) + low draw odds ({draw_odds:.2f})',
            'stake': '0 units',
            'score': score
        }
    
    # NEW FILTER: Skip Youth/U19 matches (unless HIGH)
    if 'U19' in str(match_type) or 'Youth' in str(match_type):
        if score < 50:  # Only bet if very strong signal
            return {
                'tier': 'SKIP',
                'prediction': 'SKIP',
                'confidence': 'LOW',
                'action': '❌ SKIP - Youth match (unpredictable)',
                'reason': f'Youth match - skipping unless HIGH confidence (score: {score:.0f})',
                'stake': '0 units',
                'score': score
            }
    
    # NEW FILTER: Women's matches - stricter threshold
    if 'W' in str(match_type) or 'Women' in str(match_type):
        if score < 40:  # Higher threshold for women's matches
            return {
                'tier': 'SKIP',
                'prediction': 'SKIP',
                'confidence': 'LOW',
                'action': '❌ SKIP - Women\'s match (volatile)',
                'reason': f'Women\'s match - stricter threshold (score: {score:.0f} < 40)',
                'stake': '0 units',
                'score': score
            }
    
    # Check if both teams have very low offensive output
    if total_off_ratio < 0.50 and draw_odds < 3.5:
        return {
            'tier': 'SKIP',
            'prediction': 'DRAW',
            'confidence': 'LOW',
            'action': '❌ SKIP - Low offensive output',
            'reason': f'Low offensive output ({total_off_ratio:.2f}) + low draw odds ({draw_odds:.2f})',
            'stake': '0 units',
            'score': score
        }
    
    # HIGH TIER: Strong no-draw signal (lowered threshold to 40)
    if score >= 40:
        return {
            'tier': 'HIGH',
            'prediction': 'NO_DRAW',
            'confidence': 'HIGH',
            'action': '✅ BET - Strong no-draw signal',
            'reason': f'Score: {score:.0f} (≥40) - Strong profile difference',
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
    if score >= 30 and both_weak == 0:
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
    
    # CONSIDER TIER: Secondary check (but NOT if both_weak)
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
    if draw_odds > 4.0 and (best_team_home == 1 or worst_def_away == 1) and both_weak == 0:
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
# PARSER (unchanged from previous version - keep your existing parser)
# ============================================================================
# [Your existing parse_betexplorer_data function here - unchanged]


# ============================================================================
# DISPLAY FUNCTIONS (unchanged from previous version)
# ============================================================================
# [Your existing display functions here - unchanged]


# ============================================================================
# SUPABASE OPERATIONS - get_pending, submit_result, get_results, display_records_table
# ============================================================================
# [Your existing functions here - unchanged]


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("⚔️ Advanced No-Draw Predictor")
    st.caption("Three-tier prediction system with duplicate prevention")

    with st.expander("📖 HOW IT WORKS", expanded=False):
        st.markdown("""
        ### Three-Tier Prediction System
        
        | Tier | Rating | Winrate | Stake | Action |
        |------|--------|---------|-------|--------|
        | **⭐⭐⭐ HIGH** | Best Bets | 85-90% | 2-3 units | ✅ BET |
        | **⭐⭐ CONSIDER** | Value Bets | 70-75% | 1 unit | ⚠️ BET |
        | **❌ SKIP** | Avoid | N/A | 0 units | ❌ NO BET |
        
        ### Duplicate Prevention
        - Matches are identified by **Home Team + Away Team**
        - Existing matches are **updated** with new data (not duplicated)
        - Only **pending** matches (no actual result) are updated
        - Played matches keep their results and are not overwritten
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("Predicts matches using three-tier classification system with duplicate prevention")

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
                        stored_count = updated_count = skipped_count = 0
                        predictions_count = {
                            'HIGH': 0,
                            'CONSIDER': 0,
                            'SKIP': 0
                        }
                        
                        for match in matches:
                            # Engineer features
                            features = engineer_features(match)
                            
                            # Detect match type from league
                            league = match.get("league", "")
                            match_type = "men"
                            if "U19" in league or "Youth" in league:
                                match_type = "youth"
                            elif "W" in league or "Women" in league:
                                match_type = "women"
                            
                            # Classify match into tiers with filters
                            result = classify_match(features, match_type)
                            
                            # Check if match exists (regardless of date)
                            exists = check_match_exists(
                                match.get("home_team", ""),
                                match.get("away_team", "")
                            )
                            
                            if exists:
                                # Update existing match
                                if result["tier"] in ["HIGH", "CONSIDER"]:
                                    saved_id = save_to_db(match, result, features)
                                    if saved_id == "UPDATED":
                                        updated_count += 1
                                        predictions_count[result["tier"]] += 1
                                        analyzed_results.append((match, result, features, True, "Updated"))
                                    elif saved_id:
                                        stored_count += 1
                                        predictions_count[result["tier"]] += 1
                                        analyzed_results.append((match, result, features, False, "New"))
                                    else:
                                        analyzed_results.append((match, result, features, True, "Failed"))
                                else:
                                    predictions_count['SKIP'] += 1
                                    analyzed_results.append((match, result, features, True, "Skipped (existing)"))
                            else:
                                # New match - save
                                if result["tier"] in ["HIGH", "CONSIDER"]:
                                    saved_id = save_to_db(match, result, features)
                                    if saved_id:
                                        stored_count += 1
                                        predictions_count[result["tier"]] += 1
                                        analyzed_results.append((match, result, features, False, "New"))
                                    else:
                                        analyzed_results.append((match, result, features, False, "Failed"))
                                else:
                                    predictions_count['SKIP'] += 1
                                    analyzed_results.append((match, result, features, False, "Skipped (new)"))
                        
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
                        
                        st.info(f"💾 {stored_count} new stored | 🔄 {updated_count} updated | ❌ {predictions_count['SKIP']} skipped")
                        
                        # [Rest of the display code remains the same...]
                        
                    else:
                        st.error("No matches found in the data.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())

    # [Rest of the tabs remain the same...]


if __name__ == "__main__":
    main()
