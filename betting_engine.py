import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
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
# TABLE NAME
# ============================================================================
TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Clash of Identities - No Draw Predictor", page_icon="⚔️", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .no-draw-card { border-left: 5px solid #10b981; background: linear-gradient(135deg, #0a2a1a 0%, #0a1a0a 100%); }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .consider-card { border-left: 5px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0a00 100%); }
    .draw-possible-card { border-left: 5px solid #3b82f6; background: linear-gradient(135deg, #0a1a2a 0%, #0a0a1a 100%); }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-no-draw { color: #10b981; }
    .prediction-skip { color: #f59e0b; }
    .prediction-consider { color: #fbbf24; }
    .prediction-draw-possible { color: #3b82f6; }
    .final-badge { background: #10b981; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 2px solid #10b981; }
    .no-draw-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .skip-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .consider-badge { background: #fbbf24; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .draw-possible-badge { background: #3b82f6; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .appearance-badge { background: #1e293b; border-radius: 4px; padding: 0.1rem 0.4rem; margin: 0.05rem; font-size: 0.65rem; display: inline-block; }
    .positive-badge { border-left: 3px solid #10b981; }
    .negative-badge { border-left: 3px solid #ef4444; }
    .neutral-badge { border-left: 3px solid #fbbf24; }
    .upload-container { border: 2px dashed #10b981; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .upload-container:hover { border-color: #34d399; background: rgba(16, 185, 129, 0.05); }
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
# MULTI-APPEARANCE SCORING SYSTEM
# ============================================================================
def calculate_team_score(team_data: dict) -> dict:
    """
    Calculate a team's multi-appearance score based on appearances across sections.
    Each appearance is tracked by section type and category (positive/negative/neutral).
    """
    
    # Section tracking
    sections = {
        "table_w": team_data.get("w_team", 0),
        "table_d": team_data.get("d_team", 0),
        "table_l": team_data.get("l_team", 0),
        "table_nw": team_data.get("nw_team", 0),
        "table_nd": team_data.get("nd_team", 0),
        "best_teams": team_data.get("best_team", 0),
        "worst_teams": team_data.get("worst_team", 0),
        "best_offensive": team_data.get("best_off", 0),
        "best_defensive": team_data.get("best_def", 0),
        "worst_offensive": team_data.get("worst_off", 0),
        "worst_defensive": team_data.get("worst_def", 0),
    }
    
    # Category definitions
    positive_sections = ["best_teams", "best_offensive", "best_defensive"]
    negative_sections = ["worst_teams", "worst_offensive", "worst_defensive", "table_l"]
    neutral_sections = ["table_w", "table_d", "table_nw", "table_nd"]
    
    # Count appearances and categorize
    positive_count = 0
    negative_count = 0
    neutral_count = 0
    total_appearances = 0
    appearance_details = []
    
    for section, value in sections.items():
        if value > 0:
            total_appearances += 1
            if section in positive_sections:
                positive_count += 1
                appearance_details.append({"section": section, "type": "positive", "value": value})
            elif section in negative_sections:
                negative_count += 1
                appearance_details.append({"section": section, "type": "negative", "value": value})
            else:
                neutral_count += 1
                appearance_details.append({"section": section, "type": "neutral", "value": value})
    
    # Step 2: Calculate Multi-Appearance Score
    base_score = total_appearances
    
    # Bonus for spanning multiple section types
    bonus = 0
    if total_appearances >= 5:
        bonus = 8
    elif total_appearances >= 4:
        bonus = 5
    elif total_appearances >= 3:
        bonus = 3
    
    # Penalty for being too one-sided
    penalty = 0
    if positive_count > 0 and negative_count == 0 and total_appearances >= 3:
        penalty = 3  # All positive, no negative - less reliable
    elif negative_count > 0 and positive_count == 0 and total_appearances >= 3:
        penalty = 2  # All negative, predictable but less reliable alone
    
    # Step 3: Quality Score
    quality_score = (positive_count * 2) + (negative_count * -1) + (neutral_count * 0)
    
    # Final Score
    total_score = base_score + quality_score + bonus - penalty
    
    # Determine team profile
    if positive_count > negative_count and positive_count >= 2:
        profile = "OFFENSIVE/POSITIVE"
    elif negative_count > positive_count and negative_count >= 2:
        profile = "DEFENSIVE/NEGATIVE"
    elif positive_count >= 2 and negative_count >= 2:
        profile = "VOLATILE/MIXED"
    elif total_appearances >= 3:
        profile = "ESTABLISHED"
    else:
        profile = "WEAK PROFILE"
    
    return {
        "total_appearances": total_appearances,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "base_score": base_score,
        "bonus": bonus,
        "penalty": penalty,
        "quality_score": quality_score,
        "total_score": total_score,
        "profile": profile,
        "appearance_details": appearance_details,
        "sections": sections,
        "team": team_data.get("team_name", "Unknown")
    }


def predict_match(match_data: dict) -> dict:
    """
    Predict No Draw based on multi-appearance scoring for both teams.
    """
    
    # Calculate scores for both teams
    home_score_data = calculate_team_score(match_data["home"])
    away_score_data = calculate_team_score(match_data["away"])
    
    home_score = home_score_data["total_score"]
    away_score = away_score_data["total_score"]
    match_score = home_score + away_score
    
    # Adjustments for special cases
    adjustment = 0
    
    # Case: Both teams are positive with no negative
    if home_score_data["positive_count"] >= 2 and home_score_data["negative_count"] == 0 and \
       away_score_data["positive_count"] >= 2 and away_score_data["negative_count"] == 0:
        adjustment = -5  # Both elite - draw more likely
    
    # Case: Both teams are offensive
    if home_score_data["profile"] == "OFFENSIVE/POSITIVE" and away_score_data["profile"] == "OFFENSIVE/POSITIVE":
        adjustment = -3  # Two attacking teams can cancel out
    
    # Case: One team is negative, other is positive (volatile match)
    if (home_score_data["negative_count"] >= 2 and away_score_data["positive_count"] >= 2) or \
       (away_score_data["negative_count"] >= 2 and home_score_data["positive_count"] >= 2):
        adjustment = +3  # Volatile teams rarely draw
    
    # Case: One team has 0 appearances
    if home_score_data["total_appearances"] == 0 or away_score_data["total_appearances"] == 0:
        adjustment = -2  # Unknown team - less reliable
    
    final_score = match_score + adjustment
    
    # Determine prediction
    if final_score >= 15:
        prediction = "NO_DRAW"
        confidence = "VERY HIGH"
        action = "✅ BET - Strong no-draw signal"
        reason = f"High multi-section appearance score ({final_score:.0f}) indicates extreme team profiles"
    elif final_score >= 10:
        prediction = "NO_DRAW"
        confidence = "HIGH"
        action = "✅ BET - Good no-draw signal"
        reason = f"Good multi-section score ({final_score:.0f}) suggests no-draw likely"
    elif final_score >= 6:
        prediction = "NO_DRAW"
        confidence = "MEDIUM"
        action = "⚠️ CONSIDER - Moderate no-draw signal"
        reason = f"Moderate score ({final_score:.0f}) - consider as a value bet"
    elif final_score >= 3:
        prediction = "DRAW_POSSIBLE"
        confidence = "LOW"
        action = "❌ SKIP - Draw possible"
        reason = f"Low score ({final_score:.0f}) suggests draw is possible"
    else:
        prediction = "SKIP"
        confidence = "LOW"
        action = "❌ SKIP - Insufficient evidence"
        reason = f"Very low score ({final_score:.0f}) - no clear signal"
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "action": action,
        "reason": reason,
        "final_score": final_score,
        "match_score": match_score,
        "adjustment": adjustment,
        "home_score": home_score_data,
        "away_score": away_score_data,
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
                                    elif current_page_type == 'no_losses':
                                        pass
                                    
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
def display_team_profile(team_data: dict, team_name: str):
    """Display a team's multi-appearance profile"""
    
    if not team_data:
        st.markdown(f"""
        <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.25rem 0;">
            <span style="font-weight:700;">{team_name}</span>
            <span style="color:#64748b; margin-left:0.5rem;">No appearances</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    appearances = team_data.get("appearances", [])
    
    # Map section names to display names
    section_display = {
        "wins": "W", "draws": "D", "losses": "L",
        "no_wins": "NW", "no_draws": "ND",
        "best_teams": "⭐ Best", "worst_teams": "⚠️ Worst",
        "best_offensive": "⚽ Best Off", "best_defensive": "🛡️ Best Def",
        "worst_offensive": "❌ Worst Off", "worst_defensive": "❌ Worst Def"
    }
    
    # Categorize sections
    positive_sections = ["best_teams", "best_offensive", "best_defensive"]
    negative_sections = ["worst_teams", "worst_offensive", "worst_defensive", "losses"]
    
    positive_apps = [a for a in appearances if a in positive_sections]
    negative_apps = [a for a in appearances if a in negative_sections]
    neutral_apps = [a for a in appearances if a not in positive_sections and a not in negative_sections]
    
    # Build badges
    badges = []
    for app in appearances:
        display_name = section_display.get(app, app)
        if app in positive_sections:
            badges.append(f'<span class="appearance-badge positive-badge" style="color:#10b981;">{display_name}</span>')
        elif app in negative_sections:
            badges.append(f'<span class="appearance-badge negative-badge" style="color:#ef4444;">{display_name}</span>')
        else:
            badges.append(f'<span class="appearance-badge neutral-badge" style="color:#fbbf24;">{display_name}</span>')
    
    # Calculate score
    score_data = calculate_team_score(team_data)
    
    # Determine profile color
    profile_colors = {
        "OFFENSIVE/POSITIVE": "#10b981",
        "DEFENSIVE/NEGATIVE": "#ef4444",
        "VOLATILE/MIXED": "#fbbf24",
        "ESTABLISHED": "#3b82f6",
        "WEAK PROFILE": "#64748b"
    }
    profile_color = profile_colors.get(score_data["profile"], "#64748b")
    
    st.markdown(f"""
    <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.25rem 0; border-left: 3px solid {profile_color};">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <span style="font-weight:700;">{team_name}</span>
                <span style="font-size:0.7rem; color:#94a3b8; margin-left:0.5rem;">{len(appearances)} appearances</span>
            </div>
            <div style="font-weight:800; font-size:1.2rem; color:{profile_color};">
                {score_data["total_score"]:.0f}
                <span style="font-size:0.7rem; font-weight:400; color:#94a3b8;">({score_data["profile"]})</span>
            </div>
        </div>
        <div style="display:flex; gap:0.25rem; flex-wrap:wrap; margin-top:0.25rem;">
            {' '.join(badges)}
        </div>
        <div style="display:flex; gap:1rem; font-size:0.7rem; margin-top:0.25rem; color:#94a3b8;">
            <span>✅ Positive: {score_data["positive_count"]}</span>
            <span>❌ Negative: {score_data["negative_count"]}</span>
            <span>➖ Neutral: {score_data["neutral_count"]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_prediction(match: dict, result: dict):
    """Display the prediction result"""
    prediction = result["prediction"]
    score = result["final_score"]
    home_team = match.get("home_team", "Home")
    away_team = match.get("away_team", "Away")
    
    if prediction == "NO_DRAW":
        if result["confidence"] == "VERY HIGH":
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
    elif prediction == "DRAW_POSSIBLE":
        card_class = "draw-possible-card"
        pred_class = "prediction-draw-possible"
        pred_emoji = "🤝"
        pred_text = "DRAW POSSIBLE"
        badge = f'<span class="draw-possible-badge">🤝 DRAW POSSIBLE</span>'
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
                <div style="font-size: 0.8rem; color: #94a3b8;">MULTI-APPEARANCE NO-DRAW PREDICTOR</div>
                <div class="prediction-display {pred_class}">
                    {pred_emoji} {pred_text}
                </div>
                <div>
                    {badge}
                    <span class="final-badge" style="margin-left:0.5rem;">Score: {score:.0f}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Confidence</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #10b981;">{result.get('confidence', 'LOW')}</div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            {result.get('action', '')}
            <br><span style="color:#94a3b8;">{result.get('reason', '')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ Team Multi-Appearance Profiles")
    
    col1, col2 = st.columns(2)
    with col1:
        display_team_profile(match.get("home_team_data", {}), home_team)
        st.caption(f"Home Score: {result['home_score']['total_score']:.0f}")
    with col2:
        display_team_profile(match.get("away_team_data", {}), away_team)
        st.caption(f"Away Score: {result['away_score']['total_score']:.0f}")


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, result: dict):
    """Save matches that are NO_DRAW or CONSIDER"""
    try:
        prediction = result.get("prediction", "SKIP")
        
        if prediction not in ["NO_DRAW", "DRAW_POSSIBLE"]:
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
        home_score_data = result.get("home_score", {})
        away_score_data = result.get("away_score", {})
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "league": match.get("league", "Unknown"),
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            "dc12_odds": 1 / ((1 / match.get("home_odds", 0)) + (1 / match.get("away_odds", 0))) if match.get("home_odds", 0) > 0 and match.get("away_odds", 0) > 0 else 0,
            
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
            
            # Scores
            "multi_score": result.get("final_score", 0),
            "home_score": home_score_data.get("total_score", 0),
            "away_score": away_score_data.get("total_score", 0),
            
            "home_positive_count": home_score_data.get("positive_count", 0),
            "home_negative_count": home_score_data.get("negative_count", 0),
            "home_neutral_count": home_score_data.get("neutral_count", 0),
            "home_total_appearances": home_score_data.get("total_appearances", 0),
            "home_profile": home_score_data.get("profile", "WEAK PROFILE"),
            
            "away_positive_count": away_score_data.get("positive_count", 0),
            "away_negative_count": away_score_data.get("negative_count", 0),
            "away_neutral_count": away_score_data.get("neutral_count", 0),
            "away_total_appearances": away_score_data.get("total_appearances", 0),
            "away_profile": away_score_data.get("profile", "WEAK PROFILE"),
            
            "predicted": prediction,
            "confidence": result.get("confidence", "LOW"),
            "prediction_reason": result.get("reason", ""),
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
            is_correct = (predicted == "NO_DRAW" and actual_result != "X") or (predicted == "DRAW_POSSIBLE" and actual_result == "X")
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
        pred = r.get('predicted', '?')
        actual = r.get('actual_result', '?')
        is_correct = r.get('is_correct', False)
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW POSSIBLE" if pred == "DRAW_POSSIBLE" else "❌ SKIP"
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
    st.title("⚔️ Clash of Identities - No Draw Predictor")
    st.caption("Predicts when a match is unlikely to end in a draw based on team identity gaps")

    with st.expander("📖 HOW IT WORKS", expanded=False):
        st.markdown("""
        ### Core Insight: The Power of Multiple Appearances
        
        A team appearing in **multiple sections** (especially across positive and negative categories) provides a much more reliable signal than any single appearance.
        
        ### The Multi-Appearance Scoring System
        
        **Step 1: Count Appearances Across All Sections**
        
        Track each team's appearances across these 11 sections:
        - Table W (neutral)
        - Table D (neutral)
        - Table L (negative)
        - Table NW (neutral)
        - Table ND (neutral)
        - Best teams (positive)
        - Worst teams (negative)
        - Best offensive (positive)
        - Best defensive (positive)
        - Worst offensive (negative)
        - Worst defensive (negative)
        
        **Step 2: Calculate Multi-Appearance Score**
        
        Base Score = Total Appearances
        
        Bonus:
        - 5+ sections: +8
        - 4+ sections: +5
        - 3+ sections: +3
        
        Penalty:
        - All positive (3+ sections): -3
        - All negative (3+ sections): -2
        
        **Step 3: Quality Score**
        Quality Score = (Positive × 2) + (Negative × -1) + (Neutral × 0)
        
        **Step 4: Match-Level No-Draw Score**
        Match Score = Home Total + Away Total + Adjustment
        
        ### Decision Rules
        
        | Match Score | Prediction | Confidence |
        |-------------|------------|------------|
        | ≥ 15 | NO DRAW | VERY HIGH |
        | 10 - 14 | NO DRAW | HIGH |
        | 6 - 9 | NO DRAW | MEDIUM |
        | 3 - 5 | DRAW POSSIBLE | LOW |
        | ≤ 2 | SKIP | LOW |
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("Predicts no-draw based on multi-appearance scoring across all sections")

        text_data = st.text_area(
            "Paste Betexplorer data here",
            height=300,
            key="text_paste",
            placeholder="Paste all Betexplorer page data here..."
        )

        if st.button("⚔️ PREDICT NO DRAW", type="primary"):
            if not text_data or len(text_data.strip()) < 10:
                st.error("❌ Please paste valid data.")
            else:
                try:
                    with st.spinner("Analyzing team multi-appearance profiles..."):
                        matches = parse_betexplorer_data(text_data)
                    if matches:
                        st.success(f"✅ Found {len(matches)} unique matches")
                        analyzed_results = []
                        stored_count = already_stored_count = no_draw_count = draw_possible_count = 0
                        
                        for match in matches:
                            # Prepare match data for prediction
                            match_for_prediction = {
                                "home": match.get("home_team_data", {}),
                                "away": match.get("away_team_data", {})
                            }
                            result = predict_match(match_for_prediction)
                            exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                            
                            if exists:
                                already_stored_count += 1
                                analyzed_results.append((match, result, True))
                            else:
                                if result["prediction"] in ["NO_DRAW", "DRAW_POSSIBLE"]:
                                    saved_id = save_to_db(match, result)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, result, True))
                                    elif saved_id:
                                        stored_count += 1
                                        analyzed_results.append((match, result, False))
                                        if result["prediction"] == "NO_DRAW":
                                            no_draw_count += 1
                                        elif result["prediction"] == "DRAW_POSSIBLE":
                                            draw_possible_count += 1
                                    else:
                                        analyzed_results.append((match, result, False))
                                else:
                                    analyzed_results.append((match, result, False))
                        
                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed | ⚔️ {no_draw_count} no-draw bets | 🤝 {draw_possible_count} draw possible")
                        
                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### ⚔️ PREDICTION RESULTS")
                            
                            no_draws = [(m, r, s) for m, r, s in analyzed_results if r.get("prediction") == "NO_DRAW"]
                            draw_possibles = [(m, r, s) for m, r, s in analyzed_results if r.get("prediction") == "DRAW_POSSIBLE"]
                            skips = [(m, r, s) for m, r, s in analyzed_results if r.get("prediction") == "SKIP"]

                            if no_draws:
                                st.markdown("#### ⚔️ NO DRAW PREDICTIONS")
                                for idx, (match, result, already_stored) in enumerate(no_draws, 1):
                                    st.markdown(f"##### Match #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    display_prediction(match, result)
                                    if idx < len(no_draws):
                                        st.markdown("---")

                            if draw_possibles:
                                st.markdown("#### 🤝 DRAW POSSIBLE")
                                for idx, (match, result, already_stored) in enumerate(draw_possibles, 1):
                                    with st.expander(f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')} - Score: {result.get('final_score', 0):.0f}"):
                                        display_prediction(match, result)

                            if skips:
                                st.markdown("#### ❌ SKIPPED")
                                st.caption(f"Total skipped: {len(skips)} matches")
                                for idx, (match, result, already_stored) in enumerate(skips[:5], 1):
                                    with st.expander(f"SKIP: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}"):
                                        display_prediction(match, result)
                                if len(skips) > 5:
                                    st.caption(f"... and {len(skips) - 5} more skipped matches")

                            st.markdown("---")
                            st.markdown("### 📊 Summary")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            with col1:
                                st.metric("Total Matches", len(matches))
                            with col2:
                                st.metric("⚔️ No Draw", no_draw_count)
                            with col3:
                                st.metric("🤝 Draw Possible", draw_possible_count)
                            with col4:
                                st.metric("💾 New Stored", stored_count)
                            with col5:
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
                pred = a.get('predicted', '?')
                confidence = a.get('confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                score = a.get('multi_score', 0)
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW POSSIBLE" if pred == "DRAW_POSSIBLE" else "❌ SKIP"
                badge = f"{pred_display} ({confidence}) — Score: {score:.0f}"
                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred_display} — Score: {score:.0f}")
                    c1, c2 = st.columns(2)
                    with c1: 
                        hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2: 
                        ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
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
            
            # Performance chart
            if len(results) > 1:
                df_results = pd.DataFrame(results)
                df_results['match_date'] = pd.to_datetime(df_results['match_date'])
                df_results = df_results.sort_values('match_date')
                df_results['cumulative_correct'] = df_results['is_correct'].cumsum()
                df_results['cumulative_total'] = range(1, len(df_results) + 1)
                df_results['cumulative_win_rate'] = (df_results['cumulative_correct'] / df_results['cumulative_total']) * 100
                
                chart_data = df_results[['match_date', 'cumulative_win_rate']].set_index('match_date')
                st.line_chart(chart_data)
            
            rows = []
            for r in results:
                pred = r.get('predicted', '?')
                actual = r.get('actual_result', '?')
                is_correct = r.get('is_correct', False)
                result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "🤝 DRAW POSSIBLE" if pred == "DRAW_POSSIBLE" else "❌ SKIP"
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
