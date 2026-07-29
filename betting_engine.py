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
    .prediction-consider { color: #fbbf24; }
    .final-badge { background: #10b981; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 2px solid #10b981; }
    .no-draw-badge { background: #10b981; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .skip-badge { background: #f59e0b; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .consider-badge { background: #fbbf24; color: #000; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .factor-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }
    .factor-name { color: #94a3b8; }
    .factor-value { font-weight: 600; }
    .upload-container { border: 2px dashed #10b981; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .upload-container:hover { border-color: #34d399; background: rgba(16, 185, 129, 0.05); }
    .identity-positive { color: #10b981; }
    .identity-negative { color: #ef4444; }
    .identity-neutral { color: #fbbf24; }
    .data-point { background: #0f172a; border-radius: 6px; padding: 0.25rem 0.5rem; margin: 0.1rem; display: inline-block; font-size: 0.7rem; }
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
# CLASH OF IDENTITIES FORMULA
# ============================================================================
def calculate_identity(match_data: dict, team: str) -> dict:
    """Calculate a team's identity score based on available data"""
    
    if team == "home":
        wins = match_data.get('w_home', 0)
        losses = match_data.get('l_home', 0)
        no_losses = match_data.get('nol_home', 0)
        best_team = match_data.get('best_team_home', 0)
        worst_team = match_data.get('worst_team_home', 0)
        best_offense = match_data.get('best_off_home', 0)
        worst_defense = match_data.get('worst_def_home', 0)
    else:  # away
        wins = match_data.get('w_away', 0)
        losses = match_data.get('l_away', 0)
        no_losses = match_data.get('nol_away', 0)
        best_team = match_data.get('best_team_away', 0)
        worst_team = match_data.get('worst_team_away', 0)
        best_offense = match_data.get('best_off_away', 0)
        worst_defense = match_data.get('worst_def_away', 0)
    
    # Count data points for eligibility
    data_points = 0
    if wins > 0: data_points += 1
    if losses > 0: data_points += 1
    if no_losses > 0: data_points += 1
    if best_team > 0: data_points += 1
    if worst_team > 0: data_points += 1
    if best_offense > 0: data_points += 1
    if worst_defense > 0: data_points += 1
    
    # Calculate Identity Scores
    attack_power = (wins * 2) + (no_losses * 1) + (best_team * 1) + (best_offense * 1)
    defensive_solidity = (losses * 1) + (no_losses * 2) + (worst_team * 1) + (worst_defense * 1)
    net_identity = attack_power - defensive_solidity
    
    return {
        "attack_power": attack_power,
        "defensive_solidity": defensive_solidity,
        "net_identity": net_identity,
        "data_points": data_points,
        "wins": wins,
        "losses": losses,
        "no_losses": no_losses,
        "best_team": best_team,
        "worst_team": worst_team,
        "best_offense": best_offense,
        "worst_defense": worst_defense,
    }


def calculate_no_draw_prediction(match_data: dict) -> dict:
    """Calculate No Draw prediction using Clash of Identities formula"""
    
    home_identity = calculate_identity(match_data, "home")
    away_identity = calculate_identity(match_data, "away")
    
    # Step 2: Eligibility Check - BOTH teams need ≥ 2 data points
    eligible = (home_identity["data_points"] >= 2) and (away_identity["data_points"] >= 2)
    
    if not eligible:
        return {
            "eligible": False,
            "decision": "SKIP",
            "action": "❌ SKIP - Insufficient data",
            "score": 0,
            "gap": 0,
            "home_identity": home_identity,
            "away_identity": away_identity,
            "reason": "One or both teams have insufficient data (< 2 data points)"
        }
    
    # Step 3: Calculate Match Gap
    gap = abs(home_identity["net_identity"] - away_identity["net_identity"])
    
    # Step 4: Calculate Final Score
    score = gap * 10
    
    # Step 5: Decision Rules
    if score >= 50:
        decision = "NO_DRAW"
        action = "✅ BET - No Draw expected (Double Chance 12)"
        confidence = "HIGH"
    elif score >= 30:
        decision = "CONSIDER"
        action = "⚠️ CONSIDER - Weak no-draw signal"
        confidence = "MEDIUM"
    else:
        decision = "SKIP"
        action = "❌ SKIP - Insufficient evidence for no draw"
        confidence = "LOW"
    
    return {
        "eligible": True,
        "decision": decision,
        "action": action,
        "confidence": confidence,
        "score": score,
        "gap": gap,
        "home_identity": home_identity,
        "away_identity": away_identity,
        "reason": None
    }


# ============================================================================
# COMPLETE PARSER - Extracts ALL data from ALL pages
# ============================================================================
def parse_betexplorer_data(text: str) -> list:
    """Parse Betexplorer data - extracts matches from ALL pages."""
    matches = []
    lines = text.split('\n')
    
    # Store data by match key
    match_cache = {}
    current_page_type = None
    current_country = None
    
    def get_or_create_match(home_team, away_team, home_odds=0, draw_odds=0, away_odds=0):
        match_key = f"{home_team}|{away_team}"
        if match_key not in match_cache:
            match_cache[match_key] = {
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "nd_home": 0, "nd_away": 0,
                "w_home": 0, "w_away": 0,
                "l_home": 0, "l_away": 0,
                "now_home": 0, "now_away": 0,
                "nol_home": 0, "nol_away": 0,
                "best_team_home": 0, "best_team_away": 0,
                "worst_team_home": 0, "worst_team_away": 0,
                "best_off_home": 0, "best_off_away": 0,
                "worst_def_home": 0, "worst_def_away": 0,
                "league": current_country or "Unknown",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "is_finished": False,
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
        elif 'Team\tNL\tNext match' in line or 'Team, NL, Next match' in line:
            current_page_type = 'no_losses'
            continue
        elif 'Team\tND\tNext match' in line or 'Team, ND, Next match' in line:
            current_page_type = 'no_draws'
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
        elif 'Worst defensive' in line:
            current_page_type = 'worst_defensive'
            continue
        elif 'Best defensive' in line:
            current_page_type = 'best_defensive'
            continue
        elif 'Worst offensive' in line:
            current_page_type = 'worst_offensive'
            continue
        
        # Detect country/league
        if re.match(r'^[A-Za-z\s]+$', line) and not re.search(r'[0-9.]', line):
            current_country = line
            continue
        
        # Parse page-specific data
        if current_page_type in ['wins', 'draws', 'losses', 'no_wins', 'no_losses', 'no_draws']:
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
                                    
                                    if team == home_team:
                                        if current_page_type == 'wins':
                                            match_data['w_home'] = max(match_data.get('w_home', 0), streak_value)
                                        elif current_page_type == 'losses':
                                            match_data['l_home'] = max(match_data.get('l_home', 0), streak_value)
                                        elif current_page_type == 'no_wins':
                                            match_data['now_home'] = max(match_data.get('now_home', 0), streak_value)
                                        elif current_page_type == 'no_losses':
                                            match_data['nol_home'] = max(match_data.get('nol_home', 0), streak_value)
                                        elif current_page_type == 'no_draws':
                                            match_data['nd_home'] = max(match_data.get('nd_home', 0), streak_value)
                                    elif team == away_team:
                                        if current_page_type == 'wins':
                                            match_data['w_away'] = max(match_data.get('w_away', 0), streak_value)
                                        elif current_page_type == 'losses':
                                            match_data['l_away'] = max(match_data.get('l_away', 0), streak_value)
                                        elif current_page_type == 'no_wins':
                                            match_data['now_away'] = max(match_data.get('now_away', 0), streak_value)
                                        elif current_page_type == 'no_losses':
                                            match_data['nol_away'] = max(match_data.get('nol_away', 0), streak_value)
                                        elif current_page_type == 'no_draws':
                                            match_data['nd_away'] = max(match_data.get('nd_away', 0), streak_value)
                                    continue
                except (ValueError, IndexError, AttributeError):
                    pass
        
        # Parse Best/Worst teams pages
        if current_page_type in ['best_teams', 'worst_teams', 'best_offensive', 'worst_defensive']:
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
                                
                                if team == home_team:
                                    if current_page_type == 'best_teams':
                                        match_data['best_team_home'] = 1
                                    elif current_page_type == 'worst_teams':
                                        match_data['worst_team_home'] = 1
                                    elif current_page_type == 'best_offensive':
                                        match_data['best_off_home'] = 1
                                    elif current_page_type == 'worst_defensive':
                                        match_data['worst_def_home'] = 1
                                elif team == away_team:
                                    if current_page_type == 'best_teams':
                                        match_data['best_team_away'] = 1
                                    elif current_page_type == 'worst_teams':
                                        match_data['worst_team_away'] = 1
                                    elif current_page_type == 'best_offensive':
                                        match_data['best_off_away'] = 1
                                    elif current_page_type == 'worst_defensive':
                                        match_data['worst_def_away'] = 1
                                continue
                except (ValueError, IndexError):
                    pass
    
    # Convert cache to list
    for match_data in match_cache.values():
        matches.append(match_data)
    
    return matches


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_identity(identity: dict, team_name: str, is_home: bool):
    """Display a team's identity score"""
    emoji = "🏠" if is_home else "✈️"
    
    net = identity["net_identity"]
    if net > 0:
        net_class = "identity-positive"
        net_label = "ATTACKING"
    elif net < 0:
        net_class = "identity-negative"
        net_label = "DEFENSIVE"
    else:
        net_class = "identity-neutral"
        net_label = "BALANCED"
    
    st.markdown(f"""
    <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.25rem 0;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-weight:700;">{emoji} {team_name}</span>
                <span style="font-size:0.7rem; color:#94a3b8; margin-left:0.5rem;">{identity['data_points']} data points</span>
            </div>
            <div class="{net_class}" style="font-weight:800; font-size:1.2rem;">
                {net:+.1f}
                <span style="font-size:0.7rem; font-weight:400; color:#94a3b8;">({net_label})</span>
            </div>
        </div>
        <div style="display:flex; gap:1rem; font-size:0.8rem; margin-top:0.25rem;">
            <span>⚔️ Attack: {identity['attack_power']}</span>
            <span>🛡️ Defense: {identity['defensive_solidity']}</span>
        </div>
        <div style="display:flex; gap:0.25rem; flex-wrap:wrap; margin-top:0.25rem;">
            {f'<span class="data-point">W:{identity["wins"]}</span>' if identity["wins"] > 0 else ''}
            {f'<span class="data-point">L:{identity["losses"]}</span>' if identity["losses"] > 0 else ''}
            {f'<span class="data-point">NL:{identity["no_losses"]}</span>' if identity["no_losses"] > 0 else ''}
            {f'<span class="data-point">⭐ Best Team</span>' if identity["best_team"] > 0 else ''}
            {f'<span class="data-point">⚠️ Worst Team</span>' if identity["worst_team"] > 0 else ''}
            {f'<span class="data-point">⚽ Best Offense</span>' if identity["best_offense"] > 0 else ''}
            {f'<span class="data-point">🛡️ Worst Defense</span>' if identity["worst_defense"] > 0 else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_prediction(match: dict, result: dict):
    """Display the prediction result"""
    decision = result["decision"]
    score = result["score"]
    gap = result["gap"]
    home_team = match.get("home_team", "Home")
    away_team = match.get("away_team", "Away")
    
    if decision == "NO_DRAW":
        card_class = "no-draw-card"
        pred_class = "prediction-no-draw"
        pred_emoji = "⚔️"
        pred_text = "NO DRAW EXPECTED"
        badge = f'<span class="no-draw-badge">✅ BET (Double Chance 12)</span>'
        confidence = result.get("confidence", "HIGH")
    elif decision == "CONSIDER":
        card_class = "consider-card"
        pred_class = "prediction-consider"
        pred_emoji = "⚠️"
        pred_text = "CONSIDER NO DRAW"
        badge = f'<span class="consider-badge">⚠️ CONSIDER</span>'
        confidence = result.get("confidence", "MEDIUM")
    else:
        card_class = "skip-card"
        pred_class = "prediction-skip"
        pred_emoji = "❌"
        pred_text = "SKIP - Insufficient Evidence"
        badge = f'<span class="skip-badge">❌ SKIP</span>'
        confidence = "LOW"
    
    st.markdown(f"""
    <div class="output-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">CLASH OF IDENTITIES - NO DRAW PREDICTOR</div>
                <div class="prediction-display {pred_class}">
                    {pred_emoji} {pred_text}
                </div>
                <div>
                    {badge}
                    <span class="final-badge" style="margin-left:0.5rem;">Identity Gap: {gap:.1f}</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">No Draw Score</div>
                <div style="font-size: 2.5rem; font-weight: 800;">{score:.0f}</div>
                <div>
                    <span style="font-size:0.7rem; color:#94a3b8;">{confidence} Confidence</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            {result.get('action', '')}
            {f'<br><span style="color:#ef4444;">⚠️ {result["reason"]}</span>' if result.get("reason") else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🏷️ Team Identity Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        display_identity(result["home_identity"], home_team, True)
    with col2:
        display_identity(result["away_identity"], away_team, False)
    
    if result["eligible"]:
        st.markdown(f"""
        <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.5rem 0; text-align:center;">
            <span style="color:#94a3b8;">Identity Gap:</span>
            <span style="font-weight:800; font-size:1.2rem;">{gap:.1f}</span>
            <span style="color:#94a3b8; margin-left:1rem;">→</span>
            <span style="font-weight:800; font-size:1.2rem; color:#10b981;">Score: {score:.0f}</span>
            <span style="color:#94a3b8; margin-left:1rem;">Threshold: ≥ 50 = NO DRAW</span>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, result: dict):
    """Only save matches that are NO_DRAW or CONSIDER"""
    try:
        decision = result.get("decision", "SKIP")
        
        if decision not in ["NO_DRAW", "CONSIDER"]:
            return "SKIPPED"
            
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", datetime.now().strftime("%Y-%m-%d"))
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else datetime.now().strftime("%Y-%m-%d")
        
        if check_match_exists(home_team, away_team, match_date):
            return "ALREADY_EXISTS"
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            "nd_home": match.get("nd_home", 0),
            "nd_away": match.get("nd_away", 0),
            "w_home": match.get("w_home", 0),
            "w_away": match.get("w_away", 0),
            "l_home": match.get("l_home", 0),
            "l_away": match.get("l_away", 0),
            "best_team_home": match.get("best_team_home", 0),
            "best_team_away": match.get("best_team_away", 0),
            "worst_team_home": match.get("worst_team_home", 0),
            "worst_team_away": match.get("worst_team_away", 0),
            "best_off_home": match.get("best_off_home", 0),
            "best_off_away": match.get("best_off_away", 0),
            "worst_def_home": match.get("worst_def_home", 0),
            "worst_def_away": match.get("worst_def_away", 0),
            "now_home": match.get("now_home", 0),
            "now_away": match.get("now_away", 0),
            "nol_home": match.get("nol_home", 0),
            "nol_away": match.get("nol_away", 0),
            "dc12_vs": result.get("score", 0),  # Reusing field as No Draw Score
            "dc12_odds": 1 / ((1 / match.get("home_odds", 0)) + (1 / match.get("away_odds", 0))) if match.get("home_odds", 0) > 0 and match.get("away_odds", 0) > 0 else 0,
            "predicted": decision,
            "confidence": result.get("confidence", "LOW"),
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
            is_correct = (predicted == "NO_DRAW" and actual_result != "X") or (predicted == "CONSIDER" and actual_result != "X")
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
        pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ SKIP"
        actual_display = "🤝 DRAW" if actual == "X" else "🏠 HOME" if actual == "1" else "✈️ AWAY"
        rows.append({
            "Date": r.get("match_date", ""),
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": pred_display,
            "Actual": actual_display,
            "Score": r.get('dc12_vs', 0),
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
        ### The Logic
        A draw happens when two teams are **similar**. No draw happens when teams have **extreme differences** in style or quality.
        
        ### The Formula
        **For each team:**
        ATTACK_POWER = (Wins × 2) + (No_Losses × 1) + (Best_Team) + (Best_Offense)
        DEFENSIVE_SOLIDITY = (Losses × 1) + (No_Losses × 2) + (Worst_Team) + (Worst_Defense)
        NET_IDENTITY = ATTACK_POWER - DEFENSIVE_SOLIDITY

        **For the match:**
        GAP = |NET_IDENTITY_HOME - NET_IDENTITY_AWAY|
        SCORE = GAP × 10

        ### Decision Rules
        | Score | Decision |
        |-------|----------|
        | **≥ 50** | ✅ NO DRAW (Bet Double Chance 12) |
        | **30-49** | ⚠️ CONSIDER |
        | **< 30** | ❌ SKIP |

        ### Eligibility
        - BOTH teams need ≥ 2 data points
        - Insufficient data = SKIP
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["⚔️ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Betexplorer Data")
        st.info("Only predicts when BOTH teams have sufficient data (≥ 2 data points each)")

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
                    with st.spinner("Analyzing team identities..."):
                        matches = parse_betexplorer_data(text_data)
                    if matches:
                        st.success(f"✅ Found {len(matches)} unique matches")
                        analyzed_results = []
                        stored_count = already_stored_count = no_draw_count = consider_count = 0
                        
                        for match in matches:
                            result = calculate_no_draw_prediction(match)
                            exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                            
                            if exists:
                                already_stored_count += 1
                                analyzed_results.append((match, result, True))
                            else:
                                if result["decision"] in ["NO_DRAW", "CONSIDER"] and result["eligible"]:
                                    saved_id = save_to_db(match, result)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, result, True))
                                    elif saved_id:
                                        stored_count += 1
                                        analyzed_results.append((match, result, False))
                                        if result["decision"] == "NO_DRAW":
                                            no_draw_count += 1
                                        elif result["decision"] == "CONSIDER":
                                            consider_count += 1
                                    else:
                                        analyzed_results.append((match, result, False))
                                else:
                                    analyzed_results.append((match, result, False))
                        
                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed | ⚔️ {no_draw_count} no-draw bets | ⚠️ {consider_count} considered")
                        
                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### ⚔️ PREDICTION RESULTS")
                            
                            no_draws = [(m, r, s) for m, r, s in analyzed_results if r.get("decision") == "NO_DRAW" and r.get("eligible")]
                            considers = [(m, r, s) for m, r, s in analyzed_results if r.get("decision") == "CONSIDER" and r.get("eligible")]
                            skips = [(m, r, s) for m, r, s in analyzed_results if r.get("decision") == "SKIP" or not r.get("eligible")]

                            if no_draws:
                                st.markdown("#### ✅ NO DRAW PREDICTIONS")
                                for idx, (match, result, already_stored) in enumerate(no_draws, 1):
                                    st.markdown(f"##### Match #{idx}: {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}")
                                    display_prediction(match, result)
                                    if idx < len(no_draws):
                                        st.markdown("---")

                            if considers:
                                st.markdown("#### ⚠️ CONSIDER")
                                for idx, (match, result, already_stored) in enumerate(considers, 1):
                                    with st.expander(f"{match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')} - Score: {result.get('score', 0):.0f}"):
                                        display_prediction(match, result)

                            if skips:
                                st.markdown("#### ❌ SKIPPED (Insufficient Data)")
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
                                st.metric("⚠️ Considered", consider_count)
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
                score = a.get('dc12_vs', 0)
                pred_display = "⚔️ NO DRAW" if pred == "NO_DRAW" else "⚠️ CONSIDER" if pred == "CONSIDER" else "❌ SKIP"
                badge = f"{pred_display} ({confidence}) — Score: {score:.0f}"
                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred_display} — Score: {score:.0f}")
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
                    "Score": r.get('dc12_vs', 0),
                    "Result": result_badge,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
