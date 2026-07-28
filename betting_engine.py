"""
MATCH ANALYZER V18.0 — PURE POISSON DRAW PROBABILITY
No Forebet logic. No team-specific rules. Just mathematics.
"""

import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import math
import traceback
from typing import Dict, Tuple, Optional, List

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
TABLE_NAME = "match_predictions_v18"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Poisson Draw Probability V18.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .ft-card { border-left: 5px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0505 100%); }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .metric-card { background: #0f172a; border-radius: 10px; padding: 0.75rem; text-align: center; flex: 1; }
    .metric-value { font-size: 1.5rem; font-weight: 800; }
    .metric-label { font-size: 0.7rem; color: #94a3b8; }
    .prediction-display { font-size: 2.5rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-draw { color: #f59e0b; }
    .prediction-no-draw { color: #10b981; }
    .prediction-coinflip { color: #3b82f6; }
    .final-badge { background: #8b5cf6; color: #fff; padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; border: 2px solid #8b5cf6; }
    .draw-prob-high { color: #f59e0b; font-weight: 800; }
    .draw-prob-low { color: #10b981; font-weight: 800; }
    .draw-prob-mid { color: #3b82f6; font-weight: 800; }
    .stake-badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .stake-full { background: #10b981; color: #000; }
    .stake-half { background: #f59e0b; color: #000; }
    .stake-tiny { background: #64748b; color: #fff; }
    .already-stored { background: #1a2a2a; border: 1px solid #f59e0b; border-radius: 4px; padding: 0.2rem 0.6rem; color: #fbbf24; font-size: 0.7rem; font-weight: 700; display: inline-block; }
    .league-badge { display: inline-block; padding: 0.2rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 700; }
    .league-badge.br { background: #10b981; color: #fff; }
    .league-badge.uk { background: #3b82f6; color: #fff; }
    .league-badge.es { background: #f59e0b; color: #000; }
    .league-badge.it { background: #8b5cf6; color: #fff; }
    .league-badge.de { background: #ec4899; color: #fff; }
    .league-badge.fr { background: #3b82f6; color: #fff; }
    .league-badge.tr { background: #ef4444; color: #fff; }
    .league-badge.sa { background: #10b981; color: #fff; }
    .league-badge.au { background: #f59e0b; color: #000; }
    .league-badge.no { background: #ef4444; color: #fff; }
    .league-badge.unknown { background: #64748b; color: #fff; }
    .factor-row { display: flex; justify-content: space-between; padding: 0.3rem 0; border-bottom: 1px solid #1e293b; }
    .factor-name { color: #94a3b8; }
    .factor-value { font-weight: 600; }
    .score-matrix { display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 0.5rem; }
    .score-cell { background: #1e293b; border-radius: 8px; padding: 0.5rem; text-align: center; color: #fff; }
    .score-cell .score { font-size: 1.1rem; font-weight: 800; }
    .score-cell .prob { font-size: 0.65rem; color: #94a3b8; }
    .upload-container { border: 2px dashed #3b82f6; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .upload-container:hover { border-color: #60a5fa; background: rgba(59, 130, 246, 0.05); }
    .result-win { border-left-color: #10b981; }
    .result-loss { border-left-color: #ef4444; }
    .result-draw { border-left-color: #f59e0b; }
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
    
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        pass
    
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        pass
    
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except:
        pass
    
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
    except Exception as e:
        return False


def get_stake_display(stake_value: str) -> tuple:
    stake_mapping = {
        "2 units": ("2 units", "stake-full"),
        "1 unit": ("1 unit", "stake-half"),
        "0.1 unit": ("0.1 unit", "stake-tiny"),
        "": ("0 units", "stake-tiny"),
    }
    
    if stake_value in stake_mapping:
        return stake_mapping[stake_value]
    else:
        return (stake_value, "stake-tiny")


def get_league_badge(league: str) -> str:
    league_lower = league.lower()
    if "brazil" in league_lower or "brasileiro" in league_lower:
        return "br"
    elif "premier" in league_lower or "epl" in league_lower:
        return "uk"
    elif "spain" in league_lower or "la liga" in league_lower:
        return "es"
    elif "italy" in league_lower or "serie a" in league_lower:
        return "it"
    elif "germany" in league_lower or "bundesliga" in league_lower:
        return "de"
    elif "france" in league_lower or "ligue" in league_lower:
        return "fr"
    elif "turkey" in league_lower or "super lig" in league_lower:
        return "tr"
    elif "saudi" in league_lower:
        return "sa"
    elif "australia" in league_lower or "a-league" in league_lower:
        return "au"
    elif "norway" in league_lower or "eliteserien" in league_lower:
        return "no"
    else:
        return "unknown"


# ============================================================================
# POISSON DRAW PROBABILITY ENGINE
# ============================================================================
def calculate_draw_probability(
    home_scored_avg: float,
    home_conceded_avg: float,
    away_scored_avg: float,
    away_conceded_avg: float,
    league_draw_rate: float = 0.26,
    draw_odds: Optional[float] = None
) -> dict:
    """
    Calculate draw probability using Poisson distribution.
    
    Returns:
        dict: {
            "raw_prob": float,
            "adjusted_prob": float,
            "final_prob": float,
            "decision": str,  # "DRAW", "NOT_DRAW", or "COIN_FLIP"
            "prediction": str,  # "X" or "NO_DRAW"
            "confidence": str,  # "HIGH", "MEDIUM", "LOW"
            "stake": str,
            "reason": str,
            "home_goal_exp": float,
            "away_goal_exp": float,
            "draw_scorelines": list,
        }
    """
    
    # ========================================================================
    # Step 1: Calculate Goal Expectancies (λ)
    # ========================================================================
    lambda_home = (home_scored_avg + away_conceded_avg) / 2
    lambda_away = (away_scored_avg + home_conceded_avg) / 2
    
    # Prevent impossible values
    lambda_home = max(lambda_home, 0.1)
    lambda_away = max(lambda_away, 0.1)
    
    # ========================================================================
    # Step 2: Poisson Probability Function
    # ========================================================================
    def poisson_probability(lambda_val: float, k: int) -> float:
        """P(X = k) = (λ^k × e^(-λ)) / k!"""
        if k == 0:
            return math.exp(-lambda_val)
        
        # Calculate using log to avoid overflow
        log_prob = k * math.log(lambda_val) - lambda_val - math.lgamma(k + 1)
        return math.exp(log_prob)
    
    # ========================================================================
    # Step 3: Calculate Draw Scoreline Probabilities
    # ========================================================================
    draw_scorelines = []
    raw_draw_prob = 0.0
    
    # Go up to 10 goals
    for k in range(0, 11):
        p_home = poisson_probability(lambda_home, k)
        p_away = poisson_probability(lambda_away, k)
        draw_prob = p_home * p_away
        raw_draw_prob += draw_prob
        draw_scorelines.append({
            "scoreline": f"{k}-{k}",
            "home_goals": k,
            "away_goals": k,
            "probability": draw_prob
        })
    
    # ========================================================================
    # Step 4: League Adjustment
    # ========================================================================
    global_avg_draw = 0.26
    adjustment_factor = league_draw_rate / global_avg_draw
    adjusted_draw_prob = raw_draw_prob * adjustment_factor
    
    # Normalize to reasonable range
    adjusted_draw_prob = max(0.05, min(0.45, adjusted_draw_prob))
    
    # ========================================================================
    # Step 5: Market Adjustment (if odds provided)
    # ========================================================================
    if draw_odds and draw_odds > 0:
        implied_draw_prob = 1.0 / draw_odds
        # Blend: 60% mathematical, 40% market
        final_draw_prob = (0.60 * adjusted_draw_prob) + (0.40 * implied_draw_prob)
    else:
        final_draw_prob = adjusted_draw_prob
    
    # ========================================================================
    # Step 6: Decision Rule
    # ========================================================================
    if final_draw_prob > 0.32:
        decision = "DRAW"
        prediction = "X"
        confidence = "HIGH"
        stake = "2 units"
        reason = f"Draw probability {final_draw_prob:.1%} > 32% threshold"
    elif final_draw_prob < 0.28:
        decision = "NOT_DRAW"
        prediction = "NO_DRAW"
        confidence = "MEDIUM"
        stake = "1 unit"
        reason = f"Draw probability {final_draw_prob:.1%} < 28% threshold"
    else:
        decision = "COIN_FLIP"
        prediction = "COIN_FLIP"
        confidence = "LOW"
        stake = "0.1 unit"
        reason = f"Draw probability {final_draw_prob:.1%} in coin-flip zone (28-32%)"
    
    return {
        "raw_prob": raw_draw_prob,
        "adjusted_prob": adjusted_draw_prob,
        "final_prob": final_draw_prob,
        "decision": decision,
        "prediction": prediction,
        "confidence": confidence,
        "stake": stake,
        "reason": reason,
        "home_goal_exp": lambda_home,
        "away_goal_exp": lambda_away,
        "draw_scorelines": draw_scorelines,
        "league_draw_rate": league_draw_rate,
    }


# ============================================================================
# DATA PARSER — ONLY EXTRACTS GOALS AVERAGES
# ============================================================================
def parse_match_data(text: str) -> list:
    """
    Parse match data to extract:
    - Home team name
    - Away team name
    - Home goals scored avg
    - Home goals conceded avg
    - Away goals scored avg
    - Away goals conceded avg
    - League name
    - Match date
    - FT status (if played)
    - Actual score (if played)
    """
    
    matches = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Look for match pattern: team names and stats
        # Example: "Vitória vs Palmeiras" or "Vitória - Palmeiras"
        match_name = re.search(r'([A-Za-zÀ-ÿ\s]+)\s*(?:vs|VS|[-–])\s*([A-Za-zÀ-ÿ\s]+)', line)
        if not match_name:
            i += 1
            continue
        
        home_team = match_name.group(1).strip()
        away_team = match_name.group(2).strip()
        
        # Look for goals scored avg
        # Pattern: "Goals per game: 1.1" or "1.1 goals/game" or just "1.1"
        home_scored = None
        home_conceded = None
        away_scored = None
        away_conceded = None
        league = "Unknown"
        match_date = None
        is_finished = False
        actual_home = None
        actual_away = None
        
        # Search forward for stats (up to 20 lines)
        for j in range(i + 1, min(i + 20, len(lines))):
            stat_line = lines[j].strip()
            
            # Check for FT result
            ft_match = re.search(r'FT\s+(\d+)\s*[-:]\s*(\d+)', stat_line)
            if ft_match:
                is_finished = True
                actual_home = int(ft_match.group(1))
                actual_away = int(ft_match.group(2))
            
            # Extract home goals scored avg
            if home_team in stat_line or "Home" in stat_line or "home" in stat_line:
                # Look for goals per game
                gpg = re.search(r'([\d.]+)\s*goals?\s*per\s*game', stat_line)
                if gpg:
                    if home_scored is None:
                        home_scored = float(gpg.group(1))
                
                # Look for goals conceded avg
                conceded = re.search(r'conceded\s*([\d.]+)', stat_line)
                if conceded:
                    if home_conceded is None:
                        home_conceded = float(conceded.group(1))
            
            # Extract away goals scored avg
            if away_team in stat_line or "Away" in stat_line or "away" in stat_line:
                gpg = re.search(r'([\d.]+)\s*goals?\s*per\s*game', stat_line)
                if gpg:
                    if away_scored is None:
                        away_scored = float(gpg.group(1))
                
                conceded = re.search(r'conceded\s*([\d.]+)', stat_line)
                if conceded:
                    if away_conceded is None:
                        away_conceded = float(conceded.group(1))
            
            # Extract league
            if "Brasileirão" in stat_line or "Serie A" in stat_line or "Premier" in stat_line:
                league = stat_line
            
            # Extract date
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', stat_line)
            if date_match and not match_date:
                match_date = date_match.group(1)
        
        # Calculate from table data if not found in stats
        # Look for table rows with team stats
        for j in range(i + 1, min(i + 30, len(lines))):
            stat_line = lines[j].strip()
            
            # Look for team with goals data
            if home_team in stat_line or away_team in stat_line:
                # Pattern: Team Name P W D L GF GA GD PTS
                table_match = re.search(
                    r'([A-Za-zÀ-ÿ\s]+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+)\s+(\d+)',
                    stat_line
                )
                if table_match:
                    team = table_match.group(1).strip()
                    gp = int(table_match.group(2))
                    gf = int(table_match.group(6))
                    ga = int(table_match.group(7))
                    
                    if team == home_team:
                        if home_scored is None and gp > 0:
                            home_scored = gf / gp
                        if home_conceded is None and gp > 0:
                            home_conceded = ga / gp
                    elif team == away_team:
                        if away_scored is None and gp > 0:
                            away_scored = gf / gp
                        if away_conceded is None and gp > 0:
                            away_conceded = ga / gp
        
        # Set defaults if missing
        if home_scored is None:
            home_scored = 1.0
        if home_conceded is None:
            home_conceded = 1.0
        if away_scored is None:
            away_scored = 1.0
        if away_conceded is None:
            away_conceded = 1.0
        
        # Create match entry
        matches.append({
            "home_team": home_team,
            "away_team": away_team,
            "home_scored_avg": home_scored,
            "home_conceded_avg": home_conceded,
            "away_scored_avg": away_scored,
            "away_conceded_avg": away_conceded,
            "league": league,
            "date": match_date or "Unknown",
            "is_finished": is_finished,
            "actual_home": actual_home,
            "actual_away": actual_away,
        })
        
        i += 1
    
    return matches


# ============================================================================
# ANALYSIS ENGINE
# ============================================================================
def analyze_match(match: dict, league_draw_rate: float, draw_odds: Optional[float] = None) -> dict:
    """Analyze match using Poisson draw probability engine"""
    
    if match.get("is_finished"):
        return {
            "verdict": "SKIP",
            "skip_reason": "Already played (FT)",
            "prediction": None,
            "confidence": None,
            "stake": None,
            "reason": None,
        }
    
    result = calculate_draw_probability(
        home_scored_avg=match.get("home_scored_avg", 1.0),
        home_conceded_avg=match.get("home_conceded_avg", 1.0),
        away_scored_avg=match.get("away_scored_avg", 1.0),
        away_conceded_avg=match.get("away_conceded_avg", 1.0),
        league_draw_rate=league_draw_rate,
        draw_odds=draw_odds
    )
    
    return {
        "verdict": "PROCESSED",
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "stake": result["stake"],
        "reason": result["reason"],
        "raw_prob": result["raw_prob"],
        "adjusted_prob": result["adjusted_prob"],
        "final_prob": result["final_prob"],
        "decision": result["decision"],
        "home_goal_exp": result["home_goal_exp"],
        "away_goal_exp": result["away_goal_exp"],
        "draw_scorelines": result["draw_scorelines"],
        "league_draw_rate": result["league_draw_rate"],
        "home_scored_avg": match.get("home_scored_avg", 1.0),
        "home_conceded_avg": match.get("home_conceded_avg", 1.0),
        "away_scored_avg": match.get("away_scored_avg", 1.0),
        "away_conceded_avg": match.get("away_conceded_avg", 1.0),
    }


# ============================================================================
# EVALUATION ENGINE
# ============================================================================
def evaluate_prediction(prediction: str, actual_home: int, actual_away: int) -> dict:
    try:
        home = int(actual_home) if actual_home is not None else 0
        away = int(actual_away) if actual_away is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID", "winner": "INVALID"}
    
    if home > away:
        actual = "1"
    elif away > home:
        actual = "2"
    else:
        actual = "X"
    
    # Prediction mapping: "X" = draw, "NO_DRAW" = not draw
    if prediction == "X":
        predicted = "X"
    else:
        predicted = "NOT_DRAW"
    
    if predicted == "X" and actual == "X":
        is_correct = True
    elif predicted == "NOT_DRAW" and actual != "X":
        is_correct = True
    else:
        is_correct = False
    
    return {
        "is_correct": is_correct,
        "actual": actual,
        "winner": "HOME" if home > away else "AWAY" if away > home else "DRAW",
        "score": f"{home}-{away}",
        "total_goals": home + away,
    }


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def display_analysis(match: dict, analysis: dict, league: str, already_stored: bool = False):
    if analysis.get("verdict") == "SKIP":
        st.markdown(f"""
        <div class="output-card ft-card">
            <div style="text-align:center; padding:1rem;">
                <div style="font-size:1.5rem; font-weight:800; color:#ef4444;">⏭️ SKIPPED — Already Played</div>
                <p style="color:#94a3b8; font-size:1.1rem;">
                    {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')}
                </p>
                <p style="color:#ef4444;">FT {match.get('actual_home', '?')}-{match.get('actual_away', '?')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    badge_class = get_league_badge(league)
    st.markdown(f'<span class="league-badge {badge_class}">{league}</span>', unsafe_allow_html=True)
    
    if already_stored:
        st.markdown('<span class="already-stored">📌 ALREADY STORED</span>', unsafe_allow_html=True)
    
    prediction = analysis.get("prediction", "COIN_FLIP")
    confidence = analysis.get("confidence", "LOW")
    stake = analysis.get("stake", "0.1 unit")
    reason = analysis.get("reason", "")
    final_prob = analysis.get("final_prob", 0)
    
    if prediction == "X":
        pred_text = "DRAW"
        pred_emoji = "🤝"
        pred_class = "prediction-draw"
    elif prediction == "NO_DRAW":
        pred_text = "NOT DRAW"
        pred_emoji = "🚫"
        pred_class = "prediction-no-draw"
    else:
        pred_text = "COIN FLIP"
        pred_emoji = "🪙"
        pred_class = "prediction-coinflip"
    
    confidence_color = "#10b981" if confidence == "HIGH" else "#f59e0b" if confidence == "MEDIUM" else "#64748b"
    
    prob_class = "draw-prob-high" if final_prob > 0.32 else "draw-prob-low" if final_prob < 0.28 else "draw-prob-mid"
    
    stake_display, _ = get_stake_display(stake)
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 16px; padding: 1.5rem; margin: 0.75rem 0; border-left: 4px solid {confidence_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.8rem; color: #94a3b8;">V18.0 POISSON DRAW PROBABILITY</div>
                <div class="prediction-display {pred_class}">
                    {pred_emoji} {pred_text}
                </div>
                <div>
                    <span style="background:#8b5cf6; color:#fff; padding:0.2rem 0.6rem; border-radius:4px; font-size:0.8rem; font-weight:700;">Poisson Distribution</span>
                    <span class="final-badge" style="margin-left:0.5rem;">V18.0</span>
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.8rem; color: #94a3b8;">Draw Probability</div>
                <div style="font-size: 2.5rem; font-weight: 800; class="{prob_class};">{final_prob:.1%}</div>
                <div>
                    <span class="stake-badge {stake_display.split()[1] if len(stake_display.split()) > 1 else 'stake-tiny'}">Stake: {stake_display.split()[0]}</span>
                    <span style="font-size:0.7rem; color:#94a3b8; margin-left:0.5rem;">{confidence} Confidence</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; border-top: 1px solid #1e293b; padding-top: 0.5rem;">
            📝 {reason}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics
    st.markdown("### 📊 Poisson Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        home_exp = analysis.get("home_goal_exp", 0)
        away_exp = analysis.get("away_goal_exp", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{home_exp:.2f} / {away_exp:.2f}</div><div class="metric-label">Goal Expectancy (λ H/A)</div></div>', unsafe_allow_html=True)
    with m2:
        raw_prob = analysis.get("raw_prob", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{raw_prob:.1%}</div><div class="metric-label">Raw Poisson Draw</div></div>', unsafe_allow_html=True)
    with m3:
        adj_prob = analysis.get("adjusted_prob", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{adj_prob:.1%}</div><div class="metric-label">League-Adjusted</div></div>', unsafe_allow_html=True)
    with m4:
        league_rate = analysis.get("league_draw_rate", 0)
        st.markdown(f'<div class="metric-card"><div class="metric-value">{league_rate:.1%}</div><div class="metric-label">League Draw Rate</div></div>', unsafe_allow_html=True)
    
    # Draw Scorelines
    st.markdown("### 🎯 Most Likely Draw Scorelines")
    
    draw_scorelines = analysis.get("draw_scorelines", [])
    if draw_scorelines:
        top_scores = sorted(draw_scorelines, key=lambda x: x["probability"], reverse=True)[:6]
        
        cols = st.columns(min(6, len(top_scores)))
        for idx, s in enumerate(top_scores):
            with cols[idx]:
                prob_pct = s["probability"] * 100
                st.markdown(f"""
                <div style="background:#1e293b; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;">
                    <div style="font-size:1.2rem; font-weight:800;">{s['scoreline']}</div>
                    <div style="font-size:0.7rem; color:#94a3b8;">{prob_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Input Data
    st.markdown("### 📈 Input Data Used")
    
    st.markdown(f"""
    <div style="background:#0f172a; border-radius:8px; padding:0.75rem; margin:0.25rem 0;">
        <div class="factor-row"><span class="factor-name">🏠 {match.get('home_team', 'Home')} Goals Scored Avg</span><span class="factor-value">{analysis.get('home_scored_avg', 0):.2f}</span></div>
        <div class="factor-row"><span class="factor-name">🏠 {match.get('home_team', 'Home')} Goals Conceded Avg</span><span class="factor-value">{analysis.get('home_conceded_avg', 0):.2f}</span></div>
        <div class="factor-row"><span class="factor-name">✈️ {match.get('away_team', 'Away')} Goals Scored Avg</span><span class="factor-value">{analysis.get('away_scored_avg', 0):.2f}</span></div>
        <div class="factor-row"><span class="factor-name">✈️ {match.get('away_team', 'Away')} Goals Conceded Avg</span><span class="factor-value">{analysis.get('away_conceded_avg', 0):.2f}</span></div>
        <div class="factor-row"><span class="factor-name">🏷️ League Draw Rate</span><span class="factor-value">{analysis.get('league_draw_rate', 0):.1%}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("📐 Formula: Draw Probability = Σ(λ_home^k × e^(-λ_home) / k! × λ_away^k × e^(-λ_away) / k!) × (League_Rate / 0.26)")


def display_records_table(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    
    total = len(results)
    correct = 0
    incorrect = 0
    
    for r in results:
        if r.get('predicted_1x2') and r.get('actual_1x2'):
            # Convert stored prediction back to draw/no-draw
            pred = r.get('predicted_1x2')
            actual = r.get('actual_1x2')
            
            if pred == "X" and actual == "X":
                correct += 1
            elif pred == "NO_DRAW" and actual != "X":
                correct += 1
            else:
                incorrect += 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Matches</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">V18.0 Accuracy</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Incorrect</div></div>', unsafe_allow_html=True)
    
    st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")
    
    rows = []
    for r in results:
        pred = r.get('predicted_1x2', '?')
        actual = r.get('actual_1x2', '?')
        league = r.get('league_name', '')
        badge_class = get_league_badge(league)
        
        is_correct = False
        if pred == "X" and actual == "X":
            is_correct = True
        elif pred == "NO_DRAW" and actual != "X":
            is_correct = True
        
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        
        pred_display = "🤝 DRAW" if pred == "X" else "🚫 NO DRAW"
        actual_display = "🤝" if actual == "X" else "🏠" if actual == "1" else "✈️"
        
        rows.append({
            "Date": r.get("match_date", ""),
            "League": f'<span class="league-badge {badge_class}" style="font-size:0.7rem;">{league[:15]}</span>',
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Prediction": pred_display,
            "Actual": actual_display,
            "Draw Prob": f"{r.get('draw_probability', 0):.1%}",
            "Result": result_badge,
        })
    
    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================
def save_to_db(match: dict, analysis: dict, league: str, league_draw_rate: float):
    try:
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", "")
        
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        
        exists = check_match_exists(home_team, away_team, match_date)
        if exists:
            return "ALREADY_EXISTS"
        
        record = {
            "match_date": date_part,
            "league_name": league,
            "home_team": home_team,
            "away_team": away_team,
            "home_scored_avg": match.get("home_scored_avg", 0),
            "home_conceded_avg": match.get("home_conceded_avg", 0),
            "away_scored_avg": match.get("away_scored_avg", 0),
            "away_conceded_avg": match.get("away_conceded_avg", 0),
            "league_draw_rate": league_draw_rate,
            "predicted_1x2": analysis.get("prediction"),
            "prediction_confidence": analysis.get("confidence"),
            "recommended_bet": "Draw" if analysis.get("prediction") == "X" else "No Draw",
            "stake": analysis.get("stake"),
            "draw_probability": analysis.get("final_prob", 0),
            "raw_poisson_prob": analysis.get("raw_prob", 0),
            "adjusted_poisson_prob": analysis.get("adjusted_prob", 0),
            "home_goal_exp": analysis.get("home_goal_exp", 0),
            "away_goal_exp": analysis.get("away_goal_exp", 0),
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_1x2": None,
            "is_correct": False,
        }
        
        response = supabase.table(TABLE_NAME).insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None


def get_pending():
    try:
        response = supabase.table(TABLE_NAME).select("*").is_("actual_1x2", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except Exception as e:
        st.error(f"Error fetching pending: {e}")
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        actual_1x2 = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        
        response = supabase.table(TABLE_NAME).select("predicted_1x2").eq("id", analysis_id).execute()
        if response.data:
            predicted = response.data[0].get("predicted_1x2")
            if predicted == "X" and actual_1x2 == "X":
                is_correct = True
            elif predicted == "NO_DRAW" and actual_1x2 != "X":
                is_correct = True
            else:
                is_correct = False
        else:
            is_correct = False
        
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_1x2": actual_1x2,
            "is_correct": is_correct
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False


def get_results():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_1x2", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except:
        return []


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V18.0 — Pure Poisson Draw Probability")
    st.caption(f"Mathematical draw prediction engine. No team-specific rules. Table: {TABLE_NAME}")

    with st.expander("📖 V18.0 — HOW IT WORKS", expanded=False):
        st.markdown("""
        **Pure Mathematics. No Forebet. No Team-Specific Rules.**
        
        ### The Formula:
        
        Draw Probability = Σ [P_home(k) × P_away(k)] × (League_Draw_Rate / 0.26)
        
        Where:
        - **P(k)** = Poisson probability of scoring exactly k goals
        - **λ_home** = (Home_Scored_Avg + Away_Conceded_Avg) / 2
        - **λ_away** = (Away_Scored_Avg + Home_Conceded_Avg) / 2
        - **League_Draw_Rate** = Historical draw rate for that league

        ### Decision Rules:

        | Probability | Decision | Prediction | Stake |
        |-------------|----------|------------|-------|
        | **> 32%** | DRAW | X | 2u |
        | **< 28%** | NOT DRAW | NO_DRAW | 1u |
        | **28-32%** | COIN FLIP | COIN_FLIP | 0.1u |

        ### Data Required:
        1. Home Goals Scored Avg
        2. Home Goals Conceded Avg
        3. Away Goals Scored Avg
        4. Away Goals Conceded Avg
        5. League Draw Rate (user provides)
        6. Draw Odds (optional)
        """)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Pending Matches", "📊 Records", "📈 Dashboard"])

    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info(f"V18.0: Pure Poisson Draw Probability. Saving to `{TABLE_NAME}`")

        st.markdown("""
        <div class="upload-container">
            <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">📋 Paste Match Data</p>
            <p style="color: #94a3b8; margin-bottom: 1rem;">The app will extract team names and goalscoring averages</p>
        </div>
        """, unsafe_allow_html=True)

        text_data = st.text_area(
            "Paste match data here", 
            height=300, 
            key="text_paste",
            placeholder="Paste the match data with team names and goalscoring averages..."
        )

        col1, col2 = st.columns(2)
        with col1:
            league_draw_rate = st.slider(
                "League Draw Rate (%)",
                min_value=15,
                max_value=35,
                value=28,
                step=1,
                help="Historical draw rate for this league (e.g., Brazil Serie A ~28%)"
            ) / 100.0

        with col2:
            draw_odds = st.number_input(
                "Draw Odds (Optional)",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.1,
                help="Decimal odds for the draw (e.g., 3.30). Leave 0 to skip market adjustment."
            )
            if draw_odds > 0:
                st.caption(f"Implied draw probability: {1/draw_odds:.1%}")

        if st.button("📊 ANALYZE — POISSON ENGINE", type="primary"):
            if not text_data or len(text_data.strip()) < 50:
                st.error("❌ Please paste valid data (minimum 50 characters).")
            else:
                try:
                    with st.spinner("Calculating Poisson draw probabilities..."):
                        matches = parse_match_data(text_data)

                    if matches:
                        st.success(f"✅ Found {len(matches)} matches")
                        
                        matches_sorted = sorted(matches, key=lambda x: x.get("date", ""))
                        
                        analyzed_results = []
                        stored_count = 0
                        already_stored_count = 0
                        
                        for match in matches_sorted:
                            # Skip if no valid data
                            if match.get("home_scored_avg", 0) == 0 and match.get("away_scored_avg", 0) == 0:
                                continue
                            
                            analysis = analyze_match(match, league_draw_rate, draw_odds if draw_odds > 0 else None)
                            
                            if analysis.get("verdict") != "SKIP":
                                exists = check_match_exists(match.get("home_team"), match.get("away_team"), match.get("date"))
                                
                                if exists:
                                    already_stored_count += 1
                                    analyzed_results.append((match, analysis, True))
                                else:
                                    league = match.get("league", "Unknown League")
                                    saved_id = save_to_db(match, analysis, league, league_draw_rate)
                                    if saved_id == "ALREADY_EXISTS":
                                        already_stored_count += 1
                                        analyzed_results.append((match, analysis, True))
                                    elif saved_id:
                                        stored_count += 1
                                        analyzed_results.append((match, analysis, False))
                                    else:
                                        analyzed_results.append((match, analysis, False))

                        st.info(f"💾 {stored_count} new predictions stored | {already_stored_count} already existed")

                        if analyzed_results:
                            st.markdown("---")
                            st.markdown("### 🎯 MATCH PREDICTIONS (V18.0)")
                            
                            for idx, (match, analysis, already_stored) in enumerate(analyzed_results, 1):
                                prediction = analysis.get("prediction", "COIN_FLIP")
                                confidence = analysis.get("confidence", "LOW")
                                stake = analysis.get("stake", "0.1 unit")
                                final_prob = analysis.get("final_prob", 0)
                                
                                stored_badge = " 📌 ALREADY STORED" if already_stored else " ✅ NEW"
                                
                                pred_display = "🤝 DRAW" if prediction == "X" else "🚫 NOT DRAW" if prediction == "NO_DRAW" else "🪙 COIN FLIP"
                                league = match.get("league", "Unknown League")
                                
                                date_display = format_date_display(match.get('date', ''))
                                st.markdown(f"#### Match {idx}: {match.get('home_team', 'Unknown')} vs {match.get('away_team', 'Unknown')} → {pred_display} ({confidence}) {stored_badge}")
                                st.caption(f"📅 {date_display} | Draw Prob: {final_prob:.1%} | League: {league}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Prediction", pred_display)
                                with col2:
                                    st.metric("Confidence", confidence)
                                with col3:
                                    stake_display, _ = get_stake_display(stake)
                                    st.metric("Stake", stake_display.split()[0])
                                
                                display_analysis(match, analysis, league, already_stored)
                                
                                if idx < len(analyzed_results):
                                    st.markdown("---")
                            
                            st.markdown("---")
                            st.markdown("### 📊 Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Matches", len(matches))
                            with col2:
                                st.metric("💾 New Stored", stored_count)
                            with col3:
                                st.metric("📌 Already Stored", already_stored_count)
                                
                    else:
                        st.error("No matches found in the data. Please check the format.")

                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("📝 Pending Matches")
        st.caption("Enter actual scores once matches are played.")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending result(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                pred = a.get('predicted_1x2', '?')
                confidence = a.get('prediction_confidence', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                draw_prob = a.get('draw_probability', 0)

                pred_display = "🤝 DRAW" if pred == "X" else "🚫 NO DRAW" if pred == "NO_DRAW" else "🪙 COIN FLIP"
                badge = f"{pred_display} ({confidence}) — Draw: {draw_prob:.1%}"

                with st.expander(f"📅 {date_display} | {badge} | {ht} vs {at}"):
                    st.info(f"📊 Prediction: {pred_display} — Draw Probability: {draw_prob:.1%}")
                    st.caption(f"📅 Match Date: {match_date}")
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
            return

        total = len(results)
        correct = 0
        incorrect = 0

        high_confidence = 0
        high_correct = 0
        medium_confidence = 0
        medium_correct = 0
        low_confidence = 0
        low_correct = 0

        for r in results:
            pred = r.get('predicted_1x2')
            actual = r.get('actual_1x2')
            
            if pred and actual:
                if pred == "X" and actual == "X":
                    is_correct = True
                elif pred == "NO_DRAW" and actual != "X":
                    is_correct = True
                else:
                    is_correct = False
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
                
                confidence = r.get('prediction_confidence', 'LOW')
                if confidence == 'HIGH':
                    high_confidence += 1
                    if is_correct:
                        high_correct += 1
                elif confidence == 'MEDIUM':
                    medium_confidence += 1
                    if is_correct:
                        medium_correct += 1
                else:
                    low_confidence += 1
                    if is_correct:
                        low_correct += 1

        overall_rate = round(correct / total * 100) if total > 0 else 0
        high_rate = round(high_correct / high_confidence * 100) if high_confidence > 0 else 0
        medium_rate = round(medium_correct / medium_confidence * 100) if medium_confidence > 0 else 0
        low_rate = round(low_correct / low_confidence * 100) if low_confidence > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Matches</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{overall_rate}%</div><div class="stat-label">Overall Accuracy</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{high_rate}%</div><div class="stat-label">HIGH Confidence ({high_correct}/{high_confidence})</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="stat-box"><div class="stat-number">{low_rate}%</div><div class="stat-label">LOW Confidence ({low_correct}/{low_confidence})</div></div>', unsafe_allow_html=True)

        st.markdown("#### 📊 Draw Probability Distribution")

        prob_ranges = {
            "< 20%": {"total": 0, "correct": 0, "label": "< 20%"},
            "20-24%": {"total": 0, "correct": 0, "label": "20-24%"},
            "24-28%": {"total": 0, "correct": 0, "label": "24-28%"},
            "28-32%": {"total": 0, "correct": 0, "label": "28-32%"},
            "32-36%": {"total": 0, "correct": 0, "label": "32-36%"},
            "> 36%": {"total": 0, "correct": 0, "label": "> 36%"}
        }

        for r in results:
            prob = r.get('draw_probability', 0)
            pred = r.get('predicted_1x2', '')
            actual = r.get('actual_1x2', '')
            
            if prob < 0.20:
                key = "< 20%"
            elif prob < 0.24:
                key = "20-24%"
            elif prob < 0.28:
                key = "24-28%"
            elif prob < 0.32:
                key = "28-32%"
            elif prob < 0.36:
                key = "32-36%"
            else:
                key = "> 36%"
            
            prob_ranges[key]["total"] += 1
            
            if pred == "X" and actual == "X":
                prob_ranges[key]["correct"] += 1
            elif pred == "NO_DRAW" and actual != "X":
                prob_ranges[key]["correct"] += 1

        df_probs = pd.DataFrame([
            {
                "Range": k, 
                "Total": v["total"], 
                "Correct": v["correct"], 
                "Rate": f"{round(v['correct']/v['total']*100) if v['total'] > 0 else 0}%"
            }
            for k, v in prob_ranges.items() if v["total"] > 0
        ])

        if not df_probs.empty:
            st.dataframe(df_probs, use_container_width=True)


if __name__ == "__main__":
    main()
