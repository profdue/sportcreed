"""
MATCH ANALYZER V4.0 — PROFIT-DRIVEN ENGINE
Based on performance analysis: LOW-SCORING (75%), HIGH-SCORING (100%), AWAY THREAT (100%)
Removed: DRAW PRESSURE, COMPRESSION (50% win rate = random)
Tightened: BTTS (65%+ thresholds)
Prioritized: Low-scoring patterns first
Skip rate target: 40-50% (only clear edges)
"""

import streamlit as st
from datetime import date
from supabase import create_client, Client
import pandas as pd
import re
import json

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
st.set_page_config(page_title="Match Analyzer V4.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .secondary-card { border: 2px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); opacity: 0.9; }
    .skip-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .edge-box { background: #1e293b; border-radius: 10px; padding: 0.6rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.8rem; }
    .stButton button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .badge-upgrade { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #10b981; color: #000; margin: 0.1rem; }
    .badge-caution { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #ef4444; color: #fff; margin: 0.1rem; }
    .badge-info { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #3b82f6; color: #fff; margin: 0.1rem; }
    .badge-skip { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; background: #fbbf24; color: #000; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .correct-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .incorrect-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.7rem; font-weight: 700; }
    .verdict-skip { text-align: center; padding: 1.5rem; }
    .verdict-skip .big-text { font-size: 1.5rem; font-weight: 800; color: #fbbf24; }
    .section-label { font-size: 0.9rem; font-weight: 700; color: #10b981; margin-top: 1rem; }
    .section-label-secondary { font-size: 0.9rem; font-weight: 700; color: #f59e0b; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TOP LEAGUES (same as before)
# ============================================================================
TOP_LEAGUES = [
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    "Primeira Liga", "Eredivisie", "Saudi Pro League", "Major League Soccer",
    "Championship", "Argentine Primera Division"
]

# ============================================================================
# TEAM ABBREVIATIONS (same as before - truncated for brevity)
# ============================================================================
TEAM_ABBREVIATIONS = {
    "Nottingham Forest": ["nott'm forest", "nottm forest", "notts forest"],
    "Manchester United": ["man utd", "manchester utd", "man united"],
    "Manchester City": ["man city", "manchester city"],
    # ... (keep all your existing abbreviations)
    # For brevity, I'm showing truncated version. Use your full dict.
}

def fuzzy_team_match(team_name, text):
    if not team_name or not text:
        return False
    text_lower = text.lower().strip()
    team_lower = team_name.lower().strip()
    if team_lower in text_lower or text_lower in team_lower:
        return True
    for abbr in TEAM_ABBREVIATIONS.get(team_name, []):
        if abbr in text_lower:
            return True
    return False

# ============================================================================
# PARSER (same as before - keep your existing parse_match_data function)
# ============================================================================
def parse_match_data(raw_text: str) -> dict:
    # ... (keep your existing parser exactly as before)
    # For brevity, I'm not duplicating the entire parser here.
    # Use the parser from your previous code.
    pass

# ============================================================================
# OVERHAULED STRUCTURAL ENGINE V4.0
# Profit-driven, high thresholds, eliminate garbage categories
# ============================================================================
def analyze_match(data: dict) -> dict:
    result = {
        "primary_bet": None,
        "secondary_bet": None,
        "badges": [],
        "warnings": [],
        "verdict": "PENDING",
        "classification": None,
        "skip_reasons": []
    }
    
    # Extract measurements
    home_win = data.get("home_win") or 0
    away_win = data.get("away_win") or 0
    draw_pct = data.get("draw") or 0
    btts = data.get("btts") or 0
    over_25 = data.get("over_25") or 0
    over_35 = data.get("over_35") or 0
    under_25 = data.get("under_25") or 0
    
    home_o15 = data.get("home_over_15_goals") or 0
    away_o15 = data.get("away_over_15_goals") or 0
    home_o05 = data.get("home_over_05_goals")
    away_o05 = data.get("away_over_05_goals")
    
    # Fill missing with btts as proxy
    if home_o05 is None and btts:
        home_o05 = btts
    if away_o05 is None and btts:
        away_o05 = btts
    
    home_win_trend = data.get("home_win_trend")
    away_win_trend = data.get("away_win_trend")
    btts_trend = data.get("btts_trend")
    
    home_form = data.get("home_form_all") or []
    away_form = data.get("away_form_all") or []
    home_losses = sum(1 for r in home_form[:6] if r == 'L')
    away_losses = sum(1 for r in away_form[:6] if r == 'L')
    
    score_matrix = data.get("score_matrix", [])
    
    league = data.get("league")
    is_top_league = league in TOP_LEAGUES if league else False
    is_saudi = league == "Saudi Pro League" if league else False
    
    # ========================================================================
    # SCORE MATRIX STRUCTURE ANALYSIS
    # ========================================================================
    tight_cluster = False
    btts_dominant = False
    low_scoring_cluster = False
    goals_expected = False
    modal_is_draw = False
    modal_outcome = None
    home_cluster = 0
    away_cluster = 0
    draw_cluster = 0
    
    if len(score_matrix) >= 5:
        top5_spread = score_matrix[0]["probability"] - score_matrix[4]["probability"]
        tight_cluster = top5_spread < 5.0
        
        btts_count = sum(1 for s in score_matrix[:5] if s["home_goals"] > 0 and s["away_goals"] > 0)
        btts_dominant = btts_count >= 4  # 👈 Increased from 3 to 4 (stricter)
        
        low_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] <= 2)
        low_scoring_cluster = low_count >= 4
        
        goals_count = sum(1 for s in score_matrix[:5] if s["home_goals"] + s["away_goals"] >= 3)
        goals_expected = goals_count >= 4  # 👈 Increased from 3 to 4
        
        modal_outcome = score_matrix[0]["score"]
        modal_is_draw = score_matrix[0]["home_goals"] == score_matrix[0]["away_goals"]
        
        for s in score_matrix[:5]:
            if s["home_goals"] > s["away_goals"]:
                home_cluster += s["probability"]
            elif s["away_goals"] > s["home_goals"]:
                away_cluster += s["probability"]
            else:
                draw_cluster += s["probability"]
    
    under_35_pct = (100 - over_35) if over_35 else 0
    
    # ========================================================================
    # TREND REVERSAL (keep, but high threshold)
    # ========================================================================
    trend_reversal = False
    if home_win_trend is not None and away_win_trend is not None:
        if home_win_trend <= -2.0 and away_win_trend >= 2.0:  # Stricter
            trend_reversal = True
    
    # ========================================================================
    # TRUE STRONG FAVORITE (Home) - increased thresholds
    # ========================================================================
    is_true_strong = (home_win >= 65 and home_o15 >= 60 and not tight_cluster and home_losses <= 2)
    is_true_strong_away = (away_win >= 65 and away_o15 >= 60 and not tight_cluster and away_losses <= 2)
    
    # ========================================================================
    # LOW-SCORING (PRIORITY #1 - 75% win rate historically)
    # ========================================================================
    is_low_scoring = (
        low_scoring_cluster and 
        under_25 >= 55 and 
        btts < 50 and
        not (home_losses >= 4 and away_losses >= 4)  # Avoid both teams terrible
    )
    
    # ========================================================================
    # HIGH-SCORING (PRIORITY #2 - 100% win rate historically)
    # ========================================================================
    is_high_scoring = (
        goals_expected and 
        over_25 >= 60 and 
        btts >= 55 and
        home_o15 >= 50 and 
        away_o15 >= 50
    )
    
    # ========================================================================
    # AWAY THREAT (PRIORITY #3 - 100% win rate historically)
    # ========================================================================
    is_away_threat = (home_o15 < 35 and away_win >= 35 and away_o15 >= 30)
    
    # ========================================================================
    # BTTS (DEMOTED, STRICT THRESHOLDS - target 65%+ win rate)
    # ========================================================================
    btts_contradiction = (btts >= 60 and (home_o15 < 40 or away_o15 < 40))
    is_btts_play = (
        btts >= 65 and  # 👈 Increased from 55 to 65
        btts_dominant and
        home_o05 is not None and home_o05 >= 65 and  # 👈 Both must be strong
        away_o05 is not None and away_o05 >= 65 and
        home_o15 >= 40 and  # 👈 Both can score at least 1+
        away_o15 >= 40 and
        not btts_contradiction and
        btts_trend is not None and btts_trend >= 0  # Non-negative trend
    )
    
    # ========================================================================
    # FRAGILE FAVORITE (keep but higher bar)
    # ========================================================================
    is_fragile = (home_win >= 60 and home_o15 < 55 and home_o15 >= 30)
    is_fragile_away = (away_win >= 60 and away_o15 < 55 and away_o15 >= 30)
    
    # ========================================================================
    # SAUDI LEAGUE OVERRIDE (keep)
    # ========================================================================
    is_saudi_dominant = (is_saudi and home_win >= 65)
    
    # ========================================================================
    # BETTING DECISIONS - REORDERED AND STRICT
    # ========================================================================
    
    # RULE 1: Trend Reversal (high confidence)
    if trend_reversal:
        result["classification"] = "TREND REVERSAL"
        result["badges"].append(f"Trend Reversal: Home {home_win_trend:.1f} / Away {away_win_trend:.1f}")
        result["primary_bet"] = {
            "market": "Away Win or Draw",
            "confidence": 8.0,
            "probability": away_win + draw_pct,
            "reason": f"Market reversal. Home trend {home_win_trend:.1f}, Away trend {away_win_trend:.1f}"
        }
    
    # RULE 2: TRUE Strong Favorite (Home)
    elif is_true_strong:
        result["classification"] = "TRUE STRONG FAVORITE"
        if away_o05 is not None and away_o05 < 55:
            result["primary_bet"] = {
                "market": "Home Win to Nil",
                "confidence": 8.5,
                "probability": home_win,
                "reason": f"Dominant favorite ({home_win:.0f}% win, O1.5 {home_o15:.0f}%) + underdog unlikely to score ({away_o05:.0f}%)"
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win",
                "confidence": 7.5,
                "probability": home_win,
                "reason": f"Dominant favorite ({home_win:.0f}% win, O1.5 {home_o15:.0f}%)"
            }
    
    # RULE 3: TRUE Strong Favorite (Away)
    elif is_true_strong_away:
        result["classification"] = "TRUE STRONG FAVORITE (AWAY)"
        if home_o05 is not None and home_o05 < 55:
            result["primary_bet"] = {
                "market": "Away Win to Nil",
                "confidence": 8.5,
                "probability": away_win,
                "reason": f"Dominant away favorite ({away_win:.0f}% win, O1.5 {away_o15:.0f}%) + home unlikely to score"
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win",
                "confidence": 7.5,
                "probability": away_win,
                "reason": f"Dominant away favorite ({away_win:.0f}% win, O1.5 {away_o15:.0f}%)"
            }
    
    # RULE 4: LOW-SCORING (Primary profit driver)
    elif is_low_scoring:
        result["classification"] = "LOW-SCORING"
        if under_25 >= 60:
            result["primary_bet"] = {
                "market": "Under 2.5 Goals",
                "confidence": 7.5,
                "probability": under_25,
                "reason": f"Low-scoring cluster. Under 2.5 at {under_25:.1f}%. BTTS only {btts:.1f}%."
            }
        else:
            result["primary_bet"] = {
                "market": "Under 3.5 Goals",
                "confidence": 7.0,
                "probability": under_35_pct,
                "reason": f"Low-scoring profile. Under 3.5 at {under_35_pct:.0f}%."
            }
    
    # RULE 5: HIGH-SCORING
    elif is_high_scoring:
        result["classification"] = "HIGH-SCORING"
        result["primary_bet"] = {
            "market": "Over 2.5 Goals",
            "confidence": 7.5,
            "probability": over_25,
            "reason": f"High-scoring matrix. Over 2.5 at {over_25:.1f}%. Both teams can score."
        }
        if btts >= 60:
            result["secondary_bet"] = {
                "market": "BTTS",
                "confidence": 6.5,
                "probability": btts,
                "reason": "Secondary: BTTS in high-scoring setup"
            }
    
    # RULE 6: Away Threat
    elif is_away_threat:
        result["classification"] = "AWAY THREAT"
        result["primary_bet"] = {
            "market": "Away Win or Draw (Double Chance)",
            "confidence": 7.0,
            "probability": away_win + draw_pct,
            "reason": f"Home O1.5 only {home_o15:.1f}% — home can't score. Away won't lose."
        }
    
    # RULE 7: BTTS (Strict, demoted)
    elif is_btts_play:
        result["classification"] = "BTTS"
        confidence = 7.0 if btts_trend and btts_trend >= 0.5 else 6.5
        result["primary_bet"] = {
            "market": "BTTS",
            "confidence": confidence,
            "probability": btts,
            "reason": f"Strong BTTS setup: {btts:.1f}% probability, both teams reliable scorers."
        }
    
    # RULE 8: Fragile Favorite (Home)
    elif is_fragile:
        result["classification"] = "FRAGILE FAVORITE"
        result["badges"].append(f"Fragile favorite — O1.5 only {home_o15:.0f}%")
        if is_away_threat:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": away_win + draw_pct,
                "reason": f"Fragile favorite ({home_win:.0f}% win) + away threat"
            }
        else:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": home_win + draw_pct,
                "reason": f"Fragile favorite but underdog can't score"
            }
    
    # RULE 9: Fragile Favorite (Away)
    elif is_fragile_away:
        result["classification"] = "FRAGILE FAVORITE (AWAY)"
        result["badges"].append(f"Fragile away favorite — O1.5 only {away_o15:.0f}%")
        if home_o15 >= 35:
            result["primary_bet"] = {
                "market": "Home Win or Draw (Double Chance)",
                "confidence": 7.0,
                "probability": home_win + draw_pct,
                "reason": f"Fragile away favorite + home scoring threat"
            }
        else:
            result["primary_bet"] = {
                "market": "Away Win or Draw (Double Chance)",
                "confidence": 6.5,
                "probability": away_win + draw_pct,
                "reason": f"Fragile away favorite, home can't score"
            }
    
    # RULE 10: Saudi League Dominant
    elif is_saudi_dominant:
        result["classification"] = "SAUDI DOMINANT"
        result["primary_bet"] = {
            "market": "Home Win to Nil",
            "confidence": 7.5,
            "probability": home_win,
            "reason": "Saudi league structural dominance"
        }
    
    # ========================================================================
    # FINALIZE - SKIP if no bet or confidence too low
    # ========================================================================
    if result["primary_bet"]:
        # Additional confidence filter: skip if confidence < 7.0 except for LOW-SCORING
        if result["primary_bet"]["confidence"] < 7.0 and result["classification"] != "LOW-SCORING":
            result["verdict"] = "SKIP"
            result["skip_reasons"].append(f"Confidence too low ({result['primary_bet']['confidence']}/10) - skipping marginal edge")
            result["primary_bet"] = None
            result["classification"] = "SKIP"
        else:
            result["verdict"] = "RECOMMENDED"
    else:
        result["verdict"] = "SKIP"
        if not result.get("skip_reasons"):
            result["skip_reasons"].append("No structural pattern matched with sufficient confidence")
        result["classification"] = "SKIP"
    
    # Warnings (keep useful ones)
    if draw_pct >= 28:
        result["warnings"].append(f"High draw probability ({draw_pct:.1f}%) — avoid straight match result bets")
    if not is_top_league and league:
        result["warnings"].append(f"'{league}' is not a top league — lower reliability")
    
    return result


# ============================================================================
# TRUTH-BASED EVALUATION ENGINE (same as before, with Under 2.5 fix)
# ============================================================================
def evaluate_bet(primary_pred: str, home_goals, away_goals) -> dict:
    try:
        home = int(home_goals) if home_goals is not None else 0
        away = int(away_goals) if away_goals is not None else 0
    except (ValueError, TypeError):
        return {"is_correct": False, "actual": "INVALID DATA", "message": "Non-numeric score"}
    
    total = home + away
    btts = home > 0 and away > 0
    over25 = total > 2
    over35 = total > 3
    
    if home > away:
        winner = "HOME"
    elif away > home:
        winner = "AWAY"
    else:
        winner = "DRAW"
    
    pred = primary_pred.strip()
    is_correct = False
    
    if pred == "BTTS":
        is_correct = btts
    elif pred == "Over 2.5 Goals":
        is_correct = over25
    elif pred == "Under 2.5 Goals":
        is_correct = not over25  # Fixed: 0-0 is a win
    elif pred == "Over 3.5 Goals":
        is_correct = over35
    elif pred == "Under 3.5 Goals":
        is_correct = total <= 3
    elif pred == "Home Win":
        is_correct = winner == "HOME"
    elif pred == "Away Win":
        is_correct = winner == "AWAY"
    elif pred == "Home Win to Nil":
        is_correct = (winner == "HOME" and away == 0)
    elif pred == "Away Win to Nil":
        is_correct = (winner == "AWAY" and home == 0)
    elif "Away Win or Draw" in pred:
        is_correct = winner in ["AWAY", "DRAW"]
    elif "Home Win or Draw" in pred:
        is_correct = winner in ["HOME", "DRAW"]
    elif pred == "Home Over 1.5 Goals":
        is_correct = home >= 2
    elif pred == "Away Over 1.5 Goals":
        is_correct = away >= 2
    else:
        return {"is_correct": False, "actual": f"{home}-{away}", "message": f"Unknown market: {pred}"}
    
    return {
        "is_correct": is_correct,
        "actual": f"{home}-{away}",
        "message": f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}: {pred} vs {home}-{away}"
    }


# ============================================================================
# SUPABASE OPERATIONS (keep existing)
# ============================================================================
def save_to_db(data: dict, analysis: dict):
    # ... (keep your existing save_to_db function)
    pass

def get_pending():
    # ... (keep existing)
    pass

def submit_result(analysis_id, home_goals, away_goals):
    # ... (keep existing)
    pass

def get_results():
    # ... (keep existing)
    pass


# ============================================================================
# MAIN APP (keep structure, update version to 4.0)
# ============================================================================
def main():
    st.title("📊 Match Analyzer V4.0")
    st.caption("Profit-Driven Engine | LOW-SCORING Priority | High Thresholds | No Garbage Categories")
    
    # ... (rest of your tabs and UI - same structure as before)
    # For brevity, I'm showing only the key changes. Use your existing UI code.

if __name__ == "__main__":
    main()
