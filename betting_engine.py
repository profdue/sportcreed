"""
MATCH ANALYZER V8.0 — UNIVERSAL LOGIC ENGINE
Based on analysis of 81 matches across 7 leagues
- Draw predictions are wrong 83% of the time
- Form difference > 2 points → someone wins
- Goals sweet spot: 2.00-2.40 for draws
- Desperation kills draws
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
st.set_page_config(page_title="Match Analyzer V8.0", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .primary-card { border: 3px solid #10b981; background: linear-gradient(135deg, #0a2a0a 0%, #051505 100%); }
    .lock-card { border: 3px solid #f59e0b; background: linear-gradient(135deg, #2a1a00 0%, #1a0f00 100%); }
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
    .dead-rubber-warning { background: #7c2d12; color: #fed7aa; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.8rem; margin: 0.5rem 0; }
    .win-badge { background: #10b981; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .loss-badge { background: #ef4444; color: #fff; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .skip-badge { background: #fbbf24; color: #000; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; }
    .condition-true { color: #10b981; font-weight: 700; }
    .condition-false { color: #ef4444; font-weight: 700; }
    .condition-box { background: #0f172a; border-radius: 8px; padding: 0.75rem; margin: 0.25rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# V8.0 PARSER — UNIVERSAL LOGIC
# ============================================================================
def parse_match_data_v8(raw_text: str) -> dict:
    lines = raw_text.strip().split('\n')
    
    data = {
        "home_team": None, "away_team": None, "league": None,
        "home_goals_total": None, "away_goals_total": None,
        "home_form_points": None, "away_form_points": None,
        "home_standings_position": None, "away_standings_position": None,
        "home_standings_points": None, "away_standings_points": None,
        "home_losing_streak": 0, "away_losing_streak": 0,
        "is_relegation_fight": False,
        "is_title_race": False,
        "competitive_block_home": None,
        "competitive_block_away": None,
        "score_matrix": [],
        "home_win": None, "draw": None, "away_win": None,
        "btts": None, "over_25": None, "under_25": None, "over_35": None,
    }
    
    # ================================================================
    # STEP 1: Extract Team Names
    # ================================================================
    team_names = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Predicted Lineups':
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                parts = re.split(r'\d-\d-\d-\d|\d-\d-\d', next_line)
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 1 and not any(x in part.lower() for x in ['formation', 'predicted', 'lineups']):
                        if part not in team_names:
                            team_names.append(part)
                if len(team_names) >= 2:
                    break
            break
    
    if len(team_names) < 2:
        for line in lines[:10]:
            stripped = line.strip()
            if stripped and len(stripped) > 2:
                if not any(x in stripped.lower() for x in ['predicted', 'lineups', 'formation', 'statistical', 'comparison']):
                    if stripped not in team_names:
                        team_names.append(stripped)
    
    if len(team_names) >= 2:
        data["home_team"] = team_names[0]
        data["away_team"] = team_names[1]
    
    # ================================================================
    # STEP 2: Extract Form and Stats
    # ================================================================
    in_form_data = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if 'Form Data' in stripped or 'Recent Form' in stripped:
            in_form_data = True
            continue
        
        if in_form_data:
            # Goals
            if 'Goals' in stripped and 'Shots' not in stripped and 'Tackles' not in stripped:
                match_before = re.search(r'(\d+\.?\d*)\s*Goals', stripped)
                match_after = re.search(r'Goals\s+(\d+\.?\d*)', stripped)
                if match_before:
                    val = float(match_before.group(1))
                    if data["home_goals_total"] is None and data["home_team"] and data["home_team"].lower() in stripped.lower():
                        data["home_goals_total"] = val
                    elif data["away_goals_total"] is None and data["away_team"] and data["away_team"].lower() in stripped.lower():
                        data["away_goals_total"] = val
                    elif match_after and data["home_goals_total"] is None:
                        data["home_goals_total"] = val
                    elif match_after and data["away_goals_total"] is None:
                        data["away_goals_total"] = float(match_after.group(1))
            
            # Form points (last 5 matches: W=3, D=1, L=0)
            if 'Form' in stripped or 'Last 5' in stripped:
                # Look for W/D/L pattern
                wdl_pattern = r'[WDL]{5}'
                matches = re.findall(wdl_pattern, stripped.upper())
                if matches:
                    for m in matches:
                        points = sum(3 if c == 'W' else 1 if c == 'D' else 0 for c in m)
                        if data["home_form_points"] is None and data["home_team"] and data["home_team"].lower() in stripped.lower():
                            data["home_form_points"] = points
                        elif data["away_form_points"] is None and data["away_team"] and data["away_team"].lower() in stripped.lower():
                            data["away_form_points"] = points
            
            # Losing streak detection
            if 'losing' in stripped.lower() or 'lost' in stripped.lower():
                streak_match = re.search(r'(\d+)\s*(?:losing|loss|lost)', stripped.lower())
                if streak_match:
                    streak = int(streak_match.group(1))
                    if data["home_team"] and data["home_team"].lower() in stripped.lower():
                        data["home_losing_streak"] = max(data["home_losing_streak"], streak)
                    elif data["away_team"] and data["away_team"].lower() in stripped.lower():
                        data["away_losing_streak"] = max(data["away_losing_streak"], streak)
    
    # ================================================================
    # STEP 3: Extract Standings
    # ================================================================
    in_standings = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Standings' in stripped:
            in_standings = True
            continue
        if in_standings and 'Offers' in stripped:
            break
        if in_standings and data["home_team"] and data["home_team"] in stripped:
            parts = stripped.split()
            try:
                data["home_standings_position"] = int(parts[0])
                # Points are usually at the end
                for part in reversed(parts):
                    try:
                        data["home_standings_points"] = int(part)
                        break
                    except:
                        pass
            except: pass
        if in_standings and data["away_team"] and data["away_team"] in stripped:
            parts = stripped.split()
            try:
                data["away_standings_position"] = int(parts[0])
                for part in reversed(parts):
                    try:
                        data["away_standings_points"] = int(part)
                        break
                    except:
                        pass
            except: pass
    
    # ================================================================
    # STEP 4: Competitive Blocks
    # ================================================================
    # Determine block based on position (assuming typical league size)
    def get_block(position):
        if position is None:
            return None
        if position <= 4:  # Top 4 = Europe
            return "europe"
        elif position >= 15:  # Bottom 5-6 = Relegation
            return "relegation"
        else:  # Mid-table
            return "mid"
    
    data["competitive_block_home"] = get_block(data["home_standings_position"])
    data["competitive_block_away"] = get_block(data["away_standings_position"])
    
    # ================================================================
    # STEP 5: Relegation fight detection
    # ================================================================
    if data["home_standings_position"] and data["home_standings_position"] >= 15:
        data["is_relegation_fight"] = True
    if data["away_standings_position"] and data["away_standings_position"] >= 15:
        data["is_relegation_fight"] = True
    
    # ================================================================
    # STEP 6: Score Matrix
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'Score analysis':
            for j in range(i+1, min(i+15, len(lines))):
                m = re.match(r'(\d+)-(\d+)\s*@\s*(\d+\.?\d*)\s*%', lines[j].strip())
                if m:
                    data["score_matrix"].append({
                        "score": f"{m.group(1)}-{m.group(2)}",
                        "home_goals": int(m.group(1)),
                        "away_goals": int(m.group(2)),
                        "probability": float(m.group(3))
                    })
            break
    data["score_matrix"].sort(key=lambda x: x["probability"], reverse=True)
    data["score_matrix"] = data["score_matrix"][:10]
    
    # ================================================================
    # STEP 7: Probabilities
    # ================================================================
    for i, line in enumerate(lines):
        stripped = line.strip()
        if 'Result' in stripped:
            for j in range(i+1, min(i+15, len(lines))):
                pl = lines[j].strip()
                if data["home_team"] and data["home_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["home_win"] = float(m.group(1))
                if 'Draw' in pl and not data["home_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["draw"] = float(m.group(1))
                if data["away_team"] and data["away_team"] in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["away_win"] = float(m.group(1))
                if 'Both Teams to Score' in pl:
                    m = re.search(r'(\d+\.?\d*)%', pl)
                    if m: data["btts"] = float(m.group(1))
    
    return data


# ============================================================================
# V8.0 ANALYSIS ENGINE — UNIVERSAL LOGIC
# ============================================================================
def analyze_match_v8(data: dict) -> dict:
    result = {
        "primary_bet": None,
        "classification": None,
        "verdict": "SKIP",
        "skip_reasons": [],
        "is_lock": False,
        "draw_conditions": {},
        "winner_selection": None,
        "goal_bet": None,
    }
    
    # Extract all needed values
    home_form = data.get("home_form_points", 0) or 0
    away_form = data.get("away_form_points", 0) or 0
    home_goals = data.get("home_goals_total", 0) or 0
    away_goals = data.get("away_goals_total", 0) or 0
    home_losing_streak = data.get("home_losing_streak", 0) or 0
    away_losing_streak = data.get("away_losing_streak", 0) or 0
    home_block = data.get("competitive_block_home")
    away_block = data.get("competitive_block_away")
    draw_prediction = data.get("draw", 0) or 0
    
    combined_goals = home_goals + away_goals
    avg_goals = combined_goals  # Using combined as proxy for avg (assuming 1 match)
    form_diff = abs(home_form - away_form)
    
    # ================================================================
    # TRUTH #1: Check if draw is predicted
    # ================================================================
    is_draw_predicted = draw_prediction > 0
    
    # ================================================================
    # TRUTH #5: Same competitive block
    # ================================================================
    same_block = home_block is not None and away_block is not None and home_block == away_block
    
    # ================================================================
    # TRUTH #6: Desperation kills draws
    # ================================================================
    is_home_desperate = home_losing_streak >= 3 or data.get("is_relegation_fight", False)
    is_away_desperate = away_losing_streak >= 3 or data.get("is_relegation_fight", False)
    is_any_desperate = is_home_desperate or is_away_desperate
    
    # ================================================================
    # TRUTH #3: Form difference check
    # ================================================================
    form_similar = form_diff <= 2
    
    # ================================================================
    # TRUTH #4: Goals sweet spot
    # ================================================================
    goals_in_sweet_spot = 2.00 <= avg_goals <= 2.40
    
    # ================================================================
    # DRAW CONDITIONS — ALL 4 must be TRUE
    # ================================================================
    draw_conditions = {
        "form_similar": form_similar,
        "goals_sweet_spot": goals_in_sweet_spot,
        "same_block": same_block,
        "no_desperation": not is_any_desperate,
    }
    result["draw_conditions"] = draw_conditions
    
    all_draw_conditions_met = all(draw_conditions.values())
    
    # ================================================================
    # Determine Winner Selection
    # ================================================================
    winner_selection = "HOME"
    
    # STEP 2: Determine WHICH side wins
    # └── Is Home Team in better form? (≥3 points difference)
    if home_form - away_form >= 3:
        winner_selection = "HOME"
        result["winner_reason"] = f"Home team better form: {home_form} vs {away_form} points"
    # └── Is Home Team desperate?
    elif is_home_desperate and not is_away_desperate:
        winner_selection = "HOME"
        result["winner_reason"] = "Home team desperate"
    # └── Is Away Team in better form AND desperate?
    elif away_form - home_form >= 3 and is_away_desperate:
        winner_selection = "AWAY"
        result["winner_reason"] = f"Away team better form ({away_form} vs {home_form}) and desperate"
    # └── Default (when nothing is clear) → BET HOME WIN (65% of non-draws)
    else:
        winner_selection = "HOME"
        result["winner_reason"] = "Default - home team wins 65% of non-draws"
    
    result["winner_selection"] = winner_selection
    
    # ================================================================
    # TRUTH #2: When Draw Prediction Fails — Home Wins 65% of the Time
    # ================================================================
    non_draw_winner = "HOME"  # 65% of non-draws are home wins
    
    # ================================================================
    # GOAL BETS (TRUTH #7)
    # ================================================================
    goal_bet = None
    goal_accuracy = None
    
    # STEP 1: Check Avg Goals
    if avg_goals < 2.00:
        goal_bet = "UNDER 2.5"
        goal_accuracy = "95%"
        result["goal_reason"] = f"Avg goals {avg_goals:.2f} < 2.00 → BET UNDER 2.5"
    elif avg_goals > 3.00:
        if draw_prediction > 0:
            goal_bet = "OVER 2.5"
            goal_accuracy = "100%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} > 3.00 + Draw Prediction → BET OVER 2.5"
        elif winner_selection == "AWAY":
            goal_bet = "OVER 2.5"
            goal_accuracy = "80%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} > 3.00 + Away Win → BET OVER 2.5"
        else:
            goal_bet = "OVER 2.5"
            goal_accuracy = "62%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} > 3.00 + Home Win → BET OVER 2.5 (62%)"
    elif 2.00 <= avg_goals <= 2.40:
        if draw_prediction > 0:
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} in sweet spot + Draw → BET UNDER 2.5"
        elif winner_selection == "HOME":
            goal_bet = "UNDER 2.5"
            goal_accuracy = "80%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} in sweet spot + Home Win → BET UNDER 2.5"
        else:
            goal_bet = "SKIP"
            goal_accuracy = "50%"
            result["goal_reason"] = f"Avg goals {avg_goals:.2f} in sweet spot + Away Win → SKIP (50%)"
    
    result["goal_bet"] = goal_bet
    
    # ================================================================
    # FINAL DECISION
    # ================================================================
    if is_draw_predicted and all_draw_conditions_met:
        # BET THE DRAW
        result["primary_bet"] = {
            "market": "DRAW",
            "reason": "All 4 draw conditions met: Form similar, goals in sweet spot, same block, no desperation",
            "historical_accuracy": "57%",
        }
        result["classification"] = "DRAW"
        result["verdict"] = "RECOMMENDED"
    
    elif is_draw_predicted and not all_draw_conditions_met:
        # DOUBLE CHANCE (Form diff > 2, or different blocks, or desperation)
        if form_diff > 2:
            result["primary_bet"] = {
                "market": f"DOUBLE CHANCE: {winner_selection} or Draw",
                "reason": f"Form difference {form_diff} > 2 → Double Chance wins 83% when draw predicted",
                "historical_accuracy": "83%",
            }
        elif not same_block:
            result["primary_bet"] = {
                "market": f"DOUBLE CHANCE: {winner_selection} or Draw",
                "reason": f"Different competitive blocks ({home_block} vs {away_block}) → Double Chance",
                "historical_accuracy": "83%",
            }
        elif is_any_desperate:
            result["primary_bet"] = {
                "market": f"DOUBLE CHANCE: {winner_selection} or Draw",
                "reason": "Desperation present → Double Chance",
                "historical_accuracy": "83%",
            }
        else:
            result["primary_bet"] = {
                "market": f"DOUBLE CHANCE: {winner_selection} or Draw",
                "reason": "Draw conditions not fully met → Double Chance",
                "historical_accuracy": "83%",
            }
        result["classification"] = "DOUBLE CHANCE"
        result["verdict"] = "RECOMMENDED"
    
    elif not is_draw_predicted:
        # Exact winner based on form and desperation
        result["primary_bet"] = {
            "market": f"{winner_selection} WIN",
            "reason": result.get("winner_reason", "Default selection"),
            "historical_accuracy": "89%" if "better form" in result.get("winner_reason", "") else "65%",
        }
        result["classification"] = f"{winner_selection} WIN"
        result["verdict"] = "RECOMMENDED"
    
    # Add goal bet if available
    if goal_bet and goal_bet != "SKIP":
        if result["primary_bet"]:
            result["primary_bet"]["market"] += f" | {goal_bet}"
            result["primary_bet"]["reason"] += f"; {result.get('goal_reason', '')}"
        else:
            result["primary_bet"] = {
                "market": goal_bet,
                "reason": result.get("goal_reason", ""),
                "historical_accuracy": goal_accuracy,
            }
        result["classification"] += f" | {goal_bet}"
        result["verdict"] = "RECOMMENDED"
    
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
    is_correct = False
    
    # Parse prediction for multiple bets
    if ' | ' in pred:
        bets = pred.split(' | ')
    else:
        bets = [pred]
    
    correct_count = 0
    total_bets = 0
    
    for bet in bets:
        bet = bet.strip()
        total_bets += 1
        if 'DRAW' in bet:
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
        elif 'HOME WIN' in bet or 'HOME' in bet:
            if home > away:
                correct_count += 1
        elif 'AWAY WIN' in bet or 'AWAY' in bet:
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
def save_to_db(data: dict, analysis: dict):
    try:
        primary = analysis.get("primary_bet")
        record = {
            "home_team": data.get("home_team", "Unknown"),
            "away_team": data.get("away_team", "Unknown"),
            "match_date": str(date.today()),
            "league": data.get("league"),
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
            "prediction": primary["market"] if primary else "SKIP",
            "classification": analysis.get("classification", "SKIP"),
            "pattern": "DRAW" if "DRAW" in analysis.get("classification", "") else "DOUBLE_CHANCE" if "DOUBLE" in analysis.get("classification", "") else "WIN" if "WIN" in analysis.get("classification", "") else "SKIP",
            "verdict": analysis.get("verdict", "SKIP"),
            "draw_conditions": json.dumps(analysis.get("draw_conditions", {})),
            "winner_selection": analysis.get("winner_selection"),
            "goal_bet": analysis.get("goal_bet"),
            "score_matrix": json.dumps(data.get("score_matrix", [])),
            "home_win_pct": data.get("home_win"),
            "draw_pct": data.get("draw"),
            "away_win_pct": data.get("away_win"),
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
# MAIN APP
# ============================================================================
def main():
    st.title("📊 Match Analyzer V8.0")
    st.caption("Universal Logic Engine | Based on 81 matches across 7 leagues | Draw predictions wrong 83% of the time")
    
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
        Avg Goals < 2.00 → UNDER 2.5 (95% accuracy)
        """)
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ========================================================================
    # TAB 1: ANALYZE
    # ========================================================================
    with tab1:
        st.markdown("### 📋 Paste Match Data")
        raw_text = st.text_area("Match Data", height=400, key="raw_input")
        
        if st.button("🔮 ANALYZE V8.0", type="primary"):
            if not raw_text.strip():
                st.error("Please paste the match data.")
            else:
                with st.spinner("Analyzing with Universal Logic..."):
                    data = parse_match_data_v8(raw_text)
                
                if not data.get("home_team") or not data.get("away_team"):
                    st.error("Could not detect team names.")
                else:
                    analysis = analyze_match_v8(data)
                    save_to_db(data, analysis)
                    
                    # Display Draw Conditions if draw was predicted
                    if data.get("draw", 0) > 0:
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
                            st.info("⚠️ Not all conditions met → Consider Double Chance or Exact Winner")
                    
                    v = analysis["verdict"]
                    if v == "LOCK": st.success(f"🔒 LOCK: {data['home_team']} vs {data['away_team']}")
                    elif v == "RECOMMENDED": st.success(f"✅ RECOMMENDED: {data['home_team']} vs {data['away_team']}")
                    else: st.warning(f"⚠️ SKIP: {data['home_team']} vs {data['away_team']}")
                    
                    st.markdown(f"**Classification: {analysis['classification']}**")
                    
                    if analysis.get("winner_reason"):
                        st.markdown(f"**Winner Selection:** {analysis['winner_selection']} — {analysis['winner_reason']}")
                    
                    # Key Metrics
                    st.markdown("### 📊 Key Metrics")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{data.get("home_form_points", "?")} vs {data.get("away_form_points", "?")}</div><div class="metric-label">Form Points</div></div>', unsafe_allow_html=True)
                    with m2:
                        avg_goals = (data.get("home_goals_total") or 0) + (data.get("away_goals_total") or 0)
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_goals:.2f}</div><div class="metric-label">Avg Goals</div></div>', unsafe_allow_html=True)
                    with m3:
                        block_home = data.get("competitive_block_home", "?")
                        block_away = data.get("competitive_block_away", "?")
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{block_home} vs {block_away}</div><div class="metric-label">Competitive Block</div></div>', unsafe_allow_html=True)
                    with m4:
                        desperate = "Yes" if (data.get("home_losing_streak", 0) >= 3 or data.get("away_losing_streak", 0) >= 3 or data.get("is_relegation_fight", False)) else "No"
                        st.markdown(f'<div class="metric-card"><div class="metric-value">{desperate}</div><div class="metric-label">Desperation</div></div>', unsafe_allow_html=True)
                    
                    if data.get("score_matrix"):
                        st.markdown("### 🎯 Score Matrix (Top 5)")
                        score_cols = st.columns(5)
                        for idx, s in enumerate(data["score_matrix"][:5]):
                            with score_cols[idx]:
                                bg = "#1e293b" if s["home_goals"] != s["away_goals"] else "#2a1a00"
                                st.markdown(f'<div style="background:{bg}; border-radius:8px; padding:0.5rem; text-align:center; color:#fff;"><div style="font-size:1.2rem; font-weight:800;">{s["score"]}</div><div style="font-size:0.7rem; color:#94a3b8;">{s["probability"]:.1f}%</div></div>', unsafe_allow_html=True)
                    
                    if analysis.get("primary_bet"):
                        p = analysis["primary_bet"]
                        card_class = "primary-card"
                        badge = f'<span class="accuracy-badge">📊 {p.get("historical_accuracy", "N/A")}</span>'
                        st.markdown(f'<div class="section-label">🎯 PRIMARY BET</div><div class="output-card {card_class}"><div style="display:flex;align-items:center;gap:1rem;"><div style="font-size:2.5rem;">🔥</div><div style="flex:1;"><div style="font-size:1.3rem;font-weight:800;">{p["market"]}</div><div style="font-size:0.8rem;color:#64748b;">{p["reason"]}</div><div style="margin-top:0.5rem;">{badge}</div></div></div></div>', unsafe_allow_html=True)
                    
                    if analysis["verdict"] == "SKIP":
                        st.markdown(f'<div class="output-card skip-card"><div class="verdict-skip"><div class="big-text">⚠️ SKIP — NO BET</div><p style="color:#94a3b8;">{"<br>".join(analysis.get("skip_reasons", []))}</p></div></div>', unsafe_allow_html=True)
    
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
                
                if pat == "DRAW":
                    badge = "🎯 DRAW"
                elif pat == "DOUBLE_CHANCE":
                    badge = "🔄 DOUBLE CHANCE"
                elif pat == "WIN":
                    badge = "🏆 WIN"
                else:
                    badge = "⚠️ SKIP"
                
                with st.expander(f"{badge} | {ht} vs {at} — Predicted: {pred}"):
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
            
            for r in results:
                pred = r.get('prediction', '')
                if pred == 'SKIP':
                    continue
                
                # Parse prediction for evaluation
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                evaluation = evaluate_bet(primary_pred, r.get('actual_home_goals'), r.get('actual_away_goals'))
                
                if evaluation["is_correct"]:
                    correct += 1
                else:
                    incorrect += 1
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Tracked</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{bet_count}</div><div class="stat-label">Bets Placed</div></div>', unsafe_allow_html=True)
            with col3:
                win_rate = round(correct / bet_count * 100) if bet_count > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{skip_count}</div><div class="stat-label">Skipped</div></div>', unsafe_allow_html=True)
            
            st.markdown(f"**Overall: {correct} correct | {incorrect} incorrect**")
            
            # Results table
            rows = []
            for r in results:
                pred = r.get('prediction', '')
                pattern = r.get('pattern', '')
                actual_home = r.get('actual_home_goals')
                actual_away = r.get('actual_away_goals')
                primary_pred = pred.split(' | ')[0].strip() if ' | ' in pred else pred.strip()
                
                if pred == 'SKIP':
                    badge = '<span class="skip-badge">⚪ SKIP</span>'
                    score_display = "—"
                else:
                    evaluation = evaluate_bet(primary_pred, actual_home, actual_away)
                    badge = '<span class="win-badge">🟢 WIN</span>' if evaluation["is_correct"] else '<span class="loss-badge">🔴 LOSS</span>'
                    score_display = f"{actual_home}-{actual_away}" if actual_home is not None else "—"
                
                if pattern == "DRAW":
                    match_display = f"🎯 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif pattern == "DOUBLE_CHANCE":
                    match_display = f"🔄 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                elif pattern == "WIN":
                    match_display = f"🏆 {r.get('home_team', '')} vs {r.get('away_team', '')}"
                else:
                    match_display = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
                
                rows.append({
                    "Date": r.get("match_date", ""),
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
