"""
STREAK PREDICTOR V3 - Six Checks System
Complete Engine | Audited Logic | Supabase Tracked
"""

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import date
from supabase import create_client, Client
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
st.set_page_config(page_title="Streak Predictor V3", page_icon="⚽", layout="centered")

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .tier-1 { border-left: 5px solid #10b981; }
    .tier-2 { border-left: 5px solid #fbbf24; }
    .team-header { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 12px; padding: 0.75rem; margin: 0.5rem 0; color: #ffffff; }
    .team-name { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .metric-label { color: #0f172a; font-weight: 700; font-size: 0.85rem; margin-top: 0.5rem; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .record-badge { background: #0f172a; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.8rem; color: #10b981; font-weight: 700; }
    .info-note { background: #1a3a5f; border-left: 4px solid #3b82f6; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .warning-note { background: #7f1a1a; border-left: 4px solid #ef4444; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .check-card { background: #1e293b; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; color: #ffffff; }
    .check-triggered { border: 2px solid #10b981; }
    .check-not-triggered { border: 2px solid #334155; opacity: 0.6; }
    .conf-high { color: #10b981; font-weight: 700; }
    .conf-medium { color: #fbbf24; font-weight: 700; }
    .conf-low { color: #f97316; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class MatchData:
    home_team: str
    away_team: str
    
    # Home team data
    home_last_6_wins: int
    home_last_6_draws: int
    home_last_6_losses: int
    home_last_6_goals_scored: float
    home_last_6_goals_conceded: float
    home_consecutive_wins: Optional[int]
    home_consecutive_winless: Optional[int]
    home_overall_winless: Optional[int]
    home_overall_unbeaten: Optional[int]
    home_clean_sheet_pct: Optional[float]
    
    # Away team data
    away_last_6_wins: int
    away_last_6_draws: int
    away_last_6_losses: int
    away_last_6_goals_scored: float
    away_last_6_goals_conceded: float
    away_consecutive_winless: Optional[int]
    away_consecutive_losses: Optional[int]
    away_overall_winless: Optional[int]
    away_overall_unbeaten: Optional[int]
    away_clean_sheet_pct: Optional[float]
    
    # H2H
    h2h_home_wins_last_5: Optional[int]
    h2h_away_wins_last_5: Optional[int]
    
    # Reverse fixture
    reverse_margin: Optional[int]
    reverse_winner: Optional[str]
    reverse_possession_winner: Optional[str]
    reverse_shots_winner: Optional[str]
    reverse_dangerous_attacks_winner: Optional[str]
    reverse_all_dominated: Optional[bool]
    
    # Model
    model_home_pct: Optional[float]
    model_away_pct: Optional[float]
    model_draw_pct: Optional[float]

@dataclass
class CheckResult:
    check_id: str
    check_name: str
    triggered: bool
    direction: Optional[str]
    reasoning: str

@dataclass
class BetDecision:
    bet_type: str
    team: Optional[str]
    confidence: str
    home_triggers: int
    away_triggers: int
    net_checks: int
    model_agreement: str
    checks: List[CheckResult]

# ============================================================================
# SIX CHECKS ENGINE
# ============================================================================
def run_check_1(data: MatchData) -> CheckResult:
    """WINLESS vs STREAKING"""
    # Condition A: Home winless, Away streaking
    if (data.home_overall_winless is not None and data.home_overall_winless >= 5) or \
       (data.home_consecutive_winless is not None and data.home_consecutive_winless >= 5):
        if data.away_overall_unbeaten is not None and data.away_overall_unbeaten >= 5:
            return CheckResult("C1", "Winless vs Streaking", True, "AWAY",
                             f"Home winless {data.home_overall_winless or data.home_consecutive_winless}+, Away unbeaten {data.away_overall_unbeaten}+")
    
    # Condition B: Away winless, Home streaking
    if (data.away_overall_winless is not None and data.away_overall_winless >= 5) or \
       (data.away_consecutive_winless is not None and data.away_consecutive_winless >= 5):
        if data.home_overall_unbeaten is not None and data.home_overall_unbeaten >= 5:
            return CheckResult("C1", "Winless vs Streaking", True, "HOME",
                             f"Away winless {data.away_overall_winless or data.away_consecutive_winless}+, Home unbeaten {data.home_overall_unbeaten}+")
    
    return CheckResult("C1", "Winless vs Streaking", False, None, "No extreme streak mismatch")

def run_check_2(data: MatchData) -> CheckResult:
    """FORTRESS vs TRAVELERS"""
    if data.home_consecutive_wins is not None and data.home_consecutive_wins >= 3:
        if data.away_consecutive_winless is not None and data.away_consecutive_winless >= 5:
            return CheckResult("C2", "Fortress vs Travelers", True, "HOME",
                             f"Home won {data.home_consecutive_wins} straight at home, Away winless {data.away_consecutive_winless} straight away")
    
    return CheckResult("C2", "Fortress vs Travelers", False, None, "No fortress/travelers mismatch")

def run_check_3(data: MatchData) -> CheckResult:
    """REVERSE FIXTURE DOMINATION"""
    if data.reverse_margin is None or data.reverse_winner is None:
        return CheckResult("C3", "Reverse Domination", False, None, "No reverse fixture data")
    
    if data.reverse_margin >= 2:
        if data.reverse_possession_winner == data.reverse_winner and \
           data.reverse_shots_winner == data.reverse_winner and \
           data.reverse_dangerous_attacks_winner == data.reverse_winner:
            return CheckResult("C3", "Reverse Domination", True, data.reverse_winner,
                             f"Won by {data.reverse_margin} goals, dominated all three metrics")
    
    return CheckResult("C3", "Reverse Domination", False, None, 
                      f"Margin {data.reverse_margin} or incomplete domination")

def run_check_4(data: MatchData) -> CheckResult:
    """DEFENSIVE COLLAPSE vs HOT ATTACK"""
    # Condition A: Home leaking, Away scoring
    if data.home_last_6_goals_conceded >= 2.0 and data.away_last_6_goals_scored >= 2.0:
        return CheckResult("C4", "Defensive Collapse vs Hot Attack", True, "AWAY",
                         f"Home conceding {data.home_last_6_goals_conceded}/game, Away scoring {data.away_last_6_goals_scored}/game")
    
    # Condition B: Away leaking, Home scoring
    if data.away_last_6_goals_conceded >= 2.0 and data.home_last_6_goals_scored >= 2.0:
        return CheckResult("C4", "Defensive Collapse vs Hot Attack", True, "HOME",
                         f"Away conceding {data.away_last_6_goals_conceded}/game, Home scoring {data.home_last_6_goals_scored}/game")
    
    return CheckResult("C4", "Defensive Collapse vs Hot Attack", False, None, 
                      "No 2+ conceded vs 2+ scored mismatch")

def run_check_5(data: MatchData) -> CheckResult:
    """H2H ONE-WAY TRAFFIC"""
    if data.h2h_home_wins_last_5 is not None and data.h2h_home_wins_last_5 >= 4:
        return CheckResult("C5", "H2H One-Way Traffic", True, "HOME",
                         f"Home won {data.h2h_home_wins_last_5} of last 5 at this venue")
    
    if data.h2h_away_wins_last_5 is not None and data.h2h_away_wins_last_5 >= 4:
        return CheckResult("C5", "H2H One-Way Traffic", True, "AWAY",
                         f"Away won {data.h2h_away_wins_last_5} of last 5 at this venue")
    
    return CheckResult("C5", "H2H One-Way Traffic", False, None, "No 4+ H2H dominance at venue")

def run_check_6(data: MatchData) -> CheckResult:
    """GOAL DROUGHT vs CLEAN SHEETS - DORMANT"""
    # Condition A: Home can't score, Away keeps clean sheets
    if data.home_last_6_goals_scored <= 0.5 and data.away_clean_sheet_pct is not None and data.away_clean_sheet_pct >= 40:
        return CheckResult("C6", "Goal Drought vs Clean Sheets", True, "AWAY",
                         f"Home scoring {data.home_last_6_goals_scored}/game, Away CS {data.away_clean_sheet_pct}%")
    
    # Condition B: Away can't score, Home keeps clean sheets
    if data.away_last_6_goals_scored <= 0.5 and data.home_clean_sheet_pct is not None and data.home_clean_sheet_pct >= 40:
        return CheckResult("C6", "Goal Drought vs Clean Sheets", True, "HOME",
                         f"Away scoring {data.away_last_6_goals_scored}/game, Home CS {data.home_clean_sheet_pct}%")
    
    return CheckResult("C6", "Goal Drought vs Clean Sheets", False, None, "Dormant - thresholds rarely met")

def decide_bet(home_triggers: int, away_triggers: int, checks: List[CheckResult]) -> BetDecision:
    """Decision function based on net checks"""
    net_checks = abs(home_triggers - away_triggers)
    
    if home_triggers > away_triggers:
        direction = "HOME"
    elif away_triggers > home_triggers:
        direction = "AWAY"
    else:
        return BetDecision("SKIP", None, "NONE", home_triggers, away_triggers, 0, "N/A", checks)
    
    if net_checks >= 3:
        bet_type = "WIN"
        confidence = "HIGH"
    elif net_checks == 2:
        bet_type = "DNB"
        confidence = "MEDIUM"
    else:
        bet_type = "DOUBLE_CHANCE"
        confidence = "LOW"
    
    return BetDecision(bet_type, direction, confidence, home_triggers, away_triggers, net_checks, "PENDING", checks)

def compare_to_model(decision: BetDecision, data: MatchData) -> str:
    """Firewalled model comparison"""
    if decision.bet_type == "SKIP":
        return "N/A"
    
    if decision.team == "HOME" and data.model_home_pct and data.model_home_pct >= 60:
        return "STRONG_ALIGN"
    elif decision.team == "AWAY" and data.model_away_pct and data.model_away_pct >= 60:
        return "STRONG_ALIGN"
    elif decision.team == "HOME" and data.model_draw_pct and data.model_draw_pct >= 35:
        return "DISAGREE"
    elif decision.team == "AWAY" and data.model_home_pct and data.model_home_pct >= 40:
        return "STRONG_DISAGREE"
    else:
        return "ALIGN"

def run_engine(data: MatchData) -> BetDecision:
    """Complete pipeline"""
    # Run all six checks
    checks = [
        run_check_1(data),
        run_check_2(data),
        run_check_3(data),
        run_check_4(data),
        run_check_5(data),
        run_check_6(data),
    ]
    
    # Tally triggers
    home_triggers = sum(1 for c in checks if c.triggered and c.direction == "HOME")
    away_triggers = sum(1 for c in checks if c.triggered and c.direction == "AWAY")
    
    # Decide bet
    decision = decide_bet(home_triggers, away_triggers, checks)
    
    # Model comparison (firewalled)
    decision.model_agreement = compare_to_model(decision, data)
    
    return decision

# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_match_to_db(data: MatchData, league: str, match_date: date, decision: BetDecision):
    try:
        match_record = {
            "home_team": data.home_team,
            "away_team": data.away_team,
            "league": league,
            "match_date": str(match_date),
            # Save all input data as JSON for reproducibility
            "input_data": json.dumps({
                "home": {
                    "last_6": {"wins": data.home_last_6_wins, "draws": data.home_last_6_draws, 
                              "losses": data.home_last_6_losses, "goals_scored": data.home_last_6_goals_scored,
                              "goals_conceded": data.home_last_6_goals_conceded},
                    "streaks": {"consecutive_wins": data.home_consecutive_wins, 
                               "consecutive_winless": data.home_consecutive_winless,
                               "overall_winless": data.home_overall_winless,
                               "overall_unbeaten": data.home_overall_unbeaten},
                    "clean_sheet_pct": data.home_clean_sheet_pct
                },
                "away": {
                    "last_6": {"wins": data.away_last_6_wins, "draws": data.away_last_6_draws,
                              "losses": data.away_last_6_losses, "goals_scored": data.away_last_6_goals_scored,
                              "goals_conceded": data.away_last_6_goals_conceded},
                    "streaks": {"consecutive_winless": data.away_consecutive_winless,
                               "consecutive_losses": data.away_consecutive_losses,
                               "overall_winless": data.away_overall_winless,
                               "overall_unbeaten": data.away_overall_unbeaten},
                    "clean_sheet_pct": data.away_clean_sheet_pct
                },
                "h2h": {"home_wins_last_5": data.h2h_home_wins_last_5, 
                       "away_wins_last_5": data.h2h_away_wins_last_5},
                "reverse": {"margin": data.reverse_margin, "winner": data.reverse_winner,
                           "possession_winner": data.reverse_possession_winner,
                           "shots_winner": data.reverse_shots_winner,
                           "dangerous_attacks_winner": data.reverse_dangerous_attacks_winner},
                "model": {"home_pct": data.model_home_pct, "away_pct": data.model_away_pct, 
                         "draw_pct": data.model_draw_pct}
            }),
            "decision": json.dumps({
                "bet_type": decision.bet_type,
                "team": decision.team,
                "confidence": decision.confidence,
                "home_triggers": decision.home_triggers,
                "away_triggers": decision.away_triggers,
                "net_checks": decision.net_checks,
                "model_agreement": decision.model_agreement,
                "checks": [{"id": c.check_id, "name": c.check_name, "triggered": c.triggered, 
                           "direction": c.direction, "reasoning": c.reasoning} for c in decision.checks]
            }),
        }
        response = supabase.table("matches").insert(match_record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending_matches():
    try:
        response = supabase.table("matches").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []

def submit_result(match_id, home_score, away_score):
    try:
        match = supabase.table("matches").select("*").eq("id", match_id).single().execute()
        match_data = match.data
        
        decision_data = match_data.get("decision")
        if isinstance(decision_data, str):
            decision_data = json.loads(decision_data)
        
        if decision_data:
            bet_type = decision_data.get("bet_type", "SKIP")
            team = decision_data.get("team")
            confidence = decision_data.get("confidence")
            
            if bet_type != "SKIP" and team:
                # Evaluate bet outcome
                won = evaluate_bet(bet_type, team, home_score, away_score)
                
                # Record result per check
                checks = decision_data.get("checks", [])
                for check in checks:
                    if check.get("triggered"):
                        check_id = check.get("id")
                        check_direction = check.get("direction")
                        # A check is "correct" if the match result aligns with its direction
                        check_correct = None
                        if check_direction == "HOME" and home_score > away_score:
                            check_correct = True
                        elif check_direction == "AWAY" and away_score > home_score:
                            check_correct = True
                        elif home_score == away_score:
                            check_correct = None  # Draw = push for directional bets
                        else:
                            check_correct = False
                        
                        if check_correct is not None:
                            supabase.table("check_results").insert({
                                "match_id": match_id,
                                "check_id": check_id,
                                "direction": check_direction,
                                "correct": check_correct,
                                "actual_result": f"{home_score}-{away_score}"
                            }).execute()
        
        supabase.table("matches").update({
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "result_entered": True
        }).eq("id", match_id).execute()
        
        return True
    except Exception as e:
        st.error(f"Failed: {e}")
        return False

def evaluate_bet(bet_type, team, home_score, away_score):
    """Evaluate if the bet won"""
    if bet_type == "WIN":
        if team == "HOME":
            return home_score > away_score
        else:
            return away_score > home_score
    elif bet_type == "DNB":
        if team == "HOME":
            return home_score >= away_score  # Push on draw
        else:
            return away_score >= home_score
    elif bet_type == "DOUBLE_CHANCE":
        if team == "HOME":
            return home_score >= away_score  # Home win or draw
        else:
            return away_score >= home_score  # Away win or draw
    return False

# ============================================================================
# UI INPUT
# ============================================================================
def match_input() -> MatchData:
    st.markdown("### 📋 Team Names & Context")
    c1, c2 = st.columns(2)
    with c1:
        home_name = st.text_input("🏠 Home Team", "Home", key="home_name")
    with c2:
        away_name = st.text_input("✈️ Away Team", "Away", key="away_name")
    
    league = st.text_input("🏆 League", "League", key="league")
    match_date = st.date_input("📅 Match Date", date.today(), key="match_date")
    
    st.divider()
    
    # HOME TEAM DATA
    st.markdown(f"<div class='team-header'><span class='team-name'>🏠 {home_name}</span></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">📊 Last 6 Matches</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        h_l6_w = st.number_input("Wins", 0, 6, 0, key="h_l6_w")
    with c2:
        h_l6_d = st.number_input("Draws", 0, 6, 0, key="h_l6_d")
    with c3:
        h_l6_l = st.number_input("Losses", 0, 6, 0, key="h_l6_l")
    
    c1, c2 = st.columns(2)
    with c1:
        h_l6_gs = st.number_input("Goals Scored/Game", 0.0, 6.0, 1.0, 0.1, key="h_l6_gs")
    with c2:
        h_l6_gc = st.number_input("Goals Conceded/Game", 0.0, 6.0, 1.0, 0.1, key="h_l6_gc")
    
    st.markdown('<p class="metric-label">🔄 Streaks</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        h_cons_w = st.number_input("Home Consecutive Wins", 0, 20, 0, key="h_cons_w")
        if h_cons_w == 0: h_cons_w = None
    with c2:
        h_cons_wl = st.number_input("Home Consecutive Winless", 0, 20, 0, key="h_cons_wl")
        if h_cons_wl == 0: h_cons_wl = None
    
    c1, c2 = st.columns(2)
    with c1:
        h_ov_wl = st.number_input("Overall Winless Streak", 0, 20, 0, key="h_ov_wl")
        if h_ov_wl == 0: h_ov_wl = None
    with c2:
        h_ov_ub = st.number_input("Overall Unbeaten Streak", 0, 20, 0, key="h_ov_ub")
        if h_ov_ub == 0: h_ov_ub = None
    
    h_cs_pct = st.number_input("Home Clean Sheet %", 0.0, 100.0, 0.0, 5.0, key="h_cs_pct")
    if h_cs_pct == 0.0: h_cs_pct = None
    
    st.divider()
    
    # AWAY TEAM DATA
    st.markdown(f"<div class='team-header'><span class='team-name'>✈️ {away_name}</span></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">📊 Last 6 Matches</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        a_l6_w = st.number_input("Wins", 0, 6, 0, key="a_l6_w")
    with c2:
        a_l6_d = st.number_input("Draws", 0, 6, 0, key="a_l6_d")
    with c3:
        a_l6_l = st.number_input("Losses", 0, 6, 0, key="a_l6_l")
    
    c1, c2 = st.columns(2)
    with c1:
        a_l6_gs = st.number_input("Goals Scored/Game", 0.0, 6.0, 1.0, 0.1, key="a_l6_gs")
    with c2:
        a_l6_gc = st.number_input("Goals Conceded/Game", 0.0, 6.0, 1.0, 0.1, key="a_l6_gc")
    
    st.markdown('<p class="metric-label">🔄 Streaks</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        a_cons_wl = st.number_input("Away Consecutive Winless", 0, 20, 0, key="a_cons_wl")
        if a_cons_wl == 0: a_cons_wl = None
    with c2:
        a_cons_l = st.number_input("Away Consecutive Losses", 0, 20, 0, key="a_cons_l")
        if a_cons_l == 0: a_cons_l = None
    
    c1, c2 = st.columns(2)
    with c1:
        a_ov_wl = st.number_input("Overall Winless Streak", 0, 20, 0, key="a_ov_wl")
        if a_ov_wl == 0: a_ov_wl = None
    with c2:
        a_ov_ub = st.number_input("Overall Unbeaten Streak", 0, 20, 0, key="a_ov_ub")
        if a_ov_ub == 0: a_ov_ub = None
    
    a_cs_pct = st.number_input("Away Clean Sheet %", 0.0, 100.0, 0.0, 5.0, key="a_cs_pct")
    if a_cs_pct == 0.0: a_cs_pct = None
    
    st.divider()
    
    # H2H DATA
    st.markdown(f"<div class='team-header'><span class='team-name'>🤝 H2H at Venue (Last 5)</span></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        h2h_home = st.number_input(f"{home_name} Wins", 0, 5, 0, key="h2h_home")
        if h2h_home == 0: h2h_home = None
    with c2:
        h2h_away = st.number_input(f"{away_name} Wins", 0, 5, 0, key="h2h_away")
        if h2h_away == 0: h2h_away = None
    
    st.divider()
    
    # REVERSE FIXTURE
    st.markdown(f"<div class='team-header'><span class='team-name'>🔄 Reverse Fixture</span></div>", unsafe_allow_html=True)
    rev_margin = st.number_input("Score Margin", 0, 10, 0, key="rev_margin")
    rev_winner = st.selectbox("Winner", ["None", home_name, away_name, "Draw"], key="rev_winner")
    if rev_winner == "None": 
        rev_winner = None
        rev_margin = None
    
    c1, c2, c3 = st.columns(3)
    with c1:
        rev_poss = st.selectbox("Possession Winner", ["None", home_name, away_name], key="rev_poss")
        if rev_poss == "None": rev_poss = None
    with c2:
        rev_shots = st.selectbox("Shots Winner", ["None", home_name, away_name], key="rev_shots")
        if rev_shots == "None": rev_shots = None
    with c3:
        rev_attacks = st.selectbox("Dangerous Attacks Winner", ["None", home_name, away_name], key="rev_attacks")
        if rev_attacks == "None": rev_attacks = None
    
    st.divider()
    
    # MODEL DATA
    st.markdown(f"<div class='team-header'><span class='team-name'>🤖 Model Probabilities</span></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        model_home = st.number_input("Home Win %", 0.0, 100.0, 0.0, 5.0, key="model_home")
        if model_home == 0.0: model_home = None
    with c2:
        model_away = st.number_input("Away Win %", 0.0, 100.0, 0.0, 5.0, key="model_away")
        if model_away == 0.0: model_away = None
    with c3:
        model_draw = st.number_input("Draw %", 0.0, 100.0, 0.0, 5.0, key="model_draw")
        if model_draw == 0.0: model_draw = None
    
    return MatchData(
        home_team=home_name, away_team=away_name,
        home_last_6_wins=h_l6_w, home_last_6_draws=h_l6_d, home_last_6_losses=h_l6_l,
        home_last_6_goals_scored=h_l6_gs, home_last_6_goals_conceded=h_l6_gc,
        home_consecutive_wins=h_cons_w, home_consecutive_winless=h_cons_wl,
        home_overall_winless=h_ov_wl, home_overall_unbeaten=h_ov_ub,
        home_clean_sheet_pct=h_cs_pct,
        away_last_6_wins=a_l6_w, away_last_6_draws=a_l6_d, away_last_6_losses=a_l6_l,
        away_last_6_goals_scored=a_l6_gs, away_last_6_goals_conceded=a_l6_gc,
        away_consecutive_winless=a_cons_wl, away_consecutive_losses=a_cons_l,
        away_overall_winless=a_ov_wl, away_overall_unbeaten=a_ov_ub,
        away_clean_sheet_pct=a_cs_pct,
        h2h_home_wins_last_5=h2h_home, h2h_away_wins_last_5=h2h_away,
        reverse_margin=rev_margin, reverse_winner=rev_winner,
        reverse_possession_winner=rev_poss, reverse_shots_winner=rev_shots,
        reverse_dangerous_attacks_winner=rev_attacks, reverse_all_dominated=None,
        model_home_pct=model_home, model_away_pct=model_away, model_draw_pct=model_draw
    )

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("Six Checks System | Audited Logic | Directional Betting")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        match_data = match_input()
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            decision = run_engine(match_data)
            
            # Check for duplicate
            existing = supabase.table("matches").select("id").eq(
                "home_team", match_data.home_team
            ).eq("away_team", match_data.away_team).eq(
                "match_date", str(st.session_state.get("match_date", date.today()))
            ).eq("result_entered", False).execute()
            
            if existing.data and len(existing.data) > 0:
                st.warning(f"⚠️ This analysis already exists. Submit result in Post-Match tab.")
            else:
                match_id = save_match_to_db(match_data, st.session_state.get("league", "Unknown"), 
                                           st.session_state.get("match_date", date.today()), decision)
                if match_id:
                    st.success(f"✅ Analysis saved")
            
            # Display checks
            st.markdown("### 🔍 Six Checks Results")
            triggered_checks = [c for c in decision.checks if c.triggered]
            
            for check in decision.checks:
                if check.triggered:
                    card_class = "check-triggered"
                    status = f"✅ TRIGGERED → {check.direction}"
                else:
                    card_class = "check-not-triggered"
                    status = "⏭️ Not triggered"
                
                st.markdown(f"""
                <div class="check-card {card_class}">
                    <div style="display:flex;justify-content:space-between;">
                        <div><strong>{check.check_id}: {check.check_name}</strong></div>
                        <div>{status}</div>
                    </div>
                    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{check.reasoning}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display decision
            st.markdown("### 🎯 Bet Decision")
            
            conf_class = f"conf-{decision.confidence.lower()}"
            
            if decision.bet_type == "SKIP":
                st.info("🎯 SKIP — No directional edge found. Checks conflict or don't trigger.")
            else:
                st.markdown(f"""
                <div class="output-card tier-1">
                    <div style="display:flex;justify-content:space-between;">
                        <div><strong>{decision.bet_type}</strong> → {decision.team}</div>
                        <span class="{conf_class}">{decision.confidence} CONFIDENCE</span>
                    </div>
                    <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">
                        Triggers: {decision.home_triggers} HOME, {decision.away_triggers} AWAY | Net: {decision.net_checks}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Model comparison
            if decision.bet_type != "SKIP":
                st.markdown(f"""
                <div class="info-note">
                <strong>Model Agreement:</strong> {decision.model_agreement}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_matches()
        if pending:
            match_options = {}
            for m in pending:
                label = f"{m['home_team']} vs {m['away_team']} ({m.get('match_date', '')})"
                match_options[label] = m['id']
            
            selected = st.selectbox("Select Match", list(match_options.keys()))
            c1, c2 = st.columns(2)
            with c1: home_score = st.number_input("Home Score", 0, 20, 0)
            with c2: away_score = st.number_input("Away Score", 0, 20, 0)
            
            if st.button("✅ Submit Result"):
                if submit_result(match_options[selected], home_score, away_score):
                    st.success("Result submitted!")
                    st.rerun()
        else:
            st.info("No pending matches.")
    
    with tab3:
        st.subheader("📊 Live Records by Check")
        try:
            # Aggregate results by check
            response = supabase.table("check_results").select("*").execute()
            if response.data:
                from collections import defaultdict
                check_stats = defaultdict(lambda: {"total": 0, "correct": 0, "incorrect": 0, "pushes": 0})
                
                for r in response.data:
                    check_id = r.get("check_id")
                    correct = r.get("correct")
                    if correct is True:
                        check_stats[check_id]["correct"] += 1
                        check_stats[check_id]["total"] += 1
                    elif correct is False:
                        check_stats[check_id]["incorrect"] += 1
                        check_stats[check_id]["total"] += 1
                    else:
                        check_stats[check_id]["pushes"] += 1
                
                check_names = {
                    "C1": "Winless vs Streaking",
                    "C2": "Fortress vs Travelers",
                    "C3": "Reverse Domination",
                    "C4": "Defensive Collapse vs Hot Attack",
                    "C5": "H2H One-Way Traffic",
                    "C6": "Goal Drought vs Clean Sheets"
                }
                
                for check_id in ["C1", "C2", "C3", "C4", "C5", "C6"]:
                    stats = check_stats.get(check_id, {"total": 0, "correct": 0, "incorrect": 0, "pushes": 0})
                    total = stats["total"]
                    correct = stats["correct"]
                    wr = (correct/total*100) if total > 0 else 0
                    color = "#10b981" if wr >= 80 else "#fbbf24" if wr >= 60 else "#f97316" if wr >= 40 else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                        <div><strong>{check_id}: {check_names.get(check_id, check_id)}</strong></div>
                        <div style="color:{color};">{correct}/{total} ({wr:.0f}%) | {stats['pushes']} pushes</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No results recorded yet.")
        except Exception as e:
            st.error(f"Database error: {e}")
    
    st.divider()
    st.markdown("""
    ### 📋 Six Checks System
    
    | Check | Logic | Record |
    |-------|-------|--------|
    | C1: Winless vs Streaking | Team in terminal decline vs team that can't lose | ~80% |
    | C2: Fortress vs Travelers | Home team with 3+ straight wins vs away team winless 5+ | ~100% |
    | C3: Reverse Domination | Won reverse by 2+ and dominated all stats | ~50% |
    | C4: Defensive Collapse | Team conceding 2+ vs team scoring 2+ | ~100% |
    | C5: H2H One-Way Traffic | 4+ wins in last 5 at this venue | ~100% |
    | C6: Goal Drought vs Clean Sheets | DORMANT — thresholds too tight | N/A |
    
    **Decision Logic:**
    - Net 3+ checks → WIN bet (HIGH confidence)
    - Net 2 checks → DNB (MEDIUM confidence)
    - Net 1 check → DOUBLE CHANCE (LOW confidence)
    - Net 0 / conflict → SKIP
    """)

if __name__ == "__main__":
    main()
