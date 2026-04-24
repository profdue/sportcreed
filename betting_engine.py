"""
STREAK PREDICTOR - Complete App with Supabase Integration
15-Prediction System | Condition-Based | Auto-Learning
Post-Match Result Entry | Live Records Tracking | Duplicate Protection
"""

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
st.set_page_config(
    page_title="Streak Predictor",
    page_icon="⚽",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS STYLES
# ============================================================================
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .output-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        color: #ffffff;
    }
    .tier-1-card { border-left: 5px solid #10b981; }
    .tier-2-card { border-left: 5px solid #fbbf24; }
    .tier-3-card { border-left: 5px solid #f97316; }
    .avoid-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.4rem 0;
        border-left: 4px solid #64748b;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    .team-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .team-name { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .section-header {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 0.75rem 0;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
    }
    .metric-label {
        color: #0f172a;
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        border: none;
        width: 100%;
    }
    .warning-note {
        background: #7f1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.6rem;
        margin: 0.4rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #ffffff;
    }
    .info-note {
        background: #1a3a5f;
        border-left: 4px solid #3b82f6;
        padding: 0.6rem;
        margin: 0.4rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #ffffff;
    }
    .record-badge {
        background: #0f172a;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        color: #10b981;
        font-weight: 700;
    }
    .record-lost {
        background: #0f172a;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        color: #ef4444;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamData:
    name: str
    scored_05: float; scored_15: float; scored_25: float; scored_35: float
    conceded_05: float; conceded_15: float; conceded_25: float; conceded_35: float
    btts: float; over_15: float; over_25: float; over_35: float
    btts_and_over25: float; btts_no_over25: float
    fts_pct: float; cs_pct: float
    xg: float; actual_scored: float
    scored_drop: float = 0.0; concede_drop: float = 0.0
    attack_type: str = ""; defense_type: str = ""

@dataclass
class Bet:
    market: str; bet: str; tier: int; record: str; reasoning: str

@dataclass
class MatchResult:
    tier1_bets: List[Bet]; tier2_bets: List[Bet]; tier3_bets: List[Bet]
    avoid_bets: List[str]; warnings: List[str]; profile: str
    predictions_fired: List[str] = None
    
    def __post_init__(self):
        if self.predictions_fired is None:
            self.predictions_fired = []

# ============================================================================
# SAMPLE SIZE DETECTION
# ============================================================================
def detect_sample_size(home: TeamData, away: TeamData) -> int:
    percentages = []
    for team in [home, away]:
        for attr in ['scored_05','scored_15','scored_25','scored_35',
                     'conceded_05','conceded_15','conceded_25','conceded_35',
                     'btts','over_15','over_25','over_35','btts_and_over25',
                     'btts_no_over25','fts_pct','cs_pct']:
            val = getattr(team, attr, 0)
            if isinstance(val, (int, float)) and 0 < val < 100:
                percentages.append(val)
    
    for games in [34, 33, 30, 28, 22, 17, 15, 13, 11, 10, 8, 6, 5, 4]:
        match_count = 0
        for pct in percentages:
            closest = round(pct * games / 100)
            recon = round((closest / games) * 100)
            if abs(pct - recon) <= 1:
                match_count += 1
        if match_count >= len(percentages) * 0.8:
            return games
    return 15

def get_sample_weight(games: int) -> Tuple[float, int]:
    if games < 5: return 0.0, 5
    elif games < 8: return 0.5, 3
    elif games < 15: return 0.8, 1
    else: return 1.0, 1

# ============================================================================
# DERIVED METRICS
# ============================================================================
def classify_defense(concede_05, concede_15, concede_25, concede_35):
    drop = concede_05 - concede_15
    if concede_05 >= 85 and drop <= 30: return "EXTREME_COLLAPSE"
    elif concede_05 >= 70 and drop <= 30: return "COLLAPSE"
    elif concede_05 <= 40: return "ELITE"
    elif drop >= 45: return "BEND"
    else: return "MODERATE"

def classify_attack(scored_05, scored_15, scored_25, scored_35, fts):
    drop = scored_05 - scored_15
    if scored_15 == 0 and scored_25 == 0: return "NO_GOAL"
    if fts >= 65: return "NO_GOAL"
    elif fts >= 50: return "RARELY_SCORES"
    elif scored_05 >= 85 and drop <= 25: return "SCORES_FREELY"
    elif drop <= 25: return "MODERATE_SCORER"
    elif drop >= 45: return "ONE_GOAL"
    else: return "MODERATE"

def enrich_team(team: TeamData) -> TeamData:
    team.scored_drop = team.scored_05 - team.scored_15
    team.concede_drop = team.conceded_05 - team.conceded_15
    team.attack_type = classify_attack(team.scored_05, team.scored_15, 
                                        team.scored_25, team.scored_35, team.fts_pct)
    team.defense_type = classify_defense(team.conceded_05, team.conceded_15,
                                          team.conceded_25, team.conceded_35)
    return team

# ============================================================================
# CONDITION HELPERS
# ============================================================================
def is_collapse(team: TeamData) -> bool:
    return team.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"]

def is_extreme_collapse(team: TeamData) -> bool:
    return team.defense_type == "EXTREME_COLLAPSE"

def is_dead_attack(team: TeamData) -> bool:
    return team.attack_type == "NO_GOAL"

def is_one_goal(team: TeamData) -> bool:
    return team.attack_type == "ONE_GOAL"

def is_scores_freely(team: TeamData) -> bool:
    return team.attack_type in ["SCORES_FREELY", "MODERATE_SCORER"]

def is_bend(team: TeamData) -> bool:
    return team.defense_type == "BEND"

# ============================================================================
# PREDICTION ENGINE
# ============================================================================
def run_engine(home: TeamData, away: TeamData, league: str) -> MatchResult:
    home = enrich_team(home)
    away = enrich_team(away)
    
    games = detect_sample_size(home, away)
    sample_weight, max_tier_allowed = get_sample_weight(games)
    league_excluded = "Liga MX" in league or "Mexico" in league
    
    both_o25_tg_zero = (home.scored_25 == 0 and away.scored_25 == 0)
    both_o35_tg_zero = (home.scored_35 == 0 and away.scored_35 == 0)
    combined_o15 = (home.over_15 + away.over_15) / 2
    combined_o25 = (home.over_25 + away.over_25) / 2
    combined_o35 = (home.over_35 + away.over_35) / 2
    combined_btts = (home.btts + away.btts) / 2
    
    tier1_bets = []; tier2_bets = []; tier3_bets = []
    avoid_bets = []; warnings = []; predictions_fired = []
    
    if league_excluded:
        warnings.append(f"⚠️ {league} is excluded.")
        return MatchResult([], [], [], ["ALL MARKETS"], warnings, "League Excluded", [])
    
    if sample_weight == 0.0:
        warnings.append(f"⚠️ {games} games detected. Insufficient data.")
        return MatchResult([], [], [], ["ALL MARKETS"], warnings, "Insufficient Data", [])
    
    if sample_weight <= 0.5:
        warnings.append(f"⚠️ Small sample ({games} games). Confidence reduced.")
    elif sample_weight <= 0.8:
        warnings.append(f"⚠️ Moderate sample ({games} games).")
    
    lock1_fired = False; lock2_fired = False; lock3_fired = False
    lock4_fired = False; lock5_home = False; lock5_away = False
    play6_home = False; play6_away = False
    play8_fired = False; play10_fired = False
    
    # ========================================================================
    # LOCK 1: Quadruple Lock Over 1.5 (4-0)
    # ========================================================================
    collapse_team_l1 = None; opponent_l1 = None
    if is_collapse(home) and away.fts_pct <= 25:
        collapse_team_l1, opponent_l1 = home, away
    elif is_collapse(away) and home.fts_pct <= 25:
        collapse_team_l1, opponent_l1 = away, home
    
    if collapse_team_l1 and collapse_team_l1.btts_no_over25 <= 15 and combined_o15 >= 75:
        lock1_fired = True
        predictions_fired.append("lock1")
        tier = 1 if sample_weight >= 0.8 else 2
        tier1_bets.append(Bet(
            "Over 1.5 Goals", "Over 1.5", tier, "4-0",
            f"Quadruple Lock: {collapse_team_l1.name} collapse + "
            f"{opponent_l1.name} FTS {opponent_l1.fts_pct:.0f}% + "
            f"BTTS No & O2.5 {collapse_team_l1.btts_no_over25:.0f}% + "
            f"Combined O1.5 {combined_o15:.0f}%"
        ))
        
        if combined_o25 >= 55 and not is_one_goal(opponent_l1):
            lock2_fired = True
            predictions_fired.append("lock2")
            tier1_bets.append(Bet(
                "Over 2.5 + BTTS Package", "Over 2.5 & BTTS Yes", 1, "2-0",
                f"Level 2: Combined O2.5 {combined_o25:.0f}%. "
                f"{opponent_l1.name} not one-goal attack."
            ))
    
    # ========================================================================
    # LOCK 3: No-Goal Anchor Under 3.5 (5-0)
    # ========================================================================
    if both_o25_tg_zero and both_o35_tg_zero and combined_o35 < 30:
        lock3_fired = True
        predictions_fired.append("lock3")
        tier1_bets.append(Bet(
            "Under 3.5 Goals", "Under 3.5", 1, "5-0",
            f"No-Goal Anchor: Both O2.5 TG = 0%. Both O3.5 TG = 0%. "
            f"Combined O3.5 {combined_o35:.0f}%."
        ))
    
    # ========================================================================
    # LOCK 4: Dual FTS Lock (4-0)
    # ========================================================================
    if home.fts_pct <= 10 and away.fts_pct <= 10:
        lock4_fired = True
        predictions_fired.append("lock4")
        tier1_bets.append(Bet(
            "Both Teams To Score O0.5", f"{home.name} O0.5 & {away.name} O0.5",
            1, "4-0",
            f"Dual FTS Lock: Home FTS {home.fts_pct:.0f}%. Away FTS {away.fts_pct:.0f}%."
        ))
    
    # ========================================================================
    # LOCK 5: Extreme Collapse CS No (5-0)
    # ========================================================================
    if is_extreme_collapse(home) and home.cs_pct <= 10:
        lock5_home = True
        predictions_fired.append("lock5")
        tier1_bets.append(Bet(
            f"{home.name} Clean Sheet", "No", 1, "5-0",
            f"Extreme Collapse: Concede {home.conceded_05:.0f}%. CS {home.cs_pct:.0f}%."
        ))
    if is_extreme_collapse(away) and away.cs_pct <= 10:
        lock5_away = True
        predictions_fired.append("lock5")
        tier1_bets.append(Bet(
            f"{away.name} Clean Sheet", "No", 1, "5-0",
            f"Extreme Collapse: Concede {away.conceded_05:.0f}%. CS {away.cs_pct:.0f}%."
        ))
    
    # ========================================================================
    # PLAY 6: Collapse Defense CS No (11-0)
    # ========================================================================
    if not lock5_home and is_collapse(home) and home.conceded_05 >= 75 and home.cs_pct <= 20:
        play6_home = True
        predictions_fired.append("play6")
        tier2_bets.append(Bet(
            f"{home.name} Clean Sheet", "No", 2, "11-0",
            f"Collapse: Concede {home.conceded_05:.0f}%. Drop to O1.5: {home.concede_drop:.0f}%."
        ))
    if not lock5_away and is_collapse(away) and away.conceded_05 >= 75 and away.cs_pct <= 20:
        play6_away = True
        predictions_fired.append("play6")
        tier2_bets.append(Bet(
            f"{away.name} Clean Sheet", "No", 2, "11-0",
            f"Collapse: Concede {away.conceded_05:.0f}%. Drop to O1.5: {away.concede_drop:.0f}%."
        ))
    
    # ========================================================================
    # PLAY 7: Team Total O0.5 (17-0)
    # ========================================================================
    if not lock4_fired:
        if home.fts_pct <= 25 and away.cs_pct < 35 and (home.scored_05 >= 70 or away.conceded_05 >= 70):
            predictions_fired.append("play7")
            tier2_bets.append(Bet(
                f"{home.name} Team Total O0.5", "Over 0.5", 2, "17-0",
                f"FTS {home.fts_pct:.0f}%. Opponent CS {away.cs_pct:.0f}% (< 35%)."
            ))
        if away.fts_pct <= 25 and home.cs_pct < 35 and (away.scored_05 >= 70 or home.conceded_05 >= 70):
            predictions_fired.append("play7")
            tier2_bets.append(Bet(
                f"{away.name} Team Total O0.5", "Over 0.5", 2, "17-0",
                f"FTS {away.fts_pct:.0f}%. Opponent CS {home.cs_pct:.0f}% (< 35%)."
            ))
    
    # ========================================================================
    # PLAY 8: BTTS Yes (11-0)
    # ========================================================================
    if not lock2_fired:
        both_concede = home.conceded_05 >= 60 and away.conceded_05 >= 60
        both_score = home.fts_pct <= 25 and away.fts_pct <= 25
        no_cs_block = home.cs_pct < 35 and away.cs_pct < 35
        no_bend_trap = not (is_bend(home) and is_one_goal(away)) and not (is_bend(away) and is_one_goal(home))
        
        if both_concede and both_score and no_cs_block and no_bend_trap:
            play8_fired = True
            predictions_fired.append("play8")
            tier2_bets.append(Bet(
                "BTTS", "Yes", 2, "11-0",
                "Both concede ≥60%. Both FTS ≤25%. No CS or bend traps."
            ))
    
    # ========================================================================
    # PLAY 9: Under 2.5 Both One-Goal (7-0)
    # ========================================================================
    if is_one_goal(home) and is_one_goal(away) and combined_o25 < 50:
        no_collapse_trap = not (is_collapse(home) and is_scores_freely(away)) and \
                           not (is_collapse(away) and is_scores_freely(home))
        if no_collapse_trap:
            predictions_fired.append("play9")
            tier2_bets.append(Bet(
                "Under 2.5 Goals", "Under 2.5", 2, "7-0",
                f"Both one-goal attacks. Combined O2.5 {combined_o25:.0f}%."
            ))
    
    # ========================================================================
    # PLAY 10: BTTS No Dead Attack (3-0)
    # ========================================================================
    if not play8_fired:
        dead_team = None; live_team = None
        if is_dead_attack(home): dead_team, live_team = home, away
        elif is_dead_attack(away): dead_team, live_team = away, home
        
        if dead_team and (dead_team.fts_pct >= 50 or (dead_team.scored_15 == 0 and dead_team.scored_25 == 0)):
            if live_team and live_team.cs_pct >= 25 and live_team.conceded_05 < 80:
                play10_fired = True
                predictions_fired.append("play10")
                tier2_bets.append(Bet(
                    "BTTS", "No", 2, "3-0",
                    f"Dead attack: {dead_team.name} FTS {dead_team.fts_pct:.0f}%. "
                    f"Opponent CS {live_team.cs_pct:.0f}%."
                ))
    
    # ========================================================================
    # PLAY 11: Over 2.5 Collapse + Scorer (3-0)
    # ========================================================================
    if not lock2_fired:
        coll_team = None; opp_team = None
        if is_collapse(home) and away.scored_15 >= 30: coll_team, opp_team = home, away
        elif is_collapse(away) and home.scored_15 >= 30: coll_team, opp_team = away, home
        
        if coll_team and not is_one_goal(opp_team) and combined_o25 >= 55:
            predictions_fired.append("play11")
            tier2_bets.append(Bet(
                "Over 2.5 Goals", "Over 2.5", 2, "3-0",
                f"Collapse + Scorer: {coll_team.name} collapse. "
                f"{opp_team.name} O1.5 TG {opp_team.scored_15:.0f}%. Combined O2.5 {combined_o25:.0f}%."
            ))
    
    # ========================================================================
    # PLAY 12: CS No Moderate (5-3)
    # ========================================================================
    if not lock5_home and not play6_home and 60 <= home.conceded_05 < 75 and away.scored_05 >= 70:
        predictions_fired.append("play12")
        tier3_bets.append(Bet(
            f"{home.name} Clean Sheet", "No", 3, "5-3",
            f"Moderate: Concede {home.conceded_05:.0f}%. Opponent scores {away.scored_05:.0f}%."
        ))
    if not lock5_away and not play6_away and 60 <= away.conceded_05 < 75 and home.scored_05 >= 70:
        predictions_fired.append("play12")
        tier3_bets.append(Bet(
            f"{away.name} Clean Sheet", "No", 3, "5-3",
            f"Moderate: Concede {away.conceded_05:.0f}%. Opponent scores {home.scored_05:.0f}%."
        ))
    
    # ========================================================================
    # PLAY 13: Under 2.5 Low Combined (6-3)
    # ========================================================================
    if not lock3_fired and combined_o25 < 40 and not is_collapse(home) and not is_collapse(away):
        predictions_fired.append("play13")
        tier3_bets.append(Bet(
            "Under 2.5 Goals", "Under 2.5", 3, "6-3",
            f"Low combined O2.5: {combined_o25:.0f}%. No collapse defenses."
        ))
    
    # ========================================================================
    # PLAY 14: Team Total Under 2.5 (7-2)
    # ========================================================================
    if home.scored_35 == 0 and away.conceded_25 < 25:
        predictions_fired.append("play14")
        tier3_bets.append(Bet(
            f"{home.name} Team Total Under 2.5", "Under 2.5", 3, "7-2",
            f"O3.5 TG = 0%. Opponent Concede O2.5 {away.conceded_25:.0f}%."
        ))
    if away.scored_35 == 0 and home.conceded_25 < 25:
        predictions_fired.append("play14")
        tier3_bets.append(Bet(
            f"{away.name} Team Total Under 2.5", "Under 2.5", 3, "7-2",
            f"O3.5 TG = 0%. Opponent Concede O2.5 {home.conceded_25:.0f}%."
        ))
    
    # ========================================================================
    # PLAY 15: Over 1.5 Any Signal (9-1)
    # ========================================================================
    if not lock1_fired and combined_o15 >= 75 and (home.conceded_05 >= 70 or away.conceded_05 >= 70):
        predictions_fired.append("play15")
        tier3_bets.append(Bet(
            "Over 1.5 Goals", "Over 1.5", 3, "9-1",
            f"Combined O1.5 {combined_o15:.0f}%. At least one leaky defense."
        ))
    
    # ========================================================================
    # AVOID BETS
    # ========================================================================
    if lock1_fired or lock2_fired:
        avoid_bets.append("Under 2.5 Goals (contradicts Over signal)")
    if lock2_fired:
        avoid_bets.append("Under 3.5 Goals (contradicts Quadruple Lock L2)")
    if lock3_fired:
        avoid_bets.append("Over 2.5 Goals (contradicts No-Goal Anchor)")
        avoid_bets.append("Over 3.5 Goals (contradicts No-Goal Anchor)")
    if is_collapse(home) and is_collapse(away) and combined_o25 < 55:
        avoid_bets.append("Over 2.5 Goals (combined too low for double collapse)")
    
    # ========================================================================
    # PROFILE
    # ========================================================================
    profile = f"""
**{home.name}**: {home.attack_type} attack ({home.scored_05:.0f}% score, {home.scored_15:.0f}% O1.5) | 
{home.defense_type} defense ({home.conceded_05:.0f}% concede, {home.conceded_15:.0f}% O1.5)
**{away.name}**: {away.attack_type} attack ({away.scored_05:.0f}% score, {away.scored_15:.0f}% O1.5) | 
{away.defense_type} defense ({away.conceded_05:.0f}% concede, {away.conceded_15:.0f}% O1.5)
**Combined**: O1.5: {combined_o15:.0f}% | O2.5: {combined_o25:.0f}% | O3.5: {combined_o35:.0f}% | BTTS: {combined_btts:.0f}%
**Sample**: {games} games detected
"""
    
    return MatchResult(tier1_bets, tier2_bets, tier3_bets, avoid_bets, warnings, profile, predictions_fired)

# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_match_to_db(home_data: TeamData, away_data: TeamData, league: str, 
                     sample_size: int, match_date: date, result: MatchResult) -> Optional[str]:
    try:
        match_record = {
            "home_team": home_data.name, "away_team": away_data.name,
            "league": league, "match_date": str(match_date),
            "home_scored_05": home_data.scored_05, "home_scored_15": home_data.scored_15,
            "home_scored_25": home_data.scored_25, "home_scored_35": home_data.scored_35,
            "home_conceded_05": home_data.conceded_05, "home_conceded_15": home_data.conceded_15,
            "home_conceded_25": home_data.conceded_25, "home_conceded_35": home_data.conceded_35,
            "home_btts": home_data.btts, "home_over_15": home_data.over_15,
            "home_over_25": home_data.over_25, "home_over_35": home_data.over_35,
            "home_btts_over25": home_data.btts_and_over25, "home_btts_no_over25": home_data.btts_no_over25,
            "home_fts": home_data.fts_pct, "home_cs": home_data.cs_pct,
            "home_xg": home_data.xg, "home_actual_scored": home_data.actual_scored,
            "away_scored_05": away_data.scored_05, "away_scored_15": away_data.scored_15,
            "away_scored_25": away_data.scored_25, "away_scored_35": away_data.scored_35,
            "away_conceded_05": away_data.conceded_05, "away_conceded_15": away_data.conceded_15,
            "away_conceded_25": away_data.conceded_25, "away_conceded_35": away_data.conceded_35,
            "away_btts": away_data.btts, "away_over_15": away_data.over_15,
            "away_over_25": away_data.over_25, "away_over_35": away_data.over_35,
            "away_btts_over25": away_data.btts_and_over25, "away_btts_no_over25": away_data.btts_no_over25,
            "away_fts": away_data.fts_pct, "away_cs": away_data.cs_pct,
            "away_xg": away_data.xg, "away_actual_scored": away_data.actual_scored,
            "sample_size": sample_size,
            "predictions_fired": result.predictions_fired,
            "tier1_bets": [{"market": b.market, "bet": b.bet, "record": b.record} for b in result.tier1_bets],
            "tier2_bets": [{"market": b.market, "bet": b.bet, "record": b.record} for b in result.tier2_bets],
            "tier3_bets": [{"market": b.market, "bet": b.bet, "record": b.record} for b in result.tier3_bets],
        }
        
        response = supabase.table("matches").insert(match_record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save match: {e}")
        return None

def get_pending_matches():
    try:
        response = supabase.table("matches").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to fetch pending matches: {e}")
        return []

def submit_match_result(match_id: str, home_score: int, away_score: int):
    try:
        match = supabase.table("matches").select("*").eq("id", match_id).single().execute()
        match_data = match.data
        
        all_bets = []
        for tier_key, tier_num in [("tier1_bets", 1), ("tier2_bets", 2), ("tier3_bets", 3)]:
            bets_json = match_data.get(tier_key)
            if bets_json and isinstance(bets_json, str):
                bets = json.loads(bets_json)
            elif bets_json:
                bets = bets_json
            else:
                continue
            for bet in bets:
                bet["tier"] = tier_num
                all_bets.append(bet)
        
        for bet in all_bets:
            won = evaluate_bet(bet["market"], bet["bet"], home_score, away_score, match_data)
            pred_id = map_market_to_prediction(bet["market"])
            
            if pred_id == "unknown":
                continue
            
            supabase.table("prediction_results").insert({
                "match_id": match_id,
                "prediction_id": pred_id,
                "bet_market": bet["market"],
                "bet_selection": bet["bet"],
                "tier": bet["tier"],
                "won": won,
                "actual_result": f"{home_score}-{away_score}"
            }).execute()
            
            if won:
                supabase.rpc("increment_won", {"pred_id": pred_id}).execute()
            else:
                supabase.rpc("increment_lost", {"pred_id": pred_id}).execute()
        
        supabase.table("matches").update({
            "actual_home_score": home_score,
            "actual_away_score": away_score,
            "result_entered": True
        }).eq("id", match_id).execute()
        
        return True
    except Exception as e:
        st.error(f"Failed to submit result: {e}")
        return False

def evaluate_bet(market: str, bet: str, home_score: int, away_score: int, match_data: dict) -> bool:
    total = home_score + away_score
    home_team = match_data.get("home_team", "")
    away_team = match_data.get("away_team", "")
    
    if "Over 1.5" in market and "Over 1.5" in bet and "Team Total" not in market and "Over 2.5" not in market:
        return total > 1
    if "Over 2.5" in market and "Over" in bet and "Team Total" not in market and "Package" not in market:
        return total > 2
    if "Over 2.5 & BTTS" in bet or ("Package" in market and "Over 2.5" in market):
        return total > 2 and home_score > 0 and away_score > 0
    if "Under 1.5" in market:
        return total < 2
    if "Under 2.5" in market and "Under" in bet and "Team Total" not in market:
        return total < 3
    if "Under 3.5" in market and "Under" in bet:
        return total < 4
    if market.strip() == "BTTS" and "Yes" in bet:
        return home_score > 0 and away_score > 0
    if market.strip() == "BTTS" and "No" in bet:
        return not (home_score > 0 and away_score > 0)
    if "Clean Sheet" in market and "No" in bet:
        team_name = market.split(" Clean Sheet")[0].strip()
        if team_name == home_team:
            return away_score > 0
        else:
            return home_score > 0
    if "Team Total O0.5" in market:
        team_name = market.split(" Team Total")[0].strip()
        if team_name == home_team:
            return home_score > 0
        else:
            return away_score > 0
    if "Team Total Under 2.5" in market:
        team_name = market.split(" Team Total")[0].strip()
        if team_name == home_team:
            return home_score <= 2
        else:
            return away_score <= 2
    if "Both Teams" in market and "O0.5" in market:
        return home_score > 0 and away_score > 0
    
    return False

def map_market_to_prediction(market: str) -> str:
    if "Over 1.5" in market and "Team Total" not in market and "Over 2.5" not in market:
        return "lock1" if "Quadruple" in market else "play15"
    if "Over 2.5 + BTTS" in market or "Package" in market:
        return "lock2"
    if "Under 3.5" in market:
        return "lock3"
    if "Both Teams" in market and "O0.5" in market:
        return "lock4"
    if "Clean Sheet" in market:
        return "play6"
    if "Team Total O0.5" in market:
        return "play7"
    if market.strip() == "BTTS":
        return "play8"
    if "Under 2.5" in market and "Team Total" not in market:
        return "play9"
    if "BTTS No" in market:
        return "play10"
    if "Over 2.5" in market and "Team Total" not in market and "Package" not in market:
        return "play11"
    if "Team Total Under" in market:
        return "play14"
    return "unknown"

def get_prediction_records():
    try:
        response = supabase.table("predictions").select("*").order("tier").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to fetch records: {e}")
        return []

# ============================================================================
# UI INPUT FUNCTIONS
# ============================================================================
def team_input(team_name: str, prefix: str) -> TeamData:
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    st.markdown('<p class="metric-label">⚽ Scored Per Game</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: s05 = st.number_input("Over 0.5 %", 0, 100, 80, 5, key=f"{prefix}_s05")
    with c2: s15 = st.number_input("Over 1.5 %", 0, 100, 38, 5, key=f"{prefix}_s15")
    with c3: s25 = st.number_input("Over 2.5 %", 0, 100, 15, 5, key=f"{prefix}_s25")
    with c4: s35 = st.number_input("Over 3.5 %", 0, 100, 0, 5, key=f"{prefix}_s35")
    
    st.markdown('<p class="metric-label">🛡️ Conceded / Game</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: c05 = st.number_input("Over 0.5 %", 0, 100, 77, 5, key=f"{prefix}_c05")
    with c2: c15 = st.number_input("Over 1.5 %", 0, 100, 40, 5, key=f"{prefix}_c15")
    with c3: c25 = st.number_input("Over 2.5 %", 0, 100, 10, 5, key=f"{prefix}_c25")
    with c4: c35 = st.number_input("Over 3.5 %", 0, 100, 0, 5, key=f"{prefix}_c35")
    
    st.markdown('<p class="metric-label">📊 Match Goals & Core</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: btts = st.number_input("BTTS %", 0, 100, 54, 5, key=f"{prefix}_btts")
    with c2: o15 = st.number_input("Over 1.5 %", 0, 100, 77, 5, key=f"{prefix}_o15")
    with c3: o25 = st.number_input("Over 2.5 %", 0, 100, 46, 5, key=f"{prefix}_o25")
    with c4: o35 = st.number_input("Over 3.5 %", 0, 100, 27, 5, key=f"{prefix}_o35")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: btts_o25 = st.number_input("BTTS & O2.5 %", 0, 100, 46, 5, key=f"{prefix}_btts_o25")
    with c2: btts_no = st.number_input("BTTS No & O2.5 %", 0, 100, 0, 5, key=f"{prefix}_btts_no")
    with c3: fts = st.number_input("Failed to Score %", 0, 100, 23, 5, key=f"{prefix}_fts")
    with c4: cs = st.number_input("Clean Sheet %", 0, 100, 23, 5, key=f"{prefix}_cs")
    
    st.markdown('<p class="metric-label">🎯 xG Context</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: xg = st.number_input("xG per game", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_xg")
    with c2: actual = st.number_input("Actual Scored", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_act")
    
    return TeamData(
        name=team_name, scored_05=float(s05), scored_15=float(s15),
        scored_25=float(s25), scored_35=float(s35), conceded_05=float(c05),
        conceded_15=float(c15), conceded_25=float(c25), conceded_35=float(c35),
        btts=float(btts), over_15=float(o15), over_25=float(o25), over_35=float(o35),
        btts_and_over25=float(btts_o25), btts_no_over25=float(btts_no),
        fts_pct=float(fts), cs_pct=float(cs), xg=float(xg), actual_scored=float(actual)
    )

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("15-Prediction System | Supabase-Powered | Auto-Learning")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ========================================================================
    # TAB 1: ANALYZE
    # ========================================================================
    with tab1:
        c1, c2 = st.columns(2)
        with c1: home_name = st.text_input("🏠 Home Team", "PSV Eindhoven", key="home_name")
        with c2: away_name = st.text_input("✈️ Away Team", "PEC Zwolle", key="away_name")
        
        league = st.text_input("🏆 League", "Netherlands - Eredivisie", key="league")
        match_date = st.date_input("📅 Match Date", date.today(), key="match_date")
        
        st.divider()
        st.subheader(f"🏠 {home_name}")
        home_data = team_input(home_name, "home")
        
        st.divider()
        st.subheader(f"✈️ {away_name}")
        away_data = team_input(away_name, "away")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            result = run_engine(home_data, away_data, league)
            
            # Check for duplicate
            existing = supabase.table("matches").select("id").eq(
                "home_team", home_name
            ).eq("away_team", away_name).eq(
                "match_date", str(match_date)
            ).eq("result_entered", False).execute()
            
            if existing.data and len(existing.data) > 0:
                st.warning(f"⚠️ This match already exists (ID: {existing.data[0]['id'][:8]}...). Submit result in Post-Match tab.")
            else:
                sample_size = detect_sample_size(home_data, away_data)
                match_id = save_match_to_db(home_data, away_data, league, sample_size, match_date, result)
                if match_id:
                    st.success(f"✅ Analysis saved (ID: {match_id[:8]}...)")
            
            # DISPLAY TIER 1
            if result.tier1_bets:
                st.markdown("### 🎯 TIER 1 LOCKS (100% Record)")
                for bet in result.tier1_bets:
                    st.markdown(f"""
                    <div class="output-card tier-1-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <strong style="font-size:1.1rem;">{bet.market}</strong>
                                <span style="font-size:1.2rem;font-weight:bold;margin-left:0.5rem;">→ {bet.bet}</span>
                            </div>
                            <span class="record-badge">✅ {bet.record}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # DISPLAY TIER 2
            if result.tier2_bets:
                st.markdown("### 📊 TIER 2 PLAYS (100% With Filters)")
                for bet in result.tier2_bets:
                    st.markdown(f"""
                    <div class="output-card tier-2-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <strong>{bet.market}</strong>
                                <span style="font-weight:bold;margin-left:0.5rem;">→ {bet.bet}</span>
                            </div>
                            <span class="record-badge">✅ {bet.record}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # DISPLAY TIER 3
            if result.tier3_bets:
                st.markdown("### 💡 TIER 3 VALUE (63-89% Record)")
                for bet in result.tier3_bets:
                    st.markdown(f"""
                    <div class="output-card tier-3-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div>
                                <strong>{bet.market}</strong>
                                <span style="font-weight:bold;margin-left:0.5rem;">→ {bet.bet}</span>
                            </div>
                            <span class="record-badge">📊 {bet.record}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # AVOID
            if result.avoid_bets:
                st.markdown("### ⛔ AVOID")
                for avoid in result.avoid_bets:
                    st.markdown(f'<div class="avoid-card">🚫 {avoid}</div>', unsafe_allow_html=True)
            
            # WARNINGS
            if result.warnings:
                for w in result.warnings:
                    st.markdown(f'<div class="warning-note">⚠️ {w}</div>', unsafe_allow_html=True)
            
            # PROFILE
            if result.profile:
                st.markdown(f'<div class="info-note">{result.profile}</div>', unsafe_allow_html=True)
            
            if not result.tier1_bets and not result.tier2_bets and not result.tier3_bets:
                st.info("🎯 No conditions met. PASS on this match.")
    
    # ========================================================================
    # TAB 2: POST-MATCH
    # ========================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_matches()
        
        if pending:
            match_options = {}
            for m in pending:
                label = f"{m['home_team']} vs {m['away_team']} ({m.get('match_date', m.get('created_at', ''))[:10]})"
                match_options[label] = m['id']
            
            selected_label = st.selectbox("Select Match", list(match_options.keys()))
            
            selected_match = next((m for m in pending if m['id'] == match_options[selected_label]), None)
            if selected_match:
                preds = selected_match.get('predictions_fired', [])
                if isinstance(preds, str):
                    preds = json.loads(preds)
                st.caption(f"Predictions fired: {preds}")
            
            c1, c2 = st.columns(2)
            with c1: home_score = st.number_input("Home Score", 0, 20, 0, key="post_home")
            with c2: away_score = st.number_input("Away Score", 0, 20, 0, key="post_away")
            
            if st.button("✅ Submit Result", type="primary"):
                success = submit_match_result(match_options[selected_label], home_score, away_score)
                if success:
                    st.success("Result submitted! Records updated.")
                    st.rerun()
                else:
                    st.error("Failed to submit result.")
        else:
            st.info("🎉 No pending matches. All results entered.")
    
    # ========================================================================
    # TAB 3: RECORDS
    # ========================================================================
    with tab3:
        st.subheader("📊 Live Prediction Records")
        records = get_prediction_records()
        
        if records:
            for pred in records:
                fired = pred['total_fired']
                won = pred['total_won']
                lost = pred['total_lost']
                win_rate = (won / fired * 100) if fired > 0 else 0
                
                if win_rate >= 90: color = "#10b981"
                elif win_rate >= 70: color = "#fbbf24"
                elif win_rate >= 50: color = "#f97316"
                else: color = "#ef4444"
                
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                            background:#1e293b;padding:0.75rem;border-radius:8px;margin:0.3rem 0;color:#ffffff;">
                    <div>
                        <strong>Tier {pred['tier']}: {pred['name']}</strong>
                        <span style="color:#94a3b8;margin-left:0.5rem;font-size:0.85rem;">({pred['id']})</span>
                    </div>
                    <div>
                        <span style="color:{color};font-weight:700;">{won}/{fired} ({win_rate:.0f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No prediction records yet. Analyze matches and enter results to build records.")
    
    # FOOTER
    st.divider()
    st.markdown("""
    ### 📋 Prediction Catalog (15 Total)
    
    | # | Prediction | Tier | Record |
    |---|-----------|------|--------|
    | 1 | Quadruple Lock - Over 1.5 | 1 | 4-0 |
    | 2 | Quadruple Lock - Over 2.5 + BTTS | 1 | 2-0 |
    | 3 | No-Goal Anchor - Under 3.5 | 1 | 5-0 |
    | 4 | Dual FTS Lock - Both Teams O0.5 | 1 | 4-0 |
    | 5 | Extreme Collapse CS No | 1 | 5-0 |
    | 6 | Collapse Defense CS No | 2 | 11-0 |
    | 7 | Team Total O0.5 (FTS ≤25%) | 2 | 17-0 |
    | 8 | BTTS Yes (Both Concede + Score) | 2 | 11-0 |
    | 9 | Under 2.5 (Both One-Goal) | 2 | 7-0 |
    | 10 | BTTS No (Dead Attack) | 2 | 3-0 |
    | 11 | Over 2.5 (Collapse + Scorer) | 2 | 3-0 |
    | 12 | CS No (Moderate Leak) | 3 | 5-3 |
    | 13 | Under 2.5 (Low Combined) | 3 | 6-3 |
    | 14 | Team Total Under 2.5 | 3 | 7-2 |
    | 15 | Over 1.5 (Any Signal) | 3 | 9-1 |
    """)

if __name__ == "__main__":
    main()
