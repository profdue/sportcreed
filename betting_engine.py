"""
Streak Predictor - Complete 5-Layer Architecture
Story-Driven Predictions with Data-Aware Confidence

LAYER 1: Sample Validator
LAYER 2: Shape Classifier  
LAYER 3: Story Matcher
LAYER 4: Bet Generator (DATA-AWARE - checks metrics per bet)
LAYER 5: Confidence Assigner (tier adjustments)

OUTPUT: Story Narrative + Data-Justified Bets + Risk Notes + Scoreline Range
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - Story Engine",
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
    .story-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 24px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #f59e0b;
    }
    .bet-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .tier-1 { border-left: 4px solid #10b981; }
    .tier-2 { border-left: 4px solid #fbbf24; }
    .tier-3 { border-left: 4px solid #f97316; }
    .tier-4 { border-left: 4px solid #ef4444; }
    .tier-5 { border-left: 4px solid #64748b; }
    .avoid-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #64748b;
        opacity: 0.7;
    }
    .team-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .team-name { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .section-header {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin: 1rem 0 0.5rem 0;
        font-weight: 700;
        text-align: center;
    }
    hr { margin: 1rem 0; }
    .stButton button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        border: none;
        width: 100%;
    }
    .risk-note {
        background: #7f1a1a;
        border-left: 4px solid #ef4444;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
    }
    .coexistence-note {
        background: #1a3a1a;
        border-left: 4px solid #10b981;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.85rem;
    }
    .confidence-very-high { color: #10b981; font-weight: 700; }
    .confidence-high { color: #fbbf24; font-weight: 700; }
    .confidence-medium { color: #f97316; font-weight: 600; }
    .confidence-low { color: #ef4444; font-weight: 500; }
    .confidence-pass { color: #64748b; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamMetrics:
    name: str
    # Distribution percentages
    scored_05: float = 0.0
    scored_15: float = 0.0
    scored_25: float = 0.0
    scored_35: float = 0.0
    conceded_05: float = 0.0
    conceded_15: float = 0.0
    conceded_25: float = 0.0
    conceded_35: float = 0.0
    btts: float = 0.0
    over_15: float = 0.0
    over_25: float = 0.0
    over_35: float = 0.0
    btts_and_over25: float = 0.0
    btts_no_and_over25: float = 0.0
    # Venue-specific
    fts_pct: float = 0.0
    cs_pct: float = 0.0
    games_played: int = 0
    # xG data
    xg: float = 0.0
    actual_scored: float = 0.0
    # Derived fields
    attack_type: str = ""
    defense_type: str = ""
    xg_type: str = ""


@dataclass
class Bet:
    market: str
    bet: str
    tier: int
    confidence: str
    reasoning: str


@dataclass
class AnalysisOutput:
    story_name: str
    story_description: str
    narrative: str
    bets: List[Bet]
    avoid_bets: List[str]
    risks: List[str]
    scorelines: List[str]
    warnings: List[str]
    coexistence_notes: List[str]


# ============================================================================
# LAYER 1: SAMPLE VALIDATOR
# ============================================================================
def validate_sample(home_games: int, away_games: int) -> Dict:
    min_games = min(home_games, away_games)
    
    if min_games < 5:
        return {
            "valid": False,
            "story": "INSUFFICIENT_DATA",
            "confidence_multiplier": 0.0,
            "warning": f"Only {min_games} games. Data is unreliable. PASS on all markets."
        }
    elif min_games < 8:
        return {
            "valid": True,
            "confidence_multiplier": 0.7,
            "warning": f"Only {min_games} games. Confidence downgraded by 1 tier."
        }
    elif min_games < 15:
        return {
            "valid": True,
            "confidence_multiplier": 0.9,
            "warning": None
        }
    else:
        return {
            "valid": True,
            "confidence_multiplier": 1.0,
            "warning": None
        }


# ============================================================================
# LAYER 2: SHAPE CLASSIFIER
# ============================================================================
def classify_defense(concede_05: float, concede_15: float, concede_25: float, concede_35: float) -> str:
    drop_05_to_15 = concede_05 - concede_15
    
    if concede_05 >= 85 and drop_05_to_15 <= 30:
        return "EXTREME_COLLAPSE"
    elif drop_05_to_15 <= 30:
        return "COLLAPSE"
    elif drop_05_to_15 >= 45:
        return "BEND"
    else:
        return "MODERATE"


def classify_attack(scored_05: float, scored_15: float, scored_25: float, scored_35: float, fts: float) -> str:
    drop_05_to_15 = scored_05 - scored_15
    
    if fts >= 65:
        return "NO_GOAL"
    elif fts >= 50:
        return "RARELY_SCORES"
    elif scored_05 >= 85 and drop_05_to_15 <= 25:
        return "SCORES_FREELY"
    elif drop_05_to_15 <= 25:
        return "MODERATE_SCORER"
    elif drop_05_to_15 >= 45:
        return "ONE_GOAL"
    else:
        return "MODERATE"


def classify_xg(xg: float, actual_scored: float) -> str:
    gap = actual_scored - xg
    if gap < -0.3:
        return "UNDERPERFORMING"
    elif gap > 0.3:
        return "OVERPERFORMING"
    else:
        return "NEUTRAL"


def check_binary(cs_pct: float, fts_pct: float) -> bool:
    return 40 <= cs_pct <= 55 and 40 <= fts_pct <= 55


def check_mismatch(home_concede_05: float, away_concede_05: float) -> bool:
    return abs(home_concede_05 - away_concede_05) >= 40


def classify_team(team: TeamMetrics) -> TeamMetrics:
    team.attack_type = classify_attack(
        team.scored_05, team.scored_15, team.scored_25, team.scored_35, team.fts_pct
    )
    team.defense_type = classify_defense(
        team.conceded_05, team.conceded_15, team.conceded_25, team.conceded_35
    )
    team.xg_type = classify_xg(team.xg, team.actual_scored)
    return team


# ============================================================================
# LAYER 3: STORY MATCHER
# ============================================================================
STORIES = {
    "NO_GOAL_PRESENT": {
        "priority": 1,
        "description": "At least one attack is effectively dead. Goals will be scarce.",
        "condition": lambda h, a: h.attack_type == "NO_GOAL" or a.attack_type == "NO_GOAL"
    },
    "EXTREME_COLLAPSE_DEFENSE": {
        "priority": 2,
        "description": "A defense that collapses when breached. Concedes multiple goals regularly.",
        "condition": lambda h, a: h.defense_type == "EXTREME_COLLAPSE" or a.defense_type == "EXTREME_COLLAPSE"
    },
    "MISMATCHED_DEFENSES": {
        "priority": 3,
        "description": "One defense is far superior to the other. Trust the elite defense.",
        "condition": lambda h, a: check_mismatch(h.conceded_05, a.conceded_05)
    },
    "COLLAPSE_VS_SCORER": {
        "priority": 4,
        "description": "Collapse defense meets an attack that can score multiple. Over match expected.",
        "condition": lambda h, a: (h.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] and a.attack_type in ["SCORES_FREELY", "MODERATE_SCORER"]) or \
                                   (a.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] and h.attack_type in ["SCORES_FREELY", "MODERATE_SCORER"])
    },
    "TWO_ONE_GOAL_ATTACKS": {
        "priority": 5,
        "description": "Both attacks are limited to 1 goal typically. Under match with BTTS possible.",
        "condition": lambda h, a: h.attack_type == "ONE_GOAL" and a.attack_type == "ONE_GOAL"
    },
    "ONE_GOAL_VS_COLLAPSE": {
        "priority": 6,
        "description": "One-goal attack meets collapse defense. Defense concedes but attack may not score 2+.",
        "condition": lambda h, a: (h.attack_type == "ONE_GOAL" and a.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"]) or \
                                   (a.attack_type == "ONE_GOAL" and h.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"])
    },
    "BINARY_HOME_TEAM": {
        "priority": 7,
        "description": "Home team is unpredictable. Either blanks or scores multiple. Either CS or concedes multiple.",
        "condition": lambda h, a: check_binary(h.cs_pct, h.fts_pct)
    },
    "REGRESSION_CANDIDATE": {
        "priority": 8,
        "description": "xG gap detected. One or both teams due for regression to mean.",
        "condition": lambda h, a: h.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"] or \
                                   a.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"]
    }
}


def match_stories(home: TeamMetrics, away: TeamMetrics) -> List[Dict]:
    matched = []
    for name, story in sorted(STORIES.items(), key=lambda x: x[1]["priority"]):
        if story["condition"](home, away):
            matched.append({
                "name": name,
                "description": story["description"],
                "priority": story["priority"]
            })
    
    if not matched:
        matched.append({
            "name": "NO_CLEAR_STORY",
            "description": "No clear pattern detected. Proceed with caution.",
            "priority": 99
        })
    
    return matched


# ============================================================================
# LAYER 4: BET GENERATOR (DATA-AWARE)
# ============================================================================
def apply_sample_adjustment(tier: int, sample_multiplier: float) -> int:
    """Adjust tier based on sample size multiplier"""
    if sample_multiplier == 0.7:
        tier += 1
    elif sample_multiplier == 0.5:
        tier += 2
    elif sample_multiplier == 0.0:
        return 5
    return min(tier, 5)


def get_confidence_label(tier: int) -> str:
    return {1: "VERY HIGH", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "PASS"}.get(tier, "UNKNOWN")


def generate_bets_no_goal(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 1: NO_GOAL_PRESENT"""
    bets = []
    avoid = ["Over 2.5 Goals", "BTTS Yes", "Over 3.5 Goals"]
    
    dead_team = home if home.attack_type == "NO_GOAL" else away
    live_team = away if home.attack_type == "NO_GOAL" else home
    opponent_cs = live_team.cs_pct
    
    # Under 3.5 - anchor bet
    both_zero_35 = home.scored_35 == 0 and away.scored_35 == 0
    if both_zero_35:
        under35_tier = apply_sample_adjustment(1, multiplier)
        under35_reasoning = f"Both teams 0% Over 3.5 Team Goals. {dead_team.name} has never scored 3+."
    else:
        under35_tier = apply_sample_adjustment(2, multiplier)
        under35_reasoning = f"Limited attacks. {dead_team.name} cannot score 3+."
    
    bets.append(Bet("Under 3.5 Goals", "Under 3.5", under35_tier, 
                    get_confidence_label(under35_tier), under35_reasoning))
    
    # Under 2.5
    combined_over25 = (home.over_25 + away.over_25) / 2
    if combined_over25 < 40:
        under25_tier = apply_sample_adjustment(2, multiplier)
        under25_reasoning = f"Combined Over 2.5 only {combined_over25:.0f}%. {dead_team.name} limits totals."
    else:
        under25_tier = apply_sample_adjustment(3, multiplier)
        under25_reasoning = f"Combined Over 2.5 at {combined_over25:.0f}%. {dead_team.name} attack is the ceiling."
    
    bets.append(Bet("Under 2.5 Goals", "Under 2.5", under25_tier,
                    get_confidence_label(under25_tier), under25_reasoning))
    
    # BTTS No
    if dead_team.fts_pct >= 65 and opponent_cs >= 30:
        btts_tier = apply_sample_adjustment(1, multiplier)
        btts_reasoning = f"{dead_team.name} FTS {dead_team.fts_pct:.0f}% + opponent CS {opponent_cs:.0f}%. Elite BTTS No signal."
    elif dead_team.fts_pct >= 65:
        btts_tier = apply_sample_adjustment(2, multiplier)
        btts_reasoning = f"{dead_team.name} blanks in {dead_team.fts_pct:.0f}% of games. Opponent CS only {opponent_cs:.0f}%."
    else:
        btts_tier = apply_sample_adjustment(3, multiplier)
        btts_reasoning = f"{dead_team.name} rarely scores (FTS {dead_team.fts_pct:.0f}%)."
    
    bets.append(Bet("BTTS", "No", btts_tier, get_confidence_label(btts_tier), btts_reasoning))
    
    # Live team to score
    live_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet(f"{live_team.name} Team Total O0.5", "Over 0.5", live_tier,
                    get_confidence_label(live_tier),
                    f"{live_team.name} scores in {live_team.scored_05:.0f}% of games. {dead_team.name} concedes {dead_team.conceded_05:.0f}%."))
    
    # Dead team PASS
    bets.append(Bet(f"{dead_team.name} Team Total O0.5", "PASS", 5, "PASS",
                    f"{dead_team.name} FTS {dead_team.fts_pct:.0f}%. Cannot bet them to score."))
    
    # Dead team CS No
    if dead_team.conceded_05 >= 75:
        cs_tier = apply_sample_adjustment(3, multiplier)
        bets.append(Bet(f"{dead_team.name} Clean Sheet", "No", cs_tier,
                        get_confidence_label(cs_tier),
                        f"{dead_team.name} concedes in {dead_team.conceded_05:.0f}% of games. Even dead attacks concede sometimes."))
    
    return bets, avoid


def generate_bets_extreme_collapse(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 2: EXTREME_COLLAPSE_DEFENSE"""
    bets = []
    avoid = ["Under 2.5 Goals", "Under 3.5 Goals"]
    
    collapse_team = home if home.defense_type == "EXTREME_COLLAPSE" else away
    opponent = away if home.defense_type == "EXTREME_COLLAPSE" else home
    
    # CS No for collapse team - LOCK
    cs_tier = apply_sample_adjustment(1, multiplier)
    bets.append(Bet(f"{collapse_team.name} Clean Sheet", "No", cs_tier,
                    get_confidence_label(cs_tier),
                    f"{collapse_team.name} concedes in {collapse_team.conceded_05:.0f}% of games. CS only {collapse_team.cs_pct:.0f}%. Lock."))
    
    # BTTS Yes
    if opponent.fts_pct <= 25:
        btts_tier = apply_sample_adjustment(2, multiplier)
        btts_reasoning = f"Collapse defense concedes. Opponent scores in {opponent.scored_05:.0f}% of games."
    else:
        btts_tier = apply_sample_adjustment(3, multiplier)
        btts_reasoning = f"Collapse defense concedes. But opponent FTS {opponent.fts_pct:.0f}% limits BTTS confidence."
    
    bets.append(Bet("BTTS", "Yes", btts_tier, get_confidence_label(btts_tier), btts_reasoning))
    
    # Over 2.5
    combined_over25 = (home.over_25 + away.over_25) / 2
    if combined_over25 >= 60:
        over_tier = apply_sample_adjustment(2, multiplier)
        over_reasoning = f"Combined Over 2.5 at {combined_over25:.0f}%. Collapse defense drives totals."
    else:
        over_tier = apply_sample_adjustment(3, multiplier)
        over_reasoning = f"Combined Over 2.5 at {combined_over25:.0f}%. Collapse defense helps but opponent may not contribute."
    
    bets.append(Bet("Over 2.5 Goals", "Over 2.5", over_tier, get_confidence_label(over_tier), over_reasoning))
    
    # Opponent Team Total O0.5
    opp_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet(f"{opponent.name} Team Total O0.5", "Over 0.5", opp_tier,
                    get_confidence_label(opp_tier),
                    f"{opponent.name} faces a defense that concedes {collapse_team.conceded_05:.0f}% of games."))
    
    # Opponent Team Total O1.5
    if opponent.scored_15 >= 35:
        opp15_tier = apply_sample_adjustment(3, multiplier)
        bets.append(Bet(f"{opponent.name} Team Total O1.5", "Over 1.5", opp15_tier,
                        get_confidence_label(opp15_tier),
                        f"{opponent.name} scores 2+ in {opponent.scored_15:.0f}% of games. Collapse defense concedes multiple."))
    
    return bets, avoid


def generate_bets_mismatch(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 3: MISMATCHED_DEFENSES"""
    bets = []
    avoid = ["BTTS Yes", "Over 2.5 Goals"]
    
    elite_team = home if home.conceded_05 < away.conceded_05 else away
    weak_team = away if home.conceded_05 < away.conceded_05 else home
    
    # Elite defense CS Yes
    if elite_team.cs_pct >= 35:
        cs_tier = apply_sample_adjustment(2, multiplier)
        bets.append(Bet(f"{elite_team.name} Clean Sheet", "Yes", cs_tier,
                        get_confidence_label(cs_tier),
                        f"{elite_team.name} CS {elite_team.cs_pct:.0f}%. Elite defense vs weak opponent."))
    
    # BTTS No
    if weak_team.fts_pct >= 45:
        btts_tier = apply_sample_adjustment(2, multiplier)
        btts_reasoning = f"Weak attack ({weak_team.name} FTS {weak_team.fts_pct:.0f}%) meets elite defense."
    else:
        btts_tier = apply_sample_adjustment(3, multiplier)
        btts_reasoning = f"Elite defense should control. {weak_team.name} may struggle to score."
    
    bets.append(Bet("BTTS", "No", btts_tier, get_confidence_label(btts_tier), btts_reasoning))
    
    # Under 2.5
    combined_over25 = (home.over_25 + away.over_25) / 2
    under_tier = apply_sample_adjustment(2, multiplier) if combined_over25 < 50 else apply_sample_adjustment(3, multiplier)
    bets.append(Bet("Under 2.5 Goals", "Under 2.5", under_tier,
                    get_confidence_label(under_tier),
                    f"Elite defense keeps totals low. Combined Over 2.5 at {combined_over25:.0f}%."))
    
    # Elite team to win (via scoring)
    elite_scored_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet(f"{elite_team.name} Team Total O0.5", "Over 0.5", elite_scored_tier,
                    get_confidence_label(elite_scored_tier),
                    f"{elite_team.name} scores in {elite_team.scored_05:.0f}% of games. {weak_team.name} concedes {weak_team.conceded_05:.0f}%."))
    
    return bets, avoid


def generate_bets_collapse_vs_scorer(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 4: COLLAPSE_VS_SCORER"""
    bets = []
    avoid = ["Under 2.5 Goals"]
    
    collapse_team = home if home.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] else away
    scorer_team = away if home.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] else home
    
    # CS No for collapse team
    cs_tier = apply_sample_adjustment(1 if collapse_team.defense_type == "EXTREME_COLLAPSE" else 2, multiplier)
    bets.append(Bet(f"{collapse_team.name} Clean Sheet", "No", cs_tier,
                    get_confidence_label(cs_tier),
                    f"{collapse_team.name} concedes {collapse_team.conceded_05:.0f}%. Scorer team can exploit."))
    
    # BTTS Yes
    btts_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet("BTTS", "Yes", btts_tier, get_confidence_label(btts_tier),
                    f"Collapse defense concedes. Scorer team scores in {scorer_team.scored_05:.0f}%. Both should score."))
    
    # Over 2.5
    combined_over25 = (home.over_25 + away.over_25) / 2
    over_tier = apply_sample_adjustment(2, multiplier) if combined_over25 >= 55 else apply_sample_adjustment(3, multiplier)
    bets.append(Bet("Over 2.5 Goals", "Over 2.5", over_tier,
                    get_confidence_label(over_tier),
                    f"Collapse defense + scoring attack. Combined Over 2.5 at {combined_over25:.0f}%."))
    
    # Scorer Team Total O1.5
    if scorer_team.scored_15 >= 30:
        opp15_tier = apply_sample_adjustment(3, multiplier)
        bets.append(Bet(f"{scorer_team.name} Team Total O1.5", "Over 1.5", opp15_tier,
                        get_confidence_label(opp15_tier),
                        f"{scorer_team.name} scores 2+ in {scorer_team.scored_15:.0f}%. Collapse defense concedes multiple."))
    
    # Under 3.5
    if collapse_team.conceded_35 == 0 and scorer_team.scored_35 == 0:
        under35_tier = apply_sample_adjustment(2, multiplier)
        bets.append(Bet("Under 3.5 Goals", "Under 3.5", under35_tier,
                        get_confidence_label(under35_tier),
                        "Both teams 0% Over 3.5 TG. Ceiling exists despite collapse."))
    
    return bets, avoid


def generate_bets_two_one_goal(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 5: TWO_ONE_GOAL_ATTACKS"""
    bets = []
    avoid = ["Over 2.5 Goals"]
    
    # Under 3.5 - LOCK
    both_zero_35 = home.scored_35 == 0 and away.scored_35 == 0
    if both_zero_35:
        under35_tier = apply_sample_adjustment(1, multiplier)
        under35_reasoning = "Both teams 0% Over 3.5 Team Goals. Neither can score 3+."
    else:
        under35_tier = apply_sample_adjustment(2, multiplier)
        under35_reasoning = "Both attacks limited to 1-2 goals."
    
    bets.append(Bet("Under 3.5 Goals", "Under 3.5", under35_tier,
                    get_confidence_label(under35_tier), under35_reasoning))
    
    # Under 2.5
    combined_over25 = (home.over_25 + away.over_25) / 2
    under25_tier = apply_sample_adjustment(2, multiplier) if combined_over25 < 50 else apply_sample_adjustment(3, multiplier)
    bets.append(Bet("Under 2.5 Goals", "Under 2.5", under25_tier,
                    get_confidence_label(under25_tier),
                    f"Two one-goal attacks. Combined Over 2.5 at {combined_over25:.0f}%."))
    
    # BTTS No
    if home.fts_pct >= 45 or away.fts_pct >= 45:
        btts_tier = apply_sample_adjustment(2, multiplier)
        btts_reasoning = f"One team blanks regularly. Both attacks limited."
    else:
        btts_tier = apply_sample_adjustment(3, multiplier)
        btts_reasoning = "Both attacks score 1 typically. BTTS is a coin flip."
    
    bets.append(Bet("BTTS", "No", btts_tier, get_confidence_label(btts_tier), btts_reasoning))
    
    return bets, avoid


def generate_bets_one_goal_vs_collapse(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 6: ONE_GOAL_VS_COLLAPSE"""
    bets = []
    avoid = ["Over 3.5 Goals"]
    
    one_goal_team = home if home.attack_type == "ONE_GOAL" else away
    collapse_team = away if home.attack_type == "ONE_GOAL" else home
    
    # Collapse team CS No
    cs_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet(f"{collapse_team.name} Clean Sheet", "No", cs_tier,
                    get_confidence_label(cs_tier),
                    f"{collapse_team.name} concedes {collapse_team.conceded_05:.0f}%. Defense collapses when breached."))
    
    # Under 3.5
    if one_goal_team.scored_35 == 0 and collapse_team.conceded_35 == 0:
        under35_tier = apply_sample_adjustment(2, multiplier)
        bets.append(Bet("Under 3.5 Goals", "Under 3.5", under35_tier,
                        get_confidence_label(under35_tier),
                        "One-goal attack cannot score 3+. Collapse defense rarely concedes 3+."))
    
    # BTTS Yes
    if one_goal_team.scored_05 >= 70:
        btts_tier = apply_sample_adjustment(2, multiplier)
        btts_reasoning = f"{one_goal_team.name} scores consistently. {collapse_team.name} concedes consistently."
    else:
        btts_tier = apply_sample_adjustment(3, multiplier)
        btts_reasoning = f"One-goal attack may score. Collapse defense should concede."
    
    bets.append(Bet("BTTS", "Yes", btts_tier, get_confidence_label(btts_tier), btts_reasoning))
    
    # One-goal team to score
    one_goal_tier = apply_sample_adjustment(2, multiplier)
    bets.append(Bet(f"{one_goal_team.name} Team Total O0.5", "Over 0.5", one_goal_tier,
                    get_confidence_label(one_goal_tier),
                    f"{one_goal_team.name} scores in {one_goal_team.scored_05:.0f}% of games."))
    
    return bets, avoid


def generate_bets_binary_home(home: TeamMetrics, away: TeamMetrics, multiplier: float) -> Tuple[List[Bet], List[str]]:
    """Story 7: BINARY_HOME_TEAM"""
    bets = []
    avoid = ["BTTS", "Over 2.5 Goals", "Under 2.5 Goals"]
    
    # Limited bets - binary teams are unpredictable
    if home.conceded_05 >= 70:
        cs_tier = apply_sample_adjustment(3, multiplier)
        bets.append(Bet(f"{home.name} Clean Sheet", "No", cs_tier,
                        get_confidence_label(cs_tier),
                        f"{home.name} concedes in {home.conceded_05:.0f}% of games. But binary."))
    
    if away.conceded_05 >= 80:
        away_cs_tier = apply_sample_adjustment(2, multiplier)
        bets.append(Bet(f"{away.name} Clean Sheet", "No", away_cs_tier,
                        get_confidence_label(away_cs_tier),
                        f"{away.name} concedes in {away.conceded_05:.0f}% of games."))
    
    if away.scored_05 >= 70:
        away_scored_tier = apply_sample_adjustment(2, multiplier)
        bets.append(Bet(f"{away.name} Team Total O0.5", "Over 0.5", away_scored_tier,
                        get_confidence_label(away_scored_tier),
                        f"{away.name} scores in {away.scored_05:.0f}% of games."))
    
    return bets, avoid


def generate_bets_generic(home: TeamMetrics, away: TeamMetrics, multiplier: float, matched_stories: List[Dict]) -> Tuple[List[Bet], List[str]]:
    """Fallback: Use regression story or basic signals"""
    bets = []
    avoid = []
    
    # Check xG regression for upgrades
    if home.xg_type == "UNDERPERFORMING":
        bets.append(Bet(f"{home.name} Team Total O0.5", "Over 0.5", 
                        apply_sample_adjustment(3, multiplier), "MEDIUM",
                        f"{home.name} underperforming xG by {home.actual_scored - home.xg:.2f}. Regression likely."))
    
    if away.xg_type == "UNDERPERFORMING":
        bets.append(Bet(f"{away.name} Team Total O0.5", "Over 0.5",
                        apply_sample_adjustment(3, multiplier), "MEDIUM",
                        f"{away.name} underperforming xG by {away.actual_scored - away.xg:.2f}. Regression likely."))
    
    # Basic Clean Sheet No if concede is high
    if home.conceded_05 >= 75:
        bets.append(Bet(f"{home.name} Clean Sheet", "No", apply_sample_adjustment(3, multiplier),
                        "MEDIUM", f"{home.name} concedes in {home.conceded_05:.0f}% of games."))
    
    if away.conceded_05 >= 75:
        bets.append(Bet(f"{away.name} Clean Sheet", "No", apply_sample_adjustment(3, multiplier),
                        "MEDIUM", f"{away.name} concedes in {away.conceded_05:.0f}% of games."))
    
    return bets, avoid


def generate_bets(home: TeamMetrics, away: TeamMetrics, story_name: str, 
                  sample_multiplier: float, matched_stories: List[Dict]) -> Tuple[List[Bet], List[str]]:
    """Main bet generator - dispatches to story-specific logic"""
    
    story_generators = {
        "NO_GOAL_PRESENT": generate_bets_no_goal,
        "EXTREME_COLLAPSE_DEFENSE": generate_bets_extreme_collapse,
        "MISMATCHED_DEFENSES": generate_bets_mismatch,
        "COLLAPSE_VS_SCORER": generate_bets_collapse_vs_scorer,
        "TWO_ONE_GOAL_ATTACKS": generate_bets_two_one_goal,
        "ONE_GOAL_VS_COLLAPSE": generate_bets_one_goal_vs_collapse,
        "BINARY_HOME_TEAM": generate_bets_binary_home,
    }
    
    generator = story_generators.get(story_name)
    if generator:
        bets, avoid = generator(home, away, sample_multiplier)
    else:
        bets, avoid = generate_bets_generic(home, away, sample_multiplier, matched_stories)
    
    return bets, avoid


# ============================================================================
# LAYER 5: POST-PROCESSING (xG adjustments, coexistence)
# ============================================================================
def apply_xg_adjustments(bets: List[Bet], home: TeamMetrics, away: TeamMetrics) -> List[Bet]:
    """Adjust tiers based on xG under/overperformance"""
    for bet in bets:
        # Upgrade underperforming teams
        if home.xg_type == "UNDERPERFORMING" and home.name in bet.market and "Over" in bet.bet and "Team Total" in bet.market:
            if bet.tier > 1:
                bet.tier -= 1
                bet.confidence = get_confidence_label(bet.tier)
                bet.reasoning += f" ⬆️ xG underperformance ({home.actual_scored:.1f} vs {home.xg:.1f} xG) suggests regression."
        
        if away.xg_type == "UNDERPERFORMING" and away.name in bet.market and "Over" in bet.bet and "Team Total" in bet.market:
            if bet.tier > 1:
                bet.tier -= 1
                bet.confidence = get_confidence_label(bet.tier)
                bet.reasoning += f" ⬆️ xG underperformance ({away.actual_scored:.1f} vs {away.xg:.1f} xG) suggests regression."
        
        # Downgrade overperforming teams
        if home.xg_type == "OVERPERFORMING" and home.name in bet.market and "Over" in bet.bet and "Team Total" in bet.market:
            if bet.tier < 5:
                bet.tier += 1
                bet.confidence = get_confidence_label(bet.tier)
                bet.reasoning += f" ⬇️ xG overperformance ({home.actual_scored:.1f} vs {home.xg:.1f} xG) suggests regression."
        
        if away.xg_type == "OVERPERFORMING" and away.name in bet.market and "Over" in bet.bet and "Team Total" in bet.market:
            if bet.tier < 5:
                bet.tier += 1
                bet.confidence = get_confidence_label(bet.tier)
                bet.reasoning += f" ⬇️ xG overperformance ({away.actual_scored:.1f} vs {away.xg:.1f} xG) suggests regression."
    
    return bets


def check_coexistence(bets: List[Bet]) -> List[str]:
    """Check for compatible bet combinations"""
    notes = []
    
    has_under35 = any("Under 3.5" in b.market and "Under" in b.bet for b in bets)
    has_under25 = any("Under 2.5" in b.market and "Under" in b.bet for b in bets)
    has_over25 = any("Over 2.5" in b.market and "Over" in b.bet for b in bets)
    has_btts_yes = any(b.market == "BTTS" and b.bet == "Yes" for b in bets)
    has_btts_no = any(b.market == "BTTS" and b.bet == "No" for b in bets)
    
    if has_under35 and has_btts_yes:
        notes.append("✅ Under 3.5 + BTTS Yes can coexist → 1-1, 2-1 scorelines favored.")
    if has_under35 and has_btts_no:
        notes.append("✅ Under 3.5 + BTTS No are aligned → 1-0, 2-0, 0-0 type match.")
    if has_over25 and has_btts_yes:
        notes.append("✅ Over 2.5 + BTTS Yes are correlated → 2-1, 2-2, 3-1 type match.")
    if has_over25 and has_btts_no:
        notes.append("⚠️ Over 2.5 + BTTS No is rare. Only possible if one team scores 3+ alone.")
    if has_under25 and has_btts_yes:
        notes.append("⚠️ Under 2.5 + BTTS Yes → 1-1 is the most likely scoreline.")
    
    return notes


# ============================================================================
# NARRATIVE GENERATOR
# ============================================================================
def generate_shape_description(home: TeamMetrics, away: TeamMetrics) -> str:
    """Generate data-rich shape description"""
    
    def attack_desc(team: TeamMetrics) -> str:
        drop = team.scored_05 - team.scored_15
        return f"{team.attack_type} (scores in {team.scored_05:.0f}%, 2+ in {team.scored_15:.0f}%, drop: {drop:.0f}%)"
    
    def defense_desc(team: TeamMetrics) -> str:
        drop = team.conceded_05 - team.conceded_15
        return f"{team.defense_type} (concedes in {team.conceded_05:.0f}%, 2+ in {team.conceded_15:.0f}%, drop: {drop:.0f}%)"
    
    return f"""
**SHAPE ANALYSIS:**
- **{home.name}**: {attack_desc(home)} | {defense_desc(home)} | xG: {home.xg_type} ({home.actual_scored:.1f} vs {home.xg:.1f})
- **{away.name}**: {attack_desc(away)} | {defense_desc(away)} | xG: {away.xg_type} ({away.actual_scored:.1f} vs {away.xg:.1f})
"""


def generate_matchup_analysis(home: TeamMetrics, away: TeamMetrics, story: Dict) -> str:
    """Generate data-specific matchup analysis"""
    
    if story["name"] == "NO_GOAL_PRESENT":
        dead = home if home.attack_type == "NO_GOAL" else away
        live = away if home.attack_type == "NO_GOAL" else home
        return f"""
**MATCHUP:**
- {dead.name} has scored in only {dead.scored_05:.0f}% of games. At this venue, they blank {dead.fts_pct:.0f}% of the time. They have {'never' if dead.scored_35 == 0 else 'rarely'} scored 3+ goals.
- {live.name} scores in {live.scored_05:.0f}% of games and concedes in {live.conceded_05:.0f}%.
- {live.name} only needs 1 goal. {dead.name} is unlikely to respond.
"""
    
    elif story["name"] == "EXTREME_COLLAPSE_DEFENSE":
        collapse = home if home.defense_type == "EXTREME_COLLAPSE" else away
        opponent = away if home.defense_type == "EXTREME_COLLAPSE" else home
        return f"""
**MATCHUP:**
- {collapse.name} concedes in {collapse.conceded_05:.0f}% of games. When breached, they concede multiple — only a {collapse.conceded_05 - collapse.conceded_15:.0f}% drop to O1.5.
- {opponent.name} scores in {opponent.scored_05:.0f}% of games and scores 2+ in {opponent.scored_15:.0f}%.
- Once the first goal goes in against {collapse.name}, more follow. {opponent.name} has the tools to open the floodgates.
"""
    
    elif story["name"] == "MISMATCHED_DEFENSES":
        elite = home if home.conceded_05 < away.conceded_05 else away
        weak = away if home.conceded_05 < away.conceded_05 else home
        return f"""
**MATCHUP:**
- {elite.name} defense concedes only {elite.conceded_05:.0f}% of games with a {elite.cs_pct:.0f}% clean sheet rate.
- {weak.name} defense concedes {weak.conceded_05:.0f}% of games.
- Gap between defenses: {abs(home.conceded_05 - away.conceded_05):.0f}%. {elite.name} should control this match.
"""
    
    elif story["name"] == "COLLAPSE_VS_SCORER":
        collapse = home if home.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] else away
        scorer = away if home.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] else home
        return f"""
**MATCHUP:**
- {collapse.name} defense collapses when breached (only {collapse.conceded_05 - collapse.conceded_15:.0f}% drop to O1.5).
- {scorer.name} attack scores in {scorer.scored_05:.0f}% of games and can score 2+ in {scorer.scored_15:.0f}%.
- {scorer.name} should exploit {collapse.name}'s defensive frailties. Goals expected.
"""
    
    elif story["name"] == "TWO_ONE_GOAL_ATTACKS":
        return f"""
**MATCHUP:**
- Both teams typically score exactly 1 goal. {home.name} scores 2+ in only {home.scored_15:.0f}%. {away.name} scores 2+ in only {away.scored_15:.0f}%.
- Neither attack can run away with this match. Expect a tight, low-scoring affair.
"""
    
    elif story["name"] == "ONE_GOAL_VS_COLLAPSE":
        one_goal = home if home.attack_type == "ONE_GOAL" else away
        collapse = away if home.attack_type == "ONE_GOAL" else home
        return f"""
**MATCHUP:**
- {one_goal.name} attack is limited — scores in {one_goal.scored_05:.0f}% but rarely 2+ (only {one_goal.scored_15:.0f}%).
- {collapse.name} defense concedes {collapse.conceded_05:.0f}% and concedes multiple often.
- {collapse.name} will concede, but {one_goal.name} may only score exactly 1. Total capped at 2-3 goals.
"""
    
    elif story["name"] == "BINARY_HOME_TEAM":
        return f"""
**MATCHUP:**
- {home.name} is binary at home. CS {home.cs_pct:.0f}% of games or concedes multiple. Scores in {home.scored_05:.0f}% or blanks.
- This is the most unpredictable home profile. Trust only the strongest signals.
"""
    
    else:
        return f"""
**MATCHUP:**
- No clear pattern. Both teams show mixed signals. Proceed with caution.
"""


def generate_risk_notes(home: TeamMetrics, away: TeamMetrics, story: Dict) -> List[str]:
    """Generate story-specific risk notes"""
    risks = []
    
    if story["name"] == "NO_GOAL_PRESENT":
        dead = home if home.attack_type == "NO_GOAL" else away
        risks.append(f"{dead.name} could still score a fluke goal ({dead.scored_05:.0f}% chance). BTTS No is not a lock.")
    elif story["name"] == "EXTREME_COLLAPSE_DEFENSE":
        risks.append("Collapse requires the first goal to happen. 0-0 and the collapse never triggers.")
    elif story["name"] == "MISMATCHED_DEFENSES":
        risks.append("The weak defense could overperform. Mismatches sometimes produce surprises.")
    elif story["name"] == "COLLAPSE_VS_SCORER":
        risks.append("If the collapse defense scores first, the match shape changes entirely.")
    elif story["name"] == "TWO_ONE_GOAL_ATTACKS":
        risks.append("A 2-2 is unlikely but possible if both teams hit their rare 2+ goal games simultaneously.")
    elif story["name"] == "BINARY_HOME_TEAM":
        risks.append(f"{home.name} could produce any result. Binary teams are high-variance.")
    
    # Add 2-2 risk if both score 2+ with decent probability
    if home.scored_15 >= 35 and away.scored_15 >= 35:
        both_score_2_plus = (home.scored_15 * away.scored_15) / 100
        risks.append(f"Both teams score 2+ in {both_score_2_plus:.1f}% of games → 2-2, 3-2, 2-3 are live risks.")
    
    return risks


def generate_scoreline_range(home: TeamMetrics, away: TeamMetrics, story: Dict) -> List[str]:
    """Generate story-specific scoreline range"""
    scorelines = {
        "NO_GOAL_PRESENT": ["1-0", "2-0", "0-0", "0-1"],
        "EXTREME_COLLAPSE_DEFENSE": ["2-1", "3-1", "2-2", "3-2"],
        "MISMATCHED_DEFENSES": ["1-0", "2-0", "0-0", "0-1"],
        "COLLAPSE_VS_SCORER": ["2-1", "3-1", "2-2", "3-2"],
        "TWO_ONE_GOAL_ATTACKS": ["1-1", "2-1", "1-2", "1-0"],
        "ONE_GOAL_VS_COLLAPSE": ["2-1", "1-1", "2-0", "1-0"],
        "BINARY_HOME_TEAM": ["2-0", "0-0", "2-1", "0-1"],
    }
    return scorelines.get(story["name"], ["1-1", "2-1", "1-2"])


def generate_warnings(matched_stories: List[Dict], home: TeamMetrics, away: TeamMetrics) -> List[str]:
    """Generate warnings based on sample size and other issues"""
    warnings = []
    
    min_games = min(home.games_played, away.games_played)
    if min_games < 8:
        warnings.append(f"⚠️ Small sample size ({min_games} games). All confidence levels reduced by 1 tier.")
    elif min_games < 15:
        warnings.append(f"⚠️ Moderate sample size ({min_games} games). Confidence slightly reduced.")
    
    if home.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"] or away.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"]:
        warnings.append("⚠️ xG gap detected. Regression adjustments applied to affected bets.")
    
    if check_binary(home.cs_pct, home.fts_pct):
        warnings.append(f"⚠️ {home.name} is binary at home. High variance expected. Any result possible.")
    
    return warnings


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================
def predict_match(home: TeamMetrics, away: TeamMetrics) -> AnalysisOutput:
    """Run the complete 5-layer prediction pipeline"""
    
    # LAYER 1: Sample Validator
    sample_validation = validate_sample(home.games_played, away.games_played)
    
    if not sample_validation["valid"]:
        return AnalysisOutput(
            story_name="INSUFFICIENT_DATA",
            story_description="Not enough data to make a reliable prediction.",
            narrative=f"⚠️ {sample_validation['warning']}",
            bets=[],
            avoid_bets=["ALL MARKETS"],
            risks=["Data quality insufficient. PASS on all markets."],
            scorelines=["Unknown"],
            warnings=[sample_validation["warning"]],
            coexistence_notes=[]
        )
    
    # LAYER 2: Shape Classifier
    home = classify_team(home)
    away = classify_team(away)
    
    # LAYER 3: Story Matcher
    matched_stories = match_stories(home, away)
    primary_story = matched_stories[0]
    
    # LAYER 4: Bet Generator (DATA-AWARE)
    bets, avoid_bets = generate_bets(
        home, away, primary_story["name"], 
        sample_validation["confidence_multiplier"],
        matched_stories
    )
    
    # LAYER 5: Post-Processing
    bets = apply_xg_adjustments(bets, home, away)
    coexistence_notes = check_coexistence(bets)
    
    # Sort bets by tier (1 is best)
    bets.sort(key=lambda x: x.tier)
    
    # Generate output
    narrative = f"""
**STORY:** {primary_story['name'].replace('_', ' ').title()}

**{primary_story['description']}**

{generate_shape_description(home, away)}

{generate_matchup_analysis(home, away, primary_story)}
"""
    
    risks = generate_risk_notes(home, away, primary_story)
    scorelines = generate_scoreline_range(home, away, primary_story)
    warnings = generate_warnings(matched_stories, home, away)
    
    return AnalysisOutput(
        story_name=primary_story['name'],
        story_description=primary_story['description'],
        narrative=narrative,
        bets=bets,
        avoid_bets=avoid_bets,
        risks=risks,
        scorelines=scorelines,
        warnings=warnings,
        coexistence_notes=coexistence_notes
    )


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def metric_input(team_name: str, prefix: str) -> TeamMetrics:
    """Create input fields for team metrics"""
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Distribution %**")
        scored_05 = st.number_input("Scored O0.5 %", 0, 100, 80, 5, key=f"{prefix}_s05")
        scored_15 = st.number_input("Scored O1.5 %", 0, 100, 38, 5, key=f"{prefix}_s15")
        scored_25 = st.number_input("Scored O2.5 %", 0, 100, 15, 5, key=f"{prefix}_s25")
        scored_35 = st.number_input("Scored O3.5 %", 0, 100, 0, 5, key=f"{prefix}_s35")
    
    with col2:
        st.markdown("**🛡️ Defensive %**")
        conceded_05 = st.number_input("Conceded O0.5 %", 0, 100, 77, 5, key=f"{prefix}_c05")
        conceded_15 = st.number_input("Conceded O1.5 %", 0, 100, 40, 5, key=f"{prefix}_c15")
        conceded_25 = st.number_input("Conceded O2.5 %", 0, 100, 10, 5, key=f"{prefix}_c25")
        conceded_35 = st.number_input("Conceded O3.5 %", 0, 100, 0, 5, key=f"{prefix}_c35")
    
    with col3:
        st.markdown("**📈 Match %**")
        btts = st.number_input("BTTS %", 0, 100, 54, 5, key=f"{prefix}_btts")
        over_25 = st.number_input("Over 2.5 %", 0, 100, 46, 5, key=f"{prefix}_o25")
        over_35 = st.number_input("Over 3.5 %", 0, 100, 27, 5, key=f"{prefix}_o35")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**🏟️ Venue**")
        fts_pct = st.number_input("Failed to Score %", 0, 100, 23, 5, key=f"{prefix}_fts")
        cs_pct = st.number_input("Clean Sheet %", 0, 100, 23, 5, key=f"{prefix}_cs")
        games = st.number_input("Games Played", 0, 50, 15, 1, key=f"{prefix}_gp")
    
    with col5:
        st.markdown("**🎯 xG**")
        xg = st.number_input("xG per game", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_xg")
        actual = st.number_input("Actual Scored", 0.0, 3.0, 1.2, 0.1, key=f"{prefix}_act")
    
    with col6:
        st.markdown("**📊 More %**")
        over_15 = st.number_input("Over 1.5 %", 0, 100, 77, 5, key=f"{prefix}_o15")
        btts_over25 = st.number_input("BTTS & O2.5 %", 0, 100, 46, 5, key=f"{prefix}_btts_o25")
        btts_no_over25 = st.number_input("BTTS No & O2.5 %", 0, 100, 0, 5, key=f"{prefix}_btts_no")
    
    return TeamMetrics(
        name=team_name,
        scored_05=float(scored_05), scored_15=float(scored_15),
        scored_25=float(scored_25), scored_35=float(scored_35),
        conceded_05=float(conceded_05), conceded_15=float(conceded_15),
        conceded_25=float(conceded_25), conceded_35=float(conceded_35),
        btts=float(btts), over_15=float(over_15), over_25=float(over_25),
        over_35=float(over_35), btts_and_over25=float(btts_over25),
        btts_no_and_over25=float(btts_no_over25),
        fts_pct=float(fts_pct), cs_pct=float(cs_pct),
        games_played=int(games), xg=float(xg), actual_scored=float(actual)
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("Story Engine | 5-Layer Architecture | Data-Aware Confidence")
    
    st.markdown("""
    <div class="section-header">🏗️ 5-LAYER PROCESSING ENGINE</div>
    <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <div style="background:#1e293b;padding:0.5rem;border-radius:8px;flex:1;text-align:center;">
            <strong>Layer 1</strong><br>Sample Validator
        </div>
        <div style="background:#1e293b;padding:0.5rem;border-radius:8px;flex:1;text-align:center;">
            <strong>Layer 2</strong><br>Shape Classifier
        </div>
        <div style="background:#1e293b;padding:0.5rem;border-radius:8px;flex:1;text-align:center;">
            <strong>Layer 3</strong><br>Story Matcher
        </div>
        <div style="background:#1e293b;padding:0.5rem;border-radius:8px;flex:1;text-align:center;">
            <strong>Layer 4</strong><br>Bet Generator
        </div>
        <div style="background:#1e293b;padding:0.5rem;border-radius:8px;flex:1;text-align:center;">
            <strong>Layer 5</strong><br>Confidence Assigner
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team", "AC Pisa 1909", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team", "Genoa CFC", key="away_name")
    
    st.divider()
    
    st.subheader(f"🏠 {home_name}")
    home_metrics = metric_input(home_name, "home")
    
    st.divider()
    
    st.subheader(f"✈️ {away_name}")
    away_metrics = metric_input(away_name, "away")
    
    st.divider()
    
    if st.button("🔮 ANALYZE MATCH", type="primary"):
        result = predict_match(home_metrics, away_metrics)
        
        # STORY CARD
        st.markdown(f"""
        <div class="story-card">
            <div style="font-size:1.2rem;font-weight:700;margin-bottom:0.5rem;">
                📖 STORY: {result.story_name.replace('_', ' ').title()}
            </div>
            <div style="color:#FFFFFF;margin-bottom:1rem;">
                {result.story_description}
            </div>
            <div style="white-space:pre-line;font-size:0.9rem;">
                {result.narrative}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # WARNINGS
        if result.warnings:
            for w in result.warnings:
                st.warning(w)
        
        # RECOMMENDED BETS
        if result.bets:
            st.markdown("### 🎯 RECOMMENDED BETS")
            for bet in result.bets:
                tier_class = f"tier-{bet.tier}"
                conf_class = {
                    "VERY HIGH": "confidence-very-high",
                    "HIGH": "confidence-high",
                    "MEDIUM": "confidence-medium",
                    "LOW": "confidence-low",
                    "PASS": "confidence-pass"
                }.get(bet.confidence, "")
                
                st.markdown(f"""
                <div class="bet-card {tier_class}">
                    <div style="flex:2;">
                        <strong>{bet.market}</strong>
                        <span style="font-size:1.1rem;font-weight:bold;margin-left:0.5rem;">→ {bet.bet}</span>
                        <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{bet.reasoning}</div>
                    </div>
                    <div style="flex:1;text-align:right;">
                        <span class="{conf_class}">{bet.confidence}</span>
                        <span style="background:#0f172a;padding:0.2rem 0.5rem;border-radius:12px;margin-left:0.5rem;">TIER {bet.tier}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No bets generated. PASS on this match.")
        
        # COEXISTENCE NOTES
        if result.coexistence_notes:
            for note in result.coexistence_notes:
                st.markdown(f'<div class="coexistence-note">{note}</div>', unsafe_allow_html=True)
        
        # AVOID BETS
        if result.avoid_bets:
            st.markdown("### ⛔ MARKETS TO AVOID")
            for avoid in result.avoid_bets:
                st.markdown(f'<div class="avoid-card">🚫 {avoid}</div>', unsafe_allow_html=True)
        
        # SCORELINE RANGE
        if result.scorelines:
            st.markdown(f"""
            <div style="background:#0f172a;border-radius:12px;padding:1rem;margin:1rem 0;">
                <strong>🎯 Most Likely Scorelines:</strong><br>
                {' | '.join(result.scorelines)}
            </div>
            """, unsafe_allow_html=True)
        
        # RISK NOTES
        if result.risks:
            st.markdown("### ⚠️ RISK NOTES")
            for risk in result.risks:
                st.markdown(f'<div class="risk-note">⚠️ {risk}</div>', unsafe_allow_html=True)
    
    # FOOTER
    st.divider()
    st.markdown("""
    ### 📋 Story Catalog
    
    | # | Story | Priority | Description |
    |---|-------|----------|-------------|
    | 1 | NO_GOAL_PRESENT | 1 | At least one attack is dead. Goals scarce. |
    | 2 | EXTREME_COLLAPSE_DEFENSE | 2 | Defense collapses when breached. |
    | 3 | MISMATCHED_DEFENSES | 3 | One defense far superior. |
    | 4 | COLLAPSE_VS_SCORER | 4 | Collapse defense meets scoring attack. |
    | 5 | TWO_ONE_GOAL_ATTACKS | 5 | Both attacks limited to 1 goal. |
    | 6 | ONE_GOAL_VS_COLLAPSE | 6 | One-goal attack meets collapse defense. |
    | 7 | BINARY_HOME_TEAM | 7 | Home team unpredictable. |
    | 8 | REGRESSION_CANDIDATE | 8 | xG gap detected. |
    
    ### 🎯 Confidence Tiers
    
    | Tier | Label | Criteria |
    |------|-------|----------|
    | 1 | VERY HIGH | Multiple 0%/100% signals + large sample |
    | 2 | HIGH | One 0%/100% signal + supporting data |
    | 3 | MEDIUM | Clear trend but no extreme signal |
    | 4 | LOW | Mixed signals, story favors this side |
    | 5 | PASS | No edge or high variance |
    """)

if __name__ == "__main__":
    main()
