"""
Streak Predictor - Complete 5-Layer Architecture
Based on the exact specification.

LAYER 1: Sample Validator
LAYER 2: Shape Classifier
LAYER 3: Story Matcher
LAYER 4: Bet Generator
LAYER 5: Confidence Assigner

OUTPUT: Story Narrative + Ranked Bets + Risk Notes + Scoreline Range
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Streak Predictor - 5-Layer Architecture",
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
    .confidence-very-high { color: #10b981; }
    .confidence-high { color: #fbbf24; }
    .confidence-medium { color: #f97316; }
    .confidence-low { color: #ef4444; }
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
    fts_pct: float = 0.0  # Failed to Score %
    cs_pct: float = 0.0   # Clean Sheet %
    games_played: int = 0
    # xG data
    xg: float = 0.0
    actual_scored: float = 0.0
    
    # Derived fields (set by Layer 2)
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
    risks: List[str]
    scorelines: List[str]
    warnings: List[str]


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
    gap = abs(home_concede_05 - away_concede_05)
    return gap >= 40


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
        "description": "A defense that collapses when breached. Overs are live.",
        "condition": lambda h, a: h.defense_type == "EXTREME_COLLAPSE" or a.defense_type == "EXTREME_COLLAPSE"
    },
    "MISMATCHED_DEFENSES": {
        "priority": 3,
        "description": "One defense is far superior. Trust the elite defense.",
        "condition": lambda h, a: check_mismatch(h.conceded_05, a.conceded_05)
    },
    "COLLAPSE_VS_SCORER": {
        "priority": 4,
        "description": "Collapse defense meets an attack that can score. Over match.",
        "condition": lambda h, a: (h.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] and a.attack_type in ["SCORES_FREELY", "MODERATE_SCORER"]) or \
                                   (a.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"] and h.attack_type in ["SCORES_FREELY", "MODERATE_SCORER"])
    },
    "TWO_ONE_GOAL_ATTACKS": {
        "priority": 5,
        "description": "Both attacks are one-goal. Neither scores multiple. Under match.",
        "condition": lambda h, a: h.attack_type == "ONE_GOAL" and a.attack_type == "ONE_GOAL"
    },
    "ONE_GOAL_VS_COLLAPSE": {
        "priority": 6,
        "description": "One-goal attack meets collapse defense. The defense concedes but attack may not score 2+.",
        "condition": lambda h, a: (h.attack_type == "ONE_GOAL" and a.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"]) or \
                                   (a.attack_type == "ONE_GOAL" and h.defense_type in ["COLLAPSE", "EXTREME_COLLAPSE"])
    },
    "BINARY_HOME_TEAM": {
        "priority": 7,
        "description": "Home team is binary. Either blank or score multiple. Either CS or concede multiple.",
        "condition": lambda h, a: check_binary(h.cs_pct, h.fts_pct)
    },
    "REGRESSION_CANDIDATE": {
        "priority": 8,
        "description": "xG gap detected. Regression likely. Adjust confidence accordingly.",
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
            "description": "No clear pattern detected.",
            "priority": 99
        })
    
    return matched


# ============================================================================
# LAYER 4: BET GENERATOR
# ============================================================================
STORY_BETS = {
    "NO_GOAL_PRESENT": {
        "primary": [
            {"market": "BTTS", "bet": "No", "base_tier": 2},
            {"market": "Under 2.5 Goals", "bet": "Under 2.5", "base_tier": 2},
            {"market": "Under 3.5 Goals", "bet": "Under 3.5", "base_tier": 1}
        ],
        "secondary": [],
        "avoid": ["Over 2.5 Goals", "BTTS Yes"]
    },
    "EXTREME_COLLAPSE_DEFENSE": {
        "primary": [
            {"market": "Collapse Team Clean Sheet", "bet": "No", "base_tier": 1},
            {"market": "BTTS", "bet": "Yes", "base_tier": 2},
            {"market": "Over 2.5 Goals", "bet": "Over 2.5", "base_tier": 2}
        ],
        "secondary": [
            {"market": "Opponent Team Total O1.5", "bet": "Over 1.5", "base_tier": 3}
        ],
        "avoid": ["Under 2.5 Goals", "Under 3.5 Goals"]
    },
    "MISMATCHED_DEFENSES": {
        "primary": [
            {"market": "Elite Defense Clean Sheet", "bet": "Yes", "base_tier": 2},
            {"market": "BTTS", "bet": "No", "base_tier": 2},
            {"market": "Under 2.5 Goals", "bet": "Under 2.5", "base_tier": 2}
        ],
        "secondary": [],
        "avoid": ["BTTS Yes", "Over 2.5 Goals"]
    },
    "COLLAPSE_VS_SCORER": {
        "primary": [
            {"market": "BTTS", "bet": "Yes", "base_tier": 2},
            {"market": "Over 2.5 Goals", "bet": "Over 2.5", "base_tier": 2},
            {"market": "Scorer Team Total O1.5", "bet": "Over 1.5", "base_tier": 3}
        ],
        "secondary": [],
        "avoid": ["Under 2.5 Goals"]
    },
    "TWO_ONE_GOAL_ATTACKS": {
        "primary": [
            {"market": "Under 2.5 Goals", "bet": "Under 2.5", "base_tier": 2},
            {"market": "Under 3.5 Goals", "bet": "Under 3.5", "base_tier": 1},
            {"market": "BTTS", "bet": "No", "base_tier": 3}
        ],
        "secondary": [],
        "avoid": ["Over 2.5 Goals"]
    },
    "ONE_GOAL_VS_COLLAPSE": {
        "primary": [
            {"market": "Collapse Team Clean Sheet", "bet": "No", "base_tier": 2},
            {"market": "Under 3.5 Goals", "bet": "Under 3.5", "base_tier": 2},
            {"market": "BTTS", "bet": "Yes", "base_tier": 3}
        ],
        "secondary": [],
        "avoid": ["Over 3.5 Goals"]
    },
    "BINARY_HOME_TEAM": {
        "primary": [
            {"market": "Home Team Clean Sheet", "bet": "No", "base_tier": 3},
            {"market": "Home Team Total O1.5", "bet": "Over 1.5", "base_tier": 3}
        ],
        "secondary": [],
        "avoid": []
    },
    "REGRESSION_CANDIDATE": {
        "primary": [],
        "secondary": [],
        "avoid": []
    },
    "NO_CLEAR_STORY": {
        "primary": [],
        "secondary": [],
        "avoid": []
    }
}


def generate_bets(story_name: str, home: TeamMetrics, away: TeamMetrics, sample_multiplier: float) -> List[Bet]:
    story_config = STORY_BETS.get(story_name, STORY_BETS["NO_CLEAR_STORY"])
    bets = []
    
    # Add primary bets
    for bet_config in story_config.get("primary", []):
        base_tier = bet_config["base_tier"]
        
        # Apply sample multiplier adjustment
        if sample_multiplier == 0.7:
            base_tier += 1
        elif sample_multiplier == 0.9:
            base_tier += 0  # No change, but confidence label may change
        elif sample_multiplier == 0.0:
            base_tier = 5  # PASS
        
        # Cap at 5
        base_tier = min(base_tier, 5)
        
        # Determine confidence label
        confidence_map = {1: "VERY HIGH", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "PASS"}
        
        bets.append(Bet(
            market=bet_config["market"],
            bet=bet_config["bet"],
            tier=base_tier,
            confidence=confidence_map.get(base_tier, "UNKNOWN"),
            reasoning="Generated from story logic"
        ))
    
    return bets


# ============================================================================
# LAYER 5: CONFIDENCE ASSIGNER (Integrated into generate_bets)
# ============================================================================
def tier_label(tier: int) -> Tuple[str, str]:
    return {
        1: ("VERY HIGH", "🟢"),
        2: ("HIGH", "🟡"),
        3: ("MEDIUM", "🟠"),
        4: ("LOW", "🔴"),
        5: ("PASS", "⚪")
    }.get(tier, ("UNKNOWN", "⚪"))


# ============================================================================
# NARRATIVE GENERATOR
# ============================================================================
def generate_shape_description(home: TeamMetrics, away: TeamMetrics) -> str:
    return f"""
    **SHAPE ANALYSIS:**
    - {home.name}: {home.attack_type} attack | {home.defense_type} defense | {home.xg_type} xG
    - {away.name}: {away.attack_type} attack | {away.defense_type} defense | {away.xg_type} xG
    """


def generate_matchup_analysis(home: TeamMetrics, away: TeamMetrics, story: Dict) -> str:
    if story["name"] == "NO_GOAL_PRESENT":
        dead_team = home.name if home.attack_type == "NO_GOAL" else away.name
        return f"**MATCHUP:** {dead_team} cannot score. The other team only needs 1 goal to win. 1-0, 2-0 type game."
    
    elif story["name"] == "EXTREME_COLLAPSE_DEFENSE":
        collapse_team = home.name if home.defense_type == "EXTREME_COLLAPSE" else away.name
        return f"**MATCHUP:** {collapse_team}'s defense collapses when breached. Once the first goal goes in, more follow."
    
    elif story["name"] == "MISMATCHED_DEFENSES":
        elite_team = home.name if home.conceded_05 < away.conceded_05 else away.name
        return f"**MATCHUP:** {elite_team} has the elite defense. Trust them to control the game."
    
    elif story["name"] == "COLLAPSE_VS_SCORER":
        return f"**MATCHUP:** Collapse defense meets an attack that can score. Goals expected."
    
    elif story["name"] == "TWO_ONE_GOAL_ATTACKS":
        return f"**MATCHUP:** Both teams score 1 goal typically. 1-1, 2-1 type game. Neither runs away with it."
    
    elif story["name"] == "ONE_GOAL_VS_COLLAPSE":
        return f"**MATCHUP:** One-goal attack meets collapse defense. The defense concedes but attack may not score 2+."
    
    elif story["name"] == "BINARY_HOME_TEAM":
        return f"**MATCHUP:** {home.name} is binary. Either blank or score multiple. Either CS or concede multiple."
    
    else:
        return "**MATCHUP:** No clear pattern. Proceed with caution."


def generate_risk_notes(home: TeamMetrics, away: TeamMetrics, story: Dict) -> List[str]:
    risks = []
    
    if story["name"] == "NO_GOAL_PRESENT":
        risks.append("The dead attack could still score a fluke goal. Not a lock.")
    elif story["name"] == "EXTREME_COLLAPSE_DEFENSE":
        risks.append("Collapse requires the first goal to happen. If it stays 0-0, the collapse never triggers.")
    elif story["name"] == "MISMATCHED_DEFENSES":
        risks.append("The inferior defense could play above their level. Unlikely but possible.")
    elif story["name"] == "COLLAPSE_VS_SCORER":
        risks.append("If the collapse defense scores first, the match changes shape.")
    
    # Add 2-2 risk if both teams have decent scoring
    if home.scored_15 >= 35 and away.scored_15 >= 35:
        both_score_2_plus = (home.scored_15 * away.scored_15) / 100
        risks.append(f"Both teams score 2+ in {both_score_2_plus:.1f}% of games → 2-2, 3-2, 2-3 are live risks")
    
    return risks


def generate_scoreline_range(home: TeamMetrics, away: TeamMetrics, story: Dict) -> List[str]:
    if story["name"] == "NO_GOAL_PRESENT":
        return ["1-0", "2-0", "0-0"]
    elif story["name"] == "EXTREME_COLLAPSE_DEFENSE":
        return ["2-1", "3-1", "2-2"]
    elif story["name"] == "MISMATCHED_DEFENSES":
        return ["1-0", "2-0", "0-0"]
    elif story["name"] == "COLLAPSE_VS_SCORER":
        return ["2-1", "3-1", "2-2"]
    elif story["name"] == "TWO_ONE_GOAL_ATTACKS":
        return ["1-1", "2-1", "1-2"]
    elif story["name"] == "ONE_GOAL_VS_COLLAPSE":
        return ["2-1", "1-1", "2-0"]
    elif story["name"] == "BINARY_HOME_TEAM":
        return ["2-0", "0-0", "2-1"]
    else:
        return ["1-1", "2-1", "1-2"]


def generate_warnings(matched_stories: List[Dict], home: TeamMetrics, away: TeamMetrics) -> List[str]:
    warnings = []
    
    # Check sample size warning
    if home.games_played < 8 or away.games_played < 8:
        warnings.append(f"⚠️ Small sample size ({min(home.games_played, away.games_played)} games). Confidence reduced.")
    
    # Check xG regression
    if home.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"] or away.xg_type in ["UNDERPERFORMING", "OVERPERFORMING"]:
        warnings.append(f"⚠️ xG gap detected. Regression likely. Adjust expectations.")
    
    # Check binary warning
    if check_binary(home.cs_pct, home.fts_pct):
        warnings.append(f"⚠️ {home.name} is binary. High variance expected.")
    
    return warnings


def generate_output(home: TeamMetrics, away: TeamMetrics, matched_stories: List[Dict], bets: List[Bet]) -> AnalysisOutput:
    primary_story = matched_stories[0]
    
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
        risks=risks,
        scorelines=scorelines,
        warnings=warnings
    )


# ============================================================================
# MAIN PREDICTION PIPELINE
# ============================================================================
def predict_match(home: TeamMetrics, away: TeamMetrics) -> AnalysisOutput:
    # LAYER 1: Sample Validator
    sample_validation = validate_sample(home.games_played, away.games_played)
    
    if not sample_validation["valid"]:
        return AnalysisOutput(
            story_name="INSUFFICIENT_DATA",
            story_description="Not enough data to make a reliable prediction.",
            narrative=f"⚠️ {sample_validation['warning']}",
            bets=[],
            risks=["Data quality insufficient. PASS on all markets."],
            scorelines=["Unknown"],
            warnings=[sample_validation["warning"]]
        )
    
    # LAYER 2: Shape Classifier
    home = classify_team(home)
    away = classify_team(away)
    
    # LAYER 3: Story Matcher
    matched_stories = match_stories(home, away)
    
    # LAYER 4: Bet Generator
    bets = generate_bets(matched_stories[0]["name"], home, away, sample_validation["confidence_multiplier"])
    
    # LAYER 5: Confidence Assigner (integrated)
    
    # Generate Output
    output = generate_output(home, away, matched_stories, bets)
    
    return output


# ============================================================================
# UI HELPER FUNCTIONS
# ============================================================================
def metric_input(team_name: str, prefix: str, is_home: bool = True) -> TeamMetrics:
    st.markdown(f"<div class='team-header'><span class='team-name'>{team_name}</span></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Distribution %**")
        scored_05 = st.number_input("Scored O0.5 %", min_value=0, max_value=100, value=80, step=5, key=f"{prefix}_scored_05")
        scored_15 = st.number_input("Scored O1.5 %", min_value=0, max_value=100, value=38, step=5, key=f"{prefix}_scored_15")
        scored_25 = st.number_input("Scored O2.5 %", min_value=0, max_value=100, value=15, step=5, key=f"{prefix}_scored_25")
        scored_35 = st.number_input("Scored O3.5 %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_scored_35")
    
    with col2:
        st.markdown("**🛡️ Defensive %**")
        conceded_05 = st.number_input("Conceded O0.5 %", min_value=0, max_value=100, value=77, step=5, key=f"{prefix}_conceded_05")
        conceded_15 = st.number_input("Conceded O1.5 %", min_value=0, max_value=100, value=40, step=5, key=f"{prefix}_conceded_15")
        conceded_25 = st.number_input("Conceded O2.5 %", min_value=0, max_value=100, value=10, step=5, key=f"{prefix}_conceded_25")
        conceded_35 = st.number_input("Conceded O3.5 %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_conceded_35")
    
    with col3:
        st.markdown("**📈 Match % & Context**")
        btts = st.number_input("BTTS %", min_value=0, max_value=100, value=54, step=5, key=f"{prefix}_btts")
        over_25 = st.number_input("Over 2.5 %", min_value=0, max_value=100, value=46, step=5, key=f"{prefix}_over25")
        btts_no_over25 = st.number_input("BTTS No & O2.5 %", min_value=0, max_value=100, value=0, step=5, key=f"{prefix}_btts_no_over25")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("**🏟️ Venue Metrics**")
        fts_pct = st.number_input("Failed to Score %", min_value=0, max_value=100, value=23, step=5, key=f"{prefix}_fts")
        cs_pct = st.number_input("Clean Sheet %", min_value=0, max_value=100, value=23, step=5, key=f"{prefix}_cs")
        games_played = st.number_input("Games Played", min_value=0, max_value=50, value=15, step=1, key=f"{prefix}_games")
    
    with col5:
        st.markdown("**🎯 xG Data (Optional)**")
        xg = st.number_input("xG per game", min_value=0.0, max_value=3.0, value=1.2, step=0.1, key=f"{prefix}_xg")
        actual_scored = st.number_input("Actual Scored per game", min_value=0.0, max_value=3.0, value=1.2, step=0.1, key=f"{prefix}_actual")
    
    with col6:
        st.markdown("**📊 Match % (cont.)**")
        over_15 = st.number_input("Over 1.5 %", min_value=0, max_value=100, value=77, step=5, key=f"{prefix}_over15")
        over_35 = st.number_input("Over 3.5 %", min_value=0, max_value=100, value=27, step=5, key=f"{prefix}_over35")
        btts_and_over25 = st.number_input("BTTS & O2.5 %", min_value=0, max_value=100, value=46, step=5, key=f"{prefix}_btts_over25")
    
    return TeamMetrics(
        name=team_name,
        scored_05=float(scored_05),
        scored_15=float(scored_15),
        scored_25=float(scored_25),
        scored_35=float(scored_35),
        conceded_05=float(conceded_05),
        conceded_15=float(conceded_15),
        conceded_25=float(conceded_25),
        conceded_35=float(conceded_35),
        btts=float(btts),
        over_15=float(over_15),
        over_25=float(over_25),
        over_35=float(over_35),
        btts_and_over25=float(btts_and_over25),
        btts_no_and_over25=float(btts_no_over25),
        fts_pct=float(fts_pct),
        cs_pct=float(cs_pct),
        games_played=int(games_played),
        xg=float(xg),
        actual_scored=float(actual_scored)
    )


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor")
    st.caption("5-Layer Architecture | Story-Driven Predictions")
    
    st.markdown("""
    <div class="section-header">
    🏗️ PROCESSING ENGINE (5 Layers)
    </div>
    <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Layer 1</strong><br>Sample Validator
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Layer 2</strong><br>Shape Classifier
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Layer 3</strong><br>Story Matcher
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Layer 4</strong><br>Bet Generator
        </div>
        <div style="background: #1e293b; padding: 0.5rem; border-radius: 8px; flex: 1; text-align: center;">
            <strong>Layer 5</strong><br>Confidence Assigner
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("🏠 Home Team", "Home Team", key="home_name")
    with col2:
        away_name = st.text_input("✈️ Away Team", "Away Team", key="away_name")
    
    st.divider()
    
    st.subheader(f"🏠 {home_name} - Team Metrics")
    home_metrics = metric_input(home_name, "home")
    
    st.divider()
    
    st.subheader(f"✈️ {away_name} - Team Metrics")
    away_metrics = metric_input(away_name, "away")
    
    st.divider()
    
    if st.button("🔮 RUN ANALYSIS", type="primary"):
        result = predict_match(home_metrics, away_metrics)
        
        # Display Story Card
        st.markdown(f"""
        <div class="story-card">
            <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">
                📖 {result.story_name.replace('_', ' ').title()}
            </div>
            <div style="color: #94a3b8; margin-bottom: 1rem;">
                {result.story_description}
            </div>
            <div style="white-space: pre-line; font-size: 0.9rem;">
                {result.narrative}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Bets
        if result.bets:
            st.markdown("### 🎯 RECOMMENDED BETS")
            for bet in result.bets:
                tier_class = f"tier-{bet.tier}"
                confidence_class = {
                    "VERY HIGH": "confidence-very-high",
                    "HIGH": "confidence-high",
                    "MEDIUM": "confidence-medium",
                    "LOW": "confidence-low",
                    "PASS": ""
                }.get(bet.confidence, "")
                
                st.markdown(f"""
                <div class="bet-card {tier_class}">
                    <div>
                        <strong>{bet.market}</strong>
                        <span style="font-size: 1.1rem; font-weight: bold; margin-left: 0.5rem;">{bet.bet}</span>
                    </div>
                    <div>
                        <span class="{confidence_class}">{bet.confidence}</span>
                        <span style="background: #0f172a; padding: 0.2rem 0.5rem; border-radius: 12px; margin-left: 0.5rem;">TIER {bet.tier}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No clear bets generated. Consider passing this match.")
        
        # Display Scoreline Range
        if result.scorelines:
            st.markdown(f"""
            <div style="background: #0f172a; border-radius: 12px; padding: 1rem; margin: 1rem 0;">
                <strong>🎯 Most Likely Scorelines:</strong><br>
                {' | '.join(result.scorelines)}
            </div>
            """, unsafe_allow_html=True)
        
        # Display Risk Notes
        if result.risks:
            st.markdown("### ⚠️ RISK NOTES")
            for risk in result.risks:
                st.markdown(f'<div class="risk-note">⚠️ {risk}</div>', unsafe_allow_html=True)
        
        # Display Warnings
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
    
    st.divider()
    st.markdown("""
    ### 📋 Story Catalog
    
    | # | Story Name | Priority | Description |
    |---|-----------|----------|-------------|
    | 1 | NO_GOAL_PRESENT | 1 | At least one attack is dead. Goals scarce. |
    | 2 | EXTREME_COLLAPSE_DEFENSE | 2 | Defense collapses when breached. Overs live. |
    | 3 | MISMATCHED_DEFENSES | 3 | One defense far superior. Trust elite defense. |
    | 4 | COLLAPSE_VS_SCORER | 4 | Collapse defense meets attack that can score. |
    | 5 | TWO_ONE_GOAL_ATTACKS | 5 | Both attacks score 1 goal typically. Under match. |
    | 6 | ONE_GOAL_VS_COLLAPSE | 6 | One-goal attack meets collapse defense. |
    | 7 | BINARY_HOME_TEAM | 7 | Home team either blanks or scores multiple. |
    | 8 | REGRESSION_CANDIDATE | 8 | xG gap detected. Regression likely. |
    
    ### 🎯 Confidence Tiers
    
    | Tier | Label | Criteria |
    |------|-------|----------|
    | 1 | VERY HIGH | Story priority 1-3 + sample >15 games |
    | 2 | HIGH | Story priority 4-6 + sample >8 games |
    | 3 | MEDIUM | Lower priority story or smaller sample |
    | 4 | LOW | Conflicting signals |
    | 5 | PASS | No clear edge |
    """)

if __name__ == "__main__":
    main()
