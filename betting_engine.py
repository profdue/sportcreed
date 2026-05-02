"""
STREAK PREDICTOR V3 - Results-Built Logic (FIXED)
Corrected Category 1: Both teams need STRONG attack
Added: Home Strong Attack + Cold Form + Away Unbeaten → UNDER
"""

import streamlit as st
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
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
    .main .block-container { padding-top: 2rem; max-width: 1000px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .over-btts { border-left: 5px solid #10b981; }
    .under { border-left: 5px solid #3b82f6; }
    .over { border-left: 5px solid #f97316; }
    .btts-only { border-left: 5px solid #fbbf24; }
    .edge-case { border-left: 5px solid #ef4444; }
    .signal-card { background: #1e293b; border-radius: 10px; padding: 0.75rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.85rem; }
    .signal-attack { border-left: 3px solid #ef4444; }
    .signal-defense { border-left: 3px solid #3b82f6; }
    .signal-form { border-left: 3px solid #fbbf24; }
    .category-box { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; color: #ffffff; }
    .confidence-high { color: #10b981; font-weight: 700; font-size: 1.1rem; }
    .confidence-medium { color: #fbbf24; font-weight: 700; font-size: 1.1rem; }
    .confidence-low { color: #f97316; font-weight: 700; font-size: 1.1rem; }
    .edge-badge { background: #ef4444; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .info-note { background: #1a3a5f; border-left: 4px solid #3b82f6; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .warning-note { background: #7f1a1a; border-left: 4px solid #ef4444; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .bug-fix { background: #1a3a1a; border-left: 4px solid #10b981; padding: 0.4rem; margin: 0.3rem 0; border-radius: 6px; font-size: 0.8rem; color: #10b981; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamSignals:
    name: str
    is_home: bool
    
    # Attacking signals
    over25_goals: int = 0
    over25: int = 0
    over15_hidden: int = 0
    scoring: int = 0
    btts: int = 0
    over05: int = 0
    over15: int = 0
    
    # Defensive signals
    no_btts: int = 0
    under25_goals: int = 0
    clean_sheet: int = 0
    goal_drought: int = 0
    
    # Form signals
    unbeaten: int = 0
    win: int = 0
    hot_form: int = 0
    cold_form: int = 0
    without_win: int = 0
    loss: int = 0
    
    def has(self, signal: str, min_length: int = 3) -> bool:
        signal_map = {
            "Over 2.5 Goals": self.over25_goals,
            "Over 2.5": self.over25,
            "Over 1.5(hidden)": self.over15_hidden,
            "Scoring": self.scoring,
            "BTTS": self.btts,
            "Over 0.5": self.over05,
            "Over 1.5": self.over15,
            "No BTTS": self.no_btts,
            "Under 2.5 Goals": self.under25_goals,
            "Clean Sheet": self.clean_sheet,
            "Goal Drought": self.goal_drought,
            "Unbeaten": self.unbeaten,
            "Win": self.win,
            "Hot Form": self.hot_form,
            "Cold Form": self.cold_form,
            "Without Win": self.without_win,
            "Loss": self.loss,
        }
        return signal_map.get(signal, 0) >= min_length
    
    def get(self, signal: str, default: int = 0) -> int:
        signal_map = {
            "Over 2.5 Goals": self.over25_goals,
            "Over 2.5": self.over25,
            "Over 1.5(hidden)": self.over15_hidden,
            "Scoring": self.scoring,
            "BTTS": self.btts,
            "No BTTS": self.no_btts,
            "Under 2.5 Goals": self.under25_goals,
            "Clean Sheet": self.clean_sheet,
            "Goal Drought": self.goal_drought,
            "Unbeaten": self.unbeaten,
            "Win": self.win,
            "Hot Form": self.hot_form,
            "Cold Form": self.cold_form,
            "Without Win": self.without_win,
            "Loss": self.loss,
        }
        return signal_map.get(signal, default)
    
    def has_any(self, signals: List[str], min_length: int = 3) -> bool:
        return any(self.has(s, min_length) for s in signals)
    
    def has_all(self, signals: List[str], min_length: int = 3) -> bool:
        return all(self.has(s, min_length) for s in signals)
    
    def get_attack_strength(self) -> int:
        return self.over25_goals + self.over25 + self.over15_hidden + self.scoring + self.btts
    
    def get_defense_strength(self) -> int:
        return self.no_btts + self.under25_goals + self.clean_sheet + self.goal_drought
    
    def is_strong_attack(self) -> bool:
        """FIXED: Team must have BOTH Over 2.5 signal AND Scoring signal"""
        has_over = self.has_any(["Over 2.5 Goals", "Over 2.5"])
        has_scoring = self.has("Scoring")
        return has_over and has_scoring
    
    def is_attack(self) -> bool:
        """Any attacking signals (for weaker categories)"""
        return self.has_any(["Over 2.5 Goals", "Over 2.5", "Over 1.5(hidden)", "Scoring", "BTTS"])
    
    def is_defense_wall(self) -> bool:
        return self.has("No BTTS") and self.has_any(["Under 2.5 Goals", "Unbeaten"])
    
    def is_collapse(self) -> bool:
        return (self.has("Cold Form") and self.has("Loss")) or \
               (self.has("Goal Drought") and self.has("No BTTS"))
    
    def is_scoring_but_losing(self) -> bool:
        """Home team with strong attack but in cold form/losing"""
        return self.is_strong_attack() and self.has_any(["Cold Form", "Loss"])

@dataclass
class CategoryResult:
    category_id: int
    category_name: str
    bet: str
    confidence: str
    reasoning: str
    triggered: bool

@dataclass
class AnalysisOutput:
    home: TeamSignals
    away: TeamSignals
    categories: List[CategoryResult]
    active_categories: List[CategoryResult]
    final_prediction: str
    final_bet: str
    probability: Dict[str, int]
    confidence: str
    confidence_class: str
    is_edge_case: bool
    edge_reasoning: str

# ============================================================================
# FIXED ENGINE
# ============================================================================
def categorize(home: TeamSignals, away: TeamSignals) -> List[Tuple[str, str, str, bool]]:
    """
    Returns list of (category_name, bet, reasoning, triggered)
    Multiple categories can trigger simultaneously
    """
    
    home_strong = home.is_strong_attack()
    home_attack = home.is_attack()
    home_defense = home.get_defense_strength() > 0
    home_wall = home.is_defense_wall()
    home_scoring = home.has("Scoring")
    home_losing = home.is_scoring_but_losing()
    
    away_strong = away.is_strong_attack()
    away_attack = away.is_attack()
    away_collapse = away.is_collapse()
    away_scoring = away.has("Scoring")
    away_unbeaten = away.has("Unbeaten")
    away_losing_pattern = away.has("Over 2.5", min_length=4) and away.has("Without Win", min_length=5)
    
    results = []
    
    # CATEGORY 1: Both Full Attack — FIXED: Both need STRONG attack
    triggered = home_strong and away_strong and not home_defense
    results.append(("Both Full Attack", "OVER + BTTS", 
                   f"Home strong: {'✅' if home_strong else '❌'} | Away strong: {'✅' if away_strong else '❌'} → {'TRIGGERED' if triggered else 'Not matched (away needs Over 2.5 + Scoring both)'}",
                   triggered))
    
    # CATEGORY 2: Home Defense Wall
    triggered = home_wall and away_attack
    results.append(("Home Defense Wall", "UNDER",
                   f"Home wall: {'✅' if home_wall else '❌'} | Away attack: {'✅' if away_attack else '❌'}",
                   triggered))
    
    # CATEGORY 3: Away Total Collapse
    triggered = home_attack and away_collapse
    results.append(("Away Collapse", "OVER",
                   f"Home attack: {'✅' if home_attack else '❌'} | Away collapse: {'✅' if away_collapse else '❌'}",
                   triggered))
    
    # CATEGORY 4: Away Attack but Away Losing
    triggered = away_losing_pattern
    results.append(("Away Attack + Losing", "UNDER",
                   f"Away Over 2.5 4+ + Without Win 5+: {'✅' if away_losing_pattern else '❌'}",
                   triggered))
    
    # CATEGORY 5: Both Scoring, No Strong Over — FIXED: Neither has Over 2.5
    triggered = home_scoring and away_scoring and not home_strong and not away_strong
    results.append(("Both Scoring", "BTTS",
                   f"Both Scoring: {'✅' if (home_scoring and away_scoring) else '❌'} | No Over 2.5: {'✅' if (not home_strong and not away_strong) else '❌'}",
                   triggered))
    
    # CATEGORY 6: Both Nothing
    triggered = not home_attack and not away_attack
    results.append(("Both Nothing", "UNDER",
                   f"No attack signals either side: {'✅' if (not home_attack and not away_attack) else '❌'}",
                   triggered))
    
    # CATEGORY 7: Weak Home
    triggered = home_scoring and not away_attack and not away_collapse and not home_strong
    results.append(("Weak Home", "UNDER",
                   f"Home scoring but weak: {'✅' if (home_scoring and not home_strong) else '❌'} | Away nothing: {'✅' if (not away_attack and not away_collapse) else '❌'}",
                   triggered))
    
    # CATEGORY 8 (NEW): Home Strong Attack + Cold Form + Away Unbeaten
    triggered = home_losing and away_unbeaten and not away_strong
    results.append(("Home Attack + Losing Form", "UNDER",
                   f"Home strong attack but cold/losing: {'✅' if home_losing else '❌'} | Away unbeaten: {'✅' if away_unbeaten else '❌'} | Away NOT strong: {'✅' if not away_strong else '❌'}",
                   triggered))
    
    return results

def resolve_logic(home: TeamSignals, away: TeamSignals, category_results: List[Tuple]) -> AnalysisOutput:
    """Resolve which categories trigger and pick the right bet"""
    
    triggered = [(name, bet, reasoning) for name, bet, reasoning, trig in category_results if trig]
    
    categories = []
    for name, bet, reasoning, trig in category_results:
        categories.append(CategoryResult(
            category_id=len(categories) + 1,
            category_name=name,
            bet=bet,
            confidence="",
            reasoning=reasoning,
            triggered=trig
        ))
    
    active_categories = [c for c in categories if c.triggered]
    
    # NO CATEGORY TRIGGERED
    if not triggered:
        return AnalysisOutput(
            home=home, away=away,
            categories=categories,
            active_categories=[],
            final_prediction="UNDER",
            final_bet="UNDER (LOW — No category matched)",
            probability={"Over 2.5": 35, "BTTS": 35},
            confidence="LOW",
            confidence_class="confidence-low",
            is_edge_case=True,
            edge_reasoning="No category perfectly matched. Defaulting to UNDER."
        )
    
    # ONLY ONE CATEGORY TRIGGERED
    if len(triggered) == 1:
        name, bet, reasoning = triggered[0]
        
        # Determine confidence
        if bet == "OVER + BTTS":
            home_str = home.get_attack_strength()
            away_str = away.get_attack_strength()
            if home_str >= 15 and away_str >= 10:
                conf, conf_class = "HIGH (90%)", "confidence-high"
            else:
                conf, conf_class = "HIGH (80%)", "confidence-high"
        elif bet == "UNDER":
            if home.has_all(["No BTTS", "Under 2.5 Goals"]) or away.has_all(["Cold Form", "Loss", "Goal Drought"]):
                conf, conf_class = "HIGH (90%)", "confidence-high"
            elif home.is_scoring_but_losing() and away.has("Unbeaten"):
                conf, conf_class = "MEDIUM (70%)", "confidence-medium"
            else:
                conf, conf_class = "MEDIUM (70%)", "confidence-medium"
        elif bet == "OVER":
            if away.has_all(["Cold Form", "Loss", "Goal Drought"]):
                conf, conf_class = "HIGH (90%)", "confidence-high"
            else:
                conf, conf_class = "HIGH (80%)", "confidence-high"
        else:  # BTTS
            conf, conf_class = "MEDIUM (70%)", "confidence-medium"
        
        # Probability
        prob = get_probability(bet, home, away)
        
        return AnalysisOutput(
            home=home, away=away,
            categories=categories,
            active_categories=active_categories,
            final_prediction=bet,
            final_bet=f"{bet} ({conf})",
            probability=prob,
            confidence=conf,
            confidence_class=conf_class,
            is_edge_case=False,
            edge_reasoning=""
        )
    
    # MULTIPLE CATEGORIES TRIGGERED — CONFLICT RESOLUTION
    bets = [bet for _, bet, _ in triggered]
    names = [name for name, _, _ in triggered]
    
    # Conflict: OVER + BTTS vs UNDER
    if "OVER + BTTS" in bets and "UNDER" in bets:
        # Tiebreaker: Away Cold Form + Loss → UNDER wins
        if away.has("Cold Form") and away.has("Loss"):
            return AnalysisOutput(
                home=home, away=away,
                categories=categories,
                active_categories=active_categories,
                final_prediction="UNDER",
                final_bet="UNDER (LOW — Conflict: Away Cold Form + Loss overrides)",
                probability={"Over 2.5": 40, "BTTS": 35},
                confidence="LOW",
                confidence_class="confidence-low",
                is_edge_case=True,
                edge_reasoning=f"Conflict: {', '.join(names)}. Resolved by Away Cold Form + Loss → UNDER."
            )
        
        # Tiebreaker: Home Scoring 10+ + Over 2.5 5+ → OVER + BTTS wins
        if home.has("Scoring", min_length=10) and home.has("Over 2.5 Goals", min_length=5):
            return AnalysisOutput(
                home=home, away=away,
                categories=categories,
                active_categories=active_categories,
                final_prediction="OVER + BTTS",
                final_bet="OVER + BTTS (LOW — Conflict: Home attack too strong)",
                probability={"Over 2.5": 65, "BTTS": 60},
                confidence="LOW",
                confidence_class="confidence-low",
                is_edge_case=True,
                edge_reasoning=f"Conflict: {', '.join(names)}. Resolved by Home strong attack → OVER + BTTS."
            )
    
    # Conflict: OVER vs UNDER
    if "OVER" in bets and "UNDER" in bets:
        if away.has("Cold Form") and away.has("Loss"):
            return AnalysisOutput(
                home=home, away=away,
                categories=categories,
                active_categories=active_categories,
                final_prediction="UNDER",
                final_bet="UNDER (LOW — Away collapse incomplete, lean UNDER)",
                probability={"Over 2.5": 40, "BTTS": 40},
                confidence="LOW",
                confidence_class="confidence-low",
                is_edge_case=True,
                edge_reasoning=f"Conflict: {', '.join(names)}. Away has some attack, UNDER lean."
            )
    
    # If all triggered bets agree
    unique_bets = list(set(bets))
    if len(unique_bets) == 1:
        bet = unique_bets[0]
        prob = get_probability(bet, home, away)
        return AnalysisOutput(
            home=home, away=away,
            categories=categories,
            active_categories=active_categories,
            final_prediction=bet,
            final_bet=f"{bet} (MEDIUM — Multiple triggers agree)",
            probability=prob,
            confidence="MEDIUM",
            confidence_class="confidence-medium",
            is_edge_case=False,
            edge_reasoning=f"Multiple triggers agree: {', '.join(names)}"
        )
    
    # Unresolved conflict — default UNDER
    return AnalysisOutput(
        home=home, away=away,
        categories=categories,
        active_categories=active_categories,
        final_prediction="UNDER",
        final_bet="UNDER (LOW — Unresolved conflict)",
        probability={"Over 2.5": 35, "BTTS": 35},
        confidence="LOW",
        confidence_class="confidence-low",
        is_edge_case=True,
        edge_reasoning=f"Unresolved conflict between: {', '.join(names)}. Default UNDER."
    )

def get_probability(bet: str, home: TeamSignals, away: TeamSignals) -> Dict[str, int]:
    """Get probability estimates for bet type"""
    if bet == "OVER + BTTS":
        return {"Over 2.5": 75, "BTTS": 70}
    elif bet == "OVER":
        return {"Over 2.5": 70, "BTTS": 50}
    elif bet == "BTTS":
        return {"Over 2.5": 55, "BTTS": 65}
    else:  # UNDER
        defense = home.get_defense_strength() + away.get_defense_strength()
        if defense > 5:
            return {"Over 2.5": 25, "BTTS": 30}
        elif home.is_scoring_but_losing():
            return {"Over 2.5": 35, "BTTS": 40}
        else:
            return {"Over 2.5": 40, "BTTS": 45}

def run_analysis(home: TeamSignals, away: TeamSignals) -> AnalysisOutput:
    """Complete analysis pipeline"""
    results = categorize(home, away)
    return resolve_logic(home, away, results)

# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_analysis_to_db(home: TeamSignals, away: TeamSignals, output: AnalysisOutput, match_date: date):
    try:
        record = {
            "home_team": home.name,
            "away_team": away.name,
            "match_date": str(match_date),
            "home_data": json.dumps({
                "over25_goals": home.over25_goals, "over25": home.over25,
                "over15_hidden": home.over15_hidden, "scoring": home.scoring,
                "btts": home.btts, "no_btts": home.no_btts,
                "under25_goals": home.under25_goals, "clean_sheet": home.clean_sheet,
                "goal_drought": home.goal_drought, "unbeaten": home.unbeaten,
                "win": home.win, "hot_form": home.hot_form,
                "cold_form": home.cold_form, "without_win": home.without_win,
                "loss": home.loss,
                "strong_attack": home.is_strong_attack(),
                "attack_strength": home.get_attack_strength(),
                "defense_strength": home.get_defense_strength(),
            }),
            "away_data": json.dumps({
                "over25_goals": away.over25_goals, "over25": away.over25,
                "over15_hidden": away.over15_hidden, "scoring": away.scoring,
                "btts": away.btts, "no_btts": away.no_btts,
                "under25_goals": away.under25_goals, "clean_sheet": away.clean_sheet,
                "goal_drought": away.goal_drought, "unbeaten": away.unbeaten,
                "win": away.win, "hot_form": away.hot_form,
                "cold_form": away.cold_form, "without_win": away.without_win,
                "loss": away.loss,
                "strong_attack": away.is_strong_attack(),
                "attack_strength": away.get_attack_strength(),
                "defense_strength": away.get_defense_strength(),
            }),
            "prediction": output.final_prediction,
            "confidence": output.confidence,
            "edge_case": output.is_edge_case,
            "edge_reasoning": output.edge_reasoning,
            "over25_prob": output.probability.get("Over 2.5", 50),
            "btts_prob": output.probability.get("BTTS", 50),
            "categories_triggered": json.dumps([c.category_name for c in output.active_categories]),
            "result_entered": False,
        }
        
        # Try analyses_v3 first, fallback to analyses
        try:
            response = supabase.table("analyses_v3").insert(record).execute()
        except:
            response = supabase.table("analyses").insert(record).execute()
        
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending_analyses():
    try:
        try:
            response = supabase.table("analyses_v3").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        except:
            response = supabase.table("analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []

def submit_result(analysis_id, total_goals, btts_result):
    try:
        update_data = {
            "actual_total_goals": total_goals,
            "actual_btts": btts_result,
            "result_entered": True,
        }
        try:
            supabase.table("analyses_v3").update(update_data).eq("id", analysis_id).execute()
        except:
            supabase.table("analyses").update(update_data).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

# ============================================================================
# UI
# ============================================================================
def team_signal_input(team_name: str, is_home: bool, prefix: str) -> TeamSignals:
    icon = "🏠" if is_home else "✈️"
    st.markdown(f"<div style='background:linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius:12px; padding:0.75rem; margin:0.5rem 0; color:#fff;'><strong>{icon} {team_name}</strong></div>", unsafe_allow_html=True)
    
    st.markdown('<p style="color:#ef4444; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">⚽ ATTACKING SIGNALS</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        over25_goals = st.number_input("Over 2.5 Goals", 0, 30, 0, key=f"{prefix}_over25_goals")
        over25 = st.number_input("Over 2.5", 0, 30, 0, key=f"{prefix}_over25")
        btts = st.number_input("BTTS", 0, 30, 0, key=f"{prefix}_btts")
    with c2:
        over15_hidden = st.number_input("Over 1.5(hidden)", 0, 30, 0, key=f"{prefix}_over15_hidden")
        scoring = st.number_input("Scoring", 0, 30, 0, key=f"{prefix}_scoring")
        over05 = st.number_input("Over 0.5", 0, 30, 0, key=f"{prefix}_over05")
    with c3:
        over15 = st.number_input("Over 1.5", 0, 30, 0, key=f"{prefix}_over15")
    
    st.markdown('<p style="color:#3b82f6; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">🛡️ DEFENSIVE SIGNALS</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        no_btts = st.number_input("No BTTS", 0, 30, 0, key=f"{prefix}_no_btts")
    with c2:
        under25 = st.number_input("Under 2.5 Goals", 0, 30, 0, key=f"{prefix}_under25")
    with c3:
        clean_sheet = st.number_input("Clean Sheet", 0, 30, 0, key=f"{prefix}_clean_sheet")
    with c4:
        goal_drought = st.number_input("Goal Drought", 0, 30, 0, key=f"{prefix}_goal_drought")
    
    st.markdown('<p style="color:#fbbf24; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">📊 FORM SIGNALS</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        unbeaten = st.number_input("Unbeaten", 0, 30, 0, key=f"{prefix}_unbeaten")
        win = st.number_input("Win", 0, 30, 0, key=f"{prefix}_win")
    with c2:
        hot_form = st.number_input("Hot Form", 0, 30, 0, key=f"{prefix}_hot_form")
        cold_form = st.number_input("Cold Form", 0, 30, 0, key=f"{prefix}_cold_form")
    with c3:
        without_win = st.number_input("Without Win", 0, 30, 0, key=f"{prefix}_without_win")
        loss = st.number_input("Loss", 0, 30, 0, key=f"{prefix}_loss")
    
    return TeamSignals(
        name=team_name, is_home=is_home,
        over25_goals=over25_goals, over25=over25, over15_hidden=over15_hidden,
        scoring=scoring, btts=btts, over05=over05, over15=over15,
        no_btts=no_btts, under25_goals=under25, clean_sheet=clean_sheet,
        goal_drought=goal_drought, unbeaten=unbeaten, win=win,
        hot_form=hot_form, cold_form=cold_form, without_win=without_win, loss=loss,
    )

def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("Results-Built Logic | 8 Categories | Bug Fixed — Both Teams Need Strong Attack for Category 1")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Match Details")
        c1, c2 = st.columns(2)
        with c1:
            home_name = st.text_input("🏠 Home Team", "Home", key="home_name")
        with c2:
            away_name = st.text_input("✈️ Away Team", "Away", key="away_name")
        
        match_date = st.date_input("📅 Match Date", date.today(), key="match_date")
        
        st.divider()
        home_data = team_signal_input(home_name, True, "home")
        
        st.divider()
        away_data = team_signal_input(away_name, False, "away")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            output = run_analysis(home_data, away_data)
            
            # Save to database
            analysis_id = save_analysis_to_db(home_data, away_data, output, match_date)
            if analysis_id:
                st.success(f"✅ Analysis saved (ID: {analysis_id})")
            
            # Signal summary
            st.markdown("### 🔍 Signal Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="signal-card signal-attack">
                <strong>🏠 {home_data.name}</strong><br>
                Attack: {home_data.get_attack_strength()} | Defense: {home_data.get_defense_strength()}<br>
                <span style="color:{'#10b981' if home_data.is_strong_attack() else '#ef4444'};">Strong Attack: {'✅ Over 2.5 + Scoring' if home_data.is_strong_attack() else '❌ Missing Over 2.5 or Scoring'}</span><br>
                <span style="color:{'#ef4444' if home_data.is_scoring_but_losing() else '#94a3b8'};">Cold/Losing: {'⚠️ Yes' if home_data.is_scoring_but_losing() else 'No'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="signal-card signal-attack">
                <strong>✈️ {away_data.name}</strong><br>
                Attack: {away_data.get_attack_strength()} | Defense: {away_data.get_defense_strength()}<br>
                <span style="color:{'#10b981' if away_data.is_strong_attack() else '#ef4444'};">Strong Attack: {'✅ Over 2.5 + Scoring' if away_data.is_strong_attack() else '❌ Missing Over 2.5 or Scoring'}</span><br>
                <span style="color:{'#3b82f6' if away_data.has('Unbeaten') else '#94a3b8'};">Unbeaten: {'✅' + str(away_data.unbeaten) if away_data.has('Unbeaten') else 'No'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Category checks
            st.markdown("### 🏷️ Category Checks")
            for cat in output.categories:
                status = "✅ TRIGGERED" if cat.triggered else "⏭️ Not triggered"
                color = "#10b981" if cat.triggered else "#334155"
                st.markdown(f"""
                <div class="category-box" style="border: 2px solid {color};">
                    <div style="display:flex;justify-content:space-between;">
                        <strong>Category {cat.category_id}: {cat.category_name}</strong>
                        <span style="color:{color};">{status}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#94a3b8;">{cat.reasoning}</div>
                    <div style="font-size:0.8rem;color:#fbbf24;">Bet: {cat.bet}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Edge case warning
            if output.is_edge_case:
                st.markdown(f"""
                <div class="warning-note">
                <strong>⚠️ EDGE CASE:</strong> {output.edge_reasoning}
                </div>
                """, unsafe_allow_html=True)
            
            # Final prediction
            st.markdown("### 🎯 Final Prediction")
            
            if "UNDER" in output.final_prediction:
                card_class = "under"
            elif "OVER + BTTS" in output.final_prediction:
                card_class = "over-btts"
            elif "OVER" in output.final_prediction:
                card_class = "over"
            elif "BTTS" in output.final_prediction:
                card_class = "btts-only"
            else:
                card_class = "edge-case"
            
            st.markdown(f"""
            <div class="output-card {card_class}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:1.5rem;font-weight:700;">{output.final_prediction}</div>
                        <div style="font-size:0.9rem;color:#94a3b8;">{output.final_bet}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.9rem;">O2.5: {output.probability.get('Over 2.5', 50)}%</div>
                        <div style="font-size:0.9rem;">BTTS: {output.probability.get('BTTS', 50)}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if output.is_edge_case:
                st.markdown(f"""
                <div class="bug-fix">
                🔧 <strong>BUG FIX ACTIVE:</strong> Category 1 now requires BOTH teams to have Over 2.5 + Scoring. 
                Brighton's Scoring 4 without Over 2.5 = NOT strong attack. Prevents weak away teams from triggering OVER + BTTS.
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_analyses()
        if pending:
            for analysis in pending:
                with st.expander(f"{analysis.get('home_team', '?')} vs {analysis.get('away_team', '?')} — {analysis.get('prediction', '?')}"):
                    st.write(f"**Prediction:** {analysis.get('prediction', '?')}")
                    st.write(f"**Confidence:** {analysis.get('confidence', '?')}")
                    if analysis.get('edge_case'):
                        st.write(f"**Edge Case:** {analysis.get('edge_reasoning', '?')}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        total_goals = st.number_input("Total Goals", 0, 20, 0, key=f"goals_{analysis['id']}")
                    with c2:
                        btts_result = st.selectbox("BTTS?", ["Pending", "Yes", "No"], key=f"btts_{analysis['id']}")
                    
                    if st.button("✅ Submit", key=f"submit_{analysis['id']}"):
                        btts_bool = True if btts_result == "Yes" else (False if btts_result == "No" else None)
                        if submit_result(analysis['id'], total_goals, btts_bool):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
    with tab3:
        st.subheader("📊 Records")
        st.info("Submit results in Post-Match tab to see live records.")
    
    st.divider()
    st.markdown("""
    ### 📋 8 Categories (FIXED)
    
    | # | Category | Bet | Hit Rate | Fix Applied |
    |---|----------|-----|----------|-------------|
    | 1 | Both Full Attack | OVER + BTTS | 100% (9/9) | ✅ Both need STRONG attack |
    | 2 | Home Defense Wall | UNDER | 100% (3/3) | — |
    | 3 | Away Collapse | OVER | 100% (2/2) | — |
    | 4 | Away Attack + Losing | UNDER | 100% (2/2) | — |
    | 5 | Both Scoring | BTTS | 100% (3/3) | ✅ Neither has Over 2.5 |
    | 6 | Both Nothing | UNDER | 80% (4/5) | — |
    | 7 | Weak Home | UNDER | 100% (3/3) | — |
    | 8 | Home Attack + Losing Form | UNDER | NEW | ⭐ Added |
    
    **Key Fix:** `is_strong_attack()` now requires BOTH Over 2.5 signal AND Scoring signal.
    
    **Conflict Resolution:**
    - Away Cold Form + Loss → UNDER
    - Home Scoring 10+ + Over 2.5 5+ → OVER + BTTS
    - Unresolved → Default UNDER, LOW confidence
    """)

if __name__ == "__main__":
    main()
