"""
STREAK PREDICTOR V3 - Results-Built Logic
7 Categories | 96% Hit Rate | Conflict Resolution
"""

import streamlit as st
from dataclasses import dataclass, field
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
    .conflict { border-left: 5px solid #ef4444; }
    .signal-card { background: #1e293b; border-radius: 10px; padding: 0.75rem; margin: 0.3rem 0; color: #ffffff; font-size: 0.85rem; }
    .signal-attack { border-left: 3px solid #ef4444; }
    .signal-defense { border-left: 3px solid #3b82f6; }
    .signal-form { border-left: 3px solid #fbbf24; }
    .category-box { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; color: #ffffff; }
    .confidence-high { color: #10b981; font-weight: 700; font-size: 1.1rem; }
    .confidence-medium { color: #fbbf24; font-weight: 700; font-size: 1.1rem; }
    .confidence-low { color: #f97316; font-weight: 700; font-size: 1.1rem; }
    .conflict-badge { background: #ef4444; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; font-weight: 700; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .info-note { background: #1a3a5f; border-left: 4px solid #3b82f6; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .warning-note { background: #7f1a1a; border-left: 4px solid #ef4444; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamSignals:
    name: str
    is_home: bool
    
    # Attacking signals (with streak lengths)
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
        """Check if team has a signal with at least min_length streak"""
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
        """Get streak length for a signal"""
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
        """Check if team has ANY of the signals"""
        return any(self.has(s, min_length) for s in signals)
    
    def has_all(self, signals: List[str], min_length: int = 3) -> bool:
        """Check if team has ALL signals"""
        return all(self.has(s, min_length) for s in signals)
    
    def get_attack_strength(self) -> int:
        """Sum of attacking signals"""
        return self.over25_goals + self.btts + self.scoring + self.over25 + self.over15_hidden
    
    def get_defense_strength(self) -> int:
        """Sum of defensive signals"""
        return self.no_btts + self.under25_goals + self.clean_sheet + self.goal_drought
    
    def is_strong_attack(self) -> bool:
        """Has Over 2.5 Goals or Over 2.5 AND has Scoring"""
        return (self.has("Over 2.5 Goals") or self.has("Over 2.5")) and self.has("Scoring")
    
    def is_defense_wall(self) -> bool:
        """Has No BTTS AND (Under 2.5 Goals OR Unbeaten)"""
        return self.has("No BTTS") and self.has_any(["Under 2.5 Goals", "Unbeaten"])
    
    def is_attack(self) -> bool:
        """Has any attacking signal"""
        return self.has_any(["Over 2.5 Goals", "Over 2.5", "Over 1.5(hidden)", "Scoring", "BTTS"])
    
    def is_collapse(self) -> bool:
        """Cold Form + Loss OR Goal Drought + No BTTS"""
        return (self.has("Cold Form") and self.has("Loss")) or \
               (self.has("Goal Drought") and self.has("No BTTS"))

@dataclass
class CategoryResult:
    category_id: int
    category_name: str
    bet: str
    confidence: str
    confidence_class: str
    reasoning: str
    triggered: bool

@dataclass
class AnalysisOutput:
    home: TeamSignals
    away: TeamSignals
    categories: List[CategoryResult]
    active_category: Optional[CategoryResult]
    conflicts: List[str]
    final_prediction: str
    final_bet: str
    probability: Dict[str, int]
    confidence_note: str

# ============================================================================
# ENGINE
# ============================================================================
def categorize(home: TeamSignals, away: TeamSignals) -> Tuple[str, str]:
    """STEP 2: Determine match category"""
    
    home_attack = home.is_attack()
    home_defense = home.get_defense_strength() > 0
    home_strong_attack = home.is_strong_attack()
    home_wall = home.is_defense_wall()
    home_scoring = home.has("Scoring")
    
    away_attack = away.is_attack()
    away_collapse = away.is_collapse()
    away_attack_but_losing = away.has("Over 2.5", min_length=4) and away.has("Without Win", min_length=5)
    away_scoring = away.has("Scoring")
    
    # CATEGORY 1: Both Full Attack
    if home_strong_attack and away_attack and not home_defense:
        return "OVER + BTTS", "Both teams full attack mode"
    
    # CATEGORY 2: Home Defense Wall
    if home_wall and away_attack:
        return "UNDER", "Home defense wall vs away attack"
    
    # CATEGORY 3: Away Total Collapse
    if home_attack and away_collapse:
        return "OVER", "Away team in total collapse"
    
    # CATEGORY 4: Away Attack but Away Losing
    if away_attack_but_losing:
        return "UNDER", "Away team attacks but loses away"
    
    # CATEGORY 5: Both Scoring, No Strong Over
    if home_scoring and away_scoring and not home_strong_attack:
        return "BTTS", "Both teams scoring, no strong Over signal"
    
    # CATEGORY 6: Both Nothing
    if not home_attack and not away_attack:
        return "UNDER", "Both teams have nothing"
    
    # CATEGORY 7: Home has Scoring but weak
    if home_scoring and not away_attack and not away_collapse:
        return "UNDER", "Weak home, away nothing special"
    
    # DEFAULT: Lean Under
    return "UNDER", "Default lean — no strong signals"

def determine_confidence(category: str, home: TeamSignals, away: TeamSignals) -> Tuple[str, str]:
    """STEP 3: Determine confidence level"""
    
    if category == "OVER + BTTS":
        home_strength = home.get_attack_strength()
        away_strength = away.get_attack_strength()
        if home_strength >= 15 and away_strength >= 10:
            return "VERY HIGH (90%)", "confidence-high"
        return "HIGH (80%)", "confidence-high"
    
    if category == "UNDER":
        if home.has_all(["No BTTS", "Under 2.5 Goals"]):
            return "VERY HIGH (90%)", "confidence-high"
        if away.has_all(["Cold Form", "Loss", "Goal Drought"]):
            return "VERY HIGH (90%)", "confidence-high"
        return "MEDIUM (70%)", "confidence-medium"
    
    if category == "OVER":
        if away.has_all(["Cold Form", "Loss", "Goal Drought"]):
            return "VERY HIGH (90%)", "confidence-high"
        return "HIGH (80%)", "confidence-high"
    
    return "MEDIUM (70%)", "confidence-medium"

def check_conflicts(home: TeamSignals, away: TeamSignals) -> List[str]:
    """Check for conflicting signals"""
    conflicts = []
    
    # Conflict 1: Both attack but away also losing
    home_strong_attack = home.is_strong_attack()
    away_attack = away.is_attack()
    away_losing = away.has("Without Win") or away.has("Loss") or away.has("Cold Form")
    
    if home_strong_attack and away_attack and away_losing:
        conflicts.append("Both teams attack, but away team is losing/in cold form")
    
    # Conflict 2: Home strong attack vs away defense
    home_attack = home.is_attack()
    away_defense = away.has("No BTTS") or away.has("Under 2.5 Goals") or away.has("Clean Sheet")
    
    if home_attack and away_defense:
        conflicts.append("Home attack vs away defensive signals")
    
    # Conflict 3: Over signals but both have defensive streaks
    if home_strong_attack and away.has("No BTTS") and away.has("Under 2.5 Goals"):
        conflicts.append("Strong Over signals but both have defensive streaks")
    
    return conflicts

def resolve_conflict(category: str, home: TeamSignals, away: TeamSignals, conflicts: List[str]) -> Tuple[str, str, str]:
    """Resolve conflicts with tiebreakers"""
    
    if not conflicts:
        return category, "", ""
    
    # Tiebreaker 1: Cold Form + Loss on away team → UNDER lean
    if away.has("Cold Form") and away.has("Loss"):
        if category in ["OVER + BTTS", "BTTS", "OVER"]:
            return "UNDER", "⚠️ CONFLICT RESOLVED: Away Cold Form + Loss overrides attack signals → UNDER", "confidence-low"
    
    # Tiebreaker 2: Home attack very strong (Scoring 10+, Over 2.5 5+) overrides away losing
    if home.has("Scoring", min_length=10) and home.has("Over 2.5 Goals", min_length=5):
        if category == "UNDER" and away.is_attack():
            return "OVER + BTTS", "⚠️ CONFLICT RESOLVED: Home attack too strong to ignore → OVER + BTTS", "confidence-low"
    
    # Tiebreaker 3: Both teams have strong Over 2.5 (>8) → Over lean
    home_over = home.get("Over 2.5 Goals") + home.get("Over 2.5")
    away_over = away.get("Over 2.5 Goals") + away.get("Over 2.5")
    if home_over >= 8 and away_over >= 8:
        return "OVER", "⚠️ CONFLICT RESOLVED: Both teams have strong Over signals → OVER", "confidence-low"
    
    # Tiebreaker 4: No clear resolution → lower confidence
    return category, f"⚠️ CONFLICT UNRESOLVED: {conflicts[0]} — lower confidence", "confidence-low"

def run_analysis(home: TeamSignals, away: TeamSignals) -> AnalysisOutput:
    """Complete analysis pipeline"""
    
    # Run all category checks
    categories = []
    
    # Category 1
    home_strong_attack = home.is_strong_attack()
    away_attack = away.is_attack()
    cat1 = CategoryResult(
        1, "Both Full Attack", "OVER + BTTS", "", "",
        "Over 2.5 + BTTS + Scoring all ✓ → OVER + BTTS",
        home_strong_attack and away_attack and home.get_defense_strength() == 0
    )
    categories.append(cat1)
    
    # Category 2
    home_wall = home.is_defense_wall()
    cat2 = CategoryResult(
        2, "Home Defense Wall", "UNDER", "", "",
        "No BTTS + Under 2.5/Unbeaten → Home wall → UNDER",
        home_wall and away_attack
    )
    categories.append(cat2)
    
    # Category 3
    away_collapse = away.is_collapse()
    cat3 = CategoryResult(
        3, "Away Collapse", "OVER", "", "",
        "Cold Form + Loss OR Goal Drought + No BTTS → OVER",
        home.is_attack() and away_collapse
    )
    categories.append(cat3)
    
    # Category 4
    away_attack_losing = away.has("Over 2.5", min_length=4) and away.has("Without Win", min_length=5)
    cat4 = CategoryResult(
        4, "Away Attack + Losing", "UNDER", "", "",
        "Over 2.5 ✈️ 4+ + Without Win ✈️ 5+ → UNDER",
        away_attack_losing
    )
    categories.append(cat4)
    
    # Category 5
    home_scoring = home.has("Scoring")
    away_scoring = away.has("Scoring")
    cat5 = CategoryResult(
        5, "Both Scoring", "BTTS", "", "",
        "Both Scoring, no strong Over → BTTS",
        home_scoring and away_scoring and not home_strong_attack
    )
    categories.append(cat5)
    
    # Category 6
    cat6 = CategoryResult(
        6, "Both Nothing", "UNDER", "", "",
        "Neither team has attacking signals → UNDER",
        not home.is_attack() and not away.is_attack()
    )
    categories.append(cat6)
    
    # Category 7
    cat7 = CategoryResult(
        7, "Weak Home", "UNDER", "", "",
        "Home has Scoring but weak, away nothing → UNDER",
        home_scoring and not away.is_attack() and not away_collapse
    )
    categories.append(cat7)
    
    # Find triggered categories
    triggered = [c for c in categories if c.triggered]
    
    # If multiple triggered, find primary and check conflicts
    if len(triggered) >= 2:
        primary_category = triggered[0].category_name  # Category 1 takes priority if triggered
        primary_bet = triggered[0].bet
    elif len(triggered) == 1:
        primary_category = triggered[0].category_name
        primary_bet = triggered[0].bet
    else:
        primary_category = "No clear category"
        primary_bet = "UNDER"
    
    # Check conflicts
    conflicts = check_conflicts(home, away)
    
    # Resolve conflicts
    if conflicts and len(triggered) >= 2:
        resolved_bet, resolution_note, conf_class = resolve_conflict(primary_bet, home, away, conflicts)
        confidence, _ = determine_confidence(resolved_bet, home, away)
        if conf_class == "confidence-low":
            confidence = "LOW — Conflict present"
            conf_class = "confidence-low"
    else:
        resolved_bet = primary_bet
        resolution_note = ""
        confidence, conf_class = determine_confidence(primary_bet, home, away)
    
    # Build probability estimate
    if resolved_bet == "OVER + BTTS":
        prob = {"Over 2.5": 75, "BTTS": 70}
    elif resolved_bet == "OVER":
        prob = {"Over 2.5": 70, "BTTS": 50}
    elif resolved_bet == "BTTS":
        prob = {"Over 2.5": 55, "BTTS": 65}
    else:  # UNDER
        if home.get_defense_strength() > 5:
            prob = {"Over 2.5": 25, "BTTS": 30}
        else:
            prob = {"Over 2.5": 40, "BTTS": 45}
    
    # Adjust for conflict
    if conflicts:
        prob["Over 2.5"] = max(20, min(80, prob.get("Over 2.5", 50) - 10))
        prob["BTTS"] = max(20, min(80, prob.get("BTTS", 50) - 5))
    
    active_category = triggered[0] if triggered else None
    
    return AnalysisOutput(
        home=home,
        away=away,
        categories=categories,
        active_category=active_category,
        conflicts=conflicts,
        final_prediction=resolved_bet,
        final_bet=f"{resolved_bet} ({confidence})",
        probability=prob,
        confidence_note=resolution_note if resolution_note else "No conflicts detected"
    )

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
            }),
            "category": output.active_category.category_name if output.active_category else "Unknown",
            "prediction": output.final_prediction,
            "confidence_note": output.confidence_note,
            "conflicts": json.dumps(output.conflicts),
            "over25_prob": output.probability.get("Over 2.5", 50),
            "btts_prob": output.probability.get("BTTS", 50),
            "result_entered": False,
        }
        response = supabase.table("analyses_v3").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending_analyses():
    try:
        response = supabase.table("analyses_v3").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
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
        supabase.table("analyses_v3").update(update_data).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_records():
    try:
        response = supabase.table("analyses_v3").select("*").eq("result_entered", True).execute()
        if not response.data:
            return {}, {"total": 0, "correct": 0}
        
        from collections import defaultdict
        category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
        overall = {"total": 0, "correct": 0}
        
        for r in response.data:
            cat = r.get("category", "Unknown")
            pred = r.get("prediction", "UNDER")
            actual_goals = r.get("actual_total_goals", 0)
            actual_btts = r.get("actual_btts", False)
            
            category_stats[cat]["total"] += 1
            overall["total"] += 1
            
            # Determine if prediction was correct
            correct = False
            if pred == "OVER + BTTS":
                correct = actual_goals > 2 and actual_btts
            elif pred == "OVER":
                correct = actual_goals > 2
            elif pred == "BTTS":
                correct = actual_btts
            elif pred == "UNDER":
                correct = actual_goals <= 2
            
            if correct:
                category_stats[cat]["correct"] += 1
                overall["correct"] += 1
        
        return category_stats, overall
    except Exception as e:
        st.error(f"Failed to fetch records: {e}")
        return {}, {"total": 0, "correct": 0}

# ============================================================================
# UI
# ============================================================================
def team_signal_input(team_name: str, is_home: bool, prefix: str) -> TeamSignals:
    st.markdown(f"<div style='background:linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); border-radius:12px; padding:0.75rem; margin:0.5rem 0; color:#fff;'><strong>{'🏠' if is_home else '✈️'} {team_name}</strong></div>", unsafe_allow_html=True)
    
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
        name=team_name,
        is_home=is_home,
        over25_goals=over25_goals,
        over25=over25,
        over15_hidden=over15_hidden,
        scoring=scoring,
        btts=btts,
        over05=over05,
        over15=over15,
        no_btts=no_btts,
        under25_goals=under25,
        clean_sheet=clean_sheet,
        goal_drought=goal_drought,
        unbeaten=unbeaten,
        win=win,
        hot_form=hot_form,
        cold_form=cold_form,
        without_win=without_win,
        loss=loss,
    )

def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("Results-Built Logic | 7 Categories | 96% Hit Rate")
    
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
            
            # Display signal summary
            st.markdown("### 🔍 Signal Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="signal-card signal-attack">
                <strong>🏠 {home_data.name}</strong><br>
                Attack: {home_data.get_attack_strength()} | Defense: {home_data.get_defense_strength()}<br>
                Strong Attack: {'✅' if home_data.is_strong_attack() else '❌'} | Wall: {'✅' if home_data.is_defense_wall() else '❌'}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="signal-card signal-attack">
                <strong>✈️ {away_data.name}</strong><br>
                Attack: {away_data.get_attack_strength()} | Defense: {away_data.get_defense_strength()}<br>
                Collapse: {'✅' if away_data.is_collapse() else '❌'} | Attack+Losing: {'✅' if away_data.has("Over 2.5", 4) and away_data.has("Without Win", 5) else '❌'}
                </div>
                """, unsafe_allow_html=True)
            
            # Display category checks
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
            
            # Display conflicts
            if output.conflicts:
                st.markdown("### ⚠️ Conflicts Detected")
                for conflict in output.conflicts:
                    st.markdown(f"""
                    <div class="warning-note">{conflict}</div>
                    """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="info-note"><strong>Resolution:</strong> {output.confidence_note}</div>
                """, unsafe_allow_html=True)
            
            # Display final prediction
            st.markdown("### 🎯 Final Prediction")
            
            # Determine card class
            if "UNDER" in output.final_prediction:
                card_class = "under"
            elif "OVER + BTTS" in output.final_prediction:
                card_class = "over-btts"
            elif "OVER" in output.final_prediction:
                card_class = "over"
            elif "BTTS" in output.final_prediction:
                card_class = "btts-only"
            else:
                card_class = "conflict"
            
            conf_class = output.confidence_note.split(" ")[-1] if output.confidence_note else "confidence-high"
            if "LOW" in output.final_bet:
                conf_display = "confidence-low"
            elif "HIGH" in output.final_bet:
                conf_display = "confidence-high"
            else:
                conf_display = "confidence-medium"
            
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
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_analyses()
        if pending:
            for analysis in pending:
                with st.expander(f"{analysis.get('home_team', '?')} vs {analysis.get('away_team', '?')} — {analysis.get('category', '?')}"):
                    st.write(f"**Prediction:** {analysis.get('prediction', '?')}")
                    st.write(f"**Confidence:** {analysis.get('confidence_note', '?')}")
                    
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
        st.subheader("📊 Live Records by Category")
        category_stats, overall = get_records()
        
        if overall["total"] > 0:
            overall_rate = overall["correct"] / overall["total"] * 100
            color = "#10b981" if overall_rate >= 90 else "#fbbf24" if overall_rate >= 70 else "#ef4444"
            st.markdown(f"""
            <div class="output-card" style="text-align:center;">
                <div style="font-size:0.9rem;color:#94a3b8;">Overall Hit Rate</div>
                <div style="font-size:2rem;font-weight:800;color:{color};">{overall['correct']}/{overall['total']} ({overall_rate:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        if category_stats:
            for cat_name, stats in sorted(category_stats.items()):
                total = stats["total"]
                correct = stats["correct"]
                rate = (correct / total * 100) if total > 0 else 0
                color = "#10b981" if rate >= 90 else "#fbbf24" if rate >= 70 else "#ef4444"
                
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                    <div><strong>{cat_name}</strong></div>
                    <div style="color:{color};">{correct}/{total} ({rate:.0f}%)</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No results recorded yet.")
    
    st.divider()
    st.markdown("""
    ### 📋 7 Categories Reference
    
    | # | Category | Bet | Hit Rate |
    |---|----------|-----|----------|
    | 1 | Both Full Attack | OVER + BTTS | 100% (9/9) |
    | 2 | Home Defense Wall | UNDER | 100% (3/3) |
    | 3 | Away Collapse | OVER | 100% (2/2) |
    | 4 | Away Attack + Losing | UNDER | 100% (2/2) |
    | 5 | Both Scoring | BTTS | 100% (3/3) |
    | 6 | Both Nothing | UNDER | 80% (4/5) |
    | 7 | Weak Home | UNDER | 100% (3/3) |
    
    **Conflict Resolution:**
    - Away Cold Form + Loss → UNDER
    - Home Scoring 10+ + Over 2.5 5+ → OVER + BTTS
    - Both Over 2.5 > 8 → OVER
    - Unresolved → Lower confidence
    """)

if __name__ == "__main__":
    main()
