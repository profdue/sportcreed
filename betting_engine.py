"""
STREAK PREDICTOR V3 - Label-Based Probability Engine
League Base Rates | 12 Labels | Modifier System
"""

import streamlit as st
from dataclasses import dataclass
from typing import Optional, Tuple
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
    .main .block-container { padding-top: 2rem; max-width: 900px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .max-bet { border-left: 5px solid #ef4444; }
    .bet { border-left: 5px solid #f97316; }
    .small-bet { border-left: 5px solid #fbbf24; }
    .watch { border-left: 5px solid #3b82f6; }
    .no-bet { border-left: 5px solid #6b7280; }
    .metric-badge { background: #0f172a; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.8rem; color: #10b981; font-weight: 700; }
    .info-note { background: #1a3a5f; border-left: 4px solid #3b82f6; padding: 0.6rem; margin: 0.4rem 0; border-radius: 8px; font-size: 0.85rem; color: #ffffff; }
    .formula-box { background: #0f172a; border-radius: 10px; padding: 0.8rem; margin: 0.4rem 0; color: #94a3b8; font-family: monospace; font-size: 0.85rem; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LEAGUE BASE RATES
# ============================================================================
LEAGUE_BASES = {
    "Bundesliga": {"over25": 0.72, "btts": 0.62},
    "1. Lig": {"over25": 0.70, "btts": 0.60},
    "Pro League (Saudi)": {"over25": 0.70, "btts": 0.60},
    "Ligue 1": {"over25": 0.65, "btts": 0.58},
    "2. Bundesliga": {"over25": 0.62, "btts": 0.55},
    "Premier League": {"over25": 0.55, "btts": 0.52},
    "Championship (ENG)": {"over25": 0.53, "btts": 0.50},
    "La Liga": {"over25": 0.50, "btts": 0.48},
    "Ligue 2": {"over25": 0.50, "btts": 0.48},
    "Super League 1": {"over25": 0.50, "btts": 0.48},
    "Süper Lig": {"over25": 0.55, "btts": 0.52},
    "Primeira Liga": {"over25": 0.52, "btts": 0.50},
    "Serie A (Italy)": {"over25": 0.42, "btts": 0.45},
    "Liga Argentina": {"over25": 0.38, "btts": 0.42},
    "Serie A (Brazil)": {"over25": 0.40, "btts": 0.44},
    "Liga Pro (Ecuador)": {"over25": 0.35, "btts": 0.40},
    "Primera A (Colombia)": {"over25": 0.45, "btts": 0.46},
    "Primera Div (VEN)": {"over25": 0.42, "btts": 0.44},
    "Liga Nacional (HON)": {"over25": 0.48, "btts": 0.48},
    "Liga Nacional (GUA)": {"over25": 0.48, "btts": 0.48},
    "Premiership (SCO)": {"over25": 0.55, "btts": 0.52},
    "Championship (SCO)": {"over25": 0.53, "btts": 0.50},
    "First League (RUS)": {"over25": 0.50, "btts": 0.48},
    "Liga I (ROU)": {"over25": 0.50, "btts": 0.48},
    "Liga MX": {"over25": 0.55, "btts": 0.52},
}

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================
LABELS = {
    1: {"name": "Scoring run", "needs_number": True, "has_btts": False},
    2: {"name": "Win streak", "needs_number": True, "has_btts": False},
    3: {"name": "Unbeaten", "needs_number": True, "has_btts": False},
    4: {"name": "Clean sheet", "needs_number": True, "has_btts": False},
    5: {"name": "Hot Attack 2", "needs_number": False, "has_btts": False},
    6: {"name": "High Goals", "needs_number": False, "has_btts": True},
    7: {"name": "BTTS Lock", "needs_number": False, "has_btts": True},
    8: {"name": "Clean Sheet Unlikely", "needs_number": False, "has_btts": True},
    9: {"name": "Low Quality", "needs_number": False, "has_btts": True},
    10: {"name": "Hot Clash", "needs_number": False, "has_btts": False},
    11: {"name": "Strong form clash", "needs_number": False, "has_btts": False},
    12: {"name": "Tight Game", "needs_number": False, "has_btts": False},
}

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class AnalysisInput:
    league: str
    label_id: int
    label_name: str
    streak_length: Optional[int]
    opponent_no_btts_3: bool
    opponent_cold_form: bool
    opponent_without_win_5: bool

@dataclass
class AnalysisOutput:
    over25_prob: float
    btts_prob: Optional[float]
    decision: str
    decision_class: str
    formula_breakdown: str
    modifier_details: str

# ============================================================================
# ENGINE
# ============================================================================
def calculate_over25(input_data: AnalysisInput) -> Tuple[float, str, str]:
    """Calculate Over 2.5 probability based on label and league"""
    base = LEAGUE_BASES[input_data.league]["over25"]
    label = input_data.label_id
    x = input_data.streak_length
    
    formula_parts = [f"Base: {base:.2f}"]
    modifier = 0.0
    penalty = 0.0
    
    # LABEL 1: Scoring run X
    if label == 1:
        if x is not None:
            if x >= 20:
                modifier = 0.20
                formula_parts.append(f"Scoring run ≥20: +0.20")
            elif x >= 13:
                modifier = 0.15
                formula_parts.append(f"Scoring run ≥13: +0.15")
            elif x >= 8:
                modifier = 0.10
                formula_parts.append(f"Scoring run ≥8: +0.10")
            elif x >= 5:
                modifier = 0.05
                formula_parts.append(f"Scoring run ≥5: +0.05")
            else:
                formula_parts.append(f"Scoring run <5: +0.00")
    
    # LABEL 2: Win streak X
    elif label == 2:
        if x is not None:
            if x >= 5:
                modifier = 0.10
                formula_parts.append(f"Win streak ≥5: +0.10")
            elif x >= 3:
                modifier = 0.06
                formula_parts.append(f"Win streak ≥3: +0.06")
            else:
                formula_parts.append(f"Win streak <3: +0.00")
        
        # Penalty for opponent No BTTS
        if input_data.opponent_no_btts_3:
            penalty = -0.10
            formula_parts.append(f"Opp No BTTS ≥3: -0.10")
    
    # LABEL 3: Unbeaten X
    elif label == 3:
        if x is not None:
            if x >= 16:
                modifier = -0.12
                formula_parts.append(f"Unbeaten ≥16: -0.12")
            elif x >= 10:
                modifier = -0.08
                formula_parts.append(f"Unbeaten ≥10: -0.08")
            elif x >= 5:
                modifier = -0.05
                formula_parts.append(f"Unbeaten ≥5: -0.05")
            else:
                formula_parts.append(f"Unbeaten <5: -0.00")
    
    # LABEL 4: Clean sheet X
    elif label == 4:
        if x is not None:
            if x >= 5:
                modifier = -0.20
                formula_parts.append(f"Clean sheet ≥5: -0.20")
            elif x >= 3:
                modifier = -0.15
                formula_parts.append(f"Clean sheet ≥3: -0.15")
            else:
                formula_parts.append(f"Clean sheet <3: -0.00")
    
    # LABEL 5: Hot Attack 2 (OVERRIDE)
    elif label == 5:
        formula_parts.clear()
        formula_parts.append("OVERRIDE: Hot Attack 2 = 0.90")
        return 0.90, " + ".join(formula_parts), "OVERRIDE — league base ignored"
    
    # LABEL 6: High Goals
    elif label == 6:
        modifier = 0.18
        formula_parts.append(f"High Goals: +0.18")
    
    # LABEL 7: BTTS Lock
    elif label == 7:
        modifier = 0.05
        formula_parts.append(f"BTTS Lock: +0.05")
    
    # LABEL 8: Clean Sheet Unlikely
    elif label == 8:
        modifier = 0.15
        formula_parts.append(f"Clean Sheet Unlikely: +0.15")
    
    # LABEL 9: Low Quality
    elif label == 9:
        modifier = -0.20
        formula_parts.append(f"Low Quality: -0.20")
    
    # LABEL 10: Hot Clash
    elif label == 10:
        modifier = 0.10
        formula_parts.append(f"Hot Clash: +0.10")
    
    # LABEL 11: Strong form clash
    elif label == 11:
        formula_parts.append(f"Strong form clash: no change")
    
    # LABEL 12: Tight Game
    elif label == 12:
        modifier = -0.15
        formula_parts.append(f"Tight Game: -0.15")
    
    total = base + modifier + penalty
    formula_parts.append(f"= {total:.2f}")
    
    modifier_details = f"Modifier: {modifier:+.2f}"
    if penalty != 0:
        modifier_details += f" | Penalty: {penalty:+.2f}"
    
    return total, " + ".join(formula_parts), modifier_details

def calculate_btts(input_data: AnalysisInput) -> Optional[float]:
    """Calculate BTTS probability for labels that support it"""
    label = input_data.label_id
    
    # Only labels 6-9 have BTTS
    if label not in [6, 7, 8, 9]:
        return None
    
    base = LEAGUE_BASES[input_data.league]["btts"]
    modifier = 0.0
    
    if label == 6:  # High Goals
        modifier = 0.10
    elif label == 7:  # BTTS Lock
        modifier = 0.20
    elif label == 8:  # Clean Sheet Unlikely
        modifier = 0.12
    elif label == 9:  # Low Quality
        modifier = -0.15
    
    return base + modifier

def decide_stake(probability: float, label_id: int) -> Tuple[str, str]:
    """Determine bet decision based on probability"""
    if label_id == 11:
        return "AVOID — reduce stake by 50%", "watch"
    
    if probability >= 0.85:
        return "MAX BET (4% stake)", "max-bet"
    elif probability >= 0.78:
        return "BET (3% stake)", "bet"
    elif probability >= 0.70:
        return "BET (2% stake)", "bet"
    elif probability >= 0.63:
        return "SMALL BET (1% stake)", "small-bet"
    elif probability >= 0.55:
        return "WATCH", "watch"
    else:
        return "NO BET / Consider UNDER", "no-bet"

def run_analysis(input_data: AnalysisInput) -> AnalysisOutput:
    """Complete analysis pipeline"""
    over25, formula, modifier_details = calculate_over25(input_data)
    
    # Clamp
    over25 = max(0.05, min(0.95, over25))
    
    # BTTS (if applicable)
    btts = calculate_btts(input_data)
    if btts is not None:
        btts = max(0.05, min(0.95, btts))
    
    # Decision
    decision, decision_class = decide_stake(over25, input_data.label_id)
    
    return AnalysisOutput(
        over25_prob=over25,
        btts_prob=btts,
        decision=decision,
        decision_class=decision_class,
        formula_breakdown=formula,
        modifier_details=modifier_details
    )

# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_analysis_to_db(input_data: AnalysisInput, output: AnalysisOutput, match_details: dict):
    try:
        record = {
            "league": input_data.league,
            "label_id": input_data.label_id,
            "label_name": input_data.label_name,
            "streak_length": input_data.streak_length,
            "opponent_no_btts_3": input_data.opponent_no_btts_3,
            "opponent_cold_form": input_data.opponent_cold_form,
            "opponent_without_win_5": input_data.opponent_without_win_5,
            "home_team": match_details.get("home_team", ""),
            "away_team": match_details.get("away_team", ""),
            "match_date": str(match_details.get("match_date", date.today())),
            "over25_prob": output.over25_prob,
            "btts_prob": output.btts_prob,
            "decision": output.decision,
            "formula": output.formula_breakdown,
            "result_entered": False,
        }
        response = supabase.table("analyses").insert(record).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_pending_analyses():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", False).order("created_at", desc=True).execute()
        return response.data if response.data else []
    except:
        return []

def submit_result(analysis_id, actual_over25: bool, actual_btts: Optional[bool] = None):
    try:
        update_data = {
            "actual_over25": actual_over25,
            "result_entered": True
        }
        if actual_btts is not None:
            update_data["actual_btts"] = actual_btts
        
        supabase.table("analyses").update(update_data).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_records():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", True).execute()
        if not response.data:
            return []
        
        # Calculate per-label stats
        from collections import defaultdict
        stats = defaultdict(lambda: {"total": 0, "over25_correct": 0, "over25_total": 0, 
                                       "btts_correct": 0, "btts_total": 0})
        
        for r in response.data:
            label = r.get("label_name", "Unknown")
            stats[label]["total"] += 1
            
            if r.get("actual_over25") is not None:
                stats[label]["over25_total"] += 1
                predicted = r.get("over25_prob", 0)
                actual = r.get("actual_over25")
                # Correct if predicted >= 0.50 and actual is True, or predicted < 0.50 and actual is False
                if (predicted >= 0.55 and actual) or (predicted < 0.55 and not actual):
                    stats[label]["over25_correct"] += 1
            
            if r.get("actual_btts") is not None and r.get("btts_prob") is not None:
                stats[label]["btts_total"] += 1
                predicted = r.get("btts_prob", 0)
                actual = r.get("actual_btts")
                if (predicted >= 0.50 and actual) or (predicted < 0.50 and not actual):
                    stats[label]["btts_correct"] += 1
        
        return stats
    except Exception as e:
        st.error(f"Failed to fetch records: {e}")
        return []

# ============================================================================
# UI
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("Label-Based Probability Engine | 12 Labels | League Base Rates")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    with tab1:
        st.markdown("### 📋 Match Details")
        c1, c2 = st.columns(2)
        with c1:
            home_team = st.text_input("🏠 Home Team", key="home_team")
        with c2:
            away_team = st.text_input("✈️ Away Team", key="away_team")
        
        match_date = st.date_input("📅 Match Date", date.today(), key="match_date")
        
        st.divider()
        st.markdown("### 📊 Analysis Inputs")
        
        # League selection
        league = st.selectbox(
            "🏆 League",
            list(LEAGUE_BASES.keys()),
            key="league_select"
        )
        
        # Label selection
        label_options = {v["name"]: k for k, v in LABELS.items()}
        label_name = st.selectbox(
            "🏷️ Label",
            list(label_options.keys()),
            key="label_select"
        )
        label_id = label_options[label_name]
        label_def = LABELS[label_id]
        
        # Streak length (only for labels 1-4)
        streak_length = None
        if label_def["needs_number"]:
            streak_length = st.number_input(
                f"🔢 Streak Length ({label_name})",
                min_value=0,
                max_value=50,
                value=0,
                key="streak_length"
            )
            if streak_length == 0:
                streak_length = None
        
        # Opponent checks
        st.markdown("### 🔍 Opponent Checks")
        opponent_no_btts_3 = False
        opponent_cold_form = False
        opponent_without_win_5 = False
        
        if label_id == 2:  # Only show No BTTS for Win streak
            opponent_no_btts_3 = st.checkbox("☐ Opponent has No BTTS ≥ 3? (Win Streak penalty)", key="no_btts")
        
        opponent_cold_form = st.checkbox("☐ Opponent has Cold Form?", key="cold_form")
        opponent_without_win_5 = st.checkbox("☐ Opponent has Without Win ≥ 5?", key="without_win")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            input_data = AnalysisInput(
                league=league,
                label_id=label_id,
                label_name=label_name,
                streak_length=streak_length,
                opponent_no_btts_3=opponent_no_btts_3,
                opponent_cold_form=opponent_cold_form,
                opponent_without_win_5=opponent_without_win_5,
            )
            
            output = run_analysis(input_data)
            
            # Save to database
            match_details = {
                "home_team": home_team if home_team else "TBD",
                "away_team": away_team if away_team else "TBD",
                "match_date": match_date,
            }
            analysis_id = save_analysis_to_db(input_data, output, match_details)
            if analysis_id:
                st.success(f"✅ Analysis saved (ID: {analysis_id})")
            
            # Display results
            st.markdown("### 🧮 Formula Breakdown")
            st.markdown(f"""
            <div class="formula-box">
            {output.formula_breakdown}
            </div>
            <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">{output.modifier_details}</div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                prob_color = "#10b981" if output.over25_prob >= 0.55 else "#ef4444"
                st.markdown(f"""
                <div class="output-card" style="text-align:center;">
                    <div style="font-size:0.85rem;color:#94a3b8;">Over 2.5 Goals</div>
                    <div style="font-size:2.5rem;font-weight:800;color:{prob_color};">{output.over25_prob:.0%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if output.btts_prob is not None:
                with col2:
                    btts_color = "#10b981" if output.btts_prob >= 0.50 else "#ef4444"
                    st.markdown(f"""
                    <div class="output-card" style="text-align:center;">
                        <div style="font-size:0.85rem;color:#94a3b8;">BTTS</div>
                        <div style="font-size:2.5rem;font-weight:800;color:{btts_color};">{output.btts_prob:.0%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Decision")
            st.markdown(f"""
            <div class="output-card {output.decision_class}">
                <div style="font-size:1.3rem;font-weight:700;">{output.decision}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional context
            if input_data.opponent_cold_form:
                st.markdown("""
                <div class="info-note">
                ℹ️ Opponent cold form noted — consider this additional negative signal.
                </div>
                """, unsafe_allow_html=True)
            
            if input_data.opponent_without_win_5:
                st.markdown("""
                <div class="info-note">
                ℹ️ Opponent without win 5+ — additional boost to Over probability.
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_analyses()
        if pending:
            for analysis in pending:
                with st.expander(f"{analysis.get('home_team', '?')} vs {analysis.get('away_team', '?')} — {analysis.get('label_name', '?')} ({analysis.get('league', '?')})"):
                    st.write(f"**Predicted:** Over 2.5 = {analysis.get('over25_prob', 0):.0%}")
                    if analysis.get('btts_prob'):
                        st.write(f"**Predicted BTTS:** {analysis.get('btts_prob', 0):.0%}")
                    st.write(f"**Decision:** {analysis.get('decision', '?')}")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        actual_over25 = st.selectbox(
                            "Over 2.5?",
                            ["Pending", "Yes", "No"],
                            key=f"over25_{analysis['id']}"
                        )
                    with c2:
                        if analysis.get('btts_prob'):
                            actual_btts = st.selectbox(
                                "BTTS?",
                                ["Pending", "Yes", "No"],
                                key=f"btts_{analysis['id']}"
                            )
                        else:
                            actual_btts = "Pending"
                    
                    if st.button("✅ Submit", key=f"submit_{analysis['id']}"):
                        over25_bool = True if actual_over25 == "Yes" else (False if actual_over25 == "No" else None)
                        btts_bool = True if actual_btts == "Yes" else (False if actual_btts == "No" else None)
                        
                        if submit_result(analysis['id'], over25_bool, btts_bool):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending analyses.")
    
    with tab3:
        st.subheader("📊 Live Records by Label")
        records = get_records()
        if records:
            for label_name, stats in sorted(records.items()):
                total = stats["total"]
                if total == 0:
                    continue
                
                over25_total = stats["over25_total"]
                over25_correct = stats["over25_correct"]
                over25_acc = (over25_correct / over25_total * 100) if over25_total > 0 else 0
                
                btts_total = stats["btts_total"]
                btts_correct = stats["btts_correct"]
                btts_acc = (btts_correct / btts_total * 100) if btts_total > 0 else 0
                
                color = "#10b981" if over25_acc >= 70 else "#fbbf24" if over25_acc >= 50 else "#ef4444"
                
                btts_str = f" | BTTS: {btts_correct}/{btts_total} ({btts_acc:.0f}%)" if btts_total > 0 else ""
                
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                    <div><strong>{label_name}</strong> <span style="font-size:0.8rem;color:#94a3b8;">({total} total)</span></div>
                    <div style="color:{color};">O2.5: {over25_correct}/{over25_total} ({over25_acc:.0f}%){btts_str}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No results recorded yet.")
    
    st.divider()
    st.markdown("""
    ### 📋 12 Labels Reference
    
    | # | Label | Number? | Effect on Over 2.5 |
    |---|-------|---------|---------------------|
    | 1 | Scoring run | Yes | +0.05 to +0.20 |
    | 2 | Win streak | Yes | +0.06 to +0.10 (penalty -0.10) |
    | 3 | Unbeaten | Yes | -0.05 to -0.12 |
    | 4 | Clean sheet | Yes | -0.15 to -0.20 |
    | 5 | Hot Attack 2 | No | Override to 0.90 |
    | 6 | High Goals | No | +0.18 (+ BTTS +0.10) |
    | 7 | BTTS Lock | No | +0.05 (+ BTTS +0.20) |
    | 8 | Clean Sheet Unlikely | No | +0.15 (+ BTTS +0.12) |
    | 9 | Low Quality | No | -0.20 (+ BTTS -0.15) |
    | 10 | Hot Clash | No | +0.10 |
    | 11 | Strong form clash | No | No change (AVOID) |
    | 12 | Tight Game | No | -0.15 |
    
    **Stake Tiers:**
    - ≥85% → MAX BET (4%)
    - ≥78% → BET (3%)
    - ≥70% → BET (2%)
    - ≥63% → SMALL BET (1%)
    - ≥55% → WATCH
    - <55% → NO BET / UNDER
    """)

if __name__ == "__main__":
    main()
