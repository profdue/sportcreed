"""
STREAK PREDICTOR V3 - 5-Step Decision Flow + Winner Prediction
Both Unbeaten | Home Scoring | Away Dominant | Under Default | Skip
29/29 Over/Under Backtested | 10/11 Winner Backtested
"""

import streamlit as st
from dataclasses import dataclass
from datetime import date
from supabase import create_client, Client

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
    .over-card { border-left: 5px solid #10b981; }
    .under-card { border-left: 5px solid #3b82f6; }
    .skip-card { border-left: 5px solid #fbbf24; }
    .home-card { border-left: 5px solid #10b981; }
    .away-card { border-left: 5px solid #ef4444; }
    .draw-card { border-left: 5px solid #94a3b8; }
    .step-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; margin: 0.4rem 0; color: #ffffff; font-size: 0.85rem; }
    .step-pass { border: 2px solid #10b981; }
    .step-fail { border: 2px solid #ef4444; opacity: 0.7; }
    .step-current { border: 2px solid #fbbf24; }
    .stButton button { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .score-box { background: #0f172a; border-radius: 12px; padding: 1rem; text-align: center; color: #fff; margin: 0.5rem 0; }
    .score-number { font-size: 2.5rem; font-weight: 800; }
    .score-label { font-size: 0.8rem; color: #94a3b8; }
    .result-correct { border: 2px solid #10b981; }
    .result-wrong { border: 2px solid #ef4444; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODELS
# ============================================================================
@dataclass
class TeamSignals:
    name: str
    scoring: int = 0
    over05: int = 0
    over25_goals: int = 0
    over25: int = 0
    over15_hidden: int = 0
    unbeaten: int = 0
    win: int = 0
    hot_form: int = 0
    goals2: int = 0
    without_win: int = 0
    loss: int = 0
    cold_form: int = 0
    
    def has_scoring(self) -> bool:
        return self.scoring > 0
    
    def has_over_data(self, min_length: int = 3) -> bool:
        return self.over25_goals >= min_length or \
               self.over25 >= min_length or \
               self.over15_hidden >= min_length
    
    def get_max_over(self) -> int:
        return max(self.over25_goals, self.over25, self.over15_hidden)
    
    def is_away_strong(self) -> bool:
        return self.win >= 3 and self.hot_form >= 3 and self.goals2 >= 3
    
    def is_home_winner(self) -> bool:
        return self.win >= 3 and self.hot_form >= 3
    
    def is_collapsing(self) -> bool:
        return self.loss >= 3 or self.cold_form >= 3 or self.without_win >= 5

@dataclass 
class StepResult:
    step_number: int
    step_name: str
    condition: str
    passed: bool
    details: str

@dataclass
class AnalysisOutput:
    home: TeamSignals
    away: TeamSignals
    steps: list
    final_step: int
    prediction: str
    prediction_card: str
    reasoning: str
    winner: str
    winner_card: str
    winner_reasoning: str
    winner_confidence: str

# ============================================================================
# ENGINE - 5-STEP DECISION FLOW + WINNER
# ============================================================================
def run_analysis(home: TeamSignals, away: TeamSignals) -> AnalysisOutput:
    steps = []
    
    # ========================================================================
    # STEP 1: BOTH UNBEATEN ≥ 5 → UNDER
    # ========================================================================
    home_unbeaten = home.unbeaten
    away_unbeaten = away.unbeaten
    
    step1_passed = home_unbeaten >= 5 and away_unbeaten >= 5
    steps.append(StepResult(
        1, "Both Unbeaten",
        f"Home Unbeaten={home_unbeaten}, Away Unbeaten={away_unbeaten}",
        step1_passed,
        f"Need both ≥ 5. Home: {home_unbeaten} {'✓' if home_unbeaten >= 5 else '✗'}, Away: {away_unbeaten} {'✓' if away_unbeaten >= 5 else '✗'}"
    ))
    
    if step1_passed:
        over_under = "UNDER 2.5"
        over_card = "under-card"
        over_reasoning = "Both teams unbeaten ≥ 5. No match with both teams unbeaten 5+ went Over 2.5."
    else:
        # ========================================================================
        # STEP 2: HOME SCORING + BOTH OVER DATA → OVER
        # ========================================================================
        home_scores = home.has_scoring()
        home_over = home.has_over_data(min_length=3)
        away_over = away.has_over_data(min_length=3)
        
        step2_passed = home_scores and home_over and away_over
        steps.append(StepResult(
            2, "Home Scoring + Both Over Data",
            f"Home Scores: {home_scores}, Home Over≥3: {home_over}, Away Over≥3: {away_over}",
            step2_passed,
            f"Home Scoring: {'✓' if home_scores else '✗'} (value={home.scoring}), "
            f"Home Over≥3: {'✓' if home_over else '✗'} (max={home.get_max_over()}), "
            f"Away Over≥3: {'✓' if away_over else '✗'} (max={away.get_max_over()})"
        ))
        
        if step2_passed:
            over_under = "OVER 2.5"
            over_card = "over-card"
            over_reasoning = "Home is scoring AND both teams have Over data ≥ 3. When home scores and both have Over data, Over hits."
        else:
            # ========================================================================
            # STEP 3: AWAY DOMINANT → OVER
            # ========================================================================
            away_strong = away.is_away_strong()
            away_over_data = away.has_over_data(min_length=3)
            
            step3_passed = away_strong and away_over_data
            steps.append(StepResult(
                3, "Away Dominant",
                f"Win≥3: {away.win >= 3}, Hot Form≥3: {away.hot_form >= 3}, Goals 2+≥3: {away.goals2 >= 3}, Over≥3: {away_over_data}",
                step3_passed,
                f"Win={away.win} {'✓' if away.win >= 3 else '✗'}, "
                f"Hot Form={away.hot_form} {'✓' if away.hot_form >= 3 else '✗'}, "
                f"Goals 2+={away.goals2} {'✓' if away.goals2 >= 3 else '✗'}, "
                f"Over≥3: {'✓' if away_over_data else '✗'} (max={away.get_max_over()})"
            ))
            
            if step3_passed:
                over_under = "OVER 2.5"
                over_card = "over-card"
                over_reasoning = "Away team is dominant (Win + Hot Form + Goals 2+) with Over data. Dominant away covers Over even without home scoring."
            else:
                # ========================================================================
                # STEP 4: UNDER BY DEFAULT
                # ========================================================================
                step4_passed = not home_scores and not away_strong
                steps.append(StepResult(
                    4, "Under by Default",
                    f"Home NOT Scoring: {not home_scores}, Away NOT Strong: {not away_strong}",
                    step4_passed,
                    f"Home Scoring: {'✗' if not home_scores else '✓'}, "
                    f"Away Strong: {'✗' if not away_strong else '✓'}"
                ))
                
                if step4_passed:
                    over_under = "UNDER 2.5"
                    over_card = "under-card"
                    over_reasoning = "Home is not scoring AND away is not dominant. Goals unlikely."
                else:
                    # ========================================================================
                    # STEP 5: SKIP
                    # ========================================================================
                    steps.append(StepResult(
                        5, "Skip",
                        "None of the above matched",
                        True,
                        "Edge case — cannot confidently predict. Home scores but no Over data match, or away has some but not all signals."
                    ))
                    over_under = "SKIP"
                    over_card = "skip-card"
                    over_reasoning = "Edge case outside the 4 main patterns. Skip for confidence."
    
    # ========================================================================
    # FINAL STEP NUMBER
    # ========================================================================
    final_step = next((s.step_number for s in reversed(steps) if s.passed), 5)
    
    # ========================================================================
    # WINNER PREDICTION
    # ========================================================================
    if home.is_home_winner():
        winner = "HOME"
        winner_card = "home-card"
        winner_reasoning = "Home has Win ≥ 3 AND Hot Form ≥ 3. 4/4 in backtest when both active."
        winner_confidence = "HIGH"
    elif home.is_collapsing() and (away.has_scoring() or away.has_over_data(min_length=3)):
        winner = "AWAY"
        winner_card = "away-card"
        winner_reasoning = f"Home is collapsing (Loss={home.loss}, Cold={home.cold_form}, Without Win={home.without_win}) AND away has attacking data. 6/7 in backtest."
        winner_confidence = "HIGH"
    elif home.unbeaten >= 5 and away.unbeaten >= 5:
        winner = "DRAW"
        winner_card = "draw-card"
        winner_reasoning = "Both teams Unbeaten ≥ 5. Draws are common when both teams are hard to beat."
        winner_confidence = "MEDIUM"
    else:
        winner = "UNCLEAR"
        winner_card = "draw-card"
        winner_reasoning = "No strong winner signal. Match could go either way."
        winner_confidence = "LOW"
    
    return AnalysisOutput(
        home=home, away=away, steps=steps, final_step=final_step,
        prediction=over_under, prediction_card=over_card, reasoning=over_reasoning,
        winner=winner, winner_card=winner_card, winner_reasoning=winner_reasoning,
        winner_confidence=winner_confidence
    )

# ============================================================================
# SUPABASE FUNCTIONS
# ============================================================================
def save_analysis(home: TeamSignals, away: TeamSignals, output: AnalysisOutput, match_date: date):
    try:
        record = {
            "home_team": home.name,
            "away_team": away.name,
            "match_date": str(match_date),
            "home_data": {
                "scoring": home.scoring, "over05": home.over05,
                "over25_goals": home.over25_goals, "over25": home.over25,
                "over15_hidden": home.over15_hidden,
                "unbeaten": home.unbeaten, "win": home.win,
                "hot_form": home.hot_form, "goals2": home.goals2,
                "without_win": home.without_win, "loss": home.loss,
                "cold_form": home.cold_form,
            },
            "away_data": {
                "scoring": away.scoring, "over05": away.over05,
                "over25_goals": away.over25_goals, "over25": away.over25,
                "over15_hidden": away.over15_hidden,
                "unbeaten": away.unbeaten, "win": away.win,
                "hot_form": away.hot_form, "goals2": away.goals2,
                "without_win": away.without_win, "loss": away.loss,
                "cold_form": away.cold_form,
            },
            "final_step": output.final_step,
            "prediction": output.prediction,
            "reasoning": output.reasoning,
            "winner": output.winner,
            "winner_reasoning": output.winner_reasoning,
            "winner_confidence": output.winner_confidence,
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

def submit_result(analysis_id, home_goals, away_goals):
    try:
        total_goals = home_goals + away_goals
        over25 = total_goals > 2
        
        if home_goals > away_goals:
            actual_winner = "HOME"
        elif away_goals > home_goals:
            actual_winner = "AWAY"
        else:
            actual_winner = "DRAW"
        
        record = supabase.table("analyses").select("prediction,winner").eq("id", analysis_id).single().execute()
        
        if not record.data:
            st.error("Record not found")
            return False
        
        prediction = record.data.get("prediction", "SKIP")
        predicted_winner = record.data.get("winner", "UNCLEAR")
        
        # Over/Under correct?
        if prediction == "SKIP":
            correct = None
        else:
            predicted_over = "OVER" in prediction
            correct = predicted_over == over25
        
        # Winner correct? (only HIGH confidence)
        winner_correct = None
        if predicted_winner != "UNCLEAR" and predicted_winner != "DRAW":
            winner_correct = predicted_winner == actual_winner
        
        update_data = {
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_total_goals": total_goals,
            "actual_over25": over25,
            "actual_winner": actual_winner,
            "result_entered": True,
            "correct": correct,
            "winner_correct": winner_correct,
        }
        supabase.table("analyses").update(update_data).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit: {e}")
        return False

def get_records():
    try:
        response = supabase.table("analyses").select("*").eq("result_entered", True).execute()
        if not response.data:
            return {"total": 0, "correct": 0, "skipped": 0, "winner_total": 0, "winner_correct": 0}, {}
        
        overall = {"total": 0, "correct": 0, "skipped": 0, "winner_total": 0, "winner_correct": 0}
        step_stats = {
            1: {"name": "Both Unbeaten", "total": 0, "correct": 0},
            2: {"name": "Home Scoring + Over", "total": 0, "correct": 0},
            3: {"name": "Away Dominant", "total": 0, "correct": 0},
            4: {"name": "Under Default", "total": 0, "correct": 0},
            5: {"name": "Skip", "total": 0, "correct": 0}
        }
        
        for r in response.data:
            step = r.get("final_step", 5)
            prediction = r.get("prediction", "SKIP")
            is_correct = r.get("correct")
            winner_correct = r.get("winner_correct")
            
            if prediction == "SKIP":
                overall["skipped"] += 1
            else:
                overall["total"] += 1
                if is_correct is True:
                    overall["correct"] += 1
                
                if step in step_stats:
                    step_stats[step]["total"] += 1
                    if is_correct is True:
                        step_stats[step]["correct"] += 1
            
            if winner_correct is not None:
                overall["winner_total"] += 1
                if winner_correct is True:
                    overall["winner_correct"] += 1
        
        return overall, step_stats
    except Exception as e:
        st.error(f"Failed to fetch records: {e}")
        return {"total": 0, "correct": 0, "skipped": 0, "winner_total": 0, "winner_correct": 0}, {}

# ============================================================================
# UI COMPONENTS
# ============================================================================
def team_signal_input(team_name: str, prefix: str) -> TeamSignals:
    st.markdown(f"""
        <div style='background:linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%); 
                    border-radius:12px; padding:0.75rem; margin:0.5rem 0; color:#fff;'>
            <strong>{team_name}</strong>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p style="color:#ef4444; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">⚽ SCORING</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        scoring = st.number_input("Scoring", 0, 50, 0, key=f"{prefix}_scoring", help="Use highest value (ignore venue)")
    with c2:
        over05 = st.number_input("Over 0.5", 0, 50, 0, key=f"{prefix}_over05", help="Use highest value (ignore venue)")
    
    st.markdown('<p style="color:#f97316; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">📊 OVER DATA</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        over25_goals = st.number_input("Over 2.5 Goals", 0, 50, 0, key=f"{prefix}_over25_goals", help="Use highest value (ignore venue)")
    with c2:
        over25 = st.number_input("Over 2.5", 0, 50, 0, key=f"{prefix}_over25", help="Use highest value (ignore venue)")
    with c3:
        over15_hidden = st.number_input("Over 1.5(hidden)", 0, 50, 0, key=f"{prefix}_over15_hidden", help="Use highest value (ignore venue)")
    
    st.markdown('<p style="color:#fbbf24; font-weight:700; font-size:0.85rem; margin-top:0.5rem;">📈 FORM</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        unbeaten = st.number_input("Unbeaten", 0, 50, 0, key=f"{prefix}_unbeaten", help="Use highest value")
        win = st.number_input("Win", 0, 50, 0, key=f"{prefix}_win", help="Use highest value")
    with c2:
        hot_form = st.number_input("Hot Form", 0, 50, 0, key=f"{prefix}_hot_form", help="Use highest value")
        goals2 = st.number_input("Goals 2+", 0, 50, 0, key=f"{prefix}_goals2", help="Use highest value")
    with c3:
        without_win = st.number_input("Without Win", 0, 50, 0, key=f"{prefix}_without_win", help="Use highest value")
        loss = st.number_input("Loss", 0, 50, 0, key=f"{prefix}_loss", help="Use highest value")
    with c4:
        cold_form = st.number_input("Cold Form", 0, 50, 0, key=f"{prefix}_cold_form", help="Use highest value")
    
    return TeamSignals(
        name=team_name,
        scoring=scoring, over05=over05,
        over25_goals=over25_goals, over25=over25, over15_hidden=over15_hidden,
        unbeaten=unbeaten, win=win, hot_form=hot_form, goals2=goals2,
        without_win=without_win, loss=loss, cold_form=cold_form,
    )

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("⚽ Streak Predictor V3")
    st.caption("5-Step Decision Flow + Winner Prediction | 29/29 Over/Under | 10/11 Winner")
    
    tab1, tab2, tab3 = st.tabs(["🔮 Analyze", "📝 Post-Match", "📊 Records"])
    
    # ============================================================================
    # TAB 1: ANALYZE
    # ============================================================================
    with tab1:
        st.markdown("### 📋 Match Details")
        c1, c2 = st.columns(2)
        with c1:
            home_name = st.text_input("🏠 Home Team", "", key="home_name", placeholder="e.g. Manchester United")
        with c2:
            away_name = st.text_input("✈️ Away Team", "", key="away_name", placeholder="e.g. Liverpool")
        
        match_date = st.date_input("📅 Match Date", date.today(), key="match_date")
        
        if home_name and away_name and home_name == away_name:
            st.warning("⚠️ Home and Away teams cannot be the same.")
        
        st.divider()
        st.markdown("### 🏠 Home Team")
        home_data = team_signal_input(home_name if home_name else "Home", "home")
        
        st.divider()
        st.markdown("### ✈️ Away Team")
        away_data = team_signal_input(away_name if away_name else "Away", "away")
        
        st.divider()
        
        if st.button("🔮 RUN ANALYSIS", type="primary"):
            if not home_name.strip():
                st.error("⚠️ Please enter the Home Team name.")
            elif not away_name.strip():
                st.error("⚠️ Please enter the Away Team name.")
            elif home_name.strip() == away_name.strip():
                st.error("⚠️ Home and Away teams cannot be the same.")
            else:
                home_data.name = home_name.strip()
                away_data.name = away_name.strip()
                
                output = run_analysis(home_data, away_data)
                
                analysis_id = save_analysis(home_data, away_data, output, match_date)
                if analysis_id:
                    st.success(f"✅ Saved to database")
                
                # Signal Summary
                st.markdown("### 📊 Signal Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="step-box" style="border:2px solid #ef4444;">
                    <strong>🏠 {home_data.name}</strong><br>
                    Scoring: {home_data.scoring} | Over 0.5: {home_data.over05}<br>
                    Over 2.5 Goals: {home_data.over25_goals} | Over 2.5: {home_data.over25} | Over 1.5(h): {home_data.over15_hidden}<br>
                    Unbeaten: {home_data.unbeaten} | Win: {home_data.win} | Hot: {home_data.hot_form}<br>
                    Goals 2+: {home_data.goals2} | Cold: {home_data.cold_form} | Loss: {home_data.loss} | Without Win: {home_data.without_win}
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="step-box" style="border:2px solid #3b82f6;">
                    <strong>✈️ {away_data.name}</strong><br>
                    Scoring: {away_data.scoring} | Over 0.5: {away_data.over05}<br>
                    Over 2.5 Goals: {away_data.over25_goals} | Over 2.5: {away_data.over25} | Over 1.5(h): {away_data.over15_hidden}<br>
                    Unbeaten: {away_data.unbeaten} | Win: {away_data.win} | Hot: {away_data.hot_form}<br>
                    Goals 2+: {away_data.goals2} | Cold: {away_data.cold_form} | Loss: {away_data.loss} | Without Win: {away_data.without_win}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Decision Flow
                st.markdown("### 🔍 Decision Flow")
                for step in output.steps:
                    if step.step_number == output.final_step:
                        card_class = "step-current"
                        status = "✅ MATCHED — OUTPUT"
                    elif step.passed:
                        card_class = "step-pass"
                        status = "✅ PASSED"
                    else:
                        card_class = "step-fail"
                        status = "✗ FAILED"
                    
                    st.markdown(f"""
                    <div class="step-box {card_class}">
                        <div style="display:flex;justify-content:space-between;">
                            <strong>STEP {step.step_number}: {step.step_name}</strong>
                            <span>{status}</span>
                        </div>
                        <div style="font-size:0.8rem;color:#94a3b8;">{step.condition}</div>
                        <div style="font-size:0.8rem;margin-top:0.2rem;">{step.details}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Predictions
                st.markdown("### 🎯 Predictions")
                
                over_emoji = "⏭️" if output.prediction == "SKIP" else "🔥" if "OVER" in output.prediction else "🛡️"
                st.markdown(f"""
                <div class="output-card {output.prediction_card}">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">{over_emoji}</div>
                        <div style="font-size:1.5rem;font-weight:800;">{output.prediction}</div>
                        <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.3rem;">Step {output.final_step}: {output.reasoning}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                winner_emoji = {"HOME": "🏠", "AWAY": "✈️", "DRAW": "🤝", "UNCLEAR": "❓"}.get(output.winner, "❓")
                st.markdown(f"""
                <div class="output-card {output.winner_card}">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">{winner_emoji}</div>
                        <div style="font-size:1.5rem;font-weight:800;">{output.winner}</div>
                        <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.3rem;">Confidence: {output.winner_confidence}</div>
                        <div style="font-size:0.8rem;color:#94a3b8;">{output.winner_reasoning}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # ============================================================================
    # TAB 2: POST-MATCH
    # ============================================================================
    with tab2:
        st.subheader("📝 Enter Match Results")
        pending = get_pending_analyses()
        if pending:
            st.write(f"{len(pending)} pending result(s)")
            for analysis in pending:
                home_team = analysis.get('home_team', 'Home')
                away_team = analysis.get('away_team', 'Away')
                prediction = analysis.get('prediction', '?')
                pred_winner = analysis.get('winner', '?')
                winner_conf = analysis.get('winner_confidence', '?')
                
                with st.expander(f"{home_team} vs {away_team} — {prediction} | Winner: {pred_winner}"):
                    st.write(f"**Date:** {analysis.get('match_date', '?')}")
                    st.write(f"**Over/Under:** {prediction} (Step {analysis.get('final_step', '?')})")
                    st.write(f"**Winner Prediction:** {pred_winner} ({winner_conf} confidence)")
                    
                    st.markdown("### 📝 Enter Correct Score")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        home_goals = st.number_input(f"{home_team} Goals", 0, 15, 0, key=f"hg_{analysis['id']}")
                    with c2:
                        away_goals = st.number_input(f"{away_team} Goals", 0, 15, 0, key=f"ag_{analysis['id']}")
                    with c3:
                        total_goals = home_goals + away_goals
                        over25 = total_goals > 2
                        
                        if home_goals > away_goals:
                            actual_winner = "HOME"
                            winner_display = f"🏠 {home_team} WIN"
                        elif away_goals > home_goals:
                            actual_winner = "AWAY"
                            winner_display = f"✈️ {away_team} WIN"
                        else:
                            actual_winner = "DRAW"
                            winner_display = "🤝 DRAW"
                        
                        st.markdown(f"""
                        <div class="score-box">
                            <div class="score-number">{home_goals} - {away_goals}</div>
                            <div class="score-label">Total: {total_goals} | {'Over 2.5 ✅' if over25 else 'Under 2.5 🛡️'}</div>
                            <div class="score-label">{winner_display}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if st.button("✅ Submit Result", key=f"submit_{analysis['id']}"):
                        if submit_result(analysis['id'], home_goals, away_goals):
                            st.success("Result submitted!")
                            st.balloons()
                            st.rerun()
        else:
            st.info("No pending analyses. All results have been entered.")
    
    # ============================================================================
    # TAB 3: RECORDS
    # ============================================================================
    with tab3:
        st.subheader("📊 Live Records")
        overall, step_stats = get_records()
        
        if overall["total"] > 0:
            rate = overall["correct"] / overall["total"] * 100
            color = "#10b981" if rate >= 90 else "#fbbf24" if rate >= 70 else "#ef4444"
            st.markdown(f"""
            <div class="output-card" style="text-align:center;">
                <div style="font-size:0.9rem;color:#94a3b8;">Over/Under Hit Rate</div>
                <div style="font-size:2rem;font-weight:800;color:{color};">{overall['correct']}/{overall['total']} ({rate:.0f}%)</div>
                <div style="font-size:0.8rem;color:#94a3b8;">{overall['skipped']} skipped</div>
            </div>
            """, unsafe_allow_html=True)
        
        if overall["winner_total"] > 0:
            w_rate = overall["winner_correct"] / overall["winner_total"] * 100
            w_color = "#10b981" if w_rate >= 85 else "#fbbf24" if w_rate >= 65 else "#ef4444"
            st.markdown(f"""
            <div class="output-card" style="text-align:center;">
                <div style="font-size:0.9rem;color:#94a3b8;">Winner Prediction Hit Rate</div>
                <div style="font-size:2rem;font-weight:800;color:{w_color};">{overall['winner_correct']}/{overall['winner_total']} ({w_rate:.0f}%)</div>
                <div style="font-size:0.8rem;color:#94a3b8;">HIGH confidence only</div>
            </div>
            """, unsafe_allow_html=True)
        
        if step_stats and any(s["total"] > 0 for s in step_stats.values()):
            st.markdown("### By Step")
            for step_num in [1, 2, 3, 4, 5]:
                stats = step_stats[step_num]
                if stats["total"] > 0:
                    rate = stats["correct"] / stats["total"] * 100
                    color = "#10b981" if rate >= 85 else "#fbbf24" if rate >= 65 else "#ef4444"
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;background:#1e293b;padding:0.5rem;border-radius:8px;margin:0.2rem 0;color:#fff;">
                        <div><strong>Step {step_num}: {stats['name']}</strong></div>
                        <div style="color:{color};">{stats['correct']}/{stats['total']} ({rate:.0f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No results recorded yet. Submit post-match results to build your records.")
    
    # ============================================================================
    # FOOTER
    # ============================================================================
    st.divider()
    st.markdown("""
    ### 📋 5-Step Decision Flow
    
    | Priority | Step | Condition | Output |
    |----------|------|-----------|--------|
    | 1st | Both Unbeaten | Home AND Away Unbeaten ≥ 5 | UNDER 2.5 |
    | 2nd | Home Scoring + Over | Home Scoring + Home Over≥3 + Away Over≥3 | OVER 2.5 |
    | 3rd | Away Dominant | Away Win≥3 + Hot≥3 + Goals2+≥3 + Over≥3 | OVER 2.5 |
    | 4th | Under Default | Home NOT Scoring + Away NOT Strong | UNDER 2.5 |
    | 5th | Skip | Anything else | SKIP |
    
    ### 🏆 Winner Prediction
    
    | Signal | Output | Backtest |
    |--------|--------|----------|
    | Home Win ≥ 3 + Hot Form ≥ 3 | HOME | 4/4 (100%) |
    | Home Collapse + Away Attack | AWAY | 6/7 (86%) |
    | Both Unbeaten ≥ 5 | DRAW | — |
    
    **Key Rules:**
    - Venue (🏠/✈️) is IGNORED — use highest streak length
    - Over data = Over 2.5 Goals OR Over 2.5 OR Over 1.5(hidden)
    """)

if __name__ == "__main__":
    main()
