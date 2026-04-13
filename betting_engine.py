"""
Two-Part Rule Predictor - 3 Question Version
Based on 27 matches of proven patterns
"""

import streamlit as st

st.set_page_config(
    page_title="Two-Part Rule Predictor",
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
        max-width: 600px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .bet-against {
        border-left: 6px solid #ef4444;
    }
    .bet-on {
        border-left: 6px solid #10b981;
    }
    .prediction-text {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .prediction-detail {
        font-size: 1rem;
        color: #fbbf24;
    }
    .question-box {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Two-Part Rule Predictor")
    st.caption("3 questions. 1 prediction. Based on 27 matches.")
    
    st.divider()
    
    # Team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.text_input("🏠 Home team", placeholder="e.g., Baath Bouhajla")
    
    with col2:
        away_team = st.text_input("✈️ Away team", placeholder="e.g., Sfax Railways")
    
    st.divider()
    
    # Question 1
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.markdown("**📊 Question 1 of 3**")
    more_positives = st.radio(
        "Which team has MORE positive streaks?",
        options=["Home", "Away", "Equal / Not sure"],
        index=None,
        horizontal=True
    )
    st.caption("Positive streaks = Unbeaten, Won, Scored 1+, Clean sheet, etc.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 2
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.markdown("**🚫 Question 2 of 3**")
    away_scored_none = st.radio(
        "Does AWAY team have a 'Scored none' streak?",
        options=["Yes", "No"],
        index=None,
        horizontal=True
    )
    st.caption("Example: 'Failed to score in last 5 away games'")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Question 3
    st.markdown('<div class="question-box">', unsafe_allow_html=True)
    st.markdown("**🚫 Question 3 of 3**")
    home_scored_none = st.radio(
        "Does HOME team have a 'Scored none' streak?",
        options=["Yes", "No"],
        index=None,
        horizontal=True
    )
    st.caption("Example: 'No goals scored in last 6 games'")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        # Validation
        if not home_team:
            st.error("Please enter Home team")
            return
        if not away_team:
            st.error("Please enter Away team")
            return
        if more_positives is None:
            st.error("Please answer Question 1")
            return
        if away_scored_none is None:
            st.error("Please answer Question 2")
            return
        if home_scored_none is None:
            st.error("Please answer Question 3")
            return
        
        # Logic
        prediction = ""
        detail = ""
        reasoning = []
        
        # Case 1: Both have scored none → Under 2.5
        if home_scored_none == "Yes" and away_scored_none == "Yes":
            prediction = "🏆 UNDER 2.5 GOALS"
            detail = "Both teams cannot score"
            reasoning.append("Both home and away have 'Scored none' streaks")
            reasoning.append("→ Under 2.5 goals is the strongest bet")
        
        # Case 2: Home has more positives + Away has scored none → Home wins
        elif more_positives == "Home" and away_scored_none == "Yes":
            prediction = "✅ BET ON HOME TEAM"
            detail = f"{home_team} wins"
            reasoning.append(f"{home_team} has more positive streaks")
            reasoning.append(f"{away_team} has 'Scored none' streak")
            reasoning.append("→ Situation B: Bet ON favorite (Home)")
        
        # Case 3: Away has more positives + Home has scored none → Away wins
        elif more_positives == "Away" and home_scored_none == "Yes":
            prediction = "✅ BET ON AWAY TEAM"
            detail = f"{away_team} wins"
            reasoning.append(f"{away_team} has more positive streaks")
            reasoning.append(f"{home_team} has 'Scored none' streak")
            reasoning.append("→ Situation B: Bet ON favorite (Away)")
        
        # Case 4: Home has more positives + No scored none for away → Draw risk
        elif more_positives == "Home" and away_scored_none == "No":
            prediction = "⚠️ BET AGAINST HOME"
            detail = f"Draw or {away_team} win"
            reasoning.append(f"{home_team} has more positives but {away_team} has NO 'scored none'")
            reasoning.append("→ Situation A: Clash → Favorite may fail")
        
        # Case 5: Away has more positives + No scored none for home → Draw risk
        elif more_positives == "Away" and home_scored_none == "No":
            prediction = "⚠️ BET AGAINST AWAY"
            detail = f"Draw or {home_team} win"
            reasoning.append(f"{away_team} has more positives but {home_team} has NO 'scored none'")
            reasoning.append("→ Situation A: Clash → Favorite may fail")
        
        # Case 6: Equal or unclear
        else:
            prediction = "⚖️ NO CLEAR EDGE"
            detail = "Pass or bet Under 2.5 goals"
            reasoning.append("Teams are evenly matched or unclear")
            reasoning.append("→ No confident bet")
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-text">🏆 PREDICTION</div>
            <div class="prediction-detail">{prediction}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning
        with st.expander("📋 Show reasoning"):
            for r in reasoning:
                st.markdown(f"• {r}")
    
    st.divider()
    st.caption("""
    **How to answer:**\n
    • **Question 1:** Which team has more unbeaten/won/scored streaks?\n
    • **Question 2:** Does away team have "failed to score" streak?\n
    • **Question 3:** Does home team have "failed to score" streak?
    """)


if __name__ == "__main__":
    main()
