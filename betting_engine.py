"""
Winner Predictor - Final Version
3 Situations based on 27 matches (78-80% accuracy)

Situation A: Underdog has positive streak → Bet AGAINST favorite
Situation B: Underdog has NO positive streak → Bet ON favorite
Situation C: Favorite has "Scored none" streak → Bet AGAINST favorite
"""

import streamlit as st

st.set_page_config(
    page_title="Winner Predictor",
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
        max-width: 550px;
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
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .prediction-detail {
        font-size: 1rem;
        color: #fbbf24;
        margin-top: 0.5rem;
    }
    .favorite-box {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #334155;
    }
    hr {
        margin: 1rem 0;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Winner Predictor")
    st.caption("3 situations. 1 prediction. Based on 27 matches.")
    
    st.divider()
    
    # Team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        favorite_team = st.text_input("🏆 Favorite team", placeholder="e.g., EM Mahdia")
    
    with col2:
        underdog_team = st.text_input("🐕 Underdog team", placeholder="e.g., Chebba")
    
    st.divider()
    
    # Question 1
    st.markdown("### 📊 Question 1 of 2")
    underdog_positive = st.radio(
        "Does the UNDERDOG have ANY positive streak?",
        options=["Yes", "No"],
        index=None,
        horizontal=True
    )
    st.caption("Positive streaks = Unbeaten, Won, Drawn, Scored 1+, Clean sheet, H2H unbeaten")
    
    st.divider()
    
    # Question 2
    st.markdown("### 🚫 Question 2 of 2")
    favorite_scored_none = st.radio(
        "Does the FAVORITE have a 'Scored none' streak (3+ games)?",
        options=["Yes", "No"],
        index=None,
        horizontal=True
    )
    st.caption("Example: 'Failed to score in last 3 matches' or 'No goals in last 5 games'")
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT"):
        # Validation
        if not favorite_team:
            st.error("Please enter Favorite team")
            return
        if not underdog_team:
            st.error("Please enter Underdog team")
            return
        if underdog_positive is None:
            st.error("Please answer Question 1")
            return
        if favorite_scored_none is None:
            st.error("Please answer Question 2")
            return
        
        # Logic
        if favorite_scored_none == "Yes":
            # Situation C
            prediction = "⚠️ BET AGAINST FAVORITE"
            detail = f"Draw or {underdog_team} win"
            card_class = "bet-against"
            reasoning = f"""
            • Favorite: {favorite_team} has 'Scored none' streak
            • Situation C → Bet AGAINST favorite
            """
        elif underdog_positive == "Yes":
            # Situation A
            prediction = "⚠️ BET AGAINST FAVORITE"
            detail = f"Draw or {underdog_team} win"
            card_class = "bet-against"
            reasoning = f"""
            • Underdog: {underdog_team} has positive streak
            • Favorite: {favorite_team} has NO 'Scored none' streak
            • Situation A (Clash) → Bet AGAINST favorite
            """
        else:
            # Situation B
            prediction = "✅ BET ON FAVORITE"
            detail = f"{favorite_team} wins"
            card_class = "bet-on"
            reasoning = f"""
            • Underdog: {underdog_team} has NO positive streak
            • Favorite: {favorite_team} has NO 'Scored none' streak
            • Situation B (Pure negative underdog) → Bet ON favorite
            """
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-text">{prediction}</div>
            <div class="prediction-detail">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning
        with st.expander("📋 Show reasoning"):
            st.markdown(reasoning)
    
    st.divider()
    st.caption("""
    **How to answer:**\n
    • **Question 1:** Does underdog have ANY unbeaten, won, drawn, or scored streak?\n
    • **Question 2:** Does favorite have "Failed to score in last X" (3+ games)?\n
    • **Situation A:** Underdog positive → Bet AGAINST favorite\n
    • **Situation B:** Underdog NO positive + Favorite NO scored none → Bet ON favorite\n
    • **Situation C:** Favorite has scored none → Bet AGAINST favorite
    """)

if __name__ == "__main__":
    main()
