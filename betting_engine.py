"""
Winner Prediction System
Situations A, B, C
Accuracy: ~78-80% across 27 matches
"""

import streamlit as st

st.set_page_config(
    page_title="Winner Prediction System",
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
    .situation-tag {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 0.5rem;
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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Winner Prediction System")
    st.caption("Situations A, B, C | ~78-80% accuracy across 27 matches")
    
    st.divider()
    
    # Inputs
    favorite_team = st.text_input("Favorite team", placeholder="e.g., CSKA", key="favorite")
    underdog_team = st.text_input("Underdog team", placeholder="e.g., Levski", key="underdog")
    
    st.markdown("---")
    
    underdog_positive = st.radio(
        "Does underdog have ANY positive streak?",
        options=["Yes", "No"],
        index=None,
        horizontal=True,
        help="Unbeaten, winning, drawing, scoring, H2H unbeaten, or away positive streak"
    )
    
    favorite_scored_none = st.radio(
        "Does favorite have 'Scored none' streak?",
        options=["Yes", "No"],
        index=None,
        horizontal=True,
        help="Failed to score in last 3+ games"
    )
    
    st.divider()
    
    # Predict button
    if st.button("PREDICT", type="primary", use_container_width=True):
        # Validation
        if not favorite_team:
            st.error("Please enter Favorite team")
            return
        if not underdog_team:
            st.error("Please enter Underdog team")
            return
        if underdog_positive is None:
            st.error("Please select Yes or No for underdog positive streak")
            return
        if favorite_scored_none is None:
            st.error("Please select Yes or No for favorite 'Scored none' streak")
            return
        
        # Logic: Situation C has priority (first check)
        if favorite_scored_none == "Yes":
            situation = "C"
            prediction = "Bet AGAINST favorite"
            detail = f"Draw or {underdog_team} win"
            card_class = "bet-against"
        elif underdog_positive == "Yes":
            situation = "A"
            prediction = "Bet AGAINST favorite"
            detail = f"Draw or {underdog_team} win"
            card_class = "bet-against"
        else:
            situation = "B"
            prediction = "Bet ON favorite"
            detail = f"{favorite_team} wins"
            card_class = "bet-on"
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div class="situation-tag">Situation {situation}</div>
            <div class="prediction-text">🏆 {prediction}</div>
            <div class="prediction-detail">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **Situations:**\n
    • **Situation A (Clash):** Favorite positive + Underdog positive → Bet AGAINST favorite (Draw or underdog win)\n
    • **Situation B (Pure negative underdog):** Favorite positive + Underdog NO positive → Bet ON favorite (Favorite wins)\n
    • **Situation C:** Favorite has "Scored none" streak (3+ games) → Bet AGAINST favorite (Draw or underdog win)
    """)
    st.caption("Positive streaks: Unbeaten, Winning, Drawing, Scoring 1+, Clean sheet, H2H unbeaten, Away positive")


if __name__ == "__main__":
    main()
