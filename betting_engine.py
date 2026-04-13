"""
Three-Situation Winner Predictor
Situation A (Clash): Favorite positive + Underdog positive → Bet AGAINST favorite
Situation B (Pure negative): Favorite positive + Underdog NO positive → Bet ON favorite
Situation C (Scored none): Favorite scored none streak (3+ games) → Bet AGAINST favorite
"""

import streamlit as st

st.set_page_config(
    page_title="Three-Situation Winner Predictor",
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
        font-size: 0.7rem;
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
# PREDICTION LOGIC
# ============================================================================

def predict(favorite_name: str, underdog_name: str, favorite_scored_none_streak: bool, underdog_positive_streak: bool) -> dict:
    """
    Three-Situation Rule:
    Priority 1: Favorite has scored none streak (3+ games) → Situation C
    Priority 2: Underdog has positive streak → Situation A (Clash)
    Priority 3: Else → Situation B (Pure negative underdog)
    """
    
    # Situation C: Favorite has scored none streak (3+ games)
    if favorite_scored_none_streak:
        return {
            "situation": "C",
            "situation_name": "Favorite scored none",
            "prediction": "Bet AGAINST favorite",
            "detail": f"Draw or {underdog_name} win",
            "card_class": "bet-against"
        }
    
    # Situation A: Clash (underdog has positive streak)
    if underdog_positive_streak:
        return {
            "situation": "A",
            "situation_name": "Clash",
            "prediction": "Bet AGAINST favorite",
            "detail": f"Draw or {underdog_name} win",
            "card_class": "bet-against"
        }
    
    # Situation B: Pure negative underdog
    return {
        "situation": "B",
        "situation_name": "Pure negative underdog",
        "prediction": "Bet ON favorite",
        "detail": f"{favorite_name} wins",
        "card_class": "bet-on"
    }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Three-Situation Winner Predictor")
    st.caption("Situation A (Clash) | Situation B (Pure negative) | Situation C (Scored none)")
    
    st.divider()
    
    # Team inputs
    favorite_team = st.text_input("Favorite team", placeholder="e.g., CSKA", key="favorite")
    underdog_team = st.text_input("Underdog team", placeholder="e.g., Levski", key="underdog")
    
    st.divider()
    
    # Situation C: Favorite scored none streak
    st.markdown("**Situation C: Favorite has 'Scored none' streak?**")
    st.caption("(Failed to score in last 3+ games)")
    favorite_scored_none = st.radio(
        "Select one:",
        options=["Yes", "No"],
        index=1,
        horizontal=True,
        key="scored_none"
    )
    
    st.divider()
    
    # Situation A: Underdog positive streak
    st.markdown("**Situation A: Underdog has ANY positive streak?**")
    st.caption("(Unbeaten, winning, drawing, scoring, clean sheet, H2H unbeaten, etc.)")
    underdog_positive = st.radio(
        "Select one:",
        options=["Yes", "No"],
        index=1,
        horizontal=True,
        key="underdog_positive"
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
        
        # Convert radio to boolean
        favorite_scored_none_bool = (favorite_scored_none == "Yes")
        underdog_positive_bool = (underdog_positive == "Yes")
        
        # Get prediction
        result = predict(
            favorite_team, underdog_team,
            favorite_scored_none_bool, underdog_positive_bool
        )
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card {result['card_class']}">
            <div class="situation-tag">SITUATION {result['situation']}: {result['situation_name']}</div>
            <div class="prediction-text">🏆 {result['prediction']}</div>
            <div class="prediction-detail">→ {result['detail']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **Rules (Priority Order):**\n
    • **Situation C:** Favorite has "Scored none" streak (3+ games) → Bet AGAINST favorite\n
    • **Situation A (Clash):** Underdog has positive streak → Bet AGAINST favorite\n
    • **Situation B (Pure negative):** Else → Bet ON favorite
    """)
    st.caption("⚽ Based on 27 matches at ~78-80% accuracy")


if __name__ == "__main__":
    main()
