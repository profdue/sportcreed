"""
Two-Part Rule Predictor
Logic: If underdog has positive streak → Bet AGAINST favorite (Draw or underdog win)
       Else → Bet ON favorite (Favorite wins)
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
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Two-Part Rule Predictor")
    st.caption("If underdog has positive streak → Bet AGAINST favorite | Else → Bet ON favorite")
    
    st.divider()
    
    # Inputs
    favorite_team = st.text_input("Favorite team", placeholder="e.g., Al Ahli", key="favorite")
    underdog_team = st.text_input("Underdog team", placeholder="e.g., Al-Duhail", key="underdog")
    
    st.markdown("**Does underdog have ANY positive streak?**")
    underdog_positive_streak = st.radio(
        "Select one:",
        options=["Yes", "No"],
        index=None,
        horizontal=True
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
        if underdog_positive_streak is None:
            st.error("Please select Yes or No for underdog positive streak")
            return
        
        # Two-Part Rule Logic
        if underdog_positive_streak == "Yes":
            prediction = "Bet AGAINST favorite"
            detail = f"Draw or {underdog_team} win"
            card_class = "bet-against"
        else:
            prediction = "Bet ON favorite"
            detail = f"{favorite_team} wins"
            card_class = "bet-on"
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-text">🏆 PREDICTION</div>
            <div class="prediction-detail">{prediction}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **Rule:**\n
    • If underdog has positive streak → Bet AGAINST favorite (Draw or underdog win)\n
    • If underdog has NO positive streak → Bet ON favorite (Favorite wins)
    """)


if __name__ == "__main__":
    main()
