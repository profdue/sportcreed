"""
Winner Predictor - Checkbox Version
Based on actual terms from match previews
"""

import streamlit as st

st.set_page_config(
    page_title="Winner Predictor",
    page_icon="⚽",
    layout="centered"
)

st.markdown("""
<style>
    .main .block-container { max-width: 600px; padding-top: 2rem; }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid #334155;
    }
    .bet-against { border-left: 6px solid #ef4444; }
    .bet-on { border-left: 6px solid #10b981; }
    .prediction-text { font-size: 1.3rem; font-weight: bold; }
    .prediction-detail { font-size: 1rem; color: #fbbf24; margin-top: 0.5rem; }
    .stButton button { background-color: #3b82f6; color: white; font-weight: bold; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ Winner Predictor")
st.caption("Check the streaks that appear in the match preview")

# Team names
col1, col2 = st.columns(2)
with col1:
    favorite = st.text_input("🏆 Favorite team", placeholder="e.g., CSKA Sofia")
with col2:
    underdog = st.text_input("🐕 Underdog team", placeholder="e.g., Levski Sofia")

st.divider()

# Underdog streaks
st.subheader(f"📋 {underdog or 'Underdog'} - Check ALL that appear")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**POSITIVE STREAKS**")
    unbeaten = st.checkbox("Unbeaten (last X matches)")
    won = st.checkbox("Won (last X matches)")
    scored_1 = st.checkbox("Scored 1+ (last X matches)")
    scored_2 = st.checkbox("Scored 2+ (last X matches)")
    clean_sheet = st.checkbox("Clean sheet (last X matches)")
    undefeated_ht = st.checkbox("Undefeated at half time")
    ht_ft_wins = st.checkbox("HT/FT wins")

with col2:
    st.markdown("**NEGATIVE STREAKS**")
    winless = st.checkbox("Winless (last X matches)")
    lost = st.checkbox("Lost (last X matches)")
    scored_none = st.checkbox("Scored none (last X matches)")
    lost_ht = st.checkbox("Lost at half time")
    lost_by_2 = st.checkbox("Lost by 2+ goals")

st.divider()

# Favorite scored none
st.subheader(f"🚫 {favorite or 'Favorite'} - Check if appears")
favorite_scored_none = st.checkbox("Scored none (last X matches)")

st.divider()

# Predict button
if st.button("🔮 PREDICT"):
    if not favorite or not underdog:
        st.error("Please enter both team names")
    else:
        # Determine if underdog has any positive streak
        underdog_has_positive = any([
            unbeaten, won, scored_1, scored_2, 
            clean_sheet, undefeated_ht, ht_ft_wins
        ])
        
        # Logic
        if favorite_scored_none:
            prediction = "⚠️ BET AGAINST FAVORITE"
            detail = f"Draw or {underdog} win"
            card_class = "bet-against"
        elif underdog_has_positive:
            prediction = "⚠️ BET AGAINST FAVORITE"
            detail = f"Draw or {underdog} win"
            card_class = "bet-against"
        else:
            prediction = "✅ BET ON FAVORITE"
            detail = f"{favorite} wins"
            card_class = "bet-on"
        
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-text">{prediction}</div>
            <div class="prediction-detail">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption("""
**How to use:** Read the match preview. Check EVERY streak that appears for the underdog.
Then check if favorite has 'Scored none'. Click PREDICT.
""")
