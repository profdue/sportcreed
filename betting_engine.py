"""
Winner Predictor - Checkbox Version
Based on actual terms from match previews
Classic UI Design with Team Cards
"""

import streamlit as st

st.set_page_config(
    page_title="Winner Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - CLASSIC DESIGN
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Team Card */
    .team-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #334155;
        height: 100%;
    }
    
    .team-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #fbbf24;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    
    /* Stat Grid */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    
    .stat-item {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.4rem 0.5rem;
        text-align: center;
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8;
    }
    
    .stat-value {
        font-size: 0.85rem;
        font-weight: bold;
        color: #fbbf24;
    }
    
    /* Section Header */
    .section-header {
        font-size: 1rem;
        font-weight: bold;
        color: #fbbf24;
        margin: 0.5rem 0;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        text-align: center;
        border: 1px solid #334155;
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
    }
    
    .prediction-detail {
        font-size: 1rem;
        color: #fbbf24;
        margin-top: 0.5rem;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #0f172a;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        border: none;
        border-radius: 12px;
        padding: 0.75rem;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #0f172a;
    }
    
    /* Checkbox styling */
    .stCheckbox label {
        font-size: 0.85rem;
    }
    
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Winner Predictor")
    st.caption("Check the streaks that appear in the match preview")
    
    st.divider()
    
    # Team name inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <div class="team-header">🏆 FAVORITE</div>
        </div>
        """, unsafe_allow_html=True)
        favorite = st.text_input("Team name", placeholder="e.g., CSKA Sofia", key="favorite", label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <div class="team-header">🐕 UNDERDOG</div>
        </div>
        """, unsafe_allow_html=True)
        underdog = st.text_input("Team name", placeholder="e.g., Levski Sofia", key="underdog", label_visibility="collapsed")
    
    st.divider()
    
    # Underdog streaks section
    st.markdown(f"### 📋 {underdog or 'UNDERDOG'} - Check ALL streaks that appear")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ POSITIVE STREAKS (+1 each)</div>', unsafe_allow_html=True)
        
        unbeaten = st.checkbox("Unbeaten (last X matches)", key="unbeaten")
        won = st.checkbox("Won (last X matches)", key="won")
        scored_1 = st.checkbox("Scored 1+ (last X matches)", key="scored_1")
        scored_2 = st.checkbox("Scored 2+ (last X matches)", key="scored_2")
        clean_sheet = st.checkbox("Clean sheet (last X matches)", key="clean_sheet")
        undefeated_ht = st.checkbox("Undefeated at half time", key="undefeated_ht")
        ht_ft_wins = st.checkbox("HT/FT wins", key="ht_ft_wins")
        unbeaten_h2h = st.checkbox("Unbeaten H2H (last X matches)", key="unbeaten_h2h")
        won_h2h = st.checkbox("Won H2H (last X matches)", key="won_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ NEGATIVE STREAKS (-1 each)</div>', unsafe_allow_html=True)
        
        winless = st.checkbox("Winless (last X matches)", key="winless")
        lost = st.checkbox("Lost (last X matches)", key="lost")
        scored_none = st.checkbox("Scored none (last X matches)", key="scored_none_underdog")
        conceded_1 = st.checkbox("Conceded 1+ (last X matches)", key="conceded_1")
        lost_ht = st.checkbox("Lost at half time", key="lost_ht")
        lost_by_2 = st.checkbox("Lost by 2+ goals", key="lost_by_2")
        lost_h2h = st.checkbox("Lost H2H (last X matches)", key="lost_h2h")
        winless_h2h = st.checkbox("Winless H2H (last X matches)", key="winless_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Favorite scored none
    st.markdown(f"### 🚫 {favorite or 'FAVORITE'} - Check if appears")
    
    col_center, col_empty = st.columns([1, 1])
    with col_center:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        favorite_scored_none = st.checkbox("Scored none (last X matches)", key="favorite_scored_none")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Predict button
    if st.button("🔮 PREDICT", type="primary", use_container_width=False):
        if not favorite or not underdog:
            st.error("❌ Please enter both team names")
        else:
            # Count underdog positives
            underdog_positives = sum([
                unbeaten, won, scored_1, scored_2, clean_sheet,
                undefeated_ht, ht_ft_wins, unbeaten_h2h, won_h2h
            ])
            
            # Count underdog negatives
            underdog_negatives = sum([
                winless, lost, scored_none, conceded_1, lost_ht,
                lost_by_2, lost_h2h, winless_h2h
            ])
            
            underdog_score = underdog_positives - underdog_negatives
            
            # Prediction logic
            if favorite_scored_none:
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Favorite has 'Scored none' streak → Situation C"
            elif underdog_positives >= 1:
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Underdog has {underdog_positives} positive streak(s) → Situation A"
            elif favorite_positives >= 4:
                prediction = "✅ BET ON FAVORITE"
                detail = f"{favorite} wins"
                card_class = "bet-on"
                reason = f"Favorite has {favorite_positives} positives, underdog has 0 → Situation B"
            else:
                prediction = "⚠️ CAUTION"
                detail = "Consider Draw or Under 2.5"
                card_class = "bet-against"
                reason = f"Low confidence: Favorite positives {favorite_positives}, Underdog positives {underdog_positives}"
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div class="prediction-text">{prediction}</div>
                <div class="prediction-detail">→ {detail}</div>
                <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #94a3b8;">📝 {reason}</div>
                <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #64748b;">
                    Underdog: {underdog_positives} positives | {underdog_negatives} negatives = Score {underdog_score}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **How to use:** Read the match preview. Check EVERY streak that appears for the underdog.
    Then check if favorite has 'Scored none' streak. Click PREDICT.
    
    **Positive streaks:** Unbeaten, Won, Scored 1+, Scored 2+, Clean sheet, Undefeated at HT, HT/FT wins, Unbeaten H2H, Won H2H
    **Negative streaks:** Winless, Lost, Scored none, Conceded 1+, Lost at HT, Lost by 2+, Lost H2H, Winless H2H
    """)

if __name__ == "__main__":
    main()
