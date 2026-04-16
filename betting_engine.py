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
    
    .section-header {
        font-size: 0.9rem;
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
    
    .bet-caution {
        border-left: 6px solid #f59e0b;
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
    
    # ========================================================================
    # FAVORITE STREAKS SECTION
    # ========================================================================
    
    st.markdown(f"### 📋 {favorite or 'FAVORITE'} - Check ALL streaks that appear")
    
    fav_col_left, fav_col_right = st.columns(2)
    
    with fav_col_left:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ POSITIVE STREAKS</div>', unsafe_allow_html=True)
        
        fav_unbeaten = st.checkbox("Unbeaten (last X matches)", key="fav_unbeaten")
        fav_won = st.checkbox("Won (last X matches)", key="fav_won")
        fav_scored_1 = st.checkbox("Scored 1+ (last X matches)", key="fav_scored_1")
        fav_scored_2 = st.checkbox("Scored 2+ (last X matches)", key="fav_scored_2")
        fav_clean_sheet = st.checkbox("Clean sheet (last X matches)", key="fav_clean_sheet")
        fav_undefeated_ht = st.checkbox("Undefeated at half time", key="fav_undefeated_ht")
        fav_ht_ft_wins = st.checkbox("HT/FT wins", key="fav_ht_ft_wins")
        fav_unbeaten_h2h = st.checkbox("Unbeaten H2H (last X matches)", key="fav_unbeaten_h2h")
        fav_won_h2h = st.checkbox("Won H2H (last X matches)", key="fav_won_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with fav_col_right:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ NEGATIVE STREAKS</div>', unsafe_allow_html=True)
        
        fav_winless = st.checkbox("Winless (last X matches)", key="fav_winless")
        fav_lost = st.checkbox("Lost (last X matches)", key="fav_lost")
        fav_conceded_1 = st.checkbox("Conceded 1+ (last X matches)", key="fav_conceded_1")
        fav_lost_ht = st.checkbox("Lost at half time", key="fav_lost_ht")
        fav_lost_by_2 = st.checkbox("Lost by 2+ goals", key="fav_lost_by_2")
        fav_lost_h2h = st.checkbox("Lost H2H (last X matches)", key="fav_lost_h2h")
        fav_winless_h2h = st.checkbox("Winless H2H (last X matches)", key="fav_winless_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # UNDERDOG STREAKS SECTION
    # ========================================================================
    
    st.markdown(f"### 📋 {underdog or 'UNDERDOG'} - Check ALL streaks that appear")
    
    under_col_left, under_col_right = st.columns(2)
    
    with under_col_left:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ POSITIVE STREAKS</div>', unsafe_allow_html=True)
        
        under_unbeaten = st.checkbox("Unbeaten (last X matches)", key="under_unbeaten")
        under_won = st.checkbox("Won (last X matches)", key="under_won")
        under_scored_1 = st.checkbox("Scored 1+ (last X matches)", key="under_scored_1")
        under_scored_2 = st.checkbox("Scored 2+ (last X matches)", key="under_scored_2")
        under_clean_sheet = st.checkbox("Clean sheet (last X matches)", key="under_clean_sheet")
        under_undefeated_ht = st.checkbox("Undefeated at half time", key="under_undefeated_ht")
        under_ht_ft_wins = st.checkbox("HT/FT wins", key="under_ht_ft_wins")
        under_unbeaten_h2h = st.checkbox("Unbeaten H2H (last X matches)", key="under_unbeaten_h2h")
        under_won_h2h = st.checkbox("Won H2H (last X matches)", key="under_won_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with under_col_right:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ NEGATIVE STREAKS</div>', unsafe_allow_html=True)
        
        under_winless = st.checkbox("Winless (last X matches)", key="under_winless")
        under_lost = st.checkbox("Lost (last X matches)", key="under_lost")
        under_scored_none = st.checkbox("Scored none (last X matches)", key="under_scored_none")
        under_conceded_1 = st.checkbox("Conceded 1+ (last X matches)", key="under_conceded_1")
        under_lost_ht = st.checkbox("Lost at half time", key="under_lost_ht")
        under_lost_by_2 = st.checkbox("Lost by 2+ goals", key="under_lost_by_2")
        under_lost_h2h = st.checkbox("Lost H2H (last X matches)", key="under_lost_h2h")
        under_winless_h2h = st.checkbox("Winless H2H (last X matches)", key="under_winless_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # FAVORITE SCORED NONE (SPECIAL)
    # ========================================================================
    
    st.markdown(f"### 🚫 SPECIAL - Check if appears")
    
    col_special, col_empty = st.columns([1, 1])
    with col_special:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header">Does {favorite or "FAVORITE"} have "Scored none"?</div>', unsafe_allow_html=True)
        favorite_scored_none = st.checkbox("Scored none (last X matches)", key="favorite_scored_none")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # PREDICTION BUTTON & LOGIC
    # ========================================================================
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=False):
        if not favorite or not underdog:
            st.error("❌ Please enter both team names")
        else:
            # Calculate favorite positives
            favorite_positives = sum([
                fav_unbeaten, fav_won, fav_scored_1, fav_scored_2, fav_clean_sheet,
                fav_undefeated_ht, fav_ht_ft_wins, fav_unbeaten_h2h, fav_won_h2h
            ])
            
            # Calculate favorite negatives
            favorite_negatives = sum([
                fav_winless, fav_lost, fav_conceded_1, fav_lost_ht, fav_lost_by_2,
                fav_lost_h2h, fav_winless_h2h
            ])
            
            # Calculate underdog positives
            underdog_positives = sum([
                under_unbeaten, under_won, under_scored_1, under_scored_2, under_clean_sheet,
                under_undefeated_ht, under_ht_ft_wins, under_unbeaten_h2h, under_won_h2h
            ])
            
            # Calculate underdog negatives
            underdog_negatives = sum([
                under_winless, under_lost, under_scored_none, under_conceded_1, under_lost_ht,
                under_lost_by_2, under_lost_h2h, under_winless_h2h
            ])
            
            favorite_has_positives = favorite_positives >= 1
            underdog_has_positives = underdog_positives >= 1
            
            # ================================================================
            # THREE SITUATION LOGIC (A, B, C)
            # ================================================================
            
            if favorite_scored_none:
                # Situation C: Favorite has "Scored none" streak
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Situation C: {favorite} has 'Scored none' streak → Bet AGAINST"
                
            elif underdog_has_positives:
                # Situation A: Underdog has positive streaks (Clash)
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Situation A: {underdog} has {underdog_positives} positive streak(s) → Clash → Bet AGAINST"
                
            elif favorite_has_positives:
                # Situation B: Favorite has positives, Underdog has NO positives
                prediction = "✅ BET ON FAVORITE"
                detail = f"{favorite} wins"
                card_class = "bet-on"
                reason = f"Situation B: {favorite} has {favorite_positives} positive(s), {underdog} has 0 positives → Bet ON"
                
            else:
                # No clear favorite (both weak)
                prediction = "⚠️ CAUTION"
                detail = "Consider Draw or Under 2.5 goals"
                card_class = "bet-caution"
                reason = f"No clear favorite: {favorite} has {favorite_positives} positives, {underdog} has {underdog_positives} positives"
            
            # Display prediction card
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div class="prediction-text">{prediction}</div>
                <div class="prediction-detail">→ {detail}</div>
                <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #94a3b8;">📝 {reason}</div>
                <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #64748b;">
                    📊 {favorite}: {favorite_positives} positives | {favorite_negatives} negatives<br>
                    📊 {underdog}: {underdog_positives} positives | {underdog_negatives} negatives
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **How to use:** Read the match preview. Check EVERY streak that appears for BOTH teams.
    
    **Positive streaks:** Unbeaten, Won, Scored 1+, Scored 2+, Clean sheet, Undefeated at HT, HT/FT wins, Unbeaten H2H, Won H2H
    
    **Negative streaks:** Winless, Lost, Scored none, Conceded 1+, Lost at HT, Lost by 2+, Lost H2H, Winless H2H
    
    **Logic:** Situation A (Underdog positive) → Bet AGAINST | Situation B (Only favorite positive) → Bet ON | Situation C (Favorite scored none) → Bet AGAINST
    """)


if __name__ == "__main__":
    main()
