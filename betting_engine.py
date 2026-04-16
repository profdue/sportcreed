"""
Winner Predictor - Checkbox Version
Based on actual terms from match previews
App determines favorite/underdog from streaks
"""

import streamlit as st

st.set_page_config(
    page_title="Winner Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
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
    
    hr {
        margin: 1rem 0;
    }
    
    .stCheckbox label {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_score(positives, negatives):
    """Calculate team score = positives - negatives"""
    return positives - negatives


def determine_favorite(home_score, away_score, home_team, away_team):
    """Determine favorite based on higher score"""
    if home_score > away_score:
        return home_team, away_team, home_score, away_score
    elif away_score > home_score:
        return away_team, home_team, away_score, home_score
    else:
        return None, None, home_score, away_score


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Winner Predictor")
    st.caption("Check streaks for both teams. App determines favorite and predicts.")
    
    st.divider()
    
    # Team name inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="team-card">
            <div class="team-header">🏠 HOME TEAM</div>
        </div>
        """, unsafe_allow_html=True)
        home_team = st.text_input("Home team name", placeholder="e.g., Mbarara City", key="home", label_visibility="collapsed")
    
    with col2:
        st.markdown("""
        <div class="team-card">
            <div class="team-header">✈️ AWAY TEAM</div>
        </div>
        """, unsafe_allow_html=True)
        away_team = st.text_input("Away team name", placeholder="e.g., Police FC", key="away", label_visibility="collapsed")
    
    st.divider()
    
    # ========================================================================
    # HOME TEAM STREAKS
    # ========================================================================
    
    st.markdown(f"### 📋 {home_team or 'HOME TEAM'} - Check ALL streaks that appear")
    
    home_col_left, home_col_right = st.columns(2)
    
    with home_col_left:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ POSITIVE STREAKS</div>', unsafe_allow_html=True)
        
        home_unbeaten = st.checkbox("Overall: Unbeaten / Undefeated (last X matches)", key="home_unbeaten")
        home_won = st.checkbox("Overall: Won (last X matches)", key="home_won")
        home_scored_1 = st.checkbox("Overall: Scored 1+ / Scored in last X", key="home_scored_1")
        home_scored_2 = st.checkbox("Overall: Scored 2+ (last X matches)", key="home_scored_2")
        home_clean_sheet = st.checkbox("Overall: Clean sheet / Did not concede", key="home_clean_sheet")
        home_undefeated_ht = st.checkbox("Overall: Undefeated at HT / Draws at HT", key="home_undefeated_ht")
        home_ht_ft_wins = st.checkbox("Overall: HT/FT wins", key="home_ht_ft_wins")
        home_unbeaten_h2h = st.checkbox("H2H only: Unbeaten / Undefeated vs this opponent", key="home_unbeaten_h2h")
        home_won_h2h = st.checkbox("H2H only: Won H2H (last X matches)", key="home_won_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with home_col_right:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ NEGATIVE STREAKS</div>', unsafe_allow_html=True)
        
        home_winless = st.checkbox("Overall: Winless / Won only X of last Y", key="home_winless")
        home_lost = st.checkbox("Overall: Lost (last X matches)", key="home_lost")
        home_scored_none = st.checkbox("Overall: Scored none / Failed to score", key="home_scored_none")
        home_conceded_1 = st.checkbox("Overall: Conceded 1+ / No clean sheet", key="home_conceded_1")
        home_lost_ht = st.checkbox("Overall: Lost at half time", key="home_lost_ht")
        home_lost_by_2 = st.checkbox("Overall: Lost by 2+ goals", key="home_lost_by_2")
        home_lost_h2h = st.checkbox("H2H only: Lost H2H (last X matches)", key="home_lost_h2h")
        home_winless_h2h = st.checkbox("H2H only: Winless H2H (last X matches)", key="home_winless_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # AWAY TEAM STREAKS
    # ========================================================================
    
    st.markdown(f"### 📋 {away_team or 'AWAY TEAM'} - Check ALL streaks that appear")
    
    away_col_left, away_col_right = st.columns(2)
    
    with away_col_left:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">✅ POSITIVE STREAKS</div>', unsafe_allow_html=True)
        
        away_unbeaten = st.checkbox("Overall: Unbeaten / Undefeated (last X matches)", key="away_unbeaten")
        away_won = st.checkbox("Overall: Won (last X matches)", key="away_won")
        away_scored_1 = st.checkbox("Overall: Scored 1+ / Scored in last X", key="away_scored_1")
        away_scored_2 = st.checkbox("Overall: Scored 2+ (last X matches)", key="away_scored_2")
        away_clean_sheet = st.checkbox("Overall: Clean sheet / Did not concede", key="away_clean_sheet")
        away_undefeated_ht = st.checkbox("Overall: Undefeated at HT / Draws at HT", key="away_undefeated_ht")
        away_ht_ft_wins = st.checkbox("Overall: HT/FT wins", key="away_ht_ft_wins")
        away_unbeaten_h2h = st.checkbox("H2H only: Unbeaten / Undefeated vs this opponent", key="away_unbeaten_h2h")
        away_won_h2h = st.checkbox("H2H only: Won H2H (last X matches)", key="away_won_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with away_col_right:
        st.markdown('<div class="team-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ NEGATIVE STREAKS</div>', unsafe_allow_html=True)
        
        away_winless = st.checkbox("Overall: Winless / Won only X of last Y", key="away_winless")
        away_lost = st.checkbox("Overall: Lost (last X matches)", key="away_lost")
        away_scored_none = st.checkbox("Overall: Scored none / Failed to score", key="away_scored_none")
        away_conceded_1 = st.checkbox("Overall: Conceded 1+ / No clean sheet", key="away_conceded_1")
        away_lost_ht = st.checkbox("Overall: Lost at half time", key="away_lost_ht")
        away_lost_by_2 = st.checkbox("Overall: Lost by 2+ goals", key="away_lost_by_2")
        away_lost_h2h = st.checkbox("H2H only: Lost H2H (last X matches)", key="away_lost_h2h")
        away_winless_h2h = st.checkbox("H2H only: Winless H2H (last X matches)", key="away_winless_h2h")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ========================================================================
    # PREDICTION BUTTON & LOGIC
    # ========================================================================
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=False):
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
        else:
            # Calculate home team score
            home_positives = sum([
                home_unbeaten, home_won, home_scored_1, home_scored_2, home_clean_sheet,
                home_undefeated_ht, home_ht_ft_wins, home_unbeaten_h2h, home_won_h2h
            ])
            
            home_negatives = sum([
                home_winless, home_lost, home_scored_none, home_conceded_1, home_lost_ht,
                home_lost_by_2, home_lost_h2h, home_winless_h2h
            ])
            
            home_score = home_positives - home_negatives
            
            # Calculate away team score
            away_positives = sum([
                away_unbeaten, away_won, away_scored_1, away_scored_2, away_clean_sheet,
                away_undefeated_ht, away_ht_ft_wins, away_unbeaten_h2h, away_won_h2h
            ])
            
            away_negatives = sum([
                away_winless, away_lost, away_scored_none, away_conceded_1, away_lost_ht,
                away_lost_by_2, away_lost_h2h, away_winless_h2h
            ])
            
            away_score = away_positives - away_negatives
            
            # Determine favorite and underdog
            favorite, underdog, fav_score, under_score = determine_favorite(
                home_score, away_score, home_team, away_team
            )
            
            # Check for special "Scored none" condition on the favorite
            if favorite == home_team:
                favorite_scored_none = home_scored_none
                favorite_positives = home_positives
                underdog_positives = away_positives
            elif favorite == away_team:
                favorite_scored_none = away_scored_none
                favorite_positives = away_positives
                underdog_positives = home_positives
            else:
                favorite_scored_none = False
                favorite_positives = 0
                underdog_positives = 0
            
            # THREE SITUATION LOGIC (A, B, C)
            if favorite is None:
                prediction = "⚠️ CAUTION"
                detail = "No clear favorite (scores are equal)"
                card_class = "bet-caution"
                reason = f"Scores are equal: {home_team} {home_score} vs {away_team} {away_score}"
                
            elif favorite_scored_none:
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Situation C: {favorite} has 'Scored none' streak → Bet AGAINST"
                
            elif underdog_positives >= 1:
                prediction = "⚠️ BET AGAINST FAVORITE"
                detail = f"Draw or {underdog} win"
                card_class = "bet-against"
                reason = f"Situation A: {underdog} has {underdog_positives} positive streak(s) → Clash → Bet AGAINST"
                
            elif favorite_positives >= 1:
                prediction = "✅ BET ON FAVORITE"
                detail = f"{favorite} wins"
                card_class = "bet-on"
                reason = f"Situation B: {favorite} has {favorite_positives} positive(s), {underdog} has 0 positives → Bet ON"
                
            else:
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
                    📊 {home_team}: {home_positives} positives | {home_negatives} negatives = Score {home_score}<br>
                    📊 {away_team}: {away_positives} positives | {away_negatives} negatives = Score {away_score}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **How to use:** Read the match preview. Check EVERY streak that appears for BOTH teams.
    
    **"Overall"** = form vs all opponents (e.g., "Unbeaten in last 10 matches")
    **"H2H only"** = form vs this specific opponent (e.g., "Unbeaten in last 6 vs Team X")
    
    **How it works:** App calculates score (positives - negatives) for each team. Higher score = Favorite.
    Then applies 3-situation logic to predict.
    """)


if __name__ == "__main__":
    main()