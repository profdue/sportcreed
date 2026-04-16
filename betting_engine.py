"""
Winner Predictor - Final Version
Pick exact phrases from preview. System handles duplicates and logic.
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
        max-width: 1000px;
    }
    
    .team-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1.5rem;
    }
    
    .team-header {
        font-size: 1.25rem;
        font-weight: bold;
        color: #fbbf24;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #334155;
    }
    
    .streak-row {
        background-color: #0f172a;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
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
    
    .add-button {
        background-color: #334155 !important;
        color: white !important;
        font-size: 0.8rem !important;
        padding: 0.3rem !important;
    }
    
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PHRASE MAPPING (All phrases from your matches)
# ============================================================================

PHRASE_MAP = {
    # Unbeaten / Undefeated / No losses
    "Unbeaten in last X matches (overall)": "OVERALL_UNBEATEN",
    "Undefeated in last X matches (overall)": "OVERALL_UNBEATEN",
    "Haven't lost in last X matches (overall)": "OVERALL_UNBEATEN",
    "No losses in last X matches (overall)": "OVERALL_UNBEATEN",
    "Record of X consecutive games with no losses": "OVERALL_UNBEATEN",
    
    "Unbeaten in last X home matches": "HOME_UNBEATEN",
    "Undefeated in last X home matches": "HOME_UNBEATEN",
    "Haven't lost in last X home matches": "HOME_UNBEATEN",
    "No losses in last X home matches": "HOME_UNBEATEN",
    
    "Unbeaten in last X away matches": "AWAY_UNBEATEN",
    "Undefeated in last X away matches": "AWAY_UNBEATEN",
    "Haven't lost in last X away matches": "AWAY_UNBEATEN",
    "No losses in last X away matches": "AWAY_UNBEATEN",
    
    "Unbeaten in last X H2H matches": "H2H_UNBEATEN",
    "Undefeated in last X H2H matches": "H2H_UNBEATEN",
    "Haven't lost in last X H2H matches": "H2H_UNBEATEN",
    "No losses in last X H2H matches": "H2H_UNBEATEN",
    
    # Won / Victories
    "Won last X matches (overall)": "OVERALL_WON",
    "Victories in last X matches": "OVERALL_WON",
    "Successful run of X wins": "OVERALL_WON",
    
    "Won last X home matches": "HOME_WON",
    "Won last X away matches": "AWAY_WON",
    "Won last X H2H matches": "H2H_WON",
    "Won by 2+ goals in last X H2H": "H2H_WON_BY_2",
    "Won by 3+ goals in last X away": "AWAY_WON_BY_3",
    
    # Scored
    "Scored 1+ in last X matches": "OVERALL_SCORED_1",
    "Scored in last X matches": "OVERALL_SCORED_1",
    "Found the net in last X matches": "OVERALL_SCORED_1",
    "Scored 2+ in last X matches": "OVERALL_SCORED_2",
    
    # Clean sheet
    "Clean sheet in last X matches (overall)": "OVERALL_CLEAN_SHEET",
    "Kept a clean sheet in last X matches": "OVERALL_CLEAN_SHEET",
    "Did not concede in last X matches": "OVERALL_CLEAN_SHEET",
    "Clean sheet in last X home matches": "HOME_CLEAN_SHEET",
    "Clean sheet in last X H2H matches": "H2H_CLEAN_SHEET",
    
    # Half time
    "Undefeated at half time in last X matches": "HT_UNDEFEATED",
    "Draws at half time in last X matches": "HT_UNDEFEATED",
    "Won at half time and full time": "HT_FT_WINS",
    "Lost at half time in last X matches": "HT_LOST",
    "Trailing at half time": "HT_LOST",
    
    # Negative - Winless
    "Winless: won only X of last Y matches": "OVERALL_WINLESS",
    "Won only X of last Y matches": "OVERALL_WINLESS",
    "Just X wins in last Y matches": "OVERALL_WINLESS",
    "Poor run of only X wins": "OVERALL_WINLESS",
    "Won only X of last Y home matches": "HOME_WINLESS",
    "Won only X of last Y away matches": "AWAY_WINLESS",
    
    # Negative - Lost
    "Lost last X matches (overall)": "OVERALL_LOST",
    "Defeats in last X matches": "OVERALL_LOST",
    "Suffered X straight losses": "OVERALL_LOST",
    "Lost last X home matches": "HOME_LOST",
    "Lost last X away matches": "AWAY_LOST",
    "Lost by 2+ goals in last X away": "AWAY_LOST_BY_2",
    
    # Negative - Scored none
    "Scored none in last X matches": "OVERALL_SCORED_NONE",
    "Failed to score in last X matches": "OVERALL_SCORED_NONE",
    "No goals in last X matches": "OVERALL_SCORED_NONE",
    "Blanked in last X matches": "OVERALL_SCORED_NONE",
    "Scored none in last X home matches": "HOME_SCORED_NONE",
    "Scored none in last X away matches": "AWAY_SCORED_NONE",
    
    # Negative - Conceded
    "Conceded 1+ in last X matches": "OVERALL_CONCEDED_1",
    "No clean sheet in last X matches": "OVERALL_CONCEDED_1",
    
    # Negative - H2H
    "Lost H2H in last X matches": "H2H_LOST",
    "Winless H2H in last X matches": "H2H_WINLESS",
}

# All phrases for dropdown
ALL_PHRASES = sorted(list(PHRASE_MAP.keys()))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_streak(selected_phrases, new_phrase):
    """Add a streak, handling duplicates via mapping"""
    if not new_phrase or new_phrase == "None":
        return selected_phrases
    
    new_category = PHRASE_MAP.get(new_phrase)
    
    # Check if this category already exists
    existing_categories = [PHRASE_MAP.get(p) for p in selected_phrases if p != "None"]
    
    if new_category in existing_categories:
        # Duplicate - don't add
        return selected_phrases
    else:
        # New category - add
        return selected_phrases + [new_phrase]


def calculate_score_from_phrases(phrases):
    """Calculate score from selected phrases"""
    positives = 0
    negatives = 0
    
    for phrase in phrases:
        if phrase == "None":
            continue
        category = PHRASE_MAP.get(phrase, "")
        
        # Positive categories
        if category in ["OVERALL_UNBEATEN", "HOME_UNBEATEN", "AWAY_UNBEATEN", "H2H_UNBEATEN",
                        "OVERALL_WON", "HOME_WON", "AWAY_WON", "H2H_WON", "H2H_WON_BY_2", "AWAY_WON_BY_3",
                        "OVERALL_SCORED_1", "OVERALL_SCORED_2",
                        "OVERALL_CLEAN_SHEET", "HOME_CLEAN_SHEET", "H2H_CLEAN_SHEET",
                        "HT_UNDEFEATED", "HT_FT_WINS"]:
            positives += 1
        
        # Negative categories
        elif category in ["OVERALL_WINLESS", "HOME_WINLESS", "AWAY_WINLESS",
                          "OVERALL_LOST", "HOME_LOST", "AWAY_LOST", "AWAY_LOST_BY_2",
                          "OVERALL_SCORED_NONE", "HOME_SCORED_NONE", "AWAY_SCORED_NONE",
                          "OVERALL_CONCEDED_1", "HT_LOST", "H2H_LOST", "H2H_WINLESS"]:
            negatives += 1
    
    return positives, negatives, positives - negatives


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Winner Predictor")
    st.caption("Pick exact phrases from the preview. System handles duplicates.")
    
    st.divider()
    
    # Team names
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("🏠 HOME TEAM", placeholder="e.g., Slavia Sofia II")
    with col2:
        away_team = st.text_input("✈️ AWAY TEAM", placeholder="e.g., Rilski Sportist")
    
    st.divider()
    
    # ========================================================================
    # HOME TEAM STREAKS
    # ========================================================================
    
    st.markdown(f"### 📋 {home_team or 'HOME TEAM'}")
    
    if 'home_phrases' not in st.session_state:
        st.session_state.home_phrases = []
    
    # Display existing streaks
    for i, phrase in enumerate(st.session_state.home_phrases):
        if phrase != "None":
            st.markdown(f'<div class="streak-row">✅ {phrase}</div>', unsafe_allow_html=True)
    
    # Add new streak
    col_add, col_remove = st.columns([3, 1])
    with col_add:
        new_phrase = st.selectbox(
            "Add streak (select exact phrase from preview)",
            options=["None"] + ALL_PHRASES,
            key="home_new"
        )
    with col_remove:
        if st.button("🗑 Remove last", key="home_remove"):
            if st.session_state.home_phrases:
                st.session_state.home_phrases.pop()
            st.rerun()
    
    if st.button("➕ Add this streak", key="home_add"):
        if new_phrase and new_phrase != "None":
            st.session_state.home_phrases = add_streak(
                st.session_state.home_phrases, new_phrase
            )
            st.rerun()
    
    st.divider()
    
    # ========================================================================
    # AWAY TEAM STREAKS
    # ========================================================================
    
    st.markdown(f"### 📋 {away_team or 'AWAY TEAM'}")
    
    if 'away_phrases' not in st.session_state:
        st.session_state.away_phrases = []
    
    # Display existing streaks
    for i, phrase in enumerate(st.session_state.away_phrases):
        if phrase != "None":
            st.markdown(f'<div class="streak-row">✅ {phrase}</div>', unsafe_allow_html=True)
    
    # Add new streak
    col_add, col_remove = st.columns([3, 1])
    with col_add:
        new_phrase = st.selectbox(
            "Add streak (select exact phrase from preview)",
            options=["None"] + ALL_PHRASES,
            key="away_new"
        )
    with col_remove:
        if st.button("🗑 Remove last", key="away_remove"):
            if st.session_state.away_phrases:
                st.session_state.away_phrases.pop()
            st.rerun()
    
    if st.button("➕ Add this streak", key="away_add"):
        if new_phrase and new_phrase != "None":
            st.session_state.away_phrases = add_streak(
                st.session_state.away_phrases, new_phrase
            )
            st.rerun()
    
    st.divider()
    
    # ========================================================================
    # PREDICT
    # ========================================================================
    
    if st.button("🔮 PREDICT", type="primary"):
        if not home_team or not away_team:
            st.error("❌ Please enter both team names")
        else:
            # Calculate scores
            home_pos, home_neg, home_score = calculate_score_from_phrases(st.session_state.home_phrases)
            away_pos, away_neg, away_score = calculate_score_from_phrases(st.session_state.away_phrases)
            
            # Determine favorite
            if home_score > away_score:
                favorite, underdog = home_team, away_team
                fav_score, under_score = home_score, away_score
                underdog_positives = away_pos
                favorite_scored_none = any("Scored none" in p for p in st.session_state.away_phrases)
            elif away_score > home_score:
                favorite, underdog = away_team, home_team
                fav_score, under_score = away_score, home_score
                underdog_positives = home_pos
                favorite_scored_none = any("Scored none" in p for p in st.session_state.home_phrases)
            else:
                favorite, underdog = None, None
                fav_score = under_score = home_score
                underdog_positives = 0
                favorite_scored_none = False
            
            # Three situation logic
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
                reason = f"Situation A: {underdog} has {underdog_positives} positive(s) → Clash → Bet AGAINST"
            elif fav_score > under_score:
                prediction = "✅ BET ON FAVORITE"
                detail = f"{favorite} wins"
                card_class = "bet-on"
                reason = f"Situation B: {favorite} has higher score ({fav_score}) → Bet ON"
            else:
                prediction = "⚠️ CAUTION"
                detail = "Consider Draw or Under 2.5"
                card_class = "bet-caution"
                reason = "No clear pattern"
            
            # Display
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div class="prediction-text">{prediction}</div>
                <div class="prediction-detail">→ {detail}</div>
                <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #94a3b8;">📝 {reason}</div>
                <div style="margin-top: 0.5rem; font-size: 0.7rem; color: #64748b;">
                    📊 {home_team}: {home_pos} positives | {home_neg} negatives = Score {home_score}<br>
                    📊 {away_team}: {away_pos} positives | {away_neg} negatives = Score {away_score}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("""
    **How to use:**
    1. Read the match preview
    2. For each streak you see, select the exact phrase from the dropdown
    3. Click "Add this streak"
    4. System automatically ignores duplicates (same meaning)
    5. Repeat for all streaks in the preview
    6. Click PREDICT
    
    **You don't need to think about what means what. Just pick what you see.**
    """)


if __name__ == "__main__":
    main()