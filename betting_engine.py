"""
Two-Part Rule Predictor - Unified Scoring System
Logic: 
1. Calculate score = (Positive streaks) - (Negative streaks) for each team
2. Higher score = Favorite
3. If underdog has ANY positive streak (score >= 0) → Bet AGAINST favorite (Draw or underdog win)
4. If underdog has NO positive streak (score < 0) → Bet ON favorite (Favorite wins)
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
        max-width: 700px;
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
    .score-card {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .score-positive {
        color: #10b981;
        font-weight: bold;
    }
    .score-negative {
        color: #ef4444;
        font-weight: bold;
    }
    .score-neutral {
        color: #fbbf24;
        font-weight: bold;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        font-weight: bold;
    }
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_score(positives, negatives):
    """Calculate score = positives - negatives"""
    return positives - negatives

def determine_favorite(home_score, away_score, home_team, away_team):
    """Determine which team is favorite based on higher score"""
    if home_score > away_score:
        return home_team, away_team, home_score, away_score
    elif away_score > home_score:
        return away_team, home_team, away_score, home_score
    else:
        return None, None, home_score, away_score

def get_score_style(score):
    """Return CSS style class based on score"""
    if score > 0:
        return "score-positive"
    elif score < 0:
        return "score-negative"
    else:
        return "score-neutral"

def get_score_symbol(score):
    """Return symbol for score display"""
    if score > 0:
        return f"+{score}"
    elif score < 0:
        return f"{score}"
    else:
        return "0"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Two-Part Rule Predictor")
    st.caption("Unified Scoring System: Higher score = Favorite")
    
    st.divider()
    
    # Team inputs
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.text_input("🏠 Home team", placeholder="e.g., Sirius IK", key="home")
    
    with col2:
        away_team = st.text_input("✈️ Away team", placeholder="e.g., Hammarby IF", key="away")
    
    st.markdown("---")
    st.subheader("📊 Streak Counter")
    st.caption("Count positive streaks (+1 each) and negative streaks (-1 each)")
    
    # Home team streaks
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🏠 {home_team or 'Home team'}**")
        home_positives = st.number_input(
            "Positive streaks",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="home_pos"
        )
        home_negatives = st.number_input(
            "Negative streaks",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="home_neg"
        )
    
    with col2:
        st.markdown(f"**✈️ {away_team or 'Away team'}**")
        away_positives = st.number_input(
            "Positive streaks",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="away_pos"
        )
        away_negatives = st.number_input(
            "Negative streaks",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="away_neg"
        )
    
    # Calculate scores
    home_score = calculate_score(home_positives, home_negatives)
    away_score = calculate_score(away_positives, away_negatives)
    
    # Display scores
    col1, col2 = st.columns(2)
    
    with col1:
        home_style = get_score_style(home_score)
        home_symbol = get_score_symbol(home_score)
        st.markdown(f"""
        <div class="score-card">
            <b>{home_team or 'Home'}</b><br>
            Score: <span class="{home_style}">{home_symbol}</span><br>
            <small>({home_positives} positives - {home_negatives} negatives)</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        away_style = get_score_style(away_score)
        away_symbol = get_score_symbol(away_score)
        st.markdown(f"""
        <div class="score-card">
            <b>{away_team or 'Away'}</b><br>
            Score: <span class="{away_style}">{away_symbol}</span><br>
            <small>({away_positives} positives - {away_negatives} negatives)</small>
        </div>
        """, unsafe_allow_html=True)
    
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
        
        # Determine favorite
        favorite, underdog, fav_score, under_score = determine_favorite(
            home_score, away_score, home_team, away_team
        )
        
        # Handle equal scores
        if favorite is None:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-text">⚖️ NO CLEAR FAVORITE</div>
                <div class="prediction-detail">Scores are equal ({home_score} vs {away_score})</div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">
                → Pass or bet Draw
                </div>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Determine if underdog has any positive streak (score >= 0)
        underdog_has_positive = (under_score >= 0)
        
        # Apply two-part rule
        if underdog_has_positive:
            prediction = "Bet AGAINST favorite"
            detail = f"Draw or {underdog} win"
            card_class = "bet-against"
            reasoning = f"""
            • Favorite: {favorite} (score {fav_score})
            • Underdog: {underdog} (score {under_score})
            • Underdog has positive streak? YES (score >= 0)
            • Situation A → Bet AGAINST favorite
            """
        else:
            prediction = "Bet ON favorite"
            detail = f"{favorite} wins"
            card_class = "bet-on"
            reasoning = f"""
            • Favorite: {favorite} (score {fav_score})
            • Underdog: {underdog} (score {under_score})
            • Underdog has positive streak? NO (score < 0)
            • Situation B → Bet ON favorite
            """
        
        # Display result
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-text">🏆 PREDICTION</div>
            <div class="prediction-detail">{prediction}</div>
            <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #94a3b8;">→ {detail}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning (optional expander)
        with st.expander("📋 Show reasoning"):
            st.markdown(reasoning)
            st.markdown(f"""
            **Scores:**  
            {home_team}: {home_score} ({home_positives}p - {home_negatives}n)  
            {away_team}: {away_score} ({away_positives}p - {away_negatives}n)
            """)
    
    st.divider()
    st.caption("""
    **How to count streaks:**\n
    ✅ **Positive streaks (+1 each):** Unbeaten in last X, Won last X, Drawn last X, Scored in last X, Clean sheet in last X, Unbeaten H2H, Won H2H\n
    ❌ **Negative streaks (-1 each):** Lost last X, Scored none in last X, Conceded in last X, Lost H2H, Winless in last X\n
    **Formula:** Score = Positives - Negatives | Higher score = Favorite
    """)


if __name__ == "__main__":
    main()
