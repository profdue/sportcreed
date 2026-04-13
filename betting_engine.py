"""
Streak Counter Predictor
Score = Positives - Negatives
Higher score = Bet ON favorite | Lower score = Bet AGAINST favorite
"""

import streamlit as st

st.set_page_config(
    page_title="Streak Counter Predictor",
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
    .score-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #334155;
    }
    .score-positive {
        color: #10b981;
        font-size: 2rem;
        font-weight: bold;
    }
    .score-negative {
        color: #ef4444;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 1px solid #334155;
        text-align: center;
    }
    .streak-list {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STREAK COUNTER LOGIC
# ============================================================================

def calculate_score(positives: int, negatives: int) -> int:
    """Score = Positives - Negatives"""
    return positives - negatives


def predict(home_name: str, away_name: str, home_positives: int, home_negatives: int, away_positives: int, away_negatives: int) -> dict:
    """Higher score = Favorite = Bet ON"""
    
    home_score = calculate_score(home_positives, home_negatives)
    away_score = calculate_score(away_positives, away_negatives)
    
    if home_score > away_score:
        favorite = home_name
        favorite_score = home_score
        underdog = away_name
        underdog_score = away_score
        bet = f"Bet ON favorite → {favorite} wins"
    elif away_score > home_score:
        favorite = away_name
        favorite_score = away_score
        underdog = home_name
        underdog_score = home_score
        bet = f"Bet ON favorite → {favorite} wins"
    else:
        # Scores are equal
        favorite = "Draw or underdog"
        favorite_score = home_score
        underdog = ""
        underdog_score = away_score
        bet = "Scores equal → Consider Draw or underdog"
    
    return {
        "home_name": home_name,
        "away_name": away_name,
        "home_score": home_score,
        "away_score": away_score,
        "favorite": favorite,
        "favorite_score": favorite_score,
        "underdog": underdog,
        "underdog_score": underdog_score,
        "bet": bet,
        "home_is_favorite": home_score > away_score,
        "away_is_favorite": away_score > home_score,
        "scores_equal": home_score == away_score
    }


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Streak Counter Predictor")
    st.caption("Score = Positives - Negatives | Higher score = Bet ON favorite")
    
    st.divider()
    
    # Team inputs
    col1, col2 = st.columns(2)
    with col1:
        home_name = st.text_input("Home team", "FK Tukums 2000", key="home_name")
    with col2:
        away_name = st.text_input("Away team", "Rigas Futbola skola", key="away_name")
    
    st.divider()
    
    # Home team streaks
    st.markdown(f"### 🏠 {home_name}")
    col1, col2 = st.columns(2)
    with col1:
        home_positives = st.number_input("Positive streaks (+1 each)", min_value=0, max_value=50, value=0, key="home_positives")
    with col2:
        home_negatives = st.number_input("Negative streaks (-1 each)", min_value=0, max_value=50, value=2, key="home_negatives")
    
    st.divider()
    
    # Away team streaks
    st.markdown(f"### ✈️ {away_name}")
    col1, col2 = st.columns(2)
    with col1:
        away_positives = st.number_input("Positive streaks (+1 each)", min_value=0, max_value=50, value=9, key="away_positives")
    with col2:
        away_negatives = st.number_input("Negative streaks (-1 each)", min_value=0, max_value=50, value=0, key="away_negatives")
    
    st.divider()
    
    # Predict button
    if st.button("PREDICT", type="primary", use_container_width=True):
        result = predict(
            home_name, away_name,
            home_positives, home_negatives,
            away_positives, away_negatives
        )
        
        # Display scores
        col1, col2 = st.columns(2)
        with col1:
            score_class = "score-positive" if result['home_score'] >= 0 else "score-negative"
            st.markdown(f"""
            <div class="score-card">
                <strong>{result['home_name']}</strong><br>
                Positives: +{home_positives}<br>
                Negatives: -{home_negatives}<br>
                <span class="{score_class}">Score: {result['home_score']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            score_class = "score-positive" if result['away_score'] >= 0 else "score-negative"
            st.markdown(f"""
            <div class="score-card">
                <strong>{result['away_name']}</strong><br>
                Positives: +{away_positives}<br>
                Negatives: -{away_negatives}<br>
                <span class="{score_class}">Score: {result['away_score']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-card">
            <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">🏆 PREDICTION</div>
            <div style="font-size: 1rem; color: #fbbf24;">{result['bet']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show reasoning (optional)
        with st.expander("Show reasoning"):
            st.markdown(f"**Formula:** Score = Positives - Negatives")
            st.markdown(f"**{result['home_name']}:** {home_positives} positives - {home_negatives} negatives = **{result['home_score']}**")
            st.markdown(f"**{result['away_name']}:** {away_positives} positives - {away_negatives} negatives = **{result['away_score']}**")
            if result['home_score'] > result['away_score']:
                st.markdown(f"**Higher score = {result['home_name']}** → Bet ON {result['home_name']}")
            elif result['away_score'] > result['home_score']:
                st.markdown(f"**Higher score = {result['away_name']}** → Bet ON {result['away_name']}")
            else:
                st.markdown(f"**Scores equal** → Consider Draw or underdog")
    
    st.divider()
    st.caption("""
    **Positive streaks (+1 each):** Unbeaten, Won, Drawn, Scored, Clean sheet, Unbeaten H2H, Won H2H\n
    **Negative streaks (-1 each):** Lost, Scored none, Conceded, Lost H2H, Winless\n
    **Formula:** Score = Positives - Negatives | Higher score = Favorite = Bet ON
    """)


if __name__ == "__main__":
    main()
