import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Football xG Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("⚽ Football Match Predictor")
st.markdown("""
    Predict match outcomes using Expected Goals (xG) regression analysis.
    This model adjusts for team over/underperformance and calculates probabilities using Poisson distribution.
""")

# Constants
MAX_GOALS = 8
REG_BASE_FACTOR = 0.75
REG_MATCH_THRESHOLD = 5

# Initialize session state
if 'factorial_cache' not in st.session_state:
    st.session_state.factorial_cache = {}

def factorial_cache(n):
    """Cache factorial calculations for performance"""
    if n not in st.session_state.factorial_cache:
        st.session_state.factorial_cache[n] = math.factorial(n)
    return st.session_state.factorial_cache[n]

def poisson_pmf(k, lam):
    """Calculate Poisson probability manually"""
    if lam <= 0 or k < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / factorial_cache(k)

@st.cache_data(ttl=3600)
def load_league_data(league_name):
    """Load league data from CSV with caching"""
    try:
        file_path = f"leagues/{league_name}.csv"
        df = pd.read_csv(file_path)
        
        # Basic validation
        required_cols = ['team', 'venue', 'matches', 'xg', 'xga', 'goals_vs_xg']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"CSV missing required columns: {missing_cols}")
            return None
            
        return df
    except FileNotFoundError:
        st.error(f"⚠️ League file not found: leagues/{league_name}.csv")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def prepare_team_data(df):
    """Prepare home and away stats from the data"""
    home_data = df[df['venue'] == 'home'].copy()
    away_data = df[df['venue'] == 'away'].copy()
    
    home_stats = home_data.set_index('team')
    away_stats = away_data.set_index('team')
    
    return home_stats, away_stats

def calculate_regression_factors(home_team_stats, away_team_stats, regression_factor):
    """Calculate attack regression factors"""
    home_matches = home_team_stats['matches']
    away_matches = away_team_stats['matches']
    
    if home_matches >= REG_MATCH_THRESHOLD:
        home_attack_reg = (home_team_stats['goals_vs_xg'] / home_matches) * regression_factor
    else:
        home_attack_reg = 0
    
    if away_matches >= REG_MATCH_THRESHOLD:
        away_attack_reg = (away_team_stats['goals_vs_xg'] / away_matches) * regression_factor
    else:
        away_attack_reg = 0
    
    return home_attack_reg, away_attack_reg

def calculate_expected_goals(home_stats, away_stats, home_attack_reg, away_attack_reg):
    """Calculate expected goals for both teams"""
    away_xga_per_match = away_stats['xga'] / max(away_stats['matches'], 1)
    home_expected = away_xga_per_match * (1 + home_attack_reg)
    
    home_xga_per_match = home_stats['xga'] / max(home_stats['matches'], 1)
    away_expected = home_xga_per_match * (1 + away_attack_reg)
    
    home_expected = max(home_expected, 0.1)
    away_expected = max(away_expected, 0.1)
    
    return home_expected, away_expected

def create_probability_matrix(home_lam, away_lam, max_goals=MAX_GOALS):
    """Create probability matrix for all score combinations"""
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_home = poisson_pmf(i, home_lam)
            prob_away = poisson_pmf(j, away_lam)
            prob_matrix[i, j] = prob_home * prob_away
    
    return prob_matrix

def calculate_outcome_probabilities(prob_matrix):
    """Calculate home win, draw, and away win probabilities"""
    home_win = np.sum(np.triu(prob_matrix, k=1))
    draw = np.sum(np.diag(prob_matrix))
    away_win = np.sum(np.tril(prob_matrix, k=-1))
    
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return home_win, draw, away_win

def calculate_betting_markets(prob_matrix):
    """Calculate betting market probabilities"""
    over_25 = 0
    under_25 = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            total_goals = i + j
            prob = prob_matrix[i, j]
            
            if total_goals > 2.5:
                over_25 += prob
            else:
                under_25 += prob
    
    btts_yes = 0
    btts_no = 0
    
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            prob = prob_matrix[i, j]
            
            if i >= 1 and j >= 1:
                btts_yes += prob
            else:
                btts_no += prob
    
    return over_25, under_25, btts_yes, btts_no

def get_risk_flags(home_stats, away_stats, home_xg, away_xg):
    """Generate risk flags and warnings"""
    flags = []
    
    home_perf = home_stats['goals_vs_xg'] / max(home_stats['matches'], 1)
    away_perf = away_stats['goals_vs_xg'] / max(away_stats['matches'], 1)
    
    if abs(home_perf) > 0.3:
        flags.append(f"⚠️ Home team {'over' if home_perf < 0 else 'under'}performing by {abs(home_perf):.2f} goals/match")
    
    if abs(away_perf) > 0.3:
        flags.append(f"⚠️ Away team {'over' if away_perf < 0 else 'under'}performing by {abs(away_perf):.2f} goals/match")
    
    if 'wins' in home_stats and 'wins' in away_stats:
        home_win_rate = home_stats['wins'] / max(home_stats['matches'], 1)
        away_win_rate = away_stats['wins'] / max(away_stats['matches'], 1)
        
        if abs(home_win_rate - away_win_rate) > 0.3:
            flags.append(f"⚠️ Significant form disparity: {home_win_rate:.0%} vs {away_win_rate:.0%} win rate")
    
    total_xg = home_xg + away_xg
    if total_xg > 3.0:
        flags.append("⚡ High-scoring match expected (Total xG > 3.0)")
    elif total_xg < 2.0:
        flags.append("🛡️ Low-scoring match expected (Total xG < 2.0)")
    
    if home_stats['matches'] < 5:
        flags.append("📊 Small sample size for home team home stats")
    if away_stats['matches'] < 5:
        flags.append("📊 Small sample size for away team away stats")
    
    return flags

def get_betting_suggestions(home_win_prob, draw_prob, away_win_prob, over_25_prob, under_25_prob, btts_yes_prob):
    """Generate betting suggestions based on probabilities"""
    suggestions = []
    threshold = 0.55
    
    if home_win_prob > threshold:
        suggestions.append(f"✅ Home Win ({(home_win_prob*100):.1f}%)")
    if away_win_prob > threshold:
        suggestions.append(f"✅ Away Win ({(away_win_prob*100):.1f}%)")
    if draw_prob > threshold:
        suggestions.append(f"✅ Draw ({(draw_prob*100):.1f}%)")
    
    home_draw_prob = home_win_prob + draw_prob
    away_draw_prob = away_win_prob + draw_prob
    if home_draw_prob > threshold:
        suggestions.append(f"✅ Home Win or Draw ({(home_draw_prob*100):.1f}%)")
    if away_draw_prob > threshold:
        suggestions.append(f"✅ Away Win or Draw ({(away_draw_prob*100):.1f}%)")
    
    if over_25_prob > threshold:
        suggestions.append(f"✅ Over 2.5 Goals ({(over_25_prob*100):.1f}%)")
    if under_25_prob > threshold:
        suggestions.append(f"✅ Under 2.5 Goals ({(under_25_prob*100):.1f}%)")
    
    if btts_yes_prob > threshold:
        suggestions.append(f"✅ Both Teams to Score ({(btts_yes_prob*100):.1f}%)")
    elif btts_yes_prob < (1 - threshold):
        suggestions.append(f"❌ Both Teams NOT to Score ({((1-btts_yes_prob)*100):.1f}%)")
    
    return suggestions

# ========== SIDEBAR CONTROLS ==========
with st.sidebar:
    st.header("⚙️ Match Settings")
    
    leagues = ["Bundesliga", "Premier League", "La Liga", "Serie A", "Ligue 1", "Eredivisie"]
    selected_league = st.selectbox("Select League", leagues)
    
    df = load_league_data(selected_league.lower().replace(" ", "_"))
    
    if df is not None:
        home_stats_df, away_stats_df = prepare_team_data(df)
        
        available_home_teams = sorted(home_stats_df.index.unique())
        available_away_teams = sorted(away_stats_df.index.unique())
        common_teams = sorted(list(set(available_home_teams) & set(available_away_teams)))
        
        if len(common_teams) == 0:
            st.error("❌ No teams with both home and away data available")
        else:
            home_team = st.selectbox("Home Team", common_teams)
            away_team = st.selectbox("Away Team", [t for t in common_teams if t != home_team])
            
            regression_factor = st.slider(
                "Regression Factor",
                min_value=0.0,
                max_value=2.0,
                value=REG_BASE_FACTOR,
                step=0.05,
                help="Adjust how much to regress team performance to mean (higher = more regression)"
            )
            
            calculate_btn = st.button("🎯 Calculate Predictions", type="primary", use_container_width=True)
            
            st.divider()
            st.subheader("📊 Display Options")
            show_matrix = st.checkbox("Show Score Probability Matrix", value=False)

# ========== MAIN CONTENT ==========
if df is None:
    st.warning("📁 Please add league CSV files to the 'leagues' folder")
    st.info("""
    **Your CSV format should include:**
    ```
    team,venue,matches,xg,xga,goals_vs_xg
    Team Name,home,10,25.5,12.3,-2.1
    Team Name,away,10,22.8,15.4,0.5
    ```
    """)
    st.stop()

if 'calculate_btn' not in locals() or not calculate_btn:
    st.info("👈 Select teams and click 'Calculate Predictions' to start")
    
    with st.expander("📋 Preview of Loaded Data"):
        st.dataframe(df.head(10))
    st.stop()

try:
    home_stats = home_stats_df.loc[home_team]
    away_stats = away_stats_df.loc[away_team]
except KeyError as e:
    st.error(f"❌ Team data not found: {e}")
    st.stop()

# ========== DATA PROCESSING ==========
st.header(f"📊 {home_team} vs {away_team}")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.subheader(f"🏠 {home_team} (Home)")
    st.metric("Matches", int(home_stats['matches']))
    if 'wins' in home_stats:
        st.metric("Wins", int(home_stats['wins']))
    st.metric("xG/match", f"{home_stats['xg']/max(home_stats['matches'], 1):.2f}")
    st.metric("xGA/match", f"{home_stats['xga']/max(home_stats['matches'], 1):.2f}")
    if 'gf' in home_stats and 'ga' in home_stats:
        st.metric("GF-GA", f"{home_stats['gf']}-{home_stats['ga']}")

with col2:
    st.subheader(f"✈️ {away_team} (Away)")
    st.metric("Matches", int(away_stats['matches']))
    if 'wins' in away_stats:
        st.metric("Wins", int(away_stats['wins']))
    st.metric("xG/match", f"{away_stats['xg']/max(away_stats['matches'], 1):.2f}")
    st.metric("xGA/match", f"{away_stats['xga']/max(away_stats['matches'], 1):.2f}")
    if 'gf' in away_stats and 'ga' in away_stats:
        st.metric("GF-GA", f"{away_stats['gf']}-{away_stats['ga']}")

with col3:
    home_attack_reg, away_attack_reg = calculate_regression_factors(
        home_stats, away_stats, regression_factor
    )
    
    home_xg, away_xg = calculate_expected_goals(
        home_stats, away_stats, home_attack_reg, away_attack_reg
    )
    
    st.subheader("🎯 Expected Goals")
    
    xg_data = pd.DataFrame({
        'Team': [home_team, away_team],
        'Expected Goals': [home_xg, away_xg]
    })
    
    st.bar_chart(xg_data.set_index('Team'))
    
    col_xg1, col_xg2, col_xg3 = st.columns(3)
    with col_xg1:
        st.metric("Home xG", f"{home_xg:.2f}")
    with col_xg2:
        st.metric("Away xG", f"{away_xg:.2f}")
    with col_xg3:
        total_xg = home_xg + away_xg
        st.metric("Total xG", f"{total_xg:.2f}")
    
    if total_xg > 2.6:
        st.success(f"📈 Over bias: Total xG = {total_xg:.2f} > 2.6")
    elif total_xg < 2.3:
        st.info(f"📉 Under bias: Total xG = {total_xg:.2f} < 2.3")

# ========== PROBABILITY CALCULATIONS ==========
st.divider()
st.header("📈 Probability Calculations")

prob_matrix = create_probability_matrix(home_xg, away_xg)
home_win_prob, draw_prob, away_win_prob = calculate_outcome_probabilities(prob_matrix)
over_25_prob, under_25_prob, btts_yes_prob, btts_no_prob = calculate_betting_markets(prob_matrix)

# ========== SCORE PROBABILITIES ==========
with st.expander("🎯 Most Likely Scores", expanded=True):
    score_probs = []
    for i in range(min(6, prob_matrix.shape[0])):
        for j in range(min(6, prob_matrix.shape[1])):
            prob = prob_matrix[i, j]
            if prob > 0.001:
                score_probs.append(((i, j), prob))
    
    score_probs.sort(key=lambda x: x[1], reverse=True)
    
    cols = st.columns(5)
    for idx, ((home_goals, away_goals), prob) in enumerate(score_probs[:5]):
        with cols[idx]:
            st.metric(
                label=f"{home_goals}-{away_goals}",
                value=f"{prob*100:.1f}%",
                delta="Most Likely" if idx == 0 else None
            )
    
    if score_probs:
        most_likely_score, most_likely_prob = score_probs[0]
        st.success(f"**Most Likely Score:** {most_likely_score[0]}-{most_likely_score[1]} ({(most_likely_prob*100):.1f}%)")

# ========== OUTCOME PROBABILITIES ==========
with st.expander("📊 Match Outcome Probabilities", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Home Win", f"{home_win_prob*100:.1f}%")
        st.progress(home_win_prob)
    
    with col2:
        st.metric("Draw", f"{draw_prob*100:.1f}%")
        st.progress(draw_prob)
    
    with col3:
        st.metric("Away Win", f"{away_win_prob*100:.1f}%")
        st.progress(away_win_prob)
    
    outcome_data = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [home_win_prob, draw_prob, away_win_prob]
    })
    
    st.bar_chart(outcome_data.set_index('Outcome'))

# ========== BETTING MARKETS ==========
with st.expander("💰 Betting Markets", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Over/Under 2.5 Goals")
        st.metric("Over 2.5", f"{over_25_prob*100:.1f}%")
        st.progress(over_25_prob)
        st.metric("Under 2.5", f"{under_25_prob*100:.1f}%")
        st.progress(under_25_prob)
    
    with col2:
        st.subheader("Both Teams to Score")
        st.metric("Yes", f"{btts_yes_prob*100:.1f}%")
        st.progress(btts_yes_prob)
        st.metric("No", f"{btts_no_prob*100:.1f}%")
        st.progress(btts_no_prob)

# ========== BETTING SUGGESTIONS ==========
with st.expander("💡 Betting Suggestions", expanded=True):
    suggestions = get_betting_suggestions(
        home_win_prob, draw_prob, away_win_prob,
        over_25_prob, under_25_prob, btts_yes_prob
    )
    
    if suggestions:
        st.success("**Value Bets Found:**")
        for suggestion in suggestions:
            st.write(suggestion)
    else:
        st.info("No strong value bets identified (all probabilities < 55%)")
    
    st.subheader("Double Chance")
    col_dc1, col_dc2 = st.columns(2)
    with col_dc1:
        home_draw_prob = home_win_prob + draw_prob
        st.metric("Home Win or Draw", f"{home_draw_prob*100:.1f}%")
    with col_dc2:
        away_draw_prob = away_win_prob + draw_prob
        st.metric("Away Win or Draw", f"{away_draw_prob*100:.1f}%")

# ========== RISK FLAGS ==========
with st.expander("⚠️ Risk Flags & Warnings", expanded=False):
    flags = get_risk_flags(home_stats, away_stats, home_xg, away_xg)
    
    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No significant risk flags identified")

# ========== EXPORT ==========
st.divider()
st.header("📤 Export & Share")

summary = f"""
⚽ PREDICTION SUMMARY: {home_team} vs {away_team}
League: {selected_league}

📊 Expected Goals:
• {home_team} xG: {home_xg:.2f}
• {away_team} xG: {away_xg:.2f}
• Total xG: {home_xg + away_xg:.2f}

📈 Most Likely Score: {score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'} ({(score_probs[0][1]*100 if score_probs else 0):.1f}%)

🏆 Outcome Probabilities:
• {home_team} Win: {home_win_prob*100:.1f}%
• Draw: {draw_prob*100:.1f}%
• {away_team} Win: {away_win_prob*100:.1f}%

💰 Betting Markets:
• Over 2.5 Goals: {over_25_prob*100:.1f}%
• Under 2.5 Goals: {under_25_prob*100:.1f}%
• Both Teams to Score: {btts_yes_prob*100:.1f}%

📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Regression Factor: {regression_factor}
"""

st.code(summary, language="text")

col_export1, col_export2 = st.columns(2)

with col_export1:
    st.download_button(
        label="📥 Download Summary",
        data=summary,
        file_name=f"prediction_{home_team}_vs_{away_team}.txt",
        mime="text/plain"
    )

with col_export2:
    export_data = {
        'Metric': [
            'Home Team', 'Away Team', 'League', 'Home xG', 'Away xG', 'Total xG',
            'Home Win %', 'Draw %', 'Away Win %',
            'Over 2.5 %', 'Under 2.5 %', 'BTTS Yes %', 'BTTS No %',
            'Most Likely Score', 'Regression Factor'
        ],
        'Value': [
            home_team, away_team, selected_league,
            f"{home_xg:.2f}", f"{away_xg:.2f}", f"{home_xg+away_xg:.2f}",
            f"{home_win_prob*100:.1f}", f"{draw_prob*100:.1f}", f"{away_win_prob*100:.1f}",
            f"{over_25_prob*100:.1f}", f"{under_25_prob*100:.1f}",
            f"{btts_yes_prob*100:.1f}", f"{btts_no_prob*100:.1f}",
            f"{score_probs[0][0][0] if score_probs else 'N/A'}-{score_probs[0][0][1] if score_probs else 'N/A'}",
            f"{regression_factor}"
        ]
    }
    
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"prediction_data_{home_team}_vs_{away_team}.csv",
        mime="text/csv"
    )

# ========== PROBABILITY MATRIX ==========
if show_matrix:
    with st.expander("🔢 Detailed Probability Matrix", expanded=False):
        matrix_data = []
        for i in range(6):
            row = []
            for j in range(6):
                row.append(f"{prob_matrix[i, j]*100:.2f}%")
            matrix_data.append(row)
        
        matrix_df = pd.DataFrame(
            matrix_data,
            columns=[f'Away {i}' for i in range(6)],
            index=[f'Home {i}' for i in range(6)]
        )
        
        st.dataframe(matrix_df, use_container_width=True)

# ========== FOOTER ==========
st.divider()
st.caption(f"⚡ Predictions calculated using xG regression model | Regression factor: {regression_factor} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
