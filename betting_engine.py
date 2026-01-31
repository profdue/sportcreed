#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE - THREE FILTERS ONLY
1. Under 1.5 Goals Alert
2. Under 2.5 Goals Alert  
3. No Draw Candidate (requires manual odds check)
"""

# ... (keep imports and DataLoader class unchanged)

# ============================================================================
# THREE FILTER DETECTOR
# ============================================================================

class ThreeFilterDetector:
    """Detect ONLY three filters using existing CSV data"""
    
    def __init__(self):
        # Filter 1: Under 1.5 Goals (existing proven filter)
        self.UNDER_15_GPM_THRESHOLD = 0.75
        
        # Filter 2: Under 2.5 Goals (new, less strict)
        self.UNDER_25_GPM_THRESHOLD = 1.2
        self.UNDER_25_BTTS_THRESHOLD = 50  # BTTS% < 50%
        
        # Filter 3: No Draw Candidate (for manual odds checking)
        self.NO_DRAW_GPM_MIN = 1.3          # Both teams > 1.3 GPM
        self.NO_DRAW_BTTS_MIN = 60          # Both teams BTTS% > 60%
        self.NO_DRAW_WIN_DIFF_MIN = 1       # Win difference at least 1
        self.NO_DRAW_WIN_DIFF_MAX = 3       # Win difference at most 3
        
        # High-draw leagues to flag
        self.HIGH_DRAW_LEAGUES = [
            'paraguay', 'iran', 'peru', 'venezuela', 'ecuador',
            'bolivia', 'colombia', 'uruguay', 'japan', 'saudi_arabia'
        ]
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData, league: str) -> Dict:
        """Detect ONLY three filters"""
        filters = {
            'under_15_alert': False,
            'under_25_alert': False,
            'no_draw_candidate': False,
            'high_draw_league_warning': False,
            'win_difference': team_a.last6['wins'] - team_b.last6['wins']
        }
        
        # Check if league is high-draw
        if any(high_draw_league in league.lower() for high_draw_league in self.HIGH_DRAW_LEAGUES):
            filters['high_draw_league_warning'] = True
        
        # FILTER 1: Under 1.5 Goals Alert (keep existing logic)
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # FILTER 2: Under 2.5 Goals Alert (new)
        if (metrics['team_a']['gpm'] < self.UNDER_25_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_25_GPM_THRESHOLD and
            metrics['team_a']['btts_percent'] < self.UNDER_25_BTTS_THRESHOLD and
            metrics['team_b']['btts_percent'] < self.UNDER_25_BTTS_THRESHOLD):
            filters['under_25_alert'] = True
        
        # FILTER 3: No Draw Candidate (for manual odds checking)
        abs_win_diff = abs(filters['win_difference'])
        
        if (metrics['team_a']['gpm'] > self.NO_DRAW_GPM_MIN and
            metrics['team_b']['gpm'] > self.NO_DRAW_GPM_MIN and
            metrics['team_a']['btts_percent'] > self.NO_DRAW_BTTS_MIN and
            metrics['team_b']['btts_percent'] > self.NO_DRAW_BTTS_MIN and
            self.NO_DRAW_WIN_DIFF_MIN <= abs_win_diff <= self.NO_DRAW_WIN_DIFF_MAX):
            filters['no_draw_candidate'] = True
        
        return filters

# ============================================================================
# SIMPLIFIED SCRIPT GENERATOR
# ============================================================================

class ThreeFilterScriptGenerator:
    """Generate scripts for ONLY three filters"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, is_team_a_home: bool) -> Dict:
        """Generate script for three filters only"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'predicted_score_range': [],
            'confidence_score': 0,
            'confidence_level': 'low',
            'match_narrative': '',
            'triggered_filter': None,
            'manual_check_required': False,
            'odds_to_check': {}
        }
        
        # Determine which filter triggered (in priority order)
        if filters['under_15_alert']:
            script = self._generate_under_15_script(script, metrics)
            script['confidence_score'] = 85
            script['triggered_filter'] = 'under_15'
        
        elif filters['under_25_alert']:
            script = self._generate_under_25_script(script, metrics)
            script['confidence_score'] = 75
            script['triggered_filter'] = 'under_25'
        
        elif filters['no_draw_candidate']:
            script = self._generate_no_draw_script(script, metrics, filters)
            script['confidence_score'] = 70
            script['triggered_filter'] = 'no_draw'
            script['manual_check_required'] = True
            script['odds_to_check'] = {
                'favorite_range': '1.35-1.75',
                'draw_min': '3.60',
                'check_instructions': 'Skip if favorite odds outside range or draw odds ≤ 3.60'
            }
        
        else:
            script['match_narrative'] = 'No strong filter triggered. Consider skipping.'
            script['confidence_score'] = 40
            script['triggered_filter'] = 'none'
        
        # Add high-draw league warning if applicable
        if filters['high_draw_league_warning']:
            script['match_narrative'] += ' WARNING: League has high draw rate.'
        
        # Set confidence level
        if script['confidence_score'] >= 80:
            script['confidence_level'] = 'high'
        elif script['confidence_score'] >= 65:
            script['confidence_level'] = 'medium'
        else:
            script['confidence_level'] = 'low'
        
        return script
    
    def _generate_under_15_script(self, script: Dict, metrics: Dict) -> Dict:
        script['primary_bets'].append('under_15_goals')
        script['secondary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        script['match_narrative'] = 'Both attacks extremely weak - very low scoring expected'
        return script
    
    def _generate_under_25_script(self, script: Dict, metrics: Dict) -> Dict:
        script['primary_bets'].append('under_25_goals')
        script['secondary_bets'].append('btts_no')
        
        # More varied score predictions for Under 2.5
        script['predicted_score_range'] = ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2']
        script['match_narrative'] = 'Both teams low-scoring with solid defenses'
        return script
    
    def _generate_no_draw_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        win_diff = filters['win_difference']
        
        # Determine which team is favorite based on win difference
        if win_diff > 0:
            favorite_name = self.team_a_name
            underdog_name = self.team_b_name
            script['predicted_score_range'] = ['2-1', '3-1', '2-0', '3-2', '1-0']
        else:
            favorite_name = self.team_b_name
            underdog_name = self.team_a_name
            script['predicted_score_range'] = ['1-2', '1-3', '0-2', '2-3', '0-1']
        
        # For No Draw, primary is "Double Chance" (1X for home or 12 for neutral)
        if win_diff > 0:  # Team A is favorite
            if is_team_a_home:
                script['primary_bets'].append('double_chance_1x')
            else:
                script['primary_bets'].append('double_chance_12')
        else:  # Team B is favorite
            if not is_team_a_home:
                script['primary_bets'].append('double_chance_x2')
            else:
                script['primary_bets'].append('double_chance_12')
        
        # Secondary bets that support No Draw theory
        script['secondary_bets'].append('btts_yes')
        script['secondary_bets'].append('over_25_goals')
        
        script['match_narrative'] = (
            f'High-scoring match with both teams leaking goals. '
            f'{favorite_name} has better form but both teams likely to score. '
            f'MANUAL CHECK REQUIRED: Verify odds (favorite 1.35-1.75, draw >3.60)'
        )
        
        return script

# ============================================================================
# SIMPLIFIED MAIN ENGINE
# ============================================================================

class SimplifiedBettingEngine:
    """Main orchestrator for three filters only"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = ThreeFilterDetector()
        self.script_generator = None
    
    def analyze_match(self, match_context: MatchContext, league: str) -> Dict:
        self.script_generator = ThreeFilterScriptGenerator(
            match_context.teamA.teamName, 
            match_context.teamB.teamName
        )
        
        metrics = self.metrics_calc.calculate_metrics(
            match_context.teamA, 
            match_context.teamB
        )
        
        filters = self.filter_detector.detect_filters(
            metrics, 
            match_context.teamA, 
            match_context.teamB,
            league
        )
        
        script = self.script_generator.generate_script(
            metrics, filters, match_context.isTeamAHome
        )
        
        result = {
            'match_info': {
                'team_a': match_context.teamA.teamName,
                'team_b': match_context.teamB.teamName,
                'venue': 'home' if match_context.isTeamAHome else 'away',
                'league': league
            },
            'calculated_metrics': metrics,
            'filters_triggered': filters,
            'match_script': script,
            'predicted_score_range': script['predicted_score_range'],
            'confidence': script['confidence_level'],
            'confidence_score': script['confidence_score']
        }
        
        return result

# ============================================================================
# SIMPLIFIED UI
# ============================================================================

def render_simplified_dashboard(result: Dict):
    """Display only three filter results"""
    st.title("⚽ Three-Filter Analyzer")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    
    # Confidence display
    confidence = result['confidence']
    score = result['confidence_score']
    
    if confidence == 'high':
        st.success(f"Confidence: High 🟢 ({score}/100)")
    elif confidence == 'medium':
        st.warning(f"Confidence: Medium 🟡 ({score}/100)")
    else:
        st.info(f"Confidence: Low 🔵 ({score}/100)")
    
    # Three filter indicators
    st.markdown("### 🔍 Triggered Filters")
    filters = result['filters_triggered']
    
    cols = st.columns(3)
    with cols[0]:
        if filters['under_15_alert']:
            st.error("UNDER 1.5 GOALS ✅")
        else:
            st.info("UNDER 1.5 GOALS ❌")
    
    with cols[1]:
        if filters['under_25_alert']:
            st.error("UNDER 2.5 GOALS ✅")
        else:
            st.info("UNDER 2.5 GOALS ❌")
    
    with cols[2]:
        if filters['no_draw_candidate']:
            st.warning("NO DRAW CANDIDATE ✅")
        else:
            st.info("NO DRAW CANDIDATE ❌")
    
    # High-draw league warning
    if filters['high_draw_league_warning']:
        st.error("⚠️ HIGH-DRAW LEAGUE: Be extra cautious")
    
    # Betting recommendations
    st.markdown("### 📋 Recommendations")
    script = result['match_script']
    
    if script['triggered_filter'] == 'none':
        st.info("No strong filter triggered. Consider skipping this match.")
    else:
        st.write(script['match_narrative'])
        
        # Show primary bets
        if script['primary_bets']:
            st.markdown("**Primary Bets:**")
            for bet in script['primary_bets']:
                bet_display = bet.replace('_', ' ').title()
                st.write(f"• {bet_display}")
        
        # Show secondary bets
        if script['secondary_bets']:
            st.markdown("**Secondary Bets:**")
            for bet in script['secondary_bets']:
                bet_display = bet.replace('_', ' ').title()
                st.write(f"• {bet_display}")
        
        # Manual check for No Draw
        if script['manual_check_required']:
            st.markdown("### ⚠️ MANUAL ODDS CHECK REQUIRED")
            odds = script['odds_to_check']
            st.error(f"**Verify these odds before betting:**")
            st.write(f"• Favorite odds must be: {odds['favorite_range']}")
            st.write(f"• Draw odds must be >: {odds['draw_min']}")
            st.write(f"• {odds['check_instructions']}")
    
    # Predicted scores
    if script['predicted_score_range']:
        st.markdown("### 🎯 Most Likely Scores")
        scores = script['predicted_score_range'][:5]
        score_cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores):
            with score_cols[idx]:
                st.info(f"**{score}**")

# ============================================================================
# MODIFIED MAIN FUNCTION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Three-Filter Betting Analyzer",
        page_icon="⚽",
        layout="wide"
    )
    
    st.sidebar.title("⚙️ Configuration")
    
    # League selection
    league_options = [
        "bundesliga", "premier_league", "laliga", "serie_a", 
        "ligue_1", "eredivisie", "championship"
    ]
    selected_league = st.sidebar.selectbox("Select League", league_options)
    
    if not selected_league:
        st.info("👈 Please select a league")
        return
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_from_github(selected_league)
    if df is None:
        return
    
    # Team selection
    teams = sorted(df['Team'].unique().tolist())
    col1, col2 = st.sidebar.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, key="team_a")
    with col2:
        remaining_teams = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", remaining_teams, key="team_b")
    
    # Venue selection
    venue = st.sidebar.radio("Venue", ["Team A Home", "Team B Home", "Neutral"], horizontal=True)
    is_team_a_home = venue == "Team A Home"
    
    # Parse team data
    team_a_data = parse_team_from_csv(df, team_a)
    team_b_data = parse_team_from_csv(df, team_b)
    
    if not team_a_data or not team_b_data:
        st.error("❌ Could not load team data")
        return
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        isTeamAHome=is_team_a_home
    )
    
    # Run analysis
    with st.spinner("Analyzing with three filters..."):
        engine = SimplifiedBettingEngine()
        result = engine.analyze_match(match_context, selected_league)
    
    # Display results
    render_simplified_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("Three-Filter Analyzer: Under 1.5 • Under 2.5 • No Draw Candidate")

# ============================================================================
# KEEP EXISTING HELPER CLASSES
# ============================================================================

# Keep DataLoader, MatchAnalyzer, parse_team_from_csv unchanged
# Remove BettingSlipGenerator, RiskAwareExtremeFilterDetector, etc.

if __name__ == "__main__":
    main()
