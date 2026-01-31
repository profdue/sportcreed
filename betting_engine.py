#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE - ATTACK IMBALANCE FOCUS
1. Under 1.5 Goals Alert
2. Under 2.5 Goals Alert  
3. No Draw Candidate (Attack Imbalance > 1.3x)
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
import io

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TeamFormData:
    """Team statistics data structure"""
    teamName: str
    last6: Dict[str, Any]  # Last 6 matches stats
    overall: Dict[str, Any]  # Overall season stats

@dataclass
class MatchContext:
    """Complete match context"""
    teamA: TeamFormData
    teamB: TeamFormData
    isTeamAHome: bool = True

# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Load data from GitHub repository"""
    
    @staticmethod
    def load_from_github(league_name: str) -> Optional[pd.DataFrame]:
        """Load CSV data from GitHub repository"""
        try:
            base_url = "https://raw.githubusercontent.com/profdue/sportcreed/main/leagues"
            url = f"{base_url}/{league_name}.csv"
            
            response = requests.get(url)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            st.success(f"✅ Loaded {league_name} data from GitHub")
            return df
        except Exception as e:
            st.error(f"❌ Error loading {league_name}: {str(e)}")
            try:
                df = pd.read_csv(f"{league_name}.csv")
                st.success(f"✅ Loaded {league_name} data locally")
                return df
            except:
                st.error(f"❌ Could not load {league_name} data")
                return None

# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MatchAnalyzer:
    """Calculate all metrics from raw data"""
    
    @staticmethod
    def calculate_metrics(team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """Calculate all derived metrics needed for decision making"""
        # Parse fractions
        a_cs_num, a_cs_den = MatchAnalyzer._parse_fraction(team_a.last6['cs'])
        a_btts_num, a_btts_den = MatchAnalyzer._parse_fraction(team_a.last6['btts'])
        b_cs_num, b_cs_den = MatchAnalyzer._parse_fraction(team_b.last6['cs'])
        b_btts_num, b_btts_den = MatchAnalyzer._parse_fraction(team_b.last6['btts'])
        
        # Calculate attack imbalance
        a_goals = team_a.last6['goals']
        b_goals = team_b.last6['goals']
        
        if a_goals > b_goals:
            attack_imbalance = a_goals / max(b_goals, 1)  # Avoid division by zero
            imbalance_favors = 'A'
        else:
            attack_imbalance = b_goals / max(a_goals, 1)
            imbalance_favors = 'B'
        
        metrics = {
            'team_a': {
                'goals': a_goals,
                'gpm': a_goals / 6,
                'cs_percent': (a_cs_num / 6) * 100,
                'cs_count': a_cs_num,
                'btts_percent': (a_btts_num / 6) * 100,
                'btts_count': a_btts_num,
                'win_percent': (team_a.last6['wins'] / 6) * 100,
            },
            'team_b': {
                'goals': b_goals,
                'gpm': b_goals / 6,
                'cs_percent': (b_cs_num / 6) * 100,
                'cs_count': b_cs_num,
                'btts_percent': (b_btts_num / 6) * 100,
                'btts_count': b_btts_num,
                'win_percent': (team_b.last6['wins'] / 6) * 100,
            },
            'imbalance': {
                'ratio': attack_imbalance,
                'favors': imbalance_favors,
                'higher_goals': max(a_goals, b_goals),
                'lower_goals': min(a_goals, b_goals)
            },
            'averages': {
                'avg_gpm': (a_goals/6 + b_goals/6) / 2,
                'avg_btts_percent': ((a_btts_num/6)*100 + (b_btts_num/6)*100) / 2,
                'avg_cs_percent': ((a_cs_num/6)*100 + (b_cs_num/6)*100) / 2,
            }
        }
        return metrics
    
    @staticmethod
    def _parse_fraction(frac_str: str) -> Tuple[int, int]:
        """Parse '4/6' -> (4, 6)"""
        if isinstance(frac_str, str) and '/' in frac_str:
            num, den = frac_str.split('/')
            return int(num.strip()), int(den.strip())
        return 0, 6

# ============================================================================
# ATTACK IMBALANCE FILTER DETECTOR
# ============================================================================

class AttackImbalanceDetector:
    """Detect filters with ATTACK IMBALANCE as main driver"""
    
    def __init__(self):
        # Filter 1: Under 1.5 Goals (extreme defensive)
        self.UNDER_15_GPM_THRESHOLD = 0.75
        
        # Filter 2: Under 2.5 Goals (defensive)
        self.UNDER_25_GPM_THRESHOLD = 1.5
        self.UNDER_25_BTTS_THRESHOLD = 50
        
        # Filter 3: No Draw Candidate (Attack Imbalance)
        self.NO_DRAW_IMBALANCE_MIN = 1.3  # One team scores 30%+ more
        self.NO_DRAW_BTTS_MIN = 45        # Both teams can score
        self.NO_DRAW_CS_MAX = 40          # Not both ultra-defensive
        
        # High-draw leagues
        self.HIGH_DRAW_LEAGUES = ['paraguay', 'iran', 'peru']
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData, league: str) -> Dict:
        """Detect three filters with attack imbalance focus"""
        filters = {
            'under_15_alert': False,
            'under_25_alert': False,
            'no_draw_candidate': False,
            'high_draw_league_warning': False,
            'win_difference': team_a.last6['wins'] - team_b.last6['wins'],
            'attack_imbalance_ratio': metrics['imbalance']['ratio'],
            'attack_imbalance_favors': metrics['imbalance']['favors'],
            'debug_info': {}
        }
        
        # Check if league is high-draw
        if any(high_draw_league in league.lower() for high_draw_league in self.HIGH_DRAW_LEAGUES):
            filters['high_draw_league_warning'] = True
        
        # FILTER 1: Under 1.5 Goals Alert
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
            filters['debug_info']['under_15'] = f"Both GPM < {self.UNDER_15_GPM_THRESHOLD}"
        
        # FILTER 2: Under 2.5 Goals Alert
        elif (metrics['team_a']['gpm'] < self.UNDER_25_GPM_THRESHOLD and 
              metrics['team_b']['gpm'] < self.UNDER_25_GPM_THRESHOLD and
              metrics['team_a']['btts_percent'] < self.UNDER_25_BTTS_THRESHOLD and
              metrics['team_b']['btts_percent'] < self.UNDER_25_BTTS_THRESHOLD):
            filters['under_25_alert'] = True
            filters['debug_info']['under_25'] = f"Both GPM < {self.UNDER_25_GPM_THRESHOLD} & BTTS < {self.UNDER_25_BTTS_THRESHOLD}%"
        
        # FILTER 3: No Draw Candidate (ATTACK IMBALANCE DRIVER)
        else:
            # Check attack imbalance condition
            imbalance_ok = metrics['imbalance']['ratio'] >= self.NO_DRAW_IMBALANCE_MIN
            
            # Check BTTS condition
            btts_ok = metrics['averages']['avg_btts_percent'] >= self.NO_DRAW_BTTS_MIN
            
            # Check CS condition (not both ultra-defensive)
            cs_ok = (metrics['team_a']['cs_percent'] < self.NO_DRAW_CS_MAX or 
                    metrics['team_b']['cs_percent'] < self.NO_DRAW_CS_MAX)
            
            filters['debug_info']['no_draw_check'] = {
                'imbalance': f"{metrics['imbalance']['ratio']:.2f} {'≥' if imbalance_ok else '<'} {self.NO_DRAW_IMBALANCE_MIN}",
                'btts': f"{metrics['averages']['avg_btts_percent']:.1f}% {'≥' if btts_ok else '<'} {self.NO_DRAW_BTTS_MIN}%",
                'cs': f"CS%: A={metrics['team_a']['cs_percent']:.1f}%, B={metrics['team_b']['cs_percent']:.1f}% {'✓' if cs_ok else '✗'}",
                'all_conditions': imbalance_ok and btts_ok and cs_ok
            }
            
            if imbalance_ok and btts_ok and cs_ok:
                filters['no_draw_candidate'] = True
                filters['debug_info']['no_draw'] = f"Attack imbalance {metrics['imbalance']['ratio']:.2f}x, avg BTTS {metrics['averages']['avg_btts_percent']:.1f}%"
        
        return filters

# ============================================================================
# SCRIPT GENERATOR
# ============================================================================

class AttackImbalanceScriptGenerator:
    """Generate scripts for attack imbalance strategy"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, is_team_a_home: bool) -> Dict:
        """Generate script with attack imbalance focus"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'predicted_score_range': [],
            'confidence_score': 0,
            'confidence_level': 'low',
            'match_narrative': '',
            'triggered_filter': None,
            'manual_check_required': False,
            'odds_to_check': {},
            'attack_imbalance_analysis': {}
        }
        
        # Determine which filter triggered
        if filters['under_15_alert']:
            script = self._generate_under_15_script(script, metrics, filters)
            script['confidence_score'] = 85
            script['triggered_filter'] = 'under_15'
        
        elif filters['under_25_alert']:
            script = self._generate_under_25_script(script, metrics, filters)
            script['confidence_score'] = 75
            script['triggered_filter'] = 'under_25'
        
        elif filters['no_draw_candidate']:
            script = self._generate_no_draw_script(script, metrics, filters, is_team_a_home)
            script['confidence_score'] = self._calculate_no_draw_confidence(metrics, filters)
            script['triggered_filter'] = 'no_draw'
            script['manual_check_required'] = True
            script['odds_to_check'] = {
                'favorite_range': '1.35-1.75',
                'draw_min': '3.60',
                'check_instructions': 'Skip if favorite odds outside range or draw odds ≤ 3.60'
            }
        
        else:
            script['match_narrative'] = self._generate_no_filter_narrative(metrics, filters)
            script['confidence_score'] = 40
            script['triggered_filter'] = 'none'
        
        # Add attack imbalance analysis
        script['attack_imbalance_analysis'] = {
            'ratio': metrics['imbalance']['ratio'],
            'favors': filters['attack_imbalance_favors'],
            'higher_scorer': metrics['imbalance']['higher_goals'],
            'lower_scorer': metrics['imbalance']['lower_goals'],
            'btts_avg': metrics['averages']['avg_btts_percent'],
            'cs_avg': metrics['averages']['avg_cs_percent']
        }
        
        # Set confidence level
        if script['confidence_score'] >= 80:
            script['confidence_level'] = 'high'
        elif script['confidence_score'] >= 65:
            script['confidence_level'] = 'medium'
        else:
            script['confidence_level'] = 'low'
        
        return script
    
    def _calculate_no_draw_confidence(self, metrics: Dict, filters: Dict) -> int:
        """Calculate confidence for No Draw based on attack imbalance strength"""
        base_confidence = 70
        
        # Boost for stronger imbalance
        imbalance = metrics['imbalance']['ratio']
        if imbalance > 1.8:
            base_confidence += 15
        elif imbalance > 1.5:
            base_confidence += 10
        elif imbalance > 1.3:
            base_confidence += 5
        
        # Boost for higher BTTS%
        btts_avg = metrics['averages']['avg_btts_percent']
        if btts_avg > 70:
            base_confidence += 10
        elif btts_avg > 55:
            base_confidence += 5
        
        return min(base_confidence, 85)
    
    def _generate_under_15_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        script['primary_bets'].append('under_15_goals')
        script['secondary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        script['match_narrative'] = f'EXTREME low-scoring pattern: Both teams < {0.75} GPM'
        return script
    
    def _generate_under_25_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        script['primary_bets'].append('under_25_goals')
        script['secondary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2']
        script['match_narrative'] = 'Both teams defensive with low scoring rates'
        return script
    
    def _generate_no_draw_script(self, script: Dict, metrics: Dict, filters: Dict, is_team_a_home: bool) -> Dict:
        # Determine which team has attack advantage
        if filters['attack_imbalance_favors'] == 'A':
            stronger_name = self.team_a_name
            weaker_name = self.team_b_name
            stronger_goals = metrics['team_a']['goals']
            weaker_goals = metrics['team_b']['goals']
        else:
            stronger_name = self.team_b_name
            weaker_name = self.team_a_name
            stronger_goals = metrics['team_b']['goals']
            weaker_goals = metrics['team_a']['goals']
        
        imbalance_ratio = metrics['imbalance']['ratio']
        
        # Generate predictions based on imbalance strength
        if imbalance_ratio > 1.8:
            # Strong imbalance
            if filters['attack_imbalance_favors'] == 'A':
                script['predicted_score_range'] = ['2-0', '3-0', '2-1', '3-1', '1-0']
            else:
                script['predicted_score_range'] = ['0-2', '0-3', '1-2', '1-3', '0-1']
        else:
            # Moderate imbalance
            if filters['attack_imbalance_favors'] == 'A':
                script['predicted_score_range'] = ['2-1', '1-0', '2-0', '3-2', '1-1']
            else:
                script['predicted_score_range'] = ['1-2', '0-1', '0-2', '2-3', '1-1']
        
        # Betting recommendations
        if filters['attack_imbalance_favors'] == 'A':
            if is_team_a_home:
                script['primary_bets'].append('double_chance_1x')
            else:
                script['primary_bets'].append('double_chance_12')
        else:
            if not is_team_a_home:
                script['primary_bets'].append('double_chance_x2')
            else:
                script['primary_bets'].append('double_chance_12')
        
        # Supportive bets
        script['secondary_bets'].append('btts_yes')
        if metrics['averages']['avg_gpm'] > 2.5:
            script['secondary_bets'].append('over_25_goals')
        
        # Narrative
        script['match_narrative'] = (
            f'ATTACK IMBALANCE DETECTED: {stronger_name} scores {imbalance_ratio:.1f}x more than {weaker_name} '
            f'({stronger_goals} vs {weaker_goals} goals in last 6). '
            f'Both teams score regularly (avg BTTS: {metrics["averages"]["avg_btts_percent"]:.1f}%). '
            f'MANUAL CHECK REQUIRED: Verify odds match No Draw strategy.'
        )
        
        return script
    
    def _generate_no_filter_narrative(self, metrics: Dict, filters: Dict) -> str:
        imbalance = metrics['imbalance']['ratio']
        btts_avg = metrics['averages']['avg_btts_percent']
        
        reasons = []
        
        if imbalance < 1.3:
            reasons.append(f'attack imbalance too low ({imbalance:.2f}x, need ≥1.3x)')
        
        if btts_avg < 45:
            reasons.append(f'BTTS% too low ({btts_avg:.1f}%, need ≥45%)')
        
        if metrics['team_a']['cs_percent'] >= 40 and metrics['team_b']['cs_percent'] >= 40:
            reasons.append('both teams keep clean sheets too often')
        
        if reasons:
            return f'No strong pattern: ' + ', '.join(reasons)
        else:
            return 'Match patterns unclear. Consider skipping.'

# ============================================================================
# MAIN ENGINE
# ============================================================================

class AttackImbalanceEngine:
    """Main orchestrator for attack imbalance strategy"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = AttackImbalanceDetector()
        self.script_generator = None
    
    def analyze_match(self, match_context: MatchContext, league: str) -> Dict:
        self.script_generator = AttackImbalanceScriptGenerator(
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
            'confidence_score': script['confidence_score'],
            'attack_imbalance': script['attack_imbalance_analysis']
        }
        
        return result

# ============================================================================
# DATA PARSING FUNCTION
# ============================================================================

def parse_team_from_csv(df: pd.DataFrame, team_name: str) -> Optional[TeamFormData]:
    """Convert CSV row to TeamFormData for your specific CSV format"""
    try:
        row = df[df['Team'] == team_name].iloc[0]
        
        # Parse fractions
        def parse_frac(field_name):
            value = row[field_name]
            if pd.isna(value):
                return "0/6"
            if isinstance(value, str) and '/' in value:
                return value
            return "0/6"
        
        # Calculate overall btts matches from percentage
        overall_matches = int(row['Overall Matches'])
        
        return TeamFormData(
            teamName=team_name,
            last6={
                'goals': int(row['Last 6 Goals']),
                'wins': int(row['Form W']),
                'draws': int(row['Form D']),
                'losses': int(row['Form L']),
                'cs': parse_frac('Last 6 CS'),
                'btts': parse_frac('Last 6 BTTS')
            },
            overall={
                'matches': overall_matches,
                'goals': int(row['Overall Goals']),
                'wins': int(row['Overall W']),
                'draws': int(row['Overall D']),
                'losses': int(row['Overall L']),
                'cs_percent': float(row['Overall CS%']),
                'btts_percent': float(row['Overall BTTS%'])
            }
        )
    except Exception as e:
        st.error(f"Error parsing data for {team_name}: {str(e)}")
        return None

# ============================================================================
# UI
# ============================================================================

def render_attack_imbalance_dashboard(result: Dict):
    """Display attack imbalance focused results"""
    st.title("⚽ Attack Imbalance Analyzer")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    
    # Confidence
    confidence = result['confidence']
    score = result['confidence_score']
    
    if confidence == 'high':
        st.success(f"Confidence: High 🟢 ({score}/100)")
    elif confidence == 'medium':
        st.warning(f"Confidence: Medium 🟡 ({score}/100)")
    else:
        st.info(f"Confidence: Low 🔵 ({score}/100)")
    
    # Attack Imbalance Analysis
    imbalance = result['attack_imbalance']
    st.markdown("### ⚡ Attack Imbalance Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Imbalance Ratio", f"{imbalance['ratio']:.2f}x")
    with col2:
        st.metric("Avg BTTS%", f"{imbalance['btts_avg']:.1f}%")
    with col3:
        st.metric("Avg CS%", f"{imbalance['cs_avg']:.1f}%")
    
    # Which team has attack advantage
    if imbalance['favors'] == 'A':
        st.info(f"**Attack Advantage**: {result['match_info']['team_a']} ({imbalance['higher_scorer']} vs {imbalance['lower_scorer']} goals)")
    else:
        st.info(f"**Attack Advantage**: {result['match_info']['team_b']} ({imbalance['higher_scorer']} vs {imbalance['lower_scorer']} goals)")
    
    # Three filter indicators
    st.markdown("### 🔍 Triggered Filters")
    filters = result['filters_triggered']
    
    cols = st.columns(3)
    with cols[0]:
        if filters['under_15_alert']:
            st.error("UNDER 1.5 GOALS ✅")
            st.caption(filters.get('debug_info', {}).get('under_15', ''))
        else:
            st.info("UNDER 1.5 GOALS ❌")
    
    with cols[1]:
        if filters['under_25_alert']:
            st.error("UNDER 2.5 GOALS ✅")
            st.caption(filters.get('debug_info', {}).get('under_25', ''))
        else:
            st.info("UNDER 2.5 GOALS ❌")
    
    with cols[2]:
        if filters['no_draw_candidate']:
            st.warning("NO DRAW CANDIDATE ✅")
            st.caption(filters.get('debug_info', {}).get('no_draw', ''))
        else:
            st.info("NO DRAW CANDIDATE ❌")
            debug = filters.get('debug_info', {}).get('no_draw_check', {})
            if debug:
                st.caption(f"Imbalance: {debug.get('imbalance', '')}, BTTS: {debug.get('btts', '')}")
    
    # High-draw league warning
    if filters['high_draw_league_warning']:
        st.error("⚠️ HIGH-DRAW LEAGUE: Be extra cautious")
    
    # Betting recommendations
    st.markdown("### 📋 Recommendations")
    script = result['match_script']
    
    if script['triggered_filter'] == 'none':
        st.info(script['match_narrative'])
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
    
    # Debug info expander
    with st.expander("🔍 View Detailed Analysis"):
        st.write("**Debug Info:**")
        st.json(filters.get('debug_info', {}), expanded=False)
        
        st.write("**Metrics:**")
        st.json({
            'team_a_gpm': result['calculated_metrics']['team_a']['gpm'],
            'team_b_gpm': result['calculated_metrics']['team_b']['gpm'],
            'team_a_btts': result['calculated_metrics']['team_a']['btts_percent'],
            'team_b_btts': result['calculated_metrics']['team_b']['btts_percent'],
            'team_a_cs': result['calculated_metrics']['team_a']['cs_percent'],
            'team_b_cs': result['calculated_metrics']['team_b']['cs_percent']
        }, expanded=False)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    st.set_page_config(
        page_title="Attack Imbalance Analyzer",
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
    with st.spinner("Analyzing attack imbalance..."):
        engine = AttackImbalanceEngine()
        result = engine.analyze_match(match_context, selected_league)
    
    # Display results
    render_attack_imbalance_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("Attack Imbalance Analyzer: Based on real match patterns")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
