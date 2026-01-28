#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v3.1 - Complete Single File Implementation
Streamlit app with exact 5-filter logic using real CSV data
WITH DATA VERIFICATION DISPLAY
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

# ============================================================================
# SECTION 1: DATA STRUCTURES
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
# SECTION 2: ENGINE CORE CLASSES
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
        
        metrics = {
            'team_a': {
                'gpm': team_a.last6['goals'] / 6,
                'cs_percent': (a_cs_num / 6) * 100,
                'cs_count': a_cs_num,
                'btts_percent': (a_btts_num / 6) * 100,
                'btts_count': a_btts_num,
                'win_percent': (team_a.last6['wins'] / 6) * 100,
                'loss_percent': (team_a.last6['losses'] / 6) * 100
            },
            'team_b': {
                'gpm': team_b.last6['goals'] / 6,
                'cs_percent': (b_cs_num / 6) * 100,
                'cs_count': b_cs_num,
                'btts_percent': (b_btts_num / 6) * 100,
                'btts_count': b_btts_num,
                'win_percent': (team_b.last6['wins'] / 6) * 100,
                'loss_percent': (team_b.last6['losses'] / 6) * 100
            },
            'averages': {
                'avg_gpm': (team_a.last6['goals']/6 + team_b.last6['goals']/6) / 2,
                'avg_cs_percent': ((a_cs_num/6)*100 + (b_cs_num/6)*100) / 2,
                'avg_btts_percent': ((a_btts_num/6)*100 + (b_btts_num/6)*100) / 2
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


class ExtremeFilterDetector:
    """Detect all 5 extreme filters"""
    
    # Filter constants
    UNDER_15_GPM_THRESHOLD = 0.75
    CS_PERCENT_THRESHOLD = 20
    CS_PERCENT_STRONG = 50
    OPPONENT_GPM_WEAK = 1.0
    BTTS_PERCENT_LOW = 40
    CS_PERCENT_DECENT = 30
    GPM_MODERATE = 1.5
    RECENT_BTTS_INFLATION_MIN = 70
    SEASON_BTTS_REGRESSION_MAX = 60  # CHANGED FROM 70 TO 60
    GPM_INFLATION_FACTOR = 1.5
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """Detect which extreme filters are triggered"""
        filters = {
            'under_15_alert': False,
            'btts_banker': False,
            'btts_enhanced': False,
            'clean_sheet_alert': {'team': None, 'direction': None},
            'low_scoring_alert': False,
            'low_scoring_type': None,
            'regression_alert': {'team': None, 'type': None}
        }
        
        # FILTER 1: Under 1.5 Goals Alert
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # FILTER 2: BTTS Banker Alert
        if (metrics['team_a']['cs_count'] == 0 and 
            metrics['team_b']['cs_count'] == 0):
            filters['btts_banker'] = True
            
            # Enhancement: Add Over 2.5 if high scoring form
            if (metrics['team_a']['gpm'] > 1.5 or 
                metrics['team_b']['gpm'] > 1.5):
                filters['btts_enhanced'] = True
        
        # FILTER 3: Clean Sheet Alert
        if (metrics['team_a']['cs_percent'] > self.CS_PERCENT_STRONG and 
            metrics['team_b']['gpm'] < self.OPPONENT_GPM_WEAK):
            filters['clean_sheet_alert'] = {'team': 'A', 'direction': 'win_to_nil'}
        elif (metrics['team_b']['cs_percent'] > self.CS_PERCENT_STRONG and 
              metrics['team_a']['gpm'] < self.OPPONENT_GPM_WEAK):
            filters['clean_sheet_alert'] = {'team': 'B', 'direction': 'win_to_nil'}
        
        # FILTER 4: Low-Scoring Alert
        # Condition 4A: Both teams have low BTTS percentage
        if (metrics['team_a']['btts_percent'] < self.BTTS_PERCENT_LOW and 
            metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'both_low_btts'
        
        # Condition 4B: Defensive team vs. Leaky but low-scoring team
        elif (metrics['team_a']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_a']['cs_percent'] > self.CS_PERCENT_DECENT and 
              metrics['team_b']['gpm'] < self.GPM_MODERATE):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_leaky_a'
        
        elif (metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_b']['cs_percent'] > self.CS_PERCENT_DECENT and 
              metrics['team_a']['gpm'] < self.GPM_MODERATE):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_leaky_b'
        
        # FILTER 5: Regression Alert
        # Check BTTS regression
        if team_a.overall and team_b.overall:
            a_last6_btts = metrics['team_a']['btts_percent']
            b_last6_btts = metrics['team_b']['btts_percent']
            a_season_btts = team_a.overall['btts_percent']
            b_season_btts = team_b.overall['btts_percent']
            
            if (a_last6_btts > self.RECENT_BTTS_INFLATION_MIN and 
                a_season_btts < self.SEASON_BTTS_REGRESSION_MAX):
                filters['regression_alert'] = {'team': 'A', 'type': 'btts_regression'}
            elif (b_last6_btts > self.RECENT_BTTS_INFLATION_MIN and 
                  b_season_btts < self.SEASON_BTTS_REGRESSION_MAX):
                filters['regression_alert'] = {'team': 'B', 'type': 'btts_regression'}
            
            # Check GPM inflation
            a_season_gpm = team_a.overall['goals'] / team_a.overall['matches']
            b_season_gpm = team_b.overall['goals'] / team_b.overall['matches']
            
            if metrics['team_a']['gpm'] > (a_season_gpm * self.GPM_INFLATION_FACTOR):
                filters['regression_alert'] = {'team': 'A', 'type': 'gpm_regression'}
            elif metrics['team_b']['gpm'] > (b_season_gpm * self.GPM_INFLATION_FACTOR):
                filters['regression_alert'] = {'team': 'B', 'type': 'gpm_regression'}
        
        return filters


class CurrentFormAnalyzer:
    """Analyze current form to determine favorite"""
    
    @staticmethod
    def analyze_form(team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """Analyze current form and determine favorite"""
        analysis = {
            'favorite': None,
            'form_edge': None,  # 'strong', 'moderate', 'slight', 'none'
            'goal_potential': None,  # 'high', 'medium', 'low'
            'btts_likelihood': None  # 'high', 'medium', 'low'
        }
        
        # Determine favorite based on wins
        win_diff = team_a.last6['wins'] - team_b.last6['wins']
        
        if win_diff >= 2:
            analysis['favorite'] = 'team_a'
            analysis['form_edge'] = 'strong'
        elif win_diff <= -2:
            analysis['favorite'] = 'team_b'
            analysis['form_edge'] = 'strong'
        elif win_diff == 1:
            analysis['favorite'] = 'team_a'
            analysis['form_edge'] = 'slight'
        elif win_diff == -1:
            analysis['favorite'] = 'team_b'
            analysis['form_edge'] = 'slight'
        else:
            analysis['form_edge'] = 'none'
        
        # Goal potential (average of both teams' GPM)
        avg_gpm = (team_a.last6['goals']/6 + team_b.last6['goals']/6) / 2
        if avg_gpm > 1.8:
            analysis['goal_potential'] = 'high'
        elif avg_gpm > 1.3:
            analysis['goal_potential'] = 'medium'
        else:
            analysis['goal_potential'] = 'low'
        
        # BTTS likelihood
        a_btts_num, _ = MatchAnalyzer._parse_fraction(team_a.last6['btts'])
        b_btts_num, _ = MatchAnalyzer._parse_fraction(team_b.last6['btts'])
        avg_btts = ((a_btts_num/6)*100 + (b_btts_num/6)*100) / 2
        
        if avg_btts > 60:
            analysis['btts_likelihood'] = 'high'
        elif avg_btts > 40:
            analysis['btts_likelihood'] = 'medium'
        else:
            analysis['btts_likelihood'] = 'low'
            
        return analysis


class MatchScriptGenerator:
    """Generate match script and betting narrative"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, form_analysis: Dict) -> Dict:
        """Generate complete match script and betting recommendations"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'predicted_score_range': [],
            'confidence': 'low',  # low, medium, high
            'match_narrative': '',
            'warnings': []
        }
        
        # RULE 1: Extreme Filters override everything
        if filters['under_15_alert']:
            script['primary_bets'].append('under_15_goals')
            script['primary_bets'].append('btts_no')
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams have broken attacks - expect very few goals'
            return script
            
        if filters['low_scoring_alert']:
            script['primary_bets'].append('under_25_goals')
            script['primary_bets'].append('btts_no')
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            
            if filters['low_scoring_type'] == 'both_low_btts':
                script['match_narrative'] = 'Both teams rarely see both teams score - low-scoring affair likely'
            else:
                script['match_narrative'] = 'Defensive team meets low-scoring opponent - clean sheet likely'
            return script
            
        if filters['btts_banker']:
            script['primary_bets'].append('btts_yes')
            script['secondary_bets'].append('over_15_goals')
            
            if filters['btts_enhanced']:
                script['value_bets'].append('over_25_goals')
                script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2']
            else:
                script['predicted_score_range'] = ['1-1', '2-1', '1-2']
            
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams consistently concede - expect goals at both ends'
            
        if filters['clean_sheet_alert']['team']:
            team_name = self.team_a_name if filters['clean_sheet_alert']['team'] == 'A' else self.team_b_name
            script['primary_bets'].append(f'{team_name.lower().replace(" ", "_")}_win_to_nil')
            script['secondary_bets'].append('under_25_goals')
            script['predicted_score_range'] = [f'2-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-2',
                                              f'1-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = f'{team_name} strong defensively against weak attack'
            return script
        
        # If no extreme filters triggered, use form analysis
        if not script['primary_bets']:
            script['confidence'] = 'medium'
            
            # Add BTTS based on likelihood
            if form_analysis['btts_likelihood'] == 'high':
                script['secondary_bets'].append('btts_yes')
            elif form_analysis['btts_likelihood'] == 'low':
                script['secondary_bets'].append('btts_no')
            
            # Add goal markets based on potential
            if form_analysis['goal_potential'] == 'high':
                script['secondary_bets'].append('over_25_goals')
                script['secondary_bets'].append('over_15_goals')
            elif form_analysis['goal_potential'] == 'medium':
                script['secondary_bets'].append('over_15_goals')
            else:
                script['secondary_bets'].append('under_25_goals')
            
            # Add favorite if exists
            if form_analysis['favorite']:
                favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
                script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win_or_draw')
            
            script['match_narrative'] = self._generate_form_narrative(form_analysis)
        
        # Generate score range if not already set
        if not script['predicted_score_range']:
            script['predicted_score_range'] = self._generate_score_range(script, metrics, form_analysis)
        
        # Add regression warnings
        if filters['regression_alert']['team']:
            team_name = self.team_a_name if filters['regression_alert']['team'] == 'A' else self.team_b_name
            alert_type = filters['regression_alert']['type']
            
            if alert_type == 'btts_regression':
                warning = f'⚠️ {team_name} recent BTTS rate may regress to season average'
                script['warnings'].append(warning)
            elif alert_type == 'gpm_regression':
                warning = f'⚠️ {team_name} recent scoring rate may regress to season average'
                script['warnings'].append(warning)
        
        return script
    
    def _generate_form_narrative(self, form_analysis: Dict) -> str:
        """Generate narrative based on form analysis"""
        narrative_parts = []
        
        if form_analysis['form_edge'] == 'strong':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has strong recent form advantage.')
        elif form_analysis['form_edge'] == 'slight':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has slight recent form edge.')
        
        if form_analysis['goal_potential'] == 'high':
            narrative_parts.append('High scoring potential.')
        elif form_analysis['goal_potential'] == 'low':
            narrative_parts.append('Low scoring potential.')
        
        if form_analysis['btts_likelihood'] == 'high':
            narrative_parts.append('Both teams likely to score.')
        elif form_analysis['btts_likelihood'] == 'low':
            narrative_parts.append('Clean sheet possible.')
        
        return ' '.join(narrative_parts)
    
    def _generate_score_range(self, script: Dict, metrics: Dict, form_analysis: Dict) -> List[str]:
        """Generate likely score ranges based on script"""
        scores = []
        
        if 'btts_yes' in script['primary_bets'] or 'btts_yes' in script['secondary_bets']:
            if 'over_25_goals' in script['value_bets'] or 'over_25_goals' in script['secondary_bets']:
                scores.extend(['2-1', '1-2', '2-2', '3-1', '1-3'])
            else:
                scores.extend(['1-1', '2-1', '1-2'])
        elif 'btts_no' in script['primary_bets'] or 'btts_no' in script['secondary_bets']:
            scores.extend(['1-0', '0-1', '2-0', '0-2'])
        elif 'under_15_goals' in script['primary_bets']:
            scores.extend(['0-0', '1-0', '0-1'])
        elif 'under_25_goals' in script['primary_bets']:
            scores.extend(['1-0', '0-1', '1-1', '2-0', '0-2'])
        else:
            # Default based on form analysis
            if form_analysis['goal_potential'] == 'high':
                scores.extend(['2-1', '1-2', '2-0', '0-2', '3-1'])
            elif form_analysis['goal_potential'] == 'low':
                scores.extend(['1-0', '0-1', '1-1', '0-0'])
            else:
                scores.extend(['1-0', '0-1', '1-1', '2-1', '1-2'])
        
        return scores[:5]


class BettingSlipGenerator:
    """Generate final betting recommendations"""
    
    def __init__(self):
        self.config = {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'min_confidence': 'medium'
        }
    
    def generate_slip(self, script: Dict, team_a_name: str, team_b_name: str) -> Dict:
        """Generate optimized betting slip"""
        slip = {
            'recommended_bets': [],
            'stake_suggestions': {},
            'total_units': 10,
            'match_summary': script.get('match_narrative', ''),
            'warnings': script.get('warnings', [])
        }
        
        # Skip if confidence too low
        confidence_order = {'high': 3, 'medium': 2, 'low': 1}
        min_conf_value = confidence_order.get(self.config['min_confidence'], 1)
        
        if confidence_order.get(script['confidence'], 0) < min_conf_value:
            slip['recommended_bets'].append({
                'type': 'NO_BET',
                'priority': 'none',
                'reason': 'confidence_too_low',
                'confidence': script['confidence']
            })
            return slip
        
        # Priority 1: Primary bets with high confidence
        for bet in script['primary_bets']:
            slip['recommended_bets'].append({
                'type': bet,
                'priority': 'high',
                'reason': 'extreme_filter',
                'confidence': script['confidence']
            })
        
        # Priority 2: Secondary bets if space
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'medium',
                    'reason': 'form_analysis'
                })
        
        # Limit to max bets and remove duplicates
        unique_bets = []
        seen = set()
        for bet in slip['recommended_bets']:
            if bet['type'] not in seen:
                unique_bets.append(bet)
                seen.add(bet['type'])
        
        slip['recommended_bets'] = unique_bets[:self.config['max_bets_per_match']]
        
        # Generate stake suggestions
        total_stake = slip['total_units']
        for bet in slip['recommended_bets']:
            if bet['priority'] == 'high':
                slip['stake_suggestions'][bet['type']] = round(total_stake * self.config['stake_distribution']['high'], 1)
            elif bet['priority'] == 'medium':
                slip['stake_suggestions'][bet['type']] = round(total_stake * self.config['stake_distribution']['medium'], 1)
            else:
                slip['stake_suggestions'][bet['type']] = round(total_stake * self.config['stake_distribution']['low'], 1)
        
        # Add score suggestions
        if script['predicted_score_range']:
            slip['score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip


class BettingAnalyticsEngine:
    """Main orchestrator - complete pipeline"""
    
    def __init__(self):
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = ExtremeFilterDetector()
        self.form_analyzer = CurrentFormAnalyzer()
        self.script_generator = None
        self.slip_generator = BettingSlipGenerator()
    
    def analyze_match(self, match_context: MatchContext) -> Dict:
        """Complete analysis pipeline"""
        # Initialize script generator with team names
        self.script_generator = MatchScriptGenerator(
            match_context.teamA.teamName, 
            match_context.teamB.teamName
        )
        
        # Step 1: Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(
            match_context.teamA, 
            match_context.teamB
        )
        
        # Step 2: Detect extreme filters
        filters = self.filter_detector.detect_filters(
            metrics, 
            match_context.teamA, 
            match_context.teamB
        )
        
        # Step 3: Analyze current form
        form_analysis = self.form_analyzer.analyze_form(
            match_context.teamA, 
            match_context.teamB
        )
        
        # Step 4: Generate match script
        script = self.script_generator.generate_script(
            metrics, filters, form_analysis
        )
        
        # Step 5: Generate betting slip
        slip = self.slip_generator.generate_slip(
            script,
            match_context.teamA.teamName,
            match_context.teamB.teamName
        )
        
        # Compile final result
        result = {
            'match_info': {
                'team_a': match_context.teamA.teamName,
                'team_b': match_context.teamB.teamName,
                'venue': 'home' if match_context.isTeamAHome else 'away'
            },
            'calculated_metrics': metrics,
            'filters_triggered': filters,
            'form_analysis': form_analysis,
            'match_script': script,
            'betting_slip': slip,
            'predicted_score_range': script['predicted_score_range'],
            'confidence': script['confidence']
        }
        
        return result


# ============================================================================
# SECTION 3: DATA PARSING FUNCTIONS
# ============================================================================

def parse_team_from_csv(df: pd.DataFrame, team_name: str) -> Optional[TeamFormData]:
    """Convert CSV row to TeamFormData"""
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
        overall_cs_count = int((row['Overall CS%'] / 100) * overall_matches)
        overall_btts_count = int((row['Overall BTTS%'] / 100) * overall_matches)
        
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
                'btts_percent': float(row['Overall BTTS%']),
                'cs_count': overall_cs_count,
                'btts_count': overall_btts_count
            }
        )
    except Exception as e:
        st.error(f"Error parsing data for {team_name}: {str(e)}")
        return None


def load_league_data(league_name: str) -> Optional[pd.DataFrame]:
    """Load CSV data for selected league"""
    try:
        # Try different possible file locations/names
        possible_paths = [
            f"{league_name}.csv",
            f"data/{league_name}.csv",
            f"leagues/{league_name}.csv",
            f"{league_name.lower()}.csv",
            f"leagues/{league_name.lower()}.csv"
        ]
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Loaded {league_name} data successfully")
                return df
            except:
                continue
        
        st.error(f"❌ Could not find CSV file for {league_name}")
        st.info("Looking for files named: " + ", ".join(possible_paths))
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# ============================================================================
# SECTION 4: STREAMLIT UI COMPONENTS WITH DATA VERIFICATION
# ============================================================================

def render_data_verification(team_a_data: TeamFormData, team_b_data: TeamFormData):
    """Show actual data from CSV for verification"""
    st.markdown("### 📊 DATA VERIFICATION (From CSV)")
    
    # Parse fractions for display
    a_cs_num, a_cs_den = MatchAnalyzer._parse_fraction(team_a_data.last6['cs'])
    a_btts_num, a_btts_den = MatchAnalyzer._parse_fraction(team_a_data.last6['btts'])
    b_cs_num, b_cs_den = MatchAnalyzer._parse_fraction(team_b_data.last6['cs'])
    b_btts_num, b_btts_den = MatchAnalyzer._parse_fraction(team_b_data.last6['btts'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### 🏠 {team_a_data.teamName}")
        
        st.write("**Last 6 Matches:**")
        st.write(f"- Goals: **{team_a_data.last6['goals']}** ({team_a_data.last6['goals']/6:.2f} GPM)")
        st.write(f"- Form: **{team_a_data.last6['wins']}W-{team_a_data.last6['draws']}D-{team_a_data.last6['losses']}L**")
        st.write(f"- Clean Sheets: **{a_cs_num}/{a_cs_den}** ({a_cs_num/a_cs_den*100:.1f}%)")
        st.write(f"- BTTS: **{a_btts_num}/{a_btts_den}** ({a_btts_num/a_btts_den*100:.1f}%)")
        
        st.write("**Overall Season:**")
        st.write(f"- Matches: {team_a_data.overall['matches']}")
        st.write(f"- Goals: {team_a_data.overall['goals']} ({team_a_data.overall['goals']/team_a_data.overall['matches']:.2f} GPM)")
        st.write(f"- CS%: **{team_a_data.overall['cs_percent']:.2f}%**")
        st.write(f"- BTTS%: **{team_a_data.overall['btts_percent']:.2f}%**")
    
    with col2:
        st.markdown(f"#### 🚌 {team_b_data.teamName}")
        
        st.write("**Last 6 Matches:**")
        st.write(f"- Goals: **{team_b_data.last6['goals']}** ({team_b_data.last6['goals']/6:.2f} GPM)")
        st.write(f"- Form: **{team_b_data.last6['wins']}W-{team_b_data.last6['draws']}D-{team_b_data.last6['losses']}L**")
        st.write(f"- Clean Sheets: **{b_cs_num}/{b_cs_den}** ({b_cs_num/b_cs_den*100:.1f}%)")
        st.write(f"- BTTS: **{b_btts_num}/{b_btts_den}** ({b_btts_num/b_btts_den*100:.1f}%)")
        
        st.write("**Overall Season:**")
        st.write(f"- Matches: {team_b_data.overall['matches']}")
        st.write(f"- Goals: {team_b_data.overall['goals']} ({team_b_data.overall['goals']/team_b_data.overall['matches']:.2f} GPM)")
        st.write(f"- CS%: **{team_b_data.overall['cs_percent']:.2f}%**")
        st.write(f"- BTTS%: **{team_b_data.overall['btts_percent']:.2f}%**")
    
    st.markdown("---")


def render_filter_calculations(metrics: Dict, team_a_data: TeamFormData, team_b_data: TeamFormData):
    """Show filter calculations for verification"""
    st.markdown("### 🔍 FILTER CALCULATIONS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Filter 1: Under 1.5 Goals**")
        st.write(f"- Köln GPM: {metrics['team_a']['gpm']:.2f}")
        st.write(f"- Wolfsburg GPM: {metrics['team_b']['gpm']:.2f}")
        trigger_1 = metrics['team_a']['gpm'] < 0.75 and metrics['team_b']['gpm'] < 0.75
        st.write(f"- **Trigger:** {'✅ YES' if trigger_1 else '❌ NO'}")
        if not trigger_1:
            st.caption("(Both GPM must be < 0.75)")
    
    with col2:
        st.write("**Filter 2: BTTS Banker**")
        st.write(f"- Köln CS count: {metrics['team_a']['cs_count']}")
        st.write(f"- Wolfsburg CS count: {metrics['team_b']['cs_count']}")
        trigger_2 = metrics['team_a']['cs_count'] == 0 and metrics['team_b']['cs_count'] == 0
        st.write(f"- **Trigger:** {'✅ YES' if trigger_2 else '❌ NO'}")
        if trigger_2:
            st.caption("(Both teams 0 CS in last 6)")
    
    with col3:
        st.write("**Filter 5: Regression**")
        st.write(f"- Wolfsburg Last6 BTTS: {metrics['team_b']['btts_percent']:.1f}%")
        st.write(f"- Wolfsburg Season BTTS: {team_b_data.overall['btts_percent']:.1f}%")
        trigger_5 = (metrics['team_b']['btts_percent'] > 70 and 
                     team_b_data.overall['btts_percent'] < 60)
        st.write(f"- **Trigger:** {'✅ YES' if trigger_5 else '❌ NO'}")
        if trigger_5:
            st.caption("(Last6 > 70% AND Season < 60%)")
    
    st.markdown("---")


def render_results_dashboard(result: Dict, team_a_data: TeamFormData, team_b_data: TeamFormData):
    """Display all analysis results WITH DATA VERIFICATION"""
    st.title("⚽ Betting Analytics Engine v3.1")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    
    # 1. SHOW ACTUAL DATA FROM CSV
    render_data_verification(team_a_data, team_b_data)
    
    # 2. SHOW FILTER CALCULATIONS
    render_filter_calculations(result['calculated_metrics'], team_a_data, team_b_data)
    
    # 3. Filters triggered with verification
    st.markdown("### 🔍 Detected Patterns")
    filters = result['filters_triggered']
    
    cols = st.columns(5)
    filter_config = {
        'under_15_alert': ('🔴', 'Under 1.5 Goals', filters.get('under_15_alert', False)),
        'btts_banker': ('🟢', 'BTTS Banker', filters.get('btts_banker', False)),
        'clean_sheet_alert': ('🔵', 'Clean Sheet', filters['clean_sheet_alert']['team'] is not None),
        'low_scoring_alert': ('🟣', 'Low-Scoring', filters.get('low_scoring_alert', False)),
        'regression_alert': ('🟡', 'Regression', filters['regression_alert']['team'] is not None)
    }
    
    for idx, (filter_key, (icon, label, triggered)) in enumerate(filter_config.items()):
        with cols[idx]:
            st.write(f"{icon} {label}")
            if triggered:
                st.success(f"✅ Triggered")
            else:
                st.info(f"❌ Not Triggered")
    
    # 4. Confidence level
    st.markdown("### 📈 Confidence Level")
    confidence = result['confidence']
    if confidence == 'high':
        st.success(f"Analysis Confidence: High 🟢")
    elif confidence == 'medium':
        st.warning(f"Analysis Confidence: Medium 🟡")
    else:
        st.info(f"Analysis Confidence: Low 🔵")
    
    # 5. Match analysis
    st.markdown("### 📝 Match Analysis")
    st.write(result['match_script']['match_narrative'])
    
    # 6. Warnings
    if result['betting_slip']['warnings']:
        st.markdown("### ⚠️ Warnings")
        for warning in result['betting_slip']['warnings']:
            st.warning(warning)
    
    # 7. Betting recommendations
    st.markdown("### 💰 Betting Recommendations")
    slip = result['betting_slip']
    
    if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] == 'NO_BET':
        st.info("No recommended bets - confidence too low")
    else:
        # Primary bets
        primary_bets = [b for b in slip['recommended_bets'] if b['priority'] == 'high']
        if primary_bets:
            st.markdown("#### 🎯 Primary Bets (High Confidence)")
            for bet in primary_bets:
                stake = slip['stake_suggestions'].get(bet['type'], 0)
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(f"**{bet['type'].replace('_', ' ').title()}**")
                with col2:
                    st.write(f"**{stake} units**")
                with col3:
                    st.caption(bet['reason'])
        
        # Secondary bets
        secondary_bets = [b for b in slip['recommended_bets'] if b['priority'] == 'medium']
        if secondary_bets:
            st.markdown("#### 📊 Secondary Bets (Medium Confidence)")
            for bet in secondary_bets:
                stake = slip['stake_suggestions'].get(bet['type'], 0)
                st.write(f"- **{bet['type'].replace('_', ' ').title()}** ({stake} units)")
    
    # 8. Predicted scores
    st.markdown("### 🎯 Predicted Score Range")
    scores = result['predicted_score_range']
    if scores:
        cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores[:5]):
            with cols[idx % 5]:
                st.info(f"**{score}**")
    
    # 9. Raw data toggle
    with st.expander("📋 View Raw Analysis Data"):
        st.json(result, expanded=False)


def render_sidebar() -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Render sidebar with controls"""
    st.sidebar.title("⚙️ Configuration")
    
    # League selection
    league_options = ["bundesliga", "premier_league", "la_liga", "serie_a", "ligue_1"]
    selected_league = st.sidebar.selectbox("Select League", league_options)
    
    if not selected_league:
        return None, None, None, True, None
    
    # Load data
    df = load_league_data(selected_league)
    if df is None:
        return None, None, None, True, None
    
    # Team selection
    teams = sorted(df['Team'].unique().tolist())
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, key="team_a")
    with col2:
        # Filter out selected team A
        remaining_teams = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", remaining_teams, key="team_b")
    
    # Venue selection
    st.sidebar.markdown("---")
    venue = st.sidebar.radio("Venue", ["Team A Home", "Team B Home", "Neutral"], horizontal=True)
    is_team_a_home = venue == "Team A Home"
    
    return selected_league, team_a, team_b, is_team_a_home, df


# ============================================================================
# SECTION 5: MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Betting Analytics Engine",
        page_icon="⚽",
        layout="wide"
    )
    
    st.sidebar.image("https://img.icons8.com/color/96/000000/football.png", width=80)
    
    # Sidebar
    selected_league, team_a, team_b, is_team_a_home, df = render_sidebar()
    
    if not all([selected_league, team_a, team_b, df is not None]):
        st.info("👈 Please select a league and teams to begin analysis")
        return
    
    # Main content
    st.title("📁 Data Selection")
    
    # Parse team data
    team_a_data = parse_team_from_csv(df, team_a)
    team_b_data = parse_team_from_csv(df, team_b)
    
    if not team_a_data or not team_b_data:
        st.error("❌ Could not load team data")
        return
    
    # Show what we're analyzing
    st.success(f"✅ Analyzing: **{team_a} vs {team_b}**")
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        isTeamAHome=is_team_a_home
    )
    
    # Run analysis
    with st.spinner("Running analysis..."):
        engine = BettingAnalyticsEngine()
        result = engine.analyze_match(match_context)
    
    # Display results WITH DATA VERIFICATION
    render_results_dashboard(result, team_a_data, team_b_data)
    
    # Footer
    st.markdown("---")
    st.caption("Betting Analytics Engine v3.1 • Uses exact 5-filter logic • No H2H data required")


# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
