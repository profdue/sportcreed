#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v4.0 - Production Ready
Optimized for your CSV structure with enhanced logic
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
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
# ENGINE CORE CLASSES
# ============================================================================

class DataLoader:
    """Load data from GitHub repository"""
    
    @staticmethod
    def load_from_github(league_name: str) -> Optional[pd.DataFrame]:
        """Load CSV data from GitHub repository"""
        try:
            # GitHub raw content URL
            base_url = "https://raw.githubusercontent.com/profdue/sportcreed/main/leagues"
            url = f"{base_url}/{league_name}.csv"
            
            response = requests.get(url)
            response.raise_for_status()
            
            df = pd.read_csv(io.StringIO(response.text))
            st.success(f"✅ Loaded {league_name} data from GitHub")
            return df
        except Exception as e:
            st.error(f"❌ Error loading {league_name}: {str(e)}")
            # Fallback to local file
            try:
                df = pd.read_csv(f"{league_name}.csv")
                st.success(f"✅ Loaded {league_name} data locally")
                return df
            except:
                st.error(f"❌ Could not load {league_name} data")
                return None

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
        
        # Season GPM
        a_season_gpm = team_a.overall['goals'] / max(team_a.overall['matches'], 1)
        b_season_gpm = team_b.overall['goals'] / max(team_b.overall['matches'], 1)
        
        # Scoring momentum
        a_momentum = (team_a.last6['goals']/6) / max(a_season_gpm, 0.1)
        b_momentum = (team_b.last6['goals']/6) / max(b_season_gpm, 0.1)
        
        metrics = {
            'team_a': {
                'gpm': team_a.last6['goals'] / 6,
                'cs_percent': (a_cs_num / 6) * 100,
                'cs_count': a_cs_num,
                'btts_percent': (a_btts_num / 6) * 100,
                'btts_count': a_btts_num,
                'win_percent': (team_a.last6['wins'] / 6) * 100,
                'loss_percent': (team_a.last6['losses'] / 6) * 100,
                'season_gpm': a_season_gpm,
                'scoring_momentum': a_momentum
            },
            'team_b': {
                'gpm': team_b.last6['goals'] / 6,
                'cs_percent': (b_cs_num / 6) * 100,
                'cs_count': b_cs_num,
                'btts_percent': (b_btts_num / 6) * 100,
                'btts_count': b_btts_num,
                'win_percent': (team_b.last6['wins'] / 6) * 100,
                'loss_percent': (team_b.last6['losses'] / 6) * 100,
                'season_gpm': b_season_gpm,
                'scoring_momentum': b_momentum
            },
            'averages': {
                'avg_gpm': (team_a.last6['goals']/6 + team_b.last6['goals']/6) / 2,
                'avg_cs_percent': ((a_cs_num/6)*100 + (b_cs_num/6)*100) / 2,
                'avg_btts_percent': ((a_btts_num/6)*100 + (b_btts_num/6)*100) / 2,
                'combined_cs_last12': a_cs_num + b_cs_num
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
    """Detect all 5 extreme filters with optimized logic"""
    
    def __init__(self):
        # Filter thresholds
        self.UNDER_15_GPM_THRESHOLD = 0.75
        self.CS_PERCENT_THRESHOLD = 20
        self.CS_PERCENT_STRONG = 50
        self.OPPONENT_GPM_WEAK = 1.0
        self.BTTS_PERCENT_LOW = 40
        self.CS_PERCENT_DECENT = 30
        self.GPM_MODERATE = 1.5
        self.RECENT_BTTS_INFLATION_MIN = 70
        self.SEASON_BTTS_REGRESSION_MAX = 60
        self.GPM_INFLATION_FACTOR = 1.5
        self.GPM_DEFLATION_FACTOR = 0.6
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """Detect which extreme filters are triggered"""
        filters = {
            'under_15_alert': False,
            'btts_banker': False,
            'btts_enhanced': False,
            'clean_sheet_alert': {'team': None, 'direction': None, 'strength': None},
            'low_scoring_alert': False,
            'low_scoring_type': None,
            'regression_alert': {'warnings': [], 'downgrade_factor': 1.0, 'upgrade_factor': 1.0}
        }
        
        # FILTER 1: Under 1.5 Goals Alert
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # FILTER 2: BTTS Banker Alert (ENHANCED)
        # Condition 1: Both teams have ≤1/6 CS in last 6 (changed from 0/6)
        if (metrics['team_a']['cs_count'] <= 1 and 
            metrics['team_b']['cs_count'] <= 1):
            filters['btts_banker'] = True
            
            # Check combined CS in last 12 matches
            if metrics['averages']['combined_cs_last12'] <= 2:
                filters['btts_enhanced'] = True
            
            # Enhancement: Add Over 2.5 if high scoring form
            if (metrics['team_a']['gpm'] > 1.8 or 
                metrics['team_b']['gpm'] > 1.8):
                filters['btts_enhanced'] = True
        
        # Condition 2: Both teams have season CS% < 20%
        elif (team_a.overall['cs_percent'] < self.CS_PERCENT_THRESHOLD and 
              team_b.overall['cs_percent'] < self.CS_PERCENT_THRESHOLD):
            filters['btts_banker'] = True
        
        # FILTER 3: Clean Sheet Alert (DUAL CHECK)
        # Check both Overall CS% > 50% AND/OR Last6 CS ≥ 3/6
        a_has_cs_strength = (team_a.overall['cs_percent'] > self.CS_PERCENT_STRONG or 
                            metrics['team_a']['cs_count'] >= 3)
        b_has_cs_strength = (team_b.overall['cs_percent'] > self.CS_PERCENT_STRONG or 
                            metrics['team_b']['cs_count'] >= 3)
        
        if a_has_cs_strength and metrics['team_b']['gpm'] < self.OPPONENT_GPM_WEAK:
            filters['clean_sheet_alert'] = {
                'team': 'A', 
                'direction': 'win_to_nil',
                'strength': 'strong' if team_a.overall['cs_percent'] > 50 or metrics['team_a']['cs_count'] >= 4 else 'moderate'
            }
        elif b_has_cs_strength and metrics['team_a']['gpm'] < self.OPPONENT_GPM_WEAK:
            filters['clean_sheet_alert'] = {
                'team': 'B', 
                'direction': 'win_to_nil',
                'strength': 'strong' if team_b.overall['cs_percent'] > 50 or metrics['team_b']['cs_count'] >= 4 else 'moderate'
            }
        
        # FILTER 4: Low-Scoring Alert (FIXED LOGIC)
        # Helper function to check if team is low-scoring
        def is_low_scoring_team(gpm, btts_percent, cs_count):
            return gpm < 1.5 and (btts_percent < self.BTTS_PERCENT_LOW or cs_count >= 2)
        
        # Condition 1: Both teams show low-scoring patterns
        if (is_low_scoring_team(metrics['team_a']['gpm'], metrics['team_a']['btts_percent'], metrics['team_a']['cs_count']) and
            is_low_scoring_team(metrics['team_b']['gpm'], metrics['team_b']['btts_percent'], metrics['team_b']['cs_count'])):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'both_low_scoring'
        
        # Condition 2: Defensive team vs Low-scoring opponent (WITH GPM CHECK)
        elif (metrics['team_a']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_a']['cs_count'] >= 2 and 
              metrics['team_b']['gpm'] < self.GPM_MODERATE and
              metrics['team_a']['gpm'] < self.GPM_MODERATE):  # ADDED: Check if defensive team is also low-scoring
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_weak_a'
        
        elif (metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_b']['cs_count'] >= 2 and 
              metrics['team_a']['gpm'] < self.GPM_MODERATE and
              metrics['team_b']['gpm'] < self.GPM_MODERATE):  # ADDED: Check if defensive team is also low-scoring
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_weak_b'
        
        # FILTER 5: Regression & Momentum Alert (ENHANCED)
        warnings = []
        downgrade_factors = []
        upgrade_factors = []
        
        # Check both teams
        for team_idx, (team_data, team_name) in enumerate([(team_a, 'A'), (team_b, 'B')]):
            team_metrics = metrics[f'team_{team_name.lower()}']
            
            # BTTS Regression
            if (team_metrics['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and 
                team_data.overall['btts_percent'] < self.SEASON_BTTS_REGRESSION_MAX):
                warnings.append(f"Team {team_name}: Recent BTTS {team_metrics['btts_percent']:.1f}% may regress to season {team_data.overall['btts_percent']:.1f}%")
                downgrade_factors.append(0.7)
            
            # GPM Inflation
            if team_metrics['gpm'] > (team_metrics['season_gpm'] * self.GPM_INFLATION_FACTOR):
                warnings.append(f"Team {team_name}: Recent scoring surge may be unsustainable ({team_metrics['gpm']:.2f} vs {team_metrics['season_gpm']:.2f} GPM)")
                downgrade_factors.append(0.8)
            
            # GPM Deflation (Scoring Collapse)
            elif team_metrics['gpm'] < (team_metrics['season_gpm'] * self.GPM_DEFLATION_FACTOR):
                warnings.append(f"Team {team_name}: Major scoring decline detected ({team_metrics['gpm']:.2f} vs {team_metrics['season_gpm']:.2f} GPM)")
                downgrade_factors.append(0.6)
            
            # Defensive Improvement
            if team_metrics['cs_percent'] > (team_data.overall['cs_percent'] * 1.5):
                warnings.append(f"Team {team_name}: Recent defensive improvement ({team_metrics['cs_percent']:.1f}% vs {team_data.overall['cs_percent']:.1f}% CS)")
                upgrade_factors.append(1.3)
        
        # Set regression alerts
        if warnings:
            filters['regression_alert']['warnings'] = warnings
            filters['regression_alert']['downgrade_factor'] = min(downgrade_factors) if downgrade_factors else 1.0
            filters['regression_alert']['upgrade_factor'] = max(upgrade_factors) if upgrade_factors else 1.0
        
        return filters

class CurrentFormAnalyzer:
    """Analyze current form to determine favorite with enhanced logic"""
    
    @staticmethod
    def analyze_form(team_a: TeamFormData, team_b: TeamFormData, metrics: Dict) -> Dict:
        """Analyze current form and determine favorite with momentum consideration"""
        analysis = {
            'favorite': None,
            'favorite_strength': None,  # 'very_strong', 'strong', 'slight', 'none'
            'goal_potential': None,  # 'very_high', 'high', 'medium', 'low'
            'btts_likelihood': None,  # 'very_high', 'high', 'medium', 'low'
            'scoring_potential': None  # Actual GPM value
        }
        
        # Determine favorite with margin consideration
        win_diff = team_a.last6['wins'] - team_b.last6['wins']
        
        if win_diff >= 3:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'very_strong'
        elif win_diff >= 2:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'strong'
        elif win_diff == 1:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'slight'
        elif win_diff <= -3:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'very_strong'
        elif win_diff <= -2:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'strong'
        elif win_diff == -1:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'slight'
        else:
            analysis['favorite_strength'] = 'balanced'
        
        # Goal potential using MAX of Recent or 70% of Season (accounts for scoring collapses)
        scoring_potential_a = max(metrics['team_a']['gpm'], metrics['team_a']['season_gpm'] * 0.7)
        scoring_potential_b = max(metrics['team_b']['gpm'], metrics['team_b']['season_gpm'] * 0.7)
        avg_scoring_potential = (scoring_potential_a + scoring_potential_b) / 2
        analysis['scoring_potential'] = avg_scoring_potential
        
        if avg_scoring_potential > 2.0:
            analysis['goal_potential'] = 'very_high'
        elif avg_scoring_potential > 1.6:
            analysis['goal_potential'] = 'high'
        elif avg_scoring_potential > 1.2:
            analysis['goal_potential'] = 'medium'
        else:
            analysis['goal_potential'] = 'low'
        
        # BTTS likelihood
        avg_btts = metrics['averages']['avg_btts_percent']
        
        if avg_btts > 75:
            analysis['btts_likelihood'] = 'very_high'
        elif avg_btts > 60:
            analysis['btts_likelihood'] = 'high'
        elif avg_btts > 45:
            analysis['btts_likelihood'] = 'medium'
        else:
            analysis['btts_likelihood'] = 'low'
            
        return analysis

class MatchScriptGenerator:
    """Generate match script and betting narrative with conflict resolution"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, form_analysis: Dict) -> Dict:
        """Generate complete match script with conflict resolution"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'predicted_score_range': [],
            'confidence_score': 50,  # 0-100 scale
            'confidence_level': 'low',  # low, medium, high
            'match_narrative': '',
            'warnings': [],
            'triggered_filter': None
        }
        
        # Start with filter priority (1 is highest)
        filter_priority = [
            ('under_15_alert', 1),
            ('btts_banker', 2),
            ('clean_sheet_alert', 3),
            ('low_scoring_alert', 4)
        ]
        
        triggered_filters = []
        for filter_key, priority in filter_priority:
            if filter_key == 'clean_sheet_alert':
                if filters[filter_key]['team'] is not None:
                    triggered_filters.append((filter_key, priority))
            elif filters.get(filter_key, False):
                triggered_filters.append((filter_key, priority))
        
        # Add regression as separate consideration
        regression_warnings = filters['regression_alert']['warnings']
        
        # CONFLICT RESOLUTION: Use highest priority filter
        if triggered_filters:
            triggered_filters.sort(key=lambda x: x[1])  # Sort by priority (lower = higher)
            primary_filter = triggered_filters[0][0]
            script['triggered_filter'] = primary_filter
            
            # Generate script based on primary filter
            if primary_filter == 'under_15_alert':
                script = self._generate_under_15_script(script, metrics)
            elif primary_filter == 'btts_banker':
                script = self._generate_btts_banker_script(script, metrics, filters)
            elif primary_filter == 'clean_sheet_alert':
                script = self._generate_clean_sheet_script(script, filters)
            elif primary_filter == 'low_scoring_alert':
                script = self._generate_low_scoring_script(script, metrics, filters)
            
            # Set confidence based on filter
            confidence_scores = {
                'under_15_alert': 95,
                'btts_banker': 85,
                'clean_sheet_alert': 80 if filters['clean_sheet_alert']['strength'] == 'strong' else 70,
                'low_scoring_alert': 80 if metrics['averages']['avg_gpm'] < 1.2 else 70
            }
            script['confidence_score'] = confidence_scores.get(primary_filter, 70)
        
        else:
            # No primary filters triggered - use form analysis
            script = self._generate_form_based_script(script, metrics, form_analysis)
            script['confidence_score'] = self._calculate_form_confidence(form_analysis)
        
        # Apply regression adjustments
        if regression_warnings:
            script['warnings'].extend(regression_warnings)
            downgrade = filters['regression_alert']['downgrade_factor']
            upgrade = filters['regression_alert']['upgrade_factor']
            script['confidence_score'] = int(script['confidence_score'] * downgrade * upgrade)
        
        # Convert score to level
        if script['confidence_score'] >= 80:
            script['confidence_level'] = 'high'
        elif script['confidence_score'] >= 65:
            script['confidence_level'] = 'medium'
        else:
            script['confidence_level'] = 'low'
        
        return script
    
    def _generate_under_15_script(self, script: Dict, metrics: Dict) -> Dict:
        """Generate script for Under 1.5 filter"""
        script['primary_bets'].append('under_15_goals')
        script['primary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        script['match_narrative'] = 'Both attacks completely broken - expect extremely low scoring'
        return script
    
    def _generate_btts_banker_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        """Generate script for BTTS Banker filter"""
        script['primary_bets'].append('btts_yes')
        script['secondary_bets'].append('over_15_goals')
        
        if filters.get('btts_enhanced', False):
            script['value_bets'].append('over_25_goals')
            script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2', '3-1', '1-3']
            script['match_narrative'] = 'Both defenses consistently leaky with high scoring potential'
        else:
            script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2']
            script['match_narrative'] = 'Both teams consistently concede - expect goals at both ends'
        
        return script
    
    def _generate_clean_sheet_script(self, script: Dict, filters: Dict) -> Dict:
        """Generate script for Clean Sheet filter"""
        team = filters['clean_sheet_alert']['team']
        team_name = self.team_a_name if team == 'A' else self.team_b_name
        
        if filters['clean_sheet_alert']['strength'] == 'strong':
            script['primary_bets'].append(f'{team_name.lower().replace(" ", "_")}_win_to_nil')
        else:
            script['primary_bets'].append(f'{team_name.lower().replace(" ", "_")}_to_keep_clean_sheet')
        
        script['secondary_bets'].append('under_25_goals')
        
        if team == 'A':
            script['predicted_score_range'] = ['1-0', '2-0', '0-0', '3-0']
        else:
            script['predicted_score_range'] = ['0-1', '0-2', '0-0', '0-3']
        
        script['match_narrative'] = f'{team_name} strong defense against weak attack'
        return script
    
    def _generate_low_scoring_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        """Generate script for Low-scoring filter"""
        script['primary_bets'].append('under_25_goals')
        script['secondary_bets'].append('btts_no')
        
        if filters['low_scoring_type'] == 'both_low_scoring':
            script['match_narrative'] = 'Both teams show strong low-scoring patterns'
            script['predicted_score_range'] = ['0-0', '1-0', '0-1', '1-1']
        else:
            team = 'A' if filters['low_scoring_type'] == 'defensive_vs_weak_a' else 'B'
            team_name = self.team_a_name if team == 'A' else self.team_b_name
            script['match_narrative'] = f'{team_name} defensive strength against low-scoring opponent'
            script['predicted_score_range'] = ['0-0', '1-0', '0-1', '1-1']
        
        return script
    
    def _generate_form_based_script(self, script: Dict, metrics: Dict, form_analysis: Dict) -> Dict:
        """Generate script based on form analysis when no filters trigger"""
        # Add favorite bets if strong favorite
        if form_analysis['favorite_strength'] in ['very_strong', 'strong']:
            favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win')
        
        # Goal market bets
        if form_analysis['goal_potential'] == 'very_high':
            script['secondary_bets'].append('over_25_goals')
            script['secondary_bets'].append('over_15_goals')
        elif form_analysis['goal_potential'] == 'high':
            script['secondary_bets'].append('over_15_goals')
        elif form_analysis['goal_potential'] == 'low':
            script['secondary_bets'].append('under_25_goals')
        
        # BTTS bets
        if form_analysis['btts_likelihood'] == 'very_high':
            script['secondary_bets'].append('btts_yes')
        elif form_analysis['btts_likelihood'] == 'low':
            script['secondary_bets'].append('btts_no')
        
        # Generate score predictions
        script['predicted_score_range'] = self._generate_form_scores(form_analysis)
        script['match_narrative'] = self._generate_form_narrative(form_analysis)
        
        return script
    
    def _calculate_form_confidence(self, form_analysis: Dict) -> int:
        """Calculate confidence score from form analysis"""
        confidence = 50
        
        # Adjust for favorite strength
        strength_boost = {
            'very_strong': 20,
            'strong': 15,
            'slight': 5,
            'balanced': 0
        }
        confidence += strength_boost.get(form_analysis['favorite_strength'], 0)
        
        # Adjust for goal potential clarity
        if form_analysis['goal_potential'] in ['very_high', 'low']:
            confidence += 10
        elif form_analysis['goal_potential'] in ['high', 'medium']:
            confidence += 5
        
        # Cap at 70 for form-based predictions
        return min(confidence, 70)
    
    def _generate_form_scores(self, form_analysis: Dict) -> List[str]:
        """Generate score predictions based on form"""
        if form_analysis['favorite']:
            if form_analysis['favorite'] == 'team_a':
                if form_analysis['btts_likelihood'] in ['very_high', 'high']:
                    return ['2-1', '1-0', '2-0', '3-1', '1-1']
                else:
                    return ['1-0', '2-0', '0-0', '2-1', '1-1']
            else:
                if form_analysis['btts_likelihood'] in ['very_high', 'high']:
                    return ['1-2', '0-1', '0-2', '1-3', '1-1']
                else:
                    return ['0-1', '0-2', '0-0', '1-2', '1-1']
        else:
            return ['1-1', '1-0', '0-1', '2-1', '1-2']
    
    def _generate_form_narrative(self, form_analysis: Dict) -> str:
        """Generate narrative based on form analysis"""
        narrative_parts = []
        
        if form_analysis['favorite_strength'] == 'very_strong':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has very strong recent form advantage.')
        elif form_analysis['favorite_strength'] == 'strong':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has strong recent form edge.')
        elif form_analysis['favorite_strength'] == 'slight':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has slight recent form edge.')
        
        if form_analysis['goal_potential'] == 'very_high':
            narrative_parts.append('Very high scoring potential.')
        elif form_analysis['goal_potential'] == 'high':
            narrative_parts.append('High scoring potential.')
        elif form_analysis['goal_potential'] == 'medium':
            narrative_parts.append('Moderate scoring potential.')
        else:
            narrative_parts.append('Low scoring potential.')
        
        if form_analysis['btts_likelihood'] == 'very_high':
            narrative_parts.append('Both teams very likely to score.')
        elif form_analysis['btts_likelihood'] == 'high':
            narrative_parts.append('Both teams likely to score.')
        elif form_analysis['btts_likelihood'] == 'medium':
            narrative_parts.append('Both teams may score.')
        else:
            narrative_parts.append('Clean sheet possible.')
        
        return ' '.join(narrative_parts)

class BettingSlipGenerator:
    """Generate final betting recommendations with stake suggestions"""
    
    def __init__(self):
        self.config = {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'total_units': 10,
            'min_confidence_score': 60
        }
    
    def generate_slip(self, script: Dict, team_a_name: str, team_b_name: str) -> Dict:
        """Generate optimized betting slip"""
        slip = {
            'recommended_bets': [],
            'stake_suggestions': {},
            'total_units': self.config['total_units'],
            'match_summary': script.get('match_narrative', ''),
            'warnings': script.get('warnings', []),
            'confidence': script.get('confidence_level', 'low'),
            'confidence_score': script.get('confidence_score', 50)
        }
        
        # Skip if confidence too low
        if script['confidence_score'] < self.config['min_confidence_score']:
            slip['recommended_bets'].append({
                'type': 'NO_BET',
                'priority': 'none',
                'reason': f'confidence_too_low ({script["confidence_score"]}/100)',
                'confidence': script['confidence_level']
            })
            return slip
        
        # Priority 1: Primary bets
        for bet in script['primary_bets']:
            slip['recommended_bets'].append({
                'type': bet,
                'priority': 'high',
                'reason': 'extreme_filter',
                'confidence': script['confidence_level']
            })
        
        # Priority 2: Secondary bets if space
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'medium',
                    'reason': 'form_analysis'
                })
        
        # Priority 3: Value bets
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['value_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'value_addition'
                })
        
        # Remove duplicates and limit
        unique_bets = []
        seen = set()
        for bet in slip['recommended_bets']:
            if bet['type'] not in seen:
                unique_bets.append(bet)
                seen.add(bet['type'])
        
        slip['recommended_bets'] = unique_bets[:self.config['max_bets_per_match']]
        
        # Generate stake suggestions based on confidence
        if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] != 'NO_BET':
            total_stake = slip['total_units']
            confidence_multiplier = script['confidence_score'] / 100
            
            for bet in slip['recommended_bets']:
                if bet['priority'] == 'high':
                    base_stake = total_stake * self.config['stake_distribution']['high']
                elif bet['priority'] == 'medium':
                    base_stake = total_stake * self.config['stake_distribution']['medium']
                else:
                    base_stake = total_stake * self.config['stake_distribution']['low']
                
                # Adjust by confidence
                adjusted_stake = base_stake * confidence_multiplier
                slip['stake_suggestions'][bet['type']] = round(adjusted_stake, 1)
        
        # Add score suggestions
        if script['predicted_score_range']:
            slip['score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip

# ============================================================================
# MAIN ENGINE CLASS
# ============================================================================

class BettingAnalyticsEngine:
    """Main orchestrator - complete pipeline"""
    
    def __init__(self):
        self.data_loader = DataLoader()
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
            match_context.teamB,
            metrics
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
            'confidence': script['confidence_level'],
            'confidence_score': script['confidence_score']
        }
        
        return result

# ============================================================================
# DATA PARSING FUNCTIONS
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

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def render_sidebar() -> Tuple[Optional[str], Optional[str], Optional[str], bool, Optional[pd.DataFrame]]:
    """Render sidebar with controls"""
    st.sidebar.title("⚙️ Configuration")
    
    # League selection
    league_options = [
        "bundesliga", "premier_league", "laliga", "serie_a", 
        "ligue_1", "eredivisie", "championship"
    ]
    selected_league = st.sidebar.selectbox("Select League", league_options)
    
    if not selected_league:
        return None, None, None, True, None
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_from_github(selected_league)
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

def render_results_dashboard(result: Dict):
    """Display all analysis results"""
    st.title("⚽ Betting Analytics Engine v4.0")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    
    # 1. Confidence level with score
    st.markdown("### 📈 Confidence Level")
    confidence = result['confidence']
    score = result['confidence_score']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if confidence == 'high':
            st.success(f"Analysis Confidence: High 🟢 ({score}/100)")
        elif confidence == 'medium':
            st.warning(f"Analysis Confidence: Medium 🟡 ({score}/100)")
        else:
            st.info(f"Analysis Confidence: Low 🔵 ({score}/100)")
    
    with col2:
        st.metric("Confidence Score", f"{score}/100")
    
    # 2. Filters triggered
    st.markdown("### 🔍 Detected Patterns")
    filters = result['filters_triggered']
    
    cols = st.columns(5)
    filter_config = {
        'under_15_alert': ('🔴', 'Under 1.5 Goals', filters.get('under_15_alert', False)),
        'btts_banker': ('🟢', 'BTTS Banker', filters.get('btts_banker', False)),
        'clean_sheet_alert': ('🔵', 'Clean Sheet', filters['clean_sheet_alert']['team'] is not None),
        'low_scoring_alert': ('🟣', 'Low-Scoring', filters.get('low_scoring_alert', False)),
        'regression_alert': ('🟡', 'Regression', len(filters['regression_alert']['warnings']) > 0)
    }
    
    for idx, (filter_key, (icon, label, triggered)) in enumerate(filter_config.items()):
        with cols[idx]:
            st.write(f"{icon} {label}")
            if triggered:
                st.success(f"✅ Triggered")
            else:
                st.info(f"❌ Not Triggered")
    
    # 3. Match analysis
    st.markdown("### 📝 Match Analysis")
    st.write(result['match_script']['match_narrative'])
    
    # 4. Warnings
    if result['betting_slip']['warnings']:
        st.markdown("### ⚠️ Warnings")
        for warning in result['betting_slip']['warnings']:
            st.warning(warning)
    
    # 5. Betting recommendations
    st.markdown("### 💰 Betting Recommendations")
    slip = result['betting_slip']
    
    if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] == 'NO_BET':
        st.info("No recommended bets - confidence too low")
    else:
        # Display bets with stakes
        for bet in slip['recommended_bets']:
            stake = slip['stake_suggestions'].get(bet['type'], 0)
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.write(f"**{bet['type'].replace('_', ' ').title()}**")
            with col2:
                if stake > 0:
                    st.write(f"**{stake} units**")
            with col3:
                if bet['priority'] == 'high':
                    st.success(f"Primary ({bet['reason']})")
                elif bet['priority'] == 'medium':
                    st.warning(f"Secondary ({bet['reason']})")
                else:
                    st.info(f"Value ({bet['reason']})")
    
    # 6. Predicted scores
    st.markdown("### 🎯 Predicted Score Range")
    scores = result['predicted_score_range']
    if scores:
        cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores[:5]):
            with cols[idx % 5]:
                st.info(f"**{score}**")
    
    # 7. Data verification toggle
    with st.expander("📊 View Detailed Data & Metrics"):
        st.markdown("#### Calculated Metrics")
        st.json(result['calculated_metrics'], expanded=False)
        
        st.markdown("#### Form Analysis")
        st.json(result['form_analysis'], expanded=False)

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Betting Analytics Engine v4.0",
        page_icon="⚽",
        layout="wide"
    )
    
    st.sidebar.image("https://img.icons8.com/color/96/000000/football.png", width=80)
    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.markdown("[GitHub Repository](https://github.com/profdue/sportcreed)")
    
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
    st.info(f"📍 Venue: {'Team A Home' if is_team_a_home else 'Team B Home' if not is_team_a_home else 'Neutral'}")
    
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
    
    # Display results
    render_results_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("Betting Analytics Engine v4.0 • Uses optimized 5-filter logic • GitHub Data Source")
    st.caption(f"League: {selected_league} • Teams: {team_a} vs {team_b}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
