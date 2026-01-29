#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v4.3 - RISK-AWARE CLEAN SHEET FIX
Fixed the Clean Sheet filter to avoid risky bets on high-scoring favorites
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
# ENGINE CORE CLASSES WITH RISK-AWARE CLEAN SHEET FIX
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
                'scoring_momentum': a_momentum,
                'btts_rate': a_btts_num / 6 if a_btts_den > 0 else 0
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
                'scoring_momentum': b_momentum,
                'btts_rate': b_btts_num / 6 if b_btts_den > 0 else 0
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

class RiskAwareExtremeFilterDetector:
    """Detect all 5 extreme filters with RISK-AWARE Clean Sheet fix"""
    
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
        
        # PROVEN: Warning patterns from 59 matches
        self.DEFENSIVE_IMPROVEMENT_THRESHOLD = 1.5
        self.SCORING_DECLINE_THRESHOLD = 0.6
        
        # FIXED: Stricter Low-scoring thresholds
        self.LOW_SCORING_GPM_MAX = 1.0  # WAS 1.5 - NOW STRICTER
        self.LOW_SCORING_BTTS_MAX = 50  # Combined BTTS% maximum
        self.LOW_SCORING_INDIVIDUAL_GPM_MAX = 1.2  # No single team > 1.2 GPM
        
        # v4.3: RISK-AWARE Clean Sheet thresholds
        self.HIGH_SCORING_FAVORITE_GPM = 2.0  # Teams scoring >2.0 GPM are risky for clean sheets
        self.OPPONENT_BTTS_RISK_THRESHOLD = 0.33  # Opponent scoring in >33.3% of matches
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData, league: str) -> Dict:
        """Detect which extreme filters are triggered with RISK-AWARE Clean Sheet"""
        filters = {
            'under_15_alert': False,
            'btts_banker': False,
            'btts_enhanced': False,
            'clean_sheet_alert': {'team': None, 'direction': None, 'strength': None, 'risk_level': 'low'},
            'low_scoring_alert': False,
            'low_scoring_type': None,
            'regression_alert': {'warnings': [], 'defensive_improvements': [], 'scoring_declines': []}
        }
        
        # FILTER 1: Under 1.5 Goals Alert (100% success - keep)
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # FILTER 2: BTTS Banker Alert with league-specific thresholds
        filters = self._detect_btts_banker(filters, metrics, league)
        
        # FILTER 3: Clean Sheet Alert - v4.3 RISK-AWARE version
        filters = self._detect_clean_sheet_risk_aware(filters, metrics, team_a, team_b)
        
        # FILTER 4: Low-Scoring Alert - FIXED VERSION
        filters = self._detect_low_scoring_fixed(filters, metrics)
        
        # FILTER 5: Regression & Momentum Alert
        filters = self._detect_regression(filters, metrics, team_a, team_b)
        
        return filters
    
    def _detect_btts_banker(self, filters: Dict, metrics: Dict, league: str) -> Dict:
        """Detect BTTS Banker with league-specific thresholds"""
        team_a_cs = metrics['team_a']['cs_count']
        team_b_cs = metrics['team_b']['cs_count']
        combined_cs = metrics['averages']['combined_cs_last12']
        
        if league == 'serie_a':
            # Serie A: Very strict - NO clean sheets at all
            if team_a_cs == 0 and team_b_cs == 0:
                filters['btts_banker'] = True
                filters['btts_enhanced'] = True
        else:
            # Other leagues: Moderate - ≤1/6 CS each AND ≤1 CS combined
            if (team_a_cs <= 1 and team_b_cs <= 1 and combined_cs <= 1):
                filters['btts_banker'] = True
                if team_a_cs == 0 and team_b_cs == 0:
                    filters['btts_enhanced'] = True
        
        return filters
    
    def _detect_clean_sheet_risk_aware(self, filters: Dict, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """v4.3: Detect Clean Sheet opportunities with RISK ASSESSMENT"""
        # Check Team A clean sheet opportunity
        a_has_cs_strength = (team_a.overall['cs_percent'] > self.CS_PERCENT_STRONG or 
                            metrics['team_a']['cs_count'] >= 2)
        
        # Check Team B clean sheet opportunity
        b_has_cs_strength = (team_b.overall['cs_percent'] > self.CS_PERCENT_STRONG or 
                            metrics['team_b']['cs_count'] >= 2)
        
        # Team A clean sheet opportunity
        if a_has_cs_strength and metrics['team_b']['gpm'] < self.OPPONENT_GPM_WEAK:
            risk_level = self._assess_clean_sheet_risk(
                metrics['team_a']['gpm'],  # Team A's scoring rate
                metrics['team_b']['btts_rate'],  # Team B's scoring frequency
                metrics['team_a']['btts_rate']  # Team A's conceding frequency
            )
            
            filters['clean_sheet_alert'] = {
                'team': 'A', 
                'direction': 'win_to_nil',
                'strength': 'strong' if team_a.overall['cs_percent'] > 50 or metrics['team_a']['cs_count'] >= 3 else 'moderate',
                'risk_level': risk_level
            }
        
        # Team B clean sheet opportunity (only if Team A not already triggered)
        elif b_has_cs_strength and metrics['team_a']['gpm'] < self.OPPONENT_GPM_WEAK and filters['clean_sheet_alert']['team'] is None:
            risk_level = self._assess_clean_sheet_risk(
                metrics['team_b']['gpm'],  # Team B's scoring rate
                metrics['team_a']['btts_rate'],  # Team A's scoring frequency
                metrics['team_b']['btts_rate']  # Team B's conceding frequency
            )
            
            filters['clean_sheet_alert'] = {
                'team': 'B', 
                'direction': 'win_to_nil',
                'strength': 'strong' if team_b.overall['cs_percent'] > 50 or metrics['team_b']['cs_count'] >= 3 else 'moderate',
                'risk_level': risk_level
            }
        
        return filters
    
    def _assess_clean_sheet_risk(self, favorite_gpm: float, opponent_btts_rate: float, favorite_btts_rate: float) -> str:
        """v4.3: Assess risk level for clean sheet bets"""
        # HIGH RISK: Favorite scores a lot AND opponent still scores often
        if (favorite_gpm > self.HIGH_SCORING_FAVORITE_GPM and 
            opponent_btts_rate > self.OPPONENT_BTTS_RISK_THRESHOLD):
            return 'high'
        
        # MEDIUM RISK: Favorite scores a lot BUT opponent rarely scores
        elif favorite_gpm > self.HIGH_SCORING_FAVORITE_GPM:
            return 'medium'
        
        # LOW RISK: Favorite doesn't score too much
        else:
            return 'low'
    
    def _detect_low_scoring_fixed(self, filters: Dict, metrics: Dict) -> Dict:
        """FIXED: Detect TRUE low-scoring patterns (much stricter)"""
        team_a_gpm = metrics['team_a']['gpm']
        team_b_gpm = metrics['team_b']['gpm']
        team_a_btts = metrics['team_a']['btts_percent']
        team_b_btts = metrics['team_b']['btts_percent']
        team_a_cs = metrics['team_a']['cs_count']
        team_b_cs = metrics['team_b']['cs_count']
        avg_btts = metrics['averages']['avg_btts_percent']
        
        # CRITICAL CHECK 1: No single team can have GPM > 1.2
        if team_a_gpm > self.LOW_SCORING_INDIVIDUAL_GPM_MAX or team_b_gpm > self.LOW_SCORING_INDIVIDUAL_GPM_MAX:
            return filters  # NOT low-scoring
        
        # CRITICAL CHECK 2: Combined BTTS% must be < 50%
        if avg_btts > self.LOW_SCORING_BTTS_MAX:
            return filters  # NOT low-scoring
        
        # Condition 1: Both teams TRULY low-scoring (GPM < 1.0)
        if (team_a_gpm < self.LOW_SCORING_GPM_MAX and 
            team_b_gpm < self.LOW_SCORING_GPM_MAX and
            team_a_btts < self.BTTS_PERCENT_LOW and 
            team_b_btts < self.BTTS_PERCENT_LOW):
            
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'both_truly_low_scoring'
            return filters
        
        # Condition 2: Defensive team (good CS, low BTTS) vs VERY weak attack (GPM < 1.0)
        if (team_a_btts < self.BTTS_PERCENT_LOW and 
            team_a_cs >= 2 and 
            team_b_gpm < self.LOW_SCORING_GPM_MAX and  # WAS 1.5, NOW 1.0
            team_a_gpm < self.LOW_SCORING_GPM_MAX):    # WAS 1.5, NOW 1.0
            
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_very_weak_a'
            return filters
        
        if (team_b_btts < self.BTTS_PERCENT_LOW and 
            team_b_cs >= 2 and 
            team_a_gpm < self.LOW_SCORING_GPM_MAX and  # WAS 1.5, NOW 1.0
            team_b_gpm < self.LOW_SCORING_GPM_MAX):    # WAS 1.5, NOW 1.0
            
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_very_weak_b'
            return filters
        
        return filters
    
    def _detect_regression(self, filters: Dict, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData) -> Dict:
        """Detect regression and momentum patterns"""
        warnings = []
        defensive_improvements = []
        scoring_declines = []
        
        for team_idx, (team_data, team_name) in enumerate([(team_a, 'A'), (team_b, 'B')]):
            team_metrics = metrics[f'team_{team_name.lower()}']
            
            # Defensive improvement detection
            if (team_metrics['cs_percent'] > (team_data.overall['cs_percent'] * self.DEFENSIVE_IMPROVEMENT_THRESHOLD) and
                team_data.overall['cs_percent'] > 10):
                warning = f"Team {team_name}: Recent defensive improvement ({team_metrics['cs_percent']:.1f}% vs {team_data.overall['cs_percent']:.1f}% CS)"
                warnings.append(warning)
                defensive_improvements.append(team_name)
            
            # Scoring decline detection
            if (team_metrics['gpm'] < (team_metrics['season_gpm'] * self.SCORING_DECLINE_THRESHOLD) and
                team_metrics['season_gpm'] > 0.8):
                warning = f"Team {team_name}: Major scoring decline detected ({team_metrics['gpm']:.2f} vs {team_metrics['season_gpm']:.2f} GPM)"
                warnings.append(warning)
                scoring_declines.append(team_name)
            
            # BTTS Regression
            if (team_metrics['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and 
                team_data.overall['btts_percent'] < self.SEASON_BTTS_REGRESSION_MAX):
                warnings.append(f"Team {team_name}: Recent BTTS {team_metrics['btts_percent']:.1f}% may regress to season {team_data.overall['btts_percent']:.1f}%")
            
            # GPM Inflation
            if team_metrics['gpm'] > (team_metrics['season_gpm'] * self.GPM_INFLATION_FACTOR):
                warnings.append(f"Team {team_name}: Recent scoring surge may be unsustainable ({team_metrics['gpm']:.2f} vs {team_metrics['season_gpm']:.2f} GPM)")
        
        if warnings:
            filters['regression_alert']['warnings'] = warnings
            filters['regression_alert']['defensive_improvements'] = defensive_improvements
            filters['regression_alert']['scoring_declines'] = scoring_declines
        
        return filters

class CurrentFormAnalyzer:
    """Analyze current form with PROVEN improvements"""
    
    @staticmethod
    def analyze_form(team_a: TeamFormData, team_b: TeamFormData, metrics: Dict, filters: Dict, is_team_a_home: bool) -> Dict:
        """Analyze current form with PROVEN warning-based adjustments"""
        analysis = {
            'favorite': None,
            'favorite_strength': None,
            'goal_potential': None,
            'btts_likelihood': None,
            'scoring_potential': None,
            'adjusted_confidence': 0
        }
        
        # Determine favorite with PROVEN improvements
        win_diff = team_a.last6['wins'] - team_b.last6['wins']
        
        if win_diff >= 3:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'very_strong'
            analysis['adjusted_confidence'] = 20
        elif win_diff >= 2 and is_team_a_home:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'strong'
            analysis['adjusted_confidence'] = 15
        elif win_diff <= -3:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'very_strong'
            analysis['adjusted_confidence'] = 20
        elif win_diff <= -2 and not is_team_a_home:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'strong'
            analysis['adjusted_confidence'] = 15
        elif win_diff == 1 and is_team_a_home:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'slight'
            analysis['adjusted_confidence'] = 5
        elif win_diff == -1 and not is_team_a_home:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'slight'
            analysis['adjusted_confidence'] = 5
        else:
            analysis['favorite_strength'] = 'balanced'
        
        # Adjust for scoring decline warnings
        scoring_declines = filters.get('regression_alert', {}).get('scoring_declines', [])
        if analysis['favorite']:
            fav_team = 'A' if analysis['favorite'] == 'team_a' else 'B'
            if fav_team in scoring_declines:
                analysis['adjusted_confidence'] -= 15
        
        # Goal potential with warning adjustments
        scoring_potential_a = max(metrics['team_a']['gpm'], metrics['team_a']['season_gpm'] * 0.7)
        scoring_potential_b = max(metrics['team_b']['gpm'], metrics['team_b']['season_gpm'] * 0.7)
        
        if 'A' in scoring_declines:
            scoring_potential_a *= 0.7
        if 'B' in scoring_declines:
            scoring_potential_b *= 0.7
        
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
        
        # BTTS likelihood with defensive improvement adjustments
        avg_btts = metrics['averages']['avg_btts_percent']
        
        defensive_improvements = filters.get('regression_alert', {}).get('defensive_improvements', [])
        if defensive_improvements:
            avg_btts *= 0.8
        
        if avg_btts > 75:
            analysis['btts_likelihood'] = 'very_high'
        elif avg_btts > 60:
            analysis['btts_likelihood'] = 'high'
        elif avg_btts > 45:
            analysis['btts_likelihood'] = 'medium'
        else:
            analysis['btts_likelihood'] = 'low'
            
        return analysis

class RiskAwareScriptGenerator:
    """Generate match script with RISK-AWARE Clean Sheet handling"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, form_analysis: Dict, is_team_a_home: bool) -> Dict:
        """Generate complete match script with RISK-AWARE Clean Sheet"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'predicted_score_range': [],
            'confidence_score': 50,
            'confidence_level': 'low',
            'match_narrative': '',
            'warnings': [],
            'triggered_filter': None,
            'warning_adjustments': {'applied': False, 'changes': []},
            'risk_assessment': {'clean_sheet_risk': None, 'adjusted_bets': False}
        }
        
        # Apply warning adjustments
        script = self._apply_warning_adjustments(script, filters, form_analysis, is_team_a_home)
        
        # Start with filter priority
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
        
        regression_warnings = filters['regression_alert']['warnings']
        script['warnings'] = regression_warnings
        
        if triggered_filters:
            triggered_filters.sort(key=lambda x: x[1])
            primary_filter = triggered_filters[0][0]
            script['triggered_filter'] = primary_filter
            
            if primary_filter == 'under_15_alert':
                script = self._generate_under_15_script(script, metrics)
            elif primary_filter == 'btts_banker':
                script = self._generate_btts_banker_script(script, metrics, filters)
            elif primary_filter == 'clean_sheet_alert':
                script = self._generate_clean_sheet_script_risk_aware(script, metrics, filters)
            elif primary_filter == 'low_scoring_alert':
                script = self._generate_low_scoring_script_fixed(script, metrics, filters)
            
            confidence_scores = {
                'under_15_alert': 95,
                'btts_banker': 85,
                'clean_sheet_alert': self._get_clean_sheet_confidence(filters),
                'low_scoring_alert': 90
            }
            script['confidence_score'] = confidence_scores.get(primary_filter, 70)
        
        else:
            script = self._generate_form_based_script(script, metrics, form_analysis)
            script['confidence_score'] = self._calculate_form_confidence(form_analysis)
        
        script = self._apply_final_warning_adjustments(script, filters)
        
        if script['confidence_score'] >= 80:
            script['confidence_level'] = 'high'
        elif script['confidence_score'] >= 65:
            script['confidence_level'] = 'medium'
        else:
            script['confidence_level'] = 'low'
        
        return script
    
    def _get_clean_sheet_confidence(self, filters: Dict) -> int:
        """Get confidence score based on clean sheet risk level"""
        base_confidence = 85 if filters['clean_sheet_alert']['strength'] == 'strong' else 75
        risk_level = filters['clean_sheet_alert'].get('risk_level', 'low')
        
        # Adjust confidence based on risk
        if risk_level == 'high':
            return max(60, base_confidence - 25)  # Significant reduction for high risk
        elif risk_level == 'medium':
            return max(70, base_confidence - 15)  # Moderate reduction
        else:
            return base_confidence  # No reduction for low risk
    
    def _generate_clean_sheet_script_risk_aware(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        """v4.3: Generate script for Clean Sheet with RISK AWARENESS"""
        team = filters['clean_sheet_alert']['team']
        team_name = self.team_a_name if team == 'A' else self.team_b_name
        risk_level = filters['clean_sheet_alert'].get('risk_level', 'low')
        strength = filters['clean_sheet_alert']['strength']
        
        # Clean sheet bet based on strength
        if strength == 'strong':
            clean_sheet_bet = f'{team_name.lower().replace(" ", "_")}_win_to_nil'
        else:
            clean_sheet_bet = f'{team_name.lower().replace(" ", "_")}_to_keep_clean_sheet'
        
        # v4.3 RISK-AWARE DECISION
        if risk_level == 'high':
            # HIGH RISK: Favorite scores a lot AND opponent scores often
            # Make BTTS/Over primary, clean sheet secondary
            script['primary_bets'].append('btts_yes')
            script['primary_bets'].append('over_25_goals')
            script['secondary_bets'].append(clean_sheet_bet)
            script['risk_assessment']['clean_sheet_risk'] = 'high'
            script['risk_assessment']['adjusted_bets'] = True
            script['match_narrative'] = f'{team_name} strong defense but HIGH RISK clean sheet - both teams likely to score'
            
        elif risk_level == 'medium':
            # MEDIUM RISK: Favorite scores a lot BUT opponent rarely scores
            # Keep clean sheet primary but reduce confidence
            script['primary_bets'].append(clean_sheet_bet)
            script['secondary_bets'].append('btts_no')
            script['risk_assessment']['clean_sheet_risk'] = 'medium'
            script['match_narrative'] = f'{team_name} strong defense against weak attack (moderate risk)'
            
        else:
            # LOW RISK: Favorite doesn't score too much
            # Normal clean sheet bet
            script['primary_bets'].append(clean_sheet_bet)
            script['secondary_bets'].append('under_25_goals')
            script['risk_assessment']['clean_sheet_risk'] = 'low'
            script['match_narrative'] = f'{team_name} strong defense against weak attack'
        
        # Add score predictions based on team
        if team == 'A':
            script['predicted_score_range'] = ['1-0', '2-0', '0-0', '3-0']
        else:
            script['predicted_score_range'] = ['0-1', '0-2', '0-0', '0-3']
        
        return script
    
    def _generate_low_scoring_script_fixed(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        """Generate script for FIXED Low-scoring filter"""
        script['primary_bets'].append('under_25_goals')
        script['secondary_bets'].append('btts_no')
        
        if filters['low_scoring_type'] == 'both_truly_low_scoring':
            script['match_narrative'] = 'Both teams show EXTREME low-scoring patterns'
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        else:
            team = 'A' if filters['low_scoring_type'] == 'defensive_vs_very_weak_a' else 'B'
            team_name = self.team_a_name if team == 'A' else self.team_b_name
            script['match_narrative'] = f'{team_name} defensive strength against VERY weak attack'
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        
        return script
    
    def _apply_warning_adjustments(self, script: Dict, filters: Dict, form_analysis: Dict, is_team_a_home: bool) -> Dict:
        """Apply warning-based pattern adjustments"""
        adjustments = {'applied': False, 'changes': []}
        
        defensive_improvements = filters.get('regression_alert', {}).get('defensive_improvements', [])
        scoring_declines = filters.get('regression_alert', {}).get('scoring_declines', [])
        
        home_team = 'A' if is_team_a_home else 'B'
        if home_team in defensive_improvements:
            adjustments['applied'] = True
            adjustments['changes'].append('home_defensive_improvement')
        
        if scoring_declines:
            adjustments['applied'] = True
            adjustments['changes'].append('scoring_decline_present')
        
        if form_analysis['favorite']:
            fav_team = 'A' if form_analysis['favorite'] == 'team_a' else 'B'
            if fav_team in scoring_declines:
                adjustments['applied'] = True
                adjustments['changes'].append('favorite_scoring_decline')
        
        script['warning_adjustments'] = adjustments
        return script
    
    def _apply_final_warning_adjustments(self, script: Dict, filters: Dict) -> Dict:
        """Apply final confidence adjustments"""
        original_confidence = script['confidence_score']
        warnings = filters.get('regression_alert', {}).get('warnings', [])
        
        if len(warnings) >= 3:
            script['confidence_score'] = max(40, script['confidence_score'] - 20)
        
        btts_regression_warnings = [w for w in warnings if 'BTTS may regress' in w]
        if btts_regression_warnings and any('btts' in bet.lower() for bet in script['primary_bets'] + script['secondary_bets']):
            script['confidence_score'] = max(50, script['confidence_score'] - 15)
        
        defensive_warnings = [w for w in warnings if 'defensive improvement' in w]
        if defensive_warnings and any('clean' in bet.lower() or 'nil' in bet.lower() for bet in script['primary_bets'] + script['secondary_bets']):
            script['confidence_score'] = min(100, script['confidence_score'] + 10)
        
        if original_confidence != script['confidence_score']:
            script['warning_adjustments']['changes'].append(f'confidence_adjusted_{original_confidence}→{script["confidence_score"]}')
        
        return script
    
    def _generate_under_15_script(self, script: Dict, metrics: Dict) -> Dict:
        script['primary_bets'].append('under_15_goals')
        script['primary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        script['match_narrative'] = 'Both attacks completely broken - expect extremely low scoring'
        return script
    
    def _generate_btts_banker_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        defensive_improvements = filters.get('regression_alert', {}).get('defensive_improvements', [])
        
        if defensive_improvements:
            script['match_narrative'] = 'Both defenses leaky but recent defensive improvements noted'
        else:
            script['match_narrative'] = 'Both teams consistently concede - expect goals at both ends'
        
        script['primary_bets'].append('btts_yes')
        script['secondary_bets'].append('over_15_goals')
        
        if filters.get('btts_enhanced', False):
            script['value_bets'].append('over_25_goals')
            script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2', '3-1', '1-3']
        else:
            script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2']
        
        return script
    
    def _generate_form_based_script(self, script: Dict, metrics: Dict, form_analysis: Dict) -> Dict:
        if form_analysis['favorite_strength'] in ['very_strong', 'strong']:
            favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win')
        
        if form_analysis['goal_potential'] == 'very_high':
            script['secondary_bets'].append('over_25_goals')
            script['secondary_bets'].append('over_15_goals')
        elif form_analysis['goal_potential'] == 'high':
            script['secondary_bets'].append('over_15_goals')
        elif form_analysis['goal_potential'] == 'low':
            script['secondary_bets'].append('under_25_goals')
        
        if form_analysis['btts_likelihood'] == 'very_high':
            script['secondary_bets'].append('btts_yes')
        elif form_analysis['btts_likelihood'] == 'low':
            script['secondary_bets'].append('btts_no')
        
        script['predicted_score_range'] = self._generate_form_scores(form_analysis)
        script['match_narrative'] = self._generate_form_narrative(form_analysis)
        
        return script
    
    def _calculate_form_confidence(self, form_analysis: Dict) -> int:
        confidence = 50
        confidence += form_analysis.get('adjusted_confidence', 0)
        
        if form_analysis['goal_potential'] in ['very_high', 'low']:
            confidence += 10
        elif form_analysis['goal_potential'] in ['high', 'medium']:
            confidence += 5
        
        return min(confidence, 70)
    
    def _generate_form_scores(self, form_analysis: Dict) -> List[str]:
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
    """Generate final betting recommendations"""
    
    def __init__(self):
        self.config = {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'total_units': 10,
            'min_confidence_score': 60
        }
    
    def generate_slip(self, script: Dict, team_a_name: str, team_b_name: str) -> Dict:
        slip = {
            'recommended_bets': [],
            'stake_suggestions': {},
            'total_units': self.config['total_units'],
            'match_summary': script.get('match_narrative', ''),
            'warnings': script.get('warnings', []),
            'warning_adjustments': script.get('warning_adjustments', {}),
            'confidence': script.get('confidence_level', 'low'),
            'confidence_score': script.get('confidence_score', 50),
            'risk_assessment': script.get('risk_assessment', {})
        }
        
        if script['confidence_score'] < self.config['min_confidence_score']:
            slip['recommended_bets'].append({
                'type': 'NO_BET',
                'priority': 'none',
                'reason': f'confidence_too_low ({script["confidence_score"]}/100)',
                'confidence': script['confidence_level']
            })
            return slip
        
        for bet in script['primary_bets']:
            slip['recommended_bets'].append({
                'type': bet,
                'priority': 'high',
                'reason': 'extreme_filter',
                'confidence': script['confidence_level']
            })
        
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'medium',
                    'reason': 'form_analysis'
                })
        
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['value_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'value_addition'
                })
        
        unique_bets = []
        seen = set()
        for bet in slip['recommended_bets']:
            if bet['type'] not in seen:
                unique_bets.append(bet)
                seen.add(bet['type'])
        
        slip['recommended_bets'] = unique_bets[:self.config['max_bets_per_match']]
        
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
                
                adjusted_stake = base_stake * confidence_multiplier
                
                warning_adj = script.get('warning_adjustments', {})
                if warning_adj.get('applied', False):
                    if 'home_defensive_improvement' in warning_adj.get('changes', []):
                        if 'clean' in bet['type'].lower() or 'nil' in bet['type'].lower():
                            adjusted_stake *= 1.3
                    if 'favorite_scoring_decline' in warning_adj.get('changes', []):
                        if 'win' in bet['type'].lower():
                            adjusted_stake *= 0.7
                
                # v4.3: Additional risk adjustment for high-risk clean sheets
                risk_assessment = script.get('risk_assessment', {})
                if risk_assessment.get('clean_sheet_risk') == 'high' and 'clean' in bet['type'].lower():
                    adjusted_stake *= 0.5  # Halve stake for high-risk clean sheets
                
                slip['stake_suggestions'][bet['type']] = round(adjusted_stake, 1)
        
        if script['predicted_score_range']:
            slip['score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip

# ============================================================================
# MAIN ENGINE CLASS
# ============================================================================

class BettingAnalyticsEngine:
    """Main orchestrator with RISK-AWARE Clean Sheet fix"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = RiskAwareExtremeFilterDetector()
        self.form_analyzer = CurrentFormAnalyzer()
        self.script_generator = None
        self.slip_generator = BettingSlipGenerator()
    
    def analyze_match(self, match_context: MatchContext, league: str) -> Dict:
        self.script_generator = RiskAwareScriptGenerator(
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
        
        form_analysis = self.form_analyzer.analyze_form(
            match_context.teamA, 
            match_context.teamB,
            metrics,
            filters,
            match_context.isTeamAHome
        )
        
        script = self.script_generator.generate_script(
            metrics, filters, form_analysis, match_context.isTeamAHome
        )
        
        slip = self.slip_generator.generate_slip(
            script,
            match_context.teamA.teamName,
            match_context.teamB.teamName
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
            'form_analysis': form_analysis,
            'match_script': script,
            'betting_slip': slip,
            'predicted_score_range': script['predicted_score_range'],
            'confidence': script['confidence_level'],
            'confidence_score': script['confidence_score'],
            'warning_adjustments': script.get('warning_adjustments', {}),
            'risk_assessment': script.get('risk_assessment', {})
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
        remaining_teams = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", remaining_teams, key="team_b")
    
    # Venue selection
    st.sidebar.markdown("---")
    venue = st.sidebar.radio("Venue", ["Team A Home", "Team B Home", "Neutral"], horizontal=True)
    is_team_a_home = venue == "Team A Home"
    
    return selected_league, team_a, team_b, is_team_a_home, df

def render_results_dashboard(result: Dict):
    """Display all analysis results"""
    st.title("⚽ Betting Analytics Engine v4.3 - RISK-AWARE Clean Sheet")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    st.caption(f"League: {result['match_info']['league'].replace('_', ' ').title()} • Venue: {result['match_info']['venue']}")
    
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
    
    # 2. Risk assessment if applied
    risk_assessment = result.get('risk_assessment', {})
    if risk_assessment.get('clean_sheet_risk') == 'high':
        st.markdown("### ⚠️ HIGH RISK CLEAN SHEET ALERT")
        st.error("""
        **High-scoring favorite against opponent who still scores often**  
        - Clean sheet bet downgraded to secondary
        - BTTS/Over bets prioritized
        - Stake reduced by 50% for clean sheet bets
        """)
    elif risk_assessment.get('clean_sheet_risk') == 'medium':
        st.markdown("### ⚠️ MEDIUM RISK CLEAN SHEET")
        st.warning("""
        **High-scoring favorite but opponent rarely scores**  
        - Clean sheet bet kept as primary
        - Confidence reduced by 15 points
        """)
    
    # 3. Warning adjustments if applied
    warning_adj = result.get('warning_adjustments', {})
    if warning_adj.get('applied', False):
        st.markdown("### 🔧 Applied Warning Adjustments")
        for change in warning_adj.get('changes', []):
            st.info(f"• {change.replace('_', ' ').title()}")
    
    # 4. Filters triggered
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
                if filter_key == 'clean_sheet_alert':
                    risk_level = filters['clean_sheet_alert'].get('risk_level', 'low')
                    if risk_level == 'high':
                        st.error(f"✅ Triggered (HIGH RISK)")
                    elif risk_level == 'medium':
                        st.warning(f"✅ Triggered (MEDIUM RISK)")
                    else:
                        st.success(f"✅ Triggered (LOW RISK)")
                else:
                    st.success(f"✅ Triggered")
            else:
                st.info(f"❌ Not Triggered")
    
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
        # Display bets with stakes
        for bet in slip['recommended_bets']:
            stake = slip['stake_suggestions'].get(bet['type'], 0)
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                bet_display = bet['type'].replace('_', ' ').title()
                # Highlight risk-adjusted bets
                if risk_assessment.get('adjusted_bets', False) and bet['priority'] == 'medium' and 'clean' in bet['type'].lower():
                    st.write(f"**{bet_display}** ⚠️ (Risk-Adjusted)")
                else:
                    st.write(f"**{bet_display}**")
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
    
    # 8. Predicted scores
    st.markdown("### 🎯 Predicted Score Range")
    scores = result['predicted_score_range']
    if scores:
        cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores[:5]):
            with cols[idx % 5]:
                st.info(f"**{score}**")
    
    # 9. Data verification toggle
    with st.expander("📊 View Detailed Data & Metrics"):
        st.markdown("#### Calculated Metrics")
        st.json(result['calculated_metrics'], expanded=False)
        
        st.markdown("#### Form Analysis")
        st.json(result['form_analysis'], expanded=False)
        
        if risk_assessment:
            st.markdown("#### Risk Assessment")
            st.json(risk_assessment, expanded=False)

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Betting Analytics Engine v4.3 - RISK-AWARE",
        page_icon="⚽",
        layout="wide"
    )
    
    st.sidebar.image("https://img.icons8.com/color/96/000000/football.png", width=80)
    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.markdown("[GitHub Repository](https://github.com/profdue/sportcreed)")
    st.sidebar.markdown("### 🔬 Version 4.3")
    st.sidebar.markdown("**RISK-AWARE Clean Sheet fix**")
    st.sidebar.markdown("**Avoids risky bets on high-scoring favorites**")
    
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
    st.info(f"🏆 League: {selected_league.replace('_', ' ').title()}")
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        isTeamAHome=is_team_a_home
    )
    
    # Run analysis
    with st.spinner("Running analysis with RISK-AWARE Clean Sheet fix..."):
        engine = BettingAnalyticsEngine()
        result = engine.analyze_match(match_context, selected_league)
    
    # Display results
    render_results_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("Betting Analytics Engine v4.3 • RISK-AWARE Clean Sheet fix")
    st.caption(f"League: {selected_league} • Teams: {team_a} vs {team_b}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
