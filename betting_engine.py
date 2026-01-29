#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v4.5 - Pattern Enhanced
Enhanced with proven patterns from 47-match analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
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
    league: str = ""

# ============================================================================
# LEAGUE PROFILES (Proven from 47-match analysis)
# ============================================================================

LEAGUE_PROFILES = {
    'laliga': {
        'btts_success_rate': 100,      # 4/4
        'clean_sheet_success': 100,    # 1/1
        'trust_btts': True,
        'btts_threshold': 'moderate',  # ≤1/6 CS
        'home_advantage_factor': 1.1,
        'notes': 'BTTS reliable, clean sheets predictable'
    },
    'bundesliga': {
        'btts_success_rate': 100,      # 2/2
        'away_favorite_success': 100,  # 3/3
        'trust_away_favorites': True,
        'btts_threshold': 'moderate',  # ≤1/6 CS
        'home_advantage_factor': 1.0,
        'notes': 'Away favorites strong, high scoring reliable'
    },
    'serie_a': {
        'btts_success_rate': 33,       # 1/3
        'clean_sheet_success': 100,    # 4/4
        'trust_under': True,
        'btts_threshold': 'strict',    # 0/6 CS only
        'home_advantage_factor': 1.15,
        'notes': 'Defensive league, BTTS unreliable, trust Under/Clean sheets'
    },
    'premier_league': {
        'btts_success_rate': 67,       # 2/3
        'home_advantage': True,
        'conservative': True,
        'btts_threshold': 'moderate',  # ≤1/6 CS
        'home_advantage_factor': 1.2,
        'notes': 'Home advantage strong, be conservative'
    },
    'eredivisie': {
        'btts_success_rate': 67,       # 2/3
        'volatile': True,
        'trust_warnings': True,
        'btts_threshold': 'strict',    # 0/6 CS
        'home_advantage_factor': 1.05,
        'notes': 'Volatile league, heavy reliance on warnings'
    },
    'championship': {
        'btts_success_rate': 0,        # No data
        'clean_sheet_success': 0,      # No data
        'btts_threshold': 'moderate',  # Default
        'home_advantage_factor': 1.1,
        'notes': 'Default profile - use moderate thresholds'
    }
}

# ============================================================================
# PATTERN RECOGNITION ENHANCER (Proven patterns from 47 matches)
# ============================================================================

class PatternEnhancer:
    """Apply proven pattern rules from 47-match analysis"""
    
    @staticmethod
    def detect_patterns(filters: Dict, warnings: List[str], 
                       form_analysis: Dict, venue: str, 
                       metrics: Dict, league: str) -> Dict[str, bool]:
        """Detect which proven patterns are present"""
        patterns = {
            'HOME_DEF_IMPROVE': False,
            'CLEAN_AND_LOW': False,
            'BTTS_CLEAN': False,
            'BTTS_REGRESSION': False,
            'FAV_SCORING_DECLINE': False,
            'MULTIPLE_WARNINGS': False,
            'AWAY_FAVORITE': False,
            'BTTS_PREDICTED': False
        }
        
        home_team = 'A' if venue == 'home' else 'B'
        
        # PATTERN 1: Home defensive improvement (4/4 success)
        patterns['HOME_DEF_IMPROVE'] = any(
            f'Team {home_team}: defensive improvement' in w 
            for w in warnings
        )
        
        # PATTERN 2: Clean Sheet + Low-scoring both trigger (3/3 success = 100%)
        patterns['CLEAN_AND_LOW'] = (
            filters.get('clean_sheet_alert', {}).get('team') is not None and 
            filters.get('low_scoring_alert', False)
        )
        
        # PATTERN 3: BTTS predicted with no regression warnings (4/4 success)
        patterns['BTTS_CLEAN'] = (
            filters.get('btts_banker', False) and 
            not any('BTTS may regress' in w for w in warnings)
        )
        
        # PATTERN 4: BTTS regression warning present
        patterns['BTTS_REGRESSION'] = any(
            'BTTS may regress' in w for w in warnings
        )
        
        # PATTERN 5: Favorite has scoring decline (2/2 success = 100%)
        if form_analysis.get('favorite'):
            fav_team = 'A' if form_analysis['favorite'] == 'team_a' else 'B'
            patterns['FAV_SCORING_DECLINE'] = any(
                f'Team {fav_team}: scoring decline' in w 
                for w in warnings
            )
        
        # PATTERN 6: Multiple warnings (≥2)
        patterns['MULTIPLE_WARNINGS'] = len(warnings) >= 2
        
        # PATTERN 7: Away favorite
        if form_analysis.get('favorite'):
            patterns['AWAY_FAVORITE'] = (
                (form_analysis['favorite'] == 'team_a' and venue == 'away') or
                (form_analysis['favorite'] == 'team_b' and venue == 'home')
            )
        
        # PATTERN 8: BTTS predicted
        patterns['BTTS_PREDICTED'] = filters.get('btts_banker', False)
        
        return patterns
    
    @staticmethod
    def apply_pattern_rules(filters: Dict, warnings: List[str], 
                           form_analysis: Dict, venue: str,
                           metrics: Dict, league: str) -> Dict:
        """Apply pattern-based enhancements to predictions"""
        enhancements = {
            'add_bets': [],
            'remove_bets': [],
            'confidence_boost': 0,
            'special_bets': [],
            'stake_multiplier': 1.0,
            'pattern_notes': []
        }
        
        patterns = PatternEnhancer.detect_patterns(
            filters, warnings, form_analysis, venue, metrics, league
        )
        
        home_team = 'A' if venue == 'home' else 'B'
        
        # RULE 1: Home defensive improvement = Clean sheet (4/4 success)
        if patterns['HOME_DEF_IMPROVE']:
            enhancements['add_bets'].append(f'clean_sheet_{"home" if home_team == "A" else "away"}')
            enhancements['remove_bets'].append('btts_yes')
            enhancements['confidence_boost'] += 20
            enhancements['stake_multiplier'] *= 1.3
            enhancements['pattern_notes'].append('Home defensive improvement (100% clean sheets)')
        
        # RULE 2: Clean Sheet + Low-scoring = 0-0 special (3/3 success = 100%)
        if patterns['CLEAN_AND_LOW']:
            enhancements['special_bets'].append({
                'bet': 'correct_score_0_0',
                'stake_multiplier': 2.0,
                'reason': 'Clean Sheet + Low-scoring pattern (100% success)'
            })
            enhancements['confidence_boost'] += 30
            enhancements['stake_multiplier'] *= 1.5
            enhancements['pattern_notes'].append('Clean+Low pattern (100% 0-0)')
        
        # RULE 3: BTTS regression = Remove BTTS bets
        if patterns['BTTS_REGRESSION']:
            enhancements['remove_bets'].append('btts_yes')
            enhancements['confidence_boost'] -= 20
            enhancements['stake_multiplier'] *= 0.7
            enhancements['pattern_notes'].append('BTTS regression warning')
        
        # RULE 4: Favorite scoring decline = Bet against (2/2 success)
        if patterns['FAV_SCORING_DECLINE']:
            if form_analysis.get('favorite'):
                fav_team = 'A' if form_analysis['favorite'] == 'team_a' else 'B'
                enhancements['remove_bets'].append(f'{fav_team}_win')
                enhancements['add_bets'].append('draw_or_underdog')
                enhancements['confidence_boost'] -= 25
                enhancements['stake_multiplier'] *= 0.8
                enhancements['pattern_notes'].append('Favorite scoring decline (100% underdog)')
        
        # RULE 5: Multiple warnings = Caution
        if patterns['MULTIPLE_WARNINGS']:
            enhancements['confidence_boost'] -= 15
            enhancements['stake_multiplier'] *= 0.8
            enhancements['pattern_notes'].append(f'Multiple warnings ({len(warnings)})')
        
        # RULE 6: Away favorite in Bundesliga = Boost confidence
        if patterns['AWAY_FAVORITE'] and league == 'bundesliga':
            enhancements['confidence_boost'] += 10
            enhancements['pattern_notes'].append('Bundesliga away favorite (100% success)')
        
        # RULE 7: League-specific BTTS adjustments
        if patterns['BTTS_PREDICTED']:
            league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['championship'])
            
            if league == 'serie_a':
                # Strict for Serie A
                enhancements['confidence_boost'] -= 10
                enhancements['stake_multiplier'] *= 0.9
                enhancements['pattern_notes'].append('Serie A BTTS caution (33% success)')
            
            elif league in ['laliga', 'bundesliga']:
                # Trust in Spain/Germany
                if not patterns['BTTS_REGRESSION']:
                    enhancements['confidence_boost'] += 10
                    enhancements['stake_multiplier'] *= 1.1
                    enhancements['pattern_notes'].append(f'{league.upper()} BTTS reliable (100% success)')
        
        # RULE 8: Eredivisie volatility = Extra caution
        if league == 'eredivisie' and patterns['MULTIPLE_WARNINGS']:
            enhancements['confidence_boost'] -= 10
            enhancements['stake_multiplier'] *= 0.7
            enhancements['pattern_notes'].append('Eredivisie volatility')
        
        return enhancements, patterns

# ============================================================================
# ENGINE CORE CLASSES (Enhanced)
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
        a_cs_num, a_cs_den = MatchAnalyzer._parse_fraction(team_a.last6['cs'])
        a_btts_num, a_btts_den = MatchAnalyzer._parse_fraction(team_a.last6['btts'])
        b_cs_num, b_cs_den = MatchAnalyzer._parse_fraction(team_b.last6['cs'])
        b_btts_num, b_btts_den = MatchAnalyzer._parse_fraction(team_b.last6['btts'])
        
        a_season_gpm = team_a.overall['goals'] / max(team_a.overall['matches'], 1)
        b_season_gpm = team_b.overall['goals'] / max(team_b.overall['matches'], 1)
        
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
                'season_cs': team_a.overall['cs_percent']
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
                'season_cs': team_b.overall['cs_percent']
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
    """Detect all 5 extreme filters with league-specific adjustments"""
    
    def __init__(self):
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
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, 
                      team_b: TeamFormData, league: str) -> Dict:
        """Detect which extreme filters are triggered with league adjustments"""
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
        
        # FILTER 2: BTTS Banker Alert (LEAGUE-SPECIFIC)
        league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['championship'])
        btts_threshold = league_profile['btts_threshold']
        
        if btts_threshold == 'strict':
            # Strict: Both teams must have 0/6 CS
            btts_trigger = (metrics['team_a']['cs_count'] == 0 and 
                           metrics['team_b']['cs_count'] == 0)
        else:  # 'moderate'
            # Moderate: Both teams ≤1/6 CS
            btts_trigger = (metrics['team_a']['cs_count'] <= 1 and 
                           metrics['team_b']['cs_count'] <= 1)
        
        # Alternative condition: Both teams have season CS% < 20%
        alt_trigger = (team_a.overall['cs_percent'] < self.CS_PERCENT_THRESHOLD and 
                      team_b.overall['cs_percent'] < self.CS_PERCENT_THRESHOLD)
        
        if btts_trigger or alt_trigger:
            filters['btts_banker'] = True
            
            if metrics['averages']['combined_cs_last12'] <= 2:
                filters['btts_enhanced'] = True
            
            if (metrics['team_a']['gpm'] > 1.8 or 
                metrics['team_b']['gpm'] > 1.8):
                filters['btts_enhanced'] = True
        
        # FILTER 3: Clean Sheet Alert
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
        
        # FILTER 4: Low-Scoring Alert
        def is_low_scoring_team(gpm, btts_percent, cs_count):
            return gpm < 1.5 and (btts_percent < self.BTTS_PERCENT_LOW or cs_count >= 2)
        
        if (is_low_scoring_team(metrics['team_a']['gpm'], metrics['team_a']['btts_percent'], metrics['team_a']['cs_count']) and
            is_low_scoring_team(metrics['team_b']['gpm'], metrics['team_b']['btts_percent'], metrics['team_b']['cs_count'])):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'both_low_scoring'
        
        elif (metrics['team_a']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_a']['cs_count'] >= 2 and 
              metrics['team_b']['gpm'] < self.GPM_MODERATE and
              metrics['team_a']['gpm'] < self.GPM_MODERATE):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_weak_a'
        
        elif (metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_b']['cs_count'] >= 2 and 
              metrics['team_a']['gpm'] < self.GPM_MODERATE and
              metrics['team_b']['gpm'] < self.GPM_MODERATE):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_weak_b'
        
        # FILTER 5: Regression & Momentum Alert
        warnings = []
        downgrade_factors = []
        upgrade_factors = []
        
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
        
        if warnings:
            filters['regression_alert']['warnings'] = warnings
            filters['regression_alert']['downgrade_factor'] = min(downgrade_factors) if downgrade_factors else 1.0
            filters['regression_alert']['upgrade_factor'] = max(upgrade_factors) if upgrade_factors else 1.0
        
        return filters

class CurrentFormAnalyzer:
    """Analyze current form with league-specific adjustments"""
    
    @staticmethod
    def analyze_form(team_a: TeamFormData, team_b: TeamFormData, 
                    metrics: Dict, venue: str, league: str) -> Dict:
        """Analyze current form with venue and league considerations"""
        analysis = {
            'favorite': None,
            'favorite_strength': None,
            'goal_potential': None,
            'btts_likelihood': None,
            'scoring_potential': None
        }
        
        # Determine favorite with venue consideration
        win_diff = team_a.last6['wins'] - team_b.last6['wins']
        
        # Apply league-specific home advantage
        league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['championship'])
        home_advantage = league_profile['home_advantage_factor']
        
        if venue == 'home':
            adjusted_win_diff = win_diff * home_advantage
        elif venue == 'away':
            adjusted_win_diff = win_diff / home_advantage
        else:
            adjusted_win_diff = win_diff  # Neutral
        
        if adjusted_win_diff >= 3:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'very_strong'
        elif adjusted_win_diff >= 2:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'strong'
        elif adjusted_win_diff >= 1:
            analysis['favorite'] = 'team_a'
            analysis['favorite_strength'] = 'slight'
        elif adjusted_win_diff <= -3:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'very_strong'
        elif adjusted_win_diff <= -2:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'strong'
        elif adjusted_win_diff <= -1:
            analysis['favorite'] = 'team_b'
            analysis['favorite_strength'] = 'slight'
        else:
            analysis['favorite_strength'] = 'balanced'
        
        # Goal potential (using MAX of Recent or 70% of Season)
        scoring_potential_a = max(metrics['team_a']['gpm'], metrics['team_a']['season_gpm'] * 0.7)
        scoring_potential_b = max(metrics['team_b']['gpm'], metrics['team_b']['season_gpm'] * 0.7)
        avg_scoring_potential = (scoring_potential_a + scoring_potential_b) / 2
        analysis['scoring_potential'] = avg_scoring_potential
        
        # League-adjusted goal potential
        league_adjustment = 1.0
        if league == 'bundesliga':
            league_adjustment = 1.1  # Bundesliga higher scoring
        elif league == 'serie_a':
            league_adjustment = 0.9  # Serie A lower scoring
        
        adjusted_scoring = avg_scoring_potential * league_adjustment
        
        if adjusted_scoring > 2.0:
            analysis['goal_potential'] = 'very_high'
        elif adjusted_scoring > 1.6:
            analysis['goal_potential'] = 'high'
        elif adjusted_scoring > 1.2:
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
    """Generate match script with pattern enhancements"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, 
                       form_analysis: Dict, enhancements: Dict,
                       patterns: Dict, league: str) -> Dict:
        """Generate complete match script with pattern enhancements"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'special_bets': [],
            'predicted_score_range': [],
            'confidence_score': 50,
            'confidence_level': 'low',
            'match_narrative': '',
            'warnings': [],
            'triggered_filter': None,
            'patterns_detected': patterns,
            'enhancement_notes': enhancements.get('pattern_notes', [])
        }
        
        warnings = filters['regression_alert']['warnings']
        
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
        
        # Generate base script from filters
        if triggered_filters:
            triggered_filters.sort(key=lambda x: x[1])
            primary_filter = triggered_filters[0][0]
            script['triggered_filter'] = primary_filter
            
            if primary_filter == 'under_15_alert':
                script = self._generate_under_15_script(script, metrics)
            elif primary_filter == 'btts_banker':
                script = self._generate_btts_banker_script(script, metrics, filters)
            elif primary_filter == 'clean_sheet_alert':
                script = self._generate_clean_sheet_script(script, filters)
            elif primary_filter == 'low_scoring_alert':
                script = self._generate_low_scoring_script(script, metrics, filters)
            
            # Base confidence from filter
            confidence_scores = {
                'under_15_alert': 95,
                'btts_banker': 85,
                'clean_sheet_alert': 80 if filters['clean_sheet_alert']['strength'] == 'strong' else 70,
                'low_scoring_alert': 80 if metrics['averages']['avg_gpm'] < 1.2 else 70
            }
            script['confidence_score'] = confidence_scores.get(primary_filter, 70)
        
        else:
            # No primary filters - use form analysis
            script = self._generate_form_based_script(script, metrics, form_analysis, league)
            script['confidence_score'] = self._calculate_form_confidence(form_analysis, league)
        
        # Apply pattern enhancements
        script['confidence_score'] += enhancements.get('confidence_boost', 0)
        
        # Apply pattern-based bet adjustments
        script = self._apply_enhancements(script, enhancements)
        
        # Add warnings
        if warnings:
            script['warnings'] = warnings
        
        # Apply regression adjustments
        if warnings:
            downgrade = filters['regression_alert']['downgrade_factor']
            upgrade = filters['regression_alert']['upgrade_factor']
            script['confidence_score'] = int(script['confidence_score'] * downgrade * upgrade)
        
        # Cap confidence at 0-100 and convert to level
        script['confidence_score'] = max(0, min(100, script['confidence_score']))
        
        if script['confidence_score'] >= 80:
            script['confidence_level'] = 'high'
        elif script['confidence_score'] >= 65:
            script['confidence_level'] = 'medium'
        else:
            script['confidence_level'] = 'low'
        
        return script
    
    def _apply_enhancements(self, script: Dict, enhancements: Dict) -> Dict:
        """Apply pattern-based enhancements to bets"""
        # Remove bets
        for bet_to_remove in enhancements.get('remove_bets', []):
            for bet_list in ['primary_bets', 'secondary_bets', 'value_bets']:
                if bet_to_remove in script[bet_list]:
                    script[bet_list].remove(bet_to_remove)
        
        # Add bets
        for bet_to_add in enhancements.get('add_bets', []):
            if bet_to_add not in script['primary_bets'] + script['secondary_bets'] + script['value_bets']:
                script['secondary_bets'].append(bet_to_add)
        
        # Add special bets
        script['special_bets'] = enhancements.get('special_bets', [])
        
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
    
    def _generate_form_based_script(self, script: Dict, metrics: Dict, 
                                   form_analysis: Dict, league: str) -> Dict:
        """Generate script based on form analysis"""
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
        script['predicted_score_range'] = self._generate_form_scores(form_analysis, league)
        script['match_narrative'] = self._generate_form_narrative(form_analysis, league)
        
        return script
    
    def _calculate_form_confidence(self, form_analysis: Dict, league: str) -> int:
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
        
        # League adjustment
        league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['championship'])
        if league_profile.get('conservative', False):
            confidence -= 5
        
        # Cap at 70 for form-based predictions
        return min(confidence, 70)
    
    def _generate_form_scores(self, form_analysis: Dict, league: str) -> List[str]:
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
    
    def _generate_form_narrative(self, form_analysis: Dict, league: str) -> str:
        """Generate narrative based on form analysis"""
        narrative_parts = []
        
        league_name = league.upper().replace('_', ' ')
        
        if form_analysis['favorite_strength'] == 'very_strong':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has very strong recent form advantage in {league_name}.')
        elif form_analysis['favorite_strength'] == 'strong':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has strong recent form edge in {league_name}.')
        elif form_analysis['favorite_strength'] == 'slight':
            favorite = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            narrative_parts.append(f'{favorite} has slight recent form edge in {league_name}.')
        
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
    """Generate final betting recommendations with pattern-based stake sizing"""
    
    def __init__(self):
        self.config = {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'total_units': 10,
            'min_confidence_score': 60
        }
    
    def generate_slip(self, script: Dict, team_a_name: str, 
                     team_b_name: str, patterns: Dict, 
                     enhancements: Dict) -> Dict:
        """Generate optimized betting slip with pattern-based stakes"""
        slip = {
            'recommended_bets': [],
            'stake_suggestions': {},
            'special_bets': script.get('special_bets', []),
            'total_units': self.config['total_units'],
            'match_summary': script.get('match_narrative', ''),
            'warnings': script.get('warnings', []),
            'confidence': script.get('confidence_level', 'low'),
            'confidence_score': script.get('confidence_score', 50),
            'patterns_detected': patterns,
            'enhancement_notes': enhancements.get('pattern_notes', [])
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
        
        # Collect all bets
        all_bets = []
        
        # Primary bets
        for bet in script['primary_bets']:
            all_bets.append({
                'type': bet,
                'priority': 'high',
                'reason': 'extreme_filter',
                'confidence': script['confidence_level']
            })
        
        # Secondary bets if space
        if len(all_bets) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                all_bets.append({
                    'type': bet,
                    'priority': 'medium',
                    'reason': 'form_analysis'
                })
        
        # Value bets
        if len(all_bets) < self.config['max_bets_per_match']:
            for bet in script['value_bets']:
                all_bets.append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'value_addition'
                })
        
        # Remove duplicates and limit
        unique_bets = []
        seen = set()
        for bet in all_bets:
            if bet['type'] not in seen:
                unique_bets.append(bet)
                seen.add(bet['type'])
        
        slip['recommended_bets'] = unique_bets[:self.config['max_bets_per_match']]
        
        # Generate stake suggestions with pattern-based multipliers
        if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] != 'NO_BET':
            total_stake = slip['total_units']
            base_confidence_multiplier = script['confidence_score'] / 100
            pattern_multiplier = enhancements.get('stake_multiplier', 1.0)
            
            for bet in slip['recommended_bets']:
                if bet['priority'] == 'high':
                    base_stake = total_stake * self.config['stake_distribution']['high']
                elif bet['priority'] == 'medium':
                    base_stake = total_stake * self.config['stake_distribution']['medium']
                else:
                    base_stake = total_stake * self.config['stake_distribution']['low']
                
                # Adjust by confidence and patterns
                adjusted_stake = base_stake * base_confidence_multiplier * pattern_multiplier
                slip['stake_suggestions'][bet['type']] = round(adjusted_stake, 1)
        
        # Add score suggestions
        if script['predicted_score_range']:
            slip['score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip

# ============================================================================
# MAIN ENGINE CLASS (Enhanced)
# ============================================================================

class BettingAnalyticsEngine:
    """Main orchestrator - complete pipeline with pattern enhancements"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = ExtremeFilterDetector()
        self.form_analyzer = CurrentFormAnalyzer()
        self.pattern_enhancer = PatternEnhancer()
        self.script_generator = None
        self.slip_generator = BettingSlipGenerator()
    
    def analyze_match(self, match_context: MatchContext) -> Dict:
        """Complete analysis pipeline with pattern enhancements"""
        # Initialize script generator with team names
        self.script_generator = MatchScriptGenerator(
            match_context.teamA.teamName, 
            match_context.teamB.teamName
        )
        
        venue = 'home' if match_context.isTeamAHome else 'away'
        
        # Step 1: Calculate metrics
        metrics = self.metrics_calc.calculate_metrics(
            match_context.teamA, 
            match_context.teamB
        )
        
        # Step 2: Detect extreme filters with league-specific adjustments
        filters = self.filter_detector.detect_filters(
            metrics, 
            match_context.teamA, 
            match_context.teamB,
            match_context.league
        )
        
        # Step 3: Analyze current form with league adjustments
        form_analysis = self.form_analyzer.analyze_form(
            match_context.teamA, 
            match_context.teamB,
            metrics,
            venue,
            match_context.league
        )
        
        # Step 4: Apply pattern enhancements
        enhancements, patterns = self.pattern_enhancer.apply_pattern_rules(
            filters,
            filters['regression_alert']['warnings'],
            form_analysis,
            venue,
            metrics,
            match_context.league
        )
        
        # Step 5: Generate match script with pattern enhancements
        script = self.script_generator.generate_script(
            metrics, filters, form_analysis, enhancements, patterns, match_context.league
        )
        
        # Step 6: Generate betting slip with pattern-based stakes
        slip = self.slip_generator.generate_slip(
            script,
            match_context.teamA.teamName,
            match_context.teamB.teamName,
            patterns,
            enhancements
        )
        
        # Compile final result
        result = {
            'match_info': {
                'team_a': match_context.teamA.teamName,
                'team_b': match_context.teamB.teamName,
                'venue': venue,
                'league': match_context.league
            },
            'calculated_metrics': metrics,
            'filters_triggered': filters,
            'form_analysis': form_analysis,
            'patterns_detected': patterns,
            'pattern_enhancements': enhancements,
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
        
        def parse_frac(field_name):
            value = row[field_name]
            if pd.isna(value):
                return "0/6"
            if isinstance(value, str) and '/' in value:
                return value
            return "0/6"
        
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
# STREAMLIT UI COMPONENTS (Enhanced)
# ============================================================================

def render_sidebar() -> Tuple[Optional[str], Optional[str], Optional[str], str, Optional[pd.DataFrame]]:
    """Render sidebar with controls"""
    st.sidebar.title("⚙️ Configuration")
    
    # League selection
    league_options = [
        "bundesliga", "premier_league", "laliga", "serie_a", 
        "ligue_1", "eredivisie", "championship"
    ]
    selected_league = st.sidebar.selectbox("Select League", league_options)
    
    if not selected_league:
        return None, None, None, "home", None
    
    # Load data
    data_loader = DataLoader()
    df = data_loader.load_from_github(selected_league)
    if df is None:
        return None, None, None, "home", None
    
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
    venue_options = ["Team A Home", "Team B Home", "Neutral"]
    venue = st.sidebar.radio("Venue", venue_options, horizontal=True)
    
    if venue == "Team A Home":
        venue_str = "home"
    elif venue == "Team B Home":
        venue_str = "away"
    else:
        venue_str = "neutral"
    
    return selected_league, team_a, team_b, venue_str, df

def render_results_dashboard(result: Dict):
    """Display all analysis results with pattern enhancements"""
    st.title("⚽ Betting Analytics Engine v4.5 - Pattern Enhanced")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    st.caption(f"📍 {result['match_info']['venue'].upper()} • {result['match_info']['league'].upper().replace('_', ' ')}")
    
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
    
    # 2. Patterns detected
    st.markdown("### 🧩 Patterns Detected")
    patterns = result['patterns_detected']
    
    active_patterns = [p for p, active in patterns.items() if active]
    if active_patterns:
        cols = st.columns(min(4, len(active_patterns)))
        for idx, pattern in enumerate(active_patterns[:4]):
            with cols[idx % 4]:
                st.info(f"**{pattern.replace('_', ' ').title()}**")
    else:
        st.info("No strong patterns detected")
    
    # 3. Filters triggered
    st.markdown("### 🔍 Detected Patterns (Filters)")
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
    
    # 4. Pattern enhancement notes
    enhancement_notes = result['betting_slip'].get('enhancement_notes', [])
    if enhancement_notes:
        st.markdown("### 🎯 Pattern Enhancements Applied")
        for note in enhancement_notes:
            st.info(f"• {note}")
    
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
    
    # 8. Special bets (from patterns)
    if slip.get('special_bets'):
        st.markdown("### 🎲 Special Pattern Bets")
        for special in slip['special_bets']:
            if isinstance(special, dict):
                st.success(f"**{special.get('bet', '').replace('_', ' ').title()}** - {special.get('reason', '')}")
            else:
                st.success(f"**{special.replace('_', ' ').title()}**")
    
    # 9. Predicted scores
    st.markdown("### 🎯 Predicted Score Range")
    scores = result['predicted_score_range']
    if scores:
        cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores[:5]):
            with cols[idx % 5]:
                st.info(f"**{score}**")
    
    # 10. Detailed data toggle
    with st.expander("📊 View Detailed Data & Patterns"):
        st.markdown("#### League Profile")
        league_profile = LEAGUE_PROFILES.get(result['match_info']['league'], {})
        st.json(league_profile, expanded=False)
        
        st.markdown("#### Patterns Detected")
        st.json(result['patterns_detected'], expanded=False)
        
        st.markdown("#### Pattern Enhancements")
        st.json(result['pattern_enhancements'], expanded=False)
        
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
        page_title="Betting Analytics Engine v4.5",
        page_icon="⚽",
        layout="wide"
    )
    
    st.sidebar.image("https://img.icons8.com/color/96/000000/football.png", width=80)
    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.markdown("[GitHub Repository](https://github.com/profdue/sportcreed)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Pattern Enhancements")
    st.sidebar.markdown("""
    - **League-specific** BTTS thresholds
    - **Home defensive improvement** = Clean sheet
    - **Clean+Low pattern** = 0-0 special bet
    - **Pattern-aware** confidence scoring
    - **Proven patterns** from 47-match analysis
    """)
    
    # Sidebar
    selected_league, team_a, team_b, venue_str, df = render_sidebar()
    
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
    venue_display = "Team A Home" if venue_str == "home" else "Team B Home" if venue_str == "away" else "Neutral"
    st.info(f"📍 **Venue:** {venue_display} • **League:** {selected_league.upper().replace('_', ' ')}")
    
    # Show league profile
    league_profile = LEAGUE_PROFILES.get(selected_league, {})
    if league_profile:
        with st.expander(f"📋 {selected_league.upper().replace('_', ' ')} League Profile"):
            st.write(f"**BTTS Success Rate:** {league_profile.get('btts_success_rate', 0)}%")
            st.write(f"**Clean Sheet Success:** {league_profile.get('clean_sheet_success', 0)}%")
            st.write(f"**BTTS Threshold:** {league_profile.get('btts_threshold', 'moderate').title()}")
            st.write(f"**Notes:** {league_profile.get('notes', '')}")
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        isTeamAHome=(venue_str == "home"),
        league=selected_league
    )
    
    # Run analysis
    with st.spinner("Running pattern-enhanced analysis..."):
        engine = BettingAnalyticsEngine()
        result = engine.analyze_match(match_context)
    
    # Display results
    render_results_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("Betting Analytics Engine v4.5 • Pattern Enhanced • 47-match Proven Logic")
    st.caption(f"League: {selected_league} • Teams: {team_a} vs {team_b} • Venue: {venue_display}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
