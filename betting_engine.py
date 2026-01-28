#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v3.1 - H2H-FREE VERSION
Uses ONLY team statistics, NO head-to-head data
Team A stats vs Team B stats → Predictions
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import math
import re

# ============================================================================
# SECTION 2: DATA STRUCTURES
# ============================================================================

@dataclass
class TeamFormData:
    """Team statistics ONLY - NO H2H data needed"""
    teamName: str
    # Last 6 matches - REQUIRED
    last6: Dict[str, Any]
    # Overall season stats (for regression detection)
    overall: Optional[Dict[str, Any]] = None
    # Current position (optional)
    leaguePosition: Optional[int] = None

@dataclass
class MarketOdds:
    """Market odds interface - Optional"""
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    over_15: Optional[float] = None
    under_15: Optional[float] = None
    over_25: Optional[float] = None
    under_25: Optional[float] = None
    team_a_win: Optional[float] = None
    team_b_win: Optional[float] = None
    draw: Optional[float] = None

@dataclass 
class MatchContext:
    """Match context - NO H2H data"""
    teamA: TeamFormData
    teamB: TeamFormData
    isTeamAHome: bool = True
    marketOdds: Optional[MarketOdds] = None

# ============================================================================
# SECTION 3: ENGINE CLASSES (H2H-FREE VERSION)
# ============================================================================

class MatchAnalyzer:
    """Calculate metrics from team stats ONLY"""
    
    def __init__(self, team_a: TeamFormData, team_b: TeamFormData):
        self.team_a = team_a
        self.team_b = team_b
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all metrics from team stats only"""
        metrics = {
            'team_a': {
                'gpm': self.team_a.last6['goalsScored'] / 6,
                'gcpm': self.team_a.last6.get('goalsConceded', 0) / 6 if self.team_a.last6.get('goalsConceded') is not None else None,
                'cs_percent': (self.team_a.last6['cleanSheets'] / 6) * 100,
                'btts_percent': (self.team_a.last6['bttsMatches'] / 6) * 100,
                'win_percent': (self.team_a.last6['wins'] / 6) * 100,
                'loss_percent': (self.team_a.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_a.last6['draws'] / 6) * 100,
                'avg_gpm': self.team_a.last6['goalsScored'] / 6
            },
            'team_b': {
                'gpm': self.team_b.last6['goalsScored'] / 6,
                'gcpm': self.team_b.last6.get('goalsConceded', 0) / 6 if self.team_b.last6.get('goalsConceded') is not None else None,
                'cs_percent': (self.team_b.last6['cleanSheets'] / 6) * 100,
                'btts_percent': (self.team_b.last6['bttsMatches'] / 6) * 100,
                'win_percent': (self.team_b.last6['wins'] / 6) * 100,
                'loss_percent': (self.team_b.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_b.last6['draws'] / 6) * 100,
                'avg_gpm': self.team_b.last6['goalsScored'] / 6
            }
        }
        
        # Combined metrics (NO H2H)
        metrics['combined'] = {
            'avg_gpm': (metrics['team_a']['gpm'] + metrics['team_b']['gpm']) / 2,
            'avg_cs_percent': (metrics['team_a']['cs_percent'] + metrics['team_b']['cs_percent']) / 2,
            'avg_btts_percent': (metrics['team_a']['btts_percent'] + metrics['team_b']['btts_percent']) / 2,
            'goal_potential': self._classify_goal_potential(metrics['team_a']['gpm'], metrics['team_b']['gpm']),
            'btts_likelihood': self._classify_btts_likelihood(metrics['team_a']['btts_percent'], metrics['team_b']['btts_percent'])
        }
        
        return metrics
    
    def _classify_goal_potential(self, gpm_a: float, gpm_b: float) -> str:
        """Classify goal potential based on both teams' GPM"""
        avg_gpm = (gpm_a + gpm_b) / 2
        if avg_gpm > 1.8:
            return 'high'
        elif avg_gpm > 1.3:
            return 'medium'
        else:
            return 'low'
    
    def _classify_btts_likelihood(self, btts_a: float, btts_b: float) -> str:
        """Classify BTTS likelihood based on both teams' BTTS%"""
        avg_btts = (btts_a + btts_b) / 2
        if avg_btts > 60:
            return 'high'
        elif avg_btts > 40:
            return 'medium'
        else:
            return 'low'


class ExtremeFilterDetector:
    """All 5 filters - MODIFIED for NO H2H data"""
    
    # CONSTANTS
    UNDER_15_GPM_THRESHOLD = 0.75
    
    # Filter 2: BTTS Banker Alert (MODIFIED - NO H2H requirement)
    CS_PERCENT_THRESHOLD = 20
    MIN_GPM_FOR_BTTS = 1.3  # Instead of H2H Over 1.5%
    HIGH_GPM_THRESHOLD = 1.5
    
    # Filter 3: Clean Sheet Alert
    CS_PERCENT_STRONG = 50
    OPPONENT_GPM_WEAK = 1.0
    
    # Filter 4: Low-Scoring Alert
    BTTS_PERCENT_LOW = 40
    CS_PERCENT_DECENT = 30
    GPM_MODERATE = 1.5
    
    # Filter 5: Regression Detection
    RECENT_BTTS_INFLATION_MIN = 70
    SEASON_BTTS_REGRESSION_MAX = 60
    GPM_INFLATION_FACTOR = 1.5
    
    def detect_filters(self, metrics: Dict[str, Any],
                      team_a_overall: Optional[Dict] = None,
                      team_b_overall: Optional[Dict] = None) -> Dict[str, Any]:
        """Detect all 5 filters using team stats ONLY"""
        filters = {
            'under_15_alert': False,
            'btts_banker': False,
            'btts_enhanced': False,
            'clean_sheet_alert': {'team': None, 'direction': None},
            'low_scoring_alert': False,
            'low_scoring_type': None,
            'regression_alert': {'team': None, 'type': None, 'severity': None}
        }
        
        # FILTER 1: Under 1.5 Goals Alert
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # FILTER 2: BTTS Banker Alert (MODIFIED - NO H2H)
        # Original required H2H Over 1.5% > 65%
        # Now uses: Both teams concede (CS% < 20%) AND at least one scores well (GPM > 1.3)
        if (metrics['team_a']['cs_percent'] < self.CS_PERCENT_THRESHOLD and 
            metrics['team_b']['cs_percent'] < self.CS_PERCENT_THRESHOLD and
            (metrics['team_a']['gpm'] > self.MIN_GPM_FOR_BTTS or 
             metrics['team_b']['gpm'] > self.MIN_GPM_FOR_BTTS)):
            filters['btts_banker'] = True
            
            # Enhancement: Add Over 2.5 if high scoring form
            if (metrics['team_a']['gpm'] > self.HIGH_GPM_THRESHOLD or 
                metrics['team_b']['gpm'] > self.HIGH_GPM_THRESHOLD):
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
            filters['low_scoring_type'] = 'defensive_vs_leaky'
        
        elif (metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW and 
              metrics['team_b']['cs_percent'] > self.CS_PERCENT_DECENT and 
              metrics['team_a']['gpm'] < self.GPM_MODERATE):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'defensive_vs_leaky'
        
        # FILTER 5: Regression to Mean Alert
        if team_a_overall and 'bttsMatches' in team_a_overall and 'matches' in team_a_overall:
            season_btts_percent = (team_a_overall['bttsMatches'] / team_a_overall['matches']) * 100
            if (metrics['team_a']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_percent < self.SEASON_BTTS_REGRESSION_MAX):
                filters['regression_alert'] = {'team': 'A', 'type': 'btts_regression', 'severity': 'high'}
        
        if team_b_overall and 'bttsMatches' in team_b_overall and 'matches' in team_b_overall:
            season_btts_percent = (team_b_overall['bttsMatches'] / team_b_overall['matches']) * 100
            if (metrics['team_b']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_percent < self.SEASON_BTTS_REGRESSION_MAX):
                if filters['regression_alert']['team'] == 'A':
                    filters['regression_alert'] = {'team': 'both', 'type': 'btts_regression', 'severity': 'high'}
                else:
                    filters['regression_alert'] = {'team': 'B', 'type': 'btts_regression', 'severity': 'high'}
            
        return filters


class CurrentFormAnalyzer:
    """Current form analysis - NO H2H"""
    
    def analyze_form(self, team_a_data: TeamFormData, team_b_data: TeamFormData) -> Dict[str, Any]:
        """Analyze current form using team stats only"""
        analysis = {
            'favorite': None,
            'form_edge': None,  # 'strong', 'moderate', 'slight', 'none'
            'goal_potential': None,
            'btts_likelihood': None,
            'momentum': None  # 'team_a', 'team_b', 'balanced'
        }
        
        # Determine favorite based on wins
        win_diff = team_a_data.last6['wins'] - team_b_data.last6['wins']
        
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
        
        # Goal potential
        avg_gpm = (team_a_data.last6['goalsScored']/6 + team_b_data.last6['goalsScored']/6) / 2
        if avg_gpm > 1.8:
            analysis['goal_potential'] = 'high'
        elif avg_gpm > 1.3:
            analysis['goal_potential'] = 'medium'
        else:
            analysis['goal_potential'] = 'low'
        
        # BTTS likelihood
        avg_btts = ((team_a_data.last6['bttsMatches']/6)*100 + 
                   (team_b_data.last6['bttsMatches']/6)*100) / 2
        if avg_btts > 60:
            analysis['btts_likelihood'] = 'high'
        elif avg_btts > 40:
            analysis['btts_likelihood'] = 'medium'
        else:
            analysis['btts_likelihood'] = 'low'
        
        # Momentum (recent form trend)
        momentum_score_a = (team_a_data.last6['wins'] * 3 + team_a_data.last6['draws']) - team_a_data.last6['losses']
        momentum_score_b = (team_b_data.last6['wins'] * 3 + team_b_data.last6['draws']) - team_b_data.last6['losses']
        
        if momentum_score_a > momentum_score_b + 2:
            analysis['momentum'] = 'team_a'
        elif momentum_score_b > momentum_score_a + 2:
            analysis['momentum'] = 'team_b'
        else:
            analysis['momentum'] = 'balanced'
            
        return analysis


class MatchScriptGenerator:
    """Generate betting narrative - NO H2H trends"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict[str, Any], filters: Dict[str, Any], 
                       form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete match script using team stats ONLY"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'predicted_score_range': [],
            'confidence': 'low',
            'match_narrative': '',
            'warnings': [],
            'key_insights': []
        }
        
        # Check for regression warnings
        if filters['regression_alert']['team']:
            team = filters['regression_alert']['team']
            if team == 'A' or team == 'both':
                script['warnings'].append(f"{self.team_a_name}: Recent BTTS rate inflated - expect regression to season average")
            if team == 'B' or team == 'both':
                script['warnings'].append(f"{self.team_b_name}: Recent BTTS rate inflated - expect regression to season average")
        
        # RULE 1: Extreme Filters override everything
        if filters['under_15_alert']:
            script['primary_bets'].append('under_15_goals')
            script['primary_bets'].append('btts_no')
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams have low scoring form - expect very few goals'
            script['key_insights'].append('Both teams average < 0.75 goals per game in last 6')
            return script
            
        if filters['low_scoring_alert']:
            script['primary_bets'].append('under_25_goals')
            script['primary_bets'].append('btts_no')
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            
            if filters['low_scoring_type'] == 'both_low_btts':
                script['match_narrative'] = 'Both teams rarely see both teams score - low-scoring affair likely'
                script['key_insights'].append('Both teams have BTTS% < 40% in last 6')
            else:
                script['match_narrative'] = 'Defensive team meets low-scoring opponent - clean sheet likely'
                if filters['low_scoring_type'] == 'defensive_vs_leaky':
                    script['key_insights'].append('One team strong defensively, other team low-scoring')
            return script
            
        if filters['btts_banker']:
            script['primary_bets'].append('btts_yes')
            script['secondary_bets'].append('over_15_goals')
            
            if filters['btts_enhanced']:
                script['value_bets'].append('over_25_goals')
                script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2']
                script['key_insights'].append('High scoring form suggests >2.5 goals likely')
            else:
                script['predicted_score_range'] = ['1-1', '2-1', '1-2']
            
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams consistently concede - expect goals at both ends'
            script['key_insights'].append('Both teams have clean sheet% < 20% in last 6')
            return script
        
        if filters['clean_sheet_alert']['team']:
            team_name = self.team_a_name if filters['clean_sheet_alert']['team'] == 'A' else self.team_b_name
            script['primary_bets'].append(f'{team_name.lower().replace(" ", "_")}_win_to_nil')
            script['secondary_bets'].append('under_25_goals')
            script['predicted_score_range'] = [f'2-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-2',
                                              f'1-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = f'{team_name} strong defensively against weak attack'
            script['key_insights'].append(f'{team_name} has clean sheet% > 50% while opponent scores < 1.0 GPG')
            return script
        
        # RULE 2: If no extreme filters, use form analysis
        if not script['primary_bets']:
            # Determine bets based on form analysis
            if form_analysis['favorite']:
                favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
                script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win_or_draw')
                script['key_insights'].append(f'{favorite_name} has better recent form')
            
            # Add goal market bets based on potential
            if form_analysis['goal_potential'] == 'high':
                script['secondary_bets'].append('over_15_goals')
                if metrics['combined']['avg_gpm'] > 2.0:
                    script['secondary_bets'].append('over_25_goals')
            elif form_analysis['goal_potential'] == 'low':
                script['secondary_bets'].append('under_25_goals')
            
            # Add BTTS bets based on likelihood
            if form_analysis['btts_likelihood'] == 'high':
                script['secondary_bets'].append('btts_yes')
            elif form_analysis['btts_likelihood'] == 'low':
                script['secondary_bets'].append('btts_no')
            
            script['confidence'] = 'medium' if form_analysis['form_edge'] != 'none' else 'low'
            
            # Generate narrative
            if form_analysis['goal_potential'] == 'high':
                script['match_narrative'] += 'High scoring potential. '
            elif form_analysis['goal_potential'] == 'low':
                script['match_narrative'] += 'Low scoring potential. '
                
            if form_analysis['btts_likelihood'] == 'high':
                script['match_narrative'] += 'Both teams likely to score. '
            elif form_analysis['btts_likelihood'] == 'low':
                script['match_narrative'] += 'Clean sheet possible. '
                
            if form_analysis['momentum'] != 'balanced':
                momentum_team = self.team_a_name if form_analysis['momentum'] == 'team_a' else self.team_b_name
                script['match_narrative'] += f'{momentum_team} has better recent momentum. '
        
        # Generate score range if not already set
        if not script['predicted_score_range']:
            script['predicted_score_range'] = self._generate_score_range(script, metrics, form_analysis)
        
        return script
    
    def _generate_score_range(self, script: Dict[str, Any], metrics: Dict[str, Any], 
                            form_analysis: Dict[str, Any]) -> List[str]:
        """Generate likely score ranges"""
        scores = []
        
        if 'btts_yes' in script['primary_bets'] or 'btts_yes' in script['secondary_bets']:
            if 'over_25_goals' in script['value_bets'] or 'over_25_goals' in script['secondary_bets']:
                scores.extend(['2-1', '1-2', '2-2', '3-1', '1-3'])
            else:
                scores.extend(['1-1', '2-1', '1-2'])
        elif 'under_15_goals' in script['primary_bets']:
            scores.extend(['0-0', '1-0', '0-1'])
        elif 'under_25_goals' in script['primary_bets'] or 'under_25_goals' in script['secondary_bets']:
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


class ValueDetector:
    """Value bet detection - OPTIONAL"""
    
    VALUE_EDGE_MINIMUM = 15
    HIGH_CONFIDENCE_PROB = 0.75
    MEDIUM_CONFIDENCE_PROB = 0.60
    LOW_CONFIDENCE_PROB = 0.40
    
    def __init__(self, market_odds: MarketOdds):
        self.market_odds = market_odds
    
    def calculate_value(self, script: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate value bets"""
        if not self.market_odds:
            return []
            
        value_bets = []
        
        confidence_prob = {
            'high': self.HIGH_CONFIDENCE_PROB,
            'medium': self.MEDIUM_CONFIDENCE_PROB,
            'low': self.LOW_CONFIDENCE_PROB
        }
        
        our_probability = confidence_prob.get(script['confidence'], 0.5)
        
        # Check primary bets
        for bet_type in script['primary_bets']:
            odds_key = self._map_bet_to_odds_key(bet_type)
            if hasattr(self.market_odds, odds_key) and getattr(self.market_odds, odds_key) is not None:
                odds = getattr(self.market_odds, odds_key)
                if odds > 0:
                    implied_prob = 1 / odds
                    edge = ((our_probability / implied_prob) - 1) * 100
                    
                    if edge >= self.VALUE_EDGE_MINIMUM:
                        value_bets.append({
                            'bet_type': bet_type,
                            'odds_key': odds_key,
                            'odds': odds,
                            'our_probability': round(our_probability * 100, 1),
                            'implied_probability': round(implied_prob * 100, 1),
                            'edge_percent': round(edge, 1),
                            'reason': 'primary_bet_with_edge'
                        })
        
        # Check secondary bets
        secondary_prob = our_probability * 0.8
        for bet_type in script['secondary_bets']:
            odds_key = self._map_bet_to_odds_key(bet_type)
            if hasattr(self.market_odds, odds_key) and getattr(self.market_odds, odds_key) is not None:
                odds = getattr(self.market_odds, odds_key)
                if odds > 0:
                    implied_prob = 1 / odds
                    edge = ((secondary_prob / implied_prob) - 1) * 100
                    
                    if edge >= self.VALUE_EDGE_MINIMUM * 1.5:
                        value_bets.append({
                            'bet_type': bet_type,
                            'odds_key': odds_key,
                            'odds': odds,
                            'our_probability': round(secondary_prob * 100, 1),
                            'implied_probability': round(implied_prob * 100, 1),
                            'edge_percent': round(edge, 1),
                            'reason': 'secondary_bet_with_strong_edge'
                        })
        
        return value_bets
    
    def _map_bet_to_odds_key(self, bet_type: str) -> str:
        """Map bet type to odds key"""
        mapping = {
            'under_15_goals': 'under_15',
            'over_15_goals': 'over_15',
            'under_25_goals': 'under_25',
            'over_25_goals': 'over_25',
            'btts_yes': 'btts_yes',
            'btts_no': 'btts_no',
            'draw': 'draw'
        }
        
        if '_win_to_nil' in bet_type:
            team = bet_type.replace('_win_to_nil', '')
            return f'{team}_win_to_nil'
        
        if '_win_or_draw' in bet_type:
            team = bet_type.replace('_win_or_draw', '')
            return f'{team}_win_or_draw'
        
        return mapping.get(bet_type, bet_type)


class BettingSlipGenerator:
    """Generate betting slip"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'min_confidence': 'medium'
        }
    
    def generate_slip(self, script: Dict[str, Any], value_bets: List[Dict[str, Any]], 
                     team_a_name: str, team_b_name: str) -> Dict[str, Any]:
        """Generate optimized betting slip"""
        slip = {
            'recommended_bets': [],
            'avoid_bets': [],
            'stake_suggestions': {},
            'total_units': 10,
            'match_summary': script.get('match_narrative', ''),
            'confidence': script.get('confidence', 'low'),
            'key_insights': script.get('key_insights', [])
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
        
        # Priority 1: Primary bets
        for bet in script['primary_bets']:
            slip['recommended_bets'].append({
                'type': bet,
                'priority': 'high',
                'reason': 'extreme_filter' if script['confidence'] == 'high' else 'strong_form_pattern',
                'confidence': script['confidence']
            })
        
        # Priority 2: Value bets
        for value_bet in value_bets:
            slip['recommended_bets'].append({
                'type': value_bet['bet_type'],
                'priority': 'medium',
                'reason': 'value_edge',
                'edge_percent': value_bet['edge_percent'],
                'odds': value_bet['odds']
            })
        
        # Priority 3: Secondary bets
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'supporting_bet'
                })
        
        # Remove duplicates
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
            if bet['type'] != 'NO_BET':
                stake_pct = self.config['stake_distribution'].get(bet['priority'], 0.2)
                slip['stake_suggestions'][bet['type']] = round(total_stake * stake_pct, 1)
        
        # Score suggestions
        if script['predicted_score_range']:
            slip['correct_score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip


class BettingAnalyticsEngine:
    """Main engine - NO H2H data needed"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def analyze_match(self, match_context: MatchContext) -> Dict[str, Any]:
        """Complete analysis using team stats ONLY"""
        # Initialize components
        script_generator = MatchScriptGenerator(
            match_context.teamA.teamName, 
            match_context.teamB.teamName
        )
        
        if match_context.marketOdds:
            value_detector = ValueDetector(match_context.marketOdds)
        
        # Step 1: Calculate metrics
        metrics_calc = MatchAnalyzer(match_context.teamA, match_context.teamB)
        metrics = metrics_calc.calculate_metrics()
        
        # Step 2: Detect extreme filters
        filter_detector = ExtremeFilterDetector()
        filters = filter_detector.detect_filters(
            metrics, 
            match_context.teamA.overall, 
            match_context.teamB.overall
        )
        
        # Step 3: Analyze current form
        form_analyzer = CurrentFormAnalyzer()
        form_analysis = form_analyzer.analyze_form(match_context.teamA, match_context.teamB)
        
        # Step 4: Generate match script
        script = script_generator.generate_script(metrics, filters, form_analysis)
        
        # Step 5: Detect value bets
        value_bets = []
        if match_context.marketOdds:
            value_bets = value_detector.calculate_value(script)
        
        # Step 6: Generate betting slip
        slip_generator = BettingSlipGenerator(self.config)
        slip = slip_generator.generate_slip(
            script, 
            value_bets,
            match_context.teamA.teamName,
            match_context.teamB.teamName
        )
        
        # Compile result
        result = {
            'match_info': {
                'team_a': match_context.teamA.teamName,
                'team_b': match_context.teamB.teamName,
                'venue': 'home' if match_context.isTeamAHome else 'away',
                'league_position_a': match_context.teamA.leaguePosition,
                'league_position_b': match_context.teamB.leaguePosition
            },
            'calculated_metrics': metrics,
            'filters_triggered': filters,
            'form_analysis': form_analysis,
            'match_script': script,
            'value_bets': value_bets,
            'betting_slip': slip,
            'predicted_score_range': script['predicted_score_range'],
            'confidence': script['confidence'],
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# ============================================================================
# SECTION 4: DATA PARSING
# ============================================================================

def parse_fraction_string(fraction_str: str) -> Tuple[int, int]:
    """Parse '5/6 BTTS' -> (5, 6)"""
    if isinstance(fraction_str, str):
        match = re.search(r'(\d+)/(\d+)', fraction_str)
        if match:
            return int(match.group(1)), int(match.group(2))
    return 0, 6

def parse_percentage_string(percent_str: str) -> float:
    """Parse '40.00%' -> 40.0"""
    if isinstance(percent_str, str):
        return float(percent_str.replace('%', '').strip())
    elif isinstance(percent_str, (int, float)):
        return float(percent_str)
    return 0.0

def parse_team_data(df: pd.DataFrame, team_name: str) -> Optional[TeamFormData]:
    """Parse team data from your CSV format"""
    team_row = df[df['Team'] == team_name]
    
    if team_row.empty:
        st.error(f"Team '{team_name}' not found in data")
        return None
    
    row = team_row.iloc[0]
    
    # Try to extract clean sheets and BTTS from various column formats
    clean_sheets = 0
    btts_matches = 0
    
    # Look for CS data
    for col in ['CS', 'CleanSheets', 'CS_Last6', 'Clean Sheets']:
        if col in row and pd.notna(row[col]):
            cs_numer, _ = parse_fraction_string(str(row[col]))
            clean_sheets = cs_numer
            break
    
    # Look for BTTS data
    for col in ['BTTS', 'BTTS_Last6', 'Both Teams Scored']:
        if col in row and pd.notna(row[col]):
            btts_numer, _ = parse_fraction_string(str(row[col]))
            btts_matches = btts_numer
            break
    
    # If not found, estimate from percentages
    if clean_sheets == 0 and 'CS%' in row:
        cs_percent = parse_percentage_string(row['CS%'])
        clean_sheets = round((cs_percent / 100) * 6)
    
    if btts_matches == 0 and 'BTTS%' in row:
        btts_percent = parse_percentage_string(row['BTTS%'])
        btts_matches = round((btts_percent / 100) * 6)
    
    # Build team data
    team_data = TeamFormData(
        teamName=team_name,
        last6={
            'matches': 6,
            'goalsScored': int(row['Last 6 Goals']) if 'Last 6 Goals' in row else 0,
            'goalsConceded': None,  # Optional field
            'cleanSheets': clean_sheets,
            'bttsMatches': btts_matches,
            'wins': int(row['Form W']) if 'Form W' in row else 0,
            'draws': int(row['Form D']) if 'Form D' in row else 0,
            'losses': int(row['Form L']) if 'Form L' in row else 0
        },
        overall={
            'matches': int(row['Overall Matches']) if 'Overall Matches' in row else 0,
            'goalsScored': int(row['Overall Goals']) if 'Overall Goals' in row else 0,
            'goalsConceded': None,  # Optional field
            'cleanSheets': int((parse_percentage_string(row.get('Overall CS%', 0)) / 100) * 
                             row['Overall Matches']) if 'Overall Matches' in row and 'Overall CS%' in row else 0,
            'bttsMatches': int((parse_percentage_string(row.get('Overall BTTS%', 0)) / 100) * 
                             row['Overall Matches']) if 'Overall Matches' in row and 'Overall BTTS%' in row else 0
        },
        leaguePosition=None  # Add if available
    )
    
    return team_data

def load_league_teams(league_name: str) -> Optional[pd.DataFrame]:
    """Load team data CSV"""
    possible_files = [
        f"leagues/{league_name}.csv",
        f"leagues/{league_name}_teams.csv",
        f"leagues/{league_name}_stats.csv",
        f"{league_name}.csv"
    ]
    
    for file_path in possible_files:
        try:
            df = pd.read_csv(file_path)
            # Basic validation
            if 'Team' in df.columns:
                return df
        except:
            continue
    
    st.error(f"Could not load team data for {league_name}")
    return None

# ============================================================================
# SECTION 5: STREAMLIT UI
# ============================================================================

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.title("⚽ Betting Analytics Engine v3.1")
    st.sidebar.markdown("**NO H2H Data Version**")
    st.sidebar.markdown("---")
    
    # League selection
    st.sidebar.subheader("1. Select League")
    
    # Try to find available leagues
    available_leagues = []
    try:
        import os
        if os.path.exists('leagues'):
            csv_files = [f for f in os.listdir('leagues') if f.endswith('.csv')]
            league_names = [f.replace('.csv', '').replace('_teams', '').replace('_stats', '').replace('_h2h', '') 
                           for f in csv_files]
            available_leagues = sorted(list(set(league_names)))
    except:
        # Fallback leagues based on your repository
        available_leagues = [
            'bundesliga', 'bundesliga_2', 'championship', 'eredivisie',
            'erste_divisie', 'laliga', 'laliga_2', 'ligue_1', 'ligue_2',
            'premier_league', 'serie_a', 'serie_b'
        ]
    
    if not available_leagues:
        st.sidebar.error("No league data found. Please place CSV files in 'leagues/' folder.")
        return None, None, None, None, None
    
    selected_league = st.sidebar.selectbox("League", available_leagues)
    
    # Load teams
    teams_df = load_league_teams(selected_league)
    if teams_df is None:
        return None, None, None, None, None
    
    teams = teams_df['Team'].tolist() if 'Team' in teams_df.columns else []
    
    if not teams:
        st.sidebar.error("No teams found in league data")
        return None, None, None, None, None
    
    # Team selection
    st.sidebar.subheader("2. Select Teams")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        team_a = st.selectbox("Team A", teams, key="team_a_select")
    
    with col2:
        available_teams_b = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", available_teams_b, key="team_b_select")
    
    # Venue
    st.sidebar.subheader("3. Match Details")
    is_team_a_home = st.sidebar.radio("Venue", ["Team A at Home", "Team B at Home"]) == "Team A at Home"
    
    # Market odds (optional)
    st.sidebar.subheader("4. Market Odds (Optional)")
    
    if st.sidebar.checkbox("Enter market odds for value detection"):
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            btts_yes = st.number_input("BTTS Yes", min_value=1.0, max_value=10.0, value=1.8, step=0.05)
            over_15 = st.number_input("Over 1.5", min_value=1.0, max_value=5.0, value=1.3, step=0.05)
            over_25 = st.number_input("Over 2.5", min_value=1.0, max_value=5.0, value=2.1, step=0.05)
        
        with col2:
            btts_no = st.number_input("BTTS No", min_value=1.0, max_value=10.0, value=2.0, step=0.05)
            under_15 = st.number_input("Under 1.5", min_value=1.0, max_value=5.0, value=3.4, step=0.05)
            under_25 = st.number_input("Under 2.5", min_value=1.0, max_value=5.0, value=1.7, step=0.05)
        
        team_a_win = st.sidebar.number_input(f"{team_a} Win", min_value=1.0, max_value=10.0, value=2.5, step=0.05)
        team_b_win = st.sidebar.number_input(f"{team_b} Win", min_value=1.0, max_value=10.0, value=2.8, step=0.05)
        draw = st.sidebar.number_input("Draw", min_value=1.0, max_value=10.0, value=3.2, step=0.05)
        
        market_odds = MarketOdds(
            btts_yes=btts_yes,
            btts_no=btts_no,
            over_15=over_15,
            under_15=under_15,
            over_25=over_25,
            under_25=under_25,
            team_a_win=team_a_win,
            team_b_win=team_b_win,
            draw=draw
        )
    else:
        market_odds = None
    
    return selected_league, team_a, team_b, market_odds, is_team_a_home

def render_metrics_panel(metrics: Dict[str, Any]):
    """Display metrics"""
    st.subheader("📊 Team Statistics Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Team A Metrics**")
        st.metric("Goals/Game", f"{metrics['team_a']['gpm']:.2f}")
        st.metric("Clean Sheet %", f"{metrics['team_a']['cs_percent']:.1f}%")
        st.metric("BTTS %", f"{metrics['team_a']['btts_percent']:.1f}%")
        st.metric("Win %", f"{metrics['team_a']['win_percent']:.1f}%")
    
    with col2:
        st.markdown("**Team B Metrics**")
        st.metric("Goals/Game", f"{metrics['team_b']['gpm']:.2f}")
        st.metric("Clean Sheet %", f"{metrics['team_b']['cs_percent']:.1f}%")
        st.metric("BTTS %", f"{metrics['team_b']['btts_percent']:.1f}%")
        st.metric("Win %", f"{metrics['team_b']['win_percent']:.1f}%")
    
    with col3:
        st.markdown("**Combined Analysis**")
        st.metric("Avg Goals/Game", f"{metrics['combined']['avg_gpm']:.2f}")
        st.metric("Goal Potential", metrics['combined']['goal_potential'].title())
        st.metric("BTTS Likelihood", metrics['combined']['btts_likelihood'].title())
        st.metric("Avg CS %", f"{metrics['combined']['avg_cs_percent']:.1f}%")

def render_filters_panel(filters: Dict[str, Any], team_a_name: str, team_b_name: str):
    """Display triggered filters"""
    st.subheader("🔍 Extreme Filters Detected")
    
    filter_info = {
        'under_15_alert': {'name': 'Under 1.5 Goals Alert', 'icon': '🔴', 'desc': 'Both teams average < 0.75 GPG'},
        'btts_banker': {'name': 'BTTS Banker Alert', 'icon': '🟢', 'desc': 'Both teams CS% < 20% with scoring form'},
        'clean_sheet_alert': {'name': 'Clean Sheet Alert', 'icon': '🔵', 'desc': 'Strong defense vs weak attack'},
        'low_scoring_alert': {'name': 'Low-Scoring Alert', 'icon': '🟡', 'desc': 'Low BTTS patterns detected'},
        'regression_alert': {'name': 'Regression Alert', 'icon': '🟣', 'desc': 'Recent form inflated vs season average'}
    }
    
    active_filters = []
    
    for filter_key, info in filter_info.items():
        if filter_key == 'clean_sheet_alert' and filters[filter_key]['team']:
            active_filters.append((info['icon'], info['name'], 
                                  f"{team_a_name if filters[filter_key]['team'] == 'A' else team_b_name} Win to Nil"))
        elif filter_key == 'regression_alert' and filters[filter_key]['team']:
            team = filters[filter_key]['team']
            if team == 'A':
                team_name = team_a_name
            elif team == 'B':
                team_name = team_b_name
            else:
                team_name = "Both teams"
            active_filters.append((info['icon'], info['name'], f"{team_name}: BTTS inflation"))
        elif filters.get(filter_key, False):
            active_filters.append((info['icon'], info['name'], info['desc']))
    
    if active_filters:
        for icon, name, desc in active_filters:
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"### {icon}")
            with col2:
                st.markdown(f"**{name}**")
                st.caption(desc)
            st.divider()
    else:
        st.info("No extreme filters triggered. Analysis based on team form comparison.")

def render_script_panel(script: Dict[str, Any]):
    """Display match script"""
    st.subheader("📝 Match Analysis & Predictions")
    
    # Confidence
    confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
    confidence_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{confidence_text[script['confidence']]} {confidence_colors[script['confidence']]}")
    
    with col2:
        primary_count = len(script['primary_bets'])
        st.metric("Primary Recommendations", primary_count if primary_count > 0 else "None")
    
    with col3:
        if script['predicted_score_range']:
            st.metric("Most Likely Score", script['predicted_score_range'][0])
    
    # Narrative
    st.markdown("**Match Narrative**")
    st.info(script['match_narrative'])
    
    # Warnings
    if script.get('warnings'):
        st.markdown("**⚠️ Statistical Warnings**")
        for warning in script['warnings']:
            st.warning(warning)
    
    # Key insights
    if script.get('key_insights'):
        st.markdown("**🔑 Key Insights**")
        for insight in script['key_insights']:
            st.success(f"• {insight}")
    
    # Predicted scores
    if script['predicted_score_range']:
        st.markdown("**🎯 Predicted Score Range**")
        score_cols = st.columns(min(5, len(script['predicted_score_range'])))
        for idx, score in enumerate(script['predicted_score_range']):
            with score_cols[idx % len(score_cols)]:
                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'><h3>{score}</h3></div>", 
                          unsafe_allow_html=True)

def render_betting_slip(slip: Dict[str, Any]):
    """Display betting slip"""
    st.subheader("💰 Betting Recommendations")
    
    if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] == 'NO_BET':
        st.warning("### ⚠️ NO BET RECOMMENDED")
        st.info("Confidence level too low for betting recommendations.")
        return
    
    if not slip['recommended_bets']:
        st.info("No betting recommendations for this match.")
        return
    
    # Display each recommendation
    for bet in slip['recommended_bets']:
        priority_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
        
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                bet_type_pretty = bet['type'].replace('_', ' ').title()
                st.markdown(f"### {bet_type_pretty}")
            
            with col2:
                priority_display = f"{priority_colors.get(bet['priority'], '⚪')} {bet['priority'].title()} Priority"
                st.markdown(priority_display)
            
            with col3:
                if bet['type'] in slip['stake_suggestions']:
                    stake = slip['stake_suggestions'][bet['type']]
                    st.metric("Stake", f"{stake} units")
            
            # Reason and details
            reason_text = bet['reason'].replace('_', ' ').title()
            st.caption(f"**Reason:** {reason_text}")
            
            if 'edge_percent' in bet:
                st.caption(f"**Value Edge:** {bet['edge_percent']}%")
            if 'odds' in bet:
                st.caption(f"**Odds:** {bet['odds']}")
            
            st.divider()
    
    # Correct score suggestions
    if slip.get('correct_score_suggestions'):
        st.markdown("**🎯 Correct Score Suggestions**")
        cols = st.columns(len(slip['correct_score_suggestions']))
        for idx, score in enumerate(slip['correct_score_suggestions']):
            with cols[idx]:
                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: #e6f7ff; border-radius: 5px; border: 1px solid #91d5ff;'><h4>{score}</h4></div>", 
                          unsafe_allow_html=True)

def render_value_bets(value_bets: List[Dict[str, Any]]):
    """Display value bets"""
    if not value_bets:
        return
    
    st.subheader("💎 Value Bets Detected")
    
    for bet in value_bets:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                bet_type_pretty = bet['bet_type'].replace('_', ' ').title()
                st.markdown(f"**{bet_type_pretty}**")
            
            with col2:
                st.metric("Odds", f"{bet['odds']:.2f}")
            
            with col3:
                edge = bet['edge_percent']
                st.metric("Edge", f"{edge:.1f}%")
            
            with col4:
                if edge > 25:
                    st.success("🔥 Strong Value")
                elif edge > 15:
                    st.info("📈 Good Value")
                else:
                    st.warning("⚖️ Marginal Value")
            
            # Probability comparison
            st.caption(f"Our Probability: {bet['our_probability']:.1f}% | Market Implied: {bet['implied_probability']:.1f}%")
            st.caption(f"**Reason:** {bet['reason'].replace('_', ' ').title()}")
            st.divider()

def render_match_info(match_info: Dict[str, Any]):
    """Display match header"""
    st.header(f"⚽ {match_info['team_a']} vs {match_info['team_b']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(match_info['team_a'])
        if match_info.get('league_position_a'):
            st.caption(f"League Position: {match_info['league_position_a']}")
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
        venue_text = f"{match_info['team_a']} at Home" if match_info['venue'] == 'home' else f"{match_info['team_b']} at Home"
        st.caption(venue_text)
    
    with col3:
        st.subheader(match_info['team_b'])
        if match_info.get('league_position_b'):
            st.caption(f"League Position: {match_info['league_position_b']}")
    
    st.markdown("---")

def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Betting Analytics Engine v3.1 - Team Stats Only",
        page_icon="⚽",
        layout="wide"
    )
    
    # Main title
    st.title("⚽ Betting Analytics Engine v3.1")
    st.markdown("**Team Statistics Comparison Only • No H2H Data Required**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        sidebar_result = render_sidebar()
    
    if not sidebar_result or sidebar_result[0] is None:
        st.info("👈 Select a league and teams to begin analysis")
        
        # Show available data info
        with st.expander("📁 Data Requirements"):
            st.markdown("""
            **Required CSV Format:**
            - Place CSV files in `leagues/` folder
            - File naming: `league_name.csv` (e.g., `eredivisie.csv`)
            - Required columns: `Team`, `Form W`, `Form D`, `Form L`, `Last 6 Goals`
            - Optional: `Overall Matches`, `Overall Goals`, `Overall CS%`, `Overall BTTS%`
            """)
        return
    
    selected_league, team_a, team_b, market_odds, is_team_a_home = sidebar_result
    
    # Load data
    teams_df = load_league_teams(selected_league)
    if teams_df is None:
        return
    
    # Parse team data
    team_a_data = parse_team_data(teams_df, team_a)
    team_b_data = parse_team_data(teams_df, team_b)
    
    if not team_a_data or not team_b_data:
        st.error("Failed to load team data. Check CSV format.")
        return
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        isTeamAHome=is_team_a_home,
        marketOdds=market_odds
    )
    
    # Analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Analyzing team statistics..."):
            try:
                # Initialize and run engine
                engine = BettingAnalyticsEngine()
                result = engine.analyze_match(match_context)
                
                # Display results
                st.markdown("---")
                
                # Match info
                render_match_info(result['match_info'])
                
                # Tabs for different sections
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Statistics", 
                    "🔍 Filters", 
                    "📝 Analysis", 
                    "💰 Betting", 
                    "💎 Value"
                ])
                
                with tab1:
                    render_metrics_panel(result['calculated_metrics'])
                
                with tab2:
                    render_filters_panel(
                        result['filters_triggered'], 
                        result['match_info']['team_a'], 
                        result['match_info']['team_b']
                    )
                
                with tab3:
                    render_script_panel(result['match_script'])
                
                with tab4:
                    render_betting_slip(result['betting_slip'])
                
                with tab5:
                    if market_odds:
                        render_value_bets(result['value_bets'])
                    else:
                        st.info("Enable market odds in sidebar for value bet detection")
                
                # Raw data for debugging
                with st.expander("📋 View Raw Analysis Data"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                st.info("Check your CSV format and ensure all required columns are present.")
    
    else:
        # Preview before analysis
        st.info("👆 Click 'Run Analysis' to generate predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Team A: {team_a}")
            if team_a_data:
                st.write(f"**Form:** {team_a_data.last6['wins']}W-{team_a_data.last6['draws']}D-{team_a_data.last6['losses']}L")
                st.write(f"**Goals (Last 6):** {team_a_data.last6['goalsScored']}")
                if team_a_data.last6['cleanSheets'] > 0:
                    st.write(f"**Clean Sheets:** {team_a_data.last6['cleanSheets']}/6")
                if team_a_data.last6['bttsMatches'] > 0:
                    st.write(f"**BTTS Matches:** {team_a_data.last6['bttsMatches']}/6")
        
        with col2:
            st.subheader(f"Team B: {team_b}")
            if team_b_data:
                st.write(f"**Form:** {team_b_data.last6['wins']}W-{team_b_data.last6['draws']}D-{team_b_data.last6['losses']}L")
                st.write(f"**Goals (Last 6):** {team_b_data.last6['goalsScored']}")
                if team_b_data.last6['cleanSheets'] > 0:
                    st.write(f"**Clean Sheets:** {team_b_data.last6['cleanSheets']}/6")
                if team_b_data.last6['bttsMatches'] > 0:
                    st.write(f"**BTTS Matches:** {team_b_data.last6['bttsMatches']}/6")
        
        st.markdown("---")
        st.caption("Analysis will compare team statistics only. No head-to-head history used.")

# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
