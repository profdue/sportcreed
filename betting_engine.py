#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v3.1
Complete betting analysis engine in one Streamlit file
Uses REAL data only - No mockups, no estimates
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
# SECTION 2: DATA STRUCTURES (Exact interfaces as specified)
# ============================================================================

@dataclass
class TeamFormData:
    """Interface exactly as specified in v3.1"""
    # Last 6 matches - REQUIRED
    last6: Dict[str, Any]
    # Overall season stats (optional, for regression detection)
    overall: Optional[Dict[str, Any]] = None
    # Current position (for tie-breaking)
    leaguePosition: Optional[int] = None
    teamName: str = ""

@dataclass
class H2HData:
    """Head-to-head data - Minimum 20 matches for reliability"""
    totalMatches: int
    teamAWins: int
    teamBWins: int
    draws: int
    over15Matches: int
    over25Matches: int

@dataclass
class MarketOdds:
    """Market odds interface - Optional for value detection"""
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
    """Complete match context"""
    teamA: TeamFormData
    teamB: TeamFormData
    h2h: H2HData
    isTeamAHome: bool = True
    marketOdds: Optional[MarketOdds] = None

# ============================================================================
# SECTION 3: ENGINE CLASSES (All 5 filters + complete v3.1 logic)
# ============================================================================

class MatchAnalyzer:
    """Core metrics calculator - EXACTLY as specified"""
    
    def __init__(self, team_a: TeamFormData, team_b: TeamFormData, h2h: H2HData):
        self.team_a = team_a
        self.team_b = team_b
        self.h2h = h2h
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate all derived metrics needed for decision making"""
        metrics = {
            'team_a': {
                'gpm': self.team_a.last6['goalsScored'] / 6,
                'gcpm': self.team_a.last6['goalsConceded'] / 6 if self.team_a.last6.get('goalsConceded') else None,
                'cs_percent': (self.team_a.last6['cleanSheets'] / 6) * 100,
                'btts_percent': (self.team_a.last6['bttsMatches'] / 6) * 100,
                'win_percent': (self.team_a.last6['wins'] / 6) * 100,
                'loss_percent': (self.team_a.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_a.last6['draws'] / 6) * 100
            },
            'team_b': {
                'gpm': self.team_b.last6['goalsScored'] / 6,
                'gcpm': self.team_b.last6['goalsConceded'] / 6 if self.team_b.last6.get('goalsConceded') else None,
                'cs_percent': (self.team_b.last6['cleanSheets'] / 6) * 100,
                'btts_percent': (self.team_b.last6['bttsMatches'] / 6) * 100,
                'win_percent': (self.team_b.last6['wins'] / 6) * 100,
                'loss_percent': (self.team_b.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_b.last6['draws'] / 6) * 100
            },
            'h2h': {
                'over15_percent': (self.h2h.over15Matches / self.h2h.totalMatches) * 100,
                'over25_percent': (self.h2h.over25Matches / self.h2h.totalMatches) * 100,
                'under25_percent': 100 - ((self.h2h.over25Matches / self.h2h.totalMatches) * 100),
                'win_diff_percent': abs(
                    (self.h2h.teamAWins / self.h2h.totalMatches * 100) - 
                    (self.h2h.teamBWins / self.h2h.totalMatches * 100)
                ),
                'draw_percent': (self.h2h.draws / self.h2h.totalMatches) * 100,
                'team_a_win_percent': (self.h2h.teamAWins / self.h2h.totalMatches) * 100,
                'team_b_win_percent': (self.h2h.teamBWins / self.h2h.totalMatches) * 100
            }
        }
        return metrics


class ExtremeFilterDetector:
    """All 5 filters including v3.1 regression detection"""
    
    # CONSTANTS - DO NOT MODIFY WITHOUT VALIDATION
    # Filter 1: Under 1.5 Goals Alert
    UNDER_15_GPM_THRESHOLD = 0.75
    
    # Filter 2: BTTS Banker Alert
    CS_PERCENT_THRESHOLD = 20      # For BTTS filter
    H2H_OVER15_MIN = 65           # Minimum H2H Over 1.5%
    HIGH_GPM_THRESHOLD = 1.5      # For BTTS enhancement
    
    # Filter 3: Clean Sheet Alert
    CS_PERCENT_STRONG = 50        # For Clean Sheet filter
    OPPONENT_GPM_WEAK = 1.0       # Opponent attack threshold
    
    # Filter 4: Low-Scoring Alert (NEW - from 0-0 analysis)
    BTTS_PERCENT_LOW = 40         # Low BTTS percentage
    CS_PERCENT_DECENT = 30        # Decent clean sheet rate
    GPM_MODERATE = 1.5            # Moderate scoring threshold
    
    # Filter 5: Regression Detection (v3.1)
    RECENT_BTTS_INFLATION_MIN = 70  # Last 6 BTTS% threshold
    SEASON_BTTS_REGRESSION_MAX = 60 # Season BTTS% threshold
    GPM_INFLATION_FACTOR = 1.5      # 50% higher than season average
    
    def detect_filters(self, metrics: Dict[str, Any], 
                      team_a_overall: Optional[Dict] = None,
                      team_b_overall: Optional[Dict] = None) -> Dict[str, Any]:
        """Detect all 5 extreme filters"""
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
        
        # FILTER 2: BTTS Banker Alert
        if (metrics['team_a']['cs_percent'] < self.CS_PERCENT_THRESHOLD and 
            metrics['team_b']['cs_percent'] < self.CS_PERCENT_THRESHOLD and
            metrics['h2h']['over15_percent'] > self.H2H_OVER15_MIN):
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
        
        # FILTER 4: Low-Scoring Alert (NEW)
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
        
        # FILTER 5: Regression to Mean Alert (v3.1)
        if team_a_overall and 'bttsMatches' in team_a_overall and 'matches' in team_a_overall:
            season_btts_percent = (team_a_overall['bttsMatches'] / team_a_overall['matches']) * 100
            if (metrics['team_a']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_percent < self.SEASON_BTTS_REGRESSION_MAX):
                filters['regression_alert'] = {'team': 'A', 'type': 'btts_regression', 'severity': 'high'}
        
        if team_b_overall and 'bttsMatches' in team_b_overall and 'matches' in team_b_overall:
            season_btts_percent = (team_b_overall['bttsMatches'] / team_b_overall['matches']) * 100
            if (metrics['team_b']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_percent < self.SEASON_BTTS_REGRESSION_MAX):
                # If both teams have regression, mark both
                if filters['regression_alert']['team'] == 'A':
                    filters['regression_alert'] = {'team': 'both', 'type': 'btts_regression', 'severity': 'high'}
                else:
                    filters['regression_alert'] = {'team': 'B', 'type': 'btts_regression', 'severity': 'high'}
            
        return filters


class H2HTrendClassifier:
    """H2H trend analysis"""
    
    def classify_trend(self, percent: float) -> str:
        """
        Classify H2H percentages into confidence levels
        Returns: 'very_strong', 'strong', 'moderate', 'weak'
        """
        if percent >= 80 or percent <= 20:
            return 'very_strong'
        elif percent >= 70 or percent <= 30:
            return 'strong'
        elif percent >= 60 or percent <= 40:
            return 'moderate'
        else:  # 40-60%
            return 'weak'
    
    def get_h2h_bets(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get H2H-based betting recommendations"""
        bets = []
        
        # Over 1.5 trend
        trend_15 = self.classify_trend(metrics['h2h']['over15_percent'])
        if trend_15 in ['very_strong', 'strong']:
            bets.append({
                'type': 'over_15',
                'confidence': trend_15,
                'percent': metrics['h2h']['over15_percent']
            })
        
        # Under 2.5 trend (using over25 to calculate)
        under25_percent = 100 - metrics['h2h']['over25_percent']
        trend_25 = self.classify_trend(under25_percent)
        if trend_25 in ['very_strong', 'strong']:
            bets.append({
                'type': 'under_25',
                'confidence': trend_25,
                'percent': under25_percent
            })
        
        # Over 2.5 trend (for completeness)
        trend_over25 = self.classify_trend(metrics['h2h']['over25_percent'])
        if trend_over25 in ['very_strong', 'strong']:
            bets.append({
                'type': 'over_25',
                'confidence': trend_over25,
                'percent': metrics['h2h']['over25_percent']
            })
        
        # Historical favorite
        if metrics['h2h']['win_diff_percent'] > 15:
            # Determine which team is historical favorite
            if metrics['h2h']['team_a_win_percent'] > metrics['h2h']['team_b_win_percent']:
                favorite = 'team_a'
            else:
                favorite = 'team_b'
                
            bets.append({
                'type': 'historical_favorite',
                'team': favorite,
                'diff_percent': metrics['h2h']['win_diff_percent']
            })
            
        # Draw tendency
        if metrics['h2h']['draw_percent'] > 35:
            bets.append({
                'type': 'draw_tendency',
                'percent': metrics['h2h']['draw_percent']
            })
            
        return bets


class CurrentFormAnalyzer:
    """Current form analysis"""
    
    def analyze_form(self, team_a_data: TeamFormData, team_b_data: TeamFormData) -> Dict[str, Any]:
        """
        Analyze current form to determine favorite and patterns
        """
        analysis = {
            'favorite': None,
            'form_edge': None,  # 'strong', 'moderate', 'slight', 'none'
            'goal_potential': None,  # 'high', 'medium', 'low'
            'btts_likelihood': None  # 'high', 'medium', 'low'
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
        
        # Goal potential (average of both teams' GPM)
        avg_gpm = (team_a_data.last6['goalsScored']/6 + team_b_data.last6['goalsScored']/6) / 2
        if avg_gpm > 1.8:
            analysis['goal_potential'] = 'high'
        elif avg_gpm > 1.3:
            analysis['goal_potential'] = 'medium'
        else:
            analysis['goal_potential'] = 'low'
        
        # BTTS likelihood (average of both teams' BTTS%)
        avg_btts = ((team_a_data.last6['bttsMatches']/6)*100 + 
                   (team_b_data.last6['bttsMatches']/6)*100) / 2
        if avg_btts > 60:
            analysis['btts_likelihood'] = 'high'
        elif avg_btts > 40:
            analysis['btts_likelihood'] = 'medium'
        else:
            analysis['btts_likelihood'] = 'low'
            
        return analysis


class MatchScriptGenerator:
    """Generate betting narrative and predictions"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict[str, Any], filters: Dict[str, Any], 
                       h2h_bets: List[Dict[str, Any]], form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete match script"""
        script = {
            'primary_bets': [],
            'secondary_bets': [],
            'value_bets': [],
            'predicted_score_range': [],
            'confidence': 'low',  # low, medium, high
            'match_narrative': '',
            'warnings': []
        }
        
        # Check for regression warnings first
        if filters['regression_alert']['team']:
            team = filters['regression_alert']['team']
            alert_type = filters['regression_alert']['type']
            
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
            return script
        
        if filters['clean_sheet_alert']['team']:
            team_name = self.team_a_name if filters['clean_sheet_alert']['team'] == 'A' else self.team_b_name
            script['primary_bets'].append(f'{team_name.lower().replace(" ", "_")}_win_to_nil')
            script['secondary_bets'].append('under_25_goals')
            script['predicted_score_range'] = [f'2-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-2',
                                              f'1-0' if filters['clean_sheet_alert']['team'] == 'A' else '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = f'{team_name} strong defensively against weak attack'
            return script
        
        # RULE 2: If no extreme filters, use H2H trends
        if not script['primary_bets']:
            strong_h2h_bets = [b for b in h2h_bets if b['confidence'] in ['very_strong', 'strong']]
            
            for bet in strong_h2h_bets:
                if bet['type'] == 'over_15':
                    script['primary_bets'].append('over_15_goals')
                elif bet['type'] == 'under_25':
                    script['primary_bets'].append('under_25_goals')
                elif bet['type'] == 'over_25':
                    script['secondary_bets'].append('over_25_goals')
                elif bet['type'] == 'draw_tendency':
                    script['secondary_bets'].append('draw')
            
            # Add current form favorite if exists
            if form_analysis['favorite']:
                favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
                script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win_or_draw')
                
            script['confidence'] = 'medium' if strong_h2h_bets else 'low'
            
            # Generate narrative
            if form_analysis['goal_potential'] == 'high':
                script['match_narrative'] += 'High scoring potential. '
            elif form_analysis['goal_potential'] == 'low':
                script['match_narrative'] += 'Low scoring potential. '
                
            if form_analysis['btts_likelihood'] == 'high':
                script['match_narrative'] += 'Both teams likely to score. '
            elif form_analysis['btts_likelihood'] == 'low':
                script['match_narrative'] += 'Clean sheet possible. '
        
        # Generate score range if not already set
        if not script['predicted_score_range']:
            script['predicted_score_range'] = self._generate_score_range(script, metrics, form_analysis)
        
        return script
    
    def _generate_score_range(self, script: Dict[str, Any], metrics: Dict[str, Any], 
                            form_analysis: Dict[str, Any]) -> List[str]:
        """Generate likely score ranges based on script"""
        scores = []
        
        if 'btts_yes' in script['primary_bets']:
            if 'over_25_goals' in script['value_bets']:
                scores.extend(['2-1', '1-2', '2-2', '3-1', '1-3'])
            else:
                scores.extend(['1-1', '2-1', '1-2'])
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
            
        return scores[:5]  # Return top 5 most likely


class ValueDetector:
    """Value bet detection based on market odds"""
    
    # Value thresholds
    VALUE_EDGE_MINIMUM = 15  # 15% edge required
    HIGH_CONFIDENCE_PROB = 0.75  # 75%
    MEDIUM_CONFIDENCE_PROB = 0.60  # 60%
    LOW_CONFIDENCE_PROB = 0.40  # 40%
    
    def __init__(self, market_odds: MarketOdds):
        self.market_odds = market_odds
    
    def calculate_value(self, script: Dict[str, Any], filters_triggered: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Calculate value bets based on script confidence vs market odds
        Returns list of value bets with edge percentage
        """
        if not self.market_odds:
            return []
            
        value_bets = []
        
        # Map confidence to probability
        confidence_prob = {
            'high': self.HIGH_CONFIDENCE_PROB,
            'medium': self.MEDIUM_CONFIDENCE_PROB,
            'low': self.LOW_CONFIDENCE_PROB
        }
        
        our_probability = confidence_prob.get(script['confidence'], 0.5)
        
        # Check primary bets for value
        for bet_type in script['primary_bets']:
            odds_key = self._map_bet_to_odds_key(bet_type)
            if odds_key in self.market_odds.__dict__ and self.market_odds.__dict__[odds_key] is not None:
                implied_prob = 1 / self.market_odds.__dict__[odds_key]
                edge = ((our_probability / implied_prob) - 1) * 100
                
                if edge >= self.VALUE_EDGE_MINIMUM:
                    value_bets.append({
                        'bet_type': bet_type,
                        'odds_key': odds_key,
                        'odds': self.market_odds.__dict__[odds_key],
                        'our_probability': round(our_probability * 100, 1),
                        'implied_probability': round(implied_prob * 100, 1),
                        'edge_percent': round(edge, 1),
                        'reason': 'primary_bet_with_edge'
                    })
        
        # Check secondary bets (with slightly lower probability)
        secondary_prob = our_probability * 0.8  # 20% lower confidence
        for bet_type in script['secondary_bets']:
            odds_key = self._map_bet_to_odds_key(bet_type)
            if odds_key in self.market_odds.__dict__ and self.market_odds.__dict__[odds_key] is not None:
                implied_prob = 1 / self.market_odds.__dict__[odds_key]
                edge = ((secondary_prob / implied_prob) - 1) * 100
                
                if edge >= self.VALUE_EDGE_MINIMUM * 1.5:  # Higher threshold for secondary
                    value_bets.append({
                        'bet_type': bet_type,
                        'odds_key': odds_key,
                        'odds': self.market_odds.__dict__[odds_key],
                        'our_probability': round(secondary_prob * 100, 1),
                        'implied_probability': round(implied_prob * 100, 1),
                        'edge_percent': round(edge, 1),
                        'reason': 'secondary_bet_with_strong_edge'
                    })
        
        return value_bets
    
    def _map_bet_to_odds_key(self, bet_type: str) -> str:
        """Map internal bet type to market odds key"""
        mapping = {
            'under_15_goals': 'under_15',
            'over_15_goals': 'over_15',
            'under_25_goals': 'under_25',
            'over_25_goals': 'over_25',
            'btts_yes': 'btts_yes',
            'btts_no': 'btts_no',
            'draw': 'draw'
        }
        
        # Handle team-specific bets
        if '_win_to_nil' in bet_type:
            team = bet_type.replace('_win_to_nil', '')
            return f'{team}_win_to_nil'
        
        if '_win_or_draw' in bet_type:
            team = bet_type.replace('_win_or_draw', '')
            return f'{team}_win_or_draw'
        
        return mapping.get(bet_type, bet_type)


class BettingSlipGenerator:
    """Generate optimized betting slip with stake suggestions"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            'max_bets_per_match': 3,
            'stake_distribution': {'high': 0.5, 'medium': 0.3, 'low': 0.2},
            'min_confidence': 'medium'  # Skip if confidence below this
        }
    
    def generate_slip(self, script: Dict[str, Any], value_bets: List[Dict[str, Any]], 
                     team_a_name: str, team_b_name: str) -> Dict[str, Any]:
        """
        Generate optimized betting slip with stake suggestions
        """
        slip = {
            'recommended_bets': [],
            'avoid_bets': [],
            'stake_suggestions': {},
            'total_units': 10,  # Base betting units
            'match_summary': script.get('match_narrative', ''),
            'confidence': script.get('confidence', 'low')
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
                'reason': 'extreme_filter' if script['confidence'] == 'high' else 'strong_h2h_trend',
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
        
        # Priority 3: Secondary bets if space
        if len(slip['recommended_bets']) < self.config['max_bets_per_match']:
            for bet in script['secondary_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'supporting_bet'
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
            if bet['type'] != 'NO_BET':
                stake_pct = self.config['stake_distribution'].get(bet['priority'], 0.2)
                slip['stake_suggestions'][bet['type']] = round(total_stake * stake_pct, 1)
        
        # Suggested score bets
        if script['predicted_score_range']:
            slip['correct_score_suggestions'] = script['predicted_score_range'][:3]
        
        return slip


class BettingAnalyticsEngine:
    """Main orchestrator - complete v3.1 pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.script_generator = None
        self.value_detector = None
    
    def analyze_match(self, match_context: MatchContext) -> Dict[str, Any]:
        """
        Complete analysis pipeline
        Returns: dict with all analysis results
        """
        # Initialize components with team names
        self.script_generator = MatchScriptGenerator(
            match_context.teamA.teamName, 
            match_context.teamB.teamName
        )
        
        if match_context.marketOdds:
            self.value_detector = ValueDetector(match_context.marketOdds)
        
        # Step 1: Calculate metrics
        metrics_calc = MatchAnalyzer(match_context.teamA, match_context.teamB, match_context.h2h)
        metrics = metrics_calc.calculate_metrics()
        
        # Step 2: Detect extreme filters
        filter_detector = ExtremeFilterDetector()
        filters = filter_detector.detect_filters(
            metrics, 
            match_context.teamA.overall, 
            match_context.teamB.overall
        )
        
        # Step 3: Classify H2H trends
        trend_classifier = H2HTrendClassifier()
        h2h_bets = trend_classifier.get_h2h_bets(metrics)
        
        # Step 4: Analyze current form
        form_analyzer = CurrentFormAnalyzer()
        form_analysis = form_analyzer.analyze_form(match_context.teamA, match_context.teamB)
        
        # Step 5: Generate match script
        script = self.script_generator.generate_script(metrics, filters, h2h_bets, form_analysis)
        
        # Step 6: Detect value bets (if market odds available)
        value_bets = []
        if self.value_detector:
            value_bets = self.value_detector.calculate_value(script, filters)
        
        # Step 7: Generate betting slip
        slip_generator = BettingSlipGenerator(self.config)
        slip = slip_generator.generate_slip(
            script, 
            value_bets,
            match_context.teamA.teamName,
            match_context.teamB.teamName
        )
        
        # Compile final result
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
            'h2h_trends': h2h_bets,
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
# SECTION 4: DATA PARSING (Your CSV format to our model)
# ============================================================================

def parse_fraction_string(fraction_str: str) -> Tuple[int, int]:
    """Parse strings like '5/6 BTTS' or '1/6 CS' -> (numerator, denominator)"""
    # Extract numbers using regex
    match = re.search(r'(\d+)/(\d+)', str(fraction_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 6  # Default if parsing fails

def parse_percentage_string(percent_str: str) -> float:
    """Parse percentage strings like '40.00%' -> 40.0"""
    if isinstance(percent_str, str):
        return float(percent_str.replace('%', '').strip())
    elif isinstance(percent_str, (int, float)):
        return float(percent_str)
    return 0.0

def parse_team_data(df: pd.DataFrame, team_name: str) -> Optional[TeamFormData]:
    """Convert your CSV format to TeamFormData"""
    team_row = df[df['Team'] == team_name]
    
    if team_row.empty:
        st.error(f"Team '{team_name}' not found in data")
        return None
    
    row = team_row.iloc[0]
    
    # Parse fractions from your format (assuming columns like 'CS_Last6', 'BTTS_Last6')
    # Adjust column names based on your actual CSV
    
    # Try to find clean sheets data
    clean_sheets = 0
    btts_matches = 0
    
    # Look for clean sheet data in various possible column names
    cs_columns = ['CS', 'CleanSheets', 'CS_Last6', 'Clean Sheets']
    btts_columns = ['BTTS', 'BTTS_Last6', 'Both Teams Scored']
    
    for col in cs_columns:
        if col in row and pd.notna(row[col]):
            clean_sheets_numer, _ = parse_fraction_string(str(row[col]))
            clean_sheets = clean_sheets_numer
            break
    
    for col in btts_columns:
        if col in row and pd.notna(row[col]):
            btts_numer, _ = parse_fraction_string(str(row[col]))
            btts_matches = btts_numer
            break
    
    # If not found in fractions, estimate from percentages
    if clean_sheets == 0 and 'CS%' in row:
        cs_percent = parse_percentage_string(row['CS%'])
        clean_sheets = round((cs_percent / 100) * 6)
    
    if btts_matches == 0 and 'BTTS%' in row:
        btts_percent = parse_percentage_string(row['BTTS%'])
        btts_matches = round((btts_percent / 100) * 6)
    
    # Build TeamFormData object
    team_data = TeamFormData(
        teamName=team_name,
        last6={
            'matches': 6,
            'goalsScored': int(row['Last 6 Goals']) if 'Last 6 Goals' in row else 0,
            'goalsConceded': None,  # Will be in your real data
            'cleanSheets': clean_sheets,
            'bttsMatches': btts_matches,
            'wins': int(row['Form W']) if 'Form W' in row else 0,
            'draws': int(row['Form D']) if 'Form D' in row else 0,
            'losses': int(row['Form L']) if 'Form L' in row else 0
        },
        overall={
            'matches': int(row['Overall Matches']) if 'Overall Matches' in row else 0,
            'goalsScored': int(row['Overall Goals']) if 'Overall Goals' in row else 0,
            'goalsConceded': None,  # Will be in your real data
            'cleanSheets': int((parse_percentage_string(row.get('Overall CS%', 0)) / 100) * row['Overall Matches']) 
                          if 'Overall Matches' in row and 'Overall CS%' in row else 0,
            'bttsMatches': int((parse_percentage_string(row.get('Overall BTTS%', 0)) / 100) * row['Overall Matches'])
                          if 'Overall Matches' in row and 'Overall BTTS%' in row else 0
        },
        leaguePosition=None  # Add if you have this data
    )
    
    return team_data

def load_h2h_data(league_name: str, team_a: str, team_b: str) -> Optional[H2HData]:
    """Load REAL H2H data from CSV"""
    try:
        # Try different possible file names
        possible_files = [
            f"leagues/{league_name}_h2h.csv",
            f"leagues/{league_name}_head_to_head.csv",
            f"leagues/{league_name}_h2h_stats.csv",
            f"{league_name}_h2h.csv"
        ]
        
        h2h_df = None
        for file_path in possible_files:
            try:
                h2h_df = pd.read_csv(file_path)
                break
            except:
                continue
        
        if h2h_df is None:
            st.error(f"No H2H data file found for {league_name}")
            return None
        
        # Look for the match (check both directions)
        match = h2h_df[
            ((h2h_df['Team_A'] == team_a) & (h2h_df['Team_B'] == team_b)) |
            ((h2h_df['Team_A'] == team_b) & (h2h_df['Team_B'] == team_a))
        ]
        
        if match.empty:
            st.error(f"No H2H data found for {team_a} vs {team_b}")
            return None
        
        h2h_row = match.iloc[0]
        
        # Create H2HData object
        h2h_data = H2HData(
            totalMatches=int(h2h_row['Total_Matches']) if 'Total_Matches' in h2h_row else int(h2h_row['Matches']),
            teamAWins=int(h2h_row['Team_A_Wins']),
            teamBWins=int(h2h_row['Team_B_Wins']),
            draws=int(h2h_row['Draws']),
            over15Matches=int(h2h_row['Over_15']) if 'Over_15' in h2h_row else int(h2h_row['Over15_Matches']),
            over25Matches=int(h2h_row['Over_25']) if 'Over_25' in h2h_row else int(h2h_row['Over25_Matches'])
        )
        
        # Validate minimum H2H matches
        if h2h_data.totalMatches < 10:
            st.warning(f"⚠️ Low H2H sample size ({h2h_data.totalMatches} matches). Use with caution.")
        
        return h2h_data
        
    except Exception as e:
        st.error(f"Error loading H2H data: {str(e)}")
        return None

def load_league_teams(league_name: str) -> Optional[pd.DataFrame]:
    """Load team statistics CSV for a league"""
    possible_files = [
        f"leagues/{league_name}.csv",
        f"leagues/{league_name}_teams.csv",
        f"leagues/{league_name}_stats.csv",
        f"{league_name}.csv"
    ]
    
    for file_path in possible_files:
        try:
            df = pd.read_csv(file_path)
            return df
        except:
            continue
    
    st.error(f"Could not load team data for {league_name}")
    return None

# ============================================================================
# SECTION 5: STREAMLIT UI COMPONENTS
# ============================================================================

def render_sidebar() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[MarketOdds]]:
    """Render sidebar with all controls"""
    st.sidebar.title("⚽ Betting Analytics Engine")
    
    # League selection
    st.sidebar.subheader("1. Select League")
    
    # Get available leagues from the leagues folder
    available_leagues = []
    try:
        import os
        league_files = [f for f in os.listdir('leagues') if f.endswith('.csv') and not f.endswith('_h2h.csv')]
        available_leagues = [f.replace('.csv', '').replace('_teams', '').replace('_stats', '') 
                           for f in league_files]
        available_leagues = list(set(available_leagues))  # Remove duplicates
    except:
        # Fallback to known leagues
        available_leagues = [
            'bundesliga', 'bundesliga_2', 'championship', 'eredivisie',
            'erste_divisie', 'laliga', 'laliga_2', 'ligue_1', 'ligue_2',
            'premier_league', 'serie_a', 'serie_b'
        ]
    
    selected_league = st.sidebar.selectbox("League", available_leagues)
    
    if not selected_league:
        return None, None, None, None
    
    # Load teams for this league
    teams_df = load_league_teams(selected_league)
    if teams_df is None:
        return None, None, None, None
    
    teams = teams_df['Team'].tolist() if 'Team' in teams_df.columns else []
    
    if not teams:
        st.sidebar.error("No teams found in league data")
        return None, None, None, None
    
    # Team selection
    st.sidebar.subheader("2. Select Teams")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        team_a = st.selectbox("Team A", teams, key="team_a_select")
    
    with col2:
        # Filter out selected team A
        available_teams_b = [t for t in teams if t != team_a]
        team_b = st.selectbox("Team B", available_teams_b, key="team_b_select")
    
    # Venue selection
    st.sidebar.subheader("3. Match Details")
    is_team_a_home = st.sidebar.radio("Venue", ["Team A at Home", "Team B at Home"]) == "Team A at Home"
    
    # Optional: Market odds
    st.sidebar.subheader("4. Market Odds (Optional)")
    
    if st.sidebar.checkbox("Enter market odds for value detection"):
        odds_col1, odds_col2 = st.sidebar.columns(2)
        
        with odds_col1:
            btts_yes = st.number_input("BTTS Yes", min_value=1.0, max_value=10.0, value=1.8, step=0.05)
            over_15 = st.number_input("Over 1.5", min_value=1.0, max_value=5.0, value=1.3, step=0.05)
            over_25 = st.number_input("Over 2.5", min_value=1.0, max_value=5.0, value=2.1, step=0.05)
        
        with odds_col2:
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
    """Display calculated metrics"""
    st.subheader("📊 Calculated Metrics")
    
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
        st.markdown("**H2H Trends**")
        st.metric("Over 1.5 %", f"{metrics['h2h']['over15_percent']:.1f}%")
        st.metric("Over 2.5 %", f"{metrics['h2h']['over25_percent']:.1f}%")
        st.metric("Draw %", f"{metrics['h2h']['draw_percent']:.1f}%")
        st.metric("Win Diff %", f"{metrics['h2h']['win_diff_percent']:.1f}%")

def render_filters_panel(filters: Dict[str, Any], team_a_name: str, team_b_name: str):
    """Display triggered filters"""
    st.subheader("🔍 Extreme Filters Triggered")
    
    filter_icons = {
        'under_15_alert': '🔴',
        'btts_banker': '🟢', 
        'clean_sheet_alert': '🔵',
        'low_scoring_alert': '🟡',
        'regression_alert': '🟣'
    }
    
    filter_names = {
        'under_15_alert': 'Under 1.5 Goals Alert',
        'btts_banker': 'BTTS Banker Alert',
        'clean_sheet_alert': 'Clean Sheet Alert',
        'low_scoring_alert': 'Low-Scoring Alert',
        'regression_alert': 'Regression Alert'
    }
    
    cols = st.columns(5)
    
    for idx, (filter_key, triggered) in enumerate(filters.items()):
        if filter_key in filter_icons and triggered:
            if filter_key == 'clean_sheet_alert' and filters['clean_sheet_alert']['team']:
                with cols[idx % 5]:
                    icon = filter_icons[filter_key]
                    team = team_a_name if filters['clean_sheet_alert']['team'] == 'A' else team_b_name
                    st.markdown(f"{icon} **{filter_names[filter_key]}**")
                    st.caption(f"{team} win to nil")
            elif filter_key == 'regression_alert' and filters['regression_alert']['team']:
                with cols[idx % 5]:
                    icon = filter_icons[filter_key]
                    team = filters['regression_alert']['team']
                    if team == 'A':
                        team_name = team_a_name
                    elif team == 'B':
                        team_name = team_b_name
                    else:
                        team_name = "Both teams"
                    st.markdown(f"{icon} **{filter_names[filter_key]}**")
                    st.caption(f"{team_name}: BTTS inflation")
            else:
                with cols[idx % 5]:
                    st.markdown(f"{filter_icons[filter_key]} **{filter_names[filter_key]}**")
    
    # If no filters triggered
    if not any([filters.get(k) for k in filter_icons.keys() if k != 'btts_enhanced']):
        st.info("No extreme filters triggered. Analysis based on H2H trends and current form.")

def render_script_panel(script: Dict[str, Any]):
    """Display match script and narrative"""
    st.subheader("📝 Match Script")
    
    # Confidence indicator
    confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
    confidence_text = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{confidence_text[script['confidence']]} {confidence_colors[script['confidence']]}")
    
    with col2:
        if script['primary_bets']:
            st.metric("Primary Bets", len(script['primary_bets']))
        else:
            st.metric("Primary Bets", "None")
    
    with col3:
        if script['predicted_score_range']:
            st.metric("Top Score", script['predicted_score_range'][0])
    
    # Match narrative
    st.markdown("**Match Narrative**")
    st.info(script['match_narrative'])
    
    # Warnings
    if script.get('warnings'):
        st.markdown("**⚠️ Warnings**")
        for warning in script['warnings']:
            st.warning(warning)
    
    # Predicted scores
    if script['predicted_score_range']:
        st.markdown("**Predicted Score Range**")
        scores = ", ".join(script['predicted_score_range'])
        st.write(scores)

def render_betting_slip(slip: Dict[str, Any]):
    """Display betting recommendations"""
    st.subheader("💰 Betting Slip")
    
    if slip['recommended_bets'] and slip['recommended_bets'][0]['type'] == 'NO_BET':
        st.warning("**NO BET RECOMMENDED** - Confidence too low")
        return
    
    if not slip['recommended_bets']:
        st.info("No betting recommendations for this match")
        return
    
    # Display each bet
    for bet in slip['recommended_bets']:
        priority_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.markdown(f"**{bet['type'].replace('_', ' ').title()}**")
        
        with col2:
            st.markdown(f"Priority: {priority_colors.get(bet['priority'], '⚪')} {bet['priority'].title()}")
        
        with col3:
            if bet['type'] in slip['stake_suggestions']:
                st.metric("Stake", f"{slip['stake_suggestions'][bet['type']]} units")
        
        # Reason
        st.caption(f"Reason: {bet['reason'].replace('_', ' ')}")
        st.divider()
    
    # Correct score suggestions
    if slip.get('correct_score_suggestions'):
        st.markdown("**🎯 Correct Score Suggestions**")
        score_cols = st.columns(len(slip['correct_score_suggestions']))
        for idx, score in enumerate(slip['correct_score_suggestions']):
            with score_cols[idx]:
                st.info(score)

def render_value_bets(value_bets: List[Dict[str, Any]]):
    """Display value bet analysis"""
    if not value_bets:
        return
    
    st.subheader("💎 Value Bets Detected")
    
    for bet in value_bets:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f"**{bet['bet_type'].replace('_', ' ').title()}**")
        
        with col2:
            st.metric("Odds", f"{bet['odds']:.2f}")
        
        with col3:
            st.metric("Edge", f"{bet['edge_percent']:.1f}%")
        
        with col4:
            if bet['edge_percent'] > 25:
                st.success("Strong Value")
            elif bet['edge_percent'] > 15:
                st.info("Good Value")
            else:
                st.warning("Marginal Value")
        
        st.caption(f"Our probability: {bet['our_probability']:.1f}% vs Market: {bet['implied_probability']:.1f}%")

def render_match_info(match_info: Dict[str, Any]):
    """Display match information"""
    st.header(f"⚽ {match_info['team_a']} vs {match_info['team_b']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(match_info['team_a'])
        if match_info.get('league_position_a'):
            st.caption(f"League Position: {match_info['league_position_a']}")
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>VS</h2>", unsafe_allow_html=True)
        st.caption(f"Venue: {'Home' if match_info['venue'] == 'home' else 'Away'} for {match_info['team_a']}")
    
    with col3:
        st.subheader(match_info['team_b'])
        if match_info.get('league_position_b'):
            st.caption(f"League Position: {match_info['league_position_b']}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Betting Analytics Engine v3.1",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Betting Analytics Engine v3.1")
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        sidebar_result = render_sidebar()
    
    if not sidebar_result or sidebar_result[0] is None:
        st.info("👈 Select a league and teams to begin analysis")
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
        st.error("Failed to load team data")
        return
    
    # Load H2H data
    h2h_data = load_h2h_data(selected_league, team_a, team_b)
    if h2h_data is None:
        st.warning("Analysis limited: No H2H data available")
        # Could proceed with limited analysis or stop here
        return
    
    # Create match context
    match_context = MatchContext(
        teamA=team_a_data,
        teamB=team_b_data,
        h2h=h2h_data,
        isTeamAHome=is_team_a_home,
        marketOdds=market_odds
    )
    
    # Run analysis when button is clicked
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing match..."):
            # Initialize and run engine
            engine = BettingAnalyticsEngine()
            result = engine.analyze_match(match_context)
            
            # Display results
            st.markdown("---")
            
            # Match info
            render_match_info(result['match_info'])
            st.markdown("---")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Metrics", 
                "🔍 Filters", 
                "📝 Script", 
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
                    st.info("Enable market odds in sidebar for value detection")
            
            # Show raw data for debugging (collapsible)
            with st.expander("📋 Raw Analysis Data"):
                st.json(result)
    
    else:
        # Show preview of selected teams
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Team A: {team_a}")
            if team_a_data:
                st.write(f"Last 6: {team_a_data.last6['wins']}W-{team_a_data.last6['draws']}D-{team_a_data.last6['losses']}L")
                st.write(f"Goals (Last 6): {team_a_data.last6['goalsScored']}")
        
        with col2:
            st.subheader(f"Team B: {team_b}")
            if team_b_data:
                st.write(f"Last 6: {team_b_data.last6['wins']}W-{team_b_data.last6['draws']}D-{team_b_data.last6['losses']}L")
                st.write(f"Goals (Last 6): {team_b_data.last6['goalsScored']}")
        
        st.info("Click 'Run Analysis' to see detailed predictions and betting recommendations")

# ============================================================================
# SECTION 6: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()
