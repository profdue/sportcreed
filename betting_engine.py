#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE v3.1 - TEAM STATS ONLY
Complete single-file implementation
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TeamFormData:
    """Team statistics for analysis"""
    teamName: str
    last6: Dict[str, Any]  # Last 6 matches data
    overall: Optional[Dict[str, Any]] = None  # Overall season stats
    leaguePosition: Optional[int] = None

@dataclass
class MarketOdds:
    """Market odds for value detection"""
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
    """Match analysis context"""
    teamA: TeamFormData
    teamB: TeamFormData
    isTeamAHome: bool = True
    marketOdds: Optional[MarketOdds] = None

# ============================================================================
# ENGINE CORE
# ============================================================================

class MatchAnalyzer:
    """Analyze match using team statistics only"""
    
    def __init__(self, team_a: TeamFormData, team_b: TeamFormData):
        self.team_a = team_a
        self.team_b = team_b
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate key metrics from team stats"""
        
        # Calculate per-game averages
        gpm_a = self.team_a.last6['goalsScored'] / 6
        gpm_b = self.team_b.last6['goalsScored'] / 6
        
        cs_percent_a = (self.team_a.last6['cleanSheets'] / 6) * 100
        cs_percent_b = (self.team_b.last6['cleanSheets'] / 6) * 100
        
        btts_percent_a = (self.team_a.last6['bttsMatches'] / 6) * 100
        btts_percent_b = (self.team_b.last6['bttsMatches'] / 6) * 100
        
        win_percent_a = (self.team_a.last6['wins'] / 6) * 100
        win_percent_b = (self.team_b.last6['wins'] / 6) * 100
        
        # Combined metrics
        avg_gpm = (gpm_a + gpm_b) / 2
        avg_cs_percent = (cs_percent_a + cs_percent_b) / 2
        avg_btts_percent = (btts_percent_a + btts_percent_b) / 2
        
        # Classify goal potential
        if avg_gpm > 1.8:
            goal_potential = 'high'
        elif avg_gpm > 1.3:
            goal_potential = 'medium'
        else:
            goal_potential = 'low'
        
        # Classify BTTS likelihood
        if avg_btts_percent > 60:
            btts_likelihood = 'high'
        elif avg_btts_percent > 40:
            btts_likelihood = 'medium'
        else:
            btts_likelihood = 'low'
        
        return {
            'team_a': {
                'gpm': gpm_a,
                'cs_percent': cs_percent_a,
                'btts_percent': btts_percent_a,
                'win_percent': win_percent_a,
                'loss_percent': (self.team_a.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_a.last6['draws'] / 6) * 100
            },
            'team_b': {
                'gpm': gpm_b,
                'cs_percent': cs_percent_b,
                'btts_percent': btts_percent_b,
                'win_percent': win_percent_b,
                'loss_percent': (self.team_b.last6['losses'] / 6) * 100,
                'draw_percent': (self.team_b.last6['draws'] / 6) * 100
            },
            'combined': {
                'avg_gpm': avg_gpm,
                'avg_cs_percent': avg_cs_percent,
                'avg_btts_percent': avg_btts_percent,
                'goal_potential': goal_potential,
                'btts_likelihood': btts_likelihood
            }
        }


class ExtremeFilterDetector:
    """Detect extreme betting patterns"""
    
    # Constants
    UNDER_15_GPM_THRESHOLD = 0.75
    CS_PERCENT_THRESHOLD = 20
    MIN_GPM_FOR_BTTS = 1.3
    HIGH_GPM_THRESHOLD = 1.5
    CS_PERCENT_STRONG = 50
    OPPONENT_GPM_WEAK = 1.0
    BTTS_PERCENT_LOW = 40
    CS_PERCENT_DECENT = 30
    GPM_MODERATE = 1.5
    RECENT_BTTS_INFLATION_MIN = 70
    SEASON_BTTS_REGRESSION_MAX = 60
    
    def detect_filters(self, metrics: Dict[str, Any],
                      team_a_overall: Optional[Dict] = None,
                      team_b_overall: Optional[Dict] = None) -> Dict[str, Any]:
        
        filters = {
            'under_15_alert': False,
            'btts_banker': False,
            'btts_enhanced': False,
            'clean_sheet_alert': {'team': None, 'direction': None},
            'low_scoring_alert': False,
            'low_scoring_type': None,
            'regression_alert': {'team': None, 'type': None}
        }
        
        # Filter 1: Under 1.5 Goals Alert
        if (metrics['team_a']['gpm'] < self.UNDER_15_GPM_THRESHOLD and 
            metrics['team_b']['gpm'] < self.UNDER_15_GPM_THRESHOLD):
            filters['under_15_alert'] = True
        
        # Filter 2: BTTS Banker Alert (Modified for no H2H)
        if (metrics['team_a']['cs_percent'] < self.CS_PERCENT_THRESHOLD and 
            metrics['team_b']['cs_percent'] < self.CS_PERCENT_THRESHOLD and
            (metrics['team_a']['gpm'] > self.MIN_GPM_FOR_BTTS or 
             metrics['team_b']['gpm'] > self.MIN_GPM_FOR_BTTS)):
            filters['btts_banker'] = True
            
            if (metrics['team_a']['gpm'] > self.HIGH_GPM_THRESHOLD or 
                metrics['team_b']['gpm'] > self.HIGH_GPM_THRESHOLD):
                filters['btts_enhanced'] = True
        
        # Filter 3: Clean Sheet Alert
        if (metrics['team_a']['cs_percent'] > self.CS_PERCENT_STRONG and 
            metrics['team_b']['gpm'] < self.OPPONENT_GPM_WEAK):
            filters['clean_sheet_alert'] = {'team': 'A', 'direction': 'win_to_nil'}
        elif (metrics['team_b']['cs_percent'] > self.CS_PERCENT_STRONG and 
              metrics['team_a']['gpm'] < self.OPPONENT_GPM_WEAK):
            filters['clean_sheet_alert'] = {'team': 'B', 'direction': 'win_to_nil'}
        
        # Filter 4: Low-Scoring Alert
        if (metrics['team_a']['btts_percent'] < self.BTTS_PERCENT_LOW and 
            metrics['team_b']['btts_percent'] < self.BTTS_PERCENT_LOW):
            filters['low_scoring_alert'] = True
            filters['low_scoring_type'] = 'both_low_btts'
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
        
        # Filter 5: Regression Alert
        if team_a_overall and team_a_overall.get('matches', 0) > 0:
            season_btts_a = (team_a_overall.get('bttsMatches', 0) / team_a_overall['matches']) * 100
            if (metrics['team_a']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_a < self.SEASON_BTTS_REGRESSION_MAX):
                filters['regression_alert'] = {'team': 'A', 'type': 'btts_regression'}
        
        if team_b_overall and team_b_overall.get('matches', 0) > 0:
            season_btts_b = (team_b_overall.get('bttsMatches', 0) / team_b_overall['matches']) * 100
            if (metrics['team_b']['btts_percent'] > self.RECENT_BTTS_INFLATION_MIN and
                season_btts_b < self.SEASON_BTTS_REGRESSION_MAX):
                if filters['regression_alert']['team'] == 'A':
                    filters['regression_alert'] = {'team': 'both', 'type': 'btts_regression'}
                else:
                    filters['regression_alert'] = {'team': 'B', 'type': 'btts_regression'}
        
        return filters


class CurrentFormAnalyzer:
    """Analyze current team form"""
    
    def analyze_form(self, team_a: TeamFormData, team_b: TeamFormData) -> Dict[str, Any]:
        
        # Determine favorite based on wins
        win_diff = team_a.last6['wins'] - team_b.last6['wins']
        
        if win_diff >= 2:
            favorite = 'team_a'
            form_edge = 'strong'
        elif win_diff <= -2:
            favorite = 'team_b'
            form_edge = 'strong'
        elif win_diff == 1:
            favorite = 'team_a'
            form_edge = 'slight'
        elif win_diff == -1:
            favorite = 'team_b'
            form_edge = 'slight'
        else:
            favorite = None
            form_edge = 'none'
        
        # Goal potential
        avg_gpm = (team_a.last6['goalsScored']/6 + team_b.last6['goalsScored']/6) / 2
        if avg_gpm > 1.8:
            goal_potential = 'high'
        elif avg_gpm > 1.3:
            goal_potential = 'medium'
        else:
            goal_potential = 'low'
        
        # BTTS likelihood
        avg_btts = ((team_a.last6['bttsMatches']/6)*100 + (team_b.last6['bttsMatches']/6)*100) / 2
        if avg_btts > 60:
            btts_likelihood = 'high'
        elif avg_btts > 40:
            btts_likelihood = 'medium'
        else:
            btts_likelihood = 'low'
        
        # Momentum score
        momentum_a = (team_a.last6['wins'] * 3 + team_a.last6['draws']) - team_a.last6['losses']
        momentum_b = (team_b.last6['wins'] * 3 + team_b.last6['draws']) - team_b.last6['losses']
        
        if momentum_a > momentum_b + 2:
            momentum = 'team_a'
        elif momentum_b > momentum_a + 2:
            momentum = 'team_b'
        else:
            momentum = 'balanced'
        
        return {
            'favorite': favorite,
            'form_edge': form_edge,
            'goal_potential': goal_potential,
            'btts_likelihood': btts_likelihood,
            'momentum': momentum
        }


class MatchScriptGenerator:
    """Generate match predictions and narrative"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict[str, Any], filters: Dict[str, Any],
                       form_analysis: Dict[str, Any]) -> Dict[str, Any]:
        
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
        
        # Check regression warnings
        if filters['regression_alert']['team']:
            team = filters['regression_alert']['team']
            if team == 'A' or team == 'both':
                script['warnings'].append(f"{self.team_a_name}: Recent BTTS rate may regress to season average")
            if team == 'B' or team == 'both':
                script['warnings'].append(f"{self.team_b_name}: Recent BTTS rate may regress to season average")
        
        # Apply extreme filters
        if filters['under_15_alert']:
            script['primary_bets'] = ['under_15_goals', 'btts_no']
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams have very low scoring form - expect minimal goals'
            script['key_insights'].append('Both teams average < 0.75 goals per game')
            return script
        
        if filters['low_scoring_alert']:
            script['primary_bets'] = ['under_25_goals', 'btts_no']
            script['predicted_score_range'] = ['0-0', '1-0', '0-1']
            script['confidence'] = 'high'
            
            if filters['low_scoring_type'] == 'both_low_btts':
                script['match_narrative'] = 'Low BTTS frequency suggests tight game'
                script['key_insights'].append('Both teams have BTTS% < 40%')
            else:
                script['match_narrative'] = 'Defensive strength vs limited attack'
                script['key_insights'].append('One team strong defensively, other low-scoring')
            return script
        
        if filters['btts_banker']:
            script['primary_bets'] = ['btts_yes', 'over_15_goals']
            
            if filters['btts_enhanced']:
                script['value_bets'] = ['over_25_goals']
                script['predicted_score_range'] = ['1-1', '2-1', '1-2', '2-2']
            else:
                script['predicted_score_range'] = ['1-1', '2-1', '1-2']
            
            script['confidence'] = 'high'
            script['match_narrative'] = 'Both teams leak goals consistently'
            script['key_insights'].append('Both teams have clean sheet% < 20%')
            return script
        
        if filters['clean_sheet_alert']['team']:
            team_name = self.team_a_name if filters['clean_sheet_alert']['team'] == 'A' else self.team_b_name
            script['primary_bets'] = [f'{team_name.lower().replace(" ", "_")}_win_to_nil', 'under_25_goals']
            script['predicted_score_range'] = ['2-0', '1-0'] if filters['clean_sheet_alert']['team'] == 'A' else ['0-2', '0-1']
            script['confidence'] = 'high'
            script['match_narrative'] = f'{team_name} strong defense against weak attack'
            script['key_insights'].append(f'{team_name} CS% > 50%, opponent GPM < 1.0')
            return script
        
        # No extreme filters - use form analysis
        if form_analysis['favorite']:
            favorite_name = self.team_a_name if form_analysis['favorite'] == 'team_a' else self.team_b_name
            script['secondary_bets'].append(f'{favorite_name.lower().replace(" ", "_")}_win_or_draw')
            script['key_insights'].append(f'{favorite_name} has better recent form')
        
        # Add goal market bets
        if form_analysis['goal_potential'] == 'high':
            script['secondary_bets'].append('over_15_goals')
            if metrics['combined']['avg_gpm'] > 2.0:
                script['secondary_bets'].append('over_25_goals')
        elif form_analysis['goal_potential'] == 'low':
            script['secondary_bets'].append('under_25_goals')
        
        # Add BTTS bets
        if form_analysis['btts_likelihood'] == 'high':
            script['secondary_bets'].append('btts_yes')
        elif form_analysis['btts_likelihood'] == 'low':
            script['secondary_bets'].append('btts_no')
        
        # Set confidence
        if form_analysis['form_edge'] != 'none':
            script['confidence'] = 'medium'
        
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
            script['match_narrative'] += f'{momentum_team} has momentum. '
        
        # Generate score range
        if not script['predicted_score_range']:
            script['predicted_score_range'] = self._generate_score_range(script, form_analysis)
        
        return script
    
    def _generate_score_range(self, script: Dict[str, Any], form_analysis: Dict[str, Any]) -> List[str]:
        """Generate likely score ranges"""
        
        if 'btts_yes' in script['primary_bets'] or 'btts_yes' in script['secondary_bets']:
            if 'over_25_goals' in script['value_bets'] or 'over_25_goals' in script['secondary_bets']:
                return ['2-1', '1-2', '2-2', '3-1', '1-3']
            else:
                return ['1-1', '2-1', '1-2']
        elif 'under_15_goals' in script['primary_bets']:
            return ['0-0', '1-0', '0-1']
        elif 'under_25_goals' in script['primary_bets'] or 'under_25_goals' in script['secondary_bets']:
            return ['1-0', '0-1', '1-1', '2-0', '0-2']
        else:
            if form_analysis['goal_potential'] == 'high':
                return ['2-1', '1-2', '2-0', '0-2', '3-1']
            elif form_analysis['goal_potential'] == 'low':
                return ['1-0', '0-1', '1-1', '0-0']
            else:
                return ['1-0', '0-1', '1-1', '2-1', '1-2']


class ValueDetector:
    """Detect value bets from market odds"""
    
    VALUE_EDGE_MINIMUM = 15
    
    def __init__(self, market_odds: MarketOdds):
        self.market_odds = market_odds
    
    def calculate_value(self, script: Dict[str, Any]) -> List[Dict[str, Any]]:
        
        if not self.market_odds:
            return []
        
        value_bets = []
        confidence_prob = {'high': 0.75, 'medium': 0.60, 'low': 0.40}
        our_prob = confidence_prob.get(script['confidence'], 0.5)
        
        # Check primary bets
        for bet_type in script['primary_bets']:
            odds_key = self._map_bet_type(bet_type)
            if hasattr(self.market_odds, odds_key):
                odds = getattr(self.market_odds, odds_key)
                if odds and odds > 0:
                    implied_prob = 1 / odds
                    edge = ((our_prob / implied_prob) - 1) * 100
                    if edge >= self.VALUE_EDGE_MINIMUM:
                        value_bets.append({
                            'bet_type': bet_type,
                            'odds': odds,
                            'edge_percent': round(edge, 1),
                            'reason': 'primary_bet'
                        })
        
        # Check secondary bets (lower confidence)
        secondary_prob = our_prob * 0.8
        for bet_type in script['secondary_bets']:
            odds_key = self._map_bet_type(bet_type)
            if hasattr(self.market_odds, odds_key):
                odds = getattr(self.market_odds, odds_key)
                if odds and odds > 0:
                    implied_prob = 1 / odds
                    edge = ((secondary_prob / implied_prob) - 1) * 100
                    if edge >= self.VALUE_EDGE_MINIMUM * 1.5:
                        value_bets.append({
                            'bet_type': bet_type,
                            'odds': odds,
                            'edge_percent': round(edge, 1),
                            'reason': 'secondary_bet'
                        })
        
        return value_bets
    
    def _map_bet_type(self, bet_type: str) -> str:
        """Map bet type to market odds attribute"""
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
            return bet_type.replace('_win_to_nil', '_win')
        elif '_win_or_draw' in bet_type:
            return 'draw'  # Simplified mapping
        
        return mapping.get(bet_type, bet_type)


class BettingSlipGenerator:
    """Generate final betting recommendations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_bets': 3,
            'min_confidence': 'medium'
        }
    
    def generate_slip(self, script: Dict[str, Any], value_bets: List[Dict[str, Any]],
                     team_a: str, team_b: str) -> Dict[str, Any]:
        
        slip = {
            'recommended_bets': [],
            'stake_suggestions': {},
            'total_units': 10,
            'confidence': script['confidence']
        }
        
        # Check confidence level
        confidence_levels = {'high': 3, 'medium': 2, 'low': 1}
        min_level = confidence_levels.get(self.config['min_confidence'], 1)
        
        if confidence_levels.get(script['confidence'], 0) < min_level:
            slip['recommended_bets'].append({
                'type': 'NO_BET',
                'reason': 'Low confidence',
                'priority': 'none'
            })
            return slip
        
        # Add primary bets
        for bet in script['primary_bets']:
            slip['recommended_bets'].append({
                'type': bet,
                'priority': 'high',
                'reason': 'Extreme filter match'
            })
        
        # Add value bets
        for value_bet in value_bets:
            if len(slip['recommended_bets']) < self.config['max_bets']:
                slip['recommended_bets'].append({
                    'type': value_bet['bet_type'],
                    'priority': 'medium',
                    'reason': f'Value edge: {value_bet["edge_percent"]}%',
                    'odds': value_bet.get('odds')
                })
        
        # Add secondary bets if space
        for bet in script['secondary_bets']:
            if len(slip['recommended_bets']) < self.config['max_bets']:
                slip['recommended_bets'].append({
                    'type': bet,
                    'priority': 'low',
                    'reason': 'Form-based recommendation'
                })
        
        # Remove duplicates
        unique_bets = []
        seen = set()
        for bet in slip['recommended_bets']:
            if bet['type'] not in seen:
                unique_bets.append(bet)
                seen.add(bet['type'])
        
        slip['recommended_bets'] = unique_bets[:self.config['max_bets']]
        
        # Calculate stakes
        for bet in slip['recommended_bets']:
            if bet['type'] != 'NO_BET':
                if bet['priority'] == 'high':
                    slip['stake_suggestions'][bet['type']] = 5.0
                elif bet['priority'] == 'medium':
                    slip['stake_suggestions'][bet['type']] = 3.0
                else:
                    slip['stake_suggestions'][bet['type']] = 2.0
        
        # Add score predictions
        if script['predicted_score_range']:
            slip['score_predictions'] = script['predicted_score_range'][:3]
        
        return slip


class BettingAnalyticsEngine:
    """Main analysis engine"""
    
    def __init__(self):
        pass
    
    def analyze_match(self, context: MatchContext) -> Dict[str, Any]:
        
        # Step 1: Calculate metrics
        analyzer = MatchAnalyzer(context.teamA, context.teamB)
        metrics = analyzer.calculate_metrics()
        
        # Step 2: Detect extreme filters
        filter_detector = ExtremeFilterDetector()
        filters = filter_detector.detect_filters(
            metrics, 
            context.teamA.overall, 
            context.teamB.overall
        )
        
        # Step 3: Analyze form
        form_analyzer = CurrentFormAnalyzer()
        form_analysis = form_analyzer.analyze_form(context.teamA, context.teamB)
        
        # Step 4: Generate script
        script_generator = MatchScriptGenerator(context.teamA.teamName, context.teamB.teamName)
        script = script_generator.generate_script(metrics, filters, form_analysis)
        
        # Step 5: Detect value bets
        value_bets = []
        if context.marketOdds:
            value_detector = ValueDetector(context.marketOdds)
            value_bets = value_detector.calculate_value(script)
        
        # Step 6: Generate betting slip
        slip_generator = BettingSlipGenerator()
        slip = slip_generator.generate_slip(
            script, value_bets, 
            context.teamA.teamName, context.teamB.teamName
        )
        
        return {
            'match_info': {
                'team_a': context.teamA.teamName,
                'team_b': context.teamB.teamName,
                'venue': 'home' if context.isTeamAHome else 'away'
            },
            'metrics': metrics,
            'filters': filters,
            'form_analysis': form_analysis,
            'script': script,
            'value_bets': value_bets,
            'betting_slip': slip
        }

# ============================================================================
# DATA PARSING
# ============================================================================

def parse_team_from_csv(row: pd.Series) -> TeamFormData:
    """Parse team data from CSV row"""
    
    team_name = row['Team']
    
    # Get Last 6 data
    wins = int(row['Form W'])
    draws = int(row['Form D'])
    losses = int(row['Form L'])
    goals_scored = int(row['Last 6 Goals'])
    
    # Estimate Last 6 clean sheets and BTTS from overall percentages
    # This is a limitation - ideally CSV should have Last 6 CS and BTTS
    overall_cs_pct = float(str(row.get('Overall CS%', '0')).replace('%', ''))
    overall_btts_pct = float(str(row.get('Overall BTTS%', '0')).replace('%', ''))
    
    estimated_cs = max(0, min(6, round((overall_cs_pct / 100) * 6)))
    estimated_btts = max(0, min(6, round((overall_btts_pct / 100) * 6)))
    
    # Overall data
    overall_matches = int(row.get('Overall Matches', 0))
    
    return TeamFormData(
        teamName=team_name,
        last6={
            'matches': 6,
            'goalsScored': goals_scored,
            'cleanSheets': estimated_cs,
            'bttsMatches': estimated_btts,
            'wins': wins,
            'draws': draws,
            'losses': losses
        },
        overall={
            'matches': overall_matches,
            'goalsScored': int(row.get('Overall Goals', 0)),
            'cleanSheets': int((overall_cs_pct / 100) * overall_matches) if overall_matches > 0 else 0,
            'bttsMatches': int((overall_btts_pct / 100) * overall_matches) if overall_matches > 0 else 0
        }
    )

def load_league_data(league_name: str) -> Optional[pd.DataFrame]:
    """Load league CSV data"""
    try:
        file_path = f"leagues/{league_name}.csv"
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading {league_name}: {e}")
        return None

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Betting Analytics Engine",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Betting Analytics Engine v3.1")
    st.markdown("**Team Statistics Analysis Only**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # League selection
        leagues = [
            'bundesliga', 'bundesliga_2', 'championship', 'eredivisie',
            'erste_divisie', 'laliga', 'laliga_2', 'ligue_1', 'ligue_2',
            'premier_league', 'serie_a', 'serie_b'
        ]
        
        selected_league = st.selectbox("Select League", leagues)
        
        # Load data
        df = load_league_data(selected_league)
        if df is None:
            st.error(f"Could not load {selected_league}.csv")
            st.stop()
        
        # Team selection
        teams = sorted(df['Team'].tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Team A", teams, key="team_a")
        with col2:
            # Filter out selected team A
            available_teams = [t for t in teams if t != team_a]
            team_b = st.selectbox("Team B", available_teams, key="team_b")
        
        # Venue
        venue = st.radio("Venue", ["Team A Home", "Team B Home"])
        is_team_a_home = venue == "Team A Home"
        
        # Market odds (optional)
        st.subheader("Market Odds (Optional)")
        use_odds = st.checkbox("Enter odds for value detection")
        
        market_odds = None
        if use_odds:
            col1, col2 = st.columns(2)
            with col1:
                btts_yes = st.number_input("BTTS Yes", 1.0, 10.0, 1.8, 0.05)
                over_15 = st.number_input("Over 1.5", 1.0, 5.0, 1.3, 0.05)
                over_25 = st.number_input("Over 2.5", 1.0, 5.0, 2.1, 0.05)
            with col2:
                btts_no = st.number_input("BTTS No", 1.0, 10.0, 2.0, 0.05)
                under_15 = st.number_input("Under 1.5", 1.0, 5.0, 3.4, 0.05)
                under_25 = st.number_input("Under 2.5", 1.0, 5.0, 1.7, 0.05)
            
            market_odds = MarketOdds(
                btts_yes=btts_yes, btts_no=btts_no,
                over_15=over_15, under_15=under_15,
                over_25=over_25, under_25=under_25
            )
    
    # Main content
    if df is not None:
        # Parse team data
        team_a_row = df[df['Team'] == team_a].iloc[0]
        team_b_row = df[df['Team'] == team_b].iloc[0]
        
        team_a_data = parse_team_from_csv(team_a_row)
        team_b_data = parse_team_from_csv(team_b_row)
        
        # Show team preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🏠 {team_a}")
            st.metric("Last 6 Goals", team_a_data.last6['goalsScored'])
            st.metric("Form", f"{team_a_data.last6['wins']}W-{team_a_data.last6['draws']}D-{team_a_data.last6['losses']}L")
            st.caption(f"Est. CS: {team_a_data.last6['cleanSheets']}/6")
            st.caption(f"Est. BTTS: {team_a_data.last6['bttsMatches']}/6")
        
        with col2:
            st.subheader(f"🚌 {team_b}")
            st.metric("Last 6 Goals", team_b_data.last6['goalsScored'])
            st.metric("Form", f"{team_b_data.last6['wins']}W-{team_b_data.last6['draws']}D-{team_b_data.last6['losses']}L")
            st.caption(f"Est. CS: {team_b_data.last6['cleanSheets']}/6")
            st.caption(f"Est. BTTS: {team_b_data.last6['bttsMatches']}/6")
        
        st.markdown("---")
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing match..."):
                # Create match context
                context = MatchContext(
                    teamA=team_a_data,
                    teamB=team_b_data,
                    isTeamAHome=is_team_a_home,
                    marketOdds=market_odds
                )
                
                # Run engine
                engine = BettingAnalyticsEngine()
                result = engine.analyze_match(context)
                
                # Display results
                st.header(f"📊 Analysis Results: {team_a} vs {team_b}")
                
                # Metrics
                st.subheader("Team Statistics")
                metrics = result['metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Team A GPM", f"{metrics['team_a']['gpm']:.2f}")
                    st.metric("Team A CS%", f"{metrics['team_a']['cs_percent']:.1f}%")
                with col2:
                    st.metric("Team B GPM", f"{metrics['team_b']['gpm']:.2f}")
                    st.metric("Team B CS%", f"{metrics['team_b']['cs_percent']:.1f}%")
                with col3:
                    st.metric("Avg GPM", f"{metrics['combined']['avg_gpm']:.2f}")
                    st.metric("Goal Potential", metrics['combined']['goal_potential'].title())
                
                # Filters triggered
                st.subheader("🔍 Detected Patterns")
                filters = result['filters']
                
                filter_cols = st.columns(5)
                filter_info = [
                    ('under_15_alert', '🔴', 'Under 1.5 Alert'),
                    ('btts_banker', '🟢', 'BTTS Banker'),
                    ('clean_sheet_alert', '🔵', 'Clean Sheet'),
                    ('low_scoring_alert', '🟡', 'Low Scoring'),
                    ('regression_alert', '🟣', 'Regression')
                ]
                
                for idx, (filter_key, icon, name) in enumerate(filter_info):
                    with filter_cols[idx]:
                        if filter_key == 'clean_sheet_alert' and filters[filter_key]['team']:
                            st.markdown(f"{icon} **{name}**")
                            team = team_a if filters[filter_key]['team'] == 'A' else team_b
                            st.caption(f"{team} clean sheet")
                        elif filter_key == 'regression_alert' and filters[filter_key]['team']:
                            st.markdown(f"{icon} **{name}**")
                            if filters[filter_key]['team'] == 'A':
                                st.caption(f"{team_a} BTTS inflated")
                            elif filters[filter_key]['team'] == 'B':
                                st.caption(f"{team_b} BTTS inflated")
                            elif filters[filter_key]['team'] == 'both':
                                st.caption("Both teams BTTS inflated")
                        elif filters.get(filter_key, False):
                            st.markdown(f"{icon} **{name}**")
                            st.caption("Triggered")
                
                # Match narrative
                st.subheader("📝 Match Analysis")
                script = result['script']
                
                confidence_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
                st.metric("Confidence", f"{script['confidence'].title()} {confidence_color[script['confidence']]}")
                
                st.info(script['match_narrative'])
                
                if script.get('warnings'):
                    for warning in script['warnings']:
                        st.warning(warning)
                
                if script.get('key_insights'):
                    for insight in script['key_insights']:
                        st.success(f"• {insight}")
                
                # Betting recommendations
                st.subheader("💰 Betting Recommendations")
                slip = result['betting_slip']
                
                if slip['recommended_bets'][0]['type'] == 'NO_BET':
                    st.warning("**NO BET RECOMMENDED** - Confidence too low")
                else:
                    for bet in slip['recommended_bets']:
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.markdown(f"**{bet['type'].replace('_', ' ').title()}**")
                        with col2:
                            priority_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴', 'none': '⚪'}
                            st.markdown(f"Priority: {priority_color[bet['priority']]} {bet['priority'].title()}")
                        with col3:
                            if bet['type'] in slip['stake_suggestions']:
                                st.metric("Stake", f"{slip['stake_suggestions'][bet['type']]} units")
                        
                        st.caption(f"Reason: {bet['reason']}")
                        st.divider()
                    
                    # Score predictions
                    if slip.get('score_predictions'):
                        st.markdown("**🎯 Most Likely Scores**")
                        cols = st.columns(len(slip['score_predictions']))
                        for idx, score in enumerate(slip['score_predictions']):
                            with cols[idx]:
                                st.markdown(f"<div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'><h4>{score}</h4></div>", 
                                          unsafe_allow_html=True)
                
                # Value bets
                if result['value_bets']:
                    st.subheader("💎 Value Bets")
                    for value_bet in result['value_bets']:
                        edge = value_bet['edge_percent']
                        if edge > 25:
                            color = "🟢"
                            label = "STRONG VALUE"
                        elif edge > 15:
                            color = "🟡"
                            label = "GOOD VALUE"
                        else:
                            color = "🔴"
                            label = "MARGINAL VALUE"
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{value_bet['bet_type'].replace('_', ' ').title()}**")
                        with col2:
                            st.metric("Odds", f"{value_bet['odds']:.2f}")
                        with col3:
                            st.metric("Edge", f"{edge:.1f}% {color}")
                        
                        st.caption(f"{label} | Our confidence vs market odds")
                        st.divider()
                
                # Raw data
                with st.expander("📋 View Raw Analysis Data"):
                    st.json(result)

if __name__ == "__main__":
    main()
