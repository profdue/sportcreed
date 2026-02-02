#!/usr/bin/env python3
"""
HIGH-CERTAINTY NO DRAW ANALYZER WITH BAYERN PROTECTION
Three filters only:
1. Under 1.5 Goals (High certainty defensive)
2. Under 2.5 Goals (Good certainty defensive)
3. No Draw Candidate (Tiered system with auto-reject for strong favorites)
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
# HIGH-CERTAINTY NO DRAW DETECTOR WITH BAYERN PROTECTION
# ============================================================================

class HighCertaintyNoDrawDetector:
    """Detect only HIGH-CERTAINTY No Draw patterns with Bayern protection"""
    
    def __init__(self):
        # BAYERN PROTECTION: Auto-reject thresholds
        self.AUTO_REJECT_WIN_DIFF = 4          # Win difference ≥ 4 → too one-sided
        self.AUTO_REJECT_GPM_DIFF = 1.5        # GPM difference ≥ 1.5 → scoring dominance
        self.AUTO_REJECT_GOAL_DIFF = 10        # Goal difference ≥ 10 in last 6 → too dominant
        
        # Trap teams that attract public money (customize this list)
        self.TRAP_TEAMS = [
            'bayern', 'real madrid', 'barcelona', 'psg', 'man city', 
            'liverpool', 'dortmund', 'juventus', 'milan', 'inter',
            'atletico madrid', 'chelsea', 'man united', 'arsenal',
            'napoli', 'roma', 'lyon', 'marseille', 'benfica', 'porto'
        ]
        
        # Filter 1: Under 1.5 Goals (extreme defensive - keep as is)
        self.UNDER_15_GPM_THRESHOLD = 0.75
        
        # Filter 2: Under 2.5 Goals (defensive - keep as is)
        self.UNDER_25_GPM_THRESHOLD = 1.5
        self.UNDER_25_BTTS_THRESHOLD = 50
        
        # Filter 3: HIGH-CERTAINTY No Draw Candidate
        # TIER 1: Maximum certainty (BET)
        self.NO_DRAW_TIER1_IMBALANCE = 1.4      # One team scores 40%+ more
        self.NO_DRAW_TIER1_COMBINED_GPM = 3.0   # High scoring environment
        self.NO_DRAW_TIER1_BTTS_AVG = 60        # Both teams score regularly
        self.NO_DRAW_TIER1_CS_MAX = 30          # Not defensive teams
        
        # TIER 2: Good certainty (CHECK ODDS)
        self.NO_DRAW_TIER2_IMBALANCE = 1.3      # One team scores 30%+ more
        self.NO_DRAW_TIER2_COMBINED_GPM = 2.8   # Decent scoring
        self.NO_DRAW_TIER2_BTTS_AVG = 55        # Good scoring rate
        self.NO_DRAW_TIER2_CS_MAX = 35          # Moderate defense
        
        # Special exception: Extreme high-scoring
        self.EXTREME_SCORING_GPM = 4.0          # If combined > 4.0 GPM
        self.EXTREME_SCORING_IMBALANCE = 1.25   # Allow slightly lower imbalance
        
        # High-draw leagues (always be cautious)
        self.HIGH_DRAW_LEAGUES = ['paraguay', 'iran', 'peru']
    
    def detect_filters(self, metrics: Dict, team_a: TeamFormData, team_b: TeamFormData, league: str) -> Dict:
        """Detect filters with HIGH-CERTAINTY focus and Bayern protection"""
        filters = {
            'under_15_alert': False,
            'under_25_alert': False,
            'no_draw_candidate': False,
            'no_draw_tier': 'none',  # 'tier1', 'tier2', or 'none'
            'high_draw_league_warning': False,
            'auto_rejected': False,
            'auto_reject_reasons': [],
            'trap_team_warning': False,
            'win_difference': team_a.last6['wins'] - team_b.last6['wins'],
            'attack_imbalance_ratio': metrics['imbalance']['ratio'],
            'attack_imbalance_favors': metrics['imbalance']['favors'],
            'combined_gpm': metrics['averages']['avg_gpm'] * 2,
            'debug_info': {}
        }
        
        # ===== BAYERN PROTECTION: AUTO-REJECT CHECKS =====
        auto_reject_reasons = []
        
        # 1. Check win difference (too one-sided)
        win_diff = abs(filters['win_difference'])
        if win_diff >= self.AUTO_REJECT_WIN_DIFF:
            auto_reject_reasons.append(f"Win difference too large ({win_diff} ≥ {self.AUTO_REJECT_WIN_DIFF})")
        
        # 2. Check GPM difference (scoring dominance)
        gpm_diff = abs(metrics['team_a']['gpm'] - metrics['team_b']['gpm'])
        if gpm_diff >= self.AUTO_REJECT_GPM_DIFF:
            auto_reject_reasons.append(f"Scoring dominance ({gpm_diff:.2f} GPM diff ≥ {self.AUTO_REJECT_GPM_DIFF})")
        
        # 3. Check goal difference in last 6
        goal_diff = abs(metrics['team_a']['goals'] - metrics['team_b']['goals'])
        if goal_diff >= self.AUTO_REJECT_GOAL_DIFF:
            auto_reject_reasons.append(f"Goal dominance ({goal_diff} goals diff ≥ {self.AUTO_REJECT_GOAL_DIFF})")
        
        # 4. Check for trap teams
        team_a_lower = team_a.teamName.lower()
        team_b_lower = team_b.teamName.lower()
        
        trap_team_found = None
        for trap_team in self.TRAP_TEAMS:
            if trap_team in team_a_lower or trap_team in team_b_lower:
                trap_team_found = trap_team
                break
        
        if trap_team_found:
            filters['trap_team_warning'] = True
            auto_reject_reasons.append(f"Trap team detected: {trap_team_found.title()}")
        
        # 5. Check if league is high-draw
        if any(high_draw_league in league.lower() for high_draw_league in self.HIGH_DRAW_LEAGUES):
            filters['high_draw_league_warning'] = True
        
        # If auto-rejected, skip all further analysis
        if auto_reject_reasons:
            filters['auto_rejected'] = True
            filters['auto_reject_reasons'] = auto_reject_reasons
            filters['debug_info']['auto_reject'] = auto_reject_reasons
            return filters
        
        # ===== REGULAR FILTER ANALYSIS (only if not auto-rejected) =====
        
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
        
        # FILTER 3: HIGH-CERTAINTY No Draw Candidate
        else:
            # Calculate key metrics
            imbalance = metrics['imbalance']['ratio']
            combined_gpm = filters['combined_gpm']
            btts_avg = metrics['averages']['avg_btts_percent']
            cs_avg = metrics['averages']['avg_cs_percent']
            
            # Check for EXTREME high-scoring exception first
            extreme_scoring = False
            if combined_gpm >= self.EXTREME_SCORING_GPM:
                if imbalance >= self.EXTREME_SCORING_IMBALANCE:
                    extreme_scoring = True
                    filters['debug_info']['extreme_scoring'] = f"Combined GPM {combined_gpm:.2f} ≥ {self.EXTREME_SCORING_GPM}, imbalance {imbalance:.2f} ≥ {self.EXTREME_SCORING_IMBALANCE}"
            
            # Check TIER 1: Maximum certainty
            tier1_ok = (
                imbalance >= self.NO_DRAW_TIER1_IMBALANCE and
                combined_gpm >= self.NO_DRAW_TIER1_COMBINED_GPM and
                btts_avg >= self.NO_DRAW_TIER1_BTTS_AVG and
                cs_avg < self.NO_DRAW_TIER1_CS_MAX
            )
            
            # Check TIER 2: Good certainty
            tier2_ok = (
                (imbalance >= self.NO_DRAW_TIER2_IMBALANCE or extreme_scoring) and
                combined_gpm >= self.NO_DRAW_TIER2_COMBINED_GPM and
                btts_avg >= self.NO_DRAW_TIER2_BTTS_AVG and
                cs_avg < self.NO_DRAW_TIER2_CS_MAX
            )
            
            # Store debug info
            filters['debug_info']['no_draw_check'] = {
                'imbalance': f"{imbalance:.2f}x",
                'combined_gpm': f"{combined_gpm:.2f}",
                'btts_avg': f"{btts_avg:.1f}%",
                'cs_avg': f"{cs_avg:.1f}%",
                'tier1': tier1_ok,
                'tier2': tier2_ok,
                'extreme_scoring': extreme_scoring
            }
            
            # Set result
            if tier1_ok:
                filters['no_draw_candidate'] = True
                filters['no_draw_tier'] = 'tier1'
                filters['debug_info']['no_draw'] = f"TIER 1: imbalance {imbalance:.2f}x ≥ {self.NO_DRAW_TIER1_IMBALANCE}, GPM {combined_gpm:.2f} ≥ {self.NO_DRAW_TIER1_COMBINED_GPM}"
            
            elif tier2_ok:
                filters['no_draw_candidate'] = True
                filters['no_draw_tier'] = 'tier2'
                if extreme_scoring:
                    filters['debug_info']['no_draw'] = f"TIER 2 (Extreme Scoring): GPM {combined_gpm:.2f} ≥ {self.EXTREME_SCORING_GPM}, imbalance {imbalance:.2f}x ≥ {self.EXTREME_SCORING_IMBALANCE}"
                else:
                    filters['debug_info']['no_draw'] = f"TIER 2: imbalance {imbalance:.2f}x ≥ {self.NO_DRAW_TIER2_IMBALANCE}, GPM {combined_gpm:.2f} ≥ {self.NO_DRAW_TIER2_COMBINED_GPM}"
        
        return filters

# ============================================================================
# TIERED SCRIPT GENERATOR WITH AUTO-REJECT HANDLING
# ============================================================================

class TieredScriptGenerator:
    """Generate scripts with tiered certainty levels and auto-reject handling"""
    
    def __init__(self, team_a_name: str, team_b_name: str):
        self.team_a_name = team_a_name
        self.team_b_name = team_b_name
    
    def generate_script(self, metrics: Dict, filters: Dict, is_team_a_home: bool) -> Dict:
        """Generate script with tiered certainty and auto-reject handling"""
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
            'certainty_tier': 'none',
            'attack_imbalance_analysis': {},
            'auto_rejected': False,
            'auto_reject_reasons': [],
            'trap_team_warning': False
        }
        
        # Check if auto-rejected
        if filters.get('auto_rejected', False):
            script['auto_rejected'] = True
            script['auto_reject_reasons'] = filters.get('auto_reject_reasons', [])
            script['trap_team_warning'] = filters.get('trap_team_warning', False)
            script['match_narrative'] = self._generate_auto_reject_narrative(filters)
            script['confidence_score'] = 0
            script['triggered_filter'] = 'auto_reject'
            return script
        
        # Check for trap team warning (but not auto-rejected)
        if filters.get('trap_team_warning', False):
            script['trap_team_warning'] = True
        
        # Determine which filter triggered
        if filters['under_15_alert']:
            script = self._generate_under_15_script(script, metrics, filters)
            script['confidence_score'] = 85
            script['triggered_filter'] = 'under_15'
            script['certainty_tier'] = 'high'
        
        elif filters['under_25_alert']:
            script = self._generate_under_25_script(script, metrics, filters)
            script['confidence_score'] = 75
            script['triggered_filter'] = 'under_25'
            script['certainty_tier'] = 'medium'
        
        elif filters['no_draw_candidate']:
            tier = filters['no_draw_tier']
            script = self._generate_no_draw_script(script, metrics, filters, is_team_a_home, tier)
            script['confidence_score'] = self._calculate_tiered_confidence(tier, metrics, filters)
            script['triggered_filter'] = 'no_draw'
            script['certainty_tier'] = tier
            
            # Tier-based manual check requirements
            if tier == 'tier1':
                script['manual_check_required'] = True
                script['odds_to_check'] = {
                    'favorite_range': '1.50-1.75',  # TIGHTER: Was 1.35-1.75
                    'draw_min': '3.80',  # Stricter
                    'check_instructions': 'Maximum certainty - but still verify odds strictly'
                }
            else:  # tier2
                script['manual_check_required'] = True
                script['odds_to_check'] = {
                    'favorite_range': '1.50-1.70',  # EVEN TIGHTER for tier2
                    'draw_min': '4.00',  # Higher threshold
                    'check_instructions': 'Good certainty - be EXTRA strict with odds'
                }
        
        else:
            script['match_narrative'] = self._generate_no_filter_narrative(metrics, filters)
            script['confidence_score'] = 40
            script['triggered_filter'] = 'none'
            script['certainty_tier'] = 'none'
        
        # Add attack imbalance analysis
        script['attack_imbalance_analysis'] = {
            'ratio': metrics['imbalance']['ratio'],
            'favors': filters['attack_imbalance_favors'],
            'higher_scorer': metrics['imbalance']['higher_goals'],
            'lower_scorer': metrics['imbalance']['lower_goals'],
            'combined_gpm': filters['combined_gpm'],
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
    
    def _generate_auto_reject_narrative(self, filters: Dict) -> str:
        """Generate narrative for auto-rejected matches"""
        reasons = filters.get('auto_reject_reasons', [])
        
        narrative = "🚫 **AUTO-REJECTED: ULTRA-STRONG FAVORITE DETECTED**\n\n"
        narrative += "**Reasons for rejection:**\n"
        
        for reason in reasons:
            narrative += f"• {reason}\n"
        
        narrative += "\n**Recommendation:**\n"
        narrative += "• **DO NOT BET** on No Draw for this match\n"
        narrative += "• Even if form suggests No Draw, odds will be too low (<1.50)\n"
        narrative += "• Public heavily on favorite → higher draw risk\n"
        narrative += "• Classic trap match - SKIP\n"
        
        if filters.get('trap_team_warning', False):
            narrative += "\n⚠️ **TRAP TEAM WARNING:** Known public team detected\n"
            narrative += "Casual bettors heavily on this team → bookmakers protect with draw value"
        
        return narrative
    
    def _calculate_tiered_confidence(self, tier: str, metrics: Dict, filters: Dict) -> int:
        """Calculate confidence based on tier"""
        if tier == 'tier1':
            base = 85
            
            # Boost for stronger imbalance
            imbalance = metrics['imbalance']['ratio']
            if imbalance > 1.6:
                base += 10
            elif imbalance > 1.4:
                base += 5
            
            # Boost for higher scoring
            if filters['combined_gpm'] > 3.5:
                base += 5
                
            return min(base, 95)
        
        else:  # tier2
            base = 75
            
            # Check if it's extreme scoring exception
            if filters.get('debug_info', {}).get('no_draw_check', {}).get('extreme_scoring', False):
                base += 5  # Extra confidence for extreme scoring
            
            return min(base, 85)
    
    def _generate_under_15_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        script['primary_bets'].append('under_15_goals')
        script['secondary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1']
        script['match_narrative'] = f'HIGH CERTAINTY: Extreme low-scoring pattern (both < 0.75 GPM)'
        return script
    
    def _generate_under_25_script(self, script: Dict, metrics: Dict, filters: Dict) -> Dict:
        script['primary_bets'].append('under_25_goals')
        script['secondary_bets'].append('btts_no')
        script['predicted_score_range'] = ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2']
        script['match_narrative'] = 'Good certainty: Defensive match with low scoring expected'
        return script
    
    def _generate_no_draw_script(self, script: Dict, metrics: Dict, filters: Dict, 
                                 is_team_a_home: bool, tier: str) -> Dict:
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
        
        imbalance = metrics['imbalance']['ratio']
        combined_gpm = filters['combined_gpm']
        
        # Tier-specific predictions
        if tier == 'tier1':
            # High certainty - more decisive scores
            if filters['attack_imbalance_favors'] == 'A':
                script['predicted_score_range'] = ['2-0', '3-0', '2-1', '3-1', '1-0']
            else:
                script['predicted_score_range'] = ['0-2', '0-3', '1-2', '1-3', '0-1']
            
            certainty_word = "HIGH CERTAINTY"
            confidence_word = "strongly"
            
        else:  # tier2
            # Good certainty - competitive but still no draw
            if filters['attack_imbalance_favors'] == 'A':
                script['predicted_score_range'] = ['2-1', '1-0', '2-0', '3-2']
            else:
                script['predicted_score_range'] = ['1-2', '0-1', '0-2', '2-3']
            
            certainty_word = "GOOD CERTAINTY"
            confidence_word = "likely"
            
            # Add extreme scoring note if applicable
            if filters.get('debug_info', {}).get('no_draw_check', {}).get('extreme_scoring', False):
                certainty_word = "GOOD CERTAINTY (Extreme Scoring)"
        
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
        
        # Supportive bets based on scoring level
        script['secondary_bets'].append('btts_yes')
        if combined_gpm > 3.0:
            script['secondary_bets'].append('over_25_goals')
        if combined_gpm > 3.5:
            script['secondary_bets'].append('over_35_goals')
        
        # Narrative
        if tier == 'tier1':
            script['match_narrative'] = (
                f'🎯 {certainty_word} NO DRAW: {stronger_name} scores {imbalance:.1f}x more than {weaker_name} '
                f'({stronger_goals} vs {weaker_goals} goals). '
                f'High-scoring environment ({combined_gpm:.1f} GPM combined). '
                f'Both teams score regularly (avg BTTS: {metrics["averages"]["avg_btts_percent"]:.1f}%). '
                f'Odds should {confidence_word} match No Draw criteria.'
            )
        else:
            script['match_narrative'] = (
                f'✅ {certainty_word} NO DRAW: Clear attack advantage ({imbalance:.1f}x) '
                f'in high-scoring match ({combined_gpm:.1f} GPM combined). '
                f'Both teams have scoring capability. '
                f'MANUAL CHECK: Verify odds strictly match criteria.'
            )
        
        return script
    
    def _generate_no_filter_narrative(self, metrics: Dict, filters: Dict) -> str:
        imbalance = metrics['imbalance']['ratio']
        combined_gpm = filters['combined_gpm']
        btts_avg = metrics['averages']['avg_btts_percent']
        
        reasons = []
        
        if imbalance < 1.3:
            reasons.append(f'insufficient attack imbalance ({imbalance:.2f}x)')
        
        if combined_gpm < 2.8:
            reasons.append(f'low scoring potential ({combined_gpm:.1f} GPM)')
        
        if btts_avg < 55:
            reasons.append(f'low scoring consistency (BTTS: {btts_avg:.1f}%)')
        
        if reasons:
            return f'❌ Not enough certainty for No Draw: ' + ', '.join(reasons)
        else:
            return 'Patterns unclear. No high-certainty signal detected.'

# ============================================================================
# MAIN ENGINE
# ============================================================================

class HighCertaintyEngine:
    """Main orchestrator for high-certainty strategy with Bayern protection"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.metrics_calc = MatchAnalyzer()
        self.filter_detector = HighCertaintyNoDrawDetector()
        self.script_generator = None
    
    def analyze_match(self, match_context: MatchContext, league: str) -> Dict:
        self.script_generator = TieredScriptGenerator(
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
            'certainty_tier': script['certainty_tier'],
            'auto_rejected': script['auto_rejected'],
            'auto_reject_reasons': script['auto_reject_reasons'],
            'trap_team_warning': script['trap_team_warning'],
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

def render_high_certainty_dashboard(result: Dict):
    """Display high-certainty focused results"""
    st.title("🎯 High-Certainty No Draw Analyzer")
    st.subheader(f"{result['match_info']['team_a']} vs {result['match_info']['team_b']}")
    
    # Check if auto-rejected
    if result['auto_rejected']:
        st.error("🚫 AUTO-REJECTED: ULTRA-STRONG FAVORITE DETECTED")
        
        st.markdown("### ⚠️ REJECTION REASONS")
        for reason in result['auto_reject_reasons']:
            st.write(f"• {reason}")
        
        if result['trap_team_warning']:
            st.warning("⚠️ TRAP TEAM DETECTED: Known public team")
        
        st.markdown("### 📋 RECOMMENDATION")
        st.error("**DO NOT BET ON NO DRAW FOR THIS MATCH**")
        st.write("""
        • Odds will be too low (<1.50 for favorite)
        • Public heavily on favorite → higher draw risk
        • Classic trap match - SKIP entirely
        • Even if form suggests No Draw, market says otherwise
        """)
        
        # Still show analysis for information
        st.markdown("---")
        st.markdown("### 📊 Analysis (For Reference Only)")
        
        # Attack Imbalance Analysis
        imbalance = result['attack_imbalance']
        st.markdown("#### ⚡ Attack Imbalance Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Imbalance Ratio", f"{imbalance['ratio']:.2f}x")
        with col2:
            st.metric("Combined GPM", f"{imbalance['combined_gpm']:.1f}")
        with col3:
            st.metric("Avg BTTS%", f"{imbalance['btts_avg']:.1f}%")
        with col4:
            st.metric("Avg CS%", f"{imbalance['cs_avg']:.1f}%")
        
        return
    
    # Regular analysis (not auto-rejected)
    
    # Certainty tier badge
    tier = result['certainty_tier']
    if tier == 'tier1':
        st.success("🏆 MAXIMUM CERTAINTY TIER")
    elif tier == 'tier2':
        st.warning("✅ GOOD CERTAINTY TIER")
    
    # Confidence
    confidence = result['confidence']
    score = result['confidence_score']
    
    if confidence == 'high':
        st.success(f"Confidence: High 🟢 ({score}/100)")
    elif confidence == 'medium':
        st.warning(f"Confidence: Medium 🟡 ({score}/100)")
    else:
        st.info(f"Confidence: Low 🔵 ({score}/100)")
    
    # Trap team warning (but not auto-rejected)
    if result['trap_team_warning']:
        st.warning("⚠️ TRAP TEAM DETECTED: Be extra cautious with odds check")
    
    # Attack Imbalance Analysis
    imbalance = result['attack_imbalance']
    st.markdown("### ⚡ Attack Imbalance Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Imbalance Ratio", f"{imbalance['ratio']:.2f}x")
    with col2:
        st.metric("Combined GPM", f"{imbalance['combined_gpm']:.1f}")
    with col3:
        st.metric("Avg BTTS%", f"{imbalance['btts_avg']:.1f}%")
    with col4:
        st.metric("Avg CS%", f"{imbalance['cs_avg']:.1f}%")
    
    # Attack advantage
    if imbalance['favors'] == 'A':
        st.info(f"**Attack Advantage**: {result['match_info']['team_a']} ({imbalance['higher_scorer']} vs {imbalance['lower_scorer']} goals)")
    else:
        st.info(f"**Attack Advantage**: {result['match_info']['team_b']} ({imbalance['higher_scorer']} vs {imbalance['lower_scorer']} goals)")
    
    # Three filter indicators with tier info
    st.markdown("### 🔍 Triggered Filters")
    filters = result['filters_triggered']
    
    cols = st.columns(3)
    with cols[0]:
        if filters['under_15_alert']:
            st.error("UNDER 1.5 GOALS ✅")
            st.caption("High certainty defensive match")
        else:
            st.info("UNDER 1.5 GOALS ❌")
    
    with cols[1]:
        if filters['under_25_alert']:
            st.error("UNDER 2.5 GOALS ✅")
            st.caption("Good certainty low-scoring")
        else:
            st.info("UNDER 2.5 GOALS ❌")
    
    with cols[2]:
        if filters['no_draw_candidate']:
            if filters['no_draw_tier'] == 'tier1':
                st.success("NO DRAW TIER 1 ✅")
                st.caption("Maximum certainty")
            else:
                st.warning("NO DRAW TIER 2 ✅")
                st.caption("Good certainty")
            
            debug = filters.get('debug_info', {}).get('no_draw', '')
            if debug:
                st.caption(debug)
        else:
            st.info("NO DRAW ❌")
            debug = filters.get('debug_info', {}).get('no_draw_check', {})
            if debug:
                reasons = []
                if not debug.get('tier1', False) and not debug.get('tier2', False):
                    reasons.append("Criteria not met")
                if debug.get('imbalance'):
                    reasons.append(f"Imbalance: {debug['imbalance']}")
                if reasons:
                    st.caption(", ".join(reasons))
    
    # High-draw league warning
    if filters['high_draw_league_warning']:
        st.error("⚠️ HIGH-DRAW LEAGUE: Extra caution required")
    
    # Betting recommendations
    st.markdown("### 📋 Recommendations")
    script = result['match_script']
    
    if script['triggered_filter'] == 'none':
        st.info(script['match_narrative'])
    else:
        # Show certainty level
        if script['certainty_tier'] == 'tier1':
            st.success("🎯 MAXIMUM CERTAINTY BET")
        elif script['certainty_tier'] == 'tier2':
            st.warning("✅ GOOD CERTAINTY BET")
        
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
        
        # Manual check requirements
        if script['manual_check_required']:
            st.markdown("### ⚠️ ODDS VERIFICATION REQUIRED")
            odds = script['odds_to_check']
            
            if script['certainty_tier'] == 'tier1':
                st.warning(f"**Verify these odds:**")
            else:
                st.error(f"**STRICTLY verify these odds:**")
            
            st.write(f"• Favorite odds must be: {odds['favorite_range']}")
            st.write(f"• Draw odds must be >: {odds['draw_min']}")
            st.write(f"• {odds['check_instructions']}")
            
            if result['trap_team_warning']:
                st.error("**TRAP TEAM WARNING**: Be EXTRA strict with odds. Consider skipping.")
            elif script['certainty_tier'] == 'tier2':
                st.info("**Tier 2 Note**: Be extra strict with odds. Skip if borderline.")
    
    # Predicted scores
    if script['predicted_score_range']:
        st.markdown("### 🎯 Most Likely Scores")
        scores = script['predicted_score_range'][:5]
        score_cols = st.columns(min(5, len(scores)))
        for idx, score in enumerate(scores):
            with score_cols[idx]:
                st.success(f"**{score}**")
    
    # Debug info expander
    with st.expander("🔍 View Detailed Analysis"):
        if filters.get('debug_info'):
            st.write("**Filter Analysis:**")
            st.json(filters['debug_info'], expanded=False)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    st.set_page_config(
        page_title="High-Certainty No Draw Analyzer",
        page_icon="🎯",
        layout="wide"
    )
    
    st.sidebar.title("⚙️ High-Certainty Configuration")
    st.sidebar.markdown("**Strategy**: Maximum accuracy over quantity")
    st.sidebar.markdown("**Bayern Protection**: Auto-rejects ultra-strong favorites")
    
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
    with st.spinner("Analyzing for high-certainty patterns..."):
        engine = HighCertaintyEngine()
        result = engine.analyze_match(match_context, selected_league)
    
    # Display results
    render_high_certainty_dashboard(result)
    
    # Footer
    st.markdown("---")
    st.caption("High-Certainty No Draw Analyzer • Auto-rejects ultra-strong favorites • Tiered certainty system")

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
