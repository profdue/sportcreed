#!/usr/bin/env python3
"""
BETTING ANALYTICS ENGINE - EXACT LOGIC FROM OUR CONVERSATION
No H2H, No Extras - Pure Statistical Pattern Matching
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TeamData:
    """EXACT data structure we've been using"""
    team_name: str
    last6_goals: int          # Last 6 Goals
    last6_wins: int           # Form W
    last6_draws: int          # Form D
    last6_losses: int         # Form L
    last6_clean_sheets: int   # 0/6, 1/6, etc. (from CS column)
    last6_btts_matches: int   # 5/6, 4/6, etc. (from BTTS column)
    overall_matches: int      # Overall Matches
    overall_goals: int        # Overall Goals
    overall_cs_percent: float # Overall CS% (e.g., 40.00)
    overall_btts_percent: float # Overall BTTS% (e.g., 53.33)

@dataclass
class MatchData:
    """Match to analyze"""
    team_a: TeamData
    team_b: TeamData
    team_a_home: bool = True

# ============================================================================
# EXACT 5 FILTERS LOGIC
# ============================================================================

class ExactFilterDetector:
    """IMPLEMENTING OUR EXACT 5 FILTERS LOGIC"""
    
    def __init__(self):
        # EXACT thresholds from our analysis
        self.UNDER_15_GPM = 0.75
        self.BTTS_CS_THRESHOLD = 20  # Overall CS% < 20%
        self.CLEAN_SHEET_STRONG = 50  # Overall CS% > 50%
        self.OPPONENT_GPM_WEAK = 1.0
        self.LOW_BTTS_THRESHOLD = 40  # Last6 BTTS% < 40
        self.DECENT_CS_THRESHOLD = 30  # Last6 CS% > 30
        self.MODERATE_GPM = 1.5
        self.RECENT_BTTS_INFLATED = 70  # Last6 BTTS% > 70
        self.SEASON_BTTS_NORMAL = 60   # Overall BTTS% < 60
        self.GPM_INFLATION = 1.5       # Last6 GPM > Overall GPM × 1.5
    
    def detect_filters(self, team_a: TeamData, team_b: TeamData) -> Dict[str, Any]:
        """
        EXACT 5 filters logic as we've been using
        Returns which filters trigger and betting recommendations
        """
        # Calculate metrics
        a_last6_gpm = team_a.last6_goals / 6
        b_last6_gpm = team_b.last6_goals / 6
        
        a_last6_cs_percent = (team_a.last6_clean_sheets / 6) * 100
        b_last6_cs_percent = (team_b.last6_clean_sheets / 6) * 100
        
        a_last6_btts_percent = (team_a.last6_btts_matches / 6) * 100
        b_last6_btts_percent = (team_b.last6_btts_matches / 6) * 100
        
        a_overall_gpm = team_a.overall_goals / team_a.overall_matches if team_a.overall_matches > 0 else 0
        b_overall_gpm = team_b.overall_goals / team_b.overall_matches if team_b.overall_matches > 0 else 0
        
        results = {
            'filters_triggered': [],
            'primary_bets': [],
            'secondary_bets': [],
            'warnings': [],
            'confidence': 'medium',
            'match_narrative': '',
            'predicted_scores': []
        }
        
        # ====================================================================
        # FILTER 1: Under 1.5 Goals Alert
        # ====================================================================
        if a_last6_gpm < self.UNDER_15_GPM and b_last6_gpm < self.UNDER_15_GPM:
            results['filters_triggered'].append('under_15_alert')
            results['primary_bets'].extend(['under_15_goals', 'btts_no'])
            results['confidence'] = 'high'
            results['match_narrative'] = 'Both teams have broken attacks - expect very few goals'
            results['predicted_scores'] = ['0-0', '1-0', '0-1']
            return results  # FILTER 1 overrides everything
        
        # ====================================================================
        # FILTER 2: BTTS Banker Alert
        # ====================================================================
        # Condition: BOTH teams have 0 clean sheets in last 6
        # OR BOTH teams have Overall CS% < 20%
        btts_trigger = False
        
        # Condition 2A: Both teams 0 clean sheets in last 6
        if team_a.last6_clean_sheets == 0 and team_b.last6_clean_sheets == 0:
            btts_trigger = True
            trigger_reason = f"Both teams 0 clean sheets in last 6"
        
        # Condition 2B: Both teams Overall CS% < 20%
        elif team_a.overall_cs_percent < self.BTTS_CS_THRESHOLD and team_b.overall_cs_percent < self.BTTS_CS_THRESHOLD:
            btts_trigger = True
            trigger_reason = f"Both teams Overall CS% < {self.BTTS_CS_THRESHOLD}%"
        
        if btts_trigger:
            results['filters_triggered'].append('btts_banker')
            results['primary_bets'].append('btts_yes')
            results['secondary_bets'].append('over_15_goals')
            results['confidence'] = 'high'
            results['match_narrative'] = f'Both teams consistently concede - {trigger_reason}'
            
            # Check if we should add over 2.5 (when teams score well)
            if a_last6_gpm > 1.5 or b_last6_gpm > 1.5:
                results['secondary_bets'].append('over_25_goals')
                results['predicted_scores'] = ['1-1', '2-1', '1-2', '2-2']
            else:
                results['predicted_scores'] = ['1-1', '2-1', '1-2']
            
            # Don't return - check other filters too
        
        # ====================================================================
        # FILTER 3: Clean Sheet Alert
        # ====================================================================
        # Condition: One team Overall CS% > 50% AND opponent Last6 GPM < 1.0
        
        # Team A clean sheet possibility
        if team_a.overall_cs_percent > self.CLEAN_SHEET_STRONG and b_last6_gpm < self.OPPONENT_GPM_WEAK:
            results['filters_triggered'].append('clean_sheet_alert')
            results['primary_bets'].append(f'{team_a.team_name.lower().replace(" ", "_")}_win_to_nil')
            results['secondary_bets'].append('under_25_goals')
            results['confidence'] = 'high'
            results['match_narrative'] = f'{team_a.team_name} strong defensively ({team_a.overall_cs_percent:.1f}% CS) vs {team_b.team_name} weak attack ({b_last6_gpm:.2f} GPM)'
            results['predicted_scores'] = ['1-0', '2-0', '3-0']
            return results  # Filter 3 overrides BTTS Banker
        
        # Team B clean sheet possibility
        elif team_b.overall_cs_percent > self.CLEAN_SHEET_STRONG and a_last6_gpm < self.OPPONENT_GPM_WEAK:
            results['filters_triggered'].append('clean_sheet_alert')
            results['primary_bets'].append(f'{team_b.team_name.lower().replace(" ", "_")}_win_to_nil')
            results['secondary_bets'].append('under_25_goals')
            results['confidence'] = 'high'
            results['match_narrative'] = f'{team_b.team_name} strong defensively ({team_b.overall_cs_percent:.1f}% CS) vs {team_a.team_name} weak attack ({a_last6_gpm:.2f} GPM)'
            results['predicted_scores'] = ['0-1', '0-2', '0-3']
            return results  # Filter 3 overrides BTTS Banker
        
        # ====================================================================
        # FILTER 4: Low-Scoring Alert
        # ====================================================================
        # Condition 4A: Both teams Last6 BTTS% < 40%
        if a_last6_btts_percent < self.LOW_BTTS_THRESHOLD and b_last6_btts_percent < self.LOW_BTTS_THRESHOLD:
            results['filters_triggered'].append('low_scoring_alert')
            results['primary_bets'].extend(['under_25_goals', 'btts_no'])
            results['confidence'] = 'high'
            results['match_narrative'] = f'Both teams rarely see both teams score ({a_last6_btts_percent:.1f}% vs {b_last6_btts_percent:.1f}%) - low-scoring affair likely'
            results['predicted_scores'] = ['0-0', '1-0', '0-1']
            return results  # Filter 4 overrides BTTS Banker
        
        # Condition 4B: Defensive team vs Leaky but low-scoring team
        # Team A defensive, Team B low-scoring
        elif (a_last6_btts_percent < self.LOW_BTTS_THRESHOLD and 
              a_last6_cs_percent > self.DECENT_CS_THRESHOLD and 
              b_last6_gpm < self.MODERATE_GPM):
            results['filters_triggered'].append('low_scoring_alert')
            results['primary_bets'].extend(['under_25_goals', 'btts_no'])
            results['confidence'] = 'high'
            results['match_narrative'] = f'{team_a.team_name} solid defensively vs {team_b.team_name} limited attack - clean sheet likely'
            results['predicted_scores'] = ['1-0', '2-0', '0-0']
            return results  # Filter 4 overrides BTTS Banker
        
        # Team B defensive, Team A low-scoring
        elif (b_last6_btts_percent < self.LOW_BTTS_THRESHOLD and 
              b_last6_cs_percent > self.DECENT_CS_THRESHOLD and 
              a_last6_gpm < self.MODERATE_GPM):
            results['filters_triggered'].append('low_scoring_alert')
            results['primary_bets'].extend(['under_25_goals', 'btts_no'])
            results['confidence'] = 'high'
            results['match_narrative'] = f'{team_b.team_name} solid defensively vs {team_a.team_name} limited attack - clean sheet likely'
            results['predicted_scores'] = ['0-1', '0-2', '0-0']
            return results  # Filter 4 overrides BTTS Banker
        
        # ====================================================================
        # FILTER 5: Regression Alert
        # ====================================================================
        # Check for statistical regression warnings
        warnings = []
        
        # Condition 5A: Last6 BTTS% > 70% but Overall BTTS% < 60%
        if (a_last6_btts_percent > self.RECENT_BTTS_INFLATED and 
            team_a.overall_btts_percent < self.SEASON_BTTS_NORMAL):
            warnings.append(f"{team_a.team_name}: Recent BTTS rate ({a_last6_btts_percent:.1f}%) inflated vs season ({team_a.overall_btts_percent:.1f}%) - expect regression")
        
        if (b_last6_btts_percent > self.RECENT_BTTS_INFLATED and 
            team_b.overall_btts_percent < self.SEASON_BTTS_NORMAL):
            warnings.append(f"{team_b.team_name}: Recent BTTS rate ({b_last6_btts_percent:.1f}%) inflated vs season ({team_b.overall_btts_percent:.1f}%) - expect regression")
        
        # Condition 5B: Last6 GPM > Overall GPM × 1.5
        if a_overall_gpm > 0 and a_last6_gpm > (a_overall_gpm * self.GPM_INFLATION):
            warnings.append(f"{team_a.team_name}: Recent scoring ({a_last6_gpm:.2f} GPM) inflated vs season ({a_overall_gpm:.2f} GPM)")
        
        if b_overall_gpm > 0 and b_last6_gpm > (b_overall_gpm * self.GPM_INFLATION):
            warnings.append(f"{team_b.team_name}: Recent scoring ({b_last6_gpm:.2f} GPM) inflated vs season ({b_overall_gpm:.2f} GPM)")
        
        if warnings:
            results['filters_triggered'].append('regression_alert')
            results['warnings'] = warnings
        
        # ====================================================================
        # NO FILTERS TRIGGERED - Use current form analysis
        # ====================================================================
        if not results['primary_bets']:
            # Determine favorite based on Last 6 wins
            win_diff = team_a.last6_wins - team_b.last6_wins
            
            if win_diff >= 2:
                favorite = team_a.team_name
                results['secondary_bets'].append(f'{team_a.team_name.lower().replace(" ", "_")}_win_or_draw')
                results['match_narrative'] = f'{team_a.team_name} has better recent form ({win_diff} more wins)'
            elif win_diff <= -2:
                favorite = team_b.team_name
                results['secondary_bets'].append(f'{team_b.team_name.lower().replace(" ", "_")}_win_or_draw')
                results['match_narrative'] = f'{team_b.team_name} has better recent form ({abs(win_diff)} more wins)'
            else:
                results['match_narrative'] = 'Teams have similar recent form'
            
            # Add goal market bets based on combined GPM
            avg_gpm = (a_last6_gpm + b_last6_gpm) / 2
            
            if avg_gpm > 1.8:
                results['secondary_bets'].append('over_15_goals')
                if avg_gpm > 2.2:
                    results['secondary_bets'].append('over_25_goals')
            elif avg_gpm < 1.2:
                results['secondary_bets'].append('under_25_goals')
            
            # Add BTTS based on Last 6 BTTS%
            avg_btts = (a_last6_btts_percent + b_last6_btts_percent) / 2
            
            if avg_btts > 60:
                results['secondary_bets'].append('btts_yes')
            elif avg_btts < 40:
                results['secondary_bets'].append('btts_no')
            
            # Generate score predictions
            if avg_gpm > 1.8:
                results['predicted_scores'] = ['2-1', '1-2', '2-2', '3-1', '1-3']
            elif avg_gpm < 1.2:
                results['predicted_scores'] = ['1-0', '0-1', '1-1', '0-0']
            else:
                results['predicted_scores'] = ['1-1', '2-1', '1-2', '2-0', '0-2']
            
            results['confidence'] = 'medium' if win_diff != 0 else 'low'
        
        return results

# ============================================================================
# DATA PARSING - EXACT CSV FORMAT
# ============================================================================

def parse_csv_row(row: pd.Series) -> TeamData:
    """
    Parse EXACT CSV format from your files:
    Team, Form Matches, Form W, Form D, Form L, Last 6 Goals, Overall Matches, 
    Overall Goals, Overall CS%, Overall BTTS%, ...
    """
    team_name = str(row['Team'])
    
    # Last 6 data
    last6_goals = int(row['Last 6 Goals'])
    last6_wins = int(row['Form W'])
    last6_draws = int(row['Form D'])
    last6_losses = int(row['Form L'])
    
    # Extract Last 6 CS and BTTS from fractions or calculate
    # NOTE: Your CSV doesn't have Last 6 CS/BTTS columns directly
    # We'll use Overall percentages to estimate
    overall_cs = float(str(row['Overall CS%']).replace('%', ''))
    overall_btts = float(str(row['Overall BTTS%']).replace('%', ''))
    
    # Estimate Last 6 from Overall (simplification - your CSV should add these columns)
    last6_clean_sheets = max(0, min(6, round((overall_cs / 100) * 6)))
    last6_btts_matches = max(0, min(6, round((overall_btts / 100) * 6)))
    
    # Overall data
    overall_matches = int(row['Overall Matches'])
    overall_goals = int(row['Overall Goals'])
    
    return TeamData(
        team_name=team_name,
        last6_goals=last6_goals,
        last6_wins=last6_wins,
        last6_draws=last6_draws,
        last6_losses=last6_losses,
        last6_clean_sheets=last6_clean_sheets,
        last6_btts_matches=last6_btts_matches,
        overall_matches=overall_matches,
        overall_goals=overall_goals,
        overall_cs_percent=overall_cs,
        overall_btts_percent=overall_btts
    )

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Betting Analytics Engine - Exact Logic",
        page_icon="⚽",
        layout="wide"
    )
    
    # Title
    st.title("⚽ Betting Analytics Engine")
    st.markdown("**EXACT 5-Filter Logic • No H2H Data • Pure Statistics**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Selection")
        
        # League selection
        leagues = [
            'bundesliga', 'bundesliga_2', 'championship', 'eredivisie',
            'erste_divisie', 'laliga', 'laliga_2', 'ligue_1', 'ligue_2',
            'premier_league', 'serie_a', 'serie_b'
        ]
        selected_league = st.selectbox("Select League", leagues)
        
        # Load data
        try:
            df = pd.read_csv(f"leagues/{selected_league}.csv")
        except:
            st.error(f"Could not load {selected_league}.csv")
            st.stop()
        
        # Team selection
        teams = sorted(df['Team'].tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            team_a_name = st.selectbox("Team A", teams, key="team_a")
        with col2:
            available_teams = [t for t in teams if t != team_a_name]
            team_b_name = st.selectbox("Team B", available_teams, key="team_b")
        
        # Venue
        venue = st.radio("Venue", ["Team A Home", "Team B Home"])
        team_a_home = venue == "Team A Home"
        
        # Show data warning
        st.info("⚠️ Note: Last 6 CS/BTTS estimated from Overall %")
        st.caption("For best results, add 'Last 6 CS' and 'Last 6 BTTS' columns to CSV")
    
    # Parse team data
    team_a_row = df[df['Team'] == team_a_name].iloc[0]
    team_b_row = df[df['Team'] == team_b_name].iloc[0]
    
    team_a_data = parse_csv_row(team_a_row)
    team_b_data = parse_csv_row(team_b_row)
    
    # Display team stats
    st.subheader(f"📊 {team_a_name} vs {team_b_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 🏠 {team_a_name}")
        st.metric("Last 6 Goals", team_a_data.last6_goals)
        st.metric("Form", f"{team_a_data.last6_wins}W-{team_a_data.last6_draws}D-{team_a_data.last6_losses}L")
        st.metric("Est. Last 6 CS", f"{team_a_data.last6_clean_sheets}/6")
        st.metric("Est. Last 6 BTTS", f"{team_a_data.last6_btts_matches}/6")
        st.caption(f"Overall: {team_a_data.overall_cs_percent:.1f}% CS, {team_a_data.overall_btts_percent:.1f}% BTTS")
    
    with col2:
        st.markdown(f"### 🚌 {team_b_name}")
        st.metric("Last 6 Goals", team_b_data.last6_goals)
        st.metric("Form", f"{team_b_data.last6_wins}W-{team_b_data.last6_draws}D-{team_b_data.last6_losses}L")
        st.metric("Est. Last 6 CS", f"{team_b_data.last6_clean_sheets}/6")
        st.metric("Est. Last 6 BTTS", f"{team_b_data.last6_btts_matches}/6")
        st.caption(f"Overall: {team_b_data.overall_cs_percent:.1f}% CS, {team_b_data.overall_btts_percent:.1f}% BTTS")
    
    st.markdown("---")
    
    # Run analysis button
    if st.button("🚀 Run Exact Analysis", type="primary", use_container_width=True):
        with st.spinner("Running exact 5-filter analysis..."):
            # Create match data
            match_data = MatchData(
                team_a=team_a_data,
                team_b=team_b_data,
                team_a_home=team_a_home
            )
            
            # Run detector
            detector = ExactFilterDetector()
            results = detector.detect_filters(team_a_data, team_b_data)
            
            # Display results
            st.header("🎯 Analysis Results")
            
            # Filters triggered
            st.subheader("🔍 Filters Triggered")
            filter_cols = st.columns(5)
            
            filter_info = {
                'under_15_alert': ('🔴', 'Under 1.5 Alert', 'Both teams GPM < 0.75'),
                'btts_banker': ('🟢', 'BTTS Banker', 'Both teams 0 CS or CS% < 20%'),
                'clean_sheet_alert': ('🔵', 'Clean Sheet Alert', 'Strong defense vs weak attack'),
                'low_scoring_alert': ('🟡', 'Low-Scoring Alert', 'Low BTTS patterns'),
                'regression_alert': ('🟣', 'Regression Alert', 'Recent stats inflated')
            }
            
            active_filters = []
            for filter_key, (icon, name, desc) in filter_info.items():
                if filter_key in results['filters_triggered']:
                    active_filters.append((icon, name, desc))
            
            if active_filters:
                for idx, (icon, name, desc) in enumerate(active_filters):
                    with filter_cols[idx]:
                        st.markdown(f"### {icon}")
                        st.markdown(f"**{name}**")
                        st.caption(desc)
            else:
                st.info("No extreme filters triggered")
            
            # Confidence
            st.subheader("📈 Confidence Level")
            confidence_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}
            st.metric("Analysis Confidence", 
                     f"{results['confidence'].title()} {confidence_color[results['confidence']]}")
            
            # Match narrative
            st.subheader("📝 Match Analysis")
            st.info(results['match_narrative'])
            
            # Warnings
            if results['warnings']:
                st.subheader("⚠️ Statistical Warnings")
                for warning in results['warnings']:
                    st.warning(warning)
            
            # Betting recommendations
            st.subheader("💰 Betting Recommendations")
            
            if results['primary_bets']:
                st.markdown("**🎯 Primary Bets (High Confidence)**")
                for bet in results['primary_bets']:
                    bet_display = bet.replace('_', ' ').title()
                    if '_win_to_nil' in bet:
                        team = bet.replace('_win_to_nil', '').replace('_', ' ').title()
                        st.success(f"✅ **{team} Win to Nil** - Strong defense vs weak attack")
                    elif bet == 'under_15_goals':
                        st.success(f"✅ **Under 1.5 Goals** - Both teams broken attacks")
                    elif bet == 'btts_yes':
                        st.success(f"✅ **BTTS Yes** - Both teams consistently concede")
                    elif bet == 'btts_no':
                        st.success(f"✅ **BTTS No** - Low BTTS likelihood")
                    elif bet == 'under_25_goals':
                        st.success(f"✅ **Under 2.5 Goals** - Low-scoring patterns")
                    else:
                        st.success(f"✅ **{bet_display}**")
            
            if results['secondary_bets']:
                st.markdown("**📊 Secondary Bets (Medium Confidence)**")
                for bet in results['secondary_bets']:
                    bet_display = bet.replace('_', ' ').title()
                    if '_win_or_draw' in bet:
                        team = bet.replace('_win_or_draw', '').replace('_', ' ').title()
                        st.info(f"• **{team} Win or Draw** - Better recent form")
                    elif bet == 'over_15_goals':
                        st.info(f"• **Over 1.5 Goals** - Good scoring potential")
                    elif bet == 'over_25_goals':
                        st.info(f"• **Over 2.5 Goals** - High scoring potential")
                    elif bet == 'under_25_goals':
                        st.info(f"• **Under 2.5 Goals** - Moderate scoring")
                    elif bet == 'btts_yes':
                        st.info(f"• **BTTS Yes** - Good BTTS likelihood")
                    elif bet == 'btts_no':
                        st.info(f"• **BTTS No** - Low BTTS likelihood")
                    else:
                        st.info(f"• **{bet_display}**")
            
            if not results['primary_bets'] and not results['secondary_bets']:
                st.warning("No clear betting recommendations - confidence too low")
            
            # Predicted scores
            if results['predicted_scores']:
                st.subheader("🎯 Predicted Score Range")
                score_cols = st.columns(min(5, len(results['predicted_scores'])))
                for idx, score in enumerate(results['predicted_scores']):
                    with score_cols[idx]:
                        st.markdown(f"<div style='text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'><h4>{score}</h4></div>", 
                                  unsafe_allow_html=True)
            
            # Show exact calculations
            with st.expander("📊 Show Exact Calculations"):
                st.subheader("Team A Calculations")
                st.write(f"- Last 6 GPM: {team_a_data.last6_goals / 6:.2f}")
                st.write(f"- Last 6 CS%: {(team_a_data.last6_clean_sheets / 6) * 100:.1f}%")
                st.write(f"- Last 6 BTTS%: {(team_a_data.last6_btts_matches / 6) * 100:.1f}%")
                st.write(f"- Overall GPM: {team_a_data.overall_goals / team_a_data.overall_matches if team_a_data.overall_matches > 0 else 0:.2f}")
                
                st.subheader("Team B Calculations")
                st.write(f"- Last 6 GPM: {team_b_data.last6_goals / 6:.2f}")
                st.write(f"- Last 6 CS%: {(team_b_data.last6_clean_sheets / 6) * 100:.1f}%")
                st.write(f"- Last 6 BTTS%: {(team_b_data.last6_btts_matches / 6) * 100:.1f}%")
                st.write(f"- Overall GPM: {team_b_data.overall_goals / team_b_data.overall_matches if team_b_data.overall_matches > 0 else 0:.2f}")
                
                st.subheader("Filter Checks")
                st.write(f"1. Under 1.5 Alert: {team_a_data.last6_goals / 6:.2f} < 0.75 AND {team_b_data.last6_goals / 6:.2f} < 0.75 = {results['filters_triggered'].count('under_15_alert') > 0}")
                st.write(f"2. BTTS Banker: (CS A=0 AND CS B=0) OR (CS% A<20 AND CS% B<20) = {results['filters_triggered'].count('btts_banker') > 0}")
                st.write(f"3. Clean Sheet: (CS% A>50 AND GPM B<1.0) OR (CS% B>50 AND GPM A<1.0) = {results['filters_triggered'].count('clean_sheet_alert') > 0}")
                st.write(f"4. Low-Scoring: (BTTS% A<40 AND BTTS% B<40) OR defensive patterns = {results['filters_triggered'].count('low_scoring_alert') > 0}")
                st.write(f"5. Regression: BTTS% inflation or GPM inflation = {results['filters_triggered'].count('regression_alert') > 0}")
            
            # Raw results
            with st.expander("📋 View Raw Results"):
                st.json(results)

if __name__ == "__main__":
    main()
