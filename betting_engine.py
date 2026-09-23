import streamlit as st
from datetime import date, datetime
from supabase import create_client, Client
import pandas as pd
import re
import traceback
import numpy as np
from scipy.stats import poisson
from typing import Optional

# ============================================================================
# SUPABASE SETUP
# ============================================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()

# ============================================================================
# TABLE NAME
# ============================================================================
TABLE_NAME = "match_predictions"

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Refined Prediction Strategy", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1200px; }
    .output-card { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 16px; padding: 1.25rem; margin: 0.75rem 0; color: #ffffff; }
    .bet-card { border-left: 5px solid #10b981; background: linear-gradient(135deg, #0a2a1a 0%, #0a1a0a 100%); }
    .selective-card { border-left: 5px solid #fbbf24; background: linear-gradient(135deg, #2a2a00 0%, #1a1a00 100%); }
    .skip-card { border-left: 5px solid #64748b; background: linear-gradient(135deg, #1a1a2a 0%, #0a0a1a 100%); }
    .avoid-card { border-left: 5px solid #ef4444; background: linear-gradient(135deg, #2a0a0a 0%, #1a0a0a 100%); }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 12px; padding: 0.6rem 1rem; border: none; width: 100%; }
    .stat-box { background: #1e293b; border-radius: 10px; padding: 0.8rem; text-align: center; color: #fff; }
    .stat-number { font-size: 2rem; font-weight: 800; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; }
    .prediction-display { font-size: 2rem; font-weight: 800; text-align: center; padding: 0.5rem; }
    .prediction-bet { color: #10b981; }
    .prediction-selective { color: #fbbf24; }
    .prediction-skip { color: #64748b; }
    .prediction-avoid { color: #ef4444; }
    .badge { padding: 0.3rem 0.75rem; border-radius: 8px; font-size: 0.8rem; font-weight: 700; display: inline-block; }
    .badge-bet { background: #10b981; color: #000; }
    .badge-selective { background: #fbbf24; color: #000; }
    .badge-skip { background: #64748b; color: #fff; }
    .badge-avoid { background: #ef4444; color: #fff; }
    .feature-box { background: #0f172a; border-radius: 6px; padding: 0.5rem; margin: 0.25rem 0; }
    .feature-label { color: #94a3b8; font-size: 0.7rem; }
    .feature-value { font-weight: 700; font-size: 1rem; }
    .edge-positive { color: #10b981; }
    .edge-negative { color: #ef4444; }
    .tier-header { padding: 0.5rem 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .tier-bet { background: rgba(16, 185, 129, 0.2); border-left: 4px solid #10b981; }
    .tier-selective { background: rgba(251, 191, 36, 0.2); border-left: 4px solid #fbbf24; }
    .tier-skip { background: rgba(100, 116, 139, 0.2); border-left: 4px solid #64748b; }
    .actual-score { font-size: 1.2rem; font-weight: 700; padding: 0.3rem 0.75rem; border-radius: 8px; display: inline-block; }
    .score-win { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .score-loss { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    .score-pending { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
    .xG-display { font-size: 1.5rem; font-weight: 700; color: #3b82f6; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def parse_match_date(date_val) -> datetime:
    if not date_val:
        return datetime(1900, 1, 1)
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    date_str = str(date_val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return datetime(1900, 1, 1)


def format_date_display(date_val) -> str:
    dt = parse_match_date(date_val)
    if dt.year == 1900:
        return str(date_val)
    return dt.strftime("%Y-%m-%d")


def check_match_exists(home_team: str, away_team: str, match_date: str) -> bool:
    try:
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]
        response = supabase.table(TABLE_NAME).select("id").eq("home_team", home_team).eq("away_team", away_team).eq("match_date", date_part).execute()
        return len(response.data) > 0
    except:
        return False


# ============================================================================
# CORE STRATEGY ENGINE
# ============================================================================

class RefinedPredictor:
    """
    Implements the Refined Prediction Strategy:
    - Calculate xG from home/away splits
    - Apply adjustments (form, injuries, fatigue)
    - Shrink 50% toward market total
    - Run Poisson
    - Check value (edge 5%-30%)
    - Select market (AH, BTTS, O/U, Corners)
    """
    
    def __init__(self):
        self.model_xg_home = 0.0
        self.model_xg_away = 0.0
        self.shrunk_xg_home = 0.0
        self.shrunk_xg_away = 0.0
        self.market_total = 0.0
        self.model_total = 0.0
        self.shrunk_total = 0.0
        self.probabilities = {}
        self.market_probs = {}
        self.edges = {}
        self.bets = []
        self.skips = []
        
    def calculate_base_xg(self, home_data: dict, away_data: dict) -> tuple:
        """
        Step 2: Calculate Expected Goals
        
        Home_xG = (Home_Attack_Home + Away_Defence_Away) / 2
        Away_xG = (Away_Attack_Away + Home_Defence_Home) / 2
        """
        # Home team's goals scored per game at home (blended season/last-10)
        home_attack_home = self._blend_split(
            home_data.get("home_goals_scored_season", 1.5),
            home_data.get("home_goals_scored_last10", 1.5)
        )
        
        # Away team's goals conceded per game away
        away_defence_away = self._blend_split(
            away_data.get("away_goals_conceded_season", 1.5),
            away_data.get("away_goals_conceded_last10", 1.5)
        )
        
        # Away team's goals scored per game away
        away_attack_away = self._blend_split(
            away_data.get("away_goals_scored_season", 1.2),
            away_data.get("away_goals_scored_last10", 1.2)
        )
        
        # Home team's goals conceded per game at home
        home_defence_home = self._blend_split(
            home_data.get("home_goals_conceded_season", 1.2),
            home_data.get("home_goals_conceded_last10", 1.2)
        )
        
        home_xg = (home_attack_home + away_defence_away) / 2
        away_xg = (away_attack_away + home_defence_home) / 2
        
        self.model_xg_home = max(0.1, home_xg)  # Prevent zero
        self.model_xg_away = max(0.1, away_xg)
        self.model_total = self.model_xg_home + self.model_xg_away
        
        return self.model_xg_home, self.model_xg_away
    
    def _blend_split(self, season_val: float, last10_val: float, 
                     season_weight: float = 0.30) -> float:
        """
        Blend season splits and last-10 splits.
        Default: 70% last 10, 30% season (as per strategy)
        """
        if season_val <= 0:
            return max(0.1, last10_val)
        if last10_val <= 0:
            return max(0.1, season_val)
        return (last10_val * (1 - season_weight)) + (season_val * season_weight)
    
    def apply_adjustments(self, home_data: dict, away_data: dict) -> tuple:
        """
        Step 3: Apply adjustments
        
        - Form adjustment (±0.10–0.15 xG)
        - Injury adjustment (±0.10–0.15 per key player)
        - Midweek fatigue (−0.05–0.10 xG)
        """
        home_adj = 0.0
        away_adj = 0.0
        
        # --- Form adjustment ---
        # Last 5 games: poor form (0–3 pts) → reduce; strong form (10–15 pts) → increase
        home_form_pts = home_data.get("last5_points", 7)
        away_form_pts = away_data.get("last5_points", 7)
        
        home_adj += self._form_adjustment(home_form_pts)
        away_adj += self._form_adjustment(away_form_pts)
        
        # --- Injury adjustment ---
        # Key forward out → reduce that team's xG
        # Key defender out → increase opponent's xG
        # Key midfielder out → small reduction
        home_injuries = home_data.get("injuries", [])
        away_injuries = away_data.get("injuries", [])
        
        for inj in home_injuries:
            if inj.get("position") == "forward" and inj.get("key", False):
                home_adj -= 0.15 if inj.get("confirmed_out", True) else 0.075
            elif inj.get("position") == "defender" and inj.get("key", False):
                away_adj += 0.15 if inj.get("confirmed_out", True) else 0.075
            elif inj.get("position") == "midfielder" and inj.get("key", False):
                home_adj -= 0.10 if inj.get("confirmed_out", True) else 0.05
        
        for inj in away_injuries:
            if inj.get("position") == "forward" and inj.get("key", False):
                away_adj -= 0.15 if inj.get("confirmed_out", True) else 0.075
            elif inj.get("position") == "defender" and inj.get("key", False):
                home_adj += 0.15 if inj.get("confirmed_out", True) else 0.075
            elif inj.get("position") == "midfielder" and inj.get("key", False):
                away_adj -= 0.10 if inj.get("confirmed_out", True) else 0.05
        
        # --- Midweek fatigue ---
        # If a team played a European fixture midweek, reduce their xG slightly
        if home_data.get("played_midweek", False):
            home_adj -= 0.05
        if away_data.get("played_midweek", False):
            away_adj -= 0.10  # More relevant for away teams travelling
        
        # Apply adjustments
        self.model_xg_home = max(0.1, self.model_xg_home + home_adj)
        self.model_xg_away = max(0.1, self.model_xg_away + away_adj)
        self.model_total = self.model_xg_home + self.model_xg_away
        
        return home_adj, away_adj
    
    def _form_adjustment(self, points: int) -> float:
        """Convert last-5 points to xG adjustment"""
        if points <= 1:
            return -0.15
        elif points <= 3:
            return -0.10
        elif points >= 13:
            return 0.15
        elif points >= 10:
            return 0.10
        return 0.0
    
    def shrink_toward_market(self, market_total: float) -> tuple:
        """
        Step 4: Shrink toward the market
        
        Shrunk_Total = 0.5 × Model_Total + 0.5 × Market_Total
        Scale = Shrunk_Total / Model_Total
        """
        self.market_total = max(0.5, market_total)
        self.shrunk_total = 0.5 * self.model_total + 0.5 * self.market_total
        
        if self.model_total > 0:
            scale = self.shrunk_total / self.model_total
        else:
            scale = 1.0
        
        self.shrunk_xg_home = self.model_xg_home * scale
        self.shrunk_xg_away = self.model_xg_away * scale
        
        # Ensure minimum values
        self.shrunk_xg_home = max(0.1, self.shrunk_xg_home)
        self.shrunk_xg_away = max(0.1, self.shrunk_xg_away)
        self.shrunk_total = self.shrunk_xg_home + self.shrunk_xg_away
        
        return self.shrunk_xg_home, self.shrunk_xg_away
    
    def run_poisson(self, max_goals: int = 10) -> dict:
        """
        Step 5: Run Poisson
        
        Using the shrunk xG values as Poisson means, compute:
        - P(Home win), P(Draw), P(Away win)
        - P(win by 1), P(win by 2+) for AH settlement
        """
        home_probs = [poisson.pmf(i, self.shrunk_xg_home) for i in range(max_goals + 1)]
        away_probs = [poisson.pmf(i, self.shrunk_xg_away) for i in range(max_goals + 1)]
        
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        
        # Score matrix
        score_matrix = {}
        
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                score_matrix[(h, a)] = prob
                
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
        
        # Normalize (in case of truncation)
        total_prob = p_home + p_draw + p_away
        if total_prob > 0:
            p_home /= total_prob
            p_draw /= total_prob
            p_away /= total_prob
        
        # AH probabilities
        p_home_win_by_1 = 0.0
        p_home_win_by_2plus = 0.0
        p_away_win_by_1 = 0.0
        p_away_win_by_2plus = 0.0
        
        for (h, a), prob in score_matrix.items():
            if h > a:
                if h - a == 1:
                    p_home_win_by_1 += prob
                else:
                    p_home_win_by_2plus += prob
            elif a > h:
                if a - h == 1:
                    p_away_win_by_1 += prob
                else:
                    p_away_win_by_2plus += prob
        
        # BTTS probability
        p_btts_yes = 0.0
        p_btts_no = 0.0
        
        for (h, a), prob in score_matrix.items():
            if h >= 1 and a >= 1:
                p_btts_yes += prob
            else:
                p_btts_no += prob
        
        # Over/Under probabilities
        p_over_25 = 0.0
        p_under_25 = 0.0
        
        for (h, a), prob in score_matrix.items():
            if h + a > 2.5:
                p_over_25 += prob
            else:
                p_under_25 += prob
        
        # Asian Handicap probabilities
        # Home -0.25: Half win if win, half loss if draw
        # Home -0.5: Win if win, loss if draw
        # Home +0.25: Half win if draw, half loss if loss
        # Home +0.5: Win if draw or win, loss if loss
        
        self.probabilities = {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "home_win_by_1": p_home_win_by_1,
            "home_win_by_2plus": p_home_win_by_2plus,
            "away_win_by_1": p_away_win_by_1,
            "away_win_by_2plus": p_away_win_by_2plus,
            "btts_yes": p_btts_yes,
            "btts_no": p_btts_no,
            "over_25": p_over_25,
            "under_25": p_under_25,
            "home_ah_minus_05": p_home,  # Win if home wins
            "home_ah_minus_025": p_home + (p_draw * 0.5),  # Half win if win, half loss if draw
            "away_ah_minus_05": p_away,
            "away_ah_minus_025": p_away + (p_draw * 0.5),
            "home_ah_plus_05": p_home + p_draw,
            "home_ah_plus_025": p_home + (p_draw * 0.5),
            "away_ah_plus_05": p_away + p_draw,
            "away_ah_plus_025": p_away + (p_draw * 0.5),
        }
        
        return self.probabilities
    
    def calculate_edges(self, odds: dict) -> dict:
        """
        Step 6: Compare to market
        
        Implied_Prob = 1 / Odds
        Edge = My_Prob - Implied_Prob
        """
        self.market_probs = {}
        self.edges = {}
        
        # Standard 1X2
        if odds.get("home_odds", 0) > 1:
            implied_home = 1 / odds["home_odds"]
            self.market_probs["home_win"] = implied_home
            self.edges["home_win"] = self.probabilities["home_win"] - implied_home
        
        if odds.get("draw_odds", 0) > 1:
            implied_draw = 1 / odds["draw_odds"]
            self.market_probs["draw"] = implied_draw
            self.edges["draw"] = self.probabilities["draw"] - implied_draw
        
        if odds.get("away_odds", 0) > 1:
            implied_away = 1 / odds["away_odds"]
            self.market_probs["away_win"] = implied_away
            self.edges["away_win"] = self.probabilities["away_win"] - implied_away
        
        # BTTS
        if odds.get("btts_yes_odds", 0) > 1:
            implied_btts = 1 / odds["btts_yes_odds"]
            self.market_probs["btts_yes"] = implied_btts
            self.edges["btts_yes"] = self.probabilities["btts_yes"] - implied_btts
        
        if odds.get("btts_no_odds", 0) > 1:
            implied_btts_no = 1 / odds["btts_no_odds"]
            self.market_probs["btts_no"] = implied_btts_no
            self.edges["btts_no"] = self.probabilities["btts_no"] - implied_btts_no
        
        # Over/Under
        if odds.get("over_25_odds", 0) > 1:
            implied_over = 1 / odds["over_25_odds"]
            self.market_probs["over_25"] = implied_over
            self.edges["over_25"] = self.probabilities["over_25"] - implied_over
        
        if odds.get("under_25_odds", 0) > 1:
            implied_under = 1 / odds["under_25_odds"]
            self.market_probs["under_25"] = implied_under
            self.edges["under_25"] = self.probabilities["under_25"] - implied_under
        
        # Asian Handicap (derived from 1X2 if not explicitly provided)
        if odds.get("home_ah_minus_05_odds", 0) > 1:
            implied = 1 / odds["home_ah_minus_05_odds"]
            self.edges["home_ah_minus_05"] = self.probabilities["home_ah_minus_05"] - implied
        
        if odds.get("away_ah_minus_05_odds", 0) > 1:
            implied = 1 / odds["away_ah_minus_05_odds"]
            self.edges["away_ah_minus_05"] = self.probabilities["away_ah_minus_05"] - implied
        
        return self.edges
    
    def select_markets(self, odds: dict, btts_rate: float = 0.5, 
                       corner_data: dict = None) -> list:
        """
        Step 7: Select Market
        
        Based on the edge and market conditions, determine which bets to place.
        """
        self.bets = []
        self.skips = []
        
        # --- MATCH RESULT (Core Bet) ---
        for outcome in ["home_win", "away_win", "draw"]:
            if outcome in self.edges:
                edge = self.edges[outcome]
                prob = self.probabilities[outcome]
                
                # Edge > 30% → SKIP (model error)
                if edge > 0.30:
                    self.skips.append({
                        "market": outcome,
                        "reason": f"Edge {edge:.1%} > 30% (model error signal)",
                        "edge": edge,
                        "prob": prob
                    })
                    continue
                
                # Edge < 5% → SKIP
                if edge < 0.05:
                    self.skips.append({
                        "market": outcome,
                        "reason": f"Edge {edge:.1%} < 5% (insufficient value)",
                        "edge": edge,
                        "prob": prob
                    })
                    continue
                
                # Value band: 5% < Edge < 30%
                if outcome == "home_win" and prob > 0.45:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Home -0.25 or -0.5 AH",
                        "prob": prob,
                        "edge": edge,
                        "odds": odds.get("home_odds", 0),
                        "stake": "1 unit",
                        "confidence": "High"
                    })
                elif outcome == "away_win" and prob > 0.45:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Away -0.25 or -0.5 AH",
                        "prob": prob,
                        "edge": edge,
                        "odds": odds.get("away_odds", 0),
                        "stake": "1 unit",
                        "confidence": "High"
                    })
                elif outcome == "draw" and prob > 0.28:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Draw or Underdog +0.25",
                        "prob": prob,
                        "edge": edge,
                        "odds": odds.get("draw_odds", 0),
                        "stake": "1 unit",
                        "confidence": "High"
                    })
        
        # Underdog check
        home_prob = self.probabilities.get("home_win", 0)
        away_prob = self.probabilities.get("away_win", 0)
        
        if home_prob < away_prob and away_prob >= 0.30:
            # Home is underdog
            if odds.get("home_odds", 0) >= 3.00:
                implied = 1 / odds["home_odds"]
                edge = home_prob - implied
                if 0.05 < edge < 0.30:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Home +0.75 or +1.0 AH",
                        "prob": home_prob,
                        "edge": edge,
                        "odds": odds["home_odds"],
                        "stake": "1 unit",
                        "confidence": "High"
                    })
        elif away_prob < home_prob and home_prob >= 0.30:
            # Away is underdog
            if odds.get("away_odds", 0) >= 3.00:
                implied = 1 / odds["away_odds"]
                edge = away_prob - implied
                if 0.05 < edge < 0.30:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Away +0.75 or +1.0 AH",
                        "prob": away_prob,
                        "edge": edge,
                        "odds": odds["away_odds"],
                        "stake": "1 unit",
                        "confidence": "High"
                    })
        
        # --- BTTS (Core Bet) ---
        btts_yes_prob = self.probabilities.get("btts_yes", 0.5)
        btts_no_prob = self.probabilities.get("btts_no", 0.5)
        
        if btts_rate > 0.65:
            if "btts_yes" in self.edges and self.edges["btts_yes"] > 0.05:
                self.bets.append({
                    "market": "BTTS",
                    "selection": "BTTS Yes",
                    "prob": btts_yes_prob,
                    "edge": self.edges["btts_yes"],
                    "odds": odds.get("btts_yes_odds", 0),
                    "stake": "1 unit",
                    "confidence": "High"
                })
        elif btts_rate < 0.45:
            if "btts_no" in self.edges and self.edges["btts_no"] > 0.05:
                self.bets.append({
                    "market": "BTTS",
                    "selection": "BTTS No",
                    "prob": btts_no_prob,
                    "edge": self.edges["btts_no"],
                    "odds": odds.get("btts_no_odds", 0),
                    "stake": "1 unit",
                    "confidence": "High"
                })
        
        # --- OVER/UNDER 2.5 (Selective) ---
        if self.shrunk_total > 3.00:
            if "over_25" in self.edges and self.edges["over_25"] > 0.08:
                self.bets.append({
                    "market": "Over/Under",
                    "selection": "Over 2.5 (small stake)",
                    "prob": self.probabilities["over_25"],
                    "edge": self.edges["over_25"],
                    "odds": odds.get("over_25_odds", 0),
                    "stake": "0.5 units",
                    "confidence": "Selective"
                })
        elif self.shrunk_total < 2.20:
            if "under_25" in self.edges and self.edges["under_25"] > 0.08:
                self.bets.append({
                    "market": "Over/Under",
                    "selection": "Under 2.5 (small stake)",
                    "prob": self.probabilities["under_25"],
                    "edge": self.edges["under_25"],
                    "odds": odds.get("under_25_odds", 0),
                    "stake": "0.5 units",
                    "confidence": "Selective"
                })
        else:
            self.skips.append({
                "market": "Over/Under 2.5",
                "reason": f"Shrunk total {self.shrunk_total:.2f} in neutral zone (2.20-3.00)",
                "edge": self.edges.get("over_25", 0),
                "prob": self.probabilities.get("over_25", 0)
            })
        
        # --- CORNERS (Selective) ---
        if corner_data:
            home_avg_corners = corner_data.get("home_avg_corners", 0)
            away_avg_corners = corner_data.get("away_avg_corners", 0)
            home_conceded_corners = corner_data.get("home_conceded_corners", 0)
            away_conceded_corners = corner_data.get("away_conceded_corners", 0)
            
            # Home team Over 4.5 corners
            if home_avg_corners >= 5.5 and away_conceded_corners >= 5.0:
                self.bets.append({
                    "market": "Corners",
                    "selection": "Home Over 4.5 corners",
                    "prob": 0.55,  # Estimated
                    "edge": 0.05,  # Estimated
                    "odds": corner_data.get("home_corners_odds", 0),
                    "stake": "0.5 units",
                    "confidence": "Selective"
                })
            
            # Away team Over 4.5 corners
            if away_avg_corners >= 5.5 and home_conceded_corners >= 5.0:
                self.bets.append({
                    "market": "Corners",
                    "selection": "Away Over 4.5 corners",
                    "prob": 0.55,
                    "edge": 0.05,
                    "odds": corner_data.get("away_corners_odds", 0),
                    "stake": "0.5 units",
                    "confidence": "Selective"
                })
        
        # --- NEVER BET markets (explicit rejection) ---
        self.skips.append({
            "market": "Correct Score",
            "reason": "Never bet - pure lottery (~10-15% hit rate)",
            "edge": 0,
            "prob": 0
        })
        self.skips.append({
            "market": "First Goalscorer",
            "reason": "Never bet - worse than anytime",
            "edge": 0,
            "prob": 0
        })
        self.skips.append({
            "market": "Anytime Goalscorer",
            "reason": "Avoid - high variance, team-dependent (~35-40% hit rate)",
            "edge": 0,
            "prob": 0
        })
        
        return self.bets
    
    def get_full_analysis(self) -> dict:
        """Return complete analysis for display"""
        return {
            "model_xg_home": self.model_xg_home,
            "model_xg_away": self.model_xg_away,
            "model_total": self.model_total,
            "market_total": self.market_total,
            "shrunk_xg_home": self.shrunk_xg_home,
            "shrunk_xg_away": self.shrunk_xg_away,
            "shrunk_total": self.shrunk_total,
            "probabilities": self.probabilities,
            "market_probs": self.market_probs,
            "edges": self.edges,
            "bets": self.bets,
            "skips": self.skips,
        }


# ============================================================================
# DATA EXTRACTION FROM BETEXPLORER
# ============================================================================

def parse_betexplorer_data(text: str) -> list:
    """
    Parse Betexplorer data to extract match information.
    
    This parser extracts:
    - Team names
    - Odds (1X2, and derives AH/BTTS/O-U where possible)
    - Any available stats (form, goals, etc.)
    
    Since Betexplorer's streak pages don't contain xG data, we use the
    odds as the primary market information and derive xG from them.
    """
    matches = []
    lines = text.split('\n')
    
    match_cache = {}
    current_country = None
    
    # Patterns to detect match lines
    # Format: "Home Team - Away Team  1.50  3.50  6.00"
    match_pattern = re.compile(
        r'^(.+?)\s*[-–]\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect country/league headers
        if re.match(r'^[A-Za-z\s]+$', line) and not re.search(r'[0-9.]', line) and len(line) < 40:
            if line not in ['Team', 'W', 'D', 'L', 'NW', 'ND', 'NL', 'Best teams', 
                           'Worst teams', 'Best offensive', 'Best defensive', 
                           'Worst offensive', 'Worst defensive', 'Next match',
                           'Home', 'Away', 'Draw']:
                current_country = line
                continue
        
        # Try to parse match line
        match = match_pattern.match(line)
        if match:
            home_team = match.group(1).strip()
            away_team = match.group(2).strip()
            home_odds = float(match.group(3))
            draw_odds = float(match.group(4))
            away_odds = float(match.group(5))
            
            # Validate odds
            if home_odds > 1.01 and draw_odds > 1.01 and away_odds > 1.01:
                match_key = f"{home_team}|{away_team}"
                
                if match_key not in match_cache:
                    match_cache[match_key] = {
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "league": current_country or "Unknown",
                    }
                    matches.append(match_cache[match_key])
    
    return matches


# ============================================================================
# ODDS-DERIVED xG ESTIMATION
# ============================================================================

def estimate_xg_from_odds(home_odds: float, draw_odds: float, away_odds: float) -> tuple:
    """
    Derive an estimated xG for each team from the market odds.
    
    This is a simplified approach: we use the implied probabilities to
    estimate the most likely goal distribution, then solve for xG values
    that would produce those probabilities via Poisson.
    
    In practice, you'd use a more sophisticated model. This provides a
    reasonable starting point when raw xG data isn't available.
    """
    # Remove overround (normalize to 100%)
    total_implied = (1/home_odds) + (1/draw_odds) + (1/away_odds)
    p_home = (1/home_odds) / total_implied
    p_draw = (1/draw_odds) / total_implied
    p_away = (1/away_odds) / total_implied
    
    # Estimate total goals from the odds
    # Higher draw odds generally correlate with more goals (more variance)
    # Lower draw odds correlate with fewer goals (more likely draw)
    # This is a heuristic: draw odds ~3.0 → ~2.6 goals, ~4.0 → ~3.2 goals
    if draw_odds <= 2.8:
        estimated_total = 2.2
    elif draw_odds <= 3.2:
        estimated_total = 2.5
    elif draw_odds <= 3.6:
        estimated_total = 2.8
    elif draw_odds <= 4.0:
        estimated_total = 3.1
    else:
        estimated_total = 3.4
    
    # Estimate xG split based on win probabilities
    # Stronger home team → more of the total goes to home
    home_share = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0.5
    home_share = max(0.35, min(0.65, home_share))  # Clamp to reasonable range
    
    home_xg = estimated_total * home_share
    away_xg = estimated_total * (1 - home_share)
    
    return home_xg, away_xg


def estimate_btts_rate(home_xg: float, away_xg: float) -> float:
    """Estimate BTTS rate from xG values"""
    # P(BTTS) = P(home scores) * P(away scores)
    # Using Poisson: P(team scores at least 1) = 1 - e^(-xg)
    p_home_scores = 1 - np.exp(-home_xg)
    p_away_scores = 1 - np.exp(-away_xg)
    return p_home_scores * p_away_scores


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_analysis(analysis: dict, match: dict):
    """Display the full analysis with xG, probabilities, edges, and bets"""
    
    home_team = match.get("home_team", "Home")
    away_team = match.get("away_team", "Away")
    
    # Header
    st.markdown(f"""
    <div class="output-card">
        <div style="font-size:1.2rem; font-weight:700; margin-bottom:0.5rem;">
            {home_team} vs {away_team}
        </div>
        <div style="font-size:0.8rem; color:#94a3b8;">
            {match.get('league', 'Unknown League')} | {match.get('date', '')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # xG Display
    st.markdown("### 📐 Expected Goals (xG) Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Model xG (Home)</div>
            <div class="xG-display">{analysis['model_xg_home']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Model xG (Away)</div>
            <div class="xG-display">{analysis['model_xg_away']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Model Total</div>
            <div class="xG-display">{analysis['model_total']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Market Total</div>
            <div class="xG-display">{analysis['market_total']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Shrunk xG (Home)</div>
            <div class="xG-display">{analysis['shrunk_xg_home']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">Shrunk xG (Away)</div>
            <div class="xG-display">{analysis['shrunk_xg_away']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Shrinkage info
    st.markdown(f"""
    <div style="background:#0f172a; border-radius:6px; padding:0.5rem; margin:0.5rem 0; font-size:0.8rem; color:#94a3b8;">
        <strong>Shrinkage:</strong> Shrunk Total = 0.5 × Model Total ({analysis['model_total']:.2f}) + 0.5 × Market Total ({analysis['market_total']:.2f}) = <strong style="color:#3b82f6;">{analysis['shrunk_total']:.2f}</strong>
        &nbsp;|&nbsp; Scale = {analysis['shrunk_total'] / analysis['model_total'] if analysis['model_total'] > 0 else 1:.3f}
    </div>
    """, unsafe_allow_html=True)
    
    # Probabilities
    st.markdown("### 📊 Poisson Probabilities")
    probs = analysis["probabilities"]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(Home Win)</div>
            <div class="feature-value">{probs.get('home_win', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(Draw)</div>
            <div class="feature-value">{probs.get('draw', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(Away Win)</div>
            <div class="feature-value">{probs.get('away_win', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(BTTS Yes)</div>
            <div class="feature-value">{probs.get('btts_yes', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(Over 2.5)</div>
            <div class="feature-value">{probs.get('over_25', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-label">P(Under 2.5)</div>
            <div class="feature-value">{probs.get('under_25', 0):.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Edges
    st.markdown("### 📈 Market Edges")
    edges = analysis["edges"]
    market_probs = analysis["market_probs"]
    
    if edges:
        edge_data = []
        for market, edge in edges.items():
            if edge != 0:
                market_display = {
                    "home_win": "Home Win",
                    "draw": "Draw",
                    "away_win": "Away Win",
                    "btts_yes": "BTTS Yes",
                    "btts_no": "BTTS No",
                    "over_25": "Over 2.5",
                    "under_25": "Under 2.5",
                }.get(market, market)
                
                my_prob = analysis["probabilities"].get(market, 0)
                implied = market_probs.get(market, 0)
                
                edge_class = "edge-positive" if edge > 0 else "edge-negative"
                edge_data.append({
                    "Market": market_display,
                    "My Prob": f"{my_prob:.1%}",
                    "Implied": f"{implied:.1%}",
                    "Edge": f"{edge:+.1%}",
                    "Verdict": "✅ VALUE" if 0.05 < edge < 0.30 else "⚠️ TOO HIGH" if edge > 0.30 else "❌ NO VALUE"
                })
        
        if edge_data:
            df = pd.DataFrame(edge_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Bets
    st.markdown("### 🎯 Selected Bets")
    bets = analysis["bets"]
    
    if bets:
        for bet in bets:
            confidence = bet.get("confidence", "High")
            if confidence == "High":
                card_class = "bet-card"
                badge = '<span class="badge badge-bet">CORE BET</span>'
            else:
                card_class = "selective-card"
                badge = '<span class="badge badge-selective">SELECTIVE</span>'
            
            st.markdown(f"""
            <div class="output-card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <div style="font-size:0.8rem; color:#94a3b8;">{bet.get('market', '')}</div>
                        <div style="font-size:1.5rem; font-weight:700; color:#10b981;">{bet.get('selection', '')}</div>
                        <div style="margin-top:0.3rem;">
                            {badge}
                            <span style="margin-left:0.5rem; padding:0.3rem 0.75rem; border-radius:8px; font-size:0.8rem; background:#1e293b; color:#fbbf24;">
                                Stake: {bet.get('stake', '1 unit')}
                            </span>
                            <span style="margin-left:0.5rem; padding:0.3rem 0.75rem; border-radius:8px; font-size:0.8rem; background:#1e293b; color:#3b82f6;">
                                Edge: {bet.get('edge', 0):+.1%}
                            </span>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:#94a3b8;">Model Prob</div>
                        <div style="font-size:1.8rem; font-weight:800; color:#10b981;">{bet.get('prob', 0):.1%}</div>
                        <div style="font-size:0.8rem; color:#94a3b8;">Odds: {bet.get('odds', 0):.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No value bets identified for this match. See skipped markets below.")
    
    # Skips
    with st.expander("❌ Skipped Markets", expanded=False):
        skips = analysis["skips"]
        if skips:
            for skip in skips:
                st.markdown(f"""
                <div style="background:#0f172a; border-radius:6px; padding:0.5rem; margin:0.25rem 0; border-left:3px solid #64748b;">
                    <strong>{skip.get('market', '')}</strong>: {skip.get('reason', '')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No markets skipped.")
    
    # Never bet reminder
    st.markdown("""
    <div style="background:#1a0a0a; border-radius:6px; padding:0.75rem; margin-top:1rem; border-left:3px solid #ef4444;">
        <strong style="color:#ef4444;">🚫 NEVER BET:</strong>
        <span style="color:#94a3b8;">Correct Score, First Goalscorer, Anytime Goalscorer (unless 2.00 or shorter and in strong form)</span>
    </div>
    """, unsafe_allow_html=True)


def display_bets_summary(all_results: list):
    """Display a summary table of all bets across matches"""
    if not all_results:
        return
    
    rows = []
    for match, analysis, saved_id in all_results:
        for bet in analysis.get("bets", []):
            rows.append({
                "Match": f"{match.get('home_team', '')} vs {match.get('away_team', '')}",
                "Market": bet.get("market", ""),
                "Selection": bet.get("selection", ""),
                "Edge": f"{bet.get('edge', 0):+.1%}",
                "Odds": f"{bet.get('odds', 0):.2f}",
                "Stake": bet.get("stake", "1 unit"),
                "Confidence": bet.get("confidence", "High"),
            })
    
    if rows:
        st.markdown("### 📋 Bet Summary")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Count bets
        total_bets = len(rows)
        core_bets = sum(1 for r in rows if r["Confidence"] == "High")
        selective_bets = sum(1 for r in rows if r["Confidence"] == "Selective")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bets", total_bets)
        with col2:
            st.metric("Core Bets", core_bets)
        with col3:
            st.metric("Selective Bets", selective_bets)


# ============================================================================
# SUPABASE OPERATIONS
# ============================================================================

def save_bet_to_db(match: dict, analysis: dict, bet: dict) -> Optional[str]:
    """Save a single bet to the database"""
    try:
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", datetime.now().strftime("%Y-%m-%d"))
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else datetime.now().strftime("%Y-%m-%d")
        
        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "league": match.get("league", "Unknown"),
            
            # Odds
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            
            # xG data
            "model_xg_home": analysis.get("model_xg_home", 0),
            "model_xg_away": analysis.get("model_xg_away", 0),
            "model_total": analysis.get("model_total", 0),
            "market_total": analysis.get("market_total", 0),
            "shrunk_xg_home": analysis.get("shrunk_xg_home", 0),
            "shrunk_xg_away": analysis.get("shrunk_xg_away", 0),
            "shrunk_total": analysis.get("shrunk_total", 0),
            
            # Probabilities
            "prob_home_win": analysis.get("probabilities", {}).get("home_win", 0),
            "prob_draw": analysis.get("probabilities", {}).get("draw", 0),
            "prob_away_win": analysis.get("probabilities", {}).get("away_win", 0),
            "prob_btts_yes": analysis.get("probabilities", {}).get("btts_yes", 0),
            "prob_over_25": analysis.get("probabilities", {}).get("over_25", 0),
            "prob_under_25": analysis.get("probabilities", {}).get("under_25", 0),
            
            # Bet details
            "market": bet.get("market", ""),
            "selection": bet.get("selection", ""),
            "bet_prob": bet.get("prob", 0),
            "bet_edge": bet.get("edge", 0),
            "bet_odds": bet.get("odds", 0),
            "stake": bet.get("stake", "1 unit"),
            "confidence": bet.get("confidence", "High"),
            
            # Prediction fields (for compatibility)
            "predicted": "NO_DRAW" if bet.get("market") == "Match Result" else "BET",
            "multi_score": analysis.get("shrunk_total", 0),
        }
        
        response = supabase.table(TABLE_NAME).insert(record).execute()
        return response.data[0]["id"] if response.data else None
        
    except Exception as e:
        st.error(f"Failed to save bet: {e}")
        return None


def get_pending():
    try:
        response = supabase.table(TABLE_NAME).select("*").is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except:
        return []


def submit_result(analysis_id, home_goals, away_goals):
    try:
        actual_result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"
        
        # Get the bet details
        response = supabase.table(TABLE_NAME).select("*").eq("id", analysis_id).execute()
        if not response.data:
            return False
        
        record = response.data[0]
        market = record.get("market", "")
        selection = record.get("selection", "")
        bet_odds = record.get("bet_odds", 1.0)
        
        # Determine if bet won
        is_correct = False
        
        if market == "Match Result":
            if "Home" in selection:
                if "AH" in selection:
                    # Asian Handicap
                    if "-0.5" in selection or "-0.25" in selection:
                        is_correct = home_goals > away_goals
                    elif "+0.5" in selection or "+0.25" in selection:
                        is_correct = home_goals >= away_goals
                    elif "+0.75" in selection or "+1.0" in selection:
                        # Simplified: win if draw or better
                        is_correct = home_goals >= away_goals
                else:
                    is_correct = home_goals > away_goals
            elif "Away" in selection:
                if "AH" in selection:
                    if "-0.5" in selection or "-0.25" in selection:
                        is_correct = away_goals > home_goals
                    elif "+0.5" in selection or "+0.25" in selection:
                        is_correct = away_goals >= home_goals
                    elif "+0.75" in selection or "+1.0" in selection:
                        is_correct = away_goals >= home_goals
                else:
                    is_correct = away_goals > home_goals
            elif "Draw" in selection:
                is_correct = home_goals == away_goals
        
        elif market == "BTTS":
            if "Yes" in selection:
                is_correct = home_goals >= 1 and away_goals >= 1
            else:
                is_correct = home_goals == 0 or away_goals == 0
        
        elif market == "Over/Under":
            total = home_goals + away_goals
            if "Over" in selection:
                is_correct = total > 2.5
            else:
                is_correct = total < 2.5
        
        elif market == "Corners":
            # Would need corner data; mark as pending
            is_correct = None
        
        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_result": actual_result,
            "is_correct": is_correct
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Failed to submit result: {e}")
        return False


def get_results():
    try:
        response = supabase.table(TABLE_NAME).select("*").not_.is_("actual_result", "null").execute()
        data = response.data if response.data else []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except:
        return []


def display_records_table(results: list):
    if not results:
        st.info("No results recorded yet.")
        return
    
    total = len(results)
    correct = sum(1 for r in results if r.get('is_correct'))
    incorrect = total - correct
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
    with col2:
        win_rate = round(correct / total * 100) if total > 0 else 0
        st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
    
    rows = []
    for r in results:
        is_correct = r.get('is_correct', False)
        result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
        
        rows.append({
            "Date": r.get("match_date", ""),
            "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
            "Market": r.get("market", ""),
            "Selection": r.get("selection", ""),
            "Edge": f"{r.get('bet_edge', 0):+.1%}",
            "Odds": f"{r.get('bet_odds', 0):.2f}",
            "Score": f"{r.get('actual_home_goals', '')}-{r.get('actual_away_goals', '')}",
            "Result": result_badge,
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.title("⚽ Refined Prediction Strategy")
    st.caption("xG-based Poisson model with market shrinkage, value checks, and edge caps")
    
    with st.expander("📖 STRATEGY OVERVIEW", expanded=False):
        st.markdown("""
        ### The Refined Prediction Strategy
        
        **Core Bets:** Match Result (Asian Handicap) | BTTS
        
        **Selective Bets:** Over/Under 2.5 | Corners
        
        **Never Bet:** Correct Score | First Goalscorer | Anytime Goalscorer
        
        ---
        
        ### The Seven-Step Process
        
        1. **Gather Data** — League position, form, home/away splits, H2H, injuries, lineups, odds
        2. **Calculate Expected Goals** — xG from home/away splits (blended 70% last-10, 30% season)
        3. **Apply Adjustments** — Form (±0.10–0.15), Injuries (±0.10–0.15 per key player), Fatigue (−0.05–0.10)
        4. **Shrink Toward Market** — Shrunk_Total = 0.5 × Model_Total + 0.5 × Market_Total
        5. **Run Poisson** — Using shrunk xG as Poisson means
        6. **Compare to Market** — Edge = My_Prob − Implied_Prob
        7. **Select Market** — Value bands: 5% < Edge < 30% → bet; Edge > 30% → skip (model error)
        
        ---
        
        ### The Three Principles
        
        1. **Shrinkage Principle** — Shrink model xG 50% toward market total to temper extremes
        2. **Value Check Principle** — Every market requires a trigger AND a value check
        3. **Edge Cap Principle** — Cap the maximum edge at 30% (larger edges are model errors)
        
        ---
        
        ### Discipline Rules
        
        1. **Never bet correct score. Ever.**
        2. **Never bet first goalscorer. Ever.**
        3. **Only bet anytime goalscorer if 2.00 or shorter and in strong form.** (Rarely)
        4. **Only bet Over/Under if shrunk total > 3.00 or < 2.20 AND edge > 8%.**
        5. **Only bet corners if one team averages 5.5+ in their split.**
        6. **Always bet match result when edge is between 5% and 30%.**
        7. **Always bet BTTS when rate triggers AND edge > 5%.**
        8. **Always shrink model xG 50% toward market total before running Poisson.**
        9. **Stake flat (1 unit) or Kelly (capped at 5%).**
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["⚽ Predict", "📝 Pending", "📊 Records", "📈 Dashboard"])
    
    with tab1:
        st.markdown("### 📝 Paste Match Data")
        st.info("Paste Betexplorer data or enter match details manually. The system will derive xG from odds when raw xG data isn't available.")
        
        input_method = st.radio("Input Method", ["Paste Betexplorer Data", "Manual Entry"], horizontal=True)
        
        if input_method == "Paste Betexplorer Data":
            text_data = st.text_area(
                "Paste Betexplorer data here",
                height=250,
                key="text_paste",
                placeholder="Paste all Betexplorer page data here...\n\nExample format:\nEngland\nArsenal - Chelsea  2.10  3.40  3.20\nLiverpool - Man City  2.50  3.30  2.80"
            )
        else:
            text_data = None
            st.markdown("#### Manual Match Entry")
            col1, col2 = st.columns(2)
            with col1:
                home_team = st.text_input("Home Team", key="manual_home")
                home_odds = st.number_input("Home Odds", min_value=1.01, value=2.00, step=0.01, key="manual_home_odds")
            with col2:
                away_team = st.text_input("Away Team", key="manual_away")
                away_odds = st.number_input("Away Odds", min_value=1.01, value=3.00, step=0.01, key="manual_away_odds")
            
            draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.40, step=0.01, key="manual_draw_odds")
            
            st.markdown("#### Optional: Additional Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                home_xg = st.number_input("Home xG (optional)", min_value=0.0, value=0.0, step=0.1, key="manual_home_xg")
            with col2:
                away_xg = st.number_input("Away xG (optional)", min_value=0.0, value=0.0, step=0.1, key="manual_away_xg")
            with col3:
                market_total = st.number_input("Market Total Goals (optional)", min_value=0.0, value=0.0, step=0.1, key="manual_market_total")
            
            if st.button("Analyze Manual Match", type="primary"):
                if not home_team or not away_team:
                    st.error("Please enter both team names.")
                else:
                    matches = [{
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "league": "Manual Entry",
                        "home_xg": home_xg if home_xg > 0 else None,
                        "away_xg": away_xg if away_xg > 0 else None,
                        "market_total": market_total if market_total > 0 else None,
                    }]
                    
                    process_matches(matches)
        
        if input_method == "Paste Betexplorer Data" and st.button("⚽ ANALYZE", type="primary"):
            if not text_data or len(text_data.strip()) < 10:
                st.error("❌ Please paste valid data.")
            else:
                try:
                    with st.spinner("Parsing data..."):
                        matches = parse_betexplorer_data(text_data)
                    
                    if matches:
                        process_matches(matches)
                    else:
                        st.error("No matches found in the data. Please check the format.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())
    
    with tab2:
        st.subheader("📝 Pending Bets")
        pending = get_pending()
        if pending:
            st.write(f"**{len(pending)} pending bet(s)**")
            for a in pending:
                ht = a.get('home_team', 'Home')
                at = a.get('away_team', 'Away')
                market = a.get('market', '')
                selection = a.get('selection', '')
                match_date = a.get('match_date', 'Date unknown')
                date_display = format_date_display(match_date)
                
                with st.expander(f"📅 {date_display} | {market}: {selection} | {ht} vs {at}"):
                    st.info(f"📊 Bet: {market} — {selection} @ {a.get('bet_odds', 0):.2f} | Edge: {a.get('bet_edge', 0):+.1%}")
                    c1, c2 = st.columns(2)
                    with c1:
                        hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
                    with c2:
                        ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                        if submit_result(a['id'], hg, ag):
                            st.success("Result submitted!")
                            st.rerun()
        else:
            st.info("No pending bets.")
    
    with tab3:
        st.subheader("📊 Performance Records")
        results = get_results()
        display_records_table(results)
    
    with tab4:
        st.subheader("📊 Live Dashboard")
        results = get_results()
        if not results:
            st.info("No results recorded yet.")
        else:
            total = len(results)
            correct = sum(1 for r in results if r.get('is_correct'))
            incorrect = total - correct
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total Bets</div></div>', unsafe_allow_html=True)
            with col2:
                win_rate = round(correct / total * 100) if total > 0 else 0
                st.markdown(f'<div class="stat-box"><div class="stat-number">{win_rate}%</div><div class="stat-label">Win Rate</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Wins</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="stat-box"><div class="stat-number">{incorrect}</div><div class="stat-label">Losses</div></div>', unsafe_allow_html=True)
            
            # Performance by market
            st.markdown("### Performance by Market")
            market_stats = {}
            for r in results:
                market = r.get("market", "Unknown")
                if market not in market_stats:
                    market_stats[market] = {"total": 0, "correct": 0}
                market_stats[market]["total"] += 1
                if r.get("is_correct"):
                    market_stats[market]["correct"] += 1
            
            if market_stats:
                market_rows = []
                for market, stats in market_stats.items():
                    rate = round(stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    market_rows.append({
                        "Market": market,
                        "Total": stats["total"],
                        "Wins": stats["correct"],
                        "Win Rate": f"{rate}%"
                    })
                df = pd.DataFrame(market_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            # All records
            st.markdown("### All Records")
            rows = []
            for r in results:
                is_correct = r.get('is_correct', False)
                result_badge = '🟢 WIN' if is_correct else '🔴 LOSS'
                rows.append({
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team', '')} vs {r.get('away_team', '')}",
                    "Market": r.get("market", ""),
                    "Selection": r.get("selection", ""),
                    "Edge": f"{r.get('bet_edge', 0):+.1%}",
                    "Odds": f"{r.get('bet_odds', 0):.2f}",
                    "Score": f"{r.get('actual_home_goals', '')}-{r.get('actual_away_goals', '')}",
                    "Result": result_badge,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)


def process_matches(matches: list):
    """Process a list of matches through the refined predictor"""
    st.success(f"✅ Found {len(matches)} match(es) to analyze")
    
    all_results = []
    total_bets = 0
    total_core = 0
    total_selective = 0
    total_skipped = 0
    
    for match in matches:
        # Initialize predictor
        predictor = RefinedPredictor()
        
        # Get odds
        home_odds = match.get("home_odds", 2.0)
        draw_odds = match.get("draw_odds", 3.4)
        away_odds = match.get("away_odds", 3.0)
        
        # Get or estimate xG
        home_xg = match.get("home_xg")
        away_xg = match.get("away_xg")
        market_total = match.get("market_total")
        
        # If raw xG not provided, estimate from odds
        if home_xg is None or away_xg is None or home_xg <= 0 or away_xg <= 0:
            home_xg, away_xg = estimate_xg_from_odds(home_odds, draw_odds, away_odds)
            st.info(f"ℹ️ xG estimated from odds for {match.get('home_team', '')} vs {match.get('away_team', '')}")
        
        # If market total not provided, estimate from draw odds
        if market_total is None or market_total <= 0:
            market_total = home_xg + away_xg  # Use model total as market estimate
        
        # Build data dicts for the predictor
        home_data = {
            "home_goals_scored_season": home_xg,
            "home_goals_scored_last10": home_xg,
            "home_goals_conceded_season": away_xg * 0.9,
            "home_goals_conceded_last10": away_xg * 0.9,
            "last5_points": 7,  # Default: neutral form
            "injuries": [],
            "played_midweek": False,
        }
        
        away_data = {
            "away_goals_scored_season": away_xg,
            "away_goals_scored_last10": away_xg,
            "away_goals_conceded_season": home_xg * 0.9,
            "away_goals_conceded_last10": home_xg * 0.9,
            "last5_points": 7,
            "injuries": [],
            "played_midweek": False,
        }
        
        # Step 2: Calculate base xG
        predictor.calculate_base_xg(home_data, away_data)
        
        # Step 3: Apply adjustments
        predictor.apply_adjustments(home_data, away_data)
        
        # Step 4: Shrink toward market
        predictor.shrink_toward_market(market_total)
        
        # Step 5: Run Poisson
        predictor.run_poisson()
        
        # Step 6: Calculate edges
        odds = {
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "btts_yes_odds": 1.80,  # Default estimate
            "btts_no_odds": 2.00,
            "over_25_odds": 1.90,
            "under_25_odds": 1.90,
        }
        predictor.calculate_edges(odds)
        
        # Step 7: Select markets
        btts_rate = estimate_btts_rate(predictor.shrunk_xg_home, predictor.shrunk_xg_away)
        predictor.select_markets(odds, btts_rate)
        
        # Get full analysis
        analysis = predictor.get_full_analysis()
        
        # Display
        with st.expander(f"⚽ {match.get('home_team', 'Home')} vs {match.get('away_team', 'Away')}", expanded=True):
            display_analysis(analysis, match)
        
        # Track stats
        bets = analysis.get("bets", [])
        total_bets += len(bets)
        total_core += sum(1 for b in bets if b.get("confidence") == "High")
        total_selective += sum(1 for b in bets if b.get("confidence") == "Selective")
        
        if not bets:
            total_skipped += 1
        
        # Save bets to DB
        for bet in bets:
            saved_id = save_bet_to_db(match, analysis, bet)
            if saved_id:
                all_results.append((match, analysis, saved_id))
    
    # Summary
    st.markdown("---")
    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Matches", len(matches))
    with col2:
        st.metric("Total Bets", total_bets)
    with col3:
        st.metric("Core Bets", total_core)
    with col4:
        st.metric("Selective Bets", total_selective)
    with col5:
        st.metric("No-Bet Matches", total_skipped)
    
    # Bet summary table
    if all_results:
        display_bets_summary(all_results)
    
    # Store results in session for display
    st.session_state["last_results"] = all_results


if __name__ == "__main__":
    main()
