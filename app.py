"""
Refined Prediction Strategy — single-file Streamlit app.
Parser + Predictor + Prediction-style UI.

Includes sample-size shrinkage: when a team has played few games in the
current season, the model trusts its own xG estimate less and shrinks
more aggressively toward the market total.

AH is still parsed from the page exactly as before.
On display, AH lines are converted to 3-way European Handicap form
(Home / Draw / Away) for easier interpretation.
"""

import math
import re
import traceback
from datetime import date, datetime
from typing import Optional

import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="Refined Predictor", page_icon="⚽", layout="wide")


# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1100px; }

    .team-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        color: #fff;
        margin-bottom: 1rem;
    }
    .team-names {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .team-meta {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }

    .verdict-bet {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
    }
    .verdict-nobet {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b;
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
    }
    .verdict-label {
        font-size: 0.75rem;
        letter-spacing: 2px;
        font-weight: 700;
        color: #6ee7b7;
        text-transform: uppercase;
    }
    .verdict-label-grey {
        font-size: 0.75rem;
        letter-spacing: 2px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .verdict-pick {
        font-size: 2.4rem;
        font-weight: 800;
        color: #fff;
        margin: 0.35rem 0;
        line-height: 1.1;
    }
    .verdict-noedge {
        font-size: 1.6rem;
        font-weight: 700;
        color: #cbd5e1;
        margin: 0.35rem 0;
    }
    .verdict-detail {
        font-size: 1rem;
        color: #d1fae5;
        margin-top: 0.5rem;
    }
    .verdict-detail-grey {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }

    .stat-card {
        background: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border-top: 3px solid #334155;
    }
    .stat-card-home { border-top-color: #10b981; }
    .stat-card-draw { border-top-color: #fbbf24; }
    .stat-card-away { border-top-color: #3b82f6; }
    .stat-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #fff;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-edge {
        font-size: 0.85rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .edge-pos { color: #10b981; }
    .edge-neg { color: #ef4444; }
    .edge-neutral { color: #94a3b8; }

    .alt-bet {
        background: #0f172a;
        border-left: 4px solid #fbbf24;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.5rem;
    }
    .alt-bet-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #fbbf24;
    }
    .alt-bet-meta {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.75rem 0;
    }

    .xg-row {
        background: #0f172a;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .xg-team { color: #cbd5e1; font-weight: 600; }
    .xg-value { color: #3b82f6; font-weight: 800; font-size: 1.3rem; }

    .eh-table {
        background: #0f172a;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .eh-line { color: #cbd5e1; font-weight: 600; font-size: 0.9rem; }
    .eh-prob { color: #3b82f6; font-weight: 800; font-size: 1.15rem; }
    .eh-edge { font-weight: 700; font-size: 0.85rem; margin-left: 0.5rem; }

    .trust-row {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .trust-value { font-weight: 700; color: #3b82f6; }
    .trust-warn { color: #fbbf24; }
    .trust-ok { color: #10b981; }

    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.25rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SUPABASE
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None


TABLE_NAME = "match_predictions"


# ============================================================================
# CONSTANTS
# ============================================================================
EDGE_MIN = 0.05
EDGE_MAX = 0.20
EDGE_OU_MIN = 0.08
SHRINK_WEIGHT = 0.50
LAST10_WEIGHT = 0.70
SEASON_WEIGHT = 0.30
FORM_ADJ_HIGH = 0.15
FORM_ADJ_MED = 0.10
INJURY_ADJ = 0.15
FATIGUE_HOME = 0.05
FATIGUE_AWAY = 0.10
MIN_XG = 0.10

# Sample-size shrinkage settings
TRUST_FULL_SAMPLE = 8
TRUST_MIN_WEIGHT = 0.25
MAX_EFFECTIVE_SHRINK = 0.90

# AH rules (unchanged — still drives selection logic)
AH_HOME_OUTRIGHT_MIN = 0.45
AH_AWAY_OUTRIGHT_MIN = 0.45
AH_DRAW_MIN = 0.28
AH_UNDERDOG_MIN = 0.30


# ============================================================================
# PARSER  (AH parsing is UNCHANGED from the original)
# ============================================================================
def _has_bs4():
    try:
        import bs4  # noqa
        return True
    except ImportError:
        return False


class SportsgamblerParser:
    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError("beautifulsoup4 is not installed.")
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, "html.parser")
        self.home_team: Optional[str] = None
        self.away_team: Optional[str] = None

    def parse(self) -> dict:
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        competition, venue = self._parse_league_venue()

        result = {
            "match_id": self._make_match_id(match_date),
            "match_date": match_date,
            "kickoff": kickoff,
            "competition": competition,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "venue": venue,
            "odds": self._parse_odds(),
            "home_team_last10_home": self._parse_last10_splits("home"),
            "away_team_last10_away": self._parse_last10_splits("away"),
            "season_splits": self._parse_season_splits(),
            "last5_form": self._parse_last5_form(competition),
            "injuries": self._parse_injuries(),
            "midweek_fixture": self._parse_midweek(match_date),
            "home_current_season_games": self._parse_current_season_games("home", competition),
            "away_current_season_games": self._parse_current_season_games("away", competition),
        }
        corners = self._parse_corners()
        result["home_team_last10_home"].update(corners["home"])
        result["away_team_last10_away"].update(corners["away"])
        return result

    # -- identity -----------------------------------------------------------

    def _parse_teams(self):
        teams = self.soup.select(".t_top .t_teams .t_name strong")
        if len(teams) >= 2:
            return teams[0].get_text(strip=True), teams[1].get_text(strip=True)
        h1 = self.soup.find("h1", class_="p_title")
        if h1:
            m = re.match(r"(.+?)\s+vs\s+(.+?)\s+Prediction", h1.get_text(strip=True))
            if m:
                return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _parse_league_venue(self):
        league = None
        for link in self.soup.select(".t_top .t_info_link"):
            text = link.get_text(strip=True)
            if re.search(r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Championship|MLS|Cup|Division)", text, re.I):
                league = text
                break
        venue_el = self.soup.select_one(".t_top .t_venue")
        return league, (venue_el.get_text(strip=True) if venue_el else None)

    def _parse_datetime(self):
        date_el = self.soup.select_one(".t_top .t_date span:first-child")
        time_el = self.soup.select_one(".t_top .t_date .t_time")
        raw_date = date_el.get_text(strip=True) if date_el else None
        kickoff = time_el.get_text(strip=True) if time_el else None
        iso = None
        if raw_date:
            for fmt in ("%a %d %b", "%d %b", "%a %d %B", "%d %B"):
                try:
                    dt = datetime.strptime(raw_date, fmt).replace(year=datetime.now().year)
                    iso = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        return iso, kickoff

    def _make_match_id(self, match_date):
        ht = (self.home_team or "HOME").replace(" ", "")[:3].upper()
        at = (self.away_team or "AWAY").replace(" ", "")[:3].upper()
        dt = (match_date or "").replace("-", "")
        return f"{ht}_{at}_{dt}"

    # -- odds (AH parsing kept exactly as before) ---------------------------

    def _parse_odds(self):
        flat = {
            "home": None, "draw": None, "away": None,
            "ah_home_line": None, "ah_home": None,
            "ah_away_line": None, "ah_away": None,
            "btts_yes": None, "btts_no": None,
            "over_25": None, "under_25": None,
            "corners_line": None, "corners_over": None, "corners_under": None,
            "home_corners_line": None, "home_corners_over": None,
            "away_corners_line": None, "away_corners_over": None,
        }
        for row in self.soup.select(".nlf_odds_row"):
            title_el = row.select_one(".nfl_odd_title")
            if not title_el:
                continue
            market = title_el.get_text(strip=True)
            entries = []
            for ply in row.select(".nfl_odply"):
                label_el = ply.select_one(".nfl_ply_t")
                odds_el = ply.select_one(".nfl_ply_o")
                if label_el and odds_el:
                    entries.append((label_el.get_text(strip=True),
                                    self._to_float(odds_el.get_text(strip=True))))
            self._apply_market(flat, market, entries)
        return flat

    def _apply_market(self, flat, market, entries):
        m = market.lower()
        if "full-time result" in m:
            for label, val in entries:
                if label == "1": flat["home"] = val
                elif label == "X": flat["draw"] = val
                elif label == "2": flat["away"] = val
            return
        # ---- ASIAN HANDICAP (kept as-is) ----
        if "asian handicap" in m:
            for label, val in entries:
                mm = re.match(r"(\d)\s*Hcp\s*([+-]?[\d.]+)", label)
                if not mm: continue
                side = mm.group(1)
                line = self._to_float(mm.group(2))
                if side == "1":
                    flat["ah_home_line"] = line
                    flat["ah_home"] = val
                else:
                    flat["ah_away_line"] = line
                    flat["ah_away"] = val
            return
        if "both teams to score" in m:
            for label, val in entries:
                if label.lower() == "yes": flat["btts_yes"] = val
                elif label.lower() == "no": flat["btts_no"] = val
            return
        if "total goals" in m:
            for label, val in entries:
                if "over 2.5" in label.lower(): flat["over_25"] = val
                elif "under 2.5" in label.lower(): flat["under_25"] = val
            return
        if "total corners" in m:
            for label, val in entries:
                mm = re.match(r"(Over|Under)\s+([\d.]+)", label, re.I)
                if mm:
                    flat["corners_line"] = self._to_float(mm.group(2))
                    if mm.group(1).lower() == "over": flat["corners_over"] = val
                    else: flat["corners_under"] = val
            return
        if m.endswith(" corners"):
            team_name = market[: -len(" Corners")].strip()
            is_home = self._name_matches(self.home_team, team_name)
            is_away = self._name_matches(self.away_team, team_name)
            for label, val in entries:
                mm = re.match(r"Over\s+([\d.]+)", label, re.I)
                if not mm: continue
                line = self._to_float(mm.group(1))
                if is_home:
                    flat["home_corners_line"] = line
                    flat["home_corners_over"] = val
                elif is_away:
                    flat["away_corners_line"] = line
                    flat["away_corners_over"] = val
            return

    @staticmethod
    def _name_matches(a, b):
        if not a or not b: return False
        return bool(set(a.lower().split()) & set(b.lower().split()))

    @staticmethod
    def _to_float(s):
        try: return float(str(s).strip())
        except (ValueError, TypeError): return None

    @staticmethod
    def _to_int(s):
        try: return int(str(s).strip())
        except (ValueError, TypeError): return 0

    # -- last 10 splits -----------------------------------------------------

    def _parse_last10_splits(self, side):
        split = self._empty_split()
        table = self.soup.select_one(".st-table")
        if not table: return split
        rows = table.select("tbody tr")
        idx = 0 if side == "home" else 1
        if idx >= len(rows): return split
        cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
        if len(cells) < 9: return split
        mm = re.match(r"(\d+)-(\d+)-(\d+)", cells[1])
        if mm:
            split["wins"] = int(mm.group(1))
            split["draws"] = int(mm.group(2))
            split["losses"] = int(mm.group(3))
        split["gf_per_game"] = self._to_float(cells[3]) or 0.0
        split["ga_per_game"] = self._to_float(cells[4]) or 0.0
        split["over25"] = self._to_int(cells[5])
        split["under25"] = self._to_int(cells[6])
        split["btts_yes"] = self._to_int(cells[7])
        split["btts_no"] = self._to_int(cells[8])
        return split

    @staticmethod
    def _empty_split():
        return {"wins": 0, "draws": 0, "losses": 0, "gf_per_game": 0.0, "ga_per_game": 0.0,
                "btts_yes": 0, "btts_no": 0, "over25": 0, "under25": 0,
                "corners_for": 0.0, "corners_against": 0.0}

    def _parse_season_splits(self):
        home = self._parse_last10_splits("home")
        away = self._parse_last10_splits("away")
        return {
            "home_team_home_gf_pg": home["gf_per_game"],
            "home_team_home_ga_pg": home["ga_per_game"],
            "away_team_away_gf_pg": away["gf_per_game"],
            "away_team_away_ga_pg": away["ga_per_game"],
        }

    def _parse_corners(self):
        out = {"home": {"corners_for": 0.0, "corners_against": 0.0},
               "away": {"corners_for": 0.0, "corners_against": 0.0}}
        rows = self.soup.select(".corner--stats .corner-stats-row.corners-stats-body")
        for i, row in enumerate(rows[:2]):
            side = "home" if i == 0 else "away"
            values = []
            for col in row.select(".corner-stats-column"):
                spans = col.select("span")
                if spans:
                    values.extend([s.get_text(strip=True) for s in spans])
                else:
                    values.append(col.get_text(strip=True))
            numbers = [self._to_float(v) for v in values]
            numbers = [n for n in numbers if n is not None]
            if side == "home" and len(numbers) >= 6:
                out["home"]["corners_for"] = numbers[4]
                out["home"]["corners_against"] = numbers[5]
            elif side == "away" and len(numbers) >= 9:
                out["away"]["corners_for"] = numbers[7]
                out["away"]["corners_against"] = numbers[8]
        return out

    # -- last 5 form --------------------------------------------------------

    def _parse_last5_form(self, competition):
        out = {"home_team_points": 0, "away_team_points": 0}
        container = self.soup.select_one("#last-matches #All")
        if not container: return out
        left = container.select_one(".teamstats-left")
        right = container.select_one(".teamstats-right")
        out["home_team_points"] = self._sum_points(left, competition)
        out["away_team_points"] = self._sum_points(right, competition)
        return out

    def _sum_points(self, container, competition):
        if not container: return 0
        tracked_el = container.select_one(".team-stats-team-name strong")
        tracked = tracked_el.get_text(strip=True).lower() if tracked_el else ""
        if not tracked: return 0
        comp_key = self._normalise_competition(competition)
        points = 0
        counted = 0
        for item in container.select("li.team-stat-list-item"):
            date_el = item.select_one(".team-stats-date")
            if not date_el: continue
            date_text = date_el.get_text(strip=True)
            comp = date_text.split(":", 1)[0].strip().lower() if ":" in date_text else ""
            if comp_key and comp_key not in comp: continue
            teams = item.select(".team-stats-team")
            if len(teams) < 2: continue
            home_name = teams[0].get_text(" ", strip=True).lower()
            away_name = teams[1].get_text(" ", strip=True).lower()
            h_score = self._extract_score(teams[0])
            a_score = self._extract_score(teams[1])
            if h_score is None or a_score is None: continue
            if self._token_overlap(tracked, home_name):
                if h_score > a_score: points += 3
                elif h_score == a_score: points += 1
            elif self._token_overlap(tracked, away_name):
                if a_score > h_score: points += 3
                elif a_score == h_score: points += 1
            else:
                continue
            counted += 1
            if counted >= 5: break
        return points

    def _parse_current_season_games(self, side: str, competition: Optional[str]) -> int:
        container = self.soup.select_one("#last-matches #All")
        if not container:
            return 0
        block = container.select_one(".teamstats-left" if side == "home" else ".teamstats-right")
        if not block:
            return 0
        comp_key = self._normalise_competition(competition)
        if not comp_key:
            return 0
        count = 0
        for item in block.select("li.team-stat-list-item"):
            date_el = item.select_one(".team-stats-date")
            if not date_el:
                continue
            date_text = date_el.get_text(strip=True)
            comp = date_text.split(":", 1)[0].strip().lower() if ":" in date_text else ""
            if comp_key in comp:
                count += 1
        return count

    @staticmethod
    def _token_overlap(a, b):
        return bool(set(a.split()) & set(b.split()))

    @staticmethod
    def _normalise_competition(name):
        if not name: return ""
        name = name.lower()
        if " - " in name: name = name.split(" - ", 1)[1]
        return name.strip()

    def _extract_score(self, team_el):
        score_el = team_el.select_one(".score-right")
        if not score_el: return None
        return self._to_int(score_el.get_text(strip=True))

    # -- injuries -----------------------------------------------------------

    def _parse_injuries(self):
        out = {"home_key_attackers_out": 0, "home_key_defenders_out": 0, "home_key_midfielders_out": 0,
               "away_key_attackers_out": 0, "away_key_defenders_out": 0, "away_key_midfielders_out": 0}
        for outline in self.soup.select(".inj-two-outline"):
            header = outline.select_one(".light-header strong")
            if not header: continue
            header_text = header.get_text(" ", strip=True).lower()
            side = None
            if self.home_team and self._token_overlap(self.home_team.lower(), header_text):
                side = "home"
            elif self.away_team and self._token_overlap(self.away_team.lower(), header_text):
                side = "away"
            if not side: continue
            for row in outline.select(".inj-two-row"):
                if "inj-two-title" in row.get("class", []): continue
                info_el = row.select_one(".inj-two-info")
                info = info_el.get_text(strip=True).lower() if info_el else ""
                if "doubt" in info: continue
                detail = row.find("div", class_="inj-two-hidden")
                position = self._extract_position(detail)
                importance = self._extract_importance(detail)
                if importance != "key" or position is None: continue
                key = f"{side}_key_{position}s_out"
                if key in out: out[key] += 1
        return out

    @staticmethod
    def _extract_position(detail_el):
        if not detail_el: return None
        text = detail_el.get_text(" ", strip=True).lower()
        if "forward" in text or "striker" in text or "attacker" in text: return "attacker"
        if "defender" in text or "goalkeeper" in text: return "defender"
        if "midfielder" in text: return "midfielder"
        return None

    @staticmethod
    def _extract_importance(detail_el):
        if not detail_el: return "squad"
        text = detail_el.get_text(" ", strip=True)
        matches = 0
        goals = 0
        mm = re.search(r"Matches:\s*(\d+)", text)
        if mm: matches = int(mm.group(1))
        gg = re.search(r"Goals:\s*(\d+)", text)
        if gg: goals = int(gg.group(1))
        return "key" if (matches >= 3 or goals >= 1) else "squad"

    # -- midweek ------------------------------------------------------------

    def _parse_midweek(self, match_date_iso):
        out = {"home_team_played": False, "away_team_played": False}
        if not match_date_iso: return out
        try: match_dt = datetime.strptime(match_date_iso, "%Y-%m-%d")
        except ValueError: return out
        container = self.soup.select_one("#last-matches #All")
        if not container: return out
        left = container.select_one(".teamstats-left")
        right = container.select_one(".teamstats-right")
        out["home_team_played"] = self._played_midweek(left, match_dt)
        out["away_team_played"] = self._played_midweek(right, match_dt)
        return out

    def _played_midweek(self, container, match_dt):
        if not container: return False
        first = container.select_one("li.team-stat-list-item")
        if not first: return False
        date_el = first.select_one(".team-stats-date")
        if not date_el: return False
        mm = re.search(r"(\d{2})/(\d{2})", date_el.get_text(strip=True))
        if not mm: return False
        day, month = int(mm.group(1)), int(mm.group(2))
        try: last_dt = datetime(match_dt.year, month, day)
        except ValueError: return False
        if last_dt > match_dt:
            try: last_dt = last_dt.replace(year=match_dt.year - 1)
            except ValueError: return False
        return 0 <= (match_dt - last_dt).days <= 4


# ============================================================================
# PREDICTOR — AH model unchanged, plus EH conversion for display
# ============================================================================
class RefinedPredictor:
    def __init__(self):
        self.model_xg_home = 0.0
        self.model_xg_away = 0.0
        self.model_total = 0.0
        self.market_total = 0.0
        self.shrunk_total = 0.0
        self.shrunk_xg_home = 0.0
        self.shrunk_xg_away = 0.0
        self.probabilities = {}
        self.edges = {}
        self.bets = []
        self.skips = []
        self.effective_shrink = SHRINK_WEIGHT
        self.home_trust = 1.0
        self.away_trust = 1.0
        # Poisson grids kept so we can build the EH table on demand
        self._home_pmf = []
        self._away_pmf = []
        self._max_goals = 10

    def calculate_base_xg(self, home_data, away_data):
        ha = self._blend(home_data.get("home_goals_scored_season", 1.5),
                         home_data.get("home_goals_scored_last10", 1.5))
        ad = self._blend(away_data.get("away_goals_conceded_season", 1.5),
                         away_data.get("away_goals_conceded_last10", 1.5))
        aa = self._blend(away_data.get("away_goals_scored_season", 1.2),
                         away_data.get("away_goals_scored_last10", 1.2))
        hd = self._blend(home_data.get("home_goals_conceded_season", 1.2),
                         home_data.get("home_goals_conceded_last10", 1.2))
        self.model_xg_home = max(MIN_XG, (ha + ad) / 2)
        self.model_xg_away = max(MIN_XG, (aa + hd) / 2)
        self.model_total = self.model_xg_home + self.model_xg_away

    @staticmethod
    def _blend(s, l):
        if s <= 0: return max(MIN_XG, l)
        if l <= 0: return max(MIN_XG, s)
        return LAST10_WEIGHT * l + SEASON_WEIGHT * s

    def apply_adjustments(self, home_data, away_data):
        ha = self._form_adj(home_data.get("last5_points", 7))
        aa = self._form_adj(away_data.get("last5_points", 7))
        for inj in home_data.get("injuries", []):
            if not inj.get("key"): continue
            w = 1.0 if inj.get("confirmed_out", True) else 0.5
            if inj["position"] == "forward": ha -= INJURY_ADJ * w
            elif inj["position"] == "defender": aa += INJURY_ADJ * w
            elif inj["position"] == "midfielder": ha -= INJURY_ADJ * w * 0.66
        for inj in away_data.get("injuries", []):
            if not inj.get("key"): continue
            w = 1.0 if inj.get("confirmed_out", True) else 0.5
            if inj["position"] == "forward": aa -= INJURY_ADJ * w
            elif inj["position"] == "defender": ha += INJURY_ADJ * w
            elif inj["position"] == "midfielder": aa -= INJURY_ADJ * w * 0.66
        if home_data.get("played_midweek"): ha -= FATIGUE_HOME
        if away_data.get("played_midweek"): aa -= FATIGUE_AWAY
        self.model_xg_home = max(MIN_XG, self.model_xg_home + ha)
        self.model_xg_away = max(MIN_XG, self.model_xg_away + aa)
        self.model_total = self.model_xg_home + self.model_xg_away

    @staticmethod
    def _form_adj(p):
        if p <= 1: return -FORM_ADJ_HIGH
        if p <= 3: return -FORM_ADJ_MED
        if p >= 13: return FORM_ADJ_HIGH
        if p >= 10: return FORM_ADJ_MED
        return 0.0

    @staticmethod
    def _trust_weight(current_season_games: int) -> float:
        if current_season_games >= TRUST_FULL_SAMPLE:
            return 1.0
        raw = current_season_games / TRUST_FULL_SAMPLE
        return max(TRUST_MIN_WEIGHT, raw)

    def shrink_toward_market(self, market_total, home_current_games=999, away_current_games=999):
        self.market_total = max(0.5, market_total or self.model_total)

        self.home_trust = self._trust_weight(home_current_games)
        self.away_trust = self._trust_weight(away_current_games)

        combined_trust = 0.5 * (self.home_trust + self.away_trust)
        self.effective_shrink = 1.0 - combined_trust * (1.0 - SHRINK_WEIGHT)
        self.effective_shrink = min(MAX_EFFECTIVE_SHRINK, self.effective_shrink)

        self.shrunk_total = (
            (1.0 - self.effective_shrink) * self.model_total
            + self.effective_shrink * self.market_total
        )
        scale = self.shrunk_total / self.model_total if self.model_total > 0 else 1.0
        self.shrunk_xg_home = max(MIN_XG, self.model_xg_home * scale)
        self.shrunk_xg_away = max(MIN_XG, self.model_xg_away * scale)
        self.shrunk_total = self.shrunk_xg_home + self.shrunk_xg_away

    def run_poisson(self, max_goals=10):
        self._max_goals = max_goals
        home_probs = self._pmf(self.shrunk_xg_home, max_goals)
        away_probs = self._pmf(self.shrunk_xg_away, max_goals)
        self._home_pmf = home_probs
        self._away_pmf = away_probs

        p_home = p_draw = p_away = 0.0
        p_btts_yes = p_btts_no = 0.0
        p_over = p_under = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                if h > a: p_home += prob
                elif h == a: p_draw += prob
                else: p_away += prob
                if h >= 1 and a >= 1: p_btts_yes += prob
                else: p_btts_no += prob
                if h + a > 2.5: p_over += prob
                else: p_under += prob

        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total; p_draw /= total; p_away /= total

        # ---- AH probabilities (UNCHANGED — used for selection & edges) ----
        p_home_by_1 = p_away_by_1 = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                if h - a == 1: p_home_by_1 += prob
                elif a - h == 1: p_away_by_1 += prob

        ah_home = {
            -0.5: p_home,
            -0.25: p_home + 0.5 * p_draw,
            +0.25: p_home + 0.5 * p_draw,
            +0.5: p_home + p_draw,
            +0.75: p_home + p_draw + 0.5 * p_away_by_1,
            +1.0: p_home + p_draw,
        }
        ah_away = {
            -0.5: p_away,
            -0.25: p_away + 0.5 * p_draw,
            +0.25: p_away + 0.5 * p_draw,
            +0.5: p_away + p_draw,
            +0.75: p_away + p_draw + 0.5 * p_home_by_1,
            +1.0: p_away + p_draw,
        }

        self.probabilities = {
            "home_win": p_home, "draw": p_draw, "away_win": p_away,
            "btts_yes": p_btts_yes, "btts_no": p_btts_no,
            "over_25": p_over, "under_25": p_under,
            "ah_home": ah_home, "ah_away": ah_away,
        }

    @staticmethod
    def _pmf(lam, kmax):
        return [(lam ** i) * math.exp(-lam) / math.factorial(i) for i in range(kmax + 1)]

    # ---- AH -> EH conversion (for display only) ---------------------------
    def ah_to_eh(self, ah_line, side):
        """
        Convert an AH line for a given side into a 3-way EH probability
        distribution at the equivalent whole-goal line.

        AH line convention (as parsed):
            ah_home_line: line applied to HOME (e.g. -0.5, +0.25)
            ah_away_line: line applied to AWAY (e.g. +0.5, -0.25)

        We convert to the nearest whole-goal EH line and return
        {"line": int, "home": p, "draw": p, "away": p}.
        Quarter lines are handled by splitting 50/50 between the two
        adjacent whole lines, exactly matching AH settlement.
        """
        if ah_line is None:
            return None

        max_goals = self._max_goals
        home_probs = self._home_pmf
        away_probs = self._away_pmf

        def eh_at(line_int):
            """3-way probs when 'side' receives line_int goals."""
            if side == "home":
                home_adj, away_adj = line_int, 0
            else:
                home_adj, away_adj = 0, line_int
            p_h = p_d = p_a = 0.0
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    prob = home_probs[h] * away_probs[a]
                    adj_h = h + home_adj
                    adj_a = a + away_adj
                    if adj_h > adj_a: p_h += prob
                    elif adj_h == adj_a: p_d += prob
                    else: p_a += prob
            tot = p_h + p_d + p_a
            if tot > 0:
                p_h /= tot; p_d /= tot; p_a /= tot
            return {"home": p_h, "draw": p_d, "away": p_a}

        # Split quarter lines into the two adjacent whole lines
        # e.g. +0.25 -> half at 0, half at +1 (wait: +0.25 is half 0, half +0.5, but EH only has whole lines)
        # For EH display we snap to the nearest whole line, but for quarter
        # lines we blend the two nearest whole lines weighted 50/50.
        lower = math.floor(ah_line)
        upper = math.ceil(ah_line)
        if lower == upper:
            probs = eh_at(int(ah_line))
            return {"line": int(ah_line), **probs}

        # Quarter line -> blend lower and upper
        pl = eh_at(lower)
        pu = eh_at(upper)
        blended = {
            "home": 0.5 * pl["home"] + 0.5 * pu["home"],
            "draw": 0.5 * pl["draw"] + 0.5 * pu["draw"],
            "away": 0.5 * pl["away"] + 0.5 * pu["away"],
        }
        return {"line": ah_line, **blended}

    def calculate_edges(self, odds):
        self.edges = {}
        for key, prob_key, odd_val in [
            ("home_win", "home_win", odds.get("home_odds")),
            ("draw", "draw", odds.get("draw_odds")),
            ("away_win", "away_win", odds.get("away_odds")),
            ("btts_yes", "btts_yes", odds.get("btts_yes_odds")),
            ("btts_no", "btts_no", odds.get("btts_no_odds")),
            ("over_25", "over_25", odds.get("over_25_odds")),
            ("under_25", "under_25", odds.get("under_25_odds")),
        ]:
            if not odd_val or odd_val <= 1.01: continue
            self.edges[key] = self.probabilities.get(prob_key, 0.0) - 1.0 / odd_val

    def select_markets(self, odds, btts_rate=0.5, corner_data=None):
        self.bets = []
        self.skips = []
        self._check_outright(odds)
        self._check_ah_home(odds)
        self._check_ah_away(odds)
        self._check_ah_underdog(odds)
        self._check_btts(odds, btts_rate)
        self._check_ou(odds)
        self._check_corners(corner_data)
        self.skips.append({"market": "Correct Score", "reason": "Never bet — pure lottery"})
        self.skips.append({"market": "First Goalscorer", "reason": "Never bet"})
        self.skips.append({"market": "Anytime Goalscorer", "reason": "Avoid — high variance"})
        return self.bets

    def _check_outright(self, odds):
        for outcome, label in [("home_win", "Home"), ("away_win", "Away"), ("draw", "Draw")]:
            edge = self.edges.get(outcome)
            if edge is None: continue
            prob = self.probabilities.get(outcome, 0.0)
            self.skips.append({
                "market": f"Outright 1X2: {label}",
                "reason": f"P={prob:.1%}, edge {edge:+.1%} (use AH channel)",
            })

    def _check_ah_home(self, odds):
        line = odds.get("ah_home_line")
        ah_odds = odds.get("ah_home")
        if line is None or not ah_odds or ah_odds <= 1.01: return
        p_home = self.probabilities.get("home_win", 0.0)
        effective = self._effective_ah_prob("home", line)
        if effective is None: return
        implied = 1.0 / ah_odds
        edge = effective - implied
        if edge > EDGE_MAX:
            self.skips.append({"market": f"Home {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} > 20% (model error)"})
            return
        if edge < EDGE_MIN:
            self.skips.append({"market": f"Home {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} < 5%"})
            return
        if p_home < AH_HOME_OUTRIGHT_MIN:
            self.skips.append({"market": f"Home {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} good but P(Home)={p_home:.1%} < 45%"})
            return
        self.bets.append({
            "market": "Match Result (AH)",
            "selection": f"Home {line:+g} AH",
            "prob": effective, "edge": edge, "odds": ah_odds,
            "stake": "1 unit", "confidence": "High",
        })

    def _check_ah_away(self, odds):
        line = odds.get("ah_away_line")
        ah_odds = odds.get("ah_away")
        if line is None or not ah_odds or ah_odds <= 1.01: return
        p_away = self.probabilities.get("away_win", 0.0)
        effective = self._effective_ah_prob("away", line)
        if effective is None: return
        implied = 1.0 / ah_odds
        edge = effective - implied
        if edge > EDGE_MAX:
            self.skips.append({"market": f"Away {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} > 20% (model error)"})
            return
        if edge < EDGE_MIN:
            self.skips.append({"market": f"Away {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} < 5%"})
            return
        if p_away < AH_AWAY_OUTRIGHT_MIN:
            self.skips.append({"market": f"Away {line:+g} AH",
                               "reason": f"Edge {edge:+.1%} good but P(Away)={p_away:.1%} < 45%"})
            return
        self.bets.append({
            "market": "Match Result (AH)",
            "selection": f"Away {line:+g} AH",
            "prob": effective, "edge": edge, "odds": ah_odds,
            "stake": "1 unit", "confidence": "High",
        })

    def _check_ah_underdog(self, odds):
        p_home = self.probabilities.get("home_win", 0.0)
        p_away = self.probabilities.get("away_win", 0.0)
        if p_home < p_away and p_home >= AH_UNDERDOG_MIN:
            side, line = "home", odds.get("ah_home_line")
            ah_odds = odds.get("ah_home")
        elif p_away < p_home and p_away >= AH_UNDERDOG_MIN:
            side, line = "away", odds.get("ah_away_line")
            ah_odds = odds.get("ah_away")
        else:
            return
        if line is None or not ah_odds or ah_odds <= 1.01: return
        if line <= 0: return
        effective = self._effective_ah_prob(side, line)
        if effective is None: return
        implied = 1.0 / ah_odds
        edge = effective - implied
        if not (EDGE_MIN < edge < EDGE_MAX): return
        for existing in self.bets:
            if f"{side.capitalize()} {line:+g}" in existing["selection"]:
                return
        self.bets.append({
            "market": "Match Result (AH)",
            "selection": f"{side.capitalize()} {line:+g} AH (underdog)",
            "prob": effective, "edge": edge, "odds": ah_odds,
            "stake": "0.5 units", "confidence": "Selective",
        })

    def _effective_ah_prob(self, side, line):
        ah = self.probabilities.get(f"ah_{side}", {})
        candidates = sorted(ah.keys(), key=lambda k: abs(k - line))
        if not candidates: return None
        closest = candidates[0]
        if abs(closest - line) > 0.01:
            p_win = self.probabilities.get(f"{side}_win", 0.0)
            p_draw = self.probabilities.get("draw", 0.0)
            if line >= 0.5: return p_win + p_draw
            if line >= 0.25: return p_win + 0.5 * p_draw
            if line >= -0.25: return p_win + 0.5 * p_draw
            return p_win
        return ah[closest]

    def _check_btts(self, odds, btts_rate):
        if btts_rate > 0.65:
            edge = self.edges.get("btts_yes")
            if edge is not None and EDGE_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "BTTS", "selection": "BTTS Yes",
                    "prob": self.probabilities["btts_yes"], "edge": edge,
                    "odds": odds.get("btts_yes_odds", 0),
                    "stake": "1 unit", "confidence": "High",
                })
        elif btts_rate < 0.45:
            edge = self.edges.get("btts_no")
            if edge is not None and EDGE_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "BTTS", "selection": "BTTS No",
                    "prob": self.probabilities["btts_no"], "edge": edge,
                    "odds": odds.get("btts_no_odds", 0),
                    "stake": "1 unit", "confidence": "High",
                })

    def _check_ou(self, odds):
        if self.shrunk_total > 3.00:
            edge = self.edges.get("over_25")
            if edge is not None and EDGE_OU_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "Over/Under", "selection": "Over 2.5 (small stake)",
                    "prob": self.probabilities["over_25"], "edge": edge,
                    "odds": odds.get("over_25_odds", 0),
                    "stake": "0.5 units", "confidence": "Selective",
                })
            else:
                reason = f"Total > 3.00 fired but edge {edge:+.1%} < 8%" if edge is not None else "Total > 3.00 fired"
                self.skips.append({"market": "Over/Under 2.5 (Over)", "reason": reason})
        elif self.shrunk_total < 2.20:
            edge = self.edges.get("under_25")
            if edge is not None and EDGE_OU_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "Over/Under", "selection": "Under 2.5 (small stake)",
                    "prob": self.probabilities["under_25"], "edge": edge,
                    "odds": odds.get("under_25_odds", 0),
                    "stake": "0.5 units", "confidence": "Selective",
                })
            else:
                reason = f"Total < 2.20 fired but edge {edge:+.1%} < 8%" if edge is not None else "Total < 2.20 fired"
                self.skips.append({"market": "Over/Under 2.5 (Under)", "reason": reason})
        else:
            self.skips.append({"market": "Over/Under 2.5",
                               "reason": f"Shrunk total {self.shrunk_total:.2f} in neutral zone (2.20–3.00)"})

    def _check_corners(self, corner_data):
        if not corner_data: return
        home_for = corner_data.get("home_avg_corners", 0)
        away_against = corner_data.get("away_conceded_corners", 0)
        away_for = corner_data.get("away_avg_corners", 0)
        home_against = corner_data.get("home_conceded_corners", 0)

        if home_for >= 5.5 and away_against >= 5.0:
            line = corner_data.get("home_corners_line")
            odds = corner_data.get("home_corners_over")
            if line and abs(line - 4.5) < 0.1 and odds and odds > 1.01:
                self.bets.append({
                    "market": "Corners", "selection": f"Home Over {line:g} corners",
                    "prob": 0.55, "edge": 0.05, "odds": odds,
                    "stake": "0.5 units", "confidence": "Selective",
                })
            else:
                self.skips.append({
                    "market": "Home Corners",
                    "reason": f"Trigger fired but no matching Over 4.5 line (offered line: {line})",
                })
        if away_for >= 5.5 and home_against >= 5.0:
            line = corner_data.get("away_corners_line")
            odds = corner_data.get("away_corners_over")
            if line and abs(line - 4.5) < 0.1 and odds and odds > 1.01:
                self.bets.append({
                    "market": "Corners", "selection": f"Away Over {line:g} corners",
                    "prob": 0.55, "edge": 0.05, "odds": odds,
                    "stake": "0.5 units", "confidence": "Selective",
                })
            else:
                self.skips.append({
                    "market": "Away Corners",
                    "reason": f"Trigger fired but no matching Over 4.5 line (offered line: {line})",
                })

    def get_full_analysis(self):
        return {
            "model_xg_home": self.model_xg_home, "model_xg_away": self.model_xg_away,
            "model_total": self.model_total, "market_total": self.market_total,
            "shrunk_xg_home": self.shrunk_xg_home, "shrunk_xg_away": self.shrunk_xg_away,
            "shrunk_total": self.shrunk_total,
            "probabilities": dict(self.probabilities),
            "edges": dict(self.edges),
            "bets": list(self.bets), "skips": list(self.skips),
            "effective_shrink": self.effective_shrink,
            "home_trust": self.home_trust,
            "away_trust": self.away_trust,
        }


# ============================================================================
# HELPERS
# ============================================================================
def load_parsed_match(parsed: dict) -> dict:
    odds = parsed.get("odds", {}) or {}
    h = parsed.get("home_team_last10_home", {}) or {}
    a = parsed.get("away_team_last10_away", {}) or {}
    season = parsed.get("season_splits", {}) or {}
    form = parsed.get("last5_form", {}) or {}
    inj = parsed.get("injuries", {}) or {}
    mid = parsed.get("midweek_fixture", {}) or {}

    market_total = derive_market_total(odds.get("over_25"), odds.get("under_25"))
    hg = max(1, h.get("wins", 0) + h.get("draws", 0) + h.get("losses", 0))
    ag = max(1, a.get("wins", 0) + a.get("draws", 0) + a.get("losses", 0))
    btts_rate = ((h.get("btts_yes", 0) / hg) + (a.get("btts_yes", 0) / ag)) / 2.0

    return {
        "home_team": parsed.get("home_team"), "away_team": parsed.get("away_team"),
        "league": parsed.get("competition"),
        "date": parsed.get("match_date") or datetime.now().strftime("%Y-%m-%d"),
        "home_odds": odds.get("home") or 2.0,
        "draw_odds": odds.get("draw") or 3.4,
        "away_odds": odds.get("away") or 3.0,
        "home_xg": h.get("gf_per_game") or 1.2,
        "away_xg": a.get("gf_per_game") or 1.0,
        "market_total": market_total,
        "btts_rate": btts_rate,
        "home_current_games": parsed.get("home_current_season_games", 0),
        "away_current_games": parsed.get("away_current_season_games", 0),
        "home_data": {
            "home_goals_scored_season": season.get("home_team_home_gf_pg") or h.get("gf_per_game", 1.2),
            "home_goals_scored_last10": h.get("gf_per_game", 1.2),
            "home_goals_conceded_season": season.get("home_team_home_ga_pg") or h.get("ga_per_game", 1.2),
            "home_goals_conceded_last10": h.get("ga_per_game", 1.2),
            "last5_points": form.get("home_team_points", 7),
            "injuries": _inj_list("home", inj),
            "played_midweek": mid.get("home_team_played", False),
        },
        "away_data": {
            "away_goals_scored_season": season.get("away_team_away_gf_pg") or a.get("gf_per_game", 1.0),
            "away_goals_scored_last10": a.get("gf_per_game", 1.0),
            "away_goals_conceded_season": season.get("away_team_away_ga_pg") or a.get("ga_per_game", 1.2),
            "away_goals_conceded_last10": a.get("ga_per_game", 1.2),
            "last5_points": form.get("away_team_points", 7),
            "injuries": _inj_list("away", inj),
            "played_midweek": mid.get("away_team_played", False),
        },
        "odds": {
            "btts_yes_odds": odds.get("btts_yes") or 1.80,
            "btts_no_odds": odds.get("btts_no") or 2.00,
            "over_25_odds": odds.get("over_25") or 1.90,
            "under_25_odds": odds.get("under_25") or 1.90,
            "ah_home_line": odds.get("ah_home_line"),
            "ah_home": odds.get("ah_home"),
            "ah_away_line": odds.get("ah_away_line"),
            "ah_away": odds.get("ah_away"),
        },
        "corner_data": {
            "home_avg_corners": h.get("corners_for", 0),
            "away_avg_corners": a.get("corners_for", 0),
            "home_conceded_corners": h.get("corners_against", 0),
            "away_conceded_corners": a.get("corners_against", 0),
            "home_corners_line": odds.get("home_corners_line"),
            "home_corners_over": odds.get("home_corners_over"),
            "away_corners_line": odds.get("away_corners_line"),
            "away_corners_over": odds.get("away_corners_over"),
        } if h.get("corners_for") else None,
        "_parsed": parsed,
    }


def _inj_list(side, inj):
    out = []
    for _ in range(inj.get(f"{side}_key_attackers_out", 0)):
        out.append({"position": "forward", "key": True, "confirmed_out": True})
    for _ in range(inj.get(f"{side}_key_defenders_out", 0)):
        out.append({"position": "defender", "key": True, "confirmed_out": True})
    for _ in range(inj.get(f"{side}_key_midfielders_out", 0)):
        out.append({"position": "midfielder", "key": True, "confirmed_out": True})
    return out


def derive_market_total(over_odds, under_odds):
    if not over_odds or not under_odds: return None
    p_over = (1.0 / over_odds) / ((1.0 / over_odds) + (1.0 / under_odds))
    lo, hi = 0.1, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = 1.0 - _pcdf(mid, 2)
        if p < p_over: lo = mid
        else: hi = mid
    return round((lo + hi) / 2, 2)


def _pcdf(lam, k):
    return sum((lam ** i) * math.exp(-lam) / math.factorial(i) for i in range(k + 1))


def parse_match_date(d):
    if not d: return datetime(1900, 1, 1)
    if isinstance(d, (date, datetime)): return datetime(d.year, d.month, d.day)
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try: return datetime.strptime(s, fmt)
        except ValueError: continue
    return datetime(1900, 1, 1)


# ============================================================================
# DB OPERATIONS  (AH settlement kept — selections are still stored as AH)
# ============================================================================
def save_bet_to_db(sb, match, analysis, bet):
    if sb is None: return None
    try:
        mdate = parse_match_date(match.get("date")).strftime("%Y-%m-%d")
        rec = {
            "match_date": mdate,
            "home_team": match.get("home_team", "Unknown"),
            "away_team": match.get("away_team", "Unknown"),
            "league": match.get("league", "Unknown"),
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            "model_xg_home": analysis.get("model_xg_home", 0),
            "model_xg_away": analysis.get("model_xg_away", 0),
            "model_total": analysis.get("model_total", 0),
            "market_total": analysis.get("market_total", 0),
            "shrunk_total": analysis.get("shrunk_total", 0),
            "market": bet.get("market", ""),
            "selection": bet.get("selection", ""),
            "bet_prob": bet.get("prob", 0),
            "bet_edge": bet.get("edge", 0),
            "bet_odds": bet.get("odds", 0),
            "stake": bet.get("stake", "1 unit"),
            "confidence": bet.get("confidence", "High"),
        }
        resp = sb.table(TABLE_NAME).insert(rec).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as e:
        st.warning(f"Save failed: {e}")
        return None


def get_pending(sb):
    if sb is None: return []
    try:
        resp = sb.table(TABLE_NAME).select("*").is_("actual_result", "null").execute()
        return sorted(resp.data or [], key=lambda x: parse_match_date(x.get("match_date")))
    except Exception:
        return []


def submit_result(sb, rid, hg, ag):
    if sb is None: return False
    try:
        actual = "1" if hg > ag else "2" if ag > hg else "X"
        resp = sb.table(TABLE_NAME).select("*").eq("id", rid).execute()
        if not resp.data: return False
        rec = resp.data[0]
        market = rec.get("market", "")
        sel = rec.get("selection", "")
        correct = False
        if market == "Match Result (AH)":
            m = re.search(r"(Home|Away)\s+([+-]?[\d.]+)\s+AH", sel)
            if m:
                side = m.group(1).lower()
                line = float(m.group(2))
                margin = (hg - ag) if side == "home" else (ag - hg)
                adjusted = margin + line
                if adjusted > 0: correct = True
                elif adjusted == 0: correct = None
                else: correct = False
        elif market == "Match Result":
            if "Home" in sel:
                correct = hg > ag if "−" in sel or "-" in sel else hg >= ag
            elif "Away" in sel:
                correct = ag > hg if "−" in sel or "-" in sel else ag >= hg
            elif "Draw" in sel:
                correct = hg == ag
        elif market == "BTTS":
            correct = (hg >= 1 and ag >= 1) if "Yes" in sel else (hg == 0 or ag == 0)
        elif market == "Over/Under":
            t = hg + ag
            correct = t > 2.5 if "Over" in sel else t < 2.5
        sb.table(TABLE_NAME).update({
            "actual_home_goals": hg, "actual_away_goals": ag,
            "actual_result": actual, "is_correct": correct,
        }).eq("id", rid).execute()
        return True
    except Exception as e:
        st.error(f"Submit failed: {e}")
        return False


def get_results(sb):
    if sb is None: return []
    try:
        resp = sb.table(TABLE_NAME).select("*").not_.is_("actual_result", "null").execute()
        return sorted(resp.data or [], key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except Exception:
        return []


# ============================================================================
# DISPLAY HELPERS
# ============================================================================
def edge_class(edge):
    if edge is None: return "edge-neutral"
    if edge >= 0.05: return "edge-pos"
    if edge <= -0.05: return "edge-neg"
    return "edge-neutral"


def render_prediction_card(match, parsed, analysis, predictor):
    meta_parts = []
    if match.get("league"): meta_parts.append(match["league"])
    if parsed.get("venue"): meta_parts.append(parsed["venue"])
    if match.get("date"): meta_parts.append(match["date"])
    if parsed.get("kickoff"): meta_parts.append(parsed["kickoff"])
    meta = "  ·  ".join(meta_parts)

    st.markdown(f"""
    <div class="team-header">
        <div class="team-names">{match['home_team']} &nbsp;🆚&nbsp; {match['away_team']}</div>
        <div class="team-meta">{meta}</div>
    </div>
    """, unsafe_allow_html=True)

    if analysis["bets"]:
        primary = analysis["bets"][0]
        st.markdown(f"""
        <div class="verdict-bet">
            <div class="verdict-label">⭐ Primary Pick</div>
            <div class="verdict-pick">{primary['selection']}</div>
            <div class="verdict-detail">
                {primary['market']} &nbsp;·&nbsp; @ <strong>{primary['odds']:.2f}</strong>
                &nbsp;·&nbsp; Edge <strong>{primary['edge']:+.1%}</strong>
                &nbsp;·&nbsp; Stake <strong>{primary['stake']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">Verdict</div>
            <div class="verdict-noedge">No value edge on any market</div>
            <div class="verdict-detail-grey">
                The market has priced this match efficiently. Skip it and move on.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if len(analysis["bets"]) > 1:
        st.markdown('<div class="section-title">Other Picks</div>', unsafe_allow_html=True)
        for b in analysis["bets"][1:]:
            st.markdown(f"""
            <div class="alt-bet">
                <div class="alt-bet-title">{b['selection']}</div>
                <div class="alt-bet-meta">
                    {b['market']} &nbsp;·&nbsp; @ {b['odds']:.2f}
                    &nbsp;·&nbsp; Edge <strong>{b['edge']:+.1%}</strong>
                    &nbsp;·&nbsp; {b['stake']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sample Quality & Shrinkage</div>', unsafe_allow_html=True)
    home_g = match.get("home_current_games", 0)
    away_g = match.get("away_current_games", 0)
    home_trust = analysis.get("home_trust", 1.0)
    away_trust = analysis.get("away_trust", 1.0)
    eff_shrink = analysis.get("effective_shrink", 0.5)

    def trust_class(t):
        if t >= 0.85: return "trust-ok"
        if t >= 0.50: return "trust-warn"
        return "trust-warn"

    st.markdown(f"""
    <div class="trust-row">
        <span>🏠 {match['home_team']} — current-season games</span>
        <span class="trust-value {trust_class(home_trust)}">{home_g} games · trust {home_trust:.0%}</span>
    </div>
    <div class="trust-row">
        <span>✈️ {match['away_team']} — current-season games</span>
        <span class="trust-value {trust_class(away_trust)}">{away_g} games · trust {away_trust:.0%}</span>
    </div>
    <div class="trust-row">
        <span>Effective shrinkage toward market</span>
        <span class="trust-value">{eff_shrink:.0%} <span style="color:#64748b;">(normal: 50%)</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Outcome Probabilities</div>', unsafe_allow_html=True)
    probs = analysis["probabilities"]
    edges = analysis["edges"]

    def stat_card(col, value, label, edge, css_class):
        cls = edge_class(edge)
        edge_str = f"{edge:+.1%}" if edge is not None else "—"
        col.markdown(f"""
        <div class="stat-card {css_class}">
            <div class="stat-value">{value:.1%}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-edge {cls}">Edge {edge_str}</div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    stat_card(c1, probs.get("home_win", 0), "Home Win", edges.get("home_win"), "stat-card-home")
    stat_card(c2, probs.get("draw", 0), "Draw", edges.get("draw"), "stat-card-draw")
    stat_card(c3, probs.get("away_win", 0), "Away Win", edges.get("away_win"), "stat-card-away")

    # ------------------------------------------------------------------
    # EUROPEAN HANDICAP DISPLAY (converted from the parsed AH lines)
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">European Handicap (3-way, converted from AH)</div>',
                unsafe_allow_html=True)
    eh_home = predictor.ah_to_eh(match["odds"].get("ah_home_line"), "home")
    eh_away = predictor.ah_to_eh(match["odds"].get("ah_away_line"), "away")

    def render_eh_block(col, eh, side_label, ah_line, ah_odds):
        if not eh:
            col.info("No AH line parsed for this side.")
            return
        line = eh["line"]
        # For display we always show from the perspective of the side that
        # receives the line, but label the columns as Home / Draw / Away.
        if side_label == "Home":
            p_h, p_d, p_a = eh["home"], eh["draw"], eh["away"]
            title = f"Home {line:+g} EH"
        else:
            # Away gets the line; Home is the opponent
            p_h, p_d, p_a = eh["home"], eh["draw"], eh["away"]
            title = f"Away {line:+g} EH"
        col.markdown(f"""
        <div class="eh-table" style="display:block;">
            <div class="eh-line" style="margin-bottom:0.5rem;">{title}
                <span style="color:#64748b;font-weight:400;">
                    &nbsp;·&nbsp; from AH {ah_line:+g} @ {ah_odds:.2f}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.85rem;">
                <span>Home <strong style="color:#10b981;">{p_h:.1%}</strong></span>
                <span>Draw <strong style="color:#fbbf24;">{p_d:.1%}</strong></span>
                <span>Away <strong style="color:#3b82f6;">{p_a:.1%}</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        render_eh_block(c1, eh_home, "Home",
                        match["odds"].get("ah_home_line") or 0,
                        match["odds"].get("ah_home") or 0)
    with c2:
        render_eh_block(c2, eh_away, "Away",
                        match["odds"].get("ah_away_line") or 0,
                        match["odds"].get("ah_away") or 0)

    # ------------------------------------------------------------------
    # AH panel (kept, since it's what the model actually prices)
    # ------------------------------------------------------------------
    st.markdown('<div class="section-title">Asian Handicap Probabilities (model)</div>', unsafe_allow_html=True)
    ah_home = probs.get("ah_home", {})
    ah_away = probs.get("ah_away", {})
    ah_lines = sorted(set(list(ah_home.keys()) + list(ah_away.keys())))
    if ah_lines:
        cols = st.columns(min(4, len(ah_lines)))
        for i, line in enumerate(ah_lines):
            col = cols[i % len(cols)]
            p_h = ah_home.get(line, 0)
            p_a = ah_away.get(line, 0)
            with col:
                st.markdown(f"""
                <div class="eh-table">
                    <div>
                        <div class="eh-line">Line {line:+g}</div>
                        <div class="eh-line" style="font-size:0.7rem;color:#64748b;">
                            Home: {p_h:.1%} | Away: {p_a:.1%}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Expected Goals Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="xg-row">
            <span class="xg-team">🏠 {match['home_team']} (shrunk)</span>
            <span class="xg-value">{analysis['shrunk_xg_home']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="xg-row">
            <span class="xg-team">✈️ {match['away_team']} (shrunk)</span>
            <span class="xg-value">{analysis['shrunk_xg_away']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="xg-row">
            <span class="xg-team">Model Total</span>
            <span class="xg-value">{analysis['model_total']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="xg-row">
            <span class="xg-team">Market Total</span>
            <span class="xg-value" style="color:#fbbf24;">{analysis['market_total']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        f"Model total shrunk {analysis.get('effective_shrink', 0.5):.0%} toward market "
        f"(higher = less trust in the model's small sample). Final shrunk total: "
        f"**{analysis['shrunk_total']:.2f}** goals expected."
    )

    st.markdown('<div class="section-title">Other Markets</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        e = edges.get("btts_yes")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{probs.get('btts_yes', 0):.1%}</div>
            <div class="stat-label">BTTS Yes</div>
            <div class="stat-edge {edge_class(e)}">Edge {f"{e:+.1%}" if e is not None else "—"}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        e = edges.get("over_25")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{probs.get('over_25', 0):.1%}</div>
            <div class="stat-label">Over 2.5</div>
            <div class="stat-edge {edge_class(e)}">Edge {f"{e:+.1%}" if e is not None else "—"}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        e = edges.get("under_25")
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{probs.get('under_25', 0):.1%}</div>
            <div class="stat-label">Under 2.5</div>
            <div class="stat-edge {edge_class(e)}">Edge {f"{e:+.1%}" if e is not None else "—"}</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(f"❌ Skipped markets ({len(analysis['skips'])})"):
        for s in analysis["skips"]:
            st.write(f"**{s['market']}** — {s['reason']}")


# ============================================================================
# MAIN UI
# ============================================================================
def main():
    st.title("⚽ Refined Prediction Strategy")
    st.caption("xG-based model with sample-size shrinkage · AH parsed, EH shown for interpretation")

    sb = get_supabase()
    if sb is None:
        st.info("ℹ️ Supabase not configured — predictions work, saving disabled.")

    tabs = st.tabs(["⚽ Predict", "📝 Pending", "📊 Records"])

    with tabs[0]:
        st.markdown("### Paste Sportsgambler HTML")
        st.caption("Open a Sportsgambler preview, View Source, copy the HTML, paste below.")
        text = st.text_area("HTML", height=220, key="html_input", label_visibility="collapsed")

        if st.button("⚽ Generate Prediction", type="primary"):
            if not text or len(text.strip()) < 200:
                st.error("Please paste a full Sportsgambler preview page.")
            else:
                try:
                    with st.spinner("Analysing match..."):
                        parsed = SportsgamblerParser(text).parse()
                        match = load_parsed_match(parsed)
                        p = RefinedPredictor()
                        p.calculate_base_xg(match["home_data"], match["away_data"])
                        p.apply_adjustments(match["home_data"], match["away_data"])
                        p.shrink_toward_market(
                            match["market_total"] or (match["home_xg"] + match["away_xg"]),
                            home_current_games=match.get("home_current_games", 999),
                            away_current_games=match.get("away_current_games", 999),
                        )
                        p.run_poisson()
                        edge_odds = {"home_odds": match["home_odds"], "draw_odds": match["draw_odds"],
                                     "away_odds": match["away_odds"], **match["odds"]}
                        p.calculate_edges(edge_odds)
                        p.select_markets(edge_odds, match["btts_rate"], match.get("corner_data"))
                        analysis = p.get_full_analysis()

                    st.markdown("---")
                    render_prediction_card(match, parsed, analysis, p)

                    if sb is not None and analysis["bets"]:
                        saved = 0
                        for b in analysis["bets"]:
                            if save_bet_to_db(sb, match, analysis, b):
                                saved += 1
                        if saved:
                            st.success(f"💾 Saved {saved} bet(s) to database.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

    with tabs[1]:
        st.subheader("📝 Pending Bets")
        if sb is None:
            st.info("Supabase not configured.")
        else:
            pending = get_pending(sb)
            if not pending:
                st.info("No pending bets.")
            for a in pending:
                with st.expander(f"{a.get('match_date','')} · {a.get('home_team','')} vs {a.get('away_team','')}"):
                    st.write(f"**{a.get('market','')}**: {a.get('selection','')} @ {a.get('bet_odds',0):.2f}")
                    c1, c2 = st.columns(2)
                    hg = c1.number_input(f"{a.get('home_team','')} goals", 0, 15, 0, key=f"hg_{a['id']}")
                    ag = c2.number_input(f"{a.get('away_team','')} goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("Submit", key=f"sub_{a['id']}"):
                        if submit_result(sb, a["id"], hg, ag):
                            st.success("Saved.")
                            st.rerun()

    with tabs[2]:
        st.subheader("📊 Performance Records")
        if sb is None:
            st.info("Supabase not configured.")
        else:
            results = get_results(sb)
            if not results:
                st.info("No results recorded yet.")
            else:
                total = len(results)
                wins = sum(1 for r in results if r.get("is_correct"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Bets", total)
                c2.metric("Win Rate", f"{wins/total*100:.0f}%")
                c3.metric("Wins", wins)
                df = pd.DataFrame([{
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team','')} vs {r.get('away_team','')}",
                    "Market": r.get("market", ""),
                    "Selection": r.get("selection", ""),
                    "Edge": f"{r.get('bet_edge', 0):+.1%}",
                    "Odds": f"{r.get('bet_odds', 0):.2f}",
                    "Score": f"{r.get('actual_home_goals','')}-{r.get('actual_away_goals','')}",
                    "Result": "✅" if r.get("is_correct") else "❌",
                } for r in results])
                st.dataframe(df, use_container_width=True)


main()
