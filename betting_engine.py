"""
betting_engine.py
=================

Refined Prediction Strategy — complete engine.

- Parses Sportsgambler HTML into structured data
- Computes xG from home/away splits
- Applies form, injury, and fatigue adjustments
- Shrinks toward market total
- Runs Poisson to get probabilities
- Calculates edges and selects markets

Only external dependency: beautifulsoup4 (for HTML parsing).
Uses pure-Python Poisson if scipy is unavailable.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Optional


# ============================================================================
# CONSTANTS
# ============================================================================

EDGE_MIN = 0.05
EDGE_MAX = 0.30
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


# ============================================================================
# LAZY IMPORTS
# ============================================================================

def _has_bs4() -> bool:
    try:
        import bs4  # noqa: F401
        return True
    except ImportError:
        return False


def _has_scipy() -> bool:
    try:
        import scipy.stats  # noqa: F401
        return True
    except ImportError:
        return False


# ============================================================================
# SPORTSGAMBLER HTML PARSER
# ============================================================================

class SportsgamblerParser:
    """Parse a Sportsgambler preview page into the spec JSON schema."""

    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError(
                "beautifulsoup4 is required for HTML parsing. "
                "Add 'beautifulsoup4' to your requirements.txt."
            )
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
        }

        corners = self._parse_corners()
        result["home_team_last10_home"].update(corners["home"])
        result["away_team_last10_away"].update(corners["away"])
        return result

    # -- identity ------------------------------------------------------------

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
            if re.search(
                r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Championship|MLS|Cup|Division)",
                text, re.I,
            ):
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

    # -- odds ----------------------------------------------------------------

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
                    entries.append((
                        label_el.get_text(strip=True),
                        self._to_float(odds_el.get_text(strip=True)),
                    ))
            self._apply_market(flat, market, entries)
        return flat

    def _apply_market(self, flat, market, entries):
        m = market.lower()

        if "full-time result" in m:
            for label, val in entries:
                if label == "1":
                    flat["home"] = val
                elif label == "X":
                    flat["draw"] = val
                elif label == "2":
                    flat["away"] = val
            return

        if "asian handicap" in m:
            for label, val in entries:
                mm = re.match(r"(\d)\s*Hcp\s*([+-]?[\d.]+)", label)
                if not mm:
                    continue
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
                if label.lower() == "yes":
                    flat["btts_yes"] = val
                elif label.lower() == "no":
                    flat["btts_no"] = val
            return

        if "total goals" in m:
            for label, val in entries:
                if "over 2.5" in label.lower():
                    flat["over_25"] = val
                elif "under 2.5" in label.lower():
                    flat["under_25"] = val
            return

        if "total corners" in m:
            for label, val in entries:
                mm = re.match(r"(Over|Under)\s+([\d.]+)", label, re.I)
                if mm:
                    flat["corners_line"] = self._to_float(mm.group(2))
                    if mm.group(1).lower() == "over":
                        flat["corners_over"] = val
                    else:
                        flat["corners_under"] = val
            return

        if m.endswith(" corners"):
            team_name = market[: -len(" Corners")].strip()
            is_home = self._name_matches(self.home_team, team_name)
            is_away = self._name_matches(self.away_team, team_name)
            for label, val in entries:
                mm = re.match(r"Over\s+([\d.]+)", label, re.I)
                if not mm:
                    continue
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
        if not a or not b:
            return False
        return bool(set(a.lower().split()) & set(b.lower().split()))

    @staticmethod
    def _to_float(s):
        try:
            return float(str(s).strip())
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_int(s):
        try:
            return int(str(s).strip())
        except (ValueError, TypeError):
            return 0

    # -- last-10 splits ------------------------------------------------------

    def _parse_last10_splits(self, side):
        split = self._empty_split()
        table = self.soup.select_one(".st-table")
        if not table:
            return split
        rows = table.select("tbody tr")
        idx = 0 if side == "home" else 1
        if idx >= len(rows):
            return split
        cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
        if len(cells) < 9:
            return split
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
        return {
            "wins": 0, "draws": 0, "losses": 0,
            "gf_per_game": 0.0, "ga_per_game": 0.0,
            "btts_yes": 0, "btts_no": 0,
            "over25": 0, "under25": 0,
            "corners_for": 0.0, "corners_against": 0.0,
        }

    def _parse_season_splits(self):
        home = self._parse_last10_splits("home")
        away = self._parse_last10_splits("away")
        return {
            "home_team_home_gf_pg": home["gf_per_game"],
            "home_team_home_ga_pg": home["ga_per_game"],
            "away_team_away_gf_pg": away["gf_per_game"],
            "away_team_away_ga_pg": away["ga_per_game"],
        }

    # -- corners -------------------------------------------------------------

    def _parse_corners(self):
        out = {
            "home": {"corners_for": 0.0, "corners_against": 0.0},
            "away": {"corners_for": 0.0, "corners_against": 0.0},
        }
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
            if side == "home" and len(numbers) >= 4:
                out["home"]["corners_for"] = numbers[2]
                out["home"]["corners_against"] = numbers[3]
            elif side == "away" and len(numbers) >= 6:
                out["away"]["corners_for"] = numbers[4]
                out["away"]["corners_against"] = numbers[5]
        return out

    # -- last 5 form ---------------------------------------------------------

    def _parse_last5_form(self, competition):
        out = {"home_team_points": 0, "away_team_points": 0}
        container = self.soup.select_one("#last-matches #All")
        if not container:
            return out
        left = container.select_one(".teamstats-left")
        right = container.select_one(".teamstats-right")
        out["home_team_points"] = self._sum_points(left, competition)
        out["away_team_points"] = self._sum_points(right, competition)
        return out

    def _sum_points(self, container, competition):
        if not container:
            return 0
        tracked_el = container.select_one(".team-stats-team-name strong")
        tracked = tracked_el.get_text(strip=True).lower() if tracked_el else ""
        if not tracked:
            return 0
        comp_key = self._normalise_competition(competition)
        points = 0
        counted = 0
        for item in container.select("li.team-stat-list-item"):
            date_el = item.select_one(".team-stats-date")
            if not date_el:
                continue
            date_text = date_el.get_text(strip=True)
            comp = date_text.split(":", 1)[0].strip().lower() if ":" in date_text else ""
            if comp_key and comp_key not in comp:
                continue
            teams = item.select(".team-stats-team")
            if len(teams) < 2:
                continue
            home_name = teams[0].get_text(" ", strip=True).lower()
            away_name = teams[1].get_text(" ", strip=True).lower()
            h_score = self._extract_score(teams[0])
            a_score = self._extract_score(teams[1])
            if h_score is None or a_score is None:
                continue
            if self._token_overlap(tracked, home_name):
                if h_score > a_score:
                    points += 3
                elif h_score == a_score:
                    points += 1
            elif self._token_overlap(tracked, away_name):
                if a_score > h_score:
                    points += 3
                elif a_score == h_score:
                    points += 1
            else:
                continue
            counted += 1
            if counted >= 5:
                break
        return points

    @staticmethod
    def _token_overlap(a, b):
        return bool(set(a.split()) & set(b.split()))

    @staticmethod
    def _normalise_competition(name):
        if not name:
            return ""
        name = name.lower()
        if " - " in name:
            name = name.split(" - ", 1)[1]
        return name.strip()

    def _extract_score(self, team_el):
        score_el = team_el.select_one(".score-right")
        if not score_el:
            return None
        return self._to_int(score_el.get_text(strip=True))

    # -- injuries ------------------------------------------------------------

    def _parse_injuries(self):
        out = {
            "home_key_attackers_out": 0,
            "home_key_defenders_out": 0,
            "home_key_midfielders_out": 0,
            "away_key_attackers_out": 0,
            "away_key_defenders_out": 0,
            "away_key_midfielders_out": 0,
        }
        for outline in self.soup.select(".inj-two-outline"):
            header = outline.select_one(".light-header strong")
            if not header:
                continue
            header_text = header.get_text(" ", strip=True).lower()
            side = None
            if self.home_team and self._token_overlap(self.home_team.lower(), header_text):
                side = "home"
            elif self.away_team and self._token_overlap(self.away_team.lower(), header_text):
                side = "away"
            if not side:
                continue
            for row in outline.select(".inj-two-row"):
                if "inj-two-title" in row.get("class", []):
                    continue
                info_el = row.select_one(".inj-two-info")
                info = info_el.get_text(strip=True).lower() if info_el else ""
                if "doubt" in info:
                    continue
                detail = row.find_next_sibling("div", class_="inj-two-hidden")
                position = self._extract_position(detail)
                importance = self._extract_importance(detail)
                if importance != "key" or position is None:
                    continue
                key = f"{side}_key_{position}s_out"
                if key in out:
                    out[key] += 1
        return out

    @staticmethod
    def _extract_position(detail_el):
        if not detail_el:
            return None
        text = detail_el.get_text(" ", strip=True).lower()
        if "forward" in text or "striker" in text or "attacker" in text:
            return "attacker"
        if "defender" in text or "goalkeeper" in text:
            return "defender"
        if "midfielder" in text:
            return "midfielder"
        return None

    @staticmethod
    def _extract_importance(detail_el):
        if not detail_el:
            return "squad"
        text = detail_el.get_text(" ", strip=True)
        matches = 0
        goals = 0
        mm = re.search(r"Matches:\s*(\d+)", text)
        if mm:
            matches = int(mm.group(1))
        gg = re.search(r"Goals:\s*(\d+)", text)
        if gg:
            goals = int(gg.group(1))
        return "key" if (matches >= 3 or goals >= 1) else "squad"

    # -- midweek -------------------------------------------------------------

    def _parse_midweek(self, match_date_iso):
        out = {"home_team_played": False, "away_team_played": False}
        if not match_date_iso:
            return out
        try:
            match_dt = datetime.strptime(match_date_iso, "%Y-%m-%d")
        except ValueError:
            return out
        container = self.soup.select_one("#last-matches #All")
        if not container:
            return out
        left = container.select_one(".teamstats-left")
        right = container.select_one(".teamstats-right")
        out["home_team_played"] = self._played_midweek(left, match_dt)
        out["away_team_played"] = self._played_midweek(right, match_dt)
        return out

    def _played_midweek(self, container, match_dt):
        if not container:
            return False
        first = container.select_one("li.team-stat-list-item")
        if not first:
            return False
        date_el = first.select_one(".team-stats-date")
        if not date_el:
            return False
        mm = re.search(r"(\d{2})/(\d{2})", date_el.get_text(strip=True))
        if not mm:
            return False
        day, month = int(mm.group(1)), int(mm.group(2))
        try:
            last_dt = datetime(match_dt.year, month, day)
        except ValueError:
            return False
        if last_dt > match_dt:
            try:
                last_dt = last_dt.replace(year=match_dt.year - 1)
            except ValueError:
                return False
        return 0 <= (match_dt - last_dt).days <= 4


# ============================================================================
# PARSED -> PREDICTOR INPUT
# ============================================================================

def load_parsed_match(parsed: dict) -> dict:
    odds = parsed.get("odds", {}) or {}
    home_l10 = parsed.get("home_team_last10_home", {}) or {}
    away_l10 = parsed.get("away_team_last10_away", {}) or {}
    season = parsed.get("season_splits", {}) or {}
    form = parsed.get("last5_form", {}) or {}
    inj = parsed.get("injuries", {}) or {}
    midweek = parsed.get("midweek_fixture", {}) or {}

    market_total = derive_market_total(odds.get("over_25"), odds.get("under_25"))

    home_games = max(
        1,
        home_l10.get("wins", 0) + home_l10.get("draws", 0) + home_l10.get("losses", 0),
    )
    away_games = max(
        1,
        away_l10.get("wins", 0) + away_l10.get("draws", 0) + away_l10.get("losses", 0),
    )
    btts_rate = (
        (home_l10.get("btts_yes", 0) / home_games)
        + (away_l10.get("btts_yes", 0) / away_games)
    ) / 2.0

    return {
        "home_team": parsed.get("home_team"),
        "away_team": parsed.get("away_team"),
        "league": parsed.get("competition"),
        "date": parsed.get("match_date") or datetime.now().strftime("%Y-%m-%d"),
        "home_odds": odds.get("home") or 2.0,
        "draw_odds": odds.get("draw") or 3.4,
        "away_odds": odds.get("away") or 3.0,
        "home_xg": home_l10.get("gf_per_game") or 1.2,
        "away_xg": away_l10.get("gf_per_game") or 1.0,
        "market_total": market_total,
        "btts_rate": btts_rate,
        "home_data": {
            "home_goals_scored_season": season.get("home_team_home_gf_pg") or home_l10.get("gf_per_game", 1.2),
            "home_goals_scored_last10": home_l10.get("gf_per_game", 1.2),
            "home_goals_conceded_season": season.get("home_team_home_ga_pg") or home_l10.get("ga_per_game", 1.2),
            "home_goals_conceded_last10": home_l10.get("ga_per_game", 1.2),
            "last5_points": form.get("home_team_points", 7),
            "injuries": _injuries_to_list("home", inj),
            "played_midweek": midweek.get("home_team_played", False),
        },
        "away_data": {
            "away_goals_scored_season": season.get("away_team_away_gf_pg") or away_l10.get("gf_per_game", 1.0),
            "away_goals_scored_last10": away_l10.get("gf_per_game", 1.0),
            "away_goals_conceded_season": season.get("away_team_away_ga_pg") or away_l10.get("ga_per_game", 1.2),
            "away_goals_conceded_last10": away_l10.get("ga_per_game", 1.2),
            "last5_points": form.get("away_team_points", 7),
            "injuries": _injuries_to_list("away", inj),
            "played_midweek": midweek.get("away_team_played", False),
        },
        "odds": {
            "btts_yes_odds": odds.get("btts_yes") or 1.80,
            "btts_no_odds": odds.get("btts_no") or 2.00,
            "over_25_odds": odds.get("over_25") or 1.90,
            "under_25_odds": odds.get("under_25") or 1.90,
            "home_ah_minus_05_odds": odds.get("ah_home"),
            "away_ah_minus_05_odds": odds.get("ah_away"),
            "ah_home_line": odds.get("ah_home_line"),
            "ah_away_line": odds.get("ah_away_line"),
        },
        "corner_data": {
            "home_avg_corners": home_l10.get("corners_for", 0),
            "away_avg_corners": away_l10.get("corners_for", 0),
            "home_conceded_corners": home_l10.get("corners_against", 0),
            "away_conceded_corners": away_l10.get("corners_against", 0),
            "home_corners_line": odds.get("home_corners_line"),
            "home_corners_over": odds.get("home_corners_over"),
            "away_corners_line": odds.get("away_corners_line"),
            "away_corners_over": odds.get("away_corners_over"),
        } if home_l10.get("corners_for") else None,
        "_parsed": parsed,
    }


def _injuries_to_list(side: str, inj: dict) -> list:
    out = []
    for _ in range(inj.get(f"{side}_key_attackers_out", 0)):
        out.append({"position": "forward", "key": True, "confirmed_out": True})
    for _ in range(inj.get(f"{side}_key_defenders_out", 0)):
        out.append({"position": "defender", "key": True, "confirmed_out": True})
    for _ in range(inj.get(f"{side}_key_midfielders_out", 0)):
        out.append({"position": "midfielder", "key": True, "confirmed_out": True})
    return out


def derive_market_total(over_odds, under_odds):
    if not over_odds or not under_odds:
        return None
    p_over = (1.0 / over_odds) / ((1.0 / over_odds) + (1.0 / under_odds))
    lo, hi = 0.1, 6.0
    for _ in range(50):
        mid = (lo + hi) / 2
        p = 1.0 - _poisson_cdf(mid, 2)
        if p < p_over:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _poisson_cdf(lam: float, k: int) -> float:
    total = 0.0
    for i in range(k + 1):
        total += (lam ** i) * math.exp(-lam) / math.factorial(i)
    return total


# ============================================================================
# THE PREDICTOR
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
        self.market_probs = {}
        self.edges = {}
        self.bets = []
        self.skips = []

    def calculate_base_xg(self, home_data, away_data):
        home_attack_home = self._blend(
            home_data.get("home_goals_scored_season", 1.5),
            home_data.get("home_goals_scored_last10", 1.5),
        )
        away_defence_away = self._blend(
            away_data.get("away_goals_conceded_season", 1.5),
            away_data.get("away_goals_conceded_last10", 1.5),
        )
        away_attack_away = self._blend(
            away_data.get("away_goals_scored_season", 1.2),
            away_data.get("away_goals_scored_last10", 1.2),
        )
        home_defence_home = self._blend(
            home_data.get("home_goals_conceded_season", 1.2),
            home_data.get("home_goals_conceded_last10", 1.2),
        )
        self.model_xg_home = max(MIN_XG, (home_attack_home + away_defence_away) / 2)
        self.model_xg_away = max(MIN_XG, (away_attack_away + home_defence_home) / 2)
        self.model_total = self.model_xg_home + self.model_xg_away
        return self.model_xg_home, self.model_xg_away

    @staticmethod
    def _blend(season_val, last10_val):
        if season_val <= 0:
            return max(MIN_XG, last10_val)
        if last10_val <= 0:
            return max(MIN_XG, season_val)
        return LAST10_WEIGHT * last10_val + SEASON_WEIGHT * season_val

    def apply_adjustments(self, home_data, away_data):
        home_adj = self._form_adj(home_data.get("last5_points", 7))
        away_adj = self._form_adj(away_data.get("last5_points", 7))

        for inj in home_data.get("injuries", []):
            if not inj.get("key"):
                continue
            w = 1.0 if inj.get("confirmed_out", True) else 0.5
            if inj["position"] == "forward":
                home_adj -= INJURY_ADJ * w
            elif inj["position"] == "defender":
                away_adj += INJURY_ADJ * w
            elif inj["position"] == "midfielder":
                home_adj -= INJURY_ADJ * w * 0.66

        for inj in away_data.get("injuries", []):
            if not inj.get("key"):
                continue
            w = 1.0 if inj.get("confirmed_out", True) else 0.5
            if inj["position"] == "forward":
                away_adj -= INJURY_ADJ * w
            elif inj["position"] == "defender":
                home_adj += INJURY_ADJ * w
            elif inj["position"] == "midfielder":
                away_adj -= INJURY_ADJ * w * 0.66

        if home_data.get("played_midweek"):
            home_adj -= FATIGUE_HOME
        if away_data.get("played_midweek"):
            away_adj -= FATIGUE_AWAY

        self.model_xg_home = max(MIN_XG, self.model_xg_home + home_adj)
        self.model_xg_away = max(MIN_XG, self.model_xg_away + away_adj)
        self.model_total = self.model_xg_home + self.model_xg_away
        return home_adj, away_adj

    @staticmethod
    def _form_adj(points):
        if points <= 1:
            return -FORM_ADJ_HIGH
        if points <= 3:
            return -FORM_ADJ_MED
        if points >= 13:
            return FORM_ADJ_HIGH
        if points >= 10:
            return FORM_ADJ_MED
        return 0.0

    def shrink_toward_market(self, market_total):
        self.market_total = max(0.5, market_total or self.model_total)
        self.shrunk_total = SHRINK_WEIGHT * self.model_total + (1 - SHRINK_WEIGHT) * self.market_total
        scale = self.shrunk_total / self.model_total if self.model_total > 0 else 1.0
        self.shrunk_xg_home = max(MIN_XG, self.model_xg_home * scale)
        self.shrunk_xg_away = max(MIN_XG, self.model_xg_away * scale)
        self.shrunk_total = self.shrunk_xg_home + self.shrunk_xg_away
        return self.shrunk_xg_home, self.shrunk_xg_away

    def run_poisson(self, max_goals=10):
        home_probs = self._pmf_range(self.shrunk_xg_home, max_goals)
        away_probs = self._pmf_range(self.shrunk_xg_away, max_goals)

        p_home = p_draw = p_away = 0.0
        p_btts_yes = p_btts_no = 0.0
        p_over = p_under = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                if h > a:
                    p_home += prob
                elif h == a:
                    p_draw += prob
                else:
                    p_away += prob
                if h >= 1 and a >= 1:
                    p_btts_yes += prob
                else:
                    p_btts_no += prob
                if h + a > 2.5:
                    p_over += prob
                else:
                    p_under += prob

        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        self.probabilities = {
            "home_win": p_home,
            "draw": p_draw,
            "away_win": p_away,
            "btts_yes": p_btts_yes,
            "btts_no": p_btts_no,
            "over_25": p_over,
            "under_25": p_under,
            "home_ah_plus_05": p_home + p_draw,
            "home_ah_minus_05": p_home,
            "away_ah_plus_05": p_away + p_draw,
            "away_ah_minus_05": p_away,
        }
        return self.probabilities

    @staticmethod
    def _pmf_range(lam, k_max):
        if _has_scipy():
            from scipy.stats import poisson
            return [float(poisson.pmf(i, lam)) for i in range(k_max + 1)]
        return [
            (lam ** i) * math.exp(-lam) / math.factorial(i)
            for i in range(k_max + 1)
        ]

    def calculate_edges(self, odds):
        self.market_probs = {}
        self.edges = {}

        def add(key, prob_key, odd_val):
            if not odd_val or odd_val <= 1.01:
                return
            implied = 1.0 / odd_val
            self.market_probs[key] = implied
            self.edges[key] = self.probabilities.get(prob_key, 0.0) - implied

        add("home_win", "home_win", odds.get("home_odds"))
        add("draw", "draw", odds.get("draw_odds"))
        add("away_win", "away_win", odds.get("away_odds"))
        add("btts_yes", "btts_yes", odds.get("btts_yes_odds"))
        add("btts_no", "btts_no", odds.get("btts_no_odds"))
        add("over_25", "over_25", odds.get("over_25_odds"))
        add("under_25", "under_25", odds.get("under_25_odds"))
        return self.edges

    def select_markets(self, odds, btts_rate=0.5, corner_data=None):
        self.bets = []
        self.skips = []
        self._check_match_result(odds)
        self._check_underdog(odds)
        self._check_btts(odds, btts_rate)
        self._check_ou(odds)
        self._check_corners(corner_data)

        self.skips.append({"market": "Correct Score", "reason": "Never bet — pure lottery"})
        self.skips.append({"market": "First Goalscorer", "reason": "Never bet"})
        self.skips.append({"market": "Anytime Goalscorer", "reason": "Avoid — high variance"})
        return self.bets

    def _check_match_result(self, odds):
        for outcome, label, odds_key, prob_key in [
            ("home_win", "Home", "home_odds", "home_win"),
            ("away_win", "Away", "away_odds", "away_win"),
            ("draw", "Draw", "draw_odds", "draw"),
        ]:
            edge = self.edges.get(outcome)
            if edge is None:
                continue
            prob = self.probabilities.get(prob_key, 0.0)
            if edge > EDGE_MAX:
                self.skips.append({
                    "market": f"Match Result: {label}",
                    "reason": f"Edge {edge:+.1%} > 30% (model error)",
                })
                continue
            if edge < EDGE_MIN:
                self.skips.append({
                    "market": f"Match Result: {label}",
                    "reason": f"Edge {edge:+.1%} < 5%",
                })
                continue
            if outcome == "home_win" and prob >= 0.45:
                self.bets.append({
                    "market": "Match Result",
                    "selection": "Home −0.25 or −0.5 AH",
                    "prob": prob, "edge": edge,
                    "odds": odds.get(odds_key, 0.0),
                    "stake": "1 unit", "confidence": "High",
                })
            elif outcome == "away_win" and prob >= 0.45:
                self.bets.append({
                    "market": "Match Result",
                    "selection": "Away −0.25 or −0.5 AH",
                    "prob": prob, "edge": edge,
                    "odds": odds.get(odds_key, 0.0),
                    "stake": "1 unit", "confidence": "High",
                })
            elif outcome == "draw" and prob >= 0.28:
                self.bets.append({
                    "market": "Match Result",
                    "selection": "Draw or Underdog +0.25",
                    "prob": prob, "edge": edge,
                    "odds": odds.get(odds_key, 0.0),
                    "stake": "1 unit", "confidence": "High",
                })

    def _check_underdog(self, odds):
        home_p = self.probabilities.get("home_win", 0)
        away_p = self.probabilities.get("away_win", 0)
        if home_p < away_p and away_p >= 0.30:
            odd = odds.get("home_odds", 0)
            if odd >= 3.00:
                edge = home_p - 1.0 / odd
                if EDGE_MIN < edge < EDGE_MAX:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Home +0.75 or +1.0 AH",
                        "prob": home_p, "edge": edge,
                        "odds": odd, "stake": "1 unit", "confidence": "High",
                    })
        elif away_p < home_p and home_p >= 0.30:
            odd = odds.get("away_odds", 0)
            if odd >= 3.00:
                edge = away_p - 1.0 / odd
                if EDGE_MIN < edge < EDGE_MAX:
                    self.bets.append({
                        "market": "Match Result",
                        "selection": "Away +0.75 or +1.0 AH",
                        "prob": away_p, "edge": edge,
                        "odds": odd, "stake": "1 unit", "confidence": "High",
                    })

    def _check_btts(self, odds, btts_rate):
        if btts_rate > 0.65:
            edge = self.edges.get("btts_yes")
            if edge is not None and EDGE_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "BTTS",
                    "selection": "BTTS Yes",
                    "prob": self.probabilities["btts_yes"],
                    "edge": edge,
                    "odds": odds.get("btts_yes_odds", 0),
                    "stake": "1 unit", "confidence": "High",
                })
        elif btts_rate < 0.45:
            edge = self.edges.get("btts_no")
            if edge is not None and EDGE_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "BTTS",
                    "selection": "BTTS No",
                    "prob": self.probabilities["btts_no"],
                    "edge": edge,
                    "odds": odds.get("btts_no_odds", 0),
                    "stake": "1 unit", "confidence": "High",
                })

    def _check_ou(self, odds):
        if self.shrunk_total > 3.00:
            edge = self.edges.get("over_25")
            if edge is not None and EDGE_OU_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "Over/Under",
                    "selection": "Over 2.5 (small stake)",
                    "prob": self.probabilities["over_25"],
                    "edge": edge,
                    "odds": odds.get("over_25_odds", 0),
                    "stake": "0.5 units", "confidence": "Selective",
                })
        elif self.shrunk_total < 2.20:
            edge = self.edges.get("under_25")
            if edge is not None and EDGE_OU_MIN <= edge <= EDGE_MAX:
                self.bets.append({
                    "market": "Over/Under",
                    "selection": "Under 2.5 (small stake)",
                    "prob": self.probabilities["under_25"],
                    "edge": edge,
                    "odds": odds.get("under_25_odds", 0),
                    "stake": "0.5 units", "confidence": "Selective",
                })
        else:
            self.skips.append({
                "market": "Over/Under 2.5",
                "reason": f"Shrunk total {self.shrunk_total:.2f} in neutral zone",
            })

    def _check_corners(self, corner_data):
        if not corner_data:
            return
        if corner_data.get("home_avg_corners", 0) >= 5.5 and corner_data.get("away_conceded_corners", 0) >= 5.0:
            self.bets.append({
                "market": "Corners",
                "selection": "Home Over 4.5 corners",
                "prob": 0.55, "edge": 0.05,
                "odds": corner_data.get("home_corners_over") or 0,
                "stake": "0.5 units", "confidence": "Selective",
            })
        if corner_data.get("away_avg_corners", 0) >= 5.5 and corner_data.get("home_conceded_corners", 0) >= 5.0:
            self.bets.append({
                "market": "Corners",
                "selection": "Away Over 4.5 corners",
                "prob": 0.55, "edge": 0.05,
                "odds": corner_data.get("away_corners_over") or 0,
                "stake": "0.5 units", "confidence": "Selective",
            })

    def get_full_analysis(self):
        return {
            "model_xg_home": self.model_xg_home,
            "model_xg_away": self.model_xg_away,
            "model_total": self.model_total,
            "market_total": self.market_total,
            "shrunk_xg_home": self.shrunk_xg_home,
            "shrunk_xg_away": self.shrunk_xg_away,
            "shrunk_total": self.shrunk_total,
            "probabilities": dict(self.probabilities),
            "market_probs": dict(self.market_probs),
            "edges": dict(self.edges),
            "bets": list(self.bets),
            "skips": list(self.skips),
        }


# ============================================================================
# HIGH-LEVEL ORCHESTRATION
# ============================================================================

def analyse_html(html: str):
    """End-to-end: HTML -> parsed -> predictor -> analysis."""
    parsed = SportsgamblerParser(html).parse()
    match = load_parsed_match(parsed)

    p = RefinedPredictor()
    p.calculate_base_xg(match["home_data"], match["away_data"])
    p.apply_adjustments(match["home_data"], match["away_data"])
    p.shrink_toward_market(match["market_total"] or (match["home_xg"] + match["away_xg"]))
    p.run_poisson()

    edge_odds = {
        "home_odds": match["home_odds"],
        "draw_odds": match["draw_odds"],
        "away_odds": match["away_odds"],
        **match["odds"],
    }
    p.calculate_edges(edge_odds)
    p.select_markets(edge_odds, match["btts_rate"], match.get("corner_data"))

    return parsed, match, p.get_full_analysis()
