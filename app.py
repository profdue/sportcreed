"""
Refined Prediction Strategy — single-file Streamlit app.
Wide `matches` schema: one row per match, all selections as columns.

Requires:
  st.secrets["SUPABASE_URL"]
  st.secrets["SUPABASE_KEY"]   (service_role key)

Tables:
  matches       — 1 row per match, ~233 columns
  reliability   — 10 rows, one per market type
"""

import math
import re
import traceback
import base64
import json
from datetime import date, datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Refined Predictor", page_icon="⚽", layout="wide")


# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .team-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.5rem 2rem; color: #fff; margin-bottom: 1rem;
    }
    .team-names { font-size: 2rem; font-weight: 800; margin: 0; }
    .team-meta { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }
    .verdict-bet {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0;
    }
    .verdict-nobet {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0;
    }
    .verdict-label { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #6ee7b7; text-transform: uppercase; }
    .verdict-label-grey { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
    .verdict-pick { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0.35rem 0; line-height: 1.1; }
    .verdict-noedge { font-size: 1.6rem; font-weight: 700; color: #cbd5e1; margin: 0.35rem 0; }
    .verdict-detail { font-size: 1rem; color: #d1fae5; margin-top: 0.5rem; }
    .verdict-detail-grey { font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem; }
    .cand-row { background: #0f172a; border-radius: 10px; padding: 0.75rem 1rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
    .cand-label { color: #cbd5e1; font-weight: 600; font-size: 0.9rem; }
    .cand-score { color: #3b82f6; font-weight: 800; font-size: 1.1rem; }
    .cand-detail { color: #64748b; font-size: 0.75rem; }
    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem 0; }
    .xg-row { background: #0f172a; border-radius: 10px; padding: 0.75rem 1rem; display: flex; justify-content: space-between; margin-bottom: 0.4rem; }
    .xg-team { color: #cbd5e1; font-weight: 600; }
    .xg-value { color: #3b82f6; font-weight: 800; font-size: 1.3rem; }
    .trust-row { background: #0f172a; border-radius: 8px; padding: 0.6rem 1rem; margin-bottom: 0.4rem; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; }
    .trust-value { font-weight: 700; color: #3b82f6; }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; font-weight: 700; border-radius: 10px; border: none; padding: 0.6rem 1.25rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SUPABASE
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    diag = {"url": None, "role": None, "key_prefix": None, "ok": False, "error": None}
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        diag["url"] = url
        diag["key_prefix"] = (key[:15] + "...") if key else None
        try:
            parts = key.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                payload += "=" * (-len(payload) % 4)
                decoded = json.loads(base64.urlsafe_b64decode(payload))
                diag["role"] = decoded.get("role")
            else:
                diag["role"] = "not_jwt"
        except Exception as e:
            diag["role"] = f"decode_error: {e}"
        diag["ok"] = True
        return create_client(url, key), diag
    except Exception as e:
        diag["error"] = str(e)
        return None, diag


# ============================================================================
# CONSTANTS
# ============================================================================
SHRINK_WEIGHT = 0.50
LAST10_WEIGHT = 0.70
SEASON_WEIGHT = 0.30
FORM_ADJ_HIGH = 0.15
FORM_ADJ_MED = 0.10
INJURY_ADJ = 0.15
FATIGUE_HOME = 0.05
FATIGUE_AWAY = 0.10
MIN_XG = 0.10
TRUST_FULL_SAMPLE = 8
TRUST_MIN_WEIGHT = 0.25
MAX_EFFECTIVE_SHRINK = 0.90

MIN_SCORE = 0.01
STAKE_TIER_HIGH = 0.05
STAKE_TIER_MED = 0.02
EDGE_CAP = 0.15
RELIABILITY_PRIOR_STRENGTH = 10


# ============================================================================
# PARSER
# ============================================================================
def _has_bs4():
    try:
        import bs4
        return True
    except ImportError:
        return False


class SportsgamblerParser:
    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError("beautifulsoup4 is not installed.")
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, "html.parser")
        self.home_team = None
        self.away_team = None

    def parse(self) -> dict:
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        competition, venue = self._parse_league_venue()
        return {
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
            "home_current_season_games": self._parse_current_season_games_from_table("home"),
            "away_current_season_games": self._parse_current_season_games_from_table("away"),
            "last_match_dates": self._parse_last_match_dates(),
        }

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
            if text.lower() == "football":
                continue
            if re.search(r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Championship|MLS|Cup|Division|Primera)", text, re.I):
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

    def _parse_odds(self):
        flat = {
            "home": None, "draw": None, "away": None,
            "dc_1x": None, "dc_12": None, "dc_x2": None,
            "dnb_home": None, "dnb_away": None,
            "ah_home_line": None, "ah_home": None,
            "ah_away_line": None, "ah_away": None,
            "btts_yes": None, "btts_no": None,
            "over_2.5": None, "under_2.5": None,
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
        if "half-time result" in m:
            return
        if "double chance" in m:
            for label, val in entries:
                if label == "1X": flat["dc_1x"] = val
                elif label == "12": flat["dc_12"] = val
                elif label == "X2": flat["dc_x2"] = val
            return
        if "draw no bet" in m:
            for label, val in entries:
                if label == "1": flat["dnb_home"] = val
                elif label == "2": flat["dnb_away"] = val
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
        if "total goals" in m:
            for label, val in entries:
                mm = re.match(r"(Over|Under)\s+([\d.]+)", label, re.I)
                if not mm:
                    continue
                line = mm.group(2)
                if line == "2.5":
                    if mm.group(1).lower() == "over":
                        flat["over_2.5"] = val
                    else:
                        flat["under_2.5"] = val
            return
        if "both teams to score" in m:
            for label, val in entries:
                if label.lower() == "yes": flat["btts_yes"] = val
                elif label.lower() == "no": flat["btts_no"] = val
            return

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

    def _parse_last10_splits(self, side):
        split = self._empty_split()
        table = self.soup.select_one(".st-table")
        if table:
            rows = table.select("tbody tr")
            idx = 0 if side == "home" else 1
            if idx < len(rows):
                cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
                if len(cells) >= 8:
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
                    return split
        return self._parse_last10_from_keystats(side)

    def _parse_last10_from_keystats(self, side):
        split = self._empty_split()
        target_id = f"goals-{'hometeam' if side == 'home' else 'awayteam'}"
        block = self.soup.select_one(f"#{target_id}")
        if block:
            text = block.get_text(" ", strip=True)
            m_gf = re.search(r"average of ([\d.]+) goals scored", text)
            m_ga = re.search(r"and ([\d.]+) conceded", text)
            m_o25 = re.search(r"Over 2\.5 Goals in (\d+) of", text)
            m_btts = re.search(r"BTTS Yes in (\d+) of", text)
            if m_gf: split["gf_per_game"] = float(m_gf.group(1))
            if m_ga: split["ga_per_game"] = float(m_ga.group(1))
            if m_o25: split["over25"] = int(m_o25.group(1))
            if m_btts: split["btts_yes"] = int(m_btts.group(1))
            split["under25"] = 10 - split["over25"]
            split["btts_no"] = 10 - split["btts_yes"]
        return split

    @staticmethod
    def _empty_split():
        return {"wins": 0, "draws": 0, "losses": 0, "gf_per_game": 0.0, "ga_per_game": 0.0,
                "btts_yes": 0, "btts_no": 0, "over25": 0, "under25": 0}

    def _parse_season_splits(self):
        home = self._parse_last10_splits("home")
        away = self._parse_last10_splits("away")
        return {
            "home_team_home_gf_pg": home["gf_per_game"],
            "home_team_home_ga_pg": home["ga_per_game"],
            "away_team_away_gf_pg": away["gf_per_game"],
            "away_team_away_ga_pg": away["ga_per_game"],
        }

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
                if h_score > a_score: points += 3
                elif h_score == a_score: points += 1
            elif self._token_overlap(tracked, away_name):
                if a_score > h_score: points += 3
                elif a_score == h_score: points += 1
            else:
                continue
            counted += 1
            if counted >= 5:
                break
        return points

    def _parse_last_match_dates(self):
        out = {"home": None, "away": None}
        container = self.soup.select_one("#last-matches #All")
        if not container:
            return out
        for side, selector in [("home", ".teamstats-left"), ("away", ".teamstats-right")]:
            block = container.select_one(selector)
            if not block:
                continue
            first = block.select_one("li.team-stat-list-item")
            if not first:
                continue
            date_el = first.select_one(".team-stats-date")
            if not date_el:
                continue
            mm = re.search(r"(\d{2})/(\d{2})", date_el.get_text(strip=True))
            if mm:
                try:
                    year = datetime.now().year
                    out[side] = datetime(year, int(mm.group(2)), int(mm.group(1))).strftime("%Y-%m-%d")
                except ValueError:
                    pass
        return out

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

    def _parse_current_season_games_from_table(self, side):
        team = (self.home_team if side == "home" else self.away_team) or ""
        team_lower = team.lower()
        games = None
        for table in self.soup.select("table.leage-table"):
            for row in table.select("tbody tr"):
                cells = row.select("td")
                if len(cells) < 3:
                    continue
                team_cell = cells[1].get_text(strip=True)
                if self._token_overlap(team_lower, team_cell.lower()):
                    try:
                        games = int(cells[2].get_text(strip=True))
                        break
                    except ValueError:
                        continue
            if games is not None:
                break
        if games is not None:
            return games
        return self._count_season_games_fallback(side)

    def _count_season_games_fallback(self, side):
        container = self.soup.select_one("#last-matches #All")
        if not container:
            return 0
        block = container.select_one(".teamstats-left" if side == "home" else ".teamstats-right")
        if not block:
            return 0
        return len(block.select("li.team-stat-list-item"))

    def _parse_injuries(self):
        out = {"home_key_attackers_out": 0, "home_key_defenders_out": 0, "home_key_midfielders_out": 0,
               "away_key_attackers_out": 0, "away_key_defenders_out": 0, "away_key_midfielders_out": 0}
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
                detail = row.find("div", class_="inj-two-hidden")
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
# PREDICTOR
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
        self.candidates = []
        self.bets = []
        self.skips = []
        self.effective_shrink = SHRINK_WEIGHT
        self.home_trust = 1.0
        self.away_trust = 1.0

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
    def _trust_weight(current_season_games):
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
        self.shrunk_total = ((1.0 - self.effective_shrink) * self.model_total
                             + self.effective_shrink * self.market_total)
        scale = self.shrunk_total / self.model_total if self.model_total > 0 else 1.0
        self.shrunk_xg_home = max(MIN_XG, self.model_xg_home * scale)
        self.shrunk_xg_away = max(MIN_XG, self.model_xg_away * scale)
        self.shrunk_total = self.shrunk_xg_home + self.shrunk_xg_away

    def run_poisson(self, max_goals=10):
        home_probs = self._pmf(self.shrunk_xg_home, max_goals)
        away_probs = self._pmf(self.shrunk_xg_away, max_goals)
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
        p_home_by_1 = p_away_by_1 = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                prob = home_probs[h] * away_probs[a]
                if h - a == 1: p_home_by_1 += prob
                elif a - h == 1: p_away_by_1 += prob
        ah_home = {
            -0.5: p_home, -0.25: p_home + 0.5 * p_draw,
            +0.25: p_home + 0.5 * p_draw, +0.5: p_home + p_draw,
            +0.75: p_home + p_draw + 0.5 * p_away_by_1,
            +1.0: p_home + p_draw,
        }
        ah_away = {
            -0.5: p_away, -0.25: p_away + 0.5 * p_draw,
            +0.25: p_away + 0.5 * p_draw, +0.5: p_away + p_draw,
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

    def _effective_ah_prob(self, side, line):
        ah = self.probabilities.get(f"ah_{side}", {})
        candidates = sorted(ah.keys(), key=lambda k: abs(k - line))
        if not candidates:
            return None
        closest = candidates[0]
        if abs(closest - line) > 0.01:
            p_win = self.probabilities.get(f"{side}_win", 0.0)
            p_draw = self.probabilities.get("draw", 0.0)
            if line >= 0.5: return p_win + p_draw
            if line >= 0.25: return p_win + 0.5 * p_draw
            if line >= -0.25: return p_win + 0.5 * p_draw
            return p_win
        return ah[closest]

    def generate_candidates(self, odds):
        cs = []
        P = self.probabilities

        def add(market, selection, line, model_prob, odd, reliability_key, col_key):
            if not odd or odd <= 1.01 or model_prob is None:
                return
            implied = 1.0 / odd
            edge = model_prob - implied
            conviction = abs(model_prob - 0.5) * 2
            cs.append({
                "market": market, "selection": selection, "line": line,
                "model_prob": model_prob, "implied_prob": implied, "edge": edge,
                "conviction": conviction, "odds": odd, "reliability_key": reliability_key,
                "col_key": col_key,
            })

        add("1X2", "Home Win", None, P.get("home_win"), odds.get("home"), "1X2_favourite", "1x2_home")
        add("1X2", "Draw", None, P.get("draw"), odds.get("draw"), "1X2_favourite", "1x2_draw")
        add("1X2", "Away Win", None, P.get("away_win"), odds.get("away"), "1X2_underdog", "1x2_away")
        add("DC", "1X", None, P.get("home_win", 0) + P.get("draw", 0), odds.get("dc_1x"), "DC", "dc_1x")
        add("DC", "12", None, P.get("home_win", 0) + P.get("away_win", 0), odds.get("dc_12"), "DC", "dc_12")
        add("DC", "X2", None, P.get("draw", 0) + P.get("away_win", 0), odds.get("dc_x2"), "DC", "dc_x2")
        p_h = P.get("home_win", 0); p_a = P.get("away_win", 0)
        if p_h + p_a > 0:
            add("DNB", "Home", None, p_h / (p_h + p_a), odds.get("dnb_home"), "DNB", "dnb_home")
            add("DNB", "Away", None, p_a / (p_h + p_a), odds.get("dnb_away"), "DNB", "dnb_away")
        add("BTTS", "Yes", None, P.get("btts_yes"), odds.get("btts_yes"), "BTTS_yes", "btts_yes")
        add("BTTS", "No", None, P.get("btts_no"), odds.get("btts_no"), "BTTS_no", "btts_no")
        add("O/U 2.5", "Over 2.5", 2.5, P.get("over_25"), odds.get("over_2.5"), "O/U_2.5_over", "ou_over")
        add("O/U 2.5", "Under 2.5", 2.5, P.get("under_25"), odds.get("under_2.5"), "O/U_2.5_under", "ou_under")
        ah_h_line = odds.get("ah_home_line")
        if ah_h_line is not None:
            p = self._effective_ah_prob("home", ah_h_line)
            key = "AH_positive" if ah_h_line >= 0 else "AH_negative"
            add("AH", f"Home {ah_h_line:+g}", ah_h_line, p, odds.get("ah_home"), key, "ah_home")
        ah_a_line = odds.get("ah_away_line")
        if ah_a_line is not None:
            p = self._effective_ah_prob("away", ah_a_line)
            key = "AH_positive" if ah_a_line >= 0 else "AH_negative"
            add("AH", f"Away {ah_a_line:+g}", ah_a_line, p, odds.get("ah_away"), key, "ah_away")

        self.candidates = cs

    def score_and_rank(self, reliability):
        for c in self.candidates:
            rel = reliability.get(c["reliability_key"], 0.5)
            edge_capped = min(max(c["edge"], 0.0), EDGE_CAP)
            c["reliability"] = rel
            c["score"] = c["conviction"] * edge_capped * rel
        self.candidates.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(self.candidates, start=1):
            c["rank_in_match"] = i

    def select_top(self):
        self.bets = []; self.skips = []
        if not self.candidates:
            return
        top = self.candidates[0]
        if top["score"] < MIN_SCORE or top["edge"] <= 0:
            self.skips.append({"market": top["market"], "reason": f"Top score {top['score']:.4f} below floor"})
            return
        if top["score"] >= STAKE_TIER_HIGH: stake = "1 unit"
        elif top["score"] >= STAKE_TIER_MED: stake = "0.5 units"
        else: stake = "0.25 units"
        self.bets.append({
            "market": top["market"], "selection": top["selection"],
            "prob": top["model_prob"], "edge": top["edge"], "odds": top["odds"],
            "stake": stake, "confidence": "Ranked", "score": top["score"],
        })
        for c in self.candidates[1:]:
            self.skips.append({
                "market": f"{c['market']} — {c['selection']}",
                "reason": f"Rank {c['rank_in_match']}, score {c['score']:.4f}"
            })

    def get_full_analysis(self):
        return {
            "model_xg_home": self.model_xg_home, "model_xg_away": self.model_xg_away,
            "model_total": self.model_total, "market_total": self.market_total,
            "shrunk_xg_home": self.shrunk_xg_home, "shrunk_xg_away": self.shrunk_xg_away,
            "shrunk_total": self.shrunk_total,
            "probabilities": dict(self.probabilities),
            "bets": list(self.bets), "skips": list(self.skips),
            "candidates": list(self.candidates),
            "effective_shrink": self.effective_shrink,
            "home_trust": self.home_trust, "away_trust": self.away_trust,
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
    lmd = parsed.get("last_match_dates", {}) or {}
    market_total = derive_market_total(odds.get("over_2.5"), odds.get("under_2.5"))
    return {
        "match_id": parsed.get("match_id"),
        "home_team": parsed.get("home_team"),
        "away_team": parsed.get("away_team"),
        "league": parsed.get("competition"),
        "date": parsed.get("match_date") or datetime.now().strftime("%Y-%m-%d"),
        "kickoff": parsed.get("kickoff"),
        "venue": parsed.get("venue"),
        "home_current_games": parsed.get("home_current_season_games", 0),
        "away_current_games": parsed.get("away_current_season_games", 0),
        "home_last_match_date": lmd.get("home"),
        "away_last_match_date": lmd.get("away"),
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
            "home": odds.get("home"), "draw": odds.get("draw"), "away": odds.get("away"),
            "dc_1x": odds.get("dc_1x"), "dc_12": odds.get("dc_12"), "dc_x2": odds.get("dc_x2"),
            "dnb_home": odds.get("dnb_home"), "dnb_away": odds.get("dnb_away"),
            "ah_home_line": odds.get("ah_home_line"), "ah_home": odds.get("ah_home"),
            "ah_away_line": odds.get("ah_away_line"), "ah_away": odds.get("ah_away"),
            "btts_yes": odds.get("btts_yes"), "btts_no": odds.get("btts_no"),
            "over_2.5": odds.get("over_2.5"), "under_2.5": odds.get("under_2.5"),
        },
        "market_total": market_total,
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
    if not over_odds or not under_odds:
        return None
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
# RELIABILITY
# ============================================================================
DEFAULT_RELIABILITY = {
    "AH_positive": 1.0, "AH_negative": 0.7,
    "1X2_favourite": 0.6, "1X2_underdog": 0.8,
    "BTTS_yes": 0.8, "BTTS_no": 0.5,
    "O/U_2.5_over": 0.85, "O/U_2.5_under": 0.7,
    "DC": 0.6, "DNB": 0.6,
}

COL_TO_REL = {
    "1x2_home":  "1X2_favourite",
    "1x2_draw":  "1X2_favourite",
    "1x2_away":  "1X2_underdog",
    "dc_1x":     "DC",
    "dc_12":     "DC",
    "dc_x2":     "DC",
    "dnb_home":  "DNB",
    "dnb_away":  "DNB",
    "btts_yes":  "BTTS_yes",
    "btts_no":   "BTTS_no",
    "ou_over":   "O/U_2.5_over",
    "ou_under":  "O/U_2.5_under",
}

COL_TO_MARKET_SELECTION = {
    "1x2_home":  ("1X2",     "Home Win"),
    "1x2_draw":  ("1X2",     "Draw"),
    "1x2_away":  ("1X2",     "Away Win"),
    "dc_1x":     ("DC",      "1X"),
    "dc_12":     ("DC",      "12"),
    "dc_x2":     ("DC",      "X2"),
    "dnb_home":  ("DNB",     "Home"),
    "dnb_away":  ("DNB",     "Away"),
    "btts_yes":  ("BTTS",    "Yes"),
    "btts_no":   ("BTTS",    "No"),
    "ou_over":   ("O/U 2.5", "Over 2.5"),
    "ou_under":  ("O/U 2.5", "Under 2.5"),
    "ah_home":   ("AH",      None),
    "ah_away":   ("AH",      None),
}

ALL_COL_KEYS = list(COL_TO_MARKET_SELECTION.keys())


def get_reliability(sb):
    out = dict(DEFAULT_RELIABILITY)
    if sb is None:
        return out
    try:
        resp = sb.table("reliability").select("market,weight").execute()
        for row in resp.data or []:
            out[row["market"]] = float(row["weight"])
    except Exception as e:
        st.warning(f"Reliability read failed: {e}")
    return out


def seed_reliability(sb):
    """
    Insert only markets that don't yet exist. Never overwrite weight.
    This preserves the Bayesian-updated weight across app restarts.
    """
    if sb is None:
        return False, "no client"
    try:
        existing_resp = sb.table("reliability").select("market").execute()
        existing = {row["market"] for row in (existing_resp.data or [])}

        new_rows = []
        for k, v in DEFAULT_RELIABILITY.items():
            if k not in existing:
                new_rows.append({
                    "market": k,
                    "weight": v,
                    "prior_weight": v,
                    "prior_strength": RELIABILITY_PRIOR_STRENGTH,
                    "wins": 0,
                    "losses": 0,
                    "pushes": 0,
                    "total": 0,
                })

        if not new_rows:
            return True, f"all {len(existing)} markets already present, nothing seeded"

        sb.table("reliability").insert(new_rows).execute()
        return True, f"{len(new_rows)} new markets seeded"
    except Exception as e:
        return False, str(e)


def settle_candidate(market, selection, line, hg, ag):
    total = hg + ag
    if market == "1X2":
        if selection == "Home Win": return "WON" if hg > ag else "LOST"
        if selection == "Draw": return "WON" if hg == ag else "LOST"
        if selection == "Away Win": return "WON" if hg < ag else "LOST"
    if market == "DC":
        if selection == "1X": return "WON" if hg >= ag else "LOST"
        if selection == "X2": return "WON" if hg <= ag else "LOST"
        if selection == "12": return "WON" if hg != ag else "LOST"
    if market == "DNB":
        if hg == ag: return "PUSH"
        if selection == "Home": return "WON" if hg > ag else "LOST"
        if selection == "Away": return "WON" if hg < ag else "LOST"
    if market == "BTTS":
        yes = (hg >= 1 and ag >= 1)
        if selection == "Yes": return "WON" if yes else "LOST"
        if selection == "No": return "WON" if not yes else "LOST"
    if market == "O/U 2.5":
        if "Over" in selection: return "WON" if total > 2.5 else "LOST"
        if "Under" in selection: return "WON" if total < 2.5 else "LOST"
    if market == "AH":
        m = re.match(r"(Home|Away)\s+([+-]?[\d.]+)", selection)
        if m:
            side = m.group(1).lower()
            hcp = float(m.group(2))
            margin = (hg - ag) if side == "home" else (ag - hg)
            adjusted = margin + hcp
            if adjusted > 0: return "WON"
            if adjusted < 0: return "LOST"
            return "PUSH"
    return None


def update_reliability(sb):
    """
    Read every matches row where outcome columns are populated,
    aggregate WON/LOST/PUSH per reliability key, and upsert weights.
    """
    if sb is None:
        return False, "no client"
    try:
        cols = ["ah_home_line", "ah_away_line"]
        for ck in ALL_COL_KEYS:
            cols.append(f"outcome_{ck}")
        col_csv = ",".join(cols)
        resp = sb.table("matches").select(col_csv).not_.is_("outcome_1x2_home", "null").execute()
        rows = resp.data or []

        groups = {k: {"wins": 0, "losses": 0, "pushes": 0} for k in DEFAULT_RELIABILITY}

        for r in rows:
            ah_h = r.get("ah_home_line")
            ah_a = r.get("ah_away_line")
            for ck, rel_key_static in COL_TO_REL.items():
                outcome = r.get(f"outcome_{ck}")
                if outcome == "WON": groups[rel_key_static]["wins"] += 1
                elif outcome == "LOST": groups[rel_key_static]["losses"] += 1
                elif outcome == "PUSH": groups[rel_key_static]["pushes"] += 1
            if ah_h is not None:
                rel_key = "AH_positive" if float(ah_h) >= 0 else "AH_negative"
                outcome = r.get("outcome_ah_home")
                if outcome == "WON": groups[rel_key]["wins"] += 1
                elif outcome == "LOST": groups[rel_key]["losses"] += 1
                elif outcome == "PUSH": groups[rel_key]["pushes"] += 1
            if ah_a is not None:
                rel_key = "AH_positive" if float(ah_a) >= 0 else "AH_negative"
                outcome = r.get("outcome_ah_away")
                if outcome == "WON": groups[rel_key]["wins"] += 1
                elif outcome == "LOST": groups[rel_key]["losses"] += 1
                elif outcome == "PUSH": groups[rel_key]["pushes"] += 1

        updated = 0
        for key, counts in groups.items():
            prior = DEFAULT_RELIABILITY.get(key, 0.5)
            w_, l_, p_ = counts["wins"], counts["losses"], counts["pushes"]
            total = w_ + l_ + p_
            if total == 0:
                continue
            weight = (w_ + prior * RELIABILITY_PRIOR_STRENGTH) / (total + RELIABILITY_PRIOR_STRENGTH)
            sb.table("reliability").upsert({
                "market": key, "weight": weight,
                "wins": w_, "losses": l_, "pushes": p_, "total": total,
                "prior_weight": prior, "prior_strength": RELIABILITY_PRIOR_STRENGTH,
            }, on_conflict="market").execute()
            updated += 1
        return True, f"{updated} markets updated"
    except Exception as e:
        return False, str(e)


# ============================================================================
# SUPABASE WRITE — wide row
# ============================================================================
def write_match(sb, match, analysis):
    if sb is None:
        return False, "no client"
    try:
        home_ah = match["_parsed"]["home_team_last10_home"]
        away_ah = match["_parsed"]["away_team_last10_away"]
        inj = match["_parsed"]["injuries"]

        rec = {
            "match_id": match["match_id"],
            "match_date": match["date"],
            "kickoff": match.get("kickoff"),
            "league": match.get("league"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "venue": match.get("venue"),
            "home_gf_per_game_last10": match["home_data"].get("home_goals_scored_last10"),
            "home_ga_per_game_last10": match["home_data"].get("home_goals_conceded_last10"),
            "away_gf_per_game_last10": match["away_data"].get("away_goals_scored_last10"),
            "away_ga_per_game_last10": match["away_data"].get("away_goals_conceded_last10"),
            "home_wins_last10": home_ah.get("wins"),
            "home_draws_last10": home_ah.get("draws"),
            "home_losses_last10": home_ah.get("losses"),
            "home_over25_last10": home_ah.get("over25"),
            "home_btts_yes_last10": home_ah.get("btts_yes"),
            "away_wins_last10": away_ah.get("wins"),
            "away_draws_last10": away_ah.get("draws"),
            "away_losses_last10": away_ah.get("losses"),
            "away_over25_last10": away_ah.get("over25"),
            "away_btts_yes_last10": away_ah.get("btts_yes"),
            "home_last5_points": match["home_data"].get("last5_points"),
            "away_last5_points": match["away_data"].get("last5_points"),
            "home_last_match_date": match.get("home_last_match_date"),
            "away_last_match_date": match.get("away_last_match_date"),
            "home_played_midweek": match["home_data"].get("played_midweek"),
            "away_played_midweek": match["away_data"].get("played_midweek"),
            "home_key_attackers_out": inj.get("home_key_attackers_out", 0),
            "home_key_defenders_out": inj.get("home_key_defenders_out", 0),
            "home_key_midfielders_out": inj.get("home_key_midfielders_out", 0),
            "away_key_attackers_out": inj.get("away_key_attackers_out", 0),
            "away_key_defenders_out": inj.get("away_key_defenders_out", 0),
            "away_key_midfielders_out": inj.get("away_key_midfielders_out", 0),
            "home_current_season_games": match.get("home_current_games"),
            "away_current_season_games": match.get("away_current_games"),
            "home_gf_per_game_season": match["home_data"].get("home_goals_scored_season"),
            "home_ga_per_game_season": match["home_data"].get("home_goals_conceded_season"),
            "away_gf_per_game_season": match["away_data"].get("away_goals_scored_season"),
            "away_ga_per_game_season": match["away_data"].get("away_goals_conceded_season"),
            "model_xg_home_raw": analysis["model_xg_home"],
            "model_xg_away_raw": analysis["model_xg_away"],
            "model_xg_home_shrunk": analysis["shrunk_xg_home"],
            "model_xg_away_shrunk": analysis["shrunk_xg_away"],
            "model_total_raw": analysis["model_total"],
            "model_total_shrunk": analysis["shrunk_total"],
            "shrink_weight_applied": analysis["effective_shrink"],
            "market_total": analysis["market_total"],
            "model_prob_home": analysis["probabilities"].get("home_win"),
            "model_prob_draw": analysis["probabilities"].get("draw"),
            "model_prob_away": analysis["probabilities"].get("away_win"),
            "model_prob_btts_yes": analysis["probabilities"].get("btts_yes"),
            "model_prob_over_25": analysis["probabilities"].get("over_25"),
            "model_prob_under_25": analysis["probabilities"].get("under_25"),
        }

        rec["ah_home_line"] = match["odds"].get("ah_home_line")
        rec["ah_away_line"] = match["odds"].get("ah_away_line")
        rec["line_ou_over"] = 2.5
        rec["line_ou_under"] = 2.5

        for ck in ALL_COL_KEYS:
            rec[f"odds_{ck}"] = None
            rec[f"implied_prob_{ck}"] = None
            rec[f"prob_{ck}"] = None
            rec[f"edge_{ck}"] = None
            rec[f"conviction_{ck}"] = None
            rec[f"reliability_{ck}"] = None
            rec[f"score_{ck}"] = None
            rec[f"rank_{ck}"] = None
            rec[f"was_bet_{ck}"] = False
            rec[f"stake_{ck}"] = None
            rec[f"outcome_{ck}"] = None

        for c in analysis["candidates"]:
            ck = c["col_key"]
            rec[f"odds_{ck}"] = c["odds"]
            rec[f"implied_prob_{ck}"] = c["implied_prob"]
            rec[f"prob_{ck}"] = c["model_prob"]
            rec[f"edge_{ck}"] = c["edge"]
            rec[f"conviction_{ck}"] = c["conviction"]
            rec[f"reliability_{ck}"] = c["reliability"]
            rec[f"score_{ck}"] = c["score"]
            rec[f"rank_{ck}"] = c["rank_in_match"]

        if analysis["bets"]:
            top = analysis["bets"][0]
            top_c = next((c for c in analysis["candidates"]
                          if c["market"] == top["market"] and c["selection"] == top["selection"]), None)
            if top_c:
                ck = top_c["col_key"]
                rec[f"was_bet_{ck}"] = True
                m = re.match(r"([\d.]+)", top.get("stake", "0"))
                stake_val = float(m.group(1)) if m else None
                rec[f"stake_{ck}"] = stake_val
                rec["picked_market"] = top["market"]
                rec["picked_selection"] = top["selection"]
                rec["picked_line"] = top_c["line"]
                rec["picked_odds"] = top["odds"]
                rec["picked_model_prob"] = top["prob"]
                rec["picked_implied_prob"] = 1.0 / top["odds"] if top["odds"] else None
                rec["picked_edge"] = top["edge"]
                rec["picked_conviction"] = top_c["conviction"]
                rec["picked_reliability"] = top_c["reliability"]
                rec["picked_score"] = top["score"]
                rec["picked_stake"] = stake_val
                rec["picked_outcome"] = None

        sb.table("matches").upsert(rec, on_conflict="match_id").execute()
        return True, f"match_id={rec['match_id']} ({len(analysis['candidates'])} candidates)"
    except Exception as e:
        return False, str(e)


def record_outcome(sb, match_id, hg, ag):
    if sb is None:
        return False, "no client"
    try:
        resp = sb.table("matches").select(
            "ah_home_line,ah_away_line,picked_market,picked_selection"
        ).eq("match_id", match_id).execute()
        if not resp.data:
            return False, f"match_id {match_id} not found"
        m = resp.data[0]

        updates = {"actual_home_goals": hg, "actual_away_goals": ag}
        ah_h_line = m.get("ah_home_line")
        ah_a_line = m.get("ah_away_line")

        for ck, (market, selection) in COL_TO_MARKET_SELECTION.items():
            if market == "AH":
                if ck == "ah_home":
                    if ah_h_line is None: continue
                    sel = f"Home {float(ah_h_line):+g}"
                else:
                    if ah_a_line is None: continue
                    sel = f"Away {float(ah_a_line):+g}"
            else:
                sel = selection
            outcome = settle_candidate(market, sel, None, hg, ag)
            if outcome:
                updates[f"outcome_{ck}"] = outcome

        pm = m.get("picked_market")
        ps = m.get("picked_selection")
        if pm and ps:
            picked_ck = None
            for ck, (mk, s) in COL_TO_MARKET_SELECTION.items():
                if mk == pm:
                    if mk == "AH":
                        if ck == "ah_home" and ps.startswith("Home"): picked_ck = ck
                        elif ck == "ah_away" and ps.startswith("Away"): picked_ck = ck
                    elif s == ps:
                        picked_ck = ck
            if picked_ck and f"outcome_{picked_ck}" in updates:
                updates["picked_outcome"] = updates[f"outcome_{picked_ck}"]

        sb.table("matches").update(updates).eq("match_id", match_id).execute()
        ok, msg = update_reliability(sb)
        return True, f"settled; reliability: {msg}"
    except Exception as e:
        return False, str(e)


# ============================================================================
# DISPLAY
# ============================================================================
def render_prediction_card(match, parsed, analysis):
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
            <div class="verdict-label">⭐ Primary Pick (Ranked #1)</div>
            <div class="verdict-pick">{primary['selection']}</div>
            <div class="verdict-detail">
                {primary['market']} &nbsp;·&nbsp; @ <strong>{primary['odds']:.2f}</strong>
                &nbsp;·&nbsp; Edge <strong>{primary['edge']:+.1%}</strong>
                &nbsp;·&nbsp; Score <strong>{primary.get('score', 0):.4f}</strong>
                &nbsp;·&nbsp; Stake <strong>{primary['stake']}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">Verdict</div>
            <div class="verdict-noedge">No candidate cleared the ranking floor</div>
            <div class="verdict-detail-grey">Every market scored below MIN_SCORE. Skip this match.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Top 10 Candidates</div>', unsafe_allow_html=True)
    for c in analysis["candidates"][:10]:
        st.markdown(f"""
        <div class="cand-row">
            <div>
                <div class="cand-label">#{c['rank_in_match']} · {c['market']} — {c['selection']}</div>
                <div class="cand-detail">P={c['model_prob']:.1%} · Imp={c['implied_prob']:.1%} · Edge {c['edge']:+.1%} · Rel={c['reliability']:.2f}</div>
            </div>
            <div class="cand-score">{c['score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Sample Quality & Shrinkage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="trust-row"><span>🏠 {match['home_team']} — current-season games</span>
        <span class="trust-value">{match.get('home_current_games', 0)} games · trust {analysis['home_trust']:.0%}</span></div>
    <div class="trust-row"><span>✈️ {match['away_team']} — current-season games</span>
        <span class="trust-value">{match.get('away_current_games', 0)} games · trust {analysis['away_trust']:.0%}</span></div>
    <div class="trust-row"><span>Effective shrinkage toward market</span>
        <span class="trust-value">{analysis['effective_shrink']:.0%}</span></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Expected Goals Model</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="xg-row"><span class="xg-team">🏠 {match["home_team"]} (shrunk)</span><span class="xg-value">{analysis["shrunk_xg_home"]:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="xg-row"><span class="xg-team">✈️ {match["away_team"]} (shrunk)</span><span class="xg-value">{analysis["shrunk_xg_away"]:.2f}</span></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="xg-row"><span class="xg-team">Model Total</span><span class="xg-value">{analysis["model_total"]:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="xg-row"><span class="xg-team">Market Total</span><span class="xg-value" style="color:#fbbf24;">{analysis["market_total"]:.2f}</span></div>', unsafe_allow_html=True)

    with st.expander(f"❌ Skipped candidates ({len(analysis['skips'])})"):
        for s in analysis["skips"]:
            st.write(f"**{s['market']}** — {s['reason']}")


# ============================================================================
# MAIN UI
# ============================================================================
def main():
    st.title("⚽ Refined Prediction Strategy")
    st.caption("xG-based model with ranking selection — wide matches schema")

    get_supabase.clear()
    sb, diag = get_supabase()

    with st.expander("🔍 Supabase connection", expanded=False):
        st.code(f"""
URL:        {diag.get('url')}
Key prefix: {diag.get('key_prefix')}
Connected:  {diag.get('ok')}
Error:      {diag.get('error')}
        """, language="text")

    if sb is None:
        st.info("ℹ️ Supabase not configured — predictions work, persistence disabled.")
    else:
        ok, msg = seed_reliability(sb)
        with st.expander("🔍 Reliability seed", expanded=False):
            st.code(f"ok={ok}\nmsg={msg}", language="text")

    tabs = st.tabs(["⚽ Predict", "📝 Pending", "📊 Records", "🎛️ Reliability"])

    with tabs[0]:
        st.markdown("### Paste Sportsgambler HTML")
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
                            match["market_total"] or (match["home_data"]["home_goals_scored_last10"]
                                                      + match["away_data"]["away_goals_scored_last10"]),
                            home_current_games=match.get("home_current_games", 999),
                            away_current_games=match.get("away_current_games", 999),
                        )
                        p.run_poisson()
                        p.generate_candidates(match["odds"])
                        reliability = get_reliability(sb)
                        p.score_and_rank(reliability)
                        p.select_top()
                        analysis = p.get_full_analysis()

                    st.markdown("---")
                    render_prediction_card(match, parsed, analysis)

                    with st.expander("🔍 Save to Supabase", expanded=True):
                        if sb is None:
                            st.info("Supabase not configured — nothing saved.")
                        else:
                            ok_w, msg_w = write_match(sb, match, analysis)
                            st.code(f"write_match:  ok={ok_w}\n              msg={msg_w}", language="text")
                            if ok_w:
                                st.success("💾 Saved to Supabase (1 row, all columns).")
                            else:
                                st.error("⚠️ Save failed — see message above.")

                except Exception as e:
                    st.error(f"Prediction pipeline failed: {e}")
                    st.code(traceback.format_exc(), language="python")

    with tabs[1]:
        st.subheader("📝 Pending Matches")
        if sb is None:
            st.info("Supabase not configured.")
        else:
            try:
                resp = sb.table("matches").select(
                    "match_id,match_date,home_team,away_team,picked_market,picked_selection,picked_odds"
                ).is_("actual_home_goals", "null").execute()
                pending = resp.data or []
            except Exception as e:
                st.error(f"Query failed: {e}")
                pending = []
            if not pending:
                st.info("No pending matches.")
            for m in pending:
                mid = m["match_id"]
                pm = m.get("picked_market") or "—"
                ps = m.get("picked_selection") or "—"
                po = m.get("picked_odds")
                if po is not None:
                    pick_str = f"{pm} — {ps} @ {po:.2f}"
                else:
                    pick_str = "no bet — ranked list empty"
                with st.expander(f"{m.get('match_date','')} · {m.get('home_team','')} vs {m.get('away_team','')} · {pick_str}"):
                    c1, c2 = st.columns(2)
                    hg = c1.number_input("Home goals", 0, 15, 0, key=f"hg_{mid}")
                    ag = c2.number_input("Away goals", 0, 15, 0, key=f"ag_{mid}")
                    if st.button("Submit result", key=f"sub_{mid}"):
                        ok, msg = record_outcome(sb, mid, hg, ag)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)

    with tabs[2]:
        st.subheader("📊 Performance")
        if sb is None:
            st.info("Supabase not configured.")
        else:
            try:
                resp = sb.table("matches").select(
                    "match_id,match_date,home_team,away_team,"
                    "picked_market,picked_selection,picked_odds,picked_edge,picked_score,"
                    "picked_outcome,actual_home_goals,actual_away_goals"
                ).not_.is_("picked_outcome", "null").execute()
                rows = resp.data or []
            except Exception as e:
                st.error(f"Query failed: {e}")
                rows = []
            if not rows:
                st.info("No settled picks yet.")
            else:
                total = len(rows)
                wins = sum(1 for r in rows if r.get("picked_outcome") == "WON")
                losses = sum(1 for r in rows if r.get("picked_outcome") == "LOST")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Settled picks", total)
                c2.metric("Wins", wins)
                c3.metric("Losses", losses)
                c4.metric("Win rate", f"{wins/(wins+losses)*100:.0f}%" if (wins+losses) else "—")

                df = pd.DataFrame([{
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team','')} vs {r.get('away_team','')}",
                    "Pick": f"{r.get('picked_market','')} — {r.get('picked_selection','')}",
                    "Odds": f"{r.get('picked_odds', 0):.2f}",
                    "Edge": f"{(r.get('picked_edge') or 0):+.1%}",
                    "Score": f"{(r.get('picked_score') or 0):.4f}",
                    "Result": r.get("picked_outcome", ""),
                    "Actual": f"{r.get('actual_home_goals','')}-{r.get('actual_away_goals','')}",
                } for r in rows[:200]])
                st.dataframe(df, use_container_width=True)

    with tabs[3]:
        st.subheader("🎛️ Reliability Weights")
        if sb is None:
            st.info("Supabase not configured.")
        else:
            try:
                resp = sb.table("reliability").select("*").execute()
                rows = resp.data or []
            except Exception as e:
                st.error(f"Query failed: {e}")
                rows = []
            if not rows:
                st.info("No reliability rows. Run a prediction first.")
            else:
                df = pd.DataFrame([{
                    "Market": r["market"],
                    "Weight": f"{r['weight']:.4f}",
                    "Wins": r["wins"],
                    "Losses": r["losses"],
                    "Pushes": r["pushes"],
                    "Total": r["total"],
                    "Prior": r["prior_weight"],
                } for r in rows])
                st.dataframe(df, use_container_width=True)


main()
