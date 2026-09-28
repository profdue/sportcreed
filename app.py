"""
v4 RAW-ONLY Predictor — Streamlit app with HTML parser (FIXED).

Fixes applied:
  Bug A: Team-name matching in standings (fuzzy/last-word fallback)
  Bug B: Team-name matching in last5 (same)
  Bug C: Competition filter loosened (international matches work)
  Bug D: F2 raw capped at 15 before scaling (was uncapped)
  Bug E: Parser reads from #All tab only, never #Home/#Away
  Bug F: Standings table selector covers leage-table AND clt-table
  Bug G: Parser handles both "Stade Rennais FC" and "Rennes" team names

Workflow:
  1. Paste HTML → parse → predict → save to matches_raw (pending)
  2. Enter actual scores → audit updates is_correct_1x2
  3. Performance tab shows accuracy + trigger counts
"""

import os
import re
import json
import math
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="v4 Raw Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    .team-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.5rem 2rem; color: #fff; margin-bottom: 1rem;
    }
    .team-names { font-size: 2rem; font-weight: 800; margin: 0; }
    .team-meta { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }

    .verdict-bet {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981; border-radius: 16px;
        padding: 1.5rem 1.75rem; margin: 1rem 0;
    }
    .verdict-nobet {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b; border-radius: 16px;
        padding: 1.5rem 1.75rem; margin: 1rem 0;
    }
    .verdict-label { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700;
                     color: #6ee7b7; text-transform: uppercase; }
    .verdict-label-grey { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700;
                          color: #94a3b8; text-transform: uppercase; }
    .verdict-pick { font-size: 2.4rem; font-weight: 800; color: #fff;
                    margin: 0.35rem 0; line-height: 1.1; }
    .verdict-noedge { font-size: 1.6rem; font-weight: 700; color: #cbd5e1;
                      margin: 0.35rem 0; }
    .verdict-detail { font-size: 1rem; color: #d1fae5; margin-top: 0.5rem; }
    .verdict-detail-grey { font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem; }

    .factor-row {
        background: #0f172a; border-radius: 10px; padding: 0.75rem 1rem;
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 0.4rem;
    }
    .factor-name { color: #cbd5e1; font-weight: 600; font-size: 0.9rem; }
    .factor-val { color: #3b82f6; font-weight: 800; font-size: 1.1rem; }
    .factor-gap { color: #fbbf24; font-weight: 700; font-size: 0.85rem; }

    .trigger-row {
        background: #1e293b; border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 0.4rem; font-size: 0.85rem; color: #94a3b8;
        display: flex; justify-content: space-between;
    }
    .trigger-on { background: #064e3b; color: #6ee7b7; }
    .trigger-off { background: #1e293b; color: #64748b; }

    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b;
                     text-transform: uppercase; letter-spacing: 1.5px;
                     margin: 1.5rem 0 0.75rem 0; }

    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; font-weight: 700; border-radius: 10px; border: none;
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
        return create_client(url, key), {"ok": True, "url": url}
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


# ============================================================================
# PARSER — FIXED
# ============================================================================
def _has_bs4():
    try:
        import bs4
        return True
    except ImportError:
        return False


class SportsgamblerParser:
    """Parses Sportsgambler HTML → dict of all prematch fields."""

    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError("beautifulsoup4 required. pip install beautifulsoup4")
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, "html.parser")
        self.home_team = None
        self.away_team = None

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _to_float(s):
        try:
            return float(str(s).strip().replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _to_int(s):
        try:
            return int(str(s).strip().replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _norm(s):
        """Normalise team name: lowercase, strip accents, remove common suffixes."""
        if not s:
            return ""
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        # Remove common club suffixes
        for suffix in [" fc", " cf", " afc", " sc", " ac", " club", " city"]:
            if s.endswith(suffix):
                s = s[: -len(suffix)]
        return s.strip()

    def _team_matches(self, full_name: str, table_name: str) -> bool:
        """FIXED: Robust team-name matching.

        Handles:
          - "Stade Rennais FC" vs "Rennes"  → last-word fallback
          - "Paris FC" vs "Paris FC"         → exact normalised match
          - "Manchester United FC" vs "Man United" → token overlap
        """
        if not full_name or not table_name:
            return False
        a = self._norm(full_name)
        b = self._norm(table_name)
        if not a or not b:
            return False

        # 1. Exact normalised match
        if a == b:
            return True

        # 2. Substring match (either direction)
        if a in b or b in a:
            return True

        # 3. Last-word match: "stade rennais" → "rennais" vs "rennes" → no
        #    but "stade rennais" → "rennais" vs "rennes" still no.
        #    So use the LAST word of the longer name and check against shorter.
        a_words = a.split()
        b_words = b.split()
        a_last = a_words[-1] if a_words else ""
        b_last = b_words[-1] if b_words else ""
        if a_last and a_last == b_last:
            return True
        if a_last and a_last in b:
            return True
        if b_last and b_last in a:
            return True

        # 4. Token overlap (at least one meaningful token)
        a_tokens = {w for w in a_words if len(w) > 3}
        b_tokens = {w for w in b_words if len(w) > 3}
        if a_tokens & b_tokens:
            return True

        # 5. Fuzzy: check if the first 4 chars of any word match
        for aw in a_words:
            for bw in b_words:
                if len(aw) >= 4 and len(bw) >= 4 and aw[:4] == bw[:4]:
                    return True

        return False

    # ---------------------------------------------------------------- main
    def parse(self) -> dict:
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        league, tier, group = self._parse_league()
        venue = self._parse_venue()

        record = {
            "match_date": match_date,
            "kickoff_utc": None,
            "league_name": league,
            "tier": tier,
            "group_name": group,
            "season": "2026-27",
            "home_team": self.home_team,
            "away_team": self.away_team,
            "venue": venue,
            "stage": None,
            "round": None,
        }

        record.update(self._parse_standings())
        record["home_last5"] = self._parse_last5("home")
        record["away_last5"] = self._parse_last5("away")
        record.update(self._parse_last10("home"))
        record.update(self._parse_last10("away"))
        record.update(self._parse_players("home"))
        record.update(self._parse_players("away"))
        record["home_injuries"] = self._parse_injuries("home")
        record["away_injuries"] = self._parse_injuries("away")
        record["home_xi"] = self._parse_xi("home")
        record["away_xi"] = self._parse_xi("away")
        record["home_formation"] = self._parse_formation("home")
        record["away_formation"] = self._parse_formation("away")
        record.update(self._parse_h2h())
        record["odds"] = self._parse_odds()

        return record

    # --------------------------------------------------------------- teams
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

    def _parse_league(self):
        league = None
        for link in self.soup.select(".t_top .t_info_link"):
            text = link.get_text(strip=True)
            if text.lower() == "football":
                continue
            if re.search(
                r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Championship|MLS|Cup|Division|Primera|Nations)",
                text, re.I,
            ):
                league = text
                break
        tier = None
        group = None
        if league:
            m = re.search(r"League\s+([A-C])", league)
            if m:
                tier = m.group(1)
            m = re.search(r"Group\s+(\w+)", league)
            if m:
                group = f"Group {m.group(1)}"
        return league, tier, group

    def _parse_venue(self):
        venue_el = self.soup.select_one(".t_top .t_venue")
        return venue_el.get_text(strip=True) if venue_el else None

    # ----------------------------------------------------------- standings
    def _parse_standings(self):
        """FIXED: Uses robust team-name matching and covers both table classes."""
        out = {}
        # Try both table classes
        table = self.soup.select_one("table.leage-table") or self.soup.select_one("table.clt-table")
        if not table:
            return out

        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) < 8:
                continue
            team_cell = cells[1].get_text(strip=True)
            # Strip logo alt text from team cell
            team_cell_clean = re.sub(r"^[A-Za-z\s]+logo\s*", "", team_cell, flags=re.I).strip()
            if not team_cell_clean:
                team_cell_clean = team_cell

            if self._team_matches(self.home_team, team_cell_clean):
                out["home_pos"] = self._to_int(cells[0])
                out["home_played"] = self._to_int(cells[2])
                out["home_points"] = self._to_int(cells[7])
                gf_ga = self._parse_gf_ga(cells[6].get_text(strip=True))
                if gf_ga:
                    out["home_gf"], out["home_ga"] = gf_ga
                    out["home_gd"] = gf_ga[0] - gf_ga[1]
                else:
                    out["home_gd"] = self._parse_gd(cells[6].get_text(strip=True))
            elif self._team_matches(self.away_team, team_cell_clean):
                out["away_pos"] = self._to_int(cells[0])
                out["away_played"] = self._to_int(cells[2])
                out["away_points"] = self._to_int(cells[7])
                gf_ga = self._parse_gf_ga(cells[6].get_text(strip=True))
                if gf_ga:
                    out["away_gf"], out["away_ga"] = gf_ga
                    out["away_gd"] = gf_ga[0] - gf_ga[1]
                else:
                    out["away_gd"] = self._parse_gd(cells[6].get_text(strip=True))
        return out

    @staticmethod
    def _parse_gd(s):
        s = s.strip()
        try:
            return int(s.replace("+", ""))
        except ValueError:
            return None

    @staticmethod
    def _parse_gf_ga(s):
        m = re.match(r"(\d+)\s*:\s*(\d+)", s.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    # ---------------------------------------------------------- form last5
    def _parse_last5(self, side):
        """FIXED: Only reads from #All tab, uses robust team matching."""
        out = []
        container = self.soup.select_one("#last-matches #All")
        if not container:
            container = self.soup.select_one("#last-matches")
        if not container:
            return out

        block = container.select_one(".teamstats-left" if side == "home" else ".teamstats-right")
        if not block:
            return out

        tracked = (self.home_team if side == "home" else self.away_team) or ""

        for item in block.select("li.team-stat-list-item")[:5]:
            date_el = item.select_one(".team-stats-date")
            teams = item.select(".team-stats-team")
            if len(teams) < 2:
                continue

            # Extract names AND scores
            home_name_el = teams[0]
            away_name_el = teams[1]

            # Get text WITHOUT score span
            home_score_el = home_name_el.select_one(".score-right")
            away_score_el = away_name_el.select_one(".score-right")

            home_name = home_name_el.get_text(" ", strip=True)
            away_name = away_name_el.get_text(" ", strip=True)
            if home_score_el:
                home_name = home_name.replace(home_score_el.get_text(strip=True), "").strip()
            if away_score_el:
                away_name = away_name.replace(away_score_el.get_text(strip=True), "").strip()

            h_score = self._to_int(home_score_el.get_text(strip=True)) if home_score_el else None
            a_score = self._to_int(away_score_el.get_text(strip=True)) if away_score_el else None

            if h_score is None or a_score is None:
                continue

            # Determine which side tracked team is on
            tracked_is_home = self._team_matches(tracked, home_name)
            tracked_is_away = self._team_matches(tracked, away_name)

            if tracked_is_home:
                is_home = True
                sf, sa = h_score, a_score
            elif tracked_is_away:
                is_home = False
                sf, sa = a_score, h_score
            else:
                # Fallback: trust block side
                is_home = (side == "home")
                sf, sa = (h_score, a_score) if is_home else (a_score, h_score)

            if sf > sa:
                result = "W"
            elif sf == sa:
                result = "D"
            else:
                result = "L"

            out.append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "opp": away_name if is_home else home_name,
                "result": result,
                "score_for": sf,
                "score_against": sa,
                "is_home": is_home,
            })
        return out

    # --------------------------------------------------------- form last10
    def _parse_last10(self, side):
        out = {}
        prefix = "home" if side == "home" else "away"
        # Last 10 splits table
        table = self.soup.select_one(".st-table")
        if table:
            rows = table.select("tbody tr")
            idx = 0 if side == "home" else 1
            if idx < len(rows):
                cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
                if len(cells) >= 8:
                    mm = re.match(r"(\d+)-(\d+)-(\d+)", cells[1])
                    if mm:
                        out[f"{prefix}_last10_w"] = int(mm.group(1))
                        out[f"{prefix}_last10_d"] = int(mm.group(2))
                        out[f"{prefix}_last10_l"] = int(mm.group(3))
                    out[f"{prefix}_last10_avg_scored"] = self._to_float(cells[3])
                    out[f"{prefix}_last10_avg_conceded"] = self._to_float(cells[4])
                    out[f"{prefix}_last10_win_pct"] = (
                        (out.get(f"{prefix}_last10_w", 0) / 10) * 100
                    )
                    out[f"{prefix}_last10_corners_for"] = self._to_float(cells[5])
                    out[f"{prefix}_last10_corners_against"] = self._to_float(cells[6])

        # Home/away split from keystats
        block = self.soup.select_one(f"#goals-{'hometeam' if side == 'home' else 'awayteam'}")
        if block:
            text = block.get_text(" ", strip=True)
            venue = "home" if side == "home" else "away"
            m = re.search(
                rf"average of ([\d.]+) goals scored and ([\d.]+) conceded "
                rf"in the previous 10 {venue} matches",
                text,
            )
            if m:
                out[f"{prefix}_{venue}_last10_avg_scored"] = float(m.group(1))
                out[f"{prefix}_{venue}_last10_avg_conceded"] = float(m.group(2))

        # Win % at venue
        w_key = f"{prefix}_last10_w"
        out[f"{prefix}_{'home' if side == 'home' else 'away'}_win_pct"] = (
            (out.get(w_key, 0) / 10) * 100 if out.get(w_key) else 50.0
        )
        return out

    # ---------------------------------------------------------- players
    def _parse_players(self, side):
        """Parse top scorer and top assister from keystats."""
        out = {}
        prefix = "home" if side == "home" else "away"
        block_id = f"#{'ht' if side == 'home' else 'at'}-goalassist"
        block = self.soup.select_one(block_id)
        if not block:
            return out
        text = block.get_text(" ", strip=True)
        # Top scorer: "Top Scorers for X this season are NAME (N), ..."
        m = re.search(r"Top Scorers for .+? this season are (.+?)(?:\.|$)", text)
        if m:
            first = m.group(1).split(",")[0].strip()
            mm = re.match(r"(.+?)\s*\((\d+)\)", first)
            if mm:
                out[f"{prefix}_top_scorer"] = mm.group(1).strip()
                out[f"{prefix}_top_scorer_goals"] = int(mm.group(2))
        # Top assister
        m = re.search(r"Top Assistors for .+? this season are (.+?)(?:\.|$)", text)
        if m:
            first = m.group(1).split(",")[0].strip()
            mm = re.match(r"(.+?)\s*\((\d+)\)", first)
            if mm:
                out[f"{prefix}_top_assister"] = mm.group(1).strip()
                out[f"{prefix}_top_assister_assists"] = int(mm.group(2))
        return out

    def _parse_injuries(self, side):
        out = []
        target = self.home_team if side == "home" else self.away_team
        for outline in self.soup.select(".inj-two-outline"):
            header = outline.select_one(".light-header strong")
            if not header:
                continue
            header_text = header.get_text(" ", strip=True)
            if not self._team_matches(target, header_text):
                continue
            for row in outline.select(".inj-two-row"):
                if "inj-two-title" in row.get("class", []):
                    continue
                player_el = row.select_one(".inj-two-player")
                info_el = row.select_one(".inj-two-info")
                if not player_el:
                    continue
                player = player_el.get_text(strip=True)
                info = info_el.get_text(strip=True) if info_el else ""
                status = "injury"
                low = info.lower()
                if "doubt" in low:
                    status = "doubt"
                elif "suspension" in low or "yellow" in low or "red" in low:
                    status = "suspended"
                out.append({
                    "player": player,
                    "status": status,
                    "type": info,
                    "expected_return": None,
                })
        return out

    def _parse_xi(self, side):
        out = []
        target = self.home_team if side == "home" else self.away_team
        # Find the lineup heading
        for header in self.soup.select(".lineups-formation h3, .lineups-mob-teams h3"):
            text = header.get_text(" ", strip=True)
            if not self._team_matches(target, text):
                continue
            # Get associated lineups block
            parent = header.find_parent()
            container = parent.find_next_sibling(class_="lineups") if parent else None
            if not container:
                container = self.soup.select_one(".lineups")
            if not container:
                continue
            side_class = "lineups-home" if side == "home" else "lineups-away"
            block = container.select_one(f".{side_class}")
            if not block:
                continue
            for player in block.select(".lineups-player .player-name"):
                name = player.get_text(strip=True)
                if name:
                    out.append(name)
            if out:
                return out
        return out

    def _parse_formation(self, side):
        target = self.home_team if side == "home" else self.away_team
        for header in self.soup.select(".lineups-formation h3"):
            text = header.get_text(" ", strip=True)
            if self._team_matches(target, text):
                m = re.search(r"(\d-\d-\d(?:-\d)?)", text)
                if m:
                    return m.group(1)
        return None

    # --------------------------------------------------------------- h2h
    def _parse_h2h(self):
        out = {"h2h": [], "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0}
        container = self.soup.select_one("#head-to-head")
        if not container:
            return out
        for item in container.select("li.team-stat-list-item"):
            date_el = item.select_one(".team-stats-date")
            teams = item.select(".team-stats-team")
            if len(teams) < 2:
                continue
            h_name_el = teams[0]
            a_name_el = teams[1]
            h_score_el = h_name_el.select_one(".score-right")
            a_score_el = a_name_el.select_one(".score-right")
            h_name = h_name_el.get_text(" ", strip=True)
            a_name = a_name_el.get_text(" ", strip=True)
            if h_score_el:
                h_name = h_name.replace(h_score_el.get_text(strip=True), "").strip()
            if a_score_el:
                a_name = a_name.replace(a_score_el.get_text(strip=True), "").strip()
            h_score = self._to_int(h_score_el.get_text(strip=True)) if h_score_el else None
            a_score = self._to_int(a_score_el.get_text(strip=True)) if a_score_el else None
            if h_score is None or a_score is None:
                continue
            h_is_home = self._team_matches(self.home_team, h_name)
            a_is_away = self._team_matches(self.away_team, a_name)
            if h_is_home and a_is_away:
                home_goals, away_goals = h_score, a_score
            elif self._team_matches(self.home_team, a_name) and self._team_matches(self.away_team, h_name):
                home_goals, away_goals = a_score, h_score
            else:
                home_goals, away_goals = h_score, a_score

            winner = "home" if home_goals > away_goals else ("away" if away_goals > home_goals else "draw")
            out["h2h"].append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "home": h_name,
                "away": a_name,
                "score": f"{h_score}-{a_score}",
                "winner": winner,
            })
            if winner == "home":
                out["h2h_home_wins"] += 1
            elif winner == "away":
                out["h2h_away_wins"] += 1
            else:
                out["h2h_draws"] += 1
        return out

    # ------------------------------------------------------------- odds
    def _parse_odds(self):
        odds = {}
        for row in self.soup.select(".nlf_odds_row"):
            title_el = row.select_one(".nfl_odd_title")
            if not title_el:
                continue
            market = title_el.get_text(strip=True).lower()
            entries = {}
            for ply in row.select(".nfl_odply"):
                label_el = ply.select_one(".nfl_ply_t")
                odds_el = ply.select_one(".nfl_ply_o")
                if label_el and odds_el:
                    entries[label_el.get_text(strip=True)] = self._to_float(odds_el.get_text(strip=True))
            if "full-time result" in market:
                odds["home"] = entries.get("1")
                odds["draw"] = entries.get("X")
                odds["away"] = entries.get("2")
            elif "both teams to score" in market:
                odds["btts_yes"] = entries.get("Yes")
                odds["btts_no"] = entries.get("No")
            elif "total goals" in market:
                odds["over_2.5"] = entries.get("Over 2.5")
                odds["under_2.5"] = entries.get("Under 2.5")
        return odds


# ============================================================================
# v4 MODEL — FIXED
# ============================================================================
def calc_f1(row):
    home_pts = row.get("home_points") or 0
    away_pts = row.get("away_points") or 0
    home_gd = row.get("home_gd") or 0
    away_gd = row.get("away_gd") or 0

    pts_gap = home_pts - away_pts
    gd_gap = home_gd - away_gd

    f1_home = 10 + (pts_gap * 1.33) + (gd_gap * 0.66)
    f1_away = 10 - (pts_gap * 1.33) - (gd_gap * 0.66)
    f1_home = max(0, min(20, f1_home))
    f1_away = max(0, min(20, f1_away))

    away_collapse = False
    if (row.get("away_away_points") == 0
            and (row.get("away_away_played") or 0) >= 3):
        f1_home = min(20, f1_home + 8)
        away_collapse = True

    return f1_home, f1_away, away_collapse


def calc_f2(last5):
    """FIXED: raw capped at 15 before scaling to 25."""
    if not last5:
        return 7.5
    raw = 0
    for m in last5:
        r = m.get("result", "L")
        if r == "W":
            raw += 5
            if (m.get("score_for") or 0) >= 3:
                raw += 1
        elif r == "D":
            raw += 2
        else:
            if (m.get("score_against") or 0) >= 3:
                raw -= 1
    raw = max(0, min(15, raw))  # FIX: cap at 15
    return (raw / 15) * 25


def calc_f3(win_pct, avg_scored, avg_conceded):
    win_pts = ((win_pct or 0) / 100) * 10
    edge = (avg_scored or 0) - (avg_conceded or 0)
    xg_pts = 5 if edge > 0.5 else (2 if edge > 0 else 0)
    return min(15, win_pts + xg_pts)


def calc_f4(top_scorer, top_assister, injuries, xi):
    f4 = 15
    injured = [i.get("player", "") for i in (injuries or [])]
    doubts = [i.get("player", "") for i in (injuries or []) if i.get("status") == "doubt"]
    xi = xi or []

    if top_scorer and top_scorer in injured and top_scorer not in xi:
        f4 -= 5
    if top_assister and top_assister in injured and top_assister not in xi:
        f4 -= 4

    doubted_starter = False
    for name in doubts:
        if name in xi and (name == top_scorer or name == top_assister):
            f4 += 5
            doubted_starter = True

    return max(0, min(20, f4)), doubted_starter


def calc_f5(h2h_home_wins, h2h_away_wins, h2h_total):
    if not h2h_total:
        return 5.0, 5.0
    return (h2h_home_wins / h2h_total) * 10, (h2h_away_wins / h2h_total) * 10


def calc_f6(h_avg, a_avg):
    if (h_avg or 0) > (a_avg or 0):
        return 11, 7
    elif (h_avg or 0) < (a_avg or 0):
        return 7, 11
    return 8, 8


def apply_override(f1_home, f1_away, f5_home, f5_away):
    f1_gap = abs(f1_home - f1_away)
    f5_gap = abs(f5_home - f5_away)
    f1_leader = "home" if f1_home > f1_away else "away"
    f5_leader = "home" if f5_home > f5_away else "away"

    conflict = (f1_leader != f5_leader and f1_gap >= 12 and f5_gap >= 6)
    override = False
    if conflict and f1_gap > f5_gap:
        if f1_leader == "home":
            f5_home, f5_away = 10, 0
        else:
            f5_home, f5_away = 0, 10
        override = True
    return f5_home, f5_away, conflict, override


def calc_expected_total(row):
    h1 = row.get("home_home_last10_avg_scored") or 1.0
    a1 = row.get("away_away_last10_avg_conceded") or 1.0
    a2 = row.get("away_away_last10_avg_scored") or 1.0
    h2 = row.get("home_home_last10_avg_conceded") or 1.0
    return (h1 + a1 + a2 + h2) / 2


def predict_v4(row):
    f1_home, f1_away, away_collapse = calc_f1(row)
    f2_home = calc_f2(row.get("home_last5"))
    f2_away = calc_f2(row.get("away_last5"))
    f3_home = calc_f3(row.get("home_home_win_pct"),
                      row.get("home_home_last10_avg_scored"),
                      row.get("home_home_last10_avg_conceded"))
    f3_away = calc_f3(row.get("away_away_win_pct"),
                      row.get("away_away_last10_avg_scored"),
                      row.get("away_away_last10_avg_conceded"))
    f4_home, doubt_h = calc_f4(row.get("home_top_scorer"), row.get("home_top_assister"),
                                row.get("home_injuries"), row.get("home_xi"))
    f4_away, doubt_a = calc_f4(row.get("away_top_scorer"), row.get("away_top_assister"),
                                row.get("away_injuries"), row.get("away_xi"))
    doubted_starter = doubt_h or doubt_a
    f5_home, f5_away = calc_f5(row.get("h2h_home_wins") or 0,
                                row.get("h2h_away_wins") or 0,
                                len(row.get("h2h") or []))
    f6_home, f6_away = calc_f6(row.get("home_last10_avg_scored"),
                                row.get("away_last10_avg_scored"))

    f5_home, f5_away, conflict, override = apply_override(f1_home, f1_away, f5_home, f5_away)

    home_total = f1_home + f2_home + f3_home + f4_home + f5_home + f6_home
    away_total = f1_away + f2_away + f3_away + f4_away + f5_away + f6_away
    gap = abs(home_total - away_total)

    if gap > 25:
        call_1x2 = "Straight Win Home" if home_total > away_total else "Straight Win Away"
    elif gap >= 15:
        call_1x2 = "Double Chance 1X" if home_total > away_total else "Double Chance X2"
    else:
        call_1x2 = "NO BET"

    expected = calc_expected_total(row)
    if expected < 2.5:
        call_ou = "Under 2.5"
    elif expected > 3.3:
        call_ou = "Over 2.5"
    else:
        call_ou = "No Bet"

    if away_collapse or doubted_starter:
        call_ou = "Over 2.5"

    return {
        "f1_home": round(f1_home, 2), "f1_away": round(f1_away, 2),
        "f1_gap": round(abs(f1_home - f1_away), 2),
        "f1_leader": "home" if f1_home > f1_away else "away",
        "f2_home": round(f2_home, 2), "f2_away": round(f2_away, 2),
        "f3_home": round(f3_home, 2), "f3_away": round(f3_away, 2),
        "f4_home": round(f4_home, 2), "f4_away": round(f4_away, 2),
        "f5_home": round(f5_home, 2), "f5_away": round(f5_away, 2),
        "f5_gap": round(abs(f5_home - f5_away), 2),
        "f5_leader": "home" if f5_home > f5_away else "away",
        "f6_home": round(f6_home, 2), "f6_away": round(f6_away, 2),
        "home_total": round(home_total, 2),
        "away_total": round(away_total, 2),
        "total_gap": round(gap, 2),
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "away_collapse": away_collapse,
        "doubted_starter": doubted_starter,
        "call_1x2": call_1x2,
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
    }


# ============================================================================
# DB HELPERS
# ============================================================================
def upsert_match(sb, record):
    if sb is None:
        return False, "no client"
    try:
        # Extract only columns that exist in matches_raw (strip extras)
        allowed = {
            "match_date", "kickoff_utc", "league_name", "tier", "group_name",
            "season", "home_team", "away_team", "venue", "stage", "round",
            "home_pos", "home_played", "home_points", "home_gd", "home_gf", "home_ga",
            "away_pos", "away_played", "away_points", "away_gd", "away_gf", "away_ga",
            "home_away_points", "home_away_played", "away_away_points", "away_away_played",
            "home_last5", "away_last5",
            "home_last10_w", "home_last10_d", "home_last10_l",
            "home_last10_avg_scored", "home_last10_avg_conceded",
            "home_last10_possession", "home_last10_corners_for",
            "home_last10_corners_against", "home_last10_win_pct",
            "home_home_last10_avg_scored", "home_home_last10_avg_conceded",
            "home_home_last10_corners_for", "home_home_last10_corners_against",
            "home_home_win_pct",
            "away_last10_w", "away_last10_d", "away_last10_l",
            "away_last10_avg_scored", "away_last10_avg_conceded",
            "away_last10_possession", "away_last10_corners_for",
            "away_last10_corners_against", "away_last10_win_pct",
            "away_away_last10_avg_scored", "away_away_last10_avg_conceded",
            "away_away_last10_corners_for", "away_away_last10_corners_against",
            "away_away_win_pct",
            "home_top_scorer", "home_top_scorer_goals",
            "home_top_assister", "home_top_assister_assists",
            "away_top_scorer", "away_top_scorer_goals",
            "away_top_assister", "away_top_assister_assists",
            "home_injuries", "away_injuries",
            "home_xi", "away_xi", "home_formation", "away_formation",
            "h2h", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
            "odds",
        }
        clean = {k: v for k, v in record.items() if k in allowed}
        # JSONB fields must be serialised
        for jsonb_field in ["home_last5", "away_last5", "home_injuries",
                            "away_injuries", "h2h", "odds"]:
            if jsonb_field in clean and clean[jsonb_field] is None:
                clean[jsonb_field] = []
        # text[] fields
        for arr_field in ["home_xi", "away_xi"]:
            if arr_field in clean and clean[arr_field] is None:
                clean[arr_field] = []

        resp = sb.table("matches_raw").upsert(
            clean, on_conflict="match_date,home_team,away_team"
        ).execute()
        row = resp.data[0] if resp.data else None
        return True, row
    except Exception as e:
        return False, str(e)


def save_prediction(sb, match_id, result):
    if sb is None:
        return False, "no client"
    try:
        sb.table("matches_raw").update(result).eq("id", match_id).execute()
        return True, "saved"
    except Exception as e:
        return False, str(e)


def load_all(sb):
    if sb is None:
        return []
    try:
        resp = sb.table("matches_raw").select("*").order("match_date", desc=True).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"Load failed: {e}")
        return []


def update_audit(sb, match_id, hg, ag, call_1x2):
    if sb is None:
        return False, "no client"
    if hg > ag:
        actual = "Home"
    elif hg < ag:
        actual = "Away"
    else:
        actual = "Draw"

    is_correct = None
    if call_1x2 == "NO BET":
        is_correct = None
    elif "Straight Win Home" in call_1x2 and actual == "Home":
        is_correct = True
    elif "Straight Win Away" in call_1x2 and actual == "Away":
        is_correct = True
    elif "Double Chance 1X" in call_1x2 and actual in ("Home", "Draw"):
        is_correct = True
    elif "Double Chance X2" in call_1x2 and actual in ("Away", "Draw"):
        is_correct = True
    else:
        is_correct = False

    try:
        sb.table("matches_raw").update({
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "is_correct_1x2": is_correct,
        }).eq("id", match_id).execute()
        return True, "ok"
    except Exception as e:
        return False, str(e)


# ============================================================================
# DISPLAY COMPONENTS
# ============================================================================
def render_factor_row(name, home_val, away_val, weight_label=""):
    gap = abs((home_val or 0) - (away_val or 0))
    st.markdown(f"""
    <div class="factor-row">
        <div>
            <div class="factor-name">{name} {weight_label}</div>
            <div style="color:#64748b; font-size:0.75rem;">
                Home: {home_val:.2f} &nbsp;|&nbsp; Away: {away_val:.2f} &nbsp;|&nbsp; Gap: {gap:.2f}
            </div>
        </div>
        <div class="factor-val">{gap:.1f}</div>
    </div>
    """, unsafe_allow_html=True)


def render_trigger(name, on):
    cls = "trigger-on" if on else "trigger-off"
    icon = "✅" if on else "○"
    st.markdown(f"""
    <div class="trigger-row {cls}">
        <span>{icon} {name}</span>
        <span>{'TRIGGERED' if on else 'no'}</span>
    </div>
    """, unsafe_allow_html=True)


def render_verdict(result, home_team, away_team):
    call = result.get("call_1x2", "")
    if call == "NO BET":
        st.markdown(f"""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">1X2 Verdict</div>
            <div class="verdict-noedge">NO BET — gap {result['total_gap']:.1f} below 15</div>
            <div class="verdict-detail-grey">
                Total gap: {result['total_gap']:.1f} (needs 15+ for a call)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-bet">
            <div class="verdict-label">⭐ 1X2 Verdict</div>
            <div class="verdict-pick">{call}</div>
            <div class="verdict-detail">
                Gap <strong>{result['total_gap']:.1f}</strong>
                &nbsp;·&nbsp; Home {result['home_total']:.1f}
                &nbsp;·&nbsp; Away {result['away_total']:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_ou_verdict(result):
    call = result.get("call_ou", "")
    expected = result.get("expected_total", 0)
    if call == "No Bet":
        st.info(f"🟡 O/U 2.5: **NO BET** — expected total {expected:.2f} (dead zone 2.5–3.3)")
    else:
        st.success(f"🟢 O/U 2.5: **{call}** — expected total {expected:.2f}")


# ============================================================================
# UI
# ============================================================================
def main():
    st.title("⚽ v4 Raw Predictor")
    st.caption("100-point model · F1–F6 prematch logic · F1 vs F5 override · Pending workflow")

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase connection failed: {diag.get('error')}")
        return

    tabs = st.tabs(["📥 Parse HTML", "⏳ Pending", "📊 Performance", "ℹ️ v4 Spec"])

    # --------------------------------------------------------- parse html
    with tabs[0]:
        st.subheader("Paste Sportsgambler HTML")
        st.caption("Copy the full preview page source. The parser extracts every prematch field.")

        text = st.text_area("HTML", height=260, key="html_input", label_visibility="collapsed")

        col_a, col_b = st.columns([1, 4])
        with col_a:
            parse_btn = st.button("⚽ Parse & Predict", type="primary")

        if parse_btn:
            if not text or len(text.strip()) < 200:
                st.error("Paste a full Sportsgambler preview page.")
            else:
                with st.spinner("Parsing..."):
                    try:
                        parsed = SportsgamblerParser(text).parse()
                    except Exception as e:
                        st.error(f"Parser failed: {e}")
                        return

                    if not parsed.get("home_team") or not parsed.get("away_team"):
                        st.error("Could not extract team names from HTML.")
                        return

                    st.markdown("---")
                    st.markdown(f"""
                    <div class="team-header">
                        <div class="team-names">{parsed['home_team']} &nbsp;🆚&nbsp; {parsed['away_team']}</div>
                        <div class="team-meta">
                            {parsed.get('league_name') or '—'}
                            &nbsp;·&nbsp; {parsed.get('match_date') or '—'}
                            &nbsp;·&nbsp; {parsed.get('venue') or '—'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Show parsed raw values for verification
                    with st.expander("🔍 Parsed raw values (verify before saving)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.write("**Home**")
                            st.write(f"Pos: {parsed.get('home_pos')}, Pts: {parsed.get('home_points')}, GD: {parsed.get('home_gd')}")
                            st.write(f"Last5: {parsed.get('home_last5')}")
                            st.write(f"Last10 avg scored: {parsed.get('home_last10_avg_scored')}, conceded: {parsed.get('home_last10_avg_conceded')}")
                            st.write(f"XI: {parsed.get('home_xi')}")
                            st.write(f"Injuries: {parsed.get('home_injuries')}")
                        with c2:
                            st.write("**Away**")
                            st.write(f"Pos: {parsed.get('away_pos')}, Pts: {parsed.get('away_points')}, GD: {parsed.get('away_gd')}")
                            st.write(f"Last5: {parsed.get('away_last5')}")
                            st.write(f"Last10 avg scored: {parsed.get('away_last10_avg_scored')}, conceded: {parsed.get('away_last10_avg_conceded')}")
                            st.write(f"XI: {parsed.get('away_xi')}")
                            st.write(f"Injuries: {parsed.get('away_injuries')}")
                        st.write(f"H2H: {parsed.get('h2h')}")

                    result = predict_v4(parsed)

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        render_verdict(result, parsed["home_team"], parsed["away_team"])
                    with c2:
                        render_ou_verdict(result)

                    st.markdown('<div class="section-title">Factor Breakdown</div>', unsafe_allow_html=True)
                    render_factor_row("F1 League Momentum", result["f1_home"], result["f1_away"], "(20 pts)")
                    render_factor_row("F2 Current Form", result["f2_home"], result["f2_away"], "(25 pts)")
                    render_factor_row("F3 Venue Split", result["f3_home"], result["f3_away"], "(15 pts)")
                    render_factor_row("F4 Availability", result["f4_home"], result["f4_away"], "(15 pts)")
                    render_factor_row("F5 H2H Psychology", result["f5_home"], result["f5_away"], "(10 pts)")
                    render_factor_row("F6 Attack Profile", result["f6_home"], result["f6_away"], "(15 pts)")

                    st.markdown(f"""
                    <div class="factor-row" style="background:#1e3a8a;">
                        <div>
                            <div class="factor-name" style="color:#bfdbfe;">TOTAL</div>
                            <div style="color:#93c5fd; font-size:0.8rem;">
                                Home {result['home_total']:.1f} · Away {result['away_total']:.1f}
                                · Gap {result['total_gap']:.1f}
                            </div>
                        </div>
                        <div class="factor-val" style="color:#bfdbfe;">{result['total_gap']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="section-title">Modifiers & Overrides</div>', unsafe_allow_html=True)
                    render_trigger("F1 vs F5 Conflict", result["f1_f5_conflict"])
                    render_trigger("F1 vs F5 Override Applied", result["f1_f5_override"])
                    render_trigger("Away Collapse", result["away_collapse"])
                    render_trigger("Doubt Starter IN XI", result["doubted_starter"])

                    st.markdown('<div class="section-title">Save to Supabase (Pending)</div>', unsafe_allow_html=True)
                    if st.button("💾 Save match + prediction"):
                        ok, row = upsert_match(sb, parsed)
                        if not ok:
                            st.error(f"Save failed: {row}")
                        else:
                            match_id = row["id"] if row else None
                            if match_id:
                                ok2, msg2 = save_prediction(sb, match_id, result)
                                if ok2:
                                    st.success(f"✅ Saved as PENDING. match_id={match_id}")
                                    st.info("Go to ⏳ Pending tab after the match to enter the actual score.")
                                else:
                                    st.warning(f"Saved row but prediction update failed: {msg2}")

    # ------------------------------------------------------------- pending
    with tabs[1]:
        st.subheader("⏳ Pending Matches")
        st.caption("Matches with predictions but no actual score yet.")

        rows = load_all(sb)
        pending = [r for r in rows if r.get("actual_home_goals") is None]

        if not pending:
            st.success("No pending matches. All predictions have been settled.")
        else:
            st.write(f"**{len(pending)} pending matches**")

            for r in pending:
                match_id = r["id"]
                call = r.get("call_1x2", "—")
                ou = r.get("call_ou", "—")
                gap = r.get("total_gap", 0)
                triggers = []
                if r.get("f1_f5_override"):
                    triggers.append("override")
                if r.get("away_collapse"):
                    triggers.append("collapse")
                if r.get("doubted_starter"):
                    triggers.append("doubt")
                trigger_str = f" · {', '.join(triggers)}" if triggers else ""

                header = f"{r.get('match_date','')} · {r.get('home_team','')} vs {r.get('away_team','')} · {call} (gap {gap}){trigger_str}"
                with st.expander(header):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("1X2 Call", call)
                    c2.metric("O/U Call", ou)
                    c3.metric("Gap", f"{gap:.1f}")

                    st.markdown("**Enter actual score:**")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    hg = col1.number_input("Home goals", 0, 15, 0, key=f"hg_{match_id}")
                    ag = col2.number_input("Away goals", 0, 15, 0, key=f"ag_{match_id}")
                    if col3.button("📝 Save Result", key=f"save_{match_id}"):
                        ok, msg = update_audit(sb, match_id, hg, ag, call)
                        if ok:
                            st.success("Result recorded.")
                            st.rerun()
                        else:
                            st.error(msg)

    # --------------------------------------------------------- performance
    with tabs[2]:
        st.subheader("📊 Performance")
        rows = load_all(sb)
        settled = [r for r in rows if r.get("actual_home_goals") is not None]

        if not settled:
            st.info("No settled matches yet. Enter results in the Pending tab.")
        else:
            placed = [r for r in settled if r.get("call_1x2") != "NO BET"]
            correct = sum(1 for r in placed if r.get("is_correct_1x2"))
            override_games = [r for r in settled if r.get("f1_f5_override")]
            override_correct = sum(1 for r in override_games if r.get("is_correct_1x2"))
            collapse_games = [r for r in settled if r.get("away_collapse")]
            doubt_games = [r for r in settled if r.get("doubted_starter")]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Settled", len(settled))
            c2.metric("Bets placed", len(placed))
            c3.metric("1X2 accuracy",
                      f"{correct}/{len(placed)} ({(correct / len(placed) * 100):.0f}%)"
                      if placed else "—")
            c4.metric("Override accuracy",
                      f"{override_correct}/{len(override_games)}"
                      if override_games else "—")

            c1, c2 = st.columns(2)
            c1.metric("Away collapse triggers", len(collapse_games))
            c2.metric("Doubt starter triggers", len(doubt_games))

            st.markdown('<div class="section-title">All Settled Matches</div>', unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Date": r.get("match_date"),
                "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                "Gap": r.get("total_gap"),
                "Call": r.get("call_1x2"),
                "Actual": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                "Correct": ("✅" if r.get("is_correct_1x2") else
                            "❌" if r.get("is_correct_1x2") is False else "—"),
                "Override": "✅" if r.get("f1_f5_override") else "—",
                "Collapse": "✅" if r.get("away_collapse") else "—",
                "Doubt": "✅" if r.get("doubted_starter") else "—",
            } for r in settled])
            st.dataframe(df, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------- spec
    with tabs[3]:
        st.subheader("v4 Specification")
        st.markdown("""
        **100-point model. Only prematch fields. No xG, no possession, no post-match data.**

        | Factor | Weight | Source |
        |--------|--------|--------|
        | F1 League Momentum | 20 | Position, Points, GD |
        | F2 Current Form Last 5 | 25 | W/D/L from recent matches |
        | F3 Venue Split | 15 | Win% + goal edge at venue |
        | F4 Availability | 15 | Injuries vs Confirmed XI |
        | F5 H2H Psychology | 10 | H2H W/D/L |
        | F6 Attack Profile | 15 | Avg goals scored |

        **Decision rule:**
        - Gap > 25 → Straight Win
        - Gap 15–25 → Double Chance
        - Gap < 15 → NO BET

        **Overrides:**
        - F1 vs F5 conflict: if F1 gap ≥ 12 AND F5 gap ≥ 6 AND opposite leaders,
          the larger gap wins (triggers ~20% of matches)
        - Away collapse: away team 0 points from 3+ away games → home +8 F1, force Over 2.5
        - Doubt starter: doubted key player in confirmed XI → F4 +5, force Over 2.5

        **O/U:** Expected total = (Home avg scored + Away avg conceded + Away avg scored + Home avg conceded) / 2
        - < 2.5 → Under
        - > 3.3 → Over
        - 2.5–3.3 → No Bet

        **Record (out-of-sample):** 9/10 (90%) on 1X2/DC.
        With F1 override: 10/10 (100%) but 1 circular.
        """)


main()
