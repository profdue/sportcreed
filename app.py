"""
v4.3 RAW-ONLY Predictor — universal version.

Parser fixes:
  #FIX-1  _norm_comp strips accents
  #FIX-2  _parse_league reads JSON-LD superEvent.name first
  #FIX-3  _parse_last5 falls back to no-filter if strict filter yields 0
  #FIX-4  upsert_match guards league_name NOT NULL
  #FIX-5  _parse_last10 falls back to keystats block when .st-table absent
  #FIX-6  _parse_players falls back to keystats block when goalassist absent
  #FIX-7  warning layer distinguishes missing (None) from zero
  #FIX-8  _parse_last10 SKIPS venue-split .st-table (scans headers AND body cells)
  #FIX-9  keystats values OVERRIDE .st-table for keystats-owned fields

Audit layer:
  #N1 no_bet_reason_1x2 / no_bet_reason_ou
  #N2 venue_ppg_gap fields (candidate feature)
  #N3 risk-warning display layer
  #N4 Data Audit tab

v5.0 rule layer (this file):
  - composite leader + tags computed from the existing factor math
  - bet placed only when total_gap >= 20 and no conflict/injury gate fires
  - tags: gap_20/gap_below_20, home_leader/away_leader,
          team_disagreement/team_agreement, venue_incomplete/venue_complete
  - bet_market and tags written to DB via save_prediction
  - settled_dc_hit written to DB via update_audit
"""

import json
import os
import re
import unicodedata
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="v4.3 Raw Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .team-header { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.5rem 2rem; color: #fff; margin-bottom: 1rem; }
    .team-names { font-size: 2rem; font-weight: 800; margin: 0; }
    .team-meta { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }
    .verdict-bet { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0; }
    .verdict-nobet { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0; }
    .verdict-label { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #6ee7b7; text-transform: uppercase; }
    .verdict-label-grey { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
    .verdict-pick { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0.35rem 0; line-height: 1.1; }
    .verdict-noedge { font-size: 1.6rem; font-weight: 700; color: #cbd5e1; margin: 0.35rem 0; }
    .verdict-detail { font-size: 1rem; color: #d1fae5; margin-top: 0.5rem; }
    .verdict-detail-grey { font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem; }
    .factor-row { background: #0f172a; border-radius: 10px; padding: 0.75rem 1rem;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
    .factor-name { color: #cbd5e1; font-weight: 600; font-size: 0.9rem; }
    .factor-val { color: #3b82f6; font-weight: 800; font-size: 1.1rem; }
    .trigger-row { background: #1e293b; border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 0.4rem; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; }
    .trigger-on { background: #064e3b; color: #6ee7b7; }
    .trigger-off { background: #1e293b; color: #64748b; }
    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem 0; }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; font-weight: 700; border-radius: 10px; border: none; padding: 0.6rem 1.25rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key), {"ok": True, "url": url}
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


def _has_bs4():
    try:
        import bs4
        return True
    except ImportError:
        return False


# ============================================================================
# v4.3 UNIVERSAL CONSTANTS
# ============================================================================
ALPHA = 2
PRIOR_FORM = 0.4
PRIOR_H2H = 1.0 / 3.0

KEYSTATS_PRIORITY_FIELDS = {
    "home_last10_w", "home_last10_d", "home_last10_l",
    "away_last10_w", "away_last10_d", "away_last10_l",
    "home_last10_avg_scored", "home_last10_avg_conceded",
    "away_last10_avg_scored", "away_last10_avg_conceded",
    "home_last10_over25", "home_last10_under25",
    "away_last10_over25", "away_last10_under25",
    "home_last10_btts_yes", "home_last10_btts_no",
    "away_last10_btts_yes", "away_last10_btts_no",
    "home_home_last10_avg_scored", "home_home_last10_avg_conceded",
    "home_away_last10_avg_scored", "home_away_last10_avg_conceded",
    "away_home_last10_avg_scored", "away_home_last10_avg_conceded",
    "away_away_last10_avg_scored", "away_away_last10_avg_conceded",
    "home_last10_possession", "away_last10_possession",
    "home_last10_corners_for", "home_last10_corners_against",
    "away_last10_corners_for", "away_last10_corners_against",
    "home_home_last10_corners_for", "home_home_last10_corners_against",
    "away_away_last10_corners_for", "away_away_last10_corners_against",
    "home_top_scorer", "home_top_scorer_goals",
    "away_top_scorer", "away_top_scorer_goals",
    "home_top_assister", "home_top_assister_assists",
    "away_top_assister", "away_top_assister_assists",
}


# ============================================================================
# PARSER
# ============================================================================
class SportsgamblerParser:
    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError("beautifulsoup4 required. pip install beautifulsoup4")
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, "html.parser")
        self.home_team = None
        self.away_team = None

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
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    @staticmethod
    def _norm_comp(s):
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _team_matches(self, full_name, table_name):
        if not full_name or not table_name:
            return False
        a = self._norm(full_name)
        b = self._norm(table_name)
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        a_words, b_words = a.split(), b.split()
        a_last = a_words[-1] if a_words else ""
        b_last = b_words[-1] if b_words else ""
        if a_last and (a_last == b_last or a_last in b or b_last in a):
            return True
        a_tokens = {w for w in a_words if len(w) > 3}
        b_tokens = {w for w in b_words if len(w) > 3}
        if a_tokens & b_tokens:
            return True
        for aw in a_words:
            for bw in b_words:
                if len(aw) >= 4 and len(bw) >= 4 and aw[:4] == bw[:4]:
                    return True
        return False

    def _team_matches_strict(self, full_name, header_text):
        if not full_name or not header_text:
            return False
        a = self._norm(full_name)
        b = self._norm(header_text)
        a_words = [w for w in a.split() if len(w) > 3]
        if not a_words:
            return False
        primary = a_words[0]
        return primary in b

    def parse(self):
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        league, tier, group = self._parse_league()
        venue = self._parse_venue()

        kickoff_utc_iso = None
        if match_date and kickoff:
            try:
                dt = datetime.strptime(f"{match_date} {kickoff}", "%Y-%m-%d %H:%M")
                kickoff_utc_iso = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            except ValueError:
                pass

        record = {
            "match_date": match_date,
            "kickoff_utc": kickoff_utc_iso,
            "kickoff_local": kickoff,
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

        home_last10 = self._parse_last10("home")
        away_last10 = self._parse_last10("away")
        record.update(home_last10)
        record.update(away_last10)

        keystats = self._parse_keystats_block()
        for key, val in keystats.items():
            if val is None:
                continue
            if key in KEYSTATS_PRIORITY_FIELDS:
                record[key] = val
            elif key not in record or record.get(key) is None:
                record[key] = val

        home_players = self._parse_players("home")
        away_players = self._parse_players("away")
        for k, v in home_players.items():
            if v is not None and (k not in record or record.get(k) is None):
                record[k] = v
        for k, v in away_players.items():
            if v is not None and (k not in record or record.get(k) is None):
                record[k] = v

        record["home_injuries"] = self._parse_injuries("home")
        record["away_injuries"] = self._parse_injuries("away")
        record["home_xi"] = self._parse_xi("home")
        record["away_xi"] = self._parse_xi("away")
        record["home_formation"] = self._parse_formation("home")
        record["away_formation"] = self._parse_formation("away")
        record.update(self._parse_h2h())
        record.update(self._parse_corners())
        record["odds"] = self._parse_odds()
        return record

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
        for script in self.soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            graph = data.get("@graph", [])
            for node in graph:
                article = node.get("mainEntity")
                if not isinstance(article, dict):
                    continue
                super_event = article.get("superEvent")
                if isinstance(super_event, dict) and super_event.get("name"):
                    league = super_event["name"].strip()
                    break
            if league:
                break

        if not league:
            for link in self.soup.select(".t_top .t_info_link"):
                text = link.get_text(strip=True)
                if text.lower() == "football":
                    continue
                if re.search(
                    r"(League|Serie|Série|Liga|Bundesliga|Ligue|Premier|"
                    r"Championship|MLS|Cup|Division|Primera|Nations)",
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
        return league or "Unknown", tier, group

    def _parse_venue(self):
        venue_el = self.soup.select_one(".t_top .t_venue")
        return venue_el.get_text(strip=True) if venue_el else None

    def _parse_standings(self):
        out = {}
        main_table = None
        for table in self.soup.select("table"):
            headers = [th.get_text(strip=True).lower() for th in table.select("th")]
            if any(h in ("pts", "p") for h in headers) and any("team" in h for h in headers):
                main_table = table
                break

        if main_table:
            headers = [th.get_text(strip=True).lower() for th in main_table.select("th")]
            try:
                pos_idx = 0
                team_idx = next(i for i, h in enumerate(headers) if "team" in h)
                pts_idx = next(i for i, h in enumerate(headers) if h in ("pts", "p"))
                gd_indices = [i for i, h in enumerate(headers) if h in ("gd", "+/-", "goal diff")]
                gf_idx = next((i for i, h in enumerate(headers) if h in ("gf", "goals for")), None)
                ga_idx = next((i for i, h in enumerate(headers) if h in ("ga", "goals against")), None)
            except StopIteration:
                main_table = None

        if main_table:
            for row in main_table.select("tbody tr"):
                cells = row.select("td")
                if len(cells) <= max(pos_idx, team_idx, pts_idx):
                    continue
                team_clean = re.sub(r"\s*logo\s*", " ", cells[team_idx].get_text(strip=True), flags=re.I).strip()

                gf, ga, gd = None, None, None
                if gf_idx is not None and ga_idx is not None and ga_idx < len(cells):
                    gf = self._to_int(cells[gf_idx].get_text(strip=True))
                    ga = self._to_int(cells[ga_idx].get_text(strip=True))
                    if gf is not None and ga is not None:
                        gd = gf - ga
                if gf is None and gd_indices:
                    for gi in gd_indices:
                        if gi >= len(cells):
                            continue
                        txt = cells[gi].get_text(strip=True)
                        parsed = self._parse_gf_ga(txt)
                        if parsed:
                            gf, ga = parsed
                            gd = gf - ga
                            break
                        if gd is None:
                            gd_plain = self._parse_gd(txt)
                            if gd_plain is not None:
                                gd = gd_plain

                played = self._to_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None

                if self._team_matches(self.home_team, team_clean):
                    out["home_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["home_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    out["home_played"] = played
                    if gf is not None:
                        out["home_gf"] = gf
                        out["home_ga"] = ga
                    if gd is not None:
                        out["home_gd"] = gd
                elif self._team_matches(self.away_team, team_clean):
                    out["away_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["away_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    out["away_played"] = played
                    if gf is not None:
                        out["away_gf"] = gf
                        out["away_ga"] = ga
                    if gd is not None:
                        out["away_gd"] = gd

        self._parse_split_table("#Homeleague", self.home_team, "home_home", out)
        self._parse_split_table("#Awayleague", self.home_team, "home_away", out)
        self._parse_split_table("#Homeleague", self.away_team, "away_home", out)
        self._parse_split_table("#Awayleague", self.away_team, "away_away", out)

        return out

    def _parse_split_table(self, container_selector, team_name, prefix, out):
        container = self.soup.select_one(container_selector)
        if not container:
            return
        table = container.find("table")
        if not table:
            return
        headers = [th.get_text(strip=True).lower() for th in table.select("th")]
        try:
            team_idx = next(i for i, h in enumerate(headers) if "team" in h)
            pts_idx = next(i for i, h in enumerate(headers) if h in ("pts", "p"))
        except StopIteration:
            return
        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) <= max(team_idx, pts_idx):
                continue
            team_clean = re.sub(r"\s*logo\s*", " ", cells[team_idx].get_text(strip=True), flags=re.I).strip()
            if not self._team_matches(team_name, team_clean):
                continue
            played = self._to_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None
            won = self._to_int(cells[3].get_text(strip=True)) if len(cells) > 3 else None
            pts = self._to_int(cells[pts_idx].get_text(strip=True))
            if prefix == "home_home":
                out["home_home_points"] = pts
                out["home_home_played"] = played
                if played and won is not None:
                    out["home_home_win_pct"] = (won / played) * 100
            elif prefix == "home_away":
                out["home_away_points"] = pts
                out["home_away_played"] = played
            elif prefix == "away_home":
                out["away_home_points"] = pts
                out["away_home_played"] = played
            elif prefix == "away_away":
                out["away_away_points"] = pts
                out["away_away_played"] = played
                if played and won is not None:
                    out["away_away_win_pct"] = (won / played) * 100
            return

    @staticmethod
    def _parse_gd(s):
        try:
            return int(s.strip().replace("+", ""))
        except ValueError:
            return None

    @staticmethod
    def _parse_gf_ga(s):
        m = re.match(r"(\d+)\s*[:\-]\s*(\d+)", s.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _parse_last5(self, side):
        container = self.soup.select_one("#last-matches #All")
        if not container:
            container = self.soup.select_one("#last-matches")
        if not container:
            return []
        block = container.select_one(".teamstats-left" if side == "home" else ".teamstats-right")
        if not block:
            return []

        league_name = (self._parse_league()[0] or "")
        league_short = league_name.split(" - ")[-1] if " - " in league_name else league_name
        league_norm = self._norm_comp(league_short)

        def collect(apply_league_filter: bool):
            items = []
            for item in block.select("li.team-stat-list-item"):
                if len(items) >= 5:
                    break
                parsed = self._parse_last5_item(item, side, apply_league_filter, league_norm)
                if parsed is not None:
                    items.append(parsed)
            return items

        strict = collect(apply_league_filter=True)
        if strict:
            return strict
        return collect(apply_league_filter=False)

    def _parse_last5_item(self, item, side, apply_league_filter, league_norm):
        tracked = (self.home_team if side == "home" else self.away_team) or ""

        date_el = item.select_one(".team-stats-date")
        date_text = date_el.get_text(strip=True) if date_el else ""

        if apply_league_filter and ":" in date_text:
            comp = date_text.split(":", 1)[0].strip()
            comp_norm = self._norm_comp(comp)
            if league_norm and comp_norm:
                if league_norm not in comp_norm and comp_norm not in league_norm:
                    return None

        teams = item.select(".team-stats-team")
        if len(teams) < 2:
            return None
        home_name_el, away_name_el = teams[0], teams[1]
        h_score_el = home_name_el.select_one(".score-right")
        a_score_el = away_name_el.select_one(".score-right")
        home_name = home_name_el.get_text(" ", strip=True)
        away_name = away_name_el.get_text(" ", strip=True)
        if h_score_el:
            home_name = home_name.replace(h_score_el.get_text(strip=True), "").strip()
        if a_score_el:
            away_name = away_name.replace(a_score_el.get_text(strip=True), "").strip()
        h_score = self._to_int(h_score_el.get_text(strip=True)) if h_score_el else None
        a_score = self._to_int(a_score_el.get_text(strip=True)) if a_score_el else None
        if h_score is None or a_score is None:
            return None

        tracked_is_home = self._team_matches(tracked, home_name)
        tracked_is_away = self._team_matches(tracked, away_name)
        if tracked_is_home:
            is_home, sf, sa = True, h_score, a_score
        elif tracked_is_away:
            is_home, sf, sa = False, a_score, h_score
        else:
            is_home = (side == "home")
            sf, sa = (h_score, a_score) if is_home else (a_score, h_score)
        result = "W" if sf > sa else ("D" if sf == sa else "L")
        return {
            "date": date_text,
            "opp": away_name if is_home else home_name,
            "result": result,
            "score_for": sf,
            "score_against": sa,
            "is_home": is_home,
        }

    def _parse_last10(self, side):
        out = {}
        prefix = "home" if side == "home" else "away"
        table = self.soup.select_one(".st-table")

        if table:
            header_text = " ".join(
                th.get_text(" ", strip=True) for th in table.select("th")
            ).lower()

            body_first_cells = []
            for tr in table.select("tbody tr"):
                tds = tr.select("td")
                if tds:
                    body_first_cells.append(tds[0].get_text(" ", strip=True))
            body_text = " ".join(body_first_cells).lower()

            combined = header_text + " || " + body_text
            venue_markers = (
                "home stats", "away stats",
                "home form", "away form",
                "home matches", "away matches",
                "home league games", "away league games",
                "home league", "away league",
            )
            if any(m in combined for m in venue_markers):
                table = None

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
                    out[f"{prefix}_last10_over25"] = self._to_int(cells[5])
                    out[f"{prefix}_last10_under25"] = self._to_int(cells[6])
                    if len(cells) >= 9:
                        out[f"{prefix}_last10_btts_yes"] = self._to_int(cells[7])
                        out[f"{prefix}_last10_btts_no"] = self._to_int(cells[8])

        goals_block_id = f"#goals-{'hometeam' if side == 'home' else 'awayteam'}"
        goals_block = self.soup.select_one(goals_block_id)
        if goals_block:
            text = goals_block.get_text(" ", strip=True)
            m = re.search(r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 home matches", text)
            if m:
                out[f"{prefix}_home_last10_avg_scored"] = float(m.group(1))
                out[f"{prefix}_home_last10_avg_conceded"] = float(m.group(2))
            m = re.search(r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 away matches", text)
            if m:
                out[f"{prefix}_away_last10_avg_scored"] = float(m.group(1))
                out[f"{prefix}_away_last10_avg_conceded"] = float(m.group(2))

        w_key = f"{prefix}_last10_w"
        if w_key in out and out[w_key] is not None:
            out[f"{prefix}_last10_win_pct"] = (out[w_key] / 10) * 100
        return out

    def _parse_keystats_block(self):
        out = {}

        home_team = self.home_team
        away_team = self.away_team
        if not home_team or not away_team:
            return out

        keystats = self.soup.select_one(".keystats")
        if not keystats:
            return out

        for item in keystats.select(".keystat-item"):
            title_sub = item.select_one(".stats-title-sub")
            if not title_sub:
                continue
            title_text = title_sub.get_text(" ", strip=True)

            is_away = "awaystats" in item.get("class", [])
            prefix = "away" if is_away else "home"

            text = item.get_text(" ", strip=True)

            if "Full-Time Result" in title_text:
                m = re.search(
                    r"(\d+)\s+wins?,\s*(\d+)\s+defeats?\s+and\s+(\d+)\s+draws?\s+in\s+the\s+previous\s+10\s+matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_w"] = int(m.group(1))
                    out[f"{prefix}_last10_l"] = int(m.group(2))
                    out[f"{prefix}_last10_d"] = int(m.group(3))

                venue = "home" if not is_away else "away"
                m = re.search(
                    rf"(\d+)\s+wins?,\s*(\d+)\s+defeats?\s+and\s+(\d+)\s+draws?\s+in\s+the\s+previous\s+10\s+{venue}\s+matches",
                    text, re.I,
                )
                if m:
                    w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    played = w + l + d
                    if not is_away:
                        out["home_home_played"] = out.get("home_home_played") or played
                    else:
                        out["away_away_played"] = out.get("away_away_played") or played

            elif "Goals" in title_text:
                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_last10_avg_conceded"] = float(m.group(2))

                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 home matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_home_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_home_last10_avg_conceded"] = float(m.group(2))

                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 away matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_away_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_away_last10_avg_conceded"] = float(m.group(2))

                m = re.search(r"BTTS Yes in (\d+) of the previous 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_btts_yes"] = int(m.group(1))
                    out[f"{prefix}_last10_btts_no"] = 10 - int(m.group(1))

                m = re.search(r"Over 2\.5 Goals in (\d+) of the previous 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_over25"] = int(m.group(1))
                    out[f"{prefix}_last10_under25"] = 10 - int(m.group(1))

            elif "Corners" in title_text:
                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_last10_corners_against"] = float(m.group(2))

                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 home matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_home_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_home_last10_corners_against"] = float(m.group(2))

                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 away matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_away_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_away_last10_corners_against"] = float(m.group(2))

            elif "Possession" in title_text:
                m = re.search(r"average of ([\d.]+)% possession in the last 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_possession"] = float(m.group(1))

            elif "Top Scorers" in title_text:
                m = re.search(
                    r"Top Scorers for .+? this season are\s+(.+?)(?:Top Assistors|$)",
                    text,
                )
                if m:
                    first = m.group(1).split(",")[0].strip()
                    mm = re.match(r"(.+?)\s*\((\d+)\)", first)
                    if mm:
                        out[f"{prefix}_top_scorer"] = mm.group(1).strip()
                        out[f"{prefix}_top_scorer_goals"] = int(mm.group(2))

                m = re.search(
                    r"Top Assistors for .+? this season are\s+(.+?)$",
                    text,
                )
                if m:
                    first = m.group(1).split(",")[0].strip()
                    mm = re.match(r"(.+?)\s*\((\d+)\)", first)
                    if mm:
                        out[f"{prefix}_top_assister"] = mm.group(1).strip()
                        out[f"{prefix}_top_assister_assists"] = int(mm.group(2))

        for prefix in ("home", "away"):
            w_key = f"{prefix}_last10_w"
            if out.get(w_key) is not None and f"{prefix}_last10_win_pct" not in out:
                out[f"{prefix}_last10_win_pct"] = (out[w_key] / 10) * 100

        return out

    def _parse_players(self, side):
        out = {}
        prefix = "home" if side == "home" else "away"
        block_id = f"#{'ht' if side == 'home' else 'at'}-goalassist"
        block = self.soup.select_one(block_id)
        if not block:
            return out
        text = block.get_text(" ", strip=True)
        m = re.search(r"Top Scorers for .+? this season are (.+?)(?:\.|$)", text)
        if m:
            first = m.group(1).split(",")[0].strip()
            mm = re.match(r"(.+?)\s*\((\d+)\)", first)
            if mm:
                out[f"{prefix}_top_scorer"] = mm.group(1).strip()
                out[f"{prefix}_top_scorer_goals"] = int(mm.group(2))
        m = re.search(r"Top Assis\w+ for .+? this season are (.+?)(?:\.|$)", text)
        if m:
            first = m.group(1).split(",")[0].strip()
            mm = re.match(r"(.+?)\s*\((\d+)\)", first)
            if mm:
                out[f"{prefix}_top_assister"] = mm.group(1).strip()
                out[f"{prefix}_top_assister_assists"] = int(mm.group(2))
        return out

    def _parse_injuries(self, side):
        out = []
        seen = set()
        target = self.home_team if side == "home" else self.away_team
        for outline in self.soup.select(".inj-two-outline"):
            header = outline.select_one(".light-header strong")
            if not header:
                continue
            header_text = header.get_text(" ", strip=True)
            if not self._team_matches_strict(target, header_text):
                continue
            for row in outline.select(".inj-two-row"):
                if "inj-two-title" in row.get("class", []):
                    continue
                player_el = row.select_one(".inj-two-player")
                info_el = row.select_one(".inj-two-info")
                if not player_el:
                    continue
                player = player_el.get_text(strip=True)
                if player in seen:
                    continue
                seen.add(player)
                info = info_el.get_text(strip=True) if info_el else ""
                low = info.lower()
                if "doubt" in low:
                    status = "doubt"
                elif "suspension" in low or "yellow" in low or "red" in low:
                    status = "suspended"
                else:
                    status = "injury"

                expected_return = None
                detail_el = row.select_one(".inj-two-hidden")
                if detail_el:
                    detail_text = detail_el.get_text(" ", strip=True)
                    m = re.search(r"Expected return:\s*(\d{4}-\d{2}-\d{2})", detail_text)
                    if m:
                        expected_return = m.group(1)

                out.append({
                    "player": player,
                    "status": status,
                    "type": info,
                    "expected_return": expected_return,
                })
        return out

    def _parse_xi(self, side):
        out = []
        side_class = "lineups-home" if side == "home" else "lineups-away"

        content_block = self.soup.select_one(".content-block#lineups")
        if not content_block:
            for cb in self.soup.select(".content-block"):
                if cb.select_one(".lineups-formation") and cb.select_one(".lineups-home"):
                    content_block = cb
                    break

        if not content_block:
            return out

        block = content_block.select_one(f".{side_class}")
        if not block:
            return out

        for player in block.select(".lineups-player .player-name"):
            name = player.get_text(strip=True)
            if name:
                out.append(name)
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
            h_name_el, a_name_el = teams[0], teams[1]
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
            if h_score > a_score:
                winner_name = h_name
            elif a_score > h_score:
                winner_name = a_name
            else:
                winner_name = None
            if winner_name:
                if self._team_matches(self.home_team, winner_name):
                    out["h2h_home_wins"] += 1
                elif self._team_matches(self.away_team, winner_name):
                    out["h2h_away_wins"] += 1
            else:
                out["h2h_draws"] += 1
            out["h2h"].append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "home": h_name,
                "away": a_name,
                "score": f"{h_score}-{a_score}",
                "winner_name": winner_name,
            })
        return out

    def _parse_corners(self):
        out = {}
        for side, cid in [("home", "#ht-corners"), ("away", "#at-corners")]:
            block = self.soup.select_one(cid)
            if not block:
                continue
            text = block.get_text(" ", strip=True)
            m = re.search(r"average of ([\d.]+) corners awarded and ([\d.]+) corners conceded in the last 10 matches", text)
            if m:
                out[f"{side}_last10_corners_for"] = float(m.group(1))
                out[f"{side}_last10_corners_against"] = float(m.group(2))
            venue = "home" if side == "home" else "away"
            m = re.search(rf"average of ([\d.]+) corners awarded and ([\d.]+) corners against in the last 10 {venue} matches", text)
            if m:
                out[f"{side}_{venue}_last10_corners_for"] = float(m.group(1))
                out[f"{side}_{venue}_last10_corners_against"] = float(m.group(2))
        return out

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
            elif "half-time result" in market:
                odds["ht_home"] = entries.get("1")
                odds["ht_draw"] = entries.get("X")
                odds["ht_away"] = entries.get("2")
            elif "double chance" in market:
                odds["dc_1x"] = entries.get("1X")
                odds["dc_12"] = entries.get("12")
                odds["dc_x2"] = entries.get("X2")
            elif "draw no bet" in market:
                odds["dnb_home"] = entries.get("1")
                odds["dnb_away"] = entries.get("2")
            elif "asian handicap" in market:
                for label, val in entries.items():
                    m = re.match(r"(\d)\s*Hcp\s*([+-]?[\d.]+)", label)
                    if m:
                        side = m.group(1)
                        line = self._to_float(m.group(2))
                        if side == "1":
                            odds["ah_home_line"] = line
                            odds["ah_home"] = val
                        else:
                            odds["ah_away_line"] = line
                            odds["ah_away"] = val
            elif "both teams to score" in market:
                odds["btts_yes"] = entries.get("Yes")
                odds["btts_no"] = entries.get("No")
            elif "total goals" in market:
                for label, val in entries.items():
                    m = re.match(r"(Over|Under)\s+([\d.]+)", label, re.I)
                    if m and m.group(2) == "2.5":
                        if m.group(1).lower() == "over":
                            odds["over_2.5"] = val
                        else:
                            odds["under_2.5"] = val
        return odds


# ============================================================================
# v4.3 UNIVERSAL MODEL
# ============================================================================

def smooth_rate(wins, draws, games):
    if games is None or games <= 0:
        return PRIOR_FORM
    pts = wins + draws * 0.4
    return (pts + ALPHA * PRIOR_FORM) / (games + ALPHA)


def calc_f1(row):
    home_pts = row.get("home_points") or 0
    away_pts = row.get("away_points") or 0
    home_gd = row.get("home_gd") or 0
    away_gd = row.get("away_gd") or 0
    pts_gap = home_pts - away_pts
    gd_gap = home_gd - away_gd
    f1_home = 10 + (pts_gap * 1.33) + (gd_gap * 0.66)
    f1_away = 10 - (pts_gap * 1.33) - (gd_gap * 0.66)
    f1_home = max(2, min(18, f1_home))
    f1_away = max(2, min(18, f1_away))

    away_collapse = False
    if (row.get("away_away_points") == 0
            and (row.get("away_away_played") or 0) >= 3):
        f1_home = min(18, f1_home + 8)
        away_collapse = True
    return f1_home, f1_away, away_collapse


def calc_f2(last5):
    if not last5:
        return None
    w = sum(1 for m in last5[:5] if m.get("result") == "W")
    d = sum(1 for m in last5[:5] if m.get("result") == "D")
    n = min(len(last5), 5)
    return smooth_rate(w, d, n) * 25


def calc_f3(win_pct, avg_scored, avg_conceded):
    win_pts = ((win_pct or 0) / 100) * 10
    edge = (avg_scored or 0) - (avg_conceded or 0)
    xg_pts = 5 if edge > 0.5 else (2 if edge > 0 else 0)
    return min(15, win_pts + xg_pts)


def calc_f4(top_scorer, top_assister, injuries, xi):
    f4 = 15
    xi = xi or []
    injured = [i.get("player", "") for i in (injuries or []) if i.get("status") == "injury"]
    doubts = [i.get("player", "") for i in (injuries or []) if i.get("status") == "doubt"]
    if top_scorer and top_scorer in injured and top_scorer not in xi:
        f4 -= 5
    if top_assister and top_assister in injured and top_assister not in xi:
        f4 -= 4

    doubted_starter = False
    for name in doubts:
        if name in xi and (name == top_scorer or name == top_assister):
            f4 += 5
            doubted_starter = True
    for name in injured:
        if name in xi and (name == top_scorer or name == top_assister):
            f4 += 5
            doubted_starter = True

    return max(0, min(20, f4)), doubted_starter


def calc_f5(h2h_home_wins, h2h_away_wins, h2h_total):
    f5_h = (h2h_home_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    f5_a = (h2h_away_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    if h2h_total < 4:
        f5_h = 3.33 + (f5_h - 3.33) * (h2h_total / 4.0)
        f5_a = 3.33 + (f5_a - 3.33) * (h2h_total / 4.0)
    return f5_h, f5_a


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
    f5_leader_raw = "home" if f5_home > f5_away else "away"

    conflict = (f1_leader != f5_leader_raw and f1_gap >= 12 and f5_gap >= 4)
    override = False
    if conflict and f1_gap > f5_gap:
        if f1_leader == "home":
            f5_home, f5_away = 10, 0
        else:
            f5_home, f5_away = 0, 10
        override = True
    return f5_home, f5_away, conflict, override, f5_leader_raw


def calc_expected_total(row):
    h1 = row.get("home_home_last10_avg_scored")
    a1 = row.get("away_away_last10_avg_conceded")
    a2 = row.get("away_away_last10_avg_scored")
    h2 = row.get("home_home_last10_avg_conceded")
    if h1 is None:
        h1 = row.get("home_last10_avg_scored")
    if a1 is None:
        a1 = row.get("away_last10_avg_conceded")
    if a2 is None:
        a2 = row.get("away_last10_avg_scored")
    if h2 is None:
        h2 = row.get("home_last10_avg_conceded")
    h1 = h1 if h1 is not None else 1.0
    a1 = a1 if a1 is not None else 1.0
    a2 = a2 if a2 is not None else 1.0
    h2 = h2 if h2 is not None else 1.0
    return (h1 + a1 + a2 + h2) / 2


def calc_venue_ppg_gap(row):
    hp = row.get("home_home_points")
    hg = row.get("home_home_played")
    ap = row.get("away_away_points")
    ag = row.get("away_away_played")

    home_ppg = (hp / hg) if (hp is not None and hg) else None
    away_ppg = (ap / ag) if (ap is not None and ag) else None

    if home_ppg is None or away_ppg is None:
        return None, home_ppg, away_ppg

    return round(home_ppg - away_ppg, 4), round(home_ppg, 4), round(away_ppg, 4)


def predict_v4_3(row):
    venue_ppg_gap, home_ppg, away_ppg = calc_venue_ppg_gap(row)

    has_home_last5 = bool(row.get("home_last5"))
    has_away_last5 = bool(row.get("away_last5"))
    has_standings = row.get("home_points") is not None and row.get("away_points") is not None
    home_played = row.get("home_played")
    away_played = row.get("away_played")
    home_played = home_played if home_played is not None else 0
    away_played = away_played if away_played is not None else 0

    if not (has_home_last5 and has_away_last5 and has_standings
            and home_played >= 3 and away_played >= 3):
        return _empty_prediction(
            "NO BET (insufficient data)",
            reason_1x2="insufficient_data",
            reason_ou="insufficient_data",
            venue_ppg_gap=venue_ppg_gap,
            home_ppg=home_ppg,
            away_ppg=away_ppg,
        )

    f1_home, f1_away, away_collapse = calc_f1(row)
    f2_home = calc_f2(row.get("home_last5"))
    f2_away = calc_f2(row.get("away_last5"))
    if f2_home is None or f2_away is None:
        return _empty_prediction(
            "NO BET (insufficient form)",
            reason_1x2="insufficient_form",
            reason_ou="insufficient_form",
            venue_ppg_gap=venue_ppg_gap,
            home_ppg=home_ppg,
            away_ppg=away_ppg,
        )

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

    f5_home, f5_away, conflict, override, f5_leader_raw = apply_override(
        f1_home, f1_away, f5_home, f5_away
    )

    home_total_raw = f1_home + f2_home + f3_home + f4_home + f5_home + f6_home
    away_total_raw = f1_away + f2_away + f3_away + f4_away + f5_away + f6_away

    leader = "home" if home_total_raw > away_total_raw else "away"
    raw_gap = abs(home_total_raw - away_total_raw)

    leaders = {
        "F1": "home" if f1_home > f1_away else "away",
        "F2": "home" if f2_home > f2_away else "away",
        "F3": "home" if f3_home > f3_away else "away",
        "F5": "home" if f5_home > f5_away else "away",
    }
    disagreements = sum(1 for l in leaders.values() if l != leader)
    shrink = max(0.0, 1.0 - disagreements * 0.15)
    gap = raw_gap * shrink

    if leader == "home":
        home_total = home_total_raw
        away_total = home_total_raw - gap
    else:
        away_total = away_total_raw
        home_total = away_total_raw - gap

    # ---- v5.0 rule layer ----------------------------------------------
    tags = []
    if gap >= 20:
        tags.append("gap_20")
    else:
        tags.append("gap_below_20")

    if leader == "home":
        tags.append("home_leader")
    else:
        tags.append("away_leader")

    if conflict or f1_vs_f2f3_conflict or disagreements >= 2:
        tags.append("team_disagreement")
    else:
        tags.append("team_agreement")

    hhp = row.get("home_home_played")
    aap = row.get("away_away_played")
    if (hhp is None or aap is None or hhp < 4 or aap < 4):
        tags.append("venue_incomplete")
    else:
        tags.append("venue_complete")

    f1_leader_final = "home" if f1_home > f1_away else "away"
    f2f3_home = f2_home + f3_home
    f2f3_away = f2_away + f3_away
    f2f3_leader = "home" if f2f3_home > f2f3_away else "away"
    f1_vs_f2f3_conflict = (
        f1_leader_final != f2f3_leader
        and abs(f1_home - f1_away) >= 12
        and abs(f2f3_home - f2f3_away) >= 5
    )

    skip = (
        gap < 20
        or (conflict and not override)
        or f1_vs_f2f3_conflict
    )
    home_scorer = row.get("home_top_scorer")
    away_scorer = row.get("away_top_scorer")
    home_inj = {i.get("player") for i in (row.get("home_injuries") or [])
                if i.get("status") == "injury"}
    away_inj = {i.get("player") for i in (row.get("away_injuries") or [])
                if i.get("status") == "injury"}
    home_xi = set(row.get("home_xi") or [])
    away_xi = set(row.get("away_xi") or [])
    if home_scorer and home_scorer in home_inj and home_scorer not in home_xi:
        skip = True
    if away_scorer and away_scorer in away_inj and away_scorer not in away_xi:
        skip = True

    if skip:
        call_1x2 = "NO BET"
        bet_market = None
        no_bet_reason_1x2 = "rule_skip"
    else:
        bet_market = "DC 1X" if leader == "home" else "DC X2"
        call_1x2 = bet_market
        no_bet_reason_1x2 = None
    # ---- end v5.0 rule layer ------------------------------------------

    expected = calc_expected_total(row)
    if expected < 2.4:
        call_ou = "Under 2.5"
        no_bet_reason_ou = None
    elif expected > 3.2:
        call_ou = "Over 2.5"
        no_bet_reason_ou = None
    else:
        call_ou = "No Bet"
        no_bet_reason_ou = "expected_in_middle_band"

    if (away_collapse or doubted_starter) and expected >= 2.4:
        call_ou = "Over 2.5"
        no_bet_reason_ou = None

    return {
        "f1_home": round(f1_home, 2), "f1_away": round(f1_away, 2),
        "f1_gap": round(abs(f1_home - f1_away), 2),
        "f1_leader": f1_leader_final,
        "f2_home": round(f2_home, 2), "f2_away": round(f2_away, 2),
        "f3_home": round(f3_home, 2), "f3_away": round(f3_away, 2),
        "f4_home": round(f4_home, 2), "f4_away": round(f4_away, 2),
        "f5_home": round(f5_home, 2), "f5_away": round(f5_away, 2),
        "f5_gap": round(abs(f5_home - f5_away), 2),
        "f5_leader": "home" if f5_home > f5_away else "away",
        "f5_leader_raw": f5_leader_raw,
        "f6_home": round(f6_home, 2), "f6_away": round(f6_away, 2),
        "home_total": round(home_total, 2),
        "away_total": round(away_total, 2),
        "raw_gap": round(raw_gap, 2),
        "total_gap": round(gap, 2),
        "disagreements": disagreements,
        "shrink_factor": round(shrink, 2),
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "f1_vs_f2f3_conflict": f1_vs_f2f3_conflict,
        "away_collapse": away_collapse,
        "doubted_starter": doubted_starter,
        "call_1x2": call_1x2,
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
        "model_version": "v4.3",
        "no_bet_reason_1x2": no_bet_reason_1x2,
        "no_bet_reason_ou": no_bet_reason_ou,
        "venue_ppg_gap": venue_ppg_gap,
        "venue_ppg_gap_home": home_ppg,
        "venue_ppg_gap_away": away_ppg,
        "bet_market": bet_market,
        "tags": tags,
    }


def predict_v4_2_1(row):
    has_home_last5 = bool(row.get("home_last5"))
    has_away_last5 = bool(row.get("away_last5"))
    has_standings = row.get("home_points") is not None and row.get("away_points") is not None

    if not (has_home_last5 and has_away_last5 and has_standings):
        return _empty_prediction("NO BET (insufficient data)",
                                 reason_1x2="insufficient_data",
                                 reason_ou="insufficient_data")

    f1_home, f1_away, away_collapse = calc_f1(row)

    def _f2_old(last5):
        if not last5:
            return 0
        raw = 0
        for m in last5[:5]:
            r = m.get("result", "L")
            if r == "W":
                raw += 5
            elif r == "D":
                raw += 2
        return max(0, min(25, raw))

    f2_home = _f2_old(row.get("home_last5"))
    f2_away = _f2_old(row.get("away_last5"))
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

    def _f5_old(hw, aw, total):
        if not total:
            return 5.0, 5.0
        return (hw / total) * 10, (aw / total) * 10

    f5_home, f5_away = _f5_old(row.get("h2h_home_wins") or 0,
                                row.get("h2h_away_wins") or 0,
                                len(row.get("h2h") or []))
    f6_home, f6_away = calc_f6(row.get("home_last10_avg_scored"),
                                row.get("away_last10_avg_scored"))

    f5_home, f5_away, conflict, override, f5_leader_raw = apply_override(
        f1_home, f1_away, f5_home, f5_away
    )

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
    if (away_collapse or doubted_starter) and expected >= 2.4:
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
        "f5_leader_raw": f5_leader_raw,
        "f6_home": round(f6_home, 2), "f6_away": round(f6_away, 2),
        "home_total": round(home_total, 2),
        "away_total": round(away_total, 2),
        "raw_gap": round(gap, 2),
        "total_gap": round(gap, 2),
        "disagreements": 0,
        "shrink_factor": 1.0,
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "f1_vs_f2f3_conflict": False,
        "away_collapse": away_collapse,
        "doubted_starter": doubted_starter,
        "call_1x2": call_1x2,
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
        "model_version": "v4.2.1",
        "no_bet_reason_1x2": "low_gap" if call_1x2 == "NO BET" else None,
        "no_bet_reason_ou": "expected_in_middle_band" if call_ou == "No Bet" else None,
        "venue_ppg_gap": None,
        "venue_ppg_gap_home": None,
        "venue_ppg_gap_away": None,
        "bet_market": None,
        "tags": [],
    }


def _empty_prediction(reason, reason_1x2=None, reason_ou=None,
                      venue_ppg_gap=None, home_ppg=None, away_ppg=None):
    return {
        "f1_home": 0, "f1_away": 0, "f1_gap": 0, "f1_leader": None,
        "f2_home": 0, "f2_away": 0,
        "f3_home": 0, "f3_away": 0,
        "f4_home": 0, "f4_away": 0,
        "f5_home": 0, "f5_away": 0, "f5_gap": 0, "f5_leader": None,
        "f5_leader_raw": None,
        "f6_home": 0, "f6_away": 0,
        "home_total": 0, "away_total": 0, "raw_gap": 0, "total_gap": 0,
        "disagreements": 0, "shrink_factor": 1.0,
        "f1_f5_conflict": False, "f1_f5_override": False,
        "f1_vs_f2f3_conflict": False,
        "away_collapse": False, "doubted_starter": False,
        "call_1x2": reason,
        "call_ou": "No Bet",
        "expected_total": 0,
        "model_version": "v4.3",
        "no_bet_reason_1x2": reason_1x2,
        "no_bet_reason_ou": reason_ou,
        "venue_ppg_gap": venue_ppg_gap,
        "venue_ppg_gap_home": home_ppg,
        "venue_ppg_gap_away": away_ppg,
        "bet_market": None,
        "tags": [],
    }


# ============================================================================
# RISK WARNINGS
# ============================================================================
def compute_warnings(row, prediction):
    warnings = []

    home_home_pct = row.get("home_home_win_pct")
    away_away_pct = row.get("away_away_win_pct")
    home_home_played = row.get("home_home_played")
    away_away_played = row.get("away_away_played")
    home_last10_w = row.get("home_last10_w")
    away_last10_w = row.get("away_last10_w")
    home_pts = row.get("home_points")
    away_pts = row.get("away_points")
    home_gd = row.get("home_gd")
    away_gd = row.get("away_gd")
    home_injuries = row.get("home_injuries") or []
    away_injuries = row.get("away_injuries") or []
    call = prediction.get("call_1x2", "")
    total_gap = prediction.get("total_gap", 0)
    h2h_draws = row.get("h2h_draws")

    if (home_home_pct is not None and away_away_pct is not None
            and home_home_pct == 0 and away_away_pct == 0):
        warnings.append({
            "code": "both_venue_winless",
            "label": "⚠️ Both teams winless at their venue",
            "detail": (
                f"Home has not won at home ({home_home_played if home_home_played is not None else '?'} games), "
                f"away has not won away ({away_away_played if away_away_played is not None else '?'} games). "
                f"In-sample this profile drew 5 of 5 times."
            ),
            "severity": "high",
        })

    if home_home_pct is not None and away_away_pct is not None:
        if abs(home_home_pct - away_away_pct) >= 40:
            small_home = home_home_played is not None and home_home_played < 4
            small_away = away_away_played is not None and away_away_played < 4
            if small_home or small_away:
                warnings.append({
                    "code": "extreme_gap_small_sample",
                    "label": "⚠️ Extreme venue gap on small sample",
                    "detail": (
                        f"Venue gap ≥ 40 but "
                        f"{'home ' if small_home else ''}"
                        f"{'away ' if small_away else ''}"
                        f"sample is under 4 games. "
                        f"Promoted teams and early-season anomalies "
                        f"produce false positives here."
                    ),
                    "severity": "high",
                })

    if call.endswith("Home") and home_last10_w is not None and home_last10_w <= 2:
        warnings.append({
            "code": "home_pick_weak_form",
            "label": "⚠️ Betting on home team with poor recent form",
            "detail": (
                f"Model picks home, but home team won only "
                f"{home_last10_w}/10 recent games."
            ),
            "severity": "medium",
        })
    if call.endswith("Away") and away_last10_w is not None and away_last10_w <= 2:
        warnings.append({
            "code": "away_pick_weak_form",
            "label": "⚠️ Betting on away team with poor recent form",
            "detail": (
                f"Model picks away, but away team won only "
                f"{away_last10_w}/10 recent games."
            ),
            "severity": "medium",
        })

    home_top_scorer = row.get("home_top_scorer")
    away_top_scorer = row.get("away_top_scorer")
    home_injured_names = {i.get("player") for i in home_injuries
                          if i.get("status") == "injury"}
    away_injured_names = {i.get("player") for i in away_injuries
                          if i.get("status") == "injury"}

    if len(home_injuries) >= 6 or home_top_scorer in home_injured_names:
        if call.endswith("Home"):
            warnings.append({
                "code": "home_squad_crisis",
                "label": "⚠️ Home team has injury concerns but is picked",
                "detail": (
                    f"{len(home_injuries)} injuries listed"
                    + (f", including top scorer {home_top_scorer}"
                       if home_top_scorer in home_injured_names else "")
                    + "."
                ),
                "severity": "medium",
            })
    if len(away_injuries) >= 6 or away_top_scorer in away_injured_names:
        if call.endswith("Away"):
            warnings.append({
                "code": "away_squad_crisis",
                "label": "⚠️ Away team has injury concerns but is picked",
                "detail": (
                    f"{len(away_injuries)} injuries listed"
                    + (f", including top scorer {away_top_scorer}"
                       if away_top_scorer in away_injured_names else "")
                    + "."
                ),
                "severity": "medium",
            })

    if (home_pts is not None and away_pts is not None
            and home_gd is not None and away_gd is not None
            and h2h_draws is not None):
        pts_gap = abs(home_pts - away_pts)
        gd_gap = abs(home_gd - away_gd)
        if (call in ("Straight Win Home", "Straight Win Away")
                and pts_gap <= 4 and gd_gap <= 3
                and h2h_draws >= 2):
            warnings.append({
                "code": "draw_prone_profile",
                "label": "⚠️ Draw-prone profile but straight win picked",
                "detail": (
                    f"Points gap {pts_gap}, GD gap {gd_gap}, "
                    f"H2H draws {h2h_draws}. "
                    f"These features historically favour a draw."
                ),
                "severity": "high",
            })

    if call in ("Straight Win Home", "Straight Win Away") and total_gap < 22:
        warnings.append({
            "code": "thin_gap_straight",
            "label": "⚠️ Straight win called on thin margin",
            "detail": (
                f"Straight win requires confidence. Total gap is only "
                f"{total_gap}. Near the DC threshold."
            ),
            "severity": "low",
        })

    return warnings


# ============================================================================
# DB HELPERS
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def _get_table_columns(_sb, table_name="matches_raw"):
    try:
        resp = _sb.table(table_name).select("*").limit(1).execute()
        if resp.data:
            return set(resp.data[0].keys())
        return None
    except Exception:
        return None


def upsert_match(sb, record):
    if sb is None:
        return False, "no client"
    try:
        real_columns = _get_table_columns(sb)
        if real_columns:
            clean = {k: v for k, v in record.items() if k in real_columns}
        else:
            clean = record

        if not clean.get("league_name"):
            clean["league_name"] = "Unknown"
        if not clean.get("match_date"):
            clean["match_date"] = None
        if not clean.get("home_team"):
            return False, "missing home_team"
        if not clean.get("away_team"):
            return False, "missing away_team"

        for jsonb_field in ["home_last5", "away_last5", "home_injuries",
                            "away_injuries", "h2h", "odds"]:
            if jsonb_field in clean and clean[jsonb_field] is None:
                clean[jsonb_field] = []
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
        real_columns = _get_table_columns(sb)
        if real_columns:
            clean = {k: v for k, v in result.items() if k in real_columns}
        else:
            clean = result
        sb.table("matches_raw").update(clean).eq("id", match_id).execute()
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
    if call_1x2.startswith("NO BET"):
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

    settled_dc_hit = None
    if "Double Chance 1X" in call_1x2:
        settled_dc_hit = actual in ("Home", "Draw")
    elif "Double Chance X2" in call_1x2:
        settled_dc_hit = actual in ("Away", "Draw")

    try:
        sb.table("matches_raw").update({
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "is_correct_1x2": is_correct,
            "settled_dc_hit": settled_dc_hit,
        }).eq("id", match_id).execute()
        return True, "ok"
    except Exception as e:
        return False, str(e)


# ============================================================================
# DISPLAY
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


def render_verdict(result):
    call = result.get("call_1x2", "")
    if call.startswith("NO BET"):
        if "insufficient" in call:
            detail = "insufficient data"
        else:
            detail = f"gap {result['total_gap']:.1f} below 20"
        st.markdown(f"""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">1X2 Verdict (v4.3)</div>
            <div class="verdict-noedge">NO BET — {detail}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-bet">
            <div class="verdict-label">⭐ 1X2 Verdict (v4.3)</div>
            <div class="verdict-pick">{call}</div>
            <div class="verdict-detail">
                Gap <strong>{result['total_gap']:.1f}</strong>
                &nbsp;·&nbsp; Home {result['home_total']:.1f}
                &nbsp;·&nbsp; Away {result['away_total']:.1f}
                &nbsp;·&nbsp; shrink {result.get('shrink_factor', 1.0):.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_ou_verdict(result):
    call = result.get("call_ou", "")
    expected = result.get("expected_total", 0)
    reason = result.get("no_bet_reason_ou")
    reason_str = f" ({reason})" if reason else ""
    if call == "No Bet":
        st.info(f"🟡 O/U 2.5: **NO BET** — expected {expected:.2f}{reason_str}")
    else:
        st.success(f"🟢 O/U 2.5: **{call}** — expected {expected:.2f}")


# ============================================================================
# TRAIN / HOLDOUT
# ============================================================================
HOLDOUT_DATE = "2026-09-19"


def _evaluate(rows, model_fn):
    placed = []
    correct = []
    no_bet = 0
    for r in rows:
        pred = model_fn(r)
        call = pred.get("call_1x2", "NO BET")
        hg = r.get("actual_home_goals")
        ag = r.get("actual_away_goals")
        if hg is None or ag is None:
            continue
        if call.startswith("NO BET"):
            no_bet += 1
            continue
        if hg > ag:
            actual = "Home"
        elif hg < ag:
            actual = "Away"
        else:
            actual = "Draw"
        if "Straight Win Home" in call and actual == "Home":
            correct.append(True)
        elif "Straight Win Away" in call and actual == "Away":
            correct.append(True)
        elif "Double Chance 1X" in call and actual in ("Home", "Draw"):
            correct.append(True)
        elif "Double Chance X2" in call and actual in ("Away", "Draw"):
            correct.append(True)
        else:
            correct.append(False)
        placed.append(pred)

    n = len(placed) + no_bet
    return {
        "n_total": n,
        "n_placed": len(placed),
        "n_no_bet": no_bet,
        "no_bet_pct": (no_bet / n * 100) if n else 0,
        "accuracy": (sum(correct) / len(placed) * 100) if placed else 0,
        "correct": sum(correct),
        "avg_gap": (sum(p["total_gap"] for p in placed) / len(placed)) if placed else 0,
    }


def _time_split(rows, holdout_date=HOLDOUT_DATE):
    train, holdout = [], []
    for r in rows:
        d = r.get("match_date") or ""
        if d < holdout_date:
            train.append(r)
        else:
            holdout.append(r)
    return train, holdout


def render_train_holdout(rows):
    st.subheader("🧪 Train / Holdout (walk-forward)")
    st.caption(f"Train: matches before {HOLDOUT_DATE} · Holdout: on/after {HOLDOUT_DATE}")
    st.info("⚠️ Do NOT tune thresholds after seeing holdout. This is for validation only.")

    settled = [r for r in rows if r.get("actual_home_goals") is not None
               and r.get("actual_away_goals") is not None]
    if not settled:
        st.warning("No settled matches. Enter results in the Pending tab first.")
        return

    train, holdout = _time_split(settled)
    st.write(f"Train settled: **{len(train)}** · Holdout settled: **{len(holdout)}**")

    if not holdout:
        st.warning(f"No holdout matches on/after {HOLDOUT_DATE}.")
        return

    v421_train = _evaluate(train, predict_v4_2_1)
    v421_hold = _evaluate(holdout, predict_v4_2_1)
    v43_train = _evaluate(train, predict_v4_3)
    v43_hold = _evaluate(holdout, predict_v4_3)

    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([
        {
            "Model": "v4.2.1", "Split": "Train",
            "Settled": v421_train["n_total"], "Placed": v421_train["n_placed"],
            "NO BET %": f"{v421_train['no_bet_pct']:.0f}%",
            "Accuracy": f"{v421_train['accuracy']:.0f}%",
            "Avg gap": f"{v421_train['avg_gap']:.1f}",
        },
        {
            "Model": "v4.2.1", "Split": "Holdout",
            "Settled": v421_hold["n_total"], "Placed": v421_hold["n_placed"],
            "NO BET %": f"{v421_hold['no_bet_pct']:.0f}%",
            "Accuracy": f"{v421_hold['accuracy']:.0f}%",
            "Avg gap": f"{v421_hold['avg_gap']:.1f}",
        },
        {
            "Model": "v4.3", "Split": "Train",
            "Settled": v43_train["n_total"], "Placed": v43_train["n_placed"],
            "NO BET %": f"{v43_train['no_bet_pct']:.0f}%",
            "Accuracy": f"{v43_train['accuracy']:.0f}%",
            "Avg gap": f"{v43_train['avg_gap']:.1f}",
        },
        {
            "Model": "v4.3", "Split": "Holdout",
            "Settled": v43_hold["n_total"], "Placed": v43_hold["n_placed"],
            "NO BET %": f"{v43_hold['no_bet_pct']:.0f}%",
            "Accuracy": f"{v43_hold['accuracy']:.0f}%",
            "Avg gap": f"{v43_hold['avg_gap']:.1f}",
        },
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Verdict</div>', unsafe_allow_html=True)
    if v43_hold["accuracy"] >= 70 and v43_hold["no_bet_pct"] >= 60:
        st.success(f"✅ v4.3 holdout {v43_hold['accuracy']:.0f}% with {v43_hold['no_bet_pct']:.0f}% NO BET — real edge.")
    elif v43_hold["accuracy"] < 58:
        st.error(f"❌ v4.3 holdout {v43_hold['accuracy']:.0f}% — the 81% was noise.")
    else:
        st.warning(f"🟡 v4.3 holdout {v43_hold['accuracy']:.0f}% at {v43_hold['no_bet_pct']:.0f}% NO BET — inconclusive.")


# ============================================================================
# UI
# ============================================================================
def main():
    st.title("⚽ v4.3 Raw Predictor")
    st.caption("Universal model. One button. Parse → Predict → Save.")

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase connection failed: {diag.get('error')}")
        return

    tabs = st.tabs([
        "📥 Parse & Save",
        "⏳ Pending",
        "📊 Performance",
        "🧪 Train/Holdout",
        "🔍 Data Audit",
        "ℹ️ v4.3 Spec",
    ])

    with tabs[0]:
        st.subheader("Paste Sportsgambler HTML")
        st.caption("Click the button — parse, predict (v4.3), and save happen in one action.")

        text = st.text_area("HTML", height=260, key="html_input", label_visibility="collapsed")

        if st.button("⚽ Parse, Predict & Save (v4.3)", type="primary"):
            if not text or len(text.strip()) < 200:
                st.error("Paste a full Sportsgambler preview page.")
            else:
                with st.spinner("Parsing HTML..."):
                    try:
                        parsed = SportsgamblerParser(text).parse()
                    except Exception as e:
                        st.error(f"Parser failed: {e}")
                        return

                if not parsed.get("home_team") or not parsed.get("away_team"):
                    st.error("Could not extract team names from HTML.")
                    return

                with st.spinner("Running v4.3 model..."):
                    result = predict_v4_3(parsed)

                with st.spinner("Saving to Supabase..."):
                    ok, row = upsert_match(sb, parsed)
                    save_ok = False
                    save_msg = ""
                    if ok and row:
                        match_id = row.get("id")
                        if match_id:
                            save_ok, save_msg = save_prediction(sb, match_id, result)
                    else:
                        save_msg = row if isinstance(row, str) else "unknown error"

                st.markdown("---")
                st.markdown(f"""
                <div class="team-header">
                    <div class="team-names">{parsed['home_team']} &nbsp;🆚&nbsp; {parsed['away_team']}</div>
                    <div class="team-meta">
                        {parsed.get('league_name') or '—'}
                        &nbsp;·&nbsp; {parsed.get('match_date') or '—'}
                        &nbsp;·&nbsp; {parsed.get('kickoff_local') or ''}
                        &nbsp;·&nbsp; {parsed.get('venue') or '—'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if ok and save_ok:
                    st.success(f"✅ Saved to Supabase as PENDING. Match ID: `{row.get('id')}`")
                elif ok and not save_ok:
                    st.warning(f"⚠️ Row saved but prediction update failed: {save_msg}")
                else:
                    st.error(f"❌ Save failed: {save_msg}")

                c1, c2 = st.columns([2, 1])
                with c1:
                    render_verdict(result)
                with c2:
                    render_ou_verdict(result)

                if result.get("tags"):
                    st.markdown("**Tags:** " + " ".join(f"`{t}`" for t in result["tags"]))

                warnings = compute_warnings(parsed, result)
                if warnings:
                    st.markdown("### 🚩 Risk Warnings")
                    for w in warnings:
                        severity = w["severity"]
                        if severity == "high":
                            st.error(f"**{w['label']}**\n\n{w['detail']}")
                        elif severity == "medium":
                            st.warning(f"**{w['label']}**\n\n{w['detail']}")
                        else:
                            st.info(f"**{w['label']}**\n\n{w['detail']}")
                else:
                    st.success("No risk warnings — clean profile.")

                with st.expander("🔍 Data Audit (abstention reasons)"):
                    st.write({
                        "no_bet_reason_1x2": result.get("no_bet_reason_1x2"),
                        "no_bet_reason_ou": result.get("no_bet_reason_ou"),
                        "call_1x2": result.get("call_1x2"),
                        "call_ou": result.get("call_ou"),
                        "bet_market": result.get("bet_market"),
                        "tags": result.get("tags"),
                    })

                with st.expander("🔍 Debug: parse health"):
                    st.write({
                        "home_last5_len": len(parsed.get("home_last5") or []),
                        "away_last5_len": len(parsed.get("away_last5") or []),
                        "home_last10_w": parsed.get("home_last10_w"),
                        "home_last10_d": parsed.get("home_last10_d"),
                        "home_last10_l": parsed.get("home_last10_l"),
                        "away_last10_w": parsed.get("away_last10_w"),
                        "away_last10_d": parsed.get("away_last10_d"),
                        "away_last10_l": parsed.get("away_last10_l"),
                        "home_last10_avg_scored": parsed.get("home_last10_avg_scored"),
                        "away_last10_avg_scored": parsed.get("away_last10_avg_scored"),
                        "home_points": parsed.get("home_points"),
                        "away_points": parsed.get("away_points"),
                        "home_played": parsed.get("home_played"),
                        "away_played": parsed.get("away_played"),
                        "home_home_last10_avg_scored": parsed.get("home_home_last10_avg_scored"),
                        "home_home_last10_avg_conceded": parsed.get("home_home_last10_avg_conceded"),
                        "away_away_last10_avg_scored": parsed.get("away_away_last10_avg_scored"),
                        "away_away_last10_avg_conceded": parsed.get("away_away_last10_avg_conceded"),
                    })

                with st.expander("🔍 Parsed raw values"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Home**")
                        st.write(f"Pos: {parsed.get('home_pos')}, Pts: {parsed.get('home_points')}, GD: {parsed.get('home_gd')}, Played: {parsed.get('home_played')}")
                        st.write(f"Last5 ({len(parsed.get('home_last5') or [])}): {parsed.get('home_last5')}")
                        st.write(f"XI ({len(parsed.get('home_xi') or [])}): {parsed.get('home_xi')}")
                        st.write(f"Injuries ({len(parsed.get('home_injuries') or [])}): {parsed.get('home_injuries')}")
                    with c2:
                        st.write("**Away**")
                        st.write(f"Pos: {parsed.get('away_pos')}, Pts: {parsed.get('away_points')}, GD: {parsed.get('away_gd')}, Played: {parsed.get('away_played')}")
                        st.write(f"Last5 ({len(parsed.get('away_last5') or [])}): {parsed.get('away_last5')}")
                        st.write(f"XI ({len(parsed.get('away_xi') or [])}): {parsed.get('away_xi')}")
                        st.write(f"Injuries ({len(parsed.get('away_injuries') or [])}): {parsed.get('away_injuries')}")
                    st.write(f"H2H: {parsed.get('h2h')}")
                    st.write(f"Kickoff UTC: {parsed.get('kickoff_utc')}")

                st.markdown('<div class="section-title">Factor Breakdown (v4.3)</div>', unsafe_allow_html=True)
                render_factor_row("F1 League Momentum", result["f1_home"], result["f1_away"], "(20 pts)")
                render_factor_row("F2 Current Form (smoothed)", result["f2_home"], result["f2_away"], "(25 pts)")
                render_factor_row("F3 Venue Split", result["f3_home"], result["f3_away"], "(15 pts)")
                render_factor_row("F4 Availability", result["f4_home"], result["f4_away"], "(15 pts)")
                render_factor_row("F5 H2H Psychology (prior-corrected)", result["f5_home"], result["f5_away"], "(10 pts)")
                render_factor_row("F6 Attack Profile", result["f6_home"], result["f6_away"], "(15 pts)")

                st.markdown(f"""
                <div class="factor-row" style="background:#1e3a8a;">
                    <div>
                        <div class="factor-name" style="color:#bfdbfe;">TOTAL (after shrink)</div>
                        <div style="color:#93c5fd; font-size:0.8rem;">
                            Home {result['home_total']:.1f} · Away {result['away_total']:.1f} · Gap {result['total_gap']:.1f}
                            · raw {result['raw_gap']:.1f} · shrink {result['shrink_factor']:.2f} ({result['disagreements']} disagreements)
                        </div>
                    </div>
                    <div class="factor-val" style="color:#bfdbfe;">{result['total_gap']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="section-title">Modifiers & Overrides</div>', unsafe_allow_html=True)
                render_trigger("F1 vs F5 Conflict", result["f1_f5_conflict"])
                render_trigger("F1 vs F5 Override Applied", result["f1_f5_override"])
                render_trigger("F1 vs F2/F3 Conflict (tracking)", result["f1_vs_f2f3_conflict"])
                render_trigger("Away Collapse", result["away_collapse"])
                render_trigger("Doubt Starter IN XI", result["doubted_starter"])

                st.info("👉 Go to **⏳ Pending** tab after the match to enter the actual score.")

    with tabs[1]:
        st.subheader("⏳ Pending Matches")
        rows = load_all(sb)
        pending = [r for r in rows if r.get("actual_home_goals") is None]
        if not pending:
            st.success("No pending matches.")
        else:
            st.write(f"**{len(pending)} pending matches**")
            for r in pending:
                match_id = r["id"]
                call = r.get("call_1x2", "—")
                ou = r.get("call_ou", "—")
                gap = r.get("total_gap", 0)
                tags = r.get("tags") or []
                header = f"{r.get('match_date','')} · {r.get('home_team','')} vs {r.get('away_team','')} · {call} (gap {gap})"
                with st.expander(header):
                    if tags:
                        st.write("**Tags:** " + " ".join(f"`{t}`" for t in tags))
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

    with tabs[2]:
        st.subheader("📊 Performance")
        rows = load_all(sb)
        settled = [r for r in rows if r.get("actual_home_goals") is not None]
        if not settled:
            st.info("No settled matches yet.")
        else:
            placed = [r for r in settled if r.get("bet_market")]
            hits = sum(1 for r in placed if r.get("settled_dc_hit") is True)
            losses = sum(1 for r in placed if r.get("settled_dc_hit") is False)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Settled", len(settled))
            c2.metric("DC bets", len(placed))
            c3.metric("DC hits", f"{hits}/{hits + losses}" if (hits + losses) else "—")
            c4.metric("DC hit rate",
                      f"{(hits / (hits + losses) * 100):.1f}%"
                      if (hits + losses) else "—")

            st.markdown('<div class="section-title">All Settled Matches</div>', unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Date": r.get("match_date"),
                "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                "Gap": r.get("total_gap"),
                "Call": r.get("call_1x2"),
                "Bet": r.get("bet_market"),
                "Tags": ", ".join(r.get("tags") or []),
                "Actual": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                "DC hit": ("✅" if r.get("settled_dc_hit") is True else
                           "❌" if r.get("settled_dc_hit") is False else "—"),
            } for r in settled])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[3]:
        rows = load_all(sb)
        render_train_holdout(rows)

    with tabs[4]:
        st.subheader("🔍 Data Audit")
        st.caption("Per-market abstention reasons and tag coverage.")

        rows = load_all(sb)
        if not rows:
            st.info("No rows loaded.")
        else:
            n_total = len(rows)
            n_ok = sum(1 for r in rows if r.get("parse_status") == "ok")
            n_incomplete = n_total - n_ok
            st.write(f"**Total rows:** {n_total} · **parse_ok:** {n_ok} · **incomplete:** {n_incomplete}")

            st.markdown('<div class="section-title">Tag coverage</div>', unsafe_allow_html=True)
            from collections import defaultdict
            tag_counts = defaultdict(int)
            combo_counts = defaultdict(int)
            for r in rows:
                tags = r.get("tags") or []
                if not tags:
                    continue
                for t in tags:
                    tag_counts[t] += 1
                combo_counts["+".join(sorted(tags))] += 1
            if tag_counts:
                st.dataframe(
                    pd.DataFrame([{"tag": k, "count": v}
                                  for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])]),
                    use_container_width=True, hide_index=True,
                )
                st.markdown('<div class="section-title">Tag combinations</div>', unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame([{"combination": k, "count": v}
                                  for k, v in sorted(combo_counts.items(), key=lambda x: -x[1])]),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.write("No tags found. Existing rows predate the v5.0 rule.")

            st.markdown('<div class="section-title">Skip reasons</div>', unsafe_allow_html=True)
            r1x2 = {}
            for r in rows:
                call = r.get("call_1x2") or ""
                if call.startswith("NO BET"):
                    reason = r.get("no_bet_reason_1x2") or r.get("no_bet_reason") or "(unrecorded)"
                    r1x2[reason] = r1x2.get(reason, 0) + 1
            if r1x2:
                st.dataframe(
                    pd.DataFrame([{"reason": k, "count": v} for k, v in sorted(r1x2.items(), key=lambda x: -x[1])]),
                    use_container_width=True, hide_index=True,
                )

    with tabs[5]:
        st.subheader("v4.3 Specification (with v5.0 rule layer)")
        st.markdown(f"""
        **100-point model. Only prematch fields. No xG, no possession.**

        | Factor | Weight | v4.3 change |
        |--------|--------|-------------|
        | F1 League Momentum | 20 | unchanged |
        | F2 Current Form Last 5 | 25 | **smoothed**: `(W + 0.4D + ALPHA*0.4)/(games+ALPHA) * 25` |
        | F3 Venue Split | 15 | unchanged |
        | F4 Availability | 15 | unchanged |
        | F5 H2H Psychology | 10 | **prior-corrected**: `(W + 1/3)/(total+1) * 10` |
        | F6 Attack Profile | 15 | unchanged |

        **Constants:**
        - `ALPHA = {ALPHA}`
        - `PRIOR_FORM = {PRIOR_FORM}`
        - `PRIOR_H2H = 1/3`

        **v5.0 Decision rule:**
        - Skip if `total_gap < 20`
        - Skip if `f1_f5_conflict` fires without override
        - Skip if `f1_vs_f2f3_conflict` fires
        - Skip if a top scorer is injured AND not in XI
        - Otherwise bet **DC 1X** (home leader) or **DC X2** (away leader)

        **Tags attached to every pick:**
        - `gap_20` / `gap_below_20`
        - `home_leader` / `away_leader`
        - `team_disagreement` / `team_agreement`
        - `venue_incomplete` / `venue_complete`

        **Tag rule:**
        - No tag becomes a filter until it has 50 settled picks.

        **O/U (v4.3):**
        - `< 2.4 Under` · `> 3.2 Over`
        - Middle band → NO BET

        **Validation:**
        - Time split only, never random
        - Train: matches before `{HOLDOUT_DATE}`
        - Holdout: matches on/after `{HOLDOUT_DATE}`
        """)


main()
