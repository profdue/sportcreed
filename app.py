"""
v4 RAW-ONLY Predictor — Streamlit app with HTML parser.

Single-button workflow:
  Paste HTML → click "Parse, Predict & Save" → done.
  Parser runs, model predicts, row upserted to matches_raw (pending).
  No separate save button. No duplicate saves.

Bug fixes (v4):
  #1  kickoff_local stored
  #2  GF:GA parsed from GF:GA column
  #3  Corners from corner blocks, not .st-table
  #4  Both home/away venue averages extracted
  #5  home_home_win_pct derived from split table
  #6  expected_return parsed
  #7  _parse_xi scoped fallback
  #8  away_collapse threshold per spec
  #9  AH/DNB/DC/HT odds parsed
  #10 "Assistors" typo handled
"""

import os
import re
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
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

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

    def parse(self):
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        league, tier, group = self._parse_league()
        venue = self._parse_venue()

        record = {
            "match_date": match_date,
            "kickoff_utc": None,
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
        for link in self.soup.select(".t_top .t_info_link"):
            text = link.get_text(strip=True)
            if text.lower() == "football":
                continue
            if re.search(r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Championship|MLS|Cup|Division|Primera|Nations)", text, re.I):
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
                team_clean = re.sub(
                    r"\s*logo\s*", " ",
                    cells[team_idx].get_text(strip=True),
                    flags=re.I,
                ).strip()

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

                if self._team_matches(self.home_team, team_clean):
                    out["home_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["home_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    if gf is not None:
                        out["home_gf"] = gf
                        out["home_ga"] = ga
                    if gd is not None:
                        out["home_gd"] = gd
                elif self._team_matches(self.away_team, team_clean):
                    out["away_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["away_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    if gf is not None:
                        out["away_gf"] = gf
                        out["away_ga"] = ga
                    if gd is not None:
                        out["away_gd"] = gd

        self._parse_split_table("#Awayleague", self.away_team, "away", out)
        self._parse_split_table("#Homeleague", self.home_team, "home", out)
        return out

    def _parse_split_table(self, container_selector, team_name, side_prefix, out):
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
            team_clean = re.sub(
                r"\s*logo\s*", " ",
                cells[team_idx].get_text(strip=True),
                flags=re.I,
            ).strip()
            if not self._team_matches(team_name, team_clean):
                continue
            played = self._to_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None
            won = self._to_int(cells[3].get_text(strip=True)) if len(cells) > 3 else None
            pts = self._to_int(cells[pts_idx].get_text(strip=True))
            if side_prefix == "away":
                out["away_away_points"] = pts
                out["away_away_played"] = played
                if played and won is not None:
                    out["away_away_win_pct"] = (won / played) * 100
            else:
                out["home_home_points"] = pts
                out["home_home_played"] = played
                if played and won is not None:
                    out["home_home_win_pct"] = (won / played) * 100
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
                continue
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
            out.append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "opp": away_name if is_home else home_name,
                "result": result,
                "score_for": sf,
                "score_against": sa,
                "is_home": is_home,
            })
        return out

    def _parse_last10(self, side):
        out = {}
        prefix = "home" if side == "home" else "away"
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
        if w_key in out:
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
        target = self.home_team if side == "home" else self.away_team
        side_class = "lineups-home" if side == "home" else "lineups-away"
        for header in self.soup.select(".lineups-formation h3, .lineups-mob-teams h3"):
            text = header.get_text(" ", strip=True)
            if not self._team_matches(target, text):
                continue
            parent = header.find_parent()
            if not parent:
                continue
            container = parent.find_next_sibling(class_="lineups")
            if not container:
                container = parent.find_parent(class_="lineups")
            if not container:
                container = header.find_parent(class_="content-block") or header.find_parent(class_="lineups")
            if not container:
                continue
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
# v4 MODEL
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
    if not last5:
        return 7.5
    raw = 0
    for m in last5[:5]:
        r = m.get("result", "L")
        if r == "W":
            raw += 5
        elif r == "D":
            raw += 2
    return max(0, min(25, raw))


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
        allowed = {
            "match_date", "kickoff_utc", "kickoff_local", "league_name", "tier", "group_name",
            "season", "home_team", "away_team", "venue", "stage", "round",
            "home_pos", "home_played", "home_points", "home_gd", "home_gf", "home_ga",
            "away_pos", "away_played", "away_points", "away_gd", "away_gf", "away_ga",
            "home_away_points", "home_away_played", "away_away_points", "away_away_played",
            "home_home_points", "home_home_played", "home_home_win_pct",
            "away_away_win_pct",
            "home_last5", "away_last5",
            "home_last10_w", "home_last10_d", "home_last10_l",
            "home_last10_avg_scored", "home_last10_avg_conceded",
            "home_last10_possession", "home_last10_corners_for",
            "home_last10_corners_against", "home_last10_win_pct",
            "home_home_last10_avg_scored", "home_home_last10_avg_conceded",
            "home_home_last10_corners_for", "home_home_last10_corners_against",
            "away_last10_w", "away_last10_d", "away_last10_l",
            "away_last10_avg_scored", "away_last10_avg_conceded",
            "away_last10_possession", "away_last10_corners_for",
            "away_last10_corners_against", "away_last10_win_pct",
            "away_away_last10_avg_scored", "away_away_last10_avg_conceded",
            "away_away_last10_corners_for", "away_away_last10_corners_against",
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
    if call == "NO BET":
        st.markdown(f"""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">1X2 Verdict</div>
            <div class="verdict-noedge">NO BET — gap {result['total_gap']:.1f} below 15</div>
            <div class="verdict-detail-grey">Total gap: {result['total_gap']:.1f} (needs 15+ for a call)</div>
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
    st.caption("One button. Parse → Predict → Save. Done.")

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase connection failed: {diag.get('error')}")
        return

    tabs = st.tabs(["📥 Parse & Save", "⏳ Pending", "📊 Performance", "ℹ️ v4 Spec"])

    # ---------------------------------------------------- parse & save
    with tabs[0]:
        st.subheader("Paste Sportsgambler HTML")
        st.caption("Click the button — the app parses, predicts, and saves to Supabase in one action.")

        text = st.text_area("HTML", height=260, key="html_input", label_visibility="collapsed")

        if st.button("⚽ Parse, Predict & Save", type="primary"):
            if not text or len(text.strip()) < 200:
                st.error("Paste a full Sportsgambler preview page.")
            else:
                # Step 1: Parse
                with st.spinner("Parsing HTML..."):
                    try:
                        parsed = SportsgamblerParser(text).parse()
                    except Exception as e:
                        st.error(f"Parser failed: {e}")
                        return

                if not parsed.get("home_team") or not parsed.get("away_team"):
                    st.error("Could not extract team names from HTML.")
                    return

                # Step 2: Predict
                with st.spinner("Running v4 model..."):
                    result = predict_v4(parsed)

                # Step 3: Save
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

                # Step 4: Display
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

                # Save status banner
                if ok and save_ok:
                    st.success(f"✅ Saved to Supabase as PENDING. Match ID: `{row.get('id')}`")
                elif ok and not save_ok:
                    st.warning(f"⚠️ Row saved but prediction update failed: {save_msg}")
                else:
                    st.error(f"❌ Save failed: {save_msg}")

                # Verdict
                c1, c2 = st.columns([2, 1])
                with c1:
                    render_verdict(result)
                with c2:
                    render_ou_verdict(result)

                # Debug expander
                with st.expander("🔍 Parsed raw values"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("**Home**")
                        st.write(f"Pos: {parsed.get('home_pos')}, Pts: {parsed.get('home_points')}, GD: {parsed.get('home_gd')}, GF: {parsed.get('home_gf')}, GA: {parsed.get('home_ga')}")
                        st.write(f"Last5: {parsed.get('home_last5')}")
                        st.write(f"Last10: W{parsed.get('home_last10_w')} D{parsed.get('home_last10_d')} L{parsed.get('home_last10_l')}")
                        st.write(f"Home venue avg: {parsed.get('home_home_last10_avg_scored')} / {parsed.get('home_home_last10_avg_conceded')}")
                        st.write(f"Home win%: {parsed.get('home_home_win_pct')}")
                        st.write(f"XI: {parsed.get('home_xi')}")
                        st.write(f"Injuries: {parsed.get('home_injuries')}")
                    with c2:
                        st.write("**Away**")
                        st.write(f"Pos: {parsed.get('away_pos')}, Pts: {parsed.get('away_points')}, GD: {parsed.get('away_gd')}, GF: {parsed.get('away_gf')}, GA: {parsed.get('away_ga')}")
                        st.write(f"Last5: {parsed.get('away_last5')}")
                        st.write(f"Last10: W{parsed.get('away_last10_w')} D{parsed.get('away_last10_d')} L{parsed.get('away_last10_l')}")
                        st.write(f"Away venue avg: {parsed.get('away_away_last10_avg_scored')} / {parsed.get('away_away_last10_avg_conceded')}")
                        st.write(f"Away win%: {parsed.get('away_away_win_pct')}")
                        st.write(f"XI: {parsed.get('away_xi')}")
                        st.write(f"Injuries: {parsed.get('away_injuries')}")
                    st.write(f"Away split: pts {parsed.get('away_away_points')}, played {parsed.get('away_away_played')}")
                    st.write(f"H2H: {parsed.get('h2h')}")

                # Factor breakdown
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
                            Home {result['home_total']:.1f} · Away {result['away_total']:.1f} · Gap {result['total_gap']:.1f}
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

                st.info("👉 Go to **⏳ Pending** tab after the match to enter the actual score.")

    # ------------------------------------------------------------- pending
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
                triggers = []
                if r.get("f1_f5_override"): triggers.append("override")
                if r.get("away_collapse"): triggers.append("collapse")
                if r.get("doubted_starter"): triggers.append("doubt")
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
            st.info("No settled matches yet.")
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
                      f"{correct}/{len(placed)} ({(correct/len(placed)*100):.0f}%)" if placed else "—")
            c4.metric("Override accuracy",
                      f"{override_correct}/{len(override_games)}" if override_games else "—")
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
        **100-point model. Only prematch fields. No xG, no possession.**

        | Factor | Weight | Source |
        |--------|--------|--------|
        | F1 League Momentum | 20 | Position, Points, GD |
        | F2 Current Form Last 5 | 25 | W=5, D=2, L=0, max 25 |
        | F3 Venue Split | 15 | Win% + goal edge at venue |
        | F4 Availability | 15 | Injuries (not suspensions) vs Confirmed XI |
        | F5 H2H Psychology | 10 | H2H W/D/L |
        | F6 Attack Profile | 15 | Avg goals scored |

        **Decision rule:**
        - Gap > 25 → Straight Win
        - Gap 15–25 → Double Chance
        - Gap < 15 → NO BET

        **Overrides:**
        - F1 vs F5 conflict: F1 gap ≥ 12 AND F5 gap ≥ 6 AND opposite → F1 wins
        - Away collapse: away team 0 pts from 3+ away games → home +8 F1, force Over 2.5
        - Doubt starter: doubted key player in confirmed XI → F4 +5, force Over 2.5

        **O/U:** Expected = (HomeH avg scored + AwayA avg conceded + AwayA avg scored + HomeH avg conceded) / 2
        - < 2.5 → Under
        - > 3.3 → Over
        - 2.5–3.3 → No Bet
        """)


main()
