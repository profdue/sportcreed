#!/usr/bin/env python3
"""
king.py — The Logic.

Input HTML. Extract facts. Run the logic. Show the pick.
Send to Supabase. Track pending. Settle later. See performance.

One file. Two clauses. Three tabs.
"""

import re
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from supabase import create_client, Client

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Logic",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; letter-spacing: -0.02em; }
    .match-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3d 100%);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip      { border-left-color: #64748b; opacity: 0.85; }
    .match-card.win       { border-left-color: #10b981; }
    .match-card.loss      { border-left-color: #ef4444; }
    .match-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 4px 0;
    }
    .match-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    .pick-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .badge-home { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-away { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win  { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .signal-pill {
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 8px;
        background: #0f172a;
        color: #cbd5e1;
        border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .metric-box {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #232a3d;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 0 18px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def db_insert(row: dict):
    sb = get_supabase()
    res = sb.table("matches").insert(row).execute()
    return res.data[0] if res.data else None


def db_pending():
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .is_("home_score", "null")
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_settled():
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .not_.is_("home_score", "null")
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_settle(match_id: int, home_score: int, away_score: int,
              result: str, dc_hit: int, profit_loss: float):
    sb = get_supabase()
    sb.table("matches").update({
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "dc_hit": dc_hit,
        "profit_loss": profit_loss,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("match_id", match_id).execute()


def db_delete(match_id: int):
    sb = get_supabase()
    sb.table("matches").delete().eq("match_id", match_id).execute()


# ═══════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════

_TEAM_PREFIXES = (
    "club ", "atlético ", "atletico ", "deportivo ", "real ",
    "fc ", "cf ", "cd ", "ca ", "ac ", "as ", "sv ", "sc ",
    "athletic ", "sporting ", "racing ", "olympique ", "stade ",
)

_TEAM_SUFFIXES = (
    " sco", " fc", " cf", " sc", " ac", " cd", " sv", " as",
    " 29", " 04", " 05", " 1899", " 1900",
)


def _search_term(full_name: str) -> str:
    """
    Return a short search term for a team name.
    'Atlético San Luis' → 'San Luis'
    'Club Santos Laguna' → 'Santos Laguna'
    'Stade Brestois 29' → 'Brestois'
    'Angers SCO' → 'Angers'
    'FC Famalicão' → 'Famalicão'
    """
    if not full_name:
        return ""
    name = full_name.strip()
    low = name.lower()
    for prefix in _TEAM_PREFIXES:
        if low.startswith(prefix):
            name = name[len(prefix):]
            break
    low = name.lower()
    for suffix in _TEAM_SUFFIXES:
        if low.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.split()
    if len(parts) >= 2:
        if len(parts[0]) <= 3:
            return parts[-1]
        return " ".join(parts[:2])
    return name


def _candidates(term):
    """Return candidate search terms for a team name (full, first word, cleaned)."""
    if not term:
        return []
    out = [term]
    first = term.split()[0]
    if first != term:
        out.append(first)
    clean = re.sub(r"\s+\d+$", "", term).strip()
    clean = re.sub(r"\s+(SCO|FC|CF|SC|AC|CD|SV|AS)$", "",
                   clean, flags=re.IGNORECASE).strip()
    if clean and clean not in out:
        out.append(clean)
    if clean and " " in clean:
        first2 = clean.split()[0]
        if first2 not in out:
            out.append(first2)
    return out


def _team_match(full_name, cell_text):
    if not full_name or not cell_text:
        return False
    term = _search_term(full_name).lower()
    cell = cell_text.lower()
    if not term:
        return False
    if term in cell or cell in term:
        return True
    term_words = set(term.split())
    cell_words = set(cell.split())
    return bool(term_words & cell_words)


def _word_to_int(s):
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "single": 1, "a": 1, "an": 1}
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().lower()
    if s.isdigit():
        return int(s)
    return words.get(s, 0)


def _int_any(text):
    if text is None:
        return None
    m = re.search(r"-?\d+", str(text))
    if m:
        return int(m.group())
    return _word_to_int(text)


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — TWO CLAUSES
# ═══════════════════════════════════════════════════════════════════

def compute_signals(f: dict) -> dict:
    table = "home" if (f.get("home_table_pts", 0), f.get("home_table_gd", 0)) > \
                      (f.get("away_table_pts", 0), f.get("away_table_gd", 0)) else "away"
    form  = "home" if f.get("home_form_ppg", 0) > f.get("away_form_ppg", 0) else "away"
    venue = "home" if f.get("home_home_ppg", 0) > f.get("away_away_ppg", 0) else "away"

    if f.get("h2h_home_wins", 0) > f.get("h2h_away_wins", 0):
        h2h = "home"
    elif f.get("h2h_away_wins", 0) > f.get("h2h_home_wins", 0):
        h2h = "away"
    else:
        h2h = "nothing"

    return {"table": table, "form": form, "venue": venue, "h2h": h2h}


def decide(signals: dict) -> dict:
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")
    table_leader = signals["table"]

    if home == 4 or away == 4:
        direction = "home" if home == 4 else "away"
        return {
            "direction": direction, "stake": "full", "qualifying": 1,
            "dissent_signal": None, "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if (home == 3 and away == 1) or (away == 3 and home == 1):
        dissent = next((k for k, v in signals.items()
                        if v != table_leader and v != "nothing"), None)
        return {
            "direction": table_leader, "stake": "half", "qualifying": 1,
            "dissent_signal": dissent, "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if home == 2 and away == 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": "2-2",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if (home + away) < 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": f"{home}-{away}-{nothing}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    split = f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}"
    return {
        "direction": "skip", "stake": "none", "qualifying": 0,
        "dissent_signal": None, "split": split,
        "home_count": home, "away_count": away, "nothing_count": nothing,
    }


def settle(direction, stake, home_score, away_score):
    if home_score > away_score:
        result = "home_win"
    elif away_score > home_score:
        result = "away_win"
    else:
        result = "draw"

    if direction == "skip":
        return result, None, 0.0, 0.0

    units = {"full": 1.0, "half": 0.5, "quarter": 0.25, "none": 0.0}.get(stake, 0.0)

    if direction == "home":
        hit = result in ("home_win", "draw")
    elif direction == "away":
        hit = result in ("away_win", "draw")
    else:
        hit = False

    profit = units if hit else -units
    return result, (1 if hit else 0), units, profit


# ═══════════════════════════════════════════════════════════════════
# HTML EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_facts(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    f = {"_parse_warnings": []}
    body = soup.get_text(" ", strip=True)

    # --- identity ---------------------------------------------------
    f["match_date"] = f["kickoff_time"] = None
    f["league"] = f["country"] = f["venue"] = None
    f["home_team"] = f["away_team"] = None

    t_top = soup.find("div", class_="t_top")
    if t_top:
        teams = t_top.find_all("div", class_="t_name")
        if len(teams) >= 2:
            f["home_team"] = teams[0].get_text(strip=True)
            f["away_team"] = teams[1].get_text(strip=True)
        info = t_top.find("div", class_="t_info")
        if info:
            links = info.find_all("a", class_="t_info_link")
            if len(links) >= 2:
                league_txt = links[1].get_text(strip=True)
                if " - " in league_txt:
                    _, league_txt = league_txt.split(" - ", 1)
                f["league"] = league_txt
            venue = info.find("span", class_="t_venue")
            if venue:
                f["venue"] = venue.get_text(strip=True)
            date = info.find("span", class_="t_date")
            if date:
                txt = date.get_text(" ", strip=True)
                m = re.search(r"(\d{1,2}\s+\w+)", txt)
                if m:
                    f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m:
                    f["kickoff_time"] = m.group(1)
    else:
        f["_parse_warnings"].append("t_top block not found")

    home_term = _search_term(f.get("home_team", ""))
    away_term = _search_term(f.get("away_team", ""))

    # --- standings table (overall) ----------------------------------
    # Standard SportsGambler standings row has 11 cells:
    #   pos | team | G | W | D | L | GF | GA | GD | "GF:GA" | P
    standings_found = False
    for table in soup.find_all("table", class_="clt-table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 11:
                continue
            team_cell = cells[1]
            if _team_match(f.get("home_team", ""), team_cell):
                f["home_table_pos"]    = _int_any(cells[0])
                f["home_table_played"] = _int_any(cells[2])
                f["home_table_gf"]     = _int_any(cells[6])
                f["home_table_ga"]     = _int_any(cells[7])
                f["home_table_gd"]     = _int_any(cells[8])
                f["home_table_pts"]    = _int_any(cells[10])
                standings_found = True
            elif _team_match(f.get("away_team", ""), team_cell):
                f["away_table_pos"]    = _int_any(cells[0])
                f["away_table_played"] = _int_any(cells[2])
                f["away_table_gf"]     = _int_any(cells[6])
                f["away_table_ga"]     = _int_any(cells[7])
                f["away_table_gd"]     = _int_any(cells[8])
                f["away_table_pts"]    = _int_any(cells[10])
                standings_found = True

    if not standings_found:
        for table in soup.find_all("table", class_="leage-table"):
            for row in table.find_all("tr"):
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) < 8:
                    continue
                team_cell = cells[1]
                if _team_match(f.get("home_team", ""), team_cell):
                    f["home_table_pos"]    = f.get("home_table_pos")    or _int_any(cells[0])
                    f["home_table_played"] = f.get("home_table_played") or _int_any(cells[2])
                    f["home_table_gf"]     = f.get("home_table_gf")     or _int_any(cells[4])
                    f["home_table_ga"]     = f.get("home_table_ga")     or _int_any(cells[5])
                    f["home_table_gd"]     = f.get("home_table_gd")     or _int_any(cells[6])
                    f["home_table_pts"]    = f.get("home_table_pts")    or _int_any(cells[7])
                    standings_found = True
                elif _team_match(f.get("away_team", ""), team_cell):
                    f["away_table_pos"]    = f.get("away_table_pos")    or _int_any(cells[0])
                    f["away_table_played"] = f.get("away_table_played") or _int_any(cells[2])
                    f["away_table_gf"]     = f.get("away_table_gf")     or _int_any(cells[4])
                    f["away_table_ga"]     = f.get("away_table_ga")     or _int_any(cells[5])
                    f["away_table_gd"]     = f.get("away_table_gd")     or _int_any(cells[6])
                    f["away_table_pts"]    = f.get("away_table_pts")    or _int_any(cells[7])
                    standings_found = True

    if not standings_found:
        f["_parse_warnings"].append("standings table: no matching team rows found")

    # --- split body into sentences ----------------------------------
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]

    def _team_sentences(term):
        """Return all sentences that begin with the team name (any candidate)."""
        result = []
        seen = set()
        for cand in _candidates(term):
            cand_low = cand.lower()
            for s in sentences:
                if id(s) in seen:
                    continue
                s_low = s.lower()
                if s_low.startswith(cand_low) or cand_low in s_low[:40]:
                    result.append(s)
                    seen.add(id(s))
        return result

    # --- form parsing ----------------------------------------------
    def _parse_form_from_sentences(sents):
        """Look for a sentence with the form pattern (team's last 10)."""
        patterns = [
            (r"(\d+)\s+wins?,?\s+(\d+)\s+draws?,?\s+and\s+(\d+)\s+losses?", ("w", "d", "l")),
            (r"(\d+)\s+wins?,?\s+(\d+)\s+losses?,?\s+and\s+(\d+)\s+draws?", ("w", "l", "d")),
            (r"(\d+)\s+wins?,?\s+(\d+)\s+defeats?,?\s+and\s+(\d+)\s+draws?", ("w", "l", "d")),
            (r"won\s+(\d+),?\s+lost\s+(\d+)\s+and\s+drawn\s+(\d+)", ("w", "l", "d")),
            (r"won\s+(\d+),?\s+drawn\s+(\d+)\s+and\s+lost\s+(\d+)", ("w", "d", "l")),
            (r"(\d+)\s+wins?,?\s+(\d+)\s+losses?,?\s+(\d+)\s+draws?", ("w", "l", "d")),
        ]
        for s in sents:
            for tmpl, mapping in patterns:
                m = re.search(tmpl, s, re.IGNORECASE)
                if not m:
                    continue
                vals = {}
                for i, key in enumerate(mapping):
                    vals[key] = int(m.group(i + 1))
                return vals["w"], vals["d"], vals["l"]
        return None

    home_sents = _team_sentences(home_term)
    home_form = _parse_form_from_sentences(home_sents)
    if home_form:
        f["home_form_wins"], f["home_form_draws"], f["home_form_losses"] = home_form
        f["home_form_pts"] = f["home_form_wins"] * 3 + f["home_form_draws"]
        f["home_form_ppg"] = round(f["home_form_pts"] / 10, 2)
    else:
        f["_parse_warnings"].append(f"home form not found for {f.get('home_team')}")

    away_sents = _team_sentences(away_term)
    away_form = _parse_form_from_sentences(away_sents)
    if away_form:
        f["away_form_wins"], f["away_form_draws"], f["away_form_losses"] = away_form
        f["away_form_pts"] = f["away_form_wins"] * 3 + f["away_form_draws"]
        f["away_form_ppg"] = round(f["away_form_pts"] / 10, 2)
    else:
        f["_parse_warnings"].append(f"away form not found for {f.get('away_team')}")

    # --- venue form (from same team-sentence lists) -----------------
    def _parse_venue_from_sentences(sents):
        """Look for a sentence with venue stats (wins/losses/draws in any order)."""
        win_words  = r"(?:wins?|victor(?:y|ies)|won)"
        loss_words = r"(?:losses?|defeats?|lost)"
        draw_words = r"(?:draws?|drawn)"

        patterns = [
            (rf"(\d+)\s+{win_words},?\s+(\d+)\s+{loss_words}\s+and\s+(\d+)\s+{draw_words}", ("w", "l", "d")),
            (rf"(\d+)\s+{loss_words},?\s+(\d+)\s+{win_words}\s+and\s+(\d+)\s+{draw_words}", ("l", "w", "d")),
            (rf"(\d+)\s+{win_words},?\s+(\d+)\s+{draw_words}\s+and\s+(\d+)\s+{loss_words}", ("w", "d", "l")),
            (rf"(\d+)\s+{loss_words},?\s+(\d+)\s+{draw_words}\s+and\s+(\d+)\s+{win_words}", ("l", "d", "w")),
            (rf"(\d+)\s+{draw_words},?\s+(\d+)\s+{win_words}\s+and\s+(\d+)\s+{loss_words}", ("d", "w", "l")),
            (rf"(\d+)\s+{draw_words},?\s+(\d+)\s+{loss_words}\s+and\s+(\d+)\s+{win_words}", ("d", "l", "w")),
        ]
        for s in sents:
            for tmpl, mapping in patterns:
                m = re.search(tmpl, s, re.IGNORECASE)
                if not m:
                    continue
                vals = {"w": 0, "d": 0, "l": 0}
                for i, key in enumerate(mapping):
                    vals[key] = int(m.group(i + 1))
                return vals["w"], vals["d"], vals["l"]
        return None

    # Home venue: look in home team sentences for a venue-style sentence
    # (venue sentences use "in the last 10 home games" or similar wording,
    # but our form parser also matches them, so we look for a DIFFERENT
    # sentence from the one we used for form).
    def _venue_sentence(sents):
        """Find a sentence that mentions home/away split."""
        out = []
        for s in sents:
            s_low = s.lower()
            if ("in the last 10 home" in s_low or
                "in the last 10 away" in s_low or
                "in the previous 10 home" in s_low or
                "in the previous 10 away" in s_low or
                "in the past 10 home" in s_low or
                "in the past 10 away" in s_low):
                out.append(s)
        return out

    home_venue_sents = _venue_sentence(home_sents)
    home_venue = _parse_venue_from_sentences(home_venue_sents) if home_venue_sents else None
    if home_venue:
        w, d, l = home_venue
        pts = w * 3 + d
        f["home_home_pts"]    = pts
        f["home_home_ppg"]    = round(pts / 10, 2)
        f["home_home_wins"]   = w
        f["home_home_draws"]  = d
        f["home_home_losses"] = l
    else:
        f["_parse_warnings"].append("home venue form not found")

    away_venue_sents = _venue_sentence(away_sents)
    away_venue = _parse_venue_from_sentences(away_venue_sents) if away_venue_sents else None
    if away_venue:
        w, d, l = away_venue
        pts = w * 3 + d
        f["away_away_pts"]    = pts
        f["away_away_ppg"]    = round(pts / 10, 2)
        f["away_away_wins"]   = w
        f["away_away_draws"]  = d
        f["away_away_losses"] = l
    else:
        f["_parse_warnings"].append("away venue form not found")

    # --- H2H from the H2H table -----------------------------------
    h2h_section = soup.find("div", id="head-to-head")
    h2h_found = False
    if h2h_section:
        h2h_rows = h2h_section.find_all("li", class_="team-stat-list-item")
        home_wins = away_wins = draws = 0
        for row in h2h_rows:
            text = row.get_text(" ", strip=True)
            nums = re.findall(r"\b(\d+)\b", text)
            if len(nums) < 2:
                continue
            try:
                hs = int(nums[-2])
                aws = int(nums[-1])
            except ValueError:
                continue
            idx_home = text.lower().find((home_term or "").lower())
            idx_away = text.lower().find((away_term or "").lower())
            home_listed_first = (idx_home != -1 and
                                 (idx_away == -1 or idx_home < idx_away))
            if hs > aws:
                if home_listed_first:
                    home_wins += 1
                else:
                    away_wins += 1
            elif aws > hs:
                if home_listed_first:
                    away_wins += 1
                else:
                    home_wins += 1
            else:
                draws += 1
        f["h2h_home_wins"] = home_wins
        f["h2h_away_wins"] = away_wins
        f["h2h_draws"] = draws
        f["h2h_total"] = home_wins + away_wins + draws
        h2h_found = (f["h2h_total"] > 0)

    if not h2h_found:
        f["_parse_warnings"].append("H2H not found")

    # --- defaults so the logic can run -----------------------------
    for k in [
        "home_table_pts", "away_table_pts", "home_table_gd", "away_table_gd",
        "home_form_ppg", "away_form_ppg", "home_home_ppg", "away_away_ppg",
        "h2h_home_wins", "h2h_away_wins",
    ]:
        f.setdefault(k, 0)

    return f


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

def signal_pill(name, value):
    cls = value if value in ("home", "away", "nothing") else "nothing"
    label = {"home": "H", "away": "A", "nothing": "—"}.get(value, "—")
    return f'<span class="signal-pill {cls}">{name}: {label}</span>'


def render_pending_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    direction = m.get("direction", "skip")
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "")
    date = m.get("match_date", "")

    if direction == "home":
        card_cls, badge_cls = "pick-home", "badge-home"
        pick_text = f"HOME — {home} (DC 1X)"
    elif direction == "away":
        card_cls, badge_cls = "pick-away", "badge-away"
        pick_text = f"AWAY — {away} (DC X2)"
    else:
        card_cls, badge_cls = "skip", "badge-skip"
        pick_text = "SKIP"

    pills = "".join([
        signal_pill("Table", m.get("signal_table", "nothing")),
        signal_pill("Form",  m.get("signal_form",  "nothing")),
        signal_pill("Venue", m.get("signal_venue", "nothing")),
        signal_pill("H2H",   m.get("signal_h2h",   "nothing")),
    ])

    st.markdown(f"""
    <div class="match-card {card_cls}">
        <div class="match-title">{home} vs {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge_cls}">{pick_text}</span>
        <div class="signal-row">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def render_settled_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    hs = m.get("home_score")
    aws = m.get("away_score")
    dc_hit = m.get("dc_hit")
    profit = m.get("profit_loss") or 0.0
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "")
    date = m.get("match_date", "")

    if dc_hit == 1:
        cls, badge, verdict = "win", "badge-win", "WIN"
    elif dc_hit == 0:
        cls, badge, verdict = "loss", "badge-loss", "LOSS"
    else:
        cls, badge, verdict = "skip", "badge-skip", "NO BET"

    pl = f"{profit:+.2f}u" if dc_hit is not None else "—"

    st.markdown(f"""
    <div class="match-card {cls}">
        <div class="match-title">{home} {hs}–{aws} {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge}">{verdict} · {pl}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# COLUMN MAPPERS
# ═══════════════════════════════════════════════════════════════════

def signals_to_columns(signals):
    return {
        "signal_table": signals["table"],
        "signal_form":  signals["form"],
        "signal_venue": signals["venue"],
        "signal_h2h":   signals["h2h"],
    }


def decision_columns(d):
    return {
        "direction":               d["direction"],
        "stake":                   d["stake"],
        "qualifying":              d["qualifying"],
        "dissent_signal":          d["dissent_signal"],
        "agreement_split":         d["split"],
        "agreement_home_count":    d["home_count"],
        "agreement_away_count":    d["away_count"],
        "agreement_nothing_count": d["nothing_count"],
        "table_leader":            d["direction"] if d["direction"] != "skip" else None,
    }


def build_row(facts, signals, decision):
    row = dict(facts)
    row.pop("_parse_warnings", None)
    row.update(signals_to_columns(signals))
    row.update(decision_columns(decision))
    row["extracted_at"] = datetime.now(timezone.utc).isoformat()

    allowed = {
        "match_date", "kickoff_time", "league", "country", "venue",
        "home_team", "away_team",
        "home_table_pos", "away_table_pos", "home_table_pts", "away_table_pts",
        "home_table_gd", "away_table_gd", "home_table_gf", "away_table_gf",
        "home_table_ga", "away_table_ga", "home_table_played", "away_table_played",
        "home_home_played", "home_home_wins", "home_home_draws", "home_home_losses",
        "home_home_gf", "home_home_ga", "home_home_pts", "home_home_ppg",
        "away_away_played", "away_away_wins", "away_away_draws", "away_away_losses",
        "away_away_gf", "away_away_ga", "away_away_pts", "away_away_ppg",
        "home_form_wins", "home_form_draws", "home_form_losses", "home_form_pts",
        "home_form_ppg", "home_form_gf_per_game", "home_form_ga_per_game",
        "home_form_shots_on_target_per_game", "home_form_attempts_per_game",
        "home_form_corners_per_game", "home_form_possession",
        "away_form_wins", "away_form_draws", "away_form_losses", "away_form_pts",
        "away_form_ppg", "away_form_gf_per_game", "away_form_ga_per_game",
        "away_form_shots_on_target_per_game", "away_form_attempts_per_game",
        "away_form_corners_per_game", "away_form_possession",
        "home_conversion_rate", "away_conversion_rate",
        "home_draw_rate", "away_draw_rate",
        "h2h_total", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_last_result",
        "odds_home", "odds_draw", "odds_away", "odds_home_implied", "odds_away_implied",
        "signal_table", "signal_form", "signal_venue", "signal_h2h",
        "agreement_split", "agreement_home_count", "agreement_away_count",
        "agreement_nothing_count", "table_leader", "dissent_signal",
        "direction", "pick", "stake", "qualifying",
        "source_url", "extracted_at", "notes",
    }
    return {k: v for k, v in row.items() if k in allowed}


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_perf = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance"])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — NEW MATCH
# ─────────────────────────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match HTML")
    st.caption("Paste the full SportsGambler preview HTML. The app extracts the facts, runs the logic, and shows the pick.")

    html_in = st.text_area(
        "HTML", height=220, label_visibility="collapsed",
        placeholder="<html> ... </html>"
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run = st.button("Run the Logic", type="primary", use_container_width=True)

    if run and html_in.strip():
        with st.spinner("Extracting facts..."):
            try:
                facts = extract_facts(html_in)
                signals = compute_signals(facts)
                decision = decide(signals)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

        st.markdown("### The Pick")
        preview = {
            **facts,
            **signals_to_columns(signals),
            **decision_columns(decision),
        }
        render_pending_card(preview)

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠️ Parser warnings ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.write(f"- {w}")

        st.markdown("### Extracted Facts")
        display_facts = {k: v for k, v in facts.items()
                         if v is not None and not k.startswith("_")}
        st.json(display_facts)

        row = build_row(facts, signals, decision)
        with st.spinner("Saving to Supabase..."):
            try:
                db_insert(row)
                st.success("Match saved to pending.")
            except Exception as e:
                st.error(f"Supabase insert failed: {e}")


# ─────────────────────────────────────────────────────────────────
# TAB 2 — PENDING
# ─────────────────────────────────────────────────────────────────
with tab_pending:
    st.markdown("### Pending matches")
    st.caption("Enter the final score to settle. The match moves to Performance.")

    try:
        pending = db_pending()
    except Exception as e:
        st.error(f"Could not load pending: {e}")
        pending = []

    if not pending:
        st.info("No pending matches.")
    else:
        for m in pending:
            render_pending_card(m)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    hs = st.number_input("Home", min_value=0, step=1,
                                         key=f"h_{m['match_id']}")
                with c2:
                    aws = st.number_input("Away", min_value=0, step=1,
                                          key=f"a_{m['match_id']}")
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        result, dc_hit, units, profit = settle(
                            m.get("direction"), m.get("stake"),
                            int(hs), int(aws)
                        )
                        db_settle(
                            m["match_id"], int(hs), int(aws),
                            result,
                            dc_hit if dc_hit is not None else 0,
                            profit
                        )
                        st.success("Settled.")
                        st.rerun()

                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"])
                    st.success("Deleted.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────
# TAB 3 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### Performance")
    try:
        settled = db_settled()
    except Exception as e:
        st.error(f"Could not load settled: {e}")
        settled = []

    if not settled:
        st.info("No settled matches yet.")
    else:
        df = pd.DataFrame(settled)
        df["dc_hit"] = pd.to_numeric(df["dc_hit"], errors="coerce")
        df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0)
        df["qualifying"] = pd.to_numeric(df["qualifying"], errors="coerce").fillna(0)

        qualifying = df[df["qualifying"] == 1]
        bets = qualifying[qualifying["dc_hit"].notna()]

        total = len(bets)
        wins = int((bets["dc_hit"] == 1).sum())
        losses = int((bets["dc_hit"] == 0).sum())
        net = float(df["profit_loss"].sum())
        hit_rate = (wins / total * 100) if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div>'
                        f'<div class="metric-label">Bets</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div>'
                        f'<div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{losses}</div>'
                        f'<div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
        with c4:
            color = "#10b981" if net >= 0 else "#ef4444"
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div>'
                        f'<div class="metric-label">Net P/L</div></div>', unsafe_allow_html=True)

        st.markdown("")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{hit_rate:.1f}%</div>'
                        f'<div class="metric-label">Hit rate</div></div>', unsafe_allow_html=True)
        with c6:
            skips = int((df["qualifying"] == 0).sum())
            st.markdown(f'<div class="metric-box"><div class="metric-value">{skips}</div>'
                        f'<div class="metric-label">Skips</div></div>', unsafe_allow_html=True)
        with c7:
            full = int((bets["stake"] == "full").sum()) if not bets.empty else 0
            half = int((bets["stake"] == "half").sum()) if not bets.empty else 0
            st.markdown(f'<div class="metric-box"><div class="metric-value">{full}/{half}</div>'
                        f'<div class="metric-label">Full / Half</div></div>', unsafe_allow_html=True)

        st.markdown("### By agreement split")
        if not qualifying.empty:
            by_split = (qualifying
                        .groupby("agreement_split")
                        .agg(total=("match_id", "count"),
                             wins=("dc_hit", lambda s: int((s == 1).sum())),
                             losses=("dc_hit", lambda s: int((s == 0).sum())),
                             net=("profit_loss", "sum"))
                        .reset_index())
            st.dataframe(by_split, use_container_width=True, hide_index=True)

        st.markdown("### By dissent signal")
        dissent_df = qualifying[qualifying["dissent_signal"].notna()]
        if not dissent_df.empty:
            by_dissent = (dissent_df
                          .groupby("dissent_signal")
                          .agg(total=("match_id", "count"),
                               wins=("dc_hit", lambda s: int((s == 1).sum())),
                               losses=("dc_hit", lambda s: int((s == 0).sum())),
                               net=("profit_loss", "sum"))
                          .reset_index())
            st.dataframe(by_dissent, use_container_width=True, hide_index=True)
        else:
            st.info("No dissent-signal data yet.")

        st.markdown("### All settled matches")
        for m in settled:
            render_settled_card(m)
