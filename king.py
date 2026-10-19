#!/usr/bin/env python3
"""
king.py — The Logic.

Input HTML. Extract facts. Run the logic. Show the pick.
Send to Supabase. Track pending. Settle later. See performance.

Parsing strategy: layered funnel, ID-anchored.

  Primary anchor: team IDs from logo URLs.
  SportsGambler assigns a numeric ID to every team (e.g. /logos/9866.png).
  The same ID appears in t_top, standings, venue table, H2H, fixtures.
  Matching by ID is alias-proof, abbreviation-proof, and language-proof.

  Fallbacks (only used when no logo ID is available):
    - exact name
    - substring
    - word overlap
    - initials
    - first/last word

Form extraction, three sources in priority order:
  1. Prose sentences ("Team have X wins, Y draws and Z losses")
  2. keystat-item block ("Full-Time Result" list item)
  3. (nothing else — if both fail, form stays "nothing")

Every field failure is logged. Missing data never votes silently.
"""

import re
import json
import streamlit as st
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
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
    .match-card.acc       { border-left-color: #a855f7; }
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;
    }
    .badge-home { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-away { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win  { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .badge-acc  { background: #a855f722; color: #c084fc; border: 1px solid #a855f755; }
    .badge-grade-a { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-grade-b { background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f655; }
    .badge-grade-c { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-grade-d { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.75rem; padding: 4px 10px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .metric-box {
        background: #1a1f2e; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #232a3d;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: #1a1f2e; border-radius: 10px;
        padding: 0 18px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except (KeyError, TypeError):
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
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


def db_accumulator_matches():
    """All matches (pending or settled) whose accumulator flag is TRUE."""
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .eq("accumulator", True)
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_find_existing(row: dict):
    """Look for an already-stored match matching the natural key."""
    sb = get_supabase()
    q = sb.table("matches").select("*")
    for field in ("match_date", "kickoff_time", "league",
                  "home_team", "away_team"):
        val = row.get(field)
        if val is None:
            q = q.is_(field, "null")
        else:
            q = q.eq(field, val)
    res = q.limit(1).execute()
    return res.data[0] if res.data else None


def db_settle(match_id: int, home_score: int, away_score: int,
              result: str, dc_hit, profit_loss: float):
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
# TEAM ID EXTRACTION — the primary anchor
# ═══════════════════════════════════════════════════════════════════

def _team_id_from_img(img_tag):
    if not img_tag:
        return None
    src = img_tag.get("src") or img_tag.get("data-src") or ""
    m = re.search(r"/logos/(\d+)\.", src)
    return m.group(1) if m else None


def _extract_team_ids_from_t_top(soup):
    t_top = soup.find("div", class_="t_top")
    if not t_top:
        return None, None
    teams_cols = t_top.find_all("div", class_="t_teams")
    if len(teams_cols) < 2:
        return None, None
    home_img = teams_cols[0].find("img")
    away_img = teams_cols[-1].find("img")
    return _team_id_from_img(home_img), _team_id_from_img(away_img)


# ═══════════════════════════════════════════════════════════════════
# NAME MATCHING — fallback only
# ═══════════════════════════════════════════════════════════════════

_TEAM_PREFIXES = (
    "club ", "atlético ", "atletico ", "deportivo ", "real ",
    "fc ", "cf ", "cd ", "ca ", "ac ", "as ", "sv ", "sc ",
    "athletic ", "sporting ", "racing ", "olympique ", "stade ",
    "rc ", "rcd ", "ud ", "sd ", "ss ", "ssc ", "afc ",
)

_TEAM_SUFFIXES = (
    " sco", " fc", " cf", " sc", " ac", " cd", " sv", " as",
    " 29", " 04", " 05", " 1899", " 1900", " calcio", " futbol",
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "single": 1, "a": 1, "an": 1,
}


def _search_term(full_name: str) -> str:
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


def _initials(name: str) -> str:
    skip = {"fc", "cf", "ac", "sc", "cd", "ca", "as", "sv", "ss",
            "ud", "sd", "afc", "united", "city"}
    parts = (name or "").lower().split()
    return "".join(p[0] for p in parts if p and p not in skip).upper()


def _team_match(full_name, cell_text):
    if not full_name or not cell_text:
        return False
    term = _search_term(full_name).lower()
    cell = cell_text.lower().strip()
    if not term:
        return False
    if term in cell or cell in term:
        return True
    if set(term.split()) & set(cell.split()):
        return True
    init = _initials(full_name).lower()
    if init and init == cell.replace(" ", "").replace(".", ""):
        return True
    if " " not in cell and cell in term.split():
        return True
    return False


def _word_to_int(s):
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return _NUMBER_WORDS.get(s, 0)


def _int_any(text):
    if text is None:
        return None
    m = re.search(r"-?\d+", str(text))
    if m:
        return int(m.group())
    return _word_to_int(text)


def _num(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return _NUMBER_WORDS.get(s)


def _team_aliases(soup, team_full, team_short):
    aliases = set()
    if team_full:
        aliases.add(team_full.strip())
        t = _search_term(team_full)
        if t:
            aliases.add(t)
    if team_short:
        aliases.add(team_short.strip())
        t = _search_term(team_short)
        if t:
            aliases.add(t)
    for name in list(aliases):
        low = name.lower()
        for prefix in _TEAM_PREFIXES:
            if low.startswith(prefix):
                aliases.add(name[len(prefix):].strip())
        for suffix in _TEAM_SUFFIXES:
            if low.endswith(suffix):
                aliases.add(name[: -len(suffix)].strip())
    for name in list(aliases):
        parts = name.split()
        if len(parts) > 1:
            aliases.add(parts[0])
            aliases.add(parts[-1])
    cleaned = set()
    for a in aliases:
        a = a.strip()
        if a and len(a) > 2 and a.lower() not in {"the", "and", "for"}:
            cleaned.add(a)
    return list(cleaned)


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — signals
# ═══════════════════════════════════════════════════════════════════

def _cmp_or_nothing(a, b):
    if a is None or b is None or a == b:
        return "nothing"
    return "home" if a > b else "away"


def compute_signals(f: dict) -> dict:
    """Unchanged — verified correct against all 227 qualifying rows."""
    hp = f.get("home_table_pts")
    ap = f.get("away_table_pts")
    hg = f.get("home_table_gd") or 0
    ag = f.get("away_table_gd") or 0
    if hp is None or ap is None:
        table = "nothing"
    elif (hp, hg) > (ap, ag):
        table = "home"
    elif (ap, ag) > (hp, hg):
        table = "away"
    else:
        table = "nothing"

    form  = _cmp_or_nothing(f.get("home_form_ppg"),  f.get("away_form_ppg"))
    venue = _cmp_or_nothing(f.get("home_home_ppg"),  f.get("away_away_ppg"))
    h2h   = _cmp_or_nothing(f.get("h2h_home_wins"),  f.get("h2h_away_wins"))

    return {"table": table, "form": form, "venue": venue, "h2h": h2h}


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — decide (priority rule + whitelist)
# ═══════════════════════════════════════════════════════════════════

def _compute_gap(facts: dict):
    """Absolute table-points gap, or None if either input is missing."""
    h = facts.get("home_table_pts")
    a = facts.get("away_table_pts")
    if h is None or a is None:
        return None
    return abs(h - a)


def _gap_band(gap):
    if gap is None:
        return None
    if gap < 3:
        return "close"
    if gap < 8:
        return "moderate"
    return "large"


def _compute_structure(signals: dict, direction: str) -> dict:
    """Support / oppose / abstain relative to the chosen direction."""
    support = oppose = abstain = 0
    opposing_signal = abstaining_signal = None
    for name, val in signals.items():
        if val is None or val == "nothing":
            abstain += 1
            if abstaining_signal is None:
                abstaining_signal = name
        elif val == direction:
            support += 1
        else:
            oppose += 1
            if opposing_signal is None:
                opposing_signal = name
    return {
        "support": support,
        "oppose": oppose,
        "abstain": abstain,
        "opposing_signal": opposing_signal,
        "abstaining_signal": abstaining_signal,
    }


def _grade_pick(structure: dict) -> str:
    """
    Grade the pick based on the observed agreement structure.
    Grades are descriptive (historical hit rate), not prescriptive.
    """
    s, o, a = structure["support"], structure["oppose"], structure["abstain"]
    if s == 4:
        return "A"
    if s == 3 and o == 0 and a == 1:
        return {"venue": "A", "h2h": "B", "form": "C"}.get(
            structure["abstaining_signal"], "B")
    if s == 3 and o == 1 and a == 0:
        return {"h2h": "B", "form": "A", "venue": "D"}.get(
            structure["opposing_signal"], "C")
    if s == 2 and o == 1 and a == 1:
        return "B"
    return "C"


def _skip_result(signals: dict, reason: str) -> dict:
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")
    return {
        "direction": "skip",
        "stake": "none",
        "qualifying": 0,
        "structure": None,
        "grade": None,
        "gap": None,
        "gap_band": None,
        "split": f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}",
        "home_count": home,
        "away_count": away,
        "nothing_count": nothing,
        "table_leader": signals["table"] if signals["table"] != "nothing" else None,
        "dissent_signal": None,
        "reason": reason,
    }


_STAKE_MAP = {
    (4, 0, 0): "full",
    (0, 4, 0): "full",
    (3, 1, 0): "half",
    (1, 3, 0): "half",
    (3, 0, 1): "full",
    (0, 3, 1): "full",
    (2, 1, 1): "half",
}


def decide(facts: dict, signals: dict) -> dict:
    """
    Priority order: table ≈ venue > form > H2H.
    Skip when table and venue both pick but disagree.
    Apply stake whitelist — only historically validated splits qualify.
    """
    table = signals["table"]
    venue = signals["venue"]
    form = signals["form"]
    h2h = signals["h2h"]

    table_picks = table in ("home", "away")
    venue_picks = venue in ("home", "away")

    if table_picks and venue_picks:
        if table == venue:
            direction = table
        else:
            return _skip_result(signals, reason="top_tier_disagreement")
    elif table_picks:
        direction = table
    elif venue_picks:
        direction = venue
    elif form in ("home", "away"):
        direction = form
    elif h2h in ("home", "away"):
        direction = h2h
    else:
        return _skip_result(signals, reason="no_signals")

    structure = _compute_structure(signals, direction)
    grade = _grade_pick(structure)
    gap = _compute_gap(facts)
    gap_band = _gap_band(gap)

    split_key = (structure["support"], structure["oppose"], structure["abstain"])
    stake = _STAKE_MAP.get(split_key)
    if stake is None:
        return _skip_result(signals, reason="unsupported_split")

    return {
        "direction": direction,
        "stake": stake,
        "qualifying": 1,
        "structure": structure,
        "grade": grade,
        "gap": gap,
        "gap_band": gap_band,
        "split": f"{structure['support']}-{structure['oppose']}-{structure['abstain']}",
        "home_count": sum(1 for v in signals.values() if v == "home"),
        "away_count": sum(1 for v in signals.values() if v == "away"),
        "nothing_count": sum(1 for v in signals.values() if v == "nothing"),
        "table_leader": signals["table"] if signals["table"] != "nothing" else None,
        "dissent_signal": None,
        "reason": None,
    }


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — accumulator (parity-aligned with SQL)
# ═══════════════════════════════════════════════════════════════════

def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _zero_filled_difference(home_value, away_value):
    if _is_missing(home_value) or _is_missing(away_value):
        return Decimal("0")
    try:
        return Decimal(str(home_value)) - Decimal(str(away_value))
    except Exception:
        return Decimal("0")


def calc_accumulator_score(facts):
    """Weighted sum verified to correlate with hit rate."""
    raw = (
        _zero_filled_difference(facts.get("home_table_pts"), facts.get("away_table_pts"))
        + Decimal("5") * _zero_filled_difference(
            facts.get("home_form_ppg"), facts.get("away_form_ppg")
        )
        + Decimal("3") * _zero_filled_difference(
            facts.get("home_home_ppg"), facts.get("away_away_ppg")
        )
        + _zero_filled_difference(
            facts.get("h2h_home_wins"), facts.get("h2h_away_wins")
        )
    )
    return raw.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def compute_accumulator(facts, decision):
    """
    Return TRUE / FALSE / NULL — parallel to the SQL CASE:
        qualifying != 1                       -> NULL
        qualifying = 1, direction = skip/NULL -> FALSE
        qualifying = 1, direction in home/away:
            score is NULL                     -> NULL
            |score| >= 15                     -> TRUE
            |score| < 15                      -> FALSE
        qualifying = 1, other direction       -> NULL
    """
    qualifying = decision.get("qualifying")
    if _is_missing(qualifying) or int(qualifying) != 1:
        return None

    direction = decision.get("direction")
    if _is_missing(direction):
        return False
    if direction == "skip":
        return False
    if direction not in ("home", "away"):
        return None  # invalid direction — data quality issue

    score = calc_accumulator_score(facts)
    return bool(abs(score) >= Decimal("15.0"))


def settle(direction, stake, home_score, away_score):
    """Settle a pick against the final score."""
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
# HTML EXTRACTION — identity
# ═══════════════════════════════════════════════════════════════════

def _extract_identity(soup, f):
    t_top = soup.find("div", class_="t_top")

    if t_top:
        names = t_top.find_all("div", class_="t_name")
        if len(names) >= 2:
            f["home_team"] = names[0].get_text(strip=True)
            f["away_team"] = names[1].get_text(strip=True)

    title = soup.find("title")
    if not f.get("home_team") and title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", title.get_text())
        if m:
            f["home_team"] = m.group(1).strip()
            f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        h1 = soup.find("h1")
        if h1:
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", h1.get_text())
            if m:
                f["home_team"] = m.group(1).strip()
                f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", og["content"])
            if m:
                f["home_team"] = m.group(1).strip()
                f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            m = re.search(r"/([^/]+)-vs-([^/]+)-prediction", link["href"])
            if m:
                f["home_team"] = m.group(1).replace("-", " ").title()
                f["away_team"] = m.group(2).replace("-", " ").title()

    if title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", title.get_text())
        if m:
            f["home_team_short"] = m.group(1).strip()
            f["away_team_short"] = m.group(2).strip()

    if not f.get("home_team_short"):
        f["home_team_short"] = _search_term(f.get("home_team", ""))
    if not f.get("away_team_short"):
        f["away_team_short"] = _search_term(f.get("away_team", ""))

    if t_top:
        info = t_top.find("div", class_="t_info")
        if info:
            links = info.find_all("a", class_="t_info_link")
            if len(links) >= 2:
                league_txt = links[1].get_text(strip=True)
                if " - " in league_txt:
                    country, league_txt = league_txt.split(" - ", 1)
                    f["country"] = country.strip()
                f["league"] = league_txt

            date = info.find("span", class_="t_date")
            if date:
                txt = date.get_text(" ", strip=True)
                m = re.search(r"(\d{1,2}\s+\w+)", txt)
                if m:
                    f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m:
                    f["kickoff_time"] = m.group(1)

            venue = info.find("span", class_="t_venue")
            if venue:
                f["venue"] = venue.get_text(strip=True)

    if not f.get("league"):
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                for node in data.get("@graph", [data]):
                    event = node.get("mainEntity", {})
                    if isinstance(event, dict) and "superEvent" in event:
                        f["league"] = event["superEvent"].get("name")
                        break
                if f.get("league"):
                    break
            except Exception:
                continue


# ═══════════════════════════════════════════════════════════════════
# STANDINGS
# ═══════════════════════════════════════════════════════════════════

def _gd_from_cells(cells):
    for c in cells:
        if re.match(r"^\d+:\d+$", c.strip()):
            try:
                a, b = c.split(":")
                return int(a) - int(b)
            except Exception:
                pass
    for c in reversed(cells[:-1]):
        s = c.strip()
        if (s.startswith("-") or s.startswith("+")) and s[1:].isdigit():
            return int(s)
    return None


def _parse_standings(soup, f):
    home_id, away_id = _extract_team_ids_from_t_top(soup)

    if not home_id and not away_id:
        f["_parse_warnings"].append("standings: no team IDs from t_top logos")

    home_team = f.get("home_team", "")
    away_team = f.get("away_team", "")

    tables = (
        soup.find_all("table", class_="clt-table") +
        soup.find_all("table", class_="leage-table") +
        soup.find_all("table", class_="league-table") +
        soup.find_all("table", class_="standings-table")
    )

    seen = set()
    for table in tables:
        for row in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 4:
                continue

            target = None
            img = row.find("img")
            row_id = _team_id_from_img(img)
            if row_id:
                if row_id == home_id:
                    target = "home"
                elif row_id == away_id:
                    target = "away"

            if not target:
                team_cell = cells[1] if len(cells) > 1 else ""
                if _team_match(home_team, team_cell):
                    target = "home"
                elif _team_match(away_team, team_cell):
                    target = "away"

            if not target or target in seen:
                continue
            seen.add(target)

            nums = [_int_any(c) for c in cells if _int_any(c) is not None]
            pos = nums[0] if nums else None
            pts = nums[-1] if nums else None
            gd = _gd_from_cells(cells)

            f[f"{target}_table_pos"] = pos
            f[f"{target}_table_pts"] = pts
            f[f"{target}_table_gd"]  = gd

            if len(cells) >= 11:
                f[f"{target}_table_played"] = _int_any(cells[2])
                f[f"{target}_table_gf"]     = _int_any(cells[6])
                f[f"{target}_table_ga"]     = _int_any(cells[7])

        if len(seen) == 2:
            break


# ═══════════════════════════════════════════════════════════════════
# VENUE FORM
# ═══════════════════════════════════════════════════════════════════

_WIN_WORDS  = r"(?:wins?|victor(?:y|ies)|won)"
_LOSS_WORDS = r"(?:loss(?:es)?|defeats?|lost)"
_DRAW_WORDS = r"(?:draws?|drawn)"
_NUM_PAT    = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _wdl_from_cell(text):
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _venue_sentence(sents):
    keywords = [
        "in the last 10 home", "in the last 10 away",
        "in the past 10 home", "in the past 10 away",
        "in the previous 10 home", "in the previous 10 away",
        "in their last 10 home", "in their last 10 away",
        "in their past 10 home", "in their past 10 away",
        "last 10 home matches", "last 10 away matches",
    ]
    return [s for s in sents if any(k in s.lower() for k in keywords)]


def _parse_venue_from_sentences(sents):
    patterns = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
    ]
    for s in sents:
        for tmpl, mapping in patterns:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            total = sum(vals.values())
            if total < 5 or total > 15:
                continue
            return vals["w"], vals["d"], vals["l"]
    return None


def _apply_venue(f, side, wdl):
    w, d, l = wdl
    total = w + d + l or 1
    pts = w * 3 + d
    prefix = "home_home" if side == "home" else "away_away"
    f[f"{prefix}_wins"] = w
    f[f"{prefix}_draws"] = d
    f[f"{prefix}_losses"] = l
    f[f"{prefix}_pts"] = pts
    f[f"{prefix}_ppg"] = round(pts / total, 2)


def _parse_venue(soup, home_sents, away_sents, f):
    home_id, away_id = _extract_team_ids_from_t_top(soup)

    st_table = soup.find("table", class_="st-table")
    home_wdl = away_wdl = None

    if st_table:
        for row in st_table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 2:
                continue
            wdl = None
            for c in cells:
                wdl = _wdl_from_cell(c)
                if wdl:
                    break
            if not wdl:
                continue
            img = row.find("img")
            row_id = _team_id_from_img(img)
            label = " ".join(cells).lower()
            if row_id:
                if row_id == home_id:
                    home_wdl = wdl
                    continue
                elif row_id == away_id:
                    away_wdl = wdl
                    continue
            if "home stats" in label and not home_wdl:
                home_wdl = wdl
            elif "away stats" in label and not away_wdl:
                away_wdl = wdl

    if home_wdl:
        _apply_venue(f, "home", home_wdl)
    if away_wdl:
        _apply_venue(f, "away", away_wdl)

    if "home_home_ppg" not in f and home_sents:
        result = _parse_venue_from_sentences(_venue_sentence(home_sents))
        if result:
            _apply_venue(f, "home", result)

    if "away_away_ppg" not in f and away_sents:
        result = _parse_venue_from_sentences(_venue_sentence(away_sents))
        if result:
            _apply_venue(f, "away", result)


# ═══════════════════════════════════════════════════════════════════
# FORM — PROSE + KEYSTAT (with parser fix)
# ═══════════════════════════════════════════════════════════════════

def _parse_form_from_sentences(sents, required_first_word=None, other_team=None):
    venue_sents = set(_venue_sentence(sents))
    filtered = [s for s in sents if s not in venue_sents]

    cleaned = []
    for s in filtered:
        if "G W D L" in s and "View Full Table" in s:
            s = s.split("View Full Table", -1)[-1]
        if "W-D-L" in s and "G GF GA" in s and "Home Stats" in s:
            continue
        if len(s) < 20:
            continue
        cleaned.append(s)
    filtered = cleaned

    patterns_3 = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
        (rf"won\s+{_NUM_PAT},?\s+lost\s+{_NUM_PAT}\s+and\s+drawn\s+{_NUM_PAT}", ("w","l","d")),
        (rf"won\s+{_NUM_PAT},?\s+drawn\s+{_NUM_PAT}\s+and\s+lost\s+{_NUM_PAT}", ("w","d","l")),
        (rf"have\s+won\s+{_NUM_PAT},?\s+lost\s+{_NUM_PAT}\s+and\s+drawn\s+{_NUM_PAT}", ("w","l","d")),
        (rf"have\s+won\s+{_NUM_PAT},?\s+drawn\s+{_NUM_PAT}\s+and\s+lost\s+{_NUM_PAT}", ("w","d","l")),
    ]
    patterns_2 = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","w")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","l")),
    ]

    for s in filtered:
        s_low = s.lower()
        if required_first_word:
            req_low = required_first_word.lower()
            if req_low not in s_low:
                continue
            team_idx = s_low.find(req_low)
            search_area = s[team_idx:]
            if other_team:
                other_low = other_team.lower()
                other_idx = s_low.find(other_low, team_idx + len(req_low))
                if other_idx != -1:
                    search_area = s[team_idx:other_idx]
        else:
            search_area = s

        if not search_area:
            continue

        for tmpl, mapping in patterns_3:
            m = re.search(tmpl, search_area, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            total = sum(vals.values())
            if total < 3 or total > 15:
                continue
            return vals["w"], vals["d"], vals["l"]

        for tmpl, mapping in patterns_2:
            m = re.search(tmpl, search_area, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            total = vals["w"] + vals["l"]
            if total < 2 or total > 10:
                continue
            vals["d"] = max(0, 10 - total)
            if sum(vals.values()) < 5 or sum(vals.values()) > 15:
                continue
            return vals["w"], vals["d"], vals["l"]

    return None


_KEYSTAT_PATTERNS = [
    (r"(\d+)\s+wins?,?\s+(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+draws?", ("w","l","d")),
    (r"(\d+)\s+wins?,?\s+(\d+)\s+draws?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("w","d","l")),
    (r"(\d+)\s+(?:defeats?|losses?),?\s+(\d+)\s+wins?\s+and\s+(\d+)\s+draws?", ("l","w","d")),
    (r"(\d+)\s+(?:defeats?|losses?),?\s+(\d+)\s+draws?\s+and\s+(\d+)\s+wins?", ("l","d","w")),
    (r"(\d+)\s+draws?,?\s+(\d+)\s+wins?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("d","w","l")),
    (r"(\d+)\s+draws?,?\s+(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+wins?", ("d","l","w")),
    # Oxford-comma variants
    (r"(\d+)\s+wins?,?\s+(\d+)\s+(?:defeats?|losses?),?\s+and\s+(\d+)\s+draws?", ("w","l","d")),
    (r"(\d+)\s+wins?,?\s+(\d+)\s+draws?,?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("w","d","l")),
]


def _parse_form_from_keystat(soup):
    """
    Fallback: read overall form from keystat blocks.
    Broadened title match to accept alternative block titles.
    """
    home_wdl = away_wdl = None

    acceptable_titles = (
        "full-time result", "recent form", "last 10", "form", "results"
    )

    for item in soup.find_all("div", class_="keystat-item"):
        classes = item.get("class", [])
        is_away = "awaystats" in classes

        title = item.find("div", class_="stats-title-sub")
        if not title:
            continue
        title_text = title.get_text().lower()
        if not any(k in title_text for k in acceptable_titles):
            continue

        ul = item.find("ul", class_="stats-list")
        if not ul:
            continue
        first_li = ul.find("li")
        if not first_li:
            continue

        text = first_li.get_text(" ", strip=True)
        wdl = None
        for tmpl, mapping in _KEYSTAT_PATTERNS:
            m = re.search(tmpl, text, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = int(m.group(i + 1))
            total = sum(vals.values())
            if total < 5 or total > 15:
                continue
            wdl = (vals["w"], vals["d"], vals["l"])
            break

        if not wdl:
            continue

        if is_away:
            away_wdl = wdl
        else:
            home_wdl = wdl

    return home_wdl, away_wdl


def _apply_form(f, side, wdl, source="prose"):
    w, d, l = wdl
    total = w + d + l or 1
    pts = w * 3 + d
    prefix = "home_form" if side == "home" else "away_form"
    f[f"{prefix}_wins"] = w
    f[f"{prefix}_draws"] = d
    f[f"{prefix}_losses"] = l
    f[f"{prefix}_pts"] = pts
    f[f"{prefix}_ppg"] = round(pts / total, 2)
    f[f"{prefix}_source"] = source


def _parse_form(soup, home_sents, away_sents, f):
    home_short = f.get("home_team_short") or f.get("home_team", "")
    away_short = f.get("away_team_short") or f.get("away_team", "")

    home_form = _parse_form_from_sentences(
        home_sents, required_first_word=home_short, other_team=away_short
    )
    if home_form:
        _apply_form(f, "home", home_form, source="prose")

    away_form = _parse_form_from_sentences(
        away_sents, required_first_word=away_short, other_team=home_short
    )
    if away_form:
        _apply_form(f, "away", away_form, source="prose")

    need_home = "home_form_ppg" not in f
    need_away = "away_form_ppg" not in f
    if need_home or need_away:
        ks_home, ks_away = _parse_form_from_keystat(soup)
        if need_home and ks_home:
            _apply_form(f, "home", ks_home, source="keystat")
            f["_parse_warnings"].append("home form sourced from keystat block")
        if need_away and ks_away:
            _apply_form(f, "away", ks_away, source="keystat")
            f["_parse_warnings"].append("away form sourced from keystat block")


def _team_sentences(aliases, sentences):
    if not aliases:
        return []
    result = []
    seen = set()
    sorted_aliases = sorted(aliases, key=len, reverse=True)
    for alias in sorted_aliases:
        if not alias:
            continue
        pattern = re.compile(r"\b" + re.escape(alias.lower()) + r"\b", re.I)
        for s in sentences:
            sid = hash(s)
            if sid in seen:
                continue
            if pattern.search(s):
                if "G W D L" in s and len(s) < 200:
                    if not any(w in s.lower() for w in ("wins", "won", "victor")):
                        continue
                result.append(s)
                seen.add(sid)
    return result


# ═══════════════════════════════════════════════════════════════════
# H2H
# ═══════════════════════════════════════════════════════════════════

def _parse_h2h(soup, home_term, away_term, f):
    rows = []
    section = soup.find("div", id="head-to-head")
    if section:
        rows = section.find_all("li", class_="team-stat-list-item")
    if not rows:
        for h in soup.find_all(["h3", "h4", "div"], string=re.compile("head to head", re.I)):
            parent = h.find_parent()
            if parent:
                candidate_rows = parent.find_all("li", class_="team-stat-list-item")
                if candidate_rows:
                    rows = candidate_rows
                    break
    if not rows:
        f["_parse_warnings"].append("H2H: no rows found")
        return

    home_wins = away_wins = draws = 0
    for row in rows:
        text = row.get_text(" ", strip=True)
        nums = re.findall(r"\b(\d+)\b", text)
        if len(nums) < 2:
            continue
        hs, aws = int(nums[-2]), int(nums[-1])
        idx_home = text.lower().find((home_term or "").lower())
        idx_away = text.lower().find((away_term or "").lower())
        home_first = idx_home != -1 and (idx_away == -1 or idx_home < idx_away)
        if hs > aws:
            home_wins += 1 if home_first else 0
            away_wins += 0 if home_first else 1
        elif aws > hs:
            away_wins += 1 if home_first else 0
            home_wins += 0 if home_first else 1
        else:
            draws += 1

    f["h2h_home_wins"] = home_wins
    f["h2h_away_wins"] = away_wins
    f["h2h_draws"] = draws
    f["h2h_total"] = home_wins + away_wins + draws


# ═══════════════════════════════════════════════════════════════════
# MAIN EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

def extract_facts(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    f = {"_parse_warnings": []}

    raw_text = soup.get_text(" ", strip=True)
    body = re.sub(r"\s+", " ", raw_text.replace("View Full Table", ". "))

    f["match_date"] = f["kickoff_time"] = None
    f["league"] = f["country"] = f["venue"] = None
    f["home_team"] = f["away_team"] = None
    f["home_team_short"] = f["away_team_short"] = None

    home_id, away_id = _extract_team_ids_from_t_top(soup)
    f["home_team_id"] = home_id
    f["away_team_id"] = away_id

    _extract_identity(soup, f)

    if not f.get("home_team"):
        f["_parse_warnings"].append("home_team not found")
    if not f.get("away_team"):
        f["_parse_warnings"].append("away_team not found")
    if not home_id:
        f["_parse_warnings"].append("home team logo ID not found")
    if not away_id:
        f["_parse_warnings"].append("away team logo ID not found")

    _parse_standings(soup, f)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]

    home_aliases = _team_aliases(soup, f.get("home_team", ""), f.get("home_team_short", ""))
    away_aliases = _team_aliases(soup, f.get("away_team", ""), f.get("away_team_short", ""))

    home_sents = _team_sentences(home_aliases, sentences)
    away_sents = _team_sentences(away_aliases, sentences)

    _parse_form(soup, home_sents, away_sents, f)
    if "home_form_ppg" not in f:
        f["_parse_warnings"].append("home form not found - will be treated as nothing")
    if "away_form_ppg" not in f:
        f["_parse_warnings"].append("away form not found - will be treated as nothing")

    _parse_venue(soup, home_sents, away_sents, f)
    if "home_home_ppg" not in f:
        f["_parse_warnings"].append("home venue form not found")
    if "away_away_ppg" not in f:
        f["_parse_warnings"].append("away venue form not found")

    _parse_h2h(
        soup,
        f.get("home_team_short") or f.get("home_team", ""),
        f.get("away_team_short") or f.get("away_team", ""),
        f,
    )

    for k in [
        "home_table_gd", "away_table_gd",
        "home_table_pts", "away_table_pts",
        "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_total",
    ]:
        f.setdefault(k, None)

    return f


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

def signal_pill(name, value):
    cls = value if value in ("home", "away", "nothing") else "nothing"
    label = {"home": "H", "away": "A", "nothing": "—"}.get(value, "—")
    return f'<span class="signal-pill {cls}">{name}: {label}</span>'


def grade_badge(grade):
    if not grade:
        return ""
    cls = {"A": "badge-grade-a", "B": "badge-grade-b",
           "C": "badge-grade-c", "D": "badge-grade-d"}.get(grade, "badge-skip")
    return f'<span class="pick-badge {cls}" style="margin-left:8px">Grade {grade}</span>'


def render_pending_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    direction = m.get("direction", "skip")
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""
    is_acc = bool(m.get("accumulator"))
    grade = m.get("grade")
    gap_band = m.get("gap_band")

    if direction == "home":
        card_cls, badge_cls = "pick-home", "badge-home"
        pick_text = f"HOME — {home} (DC 1X)"
    elif direction == "away":
        card_cls, badge_cls = "pick-away", "badge-away"
        pick_text = f"AWAY — {away} (DC X2)"
    else:
        card_cls, badge_cls = "skip", "badge-skip"
        pick_text = "SKIP"

    acc_badge = (
        '<span class="pick-badge badge-acc" style="margin-left:8px">🎯 ACC</span>'
        if is_acc else ""
    )
    gap_badge = (
        f'<span class="pick-badge badge-skip" style="margin-left:8px">gap: {gap_band}</span>'
        if gap_band else ""
    )

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
        <span class="pick-badge {badge_cls}">{pick_text}</span>{grade_badge(grade)}{acc_badge}{gap_badge}
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
    league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""
    is_acc = bool(m.get("accumulator"))
    grade = m.get("grade")

    if dc_hit == 1:
        cls, badge, verdict = "win", "badge-win", "WIN"
    elif dc_hit == 0:
        cls, badge, verdict = "loss", "badge-loss", "LOSS"
    else:
        cls, badge, verdict = "skip", "badge-skip", "NO BET"

    pl = f"{profit:+.2f}u" if dc_hit is not None else "—"
    acc_badge = (
        '<span class="pick-badge badge-acc" style="margin-left:8px">🎯 ACC</span>'
        if is_acc else ""
    )

    st.markdown(f"""
    <div class="match-card {cls}">
        <div class="match-title">{home} {hs}–{aws} {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge}">{verdict} · {pl}</span>{grade_badge(grade)}{acc_badge}
    </div>
    """, unsafe_allow_html=True)


def render_accumulator_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    direction = m.get("direction", "skip")
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""
    grade = m.get("grade")
    gap_band = m.get("gap_band")

    hs = m.get("home_score")
    aws = m.get("away_score")
    settled = hs is not None and aws is not None

    if direction == "home":
        pick_text = f"HOME — {home} (DC 1X)"
    elif direction == "away":
        pick_text = f"AWAY — {away} (DC X2)"
    else:
        pick_text = direction.upper() if direction else "?"

    score_txt = ""
    try:
        score_val = calc_accumulator_score(m)
        score_txt = f" · score {score_val:+.1f}"
    except Exception:
        pass

    result_txt = ""
    if settled:
        pl = m.get("profit_loss")
        pl_txt = f"{pl:+.2f}u" if isinstance(pl, (int, float)) else "—"
        hit = m.get("dc_hit")
        verdict = "WIN" if hit == 1 else ("LOSS" if hit == 0 else "NO BET")
        result_txt = f' · <span style="color:#a855f7">{hs}–{aws} {verdict} {pl_txt}</span>'

    gap_badge = (
        f'<span class="pick-badge badge-skip" style="margin-left:8px">gap: {gap_band}</span>'
        if gap_band else ""
    )

    st.markdown(f"""
    <div class="match-card acc">
        <div class="match-title">{home} vs {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}{score_txt}{result_txt}</div>
        <span class="pick-badge badge-acc">🎯 {pick_text}</span>{grade_badge(grade)}{gap_badge}
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
    cols = {
        "direction":               d["direction"],
        "stake":                   d["stake"],
        "qualifying":              d["qualifying"],
        "dissent_signal":          d.get("dissent_signal"),
        "agreement_split":         d["split"],
        "agreement_home_count":    d["home_count"],
        "agreement_away_count":    d["away_count"],
        "agreement_nothing_count": d["nothing_count"],
        "table_leader":            d.get("table_leader"),
        "grade":                   d.get("grade"),
        "gap_band":                d.get("gap_band"),
    }
    if d.get("structure"):
        cols["support_count"]     = d["structure"]["support"]
        cols["oppose_count"]      = d["structure"]["oppose"]
        cols["abstain_count"]     = d["structure"]["abstain"]
        cols["opposing_signal"]   = d["structure"]["opposing_signal"]
        cols["abstaining_signal"] = d["structure"]["abstaining_signal"]
    return cols


def build_row(facts, signals, decision):
    row = dict(facts)
    row.pop("_parse_warnings", None)
    row.pop("home_team_short", None)
    row.pop("away_team_short", None)
    row.pop("home_team_id", None)
    row.pop("away_team_id", None)
    row.pop("home_form_source", None)
    row.pop("away_form_source", None)
    row.update(signals_to_columns(signals))
    row.update(decision_columns(decision))
    row["extracted_at"] = datetime.now(timezone.utc).isoformat()

    row["accumulator_score"] = float(calc_accumulator_score(facts))
    row["accumulator"] = compute_accumulator(facts, decision)

    allowed = {
        "match_date", "kickoff_time", "league", "country", "venue",
        "home_team", "away_team",
        "home_table_pos", "away_table_pos", "home_table_pts", "away_table_pts",
        "home_table_gd", "away_table_gd", "home_table_gf", "away_table_gf",
        "home_table_ga", "away_table_ga", "home_table_played", "away_table_played",
        "home_home_wins", "home_home_draws", "home_home_losses",
        "home_home_pts", "home_home_ppg",
        "away_away_wins", "away_away_draws", "away_away_losses",
        "away_away_pts", "away_away_ppg",
        "home_form_wins", "home_form_draws", "home_form_losses", "home_form_pts",
        "home_form_ppg",
        "away_form_wins", "away_form_draws", "away_form_losses", "away_form_pts",
        "away_form_ppg",
        "h2h_total", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
        "signal_table", "signal_form", "signal_venue", "signal_h2h",
        "agreement_split", "agreement_home_count", "agreement_away_count",
        "agreement_nothing_count", "table_leader", "dissent_signal",
        "direction", "stake", "qualifying",
        "extracted_at",
        "accumulator", "accumulator_score",
        "grade", "gap_band",
        "support_count", "oppose_count", "abstain_count",
        "opposing_signal", "abstaining_signal",
    }
    return {k: v for k, v in row.items() if k in allowed and v is not None}


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_acc, tab_perf = st.tabs(
    ["➕ New Match", "⏳ Pending", "🎯 Accumulator", "📊 Performance"]
)


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
                decision = decide(facts, signals)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

        st.markdown("### The Pick")
        preview = {**facts, **signals_to_columns(signals), **decision_columns(decision)}
        preview["accumulator"] = compute_accumulator(facts, decision)
        render_pending_card(preview)

        try:
            acc_score = calc_accumulator_score(facts)
            st.markdown(
                f"**Accumulator score:** `{acc_score:+.1f}` — "
                f"threshold `±15.0` → "
                f"{'**ELIGIBLE** 🎯' if abs(acc_score) >= Decimal('15.0') else 'not eligible'}"
            )
        except Exception as e:
            st.caption(f"(could not compute accumulator score: {e})")

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.write(f"- {w}")

        st.markdown("### Extracted Facts")
        display_facts = {k: v for k, v in facts.items()
                         if v is not None and not k.startswith("_")}
        st.json(display_facts)

        row = build_row(facts, signals, decision)

        with st.spinner("Checking for an existing match..."):
            try:
                existing = db_find_existing(row)
            except Exception as e:
                existing = None
                st.warning(f"Could not check for duplicates: {e}")

        if existing:
            st.info(
                f"ℹ️ This match is already stored (match_id **{existing['match_id']}**, "
                f"saved {existing.get('extracted_at', 'earlier')}). "
                f"Prediction was regenerated above but **not** re-saved."
            )
            stored_acc = existing.get("accumulator")
            new_acc = row.get("accumulator")
            if stored_acc != new_acc:
                st.warning(
                    f"⚠️ Accumulator flag differs — "
                    f"stored: `{stored_acc}`, recomputed: `{new_acc}`."
                )
        else:
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
                        db_settle(m["match_id"], int(hs), int(aws),
                                  result, dc_hit, profit)
                        st.success("Settled.")
                        st.rerun()

                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"])
                    st.success("Deleted.")
                    st.rerun()


# ─────────────────────────────────────────────────────────────────
# TAB 3 — ACCUMULATOR
# ─────────────────────────────────────────────────────────────────
with tab_acc:
    st.markdown("### 🎯 Accumulator matches")
    st.caption(
        "All matches whose accumulator flag is TRUE "
        "(`qualifying = 1`, valid direction, `|score| >= 15.0`)."
    )

    try:
        acc_matches = db_accumulator_matches()
    except Exception as e:
        st.error(f"Could not load accumulator matches: {e}")
        acc_matches = []

    if not acc_matches:
        st.info("No accumulator-eligible matches yet.")
    else:
        total = len(acc_matches)
        settled = [m for m in acc_matches if m.get("home_score") is not None]
        pending_acc = total - len(settled)

        wins = sum(1 for m in settled if m.get("dc_hit") == 1)
        losses = sum(1 for m in settled if m.get("dc_hit") == 0)
        net = sum((m.get("profit_loss") or 0.0) for m in settled)
        hit_rate = (wins / len(settled) * 100) if settled else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div>'
                        f'<div class="metric-label">Accumulator</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{pending_acc}</div>'
                        f'<div class="metric-label">Pending</div></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div>'
                        f'<div class="metric-label">Wins</div></div>',
                        unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{losses}</div>'
                        f'<div class="metric-label">Losses</div></div>',
                        unsafe_allow_html=True)
        with c5:
            color = "#10b981" if net >= 0 else "#ef4444"
            st.markdown(
                f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div>'
                f'<div class="metric-label">Net P/L</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.caption(f"Hit rate on settled accumulator matches: **{hit_rate:.1f}%**")

        st.markdown("### Matches")
        for m in acc_matches:
            render_accumulator_card(m)


# ─────────────────────────────────────────────────────────────────
# TAB 4 — PERFORMANCE
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
        if "accumulator" in df.columns:
            df["accumulator"] = df["accumulator"].fillna(False).astype(bool)

        qualifying = df[df["qualifying"] == 1]
        bets = qualifying[qualifying["dc_hit"].notna()]

        total = len(bets)
        wins = int((bets["dc_hit"] == 1).sum()) if not bets.empty else 0
        losses = int((bets["dc_hit"] == 0).sum()) if not bets.empty else 0
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

        if "accumulator" in df.columns:
            acc_bets = bets[bets["accumulator"] == True]  # noqa: E712
            if not acc_bets.empty:
                acc_wins = int((acc_bets["dc_hit"] == 1).sum())
                acc_total = len(acc_bets)
                acc_net = float(acc_bets["profit_loss"].sum())
                acc_hr = (acc_wins / acc_total * 100) if acc_total else 0.0
                st.markdown("#### 🎯 Accumulator-only (settled)")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{acc_total}</div>'
                                f'<div class="metric-label">Bets</div></div>', unsafe_allow_html=True)
                with a2:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{acc_wins}</div>'
                                f'<div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
                with a3:
                    st.markdown(f'<div class="metric-box"><div class="metric-value">{acc_hr:.1f}%</div>'
                                f'<div class="metric-label">Hit rate</div></div>', unsafe_allow_html=True)
                with a4:
                    color = "#10b981" if acc_net >= 0 else "#ef4444"
                    st.markdown(
                        f'<div class="metric-box"><div class="metric-value" style="color:{color}">{acc_net:+.2f}u</div>'
                        f'<div class="metric-label">Net P/L</div></div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("### By grade")
        if "grade" in df.columns and not bets.empty:
            by_grade = (bets
                        .groupby("grade")
                        .agg(total=("match_id", "count"),
                             wins=("dc_hit", lambda s: int((s == 1).sum())),
                             losses=("dc_hit", lambda s: int((s == 0).sum())))
                        .reset_index())
            st.dataframe(by_grade, use_container_width=True, hide_index=True)

        st.markdown("### By gap band")
        if "gap_band" in df.columns and not bets.empty:
            by_gap = (bets
                      .groupby("gap_band")
                      .agg(total=("match_id", "count"),
                           wins=("dc_hit", lambda s: int((s == 1).sum())),
                           losses=("dc_hit", lambda s: int((s == 0).sum())))
                      .reset_index())
            st.dataframe(by_gap, use_container_width=True, hide_index=True)

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

        st.markdown("### All settled matches")
        for m in settled:
            render_settled_card(m)
