#!/usr/bin/env python3
"""
king.py — The Logic. Single-file. Learning-ready.

Two engines run in parallel on every row:

  LEGACY ENGINE (unchanged behaviour):
    Table / Form / Venue / H2H signals → decide() → direction, stake, qualifying.
    The 83.4% pick rate is preserved exactly as before.

  SIMPLE ENGINE (the 2-stat file):
    Consecutive Wins + Consecutive Unbeaten → classify_simple() →
    SOLID CLEAR / LIGHT CLEAR HOME / NOT-BEATEN SIGNAL.
    Also writes: home_consecutive_wins, away_consecutive_wins,
    home_consecutive_unbeaten, away_consecutive_unbeaten, simple_category,
    simple_pick_team, simple_bet, simple_qualifying, simple_reason.

Input formats (auto-detected in Tab 1):
  1. SportsGambler HTML (extract_facts) — legacy engine has signals; simple
     engine has no streaks (those fields don't exist in that source).
  2. Recent-form text dump (parse_recent_form_dump) — simple engine has all
     four streak numbers; legacy engine has no signals and is skipped.
"""

import re
import json
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from supabase import create_client, Client


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
        border-radius: 16px; padding: 22px 26px; margin-bottom: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip      { border-left-color: #64748b; opacity: 0.85; }
    .match-card.win       { border-left-color: #10b981; }
    .match-card.loss      { border-left-color: #ef4444; }
    .match-card.acc       { border-left-color: #a855f7; }
    .match-card.tier-1 { border-left-color: #06d6a0; }
    .match-card.tier-2 { border-left-color: #ffd166; }
    .match-card.tier-3 { border-left-color: #ef476f; }
    .match-card.tier-4 { border-left-color: #d90429; }
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
    .badge-tier-1 { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-tier-2 { background: #ffd16622; color: #ffd166; border: 1px solid #ffd16655; }
    .badge-tier-3 { background: #ef476f22; color: #ef476f; border: 1px solid #ef476f55; }
    .badge-tier-4 { background: #d9042922; color: #d90429; border: 1px solid #d9042955; }
    .badge-solid  { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-light  { background: #38bdf822; color: #38bdf8; border: 1px solid #38bdf855; }
    .badge-notbe  { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.75rem; padding: 4px 10px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .signal-pill.streak { background: #6366f115; color: #a5b4fc; border-color: #6366f140; font-family: monospace; }
    .goal-row { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
    .goal-pill {
        font-size: 0.75rem; padding: 4px 10px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
    }
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


class DecisionMismatchError(Exception):
    def __init__(self, mismatches: dict):
        self.mismatches = mismatches
        super().__init__(f"Reconciliation mismatch: {list(mismatches)}")


def db_insert(row: dict):
    audit = _reconcile_decision(row)
    if audit["status"] == "mismatch" and not _is_quarantined(row):
        raise DecisionMismatchError(audit["mismatches"])

    sb = get_supabase()
    res = sb.table("matches").insert(row).execute()
    st.cache_data.clear()

    try:
        log_entry = {
            "match_id": res.data[0]["match_id"] if res.data else None,
            "extracted_at": row.get("extracted_at"),
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "signal_table": row.get("signal_table"),
            "signal_form": row.get("signal_form"),
            "signal_venue": row.get("signal_venue"),
            "signal_h2h": row.get("signal_h2h"),
            "agreement_split": row.get("agreement_split"),
            "direction": row.get("direction"),
            "stake": row.get("stake"),
            "qualifying": row.get("qualifying"),
            "opposing_signal": row.get("opposing_signal"),
            "neutral_signal": row.get("neutral_signal"),
            "form_residual": row.get("form_residual"),
            "table_dissent": row.get("table_dissent"),
            "goal_tier": row.get("goal_tier"),
            "risk_tier": row.get("risk_tier"),
            "backtest_confidence": row.get("backtest_confidence"),
            "goal_gap": row.get("goal_gap"),
            "goal_total": row.get("goal_total"),
            "rank_gap": row.get("rank_gap"),
            # simple-engine fields
            "home_consecutive_wins": row.get("home_consecutive_wins"),
            "away_consecutive_wins": row.get("away_consecutive_wins"),
            "home_consecutive_unbeaten": row.get("home_consecutive_unbeaten"),
            "away_consecutive_unbeaten": row.get("away_consecutive_unbeaten"),
            "simple_category": row.get("simple_category"),
            "simple_pick_team": row.get("simple_pick_team"),
            "simple_bet": row.get("simple_bet"),
            "simple_qualifying": row.get("simple_qualifying"),
        }
        sb.table("learning_log").insert(log_entry).execute()
    except Exception:
        pass

    return res.data[0] if res.data else None


@st.cache_data(ttl=30)
def db_pending():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .is_("home_score", "null")
             .order("match_date", desc=True).execute())
    return res.data or []


@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .not_.is_("home_score", "null")
             .order("match_date", desc=True).execute())
    return res.data or []


@st.cache_data(ttl=60)
def db_all_matches():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .order("match_date", desc=True).execute())
    return res.data or []


def db_learning_log(limit: int = 50):
    try:
        sb = get_supabase()
        res = (sb.table("learning_log").select("*")
                 .order("extracted_at", desc=True).limit(limit).execute())
        return res.data or []
    except Exception:
        return []


def db_find_existing(row: dict):
    sb = get_supabase()
    q = sb.table("matches").select("*")
    for field in ("match_date", "kickoff_time", "league", "home_team", "away_team"):
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
        "home_score": home_score, "away_score": away_score,
        "result": result, "dc_hit": dc_hit, "profit_loss": profit_loss,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("match_id", match_id).execute()
    st.cache_data.clear()

    try:
        sb.table("learning_log").update({
            "result": result, "dc_hit": dc_hit, "profit_loss": profit_loss,
        }).eq("match_id", match_id).execute()
    except Exception:
        pass


def db_delete(match_id: int):
    sb = get_supabase()
    sb.table("matches").delete().eq("match_id", match_id).execute()
    st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════
# RECONCILIATION AUDIT
# ═══════════════════════════════════════════════════════════════════

def _reconcile_decision(row: dict) -> dict:
    stored_signals = {
        "table": row.get("signal_table"), "form": row.get("signal_form"),
        "venue": row.get("signal_venue"), "h2h": row.get("signal_h2h"),
    }
    if any(v is None for v in stored_signals.values()):
        return {"status": "skipped", "reason": "incomplete signals"}

    facts = {k: row.get(k) for k in [
        "home_table_pts", "away_table_pts", "home_table_gd", "away_table_gd",
        "home_form_ppg", "away_form_ppg",
        "home_home_ppg", "away_away_ppg",
        "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    ] if row.get(k) is not None}

    try:
        recomputed_signals = compute_signals(facts)
    except Exception:
        return {"status": "skipped", "reason": "compute_signals failed"}

    recomputed_decision = decide(recomputed_signals, facts)

    mismatches = {}
    if recomputed_signals != stored_signals:
        mismatches["signals"] = {"stored": stored_signals, "recomputed": recomputed_signals}
    if recomputed_decision["direction"] != row.get("direction"):
        mismatches["direction"] = {"stored": row.get("direction"), "recomputed": recomputed_decision["direction"]}
    if recomputed_decision["stake"] != row.get("stake"):
        mismatches["stake"] = {"stored": row.get("stake"), "recomputed": recomputed_decision["stake"]}
    if recomputed_decision.get("opposing_signal") != row.get("opposing_signal"):
        mismatches["opposing_signal"] = {"stored": row.get("opposing_signal"), "recomputed": recomputed_decision.get("opposing_signal")}
    if recomputed_decision.get("neutral_signal") != row.get("neutral_signal"):
        mismatches["neutral_signal"] = {"stored": row.get("neutral_signal"), "recomputed": recomputed_decision.get("neutral_signal")}

    if mismatches:
        return {"status": "mismatch", "mismatches": mismatches}
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════
# QUARANTINE / COHORT
# ═══════════════════════════════════════════════════════════════════

_RETRO_NOTE_MARKER = "Retrospectively reclassified"


def _is_quarantined(row: dict) -> bool:
    notes = row.get("notes")
    return bool(notes and _RETRO_NOTE_MARKER in str(notes))


def _cohort_label(signals: dict, direction: str) -> str:
    t, f, v, h = signals.get("table"), signals.get("form"), signals.get("venue"), signals.get("h2h")
    if direction == "home":
        if t == "home" and f == "home" and v == "home" and h == "home": return "A"
        if t == "home" and f == "home" and v == "home" and h in ("away", "nothing"): return "B"
        if t == "home" and f == "home": return "C"
    elif direction == "away":
        if t == "away" and f == "away" and v == "away" and h == "away": return "A"
        if t == "away" and f == "away": return "C"
    return "other"


def _cohort_from_row(row: dict) -> str:
    sig = {"table": row.get("signal_table"), "form": row.get("signal_form"),
           "venue": row.get("signal_venue"), "h2h": row.get("signal_h2h")}
    return _cohort_label(sig, row.get("direction"))


def _signal_has_data(row: dict, signal_name: str) -> bool:
    if signal_name == "table":
        return row.get("home_table_pts") is not None and row.get("away_table_pts") is not None
    if signal_name == "form":
        return row.get("home_form_ppg") is not None and row.get("away_form_ppg") is not None
    if signal_name == "venue":
        return row.get("home_home_ppg") is not None and row.get("away_away_ppg") is not None
    if signal_name == "h2h":
        total = row.get("h2h_total")
        return total is not None and total > 0
    return False


def _short_signal(sig, present):
    if sig == "home": return "H"
    if sig == "away": return "A"
    return "T" if present else "—"


def _cohort_meaning(signals: dict, direction: str, has_data: dict = None) -> str:
    has_data = has_data or {}
    return (f"T={_short_signal(signals.get('table'), has_data.get('table', True))}, "
            f"F={_short_signal(signals.get('form'), has_data.get('form', True))}, "
            f"V={_short_signal(signals.get('venue'), has_data.get('venue', True))}, "
            f"H2H={_short_signal(signals.get('h2h'), has_data.get('h2h', True))}")


def _cohort_meaning_from_row(row: dict) -> str:
    sig = {"table": row.get("signal_table"), "form": row.get("signal_form"),
           "venue": row.get("signal_venue"), "h2h": row.get("signal_h2h")}
    has_data = {k: _signal_has_data(row, k) for k in ("table", "form", "venue", "h2h")}
    return _cohort_meaning(sig, row.get("direction"), has_data)


_COHORT_MARKET = {"A": "Unanimous", "B": "H2H Fade", "C": "T+F Core", "other": None}


def _cohort_market_display(cohort: str, direction: str) -> str:
    market = "Home/Draw" if direction == "home" else ("Away/Draw" if direction == "away" else "No Bet")
    desc = _COHORT_MARKET.get(cohort)
    return f"{desc} ({market})" if desc else market


def _market_label(direction: str) -> str:
    if direction == "home": return "Home/Draw"
    if direction == "away": return "Away/Draw"
    return "No Bet"


_ACC_SPLITS = ("4-0", "0-4", "3-0-1", "0-3-1")
_ACC_GAP_MIN = 0.50
_ACC_GAP_DISSENT_MIN = 0.65
_ACC_DISSENT_SIGNALS = ("form", "venue")


def _is_accumulator_eligible(row: dict) -> bool:
    if _is_quarantined(row): return False
    if row.get("agreement_split") not in _ACC_SPLITS: return False
    gap = row.get("goal_gap")
    if gap is None: return False
    try: g = float(gap)
    except (TypeError, ValueError): return False
    if g < _ACC_GAP_MIN: return False
    opposing = row.get("opposing_signal") or row.get("dissent_signal")
    if opposing in _ACC_DISSENT_SIGNALS and g < _ACC_GAP_DISSENT_MIN: return False
    return True


# ═══════════════════════════════════════════════════════════════════
# TEAM ID + NAME MATCHING
# ═══════════════════════════════════════════════════════════════════

def _team_id_from_img(img_tag):
    if not img_tag: return None
    src = img_tag.get("src") or img_tag.get("data-src") or ""
    m = re.search(r"/logos/(\d+)\.", src)
    return m.group(1) if m else None


def _extract_team_ids_from_t_top(soup):
    t_top = soup.find("div", class_="t_top")
    if not t_top: return None, None
    teams_cols = t_top.find_all("div", class_="t_teams")
    if len(teams_cols) < 2: return None, None
    return _team_id_from_img(teams_cols[0].find("img")), _team_id_from_img(teams_cols[-1].find("img"))


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
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "single": 1, "a": 1, "an": 1,
}


def _search_term(full_name: str) -> str:
    if not full_name: return ""
    name = full_name.strip()
    low = name.lower()
    for prefix in _TEAM_PREFIXES:
        if low.startswith(prefix):
            name = name[len(prefix):]; break
    low = name.lower()
    for suffix in _TEAM_SUFFIXES:
        if low.endswith(suffix):
            name = name[: -len(suffix)]; break
    parts = name.split()
    if len(parts) >= 2:
        if len(parts[0]) <= 3: return parts[-1]
        return " ".join(parts[:2])
    return name


def _initials(name: str) -> str:
    skip = {"fc", "cf", "ac", "sc", "cd", "ca", "as", "sv", "ss", "ud", "sd", "afc", "united", "city"}
    parts = (name or "").lower().split()
    return "".join(p[0] for p in parts if p and p not in skip).upper()


def _team_match(full_name, cell_text):
    if not full_name or not cell_text: return False
    term = _search_term(full_name).lower()
    cell = cell_text.lower().strip()
    if not term: return False
    if term in cell or cell in term: return True
    term_words = set(w for w in term.split() if len(w) >= 4)
    cell_words = set(w for w in cell.split() if len(w) >= 4)
    if term_words & cell_words: return True
    init = _initials(full_name).lower()
    if init and init == cell.replace(" ", "").replace(".", ""): return True
    if " " not in cell and cell in term.split(): return True
    return False


def _word_to_int(s):
    if isinstance(s, (int, float)): return int(s)
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()): return int(s)
    return _NUMBER_WORDS.get(s, 0)


def _int_any(text):
    if text is None: return None
    m = re.search(r"-?\d+", str(text))
    if m: return int(m.group())
    return _word_to_int(text)


def _float_any(text):
    if text is None: return None
    try: return float(str(text).strip())
    except (ValueError, TypeError): return None


def _num(s):
    if s is None: return None
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()): return int(s)
    return _NUMBER_WORDS.get(s)


def _team_aliases(soup, team_full, team_short):
    aliases = set()
    if team_full:
        aliases.add(team_full.strip())
        t = _search_term(team_full)
        if t: aliases.add(t)
    if team_short:
        aliases.add(team_short.strip())
        t = _search_term(team_short)
        if t: aliases.add(t)
    for name in list(aliases):
        low = name.lower()
        for prefix in _TEAM_PREFIXES:
            if low.startswith(prefix): aliases.add(name[len(prefix):].strip())
        for suffix in _TEAM_SUFFIXES:
            if low.endswith(suffix): aliases.add(name[: -len(suffix)].strip())
    for name in list(aliases):
        parts = name.split()
        if len(parts) > 1:
            aliases.add(parts[0]); aliases.add(parts[-1])
    cleaned = set()
    for a in aliases:
        a = a.strip()
        if a and len(a) > 2 and a.lower() not in {"the", "and", "for"}: cleaned.add(a)
    return list(cleaned)


# ═══════════════════════════════════════════════════════════════════
# LEGACY PICK LOGIC (unchanged)
# ═══════════════════════════════════════════════════════════════════

def _cmp_or_nothing(a, b):
    if a is None or b is None or a == b: return "nothing"
    return "home" if a > b else "away"


def compute_signals(f: dict) -> dict:
    hp = f.get("home_table_pts"); ap = f.get("away_table_pts")
    hg = f.get("home_table_gd");  ag = f.get("away_table_gd")
    if hp is None or ap is None:
        table = "nothing"
    elif hp != ap:
        table = "home" if hp > ap else "away"
    elif hg is not None and ag is not None and hg != ag:
        table = "home" if hg > ag else "away"
    else:
        table = "nothing"
    form  = _cmp_or_nothing(f.get("home_form_ppg"),  f.get("away_form_ppg"))
    venue = _cmp_or_nothing(f.get("home_home_ppg"),  f.get("away_away_ppg"))
    h2h   = _cmp_or_nothing(f.get("h2h_home_wins"),  f.get("h2h_away_wins"))
    return {"table": table, "form": form, "venue": venue, "h2h": h2h}


def _compute_goal_profile(facts):
    h_gf = facts.get("home_home_gf"); h_ga = facts.get("home_home_ga")
    a_gf = facts.get("away_away_gf"); a_ga = facts.get("away_away_ga")
    if None in (h_gf, h_ga, a_gf, a_ga): return None
    try:
        home_exp = round((float(h_gf) + float(a_ga)) / 2.0, 2)
        away_exp = round((float(a_gf) + float(h_ga)) / 2.0, 2)
    except (ValueError, TypeError):
        return None
    return {"goal_exp_home": home_exp, "goal_exp_away": away_exp,
            "goal_total": round(home_exp + away_exp, 2),
            "goal_gap": round(abs(home_exp - away_exp), 2)}


def _pick_goal_gap(profile, direction):
    if profile is None: return None
    if direction == "home": return round(profile["goal_exp_home"] - profile["goal_exp_away"], 2)
    if direction == "away": return round(profile["goal_exp_away"] - profile["goal_exp_home"], 2)
    return None


def _split_dissent(signals: dict, direction: str):
    if direction is None or direction == "skip": return None, None
    opposing = next((k for k, v in signals.items() if v != direction and v != "nothing"), None)
    neutral = next((k for k, v in signals.items() if v == "nothing"), None)
    return opposing, neutral


_DECISION_TABLE = {
    (4, 0, 0): {"direction": "home", "stake": "full", "qualifying": 1},
    (0, 4, 0): {"direction": "away", "stake": "full", "qualifying": 1},
    (3, 1, 0): {"direction": "majority", "stake": "half", "qualifying": 1},
    (1, 3, 0): {"direction": "majority", "stake": "half", "qualifying": 1},
    (3, 0, 1): {"direction": "home", "stake": "full", "qualifying": 1},
    (0, 3, 1): {"direction": "away", "stake": "full", "qualifying": 1},
    (2, 2, 0): {"direction": "skip", "stake": "none", "qualifying": 0},
    (2, 1, 1): {"direction": "home", "stake": "half", "qualifying": 1},
    (1, 2, 1): {"direction": "skip", "stake": "none", "qualifying": 0},
    (2, 0, 2): {"direction": "skip", "stake": "none", "qualifying": 0},
    (0, 2, 2): {"direction": "skip", "stake": "none", "qualifying": 0},
    (1, 1, 2): {"direction": "skip", "stake": "none", "qualifying": 0},
}


def _split_label(home, away, nothing):
    if home == 4 and nothing == 0: return "4-0"
    if away == 4 and nothing == 0: return "0-4"
    if home == 3 and away == 0 and nothing == 1: return "3-0-1"
    if away == 3 and home == 0 and nothing == 1: return "0-3-1"
    if home == 3 and away == 1: return "3-1"
    if away == 3 and home == 1: return "1-3"
    if home == 2 and away == 1 and nothing == 1: return "2-1-1"
    if away == 2 and home == 1 and nothing == 1: return "1-2-1"
    if home == 2 and away == 2: return "2-2"
    if home == 2 and away == 0 and nothing == 2: return "2-0-2"
    if away == 2 and home == 0 and nothing == 2: return "0-2-2"
    if home == 1 and away == 1 and nothing == 2: return "1-1-2"
    return f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}"


def decide(signals: dict, facts: dict = None) -> dict:
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")

    spec = _DECISION_TABLE.get((home, away, nothing),
                               {"direction": "skip", "stake": "none", "qualifying": 0})
    direction = spec["direction"]
    if direction == "majority":
        direction = "home" if home > away else "away" if away > home else "skip"

    if direction in ("home", "away"):
        opposing, neutral = _split_dissent(signals, direction)
    else:
        opposing, neutral = None, None

    return {"direction": direction, "stake": spec["stake"], "qualifying": spec["qualifying"],
            "dissent_signal": None, "opposing_signal": opposing, "neutral_signal": neutral,
            "split": _split_label(home, away, nothing),
            "home_count": home, "away_count": away, "nothing_count": nothing,
            "table_leader": signals["table"] if signals["table"] != "nothing" else None}


def settle(direction, stake, home_score, away_score):
    if home_score > away_score: result = "home_win"
    elif away_score > home_score: result = "away_win"
    else: result = "draw"
    if direction == "skip": return result, None, 0.0, 0.0
    units = {"full": 1.0, "half": 0.5, "quarter": 0.25, "none": 0.0}.get(stake, 0.0)
    if direction == "home": hit = result in ("home_win", "draw")
    elif direction == "away": hit = result in ("away_win", "draw")
    else: hit = False
    profit = units if hit else -units
    return result, (1 if hit else 0), units, profit


# ═══════════════════════════════════════════════════════════════════
# SIMPLE CLASSIFIER — SOLID / LIGHT-HOME / NOT-BEATEN
# Uses ONLY: Consecutive Wins + Consecutive Unbeaten (4 numbers).
# Ignored by design: Consecutive Scoring, H2H.
# ═══════════════════════════════════════════════════════════════════

NOT_BEATEN_MIN_GAP = 3


def _to_int_or_none(v):
    if v is None: return None
    try:
        if v != v: return None
    except Exception:
        pass
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def classify_simple(home_wins, away_wins, home_unbeaten, away_unbeaten):
    hw = _to_int_or_none(home_wins)
    aw = _to_int_or_none(away_wins)
    hu = _to_int_or_none(home_unbeaten)
    au = _to_int_or_none(away_unbeaten)

    if None in (hw, aw, hu, au):
        return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
                "reason": "Missing one or more streak inputs."}

    home_has = hw > 0
    away_has = aw > 0

    if home_has and not away_has:
        if hu >= 2 and au == 0:
            return {"category": "SOLID CLEAR", "pick_team": "home", "bet": "1X",
                    "qualifying": 1,
                    "reason": f"Home {hw}W/{hu}U vs Away {aw}W/{au}U — SOLID (opponent 0/0)."}
        if hu == hw:
            return {"category": "LIGHT CLEAR HOME", "pick_team": "home", "bet": "1X",
                    "qualifying": 1,
                    "reason": f"Home {hw}W/{hu}U (unbeaten = wins) vs Away {aw}W/{au}U — LIGHT-HOME."}
        return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
                "reason": f"Home leader {hw}W/{hu}U but unbeaten != wins and SOLID not met."}

    if away_has and not home_has:
        if au >= 2 and hu == 0:
            return {"category": "SOLID CLEAR", "pick_team": "away", "bet": "X2",
                    "qualifying": 1,
                    "reason": f"Away {aw}W/{au}U vs Home {hw}W/{hu}U — SOLID (opponent 0/0)."}
        return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
                "reason": f"Away leader {aw}W/{au}U — LIGHT-away removed; SOLID not met."}

    if not home_has and not away_has:
        gap = abs(hu - au)
        if gap < NOT_BEATEN_MIN_GAP:
            return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
                    "reason": f"Both 0 wins, unbeaten gap {gap} < {NOT_BEATEN_MIN_GAP}."}
        if hu > au:
            return {"category": "NOT-BEATEN SIGNAL", "pick_team": "home", "bet": "1X",
                    "qualifying": 1,
                    "reason": f"Both 0W; Home unbeaten {hu} vs Away {au} (gap {gap} >= {NOT_BEATEN_MIN_GAP})."}
        return {"category": "NOT-BEATEN SIGNAL", "pick_team": "away", "bet": "X2",
                "qualifying": 1,
                "reason": f"Both 0W; Away unbeaten {au} vs Home {hu} (gap {gap} >= {NOT_BEATEN_MIN_GAP})."}

    return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
            "reason": f"Both teams have wins ({hw} vs {aw}) — outside the file."}


def settle_simple(simple_bet: str, home_score: int, away_score: int):
    if home_score > away_score: result = "home_win"
    elif away_score > home_score: result = "away_win"
    else: result = "draw"
    if simple_bet == "1X":   hit = result in ("home_win", "draw")
    elif simple_bet == "X2": hit = result in ("away_win", "draw")
    else: return result, None, 0.0, 0.0
    profit = 1.0 if hit else -1.0
    return result, (1 if hit else 0), 1.0, profit


def compute_simple_row(facts: dict) -> dict:
    """Read 4 streak numbers from a facts dict; return classifier output + canonical inputs."""
    def _get(*keys):
        for k in keys:
            if k in facts and facts[k] is not None:
                return facts[k]
        return None

    hw = _get("home_consecutive_wins", "Home_Consecutive_Wins", "home_wins_streak")
    aw = _get("away_consecutive_wins", "Away_Consecutive_Wins", "away_wins_streak")
    hu = _get("home_consecutive_unbeaten", "Home_Consecutive_Unbeaten", "home_unbeaten_streak")
    au = _get("away_consecutive_unbeaten", "Away_Consecutive_Unbeaten", "away_unbeaten_streak")

    c = classify_simple(hw, aw, hu, au)
    return {
        "simple_category": c["category"], "simple_pick_team": c["pick_team"],
        "simple_bet": c["bet"], "simple_qualifying": c["qualifying"],
        "simple_reason": c["reason"],
        "home_consecutive_wins": _to_int_or_none(hw),
        "away_consecutive_wins": _to_int_or_none(aw),
        "home_consecutive_unbeaten": _to_int_or_none(hu),
        "away_consecutive_unbeaten": _to_int_or_none(au),
    }


# ═══════════════════════════════════════════════════════════════════
# RISK TAXONOMY (legacy engine)
# ═══════════════════════════════════════════════════════════════════

def _compute_form_residual(row: dict, direction: str):
    if direction not in ("home", "away"): return None
    if direction == "home":
        pts = row.get("home_table_pts"); played = row.get("home_table_played"); form = row.get("home_form_ppg")
    else:
        pts = row.get("away_table_pts"); played = row.get("away_table_played"); form = row.get("away_form_ppg")
    if pts is None or played is None or played == 0 or form is None: return None
    try: return round(float(form) - float(pts) / float(played), 3)
    except (TypeError, ValueError): return None


def _compute_table_dissent(row: dict, direction: str) -> bool:
    if direction == "home": return row.get("signal_table") == "away"
    if direction == "away": return row.get("signal_table") == "home"
    return False


def _compute_rank_gap(row: dict, direction: str):
    if direction == "home":
        hp, ap = row.get("home_table_pos"), row.get("away_table_pos")
    elif direction == "away":
        hp, ap = row.get("away_table_pos"), row.get("home_table_pos")
    else: return None
    if hp is None or ap is None: return None
    try: return int(hp) - int(ap)
    except (TypeError, ValueError): return None


def _compute_goal_tier(row: dict) -> str:
    split = row.get("agreement_split"); gap = row.get("goal_gap")
    if split in ("4-0", "0-4"):
        if gap is None: return "unanimous_no_gap"
        try: g = float(gap)
        except (TypeError, ValueError): return "unanimous_no_gap"
        if g >= 0.75: return "unanimous_goal_075_plus"
        if g >= 0.25: return "unanimous_goal_025_075"
        return "unanimous_goal_under_025"
    return "non_unanimous"


_RISK_TIER_RULES = (
    ("unanimous_strong_gap_no_form_risk", lambda fa, d, gt: gt == "unanimous_goal_075_plus" and not fa, 1),
    ("unanimous_strong_gap_with_form_risk", lambda fa, d, gt: gt == "unanimous_goal_075_plus", 2),
    ("form_above_and_dissent", lambda fa, d, gt: fa and d, 4),
    ("form_above", lambda fa, d, gt: fa, 3),
    ("dissent", lambda fa, d, gt: d, 3),
    ("clean", lambda fa, d, gt: True, 2),
)


def _compute_risk_tier(row: dict, direction: str) -> int:
    if direction not in ("home", "away"): return 0
    form_above = ((_compute_form_residual(row, direction) or 0) > 0)
    dissent = _compute_table_dissent(row, direction)
    goal_tier = _compute_goal_tier(row)
    for _, predicate, tier in _RISK_TIER_RULES:
        if predicate(form_above, dissent, goal_tier):
            return tier
    return 2


def _compute_backtest_confidence(row: dict, direction: str) -> float:
    if direction not in ("home", "away"): return 0.0
    base = 0.87 if direction == "home" else 0.78
    fr = _compute_form_residual(row, direction)
    if fr is not None and fr > 0: base -= 0.08
    elif fr is not None and fr <= 0: base += 0.03
    if _compute_table_dissent(row, direction): base -= 0.15
    gt = _compute_goal_tier(row)
    if gt == "unanimous_goal_075_plus": base += 0.10
    elif gt == "unanimous_goal_under_025": base -= 0.05
    return round(max(0.0, min(1.0, base)), 3)


def _compute_risk_profile(row: dict, direction: str):
    if direction not in ("home", "away"): return None
    return {
        "form_residual": _compute_form_residual(row, direction),
        "table_dissent": _compute_table_dissent(row, direction),
        "goal_tier": _compute_goal_tier(row),
        "risk_tier": _compute_risk_tier(row, direction),
        "backtest_confidence": _compute_backtest_confidence(row, direction),
        "rank_gap": _compute_rank_gap(row, direction),
    }


# ═══════════════════════════════════════════════════════════════════
# ACCUMULATOR SCORE (legacy engine)
# ═══════════════════════════════════════════════════════════════════

def _is_missing(value):
    if value is None: return True
    try: return bool(pd.isna(value))
    except (TypeError, ValueError): return False


def _zero_filled_difference(home_value, away_value):
    if _is_missing(home_value) or _is_missing(away_value): return Decimal("0")
    try: return Decimal(str(home_value)) - Decimal(str(away_value))
    except Exception: return Decimal("0")


def calc_accumulator_score(facts):
    raw = (_zero_filled_difference(facts.get("home_table_pts"), facts.get("away_table_pts"))
           + Decimal("5") * _zero_filled_difference(facts.get("home_form_ppg"), facts.get("away_form_ppg"))
           + Decimal("3") * _zero_filled_difference(facts.get("home_home_ppg"), facts.get("away_away_ppg"))
           + _zero_filled_difference(facts.get("h2h_home_wins"), facts.get("h2h_away_wins")))
    return raw.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def compute_accumulator(facts, decision):
    qualifying = decision.get("qualifying")
    if _is_missing(qualifying) or int(qualifying) != 1: return None
    direction = decision.get("direction")
    if _is_missing(direction) or direction == "skip": return False
    return bool(abs(calc_accumulator_score(facts)) >= Decimal("15.0"))


# ═══════════════════════════════════════════════════════════════════
# GOAL PROXY (legacy engine)
# ═══════════════════════════════════════════════════════════════════

def _safe_rate(gf, played):
    if gf is None or played is None or played == 0: return None
    try: return float(gf) / float(played)
    except (TypeError, ValueError, ZeroDivisionError): return None


def _compute_goal_proxy(f):
    home_att = _safe_rate(f.get("home_table_gf"), f.get("home_table_played"))
    away_att = _safe_rate(f.get("away_table_gf"), f.get("away_table_played"))
    home_def = _safe_rate(f.get("home_table_ga"), f.get("home_table_played"))
    away_def = _safe_rate(f.get("away_table_ga"), f.get("away_table_played"))
    if None not in (home_att, away_att, home_def, away_def):
        f["attack_sum"] = round(home_att + away_att, 4)
        f["exp_proxy_avg"] = round((home_att + away_att + home_def + away_def) / 2.0, 4)
    else:
        f["attack_sum"] = None
        f["exp_proxy_avg"] = None


# ═══════════════════════════════════════════════════════════════════
# SPORTSGAMBLER HTML EXTRACTION (legacy engine)
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
        if m: f["home_team"] = m.group(1).strip(); f["away_team"] = m.group(2).strip()
    if not f.get("home_team"):
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", og["content"])
            if m: f["home_team"] = m.group(1).strip(); f["away_team"] = m.group(2).strip()
    if title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", title.get_text())
        if m:
            f["home_team_short"] = m.group(1).strip()
            f["away_team_short"] = m.group(2).strip()
    if not f.get("home_team_short"): f["home_team_short"] = _search_term(f.get("home_team", ""))
    if not f.get("away_team_short"): f["away_team_short"] = _search_term(f.get("away_team", ""))
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
                if m: f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m: f["kickoff_time"] = m.group(1)
            venue = info.find("span", class_="t_venue")
            if venue: f["venue"] = venue.get_text(strip=True)


def _gd_from_cells(cells, row=None):
    if row is not None:
        gd_cell = row.find("td", class_="clt-gd")
        if gd_cell:
            try: return int(gd_cell.get_text(strip=True))
            except (TypeError, ValueError): pass
    for c in reversed(cells[:-1]):
        s = c.strip()
        if (s.startswith("-") or s.startswith("+")) and s[1:].isdigit(): return int(s)
    return None


def _parse_standings(soup, f):
    home_id, away_id = _extract_team_ids_from_t_top(soup)
    if not home_id and not away_id:
        f["_parse_warnings"].append("standings: no team IDs from t_top logos")
    home_team = f.get("home_team", ""); away_team = f.get("away_team", "")
    tables = (soup.find_all("table", class_="clt-table")
              + soup.find_all("table", class_="leage-table")
              + soup.find_all("table", class_="league-table")
              + soup.find_all("table", class_="standings-table"))
    seen = set()
    for table in tables:
        for row in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 4: continue
            target = None
            img = row.find("img"); row_id = _team_id_from_img(img)
            if row_id:
                if row_id == home_id: target = "home"
                elif row_id == away_id: target = "away"
            if not target:
                team_cell = cells[1] if len(cells) > 1 else ""
                if _team_match(home_team, team_cell): target = "home"
                elif _team_match(away_team, team_cell): target = "away"
            if not target or target in seen: continue
            seen.add(target)
            nums = [_int_any(c) for c in cells if _int_any(c) is not None]
            f[f"{target}_table_pos"] = nums[0] if nums else None
            f[f"{target}_table_pts"] = nums[-1] if nums else None
            f[f"{target}_table_gd"] = _gd_from_cells(cells, row=row)
            if len(cells) >= 11:
                f[f"{target}_table_played"] = _int_any(cells[2])
                f[f"{target}_table_gf"] = _int_any(cells[6])
                f[f"{target}_table_ga"] = _int_any(cells[7])
        if len(seen) == 2: break


_WIN_WORDS  = r"(?:wins?|victor(?:y|ies)|won)"
_LOSS_WORDS = r"(?:loss(?:es)?|defeats?|lost)"
_DRAW_WORDS = r"(?:draws?|drawn)"
_NUM_PAT    = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

_WDL_PATTERNS = [
    (re.compile(rf"{_NUM_PAT}\s+{_WIN_WORDS}[,.]?\s+{_NUM_PAT}\s+{_DRAW_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", re.I), ("w","d","l")),
    (re.compile(rf"{_NUM_PAT}\s+{_WIN_WORDS}[,.]?\s+{_NUM_PAT}\s+{_LOSS_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", re.I), ("w","l","d")),
    (re.compile(rf"{_NUM_PAT}\s+{_LOSS_WORDS}[,.]?\s+{_NUM_PAT}\s+{_WIN_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", re.I), ("l","w","d")),
    (re.compile(rf"{_NUM_PAT}\s+{_LOSS_WORDS}[,.]?\s+{_NUM_PAT}\s+{_DRAW_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", re.I), ("l","d","w")),
    (re.compile(rf"{_NUM_PAT}\s+{_DRAW_WORDS}[,.]?\s+{_NUM_PAT}\s+{_WIN_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", re.I), ("d","w","l")),
    (re.compile(rf"{_NUM_PAT}\s+{_DRAW_WORDS}[,.]?\s+{_NUM_PAT}\s+{_LOSS_WORDS}[,.]?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", re.I), ("d","l","w")),
    (re.compile(rf"won\s+{_NUM_PAT}[,.]?\s+lost\s+{_NUM_PAT}\s+and\s+drawn\s+{_NUM_PAT}", re.I), ("w","l","d")),
    (re.compile(rf"won\s+{_NUM_PAT}[,.]?\s+drawn\s+{_NUM_PAT}\s+and\s+lost\s+{_NUM_PAT}", re.I), ("w","d","l")),
    (re.compile(rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", re.I), ("w","l")),
    (re.compile(rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", re.I), ("l","w")),
    (re.compile(rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", re.I), ("l","d")),
    (re.compile(rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", re.I), ("d","l")),
    (re.compile(rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", re.I), ("w","d")),
    (re.compile(rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", re.I), ("d","w")),
]


def _extract_window_size(text, default=10):
    m = re.search(r"(?:previous|last|past|in\s+the\s+last|in\s+the\s+past|in\s+the\s+previous)\s+(\d+)\s+(?:home\s+|away\s+)?(?:matches?|games?)", text, re.I)
    if m: return int(m.group(1))
    m = re.search(r"in\s+the\s+(?:last|previous|past)\s+(\d+)", text, re.I)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s+(?:home\s+|away\s+)?(?:matches?|games?)", text, re.I)
    if m: return int(m.group(1))
    return default


def _find_wdl(text: str, min_total: int, max_total: int):
    for pattern, mapping in _WDL_PATTERNS:
        m = pattern.search(text)
        if not m: continue
        vals = {"w": 0, "d": 0, "l": 0}
        for i, key in enumerate(mapping):
            vals[key] = _num(m.group(i + 1)) or 0
        if len(mapping) == 2:
            window = _extract_window_size(text, default=sum(vals.values()))
            missing = window - (vals["w"] + vals["d"] + vals["l"])
            if missing < 0: continue
            for slot in ("w", "d", "l"):
                if slot not in mapping: vals[slot] = missing; break
        total = vals["w"] + vals["d"] + vals["l"]
        if total < min_total or total > max_total: continue
        return vals["w"], vals["d"], vals["l"]
    return None


def _wdl_from_cell(text):
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text)
    if m: return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _venue_sentence(sents):
    keywords = ["in the last 10 home", "in the last 10 away", "in the past 10 home", "in the past 10 away",
                "in the previous 10 home", "in the previous 10 away", "in their last 10 home", "in their last 10 away",
                "in their past 10 home", "in their past 10 away", "last 10 home matches", "last 10 away matches"]
    return [s for s in sents if any(k in s.lower() for k in keywords)]


def _parse_venue_from_sentences(sents):
    for s in sents:
        result = _find_wdl(s, min_total=5, max_total=15)
        if result: return result
    return None


def _apply_venue(f, side, wdl):
    w, d, l = wdl
    total = w + d + l or 1
    pts = w * 3 + d
    prefix = "home_home" if side == "home" else "away_away"
    f[f"{prefix}_wins"] = w; f[f"{prefix}_draws"] = d; f[f"{prefix}_losses"] = l
    f[f"{prefix}_pts"] = pts; f[f"{prefix}_ppg"] = round(pts / total, 2)


def _parse_venue_from_keystat(soup):
    home_wdl = away_wdl = None
    home_gf = home_ga = away_gf = away_ga = None
    for item in soup.find_all("div", class_="keystat-item"):
        classes = item.get("class", [])
        is_away = "awaystats" in classes
        title = item.find("div", class_="stats-title-sub")
        if not title or "Full-Time Result" not in title.get_text(): continue
        ul = item.find("ul", class_="stats-list")
        if not ul: continue
        lis = ul.find_all("li")
        if len(lis) < 2: continue
        text = lis[1].get_text(" ", strip=True)
        wdl = _find_wdl(text, min_total=5, max_total=15)
        if not wdl: continue
        if is_away: away_wdl = wdl
        else: home_wdl = wdl
    for item in soup.find_all("div", class_="keystat-item"):
        classes = item.get("class", [])
        is_away = "awaystats" in classes
        title = item.find("div", class_="stats-title-sub")
        if not title or "Goals" not in title.get_text(): continue
        ul = item.find("ul", class_="stats-list")
        if not ul: continue
        lis = ul.find_all("li")
        if len(lis) < 2: continue
        text = lis[1].get_text(" ", strip=True)
        m = re.search(r"(\d+\.\d+)\s+goals?\s+scored\s+and\s+(\d+\.\d+)\s+conceded", text, re.I)
        if not m: continue
        gf = float(m.group(1)); ga = float(m.group(2))
        if is_away: away_gf, away_ga = gf, ga
        else: home_gf, home_ga = gf, ga
    return home_wdl, away_wdl, home_gf, home_ga, away_gf, away_ga


def _parse_venue(soup, home_sents, away_sents, f):
    home_id, away_id = _extract_team_ids_from_t_top(soup)
    st_table = soup.find("table", class_="st-table")
    home_wdl = away_wdl = None
    if st_table:
        for row in st_table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 2: continue
            wdl = None
            for c in cells:
                wdl = _wdl_from_cell(c)
                if wdl: break
            if not wdl: continue
            img = row.find("img"); row_id = _team_id_from_img(img)
            label = " ".join(cells).lower()
            target = None
            if row_id:
                if row_id == home_id: target = "home"
                elif row_id == away_id: target = "away"
            if not target:
                if "home stats" in label: target = "home"
                elif "away stats" in label: target = "away"
            if target == "home" and not home_wdl:
                home_wdl = wdl
                if len(cells) >= 5:
                    f["home_home_gf"] = _float_any(cells[3]); f["home_home_ga"] = _float_any(cells[4])
            elif target == "away" and not away_wdl:
                away_wdl = wdl
                if len(cells) >= 5:
                    f["away_away_gf"] = _float_any(cells[3]); f["away_away_ga"] = _float_any(cells[4])
    if home_wdl: _apply_venue(f, "home", home_wdl)
    if away_wdl: _apply_venue(f, "away", away_wdl)
    if "home_home_ppg" not in f or "away_away_ppg" not in f:
        ks_home, ks_away, h_gf, h_ga, a_gf, a_ga = _parse_venue_from_keystat(soup)
        if "home_home_ppg" not in f and ks_home: _apply_venue(f, "home", ks_home)
        if "away_away_ppg" not in f and ks_away: _apply_venue(f, "away", ks_away)
        if f.get("home_home_gf") is None and h_gf is not None:
            f["home_home_gf"] = h_gf; f["home_home_ga"] = h_ga
        if f.get("away_away_gf") is None and a_gf is not None:
            f["away_away_gf"] = a_gf; f["away_away_ga"] = a_ga
    if "home_home_ppg" not in f and home_sents:
        result = _parse_venue_from_sentences(_venue_sentence(home_sents))
        if result: _apply_venue(f, "home", result)
    if "away_away_ppg" not in f and away_sents:
        result = _parse_venue_from_sentences(_venue_sentence(away_sents))
        if result: _apply_venue(f, "away", result)


def _parse_form_from_sentences(sents, required_first_word=None, other_team=None):
    venue_sents = set(_venue_sentence(sents))
    filtered = [s for s in sents if s not in venue_sents]
    cleaned = []
    for s in filtered:
        if "G W D L" in s and "View Full Table" in s:
            s = s.split("View Full Table", -1)[-1]
        if "W-D-L" in s and "G GF GA" in s and "Home Stats" in s: continue
        if len(s) < 20: continue
        cleaned.append(s)
    for s in cleaned:
        result = _find_wdl(s, min_total=3, max_total=15)
        if result: return result
    return None


def _parse_form_from_keystat(soup):
    home_wdl = away_wdl = None
    for item in soup.find_all("div", class_="keystat-item"):
        classes = item.get("class", [])
        is_away = "awaystats" in classes
        title = item.find("div", class_="stats-title-sub")
        if not title or "Full-Time Result" not in title.get_text(): continue
        ul = item.find("ul", class_="stats-list")
        if not ul: continue
        first_li = ul.find("li")
        if not first_li: continue
        text = first_li.get_text(" ", strip=True)
        wdl = _find_wdl(text, min_total=5, max_total=15)
        if not wdl: continue
        if is_away: away_wdl = wdl
        else: home_wdl = wdl
    return home_wdl, away_wdl


def _apply_form(f, side, wdl, source="prose"):
    w, d, l = wdl
    total = w + d + l or 1
    pts = w * 3 + d
    prefix = "home_form" if side == "home" else "away_form"
    f[f"{prefix}_wins"] = w; f[f"{prefix}_draws"] = d; f[f"{prefix}_losses"] = l
    f[f"{prefix}_pts"] = pts; f[f"{prefix}_ppg"] = round(pts / total, 2)
    f[f"{prefix}_source"] = source


def _parse_form(soup, home_sents, away_sents, f):
    home_short = f.get("home_team_short") or f.get("home_team", "")
    away_short = f.get("away_team_short") or f.get("away_team", "")
    home_form = _parse_form_from_sentences(home_sents)
    if home_form: _apply_form(f, "home", home_form, source="prose")
    away_form = _parse_form_from_sentences(away_sents)
    if away_form: _apply_form(f, "away", away_form, source="prose")
    need_home = "home_form_ppg" not in f
    need_away = "away_form_ppg" not in f
    if need_home or need_away:
        ks_home, ks_away = _parse_form_from_keystat(soup)
        if need_home and ks_home: _apply_form(f, "home", ks_home, source="keystat")
        if need_away and ks_away: _apply_form(f, "away", ks_away, source="keystat")


def _team_sentences(aliases, sentences):
    if not aliases: return []
    result = []; seen = set()
    for alias in sorted(aliases, key=len, reverse=True):
        if not alias: continue
        pattern = re.compile(r"\b" + re.escape(alias.lower()) + r"\b", re.I)
        for s in sentences:
            sid = hash(s)
            if sid in seen: continue
            if pattern.search(s):
                if "G W D L" in s and len(s) < 200:
                    if not any(w in s.lower() for w in ("wins", "won", "victor")): continue
                result.append(s); seen.add(sid)
    return result


def _parse_h2h(soup, home_term, away_term, f):
    rows = []
    section = soup.find("div", id="head-to-head")
    if section: rows = section.find_all("li", class_="team-stat-list-item")
    if not rows:
        for container in soup.find_all("div", class_="h2h-inside"):
            candidate = [r for r in container.find_all("li", class_="team-stat-list-item")
                         if re.search(r"\b(19|20)\d{2}\b", r.get_text(" ", strip=True))]
            if len(candidate) >= 5: rows = candidate; break
    if not rows:
        f["h2h_present"] = False
        f["_parse_warnings"].append("H2H: no structured rows found.")
        return
    home_wins = away_wins = draws = 0
    for row in rows:
        sides = row.find_all("span", class_="team-stats-team")
        if len(sides) < 2: continue
        def _side(span):
            score_span = span.find("span", class_="score-right")
            if not score_span: return None, None
            try: score = int(score_span.get_text(strip=True))
            except (TypeError, ValueError): return None, None
            name = re.sub(r"\s*\d+\s*$", "", span.get_text(" ", strip=True)).strip()
            return name, score
        name_a, score_a = _side(sides[0]); name_b, score_b = _side(sides[1])
        if score_a is None or score_b is None: continue
        a_is_home = _team_match(home_term, name_a); a_is_away = _team_match(away_term, name_a)
        b_is_home = _team_match(home_term, name_b); b_is_away = _team_match(away_term, name_b)
        if a_is_home and b_is_away and not a_is_away and not b_is_home: hs, aws = score_a, score_b
        elif b_is_home and a_is_away and not b_is_away and not a_is_home: hs, aws = score_b, score_a
        elif a_is_home and b_is_away: hs, aws = score_a, score_b
        elif b_is_home and a_is_away: hs, aws = score_b, score_a
        else: hs, aws = score_a, score_b
        if hs > aws: home_wins += 1
        elif aws > hs: away_wins += 1
        else: draws += 1
    f["h2h_home_wins"] = home_wins; f["h2h_away_wins"] = away_wins
    f["h2h_draws"] = draws; f["h2h_total"] = home_wins + away_wins + draws
    f["h2h_present"] = f["h2h_total"] > 0


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
    f["home_team_id"] = home_id; f["away_team_id"] = away_id
    _extract_identity(soup, f)
    if not f.get("home_team"): f["_parse_warnings"].append("home_team not found")
    if not f.get("away_team"): f["_parse_warnings"].append("away_team not found")
    _parse_standings(soup, f)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
    home_aliases = _team_aliases(soup, f.get("home_team", ""), f.get("home_team_short", ""))
    away_aliases = _team_aliases(soup, f.get("away_team", ""), f.get("away_team_short", ""))
    home_sents = _team_sentences(home_aliases, sentences)
    away_sents = _team_sentences(away_aliases, sentences)
    _parse_form(soup, home_sents, away_sents, f)
    _parse_venue(soup, home_sents, away_sents, f)
    _parse_h2h(soup, f.get("home_team_short") or f.get("home_team", ""),
               f.get("away_team_short") or f.get("away_team", ""), f)
    for k in ["home_table_gd", "away_table_gd", "home_table_pts", "away_table_pts",
              "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_total",
              "home_home_gf", "home_home_ga", "away_away_gf", "away_away_ga"]:
        f.setdefault(k, None)
    _compute_goal_proxy(f)
    return f


# ═══════════════════════════════════════════════════════════════════
# TEXT-DUMP PARSER — Recent form / Consecutive blocks / H2H tail
# ═══════════════════════════════════════════════════════════════════

_CONSEC_HEADER = re.compile(r"^Consecutive\s+(scoring|wins|unbeaten)\s*$", re.I)
_RECORD_PAREN  = re.compile(r"\(Record\s+(\d+)\)", re.I)
_SCORE_LINE    = re.compile(r"^\d+\s*-\s*\d+$")
_H2H_TAIL      = re.compile(r"^(\d+)\s+Wins?\s*$", re.I)
_FORM_RESULT   = re.compile(r"^[WDL]$")


def _strip_lines(text: str) -> list:
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]


def _next_nonblank(lines, start):
    i = start
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines): return None, i
    return lines[i].strip(), i + 1


def _parse_match_information(lines, f):
    """Venue + referee names + league (best effort)."""
    for i, ln in enumerate(lines):
        if ln.strip().lower() != "match information":
            continue
        val, j = _next_nonblank(lines, i + 1)
        if not val: return
        # The next 1-2 non-blank lines may be venue + referees, before "Recent form".
        entries = []
        while val and val.strip().lower() != "recent form" and len(entries) < 4:
            entries.append(val.strip())
            val, j = _next_nonblank(lines, j)
        if entries:
            f["venue"] = entries[0]
        return


def _parse_teams_from_recent_form(lines, rf_idx):
    """
    Derive home/away team names from the recent-form block.
    Two heuristic sources:
      1. The most frequent team names in the triplets (both teams appear across their own games).
      2. If the H2H header lists two names before scores, use those.
    Returns (home_name, away_name) or (None, None).
    """
    # Frequency approach over triplets.
    _, i = _next_nonblank(lines, rf_idx + 1)  # league
    nxt, i = _next_nonblank(lines, i)
    if nxt and not _FORM_RESULT.match(nxt):
        nxt, i = _next_nonblank(lines, i)
    cur = nxt
    names = []
    while cur and _FORM_RESULT.match(cur):
        team, i = _next_nonblank(lines, i)
        score, i = _next_nonblank(lines, i)
        if not team or not score or not _SCORE_LINE.match(score): break
        names.append(team.strip())
        cur, i = _next_nonblank(lines, i)
        if not cur or not _FORM_RESULT.match(cur): break

    # Look ahead to H2H header: two names immediately after "H2H last matches".
    h2h_names = []
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("h2h last matches"):
            a, j = _next_nonblank(lines, i + 1)
            b, _ = _next_nonblank(lines, j)
            if a and b and not _SCORE_LINE.match(a) and not _SCORE_LINE.match(b):
                h2h_names = [a.strip(), b.strip()]
            break

    if len(h2h_names) == 2:
        return h2h_names[0], h2h_names[1]

    # Frequency fallback: two most-common names (preserving first-seen order).
    from collections import Counter
    if not names: return None, None
    c = Counter(names)
    if len(c) < 2: return None, None
    top = [n for n, _ in c.most_common(2)]
    first_seen = {}
    for idx, nm in enumerate(names):
        first_seen.setdefault(nm, idx)
    top.sort(key=lambda nm: first_seen[nm])
    return top[0], top[1]


def parse_recent_form_dump(text: str) -> dict:
    """
    Parse the text dump. Returns:
      home_team, away_team, venue
      home_consecutive_wins, away_consecutive_wins
      home_consecutive_unbeaten, away_consecutive_unbeaten
      home_consecutive_scoring, away_consecutive_scoring
      h2h_home_wins, h2h_draws, h2h_away_wins
      recent_form_triplets  (raw list — cosmetic, not split by team)
    """
    f = {
        "home_team": None, "away_team": None, "venue": None,
        "home_consecutive_scoring": None, "away_consecutive_scoring": None,
        "home_consecutive_wins": None, "away_consecutive_wins": None,
        "home_consecutive_unbeaten": None, "away_consecutive_unbeaten": None,
        "h2h_home_wins": None, "h2h_draws": None, "h2h_away_wins": None,
        "recent_form_triplets": [],
        "_parse_warnings": [],
    }
    lines = _strip_lines(text)
    n = len(lines)

    _parse_match_information(lines, f)

    # Consecutive blocks — hardened: skip any run of "(Record N)" lines.
    for i, ln in enumerate(lines):
        m = _CONSEC_HEADER.match(ln.strip())
        if not m: continue
        kind = m.group(1).lower()
        home_val, j = _next_nonblank(lines, i + 1)
        if home_val is None: continue
        while j < n and _RECORD_PAREN.search(lines[j]):
            j += 1
        away_val, _ = _next_nonblank(lines, j)
        hv = _to_int_or_none(home_val); av = _to_int_or_none(away_val)
        if kind == "scoring":
            f["home_consecutive_scoring"], f["away_consecutive_scoring"] = hv, av
        elif kind == "wins":
            f["home_consecutive_wins"], f["away_consecutive_wins"] = hv, av
        elif kind == "unbeaten":
            f["home_consecutive_unbeaten"], f["away_consecutive_unbeaten"] = hv, av

    # Team names — first from H2H header, fall back to recent-form frequency.
    try:
        rf_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower() == "recent form")
    except StopIteration:
        rf_idx = -1
    if rf_idx >= 0:
        home_name, away_name = _parse_teams_from_recent_form(lines, rf_idx)
        f["home_team"], f["away_team"] = home_name, away_name

        # Recent form triplets — raw list, cosmetic only.
        _, i = _next_nonblank(lines, rf_idx + 1)
        nxt, i = _next_nonblank(lines, i)
        if nxt and not _FORM_RESULT.match(nxt):
            nxt, i = _next_nonblank(lines, i)
        cur = nxt
        while cur and _FORM_RESULT.match(cur) and len(f["recent_form_triplets"]) < 20:
            team, i = _next_nonblank(lines, i)
            score, i = _next_nonblank(lines, i)
            if not team or not score or not _SCORE_LINE.match(score): break
            f["recent_form_triplets"].append(
                {"result": cur.upper(), "team": team.strip(), "score": score.strip()})
            cur, i = _next_nonblank(lines, i)
            if not cur or not _FORM_RESULT.match(cur): break

    # H2H tail: "N Wins" / "M Draws" / "K Wins"
    tail = [ln.strip() for ln in lines if ln.strip()]
    if len(tail) >= 3:
        h = d = a = None
        idx = len(tail) - 1
        if m := _H2H_TAIL.match(tail[idx]): a = int(m.group(1)); idx -= 1
        if idx >= 0:
            if m := re.match(r"^(\d+)\s+Draws?\s*$", tail[idx], re.I):
                d = int(m.group(1)); idx -= 1
        if idx >= 0:
            if m := _H2H_TAIL.match(tail[idx]): h = int(m.group(1))
        if None not in (h, d, a):
            f["h2h_home_wins"], f["h2h_draws"], f["h2h_away_wins"] = h, d, a

    if not f["home_team"]: f["_parse_warnings"].append("home_team not found in text dump")
    if not f["away_team"]: f["_parse_warnings"].append("away_team not found in text dump")

    return f


def _looks_like_html(text: str) -> bool:
    head = text.lstrip()[:200].lower()
    return head.startswith("<") or "<html" in head or "<div" in head


def _looks_like_text_dump(text: str) -> bool:
    low = text.lower()
    return "recent form" in low and "consecutive" in low


# ═══════════════════════════════════════════════════════════════════
# COLUMN MAPPERS
# ═══════════════════════════════════════════════════════════════════

def signals_to_columns(signals):
    return {"signal_table": signals["table"], "signal_form": signals["form"],
            "signal_venue": signals["venue"], "signal_h2h": signals["h2h"]}


def decision_columns(d):
    return {
        "direction": d["direction"], "stake": d["stake"], "qualifying": d["qualifying"],
        "dissent_signal": None,
        "opposing_signal": d.get("opposing_signal"),
        "neutral_signal": d.get("neutral_signal"),
        "agreement_split": d["split"],
        "agreement_home_count": d["home_count"],
        "agreement_away_count": d["away_count"],
        "agreement_nothing_count": d["nothing_count"],
        "table_leader": d.get("table_leader"),
    }


def build_row(facts, signals, decision):
    row = dict(facts)
    for k in ("_parse_warnings", "home_team_short", "away_team_short",
              "home_team_id", "away_team_id", "home_form_source", "away_form_source",
              "recent_form_triplets"):
        row.pop(k, None)

    row.update(signals_to_columns(signals))
    row.update(decision_columns(decision))

    # Simple-engine fields
    simple = compute_simple_row(facts)
    row.update(simple)

    now_iso = datetime.now(timezone.utc).isoformat()
    row["extracted_at"] = now_iso
    row["extracted_at_ts"] = now_iso

    row["accumulator"] = compute_accumulator(facts, decision)
    profile = _compute_goal_profile(facts)
    if profile:
        row["goal_exp_home"] = profile["goal_exp_home"]
        row["goal_exp_away"] = profile["goal_exp_away"]
        row["goal_total"] = profile["goal_total"]
        row["goal_gap"] = profile["goal_gap"]

    direction = decision.get("direction")
    venue = signals.get("venue")
    row["venue_opposition"] = (direction in ("home", "away") and venue not in ("nothing", direction))
    row["would_have_skipped_away_table"] = (direction == "away" and decision.get("opposing_signal") == "table")

    risk = _compute_risk_profile(row, direction)
    if risk:
        row.update(risk)

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
        "home_form_wins", "home_form_draws", "home_form_losses", "home_form_pts", "home_form_ppg",
        "away_form_wins", "away_form_draws", "away_form_losses", "away_form_pts", "away_form_ppg",
        "h2h_total", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_present",
        "signal_table", "signal_form", "signal_venue", "signal_h2h",
        "agreement_split", "agreement_home_count", "agreement_away_count",
        "agreement_nothing_count", "table_leader", "dissent_signal",
        "opposing_signal", "neutral_signal",
        "direction", "stake", "qualifying", "extracted_at", "extracted_at_ts",
        "accumulator", "attack_sum", "exp_proxy_avg",
        "would_have_skipped_away_table", "venue_opposition",
        "home_home_gf", "home_home_ga", "away_away_gf", "away_away_ga",
        "goal_gap", "goal_exp_home", "goal_exp_away", "goal_total",
        "form_residual", "table_dissent", "goal_tier", "risk_tier",
        "backtest_confidence", "rank_gap",
        # simple engine
        "home_consecutive_wins", "away_consecutive_wins",
        "home_consecutive_unbeaten", "away_consecutive_unbeaten",
        "home_consecutive_scoring", "away_consecutive_scoring",
        "simple_category", "simple_pick_team", "simple_bet",
        "simple_qualifying", "simple_reason",
    }
    out = {k: v for k, v in row.items() if k in allowed and v is not None}

    # Defaults to satisfy legacy NOT NULL constraints when the legacy engine
    # didn't fire (text-dump rows have no table/form/venue/h2h signals).
    out.setdefault("direction", "skip")
    out.setdefault("stake", "none")
    out.setdefault("qualifying", 0)

    return out


# ═══════════════════════════════════════════════════════════════════
# UI RENDERERS
# ═══════════════════════════════════════════════════════════════════

def signal_pill(name, value, has_data=True):
    if value == "home": cls, label = "home", "H"
    elif value == "away": cls, label = "away", "A"
    else: cls, label = ("nothing", "T" if has_data else "—")
    return f'<span class="signal-pill {cls}">{name}: {label}</span>'


def streak_pill(name, value):
    if value is None: return ""
    return f'<span class="signal-pill streak">{name}: {value}</span>'


def simple_badge_html(simple_cat, simple_bet, pick_team):
    if not simple_cat: return ""
    cls = {"SOLID CLEAR": "badge-solid", "LIGHT CLEAR HOME": "badge-light",
           "NOT-BEATEN SIGNAL": "badge-notbe"}.get(simple_cat, "badge-skip")
    pick_lbl = "HOME" if pick_team == "home" else ("AWAY" if pick_team == "away" else "")
    return f'<span class="pick-badge {cls}" style="margin-left:8px">{simple_cat} · {simple_bet} {pick_lbl}</span>'


def _goal_profile_html(m: dict) -> str:
    home_exp = m.get("goal_exp_home"); away_exp = m.get("goal_exp_away")
    total = m.get("goal_total"); gap = m.get("goal_gap")
    if home_exp is None or away_exp is None: return ""
    total_pill = f'<span class="goal-pill">Total <strong style="color:#c084fc;">{total:.2f}</strong></span>' if total is not None else ""
    gap_pill = f'<span class="goal-pill">Gap <strong>{gap:.2f}</strong></span>' if gap is not None else ""
    return ('<div class="goal-row">'
            f'<span class="goal-pill">Home exp <strong style="color:#34d399;">{home_exp:.2f}</strong></span>'
            f'<span class="goal-pill">Away exp <strong style="color:#fbbf24;">{away_exp:.2f}</strong></span>'
            f'{total_pill}{gap_pill}</div>')


def _verdict_label(dc_hit) -> str:
    if dc_hit == 1: return "WIN"
    if dc_hit == 0: return "LOSS"
    return "NO BET"


def _render_signal_row(m: dict) -> str:
    pills = [
        signal_pill("Table", m.get("signal_table", "nothing"), _signal_has_data(m, "table")),
        signal_pill("Form", m.get("signal_form", "nothing"), _signal_has_data(m, "form")),
        signal_pill("Venue", m.get("signal_venue", "nothing"), _signal_has_data(m, "venue")),
        signal_pill("H2H", m.get("signal_h2h", "nothing"), _signal_has_data(m, "h2h")),
    ]
    hw = m.get("home_consecutive_wins"); aw = m.get("away_consecutive_wins")
    hu = m.get("home_consecutive_unbeaten"); au = m.get("away_consecutive_unbeaten")
    if any(v is not None for v in (hw, aw, hu, au)):
        pills.append(streak_pill("W", f"{hw if hw is not None else '?'}/{aw if aw is not None else '?'}"))
        pills.append(streak_pill("U", f"{hu if hu is not None else '?'}/{au if au is not None else '?'}"))
    return "".join(pills)


def _tier_badge_html(row: dict) -> str:
    tier = row.get("risk_tier")
    if tier is None or tier == 0: return ""
    labels = {1: "TIER 1 · elite", 2: "TIER 2 · balanced", 3: "TIER 3 · caution", 4: "TIER 4 · dangerous"}
    return f'<span class="pick-badge badge-tier-{tier}" style="margin-left:8px">{labels.get(tier, "")}</span>'


def render_pending_card(m: dict):
    home = m.get("home_team") or "?"; away = m.get("away_team") or "?"
    direction = m.get("direction", "skip"); stake = m.get("stake", "none")
    split = m.get("agreement_split", "?") or "?"
    league = m.get("league") or ""
    date = m.get("match_date") or ""
    is_acc = bool(m.get("accumulator_eligible", False))

    simple_cat = m.get("simple_category"); simple_bet = m.get("simple_bet"); simple_pick = m.get("simple_pick_team")

    if simple_cat:
        # Simple engine drives the card display when present.
        pick_text = f"{simple_cat} — {simple_bet} " + ("HOME" if simple_pick == "home" else "AWAY")
        card_cls = "pick-home" if simple_pick == "home" else "pick-away"
        badge_cls = "badge-solid" if simple_cat == "SOLID CLEAR" else \
                    "badge-light" if simple_cat == "LIGHT CLEAR HOME" else "badge-notbe"
    elif direction == "home":
        pick_text, card_cls, badge_cls = f"HOME — {home} · Home/Draw", "pick-home", "badge-home"
    elif direction == "away":
        pick_text, card_cls, badge_cls = f"AWAY — {away} · Away/Draw", "pick-away", "badge-away"
    else:
        pick_text, card_cls, badge_cls = "SKIP", "skip", "badge-skip"

    acc_badge = '<span class="pick-badge badge-acc" style="margin-left:8px">🎯 ACCUMULATOR</span>' if is_acc else ""
    pills = _render_signal_row(m)
    goal_html = _goal_profile_html(m)
    tier_badge = _tier_badge_html(m)

    st.markdown(f"""
    <div class="match-card {card_cls}">
        <div class="match-title">{home} vs {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge_cls}">{pick_text}</span>{acc_badge}{tier_badge}
        <div class="signal-row">{pills}</div>
        {goal_html}
    </div>
    """, unsafe_allow_html=True)


def render_settled_card(m: dict, is_acc: bool = False):
    home = m.get("home_team") or "?"; away = m.get("away_team") or "?"
    hs = m.get("home_score"); aws = m.get("away_score")
    dc_hit = m.get("dc_hit"); profit = m.get("profit_loss") or 0.0
    stake = m.get("stake", "none"); split = m.get("agreement_split", "?") or "?"
    league = m.get("league") or ""; date = m.get("match_date") or ""

    simple_cat = m.get("simple_category"); simple_bet = m.get("simple_bet"); simple_pick = m.get("simple_pick_team")
    if simple_cat:
        market = f"{simple_cat} · {simple_bet} " + ("HOME" if simple_pick == "home" else "AWAY")
    else:
        market = _market_label(m.get("direction"))

    if dc_hit == 1: cls, badge = "win", "badge-win"
    elif dc_hit == 0: cls, badge = "loss", "badge-loss"
    else: cls, badge = "skip", "badge-skip"

    verdict = _verdict_label(dc_hit)
    pl = f"{profit:+.2f}u" if dc_hit is not None else "—"
    acc_badge = '<span class="pick-badge badge-acc" style="margin-left:8px">🎯 ACCUMULATOR</span>' if is_acc else ""
    tier_badge = _tier_badge_html(m)
    pills = _render_signal_row(m)
    goal_html = _goal_profile_html(m)

    st.markdown(f"""
    <div class="match-card {cls}">
        <div class="match-title">{home} {hs}–{aws} {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge}">{verdict} · {market} · {pl}</span>{acc_badge}{tier_badge}
        <div class="signal-row">{pills}</div>
        {goal_html}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SUMMARIES
# ═══════════════════════════════════════════════════════════════════

def cohort_summary(rows):
    records = []
    for r in rows:
        cohort = _cohort_from_row(r)
        if cohort == "other": continue
        result = r.get("result"); direction = r.get("direction")
        if result is None or direction not in ("home", "away"): continue
        if cohort in ("A", "C"):
            hit = 1 if (direction == "home" and result in ("home_win", "draw")) or \
                       (direction == "away" and result in ("away_win", "draw")) else 0
        elif cohort == "B":
            hit = 1 if result in ("home_win", "draw") else 0
        else: continue
        records.append({"cohort": cohort, "meaning": _cohort_meaning_from_row(r),
                        "market": _cohort_market_display(cohort, direction), "hit": hit})
    if not records:
        return pd.DataFrame(columns=["cohort", "meaning", "market", "n", "wins", "losses", "DC hit %"])
    df = pd.DataFrame(records)
    out = df.groupby(["cohort", "meaning", "market"]).agg(n=("hit", "count"), wins=("hit", "sum")).reset_index()
    out["losses"] = out["n"] - out["wins"]
    out["DC hit %"] = (out["wins"] / out["n"] * 100).round(1)
    return out.sort_values(by=["cohort", "n"], ascending=[True, False])[["cohort", "meaning", "market", "n", "wins", "losses", "DC hit %"]]


def tier_summary(rows):
    records = []
    for r in rows:
        tier = r.get("risk_tier")
        if tier is None or tier == 0: continue
        if r.get("dc_hit") is None: continue
        records.append({"tier": tier, "hit": int(r["dc_hit"]),
                        "pl": float(r.get("profit_loss") or 0.0)})
    if not records:
        return pd.DataFrame(columns=["tier", "label", "n", "wins", "losses", "hit_pct", "net_pl"])
    df = pd.DataFrame(records)
    out = df.groupby("tier").agg(n=("hit", "count"), wins=("hit", "sum"), net_pl=("pl", "sum")).reset_index()
    out["losses"] = out["n"] - out["wins"]
    out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1)
    labels = {1: "1 · elite", 2: "2 · balanced", 3: "3 · caution", 4: "4 · dangerous"}
    out["label"] = out["tier"].map(labels)
    return out.sort_values("tier")[["tier", "label", "n", "wins", "losses", "hit_pct", "net_pl"]]


def simple_summary(rows):
    records = []
    for r in rows:
        if r.get("simple_qualifying") != 1: continue
        if r.get("dc_hit") is None: continue
        records.append({
            "category": r.get("simple_category"),
            "bet": r.get("simple_bet"),
            "hit": int(r["dc_hit"]),
            "pl": float(r.get("profit_loss") or 0.0),
        })
    if not records:
        return pd.DataFrame(columns=["category", "bet", "n", "wins", "losses", "hit_pct", "net_pl"])
    df = pd.DataFrame(records)
    out = df.groupby(["category", "bet"]).agg(n=("hit", "count"), wins=("hit", "sum"),
                                               net_pl=("pl", "sum")).reset_index()
    out["losses"] = out["n"] - out["wins"]
    out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1)
    return out.sort_values("category")[["category", "bet", "n", "wins", "losses", "hit_pct", "net_pl"]]


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_acc, tab_perf, tab_learn, tab_simple = st.tabs(
    ["➕ New Match", "⏳ Pending", "🎯 Accumulator", "📊 Performance",
     "🧠 Learning Log", "🧪 Simple Engine Lab"]
)


# ─── TAB 1 — NEW MATCH ─────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Accepts either a SportsGambler HTML page or a recent-form text dump "
               "(the format with 'Recent form', 'Consecutive scoring/wins/unbeaten', and H2H last matches).")

    raw_in = st.text_area("Input", height=240, label_visibility="collapsed",
                          placeholder="<html> ... </html>   OR   Match information\\nEstadio ...\\nRecent form\\n...")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run = st.button("Run the Logic", type="primary", use_container_width=True)

    if run and raw_in.strip():
        raw = raw_in.strip()
        with st.spinner("Parsing..."):
            try:
                if _looks_like_html(raw):
                    facts = extract_facts(raw)
                    source = "html"
                elif _looks_like_text_dump(raw):
                    facts = parse_recent_form_dump(raw)
                    source = "text"
                else:
                    st.error("Input is neither SportsGambler HTML nor a recent-form text dump.")
                    st.stop()
                signals = compute_signals(facts)
                decision = decide(signals, facts)
                simple = compute_simple_row(facts)
            except Exception as e:
                st.error(f"Parse failed: {e}")
                st.stop()

        st.caption(f"Source detected: **{source}**")

        # Simple-engine verdict
        if simple["simple_category"]:
            st.success(f"**Simple → {simple['simple_category']}** · bet **{simple['simple_bet']}** "
                       f"· {simple['simple_reason']}")
        elif simple.get("home_consecutive_wins") is not None:
            st.info(f"**Simple → no bet.** {simple['simple_reason']}")
        else:
            st.caption("Simple engine: no streak inputs in this source.")

        # Legacy-engine verdict
        if decision["direction"] in ("home", "away"):
            st.markdown("### Legacy engine pick")
            preview = {**facts, **signals_to_columns(signals),
                       **decision_columns(decision), **simple}
            preview["accumulator_eligible"] = _is_accumulator_eligible(preview)
            profile = _compute_goal_profile(facts)
            if profile: preview.update(profile)
            risk = _compute_risk_profile(preview, decision["direction"])
            if risk: preview.update(risk)
            render_pending_card(preview)
        else:
            st.markdown("**Legacy engine → skip** (no qualifying signal set).")

        # Show recent-form triplets if present
        if facts.get("recent_form_triplets"):
            with st.expander("Recent form (source order, not split by team)"):
                st.dataframe(pd.DataFrame(facts["recent_form_triplets"]),
                             use_container_width=True, hide_index=True)

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})"):
                for w in warnings: st.write(f"- {w}")

        # Save
        row = build_row(facts, signals, decision)

        if not (row.get("home_team") and row.get("away_team")):
            st.warning("Team names missing — cannot save to DB. Check the paste format.")
        else:
            with st.spinner("Checking for existing match..."):
                try:
                    existing = db_find_existing(row)
                except Exception as e:
                    existing = None
                    st.warning(f"Duplicate check failed: {e}")

            if existing:
                st.info(f"Match already stored (match_id **{existing['match_id']}**). Not re-saved.")
            else:
                with st.spinner("Saving..."):
                    try:
                        db_insert(row)
                        st.success("Saved.")
                    except DecisionMismatchError as e:
                        st.error("Reconciliation failed — row not saved.")
                        st.json(e.mismatches)
                    except Exception as e:
                        st.error(f"Insert failed: {e}")


# ─── TAB 2 — PENDING ───────────────────────────────────────────────
with tab_pending:
    st.markdown("### Pending matches")
    st.caption("Enter the final score. Simple-engine picks settle via 1X/X2 logic; "
               "legacy picks settle via direction+stake.")

    try:
        pending = db_pending()
    except Exception as e:
        st.error(f"Could not load pending: {e}")
        pending = []

    if not pending:
        st.info("No pending matches.")
    else:
        for m in pending:
            pending_view = dict(m)
            pending_view["accumulator_eligible"] = _is_accumulator_eligible(m)
            render_pending_card(pending_view)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2:
                    aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        # Prefer simple settlement when the simple engine qualified.
                        if m.get("simple_qualifying") == 1 and m.get("simple_bet"):
                            result, dc_hit, units, profit = settle_simple(
                                m["simple_bet"], int(hs), int(aws))
                        else:
                            result, dc_hit, units, profit = settle(
                                m.get("direction"), m.get("stake"), int(hs), int(aws))
                        db_settle(m["match_id"], int(hs), int(aws), result, dc_hit, profit)
                        st.success("Settled.")
                        st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"])
                    st.success("Deleted.")
                    st.rerun()


# ─── TAB 3 — ACCUMULATOR ───────────────────────────────────────────
with tab_acc:
    st.markdown("### 🎯 Accumulator matches")
    st.caption("Legacy engine: split in {4-0, 0-4, 3-0-1, 0-3-1} and goal_gap ≥ 0.50.")

    show_elite_only = st.checkbox("Elite tier only (risk_tier = 1)", value=False)

    try:
        all_matches = db_all_matches()
    except Exception as e:
        st.error(f"Could not load: {e}")
        all_matches = []

    acc_matches = [m for m in all_matches if _is_accumulator_eligible(m)]
    if show_elite_only:
        acc_matches = [m for m in acc_matches if m.get("risk_tier") == 1]

    if not acc_matches:
        st.info("No accumulator-eligible matches.")
    else:
        total = len(acc_matches)
        settled = [m for m in acc_matches if m.get("home_score") is not None]
        wins = sum(1 for m in settled if m.get("dc_hit") == 1)
        losses = sum(1 for m in settled if m.get("dc_hit") == 0)
        net = sum((m.get("profit_loss") or 0.0) for m in settled)
        pending_n = total - len(settled)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div><div class="metric-label">Accumulator</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-value">{pending_n}</div><div class="metric-label">Pending</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div><div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#ef4444">{losses}</div><div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
        with c5:
            color = "#10b981" if net >= 0 else "#ef4444"
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div><div class="metric-label">Net P/L</div></div>', unsafe_allow_html=True)

        for m in acc_matches:
            if m.get("home_score") is not None:
                render_settled_card(m, is_acc=True)
            else:
                render_pending_card({**m, "accumulator_eligible": True})


# ─── TAB 4 — PERFORMANCE ───────────────────────────────────────────
with tab_perf:
    st.markdown("### Performance")
    try:
        settled = db_settled()
    except Exception as e:
        st.error(f"Could not load: {e}")
        settled = []

    if not settled:
        st.info("No settled matches yet.")
    else:
        clean = [m for m in settled if not _is_quarantined(m)]

        # Overall (both engines)
        df = pd.DataFrame(clean)
        if df.empty:
            st.info("No settled after quarantine.")
        else:
            df["dc_hit"] = pd.to_numeric(df["dc_hit"], errors="coerce")
            df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0)

            simple_bets = [m for m in clean if m.get("simple_qualifying") == 1 and m.get("dc_hit") is not None]
            legacy_bets = [m for m in clean if m.get("simple_qualifying") != 1
                           and m.get("qualifying") == 1 and m.get("dc_hit") is not None]

            st.markdown("### 🧪 Simple engine")
            simple_tbl = simple_summary(clean)
            if simple_tbl.empty:
                st.info("No settled simple-engine picks yet.")
            else:
                st.dataframe(simple_tbl, use_container_width=True, hide_index=True)
                s_n = len(simple_bets)
                s_w = sum(1 for m in simple_bets if m.get("dc_hit") == 1)
                s_net = sum((m.get("profit_loss") or 0.0) for m in simple_bets)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Simple bets", s_n)
                c2.metric("Wins", s_w)
                c3.metric("Losses", s_n - s_w)
                c4.metric("Net P/L", f"{s_net:+.2f}u")
                if s_n: st.caption(f"Hit rate: **{s_w/s_n*100:.1f}%**")

            st.markdown("### 📊 Legacy engine")
            if legacy_bets:
                l_n = len(legacy_bets)
                l_w = sum(1 for m in legacy_bets if m.get("dc_hit") == 1)
                l_net = sum((m.get("profit_loss") or 0.0) for m in legacy_bets)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Legacy bets", l_n)
                c2.metric("Wins", l_w)
                c3.metric("Losses", l_n - l_w)
                c4.metric("Net P/L", f"{l_net:+.2f}u")

                st.markdown("#### Risk tier breakdown (legacy)")
                tier_df = tier_summary(clean)
                if not tier_df.empty:
                    st.dataframe(tier_df, use_container_width=True, hide_index=True)

                st.markdown("#### Cohorts (legacy)")
                cohort_df = cohort_summary(clean)
                if not cohort_df.empty:
                    st.dataframe(cohort_df, use_container_width=True, hide_index=True)
            else:
                st.info("No settled legacy picks yet.")

            st.markdown("### All settled matches")
            for m in clean:
                render_settled_card(m, is_acc=_is_accumulator_eligible(m))


# ─── TAB 5 — LEARNING LOG ─────────────────────────────────────────
with tab_learn:
    st.markdown("### 🧠 Learning Log")
    st.caption("Every extraction is logged with its feature vector and eventual outcome.")

    log = db_learning_log(limit=100)
    if not log:
        st.info("No entries yet. The log fills as you save matches.")
    else:
        st.markdown(f"**{len(log)} most recent:**")
        df = pd.DataFrame(log)
        cols = [c for c in [
            "extracted_at", "league", "home_team", "away_team",
            "simple_category", "simple_bet",
            "home_consecutive_wins", "away_consecutive_wins",
            "home_consecutive_unbeaten", "away_consecutive_unbeaten",
            "direction", "stake", "risk_tier", "result", "dc_hit", "profit_loss",
        ] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)


# ─── TAB 6 — SIMPLE ENGINE LAB ────────────────────────────────────
with tab_simple:
    st.markdown("### 🧪 Simple Engine Lab")
    st.caption("Paste a recent-form text dump. No DB write — inspect parser + classifier output.")

    text_in = st.text_area("Text dump", height=300, label_visibility="collapsed")

    if text_in.strip():
        try:
            facts = parse_recent_form_dump(text_in)
            simple = compute_simple_row(facts)
        except Exception as e:
            st.error(f"Parse failed: {e}")
            st.stop()

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Home W / U", f"{simple.get('home_consecutive_wins')} / {simple.get('home_consecutive_unbeaten')}")
        cB.metric("Away W / U", f"{simple.get('away_consecutive_wins')} / {simple.get('away_consecutive_unbeaten')}")
        cC.metric("Category", simple.get("simple_category") or "—")
        cD.metric("Bet", simple.get("simple_bet") or "—")

        st.caption(simple.get("simple_reason") or "")

        if facts.get("recent_form_triplets"):
            st.markdown("**Recent form (source order, not split by team)**")
            st.dataframe(pd.DataFrame(facts["recent_form_triplets"]),
                         use_container_width=True, hide_index=True)

        with st.expander("Raw parsed fields"):
            st.json({k: v for k, v in facts.items()
                     if v not in (None, [], "") and k != "recent_form_triplets"})
