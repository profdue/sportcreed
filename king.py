#!/usr/bin/env python3
"""
king.py — The Logic. Learning-ready.

PICKING RULE: UNCHANGED. Produces the same direction, stake, and qualifying
that it produced before. The 83.4% pick rate is preserved.

LEARNING LAYER: New fields are captured at extraction time so the system
accumulates data for the risk classification model:
  - form_residual     (team form PPG minus team table PPG)
  - table_dissent     (does table signal oppose the pick?)
  - goal_tier         (unanimous + goal_gap band)
  - risk_tier         (1=elite, 2=balanced, 3=caution, 4=dangerous)
  - backtest_confidence (0-1 score)

New table: learning_log
  - One row per new match with the full feature vector and eventual outcome.
  - This is the raw material for future retraining.

The app's Performance tab gains two new panels:
  - "Risk Tier Breakdown" — hit rate per tier on settled picks.
  - "Learning Log" — recent rows from learning_log.

The app's Accumulator tab gains one new filter: "Elite tier only" (default off).
This filter is OPT-IN. It does not change default behaviour.
"""

import re
import json
import streamlit as st
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
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
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.75rem; padding: 4px 10px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
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


def db_insert(row: dict):
    """Insert a new match. Also appends to learning_log if the tables exist."""
    audit = _reconcile_decision(row)
    if audit["status"] == "mismatch":
        print(f"AUDIT MISMATCH on insert: {audit['mismatches']}")

    sb = get_supabase()
    res = sb.table("matches").insert(row).execute()

    # Learning log — append the feature vector. Non-fatal if the table
    # does not yet exist (the app should still work without it).
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
        }
        sb.table("learning_log").insert(log_entry).execute()
    except Exception:
        # Learning log is best-effort. If the table doesn't exist yet,
        # or the schema differs, we don't block the main insert.
        pass

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


def db_all_matches():
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_learning_log(limit: int = 50):
    """Best-effort read of the learning log. Returns empty list on error."""
    try:
        sb = get_supabase()
        res = (sb.table("learning_log")
                 .select("*")
                 .order("extracted_at", desc=True)
                 .limit(limit)
                 .execute())
        return res.data or []
    except Exception:
        return []


def db_find_existing(row: dict):
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

    # Update learning log if it exists (best-effort)
    try:
        sb.table("learning_log").update({
            "result": result,
            "dc_hit": dc_hit,
            "profit_loss": profit_loss,
        }).eq("match_id", match_id).execute()
    except Exception:
        pass


def db_delete(match_id: int):
    sb = get_supabase()
    sb.table("matches").delete().eq("match_id", match_id).execute()


# ═══════════════════════════════════════════════════════════════════
# RECONCILIATION AUDIT
# ═══════════════════════════════════════════════════════════════════

def _reconcile_decision(row: dict) -> dict:
    stored_signals = {
        "table": row.get("signal_table"),
        "form":  row.get("signal_form"),
        "venue": row.get("signal_venue"),
        "h2h":   row.get("signal_h2h"),
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
    if row.get("dissent_signal") is not None:
        mismatches["dissent_signal"] = {"stored": row.get("dissent_signal"), "recomputed": None, "note": "deprecated"}

    if mismatches:
        return {"status": "mismatch", "mismatches": mismatches}
    return {"status": "ok"}


# ═══════════════════════════════════════════════════════════════════
# QUARANTINE / COHORT
# ═══════════════════════════════════════════════════════════════════

_RETRO_NOTE_MARKER = "Retrospectively reclassified"


def _is_quarantined(row: dict) -> bool:
    notes = row.get("notes")
    if not notes:
        return False
    return _RETRO_NOTE_MARKER in str(notes)


def _cohort_label(signals: dict, direction: str) -> str:
    t, f, v, h = signals.get("table"), signals.get("form"), signals.get("venue"), signals.get("h2h")
    if direction == "home":
        if t == "home" and f == "home" and v == "home" and h == "home":
            return "A"
        if t == "home" and f == "home" and v == "home" and h in ("away", "nothing"):
            return "B"
        if t == "home" and f == "home":
            return "C"
    elif direction == "away":
        if t == "away" and f == "away" and v == "away" and h == "away":
            return "A"
        if t == "away" and f == "away":
            return "C"
    return "other"


def _cohort_from_row(row: dict) -> str:
    sig = {
        "table": row.get("signal_table"),
        "form":  row.get("signal_form"),
        "venue": row.get("signal_venue"),
        "h2h":   row.get("signal_h2h"),
    }
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
    return (
        f"T={_short_signal(signals.get('table'), has_data.get('table', True))}, "
        f"F={_short_signal(signals.get('form'), has_data.get('form', True))}, "
        f"V={_short_signal(signals.get('venue'), has_data.get('venue', True))}, "
        f"H2H={_short_signal(signals.get('h2h'), has_data.get('h2h', True))}"
    )


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
    split = row.get("agreement_split")
    if split not in _ACC_SPLITS: return False
    gap = row.get("goal_gap")
    if gap is None: return False
    try: g = float(gap)
    except (TypeError, ValueError): return False
    if g < _ACC_GAP_MIN: return False
    opposing = row.get("opposing_signal") or row.get("dissent_signal")
    if opposing in _ACC_DISSENT_SIGNALS and g < _ACC_GAP_DISSENT_MIN: return False
    return True


def _accumulator_shadow_flags(row: dict) -> list:
    flags = []
    split = row.get("agreement_split")
    if split not in _ACC_SPLITS: return flags
    gap = row.get("goal_gap")
    if gap is None: return flags
    try: g = float(gap)
    except (TypeError, ValueError): return flags
    if g < _ACC_GAP_MIN: return flags
    if _is_quarantined(row): return flags
    dissent = row.get("opposing_signal") or row.get("dissent_signal")
    if dissent in _ACC_DISSENT_SIGNALS and g < _ACC_GAP_DISSENT_MIN:
        flags.append(f"⛔ Dissent on '{dissent}' with gap {g:.2f} < {_ACC_GAP_DISSENT_MIN:.2f} — not accumulator-eligible.")
    return flags


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
    home_img = teams_cols[0].find("img")
    away_img = teams_cols[-1].find("img")
    return _team_id_from_img(home_img), _team_id_from_img(away_img)


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
    "eleven": 11, "twelve": 12, "single": 1, "a": 1, "an": 1,
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
# THE PICK LOGIC — UNCHANGED
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
    except (ValueError, TypeError): return None
    return {
        "goal_exp_home": home_exp, "goal_exp_away": away_exp,
        "goal_total": round(home_exp + away_exp, 2),
        "goal_gap": round(abs(home_exp - away_exp), 2),
    }


def _pick_goal_gap(profile, direction):
    if profile is None: return None
    if direction == "home": return round(profile["goal_exp_home"] - profile["goal_exp_away"], 2)
    if direction == "away": return round(profile["goal_exp_away"] - profile["goal_exp_home"], 2)
    return None


def _split_dissent(signals: dict, direction: str) -> tuple:
    if direction is None or direction == "skip": return None, None
    opposing = next((k for k, v in signals.items() if v != direction and v != "nothing"), None)
    neutral = next((k for k, v in signals.items() if v == "nothing"), None)
    return opposing, neutral


def decide(signals: dict, facts: dict = None) -> dict:
    """THE PICK LOGIC — BYTE-IDENTICAL TO BEFORE."""
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")
    table_leader = signals["table"]
    table_leader_value = table_leader if table_leader != "nothing" else None

    if home == 4 and nothing == 0: split_str = "4-0"
    elif away == 4 and nothing == 0: split_str = "0-4"
    elif home == 3 and away == 0 and nothing == 1: split_str = "3-0-1"
    elif away == 3 and home == 0 and nothing == 1: split_str = "0-3-1"
    elif home == 3 and away == 1: split_str = "3-1"
    elif away == 3 and home == 1: split_str = "1-3"
    elif home == 2 and away == 1 and nothing == 1: split_str = "2-1-1"
    elif away == 2 and home == 1 and nothing == 1: split_str = "1-2-1"
    elif home == 2 and away == 2: split_str = "2-2"
    elif home == 2 and away == 0 and nothing == 2: split_str = "2-0-2"
    elif away == 2 and home == 0 and nothing == 2: split_str = "0-2-2"
    elif home == 1 and away == 1 and nothing == 2: split_str = "1-1-2"
    else: split_str = f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}"

    if home > away: majority = "home"
    elif away > home: majority = "away"
    else: majority = None

    base = None

    if home == 4 or away == 4:
        direction = "home" if home == 4 else "away"
        opp, neu = _split_dissent(signals, direction)
        base = {"direction": direction, "stake": "full", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif home == 3 and away == 1:
        opp, neu = _split_dissent(signals, majority)
        base = {"direction": majority, "stake": "half", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif away == 3 and home == 1:
        opp, neu = _split_dissent(signals, majority)
        base = {"direction": majority, "stake": "half", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif home == 3 and away == 0 and nothing == 1:
        opp, neu = _split_dissent(signals, "home")
        base = {"direction": "home", "stake": "full", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif away == 3 and home == 0 and nothing == 1:
        opp, neu = _split_dissent(signals, "away")
        base = {"direction": "away", "stake": "full", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif home == 2 and away == 2:
        base = {"direction": "skip", "stake": "none", "qualifying": 0,
                "dissent_signal": None, "opposing_signal": None, "neutral_signal": None,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif home == 2 and away == 1 and nothing == 1:
        opp, neu = _split_dissent(signals, "home")
        base = {"direction": "home", "stake": "half", "qualifying": 1,
                "dissent_signal": None, "opposing_signal": opp, "neutral_signal": neu,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif away == 2 and home == 1 and nothing == 1:
        base = {"direction": "skip", "stake": "none", "qualifying": 0,
                "dissent_signal": None, "opposing_signal": None, "neutral_signal": None,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif home == 2 and away == 0 and nothing == 2:
        base = {"direction": "skip", "stake": "none", "qualifying": 0,
                "dissent_signal": None, "opposing_signal": None, "neutral_signal": None,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    elif away == 2 and home == 0 and nothing == 2:
        base = {"direction": "skip", "stake": "none", "qualifying": 0,
                "dissent_signal": None, "opposing_signal": None, "neutral_signal": None,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    else:
        base = {"direction": "skip", "stake": "none", "qualifying": 0,
                "dissent_signal": None, "opposing_signal": None, "neutral_signal": None,
                "split": split_str, "home_count": home, "away_count": away,
                "nothing_count": nothing, "table_leader": table_leader_value}
    return base


def settle(direction, stake, home_score, away_score):
    if home_score > away_score: result = "home_win"
    elif away_score > home_score: result = "away_win"
    else: result = "draw"
    if direction == "skip": return result, None, 0.0, 0.0
    units = {"full": 1.0, "half": 0.5, "quarter": 0.25, "none": 0.0}[stake]
    if direction == "home": hit = result in ("home_win", "draw")
    elif direction == "away": hit = result in ("away_win", "draw")
    else: hit = False
    profit = units if hit else -units
    return result, (1 if hit else 0), units, profit


# ═══════════════════════════════════════════════════════════════════
# RISK TAXONOMY — NEW, LEARNING-READY
# ═══════════════════════════════════════════════════════════════════

def _compute_form_residual(row: dict, direction: str):
    """
    Selected team's form PPG minus table PPG. Positive = overperforming
    the table (risky per the audit's H3 finding). Negative = underperforming
    (safe). None if either input is missing or played=0.
    """
    if direction not in ("home", "away"): return None
    if direction == "home":
        pts = row.get("home_table_pts"); played = row.get("home_table_played")
        form = row.get("home_form_ppg")
    else:
        pts = row.get("away_table_pts"); played = row.get("away_table_played")
        form = row.get("away_form_ppg")
    if pts is None or played is None or played == 0 or form is None:
        return None
    try:
        return round(float(form) - float(pts) / float(played), 3)
    except (TypeError, ValueError):
        return None


def _compute_table_dissent(row: dict, direction: str) -> bool:
    """True if table signal opposes the pick."""
    if direction == "home": return row.get("signal_table") == "away"
    if direction == "away": return row.get("signal_table") == "home"
    return False


def _compute_rank_gap(row: dict, direction: str):
    """Positive = selected side ranked higher. None if positions missing."""
    if direction == "home":
        hp, ap = row.get("home_table_pos"), row.get("away_table_pos")
    elif direction == "away":
        hp, ap = row.get("away_table_pos"), row.get("home_table_pos")
    else: return None
    if hp is None or ap is None: return None
    try: return int(hp) - int(ap)
    except (TypeError, ValueError): return None


def _compute_goal_tier(row: dict) -> str:
    """Classify the goal gap for unanimous picks, or 'non_unanimous'."""
    split = row.get("agreement_split")
    gap = row.get("goal_gap")
    if split in ("4-0", "0-4"):
        if gap is None: return "unanimous_no_gap"
        try: g = float(gap)
        except (TypeError, ValueError): return "unanimous_no_gap"
        if g >= 0.75: return "unanimous_goal_075_plus"
        if g >= 0.25: return "unanimous_goal_025_075"
        return "unanimous_goal_under_025"
    return "non_unanimous"


def _compute_risk_tier(row: dict) -> int:
    """
    Returns 1, 2, 3, or 4.
      1 = elite (highest confidence)
      2 = balanced
      3 = caution
      4 = dangerous (highest loss rate)

    Based on the audit's cross-tab findings:
      - form_residual > 0 AND table_dissent  → tier 4 (53.8% loss)
      - form_residual > 0 only               → tier 3 (21.5% loss)
      - table_dissent only                   → tier 3 (50.0% loss, small n)
      - unanimous + goal_075_plus            → tier 1 (100% hit, the 33/33)
      - neither factor, unanimous + moderate → tier 2
      - neither factor, non-unanimous        → tier 2
    """
    direction = row.get("direction")
    if direction not in ("home", "away"): return 0

    form_res = _compute_form_residual(row, direction)
    dissent = _compute_table_dissent(row, direction)
    goal_tier = _compute_goal_tier(row)

    form_above = (form_res is not None and form_res > 0)

    if goal_tier == "unanimous_goal_075_plus":
        # The 33/33 bucket
        if form_above: return 2   # unlikely but possible; still strong
        return 1                   # elite

    if form_above and dissent: return 4
    if form_above: return 3
    if dissent: return 3

    return 2


def _compute_backtest_confidence(row: dict) -> float:
    """
    0-1 confidence score. Derived from the three risk dimensions using the
    empirical hit rates from the audit as weights. This is NOT used for
    picking; it is stored for downstream analysis.
    """
    direction = row.get("direction")
    if direction not in ("home", "away"): return 0.0

    # Base rate by direction (from audit)
    base = 0.87 if direction == "home" else 0.78

    # Form residual adjustment (H3: -14.7pp when form above)
    form_res = _compute_form_residual(row, direction)
    if form_res is not None and form_res > 0:
        base -= 0.08  # half the observed gap (conservative)
    elif form_res is not None and form_res <= 0:
        base += 0.03

    # Table dissent adjustment (H12a: large negative)
    if _compute_table_dissent(row, direction):
        base -= 0.15

    # Goal gap adjustment (for unanimous picks)
    goal_tier = _compute_goal_tier(row)
    if goal_tier == "unanimous_goal_075_plus":
        base += 0.10
    elif goal_tier == "unanimous_goal_under_025":
        base -= 0.05

    return round(max(0.0, min(1.0, base)), 3)


# ═══════════════════════════════════════════════════════════════════
# ACCUMULATOR SCORE
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
    raw = (
        _zero_filled_difference(facts.get("home_table_pts"), facts.get("away_table_pts"))
        + Decimal("5") * _zero_filled_difference(facts.get("home_form_ppg"), facts.get("away_form_ppg"))
        + Decimal("3") * _zero_filled_difference(facts.get("home_home_ppg"), facts.get("away_away_ppg"))
        + _zero_filled_difference(facts.get("h2h_home_wins"), facts.get("h2h_away_wins"))
    )
    return raw.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def compute_accumulator(facts, decision):
    qualifying = decision.get("qualifying")
    if _is_missing(qualifying) or int(qualifying) != 1: return None
    direction = decision.get("direction")
    direction_is_valid = (not _is_missing(direction)) and direction != "skip"
    score = calc_accumulator_score(facts)
    eligible = direction_is_valid and abs(score) >= Decimal("15.0")
    return bool(eligible)


# ═══════════════════════════════════════════════════════════════════
# GOAL PROXY
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
# HTML EXTRACTION (largely unchanged)
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
            f["home_team"] = m.group(1).strip(); f["away_team"] = m.group(2).strip()
    if not f.get("home_team"):
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", og["content"])
            if m:
                f["home_team"] = m.group(1).strip(); f["away_team"] = m.group(2).strip()
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
                if m: f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m: f["kickoff_time"] = m.group(1)
            venue = info.find("span", class_="t_venue")
            if venue: f["venue"] = venue.get_text(strip=True)


def _gd_from_cells(cells, row=None):
    if row is not None:
        gd_cell = row.find("td", class_="clt-gd")
        if gd_cell:
            txt = gd_cell.get_text(strip=True)
            try: return int(txt)
            except (TypeError, ValueError): pass
    for c in reversed(cells[:-1]):
        s = c.strip()
        if (s.startswith("-") or s.startswith("+")) and s[1:].isdigit():
            return int(s)
    return None


def _parse_standings(soup, f):
    home_id, away_id = _extract_team_ids_from_t_top(soup)
    if not home_id and not away_id:
        f["_parse_warnings"].append("standings: no team IDs from t_top logos")
    home_team = f.get("home_team", ""); away_team = f.get("away_team", "")
    tables = (soup.find_all("table", class_="clt-table") +
              soup.find_all("table", class_="leage-table") +
              soup.find_all("table", class_="league-table") +
              soup.find_all("table", class_="standings-table"))
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
            pos = nums[0] if nums else None
            pts = nums[-1] if nums else None
            gd = _gd_from_cells(cells, row=row)
            f[f"{target}_table_pos"] = pos
            f[f"{target}_table_pts"] = pts
            f[f"{target}_table_gd"]  = gd
            if len(cells) >= 11:
                f[f"{target}_table_played"] = _int_any(cells[2])
                f[f"{target}_table_gf"]     = _int_any(cells[6])
                f[f"{target}_table_ga"]     = _int_any(cells[7])
        if len(seen) == 2: break


_WIN_WORDS  = r"(?:wins?|victor(?:y|ies)|won)"
_LOSS_WORDS = r"(?:loss(?:es)?|defeats?|lost)"
_DRAW_WORDS = r"(?:draws?|drawn)"
_NUM_PAT    = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _wdl_from_cell(text):
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text)
    if m: return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _venue_sentence(sents):
    keywords = ["in the last 10 home", "in the last 10 away",
                "in the past 10 home", "in the past 10 away",
                "in the previous 10 home", "in the previous 10 away",
                "in their last 10 home", "in their last 10 away",
                "in their past 10 home", "in their past 10 away",
                "last 10 home matches", "last 10 away matches"]
    return [s for s in sents if any(k in s.lower() for k in keywords)]


def _extract_window_size(text, default=10):
    m = re.search(r"(?:previous|last|past|in\s+the\s+last|in\s+the\s+past|in\s+the\s+previous)\s+(\d+)\s+(?:home\s+|away\s+)?(?:matches?|games?)", text, re.I)
    if m: return int(m.group(1))
    m = re.search(r"in\s+the\s+(?:last|previous|past)\s+(\d+)", text, re.I)
    if m: return int(m.group(1))
    m = re.search(r"(\d+)\s+(?:home\s+|away\s+)?(?:matches?|games?)", text, re.I)
    if m: return int(m.group(1))
    return default


def _parse_venue_from_sentences(sents):
    patterns = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","w")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","d")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","l")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","d")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","w")),
    ]
    for s in sents:
        for tmpl, mapping in patterns:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m: continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            if len(mapping) == 2:
                window = _extract_window_size(s, default=sum(vals.values()))
                known = vals["w"] + vals["d"] + vals["l"]
                missing = window - known
                if missing < 0: continue
                if "w" not in mapping: vals["w"] = missing
                elif "d" not in mapping: vals["d"] = missing
                else: vals["l"] = missing
            total = sum(vals.values())
            if total < 5 or total > 15: continue
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


_KEYSTAT_PATTERNS = [
    (r"(\d+)\s+wins?,?\s+(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+draws?", ("w","l","d")),
    (r"(\d+)\s+wins?,?\s+(\d+)\s+draws?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("w","d","l")),
    (r"(\d+)\s+(?:defeats?|losses?),?\s+(\d+)\s+wins?\s+and\s+(\d+)\s+draws?", ("l","w","d")),
    (r"(\d+)\s+(?:defeats?|losses?),?\s+(\d+)\s+draws?\s+and\s+(\d+)\s+wins?", ("l","d","w")),
    (r"(\d+)\s+draws?,?\s+(\d+)\s+wins?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("d","w","l")),
    (r"(\d+)\s+draws?,?\s+(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+wins?", ("d","l","w")),
    (r"(\d+)\s+wins?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("w","l")),
    (r"(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+wins?", ("l","w")),
    (r"(\d+)\s+(?:defeats?|losses?)\s+and\s+(\d+)\s+draws?", ("l","d")),
    (r"(\d+)\s+draws?\s+and\s+(\d+)\s+(?:defeats?|losses?)", ("d","l")),
    (r"(\d+)\s+wins?\s+and\s+(\d+)\s+draws?", ("w","d")),
    (r"(\d+)\s+draws?\s+and\s+(\d+)\s+wins?", ("d","w")),
]


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
        wdl = None
        for tmpl, mapping in _KEYSTAT_PATTERNS:
            m = re.search(tmpl, text, re.IGNORECASE)
            if not m: continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = int(m.group(i + 1))
            if len(mapping) == 2:
                window = _extract_window_size(text, default=sum(vals.values()))
                known = vals["w"] + vals["d"] + vals["l"]
                missing = window - known
                if missing < 0: continue
                if "w" not in mapping: vals["w"] = missing
                elif "d" not in mapping: vals["d"] = missing
                else: vals["l"] = missing
            total = sum(vals.values())
            if total < 5 or total > 15: continue
            wdl = (vals["w"], vals["d"], vals["l"]); break
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
                    f["home_home_gf"] = _float_any(cells[3])
                    f["home_home_ga"] = _float_any(cells[4])
            elif target == "away" and not away_wdl:
                away_wdl = wdl
                if len(cells) >= 5:
                    f["away_away_gf"] = _float_any(cells[3])
                    f["away_away_ga"] = _float_any(cells[4])
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
    filtered = cleaned

    patterns_3 = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
        (rf"won\s+{_NUM_PAT},?\s+lost\s+{_NUM_PAT}\s+and\s+drawn\s+{_NUM_PAT}", ("w","l","d")),
        (rf"won\s+{_NUM_PAT},?\s+drawn\s+{_NUM_PAT}\s+and\s+lost\s+{_NUM_PAT}", ("w","d","l")),
    ]
    patterns_2 = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","w")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","d")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","l")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","d")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","w")),
    ]
    for s in filtered:
        for tmpl, mapping in patterns_3:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m: continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            total = sum(vals.values())
            if total < 3 or total > 15: continue
            return vals["w"], vals["d"], vals["l"]
        for tmpl, mapping in patterns_2:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m: continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            window = _extract_window_size(s, default=sum(vals.values()))
            known = vals["w"] + vals["d"] + vals["l"]
            missing = window - known
            if missing < 0: continue
            if "w" not in mapping: vals["w"] = missing
            elif "d" not in mapping: vals["d"] = missing
            else: vals["l"] = missing
            total = sum(vals.values())
            if total < 5 or total > 15: continue
            return vals["w"], vals["d"], vals["l"]
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
        wdl = None
        for tmpl, mapping in _KEYSTAT_PATTERNS:
            m = re.search(tmpl, text, re.IGNORECASE)
            if not m: continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = int(m.group(i + 1))
            if len(mapping) == 2:
                window = _extract_window_size(text, default=sum(vals.values()))
                known = vals["w"] + vals["d"] + vals["l"]
                missing = window - known
                if missing < 0: continue
                if "w" not in mapping: vals["w"] = missing
                elif "d" not in mapping: vals["d"] = missing
                else: vals["l"] = missing
            total = sum(vals.values())
            if total < 5 or total > 15: continue
            wdl = (vals["w"], vals["d"], vals["l"]); break
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
    f[f"{prefix}_pts"] = pts
    f[f"{prefix}_ppg"] = round(pts / total, 2)
    f[f"{prefix}_source"] = source


def _parse_form(soup, home_sents, away_sents, f):
    home_short = f.get("home_team_short") or f.get("home_team", "")
    away_short = f.get("away_team_short") or f.get("away_team", "")
    home_form = _parse_form_from_sentences(home_sents, required_first_word=home_short, other_team=away_short)
    if home_form: _apply_form(f, "home", home_form, source="prose")
    away_form = _parse_form_from_sentences(away_sents, required_first_word=away_short, other_team=home_short)
    if away_form: _apply_form(f, "away", away_form, source="prose")
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
    if "home_form_ppg" not in f:
        loose = _parse_form_from_sentences(home_sents) if home_sents else None
        if loose:
            _apply_form(f, "home", loose, source="loose-prose")
            f["_parse_warnings"].append("home form sourced from loose prose")
    if "away_form_ppg" not in f:
        loose = _parse_form_from_sentences(away_sents) if away_sents else None
        if loose:
            _apply_form(f, "away", loose, source="loose-prose")
            f["_parse_warnings"].append("away form sourced from loose prose")


def _team_sentences(aliases, sentences):
    if not aliases: return []
    result = []; seen = set()
    sorted_aliases = sorted(aliases, key=len, reverse=True)
    for alias in sorted_aliases:
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
    if section:
        rows = section.find_all("li", class_="team-stat-list-item")
    if not rows:
        for container in soup.find_all("div", class_="h2h-inside"):
            candidate = container.find_all("li", class_="team-stat-list-item")
            candidate = [r for r in candidate if re.search(r"\b(19|20)\d{2}\b", r.get_text(" ", strip=True))]
            if len(candidate) >= 5:
                rows = candidate; break
    if not rows:
        f["h2h_present"] = False
        f["_parse_warnings"].append("H2H: no structured rows found in head-to-head container. H2H may exist in prose; not currently parsed.")
        return
    home_wins = away_wins = draws = 0
    unmatched = 0; ambiguous = 0
    for row in rows:
        sides = row.find_all("span", class_="team-stats-team")
        if len(sides) < 2: continue
        def _side(span):
            score_span = span.find("span", class_="score-right")
            if not score_span: return None, None
            try: score = int(score_span.get_text(strip=True))
            except (TypeError, ValueError): return None, None
            name = span.get_text(" ", strip=True)
            name = re.sub(r"\s*\d+\s*$", "", name).strip()
            return name, score
        name_a, score_a = _side(sides[0]); name_b, score_b = _side(sides[1])
        if score_a is None or score_b is None: continue
        a_is_home = _team_match(home_term, name_a); a_is_away = _team_match(away_term, name_a)
        b_is_home = _team_match(home_term, name_b); b_is_away = _team_match(away_term, name_b)
        if a_is_home and b_is_away and not a_is_away and not b_is_home:
            hs, aws = score_a, score_b
        elif b_is_home and a_is_away and not b_is_away and not a_is_home:
            hs, aws = score_b, score_a
        elif a_is_home and b_is_away:
            hs, aws = score_a, score_b
        elif b_is_home and a_is_away:
            hs, aws = score_b, score_a
        else:
            hs, aws = score_a, score_b
            if a_is_home or b_is_home or a_is_away or b_is_away: ambiguous += 1
            else: unmatched += 1
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
# COLUMN MAPPERS — NEW FIELDS ADDED
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
              "home_team_id", "away_team_id", "home_form_source", "away_form_source"):
        row.pop(k, None)
    row.update(signals_to_columns(signals))
    row.update(decision_columns(decision))
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

    # ── NEW: risk taxonomy fields ──
    if direction in ("home", "away"):
        row["form_residual"] = _compute_form_residual(row, direction)
        row["table_dissent"] = _compute_table_dissent(row, direction)
        row["goal_tier"] = _compute_goal_tier(row)
        row["risk_tier"] = _compute_risk_tier(row)
        row["backtest_confidence"] = _compute_backtest_confidence(row)
        row["rank_gap"] = _compute_rank_gap(row, direction)

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
        # ── NEW ──
        "form_residual", "table_dissent", "goal_tier", "risk_tier",
        "backtest_confidence", "rank_gap",
    }
    return {k: v for k, v in row.items() if k in allowed and v is not None}


# ═══════════════════════════════════════════════════════════════════
# UI RENDERERS
# ═══════════════════════════════════════════════════════════════════

def signal_pill(name, value, has_data=True):
    if value == "home": cls, label = "home", "H"
    elif value == "away": cls, label = "away", "A"
    else: cls, label = ("nothing", "T" if has_data else "—")
    return f'<span class="signal-pill {cls}">{name}: {label}</span>'


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
    return "".join([
        signal_pill("Table", m.get("signal_table", "nothing"), _signal_has_data(m, "table")),
        signal_pill("Form",  m.get("signal_form",  "nothing"), _signal_has_data(m, "form")),
        signal_pill("Venue", m.get("signal_venue", "nothing"), _signal_has_data(m, "venue")),
        signal_pill("H2H",   m.get("signal_h2h",   "nothing"), _signal_has_data(m, "h2h")),
    ])


def _tier_badge_html(row: dict) -> str:
    tier = row.get("risk_tier")
    if tier is None: return ""
    labels = {1: "TIER 1 · elite", 2: "TIER 2 · balanced", 3: "TIER 3 · caution", 4: "TIER 4 · dangerous"}
    return f'<span class="pick-badge badge-tier-{tier}" style="margin-left:8px">{labels.get(tier, "")}</span>'


def render_pending_card(m: dict):
    home = m.get("home_team", "?"); away = m.get("away_team", "?")
    direction = m.get("direction", "skip"); stake = m.get("stake", "none")
    split = m.get("agreement_split", "?"); league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""
    is_acc = bool(m.get("accumulator_eligible", False))

    if direction == "home": card_cls, badge_cls = "pick-home", "badge-home"
    elif direction == "away": card_cls, badge_cls = "pick-away", "badge-away"
    else: card_cls, badge_cls = "skip", "badge-skip"

    pick_text = f"HOME — {home} · Home/Draw" if direction == "home" else \
                f"AWAY — {away} · Away/Draw" if direction == "away" else "SKIP"
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
    home = m.get("home_team", "?"); away = m.get("away_team", "?")
    hs = m.get("home_score"); aws = m.get("away_score")
    dc_hit = m.get("dc_hit"); profit = m.get("profit_loss") or 0.0
    stake = m.get("stake", "none"); split = m.get("agreement_split", "?")
    league = m.get("league", "") or ""; date = m.get("match_date", "") or ""
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
# COHORT / TIER SUMMARIES
# ═══════════════════════════════════════════════════════════════════

def cohort_summary(rows):
    records = []
    for r in rows:
        cohort = _cohort_from_row(r)
        if cohort == "other": continue
        result = r.get("result")
        direction = r.get("direction")
        if result is None or direction not in ("home", "away"): continue
        hit = None
        if cohort == "A":
            hit = 1 if (direction == "home" and result in ("home_win", "draw")) or \
                       (direction == "away" and result in ("away_win", "draw")) else 0
        elif cohort == "B":
            hit = 1 if result in ("home_win", "draw") else 0
        elif cohort == "C":
            hit = 1 if (direction == "home" and result in ("home_win", "draw")) or \
                       (direction == "away" and result in ("away_win", "draw")) else 0
        if hit is None: continue
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
    """The new 'Risk Tier Breakdown' panel. Hit rate per tier."""
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
    out = df.groupby("tier").agg(n=("hit", "count"), wins=("hit", "sum"),
                                  net_pl=("pl", "sum")).reset_index()
    out["losses"] = out["n"] - out["wins"]
    out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1)
    labels = {1: "1 · elite", 2: "2 · balanced", 3: "3 · caution", 4: "4 · dangerous"}
    out["label"] = out["tier"].map(labels)
    return out.sort_values("tier")[["tier", "label", "n", "wins", "losses", "hit_pct", "net_pl"]]


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_acc, tab_perf, tab_learn = st.tabs(
    ["➕ New Match", "⏳ Pending", "🎯 Accumulator", "📊 Performance", "🧠 Learning Log"]
)


# ─── TAB 1 — NEW MATCH ──────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match HTML")
    st.caption("Paste the full SportsGambler preview HTML. The app extracts the facts, runs the logic, and shows the pick.")

    html_in = st.text_area("HTML", height=220, label_visibility="collapsed", placeholder="<html> ... </html>")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run = st.button("Run the Logic", type="primary", use_container_width=True)

    if run and html_in.strip():
        with st.spinner("Extracting facts..."):
            try:
                facts = extract_facts(html_in)
                signals = compute_signals(facts)
                decision = decide(signals, facts)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

        cohort = _cohort_label(signals, decision.get("direction"))
        market = _cohort_market_display(cohort, decision.get("direction"))
        has_data = {
            "table": facts.get("home_table_pts") is not None and facts.get("away_table_pts") is not None,
            "form":  facts.get("home_form_ppg") is not None and facts.get("away_form_ppg") is not None,
            "venue": facts.get("home_home_ppg") is not None and facts.get("away_away_ppg") is not None,
            "h2h":   facts.get("h2h_total") is not None and facts.get("h2h_total", 0) > 0,
        }
        meaning = _cohort_meaning(signals, decision.get("direction"), has_data)
        direction = decision.get("direction")
        venue = signals.get("venue")
        venue_opposition = (direction in ("home", "away") and venue not in ("nothing", direction))

        preview = {**facts, **signals_to_columns(signals), **decision_columns(decision)}
        preview["accumulator"] = compute_accumulator(facts, decision)
        preview["accumulator_eligible"] = _is_accumulator_eligible(preview)

        profile = _compute_goal_profile(facts)
        if profile:
            preview["goal_exp_home"] = profile["goal_exp_home"]
            preview["goal_exp_away"] = profile["goal_exp_away"]
            preview["goal_total"]    = profile["goal_total"]
            preview["goal_gap"]      = profile["goal_gap"]

        # Compute risk tier for the preview
        if direction in ("home", "away"):
            preview["form_residual"] = _compute_form_residual(preview, direction)
            preview["table_dissent"] = _compute_table_dissent(preview, direction)
            preview["goal_tier"] = _compute_goal_tier(preview)
            preview["risk_tier"] = _compute_risk_tier(preview)
            preview["backtest_confidence"] = _compute_backtest_confidence(preview)
            preview["rank_gap"] = _compute_rank_gap(preview, direction)

        st.markdown("### The Pick")
        render_pending_card(preview)

        if venue_opposition:
            st.warning(f"⚠️ Venue Opposition: Venue={venue.upper()}, Pick={direction.upper()} (tracking only)")

        if direction in ("home", "away"):
            tier = preview.get("risk_tier")
            if tier == 4:
                st.error(f"🚨 **Risk tier 4 (dangerous)** — form residual above table + table dissent. "
                         f"Historical hit rate in this bucket: 53.8% (7 losses in 13 picks).")
            elif tier == 3:
                st.warning(f"⚠️ **Risk tier 3 (caution)** — one risk factor present. "
                          f"Historical hit rate in this bucket: 78.5%.")
            elif tier == 2:
                st.info(f"✅ **Risk tier 2 (balanced)** — no negative risk factors. "
                       f"Historical hit rate: 85-90%.")
            elif tier == 1:
                st.success(f"💎 **Risk tier 1 (elite)** — unanimous + strong goal gap. "
                          f"Historical hit rate: 100% (33 wins in 33 picks).")

            col1, col2, col3 = st.columns(3)
            with col1:
                fr = preview.get("form_residual")
                if fr is not None:
                    st.metric("Form residual", f"{fr:+.2f}",
                              delta="⚠ risky" if fr > 0 else "✓ safe",
                              delta_color="inverse" if fr > 0 else "normal")
            with col2:
                td = preview.get("table_dissent")
                st.metric("Table dissent", "Yes" if td else "No",
                          delta="⚠ risky" if td else "✓ safe",
                          delta_color="inverse" if td else "normal")
            with col3:
                bc = preview.get("backtest_confidence")
                if bc is not None:
                    st.metric("Backtest confidence", f"{bc:.2f}")

        if profile:
            pick_gap = _pick_goal_gap(profile, direction)
            pick_txt = f" · pick gap **{pick_gap:+.2f}**" if pick_gap is not None else ""
            acc_note = " · **Accumulator-qualified**" if preview["accumulator_eligible"] else ""
            st.markdown(
                f"**Goal profile:** Home exp **{profile['goal_exp_home']:.2f}** · "
                f"Away exp **{profile['goal_exp_away']:.2f}** · "
                f"Total **{profile['goal_total']:.2f}** · Gap **{profile['goal_gap']:.2f}**"
                f"{pick_txt}{acc_note}"
            )

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})"):
                for w in warnings: st.write(f"- {w}")

        st.markdown("### Extracted Facts")
        display_facts = {k: v for k, v in facts.items() if v is not None and not k.startswith("_")}
        st.json(display_facts)

        row = build_row(facts, signals, decision)

        with st.spinner("Checking for existing match..."):
            try:
                existing = db_find_existing(row)
            except Exception as e:
                existing = None
                st.warning(f"Could not check duplicates: {e}")

        if existing:
            st.info(f"Match already stored (match_id **{existing['match_id']}**). "
                    f"Prediction was regenerated but not saved.")
        else:
            with st.spinner("Saving to Supabase..."):
                try:
                    db_insert(row)
                    st.success("Match saved to pending. Learning log updated.")
                except Exception as e:
                    st.error(f"Supabase insert failed: {e}")


# ─── TAB 2 — PENDING ───────────────────────────────────────────────
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
                        result, dc_hit, units, profit = settle(
                            m.get("direction"), m.get("stake"), int(hs), int(aws)
                        )
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
    st.caption("Split in {4-0, 0-4, 3-0-1, 0-3-1} AND goal_gap ≥ 0.50. Form/venue dissent requires gap ≥ 0.65. Retrospective edits excluded.")

    show_elite_only = st.checkbox("Show elite tier only (risk_tier = 1)", value=False,
        help="Filters the accumulator list to unanimous + goal_gap ≥ 0.75 picks. "
             "This is the 33/33 bucket. Opt-in only — the default shows all eligible.")

    try:
        all_matches = db_all_matches()
    except Exception as e:
        st.error(f"Could not load matches: {e}")
        all_matches = []

    acc_matches = [m for m in all_matches if _is_accumulator_eligible(m)]
    if show_elite_only:
        acc_matches = [m for m in acc_matches if m.get("risk_tier") == 1]

    if not acc_matches:
        st.info("No accumulator-eligible matches (with the current filter).")
    else:
        total = len(acc_matches)
        settled = [m for m in acc_matches if m.get("home_score") is not None]
        pending_acc = total - len(settled)
        wins = sum(1 for m in settled if m.get("dc_hit") == 1)
        losses = sum(1 for m in settled if m.get("dc_hit") == 0)
        net = sum((m.get("profit_loss") or 0.0) for m in settled)
        hit_rate = (wins / len(settled) * 100) if settled else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div><div class="metric-label">Accumulator</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-box"><div class="metric-value">{pending_acc}</div><div class="metric-label">Pending</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div><div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:#ef4444">{losses}</div><div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
        with c5:
            color = "#10b981" if net >= 0 else "#ef4444"
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div><div class="metric-label">Net P/L</div></div>', unsafe_allow_html=True)
        st.caption(f"Hit rate: **{hit_rate:.1f}%**")

        for m in acc_matches:
            render_settled_card(m, is_acc=True) if m.get("home_score") is not None else render_pending_card(
                {**m, "accumulator_eligible": True}
            )


# ─── TAB 4 — PERFORMANCE ───────────────────────────────────────────
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
        clean = [m for m in settled if not _is_quarantined(m)]
        df = pd.DataFrame(clean)
        if df.empty:
            st.info("No settled after quarantine.")
        else:
            df["dc_hit"] = pd.to_numeric(df["dc_hit"], errors="coerce")
            df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0)
            df["qualifying"] = pd.to_numeric(df["qualifying"], errors="coerce").fillna(0)
            qualifying = df[df["qualifying"] == 1]
            bets = qualifying[qualifying["dc_hit"].notna()]

            total = len(bets)
            wins = int((bets["dc_hit"] == 1).sum()) if not bets.empty else 0
            losses = int((bets["dc_hit"] == 0).sum()) if not bets.empty else 0
            net = float(df["profit_loss"].sum())
            hit_rate = (wins / total * 100) if total else 0.0

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div><div class="metric-label">Bets</div></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div><div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="metric-box"><div class="metric-value">{losses}</div><div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
            with c4:
                color = "#10b981" if net >= 0 else "#ef4444"
                st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div><div class="metric-label">Net P/L</div></div>', unsafe_allow_html=True)

            st.markdown(f"**Hit rate:** {hit_rate:.1f}%")

            st.markdown("### 🎯 Risk Tier Breakdown")
            st.caption("Hit rate per tier. Tier 1 = elite (unanimous + strong goal gap). "
                       "Tier 4 = dangerous (form above table + table dissent).")
            tier_df = tier_summary(clean)
            if tier_df.empty:
                st.info("No risk-tier data yet.")
            else:
                st.dataframe(tier_df, use_container_width=True, hide_index=True)

            st.markdown("### Cohorts")
            cohort_df = cohort_summary(clean)
            if not cohort_df.empty:
                st.dataframe(cohort_df, use_container_width=True, hide_index=True)

            st.markdown("### All settled matches")
            for m in clean:
                render_settled_card(m, is_acc=_is_accumulator_eligible(m))


# ─── TAB 5 — LEARNING LOG ─────────────────────────────────────────
with tab_learn:
    st.markdown("### 🧠 Learning Log")
    st.caption("Every new extraction is logged with its full feature vector. "
               "This is the raw material for future retraining and analysis. "
               "New rows appear here as you extract matches.")

    log = db_learning_log(limit=100)
    if not log:
        st.info("No learning log entries yet. The log fills as you extract new matches. "
                "If the `learning_log` table does not exist yet, create it with the schema below.")
        with st.expander("Show learning_log schema"):
            st.code("""
CREATE TABLE public.learning_log (
    id                BIGSERIAL PRIMARY KEY,
    match_id          INTEGER REFERENCES public.matches(match_id),
    extracted_at      TIMESTAMPTZ,
    league            TEXT,
    home_team         TEXT,
    away_team         TEXT,
    signal_table      TEXT,
    signal_form       TEXT,
    signal_venue      TEXT,
    signal_h2h        TEXT,
    agreement_split   TEXT,
    direction         TEXT,
    stake             TEXT,
    qualifying        INTEGER,
    opposing_signal   TEXT,
    neutral_signal    TEXT,
    form_residual     NUMERIC,
    table_dissent     BOOLEAN,
    goal_tier         TEXT,
    risk_tier         INTEGER,
    backtest_confidence NUMERIC,
    goal_gap          NUMERIC,
    goal_total        NUMERIC,
    rank_gap          INTEGER,
    result            TEXT,
    dc_hit            INTEGER,
    profit_loss       NUMERIC
);

CREATE INDEX idx_learning_log_extracted_at ON public.learning_log(extracted_at DESC);
CREATE INDEX idx_learning_log_risk_tier ON public.learning_log(risk_tier);
""", language="sql")
    else:
        st.markdown(f"**{len(log)} most recent entries:**")
        df = pd.DataFrame(log)
        # Show the useful columns
        display_cols = [c for c in [
            "extracted_at", "league", "home_team", "away_team",
            "direction", "stake", "risk_tier", "form_residual",
            "table_dissent", "goal_tier", "backtest_confidence",
            "result", "dc_hit", "profit_loss",
        ] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("### Aggregate by Risk Tier")
        if "risk_tier" in df.columns and "dc_hit" in df.columns:
            tier_df = df[df["dc_hit"].notna()].groupby("risk_tier").agg(
                n=("dc_hit", "count"),
                wins=("dc_hit", "sum"),
                pl=("profit_loss", "sum"),
            ).reset_index()
            tier_df["losses"] = tier_df["n"] - tier_df["wins"]
            tier_df["hit_pct"] = (tier_df["wins"] / tier_df["n"] * 100).round(1)
            st.dataframe(tier_df, use_container_width=True, hide_index=True)
