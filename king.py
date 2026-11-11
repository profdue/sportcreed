#!/usr/bin/env python3
"""
king.py — The Logic (text-dump only, header-driven home/away)

Input: a text dump that contains a "Pre-match HOME vs AWAY" line plus:
    Match information / venue
    Recent form / results
    Consecutive scoring / wins / unbeaten
    H2H last matches

Classifier (only logic): 2 stats — consecutive wins + consecutive unbeaten.

    SOLID CLEAR        home or away leader, U >= 2, opponent 0/0  -> 1X / X2
    LIGHT CLEAR HOME   home leader, U == W                         -> 1X
    NOT-BEATEN SIGNAL  both 0 wins, unbeaten gap >= 3              -> 1X / X2
    outside the file   anything else                               -> no bet

One table: public.matches. No learning log, no accumulator, no legacy engine.
"""

import re
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from supabase import create_client, Client


st.set_page_config(page_title="The Logic", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

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
    .match-card.skip      { border-left-color: #64748b; opacity: 0.9; }
    .match-card.win       { border-left-color: #10b981; }
    .match-card.loss      { border-left-color: #ef4444; }
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta  { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;
    }
    .badge-solid  { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-light  { background: #38bdf822; color: #38bdf8; border: 1px solid #38bdf855; }
    .badge-notbe  { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
    .badge-nobet  { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win    { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss   { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.8rem; padding: 5px 11px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
        font-family: monospace;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
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
    st.cache_data.clear()
    return res.data[0] if res.data else None


@st.cache_data(ttl=30)
def db_pending():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .is_("home_score", "null")
             .order("extracted_at", desc=True).execute())
    return res.data or []


@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .not_.is_("home_score", "null")
             .order("settled_at", desc=True).execute())
    return res.data or []


def db_find_existing(row: dict):
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .eq("home_team", row.get("home_team"))
             .eq("away_team", row.get("away_team"))
             .limit(1).execute())
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


def db_delete(match_id: int):
    sb = get_supabase()
    sb.table("matches").delete().eq("match_id", match_id).execute()
    st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _to_int_or_none(v):
    if v is None:
        return None
    try:
        if v != v:
            return None
    except Exception:
        pass
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"):
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _strip_lines(text: str) -> list:
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]


def _next_nonblank(lines, start):
    i = start
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i >= len(lines):
        return None, i
    return lines[i].strip(), i + 1


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

NOT_BEATEN_MIN_GAP = 3


def classify_simple(home_wins, away_wins, home_unbeaten, away_unbeaten):
    hw = _to_int_or_none(home_wins)
    aw = _to_int_or_none(away_wins)
    hu = _to_int_or_none(home_unbeaten)
    au = _to_int_or_none(away_unbeaten)

    if None in (hw, aw, hu, au):
        return {"category": None, "pick_team": None, "bet": None, "qualifying": 0,
                "reason": "Missing one or more streak inputs."}

    home_has, away_has = hw > 0, aw > 0

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


def compute_simple_row(facts: dict) -> dict:
    c = classify_simple(
        facts.get("home_consecutive_wins"),
        facts.get("away_consecutive_wins"),
        facts.get("home_consecutive_unbeaten"),
        facts.get("away_consecutive_unbeaten"),
    )
    return {
        "simple_category":     c["category"],
        "simple_pick_team":    c["pick_team"],
        "simple_bet":          c["bet"],
        "simple_qualifying":   c["qualifying"],
        "simple_reason":       c["reason"],
        "home_consecutive_wins":     _to_int_or_none(facts.get("home_consecutive_wins")),
        "away_consecutive_wins":     _to_int_or_none(facts.get("away_consecutive_wins")),
        "home_consecutive_unbeaten": _to_int_or_none(facts.get("home_consecutive_unbeaten")),
        "away_consecutive_unbeaten": _to_int_or_none(facts.get("away_consecutive_unbeaten")),
    }


def settle_simple(simple_bet: str, home_score: int, away_score: int):
    if home_score > away_score:     result = "home_win"
    elif away_score > home_score:   result = "away_win"
    else:                           result = "draw"
    if simple_bet == "1X":   hit = result in ("home_win", "draw")
    elif simple_bet == "X2": hit = result in ("away_win", "draw")
    else:                    return result, None, 0.0
    profit = 1.0 if hit else -1.0
    return result, (1 if hit else 0), profit


# ═══════════════════════════════════════════════════════════════════
# TEXT-DUMP PARSER
# ═══════════════════════════════════════════════════════════════════

_CONSEC_HEADER = re.compile(r"^Consecutive\s+(scoring|wins|unbeaten)\s*$", re.I)
_RECORD_PAREN  = re.compile(r"\(Record\s+(\d+)\)", re.I)
_SCORE_LINE    = re.compile(r"^\d+\s*-\s*\d+$")
_FORM_RESULT   = re.compile(r"^[WDL]$")

# "Pre-match Leeds United vs Tottenham Hotspur" — also tolerates missing prefix.
# Requires whitespace around "vs" so team names containing "vs" aren't broken.
_VS_HEADER = re.compile(r"^(?:Pre-?match\s+)?(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)


def _looks_like_text_dump(text: str) -> bool:
    low = text.lower()
    if "pre-match" in low and " vs " in low:
        return True
    return "recent form" in low and "consecutive" in low


def _teams_from_header(lines):
    """
    Scan the first non-blank lines for 'Pre-match HOME vs AWAY' (or bare
    'HOME vs AWAY'). Returns (home, away) or (None, None).
    Rejects lines containing digits or odds glyphs.
    """
    for ln in lines[:8]:
        s = ln.strip()
        if not s:
            continue
        # Ignore "Match information" and similar structural labels.
        low = s.lower()
        if low in ("match information", "recent form", "general stats",
                   "pre-match odds", "h2h last matches", "odds"):
            continue
        m = _VS_HEADER.match(s)
        if not m:
            continue
        home, away = m.group(1).strip(), m.group(2).strip()
        if not home or not away:
            continue
        if any(ch.isdigit() for ch in home + away):
            continue
        if any(sym in home + away for sym in "➜➔→•/"):
            continue
        return home, away
    return None, None


def _parse_match_information(lines, f):
    for i, ln in enumerate(lines):
        if ln.strip().lower() != "match information":
            continue
        val, j = _next_nonblank(lines, i + 1)
        if not val:
            return
        entries = []
        while val and val.strip().lower() != "recent form" and len(entries) < 4:
            entries.append(val.strip())
            val, j = _next_nonblank(lines, j)
        if entries:
            f["venue"] = entries[0]
        return


def _teams_frequency_fallback(lines):
    """
    If no 'Pre-match X vs Y' header exists, derive teams from the recent-form
    block's most frequent team names. This is best-effort: the source does not
    guarantee which of the two is home. Use only when header is absent.
    """
    try:
        rf_idx = next(i for i, ln in enumerate(lines)
                      if ln.strip().lower() == "recent form")
    except StopIteration:
        return None, None

    _, i = _next_nonblank(lines, rf_idx + 1)
    nxt, i = _next_nonblank(lines, i)
    if nxt and not _FORM_RESULT.match(nxt):
        nxt, i = _next_nonblank(lines, i)

    cur = nxt
    names = []
    while cur and _FORM_RESULT.match(cur):
        team, i = _next_nonblank(lines, i)
        score, i = _next_nonblank(lines, i)
        if not team or not score or not _SCORE_LINE.match(score):
            break
        names.append(team.strip())
        cur, i = _next_nonblank(lines, i)
        if not cur or not _FORM_RESULT.match(cur):
            break

    from collections import Counter
    if not names:
        return None, None
    c = Counter(names)
    if len(c) < 2:
        return None, None
    top = [n for n, _ in c.most_common(2)]
    first_seen = {}
    for idx, nm in enumerate(names):
        first_seen.setdefault(nm, idx)
    top.sort(key=lambda nm: first_seen[nm])
    return top[0], top[1]


def parse_recent_form_dump(text: str) -> dict:
    f = {
        "home_team": None, "away_team": None, "venue": None,
        "league": None, "match_date": None, "kickoff_time": None, "country": None,
        "home_consecutive_scoring": None, "away_consecutive_scoring": None,
        "home_consecutive_wins": None,    "away_consecutive_wins": None,
        "home_consecutive_unbeaten": None, "away_consecutive_unbeaten": None,
        "recent_form_triplets": [],
        "_parse_warnings": [],
    }
    lines = _strip_lines(text)
    n = len(lines)

    _parse_match_information(lines, f)

    # Teams: header is authoritative; frequency is a fallback for legacy dumps.
    header_home, header_away = _teams_from_header(lines)
    if header_home and header_away:
        f["home_team"], f["away_team"] = header_home, header_away
    else:
        h, a = _teams_frequency_fallback(lines)
        f["home_team"], f["away_team"] = h, a
        if h and a:
            f["_parse_warnings"].append(
                "No 'Pre-match X vs Y' header — team order guessed from recent-form "
                "frequency. Home/away may be swapped.")

    # Consecutive blocks — skip any run of "(Record N)" between the two values.
    for i, ln in enumerate(lines):
        m = _CONSEC_HEADER.match(ln.strip())
        if not m:
            continue
        kind = m.group(1).lower()
        home_val, j = _next_nonblank(lines, i + 1)
        if home_val is None:
            continue
        while j < n and _RECORD_PAREN.search(lines[j]):
            j += 1
        away_val, _ = _next_nonblank(lines, j)
        hv = _to_int_or_none(home_val)
        av = _to_int_or_none(away_val)
        if kind == "scoring":
            f["home_consecutive_scoring"], f["away_consecutive_scoring"] = hv, av
        elif kind == "wins":
            f["home_consecutive_wins"], f["away_consecutive_wins"] = hv, av
        elif kind == "unbeaten":
            f["home_consecutive_unbeaten"], f["away_consecutive_unbeaten"] = hv, av

    # Recent form as raw triplets — cosmetic, not used by the classifier.
    try:
        rf_idx = next(i for i, ln in enumerate(lines)
                      if ln.strip().lower() == "recent form")
    except StopIteration:
        rf_idx = -1
    if rf_idx >= 0:
        _, i = _next_nonblank(lines, rf_idx + 1)
        nxt, i = _next_nonblank(lines, i)
        if nxt and not _FORM_RESULT.match(nxt):
            nxt, i = _next_nonblank(lines, i)
        cur = nxt
        while cur and _FORM_RESULT.match(cur) and len(f["recent_form_triplets"]) < 20:
            team, i = _next_nonblank(lines, i)
            score, i = _next_nonblank(lines, i)
            if not team or not score or not _SCORE_LINE.match(score):
                break
            f["recent_form_triplets"].append(
                {"result": cur.upper(), "team": team.strip(), "score": score.strip()}
            )
            cur, i = _next_nonblank(lines, i)
            if not cur or not _FORM_RESULT.match(cur):
                break

    if not f["home_team"]:
        f["_parse_warnings"].append("home_team not found in text dump")
    if not f["away_team"]:
        f["_parse_warnings"].append("away_team not found in text dump")

    return f


# ═══════════════════════════════════════════════════════════════════
# ROW BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_row(facts: dict) -> dict:
    simple = compute_simple_row(facts)
    now_iso = datetime.now(timezone.utc).isoformat()

    row = {
        "home_team": facts.get("home_team"),
        "away_team": facts.get("away_team"),
        "venue":     facts.get("venue"),
        "league":    facts.get("league"),
        "match_date": facts.get("match_date"),
        "kickoff_time": facts.get("kickoff_time"),
        "country":   facts.get("country"),

        "home_consecutive_wins":     simple["home_consecutive_wins"],
        "away_consecutive_wins":     simple["away_consecutive_wins"],
        "home_consecutive_unbeaten": simple["home_consecutive_unbeaten"],
        "away_consecutive_unbeaten": simple["away_consecutive_unbeaten"],

        "simple_category":     simple["simple_category"],
        "simple_pick_team":    simple["simple_pick_team"],
        "simple_bet":          simple["simple_bet"],
        "simple_qualifying":   simple["simple_qualifying"],
        "simple_reason":       simple["simple_reason"],

        "extracted_at": now_iso,
    }
    return {k: v for k, v in row.items() if v is not None}


# ═══════════════════════════════════════════════════════════════════
# CARD RENDERER
# ═══════════════════════════════════════════════════════════════════

def _card_html(m: dict, settled: bool = False) -> str:
    home = m.get("home_team") or "?"
    away = m.get("away_team") or "?"
    hs = m.get("home_score"); aws = m.get("away_score")
    dc_hit = m.get("dc_hit")
    profit = m.get("profit_loss") or 0.0

    cat   = m.get("simple_category")
    bet   = m.get("simple_bet")
    pick  = m.get("simple_pick_team")
    hw    = m.get("home_consecutive_wins")
    aw    = m.get("away_consecutive_wins")
    hu    = m.get("home_consecutive_unbeaten")
    au    = m.get("away_consecutive_unbeaten")

    cat_cls = {"SOLID CLEAR": "badge-solid",
               "LIGHT CLEAR HOME": "badge-light",
               "NOT-BEATEN SIGNAL": "badge-notbe"}.get(cat, "badge-nobet")
    pick_lbl = "HOME" if pick == "home" else ("AWAY" if pick == "away" else "")
    card_cls = "pick-home" if pick == "home" else ("pick-away" if pick == "away" else "skip")

    meta_bits = [b for b in [m.get("league"), m.get("match_date"), m.get("venue")] if b]
    meta = " · ".join(meta_bits) if meta_bits else "—"

    title = f"{home} {hs}–{aws} {away}" if (settled and hs is not None) else f"{home} vs {away}"

    if settled and dc_hit is not None:
        vcls = "badge-win" if dc_hit == 1 else "badge-loss"
        vlabel = "WIN" if dc_hit == 1 else "LOSS"
        verdict_html = (f'<span class="pick-badge {vcls}" style="margin-left:8px">'
                        f'{vlabel} · {profit:+.2f}u</span>')
    else:
        verdict_html = ""

    badge_text = f"{cat} · {bet} {pick_lbl}" if cat else "NO BET"

    return f"""
    <div class="match-card {card_cls}">
        <div class="match-title">{title}</div>
        <div class="match-meta">{meta}</div>
        <span class="pick-badge {cat_cls}">{badge_text}</span>{verdict_html}
        <div class="signal-row">
            <span class="signal-pill home">Home W/U: {hw} / {hu}</span>
            <span class="signal-pill away">Away W/U: {aw} / {au}</span>
        </div>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(
    ["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"]
)


# ─── TAB 1 — NEW MATCH ────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Include the `Pre-match HOME vs AWAY` header — that is what decides "
               "which side is home. Then the four consecutive stats decide the pick.")

    raw_in = st.text_area("Input", height=320, label_visibility="collapsed",
                          placeholder="Pre-match Leeds United vs Tottenham Hotspur\\n"
                                      "Match information\\nElland Road\\n...")
    run = st.button("Run the Logic", type="primary")

    if run and raw_in.strip():
        raw = raw_in.strip()
        if not _looks_like_text_dump(raw):
            st.error("Input must contain 'Pre-match X vs Y' or "
                     "'Recent form' + 'Consecutive' sections.")
            st.stop()

        try:
            facts = parse_recent_form_dump(raw)
            simple = compute_simple_row(facts)
        except Exception as e:
            st.error(f"Parse failed: {e}")
            st.stop()

        st.markdown("### The Pick")
        preview = {**facts, **simple}
        st.markdown(_card_html(preview, settled=False), unsafe_allow_html=True)

        if simple["simple_category"]:
            st.success(simple["simple_reason"])
        elif simple.get("home_consecutive_wins") is not None:
            st.info(simple["simple_reason"])
        else:
            st.warning("Missing streak values — cannot classify.")

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})"):
                for w in warnings:
                    st.write(f"- {w}")

        has_required = (
            facts.get("home_team") and facts.get("away_team")
            and simple.get("home_consecutive_wins") is not None
            and simple.get("away_consecutive_wins") is not None
            and simple.get("home_consecutive_unbeaten") is not None
            and simple.get("away_consecutive_unbeaten") is not None
        )

        if not has_required:
            st.warning("Missing required fields (team names or streaks) — cannot save.")
        elif not simple["simple_qualifying"]:
            st.warning("No bet — this match is outside the file. Nothing to save.")
        else:
            row = build_row(facts)
            existing = db_find_existing(row)
            if existing:
                st.info(f"Already stored (match_id **{existing['match_id']}**). Not re-saved.")
            else:
                if st.button("Save to database", type="primary"):
                    try:
                        db_insert(row)
                        st.success("Saved.")
                    except Exception as e:
                        st.error(f"Insert failed: {e}")


# ─── TAB 2 — PENDING ──────────────────────────────────────────────
with tab_pending:
    st.markdown("### Pending matches")

    try:
        pending = db_pending()
    except Exception as e:
        st.error(f"Could not load pending: {e}")
        pending = []

    if not pending:
        st.info("No pending matches.")
    else:
        for m in pending:
            st.markdown(_card_html(m, settled=False), unsafe_allow_html=True)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2:
                    aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        result, dc_hit, profit = settle_simple(
                            m.get("simple_bet") or "", int(hs), int(aws))
                        db_settle(m["match_id"], int(hs), int(aws), result, dc_hit, profit)
                        st.success("Settled.")
                        st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"])
                    st.success("Deleted.")
                    st.rerun()


# ─── TAB 3 — PERFORMANCE ──────────────────────────────────────────
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
        bets = [m for m in settled
                if m.get("simple_qualifying") == 1 and m.get("dc_hit") is not None]
        n = len(bets)
        wins = sum(1 for m in bets if m.get("dc_hit") == 1)
        losses = n - wins
        net = sum((m.get("profit_loss") or 0.0) for m in bets)
        hit = (wins / n * 100) if n else 0.0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bets", n)
        c2.metric("Wins", wins)
        c3.metric("Losses", losses)
        c4.metric("Hit %", f"{hit:.1f}%")
        c5.metric("Net P/L", f"{net:+.2f}u")

        if bets:
            st.markdown("### By category")
            df = pd.DataFrame([
                {"category": m.get("simple_category"),
                 "bet": m.get("simple_bet"),
                 "hit": int(m["dc_hit"]),
                 "pl": float(m.get("profit_loss") or 0.0)}
                for m in bets
            ])
            out = df.groupby(["category", "bet"]).agg(
                n=("hit", "count"), wins=("hit", "sum"), net_pl=("pl", "sum")
            ).reset_index()
            out["losses"] = out["n"] - out["wins"]
            out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1)
            st.dataframe(out[["category", "bet", "n", "wins", "losses", "hit_pct", "net_pl"]],
                         use_container_width=True, hide_index=True)

        st.markdown("### All settled")
        for m in settled:
            st.markdown(_card_html(m, settled=True), unsafe_allow_html=True)


# ─── TAB 4 — LAB ──────────────────────────────────────────────────
with tab_lab:
    st.markdown("### 🧪 Lab")
    st.caption("Parse a dump without saving. Inspect the extracted values "
               "and the classifier's decision.")

    text_in = st.text_area("Text dump", height=320, label_visibility="collapsed",
                           key="lab_input")

    if text_in.strip():
        if not _looks_like_text_dump(text_in):
            st.error("Input must contain 'Pre-match X vs Y' or "
                     "'Recent form' + 'Consecutive' sections.")
            st.stop()
        try:
            facts = parse_recent_form_dump(text_in)
            simple = compute_simple_row(facts)
        except Exception as e:
            st.error(f"Parse failed: {e}")
            st.stop()

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Home W / U",
                  f"{simple.get('home_consecutive_wins')} / {simple.get('home_consecutive_unbeaten')}")
        cB.metric("Away W / U",
                  f"{simple.get('away_consecutive_wins')} / {simple.get('away_consecutive_unbeaten')}")
        cC.metric("Category", simple.get("simple_category") or "—")
        cD.metric("Bet", simple.get("simple_bet") or "—")

        st.caption(simple.get("simple_reason") or "")

        with st.expander("Parsed fields"):
            st.json({k: v for k, v in facts.items()
                     if v not in (None, [], "") and k != "recent_form_triplets"})

        if facts.get("recent_form_triplets"):
            with st.expander("Recent form (raw, source order)"):
                st.dataframe(pd.DataFrame(facts["recent_form_triplets"]),
                             use_container_width=True, hide_index=True)
