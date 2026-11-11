#!/usr/bin/env python3
"""
king.py — The Logic (text-dump only, header-driven home/away)

Classifier: 2 stats — consecutive wins + consecutive unbeaten.

  QUALIFYING (simple_qualifying=1):
    SOLID CLEAR        leader U >= 2, opponent 0/0      -> 1X / X2
    LIGHT CLEAR HOME   home leader, U == W               -> 1X
    NOT-BEATEN SIGNAL  both 0 wins, unbeaten gap >= 3    -> 1X / X2

  STUDY (simple_qualifying=0, rejected matches kept for analysis):
    HOME_LEADER_NO_PADDING, AWAY_LEADER_NO_SOLID,
    NOT_BEATEN_GAP_LOW, BOTH_HAVE_WINS, MISSING

Every parsed row inserts. Qualifying bets go to Pending. No-bets go to Study.
Study rows record the actual final score; no hypothetical bets, no predictions.
One table: public.matches.
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
             .eq("simple_qualifying", 1)
             .is_("home_score", "null")
             .order("extracted_at", desc=True).execute())
    return res.data or []


@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .eq("simple_qualifying", 1)
             .not_.is_("home_score", "null")
             .order("settled_at", desc=True).execute())
    return res.data or []


@st.cache_data(ttl=30)
def db_study():
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .eq("simple_qualifying", 0)
             .order("extracted_at", desc=True).execute())
    return res.data or []


def db_find_existing(row: dict):
    sb = get_supabase()
    res = (sb.table("matches").select("*")
             .eq("home_team", row.get("home_team"))
             .eq("away_team", row.get("away_team"))
             .limit(1).execute())
    return res.data[0] if res.data else None


def db_settle(match_id: int, home_score: int, away_score: int,
              result: str, dc_hit, profit_loss):
    """
    Records the final score. dc_hit and profit_loss are only meaningful for
    qualifying bets — study rows pass None/None.
    """
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


def _clean_line(s: str) -> str:
    return s.strip().strip("\ufeff\u200b\u200c\u200d\u00a0 ")


def _strip_lines(text: str) -> list:
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]


def _next_nonblank(lines, start):
    i = start
    while i < len(lines) and not _clean_line(lines[i]):
        i += 1
    if i >= len(lines):
        return None, i
    return _clean_line(lines[i]), i + 1


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIER
# ═══════════════════════════════════════════════════════════════════

NOT_BEATEN_MIN_GAP = 3


def classify_simple(home_wins, away_wins, home_unbeaten, away_unbeaten):
    hw = _to_int_or_none(home_wins)
    aw = _to_int_or_none(away_wins)
    hu = _to_int_or_none(home_unbeaten)
    au = _to_int_or_none(away_unbeaten)

    def result(category=None, pick=None, bet=None, qualifying=0, reason="", code=None):
        return {"category": category, "pick_team": pick, "bet": bet,
                "qualifying": qualifying, "reason": reason, "rejection_code": code}

    if None in (hw, aw, hu, au):
        return result(reason="Missing one or more streak inputs.", code="MISSING")

    home_has, away_has = hw > 0, aw > 0

    if home_has and not away_has:
        if hu >= 2 and au == 0:
            return result("SOLID CLEAR", "home", "1X", 1,
                          f"Home {hw}W/{hu}U vs Away {aw}W/{au}U — SOLID (opponent 0/0).")
        if hu == hw:
            return result("LIGHT CLEAR HOME", "home", "1X", 1,
                          f"Home {hw}W/{hu}U (unbeaten = wins) vs Away {aw}W/{au}U — LIGHT-HOME.")
        return result(reason=f"Home leader {hw}W/{hu}U but unbeaten != wins and SOLID not met.",
                      code="HOME_LEADER_NO_PADDING")

    if away_has and not home_has:
        if au >= 2 and hu == 0:
            return result("SOLID CLEAR", "away", "X2", 1,
                          f"Away {aw}W/{au}U vs Home {hw}W/{hu}U — SOLID (opponent 0/0).")
        return result(reason=f"Away leader {aw}W/{au}U — LIGHT-away removed; SOLID not met.",
                      code="AWAY_LEADER_NO_SOLID")

    if not home_has and not away_has:
        gap = abs(hu - au)
        if gap < NOT_BEATEN_MIN_GAP:
            return result(reason=f"Both 0 wins, unbeaten gap {gap} < {NOT_BEATEN_MIN_GAP}.",
                          code="NOT_BEATEN_GAP_LOW")
        if hu > au:
            return result("NOT-BEATEN SIGNAL", "home", "1X", 1,
                          f"Both 0W; Home unbeaten {hu} vs Away {au} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")
        return result("NOT-BEATEN SIGNAL", "away", "X2", 1,
                      f"Both 0W; Away unbeaten {au} vs Home {hu} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")

    return result(reason=f"Both teams have wins ({hw} vs {aw}) — outside the file.",
                  code="BOTH_HAVE_WINS")


def compute_simple_row(facts: dict) -> dict:
    c = classify_simple(
        facts.get("home_consecutive_wins"),
        facts.get("away_consecutive_wins"),
        facts.get("home_consecutive_unbeaten"),
        facts.get("away_consecutive_unbeaten"),
    )
    return {
        "simple_category":       c["category"],
        "simple_pick_team":      c["pick_team"],
        "simple_bet":            c["bet"],
        "simple_qualifying":     c["qualifying"],
        "simple_reason":         c["reason"],
        "simple_rejection_code": c["rejection_code"],
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
_VS_HEADER     = re.compile(r"^(?:Pre-?match\s+)?(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)

_ODDS_GLYPHS = "➜➔→•€$£"

_STRUCTURAL_LABELS = {
    "match information", "recent form", "general stats",
    "pre-match odds", "h2h last matches", "odds", "advertisement",
}


def _looks_like_text_dump(text: str) -> bool:
    low = text.lower()
    if "pre-match" in low and " vs " in low:
        return True
    return "recent form" in low and "consecutive" in low


def _looks_like_odds_line(s: str) -> bool:
    if any(sym in s for sym in _ODDS_GLYPHS):
        return True
    decimals = re.findall(r"\b\d+\.\d+\b", s)
    if len(decimals) >= 2:
        return True
    return False


def _teams_from_header(lines):
    for ln in lines:
        s = _clean_line(ln)
        if not s:
            continue
        if s.lower() in _STRUCTURAL_LABELS:
            continue
        if _looks_like_odds_line(s):
            continue
        m = _VS_HEADER.match(s)
        if not m:
            continue
        home, away = _clean_line(m.group(1)), _clean_line(m.group(2))
        if not home or not away:
            continue
        if home.isdigit() or away.isdigit():
            continue
        return home, away
    return None, None


def _parse_match_information(lines, f):
    for i, ln in enumerate(lines):
        if _clean_line(ln).lower() != "match information":
            continue
        val, j = _next_nonblank(lines, i + 1)
        if not val:
            return
        entries = []
        while val and val.lower() != "recent form" and len(entries) < 4:
            entries.append(val)
            val, j = _next_nonblank(lines, j)
        if entries:
            f["venue"] = entries[0]
        return


def _teams_frequency_fallback(lines):
    try:
        rf_idx = next(i for i, ln in enumerate(lines)
                      if _clean_line(ln).lower() == "recent form")
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
        names.append(team)
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

    for i, ln in enumerate(lines):
        m = _CONSEC_HEADER.match(_clean_line(ln))
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

    try:
        rf_idx = next(i for i, ln in enumerate(lines)
                      if _clean_line(ln).lower() == "recent form")
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
                {"result": cur.upper(), "team": team, "score": score}
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

        "simple_category":       simple["simple_category"],
        "simple_pick_team":      simple["simple_pick_team"],
        "simple_bet":            simple["simple_bet"],
        "simple_qualifying":     simple["simple_qualifying"],
        "simple_reason":         simple["simple_reason"],
        "simple_rejection_code": simple["simple_rejection_code"],

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

tab_new, tab_pending, tab_perf, tab_study, tab_lab = st.tabs(
    ["➕ New Match", "⏳ Pending", "📊 Performance", "📚 Study", "🧪 Lab"]
)


# ─── TAB 1 — NEW MATCH ────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Include the `Pre-match HOME vs AWAY` header — that decides home/away.")

    raw_in = st.text_area("Input", height=320, label_visibility="collapsed",
                          placeholder="Pre-match Mainz 05 vs Bayern München\\n"
                                      "Match information\\nMewa Arena\\n...")
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

        if simple["simple_qualifying"]:
            st.success(simple["simple_reason"])
        elif simple.get("home_consecutive_wins") is not None:
            st.info(f"No bet [{simple['simple_rejection_code']}] — {simple['simple_reason']}")
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
            st.warning("Missing required fields (team names or streaks) — not saved.")
        else:
            row = build_row(facts)
            try:
                existing = db_find_existing(row)
            except Exception as e:
                existing = None
                st.warning(f"Duplicate check failed: {e}")

            if existing:
                st.info(f"Already stored (match_id **{existing['match_id']}**, "
                        f"qualifying={existing.get('simple_qualifying')}).")
            else:
                try:
                    db_insert(row)
                    if simple["simple_qualifying"]:
                        st.success("Saved to Pending.")
                    else:
                        st.success("Saved to Study (no bet).")
                except Exception as e:
                    st.error(f"Insert failed: {e}")


# ─── TAB 2 — PENDING ──────────────────────────────────────────────
with tab_pending:
    st.markdown("### Pending matches")
    st.caption("Qualifying bets awaiting settlement.")

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
    st.caption("Qualifying bets only.")

    try:
        settled = db_settled()
    except Exception as e:
        st.error(f"Could not load: {e}")
        settled = []

    if not settled:
        st.info("No settled matches yet.")
    else:
        bets = [m for m in settled if m.get("dc_hit") is not None]
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


# ─── TAB 4 — STUDY ────────────────────────────────────────────────
with tab_study:
    st.markdown("### 📚 Study — no-bet matches with actual results")
    st.caption("Every parsed match the classifier rejected, plus the actual final score. "
               "Raw data only — no hypotheticals, no predicted bets.")

    try:
        study = db_study()
    except Exception as e:
        st.error(f"Could not load study rows: {e}")
        study = []

    if not study:
        st.info("No study rows yet. Paste a no-bet match into New Match to populate.")
    else:
        settled = [m for m in study if m.get("home_score") is not None]
        unsettled = [m for m in study if m.get("home_score") is None]

        c1, c2, c3 = st.columns(3)
        c1.metric("Study rows", len(study))
        c2.metric("Settled", len(settled))
        c3.metric("Awaiting result", len(unsettled))

        # ── Actual outcome counts by rejection code ──
        if settled:
            st.markdown("### Actual outcome counts by rejection code")
            rows = []
            for m in settled:
                rows.append({
                    "rejection_code": m.get("simple_rejection_code") or "(none)",
                    "result": m.get("result") or "(unknown)",
                })
            df = pd.DataFrame(rows)
            pivot = (df.groupby(["rejection_code", "result"])
                     .size().unstack(fill_value=0))
            for col in ("home_win", "draw", "away_win"):
                if col not in pivot.columns:
                    pivot[col] = 0
            pivot = pivot[["home_win", "draw", "away_win"]]
            pivot["total"] = pivot.sum(axis=1)
            pivot = pivot.reset_index().sort_values("total", ascending=False)
            st.dataframe(pivot, use_container_width=True, hide_index=True)

            # ── Unbeaten gap vs actual outcome (NOT-BEATEN cases) ──
            nb_rows = []
            for m in settled:
                if m.get("simple_rejection_code") != "NOT_BEATEN_GAP_LOW":
                    continue
                hu = m.get("home_consecutive_unbeaten") or 0
                au = m.get("away_consecutive_unbeaten") or 0
                higher = "home" if hu > au else ("away" if au > hu else "tie")
                nb_rows.append({
                    "home_team": m.get("home_team"),
                    "away_team": m.get("away_team"),
                    "home_u": hu,
                    "away_u": au,
                    "gap": abs(hu - au),
                    "higher_unbeaten_side": higher,
                    "result": m.get("result"),
                })
            if nb_rows:
                st.markdown("### Unbeaten gap vs actual outcome (NOT-BEATEN cases)")
                st.caption("For rows where both teams had 0 wins. The gap is the "
                           "absolute difference in consecutive unbeaten.")
                st.dataframe(pd.DataFrame(nb_rows), use_container_width=True,
                             hide_index=True)

        # ── Individual rows with settle form ──
        st.markdown("### Rows")
        all_codes = sorted({m.get("simple_rejection_code") or "(none)" for m in study})
        selected = st.multiselect("Filter by rejection code", all_codes,
                                   default=all_codes, key="study_filter")

        for m in study:
            code = m.get("simple_rejection_code") or "(none)"
            if code not in selected:
                continue

            st.markdown(_card_html(m, settled=(m.get("home_score") is not None)),
                        unsafe_allow_html=True)

            if m.get("home_score") is not None:
                result_lbl = {"home_win": "Home win",
                              "away_win": "Away win",
                              "draw": "Draw"}.get(m.get("result"),
                                                   m.get("result") or "—")
                st.caption(f"Final: **{m['home_score']}–{m['away_score']}** · {result_lbl}")
                with st.expander("Correct result"):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        hs = st.number_input("Home", min_value=0, step=1,
                                              value=int(m["home_score"]),
                                              key=f"ch_{m['match_id']}")
                    with c2:
                        aws = st.number_input("Away", min_value=0, step=1,
                                               value=int(m["away_score"]),
                                               key=f"ca_{m['match_id']}")
                    with c3:
                        st.write(""); st.write("")
                        if st.button("Update", key=f"cu_{m['match_id']}"):
                            if hs > aws:     result = "home_win"
                            elif aws > hs:   result = "away_win"
                            else:            result = "draw"
                            db_settle(m["match_id"], int(hs), int(aws),
                                      result, None, None)
                            st.success("Updated.")
                            st.rerun()
            else:
                with st.expander("Enter final score"):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1:
                        hs = st.number_input("Home", min_value=0, step=1,
                                              key=f"sh_{m['match_id']}")
                    with c2:
                        aws = st.number_input("Away", min_value=0, step=1,
                                               key=f"sa_{m['match_id']}")
                    with c3:
                        st.write(""); st.write("")
                        if st.button("Save result", key=f"ss_{m['match_id']}",
                                     type="primary"):
                            if hs > aws:     result = "home_win"
                            elif aws > hs:   result = "away_win"
                            else:            result = "draw"
                            db_settle(m["match_id"], int(hs), int(aws),
                                      result, None, None)
                            st.success("Result saved.")
                            st.rerun()

        with st.expander("Manage rows"):
            ids = st.multiselect("Select match_id to delete",
                                 [m["match_id"] for m in study])
            if ids and st.button("Delete selected"):
                for mid in ids:
                    db_delete(mid)
                st.success(f"Deleted {len(ids)}.")
                st.rerun()


# ─── TAB 5 — LAB ──────────────────────────────────────────────────
with tab_lab:
    st.markdown("### 🧪 Lab")
    st.caption("Parse a dump without saving.")

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

        if simple["simple_qualifying"]:
            st.success(simple.get("simple_reason") or "")
        else:
            st.info(f"No bet [{simple.get('simple_rejection_code')}] — "
                    f"{simple.get('simple_reason')}")

        with st.expander("Parsed fields"):
            st.json({k: v for k, v in facts.items()
                     if v not in (None, [], "") and k != "recent_form_triplets"})

        if facts.get("recent_form_triplets"):
            with st.expander("Recent form (raw, source order)"):
                st.dataframe(pd.DataFrame(facts["recent_form_triplets"]),
                             use_container_width=True, hide_index=True)
