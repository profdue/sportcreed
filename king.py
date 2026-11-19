#!/usr/bin/env python3
"""
king.py — The Logic (DC + BTTS + Trap Warnings)
Classifier: 2 stats for DC — consecutive wins + consecutive unbeaten.
Analytical Flags:
  - Market veto (hard): DC pick must match moneyline favourite
  - Unproven Streak (soft): warns if current win streak == all-time record
  - BTTS Prediction (hard prediction): BTTS YES if both teams scoring streak >= 3 AND max(odds_1, odds_2) <= 4.00. Settled like DC: hit if both teams score.
  - LIGHT CLEAR HOME: requires hu == hw AND hw >= 2.
No learning / no Study tab. Save if DC qualifies OR BTTS qualifies. Matches with neither are NOT saved.
"""
import re
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
from supabase import create_client, Client
st.set_page_config(page_title="The Logic", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; letter-spacing: -0.02em; }
    .match-card { background: linear-gradient(135deg, #1a1f2e 0%, #232a3d 100%); border-radius: 16px; padding: 22px 26px; margin-bottom: 16px; border-left: 4px solid #6366f1; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip { border-left-color: #64748b; opacity: 0.9; }
    .match-card.win { border-left-color: #10b981; }
    .match-card.loss { border-left-color: #ef4444; }
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em; }
    .badge-solid { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-light { background: #38bdf822; color: #38bdf8; border: 1px solid #38bdf855; }
    .badge-notbe { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
    .badge-nobet { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .badge-btts { background: #8b5cf622; color: #a78bfa; border: 1px solid #8b5cf655; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill { font-size: 0.8rem; padding: 5px 11px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b; font-family: monospace; }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.odds { background: #6366f115; color: #a5b4fc; border-color: #6366f140; }
    .signal-pill.scoring { background: #8b5cf615; color: #a78bfa; border-color: #8b5cf640; }
    .warning-box { background: #ef444422; color: #fca5a5; padding: 8px 12px; border-radius: 8px; margin-top: 10px; font-size: 0.85rem; font-weight: 600; border: 1px solid #ef444455; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 44px; background-color: #1a1f2e; border-radius: 10px; padding: 0 18px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_supabase() -> Client:
    try:
        url = st.secrets["supabase"]["url"]; key = st.secrets["supabase"]["key"]
    except (KeyError, TypeError):
        url = st.secrets["SUPABASE_URL"]; key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def db_insert(row: dict):
    sb = get_supabase()
    try:
        res = sb.table("matches").insert(row).execute()
    except Exception as e:
        if "btts" in str(e).lower():
            row = {k: v for k, v in row.items() if not k.startswith("btts")}
            res = sb.table("matches").insert(row).execute()
        else:
            raise
    st.cache_data.clear()
    return res.data[0] if res.data else None

@st.cache_data(ttl=30)
def db_pending():
    sb = get_supabase()
    return (sb.table("matches").select("*").or_("simple_qualifying.eq.1,btts_qualifying.eq.1").is_("home_score", "null").order("extracted_at", desc=True).execute().data or [])

@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    return (sb.table("matches").select("*").or_("simple_qualifying.eq.1,btts_qualifying.eq.1").not_.is_("home_score", "null").order("settled_at", desc=True).execute().data or [])

def db_find_existing(row: dict):
    sb = get_supabase()
    res = (sb.table("matches").select("*").eq("home_team", row.get("home_team")).eq("away_team", row.get("away_team")).limit(1).execute())
    return res.data[0] if res.data else None

def db_settle(match_id: int, home_score: int, away_score: int, result: str, dc_hit, profit_loss, btts_hit=None, btts_profit=None):
    sb = get_supabase()
    payload = {"home_score": home_score, "away_score": away_score, "result": result, "dc_hit": dc_hit, "profit_loss": profit_loss, "settled_at": datetime.now(timezone.utc).isoformat()}
    # BTTS columns: add these columns in Supabase (btts_hit int, btts_profit float, btts_qualifying int, btts_pick text). If columns don't exist yet, settle still works for DC — wrap BTTS update separately.
    if btts_hit is not None:
        payload["btts_hit"] = btts_hit; payload["btts_profit"] = btts_profit
    try:
        sb.table("matches").update(payload).eq("match_id", match_id).execute()
    except Exception as e:
        if "btts" in str(e).lower():
            payload = {k: v for k, v in payload.items() if not k.startswith("btts")}
            sb.table("matches").update(payload).eq("match_id", match_id).execute()
        else:
            raise
    st.cache_data.clear()

def db_delete(match_id: int):
    get_supabase().table("matches").delete().eq("match_id", match_id).execute(); st.cache_data.clear()

def _to_int_or_none(v):
    if v is None: return None
    try:
        if v != v: return None
    except Exception: pass
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"): return None
    try: return int(float(s))
    except (TypeError, ValueError): return None

def _clean_line(s: str) -> str: return s.strip().strip("\ufeff\u200b\u200c\u200d\u00a0 ")
def _strip_lines(text: str) -> list: return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
def _next_nonblank(lines, start):
    i = start
    while i < len(lines) and not _clean_line(lines[i]): i += 1
    if i >= len(lines): return None, i
    return _clean_line(lines[i]), i + 1
_ARROWS = {"➜", "➔", "→", "->", "=>", ">", "➡", "⬈", "⬊"}
_MARKET_MAP = {"1": "1", "X": "X", "x": "X", "2": "2", "×": "X", "✕": "X"}
_DASHES = {"-", "–", "—", "N/A", "n/a"}
_ODDS_STOP_LABELS = {"general stats", "match information", "recent form", "h2h last matches", "consecutive scoring", "consecutive wins", "consecutive unbeaten"}
_OTHER_BOOKMAKERS = {"betking", "bet9ja", "betway", "sportybet", "bet365", "pinnacle", "1xbet"}
def _find_odds_header(lines):
    for i, raw in enumerate(lines):
        s = _clean_line(raw)
        if not s: continue
        low = s.lower()
        if low == "pre-match odds" or ("odds" in low and "pre" in low and "match" in low): return i + 1
    return None
def _parse_odds_block(lines):
    start = _find_odds_header(lines)
    if start is None: return {}
    out = {}; current_book = {}
    def has_prices(d): return d.get("1") is not None or d.get("X") is not None or d.get("2") is not None
    i = start; guard = 0
    while i < len(lines) and guard < 200:
        guard += 1; s = _clean_line(lines[i])
        if not s: i += 1; continue
        low = s.lower()
        if low in _ODDS_STOP_LABELS: break
        if low in _OTHER_BOOKMAKERS:
            if has_prices(current_book): break
            else: current_book = {}; i += 1; continue
        if s in _MARKET_MAP:
            market = _MARKET_MAP[s]
            if market in current_book:
                if has_prices(current_book): break
                else: current_book = {}
            nxt, j = _next_nonblank(lines, i + 1)
            if nxt is None: i += 1; continue
            if nxt in _DASHES: i = j; continue
            if nxt in _ARROWS: price, k = _next_nonblank(lines, j); i = k
            else: price, k = nxt, j; i = k
            if price and price not in _DASHES:
                try: current_book[market] = float(price)
                except (TypeError, ValueError): pass
            continue
        i += 1
    if has_prices(current_book): return current_book
    return {}
def _market_favourite(odds):
    h, a = odds.get("1"), odds.get("2")
    if h is None or a is None: return None
    if abs(h - a) < 0.05: return "tie"
    return "home" if h < a else "away"

NOT_BEATEN_MIN_GAP = 3
def _classify_streaks(hw, aw, hu, au):
    def result(category=None, pick=None, bet=None, qualifying=0, reason="", code=None):
        return {"category": category, "pick_team": pick, "bet": bet, "qualifying": qualifying, "reason": reason, "rejection_code": code}
    home_has, away_has = hw > 0, aw > 0
    if home_has and not away_has:
        if hu >= 2 and au == 0: return result("SOLID CLEAR", "home", "1X", 1, f"Home {hw}W/{hu}U vs Away {aw}W/{au}U — SOLID (opponent 0/0).")
        # LIGHT CLEAR HOME: hu == hw AND hw >= 2
        if hu == hw and hw >= 2: return result("LIGHT CLEAR HOME", "home", "1X", 1, f"Home {hw}W/{hu}U (unbeaten = wins, wins >= 2) vs Away {aw}W/{au}U — LIGHT-HOME.")
        return result(reason=f"Home leader {hw}W/{hu}U but unbeaten != wins or < 2 wins and SOLID not met.", code="HOME_LEADER_NO_PADDING")
    if away_has and not home_has:
        if au >= 2 and hu == 0: return result("SOLID CLEAR", "away", "X2", 1, f"Away {aw}W/{au}U vs Home {hw}W/{hu}U — SOLID (opponent 0/0).")
        return result(reason=f"Away leader {aw}W/{au}U — LIGHT-away removed; SOLID not met.", code="AWAY_LEADER_NO_SOLID")
    if not home_has and not away_has:
        gap = abs(hu - au)
        if gap < NOT_BEATEN_MIN_GAP: return result(reason=f"Both 0 wins, unbeaten gap {gap} < {NOT_BEATEN_MIN_GAP}.", code="NOT_BEATEN_GAP_LOW")
        if hu > au: return result("NOT-BEATEN SIGNAL", "home", "1X", 1, f"Both 0W; Home unbeaten {hu} vs Away {au} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")
        return result("NOT-BEATEN SIGNAL", "away", "X2", 1, f"Both 0W; Away unbeaten {au} vs Home {hu} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")
    return result(reason=f"Both teams have wins ({hw} vs {aw}) — outside the file.", code="BOTH_HAVE_WINS")

def classify_simple(home_wins, away_wins, home_unbeaten, away_unbeaten, market_favourite=None):
    hw, aw, hu, au = _to_int_or_none(home_wins), _to_int_or_none(away_wins), _to_int_or_none(home_unbeaten), _to_int_or_none(away_unbeaten)
    def result(category=None, pick=None, bet=None, qualifying=0, reason="", code=None):
        return {"category": category, "pick_team": pick, "bet": bet, "qualifying": qualifying, "reason": reason, "rejection_code": code}
    if None in (hw, aw, hu, au): return result(reason="Missing one or more streak inputs.", code="MISSING")
    streak = _classify_streaks(hw, aw, hu, au)
    if not streak["qualifying"]: return streak
    pick = streak["pick_team"]
    if market_favourite is None: return result(reason=f"{streak['reason']} [no odds available]", code="MARKET_UNAVAILABLE")
    if market_favourite != "tie" and market_favourite != pick: return result(reason=f"{streak['reason']} [market favourite={market_favourite}, pick={pick}]", code="MARKET_DISAGREES")
    return dict(streak)

def compute_simple_row(facts: dict) -> dict:
    c = classify_simple(facts.get("home_consecutive_wins"), facts.get("away_consecutive_wins"), facts.get("home_consecutive_unbeaten"), facts.get("away_consecutive_unbeaten"), market_favourite=facts.get("market_favourite"))
    return {"simple_category": c["category"], "simple_pick_team": c["pick_team"], "simple_bet": c["bet"], "simple_qualifying": c["qualifying"], "simple_reason": c["reason"], "simple_rejection_code": c["rejection_code"], "home_consecutive_wins": _to_int_or_none(facts.get("home_consecutive_wins")), "away_consecutive_wins": _to_int_or_none(facts.get("away_consecutive_wins")), "home_consecutive_unbeaten": _to_int_or_none(facts.get("home_consecutive_unbeaten")), "away_consecutive_unbeaten": _to_int_or_none(facts.get("away_consecutive_unbeaten"))}

def settle_simple(simple_bet, home_score, away_score):
    if home_score > away_score: result = "home_win"
    elif away_score > home_score: result = "away_win"
    else: result = "draw"
    if simple_bet == "1X": hit = result in ("home_win", "draw")
    elif simple_bet == "X2": hit = result in ("away_win", "draw")
    else: return result, None, 0.0
    return result, (1 if hit else 0), (1.0 if hit else -1.0)

def settle_btts(home_score, away_score, btts_qualifying):
    """BTTS YES prediction: hit iff BOTH teams score. Only settled if it was a prediction."""
    if not btts_qualifying: return None, None
    hit = 1 if (home_score > 0 and away_score > 0) else 0
    return hit, (1.0 if hit else -1.0)

_CONSEC_HEADER = re.compile(r"^Consecutive\s+(scoring|wins|unbeaten)\s*$", re.I)
_RECORD_PAREN = re.compile(r"\(Record\s+(\d+)\)", re.I)
_SCORE_LINE = re.compile(r"^\d+\s*-\s*\d+$")
_FORM_RESULT = re.compile(r"^[WDL]$")
_VS_HEADER = re.compile(r"^(?:(?:pre|re|e)[-–— ]?match\s+|match\s+)?(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)
_ODDS_GLYPHS = "➜➔→➡•€$£"
_STRUCTURAL_LABELS = {"match information", "recent form", "general stats", "pre-match odds", "h2h last matches", "odds", "advertisement"}
_HEADER_JUNK_PREFIXES = ("pre-match ", "pre match ", "pre–match ", "re-match ", "re match ", "e-match ", "e match ", "match ")
_STADIUM_KEYWORDS = ("stadium", "arena", "park", "ground", "field", "centre", "center", "stadion", "estadio", "estádio", "stade", "stadio", "olympic", "olympique", "coliseum", "sports complex")
def _looks_like_text_dump(text: str) -> bool:
    low = text.lower(); return ("pre-match" in low and " vs " in low) or ("re-match" in low and " vs " in low) or ("recent form" in low and "consecutive" in low)
def _looks_like_odds_line(s: str) -> bool:
    if any(sym in s for sym in _ODDS_GLYPHS): return True
    return len(re.findall(r"\b\d+\.\d+\b", s)) >= 2
def _looks_like_person_name(s: str) -> bool:
    tokens = s.split()
    if len(tokens) < 2 or len(tokens) > 4: return False
    for tok in tokens:
        clean = tok.replace("'", "").replace("’", "").replace("-", "").replace(".", "").replace(" ", "")
        if not clean or not clean[0].isupper() or not all(c.isalpha() for c in clean): return False
    return True
def _strip_header_prefix(name: str) -> str:
    low = name.lower()
    for marker in _HEADER_JUNK_PREFIXES:
        if low.startswith(marker): return name[len(marker):].strip()
    return name
def _teams_from_header(lines):
    for ln in lines:
        s = _clean_line(ln)
        if not s or s.lower() in _STRUCTURAL_LABELS or _looks_like_odds_line(s): continue
        m = _VS_HEADER.match(s)
        if not m: continue
        home = _strip_header_prefix(_clean_line(m.group(1))); away = _strip_header_prefix(_clean_line(m.group(2)))
        if not home or not away or home.isdigit() or away.isdigit(): continue
        return home, away
    return None, None
def _parse_match_information(lines, f):
    for i, ln in enumerate(lines):
        if _clean_line(ln).lower() != "match information": continue
        val, j = _next_nonblank(lines, i + 1)
        if not val: return
        entries = []
        while val and val.lower() != "recent form" and len(entries) < 5:
            entries.append(val); val, j = _next_nonblank(lines, j)
        if not entries: return
        for entry in entries:
            if any(kw in entry.lower() for kw in _STADIUM_KEYWORDS): f["venue"] = entry; return
        if not _looks_like_person_name(entries[0]): f["venue"] = entries[0]
        return
def _teams_frequency_fallback(lines):
    try: rf_idx = next(i for i, ln in enumerate(lines) if _clean_line(ln).lower() == "recent form")
    except StopIteration: return None, None
    _, i = _next_nonblank(lines, rf_idx + 1); nxt, i = _next_nonblank(lines, i)
    if nxt and not _FORM_RESULT.match(nxt): nxt, i = _next_nonblank(lines, i)
    cur = nxt; names = []
    while cur and _FORM_RESULT.match(cur):
        team, i = _next_nonblank(lines, i); score, i = _next_nonblank(lines, i)
        if not team or not score or not _SCORE_LINE.match(score): break
        names.append(team); cur, i = _next_nonblank(lines, i)
        if not cur or not _FORM_RESULT.match(cur): break
    from collections import Counter
    if not names: return None, None
    c = Counter(names)
    if len(c) < 2: return None, None
    top = [n for n, _ in c.most_common(2)]; first_seen = {}
    for idx, nm in enumerate(names): first_seen.setdefault(nm, idx)
    top.sort(key=lambda nm: first_seen[nm]); return top[0], top[1]
def parse_recent_form_dump(text: str) -> dict:
    f = {"home_team": None, "away_team": None, "venue": None, "league": None, "match_date": None, "kickoff_time": None, "country": None, "home_consecutive_scoring": None, "away_consecutive_scoring": None, "home_consecutive_wins": None, "away_consecutive_wins": None, "home_consecutive_unbeaten": None, "away_consecutive_unbeaten": None, "home_consecutive_wins_record": None, "away_consecutive_wins_record": None, "home_consecutive_unbeaten_record": None, "away_consecutive_unbeaten_record": None, "odds": {}, "market_favourite": None, "recent_form_triplets": [], "_parse_warnings": []}
    lines = _strip_lines(text); n = len(lines)
    _parse_match_information(lines, f)
    header_home, header_away = _teams_from_header(lines)
    if header_home and header_away: f["home_team"], f["away_team"] = header_home, header_away
    else:
        h, a = _teams_frequency_fallback(lines); f["home_team"], f["away_team"] = h, a
        if h and a: f["_parse_warnings"].append("No 'Pre-match X vs Y' header — team order guessed.")
    odds = _parse_odds_block(lines); f["odds"] = odds; f["market_favourite"] = _market_favourite(odds)
    for i, ln in enumerate(lines):
        m = _CONSEC_HEADER.match(_clean_line(ln))
        if not m: continue
        kind = m.group(1).lower(); home_val, j = _next_nonblank(lines, i + 1)
        if home_val is None: continue
        home_rec_val = None
        if j < n:
            rec_m = _RECORD_PAREN.search(lines[j])
            if rec_m: home_rec_val = _to_int_or_none(rec_m.group(1)); j += 1
        while j < n and _RECORD_PAREN.search(lines[j]): j += 1
        away_val, j = _next_nonblank(lines, j)
        if away_val is None: continue
        away_rec_val = None
        if j < n:
            rec_m = _RECORD_PAREN.search(lines[j])
            if rec_m: away_rec_val = _to_int_or_none(rec_m.group(1)); j += 1
        hv = _to_int_or_none(home_val); av = _to_int_or_none(away_val)
        if kind == "scoring": f["home_consecutive_scoring"], f["away_consecutive_scoring"] = hv, av
        elif kind == "wins": f["home_consecutive_wins"], f["away_consecutive_wins"] = hv, av; f["home_consecutive_wins_record"], f["away_consecutive_wins_record"] = home_rec_val, away_rec_val
        elif kind == "unbeaten": f["home_consecutive_unbeaten"], f["away_consecutive_unbeaten"] = hv, av; f["home_consecutive_unbeaten_record"], f["away_consecutive_unbeaten_record"] = home_rec_val, away_rec_val
    try: rf_idx = next(i for i, ln in enumerate(lines) if _clean_line(ln).lower() == "recent form")
    except StopIteration: rf_idx = -1
    if rf_idx >= 0:
        _, i = _next_nonblank(lines, rf_idx + 1); nxt, i = _next_nonblank(lines, i)
        if nxt and not _FORM_RESULT.match(nxt): nxt, i = _next_nonblank(lines, i)
        cur = nxt
        while cur and _FORM_RESULT.match(cur) and len(f["recent_form_triplets"]) < 20:
            team, i = _next_nonblank(lines, i); score, i = _next_nonblank(lines, i)
            if not team or not score or not _SCORE_LINE.match(score): break
            f["recent_form_triplets"].append({"result": cur.upper(), "team": team, "score": score}); cur, i = _next_nonblank(lines, i)
            if not cur or not _FORM_RESULT.match(cur): break
    if not f["home_team"]: f["_parse_warnings"].append("home_team not found")
    if not f["away_team"]: f["_parse_warnings"].append("away_team not found")
    if f["market_favourite"] is None: f["_parse_warnings"].append("Odds block missing or unparseable.")
    return f

# --- BTTS as prediction (not a signal / not learning) ---
def compute_btts_prediction(facts: dict) -> dict:
    hsc = _to_int_or_none(facts.get("home_consecutive_scoring")); asc = _to_int_or_none(facts.get("away_consecutive_scoring"))
    odds = facts.get("odds", {}) or {}
    o1, o2 = odds.get("1"), odds.get("2")
    if hsc is None or asc is None or o1 is None or o2 is None: return {"btts_pick": None, "btts_qualifying": 0, "btts_reason": "Missing scoring streak or odds."}
    try: max_odds = max(float(o1), float(o2))
    except (TypeError, ValueError): return {"btts_pick": None, "btts_qualifying": 0, "btts_reason": "Odds unparseable."}
    if hsc >= 3 and asc >= 3 and max_odds <= 4.00:
        return {"btts_pick": "BTTS YES", "btts_qualifying": 1, "btts_reason": f"Scoring {hsc} vs {asc}, max odds {max_odds:.2f} <= 4.00 — BTTS YES."}
    return {"btts_pick": None, "btts_qualifying": 0, "btts_reason": f"Scoring {hsc} vs {asc}, max odds {max_odds:.2f} — no BTTS."}

def build_row(facts: dict) -> dict:
    simple = compute_simple_row(facts); btts = compute_btts_prediction(facts)
    now_iso = datetime.now(timezone.utc).isoformat(); odds = facts.get("odds", {})
    row = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team"), "venue": facts.get("venue"), "league": facts.get("league"), "match_date": facts.get("match_date"), "kickoff_time": facts.get("kickoff_time"), "country": facts.get("country"), "home_consecutive_scoring": facts.get("home_consecutive_scoring"), "away_consecutive_scoring": facts.get("away_consecutive_scoring"), "home_consecutive_wins": simple["home_consecutive_wins"], "away_consecutive_wins": simple["away_consecutive_wins"], "home_consecutive_unbeaten": simple["home_consecutive_unbeaten"], "away_consecutive_unbeaten": simple["away_consecutive_unbeaten"], "home_consecutive_wins_record": facts.get("home_consecutive_wins_record"), "away_consecutive_wins_record": facts.get("away_consecutive_wins_record"), "home_consecutive_unbeaten_record": facts.get("home_consecutive_unbeaten_record"), "away_consecutive_unbeaten_record": facts.get("away_consecutive_unbeaten_record"), "simple_category": simple["simple_category"], "simple_pick_team": simple["simple_pick_team"], "simple_bet": simple["simple_bet"], "simple_qualifying": simple["simple_qualifying"], "simple_reason": simple["simple_reason"], "simple_rejection_code": simple["simple_rejection_code"], "btts_pick": btts["btts_pick"], "btts_qualifying": btts["btts_qualifying"], "btts_reason": btts["btts_reason"], "odds_1": odds.get("1"), "odds_x": odds.get("X"), "odds_2": odds.get("2"), "market_favourite": facts.get("market_favourite"), "extracted_at": now_iso}
    return {k: v for k, v in row.items() if v is not None}

def _compute_trap_warning_from_row(m: dict) -> str:
    pick = m.get("simple_pick_team")
    if pick == "home": wins, wins_rec, unbeaten, unbeaten_rec, team = m.get("home_consecutive_wins"), m.get("home_consecutive_wins_record"), m.get("home_consecutive_unbeaten"), m.get("home_consecutive_unbeaten_record"), m.get("home_team")
    elif pick == "away": wins, wins_rec, unbeaten, unbeaten_rec, team = m.get("away_consecutive_wins"), m.get("away_consecutive_wins_record"), m.get("away_consecutive_unbeaten"), m.get("away_consecutive_unbeaten_record"), m.get("away_team")
    else: return ""
    if wins is not None and wins_rec is not None and wins == wins_rec and wins >= 3: return f"⚠ UNPROVEN STREAK ({team} Win Streak {wins} == Record {wins_rec})"
    if unbeaten is not None and unbeaten_rec is not None and unbeaten == unbeaten_rec and unbeaten >= 4: return f"⚠ UNPROVEN STREAK ({team} Unbeaten {unbeaten} == Record {unbeaten_rec})"
    return ""
def _compute_double_trap_warning(m: dict) -> str:
    hw, hwr, aw, awr = m.get("home_consecutive_wins"), m.get("home_consecutive_wins_record"), m.get("away_consecutive_wins"), m.get("away_consecutive_wins_record")
    if hw is not None and hwr is not None and aw is not None and awr is not None and hw == hwr and aw == awr and hw >= 3 and aw >= 3: return "⚠ DOUBLE TRAP: Both teams are at peak win records. Danger zone for DC."
    return ""
def _compute_btts_signal(m: dict) -> bool:
    # Back-compat: for rows saved before btts_qualifying existed, recompute from stored fields.
    if m.get("btts_qualifying") is not None: return bool(m.get("btts_qualifying"))
    hsc, asc, o1, o2 = m.get("home_consecutive_scoring"), m.get("away_consecutive_scoring"), m.get("odds_1"), m.get("odds_2")
    if hsc is None or asc is None or o1 is None or o2 is None: return False
    try: return int(hsc) >= 3 and int(asc) >= 3 and max(float(o1), float(o2)) <= 4.00
    except (TypeError, ValueError): return False

def _card_html(m: dict, settled: bool = False) -> str:
    home, away = m.get("home_team") or "?", m.get("away_team") or "?"
    hs, aws = m.get("home_score"), m.get("away_score"); dc_hit, profit = m.get("dc_hit"), m.get("profit_loss") or 0.0
    cat, bet, pick = m.get("simple_category"), m.get("simple_bet"), m.get("simple_pick_team")
    hw, aw = m.get("home_consecutive_wins"), m.get("away_consecutive_wins"); hwr, awr = m.get("home_consecutive_wins_record"), m.get("away_consecutive_wins_record")
    hu, au = m.get("home_consecutive_unbeaten"), m.get("away_consecutive_unbeaten"); hur, aur = m.get("home_consecutive_unbeaten_record"), m.get("away_consecutive_unbeaten_record")
    hsc, asc = m.get("home_consecutive_scoring"), m.get("away_consecutive_scoring")
    cat_cls = {"SOLID CLEAR": "badge-solid", "LIGHT CLEAR HOME": "badge-light", "NOT-BEATEN SIGNAL": "badge-notbe"}.get(cat, "badge-nobet")
    pick_lbl = "HOME" if pick == "home" else ("AWAY" if pick == "away" else "")
    card_cls = "pick-home" if pick == "home" else ("pick-away" if pick == "away" else "skip")
    meta_bits = [b for b in [m.get("league"), m.get("match_date"), m.get("venue")] if b]; meta = " · ".join(meta_bits) if meta_bits else "—"
    title = f"{home} {hs}–{aws} {away}" if (settled and hs is not None) else f"{home} vs {away}"
    if settled and dc_hit is not None:
        vcls = "badge-win" if dc_hit == 1 else "badge-loss"; verdict_html = f'<span class="pick-badge {vcls}" style="margin-left:8px">{"WIN" if dc_hit == 1 else "LOSS"} · {profit:+.2f}u</span>'
    else: verdict_html = ""
    badge_text = f"{cat} · {bet} {pick_lbl}" if cat else "NO BET"
    btts_badge = ""; btts_verdict = ""
    if _compute_btts_signal(m):
        btts_badge = '<span class="pick-badge badge-btts" style="margin-left:8px">🎯 BTTS YES — PREDICTION</span>'
        if settled and m.get("btts_hit") is not None:
            bv = "badge-win" if m.get("btts_hit") == 1 else "badge-loss"; btts_verdict = f'<span class="pick-badge {bv}" style="margin-left:8px">BTTS {"WIN" if m.get("btts_hit")==1 else "LOSS"} · {(m.get("btts_profit") or 0.0):+.2f}u</span>'
    warn_html = ""
    dbl_trap = _compute_double_trap_warning(m)
    if dbl_trap: warn_html = f'<div class="warning-box">{dbl_trap}</div>'
    if cat:
        trap_warn = _compute_trap_warning_from_row(m)
        if trap_warn: warn_html = f'<div class="warning-box">{trap_warn}</div>'
    o1, ox, o2, mf = m.get("odds_1"), m.get("odds_x"), m.get("odds_2"), m.get("market_favourite"); odds_row = ""
    if any(v is not None for v in (o1, ox, o2)):
        mf_lbl = {"home": "HOME", "away": "AWAY", "tie": "LEVEL"}.get(mf, "?")
        odds_row = ('<div class="signal-row">' f'<span class="signal-pill odds">1: {o1 if o1 is not None else "—"}</span>' f'<span class="signal-pill odds">X: {ox if ox is not None else "—"}</span>' f'<span class="signal-pill odds">2: {o2 if o2 is not None else "—"}</span>' f'<span class="signal-pill odds">Mkt: <b>{mf_lbl}</b></span>' '</div>')
    win_pill = f'<span class="signal-pill home">Home W/U: {hw} / {hu} {f"({hwr}/{hur})" if hwr or hur else ""}</span>'; away_pill = f'<span class="signal-pill away">Away W/U: {aw} / {au} {f"({awr}/{aur})" if awr or aur else ""}</span>'
    scoring_row = ""
    if hsc is not None and asc is not None: scoring_row = ('<div class="signal-row" style="margin-top: 4px;">' f'<span class="signal-pill scoring">Scoring Streak: {hsc} vs {asc}</span>' '</div>')
    html = f"""<div class="match-card {card_cls}"><div class="match-title">{title}</div><div class="match-meta">{meta}</div><span class="pick-badge {cat_cls}">{badge_text}</span>{btts_badge}{verdict_html}{btts_verdict}{warn_html}<div class="signal-row">{win_pill}{away_pill}</div>{scoring_row}{odds_row}</div>"""
    return "".join(line.strip() for line in html.split("\n"))

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"])
with tab_new:
    st.markdown("### Paste the match data"); st.caption("Include `Pre-match HOME vs AWAY` header and the `Pre-match odds` block.")
    raw_in = st.text_area("Input", height=340, label_visibility="collapsed", placeholder="Pre-match Real Zaragoza vs Huesca\n...")
    run = st.button("Run the Logic", type="primary")
    if run and raw_in.strip():
        if not _looks_like_text_dump(raw_in): st.error("Input must contain 'Pre-match X vs Y' or 'Recent form' + 'Consecutive' sections."); st.stop()
        try: facts = parse_recent_form_dump(raw_in.strip()); simple = compute_simple_row(facts); btts = compute_btts_prediction(facts)
        except Exception as e: st.error(f"Parse failed: {e}"); st.stop()
        st.markdown("### The Pick"); preview = {**facts, **simple, "btts_qualifying": btts["btts_qualifying"], "btts_pick": btts["btts_pick"]}
        st.markdown(_card_html(preview, settled=False), unsafe_allow_html=True)
        if simple["simple_qualifying"]: st.success(simple["simple_reason"])
        elif simple.get("home_consecutive_wins") is not None: st.info(f"No DC bet [{simple['simple_rejection_code']}] — {simple['simple_reason']}")
        else: st.warning("Missing streak values — cannot classify.")
        if btts["btts_qualifying"]: st.success(f"🎯 BTTS PREDICTION: {btts['btts_reason']}")
        else: st.caption(f"BTTS: {btts['btts_reason']}")
        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})"):
                for w in warnings: st.write(f"- {w}")
        has_required = facts.get("home_team") and facts.get("away_team") and simple.get("home_consecutive_wins") is not None
        if not has_required: st.warning("Missing required fields — not saved.")
        elif not simple["simple_qualifying"] and not btts["btts_qualifying"]: st.info("Not saved — no DC bet and no BTTS prediction.")
        elif not simple["simple_qualifying"]: st.info("No DC bet — saving for BTTS prediction only.")
        else:
            row = build_row(facts); existing = db_find_existing(row)
            if existing: st.info(f"Already stored (match_id **{existing['match_id']}**, DC={existing.get('simple_qualifying')}, BTTS={existing.get('btts_qualifying')}).")
            else:
                try: db_insert(row); st.success("Saved to Pending.")
                except Exception as e: st.error(f"Insert failed: {e}")
with tab_pending:
    st.markdown("### Pending matches"); pending = db_pending()
    if not pending: st.info("No pending matches.")
    else:
        for m in pending:
            st.markdown(_card_html(m, settled=False), unsafe_allow_html=True)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        result, dc_hit, profit = settle_simple(m.get("simple_bet") or "", int(hs), int(aws))
                        if not m.get("simple_qualifying"): dc_hit, profit = None, None
                        btts_hit, btts_profit = settle_btts(int(hs), int(aws), m.get("btts_qualifying"))
                        db_settle(m["match_id"], int(hs), int(aws), result, dc_hit, profit, btts_hit, btts_profit); st.success("Settled."); st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"]); st.success("Deleted."); st.rerun()
with tab_perf:
    st.markdown("### Performance"); settled = db_settled()
    if not settled: st.info("No settled matches yet.")
    else:
        bets = [m for m in settled if m.get("dc_hit") is not None]; n = len(bets); wins = sum(1 for m in bets if m.get("dc_hit") == 1); net = sum((m.get("profit_loss") or 0.0) for m in bets)
        st.markdown("#### DC"); c1, c2, c3, c4, c5 = st.columns(5); c1.metric("Bets", n); c2.metric("Wins", wins); c3.metric("Losses", n - wins); c4.metric("Hit %", f"{(wins / n * 100) if n else 0:.1f}%"); c5.metric("Net P/L", f"{net:+.2f}u")
        if bets:
            df = pd.DataFrame([{"category": m.get("simple_category"), "bet": m.get("simple_bet"), "hit": int(m["dc_hit"]), "pl": float(m.get("profit_loss") or 0.0)} for m in bets]); out = df.groupby(["category", "bet"]).agg(n=("hit", "count"), wins=("hit", "sum"), net_pl=("pl", "sum")).reset_index(); out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1); st.dataframe(out[["category", "bet", "n", "wins", "hit_pct", "net_pl"]], use_container_width=True, hide_index=True)
        btts_bets = [m for m in settled if m.get("btts_hit") is not None]; bn = len(btts_bets); bwins = sum(1 for m in btts_bets if m.get("btts_hit") == 1); bnet = sum((m.get("btts_profit") or 0.0) for m in btts_bets)
        st.markdown("#### BTTS YES — Prediction"); c1, c2, c3, c4, c5 = st.columns(5); c1.metric("BTTS Bets", bn); c2.metric("Wins", bwins); c3.metric("Losses", bn - bwins); c4.metric("Hit %", f"{(bwins / bn * 100) if bn else 0:.1f}%"); c5.metric("Net P/L", f"{bnet:+.2f}u")
        st.markdown("### All settled")
        for m in settled: st.markdown(_card_html(m, settled=True), unsafe_allow_html=True)
with tab_lab:
    st.markdown("### 🧪 Lab"); text_in = st.text_area("Text dump", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        try:
            facts = parse_recent_form_dump(text_in); simple = compute_simple_row(facts); btts = compute_btts_prediction(facts)
            cA, cB, cC, cD = st.columns(4); cA.metric("Home W / U", f"{simple.get('home_consecutive_wins')} / {simple.get('home_consecutive_unbeaten')}"); cB.metric("Away W / U", f"{simple.get('away_consecutive_wins')} / {simple.get('away_consecutive_unbeaten')}"); cC.metric("Category", simple.get("simple_category") or "—"); cD.metric("Bet", simple.get("simple_bet") or "—")
            if simple["simple_qualifying"]: st.success(simple.get("simple_reason") or "")
            else: st.info(f"No bet [{simple.get('simple_rejection_code')}] — {simple.get('simple_reason')}")
            if btts["btts_qualifying"]: st.success(f"🎯 BTTS YES — PREDICTION: {btts['btts_reason']}")
            else: st.caption(f"BTTS: {btts['btts_reason']}")
            with st.expander("Parsed fields"): st.json({k: v for k, v in facts.items() if v not in (None, [], {}, "") and k != "recent_form_triplets"})
        except Exception as e: st.error(f"Parse failed: {e}")
