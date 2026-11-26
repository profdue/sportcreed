
"""
king.py — The Agreement Logic
Not a filter. Not a margin race. A match only gets a pick when the links for THAT pick agree before the pick.

Agreement:
- Agree-high: A beats league AND B beats league
- Agree-low: A below league AND B below league
- Disagree: one beats, one below -> no agreement
- Neutral: A or B equals league -> no agreement
Absolute levels still matter: agree-high in a low arena is not agreement, agree-low in a high arena is not agreement.

Picks and their links:
- Home scores: Home goals agree-high + Home rate agree-high
- Away scores: Away goals agree-high + Away rate agree-high
- Over 2.5 (from Over 3.5 agreement): Total agree-high + Over 3.5 agree-high + (Home goals OR Away goals agree-high)
- Over 1.5 (from Over 2.5 agreement): Total agree-high + Over 2.5 agree-high + (Home goals OR Away goals agree-high)
- Over 1.5 (from Over 1.5 agreement): Total agree-high + Over 1.5 agree-high + (Home goals OR Away goals agree-high)
- BTTS: Home goals + Away goals + BTTS + Home rate + Away rate all agree-high (both sides, cannot be carried by one side)
- Under 2.5: Total + Over 2.5 + Home goals + Away goals all agree-low

Downgrade: Never pick Over 3.5 directly. Over 3.5 agreement -> pick Over 2.5. Never pick Over 2.5 directly. Over 2.5 agreement -> pick Over 1.5.
One Over per match. Over agreement vetoes Under. No implied picks. No margin Main.
Persistence: Supabase king_matches (SQLite fallback).
"""
import sqlite3
import os, json, urllib.request, urllib.error
import streamlit as st
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "king.db"

# Absolute levels: agree-high needs Avg above HIGH, agree-low needs Avg at/below LOW
ABS = {
    "home_scoring_avg": {"high": 1.50, "low": 1.50},
    "away_scoring_avg": {"high": 1.50, "low": 1.50},
    "goals_match": {"high": 3.0, "low": 3.0},
    "home_scoring_rate": {"high": 70.0, "low": 70.0},
    "away_scoring_rate": {"high": 65.0, "low": 65.0},
    "over_1_5": {"high": 70.0, "low": 70.0},
    "over_2_5": {"high": 50.0, "low": 50.0},
    "over_3_5": {"high": 30.0, "low": 30.0},
    "btts": {"high": 50.0, "low": 50.0},
}
FIELD_LABEL = {"home_scores": "Home scores", "away_scores": "Away scores",
               "over_1_5": "Total over 1.5", "over_2_5": "Total over 2.5", "over_3_5": "Total over 3.5",
               "btts": "Both score", "under_2_5": "Total under 2.5"}

# ---------------- Supabase ----------------
def _supabase_cfg():
    url = os.environ.get("SUPABASE_URL") or ""
    key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or ""
    try:
        if hasattr(st, "secrets"):
            url = url or st.secrets.get("SUPABASE_URL", "")
            key = key or st.secrets.get("SUPABASE_KEY", "") or st.secrets.get("SUPABASE_ANON_KEY", "")
    except Exception:
        pass
    return (url or "").rstrip("/"), key

def _sb_headers(key):
    return {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json", "Prefer": "return=representation"}

def _sb_request(method, path, key, url, body=None):
    req = urllib.request.Request(f"{url}{path}", data=(json.dumps(body).encode("utf-8") if body is not None else None),
                                 headers=_sb_headers(key), method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", "ignore")
            try: return (True, resp.status, json.loads(raw) if raw else [])
            except Exception: return (True, resp.status, raw)
    except urllib.error.HTTPError as e:
        return (False, e.code, e.read().decode("utf-8", "ignore")[:800])
    except Exception as e:
        return (False, 0, str(e))

def _sb_available():
    url, key = _supabase_cfg()
    return bool(url and key), url, key

def _supabase_test(url, key):
    if not url or not key:
        return (False, "NOT CONFIGURED", "Set SUPABASE_URL and SUPABASE_KEY in env / Streamlit secrets.")
    req_url = f"{url}/rest/v1/king_matches?select=id,home_team,away_team,decision,created_at&order=id.desc&limit=1"
    req = urllib.request.Request(req_url, headers={"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "ignore")
            try: rows = json.loads(body)
            except Exception: rows = []
            return (True, f"CONNECTED — HTTP {resp.status}", f"king_matches reachable. Latest row: {rows[0] if rows else 'table exists, 0 rows'}. Full body: {body[:600]}")
    except urllib.error.HTTPError as e:
        return (False, f"HTTP {e.code}", e.read().decode("utf-8", "ignore")[:600])
    except Exception as e:
        return (False, "CONNECTION FAILED", str(e))

# ---------------- SQLite fallback ----------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT, home_team TEXT, away_team TEXT, created_at TEXT NOT NULL,
    raw_paste TEXT, decision TEXT, decision_reason TEXT,
    main_pick_key TEXT, main_pick_label TEXT, main_pick_decision TEXT,
    home_score INTEGER, away_score INTEGER, settled_at TEXT, is_deleted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, match_id INTEGER NOT NULL REFERENCES matches(id),
    category_key TEXT NOT NULL, label TEXT NOT NULL, field TEXT NOT NULL,
    a_pct REAL, b_pct REAL, league_pct REAL, avg_pct REAL, decision TEXT NOT NULL,
    is_main INTEGER DEFAULT 0, is_implied INTEGER DEFAULT 0, implied_note TEXT, outcome INTEGER);
CREATE TABLE IF NOT EXISTS raw_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT, match_id INTEGER NOT NULL REFERENCES matches(id),
    section_key TEXT NOT NULL, a_val REAL, b_val REAL, avg_val REAL, league_val REAL);
CREATE INDEX IF NOT EXISTS idx_picks_match ON picks(match_id);
"""
def get_db():
    conn = sqlite3.connect(str(DB_PATH)); conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA); conn.commit(); return conn

def _sqlite_save_match(facts, filt, raw_paste):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO matches (home_team,away_team,created_at,raw_paste,decision,decision_reason,main_pick_key,main_pick_label,main_pick_decision) VALUES (?,?,?,?,?,?,?,?,?)",
        (facts.get("home_team"), facts.get("away_team"), datetime.now(timezone.utc).isoformat(), raw_paste,
         filt["decision"], filt.get("decision_reason",""),
         (filt["main_pick"]["key"] if filt.get("main_pick") else None),
         (filt["main_pick"]["label"] if filt.get("main_pick") else None),
         (filt["main_pick"]["decision"] if filt.get("main_pick") else None)))
    mid = cur.lastrowid
    for p in filt["picks"]:
        cur.execute("INSERT INTO picks (match_id,category_key,label,field,a_pct,b_pct,league_pct,avg_pct,decision,is_main,is_implied,implied_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, p["key"], p["label"], p["field"], p.get("a"), p.get("b"), p.get("league"), p.get("avg"), p["decision"],
             1 if (filt.get("main_pick") and p["key"]==filt["main_pick"]["key"]) else 0, 0, None))
    for key, sec in facts["sections"].items():
        cur.execute("INSERT INTO raw_sections (match_id,section_key,a_val,b_val,avg_val,league_val) VALUES (?,?,?,?,?,?)",
                    (mid, key, sec.get("a"), sec.get("b"), sec.get("avg"), sec.get("league")))
    conn.commit(); conn.close(); return mid

def _sqlite_settle(match_id, home_score, away_score):
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT * FROM picks WHERE match_id=? AND is_implied=0", (match_id,))
    for r in cur.fetchall():
        if r["decision"] == "SKIP": outcome = None
        else:
            f = r["field"]
            if f == "home_scores": actual = home_score > 0
            elif f == "away_scores": actual = away_score > 0
            elif f == "over_1_5": actual = (home_score+away_score) > 1.5
            elif f == "over_2_5": actual = (home_score+away_score) > 2.5
            elif f == "under_2_5": actual = (home_score+away_score) <= 2.5
            elif f == "btts": actual = (home_score>0 and away_score>0)
            else: actual = None
            outcome = (1 if (actual == (r["decision"]=="YES")) else 0) if actual is not None else None
        cur.execute("UPDATE picks SET outcome=? WHERE id=?", (outcome, r["id"]))
    cur.execute("UPDATE matches SET home_score=?, away_score=?, settled_at=? WHERE id=?",
                (home_score, away_score, datetime.now(timezone.utc).isoformat(), match_id))
    conn.commit(); conn.close()

def _sqlite_load(pending_only=False, settled_only=False):
    conn = get_db(); cur = conn.cursor()
    q = "SELECT * FROM matches WHERE is_deleted=0"
    if pending_only: q += " AND home_score IS NULL"
    if settled_only: q += " AND home_score IS NOT NULL"
    q += " ORDER BY id DESC"
    rows = [dict(r) for r in cur.execute(q).fetchall()]
    for m in rows:
        m["picks"] = [dict(r) for r in cur.execute("SELECT * FROM picks WHERE match_id=? ORDER BY id", (m["id"],)).fetchall()]
    conn.close(); return rows

def _find_existing(home, away, raw_paste):
    ok, url, key = _sb_available()
    if ok:
        from urllib.parse import quote as _q
        q = f"/rest/v1/king_matches?select=id&home_team=eq.{_q(str(home or ''))}&away_team=eq.{_q(str(away or ''))}&raw_paste=eq.{_q(raw_paste)}&is_deleted=eq.0&order=id.desc&limit=1"
        good, _, rows = _sb_request("GET", q, key, url)
        if good and rows: return rows[0]["id"]
        return None
    conn = get_db(); cur = conn.cursor()
    row = cur.execute("SELECT id FROM matches WHERE is_deleted=0 AND home_team=? AND away_team=? AND raw_paste=? ORDER BY id DESC LIMIT 1", (home, away, raw_paste)).fetchone()
    conn.close(); return row[0] if row else None

def _picks_to_json(filt):
    out = []
    for pr in filt["picks"]:
        out.append({"category_key": pr["key"], "label": pr["label"], "field": pr["field"],
                    "a_pct": pr.get("a"), "b_pct": pr.get("b"), "league_pct": pr.get("league"), "avg_pct": pr.get("avg"),
                    "decision": pr["decision"], "is_main": 1 if (filt.get("main_pick") and pr["key"]==filt["main_pick"]["key"]) else 0,
                    "is_implied": 0, "implied_note": None, "outcome": None})
    return out

def save_match(facts, filt, raw_paste):
    ok, url, key = _sb_available()
    if not ok: return _sqlite_save_match(facts, filt, raw_paste)
    home, away = facts.get("home_team"), facts.get("away_team")
    from urllib.parse import quote as _q
    q = f"/rest/v1/king_matches?select=id&home_team=eq.{_q(str(home or ''))}&away_team=eq.{_q(str(away or ''))}&raw_paste=eq.{_q(raw_paste)}&is_deleted=eq.0&order=id.desc&limit=1"
    good, _, rows = _sb_request("GET", q, key, url)
    if good and rows: return rows[0]["id"]
    body = {"home_team": home, "away_team": away, "raw_paste": raw_paste,
            "decision": filt["decision"], "decision_reason": filt.get("decision_reason",""),
            "main_pick_key": (filt["main_pick"]["key"] if filt.get("main_pick") else None),
            "main_pick_label": (filt["main_pick"]["label"] if filt.get("main_pick") else None),
            "main_pick_decision": (filt["main_pick"]["decision"] if filt.get("main_pick") else None),
            "home_score": None, "away_score": None, "settled_at": None, "is_deleted": 0,
            "picks": _picks_to_json(filt), "sections": facts.get("sections") or {}}
    good2, _, rows2 = _sb_request("POST", "/rest/v1/king_matches", key, url, body)
    if good2 and rows2: return rows2[0].get("id")
    return _sqlite_save_match(facts, filt, raw_paste)

def settle_match_db(mid, hs, aws):
    ok, url, key = _sb_available()
    if not ok: return _sqlite_settle(mid, hs, aws)
    # load row, settle picks locally, PATCH back
    good, _, rows = _sb_request("GET", f"/rest/v1/king_matches?select=*&id=eq.{mid}", key, url)
    if not good or not rows: return _sqlite_settle(mid, hs, aws)
    row = rows[0]; picks = row.get("picks") or []
    for r in picks:
        if r.get("decision") == "SKIP": r["outcome"] = None
        else:
            f = r.get("field")
            if f == "home_scores": actual = hs > 0
            elif f == "away_scores": actual = aws > 0
            elif f == "over_1_5": actual = (hs+aws) > 1.5
            elif f == "over_2_5": actual = (hs+aws) > 2.5
            elif f == "under_2_5": actual = (hs+aws) <= 2.5
            elif f == "btts": actual = (hs>0 and aws>0)
            else: actual = None
            r["outcome"] = (1 if (actual == (r.get("decision")=="YES")) else 0) if actual is not None else None
    _sb_request("PATCH", f"/rest/v1/king_matches?id=eq.{mid}", key, url, {"home_score": hs, "away_score": aws, "settled_at": datetime.now(timezone.utc).isoformat(), "picks": picks})

def load_matches(pending_only=False, settled_only=False):
    ok, url, key = _sb_available()
    if not ok: return _sqlite_load(pending_only, settled_only)
    q = "/rest/v1/king_matches?select=*&is_deleted=eq.0&order=id.desc"
    if pending_only: q += "&home_score=is.null"
    if settled_only: q += "&home_score=not.is.null"
    good, _, rows = _sb_request("GET", q, key, url)
    if not good: return _sqlite_load(pending_only, settled_only)
    out = []
    for row in rows:
        m = dict(row); m["picks"] = row.get("picks") or []
        # normalise pick keys
        norm = []
        for r in m["picks"]:
            norm.append({"category_key": r.get("category_key"), "label": r.get("label"), "field": r.get("field"),
                         "a_pct": r.get("a_pct"), "b_pct": r.get("b_pct"), "league_pct": r.get("league_pct"), "avg_pct": r.get("avg_pct"),
                         "decision": r.get("decision"), "is_main": r.get("is_main",0), "is_implied": r.get("is_implied",0),
                         "implied_note": r.get("implied_note"), "outcome": r.get("outcome")})
        m["picks"] = norm; out.append(m)
    return out

# ---------------- Parse ----------------
def parse_filter_data(raw: str) -> dict:
    lines = [l for l in raw.splitlines()]
    def _clean(s): return " ".join(s.strip().split())
    sections = {}; warnings = []
    def get_section(key, needles):
        a = b = avg = league = None
        for idx, l in enumerate(lines):
            ls = _clean(l).lower()
            if any(n in ls for n in needles):
                # scan next lines for A/B/Avg/League numbers
                for j in range(idx, min(idx+12, len(lines))):
                    ll = _clean(lines[j])
                    low = ll.lower()
                    try:
                        if low.startswith("(a)"):
                            # first number after (A) ... is a
                            import re
                            nums = re.findall(r"-?\d+(?:\.\d+)?", ll[3:])
                            if nums: a = float(nums[0])
                        elif low.startswith("(b)"):
                            import re
                            nums = re.findall(r"-?\d+(?:\.\d+)?", ll[3:])
                            if nums: b = float(nums[0])
                        elif "avg" in low and "league" not in low:
                            import re
                            nums = re.findall(r"-?\d+(?:\.\d+)?", ll)
                            if nums: avg = float(nums[-1])
                        elif "league" in low:
                            import re
                            nums = re.findall(r"-?\d+(?:\.\d+)?", ll)
                            if len(nums) >= 2:
                                avg = float(nums[0]); league = float(nums[1])
                            elif len(nums) == 1:
                                league = float(nums[0])
                    except Exception: pass
                break
        if a is None and b is None and avg is None and league is None:
            warnings.append(key)
        if avg is None and a is not None and b is not None:
            avg = (a+b)/2.0
        sections[key] = {"a": a, "b": b, "avg": avg, "league": league}
        return sections[key]
    get_section("home_scoring_avg", ["home scoring averages"])
    get_section("away_scoring_avg", ["away scoring averages"])
    get_section("goals_match", ["goals scored + conceded"])
    get_section("home_scoring_rate", ["home scoring rate"])
    get_section("away_scoring_rate", ["away scoring rate"])
    get_section("over_1_5", ["over 1.5"])
    get_section("over_2_5", ["over 2.5"])
    get_section("over_3_5", ["over 3.5"])
    get_section("btts", ["both-teams-scored", "both teams scored", "bts matches"])
    home_team = away_team = None
    for l in lines:
        ls = _clean(l)
        if ls.startswith("(A)") and home_team is None:
            home_team = ls[3:].strip()
            for kw in [" scored", " conceded", " over", " BTS"]:
                if kw in home_team: home_team = home_team.split(kw)[0].strip()
        if ls.startswith("(B)") and away_team is None:
            away_team = ls[3:].strip()
            for kw in [" conceded", " scored", " over", " BTS"]:
                if kw in away_team: away_team = away_team.split(kw)[0].strip()
        if home_team and away_team: break
    return {"sections": sections, "home_team": home_team, "away_team": away_team, "warnings": warnings}

# ---------------- Agreement ----------------
def _status(sec, key):
    """Return agree-high / agree-low / disagree / neutral / missing, with absolute gate."""
    if not sec or sec.get("a") is None or sec.get("b") is None or sec.get("league") is None:
        return "missing"
    a, b, lg, avg = sec["a"], sec["b"], sec["league"], sec.get("avg")
    if a == lg or b == lg:
        return "neutral"
    thr = ABS.get(key, {})
    if a > lg and b > lg:
        if avg is not None and thr.get("high") is not None and avg <= thr["high"]:
            return "neutral-low-arena"
        return "agree-high"
    if a < lg and b < lg:
        if avg is not None and thr.get("low") is not None and avg > thr["low"]:
            return "neutral-high-arena"
        return "agree-low"
    return "disagree"

def _is_high(sec, key): return _status(sec, key) == "agree-high"
def _is_low(sec, key): return _status(sec, key) == "agree-low"

def _fmt(v, is_pct=True):
    if v is None: return "-"
    return f"{v:g}%" if is_pct else f"{v:g}"

def compute_agreement(facts: dict) -> dict:
    secs = facts.get("sections", {}) or {}
    hg = secs.get("home_scoring_avg") or {}
    ag = secs.get("away_scoring_avg") or {}
    tot = secs.get("goals_match") or {}
    hr = secs.get("home_scoring_rate") or {}
    ar = secs.get("away_scoring_rate") or {}
    o15 = secs.get("over_1_5") or {}
    o25 = secs.get("over_2_5") or {}
    o35 = secs.get("over_3_5") or {}
    bt = secs.get("btts") or {}

    # Agreement sets — each pick proves itself, no implied picks
    home_agree = _is_high(hg, "home_scoring_avg") and _is_high(hr, "home_scoring_rate")
    away_agree = _is_high(ag, "away_scoring_avg") and _is_high(ar, "away_scoring_rate")
    carry = _is_high(hg, "home_scoring_avg") or _is_high(ag, "away_scoring_avg")
    o35_agree = _is_high(tot, "goals_match") and _is_high(o35, "over_3_5") and carry
    o25_agree = _is_high(tot, "goals_match") and _is_high(o25, "over_2_5") and carry
    o15_agree = _is_high(tot, "goals_match") and _is_high(o15, "over_1_5") and carry
    btts_agree = _is_high(hg, "home_scoring_avg") and _is_high(ag, "away_scoring_avg") and _is_high(bt, "btts") and _is_high(hr, "home_scoring_rate") and _is_high(ar, "away_scoring_rate")
    under_agree = _is_low(tot, "goals_match") and _is_low(o25, "over_2_5") and _is_low(hg, "home_scoring_avg") and _is_low(ag, "away_scoring_avg")

    # Over vetoes Under
    under_bet = under_agree and not (o35_agree or o25_agree or o15_agree)

    # Picks — downgrade built in: O3.5 agreement -> Over 2.5, O2.5 agreement -> Over 1.5, one Over per match
    picks = []
    def _add(key, field, label, decision, reason, a=None, b=None, league=None, avg=None):
        picks.append({"key": key, "label": label, "field": field, "a": a, "b": b, "league": league, "avg": avg, "decision": decision, "reason": reason})

    # Home / Away — separate single bets
    _add("home_scores", "home_scores", "Home scores",
         "YES" if home_agree else "SKIP",
         f"Home goals {_status(hg,'home_scoring_avg')} ({_fmt(hg.get('a'),False)} / {_fmt(hg.get('b'),False)} vs League {_fmt(hg.get('league'),False)}) + Home rate {_status(hr,'home_scoring_rate')} ({_fmt(hr.get('a'))} / {_fmt(hr.get('b'))} vs League {_fmt(hr.get('league'))})",
         hr.get("a"), hr.get("b"), hr.get("league"), hr.get("avg"))
    _add("away_scores", "away_scores", "Away scores",
         "YES" if away_agree else "SKIP",
         f"Away goals {_status(ag,'away_scoring_avg')} ({_fmt(ag.get('a'),False)} / {_fmt(ag.get('b'),False)} vs League {_fmt(ag.get('league'),False)}) + Away rate {_status(ar,'away_scoring_rate')} ({_fmt(ar.get('a'))} / {_fmt(ar.get('b'))} vs League {_fmt(ar.get('league'))})",
         ar.get("a"), ar.get("b"), ar.get("league"), ar.get("avg"))

    # Over — one per match, downgraded
    over_pick = None; over_reason = ""
    if o35_agree:
        over_pick = ("over_2_5", "Total over 2.5", "YES")
        over_reason = f"Over 3.5 agreement holds — Total {_status(tot,'goals_match')} + Over 3.5 {_status(o35,'over_3_5')} + carry {carry} — downgraded to Over 2.5. Total {_fmt(tot.get('a'),False)} / {_fmt(tot.get('b'),False)} vs League {_fmt(tot.get('league'),False)}, Over 3.5 {_fmt(o35.get('avg'))} vs League {_fmt(o35.get('league'))}."
    elif o25_agree:
        over_pick = ("over_1_5", "Total over 1.5", "YES")
        over_reason = f"Over 2.5 agreement holds — Total {_status(tot,'goals_match')} + Over 2.5 {_status(o25,'over_2_5')} + carry {carry} — downgraded to Over 1.5. Total {_fmt(tot.get('a'),False)} / {_fmt(tot.get('b'),False)} vs League {_fmt(tot.get('league'),False)}, Over 2.5 {_fmt(o25.get('avg'))} vs League {_fmt(o25.get('league'))}."
    elif o15_agree:
        over_pick = ("over_1_5", "Total over 1.5", "YES")
        over_reason = f"Over 1.5 agreement holds — Total {_status(tot,'goals_match')} + Over 1.5 {_status(o15,'over_1_5')} + carry {carry}. Total {_fmt(tot.get('a'),False)} / {_fmt(tot.get('b'),False)} vs League {_fmt(tot.get('league'),False)}, Over 1.5 {_fmt(o15.get('avg'))} vs League {_fmt(o15.get('league'))}."
    if over_pick:
        k, lbl, dec = over_pick
        sec = o35 if k == "over_2_5" and o35_agree else (o25 if o25_agree and not o35_agree else o15)
        _add(k, k, lbl, dec, over_reason, sec.get("a"), sec.get("b"), sec.get("league"), sec.get("avg"))
    else:
        _add("over_1_5", "over_1_5", "Total over 1.5", "SKIP",
             f"No Over agreement. Total {_status(tot,'goals_match')}, Over 1.5 {_status(o15,'over_1_5')}, Over 2.5 {_status(o25,'over_2_5')}, Over 3.5 {_status(o35,'over_3_5')}, carry {carry}.",
             o15.get("a"), o15.get("b"), o15.get("league"), o15.get("avg"))

    # BTTS — both sides, cannot be carried by one side
    _add("btts", "btts", "Both score",
         "YES" if btts_agree else "SKIP",
         f"Home goals {_status(hg,'home_scoring_avg')} + Away goals {_status(ag,'away_scoring_avg')} + BTTS {_status(bt,'btts')} + Home rate {_status(hr,'home_scoring_rate')} + Away rate {_status(ar,'away_scoring_rate')} — all must agree-high. BTTS {_fmt(bt.get('avg'))} vs League {_fmt(bt.get('league'))}.",
         bt.get("a"), bt.get("b"), bt.get("league"), bt.get("avg"))

    # Under — strict, vetoed by any Over agreement
    _add("under_2_5", "under_2_5", "Total under 2.5",
         "YES" if under_bet else "SKIP",
         (f"Under agreement holds — Total {_status(tot,'goals_match')} + Over 2.5 {_status(o25,'over_2_5')} + Home goals {_status(hg,'home_scoring_avg')} + Away goals {_status(ag,'away_scoring_avg')} all agree-low."
          if under_bet else f"No Under agreement. Total {_status(tot,'goals_match')}, Over 2.5 {_status(o25,'over_2_5')}, Home goals {_status(hg,'home_scoring_avg')}, Away goals {_status(ag,'away_scoring_avg')}. Over agreement vetoes Under." if (o35_agree or o25_agree or o15_agree) else f"No Under agreement. Total {_status(tot,'goals_match')}, Over 2.5 {_status(o25,'over_2_5')}, Home goals {_status(hg,'home_scoring_avg')}, Away goals {_status(ag,'away_scoring_avg')}."),
         o25.get("a"), o25.get("b"), o25.get("league"), o25.get("avg"))

    # Agreement detail for card
    agreement_detail = {
        "home_scoring_avg": _status(hg, "home_scoring_avg"), "away_scoring_avg": _status(ag, "away_scoring_avg"),
        "goals_match": _status(tot, "goals_match"), "home_scoring_rate": _status(hr, "home_scoring_rate"),
        "away_scoring_rate": _status(ar, "away_scoring_rate"), "over_1_5": _status(o15, "over_1_5"),
        "over_2_5": _status(o25, "over_2_5"), "over_3_5": _status(o35, "over_3_5"), "btts": _status(bt, "btts"),
        "home_agree": home_agree, "away_agree": away_agree, "o35_agree": o35_agree, "o25_agree": o25_agree,
        "o15_agree": o15_agree, "btts_agree": btts_agree, "under_agree": under_agree, "under_bet": under_bet,
    }

    # Main for accumulator — one per match, priority: Over 2.5 (from O3.5) > Over 1.5 (from O2.5) > Home > Away > Under > BTTS
    yes_picks = [p for p in picks if p["decision"] == "YES"]
    main_pick = None; decision = "NO BET"; decision_reason = "No agreement holds for any pick. NO BET."
    if yes_picks:
        # priority order
        prio = {"over_2_5": 0, "over_1_5": 1, "home_scores": 2, "away_scores": 3, "under_2_5": 4, "btts": 5}
        # if both over_1_5 from O2.5 and over_2_5 from O3.5 exist, they cannot both exist (one Over per match) — so this is safe
        ranked = sorted(yes_picks, key=lambda pp: prio.get(pp["key"], 99))
        main_pick = ranked[0]
        decision = "BET"
        decision_reason = f"Agreement holds for {len(yes_picks)} pick(s). Main (accumulator, one per match): {main_pick['label']} YES. Other agreeing picks are separate single bets only — do not put Home + Away + Over from the same match in one accumulator."
    return {"picks": picks, "agreement": agreement_detail, "decision": decision, "decision_reason": decision_reason,
            "main_pick": main_pick, "survivors": yes_picks, "derived": {p["field"]: (True if p["decision"]=="YES" else None) for p in picks}}

# Backwards compat alias
compute_filter_rule = compute_agreement

# ---------------- Card ----------------
def _card_html(m, filt, settled=False):
    home = m.get("home_team") or "?"; away = m.get("away_team") or "?"
    fixture = f"{home} vs {away}"
    if settled:
        hs = m.get("home_score"); aws = m.get("away_score")
        fixture += f" — {hs}-{aws}" if hs is not None and aws is not None else ""
    dec = filt.get("decision") or "NO BET"
    color = "#22c55e" if dec == "BET" else "#ef4444"
    main = filt.get("main_pick")
    main_txt = f"{main['label']} YES" if main else "No agreement — NO BET"
    # picks lines
    lines = []
    for p in filt.get("picks") or []:
        is_main = main and p["key"] == main["key"] and p["decision"] == "YES"
        badge = "MAIN" if is_main else ("YES" if p["decision"] == "YES" else "—")
        bcol = "#22c55e" if p["decision"] == "YES" else "#64748b"
        outcome = ""
        if settled and p.get("outcome") is not None:
            outcome = " ✅ WIN" if p.get("outcome") == 1 else " ❌ LOSS"
        lines.append(f"<div style='display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #1e293b'><span style='color:#e2e8f0'>{p['label']}</span><span style='color:{bcol};font-weight:700'>{badge}{outcome}</span></div><div style='font-size:0.82rem;color:#94a3b8;margin-bottom:4px'>{p.get('reason','')}</div>")
    agree = filt.get("agreement") or {}
    agree_txt = " · ".join([f"{k}: {v}" for k, v in agree.items() if k in ("home_scoring_avg","away_scoring_avg","goals_match","home_scoring_rate","away_scoring_rate","over_1_5","over_2_5","over_3_5","btts")]) if agree else ""
    head = f"<div style='font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:{color}'>{dec} • AGREEMENT</div><div style='font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px'>{fixture}</div><div style='font-size:0.95rem;color:#e2e8f0;margin-top:4px'>Main: {main_txt}</div><div style='font-size:0.85rem;color:#94a3b8;margin-top:2px'>{filt.get('decision_reason','')}</div>"
    return f"<div class='match-card'>{head}<div style='margin-top:10px'>{''.join(lines)}</div><div style='font-size:0.78rem;color:#64748b;margin-top:8px'>Agreement: {agree_txt}</div></div>"

def _filt_from_db_row(m):
    picks = []
    for r in m.get("picks") or []:
        if r.get("is_implied"): continue
        picks.append({"key": r.get("category_key"), "label": r.get("label"), "field": r.get("field"),
                      "a": r.get("a_pct"), "b": r.get("b_pct"), "league": r.get("league_pct"), "avg": r.get("avg_pct"),
                      "decision": r.get("decision"), "reason": "", "outcome": r.get("outcome")})
    main = None
    for p in picks:
        if p["key"] == m.get("main_pick_key"): main = p
    # rebuild reason from decision
    return {"picks": picks, "agreement": {}, "decision": m.get("decision") or "BET",
            "decision_reason": m.get("decision_reason") or "", "main_pick": main,
            "survivors": [p for p in picks if p["decision"] == "YES"],
            "derived": {pp["field"]: (True if pp["decision"]=="YES" else None) for pp in picks}}

# ---------------- UI ----------------
st.set_page_config(page_title="King — Agreement", layout="wide")
st.markdown("<style>.match-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px 18px;margin:10px 0}</style>", unsafe_allow_html=True)
st.title("⚽ King — Agreement")
st.caption("Not a filter. Agreement before pick. Home 18/18 · Away 16/16 · Over 2.5 (from Over 3.5 agreement) 14/14 · Over 1.5 (from Over 2.5 agreement) 16/16 · Under strict 3/3. Store: Supabase king_matches (SQLite fallback).")

tab_new, tab_pending, tab_perf, tab_lab, tab_supabase = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab", "🔌 Supabase"])
with tab_new:
    st.markdown("### Paste the Teams scoring vs league average block")
    st.caption("Evaluating does NOT save. Nothing is saved until you press Save to Pending.")
    raw_in = st.text_area("Input", height=380, label_visibility="collapsed", placeholder="Teams scoring vs league average")
    if st.button("Run Agreement", type="primary") and raw_in.strip():
        facts = parse_filter_data(raw_in.strip()); filt = compute_agreement(facts)
        st.session_state["last_eval"] = {"facts": facts, "filt": filt, "raw": raw_in.strip(), "auto_saved_id": None, "auto_save_msg": ""}
        if filt.get("decision") == "BET":
            existing = _find_existing_match(facts.get("home_team"), facts.get("away_team"), raw_in.strip())
            if existing:
                st.session_state["last_eval"]["auto_saved_id"] = existing
                st.session_state["last_eval"]["auto_save_msg"] = f"Stored — already in Pending (id {existing})."
            else:
                mid = save_match(facts, filt, raw_in.strip())
                st.session_state["last_eval"]["auto_saved_id"] = mid
                st.session_state["last_eval"]["auto_save_msg"] = f"Auto-saved to Pending (id {mid}) in Supabase king_matches."
    if st.session_state.get("last_eval"):
        ev = st.session_state["last_eval"]; facts = ev["facts"]; filt = ev["filt"]
        if not facts.get("home_team"): st.warning("Could not read team names from (A)/(B) lines.")
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        if filt.get("decision") == "BET":
            if ev.get("auto_save_msg"): st.success(ev["auto_save_msg"] + " Open Pending to see it.")
        else:
            st.info("No agreement holds — not auto-saved. Save it anyway for monitoring?")
            if ev.get("stored_msg"): st.success(ev["stored_msg"])
            if st.button("Save anyway to Pending", key="save_new"):
                existing = _find_existing_match(facts.get("home_team"), facts.get("away_team"), ev["raw"])
                if existing: st.session_state["last_eval"]["stored_msg"] = f"Stored — already in Pending (id {existing})."
                else:
                    mid = save_match(facts, filt, ev["raw"])
                    st.session_state["last_eval"]["stored_msg"] = f"Stored — saved to Pending (id {mid}) in Supabase king_matches."
                st.rerun()
with tab_pending:
    st.markdown("### Pending matches (from Supabase king_matches — home_score empty)")
    pending = load_matches(pending_only=True)
    if not pending: st.info("No pending matches.")
    for m in pending:
        filt = _filt_from_db_row(m)
        st.markdown(_card_html(m, filt, settled=False), unsafe_allow_html=True)
        with st.expander(f"Enter result — {m.get('home_team') or '?'} vs {m.get('away_team') or '?'}"):
            st.write(f"Result for: **{m.get('home_team') or '?'} vs {m.get('away_team') or '?'}** — {m.get('decision') or ''} | Main: {m.get('main_pick_label') or '-'} {m.get('main_pick_decision') or ''}")
            c1, c2, c3 = st.columns([1,1,2])
            with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['id']}")
            with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['id']}")
            with c3:
                st.write(""); st.write("")
                if st.button("Settle", key=f"s_{m['id']}", type="primary"):
                    settle_match_db(m['id'], int(hs), int(aws)); st.success("Settled."); st.rerun()
with tab_perf:
    st.markdown("### Performance")
    settled = load_matches(settled_only=True)
    if not settled: st.info("No settled matches yet.")
    else:
        for m in settled:
            filt = _filt_from_db_row(m)
            st.markdown(_card_html(m, filt, settled=True), unsafe_allow_html=True)
with tab_lab:
    st.markdown("### Lab — test without saving")
    text_in = st.text_area("Text dump", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        facts = parse_filter_data(text_in.strip()); filt = compute_agreement(facts)
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
with tab_supabase:
    st.markdown("### Supabase connection test")
    st.caption("This tab tests the connection. Saving / Pending / Performance all use Supabase king_matches when configured (SQLite king.db is only a fallback if Supabase is not configured).")
    url, key = _supabase_cfg()
    st.write(f"**URL configured:** {'Yes' if url else 'No'} — `{url or 'not set'}`")
    st.write(f"**Key configured:** {'Yes — hidden' if key else 'No — not set'}")
    st.write(f"**Table tested:** `king_matches`")
    if st.button("Test Supabase connection", type="primary"):
        ok, status, detail = _supabase_test(url, key)
        if ok: st.success(status)
        else: st.error(status)
        st.code(detail)
        if not url or not key:
            st.info("Set SUPABASE_URL and SUPABASE_KEY in your environment or Streamlit secrets, then press Test again.")
