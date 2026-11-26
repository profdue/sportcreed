
"""
king.py — Final Logic (verified on 71 + 73)
Gate 0 Everything-High | Gate 1 Margin (Over 2.5 only dial) | Gate 2 Thresholds
Gate 3 Winner 1X / X2 (HIGH+LOW, asymmetric) | Gate 4 Team to Score (asymmetric)
Gate 5 Under / Skip | Priority: Gate 0 > Margin (except BTTS/Over 1.5) > Thresholds
Home: Pool alone is YES. Away: Rate(full) alone is YES, Pool alone is LEAN.
Rate(full) for Away = Away Rate Avg >= 70 AND Over 1.5 Avg >= 80.
Persistence: Supabase king_matches (SQLite fallback). Design / card / tabs unchanged.
"""
import sqlite3
import re
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
            elif f == "over_3_5": actual = (home_score+away_score) > 3.5
            elif f == "under_2_5": actual = (home_score+away_score) <= 2.5
            elif f == "btts": actual = (home_score>0 and away_score>0)
            elif f == "1x": actual = (home_score >= away_score)
            elif f == "x2": actual = (away_score >= home_score)
            else: actual = None
            # LEAN counts as a YES call for outcome, NO/LEAN NO/SKIP do not get a win/loss
            _call = (r["decision"] in ("YES", "LEAN YES", "LEAN"))
            outcome = (1 if (actual == _call) else 0) if actual is not None and r["decision"] in ("YES", "NO", "LEAN YES", "LEAN NO", "LEAN") else None
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

# Alias kept for UI calls
_find_existing_match = _find_existing

def _picks_to_json(filt):
    out = []
    for pr in filt["picks"]:
        out.append({"category_key": pr["key"], "label": pr["label"], "field": pr["field"],
                    "a_pct": pr.get("a"), "b_pct": pr.get("b"), "league_pct": pr.get("league"), "avg_pct": pr.get("avg"),
                    "decision": pr["decision"], "confidence": pr.get("confidence", ""), "is_main": 1 if (filt.get("main_pick") and pr["key"]==filt["main_pick"]["key"]) else 0,
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
            elif f == "over_3_5": actual = (hs+aws) > 3.5
            elif f == "under_2_5": actual = (hs+aws) <= 2.5
            elif f == "btts": actual = (hs>0 and aws>0)
            elif f == "1x": actual = (hs >= aws)
            elif f == "x2": actual = (aws >= hs)
            else: actual = None
            _call = (r.get("decision") in ("YES", "LEAN YES", "LEAN"))
            r["outcome"] = (1 if (actual == _call) else 0) if actual is not None and r.get("decision") in ("YES", "NO", "LEAN YES", "LEAN NO", "LEAN") else None
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
    def _pure_num(s):
        s2 = s.strip().replace("%", "").strip()
        try:
            return float(s2)
        except Exception:
            return None
    sections = {}; warnings = []
    # Headers in order, to bound each section block
    HEADERS = [
        ("home_scoring_avg", ["home scoring averages"]),
        ("away_scoring_avg", ["away scoring averages"]),
        ("goals_match", ["goals scored + conceded"]),
        ("home_scoring_rate", ["home scoring rate"]),
        ("away_scoring_rate", ["away scoring rate"]),
        ("over_1_5", ["over 1.5"]),
        ("over_2_5", ["over 2.5"]),
        ("over_3_5", ["over 3.5"]),
        ("btts", ["both-teams-scored", "both teams scored", "bts matches"]),
    ]
    # Find header index for each key
    header_idx = {}
    for key, needles in HEADERS:
        for idx, l in enumerate(lines):
            if any(n in _clean(l).lower() for n in needles):
                header_idx[key] = idx; break
    for key, needles in HEADERS:
        a = b = avg = league = None
        if key not in header_idx:
            warnings.append(key); sections[key] = {"a": None, "b": None, "avg": None, "league": None}; continue
        idx = header_idx[key]
        # End = next header after idx, or idx+20
        ends = [v for v in header_idx.values() if v > idx]
        end_idx = min(ends) if ends else min(idx+20, len(lines))
        end_idx = min(end_idx, idx+20)
        block = lines[idx+1:end_idx]
        # Walk block: (A) label -> number on same line (unless Over) or next pure-number line
        # (B) label -> same, League/Avg label -> next two pure numbers
        i = 0
        is_over = key.startswith("over_")
        while i < len(block):
            ll = _clean(block[i]); low = ll.lower()
            if low.startswith("(a)") and not low.startswith("(a) (b)"):
                if a is None:
                    if not is_over:
                        nums = re.findall(r"-?\d+(?:\.\d+)?", ll[3:])
                        # ignore numbers that are part of "scoring rate" etc? (A) lines have no numbers except Over threshold, so this is safe for non-Over
                        if nums: a = float(nums[0])
                    if a is None:
                        # next pure-number line
                        for k in range(i+1, min(i+3, len(block))):
                            v = _pure_num(block[k])
                            if v is not None: a = v; i = k; break
            elif low.startswith("(b)"):
                if b is None:
                    if not is_over:
                        nums = re.findall(r"-?\d+(?:\.\d+)?", ll[3:])
                        if nums: b = float(nums[0])
                    if b is None:
                        for k in range(i+1, min(i+3, len(block))):
                            v = _pure_num(block[k])
                            if v is not None: b = v; i = k; break
            elif "league" in low:
                nums = re.findall(r"-?\d+(?:\.\d+)?", ll)
                # League label lines like "League Avg home scoring" have no numbers, or "stats.league_..." have no numbers
                # If numbers present on same line (old format), use them
                if len(nums) >= 2:
                    avg = float(nums[0]); league = float(nums[1])
                elif len(nums) == 1 and "avg" not in low:
                    league = float(nums[0])
                else:
                    # next two pure-number lines: Avg then League
                    vals = []
                    for k in range(i+1, min(i+4, len(block))):
                        v = _pure_num(block[k])
                        if v is not None: vals.append(v)
                        if len(vals) >= 2: break
                    if len(vals) >= 2:
                        avg = vals[0]; league = vals[1]; i = i + len(vals)
                    elif len(vals) == 1 and league is None:
                        league = vals[0]
            i += 1
        if avg is None and a is not None and b is not None:
            avg = (a+b)/2.0
        sections[key] = {"a": a, "b": b, "avg": avg, "league": league}
    home_team = away_team = None
    for l in lines:
        ls = _clean(l)
        if ls.startswith("(A)") and home_team is None and "(B)" not in ls:
            home_team = ls[3:].strip()
            for kw in [" scored", " conceded", " over", " BTS"]:
                if kw in home_team: home_team = home_team.split(kw)[0].strip()
        if ls.startswith("(B)") and away_team is None:
            away_team = ls[3:].strip()
            for kw in [" conceded", " scored", " over", " BTS"]:
                if kw in away_team: away_team = away_team.split(kw)[0].strip()
        if home_team and away_team: break
    return {"sections": sections, "home_team": home_team, "away_team": away_team, "warnings": warnings}

# ---------------- Final Logic ----------------
def _fmt(v, is_pct=True):
    if v is None: return "-"
    return f"{v:g}%" if is_pct else f"{v:g}"

def _num(sec, k):
    try: return sec.get(k)
    except Exception: return None

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
    # ---- Inputs ----
    KEYS = ["home_scoring_avg","away_scoring_avg","goals_match","home_scoring_rate","away_scoring_rate","over_1_5","over_2_5","over_3_5","btts"]
    PCT_KEYS = ["home_scoring_rate","away_scoring_rate","over_1_5","over_2_5","over_3_5","btts"]
    def _avg(k): return (secs.get(k) or {}).get("avg")
    def _lg(k): return (secs.get(k) or {}).get("league")
    def _a(k): return (secs.get(k) or {}).get("a")
    def _b(k): return (secs.get(k) or {}).get("b")
    home_pool = (hg.get("a") or 0) + (hg.get("b") or 0) if hg.get("a") is not None and hg.get("b") is not None else None
    away_pool = (ag.get("a") or 0) + (ag.get("b") or 0) if ag.get("a") is not None and ag.get("b") is not None else None
    avgs_above = sum(1 for k in KEYS if _avg(k) is not None and _lg(k) is not None and _avg(k) > _lg(k))
    sides_above = sum(1 for k in KEYS for v in [_a(k), _b(k)] if v is not None and _lg(k) is not None and v > _lg(k))
    avg80 = sum(1 for k in PCT_KEYS if _avg(k) is not None and _avg(k) >= 80)
    side80 = sum(1 for k in PCT_KEYS for v in [_a(k), _b(k)] if v is not None and v >= 80)
    def _margin(k):
        return (_avg(k) - _lg(k)) if (_avg(k) is not None and _lg(k) is not None) else None
    over25_margin = _margin("over_2_5"); btts_margin = _margin("btts")
    over35_margin = _margin("over_3_5"); over15_margin = _margin("over_1_5")
    rate_full = (ar.get("avg") is not None and ar.get("avg") >= 70 and o15.get("avg") is not None and o15.get("avg") >= 80)
    hr_avg = hr.get("avg"); ar_avg = ar.get("avg")
    picks = []
    def _add(key, field, label, decision, confidence, reason, a=None, b=None, league=None, avg=None):
        picks.append({"key": key, "label": label, "field": field, "a": a, "b": b, "league": league, "avg": avg, "decision": decision, "confidence": confidence, "reason": reason})
    def _rank_conf(c):
        return {"Highest": 4, "High": 3, "Medium": 2, "Low": 1}.get(c, 0)
    # ---- Gate 0 ----
    gate0_over = avgs_above == 9
    gate0_1x = avgs_above == 8
    # ---- Gate 1 Margin: Over 2.5 / Over 3.5 / Over 1.5 ----
    # Over 2.5 margin gate (Gate 1 overrides Gate 2 for Over 2.5)
    if gate0_over:
        o25_dec, o25_conf, o25_reason = "YES", "Highest", "Gate 0 — avgs_above == 9. Over 2.5 YES, Highest. Stop for Over."
        o35_dec, o35_conf, o35_reason = "YES", "Highest", "Gate 0 — avgs_above == 9. Over 3.5 YES, Highest."
    else:
        if over25_margin is None: o25_dec, o25_conf, o25_reason = "SKIP", "Low", "Over 2.5 not in paste."
        elif over25_margin >= 20: o25_dec, o25_conf, o25_reason = "YES", "Highest", f"Gate 1 — over25_margin {over25_margin:+.0f} >= +20. Over 2.5 YES, Highest."
        elif over25_margin >= 10: o25_dec, o25_conf, o25_reason = "YES", "High", f"Gate 1 — over25_margin {over25_margin:+.0f} >= +10. Over 2.5 YES, High."
        elif over25_margin >= 0: o25_dec, o25_conf, o25_reason = "YES", "Medium", f"Gate 1 — over25_margin {over25_margin:+.0f} in 0 to +10. Over 2.5 YES, Medium. Margin is a dial only here. Bucket 68% in file."
        else: o25_dec, o25_conf, o25_reason = "NO", "Medium", f"Gate 1 — over25_margin {over25_margin:+.0f} < 0. Over 2.5 NO. Under lean."
        if over35_margin is None: o35_dec, o35_conf, o35_reason = "SKIP", "Low", "Over 3.5 not in paste."
        elif over35_margin >= 20: o35_dec, o35_conf, o35_reason = "YES", "Highest", f"Gate 1 — over35_margin {over35_margin:+.0f} >= +20. Over 3.5 YES, Highest."
        else: o35_dec, o35_conf, o35_reason = "NO", "Medium", f"Gate 1 — over35_margin {over35_margin if over35_margin is not None else 0:+.0f} < +20. Over 3.5 NO. Margin below +20 is not enough."
    # Over 1.5 — no downgrade, no raise
    if over15_margin is None: o15_dec, o15_conf, o15_reason = "SKIP", "Low", "Over 1.5 not in paste."
    elif over15_margin >= 0: o15_dec, o15_conf, o15_reason = "YES", "Highest", f"Gate 1 — over15_margin {over15_margin:+.0f} >= 0. Over 1.5 YES, Highest. Do NOT downgrade if margin <+10. Do NOT raise to +10."
    else: o15_dec, o15_conf, o15_reason = "NO", "Medium", f"Gate 1 — over15_margin {over15_margin:+.0f} < 0. Over 1.5 NO."
    # Gate 2 Over 2.5 threshold — margin already decided, so threshold can only downgrade Over 2.5 confidence when margin <+10 (already Medium) — it never overrides a Gate 1 NO to YES, and never upgrades.
    # (Kept as a reason note: threshold agreement is recorded in the reason, decision stays with Gate 1.)
    thr_o25 = None
    if o25.get("avg") is not None:
        if away_pool is not None and away_pool >= 3.0 and o25.get("avg", 0) >= 60: thr_o25 = "High"
        elif o15.get("avg") is not None and o15.get("avg", 0) >= 80 and o25.get("avg", 0) >= 60: thr_o25 = "High"
        elif o25.get("avg", 0) >= 60: thr_o25 = "Medium"
    if thr_o25 and o25_dec == "YES" and over25_margin is not None and over25_margin < 10:
        o25_conf = "Medium"
    # Combined margin (strongest Over + BTTS together) — overrides individual Over/BTTS confidence to Highest when both fire
    combined = (over25_margin is not None and btts_margin is not None and over25_margin >= 15 and btts_margin >= 15)
    combined_top = (over25_margin is not None and btts_margin is not None and over25_margin >= 20 and btts_margin >= 20)
    if combined and not gate0_over:
        o25_dec, o25_conf = "YES", "Highest"
        o35_dec, o35_conf = "YES", "Highest"
        o25_reason += " Combined margin >= +15 both — Over 2.5 + Over 3.5 + BTTS Highest."
        o35_reason += " Combined margin >= +15 both — Highest."
    # ---- Gate 2 BTTS ----
    btts_dec, btts_conf, btts_reason = "SKIP", "Low", "No BTTS gate fired."
    btts_agree_rates = (hr_avg is not None and ar_avg is not None and hr_avg >= 70 and ar_avg >= 70)
    btts_agree_pool = (home_pool is not None and home_pool >= 3.0 and ar_avg is not None and ar_avg >= 70)
    btts_both_scorers = (hr.get("a") is not None and ar.get("b") is not None and hr.get("a") >= 70 and ar.get("b") >= 70)
    btts_only_home = (hr_avg is not None and ar_avg is not None and hr_avg >= 70 and ar_avg < 70)
    btts_only_away = (hr_avg is not None and ar_avg is not None and ar_avg >= 70 and hr_avg < 70)
    if btts_agree_rates:
        btts_dec, btts_conf = "YES", "Highest"
        btts_reason = f"Gate 2 — Both Rates >= 70 ({_fmt(hr_avg)} / {_fmt(ar_avg)}). BTTS YES, Highest. No margin filter. No downgrade if btts_margin <+10."
    elif btts_agree_pool:
        btts_dec, btts_conf = "YES", "High"
        btts_reason = f"Gate 2 — Home Pool {home_pool:.2f} >= 3.0 + Away Rate {_fmt(ar_avg)} >= 70. BTTS YES, High."
    elif btts_both_scorers:
        btts_dec, btts_conf = "YES", "Medium"
        btts_reason = f"Gate 2 — Both Scorers >= 70 (Home Rate A {_fmt(hr.get('a'))} + Away Rate B {_fmt(ar.get('b'))}). BTTS YES, Medium."
    elif btts_only_home:
        btts_dec, btts_conf = "NO", "Medium"
        btts_reason = "Gate 2 — Only Home Rate >= 70. BTTS NO. Away blanks."
    elif btts_only_away:
        btts_dec, btts_conf = "LEAN NO", "Medium"
        btts_reason = "Gate 2 — Only Away Rate >= 70. BTTS NO / LEAN NO."
    # 80% match gate — BTTS YES Highest, Over NOT automatic
    is_80 = (side80 >= 7 or avg80 >= 4)
    if is_80 and btts_dec != "YES":
        btts_dec, btts_conf = "YES", "Highest"
        btts_reason = f"Gate 2 — 80% match (side80 {side80}, avg80 {avg80}). BTTS YES, Highest. Over is NOT automatic."
    elif is_80 and btts_dec == "YES":
        btts_reason += f" 80% match also fires (side80 {side80}, avg80 {avg80})."
    # BTTS margin alone never creates YES; margin >=0 with no agreement is NO (already handled: if no YES above, fall through)
    if btts_dec in ("SKIP", "NO", "LEAN NO"):
        if btts_margin is not None and btts_margin >= 0 and not (btts_agree_rates or btts_agree_pool or btts_both_scorers or is_80):
            # If Gate 2 said NO/LEAN NO, keep it. If SKIP, margin-only is NO.
            if btts_dec == "SKIP":
                btts_dec, btts_conf = "LEAN NO", "Medium"
                btts_reason = f"Gate 1 — btts_margin {btts_margin:+.0f} >= 0 with no Rates / Pool agreement. BTTS NO / LEAN NO. Margin alone is not a BTTS pick (29% in file)."
            else:
                btts_reason += f" BTTS margin {btts_margin:+.0f} does not override."
    if combined:
        btts_dec, btts_conf = "YES", "Highest"
        btts_reason += " Combined margin >= +15 both — BTTS Highest."
    # 80% + Over: only if both pools >= 3.0 — add a note, Over decisions already come from Gate 1; do NOT add Over from 80% alone
    if is_80 and home_pool is not None and away_pool is not None and home_pool >= 3.0 and away_pool >= 3.0:
        btts_reason += " Both pools >= 3.0 — Over may be added via Gate 1, not from 80% alone."
    # ---- Gate 3 Winner ----
    # 1X — Home: Pool alone is YES
    if gate0_1x:
        x1_dec, x1_conf, x1_reason = "YES", "Highest", "Gate 0 — avgs_above == 8. 1X YES, Highest. Gate 0 1X beats Gate 3 No winner."
    elif home_pool is not None and away_pool is not None and home_pool >= 3.0 and away_pool < 3.0:
        x1_dec, x1_conf, x1_reason = "YES", "Highest", f"Gate 3 — Home Pool {home_pool:.2f} >= 3.0 + Away Pool {away_pool:.2f} < 3.0. 1X YES, Highest. 27 / 27 in file."
    elif home_pool is not None and hr_avg is not None and home_pool >= 3.0 and hr_avg >= 70:
        x1_dec, x1_conf, x1_reason = "YES", "High", f"Gate 3 — Home Pool {home_pool:.2f} >= 3.0 + Home Rate {_fmt(hr_avg)} >= 70. 1X YES, High."
    elif hr_avg is not None and ar_avg is not None and hr_avg >= 70 and ar_avg < 70:
        x1_dec, x1_conf, x1_reason = "YES", "Medium", "Gate 3 — Home Rate >= 70 + Away Rate < 70. 1X YES, Medium."
    else:
        # No winner rule: Both pools >= 3.0 (and not Gate 0 8/9) -> No 1X pick; avg80 >= 4 -> No winner (unless Gate 0)
        if home_pool is not None and away_pool is not None and home_pool >= 3.0 and away_pool >= 3.0:
            x1_dec, x1_conf, x1_reason = "SKIP", "Low", "Gate 3 — Both pools >= 3.0. No winner pick. Both can score."
        elif avg80 >= 4:
            x1_dec, x1_conf, x1_reason = "SKIP", "Low", "Gate 3 — avg80 >= 4. No winner pick."
        else:
            x1_dec, x1_conf, x1_reason = "SKIP", "Low", "Gate 3 — No 1X gate fired."
    # X2 — Away: Pool alone is LEAN, Rate(full) must agree for YES
    if away_pool is not None and home_pool is not None and away_pool >= 3.0 and home_pool < 3.0:
        if rate_full:
            x2_dec, x2_conf, x2_reason = "YES", "High", f"Gate 3 — Away Pool {away_pool:.2f} >= 3.0 + Home Pool {home_pool:.2f} < 3.0 + Rate(full) (Away Rate {_fmt(ar_avg)} >= 70 + Over 1.5 {_fmt(o15.get('avg'))} >= 80). X2 YES, High."
        else:
            x2_dec, x2_conf, x2_reason = "LEAN", "Medium", f"Gate 3 — Away Pool {away_pool:.2f} >= 3.0 + Home Pool {home_pool:.2f} < 3.0 but Rate(full) fails (Away Rate {_fmt(ar_avg)} / Over 1.5 {_fmt(o15.get('avg'))}). X2 LEAN, Medium. Not YES, not No Pick."
    elif ar_avg is not None and hr_avg is not None and ar_avg >= 70 and hr_avg < 70:
        x2_dec, x2_conf, x2_reason = "LEAN", "Medium", "Gate 3 — Away Rate >= 70 + Home Rate < 70. X2 LEAN, Medium."
    else:
        if home_pool is not None and away_pool is not None and home_pool >= 3.0 and away_pool >= 3.0 and not gate0_1x and avgs_above != 9:
            x2_dec, x2_conf, x2_reason = "SKIP", "Low", "Gate 3 — Both pools >= 3.0. No winner pick."
        elif avg80 >= 4 and not gate0_1x:
            x2_dec, x2_conf, x2_reason = "SKIP", "Low", "Gate 3 — avg80 >= 4. No winner pick."
        else:
            x2_dec, x2_conf, x2_reason = "SKIP", "Low", "Gate 3 — No X2 gate fired."
    # Gate 0 9 -> No winner (already: 1X/X2 stay SKIP unless Gate 0 8; if avgs==9, force SKIP for winner)
    if gate0_over:
        x1_dec, x1_conf, x1_reason = "SKIP", "Low", "Gate 0 — avgs_above == 9. No winner pick."
        x2_dec, x2_conf, x2_reason = "SKIP", "Low", "Gate 0 — avgs_above == 9. No winner pick."
    # ---- Gate 4 Team to Score (asymmetric) ----
    # Home: Pool alone is YES
    if home_pool is not None and home_pool >= 3.0:
        hs_dec, hs_conf, hs_reason = "YES", "High", f"Gate 4 — Home Pool {home_pool:.2f} >= 3.0. Home Scores YES, High. 95% in file."
    elif hr.get("a") is not None and hr.get("b") is not None and hr.get("a") >= 70 and hr.get("b") >= 70:
        hs_dec, hs_conf, hs_reason = "YES", "High", f"Gate 4 — Home Rate A {_fmt(hr.get('a'))} >= 70 + Away Conceding B {_fmt(hr.get('b'))} >= 70. Home Scores YES, High."
    else:
        hs_dec, hs_conf, hs_reason = "SKIP", "Low", "Gate 4 — No Home Scores gate fired."
    # Away: Rate(full) alone is YES. Pool adds nothing.
    if rate_full:
        as_dec, as_conf, as_reason = "YES", "Highest", f"Gate 4 — Rate(full) passes (Away Rate {_fmt(ar_avg)} >= 70 + Over 1.5 {_fmt(o15.get('avg'))} >= 80). Away Scores YES, Highest. Pool does not matter."
    elif away_pool is not None and away_pool >= 3.0:
        as_dec, as_conf, as_reason = "LEAN", "Medium", f"Gate 4 — Rate(full) fails but Away Pool {away_pool:.2f} >= 3.0. Away Scores LEAN, Medium. Conflict bucket 67%. LEAN, not No Pick."
    else:
        as_dec, as_conf, as_reason = "SKIP", "Low", "Gate 4 — Rate(full) fails and Away Pool < 3.0. No pick."
    # ---- Gate 5 Under / Skip ----
    if hr_avg is not None and ar_avg is not None and hr_avg < 70 and ar_avg < 70:
        un_dec, un_conf, un_reason = "YES", "Highest", f"Gate 5 — Home Rate {_fmt(hr_avg)} < 70 + Away Rate {_fmt(ar_avg)} < 70. Under 2.5 YES, Highest. Over 0 / 6 in file."
    elif bt.get("avg") is not None and o25.get("avg") is not None and bt.get("avg") < 50 and o25.get("avg") < 60:
        un_dec, un_conf, un_reason = "LEAN", "Medium", "Gate 5 — BTTS Avg < 50 + Over 2.5 Avg < 60. Under 2.5 LEAN."
    else:
        un_dec, un_conf, un_reason = "SKIP", "Low", "Gate 5 — No Under gate fired."
    # ---- Assemble picks (design: same card order, now 9 markets) ----
    _add("over_1_5", "over_1_5", "Total over 1.5", o15_dec, o15_conf, o15_reason, o15.get("a"), o15.get("b"), o15.get("league"), o15.get("avg"))
    _add("over_2_5", "over_2_5", "Total over 2.5", o25_dec, o25_conf, o25_reason, o25.get("a"), o25.get("b"), o25.get("league"), o25.get("avg"))
    _add("over_3_5", "over_3_5", "Total over 3.5", o35_dec, o35_conf, o35_reason, o35.get("a"), o35.get("b"), o35.get("league"), o35.get("avg"))
    _add("btts", "btts", "Both score", btts_dec, btts_conf, btts_reason, bt.get("a"), bt.get("b"), bt.get("league"), bt.get("avg"))
    _add("home_scores", "home_scores", "Home scores", hs_dec, hs_conf, hs_reason, hr.get("a"), hr.get("b"), hr.get("league"), hr.get("avg"))
    _add("away_scores", "away_scores", "Away scores", as_dec, as_conf, as_reason, ar.get("a"), ar.get("b"), ar.get("league"), ar.get("avg"))
    _add("1x", "1x", "1X (Home does not lose)", x1_dec, x1_conf, x1_reason, hg.get("a"), hg.get("b"), hg.get("league"), hg.get("avg"))
    _add("x2", "x2", "X2 (Away does not lose)", x2_dec, x2_conf, x2_reason, ag.get("a"), ag.get("b"), ag.get("league"), ag.get("avg"))
    _add("under_2_5", "under_2_5", "Total under 2.5", un_dec, un_conf, un_reason, o25.get("a"), o25.get("b"), o25.get("league"), o25.get("avg"))
    agreement_detail = {
        "home_scoring_avg": f"Pool {home_pool:.2f}" if home_pool is not None else "not in paste",
        "away_scoring_avg": f"Pool {away_pool:.2f}" if away_pool is not None else "not in paste",
        "goals_match": f"avgs_above {avgs_above} / 9, sides_above {sides_above} / 18",
        "home_scoring_rate": f"Avg {_fmt(hr_avg)}",
        "away_scoring_rate": f"Avg {_fmt(ar_avg)}",
        "over_1_5": f"margin {over15_margin:+.0f}" if over15_margin is not None else "not in paste",
        "over_2_5": f"margin {over25_margin:+.0f}" if over25_margin is not None else "not in paste",
        "over_3_5": f"margin {over35_margin:+.0f}" if over35_margin is not None else "not in paste",
        "btts": f"margin {btts_margin:+.0f}" if btts_margin is not None else "not in paste",
        "rate_full": "passes" if rate_full else "fails",
        "avg80": str(avg80), "side80": str(side80),
    }
    # Main for accumulator — one per match, YES only (LEAN never a main pick). Priority: Gate 0 1X > Over 3.5 > Over 2.5 > Over 1.5 > Home > Away > 1X > X2 > BTTS > Under
    yes_picks = [p for p in picks if p["decision"] == "YES"]
    main_pick = None; decision = "NO BET"; decision_reason = "No gate fires with at least Medium confidence — SKIP. Do not force a pick."
    if yes_picks:
        prio = {"1x": 0, "over_3_5": 1, "over_2_5": 2, "over_1_5": 3, "home_scores": 4, "away_scores": 5, "x2": 6, "btts": 7, "under_2_5": 8}
        # Gate 0 1X should outrank Over when avgs==8; otherwise Over outranks 1X — handle by boosting Gate 0 1X to top, else demote 1X below Over
        def _prio(pp):
            if pp["key"] == "1x" and gate0_1x: return -1
            if pp["key"] == "1x": return 6
            return prio.get(pp["key"], 99)
        ranked = sorted(yes_picks, key=lambda pp: (_prio(pp), -_rank_conf(pp.get("confidence", "Medium"))))
        main_pick = ranked[0]
        decision = "BET"
        decision_reason = f"Agreement holds for {len(yes_picks)} pick(s). Main (accumulator, one per match): {main_pick['label']} YES ({main_pick.get('confidence','')}). Other YES picks are separate single bets only — do not put Home + Away + Over from the same match in one accumulator."
    else:
        leans = [p for p in picks if p["decision"] in ("LEAN", "LEAN YES", "LEAN NO")]
        if leans:
            decision_reason = f"No YES pick. {len(leans)} LEAN(s) only — not a main pick. NO BET."
    return {"picks": picks, "agreement": agreement_detail, "decision": decision, "decision_reason": decision_reason,
            "main_pick": main_pick, "survivors": yes_picks, "derived": {p["field"]: (True if p["decision"]=="YES" else None) for p in picks}}

# Backwards compat alias
compute_filter_rule = compute_agreement

# ---------------- Card ----------------
# Same display card as the last green code — green = BET, red = NO BET. Only the logic underneath changed to agreement.
def _card_html(m, filt, settled=False):
    home = m.get("home_team") or "?"; away = m.get("away_team") or "?"
    fixture = f"{home} vs {away}"
    if settled:
        hs = m.get("home_score"); aws = m.get("away_score")
        fixture += f" — {hs}-{aws}" if hs is not None and aws is not None else ""
    dec = filt.get("decision") or "NO BET"
    is_bet = dec == "BET"
    color = "#22c55e" if is_bet else "#ef4444"
    bg = "#22c55e22" if is_bet else "#ef444422"
    main = filt.get("main_pick")
    main_txt = f"{main['label']} YES" if main else "No agreement — NO BET"
    lines = []
    for p in filt.get("picks") or []:
        is_main = main and p["key"] == main["key"] and p["decision"] == "YES"
        _d = p["decision"]
        badge = "MAIN" if is_main else ("YES" if _d == "YES" else ("LEAN" if _d in ("LEAN", "LEAN YES") else ("LEAN NO" if _d == "LEAN NO" else ("NO" if _d == "NO" else "—"))))
        bcol = "#22c55e" if _d == "YES" else ("#f59e0b" if _d in ("LEAN", "LEAN YES", "LEAN NO") else ("#ef4444" if _d == "NO" else "#64748b"))
        _conf = p.get("confidence") or ""
        if _conf: badge = f"{badge} · {_conf}"
        outcome = ""
        if settled and p.get("outcome") is not None:
            outcome = " ✅ WIN" if p.get("outcome") == 1 else " ❌ LOSS"
        lines.append(f"<div style='display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #1e293b'><span style='color:#e2e8f0'>{p['label']}</span><span style='color:{bcol};font-weight:700'>{badge}{outcome}</span></div><div style='font-size:0.82rem;color:#94a3b8;margin-bottom:4px'>{p.get('reason','')}</div>")
    agree = filt.get("agreement") or {}
    agree_txt = " · ".join([f"{k}: {v}" for k, v in agree.items() if k in ("home_scoring_avg","away_scoring_avg","goals_match","home_scoring_rate","away_scoring_rate","over_1_5","over_2_5","over_3_5","btts")]) if agree else ""
    head = f"<div style='background:{bg};border:1px solid {color};border-radius:12px;padding:14px 16px'><div style='font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:{color}'>{dec} • AGREEMENT</div><div style='font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px'>{fixture}</div><div style='font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-top:4px'>Main: {main_txt}</div><div style='font-size:0.88rem;color:#94a3b8;margin-top:2px'>{filt.get('decision_reason','')}</div></div>"
    return f"<div class='match-card'>{head}<div style='margin-top:10px'>{''.join(lines)}</div><div style='font-size:0.78rem;color:#64748b;margin-top:8px'>Agreement: {agree_txt}</div></div>"

def _filt_from_db_row(m):
    picks = []
    for r in m.get("picks") or []:
        if r.get("is_implied"): continue
        picks.append({"key": r.get("category_key"), "label": r.get("label"), "field": r.get("field"), "confidence": r.get("confidence", ""),
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
st.caption("Final logic — Gate 0 · Margin (Over 2.5 only) · Thresholds · 1X / X2 · Team to Score (asymmetric) · Under. Home Pool YES · Away Rate(full) YES, Pool LEAN. Store: Supabase king_matches (SQLite fallback).")

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
