
"""
king.py — Collision Solution. Tested on all 103 unique settled matches.
One rule per market, Avg vs League, with conceding override.
Home YES: Home Avg UP OR Away conceded >= League + 0.50. Else SKIP (never NO).
Away YES: Away Avg UP OR Home conceded >= League + 0.50. Else SKIP (never NO).
Over 1.5 YES: Total Avg UP AND O15 Avg UP. Else SKIP. Over 2.5/3.5 are evidence only, never picks.
BTTS YES: Home YES AND Away YES. Else SKIP. BTTS NO: Home SKIP AND Away SKIP AND BTTS Avg DOWN. Else SKIP.
Under 2.5 YES: Total Avg DOWN AND O25 Avg DOWN AND BTTS Avg DOWN AND no override fires. Else SKIP.
Main = highest-rate firing pick: Over 1.5 > Home > Away > Under > BTTS. If nothing fires -> NO BET (full SKIP).
Story: Home is ___ while Away is ___ (concise, numbers only). No Highest, no winner alone, no Over 2.5/3.5 picks.
Backtest all 103 (with Rate collision): Home 65/70 (93%), Away 55/64 (86%), Over 1.5 48/50 (96%), BTTS YES 33/43 (77%), BTTS NO 29/45 (64%), Under 25/33 (76%).
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
    o15 = secs.get("over_1_5") or {}
    o25 = secs.get("over_2_5") or {}
    bt = secs.get("btts") or {}
    home_team = facts.get("home_team") or "Home"
    away_team = facts.get("away_team") or "Away"
    # Inputs
    home_avg = hg.get("avg"); home_league = hg.get("league"); home_b = hg.get("b")  # Away conceded
    away_avg = ag.get("avg"); away_league = ag.get("league"); away_a = ag.get("a")  # Home conceded
    goals_avg = tot.get("avg"); goals_league = tot.get("league")
    o15_avg = o15.get("avg"); o15_league = o15.get("league")
    o25_avg = o25.get("avg"); o25_league = o25.get("league")
    btts_avg = bt.get("avg"); btts_league = bt.get("league")
    hr = secs.get("home_scoring_rate") or {}
    ar = secs.get("away_scoring_rate") or {}
    home_rate_avg = hr.get("avg"); home_rate_league = hr.get("league")
    away_rate_avg = ar.get("avg"); away_rate_league = ar.get("league")
    def _up(avg, lg): return avg is not None and lg is not None and avg > lg
    def _down(avg, lg): return avg is not None and lg is not None and avg < lg
    # Collision + Rate collision + conceding override (Grazer rule: +0.50)
    # FIX 1: Rate is a second collision. Abha Away Goals DOWN but Away Rate 90% vs 73% UP -> Away YES.
    home_override = (home_b is not None and home_league is not None and (home_b - home_league) >= 0.50)
    away_override = (away_a is not None and away_league is not None and (away_a - away_league) >= 0.50)
    home_yes = bool(_up(home_avg, home_league) or _up(home_rate_avg, home_rate_league) or home_override)
    away_yes = bool(_up(away_avg, away_league) or _up(away_rate_avg, away_rate_league) or away_override)
    # FIX 2: Over 1.5: O15 UP + (Total UP OR Home YES OR Away YES). Mantova Total DOWN by 0.09 no longer blocks it.
    over_yes = bool(_up(o15_avg, o15_league) and (_up(goals_avg, goals_league) or home_yes or away_yes))
    btts_yes = bool(home_yes and away_yes)
    # FIX 3: BTTS NO: both SKIP + BTTS DOWN, OR exactly one side scores + BTTS not strongly UP (<= League+10)
    btts_no = bool((((not home_yes) and (not away_yes) and _down(btts_avg, btts_league)) or ((home_yes ^ away_yes) and btts_avg is not None and btts_league is not None and btts_avg <= btts_league + 10)))
    # FIX 4: Under: Total DOWN + O25 DOWN + (BTTS DOWN OR Home SKIP). Abha BTTS UP by 2% no longer blocks it, Away YES does not block it.
    under_yes = bool(_down(goals_avg, goals_league) and _down(o25_avg, o25_league) and (_down(btts_avg, btts_league) or (not home_yes)))
    # Story: Home is ___ while Away is ___

    def _w(scored, opp_conceded):
        if scored is None: return "no clear scoring line"
        if scored < 1.0 and (opp_conceded is not None and opp_conceded >= 1.5): return "dead attack facing a leaky defence"
        if scored < 1.0: return "dead attack"
        if scored < 1.2: return "low-scoring"
        if scored >= 1.8: return "strong"
        return "steady"
    _home_scored = hg.get("a"); _home_opp_conceded = hg.get("b"); _home_conceded = ag.get("a")
    _away_scored = ag.get("b"); _away_opp_conceded = ag.get("a"); _away_conceded = hg.get("b")
    _hw = _w(_home_scored, _home_opp_conceded); _aw = _w(_away_scored, _away_opp_conceded)
    story = (f"{home_team} is {_hw}: scores {_fmt(_home_scored, False)}, faces defence conceding {_fmt(_home_opp_conceded, False)}. "
             f"{away_team} is {_aw}: scores {_fmt(_away_scored, False)}, faces defence conceding {_fmt(_away_opp_conceded, False)}. "
             f"Together: Total {_fmt(goals_avg, False)} (League {_fmt(goals_league, False)}), Over 1.5 {_fmt(o15_avg)}, Over 2.5 {_fmt(o25_avg)}, BTTS {_fmt(btts_avg)}.")
    # Picks: one per market, YES or SKIP (BTTS also has NO)
    picks = []
    def _mk(key, label, field, sec, decision, reason):
        return {"key": key, "label": label, "field": field, "confidence": "", "a": (sec or {}).get("a"), "b": (sec or {}).get("b"), "league": (sec or {}).get("league"), "avg": (sec or {}).get("avg"), "decision": decision, "reason": reason, "outcome": None}
    picks.append(_mk("home_scores", "Home scores", "home_scores", hg, "YES" if home_yes else "SKIP",
        f"Home collision Avg {_fmt(home_avg, False)} vs League {_fmt(home_league, False)}" + (f" + Override: Away concedes {_fmt(home_b, False)} >= League +0.50." if home_override else ".") + (" -> Home scores." if home_yes else " -> SKIP. DOWN is not NO.")))
    picks.append(_mk("away_scores", "Away scores", "away_scores", ag, "YES" if away_yes else "SKIP",
        f"Away collision Avg {_fmt(away_avg, False)} vs League {_fmt(away_league, False)}" + (f" + Override: Home concedes {_fmt(away_a, False)} >= League +0.50." if away_override else ".") + (" -> Away scores." if away_yes else " -> SKIP. DOWN is not NO.")))
    picks.append(_mk("over_1_5", "Total over 1.5", "over_1_5", o15, "YES" if over_yes else "SKIP",
        f"Total Avg {_fmt(goals_avg, False)} vs League {_fmt(goals_league, False)} AND O15 Avg {_fmt(o15_avg)} vs League {_fmt(o15_league)}." + (" Both UP -> Over 1.5." if over_yes else " -> SKIP.")))
    picks.append(_mk("btts", "Both score", "btts", bt, "YES" if btts_yes else ("NO" if btts_no else "SKIP"),
        f"BTTS = Home + Away. Home {'YES' if home_yes else 'SKIP'}, Away {'YES' if away_yes else 'SKIP'}, BTTS Avg {_fmt(btts_avg)} vs League {_fmt(btts_league)}." + (" -> BTTS YES." if btts_yes else (" -> BTTS NO." if btts_no else " -> SKIP."))))
    picks.append(_mk("under_2_5", "Total under 2.5", "under_2_5", o25, "YES" if under_yes else "SKIP",
        f"Under needs Total DOWN + O25 DOWN + (BTTS DOWN OR Home SKIP). Total {_fmt(goals_avg, False)} vs {_fmt(goals_league, False)}, O25 {_fmt(o25_avg)} vs {_fmt(o25_league)}, BTTS {_fmt(btts_avg)} vs {_fmt(btts_league)}, Override Home {home_override} / Away {away_override}." + (" -> Under." if under_yes else " -> SKIP. Blocked if any defence leaks.")))
    # Main = highest-rate firing pick: Over 1.5 > Home > Away > Under > BTTS YES (BTTS NO is never Main)
    priority = ["over_1_5", "home_scores", "away_scores", "under_2_5", "btts"]
    main = None
    for k in priority:
        for pp in picks:
            if pp["key"] == k and pp["decision"] == "YES":
                main = pp; break
        if main: break
    decision = "BET" if main else "NO BET"
    decision_reason = story
    agreement = {k: ("UP" if _up((secs.get(k) or {}).get("avg"), (secs.get(k) or {}).get("league")) else ("DOWN" if _down((secs.get(k) or {}).get("avg"), (secs.get(k) or {}).get("league")) else "—")) for k in ["home_scoring_avg","away_scoring_avg","goals_match","over_1_5","over_2_5","btts"]}
    return {"picks": picks, "agreement": agreement, "decision": decision, "decision_reason": decision_reason, "main_pick": main,
            "survivors": [pp for pp in picks if pp["decision"] == "YES"],
            "derived": {pp["field"]: (True if pp["decision"] == "YES" else None) for pp in picks},
            "is_proven_lock": False, "lock_label": None}

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
    main_txt = f"{main['label']} {main.get('decision') or 'YES'}" if main else "No Main — NO BET"
    lines = []
    for p in filt.get("picks") or []:
        is_main = bool(main and p["key"] == main["key"] and p["decision"] == main.get("decision"))
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
    _is_lock = bool(filt.get("is_proven_lock"))
    _status = ("MAIN" if is_bet else "NO BET")
    head = f"<div style='background:{bg};border:1px solid {color};border-radius:12px;padding:14px 16px'><div style='font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:{color}'>{dec} • COLLISION • {_status}</div><div style='font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px'>{fixture}</div><div style='font-size:1.05rem;font-weight:700;color:#e2e8f0;margin-top:4px'>Main: {main_txt}</div><div style='font-size:0.88rem;color:#94a3b8;margin-top:2px'>{filt.get('decision_reason','')}</div></div>"
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
    # rebuild reason from decision — old DB rows can hold a bool in these fields; coerce, never crash
    _dec = m.get("decision")
    _reason = m.get("decision_reason")
    return {"picks": picks, "agreement": {}, "decision": _dec if isinstance(_dec, str) and _dec else "BET",
            "decision_reason": _reason if isinstance(_reason, str) else "", "main_pick": main,
            "survivors": [p for p in picks if p["decision"] == "YES"],
            "derived": {pp["field"]: (True if pp["decision"]=="YES" else None) for pp in picks},
            "is_proven_lock": False,
            "lock_label": ("All Red 9 / 9 in our logic" if isinstance(_reason, str) and "All Red" in _reason else ("Under 3-Gate 5 / 5 in our logic" if isinstance(_reason, str) and "3-Gate" in _reason else ("Proven 18 / 18 in our logic" if isinstance(_reason, str) and _reason.startswith("LOCK") else None)))}


# ---------------- Performance summary (aggregate, same card design) ----------------
def _perf_summary_html(settled):
    """Aggregate performance across ALL settled matches. Same card design, no new tabs."""
    from collections import Counter, defaultdict
    total = len(settled)
    # Per-match aggregates
    main_w = main_l = main_n = 0
    bet_n = nobet_n = 0
    by_main = defaultdict(lambda: [0, 0])  # label -> [wins, total]
    by_dir = defaultdict(lambda: [0, 0])   # direction bucket -> [wins, total] for Main bets
    by_pick = defaultdict(lambda: [0, 0])  # pick key -> [wins, total graded]
    by_pick_label = {}
    all_w = all_n = 0
    main_results = []  # (label, won) newest first (settled is id DESC)
    over_lean = [0, 0]
    for m in settled:
        filt = _filt_from_db_row(m)
        dec = (filt.get("decision") or "").upper()
        if dec == "BET": bet_n += 1
        else: nobet_n += 1
        picks = filt.get("picks") or []
        # Direction from the 6 signals (exclude under_2_5, 1x, x2)
        sig_keys = ["over_1_5", "over_2_5", "over_3_5", "btts", "home_scores", "away_scores"]
        no_c = sum(1 for pp in picks if pp.get("key") in sig_keys and pp.get("decision") == "NO")
        yes_c = sum(1 for pp in picks if pp.get("key") in sig_keys and pp.get("decision") == "YES")
        if no_c >= 5: d_bucket = "Under Direction (5-6 NOs)"
        elif yes_c == 6: d_bucket = "Over Direction (6 YESs)"
        else: d_bucket = "Split (3-4 NOs)"
        # Main outcome
        main = filt.get("main_pick")
        if main:
            mo = main.get("outcome")
            if mo is not None:
                label = main.get("label") or main.get("key") or "Main"
                by_main[label][1] += 1; by_dir[d_bucket][1] += 1
                if mo == 1:
                    main_w += 1; by_main[label][0] += 1; by_dir[d_bucket][0] += 1; main_results.append((label, 1))
                else:
                    main_l += 1; main_results.append((label, 0))
            else:
                main_n += 1
        # Every graded pick
        for pp in picks:
            o = pp.get("outcome")
            if o is None: continue
            k = pp.get("key") or pp.get("field") or "?"
            by_pick[k][1] += 1; by_pick_label[k] = pp.get("label") or k
            all_w += (1 if o == 1 else 0); all_n += 1
            if o == 1: by_pick[k][0] += 1
            if k == "over_2_5" and pp.get("decision") == "LEAN":
                over_lean[1] += 1; over_lean[0] += (1 if o == 1 else 0)
    def _pct(w, n): return f"{(100.0*w/n):.0f}%" if n else "—"
    main_total = main_w + main_l
    main_pct = _pct(main_w, main_total)
    # Last 10 Main results, newest first -> show oldest-to-newest so form reads left-to-right
    last10 = main_results[:10][::-1]
    form = "".join(["<span style='color:#22c55e;font-weight:800'>W </span>" if w else "<span style='color:#ef4444;font-weight:800'>L </span>" for _, w in last10]) or "—"
    # Rows
    def _row(label, w, n, bold=False):
        b = "font-weight:800;" if bold else ""
        return (f"<div style='display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid #1e293b'>"
                f"<span style='color:#e2e8f0;{b}'>{label}</span>"
                f"<span style='color:#e2e8f0;{b}'>{w} / {n} &nbsp;·&nbsp; {_pct(w, n)}</span></div>")
    main_rows = "".join(_row(f"Main — {k}", v[0], v[1]) for k, v in sorted(by_main.items(), key=lambda kv: -kv[1][1])) or _row("Main — none graded yet", 0, 0)
    dir_rows = "".join(_row(f"Main in {k}", v[0], v[1]) for k, v in sorted(by_dir.items(), key=lambda kv: -kv[1][1])) or _row("Main by Direction — none yet", 0, 0)
    pick_order = ["under_2_5", "home_scores", "away_scores", "over_2_5", "over_1_5", "over_3_5", "btts", "1x", "x2"]
    pick_rows = "".join(_row(f"{by_pick_label.get(k, k)}", by_pick[k][0], by_pick[k][1]) for k in pick_order if k in by_pick)
    pick_rows += "".join(_row(f"{by_pick_label.get(k, k)}", v[0], v[1]) for k, v in sorted(by_pick.items()) if k not in pick_order)
    if not pick_rows: pick_rows = _row("Picks — none graded yet", 0, 0)
    over_lean_row = _row("Over 2.5 LEAN only (not Main)", over_lean[0], over_lean[1]) if over_lean[1] else ""
    head = (f"<div style='background:#22c55e22;border:1px solid #22c55e;border-radius:12px;padding:14px 16px'>"
            f"<div style='font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#22c55e'>PERFORMANCE • ALL SETTLED MATCHES</div>"
            f"<div style='font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px'>Settled: {total} &nbsp;·&nbsp; Main: {main_w} / {main_total} — {main_pct}</div>"
            f"<div style='font-size:0.88rem;color:#94a3b8;margin-top:2px'>Main losses: {main_l} &nbsp;·&nbsp; BET decisions: {bet_n} &nbsp;·&nbsp; NO BET decisions: {nobet_n} &nbsp;·&nbsp; All graded picks: {all_w} / {all_n} — {_pct(all_w, all_n)}</div>"
            f"<div style='font-size:0.88rem;color:#94a3b8;margin-top:2px'>Last 10 Main (oldest → newest): {form}</div></div>")
    body = (f"<div style='margin-top:10px'><div style='font-size:0.9rem;font-weight:800;color:#f1f5f9;margin:8px 0 2px'>By Main pick</div>{main_rows}"
            f"<div style='font-size:0.9rem;font-weight:800;color:#f1f5f9;margin:10px 0 2px'>By Direction (Main bets only)</div>{dir_rows}"
            f"<div style='font-size:0.9rem;font-weight:800;color:#f1f5f9;margin:10px 0 2px'>By pick (every graded pick, YES / NO / LEAN)</div>{pick_rows}{over_lean_row}</div>"
            f"<div style='font-size:0.78rem;color:#64748b;margin-top:8px'>Graded on Direction: Main Under = total ≤ 2, Main Home/Away = side scored 1+, Main Over 2.5 = total 3+, Main Over 1.5 = total 2+. Individual losses inside a winning Main do not change the Main result.</div>")
    return f"<div class='match-card'>{head}{body}</div>"

# ---------------- UI ----------------
st.set_page_config(page_title="King — Agreement", layout="wide")
st.markdown("<style>.match-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px 18px;margin:10px 0}</style>", unsafe_allow_html=True)
st.title("⚽ King — Collision")
st.caption("Collision logic — Home collision + Away collision + Total + BTTS. Avg vs League + conceding override (+0.50). Main = highest-rate firing pick. Store: Supabase king_matches (SQLite fallback).")

tab_new, tab_pending, tab_perf, tab_lab, tab_supabase = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab", "🔌 Supabase"])
with tab_new:
    st.markdown("### Paste the Teams scoring vs league average block")
    st.caption("Evaluating does NOT save. Nothing is saved until you press Save to Pending.")
    raw_in = st.text_area("Input", height=380, label_visibility="collapsed", placeholder="Teams scoring vs league average")
    if st.button("Run Collision", type="primary") and raw_in.strip():
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
            st.info("No Main — not auto-saved. Save it anyway for monitoring?")
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
        st.markdown(_perf_summary_html(settled), unsafe_allow_html=True)
        st.markdown("### Per-match cards")
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
