
"""
king.py — CORRELATION LOGIC (V4.2)
Don't bet the average. Bet the correlation: BOTH teams must independently
agree on the same outcome, for their own reasons.

RULES (V4.2):
- TO SCORE (Home / Away): RATE correlation only.
  Both rates >= league rate - 2pts (tolerance) AND both rates >= 65%
  AND scorer volume >= 0.80 x league scoring volume.
  Scorer = Home scored avg for Home scores, Away scored avg for Away scores.
  Volume does NOT need to beat league, it just must not collapse (<80%).
  Example that fails: Mainz vs Nurnberg, Nurnberg scored 1.33 vs league 1.68
  (79.2% of league) -> bottleneck, SPOTTING, SKIP.
  Example that passes: Guadalajara 1.78 vs league 1.55 -> LOGIC.
- OVER 1.5 / OVER 2.5: RATE + VOLUME correlation.
  Both rates >= league rate - 2pts AND both rates >= 60% (Over 1.5) / >= 20% (Over 2.5)
  AND total volume (Goals scored+conceded avg) > 2.50.
  Example that fails: Municipal Over 1.5, total volume 2.37 < 2.50 -> SKIP.
- BTTS: LOGIC only if Home-scores LOGIC AND Away-scores LOGIC both pass.
  BTTS rate alone is never enough.
- Over 3.5 always ignored. No scoreline / exact score is ever predicted.
- Picks are YES (LOGIC) or SKIP (SPOTTING). There is no NO pick.
- Decision: BET if >=1 LOGIC YES pick, else NO BET - NO LOGIC.
- Main Pick: biggest relative margin among LOGIC YES picks.
  Margin = avg((A-Lg)/Lg, (B-Lg)/Lg), absolute value.
  Tie hierarchy: Away > Over 1.5 > Over 2.5 > BTTS > Home.
- Evaluating does NOT save. Only "Save to Pending" saves.
- Performance settles per-signal only, never scoreline.
Persistence: Supabase ONE table king_matches if SUPABASE_URL/KEY set, else SQLite king.db.
Backtest (V4.2, 67 triggers): 63 kept, 63 wins on kept after filtering
Municipal, Soroksar, Mainz x2 as SPOTTING. Old 80%-average rule: 63/67.
"""
import os
import json
import sqlite3
import urllib.request
import urllib.error
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "king.db"

# ---------------------------------------------------------------- Supabase — ONE TABLE: public.king_matches
# Secrets: SUPABASE_URL, SUPABASE_KEY. If not set -> fallback to local SQLite king.db
SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
def _sb_configured():
    return bool(SUPABASE_URL and SUPABASE_KEY)
def _sb_req(method, table, payload=None, query=""):
    url = f"{SUPABASE_URL}/rest/v1/{table}{query}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("apikey", SUPABASE_KEY)
    req.add_header("Authorization", f"Bearer {SUPABASE_KEY}")
    req.add_header("Content-Type", "application/json")
    if method in ("POST", "PATCH"):
        req.add_header("Prefer", "return=representation")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode()
            return json.loads(body) if body.strip() else []
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Supabase {method} {table} failed {e.code}: {e.read().decode()[:500]}")
def _sb_save_match(facts, filt, raw_paste):
    picks = []
    for pr in filt["picks"]:
        picks.append({"category_key": pr["key"], "label": pr["label"], "field": pr["field"], "a_pct": pr["a"], "b_pct": pr["b"], "league_pct": pr["league"], "avg_pct": pr.get("avg"), "decision": pr["decision"], "margin_pct": pr.get("margin_pct"), "is_main": 1 if (filt.get("main_pick") and pr["key"]==filt["main_pick"]["key"]) else 0, "is_implied": 0, "implied_note": None, "outcome": None})
    for field, note in (filt.get("implied_notes") or {}).items():
        picks.append({"category_key": field, "label": FIELD_LABEL.get(field, field), "field": field, "a_pct": None, "b_pct": None, "league_pct": None, "avg_pct": None, "decision": "YES" if filt["derived"].get(field) else "NO", "margin_pct": None, "is_main": 0, "is_implied": 1, "implied_note": note, "outcome": None})
    row = _sb_req("POST", "king_matches", {"home_team": facts.get("home_team"), "away_team": facts.get("away_team"), "created_at": datetime.now(timezone.utc).isoformat(), "raw_paste": raw_paste, "decision": filt["decision"], "decision_reason": filt.get("decision_reason", ""), "main_pick_key": (filt["main_pick"]["key"] if filt.get("main_pick") else None), "main_pick_label": (filt["main_pick"]["label"] if filt.get("main_pick") else None), "main_pick_decision": (filt["main_pick"]["decision"] if filt.get("main_pick") else None), "main_pick_margin_pct": (filt["main_pick"]["margin_pct"] if filt.get("main_pick") else None), "survivors_count": len(filt["survivors"]), "picks": picks, "sections": facts.get("sections") or {}})[0]
    return row["id"]
def _sb_load_matches(pending_only=False, settled_only=False):
    rows = _sb_req("GET", "king_matches", query="?select=*&order=id.desc")
    out = []
    for m in rows:
        if pending_only and m.get("home_score") is not None: continue
        if settled_only and m.get("home_score") is None: continue
        if m.get("is_deleted"): continue
        m = dict(m); m["picks"] = m.get("picks") or []
        out.append(m)
    return out
def _sb_settle(match_id, home_score, away_score):
    rows = _sb_req("GET", "king_matches", query=f"?select=*&id=eq.{match_id}")
    if not rows: return
    picks = rows[0].get("picks") or []
    for r in picks:
        if r.get("is_implied"): continue
        if r.get("decision") == "SKIP":
            r["outcome"] = None
        else:
            f = r.get("field")
            if f == "home_scores": actual = home_score > 0
            elif f == "away_scores": actual = away_score > 0
            elif f == "over_1_5": actual = (home_score+away_score) > 1.5
            elif f == "over_2_5": actual = (home_score+away_score) > 2.5
            elif f == "btts": actual = (home_score>0 and away_score>0)
            else: actual = None
            r["outcome"] = (1 if (actual == (r.get("decision")=="YES")) else 0) if actual is not None else None
    _sb_req("PATCH", "king_matches", {"home_score": home_score, "away_score": away_score, "settled_at": datetime.now(timezone.utc).isoformat(), "picks": picks}, query=f"?id=eq.{match_id}")

PROVEN = {
    "home_scoring_rate": "V4.2 Correlation",
    "away_scoring_rate": "V4.2 Correlation",
    "over_1_5": "V4.2 Correlation",
    "over_2_5": "V4.2 Correlation",
    "btts": "V4.2 Correlation",
}
FILTER_CATEGORIES = [
    ("home_scoring_rate", "Home Scoring Rate", "Does the home team score?", "home_scores"),
    ("away_scoring_rate", "Away Scoring Rate", "Does the away team score?", "away_scores"),
    ("over_1_5", "Over 1.5 Goals", "Total > 1.5?", "over_1_5"),
    ("over_2_5", "Over 2.5 Goals", "Total > 2.5?", "over_2_5"),
    ("btts", "BTTS", "Do both teams score?", "btts"),
]
HIER = {"away_scoring_rate": 0, "over_1_5": 1, "over_2_5": 2, "btts": 3, "home_scoring_rate": 4}
LABEL = {"Home Scoring Rate": "Home scores", "Away Scoring Rate": "Away scores",
         "Over 1.5 Goals": "Total over 1.5", "Over 2.5 Goals": "Total over 2.5", "BTTS": "Both score"}
FIELD_LABEL = {"home_scores": "Home scores", "away_scores": "Away scores",
               "over_1_5": "Total over 1.5", "over_2_5": "Total over 2.5", "btts": "Both score"}

# ---------------------------------------------------------------- SQL
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team TEXT, away_team TEXT,
    created_at TEXT NOT NULL,
    raw_paste TEXT,
    decision TEXT,               -- BET / NO BET - WEAK / NO BET - CONFLICT
    decision_reason TEXT,
    main_pick_key TEXT, main_pick_label TEXT, main_pick_decision TEXT, main_pick_margin_pct REAL,
    survivors_count INTEGER,
    home_score INTEGER, away_score INTEGER, settled_at TEXT,
    is_deleted INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    category_key TEXT NOT NULL, label TEXT NOT NULL, field TEXT NOT NULL,
    a_pct REAL, b_pct REAL, league_pct REAL, avg_pct REAL,
    decision TEXT NOT NULL,      -- YES / NO / SKIP
    margin_pct REAL,
    is_main INTEGER DEFAULT 0,
    is_implied INTEGER DEFAULT 0,
    implied_note TEXT,
    outcome INTEGER,             -- 1 WIN / 0 LOSS / NULL SKIP-or-unsettled
    FOREIGN KEY(match_id) REFERENCES matches(id)
);
CREATE TABLE IF NOT EXISTS raw_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    section_key TEXT NOT NULL, a_val REAL, b_val REAL, avg_val REAL, league_val REAL,
    FOREIGN KEY(match_id) REFERENCES matches(id)
);
CREATE INDEX IF NOT EXISTS idx_picks_match ON picks(match_id);
CREATE INDEX IF NOT EXISTS idx_picks_field ON picks(field);
"""

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

def save_match(facts, filt, raw_paste):
    if _sb_configured():
        try:
            return _sb_save_match(facts, filt, raw_paste)
        except Exception as e:
            st.warning(f'Supabase save failed, fell back to local king.db: {e}')
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO matches (home_team,away_team,created_at,raw_paste,decision,decision_reason,main_pick_key,main_pick_label,main_pick_decision,main_pick_margin_pct,survivors_count) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (facts.get("home_team"), facts.get("away_team"), datetime.now(timezone.utc).isoformat(), raw_paste,
         filt["decision"], filt.get("decision_reason",""), (filt["main_pick"]["key"] if filt.get("main_pick") else None),
         (filt["main_pick"]["label"] if filt.get("main_pick") else None),
         (filt["main_pick"]["decision"] if filt.get("main_pick") else None),
         (filt["main_pick"]["margin_pct"] if filt.get("main_pick") else None), len(filt["survivors"])))
    mid = cur.lastrowid
    for p in filt["picks"]:
        cur.execute(
            "INSERT INTO picks (match_id,category_key,label,field,a_pct,b_pct,league_pct,avg_pct,decision,margin_pct,is_main,is_implied,implied_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, p["key"], p["label"], p["field"], p["a"], p["b"], p["league"], p.get("avg"), p["decision"],
             p.get("margin_pct"), 1 if (filt.get("main_pick") and p["key"]==filt["main_pick"]["key"]) else 0, 0, None))
    for field, note in (filt.get("implied_notes") or {}).items():
        lbl = FIELD_LABEL.get(field, field)
        cur.execute(
            "INSERT INTO picks (match_id,category_key,label,field,a_pct,b_pct,league_pct,avg_pct,decision,margin_pct,is_main,is_implied,implied_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, field, lbl, field, None, None, None, None, "YES" if filt["derived"].get(field) else "NO", None, 0, 1, note))
    for key, sec in facts["sections"].items():
        cur.execute("INSERT INTO raw_sections (match_id,section_key,a_val,b_val,avg_val,league_val) VALUES (?,?,?,?,?,?)",
                    (mid, key, sec.get("a"), sec.get("b"), sec.get("avg"), sec.get("league")))
    conn.commit(); conn.close()
    return mid


def save_match_dedup(facts, filt, raw_paste):
    """Auto-save on Generate, but never duplicate.
    If the same raw paste (same teams + same numbers) is already in Pending/Settled, return (existing_id, True).
    Otherwise save and return (new_id, False)."""
    key = (raw_paste or "").strip()
    try:
        existing = load_matches()
    except Exception:
        existing = []
    for m in existing:
        if (m.get("raw_paste") or "").strip() == key:
            return m["id"], True
        # Fallback: same teams + same decision + same survivors, no raw stored (old rows)
        if key and m.get("home_team")==facts.get("home_team") and m.get("away_team")==facts.get("away_team") and (m.get("raw_paste") or "").strip()==key:
            return m["id"], True
    return save_match(facts, filt, raw_paste), False

def settle_match_db(match_id, home_score, away_score):
    if _sb_configured():
        try:
            _sb_settle(match_id, home_score, away_score); return
        except Exception as e:
            st.warning(f'Supabase settle failed, fell back to local king.db: {e}')
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT * FROM picks WHERE match_id=? AND is_implied=0", (match_id,))
    rows = cur.fetchall()
    for r in rows:
        if r["decision"] == "SKIP":
            outcome = None
        else:
            f = r["field"]
            if f == "home_scores": actual = home_score > 0
            elif f == "away_scores": actual = away_score > 0
            elif f == "over_1_5": actual = (home_score+away_score) > 1.5
            elif f == "over_2_5": actual = (home_score+away_score) > 2.5
            elif f == "btts": actual = (home_score>0 and away_score>0)
            else: actual = None
            outcome = (1 if (actual == (r["decision"]=="YES")) else 0) if actual is not None else None
        cur.execute("UPDATE picks SET outcome=? WHERE id=?", (outcome, r["id"]))
    cur.execute("UPDATE matches SET home_score=?, away_score=?, settled_at=? WHERE id=?",
                (home_score, away_score, datetime.now(timezone.utc).isoformat(), match_id))
    conn.commit(); conn.close()

def load_matches(pending_only=False, settled_only=False):
    if _sb_configured():
        try:
            return _sb_load_matches(pending_only, settled_only)
        except Exception as e:
            st.warning(f'Supabase load failed, fell back to local king.db: {e}')
    conn = get_db(); cur = conn.cursor()
    q = "SELECT * FROM matches WHERE is_deleted=0"
    if pending_only: q += " AND home_score IS NULL"
    if settled_only: q += " AND home_score IS NOT NULL"
    q += " ORDER BY id DESC"
    rows = [dict(r) for r in cur.execute(q).fetchall()]
    for m in rows:
        m["picks"] = [dict(r) for r in cur.execute("SELECT * FROM picks WHERE match_id=? ORDER BY id", (m["id"],)).fetchall()]
    conn.close()
    return rows

# ---------------------------------------------------------------- parsing
def _clean(s: str) -> str:
    return s.replace("\u00a0"," ").strip()

def _nums_in_section(lines, start_idx, end_idx, want_pct: bool):
    nums=[]
    for i in range(start_idx, end_idx):
        ls=_clean(lines[i])
        if not ls: continue
        low=ls.lower()
        if low.startswith("(a)") or low.startswith("(b)") or "average" in low or "league" in low or "stats." in low:
            continue
        try:
            v=float(ls.replace("%","").strip())
            nums.append(v)
        except Exception:
            pass
    return nums

def _find_section(lines, keywords):
    for i, l in enumerate(lines):
        low=_clean(l).lower()
        if any(k in low for k in keywords):
            return i
    return None

def parse_filter_data(text: str) -> dict:
    lines=text.splitlines()
    sections={}
    warnings=[]
    def get_section(key, keywords):
        idx=_find_section(lines, keywords)
        if idx is None:
            warnings.append(f"Section not found: {key}")
            return None
        # find (A) line, (B) line, average line, league line after idx
        a=b=avg=league=None; avg_paste=None
        j=idx+1
        while j < len(lines) and j < idx+30:
            low=_clean(lines[j]).lower()
            if low.startswith("(a)"):
                # next numeric line
                k=j+1
                while k < len(lines) and not _clean(lines[k]): k+=1
                try: a=float(_clean(lines[k]).replace("%",""))
                except Exception: pass
            elif low.startswith("(b)"):
                k=j+1
                while k < len(lines) and not _clean(lines[k]): k+=1
                try: b=float(_clean(lines[k]).replace("%",""))
                except Exception: pass
            elif "average" in low and "(a)" not in low and "(b)" not in low:
                # the number after this line, or numbers in following lines until league
                pass
            elif "league" in low or "stats." in low:
                # FIX: after the League label there are TWO numbers:
                #   1st = (A)(B) average (from the paste)
                #   2nd = League (the one the Filter must use)
                # Old code took the 1st number as league -> compared A/B to the average.
                # Correct: skip the 1st, take the 2nd as league. If only 1 number exists, use it.
                nums_after=[]
                k=j+1
                while k < len(lines) and len(nums_after) < 2:
                    ls2=_clean(lines[k])
                    if not ls2:
                        k+=1; continue
                    try:
                        nums_after.append(float(ls2.replace("%","")))
                    except Exception:
                        break
                    k+=1
                if len(nums_after) >= 2:
                    avg_paste=nums_after[0]; league=nums_after[1]
                elif len(nums_after) == 1:
                    league=nums_after[0]
                break
            j+=1
        # avg: use the paste's own (A)(B) average (1st number after League label) if we found it,
        # else fall back to (a+b)/2
        if 'avg_paste' in locals() and avg_paste is not None:
            avg=avg_paste
        elif a is not None and b is not None:
            avg=(a+b)/2.0
        sections[key]={"a":a,"b":b,"avg":avg,"league":league}
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
    # team names from (A) / (B) lines: first (A) line is home team, first (B) line is away team
    home_team=away_team=None
    for l in lines:
        ls=_clean(l)
        if ls.startswith("(A)") and home_team is None:
            home_team=ls[3:].replace("scored at home","").replace("conceded at home","").replace("scored","").strip().split("  ")[0].strip()
            # keep full name before keywords
            for kw in [" scored"," conceded"," over"," BTS"]:
                if kw in home_team: home_team=home_team.split(kw)[0].strip()
        if ls.startswith("(B)") and away_team is None:
            away_team=ls[3:].strip()
            for kw in [" conceded"," scored"," over"," BTS"]:
                if kw in away_team: away_team=away_team.split(kw)[0].strip()
        if home_team and away_team: break
    # cleaner: use first (A)/(B) pair from Home scoring averages block
    return {"sections": sections, "home_team": home_team, "away_team": away_team, "warnings": warnings}

# ---------------------------------------------------------------- CORRELATION LOGIC (V4.2)
def _rate_ok(a, b, league, min_rate=65.0, tol=2.0):
    """Both rates within 2pts of league (or above) AND both >= min_rate.
    This is the correlation: both teams independently agree on rate."""
    if a is None or b is None or league is None:
        return False
    return a >= league - tol and b >= league - tol and a >= min_rate and b >= min_rate

def _scorer_volume_ok(scorer_avg, league_vol, floor=0.80):
    """Scorer must not collapse: scorer volume >= 80% of league volume.
    Mainz/Nurnberg: 1.33 / 1.68 = 79.2% -> FAIL. Guadalajara: 1.78 / 1.55 -> PASS."""
    if scorer_avg is None or league_vol is None or league_vol == 0:
        return False
    return scorer_avg >= floor * league_vol

def _decide(a, league, b):
    # Kept for backwards compat — old both-beat-league is NO LONGER used.
    # Correlation logic lives in compute_filter_rule.
    if a is None or b is None or league is None:
        return "SKIP"
    if a > league and b > league:
        return "YES"
    if a < league and b < league:
        return "NO"
    return "SKIP"

def _lbl(v, lg):
    return f"{v:g}% BEATS league {lg:g}%" if v > lg else f"{v:g}% BELOW league {lg:g}%" if v < lg else f"{v:g}% = league {lg:g}% (neutral)"

def _margin(a, b, league):
    if league in (None, 0):
        return 0.0
    return abs(((a - league) / league + (b - league) / league) / 2.0)

def compute_filter_rule(facts: dict) -> dict:
    S = facts.get("sections") or {}
    def sec(k):
        return S.get(k) or {}
    # Volume sections
    hs_avg = sec("home_scoring_avg")   # a=home scored, b=away conceded, league=league home scoring
    aw_avg = sec("away_scoring_avg")   # a=home conceded, b=away scored, league=league away scoring
    gm = sec("goals_match")            # avg = total goals (A+B)/2
    total_vol = gm.get("avg")
    # Rate sections
    hs_rate = sec("home_scoring_rate") # a=home scoring rate, b=away conceding rate
    aw_rate = sec("away_scoring_rate") # a=home conceding rate, b=away scoring rate
    o15 = sec("over_1_5")
    o25 = sec("over_2_5")
    btts = sec("btts")

    results = {}  # key -> (decision, reason, margin)
    # --- Home scores: RATE correlation + scorer-volume floor ---
    if hs_rate.get("a") is None or hs_rate.get("b") is None or hs_rate.get("league") is None:
        results["home_scoring_rate"] = ("SKIP", "Section not found in paste.", 0.0)
    else:
        ok_rate = _rate_ok(hs_rate["a"], hs_rate["b"], hs_rate["league"], min_rate=65.0)
        ok_vol = _scorer_volume_ok(hs_avg.get("a"), hs_avg.get("league"))
        if ok_rate and ok_vol:
            results["home_scoring_rate"] = ("YES", f"LOGIC: both rates agree ({hs_rate['a']:g}% / {hs_rate['b']:g}% vs league {hs_rate['league']:g}% -2 tol) + scorer volume {hs_avg.get('a')} vs league {hs_avg.get('league')} >=80%", _margin(hs_rate["a"], hs_rate["b"], hs_rate["league"]))
        else:
            why = []
            if not ok_rate: why.append(f"RATE split: {hs_rate['a']:g}% / {hs_rate['b']:g}% vs league {hs_rate['league']:g}% (need both >= league-2 and >=65%)")
            if not ok_vol: why.append(f"VOLUME bottleneck: scorer {hs_avg.get('a')} vs league {hs_avg.get('league')} <80%")
            results["home_scoring_rate"] = ("SKIP", "SPOTTING: " + " | ".join(why), 0.0)
    # --- Away scores ---
    if aw_rate.get("a") is None or aw_rate.get("b") is None or aw_rate.get("league") is None:
        results["away_scoring_rate"] = ("SKIP", "Section not found in paste.", 0.0)
    else:
        ok_rate = _rate_ok(aw_rate["a"], aw_rate["b"], aw_rate["league"], min_rate=65.0)
        ok_vol = _scorer_volume_ok(aw_avg.get("b"), aw_avg.get("league"))
        if ok_rate and ok_vol:
            results["away_scoring_rate"] = ("YES", f"LOGIC: both rates agree ({aw_rate['a']:g}% / {aw_rate['b']:g}% vs league {aw_rate['league']:g}% -2 tol) + scorer volume {aw_avg.get('b')} vs league {aw_avg.get('league')} >=80%", _margin(aw_rate["a"], aw_rate["b"], aw_rate["league"]))
        else:
            why = []
            if not ok_rate: why.append(f"RATE split: {aw_rate['a']:g}% / {aw_rate['b']:g}% vs league {aw_rate['league']:g}% (need both >= league-2 and >=65%)")
            if not ok_vol: why.append(f"VOLUME bottleneck: scorer {aw_avg.get('b')} vs league {aw_avg.get('league')} <80% (e.g. Nurnberg 1.33 vs 1.68)")
            results["away_scoring_rate"] = ("SKIP", "SPOTTING: " + " | ".join(why), 0.0)
    # --- Over 1.5: RATE + TOTAL VOLUME > 2.50 ---
    if o15.get("a") is None or o15.get("b") is None or o15.get("league") is None:
        results["over_1_5"] = ("SKIP", "Section not found in paste.", 0.0)
    else:
        ok_rate = _rate_ok(o15["a"], o15["b"], o15["league"], min_rate=60.0)
        ok_vol = total_vol is not None and total_vol > 2.50
        if ok_rate and ok_vol:
            results["over_1_5"] = ("YES", f"LOGIC: both Over 1.5 rates agree ({o15['a']:g}% / {o15['b']:g}% vs league {o15['league']:g}% -2 tol) + total volume {total_vol:g} > 2.50", _margin(o15["a"], o15["b"], o15["league"]))
        else:
            why = []
            if not ok_rate: why.append(f"RATE split: {o15['a']:g}% / {o15['b']:g}% vs league {o15['league']:g}%")
            if not ok_vol: why.append(f"VOLUME: total goals {total_vol} <= 2.50 (e.g. Municipal 2.37)")
            results["over_1_5"] = ("SKIP", "SPOTTING: " + " | ".join(why), 0.0)
    # --- Over 2.5: RATE + TOTAL VOLUME > 2.50 ---
    if o25.get("a") is None or o25.get("b") is None or o25.get("league") is None:
        results["over_2_5"] = ("SKIP", "Section not found in paste.", 0.0)
    else:
        ok_rate = _rate_ok(o25["a"], o25["b"], o25["league"], min_rate=20.0)
        ok_vol = total_vol is not None and total_vol > 2.50
        if ok_rate and ok_vol:
            results["over_2_5"] = ("YES", f"LOGIC: both Over 2.5 rates agree ({o25['a']:g}% / {o25['b']:g}% vs league {o25['league']:g}% -2 tol) + total volume {total_vol:g} > 2.50", _margin(o25["a"], o25["b"], o25["league"]))
        else:
            why = []
            if not ok_rate: why.append(f"RATE split: {o25['a']:g}% / {o25['b']:g}% vs league {o25['league']:g}%")
            if not ok_vol: why.append(f"VOLUME: total goals {total_vol} <= 2.50")
            results["over_2_5"] = ("SKIP", "SPOTTING: " + " | ".join(why), 0.0)
    # --- BTTS: LOGIC only if BOTH Home-scores LOGIC and Away-scores LOGIC ---
    if btts.get("a") is None or btts.get("b") is None or btts.get("league") is None:
        results["btts"] = ("SKIP", "Section not found in paste.", 0.0)
    else:
        home_ok = results["home_scoring_rate"][0] == "YES"
        away_ok = results["away_scoring_rate"][0] == "YES"
        ok_rate = _rate_ok(btts["a"], btts["b"], btts["league"], min_rate=40.0)
        if home_ok and away_ok and ok_rate:
            results["btts"] = ("YES", f"LOGIC: Home LOGIC + Away LOGIC + BTTS rates agree ({btts['a']:g}% / {btts['b']:g}% vs league {btts['league']:g}%)", _margin(btts["a"], btts["b"], btts["league"]))
        else:
            why = []
            if not home_ok: why.append("Home-scores not LOGIC")
            if not away_ok: why.append("Away-scores not LOGIC")
            if not ok_rate: why.append(f"BTTS RATE split: {btts['a']:g}% / {btts['b']:g}% vs league {btts['league']:g}%")
            results["btts"] = ("SKIP", "SPOTTING: " + " | ".join(why), 0.0)

    picks = []
    derived = {}
    for key, label, question, field in FILTER_CATEGORIES:
        sec_r = S.get(key) or {}
        dec, reason, marg = results[key]
        derived[field] = (True if dec == "YES" else None)
        picks.append({"key": key, "label": label, "question": question, "field": field,
                      "a": sec_r.get("a"), "b": sec_r.get("b"), "league": sec_r.get("league"),
                      "avg": sec_r.get("avg"), "decision": dec, "reason": reason,
                      "margin": marg, "margin_pct": marg * 100.0})
    survivors = [x for x in picks if x["decision"] == "YES"]
    weak = len(survivors) == 0
    conflicts = []  # No NO picks in V4.2, so no direct contradictions to check.
    main_pick = None; decision = "BET"; decision_reason = ""
    if weak:
        decision = "NO BET - WEAK"; decision_reason = "No LOGIC: no signal has both teams agreeing. All >=80% averages here are SPOTTING."
    else:
        ranked = sorted(survivors, key=lambda pp: (-pp["margin"], HIER.get(pp["key"], 99)))
        main_pick = ranked[0]
        decision_reason = f"Main Pick {LABEL.get(main_pick['label'], main_pick['label'])} {main_pick['decision']} at {main_pick['margin_pct']:.1f}% margin (LOGIC)."
    return {"picks": picks, "derived": derived, "survivors": survivors, "weak_test": weak,
            "conflicts": conflicts, "decision": decision, "decision_reason": decision_reason,
            "main_pick": main_pick, "implied_notes": {}}

# ---------------------------------------------------------------- card (DB-shaped picks)
def _card_html(m, filt, settled=False):
    home, away = m.get('home_team') or '?', m.get('away_team') or '?'
    picks=filt['picks']; main=filt.get('main_pick')
    fixture=f'<div style="font-size:1.15rem;font-weight:700;color:#f1f5f9">{home} <span style="font-size:0.72rem;color:#94a3b8">(HOME)</span> vs {away} <span style="font-size:0.72rem;color:#94a3b8">(AWAY)</span></div>'
    if settled and m.get('home_score') is not None:
        head=fixture+f'<div style="font-size:1rem;color:#94a3b8;margin-top:6px">Result: {home} {m["home_score"]}-{m["away_score"]} {away}</div>'
    else:
        if filt['decision']=="BET":
            ml=LABEL.get(main['label'], main['label']) if main else '-'
            mv='YES' if main['decision']=='YES' else 'NO'
            head=fixture+f'<div style="margin-top:10px;background:#10b98122;border:1px solid #10b981;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#10b981">DECISION: BET &nbsp;|&nbsp; MAIN PICK</div><div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-top:4px">{ml} - {mv}</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">Strongest signal today: {main["margin_pct"]:.1f}% above/below league on average. History only breaks ties.</div></div>'
        elif filt['decision']=="NO BET - CONFLICT":
            head=fixture+f'<div style="margin-top:10px;background:#f59e0b22;border:1px solid #f59e0b;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#f59e0b">DECISION: NO BET — CONFLICT</div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;margin-top:4px">Survivors contradict each other. Do not bet.</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">{filt.get("decision_reason","")}</div></div>'
        else:
            head=fixture+f'<div style="margin-top:10px;background:#ef444422;border:1px solid #ef4444;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#ef4444">DECISION: NO BET — WEAK</div><div style="font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px">Weak test - do not bet</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">No LOGIC signal. All >=80% averages here are SPOTTING — one team disagrees on rate/volume.</div></div>'
    rows=''
    if main and filt['decision']=="BET":
        rows+=f'<tr><td style="padding:6px 10px 6px 0;color:#e2e8f0;font-weight:700">MAIN: {LABEL.get(main["label"], main["label"])}</td><td style="padding:6px 10px;color:#f1f5f9;font-weight:800">{"YES" if main["decision"]=="YES" else "NO"}</td><td style="padding:6px 0;color:#94a3b8;font-size:0.85rem">{main["a"]:g}% vs league {main["league"]:g}% | {main["b"]:g}% vs league {main["league"]:g}% - margin {main["margin_pct"]:.1f}%</td></tr>'
    for p in picks:
        if p['decision']=="SKIP": continue
        if main and p['key']==main['key']: continue
        rows+=f'<tr><td style="padding:6px 10px 6px 0;color:#e2e8f0">{LABEL.get(p["label"], p["label"])}</td><td style="padding:6px 10px;color:#f1f5f9;font-weight:700">{"YES" if p["decision"]=="YES" else "NO"}</td><td style="padding:6px 0;color:#94a3b8;font-size:0.85rem">{p["a"]:g}% vs league {p["league"]:g}% | {p["b"]:g}% vs league {p["league"]:g}% - margin {p.get("margin_pct",0):.1f}%</td></tr>'
    for field, note in (filt.get('implied_notes') or {}).items():
        v=filt['derived'].get(field)
        rows+=f'<tr><td style="padding:6px 10px 6px 0;color:#e2e8f0">{FIELD_LABEL.get(field,field)}</td><td style="padding:6px 10px;color:#f1f5f9;font-weight:700">{"YES" if v else "NO"} *</td><td style="padding:6px 0;color:#94a3b8;font-size:0.85rem">{note} - proved by another pick</td></tr>'
    if not rows: rows='<tr><td colspan=3 style="color:#94a3b8">No signals agree.</td></tr>'
    skipped=[p['label'] for p in picks if p['decision']=='SKIP']
    skipped_line='Skipped (split / neutral, not used): '+', '.join(skipped) if skipped else 'Nothing skipped.'
    verdict=''
    if settled:
        parts=[]
        for p in (m.get("picks") or []):
            if p.get("is_implied"): continue
            if p.get("outcome") is not None:
                parts.append(f"{LABEL.get(p['label'], p['label'])}: {'WIN' if p['outcome']==1 else 'LOSS'}")
        if parts: verdict='<div style="margin-top:10px;font-size:0.85rem;color:#94a3b8">'+'  |  '.join(parts)+'</div>'
    legend='<div style="margin-top:6px;font-size:0.82rem;color:#94a3b8">First number = home-team side (A) | Second = away-team side (B) | * = proved by another pick | Margin = avg relative distance from league | = league is neutral, not BELOW</div>'
    title='MAIN PICK + OTHERS' if (main and filt['decision']=="BET") else 'SIGNALS THAT AGREE'
    return f'<div class="match-card">{head}<div style="margin-top:14px;font-size:0.78rem;letter-spacing:0.06em;font-weight:700;color:#94a3b8">{title}</div>{legend}<table style="margin-top:4px">{rows}</table><div style="margin-top:10px;font-size:0.82rem;color:#64748b">{skipped_line}</div>{verdict}</div>'

def _filt_from_db_row(m):
    picks=[]
    for r in m.get("picks") or []:
        if r.get("is_implied"): continue
        picks.append({"key":r["category_key"],"label":r["label"],"question":"","field":r["field"],
                      "a":r["a_pct"],"b":r["b_pct"],"league":r["league_pct"],"avg":r["avg_pct"],
                      "decision":r["decision"],"reason":"","margin_pct":r["margin_pct"] or 0,"margin":(r["margin_pct"] or 0)/100.0})
    main=None
    for p in picks:
        if p["key"]==m.get("main_pick_key"): main=p
    implied={}
    for r in m.get("picks") or []:
        if r.get("is_implied"): implied[r["field"]]=r.get("implied_note") or ""
    return {"picks":picks,"derived":{},"survivors":[p for p in picks if p["decision"] in ("YES","NO")],
            "weak_test": m.get("decision")=="NO BET - WEAK","conflicts":[],"decision":m.get("decision") or "BET",
            "decision_reason":m.get("decision_reason") or "","main_pick":main,"implied_notes":implied}

# ---------------------------------------------------------------- UI
st.set_page_config(page_title="King — The Filter Rule", layout="wide")
st.markdown("<style>.match-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px 18px;margin:10px 0}</style>", unsafe_allow_html=True)
st.title("⚽ King — Correlation Logic (V4.2)")
st.caption("Correlation Logic V4.2: BOTH teams must independently agree on RATE (within 2pts of league). TO SCORE: scorer volume >=80% of league. OVER 1.5/2.5: total volume >2.50. BTTS: needs Home LOGIC + Away LOGIC. Average >=80% alone = SPOTTING, not LOGIC.")

tab_new, tab_pending, tab_perf, tab_lab, tab_sb = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab", "🔌 Supabase"])
with tab_new:
    st.markdown("### Paste the Teams scoring vs league average block")
    st.caption("Generate = auto-saves to Pending immediately. If the same paste was already saved, it says Already stored — no duplicate.")
    raw_in = st.text_area("Input", height=380, label_visibility="collapsed", placeholder="Teams scoring vs league average")
    if st.button("Run the Filter", type="primary") and raw_in.strip():
        facts = parse_filter_data(raw_in.strip()); filt = compute_filter_rule(facts)
        st.session_state["pending_eval"] = {"facts": facts, "filt": filt, "raw": raw_in.strip(), "save_status": None, "save_id": None}
        # Auto-save immediately on Generate (deduped)
        try:
            mid, dup = save_match_dedup(facts, filt, raw_in.strip())
            st.session_state["pending_eval"]["save_status"] = "dup" if dup else "saved"
            st.session_state["pending_eval"]["save_id"] = mid
        except Exception as e:
            st.session_state["pending_eval"]["save_status"] = "error"
            st.session_state["pending_eval"]["save_error"] = str(e)
    ev = st.session_state.get("pending_eval")
    if ev:
        facts = ev["facts"]; filt = ev["filt"]
        if ev.get("save_status") == "saved":
            st.success(f"✅ Saved to Pending (id {ev.get('save_id')}) — in Supabase king_matches.")
        elif ev.get("save_status") == "dup":
            st.info(f"ℹ️ Already stored (id {ev.get('save_id')}) — not saved again. No duplicate created.")
        elif ev.get("save_status") == "error":
            st.error(f"Save FAILED: {ev.get('save_error')}")
        if not facts.get("home_team"): st.warning("Could not read team names from (A)/(B) lines.")
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        if st.button("Clear", key="clear_new"):
            st.session_state.pop("pending_eval", None)
            st.rerun()
with tab_pending:
    st.markdown("### Pending matches (from king.db)")
    pending = load_matches(pending_only=True)
    if not pending: st.info("No pending matches.")
    for m in pending:
        filt = _filt_from_db_row(m)
        st.markdown(_card_html(m, filt, settled=False), unsafe_allow_html=True)
        with st.expander("Enter result"):
            c1, c2, c3 = st.columns([1,1,2])
            with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['id']}")
            with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['id']}")
            with c3:
                st.write(""); st.write("")
                if st.button("Settle", key=f"s_{m['id']}", type="primary"):
                    settle_match_db(m['id'], int(hs), int(aws)); st.success("Settled."); st.rerun()
with tab_perf:
    st.markdown("### Performance — per-signal only, SKIP excluded")
    settled = load_matches(settled_only=True)
    if not settled: st.info("No settled matches yet.")
    else:
        rows=[]
        for key, label, _, field in FILTER_CATEGORIES:
            n=wins=0
            for m in settled:
                for p in m["picks"]:
                    if p["field"]==field and not p.get("is_implied") and p.get("outcome") is not None:
                        n+=1; wins+=p["outcome"]
            rows.append({"Category": label, "n (both agree)": n, "Wins": wins, "Losses": n-wins, "Hit %": f"{(wins/n*100) if n else 0:.1f}%", "Proven": PROVEN.get(key,"")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        # Main Pick hit rate
        mp=[m for m in settled if m.get("main_pick_key")]
        mp_wins=0; mp_n=0
        for m in mp:
            for p in m["picks"]:
                if p.get("is_main") and p.get("outcome") is not None:
                    mp_n+=1; mp_wins+=p["outcome"]
        st.metric("Settled matches", len(settled))
        st.metric("Main Pick hit rate", f"{mp_wins}/{mp_n} = {(mp_wins/mp_n*100) if mp_n else 0:.1f}%")
        st.markdown("### All settled")
        for m in settled:
            filt=_filt_from_db_row(m)
            st.markdown(_card_html(m, filt, settled=True), unsafe_allow_html=True)
with tab_lab:
    st.markdown("### Lab — test without saving")
    text_in = st.text_area("Text dump", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        facts = parse_filter_data(text_in.strip()); filt = compute_filter_rule(facts)
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        with st.expander("Parsed sections"):
            st.json(facts["sections"])
            for w in facts.get("warnings", []): st.write(f"- {w}")

with tab_sb:
    st.markdown("### Supabase — why is data not saving?")
    st.write(f"**SUPABASE_URL set:** {'YES — ' + (SUPABASE_URL[:35] + '...') if SUPABASE_URL else 'NO ❌ — add it in Streamlit Secrets'}")
    st.write(f"**SUPABASE_KEY set:** {'YES — len ' + str(len(SUPABASE_KEY)) if SUPABASE_KEY else 'NO ❌ — add it in Streamlit Secrets'}")
    st.write(f"**Backend in use:** {'Supabase `king_matches`' if _sb_configured() else 'Local SQLite king.db (NOT Supabase — this is why nothing appears in Supabase)'}")
    st.caption("If Backend says SQLite, your Secrets are not reaching the app. Secrets need a reboot to apply.")
    if st.button("Test Supabase connection", type="primary"):
        if not _sb_configured():
            st.error("Secrets missing. Add SUPABASE_URL + SUPABASE_KEY in Streamlit > Settings > Secrets, then Reboot.")
        else:
            try:
                rows = _sb_req("GET", "king_matches", query="?select=id&limit=1")
                st.success(f"Connected ✅ Table king_matches exists. Rows found: {len(rows)}")
                # Try a test write + delete
                test = _sb_req("POST", "king_matches", {"home_team": "__TEST__", "away_team": "__TEST__", "decision": "TEST", "picks": [], "sections": {}})[0]
                tid = test["id"]
                _sb_req("DELETE", "king_matches", query=f"?id=eq.{tid}")
                st.success(f"Write + delete test PASSED ✅ (test id {tid} created then deleted). Saves should work.")
            except Exception as e:
                st.error(f"FAILED ❌: {e}")
                st.caption("Copy this exact error — it tells us if it's: table missing (404 / PGRST205), RLS blocked (401/403), wrong URL, or wrong key.")
    st.markdown("#### Most common causes")
    st.markdown("- **1. Old king.py still deployed** — you must replace the file on Streamlit and Reboot. If the Supabase tab is missing, you are on the old file.\n- **2. Secrets not set / not rebooted** — Streamlit only reads Secrets on reboot.\n- **3. Table not created** — run supabase_single_table.sql in Supabase SQL Editor. Error will say table not found / PGRST205.\n- **4. Wrong key** — use the `anon public` key from Supabase Settings > API, full string, no quotes, no spaces.\n- **5. RLS blocked** — if the policy `king_all` was not created, anon writes are blocked. Re-run the SQL file.\n- **6. Local SQLite looks like saving** — on Streamlit Cloud, king.db is wiped on reboot, so saves disappear. That is why it looks like 'not saved'.")

