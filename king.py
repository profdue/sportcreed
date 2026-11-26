
"""
king.py — The Filter Rule — 100% Logic
100% Logic (29/29 Main YES): Only bet when Main is YES. Only Main gate counts. Side signals never veto Main.
Rules:
- Only (A) vs League AND (B) vs League. Both beat -> YES. Both below -> NO. Else SKIP.
- Equality (= league) is NEUTRAL, not BELOW. Any side = league -> SKIP for that category.
- Over 3.5 always ignored.
- No scoreline / exact score is ever predicted.
- UNDER callout: an Over 1.5 NO / Over 2.5 NO survivor is shown as its own highlighted ALSO: UNDER line on the card, never buried in Others. Main Pick rule is unchanged.
- ABSOLUTE GATE (Tromso rule): a relative NO only counts if Avg is at/below the LOW ceiling for that category;
  a relative YES only counts if Avg is above LOW. Below league but above LOW (e.g. 86% Over 1.5 vs 87% league)
  is NOT a red - it is a high arena, downgraded to SKIP. This is what filters Tromso-type matches out of BET.
- Weak test: <3 survivors -> NO BET.
- Conflict test: direct survivors that logically contradict -> NO BET - CONFLICT.
- Main Pick: biggest relative margin among survivors. History only breaks ties.
  Margin = avg( (A-Lg)/Lg , (B-Lg)/Lg ), absolute value.
  Tie hierarchy: Away > Over 1.5 > Over 2.5 > BTTS > Home.
- Evaluating does NOT save. Only "Save to Pending" saves.
- Performance settles per-signal only, never scoreline.
Persistence: SQLite (king.db) via stdlib sqlite3.
"""
import sqlite3
import os, json, urllib.request, urllib.error
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "king.db"
PROVEN = {
    "home_scoring_rate": "7/8 (87.5%)",
    "away_scoring_rate": "10/10 (100%)",
    "over_1_5": "9/10 (90%)",
    "over_2_5": "8/9 (88.9%)",
    "btts": "8/9 (88.9%)",
}
FILTER_CATEGORIES = [
    ("home_scoring_rate", "Home Scoring Rate", "Does the home team score?", "home_scores"),
    ("away_scoring_rate", "Away Scoring Rate", "Does the away team score?", "away_scores"),
    ("over_1_5", "Over 1.5 Goals", "Total > 1.5?", "over_1_5"),
    ("over_2_5", "Over 2.5 Goals", "Total > 2.5?", "over_2_5"),
    ("btts", "BTTS", "Do both teams score?", "btts"),
]
HIER = {"away_scoring_rate": 0, "over_1_5": 1, "over_2_5": 2, "btts": 3, "home_scoring_rate": 4}

# --- Absolute gate (the Tromso rule) ---
# A relative call (below/above league) only counts if the ABSOLUTE level (Avg) agrees.
# Below league alone is not enough: 86% Over 1.5 in an 87% league is a HIGH arena, not a red.
# LOW = ceiling for a "true low" arena in that category. A NO only counts if Avg is at/below LOW;
# a YES only counts if Avg is above LOW. Otherwise the pick is downgraded to SKIP.
# Values are per-category because base rates differ: Over 1.5 lives near 80%, Over 2.5 / BTTS near 55%,
# so one universal 70 would call a 65% Over 2.5 "red" when it is actually high.

def _main_passes_100_gate(pick, derived):
    if pick["decision"] != "YES":
        return False
    avg = pick.get("avg"); lg = pick.get("league"); f = pick["field"]
    if f == "home_scores": return True
    if f == "over_1_5": return True
    if f == "over_2_5": return (avg is not None and lg is not None and avg >= 70 and lg >= 50)
    if f == "btts": return (lg is not None and lg >= 60) or (avg is not None and 60 <= avg <= 70)
    if f == "away_scores": return derived.get("btts") is True
    return False

def _main_gate_reason(pick, derived):
    f = pick["field"]; avg = pick.get("avg"); lg = pick.get("league")
    if f == "over_2_5": return f"Over 2.5 YES needs Avg >= 70 AND League >= 50. Got Avg {avg:g}% / League {lg:g}%."
    if f == "btts": return f"BTTS YES needs League >= 60 OR Avg 60-70. Got Avg {avg:g}% / League {lg:g}%."
    if f == "away_scores": return "Away YES needs BTTS also YES."
    return ""

ABSOLUTE_LOW = {
    "home_scoring_rate": 70.0,  # % - no Home-rate league in our 71 ever went below 71
    "away_scoring_rate": 65.0,  # % - 70 is still average for Away rate
    "over_1_5": 70.0,           # % - league median ~79
    "over_2_5": 50.0,           # % - league median ~57, max 68; 70 would be HIGH here
    "btts": 50.0,                # % - league median ~56, max 61; 70 would be HIGH here
}
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

def settle_match_db(match_id, home_score, away_score):
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

# ---------------------------------------------------------------- filter rule
def _decide(a, league, b, avg=None, key=None):
    # Equality is NEUTRAL. Only strict > / strict < counts.
    # ABSOLUTE GATE: a relative call only stands if the absolute level (Avg) agrees.
    #   NO needs Avg at/below LOW (true low arena). YES needs Avg above LOW (not a low arena
    #   beating an even lower league). Otherwise downgraded to SKIP.
    #   Tromso Over 1.5: 86/86 vs 87 -> candidate NO, but Avg 86 > LOW 70 -> SKIP, not NO.
    if a is None or b is None or league is None:
        return "SKIP"
    low = ABSOLUTE_LOW.get(key) if key else None
    if a > league and b > league:
        if low is not None and avg is not None and avg <= low:
            return "SKIP"
        return "YES"
    if a < league and b < league:
        if low is not None and avg is not None and avg > low:
            return "SKIP"
        return "NO"
    return "SKIP"

def _lbl(v, lg):
    return f"{v:g}% BEATS league {lg:g}%" if v > lg else f"{v:g}% BELOW league {lg:g}%" if v < lg else f"{v:g}% = league {lg:g}% (neutral)"

def compute_filter_rule(facts: dict) -> dict:
    picks=[]; derived={}
    for key, label, question, field in FILTER_CATEGORIES:
        sec=facts["sections"].get(key)
        if not sec or sec.get("a") is None or sec.get("b") is None or sec.get("league") is None:
            picks.append({"key":key,"label":label,"question":question,"field":field,"a":None,"b":None,"league":None,"decision":"SKIP","reason":"Section not found in paste."})
            derived[field]=None; continue
        dec=_decide(sec["a"], sec["league"], sec["b"], avg=sec.get("avg"), key=key)
        if dec=="YES":
            derived[field]=True; reason=f"Both BEAT league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> YES"
        elif dec=="NO":
            derived[field]=False; reason=f"Both BELOW league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> NO"
        else:
            derived[field]=None
            # If both sides were on the same side of league but the absolute gate blocked it,
            # say so - otherwise a Tromso-type downgrade looks like an ordinary split.
            _low = ABSOLUTE_LOW.get(key)
            if sec["a"] is not None and sec["b"] is not None and _low is not None and sec.get("avg") is not None:
                if sec["a"] < sec["league"] and sec["b"] < sec["league"] and sec["avg"] > _low:
                    reason=f"Below league but Avg {sec['avg']:g}% above LOW {_low:g}% - high arena, not a true NO -> SKIP"
                elif sec["a"] > sec["league"] and sec["b"] > sec["league"] and sec["avg"] <= _low:
                    reason=f"Above league but Avg {sec['avg']:g}% at/below LOW {_low:g}% - low arena, not a true YES -> SKIP"
                else:
                    reason=f"Split: A {_lbl(sec['a'], sec['league'])} | B {_lbl(sec['b'], sec['league'])} -> SKIP"
            else:
                reason=f"Split: A {_lbl(sec['a'], sec['league'])} | B {_lbl(sec['b'], sec['league'])} -> SKIP"
        picks.append({"key":key,"label":label,"question":question,"field":field,"a":sec["a"],"b":sec["b"],"league":sec["league"],"avg":sec.get("avg"),"decision":dec,"reason":reason})
    # implied (only fills None, never overrides a direct pick)
    implied_notes={}
    if derived.get('btts') is True:
        if derived.get('home_scores') is None: derived['home_scores']=True; implied_notes['home_scores']='via BTTS YES'
        if derived.get('away_scores') is None: derived['away_scores']=True; implied_notes['away_scores']='via BTTS YES'
    if derived.get('btts') is False:
        if derived.get('home_scores') is True and derived.get('away_scores') is None: derived['away_scores']=False; implied_notes['away_scores']='via BTTS NO + Home scores'
        if derived.get('away_scores') is True and derived.get('home_scores') is None: derived['home_scores']=False; implied_notes['home_scores']='via BTTS NO + Away scores'
    if derived.get('home_scores') is False or derived.get('away_scores') is False:
        if derived.get('btts') is None: derived['btts']=False; implied_notes['btts']='via one side NO'
    if derived.get('over_2_5') is True and derived.get('over_1_5') is None: derived['over_1_5']=True; implied_notes['over_1_5']='via Over 2.5 YES'
    if derived.get('over_1_5') is False and derived.get('over_2_5') is None: derived['over_2_5']=False; implied_notes['over_2_5']='via Over 1.5 NO'
    survivors=[x for x in picks if x['decision'] in ("YES","NO")]
    weak=len(survivors) < 3
    # ---- coherence: direct picks that logically contradict ----
    conflicts=[]
    d={p["field"]: p["decision"] for p in picks}  # direct only
    if d.get("btts")=="YES":
        if d.get("home_scores")=="NO": conflicts.append("BTTS YES needs Home scores YES, but Home scores is NO")
        if d.get("away_scores")=="NO": conflicts.append("BTTS YES needs Away scores YES, but Away scores is NO")
        if d.get("over_1_5")=="NO": conflicts.append("BTTS YES needs Over 1.5 YES (total at least 2), but Over 1.5 is NO")
    if d.get("over_1_5")=="NO":
        if d.get("btts")=="YES": conflicts.append("Over 1.5 NO needs BTTS NO (total 0 or 1), but BTTS is YES")
        if d.get("over_2_5")=="YES": conflicts.append("Over 1.5 NO needs Over 2.5 NO, but Over 2.5 is YES")
    if d.get("over_2_5")=="YES" and d.get("over_1_5")=="NO":
        conflicts.append("Over 2.5 YES needs Over 1.5 YES, but Over 1.5 is NO")
    if d.get("home_scores")=="NO" and d.get("away_scores")=="NO" and d.get("btts")=="YES":
        conflicts.append("Home NO + Away NO needs BTTS NO, but BTTS is YES")
    # de-duplicate preserving order
    seen=set(); conflicts=[c for c in conflicts if not (c in seen or seen.add(c))]
    # ---- margin + Main Pick — 100% Logic ----
    for x in survivors:
        lg=x['league'] if x['league'] else 1
        x['margin']=abs(((x['a']-lg)/lg + (x['b']-lg)/lg)/2.0)
        x['margin_pct']=x['margin']*100.0
    main_pick=None; decision="BET"; decision_reason=""
    yes_survivors=[x for x in survivors if x['decision']=="YES"]
    gate_pass=[x for x in yes_survivors if _main_passes_100_gate(x, derived)]
    gate_fail=[x for x in yes_survivors if not _main_passes_100_gate(x, derived)]
    if weak:
        decision="NO BET - WEAK"; decision_reason=f"Only {len(survivors)}/5 signals agree. Not enough for a Main Pick."
    elif conflicts:
        decision="NO BET - CONFLICT"; decision_reason="; ".join(conflicts)
    elif not yes_survivors:
        decision="NO BET - MAIN NO"; decision_reason="No YES signal. Main NO is 24/30 (80.0%) — not a 100% bet."
    elif not gate_pass:
        decision="NO BET - NOT 100%"; decision_reason="Main YES exists but none clears its own 100% gate: " + "; ".join(f"{LABEL.get(x['label'], x['label'])} YES — {_main_gate_reason(x, derived)}" for x in gate_fail)
    else:
        ranked=sorted(gate_pass, key=lambda pp: (-pp['margin'], HIER.get(pp['key'],99)))
        main_pick=ranked[0]
        decision_reason=f"100% Main: {LABEL.get(main_pick['label'], main_pick['label'])} YES. Side signals do not veto Main."
    return {"picks":picks,"derived":derived,"survivors":survivors,"weak_test":weak,
            "conflicts":conflicts,"decision":decision,"decision_reason":decision_reason,
            "main_pick":main_pick,"implied_notes":implied_notes,"gate_pass":gate_pass,"gate_fail":gate_fail}


# ---------------------------------------------------------------- analyst reason
def _analyst_reason(main, picks):
    """Analyst-style reason for the Main. Specific numbers, not generic margin talk."""
    if not main:
        return ""
    a, b, lg, avg = main.get("a"), main.get("b"), main.get("league"), main.get("avg")
    weaker = min(a, b) if a is not None and b is not None else None
    by_field = {x["field"]: x for x in picks}
    def _fmt(v):
        return f"{v:g}%" if v is not None else "-"
    # supporting signals, in analyst language
    support = []
    o15 = by_field.get("over_1_5"); o25 = by_field.get("over_2_5"); btts = by_field.get("btts")
    hs = by_field.get("home_scores"); aws = by_field.get("away_scores")
    if o15 and o15["decision"] == "YES" and main["field"] != "over_1_5":
        support.append(f"the floor is there too — Over 1.5 at {_fmt(o15.get('avg'))} (league {_fmt(o15.get('league'))})")
    if btts and btts["decision"] == "YES" and main["field"] != "btts":
        support.append(f"both sides score — BTTS {_fmt(btts.get('avg'))} in a {_fmt(btts.get('league'))} league")
    if hs and hs["decision"] == "YES" and main["field"] != "home_scores":
        support.append(f"home scores {_fmt(hs.get('avg'))}")
    if aws and aws["decision"] == "YES" and main["field"] != "away_scores":
        support.append(f"away scores {_fmt(aws.get('avg'))}")
    tail = (" Support — " + "; ".join(support) + ".") if support else ""
    f = main["field"]
    if f == "over_2_5":
        return (f"Over 2.5 is the bet because both sides live up there in absolute terms — A {_fmt(a)}, B {_fmt(b)}, Avg {_fmt(avg)} — in a league that itself goes over 2.5 {_fmt(lg)} of the time. "
                f"The weaker side is still {_fmt(weaker)}, so this is not one team carrying the number. This is a high-scoring environment, not a low league beaten on margin.{tail} "
                f"Main YES is 29/29 in this shape.")
    if f == "btts":
        return (f"BTTS is the bet because both attacks are real — A {_fmt(a)}, B {_fmt(b)}, Avg {_fmt(avg)} — against a league that sees both score {_fmt(lg)} of the time. "
                f"The weaker side is {_fmt(weaker)}, so neither attack is a passenger. Both teams have to be kept honest here.{tail} "
                f"Main YES is 29/29 in this shape.")
    if f == "home_scores":
        return (f"Home scores is the bet because the home attack is outright — A {_fmt(a)} against a league rate of {_fmt(lg)}, Avg {_fmt(avg)}. "
                f"This is not a league-inflated number, the home side scores on its own. Home YES in this shape is 26/26.{tail}")
    if f == "away_scores":
        return (f"Away scores is the bet because the away side scores on the road — B {_fmt(b)} against a league rate of {_fmt(lg)}, Avg {_fmt(avg)} — and BTTS is also YES, so the away goal does not need a clean sheet to land. "
                f"The away attack travels.{tail} Main YES is 29/29 in this shape.")
    if f == "over_1_5":
        return (f"Over 1.5 is the bet because the floor is high — A {_fmt(a)}, B {_fmt(b)}, Avg {_fmt(avg)} in a league that clears 1.5 {_fmt(lg)} of the time. "
                f"You only need two goals, and both sides live above the league on that floor. Over 1.5 YES in this shape is 24/24.{tail}")
    return f"Main YES is 29/29 in this shape."

# ---------------------------------------------------------------- card (DB-shaped picks)
def _card_html(m, filt, settled=False):
    # MINIMAL card: fixture + one decision box. Nothing else.
    # Minimal: no section title, no legend line, no detail rows, no skipped line.
    # The table is what made the card too tall. All signals are still computed and still
    # settle in Performance - they are just not printed on the card.
    home, away = m.get('home_team') or '?', m.get('away_team') or '?'
    picks=filt['picks']; main=filt.get('main_pick')
    fixture=f'<div style="font-size:1.15rem;font-weight:700;color:#f1f5f9">{home} <span style="font-size:0.72rem;color:#94a3b8">(HOME)</span> vs {away} <span style="font-size:0.72rem;color:#94a3b8">(AWAY)</span></div>'
    if settled and m.get('home_score') is not None:
        # Settled: fixture + result + Main outcome only. No table.
        _main_outcome=''
        if main:
            for _r in (m.get("picks") or []):
                if _r.get("is_main") and _r.get("outcome") is not None:
                    _main_outcome=f'<div style="font-size:0.88rem;color:#94a3b8;margin-top:4px">Main: {LABEL.get(main["label"], main["label"])} {main["decision"]} \u2014 {"WIN" if _r["outcome"]==1 else "LOSS"}</div>'
                    break
        head=fixture+f'<div style="font-size:1rem;color:#94a3b8;margin-top:6px">Result: {home} {m["home_score"]}-{m["away_score"]} {away}</div>{_main_outcome}'
    else:
        if filt['decision']=="BET":
            ml=LABEL.get(main['label'], main['label']) if main else '-'
            mv='YES' if main['decision']=='YES' else 'NO'
            head=fixture+f'<div style="margin-top:10px;background:#10b98122;border:1px solid #10b981;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#10b981">BET \u2022 100% MAIN</div><div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;margin-top:4px">{ml} \u2014 {mv}</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">Analyst: {_analyst_reason(main, picks)} Margin: {main["margin_pct"]:.1f}% above/below league on average. History only breaks ties.</div></div>'
            # UNDER callout stays: it is a bet, not table clutter. One box, same style.
            _unders=[x for x in filt['picks'] if x['field'] in ('over_1_5','over_2_5') and x['decision']=='NO' and not (main and x['key']==main['key'])]
            _unders.sort(key=lambda x: 0 if x['field']=='over_2_5' else 1)
            if _unders:
                _up=_unders[0]
                _under = 'UNDER 1.5' if _up['field']=='over_1_5' else 'UNDER 2.5'
                _over_lbl = '1.5' if _up['field']=='over_1_5' else '2.5'
                head += f'<div style="margin-top:8px;background:#0ea5e922;border:1px solid #0ea5e9;border-radius:12px;padding:12px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#38bdf8">ALSO: {_under}</div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;margin-top:2px">{_under} \u2014 Over {_over_lbl} NO</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">Margin {_up.get("margin_pct",0):.1f}% \u2014 both sides below league, Avg at/below LOW. Settles as its own WIN/LOSS.</div></div>'
        elif filt['decision'] in ("NO BET - NOT 100%", "NO BET - MAIN NO"):
            _lbl2 = "NOT 100%" if filt['decision']=="NO BET - NOT 100%" else "MAIN NO"
            head=fixture+f'<div style="margin-top:10px;background:#f59e0b22;border:1px solid #f59e0b;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#f59e0b">NO BET \u2022 {_lbl2} \u2022 MONITORING</div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;margin-top:4px">Not a 100% bet — do not bet.</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">{filt.get("decision_reason","")}</div></div>'
        elif filt['decision']=="NO BET - CONFLICT":
            head=fixture+f'<div style="margin-top:10px;background:#f59e0b22;border:1px solid #f59e0b;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#f59e0b">NO BET \u2022 CONFLICT</div><div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;margin-top:4px">Survivors contradict each other. Do not bet.</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">{filt.get("decision_reason","")}</div></div>'
        else:
            head=fixture+f'<div style="margin-top:10px;background:#ef444422;border:1px solid #ef4444;border-radius:12px;padding:14px 16px"><div style="font-size:0.75rem;letter-spacing:0.08em;font-weight:700;color:#ef4444">NO BET \u2022 WEAK</div><div style="font-size:1.25rem;font-weight:800;color:#f1f5f9;margin-top:4px">Weak test - do not bet</div><div style="font-size:0.88rem;color:#94a3b8;margin-top:2px">Only {len(filt["survivors"])}/5 signals agree. Not enough for a Main Pick.</div></div>'
    return f'<div class="match-card">{head}</div>'

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
    derived={}
    for pp in picks:
        derived[pp["field"]] = True if pp["decision"]=="YES" else False if pp["decision"]=="NO" else None
    for r in m.get("picks") or []:
        if r.get("is_implied"):
            implied[r["field"]]=r.get("implied_note") or ""
            # Implied rows were saved with their true YES/NO decision - restore it.
            # Without this, derived is empty and every implied row renders as NO (the ENPPI bug:
            # Home scores NO * via BTTS YES, when BTTS YES must mean Home scores YES).
            derived[r["field"]] = (r.get("decision")=="YES")
    return {"picks":picks,"derived":derived,"survivors":[p for p in picks if p["decision"] in ("YES","NO")],
            "weak_test": m.get("decision")=="NO BET - WEAK","conflicts":[],"decision":m.get("decision") or "BET",
            "decision_reason":m.get("decision_reason") or "","main_pick":main,"implied_notes":implied}


# ---------------------------------------------------------------- Supabase connection test
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

def _supabase_test(url, key):
    """GET king_matches via Supabase REST. Returns (ok, status, detail). Stdlib only."""
    if not url or not key:
        return (False, "NOT CONFIGURED", "Set SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) in env / Streamlit secrets.")
    req_url = f"{url}/rest/v1/king_matches?select=id,home_team,away_team,decision,created_at&order=id.desc&limit=1"
    req = urllib.request.Request(req_url, headers={"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8", "ignore")
            try:
                rows = json.loads(body)
            except Exception:
                rows = []
            return (True, f"CONNECTED — HTTP {resp.status}", f"king_matches reachable. Latest row: {rows[0] if rows else 'table exists, 0 rows'}. Full body: {body[:600]}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")[:600]
        return (False, f"HTTP {e.code}", body or "No body returned.")
    except Exception as e:
        return (False, "CONNECTION FAILED", str(e))

# ---------------------------------------------------------------- UI
st.set_page_config(page_title="King — The Filter Rule", layout="wide")
st.markdown("<style>.match-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:16px 18px;margin:10px 0}</style>", unsafe_allow_html=True)
st.title("⚽ King — The Filter Rule")
st.caption("100% Logic: Only bet when Main is YES (29/29) and Main passes its OWN 100% gate. Over 2.5 Main needs Avg >= 70 & League >= 50. BTTS Main needs League >= 60 or Avg 60-70. Side signals never veto Main. SQLite: king.db")

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab", "🔌 Supabase"])
with tab_new:
    st.markdown("### Paste the Teams scoring vs league average block")
    st.caption("Evaluating does NOT save. Nothing is saved until you press Save to Pending.")
    raw_in = st.text_area("Input", height=380, label_visibility="collapsed", placeholder="Teams scoring vs league average")
    if st.button("Run the Filter", type="primary") and raw_in.strip():
        facts = parse_filter_data(raw_in.strip()); filt = compute_filter_rule(facts)
        # Persist the evaluation in session_state. Streamlit buttons are momentary:
        # on the next rerun "Run" is False again, so a Save button nested inside
        # the Run block never survives to be clicked — that is why Pending stayed empty.
        st.session_state["last_eval"] = {"facts": facts, "filt": filt, "raw": raw_in.strip()}
    if st.session_state.get("last_eval"):
        ev = st.session_state["last_eval"]; facts = ev["facts"]; filt = ev["filt"]
        if not facts.get("home_team"): st.warning("Could not read team names from (A)/(B) lines.")
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        if st.button("Save to Pending", key="save_new"):
            mid = save_match(facts, filt, ev["raw"])
            st.session_state.pop("last_eval", None)
            st.success(f"Saved to Pending (id {mid}) in king.db. Open Pending to see it.")
            st.rerun()
with tab_pending:
    st.markdown("### Pending matches (from king.db)")
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
    # Performance = cards only. No per-signal table, no metrics text, no headings.
    # The table used to sit above the cards, so if it failed the DB cards below it never rendered.
    st.markdown("### Performance")
    settled = load_matches(settled_only=True)
    if not settled:
        st.info("No settled matches yet.")
    else:
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

with tab_supabase:
    st.markdown("### Supabase connection test")
    st.caption("This tab only tests the connection. It does not save, settle, or move matches. SQLite (king.db) is still where this app saves.")
    url, key = _supabase_cfg()
    st.write(f"**URL configured:** {'Yes' if url else 'No'} — `{url or 'not set'}`")
    st.write(f"**Key configured:** {'Yes — hidden' if key else 'No — not set'}")
    st.write(f"**Table tested:** `king_matches`")
    if st.button("Test Supabase connection", type="primary"):
        ok, status, detail = _supabase_test(url, key)
        if ok:
            st.success(status)
        else:
            st.error(status)
        st.code(detail)
        if not url or not key:
            st.info("Set SUPABASE_URL and SUPABASE_KEY in your environment or Streamlit secrets, then press Test again. Example secret names: SUPABASE_URL = https://xyz.supabase.co , SUPABASE_KEY = your anon key.")
