
"""
king.py — The Filter Rule
Rules:
- Only (A) vs League AND (B) vs League. Both beat -> YES. Both below -> NO. Else SKIP.
- Equality (= league) is NEUTRAL, not BELOW. Any side = league -> SKIP for that category.
- Over 3.5 always ignored.
- No scoreline / exact score is ever predicted.
- Weak test: <3 survivors -> NO BET.
- Conflict test: direct survivors that logically contradict -> NO BET - CONFLICT.
- Main Pick: biggest relative margin among survivors. History only breaks ties.
  Margin = avg( (A-Lg)/Lg , (B-Lg)/Lg ), absolute value.
  Tie hierarchy: Away > Over 1.5 > Over 2.5 > BTTS > Home.
- BET = Main YES (any margin) OR Main NO margin >=40% (Strong NO). Weak NO -> NO BET - WEAK NO, not counted in Main.
- WEAK = Filter failed (<3 survivors, or Weak NO, or CONFLICT). WEAK is DATA COLLECTION ONLY, never a bet, never counted in BET Main.
- WEAK RESCUE (Average Signal, AFTER Filter fails only; never used when Filter gives BET):
  For each of the 9 Average rows compare Average vs League: Average > League = green, < League = red, = League = NEUTRAL.
  Neutral does NOT break the lean. Only the opposite colour breaks it.
  0 red + 1+ green + rest neutral = ALL_GREEN. 0 green + 1+ red + rest neutral = ALL_RED. Green and red both present = MIXED.
  WEAK + ALL_RED -> Main under WEAK = Under 2.5 (the pick, singular).
  WEAK + ALL_GREEN -> Main under WEAK = Over 2.5 YES, secondary = BTTS YES.
  WEAK + MIXED -> no rescue, stays NO BET - WEAK.
  Rescue lives in its own bucket (Main under WEAK), never mixed with BET Main.
- Order, every time: Filter first. If Filter gives BET, stop - Average is never used. If Filter fails -> WEAK -> check Average -> rescue or no rescue.
- Save: BET auto-saves. Non-rule asks Still save? first.
- Performance settles per-signal only, never scoreline.
Persistence: SQLite (king.db) via stdlib sqlite3.
"""
import re
import sqlite3
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
    if filt.get("weak_main"):
        _wm = filt["weak_main"]
        cur.execute(
            "INSERT INTO picks (match_id,category_key,label,field,a_pct,b_pct,league_pct,avg_pct,decision,margin_pct,is_main,is_implied,implied_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, _wm["key"], _wm["label"], _wm["field"], None, None, None, None, _wm["decision"], 0.0, 2, 0, "WEAK RESCUE - Main under WEAK"))
    if filt.get("weak_secondary"):
        _ws = filt["weak_secondary"]
        cur.execute(
            "INSERT INTO picks (match_id,category_key,label,field,a_pct,b_pct,league_pct,avg_pct,decision,margin_pct,is_main,is_implied,implied_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (mid, _ws["key"], _ws["label"], _ws["field"], None, None, None, None, _ws["decision"], 0.0, 3, 0, "WEAK RESCUE - secondary"))
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
        try:
            secs = {}
            for r in cur.execute("SELECT * FROM raw_sections WHERE match_id=?", (m["id"],)).fetchall():
                secs[r["section_key"]] = {"a": r["a_val"], "b": r["b_val"], "avg": r["avg_val"], "league": r["league_val"]}
            m["sections"] = secs
        except Exception:
            m["sections"] = {}
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
    home_team=re.sub(r"\s+", " ", re.sub(r"\s*\((HOME|AWAY|home|away)\)\s*", " ", home_team or "")).strip(" -") if home_team else home_team
    away_team=re.sub(r"\s+", " ", re.sub(r"\s*\((HOME|AWAY|home|away)\)\s*", " ", away_team or "")).strip(" -") if away_team else away_team
    return {"sections": sections, "home_team": home_team, "away_team": away_team, "warnings": warnings}

# ---------------------------------------------------------------- filter rule
def _decide(a, league, b):
    # Equality is NEUTRAL. Only strict > / strict < counts.
    if a is None or b is None or league is None:
        return "SKIP"
    if a > league and b > league:
        return "YES"
    if a < league and b < league:
        return "NO"
    return "SKIP"

def _lbl(v, lg):
    # Lab must never crash on a missing / non-numeric value: return neutral text instead.
    try:
        fv = float(v); fl = float(lg)
    except Exception:
        return "missing (neutral)"
    return f"{fv:g}% BEATS league {fl:g}%" if fv > fl else f"{fv:g}% BELOW league {fl:g}%" if fv < fl else f"{fv:g}% = league {fl:g}% (neutral)"

def compute_filter_rule(facts: dict) -> dict:
    picks=[]; derived={}
    for key, label, question, field in FILTER_CATEGORIES:
        sec=facts["sections"].get(key)
        if not sec or sec.get("a") is None or sec.get("b") is None or sec.get("league") is None:
            picks.append({"key":key,"label":label,"question":question,"field":field,"a":None,"b":None,"league":None,"decision":"SKIP","reason":"Section not found in paste."})
            derived[field]=None; continue
        dec=_decide(sec["a"], sec["league"], sec["b"])
        if dec=="YES":
            derived[field]=True; reason=f"Both BEAT league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> YES"
        elif dec=="NO":
            derived[field]=False; reason=f"Both BELOW league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> NO"
        else:
            derived[field]=None
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
    # ---- margin + Main Pick ----
    for x in survivors:
        lg=x['league'] if x['league'] else 1
        x['margin']=abs(((x['a']-lg)/lg + (x['b']-lg)/lg)/2.0)
        x['margin_pct']=x['margin']*100.0
    main_pick=None; decision="BET"; decision_reason=""
    if weak:
        decision="NO BET - WEAK"; decision_reason=f"Only {len(survivors)}/5 signals agree. Not enough for a Main Pick."
    elif conflicts:
        decision="NO BET - CONFLICT"; decision_reason="; ".join(conflicts)
    else:
        ranked=sorted(survivors, key=lambda pp: (-pp['margin'], HIER.get(pp['key'],99)))
        main_pick=ranked[0]
        if main_pick["decision"]=="NO" and main_pick["margin_pct"]<40.0:
            decision="NO BET - WEAK NO"; decision_reason="Weak NO - Strong NO only (40%+). Not a bet."
        else:
            decision_reason=f"Main Pick {LABEL.get(main_pick['label'], main_pick['label'])} {main_pick['decision']} at {main_pick['margin_pct']:.1f}% margin."
    _avg_sig = _avg_signal_from_sections(facts.get("sections") or {})[0]
    # WEAK RESCUE: Filter failed -> check Average. If ALL_RED / ALL_GREEN, it counts as MAIN UNDER WEAK.
    weak_main = None; weak_secondary = None
    if decision != "BET":
        if _avg_sig == "ALL_RED":
            weak_main = {"key":"over_2_5","label":"Over 2.5 Goals","field":"over_2_5","decision":"NO","margin_pct":0.0}
            decision_reason += " | WEAK RESCUE: ALL_RED -> Main under WEAK = Under 2.5."
        elif _avg_sig == "ALL_GREEN":
            weak_main = {"key":"over_2_5","label":"Over 2.5 Goals","field":"over_2_5","decision":"YES","margin_pct":0.0}
            weak_secondary = {"key":"btts","label":"BTTS","field":"btts","decision":"YES","margin_pct":0.0}
            decision_reason += " | WEAK RESCUE: ALL_GREEN -> Main under WEAK = Over 2.5 YES, secondary BTTS YES."
    return {"picks":picks,"derived":derived,"survivors":survivors,"weak_test":weak,
            "conflicts":conflicts,"decision":decision,"decision_reason":decision_reason,
            "main_pick":main_pick,"implied_notes":implied_notes,"avg_signal":_avg_sig,"avg_green_count":_avg_signal_from_sections(facts.get("sections") or {})[1],
            "weak_main":weak_main,"weak_secondary":weak_secondary}

# ---------------------------------------------------------------- Average Signal (single canonical definition)
# Neutral (=) does NOT break an Average lean. Only the opposite colour breaks it.
# Talleres rule: 8 red + 1 neutral + 0 green = ALL_RED in substance.
def _avg_signal_from_sections(sections):
    keys=['home_scoring_avg','away_scoring_avg','goals_match','home_scoring_rate','away_scoring_rate','over_1_5','over_2_5','over_3_5','btts']
    stts=[]
    for k in keys:
        s=(sections or {}).get(k)
        if s and s.get('avg') is not None and s.get('league') is not None:
            stts.append('G' if s['avg']>s['league'] else 'R' if s['avg']<s['league'] else '=')
    if stts and 'R' not in stts and 'G' in stts: return 'ALL_GREEN',stts.count('G')
    if stts and 'G' not in stts and 'R' in stts: return 'ALL_RED',0
    return 'MIXED',stts.count('G')

def _row_avg_signal(m):
    return _avg_signal_from_sections(m.get('sections') or {})

def _avg_breakdown(m):
    secs=m.get('sections') or {}
    g=r=e=0; neut=[]
    for k in ['home_scoring_avg','away_scoring_avg','goals_match','home_scoring_rate','away_scoring_rate','over_1_5','over_2_5','over_3_5','btts']:
        s=secs.get(k)
        if not s or s.get('avg') is None or s.get('league') is None: continue
        if s['avg']>s['league']: g+=1
        elif s['avg']<s['league']: r+=1
        else: e+=1; neut.append(k)
    return g,r,e,neut

def _card_html(m, filt, settled=False):
    home, away = m.get('home_team') or '?', m.get('away_team') or '?'
    picks = filt['picks']; main = filt.get('main_pick')
    hs, aws = m.get('home_score'), m.get('away_score')
    is_settled = settled and hs is not None and aws is not None
    total = (hs + aws) if is_settled else None
    outcome = {}
    for r in (m.get('picks') or []):
        if not r.get('is_implied'):
            outcome[r.get('field')] = r.get('outcome')
    # header: one line
    if is_settled:
        head = f'<div style="display:flex;justify-content:space-between;align-items:center"><div style="font-size:1rem;font-weight:800;color:#f1f5f9">{home} <span style="color:#64748b">vs</span> {away}</div><span style="background:#0b1220;border:1px solid #334155;border-radius:999px;padding:2px 10px;font-size:0.85rem;font-weight:800">{hs}-{aws} &middot; {total} goals</span></div>'
    else:
        head = f'<div style="font-size:1rem;font-weight:800;color:#f1f5f9">{home} <span style="color:#64748b">vs</span> {away}</div>'
    dec = filt['decision']
    if dec == 'BET' and main:
        ml = LABEL.get(main['label'], main['label']); mv = 'YES' if main['decision'] == 'YES' else 'NO'
        banner = f'<div style="margin-top:8px;background:#065f46;border-radius:8px;padding:8px 10px;display:flex;justify-content:space-between;align-items:center"><span style="font-size:0.85rem;font-weight:900;color:#fff">BET &middot; MAIN {ml} — {mv}</span><span style="font-size:0.78rem;color:#d1fae5">{main["margin_pct"]:.1f}% margin</span></div>'
    elif dec == 'NO BET - WEAK NO':
        banner = f'<div style="margin-top:8px;background:#78350f;border-radius:8px;padding:8px 10px;font-size:0.85rem;font-weight:800;color:#fef3c7">NO BET — WEAK NO &middot; Strong NO only (40%+)</div>'
    elif dec == 'NO BET - CONFLICT':
        banner = f'<div style="margin-top:8px;background:#78350f;border-radius:8px;padding:8px 10px;font-size:0.85rem;font-weight:800;color:#fef3c7">NO BET — CONFLICT &middot; Survivors contradict</div>'
    else:
        banner = f'<div style="margin-top:8px;background:#7f1d1d;border-radius:8px;padding:8px 10px;font-size:0.85rem;font-weight:800;color:#fecaca">NO BET — WEAK &middot; Only {len(filt["survivors"])}/5 agree</div>'
    # Average display rule: BET -> show NOTHING about Average (Filter passed, Average never used).
    # WEAK + rescue -> show the rescue basis. WEAK + MIXED -> show nothing (no rescue).
    # The old "MONITORING ONLY" strip is removed: it contradicted the rescue strip on the same card.
    # Average display rule: BET -> show NOTHING about Average (Filter passed, Average never used).
    # WEAK + rescue -> show the rescue basis. WEAK + MIXED -> show nothing (no rescue).
    # The old "MONITORING ONLY" strip is removed: it contradicted the rescue strip on the same card.
    _avg = filt.get('avg_signal') or 'MIXED'
    _has_rescue = bool(filt.get('weak_main')) and dec != 'BET'
    if dec == 'BET' or not _has_rescue:
        avg_html = ''
    elif _avg == 'ALL_GREEN':
        _gc = filt.get('avg_green_count')
        avg_html = f'<div style="margin-top:6px;font-size:0.8rem;color:#86efac">Average: ALL GREEN — {str(_gc) + " green, " if _gc is not None else ""}0 red, neutral does not break the lean — basis for WEAK rescue</div>'
    elif _avg == 'ALL_RED':
        avg_html = '<div style="margin-top:6px;font-size:0.8rem;color:#fca5a5">Average: ALL RED — 0 green, neutral does not break the lean — basis for WEAK rescue</div>'
    else:
        avg_html = ''
    # WEAK RESCUE strip: Main under WEAK (counts as Main, separate bucket from BET Main)
    _wm = filt.get('weak_main'); _ws = filt.get('weak_secondary')
    _rescue_html = ''
    if _wm and dec != 'BET':
        _wm_label = LABEL.get(_wm.get('label'), _wm.get('label'))
        _wm_disp = 'Under 2.5' if (_wm.get('field') == 'over_2_5' and _wm.get('decision') == 'NO') else f"{_wm_label} — {_wm.get('decision')}"
        _rescue_html = f'<div style="margin-top:6px;background:#312e81;border:1px solid #6366f1;border-radius:8px;padding:8px 10px;font-size:0.85rem;font-weight:800;color:#e0e7ff">WEAK RESCUE — MAIN UNDER WEAK: {_wm_disp} &nbsp; <span style="font-weight:600;color:#c7d2fe">Counts as Main under WEAK, not BET Main.</span></div>'
        if _ws:
            _ws_label = LABEL.get(_ws.get('label'), _ws.get('label'))
            _rescue_html += f'<div style="font-size:0.78rem;color:#a5b4fc;margin-top:2px">Secondary rescue: {_ws_label} — {_ws.get("decision")}</div>'
    # picks as compact rows: one line each
    rows = ''
    ordered = []
    if main and dec == 'BET': ordered.append(main)
    for p in picks:
        if p['decision'] == 'SKIP': continue
        if main and p['key'] == main['key'] and dec == 'BET': continue
        ordered.append(p)
    for p in ordered:
        is_main_row = (main and p['key'] == main['key'] and dec == 'BET')
        label = LABEL.get(p['label'], p['label'])
        d = p['decision']
        a, b, lg = p.get('a'), p.get('b'), p.get('league')
        if a is not None and b is not None and lg is not None:
            why = f'Both {"beat" if d=="YES" else "below"} league: {a:g}% / {b:g}% vs {lg:g}%'
        else: why = ''
        if is_settled:
            oc = outcome.get(p.get('field'))
            is_bet_card = (dec == 'BET')
            if oc == 1: verdict = '<span style="background:#16a34a;color:#fff;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:900">WIN</span>' if is_bet_card else '<span style="background:#334155;color:#cbd5e1;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:800">hit — research</span>'
            elif oc == 0: verdict = '<span style="background:#dc2626;color:#fff;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:900">LOSS</span>' if is_bet_card else '<span style="background:#334155;color:#cbd5e1;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:800">missed — research</span>'
            else: verdict = ''
            if p.get('field') == 'home_scores': act = f'Home {hs}'
            elif p.get('field') == 'away_scores': act = f'Away {aws}'
            elif p.get('field') in ('over_1_5','over_2_5'): act = f'Total {total}'
            elif p.get('field') == 'btts': act = 'Both scored' if (hs > 0 and aws > 0) else 'Not both'
            else: act = ''
            right = f'{act} {verdict}'
        else:
            right = f'{p.get("margin_pct",0):.1f}%'
        main_tag = ' <span style="color:#f59e0b;font-size:0.68rem;font-weight:900">MAIN</span>' if is_main_row else ''
        rows += f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1e293b"><div><span style="font-weight:700;font-size:0.85rem;color:#f1f5f9">{label}{main_tag}</span> <span style="font-size:0.8rem;color:#94a3b8">{why}</span></div><div style="display:flex;gap:8px;align-items:center;white-space:nowrap"><span style="background:{"#065f46" if d=="YES" else "#7f1d1d"};color:#fff;border-radius:999px;padding:1px 8px;font-size:0.75rem;font-weight:800">{d}</span><span style="font-size:0.8rem;color:#cbd5e1">{right}</span></div></div>'
    # WEAK Main row (settled: hit/missed as Main under WEAK, never WIN/LOSS bet pill confusion - use Main pill)
    if _wm and dec != 'BET':
        _wm_label2 = LABEL.get(_wm.get('label'), _wm.get('label'))
        _wm_disp2 = 'Under 2.5' if (_wm.get('field') == 'over_2_5' and _wm.get('decision') == 'NO') else f"{_wm_label2} — {_wm.get('decision')}"
        if is_settled:
            # outcome from stored picks where is_main==2
            _oc = None
            for _r in (m.get('picks') or []):
                if _r.get('is_main') == 2: _oc = _r.get('outcome')
            if _oc == 1: _vp = '<span style="background:#6366f1;color:#fff;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:900">HIT — WEAK MAIN</span>'
            elif _oc == 0: _vp = '<span style="background:#334155;color:#cbd5e1;border-radius:999px;padding:1px 8px;font-size:0.72rem;font-weight:800">MISSED — WEAK MAIN</span>'
            else: _vp = ''
            rows += f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #1e293b;background:#1e1b4b;border-radius:6px;padding:5px 8px"><div><span style="font-weight:800;font-size:0.85rem;color:#e0e7ff">{_wm_disp2} <span style="color:#a5b4fc;font-size:0.68rem;font-weight:900">WEAK MAIN</span></span> <span style="font-size:0.8rem;color:#a5b4fc">Average rescue — counts as Main under WEAK</span></div><div style="display:flex;gap:8px;align-items:center">{_vp}</div></div>'
        else:
            rows += f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:5px 8px;background:#1e1b4b;border-radius:6px"><div><span style="font-weight:800;font-size:0.85rem;color:#e0e7ff">{_wm_disp2} <span style="color:#a5b4fc;font-size:0.68rem;font-weight:900">WEAK MAIN</span></span></div><div style="font-size:0.8rem;color:#c7d2fe">Needs: {"0-2 goals" if _wm_disp2=="Under 2.5" else "3+ goals"}</div></div>'
    implied = ''
    for field, note in (filt.get('implied_notes') or {}).items():
        v = filt['derived'].get(field)
        implied += f'<div style="font-size:0.76rem;color:#64748b;padding:2px 0">* {FIELD_LABEL.get(field,field)} {"YES" if v else "NO"} — implied by logic ({note}), never counted, never a pick</div>'
    _implied_fields = set((filt.get('implied_notes') or {}).keys())
    skipped = [p['label'] for p in picks if p['decision'] == 'SKIP' and p.get('field') not in _implied_fields]
    skip_html = f'<div style="font-size:0.78rem;color:#64748b;margin-top:4px">Skipped: {", ".join(skipped)}</div>' if skipped else ''
    _note = ''
    if is_settled and dec != 'BET':
        if filt.get('weak_main'):
            _note = '<div style="font-size:0.76rem;color:#64748b;margin-top:4px">WEAK — data collection only. Final score used only to settle the WEAK Main, never predicted.</div>'
        else:
            _note = '<div style="font-size:0.76rem;color:#64748b;margin-top:4px">WEAK — data collection only, no rescue. Final score used only to settle signals, never predicted.</div>'
    return f'<div class="match-card">{head}{banner}{_rescue_html}{avg_html}<div style="margin-top:6px">{rows}</div>{implied}{skip_html}{_note}</div>'


def _filt_from_db_row(m):
    picks=[]
    for r in m.get("picks") or []:
        if r.get("is_implied"): continue
        if r.get("is_main") in (2, 3): continue  # WEAK rescue rows: shown only in the WEAK MAIN row, never as Filter picks
        picks.append({"key":r["category_key"],"label":r["label"],"question":"","field":r["field"],
                      "a":r["a_pct"],"b":r["b_pct"],"league":r["league_pct"],"avg":r["avg_pct"],
                      "decision":r["decision"],"reason":"","margin_pct":r["margin_pct"] or 0,"margin":(r["margin_pct"] or 0)/100.0})
    main=None
    for p in picks:
        if p["key"]==m.get("main_pick_key"): main=p
    implied={}
    for r in m.get("picks") or []:
        if r.get("is_implied"): implied[r["field"]]=r.get("implied_note") or ""
    _wm = None; _ws = None
    for r in m.get("picks") or []:
        if r.get("is_main") == 2: _wm = {"key":r["category_key"],"label":r["label"],"field":r["field"],"decision":r["decision"],"margin_pct":0.0}
        if r.get("is_main") == 3: _ws = {"key":r["category_key"],"label":r["label"],"field":r["field"],"decision":r["decision"],"margin_pct":0.0}
    return {"picks":picks,"derived":{},"survivors":[p for p in picks if p["decision"] in ("YES","NO")],
            "weak_test": m.get("decision")=="NO BET - WEAK","conflicts":[],"decision":m.get("decision") or "BET",
            "decision_reason":m.get("decision_reason") or "","main_pick":main,"weak_main":_wm,"weak_secondary":_ws,"implied_notes":implied,"avg_signal":_row_avg_signal(m)[0],"avg_green_count":_row_avg_signal(m)[1]}

# ---------------------------------------------------------------- UI
st.set_page_config(page_title="King — The Filter Rule", layout="wide")
st.markdown("<style>.match-card{background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:10px 12px;margin:6px 0}</style>", unsafe_allow_html=True)
st.title("⚽ King — The Filter Rule")
st.caption("Only logic: (A) vs League AND (B) vs League. Both beat → YES. Both below → NO. Split or = league → SKIP. Over 3.5 ignored. No exact score predicted. SQLite: king.db")

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"])
with tab_new:
    st.markdown("### Paste the Teams scoring vs league average block")
    st.caption("BET auto-saves. Non-rule asks Still save? first — you may not know beforehand which will meet the rule.")
    raw_in = st.text_area("Input", height=220, label_visibility="collapsed", placeholder="Teams scoring vs league average")
    if st.button("Run the Filter", type="primary") and raw_in.strip():
        facts = parse_filter_data(raw_in.strip()); filt = compute_filter_rule(facts)
        st.session_state["pending_eval"] = {"facts": facts, "filt": filt, "raw": raw_in.strip(), "save_status": None, "save_id": None}
        if filt["decision"] == "BET":
            try:
                mid = save_match(facts, filt, raw_in.strip())
                st.session_state["pending_eval"]["save_status"] = "saved"; st.session_state["pending_eval"]["save_id"] = mid
            except Exception as e:
                st.session_state["pending_eval"]["save_status"] = "error"; st.session_state["pending_eval"]["save_error"] = str(e)
        else:
            st.session_state["pending_eval"]["save_status"] = "needs_confirm"
    ev = st.session_state.get("pending_eval")
    if ev:
        facts = ev["facts"]; filt = ev["filt"]
        if ev.get("save_status") == "saved":
            st.success(f"Saved to Pending (id {ev.get('save_id')}).")
        elif ev.get("save_status") == "error":
            st.error(f"Save FAILED: {ev.get('save_error')}")
        elif ev.get("save_status") == "needs_confirm":
            st.warning(f"This match does NOT meet the rule — {filt.get('decision')}. It will NOT count in Main. Still save for monitoring?")
            _c1, _c2 = st.columns(2)
            with _c1:
                if st.button("Yes, still save", type="primary", key="confirm_save_yes"):
                    try:
                        mid = save_match(facts, filt, ev["raw"])
                        st.session_state["pending_eval"]["save_status"] = "saved"; st.session_state["pending_eval"]["save_id"] = mid; st.rerun()
                    except Exception as e:
                        st.session_state["pending_eval"]["save_status"] = "error"; st.session_state["pending_eval"]["save_error"] = str(e); st.rerun()
            with _c2:
                if st.button("No, don't save", key="confirm_save_no"):
                    st.session_state["pending_eval"]["save_status"] = "skipped"; st.rerun()
        elif ev.get("save_status") == "skipped":
            st.info("Not saved — you chose not to save this non-rule match.")
        if not facts.get("home_team"): st.warning("Could not read team names from (A)/(B) lines.")
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        if st.button("Clear", key="clear_new"):
            st.session_state.pop("pending_eval", None); st.rerun()
with tab_pending:
    st.markdown("### Pending matches (from king.db)")
    pending = load_matches(pending_only=True)
    if not pending: st.info("No pending matches.")
    _pc1, _pc2 = st.columns(2)
    for _i, m in enumerate(pending):
        filt = _filt_from_db_row(m)
        with (_pc1 if _i % 2 == 0 else _pc2):
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
    st.markdown("### Performance — MAIN and WEAK never mixed")
    settled = load_matches(settled_only=True)
    if not settled: st.info("No settled matches yet.")
    else:
        main_settled = [m for m in settled if m.get("decision") == "BET"]
        weak_settled = [m for m in settled if m.get("decision") != "BET"]
        mp_wins = mp_n = 0
        for m in settled:
            for p in m["picks"]:
                if p.get("is_main") == 1 and p.get("outcome") is not None: mp_n += 1; mp_wins += p["outcome"]
        yes_n = yes_w = no_n = no_w = 0
        for m in main_settled:
            for p in m["picks"]:
                if p.get("is_main") == 1 and p.get("outcome") is not None:
                    if m.get("main_pick_decision") == "YES": yes_n += 1; yes_w += p["outcome"]
                    elif m.get("main_pick_decision") == "NO": no_n += 1; no_w += p["outcome"]
        def _mini(title, big, sub, color="#f1f5f9"):
            return f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:10px 12px"><div style="font-size:0.68rem;letter-spacing:0.08em;color:#94a3b8;font-weight:800">{title}</div><div style="font-size:1.3rem;font-weight:900;color:{color};margin-top:2px">{big}</div><div style="font-size:0.78rem;color:#94a3b8">{sub}</div></div>'
        # WEAK Main (Average rescue) - separate bucket, never mixed with BET Main
        _wm_n = _wm_w = 0
        for m in settled:
            for pp in m["picks"]:
                if pp.get("is_main") == 2 and pp.get("outcome") is not None: _wm_n += 1; _wm_w += pp["outcome"]
        _k1, _k2, _k3, _k4 = st.columns(4)
        with _k1: st.markdown(_mini("MAIN PICK", f"{mp_wins}/{mp_n}" if mp_n else "—", f"{(mp_wins/mp_n*100) if mp_n else 0:.1f}% hit · {len(main_settled)} settled", "#4ade80"), unsafe_allow_html=True)
        with _k2: st.markdown(_mini("YES MAIN", f"{yes_w}/{yes_n}" if yes_n else "—", f"{(yes_w/yes_n*100) if yes_n else 0:.1f}% · any margin", "#4ade80"), unsafe_allow_html=True)
        with _k3: st.markdown(_mini("STRONG NO MAIN", f"{no_w}/{no_n}" if no_n else "—", f"{(no_w/no_n*100) if no_n else 0:.1f}% · 40%+", "#fbbf24"), unsafe_allow_html=True)
        with _k4: st.markdown(_mini("SETTLED", f"{len(settled)}", f"MAIN {len(main_settled)} · WEAK {len(weak_settled)}", "#f1f5f9"), unsafe_allow_html=True)
        _w1, _w2 = st.columns(2)
        with _w1: st.markdown(_mini("WEAK MAIN (Average rescue)", f"{_wm_w}/{_wm_n}" if _wm_n else "—", f"{(_wm_w/_wm_n*100) if _wm_n else 0:.1f}% · ALL_RED Under + ALL_GREEN Over", "#a5b4fc"), unsafe_allow_html=True)
        with _w2: st.markdown(_mini("WEAK RESCUE RATE", f"{_wm_n}/{len(weak_settled)}" if weak_settled else "—", "WEAK games that got a rescue Main", "#a5b4fc"), unsafe_allow_html=True)
        st.markdown("#### By category — small cards")
        _cols = st.columns(5)
        for _i, (key, label, _, field) in enumerate(FILTER_CATEGORIES):
            n = wins = 0
            for m in main_settled:
                for p in m["picks"]:
                    if p["field"] == field and not p.get("is_implied") and p.get("outcome") is not None: n += 1; wins += p["outcome"]
            with _cols[_i]: st.markdown(_mini(label.upper(), f"{wins}/{n}" if n else "—", f"{(wins/n*100) if n else 0:.1f}% · MAIN only"), unsafe_allow_html=True)
        st.markdown("#### WEAK Average check — ALL_GREEN / ALL_RED, WEAK matches only (BET matches never used Average)")
        def _avg_out(m, want):
            hs, aws = m.get("home_score"), m.get("away_score")
            if hs is None or aws is None: return None
            if want == "OVER": return 1 if (hs + aws) > 2.5 else 0
            if want == "BTTS": return 1 if (hs > 0 and aws > 0) else 0
            if want == "UNDER": return 1 if (hs + aws) <= 2.5 else 0
            return None
        _m1, _m2, _m3 = st.columns(3)
        for _col, _sig, _want, _avg_lbl in [(_m1, "ALL_GREEN", "OVER", "ALL GREEN → Over"), (_m2, "ALL_GREEN", "BTTS", "ALL GREEN → BTTS"), (_m3, "ALL_RED", "UNDER", "ALL RED → Under")]:
            # WEAK only: BET matches never used Average, so they must never feed Average stats.
            ss = [m for m in settled if m.get("decision") != "BET" and _row_avg_signal(m)[0] == _sig]
            outs = [o for o in [_avg_out(m, _want) for m in ss] if o is not None]
            with _col: st.markdown(_mini(_avg_lbl, f"{sum(outs)}/{len(outs)}" if outs else "—", f"{(sum(outs)/len(outs)*100) if outs else 0:.1f}% · monitoring only"), unsafe_allow_html=True)
        st.markdown("#### All settled")
        _g1, _g2 = st.columns(2)
        for _i, m in enumerate(settled):
            filt = _filt_from_db_row(m)
            with (_g1 if _i % 2 == 0 else _g2):
                st.markdown(_card_html(m, filt, settled=True), unsafe_allow_html=True)
with tab_lab:
    st.markdown("### Lab — test without saving")
    text_in = st.text_area("Text dump", height=180, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        facts = parse_filter_data(text_in.strip()); filt = compute_filter_rule(facts)
        tmp = {"home_team": facts.get("home_team"), "away_team": facts.get("away_team")}
        st.markdown(_card_html(tmp, filt, settled=False), unsafe_allow_html=True)
        with st.expander("Parsed sections"):
            st.json(facts["sections"])
            for w in facts.get("warnings", []): st.write(f"- {w}")
