#!/usr/bin/env python3
"""
King_1.py — The Filter Rule (Both Agree vs League)

ONLY logic in this file:
  For each of 5 categories, compare (A) vs League and (B) vs League separately.
    Both BEAT league  -> Predict YES (event happens)
    Both BELOW league -> Predict NO  (event does NOT happen)
    Split             -> SKIP (coin flip, do not predict)
  Categories: Away Scoring Rate, Over 1.5, Over 2.5, BTTS, Home Scoring Rate
  Over 3.5 is ALWAYS ignored (was 50% even when both agreed).
  Goals averages (Home / Away scoring averages) are ONLY a tie-breaker
  for the scoreline, never a filtered pick.

Proven record when both agree (from conversation testing):
  Away Scoring Rate 10/10, Over 1.5 7/7, Over 2.5 6/6, BTTS 5/5, Home Scoring Rate 7/8
  Total 35/36 = 97%
"""
import re
import math
from datetime import datetime, timezone
import streamlit as st
import pandas as pd

st.set_page_config(page_title="The Filter Rule", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; letter-spacing: -0.02em; }
    .match-card { background: linear-gradient(135deg, #1a1f2e 0%, #232a3d 100%); border-radius: 16px; padding: 22px 26px; margin-bottom: 16px; border-left: 4px solid #6366f1; box-shadow: 0 4px 20px rgba(0,0,0,0.25); }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip { border-left-color: #64748b; opacity: 0.9; }
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge { display: inline-block; padding: 6px 14px; border-radius: 999px; font-weight: 700; font-size: 0.9rem; }
    .badge-yes { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-no { background: #ef444422; color: #fca5a5; border: 1px solid #ef444455; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill { font-size: 0.8rem; padding: 5px 11px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b; font-family: monospace; }
    .signal-pill.yes { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.no { background: #ef444415; color: #fca5a5; border-color: #ef444440; }
    .signal-pill.skip { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 44px; background-color: #1a1f2e; border-radius: 10px; padding: 0 18px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- parsing
def _clean(s: str) -> str:
    return s.strip().strip("\ufeff\u200b\u200c\u200d\u00a0 ")

def _nums_in_section(lines, start_idx, end_idx, want_pct: bool):
    """Return numbers in order in a section. For pct sections, only % numbers.
    For goals sections, only plain numbers (no %)."""
    out = []
    for i in range(start_idx, min(end_idx, len(lines))):
        s = _clean(lines[i])
        if not s:
            continue
        # skip labels that are section headers
        low = s.lower()
        if "league avg" in low or "(a) (b) average" in low or "average" in low and "league" in low:
            continue
        if want_pct:
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", s):
                out.append(float(m.group(1)))
        else:
            # plain number line: e.g. "1.67" or "1.67" alone, ignore lines with % or letters (except pure numbers)
            if "%" in s:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", s):
                out.append(float(s))
    return out

# Section definitions: (key, header keywords, want_pct)
SECTIONS = [
    ("home_scoring_avg", ["home scoring averages"], False),
    ("away_scoring_avg", ["away scoring averages"], False),
    ("total_goals_avg", ["goals scored + conceded", "goals scored + conceded per match"], False),
    ("home_scoring_rate", ["home scoring rate"], True),
    ("away_scoring_rate", ["away scoring rate"], True),
    ("over_1_5", ["over 1.5"], True),
    ("over_2_5", ["over 2.5"], True),
    ("over_3_5", ["over 3.5"], True),
    ("btts", ["both-teams-scored", "both teams scored", "bts"], True),
]

def _find_section(lines, keywords):
    for i, raw in enumerate(lines):
        low = _clean(raw).lower()
        if any(k in low for k in keywords):
            # avoid matching "home scoring averages" when looking for "home scoring rate" etc.
            # require exact intent: if keywords contain "rate", header must contain "rate"
            return i
    return None

def parse_filter_data(text: str) -> dict:
    lines = [_clean(ln) for ln in text.replace("\r\n", "\n").split("\n")]
    # team names from first (A) / (B) lines
    home_team, away_team = None, None
    for raw in lines:
        s = _clean(raw)
        m = re.match(r"\(A\)\s*(.+?)\s+(?:scored|scoring|conceding|conceded)\b", s, re.I)
        if m and not home_team:
            home_team = m.group(1).strip()
        m2 = re.match(r"\(B\)\s*(.+?)\s+(?:scored|scoring|conceding|conceded)\b", s, re.I)
        if m2 and not away_team:
            away_team = m2.group(1).strip()
        if home_team and away_team:
            break
    # also support "Pre-match X vs Y" header
    if not home_team:
        for raw in lines:
            m = re.match(r"^(?:pre-match\s+)?(.+?)\s+vs\.?\s+(.+?)\s*$", _clean(raw), re.I)
            if m:
                home_team, away_team = m.group(1).strip(), m.group(2).strip()
                break
    facts = {"home_team": home_team, "away_team": away_team, "sections": {}, "warnings": []}
    # find section boundaries in order
    # Build list of (key, start) sorted by appearance, using specific matching to avoid overlap
    # We scan sequentially: use header detection per section in order of appearance in text
    # Simpler: for each section, find its header index, then numbers until next header
    header_idxs = []
    for key, kws, want_pct in SECTIONS:
        idx = None
        for i, raw in enumerate(lines):
            low = _clean(raw).lower()
            # precise matching
            if key == "home_scoring_avg" and "home scoring averages" in low:
                idx = i; break
            if key == "away_scoring_avg" and "away scoring averages" in low:
                idx = i; break
            if key == "total_goals_avg" and "goals scored + conceded" in low:
                idx = i; break
            if key == "home_scoring_rate" and "home scoring rate" in low:
                idx = i; break
            if key == "away_scoring_rate" and "away scoring rate" in low:
                idx = i; break
            if key == "over_1_5" and "over 1.5" in low:
                idx = i; break
            if key == "over_2_5" and "over 2.5" in low:
                idx = i; break
            if key == "over_3_5" and "over 3.5" in low:
                idx = i; break
            if key == "btts" and ("both-teams-scored" in low or "both teams" in low or ("bts" in low and "matches" in low)):
                idx = i; break
        if idx is not None:
            header_idxs.append((idx, key, want_pct))
    header_idxs.sort()
    for pos, (idx, key, want_pct) in enumerate(header_idxs):
        end = header_idxs[pos+1][0] if pos+1 < len(header_idxs) else len(lines)
        nums = _nums_in_section(lines, idx+1, end, want_pct)
        # Expected order in pasted data: A, B, (A)(B) average, League
        # Some pastes put League label before League value, and average label before average value.
        # _nums_in_section already skips label lines, so nums should be [A, B, avg, league]
        if len(nums) >= 4:
            a, b, avg, league = nums[0], nums[1], nums[2], nums[3]
            facts["sections"][key] = {"a": a, "b": b, "avg": avg, "league": league}
        elif len(nums) >= 2:
            facts["sections"][key] = {"a": nums[0], "b": nums[1], "avg": None, "league": None}
            facts["warnings"].append(f"{key}: only {len(nums)} numbers found, need A, B, avg, League.")
        else:
            facts["warnings"].append(f"{key}: not found / no numbers.")
    return facts

# ---------------------------------------------------------------- filter rule
# Only these 5 are used. Over 3.5 is ALWAYS ignored.
FILTER_CATEGORIES = [
    ("away_scoring_rate", "Away Scoring Rate", "Away team scores?", "away_scores"),
    ("over_1_5", "Over 1.5 Goals", "Total > 1.5?", "over_1_5"),
    ("over_2_5", "Over 2.5 Goals", "Total > 2.5?", "over_2_5"),
    ("btts", "BTTS", "Both teams score?", "btts"),
    ("home_scoring_rate", "Home Scoring Rate", "Home team scores?", "home_scores"),
]
# Proven hit rates when both agree (from conversation testing)
PROVEN = {"away_scoring_rate": "10/10", "over_1_5": "7/7", "over_2_5": "6/6", "btts": "5/5", "home_scoring_rate": "7/8"}

def _decide(a, league, b):
    """Return (decision, a_beats, b_beats) where decision in YES/NO/SKIP"""
    if a is None or b is None or league is None:
        return "SKIP", None, None
    # equality -> SKIP (neutral, not a beat, not below)
    if a == league or b == league:
        return "SKIP", (a > league), (b > league)
    a_beats = a > league
    b_beats = b > league
    if a_beats and b_beats:
        return "YES", True, True
    if (not a_beats) and (not b_beats):
        return "NO", False, False
    return "SKIP", a_beats, b_beats

def compute_filter_rule(facts: dict) -> dict:
    picks = []
    derived = {}  # home_scores, away_scores, over_1_5, over_2_5, btts -> True/False/None
    for key, label, question, field in FILTER_CATEGORIES:
        sec = facts["sections"].get(key)
        if not sec:
            picks.append({"key": key, "label": label, "question": question, "field": field, "a": None, "b": None, "league": None, "decision": "SKIP", "reason": "Section not found in paste."})
            derived[field] = None
            continue
        dec, a_beats, b_beats = _decide(sec["a"], sec["league"], sec["b"])
        if dec == "YES":
            derived[field] = True
            reason = f"Both BEAT league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> YES"
        elif dec == "NO":
            derived[field] = False
            reason = f"Both BELOW league ({sec['a']:g}% vs {sec['league']:g}% and {sec['b']:g}% vs {sec['league']:g}%) -> NO"
        else:
            derived[field] = None
            def _lbl(v, lg): return f"{v:g}% BEATS league {lg:g}%" if v > lg else f"{v:g}% BELOW league {lg:g}%" if v < lg else f"{v:g}% = league {lg:g}% (neutral)"
            a_txt = _lbl(sec['a'], sec['league']); b_txt = _lbl(sec['b'], sec['league'])
            reason = f"Split: A {a_txt} | B {b_txt} -> SKIP"
        picks.append({"key": key, "label": label, "question": question, "field": field, "a": sec["a"], "b": sec["b"], "league": sec["league"], "avg": sec.get("avg"), "decision": dec, "reason": reason})
    survivors = [p for p in picks if p["decision"] in ("YES", "NO")]
    weak = len(survivors) < 3
    scoreline, scoreline_reason = predict_scoreline(facts, derived)
    return {"picks": picks, "derived": derived, "survivors": survivors, "weak_test": weak, "scoreline": scoreline, "scoreline_reason": scoreline_reason}

def predict_scoreline(facts: dict, derived: dict):
    # Expected goals from goals averages (tie-breaker ONLY)
    h_avg = facts["sections"].get("home_scoring_avg", {}).get("avg")
    a_avg = facts["sections"].get("away_scoring_avg", {}).get("avg")
    # Fallback: use A values if avg missing
    if h_avg is None:
        h_avg = facts["sections"].get("home_scoring_avg", {}).get("a")
    if a_avg is None:
        a_avg = facts["sections"].get("away_scoring_avg", {}).get("a")
    h_exp = h_avg if h_avg is not None else 1.0
    a_exp = a_avg if a_avg is not None else 1.0
    # Start from rounded expected goals
    h = int(round(h_exp)); a = int(round(a_exp))
    h = max(0, h); a = max(0, a)
    def satisfies(hh, aa):
        if derived.get("home_scores") is True and hh == 0: return False
        if derived.get("home_scores") is False and hh != 0: return False
        if derived.get("away_scores") is True and aa == 0: return False
        if derived.get("away_scores") is False and aa != 0: return False
        if derived.get("btts") is True and (hh == 0 or aa == 0): return False
        if derived.get("btts") is False and (hh > 0 and aa > 0): return False
        if derived.get("over_1_5") is True and (hh + aa) <= 1: return False
        if derived.get("over_1_5") is False and (hh + aa) > 1: return False
        if derived.get("over_2_5") is True and (hh + aa) <= 2: return False
        if derived.get("over_2_5") is False and (hh + aa) > 2: return False
        return True
    # Search nearby scorelines (0..5 each) closest to expected goals that satisfy filter
    best = None; best_dist = 1e9
    for hh in range(0, 6):
        for aa in range(0, 6):
            if not satisfies(hh, aa):
                continue
            dist = abs(hh - h_exp) + abs(aa - a_exp)
            # slight preference for lower total when tied (more common)
            if dist < best_dist:
                best_dist = dist; best = (hh, aa)
    if best is None:
        # No scoreline satisfies all (e.g. contradictory derived) -> fall back to expected, note conflict
        return (h, a), f"Filter constraints conflict (no scoreline satisfies all survivors). Fallback to expected goals {h_exp:g}-{a_exp:g}. Treat as weak test."
    hh, aa = best
    return (hh, aa), f"Closest to expected goals {h_exp:g}-{a_exp:g} that satisfies all surviving picks."

# ---------------------------------------------------------------- settling (filter only)
def settle_filter(match: dict, home_score: int, away_score: int):
    """Settle each surviving pick. Returns per-pick hit (1/0/None for SKIP)."""
    results = {}
    for p in match["filter"]["picks"]:
        field = p["field"]; dec = p["decision"]
        if dec == "SKIP":
            results[field] = None; continue
        if field == "home_scores":
            actual = home_score > 0
        elif field == "away_scores":
            actual = away_score > 0
        elif field == "over_1_5":
            actual = (home_score + away_score) > 1.5
        elif field == "over_2_5":
            actual = (home_score + away_score) > 2.5
        elif field == "btts":
            actual = (home_score > 0 and away_score > 0)
        else:
            actual = None
        if actual is None:
            results[field] = None
        else:
            want = (dec == "YES")
            results[field] = 1 if (actual == want) else 0
    # scoreline hit
    pred = match["filter"]["scoreline"]
    results["_scoreline_hit"] = 1 if (pred[0] == home_score and pred[1] == away_score) else 0
    # result (win/draw) hit: compare predicted winner vs actual winner
    def winner(h, a):
        return "home" if h > a else ("away" if a > h else "draw")
    results["_result_hit"] = 1 if winner(pred[0], pred[1]) == winner(home_score, away_score) else 0
    return results

# ---------------------------------------------------------------- storage (session_state, no external DB)
if "matches" not in st.session_state:
    st.session_state.matches = []  # each: {id, home, away, filter, home_score, away_score, settled_at, per_pick}
if "next_id" not in st.session_state:
    st.session_state.next_id = 1

def _action_line(m):
    f = m['filter']; d = f['derived']; pred = f['scoreline']
    home, away = m.get('home_team') or 'Home', m.get('away_team') or 'Away'
    h, a = pred
    result = f"{home} WIN" if h > a else (f"{away} WIN" if a > h else 'DRAW')
    headline = f"PREDICT: {home} {h}-{a} {away}  |  {result}"
    bits = []
    if d.get('home_scores') is True: bits.append(f"{home} SCORES")
    elif d.get('home_scores') is False: bits.append(f"{home} NO GOAL")
    if d.get('away_scores') is True: bits.append(f"{away} SCORES")
    elif d.get('away_scores') is False: bits.append(f"{away} NO GOAL")
    if d.get('btts') is True: bits.append('BTTS YES')
    elif d.get('btts') is False: bits.append('BTTS NO')
    if d.get('over_1_5') is True: bits.append('OVER 1.5')
    elif d.get('over_1_5') is False: bits.append('UNDER 1.5')
    if d.get('over_2_5') is True: bits.append('OVER 2.5')
    elif d.get('over_2_5') is False: bits.append('UNDER 2.5')
    sub = '  |  '.join(bits) if bits else 'No surviving picks - no bet'
    h_pred, a_pred = pred
    implied = []
    if h_pred > 0 and d.get('home_scores') is None: implied.append(f"{home} scores (implied by scoreline, NOT a filtered pick)")
    if a_pred > 0 and d.get('away_scores') is None: implied.append(f"{away} scores (implied by scoreline, NOT a filtered pick)")
    if h_pred == 0 and d.get('home_scores') is None: implied.append(f"{home} no goal (implied by scoreline, NOT a filtered pick)")
    if a_pred == 0 and d.get('away_scores') is None: implied.append(f"{away} no goal (implied by scoreline, NOT a filtered pick)")
    if implied: sub = sub + '<br><span style="color:#94a3b8;font-weight:500">Scoreline adds: ' + '  |  '.join(implied) + '</span>'
    return headline, sub

def _card_html(m, settled=False):
    home, away = m.get('home_team') or '?', m.get('away_team') or '?'
    f = m['filter']; picks = f['picks']; pred = f['scoreline']
    headline, sub = _action_line(m)
    fixture = '<div style="font-size:1.15rem;font-weight:700;color:#f1f5f9">' + home + ' <span style="font-size:0.72rem;color:#38bdf8;background:#38bdf822;border:1px solid #38bdf855;padding:2px 7px;border-radius:999px">HOME</span>  vs  ' + away + ' <span style="font-size:0.72rem;color:#fbbf24;background:#f59e0b22;border:1px solid #f59e0b55;padding:2px 7px;border-radius:999px">AWAY</span></div>'
    fixture_sub = '<div style="font-size:0.82rem;color:#94a3b8;margin-top:2px">Fixture: ' + home + ' (home) vs ' + away + ' (away)</div>'
    if settled and m.get('home_score') is not None:
        top = fixture + fixture_sub + '<div style="font-size:1.35rem;font-weight:800;color:#f1f5f9;margin-top:10px">' + f"{home} {m['home_score']}-{m['away_score']} {away}" + '</div><div style="font-size:0.95rem;color:#94a3b8">PREDICTED: ' + f"{pred[0]}-{pred[1]} - {headline.replace('PREDICT: ', '')}" + '</div>'
    else:
        top = fixture + fixture_sub + '<div style="font-size:1.45rem;font-weight:800;color:#f1f5f9;margin-top:10px">' + headline + '</div><div style="font-size:1rem;font-weight:600;color:#34d399;margin-top:6px">' + sub + '</div>'
    weak = '<div style="margin-top:8px"><span class="signal-pill" style="background:#f59e0b22;color:#fbbf24">WEAK TEST - ' + str(len(f['survivors'])) + '/5 survivors</span></div>' if f['weak_test'] else '<div style="margin-top:8px"><span class="signal-pill yes">' + str(len(f['survivors'])) + '/5 survivors</span></div>'
    take_rows = ''
    for p in picks:
        if p['decision'] == 'SKIP': continue
        cls = 'yes' if p['decision'] == 'YES' else 'no'
        take_rows += '<span class="signal-pill ' + cls + '">TAKE ' + p['label'] + ': ' + p['decision'] + ' - A ' + format(p['a'], 'g') + '% vs Lg ' + format(p['league'], 'g') + '% | B ' + format(p['b'], 'g') + '% vs Lg ' + format(p['league'], 'g') + '%</span>'
    if not take_rows: take_rows = '<span class="signal-pill skip">No surviving picks</span>'
    skip_rows = ''
    for p in picks:
        if p['decision'] != 'SKIP': continue
        skip_rows += '<span class="signal-pill skip">SKIP ' + p['label'] + ': ' + p['reason'] + '</span>'
    verdict = ''
    if settled and m.get('per_pick') is not None:
        pp = m['per_pick']
        verdict = '<div class="signal-row"><span class="signal-pill">Scoreline: ' + ('WIN' if pp.get('_scoreline_hit')==1 else 'LOSS') + '</span><span class="signal-pill">Result: ' + ('WIN' if pp.get('_result_hit')==1 else 'LOSS') + '</span></div>'
    return '<div class="match-card">' + top + weak + '<div style="font-size:0.78rem;color:#94a3b8;margin-top:14px;font-weight:700">TAKE - WHY</div><div class="signal-row">' + take_rows + '</div><div style="font-size:0.78rem;color:#64748b;margin-top:12px;font-weight:700">SKIPPED - SPLIT, DO NOT BET</div><div class="signal-row">' + (skip_rows if skip_rows else '<span class="signal-pill skip">None skipped</span>') + '</div><div style="font-size:0.82rem;color:#94a3b8;margin-top:10px">' + f['scoreline_reason'] + '</div>' + verdict + '</div>'

# ---------------------------------------------------------------- UI
st.title("⚽ The Filter Rule — Both Agree vs League")
st.caption("Only logic: (A) vs League AND (B) vs League. Both beat → YES. Both below → NO. Split → SKIP. Over 3.5 always ignored. Proven when both agree: 35/36 = 97%.")

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"])

with tab_new:
    st.markdown('### Paste the Teams scoring vs league average block')
    st.caption('Evaluating does NOT save. Nothing is saved until you press Save to Pending.')
    raw_in = st.text_area('Input', height=380, label_visibility='collapsed', placeholder='Teams scoring vs league average')
    run = st.button('Run the Filter', type='primary')
    if run and raw_in.strip():
        facts = parse_filter_data(raw_in.strip())
        filt = compute_filter_rule(facts)
        if not facts.get('home_team'): st.warning('Could not read team names from (A)/(B) lines.')
        tmp = {'id': 0, 'home_team': facts.get('home_team'), 'away_team': facts.get('away_team'), 'filter': filt, 'home_score': None, 'away_score': None, 'per_pick': None}
        st.markdown(_card_html(tmp, settled=False), unsafe_allow_html=True)
        if st.button('Save to Pending', key='save_new'):
            mid = st.session_state.next_id; st.session_state.next_id += 1
            m = {'id': mid, 'home_team': facts.get('home_team'), 'away_team': facts.get('away_team'), 'filter': filt, 'home_score': None, 'away_score': None, 'settled_at': None, 'per_pick': None, 'created_at': datetime.now(timezone.utc).isoformat()}
            st.session_state.matches.append(m)
            st.success(f'Saved to Pending (id {mid}).')

with tab_pending:
    st.markdown("### Pending matches")
    pending = [m for m in st.session_state.matches if m.get("home_score") is None]
    if not pending:
        st.info("No pending matches.")
    else:
        for m in pending:
            st.markdown(_card_html(m, settled=False), unsafe_allow_html=True)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['id']}")
                with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['id']}", type="primary"):
                        m["home_score"] = int(hs); m["away_score"] = int(aws)
                        m["per_pick"] = settle_filter(m, int(hs), int(aws))
                        m["settled_at"] = datetime.now(timezone.utc).isoformat()
                        st.success("Settled."); st.rerun()
                if st.button("Delete match", key=f"d_{m['id']}"):
                    st.session_state.matches = [x for x in st.session_state.matches if x["id"] != m["id"]]
                    st.success("Deleted."); st.rerun()

with tab_perf:
    st.markdown("### Performance — Filter picks only (both-agree picks, SKIP excluded)")
    settled = [m for m in st.session_state.matches if m.get("home_score") is not None]
    if not settled:
        st.info("No settled matches yet.")
    else:
        # per-category stats
        rows = []
        for key, label, _, field in FILTER_CATEGORIES:
            n = 0; wins = 0
            for m in settled:
                v = (m.get("per_pick") or {}).get(field)
                if v is not None:
                    n += 1; wins += v
            rows.append({"Category": label, "n (both agree)": n, "Wins": wins, "Losses": n-wins, "Hit %": f"{(wins/n*100) if n else 0:.1f}%", "Proven": PROVEN.get(key, "")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        # scoreline / result
        ns = len(settled); sc_wins = sum(1 for m in settled if (m.get("per_pick") or {}).get("_scoreline_hit")==1)
        res_wins = sum(1 for m in settled if (m.get("per_pick") or {}).get("_result_hit")==1)
        c1, c2, c3 = st.columns(3); c1.metric("Settled matches", ns); c2.metric("Exact scoreline", f"{sc_wins}/{ns}"); c3.metric("Result (W/D)", f"{res_wins}/{ns}")
        st.markdown("### All settled")
        for m in settled:
            st.markdown(_card_html(m, settled=True), unsafe_allow_html=True)

with tab_lab:
    st.markdown("### 🧪 Lab — test without saving")
    text_in = st.text_area("Text dump", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        facts = parse_filter_data(text_in.strip()); filt = compute_filter_rule(facts)
        df = pd.DataFrame([{"Category": p["label"], "A": p["a"], "B": p["b"], "League": p["league"], "Decision": p["decision"], "Reason": p["reason"]} for p in filt["picks"]])
        st.dataframe(df, use_container_width=True, hide_index=True)
        h, a = filt["scoreline"]
        st.success(f"Predicted: {facts.get('home_team') or 'Home'} {h}-{a} {facts.get('away_team') or 'Away'} — {filt['scoreline_reason']}")
        if filt["weak_test"]:
            st.warning(f"WEAK TEST — only {len(filt['survivors'])}/5 survivors.")
        with st.expander("Parsed sections"):
            st.json(facts["sections"])
            if facts.get("warnings"):
                for w in facts["warnings"]:
                    st.write(f"- {w}")
