#!/usr/bin/env python3
"""
king.py — 6-Vote Logic (Home/Away only)

For every match, compare both teams on Home/Away stats only.
Give 1 vote to the team that wins each category:

1. More Shots             — higher Avg Shots
2. Higher ON%            — higher % ON Target
3. Lower Blocked          — lower Avg Blocked
4. Higher Scored Avg     — higher Avg Goals Scored
5. Lower Conceded Avg    — lower Avg Goals Conceded
6. Higher Scored%        — higher % Scored In

No other metrics count. Not possession, not pass accuracy, not corners,
not total attacks.

Decision (counted PRE-match, on who has more votes):
  Leader 6-0, 5-1, 4-2  -> TAKE leader Win/Draw
  3-3                   -> LEAN ONLY, NO BET
  Leader 2 or less      -> SKIP (cannot happen with 6 votes, kept for safety)

Counting rule: TAKE wins / TAKE losses only.
3-3 is NOT a win, it is a non-bet.
A game where the 4+ team loses is a LOSS, not a SKIP.
"""
import re
from datetime import datetime, timezone
import streamlit as st
import pandas as pd

st.set_page_config(page_title="King — 6 Votes", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")
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
    .badge-take { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-lean { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .vote-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .vote-pill { font-size: 0.8rem; padding: 5px 11px; border-radius: 8px; background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b; font-family: monospace; }
    .vote-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .vote-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .vote-pill.tie { background: #64748b22; color: #94a3b8; border-color: #64748b55; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 44px; background-color: #1a1f2e; border-radius: 10px; padding: 0 18px; color: #94a3b8; }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Supabase (optional, same table) ----------------
def get_supabase():
    try:
        from supabase import create_client
        try:
            url = st.secrets["supabase"]["url"]; key = st.secrets["supabase"]["key"]
        except (KeyError, TypeError):
            url = st.secrets["SUPABASE_URL"]; key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception:
        return None

def db_insert(row: dict):
    sb = get_supabase()
    if sb is None: return None
    res = sb.table("matches").insert(row).execute(); st.cache_data.clear()
    return res.data[0] if res.data else None

@st.cache_data(ttl=30)
def db_pending():
    sb = get_supabase()
    if sb is None: return []
    try:
        return (sb.table("matches").select("*").eq("vote_decision", "TAKE").is_("home_score", "null").order("extracted_at", desc=True).execute().data or [])
    except Exception:
        return []

@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    if sb is None: return []
    try:
        return (sb.table("matches").select("*").eq("vote_decision", "TAKE").not_.is_("home_score", "null").order("settled_at", desc=True).execute().data or [])
    except Exception:
        return []

def db_find_existing(home, away):
    sb = get_supabase()
    if sb is None: return None
    try:
        res = sb.table("matches").select("*").eq("home_team", home).eq("away_team", away).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

def db_settle(match_id, hs, aws, result, hit, profit):
    sb = get_supabase()
    if sb is None: return
    payload = {"home_score": hs, "away_score": aws, "result": result, "dc_hit": hit, "profit_loss": profit, "settled_at": datetime.now(timezone.utc).isoformat()}
    sb.table("matches").update(payload).eq("match_id", match_id).execute(); st.cache_data.clear()

def db_delete(match_id):
    sb = get_supabase()
    if sb is None: return
    sb.table("matches").delete().eq("match_id", match_id).execute(); st.cache_data.clear()

# ---------------- 6-Vote Core ----------------
def compute_votes(h, a):
    """h, a: dicts with shots, on_pct, blocked, scored_avg, conceded_avg, scored_pct.
    Returns (home_votes, away_votes, breakdown list of (label, winner, h_val, a_val))"""
    tests = [
        ("Shots", h["shots"], a["shots"], "higher"),
        ("ON%", h["on_pct"], a["on_pct"], "higher"),
        ("Blocked", h["blocked"], a["blocked"], "lower"),
        ("Scored Avg", h["scored_avg"], a["scored_avg"], "higher"),
        ("Conceded Avg", h["conceded_avg"], a["conceded_avg"], "lower"),
        ("Scored%", h["scored_pct"], a["scored_pct"], "higher"),
    ]
    hv = av = 0
    breakdown = []
    for label, hv_val, av_val, direction in tests:
        if hv_val is None or av_val is None:
            breakdown.append((label, "MISSING", hv_val, av_val)); continue
        if hv_val == av_val:
            breakdown.append((label, "TIE", hv_val, av_val)); continue
        if direction == "higher":
            w = "HOME" if hv_val > av_val else "AWAY"
        else:
            w = "HOME" if hv_val < av_val else "AWAY"
        if w == "HOME": hv += 1
        else: av += 1
        breakdown.append((label, w, hv_val, av_val))
    return hv, av, breakdown

def decide(hv, av):
    leader = max(hv, av)
    if hv == av:
        return "LEAN", None, "3-3 TIE — LEAN ONLY, NO BET"
    if leader >= 4:
        pick = "HOME" if hv > av else "AWAY"
        return "TAKE", pick, f"{hv}-{av} — TAKE {pick} Win/Draw"
    return "SKIP", None, f"{hv}-{av} — SKIP"

def settle_vote(pick, hs, aws):
    """Win/Draw: hit if pick team did not lose."""
    if pick == "HOME":
        hit = 1 if hs >= aws else 0
    elif pick == "AWAY":
        hit = 1 if aws >= hs else 0
    else:
        return None, 0.0
    # Flat 1u, no odds in this version — P/L is +1 / -1 for hit-rate tracking.
    return hit, (1.0 if hit else -1.0)

def result_str(hs, aws):
    if hs > aws: return "1"
    if hs < aws: return "2"
    return "X"

# ---------------- Parser for Besoccer-style dump ----------------
def _f(x):
    try: return float(str(x).replace(",", ".").strip().rstrip("%"))
    except Exception: return None

def parse_dump(text: str):
    """Best-effort parser for the dump format you paste.
    Extracts, in order, the two teams' blocks:
      Shots Avg, Blocked, ON%, Scored Avg, Conceded Avg, Scored%
    Never invents numbers — if a field is missing, it errors."""
    t = text.replace("\r\n", "\n")
    # Team codes: look for "Played games" section — two 2-4 letter codes
    codes = []
    m = re.search(r"Played games\s*\n+\s*([A-Z]{2,4})\s*\n\s*\d+", t)
    if m: codes.append(m.group(1))
    # Second code: after first block, search for next code before its played number
    # Simpler: find all lines that are 2-4 uppercase letters followed by a number line
    lines = [ln.strip() for ln in t.split("\n")]
    for i, ln in enumerate(lines):
        if re.fullmatch(r"[A-Z]{2,4}", ln) and i+1 < len(lines) and re.fullmatch(r"\d+", lines[i+1]):
            if ln not in codes: codes.append(ln)
    # Goals: Scored Avg / Conceded Avg — two blocks in order
    # Pattern: "Scored\n<number>\nAvg.\nper game\n<number>" appears twice (home then away)
    scored_avgs = re.findall(r"Scored\s*\n\s*[\d.]+\s*\n\s*Avg\.\s*\n\s*per game\s*\n\s*([\d.]+)", t)
    conceded_avgs = re.findall(r"Conceded\s*\n\s*[\d.]+\s*\n\s*Avg\.\s*\n\s*per game\s*\n\s*([\d.]+)", t)
    # Scored% : "Yes\nNo\n<number> (xx%)" — first % after Yes/No is scored%
    scored_pcts = re.findall(r"Yes\s*\n\s*No\s*\n\s*\d+\s*\((\d+)%\)", t)
    # Shots: "Total shots\n<number>\n<avg>" — two blocks
    shots_avgs = re.findall(r"Total shots\s*\n\s*[\d.]+\s*\n\s*([\d.]+)", t)
    # Blocked: "Blocked\n<number>\n<avg>" — two blocks
    blocked_avgs = re.findall(r"Blocked\s*\n\s*[\d.]+\s*\n\s*([\d.]+)", t)
    # ON% : "<x>%ON target" — two blocks
    on_pcts = re.findall(r"(\d+)%ON target", t)
    # Full names: they appear in Total attacks / Dangerous attacks, e.g.
    # "Total attacks\n\nGAZ\nGenclerbirligi\n455\nAvg.\n75.83\nGaziantep\n334..."
    # Codes (GNC/GAZ) are NOT team names — never use them as team names.
    full_names = []
    m_att = re.search(r"Total attacks(.{0,1200})", t, re.S)
    if m_att:
        sec = m_att.group(1)
        # Full name = line of letters/spaces (3+ chars, not all caps) followed by a number line
        for mm in re.finditer(r"\n\s*([A-Z][a-zA-Z\u00C0-\u017F ]{2,40})\s*\n\s*\d+\s*\n\s*Avg\.", sec):
            nm = mm.group(1).strip()
            if nm.upper() not in codes and nm not in full_names and len(nm) >= 3:
                full_names.append(nm)
            if len(full_names) >= 2: break
    # Fallback: Dangerous attacks section has the same two names
    if len(full_names) < 2:
        m_d = re.search(r"Dangerous attacks(.{0,1200})", t, re.S)
        if m_d:
            sec = m_d.group(1)
            for mm in re.finditer(r"\n\s*([A-Z][a-zA-Z\u00C0-\u017F ]{2,40})\s*\n\s*\d+\s*\n\s*Avg\.", sec):
                nm = mm.group(1).strip()
                if nm.upper() not in codes and nm not in full_names and len(nm) >= 3:
                    full_names.append(nm)
                if len(full_names) >= 2: break
    home_name = full_names[0] if len(full_names) > 0 else (codes[0] if len(codes) > 0 else "")
    away_name = full_names[1] if len(full_names) > 1 else (codes[1] if len(codes) > 1 else "")
    out = {"home_code": home_name, "away_code": away_name, "_warnings": []}
    def pick(arr, idx, name):
        if len(arr) > idx: return _f(arr[idx])
        out["_warnings"].append(f"Missing {name} for {'home' if idx==0 else 'away'} — paste the full dump.")
        return None
    out["home"] = {
        "shots": pick(shots_avgs, 0, "Shots"), "on_pct": pick(on_pcts, 0, "ON%"),
        "blocked": pick(blocked_avgs, 0, "Blocked"), "scored_avg": pick(scored_avgs, 0, "Scored Avg"),
        "conceded_avg": pick(conceded_avgs, 0, "Conceded Avg"), "scored_pct": pick(scored_pcts, 0, "Scored%"),
    }
    out["away"] = {
        "shots": pick(shots_avgs, 1, "Shots"), "on_pct": pick(on_pcts, 1, "ON%"),
        "blocked": pick(blocked_avgs, 1, "Blocked"), "scored_avg": pick(scored_avgs, 1, "Scored Avg"),
        "conceded_avg": pick(conceded_avgs, 1, "Conceded Avg"), "scored_pct": pick(scored_pcts, 1, "Scored%"),
    }
    return out

def _card_html(d, settled=False):
    hv, av = d.get("home_votes", 0), d.get("away_votes", 0)
    dec = d.get("vote_decision", "LEAN")
    pick = d.get("vote_pick", "")
    title = f"{d.get('home_team','?')} vs {d.get('away_team','?')}"
    meta = f"{hv}-{av} &nbsp;·&nbsp; {dec}" + (f" &nbsp;·&nbsp; {pick} Win/Draw" if pick else "")
    if dec == "TAKE": badge = f'<span class="pick-badge badge-take">TAKE {pick} Win/Draw</span>'; cls = "pick-home" if pick=="HOME" else "pick-away"
    elif dec == "LEAN": badge = '<span class="pick-badge badge-lean">LEAN ONLY — NO BET</span>'; cls = "skip"
    else: badge = '<span class="pick-badge badge-skip">SKIP</span>'; cls = "skip"
    verdict = ""
    if settled and d.get("home_score") is not None:
        verdict = f'<div style="margin-top:10px;"><span class="pick-badge {"badge-win" if d.get("dc_hit")==1 else "badge-loss"}">Result {d.get("home_score")}-{d.get("away_score")} — {"HIT" if d.get("dc_hit")==1 else "MISS"}</span></div>'
    pills = ""
    for label, w, hval, aval in d.get("breakdown", []):
        cls2 = "home" if w=="HOME" else ("away" if w=="AWAY" else "tie")
        pills += f'<span class="vote-pill {cls2}">{label}: {hval} vs {aval} → {w}</span>'
    return f'<div class="match-card {cls}"><div class="match-title">{title}</div><div class="match-meta">{meta}</div>{badge}{verdict}<div class="vote-row">{pills}</div></div>'

# ---------------- UI ----------------
st.title("King — 6 Votes")
st.caption("Home/Away only. 6 votes. 4+ = TAKE leader Win/Draw. 3-3 = LEAN only. No possession, no pass accuracy, no H2H, no double zero, no UNPROVEN.")

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"])

with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Include the Overall / Home-Away dump — Shots, ON target, Blocked, Goals, Scored a goal.")
    raw_in = st.text_area("Input", height=340, label_visibility="collapsed", placeholder="Paste the match data\n...")
    run = st.button("Run the Logic", type="primary")
    if run and raw_in.strip():
        try:
            parsed = parse_dump(raw_in.strip())
            h, a = parsed["home"], parsed["away"]
            missing = [k for k in h if h[k] is None] + [k for k in a if a[k] is None]
            if missing:
                st.error(f"Parse failed — could not find: {', '.join(sorted(set(missing)))}. Paste the full dump.")
                st.stop()
            home_team = parsed["home_code"] or "HOME"
            away_team = parsed["away_code"] or "AWAY"
            hv, av, br = compute_votes(h, a)
            dec, pick, reason = decide(hv, av)
            d = {"home_team": home_team, "away_team": away_team, "home_votes": hv, "away_votes": av, "vote_decision": dec, "vote_pick": pick or "", "breakdown": br}
            st.markdown("### The Pick")
            st.markdown(_card_html(d), unsafe_allow_html=True)
            if dec == "TAKE":
                st.success(reason)
            elif dec == "LEAN":
                st.info(reason)
            else:
                st.info(reason)
            if parsed.get("_warnings"):
                with st.expander(f"Parser warnings ({len(parsed['_warnings'])})"):
                    for w in parsed["_warnings"]: st.write(f"- {w}")
            if dec == "TAKE":
                row = {"home_team": home_team, "away_team": away_team, "home_votes": hv, "away_votes": av, "vote_decision": dec, "vote_pick": pick, "simple_qualifying": 1, "simple_category": "6-VOTE", "simple_bet": f"{pick} Win/Draw", "extracted_at": datetime.now(timezone.utc).isoformat()}
                existing = db_find_existing(home_team, away_team)
                if existing:
                    st.info(f"Already stored (match_id {existing.get('match_id')}).")
                else:
                    try: db_insert(row); st.success("Saved to Pending.")
                    except Exception as e: st.error(f"Insert failed: {e}")
            else:
                st.info("Not saved — LEAN / SKIP are not bets.")
        except Exception as e:
            st.error(f"Parse failed: {e}")

with tab_pending:
    st.markdown("### Pending — TAKE only")
    pending = db_pending()
    if not pending: st.info("No pending TAKE matches.")
    else:
        for m in pending:
            # Rebuild breakdown from stored votes if breakdown not stored
            d = dict(m); d["breakdown"] = []
            st.markdown(_card_html(d), unsafe_allow_html=True)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        hit, prof = settle_vote(m.get("vote_pick") or m.get("simple_bet","")[:4], int(hs), int(aws))
                        # vote_pick stored as HOME/AWAY; simple_bet fallback
                        pick = m.get("vote_pick") or ("HOME" if "HOME" in str(m.get("simple_bet","")) else "AWAY" if "AWAY" in str(m.get("simple_bet","")) else None)
                        hit, prof = settle_vote(pick, int(hs), int(aws))
                        db_settle(m["match_id"], int(hs), int(aws), result_str(int(hs), int(aws)), hit, prof); st.success("Settled."); st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"]); st.success("Deleted."); st.rerun()

with tab_perf:
    st.markdown("### Performance — TAKE only, counted PRE-match")
    settled = db_settled()
    if not settled: st.info("No settled TAKE matches yet.")
    else:
        bets = [m for m in settled if m.get("dc_hit") is not None]; n = len(bets); wins = sum(1 for m in bets if m.get("dc_hit")==1); net = sum((m.get("profit_loss") or 0.0) for m in bets)
        c1, c2, c3, c4, c5 = st.columns(5); c1.metric("TAKE Bets", n); c2.metric("Wins", wins); c3.metric("Losses", n-wins); c4.metric("Hit %", f"{(wins/n*100) if n else 0:.1f}%"); c5.metric("Net P/L", f"{net:+.2f}u")
        st.caption("3-3 LEAN games are NOT counted here — they are non-bets. A 4+ team that loses is a LOSS, not a SKIP.")
        for m in settled:
            d = dict(m); d["breakdown"] = []
            st.markdown(_card_html(d, settled=True), unsafe_allow_html=True)

with tab_lab:
    st.markdown("### Lab")
    text_in = st.text_area("Input", height=300, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        try:
            p = parse_dump(text_in)
            h, a = p["home"], p["away"]
            if any(v is None for v in list(h.values())+list(a.values())):
                st.warning("Some fields missing — parser could not find them. Paste the full dump.")
            else:
                hv, av, br = compute_votes(h, a)
                dec, pick, reason = decide(hv, av)
                st.markdown(f"**{p['home_code']} vs {p['away_code']} — {hv}-{av} — {reason}**")
                for label, w, hv_val, av_val in br:
                    st.write(f"- {label}: {hv_val} vs {av_val} → {w}")
                if dec == "TAKE": st.success(f"TAKE {pick} Win/Draw")
                else: st.info(reason)
        except Exception as e: st.error(f"Parse failed: {e}")
