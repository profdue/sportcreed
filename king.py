#!/usr/bin/env python3
"""
king.py — 6-Vote Logic, ALL DATA STORED
Paste-only. No manual inputs.

6 votes (Home/Away only):
 1 More Shots, 2 Higher ON%, 3 Lower Blocked, 4 Higher Scored Avg,
 5 Lower Conceded Avg, 6 Higher Scored%
Decision PRE-match:
  Leader 6-0,5-1,4-2 -> TAKE leader Win/Draw
  3-3 -> LEAN ONLY, NO BET
  Leader <=2 -> SKIP

STORAGE: ALL decisions saved (TAKE + LEAN + SKIP), not just TAKE.
ALL data saved: raw_dump (full paste) + all_data (JSON of every parsed field)
+ individual columns for every metric, so nothing is thrown away.
"""
import re, json
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
        return (sb.table("matches").select("*").is_("home_score", "null").order("extracted_at", desc=True).execute().data or [])
    except Exception: return []

@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    if sb is None: return []
    try:
        return (sb.table("matches").select("*").not_.is_("home_score", "null").order("settled_at", desc=True).execute().data or [])
    except Exception: return []

def db_find_existing(home, away):
    sb = get_supabase()
    if sb is None: return None
    try:
        res = sb.table("matches").select("*").eq("home_team", home).eq("away_team", away).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception: return None

def db_settle(match_id, hs, aws, result, hit, profit):
    sb = get_supabase()
    if sb is None: return
    payload = {"home_score": hs, "away_score": aws, "result": result, "dc_hit": hit, "profit_loss": profit, "settled_at": datetime.now(timezone.utc).isoformat()}
    sb.table("matches").update(payload).eq("match_id", match_id).execute(); st.cache_data.clear()

def db_delete(match_id):
    sb = get_supabase()
    if sb is None: return
    sb.table("matches").delete().eq("match_id", match_id).execute(); st.cache_data.clear()

# ---------- 6-Vote Core ----------
def compute_votes(h, a):
    tests = [
        ("Shots", h["shots"], a["shots"], "higher"),
        ("ON%", h["on_pct"], a["on_pct"], "higher"),
        ("Blocked", h["blocked"], a["blocked"], "lower"),
        ("Scored Avg", h["scored_avg"], a["scored_avg"], "higher"),
        ("Conceded Avg", h["conceded_avg"], a["conceded_avg"], "lower"),
        ("Scored%", h["scored_pct"], a["scored_pct"], "higher"),
    ]
    hv = av = 0; breakdown = []
    for label, hv_val, av_val, direction in tests:
        if hv_val is None or av_val is None:
            breakdown.append((label, "MISSING", hv_val, av_val)); continue
        if hv_val == av_val:
            breakdown.append((label, "TIE", hv_val, av_val)); continue
        w = "HOME" if (hv_val > av_val if direction=="higher" else hv_val < av_val) else "AWAY"
        if w=="HOME": hv+=1
        else: av+=1
        breakdown.append((label, w, hv_val, av_val))
    return hv, av, breakdown

def decide(hv, av):
    if hv==av: return "LEAN", None, "3-3 TIE — LEAN ONLY, NO BET"
    if max(hv,av) >= 4:
        pick = "HOME" if hv>av else "AWAY"
        return "TAKE", pick, f"{hv}-{av} — TAKE {pick} Win/Draw"
    return "SKIP", None, f"{hv}-{av} — SKIP"

def settle_vote(pick, hs, aws):
    if pick=="HOME": hit = 1 if hs>=aws else 0
    elif pick=="AWAY": hit = 1 if aws>=hs else 0
    else: return None, 0.0
    return hit, (1.0 if hit else -1.0)

def result_str(hs, aws):
    return "1" if hs>aws else ("2" if hs<aws else "X")

# ---------- Full-Data Parser ----------
def parse_dump(text: str):
    """Parses EVERYTHING in the paste. Never invents.
    Returns home/away dicts with all fields + raw_dump.
    Missing fields stay None — raw_dump preserves the full paste."""
    t = text.replace("\r\n","\n")
    lines = [ln.strip() for ln in t.split("\n")]
    codes=[]
    for i,ln in enumerate(lines):
        if re.fullmatch(r"[A-Z]{2,4}", ln) and i+1<len(lines) and re.fullmatch(r"\d+", lines[i+1]):
            if ln not in codes: codes.append(ln)
    # Full names from Total attacks / Dangerous attacks — never codes as names
    full_names=[]
    for sec_name in ["Total attacks","Dangerous attacks"]:
        m = re.search(sec_name+r"(.{0,1500})", t, re.S)
        if m:
            sec=m.group(1)
            for mm in re.finditer(r"\n\s*([A-Z][a-zA-Z\u00C0-\u017F ]{2,40})\s*\n\s*\d+\s*\n\s*Avg\.", sec):
                nm=mm.group(1).strip()
                if nm.upper() not in codes and nm not in full_names and len(nm)>=3:
                    full_names.append(nm)
                if len(full_names)>=2: break
        if len(full_names)>=2: break
    home_name = full_names[0] if len(full_names)>0 else (codes[0] if len(codes)>0 else "")
    away_name = full_names[1] if len(full_names)>1 else (codes[1] if len(codes)>1 else "")
    def _f(x):
        try: return float(str(x).replace(",",".").strip().rstrip("%"))
        except Exception: return None
    def pick(arr, idx): return _f(arr[idx]) if len(arr)>idx else None
    played = re.findall(r"Played games\s*\n+\s*[A-Z]{2,4}\s*\n\s*(\d+)", t)
    scored_tot = re.findall(r"Scored\s*\n\s*([\d.]+)\s*\n\s*Avg\.", t)
    scored_avgs = re.findall(r"Scored\s*\n\s*[\d.]+\s*\n\s*Avg\.\s*\n\s*per game\s*\n\s*([\d.]+)", t)
    conceded_tot = re.findall(r"Conceded\s*\n\s*([\d.]+)\s*\n\s*Avg\.", t)
    conceded_avgs = re.findall(r"Conceded\s*\n\s*[\d.]+\s*\n\s*Avg\.\s*\n\s*per game\s*\n\s*([\d.]+)", t)
    scored_pcts = re.findall(r"Yes\s*\n\s*No\s*\n\s*\d+\s*\((\d+)%\)", t)
    shots_avgs = re.findall(r"Total shots\s*\n\s*[\d.]+\s*\n\s*([\d.]+)", t)
    shots_tot = re.findall(r"Total shots\s*\n\s*([\d.]+)", t)
    blocked_avgs = re.findall(r"Blocked\s*\n\s*[\d.]+\s*\n\s*([\d.]+)", t)
    blocked_tot = re.findall(r"Blocked\s*\n\s*([\d.]+)", t)
    on_pcts = re.findall(r"(\d+)%ON target", t)
    possessions = re.findall(r"Ball Possession\s*\n\s*(\d+)%", t)
    pass_accs = re.findall(r"Accurate\s*\n\s*\d+\s*\n\s*(\d+)%", t)
    def _attack_avgs(section):
        m=re.search(section+r"(.{0,1500})", t, re.S)
        if not m: return []
        return re.findall(r"[A-Z][a-zA-Z ]+\s*\n\s*\d+\s*\n\s*Avg\.\s*\n\s*([\d.]+)", m.group(1))
    total_attacks=_attack_avgs("Total attacks")
    dangerous_attacks=_attack_avgs("Dangerous attacks")
    btts = re.findall(r"Both scored.*?Yes\s*\n\s*\d+\s*\n\s*(\d+)%", t, re.S)
    warnings=[]
    def need(arr, idx, name):
        if len(arr)<=idx: warnings.append(f"Missing {name} — paste the full dump.")
        return pick(arr, idx)
    home={"shots":need(shots_avgs,0,"Shots home"),"shots_total":need(shots_tot,0,"Shots total home"),
        "on_pct":need(on_pcts,0,"ON% home"),"blocked":need(blocked_avgs,0,"Blocked home"),"blocked_total":need(blocked_tot,0,"Blocked total home"),
        "scored_avg":need(scored_avgs,0,"Scored Avg home"),"scored_total":need(scored_tot,0,"Scored total home"),
        "conceded_avg":need(conceded_avgs,0,"Conceded Avg home"),"conceded_total":need(conceded_tot,0,"Conceded total home"),
        "scored_pct":need(scored_pcts,0,"Scored% home"),"played":need(played,0,"Played home"),
        "possession":need(possessions,0,"Possession home"),"pass_acc":need(pass_accs,0,"Pass Acc home"),
        "total_attacks":need(total_attacks,0,"Total attacks home"),"dangerous_attacks":need(dangerous_attacks,0,"Dangerous attacks home"),
        "btts_yes_pct":need(btts,0,"BTTS Yes% home")}
    away={"shots":need(shots_avgs,1,"Shots away"),"shots_total":need(shots_tot,1,"Shots total away"),
        "on_pct":need(on_pcts,1,"ON% away"),"blocked":need(blocked_avgs,1,"Blocked away"),"blocked_total":need(blocked_tot,1,"Blocked total away"),
        "scored_avg":need(scored_avgs,1,"Scored Avg away"),"scored_total":need(scored_tot,1,"Scored total away"),
        "conceded_avg":need(conceded_avgs,1,"Conceded Avg away"),"conceded_total":need(conceded_tot,1,"Conceded total away"),
        "scored_pct":need(scored_pcts,1,"Scored% away"),"played":need(played,1,"Played away"),
        "possession":need(possessions,1,"Possession away"),"pass_acc":need(pass_accs,1,"Pass Acc away"),
        "total_attacks":need(total_attacks,1,"Total attacks away"),"dangerous_attacks":need(dangerous_attacks,1,"Dangerous attacks away"),
        "btts_yes_pct":need(btts,1,"BTTS Yes% away")}
    return {"home_code":home_name,"away_code":away_name,"home":home,"away":away,"_warnings":warnings,"raw_dump":text}

def _card_html(d, settled=False):
    hv,av=d.get("home_votes",0),d.get("away_votes",0); dec=d.get("vote_decision","LEAN"); pick=d.get("vote_pick","")
    title=f"{d.get('home_team','?')} vs {d.get('away_team','?')}"
    meta=f"{hv}-{av} &nbsp;·&nbsp; {dec}"+(f" &nbsp;·&nbsp; {pick} Win/Draw" if pick else "")
    if dec=="TAKE": badge=f'<span class="pick-badge badge-take">TAKE {pick} Win/Draw</span>'; cls="pick-home" if pick=="HOME" else "pick-away"
    elif dec=="LEAN": badge='<span class="pick-badge badge-lean">LEAN ONLY — NO BET</span>'; cls="skip"
    else: badge='<span class="pick-badge badge-skip">SKIP</span>'; cls="skip"
    verdict=""
    if settled and d.get("home_score") is not None:
        verdict=f'<div style="margin-top:10px;"><span class="pick-badge {"badge-win" if d.get("dc_hit")==1 else "badge-loss"}">Result {d.get("home_score")}-{d.get("away_score")} — {"HIT" if d.get("dc_hit")==1 else "MISS"}</span></div>'
    pills=""
    for label,w,hval,aval in d.get("breakdown",[]):
        cls2="home" if w=="HOME" else ("away" if w=="AWAY" else "tie")
        pills+=f'<span class="vote-pill {cls2}">{label}: {hval} vs {aval} → {w}</span>'
    return f'<div class="match-card {cls}"><div class="match-title">{title}</div><div class="match-meta">{meta}</div>{badge}{verdict}<div class="vote-row">{pills}</div></div>'

st.title("King — 6 Votes")
st.caption("Home/Away only. 6 votes. 4+ = TAKE leader Win/Draw. 3-3 = LEAN only. ALL data stored: TAKE + LEAN + SKIP, full dump + all metrics.")

tab_new, tab_pending, tab_perf, tab_lab = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance", "🧪 Lab"])
with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Include the full dump — Shots, ON target, Blocked, Goals, Scored a goal, Passes, Attacks, Others, Disciplinary. Everything is stored.")
    raw_in = st.text_area("Input", height=340, label_visibility="collapsed", placeholder="Paste the match data\n...")
    run = st.button("Run the Logic", type="primary")
    if run and raw_in.strip():
        try:
            parsed = parse_dump(raw_in.strip())
            h,a = parsed["home"], parsed["away"]
            # Only 6 vote fields are required to decide. Other fields may be None — raw_dump still saves them.
            req = ["shots","on_pct","blocked","scored_avg","conceded_avg","scored_pct"]
            missing = [k for k in req if h[k] is None] + [k for k in req if a[k] is None]
            if missing:
                st.error(f"Parse failed — could not find 6-vote fields: {', '.join(sorted(set(missing)))}. Paste the full dump.")
                st.stop()
            home_team = parsed["home_code"] or "HOME"; away_team = parsed["away_code"] or "AWAY"
            hv,av,br = compute_votes(h,a); dec,pick,reason = decide(hv,av)
            d = {"home_team":home_team,"away_team":away_team,"home_votes":hv,"away_votes":av,"vote_decision":dec,"vote_pick":pick or "","breakdown":br}
            st.markdown("### The Pick"); st.markdown(_card_html(d), unsafe_allow_html=True)
            if dec=="TAKE": st.success(reason)
            elif dec=="LEAN": st.info(reason)
            else: st.info(reason)
            if parsed.get("_warnings"):
                with st.expander(f"Parser warnings ({len(parsed['_warnings'])})"):
                    for w in parsed["_warnings"]: st.write(f"- {w}")
            # ALL DATA STORED: TAKE + LEAN + SKIP, full dump + all metrics
            row = {"home_team":home_team,"away_team":away_team,"home_votes":hv,"away_votes":av,"vote_decision":dec,"vote_pick":pick or "",
                   "simple_qualifying":1 if dec=="TAKE" else 0,"simple_category":"6-VOTE","simple_bet":f"{pick} Win/Draw" if pick else "LEAN/SKIP",
                   "extracted_at":datetime.now(timezone.utc).isoformat(),
                   "raw_dump":parsed.get("raw_dump",""),
                   "all_data":json.dumps({"home":h,"away":a,"breakdown":br}, default=str),
                   "home_shots":h.get("shots"),"away_shots":a.get("shots"),
                   "home_on_pct":h.get("on_pct"),"away_on_pct":a.get("on_pct"),
                   "home_blocked":h.get("blocked"),"away_blocked":a.get("blocked"),
                   "home_scored_avg":h.get("scored_avg"),"away_scored_avg":a.get("scored_avg"),
                   "home_conceded_avg":h.get("conceded_avg"),"away_conceded_avg":a.get("conceded_avg"),
                   "home_scored_pct":h.get("scored_pct"),"away_scored_pct":a.get("scored_pct"),
                   "home_possession":h.get("possession"),"away_possession":a.get("possession"),
                   "home_total_attacks":h.get("total_attacks"),"away_total_attacks":a.get("total_attacks"),
                   "home_dangerous_attacks":h.get("dangerous_attacks"),"away_dangerous_attacks":a.get("dangerous_attacks")}
            existing = db_find_existing(home_team, away_team)
            if existing: st.info(f"Already stored (match_id {existing.get('match_id')}).")
            else:
                try: db_insert(row); st.success(f"Saved — ALL data stored ({dec}). Full dump + all metrics in DB.")
                except Exception as e: st.error(f"Insert failed: {e} — run the ALTER TABLE SQL for new columns.")
        except Exception as e:
            st.error(f"Parse failed: {e}")

with tab_pending:
    st.markdown("### Pending — ALL decisions")
    pending = db_pending()
    if not pending: st.info("No pending matches.")
    else:
        for m in pending:
            d=dict(m); d["breakdown"]=[]
            st.markdown(_card_html(d), unsafe_allow_html=True)
            # Show stored full data if present
            if m.get("all_data"):
                with st.expander("Stored full data"):
                    try: st.json(json.loads(m["all_data"]))
                    except Exception: st.write(m["all_data"])
            if m.get("raw_dump"):
                with st.expander("Stored raw dump"):
                    st.text(m["raw_dump"][:4000])
            with st.expander("Enter result"):
                c1,c2,c3 = st.columns([1,1,2])
                with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        pick = m.get("vote_pick") or ("HOME" if "HOME" in str(m.get("simple_bet","")) else "AWAY" if "AWAY" in str(m.get("simple_bet","")) else None)
                        hit,prof = settle_vote(pick, int(hs), int(aws))
                        # LEAN/SKIP have no pick — settling still stores result, hit stays None for non-TAKE
                        if m.get("vote_decision")!="TAKE":
                            hit,prof = None, 0.0
                        db_settle(m["match_id"], int(hs), int(aws), result_str(int(hs), int(aws)), hit, prof); st.success("Settled."); st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"]); st.success("Deleted."); st.rerun()

with tab_perf:
    st.markdown("### Performance — TAKE only, counted PRE-match")
    settled = db_settled()
    if not settled: st.info("No settled matches yet.")
    else:
        bets=[m for m in settled if m.get("vote_decision")=="TAKE" and m.get("dc_hit") is not None]
        n=len(bets); wins=sum(1 for m in bets if m.get("dc_hit")==1); net=sum((m.get("profit_loss") or 0.0) for m in bets)
        c1,c2,c3,c4,c5=st.columns(5); c1.metric("TAKE Bets",n); c2.metric("Wins",wins); c3.metric("Losses",n-wins); c4.metric("Hit %",f"{(wins/n*100) if n else 0:.1f}%"); c5.metric("Net P/L",f"{net:+.2f}u")
        st.caption("LEAN / SKIP are stored but NOT counted as bets. A 4+ team that loses is a LOSS, not a SKIP.")
        for m in settled:
            d=dict(m); d["breakdown"]=[]
            st.markdown(_card_html(d, settled=True), unsafe_allow_html=True)

with tab_lab:
    st.markdown("### Lab")
    text_in = st.text_area("Input", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        try:
            p=parse_dump(text_in); h,a=p["home"],p["away"]
            hv,av,br=compute_votes(h,a); dec,pick,reason=decide(hv,av)
            st.markdown(f"**{p['home_code']} vs {p['away_code']} — {hv}-{av} — {reason}**")
            for label,w,hv_val,av_val in br: st.write(f"- {label}: {hv_val} vs {av_val} → {w}")
            if dec=="TAKE": st.success(f"TAKE {pick} Win/Draw")
            else: st.info(reason)
            with st.expander("All parsed data (what would be stored)"):
                st.json({"home":h,"away":a,"raw_dump_chars":len(text_in)})
        except Exception as e: st.error(f"Parse failed: {e}")
