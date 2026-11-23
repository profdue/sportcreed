import re
from datetime import datetime, timezone
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="King — Lead / Defend / Equalize", page_icon="⚽", layout="wide")
st.title("King — Lead / Defend / Equalize")
st.caption("Step 1 scores first · Step 2 holds lead · Step 3 comes back · Step 4 shapes scoreline. Gaps decide, not Step 1 alone.")

@st.cache_resource
def get_supabase():
    try:
        url = st.secrets["supabase"]["url"]; key = st.secrets["supabase"]["key"]
    except Exception:
        url = st.secrets["SUPABASE_URL"]; key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def now_iso(): return datetime.now(timezone.utc).isoformat()
def pct(s):
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s or "")
    return float(m.group(1)) if m else None

# ---------- PARSER: source headings only, home split for home team, away split for away team ----------
def parse_dump(t: str) -> dict:
    f = {"raw": t}
    # Teams: from "Monza" / "Fiorentina" header + "Monza (HOME)" / "Fiorentina (AWAY)" tables.
    # This dump has no "X vs Y predictions" line, so take first two team names from Leading and Trailing blocks.
    # Fallback: search "Monza" / "Fiorentina" style — use first "Team Stats" header pattern.
    # Teams: (HOME)/(AWAY) is reliable in this format. "flagMonza vs Fiorentina 3:0" is NOT.
    # Old code took the vs-line first and got home = "flagMonza".
    hm = re.search(r"([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})\s+\(HOME\)", t)
    am = re.search(r"([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})\s+\(AWAY\)", t)
    if hm and am:
        f["home_team"] = hm.group(1).strip()
        f["away_team"] = am.group(1).strip()
    else:
        m = re.search(r"([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})\s+vs\s+([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})", t)
        if m:
            f["home_team"] = m.group(1).strip().replace("flag","").replace(" predictions","").strip()
            f["away_team"] = m.group(2).strip().replace(" predictions","").strip()
    for k in ("home_team","away_team"):
        if f.get(k):
            f[k] = re.sub(r"^flag", "", f[k]).strip()
            f[k] = re.sub(r"\s+\d+:\d+.*$", "", f[k]).strip()
    # --- ANCHOR: only parse Steps 1-3 from "Leading and Trailing" block, NOT from earlier sections ---
    # Earlier in this dump, "% Scored first" appears in "Team Stats vs League Average" (40% / 20%)
    # and in "Contextual averages" — those are NOT the source. Source is Leading and Trailing.
    lt_idx = t.find("Leading and Trailing")
    lt = t[lt_idx:] if lt_idx >= 0 else t
    # Split LT into Home block (Monza Home) and Away block (Fiorentina Away)
    home_block, away_block = lt, lt
    # Home block: from "Monza" + "Home" in LT to "Fiorentina" + "Away"
    if f.get("home_team") and f.get("away_team"):
        hm2 = re.search(re.escape(f["home_team"]) + r"\s*\n\s*Home", lt)
        am2 = re.search(re.escape(f["away_team"]) + r"\s*\n\s*Away", lt)
        if hm2 and am2:
            home_block = lt[hm2.start():am2.start()]
            away_block = lt[am2.start():]
    def pct_after(block: str, pat: str):
        m = re.search(pat, block, re.I | re.S)
        if not m: return None
        seg = block[m.end():m.end()+600]
        p = re.search(r"(\d+(?:\.\d+)?)\s*%", seg)
        return float(p.group(1)) if p else None
    def int_after(block: str, pat: str):
        m = re.search(pat, block, re.I | re.S)
        if not m: return None
        seg = block[m.end():m.end()+300]
        n = re.search(r"(\d+)", seg)
        return int(n.group(1)) if n else None
    # Step 1: OPENING GOALS > Team scored first — Home block vs Away block
    f["step1_home"] = pct_after(home_block, r"Team scored first")
    f["step1_away"] = pct_after(away_block, r"Team scored first")
    # Step 2: DEFENDING THE LEAD > Lead-defending rate
    f["step2_home"] = pct_after(home_block, r"Lead-defending rate")
    f["step2_away"] = pct_after(away_block, r"Lead-defending rate")
    # Step 3: EQUALIZING > Equalizing rate
    f["step3_home"] = pct_after(home_block, r"Equalizing rate")
    f["step3_away"] = pct_after(away_block, r"Equalizing rate")
    # Raw counts
    f["leads_home"] = int_after(home_block, r"Goals giving the lead")
    f["leads_away"] = int_after(away_block, r"Goals giving the lead")
    f["trails_home"] = int_after(home_block, r"Goals giving lead to opponent")
    f["trails_away"] = int_after(away_block, r"Goals giving lead to opponent")
    # Step 4: Contextual averages — Clean sheets / Failed to score (home split / away split)
    # Use "Contextual averages" block if present, else Team Stats vs League Average
    ctx_idx = t.find("Contextual averages")
    ctx = t[ctx_idx:ctx_idx+2500] if ctx_idx >= 0 else t
    # In this dump Contextual table is transposed, so fall back to explicit Home vs Away Stats table:
    # "Clean sheets 20% 0%" and "Failed to score 20% 20%" under "Home vs Away Stats"
    hvs_idx = t.find("Home vs Away Stats")
    hvs = t[hvs_idx:hvs_idx+2000] if hvs_idx >= 0 else ""
    def two_pcts(block, pat):
        m = re.search(pat, block, re.I | re.S)
        if not m: return None, None
        seg = block[m.end():m.end()+400]
        ps = re.findall(r"(\d+(?:\.\d+)?)\s*%", seg)
        return (float(ps[0]), float(ps[1])) if len(ps) >= 2 else (None, None)
    cs_h, cs_a = two_pcts(hvs, r"Clean sheets")
    fs_h, fs_a = two_pcts(hvs, r"Failed to score")
    f["cs_home"], f["cs_away"], f["fts_home"], f["fts_away"] = cs_h, cs_a, fs_h, fs_a
    # Veto: conceded per game in split — from "Comparison with league average" / "Goals conceded per game"
    # Monza home 1.80, Fiorentina away 2.40 in this dump
    m = re.search(r"Goals conceded per game\s*(\d+(?:\.\d+)?)\s*\n?\s*(\d+(?:\.\d+)?)", t)
    if m:
        f["conceded_home"], f["conceded_away"] = float(m.group(1)), float(m.group(2))
    else:
        m2 = re.search(r"Avg Conceded\s*(\d+(?:\.\d+)?)", t)
        if m2: f["conceded_home"] = float(m2.group(1))
    return f

# ---------- EVALUATION ----------
def evaluate(f: dict):
    home, away = f.get("home_team","Home"), f.get("away_team","Away")
    steps = []
    for i, (hk, ak, label) in enumerate([
        ("step1_home","step1_away","Step 1 — Scores first"),
        ("step2_home","step2_away","Step 2 — Holds lead"),
        ("step3_home","step3_away","Step 3 — Comes back")], start=1):
        h, a = f.get(hk), f.get(ak)
        if h is None or a is None:
            steps.append({"step": i, "label": label, "home": h, "away": a, "winner": None, "gap": None, "status": "MISSING"})
        else:
            winner = "HOME" if h > a else ("AWAY" if a > h else "TIE")
            gap = round(abs(h - a), 1)
            steps.append({"step": i, "label": label, "home": h, "away": a, "winner": winner, "gap": gap, "status": "OK"})
    # Totals: sum gaps where team wins the step
    total_home = sum(s["gap"] for s in steps if s["winner"] == "HOME")
    total_away = sum(s["gap"] for s in steps if s["winner"] == "AWAY")
    net = round(abs(total_home - total_away), 1)
    total_winner = "HOME" if total_home > total_away else ("AWAY" if total_away > total_home else "TIE")
    # Alignment tiers
    winners = [s["winner"] for s in steps if s["winner"] in ("HOME","AWAY")]
    all_agree = len(set(winners)) == 1 and len(winners) == 3
    if any(s["status"] == "MISSING" for s in steps):
        tier = "NO BET"; tier_reason = "Missing Step 1/2/3 — cannot decide."
    elif net <= 10:
        tier = "DRAW / NO BET"; tier_reason = f"Total gaps within ~10 (Home {total_home} vs Away {total_away}, net {net}) — no clean winner."
    elif all_agree and net >= 30:
        tier = "LOCK"; tier_reason = f"All 3 steps agree ({total_winner}), net gap {net} large."
    else:
        tier = "LEAN"; tier_reason = f"Steps split, but {total_winner} wins total gap {total_home} vs {total_away} (net {net})."
    # Veto: favourite concedes >1.60 in split -> downgrade one tier, never upgrade
    fav_team = home if total_winner == "HOME" else away if total_winner == "AWAY" else None
    fav_conceded = f.get("conceded_home") if total_winner == "HOME" else f.get("conceded_away") if total_winner == "AWAY" else None
    veto = bool(fav_conceded is not None and fav_conceded > 1.60)
    if veto:
        if tier == "LOCK": tier = "LEAN"
        elif tier == "LEAN": tier = "DRAW / NO BET"
        tier_reason += f" VETO: {fav_team} concedes {fav_conceded}/game in split (>1.60) — downgraded one tier."
    # Raw-count flag: any key % on <3 leads/trails
    # Raw-count rule: 100%/0% on <3 leads/trails = flag, not evidence.
    # Check the denominator that backs each step: Step 2 uses leads, Step 3 uses trails.
    low_n = False
    if f.get("leads_home") is not None and f["leads_home"] < 3: low_n = True
    if f.get("leads_away") is not None and f["leads_away"] < 3: low_n = True
    if f.get("trails_home") is not None and f["trails_home"] < 3: low_n = True
    if f.get("trails_away") is not None and f["trails_away"] < 3: low_n = True
    # Confidence
    if tier in ("DRAW / NO BET","NO BET"): conf = "No bet"
    elif all_agree and net >= 30 and not veto and not low_n: conf = "High"
    elif net > 10 and not veto and not low_n: conf = "Medium"
    else: conf = "Low"
    # Step 4 shape: to-nil vs both-to-score
    cs_fav = f.get("cs_home") if total_winner == "HOME" else f.get("cs_away")
    fts_opp = f.get("fts_away") if total_winner == "HOME" else f.get("fts_home")
    if cs_fav is not None and fts_opp is not None and cs_fav >= 40 and fts_opp >= 40:
        shape = "To-nil / low-score shape"
    elif cs_fav is not None and fts_opp is not None and cs_fav < 30 and fts_opp < 30:
        shape = "Both teams to score shape"
    else: shape = "Mixed shape — Step 4 does not pick winner"
    # Predicted script
    s1w = next((s for s in steps if s["step"]==1), {})
    script = f"{home if s1w.get('winner')=='HOME' else away if s1w.get('winner')=='AWAY' else '?'} scores first. "
    script += f"{fav_team or '?'} holds / comes back on total gap. {shape}."
    return {"steps": steps, "total_home": round(total_home,1), "total_away": round(total_away,1),
            "net": net, "total_winner": total_winner, "tier": tier, "tier_reason": tier_reason,
            "veto": veto, "fav_conceded": fav_conceded, "low_n": low_n, "confidence": conf,
            "shape": shape, "script": script, "fav_team": fav_team}

# ---------- UI ----------
tab1, tab2, tab3 = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance"])
with tab1:
    dump = st.text_area("Paste the match data (Leading and Trailing + Contextual averages)", height=300)
    if st.button("Evaluate", type="primary") and dump.strip():
        f = parse_dump(dump); r = evaluate(f)
        # ===== DECISION CARD — the bet, first, impossible to miss =====
        _tier = r['tier']
        _conf = r['confidence']
        _winner = r['total_winner']
        _winner_name = f.get('home_team') if _winner=='HOME' else (f.get('away_team') if _winner=='AWAY' else 'No winner')
        if _tier == 'LOCK':
            _bg, _border, _badge, _big, _action = '#0f5132','#198754','#198754','✅ LOCK — BET','BET: NO DRAW — ' + (_winner_name or '')
        elif _tier == 'LEAN':
            _bg, _border, _badge, _big, _action = '#664d03','#ffc107','#ffc107','⚠️ LEAN — SMALL BET','BET: NO DRAW — ' + (_winner_name or '')
        elif _tier == 'DRAW / NO BET':
            _bg, _border, _badge, _big, _action = '#842029','#dc3545','#dc3545','⛔ DRAW / NO BET','NO BET — SKIP THIS MATCH'
        else:
            _bg, _border, _badge, _big, _action = '#842029','#dc3545','#dc3545','⛔ NO BET','NO BET — SKIP THIS MATCH'
        _veto_txt = f"🚨 VETO APPLIED — downgraded (fav concedes {r['fav_conceded']}/game)" if r['veto'] else "No veto"
        _lown_txt = "⚠️ LOW-N — key % on &lt;3 leads/trails" if r['low_n'] else "Sample OK"
        st.markdown(f'''
        <div style="border:3px solid {_border}; border-radius:16px; padding:20px 22px; background:{_bg}; color:white; margin:8px 0 16px 0;">
          <div style="font-size:13px; letter-spacing:2px; opacity:0.85;">BETTING DECISION &nbsp;•&nbsp; {f.get('home_team','?')} vs {f.get('away_team','?')}</div>
          <div style="font-size:36px; font-weight:900; line-height:1.1; margin:8px 0 4px 0;">{_big}</div>
          <div style="font-size:20px; font-weight:700; margin-bottom:10px;">{_action}</div>
          <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;">
            <span style="background:rgba(255,255,255,0.18); padding:6px 12px; border-radius:999px; font-weight:700;">Winner: {_winner_name} ({_winner})</span>
            <span style="background:rgba(255,255,255,0.18); padding:6px 12px; border-radius:999px; font-weight:700;">Total: {r['total_home']} vs {r['total_away']} &nbsp;•&nbsp; Net {r['net']}</span>
            <span style="background:rgba(255,255,255,0.18); padding:6px 12px; border-radius:999px; font-weight:700;">Confidence: {_conf}</span>
          </div>
          <div style="font-size:14px; line-height:1.5; opacity:0.95;">{_veto_txt} &nbsp;•&nbsp; {_lown_txt}</div>
          <div style="font-size:14px; line-height:1.5; margin-top:6px;"><b>Script:</b> {r['script']}</div>
          <div style="font-size:13px; line-height:1.5; margin-top:4px; opacity:0.9;"><b>Shape (Step 4):</b> {r['shape']}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown("**Why — Steps 1/2/3**")
        for s in r["steps"]:
            _w = s['winner'] or 'MISSING'
            _wc = '#198754' if _w=='HOME' else ('#fd7e14' if _w=='AWAY' else '#6c757d')
            st.markdown(f"<div style='border-left:4px solid {_wc}; padding:6px 10px; margin:4px 0; background:#f8f9fa; border-radius:6px;'><b>{s['label']}</b> — Home {s['home']}% vs Away {s['away']}% → <b>{_w}</b> · Gap {s['gap']}</div>", unsafe_allow_html=True)
        st.write(f"**Tier reason:** {r['tier_reason']}")
        st.write(f"**Step 4 detail:** CS Home {f.get('cs_home')}% / Away {f.get('cs_away')}% · FTS Home {f.get('fts_home')}% / Away {f.get('fts_away')}%")
        row = {"home_team": f.get("home_team"), "away_team": f.get("away_team"),
               "tier": r["tier"],
               "confidence": r["confidence"],
               "decision": "NO DRAW" if r["tier"] in ("LOCK","LEAN") else "NO BET",
               "checks": [{"name": s["label"], "status": s["winner"] or "MISSING", "detail": f"Home {s['home']}% vs Away {s['away']}% · Gap {s['gap']}"} for s in r["steps"]],
               "parsed": {**f, **r}, "raw_dump": dump,
               "all_data": {"parsed": f, "result": r, "tier": r["tier"], "confidence": r["confidence"]},
               "extracted_at": now_iso()}
        row = {k: v for k, v in row.items() if v is not None}
        try:
            sb = get_supabase()
            ex = sb.table("matches").select("match_id").eq("home_team", f.get("home_team")).eq("away_team", f.get("away_team")).limit(1).execute()
            if ex.data: st.info(f"Already stored (match_id {ex.data[0]['match_id']}). Delete first to re-save.")
            else: sb.table("matches").insert(row).execute(); st.success("Saved — Step 1/2/3 + gaps + tier + script stored.")
        except Exception as e: st.error(f"Insert failed: {e}")
with tab2:
    try: data = get_supabase().table("matches").select("*").is_("home_score","null").order("extracted_at", desc=True).execute().data or []
    except Exception as e: data=[]; st.error(str(e))
    if not data: st.info("No pending matches.")
    for m in data:
        p = m.get("parsed") or {}
        st.markdown(f"**{m.get('home_team')} vs {m.get('away_team')}** · {p.get('tier','?')} · {p.get('confidence','?')}")
        st.write(p.get("script",""))
        c1,c2,c3 = st.columns(3)
        hs=c1.number_input("Home",min_value=0,step=1,key=f"h_{m['match_id']}")
        aws=c2.number_input("Away",min_value=0,step=1,key=f"a_{m['match_id']}")
        if c3.button("Settle",key=f"s_{m['match_id']}"):
            res="HOME" if hs>aws else ("AWAY" if aws>hs else "DRAW")
            get_supabase().table("matches").update({"home_score":int(hs),"away_score":int(aws),"result":res,"settled_at":now_iso()}).eq("match_id",m["match_id"]).execute(); st.rerun()
        if st.button("Delete",key=f"d_{m['match_id']}"):
            get_supabase().table("matches").delete().eq("match_id",m["match_id"]).execute(); st.rerun()
with tab3:
    try: data = get_supabase().table("matches").select("*").not_.is_("home_score","null").order("settled_at",desc=True).execute().data or []
    except Exception as e: data=[]; st.error(str(e))
    picks=[m for m in data if (m.get("parsed") or {}).get("tier") in ("LOCK","LEAN")]
    if not picks: st.info("No settled LOCK/LEAN picks yet.")
    else:
        st.metric("LOCK/LEAN settled", len(picks))
        for m in picks:
            p=m.get("parsed") or {}
            st.write(f"{m.get('home_team')} {m.get('home_score')}-{m.get('away_score')} {m.get('away_team')} · {p.get('tier')} · {p.get('confidence')} · Result {m.get('result')} · Script: {p.get('script','')}")
