import re
from datetime import datetime, timezone
import streamlit as st
from supabase import create_client

st.set_page_config(page_title="King — NO DRAW", page_icon="⚽", layout="wide")
st.title("King — NO DRAW")
st.caption("Gap + X% + Fav% + Form + Trap. 5 Pass + 1 Warning = NO DRAW. Any FAIL = NO BET.")

@st.cache_resource
def get_supabase():
    try:
        url = st.secrets["supabase"]["url"]; key = st.secrets["supabase"]["key"]
    except Exception:
        url = st.secrets["SUPABASE_URL"]; key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# ---------- PARSER ----------
def parse_dump(t: str) -> dict:
    f = {}
    # Teams: "Werder Bremen vs Hamburger SV predictions"
    m = re.search(r"([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})\s+vs\s+([A-Z][A-Za-zÀ-ÿ' \-\.]{2,40})\s+predictions", t)
    if m:
        f["home_team"] = m.group(1).strip()
        f["away_team"] = m.group(2).strip().replace(" predictions", "").strip()
    # League: "Bundesliga Round 10" / "HNL Round 14"
    m = re.search(r"^([A-Z][A-Za-z ]+)\s+Round\s+\d+", t, re.M)
    if m: f["league"] = m.group(1).strip()
    # ELO: "82\tELO\t81" or "85 ELO 76"
    m = re.search(r"(\d+(?:\.\d+)?)\s+ELO\s+(\d+(?:\.\d+)?)", t)
    if m:
        f["home_elo"] = float(m.group(1)); f["away_elo"] = float(m.group(2))
    # ELO win probability block: "37.2% Werder / 24.7% Draw / 38% Hamburger"
    blk = re.search(r"ELO win probability(.{0,400})", t, re.S | re.I)
    if blk:
        probs = re.findall(r"(\d+(?:\.\d+)?)%\s*\n?\s*([A-Za-zÀ-ÿ' \-\.]{2,40})", blk.group(1))
        clean = [(float(p), n.strip().replace(" predictions","").strip()) for p, n in probs if "Draw" not in n]
        dm = re.search(r"(\d+(?:\.\d+)?)%\s*\n?\s*Draw\b", blk.group(1))
        if dm: f["draw_prob"] = float(dm.group(1))
        if clean:
            best = max(clean, key=lambda x: x[0])
            f["fav_prob"] = best[0]; f["fav_team"] = best[1]
    if f.get("home_team") and f.get("fav_team"):
        f["home_is_fav"] = f["fav_team"].strip() == f["home_team"].strip()
    # Last 5: ONLY count scores AFTER "Matches\n<team>" header (ignores live 0-0 header)
    def count_wdl(section: str, names: list):
        lines = [ln.strip() for ln in section.splitlines()]
        res = []
        for i, ln in enumerate(lines):
            if re.fullmatch(r"\d+\s*-\s*\d+", ln):
                prev = lines[i-1] if i > 0 else ""
                nxt = lines[i+1] if i+1 < len(lines) else ""
                is_home = any(n and n in prev for n in names)
                is_away = any(n and n in nxt for n in names)
                if not (is_home or is_away):
                    continue
                a, b = [int(x) for x in re.findall(r"\d+", ln)]
                if a == b: res.append("D")
                elif is_home: res.append("W" if a > b else "L")
                else: res.append("W" if b > a else "L")
                if len(res) == 5: break
        if len(res) != 5: return None
        return [res.count("W"), res.count("D"), res.count("L")]
    if f.get("home_team") and f.get("away_team"):
        home_sec = re.split(r"Matches\s*\n\s*" + re.escape(f["home_team"]), t)
        away_sec = re.split(r"Matches\s*\n\s*" + re.escape(f["away_team"]), t)
        home_part = home_sec[1] if len(home_sec) > 1 else ""
        away_part = away_sec[1] if len(away_sec) > 1 else ""
        # home_part may contain away section too — cut at away Matches header
        if away_part and home_part:
            cut = re.split(r"Matches\s*\n\s*" + re.escape(f["away_team"]), home_part)
            if len(cut) > 1: home_part = cut[0]
        home_names = [f["home_team"], f["home_team"].split()[-1]]
        away_names = [f["away_team"], f["away_team"].split()[-1]]
        home_c = count_wdl(home_part, home_names) if home_part else None
        away_c = count_wdl(away_part, away_names) if away_part else None
        if f.get("home_is_fav", True):
            if home_c: f["fav_last5"] = home_c; f["fav_last5_team"] = f["home_team"]
            if away_c: f["und_last5"] = away_c; f["und_last5_team"] = f["away_team"]
        else:
            if away_c: f["fav_last5"] = away_c; f["fav_last5_team"] = f["away_team"]
            if home_c: f["und_last5"] = home_c; f["und_last5_team"] = f["home_team"]
    if f.get("fav_last5"): f["fav_draw_rate"] = round(f["fav_last5"][1] / 5 * 100, 1)
    if f.get("und_last5"): f["und_draw_rate"] = round(f["und_last5"][1] / 5 * 100, 1)
    return f

# ---------- CHECKS ----------
def evaluate(f: dict):
    checks = []
    gap = (f.get("home_elo") or 0) - (f.get("away_elo") or 0)
    checks.append({"name": "1. Gap (dELO)", "status": "PASS" if gap >= 10 else "FAIL",
                   "detail": f"{f.get('home_elo')} vs {f.get('away_elo')} = {gap} — needs 10+"})
    x = f.get("draw_prob")
    checks.append({"name": "2. Draw Prob (X%)", "status": "PASS" if x is not None and x <= 20 else ("MISSING" if x is None else "FAIL"),
                   "detail": f"{x}% — needs 20% or less" if x is not None else "Draw prob missing"})
    fp = f.get("fav_prob")
    checks.append({"name": "3. Favourite Prob (1%)", "status": "PASS" if fp is not None and fp >= 65 else ("MISSING" if fp is None else "FAIL"),
                   "detail": f"{f.get('fav_team')} {fp}% — needs 65%+" if fp is not None else "Favourite prob missing"})
    for key, label in [("fav_last5", "4. Last Matches — Favourite"), ("und_last5", "5. Last Matches — Underdog")]:
        v = f.get(key)
        if v is None:
            checks.append({"name": label, "status": "MISSING", "detail": "Last 5 missing"})
        else:
            dr = round(v[1] / 5 * 100, 1)
            if dr <= 20: stt, txt = "PASS", "Decisive"
            elif dr <= 40: stt, txt = "WARNING", "Draw-prone"
            else: stt, txt = "FAIL", "Draw-prone"
            team = f.get("fav_last5_team") if key == "fav_last5" else f.get("und_last5_team")
            checks.append({"name": label, "status": stt,
                           "detail": f"{team} — Last 5: {v[0]}W {v[1]}D {v[2]}L, Draw rate {dr}% — {txt}"})
    trap_pass = bool(f.get("home_is_fav"))
    checks.append({"name": "6. Home Underdog Trap", "status": "PASS" if trap_pass else "FAIL",
                   "detail": f"Home = {f.get('home_team')} (favourite) — No Hull / Dynamo trap" if trap_pass else f"Home = {f.get('home_team')} (underdog) — Hull / Dynamo trap, NO BET"})
    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_warn = sum(1 for c in checks if c["status"] == "WARNING")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_miss = sum(1 for c in checks if c["status"] == "MISSING")
    decision = "NO DRAW" if (n_pass == 5 and n_warn == 1 and n_fail == 0 and n_miss == 0) else "NO BET"
    return checks, decision, (n_pass, n_warn, n_fail, n_miss)

# ---------- UI ----------
tab1, tab2, tab3 = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance"])
with tab1:
    dump = st.text_area("Paste the match data", height=300, placeholder="Paste BeSoccer dump here...")
    if st.button("Evaluate", type="primary") and dump.strip():
        f = parse_dump(dump)
        checks, decision, counts = evaluate(f)
        st.subheader(f"{f.get('home_team','?')} vs {f.get('away_team','?')} — {decision}")
        for c in checks:
            icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "MISSING": "❓"}.get(c["status"], "")
            st.write(f"{icon} **{c['name']}**: {c['status']} — {c['detail']}")
        st.write(f"{counts[0]} Pass, {counts[1]} Warning, {counts[2]} Fail, {counts[3]} Missing — {decision}")
        # Save ALL decisions (NO DRAW + NO BET) — NO DRAW-only columns, no DC/BTTS
        row = {
            "home_team": f.get("home_team"), "away_team": f.get("away_team"),
            "league": f.get("league"),
            "home_elo": f.get("home_elo"), "away_elo": f.get("away_elo"),
            "draw_prob": f.get("draw_prob"), "fav_prob": f.get("fav_prob"),
            "fav_team": f.get("fav_team"), "home_is_fav": f.get("home_is_fav"),
            "fav_last5": f.get("fav_last5"), "fav_last5_team": f.get("fav_last5_team"),
            "fav_draw_rate": f.get("fav_draw_rate"),
            "und_last5": f.get("und_last5"), "und_last5_team": f.get("und_last5_team"),
            "und_draw_rate": f.get("und_draw_rate"),
            "decision": decision, "checks": checks, "parsed": f, "raw_dump": dump,
            "all_data": {"parsed": f, "checks": checks, "decision": decision},
            "extracted_at": now_iso(),
        }
        row = {k: v for k, v in row.items() if v is not None}
        try:
            sb = get_supabase()
            ex = sb.table("matches").select("match_id").eq("home_team", f.get("home_team")).eq("away_team", f.get("away_team")).limit(1).execute()
            if ex.data:
                st.info(f"Already stored (match_id {ex.data[0]['match_id']}). Delete it first to re-save.")
            else:
                sb.table("matches").insert(row).execute()
                st.success("Saved — ALL data stored.")
        except Exception as e:
            st.error(f"Insert failed: {e}")
with tab2:
    try:
        data = get_supabase().table("matches").select("*").is_("home_score", "null").order("extracted_at", desc=True).execute().data or []
    except Exception as e:
        data = []; st.error(str(e))
    if not data: st.info("No pending matches.")
    for m in data:
        st.markdown(f"**{m.get('home_team')} vs {m.get('away_team')}** · {m.get('decision')}")
        for c in (m.get("checks") or []):
            st.write(f"{c.get('status')} — {c.get('name')}: {c.get('detail')}")
        c1, c2, c3 = st.columns(3)
        hs = c1.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
        aws = c2.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
        if c3.button("Settle", key=f"s_{m['match_id']}"):
            res = "HOME" if hs > aws else ("AWAY" if aws > hs else "DRAW")
            get_supabase().table("matches").update({
                "home_score": int(hs), "away_score": int(aws), "result": res,
                "nodraw_hit": (res != "DRAW") if m.get("decision") == "NO DRAW" else None,
                "settled_at": now_iso()}).eq("match_id", m["match_id"]).execute()
            st.rerun()
        if st.button("Delete", key=f"d_{m['match_id']}"):
            get_supabase().table("matches").delete().eq("match_id", m["match_id"]).execute(); st.rerun()
with tab3:
    try:
        data = get_supabase().table("matches").select("*").not_.is_("home_score", "null").order("settled_at", desc=True).execute().data or []
    except Exception as e:
        data = []; st.error(str(e))
    if not data: st.info("No settled matches yet.")
    for m in data:
        st.write(f"{m.get('home_team')} {m.get('home_score')}-{m.get('away_score')} {m.get('away_team')} · {m.get('decision')} · {m.get('result')} · NO DRAW hit: {m.get('nodraw_hit')}")
