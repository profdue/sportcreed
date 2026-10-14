#!/usr/bin/env python3
"""
king.py — The Logic.

Input HTML. Extract facts. Run the logic. Show the pick.
Send to Supabase. Track pending. Settle later. See performance.

One file. Two clauses. Three tabs.
"""

import re
import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from supabase import create_client, Client

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="The Logic",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; letter-spacing: -0.02em; }
    .match-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3d 100%);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip      { border-left-color: #64748b; opacity: 0.85; }
    .match-card.win       { border-left-color: #10b981; }
    .match-card.loss      { border-left-color: #ef4444; }
    .match-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 4px 0;
    }
    .match-meta {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    .pick-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .badge-home { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-away { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win  { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .signal-pill {
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 8px;
        background: #0f172a;
        color: #cbd5e1;
        border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .metric-box {
        background: #1a1f2e;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #232a3d;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 0 18px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


def db_insert(row: dict):
    sb = get_supabase()
    res = sb.table("matches").insert(row).execute()
    return res.data[0] if res.data else None


def db_pending():
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .is_("home_score", "null")
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_settled():
    sb = get_supabase()
    res = (sb.table("matches")
             .select("*")
             .not_.is_("home_score", "null")
             .order("match_date", desc=True)
             .execute())
    return res.data or []


def db_settle(match_id: int, home_score: int, away_score: int,
              result: str, dc_hit: int, profit_loss: float):
    sb = get_supabase()
    sb.table("matches").update({
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "dc_hit": dc_hit,
        "profit_loss": profit_loss,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("match_id", match_id).execute()


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — TWO CLAUSES
# ═══════════════════════════════════════════════════════════════════

def compute_signals(f: dict) -> dict:
    """Step 2 — four leaders."""
    table = "home" if (f["home_table_pts"], f["home_table_gd"]) > \
                      (f["away_table_pts"], f["away_table_gd"]) else "away"
    form  = "home" if f["home_form_ppg"] > f["away_form_ppg"] else "away"
    venue = "home" if f["home_home_ppg"] > f["away_away_ppg"] else "away"

    if f["h2h_home_wins"] > f["h2h_away_wins"]:
        h2h = "home"
    elif f["h2h_away_wins"] > f["h2h_home_wins"]:
        h2h = "away"
    else:
        h2h = "nothing"

    return {"table": table, "form": form, "venue": venue, "h2h": h2h}


def decide(signals: dict) -> dict:
    """Step 3 & 4 — direction and stake."""
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")
    table_leader = signals["table"]

    # 4–0
    if home == 4 or away == 4:
        direction = "home" if home == 4 else "away"
        return {
            "direction": direction,
            "stake": "full",
            "qualifying": 1,
            "dissent_signal": None,
            "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    # 3–1
    if (home == 3 and away == 1) or (away == 3 and home == 1):
        dissent = next((k for k, v in signals.items()
                        if v != table_leader and v != "nothing"), None)
        return {
            "direction": table_leader,
            "stake": "half",
            "qualifying": 1,
            "dissent_signal": dissent,
            "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    # 2–2
    if home == 2 and away == 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": "2-2",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    # fewer than two available
    if (home + away) < 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": f"{home}-{away}-{nothing}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    # anything else
    split = f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}"
    return {
        "direction": "skip", "stake": "none", "qualifying": 0,
        "dissent_signal": None, "split": split,
        "home_count": home, "away_count": away, "nothing_count": nothing,
    }


def settle(direction, stake, home_score, away_score):
    """Step 5 — settle."""
    if home_score > away_score:
        result = "home_win"
    elif away_score > home_score:
        result = "away_win"
    else:
        result = "draw"

    if direction == "skip":
        return result, None, 0.0, 0.0

    units = {"full": 1.0, "half": 0.5, "quarter": 0.25, "none": 0.0}.get(stake, 0.0)

    if direction == "home":
        hit = result in ("home_win", "draw")   # DC 1X
    elif direction == "away":
        hit = result in ("away_win", "draw")   # DC X2
    else:
        hit = False

    profit = units if hit else -units
    return result, (1 if hit else 0), units, profit


# ═══════════════════════════════════════════════════════════════════
# HTML EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _int(text):
    if text is None:
        return None
    m = re.search(r"-?\d+", str(text))
    return int(m.group()) if m else None


def _float(text):
    if text is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(text))
    return float(m.group()) if m else None


def extract_facts(html: str) -> dict:
    """
    Parse the SportsGambler HTML into the raw facts.

    Tolerant: if a section is missing, the field stays None and the
    logic will skip or flag accordingly.
    """
    soup = BeautifulSoup(html, "html.parser")
    f = {}

    # --- identity ---------------------------------------------------
    f["match_date"]   = None
    f["kickoff_time"] = None
    f["league"]       = None
    f["country"]      = None
    f["venue"]        = None
    f["home_team"]    = None
    f["away_team"]    = None

    t_top = soup.find("div", class_="t_top")
    if t_top:
        teams = t_top.find_all("div", class_="t_name")
        if len(teams) >= 2:
            f["home_team"] = teams[0].get_text(strip=True)
            f["away_team"] = teams[1].get_text(strip=True)

        info = t_top.find("div", class_="t_info")
        if info:
            links = info.find_all("a", class_="t_info_link")
            if len(links) >= 2:
                league_txt = links[1].get_text(strip=True)
                if " - " in league_txt:
                    _, league_txt = league_txt.split(" - ", 1)
                f["league"] = league_txt
            venue = info.find("span", class_="t_venue")
            if venue:
                f["venue"] = venue.get_text(strip=True)
            date = info.find("span", class_="t_date")
            if date:
                txt = date.get_text(" ", strip=True)
                m = re.search(r"(\d{1,2}\s+\w+)", txt)
                if m:
                    f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m:
                    f["kickoff_time"] = m.group(1)

    # --- standings table (overall) ----------------------------------
    for table in soup.find_all("table", class_="clt-table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 10:
                continue
            team = cells[1]
            if f["home_team"] and f["home_team"].split()[0] in team:
                f["home_table_pos"]    = _int(cells[0])
                f["home_table_played"] = _int(cells[2])
                f["home_table_gf"]     = _int(cells[6])
                f["home_table_ga"]     = _int(cells[7])
                f["home_table_gd"]     = _int(cells[8])
                f["home_table_pts"]    = _int(cells[9])
            elif f["away_team"] and f["away_team"].split()[0] in team:
                f["away_table_pos"]    = _int(cells[0])
                f["away_table_played"] = _int(cells[2])
                f["away_table_gf"]     = _int(cells[6])
                f["away_table_ga"]     = _int(cells[7])
                f["away_table_gd"]     = _int(cells[8])
                f["away_table_pts"]    = _int(cells[9])

    body = soup.get_text(" ", strip=True)

    # --- last 10 form -----------------------------------------------
    if f["home_team"]:
        first = f["home_team"].split()[0]
        m = re.search(rf"{first}[^.]*?(\d+)\s+wins?,?\s+(\d+)\s+losses?,?\s+and\s+(\d+)\s+draws?",
                      body, re.IGNORECASE)
        if not m:
            m = re.search(rf"{first}[^.]*?(\d+)\s+wins?,?\s+(\d+)\s+draws?,?\s+and\s+(\d+)\s+losses?",
                          body, re.IGNORECASE)
            if m:
                f["home_form_wins"]   = int(m.group(1))
                f["home_form_draws"]  = int(m.group(2))
                f["home_form_losses"] = int(m.group(3))
        elif m:
            f["home_form_wins"]   = int(m.group(1))
            f["home_form_losses"] = int(m.group(2))
            f["home_form_draws"]  = int(m.group(3))
        if "home_form_wins" in f:
            f["home_form_pts"] = f["home_form_wins"] * 3 + f.get("home_form_draws", 0)
            f["home_form_ppg"] = round(f["home_form_pts"] / 10, 2)

    if f["away_team"]:
        first = f["away_team"].split()[0]
        m = re.search(rf"{first}[^.]*?(\d+)\s+wins?,?\s+(\d+)\s+losses?,?\s+and\s+(\d+)\s+draws?",
                      body, re.IGNORECASE)
        if not m:
            m = re.search(rf"{first}[^.]*?(\d+)\s+wins?,?\s+(\d+)\s+draws?,?\s+and\s+(\d+)\s+losses?",
                          body, re.IGNORECASE)
            if m:
                f["away_form_wins"]   = int(m.group(1))
                f["away_form_draws"]  = int(m.group(2))
                f["away_form_losses"] = int(m.group(3))
        elif m:
            f["away_form_wins"]   = int(m.group(1))
            f["away_form_losses"] = int(m.group(2))
            f["away_form_draws"]  = int(m.group(3))
        if "away_form_wins" in f:
            f["away_form_pts"] = f["away_form_wins"] * 3 + f.get("away_form_draws", 0)
            f["away_form_ppg"] = round(f["away_form_pts"] / 10, 2)

    # --- H2H --------------------------------------------------------
    m = re.search(
        r"past\s+(\d+)\s+H2H[^.]*?(\d+)\s+triumphs?\s+for\s+([\w\s]+?),?\s+(\d+)\s+for\s+([\w\s]+?)\s+and\s+(\d+)\s+draws?",
        body, re.IGNORECASE
    )
    if m:
        f["h2h_total"] = int(m.group(1))
        home_w = int(m.group(2))
        home_name = m.group(3).strip()
        away_w = int(m.group(4))
        f["h2h_draws"] = int(m.group(6))
        if f["home_team"] and f["home_team"].split()[0] in home_name:
            f["h2h_home_wins"] = home_w
            f["h2h_away_wins"] = away_w
        else:
            f["h2h_home_wins"] = away_w
            f["h2h_away_wins"] = home_w

    # --- venue-only PPG --------------------------------------------
    if f["home_team"]:
        first = f["home_team"].split()[0]
        m = re.search(
            rf"{first}[^.]*?(\d+)\s+victor(?:y|ies),?\s+(\d+)\s+(?:losses|defeats)\s+and\s+(\d+)\s+draws",
            body, re.IGNORECASE
        )
        if m:
            w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pts = w * 3 + d
            f["home_home_pts"]   = pts
            f["home_home_ppg"]   = round(pts / 10, 2)
            f["home_home_wins"]  = w
            f["home_home_draws"] = d
            f["home_home_losses"] = l

    if f["away_team"]:
        first = f["away_team"].split()[0]
        m = re.search(
            rf"{first}[^.]*?(\d+)\s+victor(?:y|ies),?\s+(\d+)\s+(?:losses|defeats)\s+and\s+(\d+)\s+draws",
            body, re.IGNORECASE
        )
        if m:
            w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pts = w * 3 + d
            f["away_away_pts"]   = pts
            f["away_away_ppg"]   = round(pts / 10, 2)
            f["away_away_wins"]  = w
            f["away_away_draws"] = d
            f["away_away_losses"] = l

    # --- defaults so the logic can run -----------------------------
    for k in [
        "home_table_pts", "away_table_pts", "home_table_gd", "away_table_gd",
        "home_form_ppg", "away_form_ppg", "home_home_ppg", "away_away_ppg",
        "h2h_home_wins", "h2h_away_wins",
    ]:
        f.setdefault(k, 0)

    return f


# ═══════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════

def signal_pill(name, value):
    cls = value if value in ("home", "away", "nothing") else "nothing"
    label = {"home": "H", "away": "A", "nothing": "—"}.get(value, "—")
    return f'<span class="signal-pill {cls}">{name}: {label}</span>'


def render_pending_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    direction = m.get("direction", "skip")
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "")
    date = m.get("match_date", "")

    if direction == "home":
        card_cls, badge_cls = "pick-home", "badge-home"
        pick_text = f"HOME — {home} (DC 1X)"
    elif direction == "away":
        card_cls, badge_cls = "pick-away", "badge-away"
        pick_text = f"AWAY — {away} (DC X2)"
    else:
        card_cls, badge_cls = "skip", "badge-skip"
        pick_text = "SKIP"

    pills = "".join([
        signal_pill("Table", m.get("signal_table", "nothing")),
        signal_pill("Form",  m.get("signal_form",  "nothing")),
        signal_pill("Venue", m.get("signal_venue", "nothing")),
        signal_pill("H2H",   m.get("signal_h2h",   "nothing")),
    ])

    st.markdown(f"""
    <div class="match-card {card_cls}">
        <div class="match-title">{home} vs {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge_cls}">{pick_text}</span>
        <div class="signal-row">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


def render_settled_card(m: dict):
    home = m.get("home_team", "?")
    away = m.get("away_team", "?")
    hs = m.get("home_score")
    aws = m.get("away_score")
    dc_hit = m.get("dc_hit")
    profit = m.get("profit_loss") or 0.0
    stake = m.get("stake", "none")
    split = m.get("agreement_split", "?")
    league = m.get("league", "")
    date = m.get("match_date", "")

    if dc_hit == 1:
        cls, badge, verdict = "win", "badge-win", "WIN"
    elif dc_hit == 0:
        cls, badge, verdict = "loss", "badge-loss", "LOSS"
    else:
        cls, badge, verdict = "skip", "badge-skip", "NO BET"

    pl = f"{profit:+.2f}u" if dc_hit is not None else "—"

    st.markdown(f"""
    <div class="match-card {cls}">
        <div class="match-title">{home} {hs}–{aws} {away}</div>
        <div class="match-meta">{league} · {date} · split {split} · stake {stake}</div>
        <span class="pick-badge {badge}">{verdict} · {pl}</span>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# COLUMN MAPPERS
# ═══════════════════════════════════════════════════════════════════

def signals_to_columns(signals):
    return {
        "signal_table": signals["table"],
        "signal_form":  signals["form"],
        "signal_venue": signals["venue"],
        "signal_h2h":   signals["h2h"],
    }


def decision_columns(d):
    return {
        "direction":               d["direction"],
        "stake":                   d["stake"],
        "qualifying":              d["qualifying"],
        "dissent_signal":          d["dissent_signal"],
        "agreement_split":         d["split"],
        "agreement_home_count":    d["home_count"],
        "agreement_away_count":    d["away_count"],
        "agreement_nothing_count": d["nothing_count"],
        "table_leader":            d["direction"] if d["direction"] != "skip" else None,
    }


def build_row(facts, signals, decision):
    row = dict(facts)
    row.update(signals_to_columns(signals))
    row.update(decision_columns(decision))
    row["extracted_at"] = datetime.now(timezone.utc).isoformat()

    allowed = {
        "match_date", "kickoff_time", "league", "country", "venue",
        "home_team", "away_team",
        "home_table_pos", "away_table_pos", "home_table_pts", "away_table_pts",
        "home_table_gd", "away_table_gd", "home_table_gf", "away_table_gf",
        "home_table_ga", "away_table_ga", "home_table_played", "away_table_played",
        "home_home_played", "home_home_wins", "home_home_draws", "home_home_losses",
        "home_home_gf", "home_home_ga", "home_home_pts", "home_home_ppg",
        "away_away_played", "away_away_wins", "away_away_draws", "away_away_losses",
        "away_away_gf", "away_away_ga", "away_away_pts", "away_away_ppg",
        "home_form_wins", "home_form_draws", "home_form_losses", "home_form_pts",
        "home_form_ppg", "home_form_gf_per_game", "home_form_ga_per_game",
        "home_form_shots_on_target_per_game", "home_form_attempts_per_game",
        "home_form_corners_per_game", "home_form_possession",
        "away_form_wins", "away_form_draws", "away_form_losses", "away_form_pts",
        "away_form_ppg", "away_form_gf_per_game", "away_form_ga_per_game",
        "away_form_shots_on_target_per_game", "away_form_attempts_per_game",
        "away_form_corners_per_game", "away_form_possession",
        "home_conversion_rate", "away_conversion_rate",
        "home_draw_rate", "away_draw_rate",
        "h2h_total", "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_last_result",
        "odds_home", "odds_draw", "odds_away", "odds_home_implied", "odds_away_implied",
        "signal_table", "signal_form", "signal_venue", "signal_h2h",
        "agreement_split", "agreement_home_count", "agreement_away_count",
        "agreement_nothing_count", "table_leader", "dissent_signal",
        "direction", "pick", "stake", "qualifying",
        "source_url", "extracted_at", "notes",
    }
    return {k: v for k, v in row.items() if k in allowed}


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_perf = st.tabs(["➕ New Match", "⏳ Pending", "📊 Performance"])


# ─────────────────────────────────────────────────────────────────
# TAB 1 — NEW MATCH
# ─────────────────────────────────────────────────────────────────
with tab_new:
    st.markdown("### Paste the match HTML")
    st.caption("Paste the full SportsGambler preview HTML. The app extracts the facts, runs the logic, and shows the pick.")

    html_in = st.text_area(
        "HTML", height=220, label_visibility="collapsed",
        placeholder="<html> ... </html>"
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run = st.button("Run the Logic", type="primary", use_container_width=True)

    if run and html_in.strip():
        with st.spinner("Extracting facts..."):
            try:
                facts = extract_facts(html_in)
                signals = compute_signals(facts)
                decision = decide(signals)
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()

        st.markdown("### The Pick")
        preview = {
            **facts,
            **signals_to_columns(signals),
            **decision_columns(decision),
        }
        render_pending_card(preview)

        st.markdown("### Extracted Facts")
        st.json({k: v for k, v in facts.items() if v is not None})

        row = build_row(facts, signals, decision)
        with st.spinner("Saving to Supabase..."):
            try:
                db_insert(row)
                st.success("Match saved to pending.")
            except Exception as e:
                st.error(f"Supabase insert failed: {e}")


# ─────────────────────────────────────────────────────────────────
# TAB 2 — PENDING
# ─────────────────────────────────────────────────────────────────
with tab_pending:
    st.markdown("### Pending matches")
    st.caption("Enter the final score to settle. The match moves to Performance.")

    try:
        pending = db_pending()
    except Exception as e:
        st.error(f"Could not load pending: {e}")
        pending = []

    if not pending:
        st.info("No pending matches.")
    else:
        for m in pending:
            render_pending_card(m)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    hs = st.number_input("Home", min_value=0, step=1,
                                         key=f"h_{m['match_id']}")
                with c2:
                    aws = st.number_input("Away", min_value=0, step=1,
                                          key=f"a_{m['match_id']}")
                with c3:
                    st.write("")
                    st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        result, dc_hit, units, profit = settle(
                            m.get("direction"), m.get("stake"),
                            int(hs), int(aws)
                        )
                        db_settle(
                            m["match_id"], int(hs), int(aws),
                            result,
                            dc_hit if dc_hit is not None else 0,
                            profit
                        )
                        st.success("Settled.")
                        st.rerun()


# ─────────────────────────────────────────────────────────────────
# TAB 3 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### Performance")
    try:
        settled = db_settled()
    except Exception as e:
        st.error(f"Could not load settled: {e}")
        settled = []

    if not settled:
        st.info("No settled matches yet.")
    else:
        df = pd.DataFrame(settled)
        df["dc_hit"] = pd.to_numeric(df["dc_hit"], errors="coerce")
        df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0)

        qualifying = df[df["qualifying"] == 1]
        bets = qualifying[qualifying["dc_hit"].notna()]

        total = len(bets)
        wins = int((bets["dc_hit"] == 1).sum())
        losses = int((bets["dc_hit"] == 0).sum())
        net = float(df["profit_loss"].sum())
        hit_rate = (wins / total * 100) if total else 0.0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{total}</div>'
                        f'<div class="metric-label">Bets</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{wins}</div>'
                        f'<div class="metric-label">Wins</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{losses}</div>'
                        f'<div class="metric-label">Losses</div></div>', unsafe_allow_html=True)
        with c4:
            color = "#10b981" if net >= 0 else "#ef4444"
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color:{color}">{net:+.2f}u</div>'
                        f'<div class="metric-label">Net P/L</div></div>', unsafe_allow_html=True)

        st.markdown("")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{hit_rate:.1f}%</div>'
                        f'<div class="metric-label">Hit rate</div></div>', unsafe_allow_html=True)
        with c6:
            skips = int((df["qualifying"] == 0).sum())
            st.markdown(f'<div class="metric-box"><div class="metric-value">{skips}</div>'
                        f'<div class="metric-label">Skips</div></div>', unsafe_allow_html=True)
        with c7:
            full = int((bets["stake"] == "full").sum()) if not bets.empty else 0
            half = int((bets["stake"] == "half").sum()) if not bets.empty else 0
            st.markdown(f'<div class="metric-box"><div class="metric-value">{full}/{half}</div>'
                        f'<div class="metric-label">Full / Half</div></div>', unsafe_allow_html=True)

        st.markdown("### By agreement split")
        if not qualifying.empty:
            by_split = (qualifying
                        .groupby("agreement_split")
                        .agg(total=("match_id", "count"),
                             wins=("dc_hit", lambda s: int((s == 1).sum())),
                             losses=("dc_hit", lambda s: int((s == 0).sum())),
                             net=("profit_loss", "sum"))
                        .reset_index())
            st.dataframe(by_split, use_container_width=True, hide_index=True)

        st.markdown("### By dissent signal")
        dissent_df = qualifying[qualifying["dissent_signal"].notna()]
        if not dissent_df.empty:
            by_dissent = (dissent_df
                          .groupby("dissent_signal")
                          .agg(total=("match_id", "count"),
                               wins=("dc_hit", lambda s: int((s == 1).sum())),
                               losses=("dc_hit", lambda s: int((s == 0).sum())),
                               net=("profit_loss", "sum"))
                          .reset_index())
            st.dataframe(by_dissent, use_container_width=True, hide_index=True)
        else:
            st.info("No dissent-signal data yet.")

        st.markdown("### All settled matches")
        for m in settled:
            render_settled_card(m)
