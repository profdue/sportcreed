"""
v4 RAW-ONLY Predictor — Streamlit app.
Implements the exact 100-point logic from the strategy breakdown.
Only uses prematch fields stored in matches_raw.

Requires:
  st.secrets["SUPABASE_URL"]
  st.secrets["SUPABASE_KEY"]  (service_role key)

Table: public.matches_raw (one row per match)
View:  public.v4_performance
"""

import os
import re
import json
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="v4 Raw Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.75rem 2rem; color: #fff;
        margin-bottom: 1.25rem; border-left: 6px solid #10b981;
    }
    .header-title { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .header-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem; }
    .header-badge {
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 999px;
        background: #064e3b; color: #6ee7b7; font-weight: 700;
        font-size: 0.75rem; letter-spacing: 1.5px; text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .match-card {
        background: #0f172a; border-radius: 14px; padding: 1.25rem 1.5rem;
        margin-bottom: 1rem; border: 1px solid #1e293b;
    }
    .match-card-bet { border-left: 6px solid #10b981; }
    .match-card-nobet { border-left: 6px solid #64748b; }

    .match-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 1rem; flex-wrap: wrap; gap: 0.75rem;
    }
    .match-teams { font-size: 1.35rem; font-weight: 800; color: #f1f5f9; margin: 0; }
    .match-meta { color: #64748b; font-size: 0.85rem; margin-top: 0.25rem; }

    .call-pill {
        display: inline-block; padding: 0.55rem 1.1rem; border-radius: 999px;
        font-weight: 800; font-size: 0.95rem; letter-spacing: 0.5px;
    }
    .call-win { background: #064e3b; color: #6ee7b7; }
    .call-dc { background: #78350f; color: #fde68a; }
    .call-nobet { background: #1e293b; color: #94a3b8; }

    .factor-grid {
        display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.6rem; margin-top: 0.75rem;
    }
    .factor-box {
        background: #020617; border-radius: 10px; padding: 0.65rem 0.85rem;
        border: 1px solid #1e293b;
    }
    .factor-label {
        color: #64748b; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase;
    }
    .factor-value {
        color: #3b82f6; font-size: 1.15rem; font-weight: 800;
        margin-top: 0.2rem;
    }
    .factor-value-home { color: #10b981; }
    .factor-value-away { color: #f59e0b; }

    .stat-row {
        display: flex; justify-content: space-between; padding: 0.4rem 0;
        border-bottom: 1px solid #1e293b; font-size: 0.9rem;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: #94a3b8; }
    .stat-value { color: #f1f5f9; font-weight: 700; }

    .trigger-badge {
        display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px;
        margin-right: 0.35rem; margin-top: 0.35rem;
    }
    .trigger-override { background: #7f1d1d; color: #fecaca; }
    .trigger-collapse { background: #78350f; color: #fde68a; }
    .trigger-doubt { background: #1e3a8a; color: #bfdbfe; }
    .trigger-none { background: #1e293b; color: #64748b; }

    .section-title {
        font-size: 0.8rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 1.5rem 0 0.75rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 14px; padding: 1.25rem 1.5rem; border: 1px solid #1e293b;
    }
    .metric-value {
        font-size: 2.2rem; font-weight: 800; color: #3b82f6;
        margin: 0.25rem 0; line-height: 1;
    }
    .metric-label {
        color: #64748b; font-size: 0.75rem; font-weight: 700;
        letter-spacing: 1.5px; text-transform: uppercase;
    }
    .metric-detail { color: #94a3b8; font-size: 0.85rem; margin-top: 0.35rem; }

    .stButton button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; font-weight: 700; border-radius: 10px;
        border: none; padding: 0.6rem 1.5rem; font-size: 1rem;
    }
    .stButton button:hover { background: linear-gradient(135deg, #059669 0%, #047857 100%); }

    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] {
        background: #0f172a; border-radius: 10px; padding: 0.5rem 1.2rem;
        color: #94a3b8; font-weight: 600; border: 1px solid #1e293b;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important; color: #10b981 !important;
        border-color: #10b981 !important;
    }

    .empty-state {
        background: #0f172a; border-radius: 14px; padding: 2rem;
        text-align: center; color: #64748b; border: 1px dashed #334155;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SUPABASE
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    diag = {"url": None, "ok": False, "error": None}
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        diag["url"] = url
        diag["ok"] = True
        return create_client(url, key), diag
    except Exception as e:
        diag["error"] = str(e)
        return None, diag


# ============================================================================
# v4 MODEL — EXACT LOGIC
# ============================================================================
def calc_f1(row):
    """F1: League Momentum — 20 pts"""
    home_pts = row.get("home_points") or 0
    away_pts = row.get("away_points") or 0
    home_gd = row.get("home_gd") or 0
    away_gd = row.get("away_gd") or 0

    pts_gap = home_pts - away_pts
    gd_gap = home_gd - away_gd

    f1_home = 10 + (pts_gap * 1.33) + (gd_gap * 0.66)
    f1_away = 10 - (pts_gap * 1.33) - (gd_gap * 0.66)

    f1_home = max(0, min(20, f1_home))
    f1_away = max(0, min(20, f1_away))

    away_collapse = False
    away_away_pts = row.get("away_away_points")
    away_away_played = row.get("away_away_played") or 0
    if away_away_pts == 0 and away_away_played >= 3:
        f1_home = min(20, f1_home + 8)
        away_collapse = True

    return f1_home, f1_away, away_collapse


def calc_f2(last5_json):
    """F2: Current Form Last 5 — 25 pts"""
    if not last5_json:
        return 7.5
    raw = 0
    for m in last5_json:
        result = m.get("result", "L")
        if result == "W":
            raw += 5
            if (m.get("score_for") or 0) >= 3:
                raw += 1
        elif result == "D":
            raw += 2
        else:
            if (m.get("score_against") or 0) >= 3:
                raw -= 1
    raw = max(0, min(15, raw))
    return (raw / 15) * 25


def calc_f3(win_pct, avg_scored, avg_conceded):
    """F3: Venue Split — 15 pts"""
    win_pts = ((win_pct or 0) / 100) * 10
    xg_edge = (avg_scored or 0) - (avg_conceded or 0)
    if xg_edge > 0.5:
        xg_pts = 5
    elif xg_edge > 0:
        xg_pts = 2
    else:
        xg_pts = 0
    return min(15, win_pts + xg_pts)


def calc_f4(top_scorer, top_assister, injuries, xi):
    """F4: Availability — 15 pts"""
    f4 = 15
    injured_names = [i.get("player", "") for i in (injuries or [])]
    doubt_names = [i.get("player", "") for i in (injuries or []) if i.get("status") == "doubt"]
    xi = xi or []

    if top_scorer and top_scorer in injured_names and top_scorer not in xi:
        f4 -= 5
    if top_assister and top_assister in injured_names and top_assister not in xi:
        f4 -= 4

    doubted_starter = False
    for name in doubt_names:
        if name in xi:
            if name == top_scorer or name == top_assister:
                f4 += 5
                doubted_starter = True

    return max(0, min(20, f4)), doubted_starter


def calc_f5(h2h_home_wins, h2h_away_wins, h2h_total):
    """F5: H2H Psychology — 10 pts"""
    if not h2h_total or h2h_total == 0:
        return 5.0, 5.0
    f5_home = (h2h_home_wins / h2h_total) * 10
    f5_away = (h2h_away_wins / h2h_total) * 10
    return f5_home, f5_away


def calc_f6(home_avg, away_avg):
    """F6: Attack Profile — 15 pts"""
    if (home_avg or 0) > (away_avg or 0):
        return 11, 7
    elif (home_avg or 0) < (away_avg or 0):
        return 7, 11
    return 8, 8


def apply_f1_f5_override(f1_home, f1_away, f5_home, f5_away):
    """v4 Override: F1 vs F5 conflict"""
    f1_gap = abs(f1_home - f1_away)
    f5_gap = abs(f5_home - f5_away)
    f1_leader = "home" if f1_home > f1_away else "away"
    f5_leader = "home" if f5_home > f5_away else "away"

    conflict = f1_leader != f5_leader and f1_gap >= 12 and f5_gap >= 6
    override = False
    if conflict and f1_gap > f5_gap:
        if f1_leader == "home":
            f5_home, f5_away = 10, 0
        else:
            f5_home, f5_away = 0, 10
        override = True

    return f5_home, f5_away, conflict, override


def calc_expected_total(row):
    """O/U expected total from prematch averages"""
    h_avg_scored = row.get("home_home_last10_avg_scored") or 1.0
    a_avg_conceded = row.get("away_away_last10_avg_conceded") or 1.0
    a_avg_scored = row.get("away_away_last10_avg_scored") or 1.0
    h_avg_conceded = row.get("home_home_last10_avg_conceded") or 1.0
    return (h_avg_scored + a_avg_conceded + a_avg_scored + h_avg_conceded) / 2


def predict_v4(row):
    """Run the full v4 model on one row"""
    f1_home, f1_away, away_collapse = calc_f1(row)
    f2_home = calc_f2(row.get("home_last5"))
    f2_away = calc_f2(row.get("away_last5"))
    f3_home = calc_f3(
        row.get("home_home_win_pct"),
        row.get("home_home_last10_avg_scored"),
        row.get("home_home_last10_avg_conceded"),
    )
    f3_away = calc_f3(
        row.get("away_away_win_pct"),
        row.get("away_away_last10_avg_scored"),
        row.get("away_away_last10_avg_conceded"),
    )
    f4_home, doubt_home = calc_f4(
        row.get("home_top_scorer"),
        row.get("home_top_assister"),
        row.get("home_injuries"),
        row.get("home_xi"),
    )
    f4_away, doubt_away = calc_f4(
        row.get("away_top_scorer"),
        row.get("away_top_assister"),
        row.get("away_injuries"),
        row.get("away_xi"),
    )
    doubted_starter = doubt_home or doubt_away
    f5_home, f5_away = calc_f5(
        row.get("h2h_home_wins") or 0,
        row.get("h2h_away_wins") or 0,
        len(row.get("h2h") or []),
    )
    f6_home, f6_away = calc_f6(
        row.get("home_last10_avg_scored"),
        row.get("away_last10_avg_scored"),
    )

    f5_home, f5_away, conflict, override = apply_f1_f5_override(
        f1_home, f1_away, f5_home, f5_away
    )

    home_total = f1_home + f2_home + f3_home + f4_home + f5_home + f6_home
    away_total = f1_away + f2_away + f3_away + f4_away + f5_away + f6_away
    gap = abs(home_total - away_total)

    if gap > 25:
        call_1x2 = "Straight Win Home" if home_total > away_total else "Straight Win Away"
    elif gap >= 15:
        call_1x2 = "Double Chance 1X" if home_total > away_total else "Double Chance X2"
    else:
        call_1x2 = "NO BET"

    expected = calc_expected_total(row)
    if expected < 2.5:
        call_ou = "Under 2.5"
    elif expected > 3.3:
        call_ou = "Over 2.5"
    else:
        call_ou = "No Bet"

    if away_collapse or doubted_starter:
        call_ou = "Over 2.5"

    return {
        "f1_home": round(f1_home, 2), "f1_away": round(f1_away, 2),
        "f1_gap": round(abs(f1_home - f1_away), 2),
        "f1_leader": "home" if f1_home > f1_away else "away",
        "f2_home": round(f2_home, 2), "f2_away": round(f2_away, 2),
        "f3_home": round(f3_home, 2), "f3_away": round(f3_away, 2),
        "f4_home": round(f4_home, 2), "f4_away": round(f4_away, 2),
        "f5_home": round(f5_home, 2), "f5_away": round(f5_away, 2),
        "f5_gap": round(abs(f5_home - f5_away), 2),
        "f5_leader": "home" if f5_home > f5_away else "away",
        "f6_home": round(f6_home, 2), "f6_away": round(f6_away, 2),
        "home_total": round(home_total, 2),
        "away_total": round(away_total, 2),
        "total_gap": round(gap, 2),
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "away_collapse": away_collapse,
        "doubted_starter": doubted_starter,
        "call_1x2": call_1x2,
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
    }


def save_prediction(sb, match_id, result):
    if sb is None:
        return False, "no client"
    try:
        sb.table("matches_raw").update(result).eq("id", match_id).execute()
        return True, "saved"
    except Exception as e:
        return False, str(e)


def save_actual(sb, match_id, hg, ag):
    if sb is None:
        return False, "no client"
    try:
        row_resp = sb.table("matches_raw").select("call_1x2").eq("id", match_id).single().execute()
        row = row_resp.data
        call = row.get("call_1x2") or ""

        if hg > ag:
            actual = "Home"
        elif hg < ag:
            actual = "Away"
        else:
            actual = "Draw"

        is_correct = None
        if call == "NO BET":
            is_correct = None
        elif call == "Straight Win Home":
            is_correct = actual == "Home"
        elif call == "Straight Win Away":
            is_correct = actual == "Away"
        elif call == "Double Chance 1X":
            is_correct = actual in ("Home", "Draw")
        elif call == "Double Chance X2":
            is_correct = actual in ("Away", "Draw")

        sb.table("matches_raw").update({
            "actual_home_goals": hg,
            "actual_away_goals": ag,
            "is_correct_1x2": is_correct,
        }).eq("id", match_id).execute()
        return True, "saved"
    except Exception as e:
        return False, str(e)


# ============================================================================
# RENDER HELPERS
# ============================================================================
def call_pill(call):
    if call.startswith("Straight Win"):
        return f'<span class="call-pill call-win">{call}</span>'
    elif call.startswith("Double Chance"):
        return f'<span class="call-pill call-dc">{call}</span>'
    else:
        return f'<span class="call-pill call-nobet">{call}</span>'


def render_triggers(row):
    badges = []
    if row.get("f1_f5_override"):
        badges.append('<span class="trigger-badge trigger-override">F1 OVERRIDE</span>')
    if row.get("away_collapse"):
        badges.append('<span class="trigger-badge trigger-collapse">AWAY COLLAPSE</span>')
    if row.get("doubted_starter"):
        badges.append('<span class="trigger-badge trigger-doubt">DOUBT STARTER</span>')
    if not badges:
        badges.append('<span class="trigger-badge trigger-none">NO TRIGGERS</span>')
    return "".join(badges)


def render_factor_grid(row):
    return f"""
    <div class="factor-grid">
        <div class="factor-box">
            <div class="factor-label">F1 Momentum</div>
            <div class="factor-value factor-value-home">{row.get('f1_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f1_away', 0):.1f}</div>
        </div>
        <div class="factor-box">
            <div class="factor-label">F2 Form L5</div>
            <div class="factor-value factor-value-home">{row.get('f2_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f2_away', 0):.1f}</div>
        </div>
        <div class="factor-box">
            <div class="factor-label">F3 Venue</div>
            <div class="factor-value factor-value-home">{row.get('f3_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f3_away', 0):.1f}</div>
        </div>
        <div class="factor-box">
            <div class="factor-label">F4 Availability</div>
            <div class="factor-value factor-value-home">{row.get('f4_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f4_away', 0):.1f}</div>
        </div>
        <div class="factor-box">
            <div class="factor-label">F5 H2H</div>
            <div class="factor-value factor-value-home">{row.get('f5_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f5_away', 0):.1f}</div>
        </div>
        <div class="factor-box">
            <div class="factor-label">F6 Attack</div>
            <div class="factor-value factor-value-home">{row.get('f6_home', 0):.1f}</div>
            <div class="factor-value factor-value-away" style="font-size:0.8rem;">{row.get('f6_away', 0):.1f}</div>
        </div>
    </div>
    """


def render_match_card(row, show_actual=False):
    gap = row.get("total_gap", 0)
    card_class = "match-card match-card-bet" if row.get("call_1x2") != "NO BET" else "match-card match-card-nobet"

    home_total = row.get("home_total", 0)
    away_total = row.get("away_total", 0)

    actual_html = ""
    if show_actual and row.get("actual_home_goals") is not None:
        correct = row.get("is_correct_1x2")
        correct_str = "✅" if correct else ("⬜" if correct is None else "❌")
        actual_html = f"""
        <div class="stat-row">
            <span class="stat-label">Actual</span>
            <span class="stat-value">{row.get('actual_home_goals')}-{row.get('actual_away_goals')} {correct_str}</span>
        </div>
        """

    html = f"""
    <div class="{card_class}">
        <div class="match-header">
            <div>
                <div class="match-teams">{row.get('home_team')} <span style="color:#64748b;">vs</span> {row.get('away_team')}</div>
                <div class="match-meta">{row.get('match_date', '')} · {row.get('league_name', '')}</div>
            </div>
            <div>{call_pill(row.get('call_1x2', 'NO BET'))}</div>
        </div>

        <div style="display:flex; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.75rem;">
            {render_triggers(row)}
        </div>

        <div class="factor-grid" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">
            <div class="factor-box">
                <div class="factor-label">Gap</div>
                <div class="factor-value">{gap:.1f}</div>
            </div>
            <div class="factor-box">
                <div class="factor-label">Home Total</div>
                <div class="factor-value factor-value-home">{home_total:.1f}</div>
            </div>
            <div class="factor-box">
                <div class="factor-label">Away Total</div>
                <div class="factor-value factor-value-away">{away_total:.1f}</div>
            </div>
            <div class="factor-box">
                <div class="factor-label">O/U Call</div>
                <div class="factor-value" style="font-size:0.95rem;">{row.get('call_ou', '—')}</div>
            </div>
            <div class="factor-box">
                <div class="factor-label">Expected</div>
                <div class="factor-value">{row.get('expected_total', 0):.2f}</div>
            </div>
        </div>

        {render_factor_grid(row)}

        {actual_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# UI
# ============================================================================
def main():
    st.markdown("""
    <div class="header-banner">
        <div class="header-title">⚽ v4 Raw-Only Predictor</div>
        <div class="header-sub">F1–F6 prematch logic · No xG · No possession · No post-match data</div>
        <div class="header-badge">9/10 on 1X2/DC out-of-sample</div>
    </div>
    """, unsafe_allow_html=True)

    sb, diag = get_supabase()

    if sb is None:
        st.error(f"Supabase connection failed: {diag.get('error')}")
        st.info("Add SUPABASE_URL and SUPABASE_KEY to Streamlit secrets.")
        return

    # Load matches
    try:
        resp = sb.table("matches_raw").select("*").order("match_date", desc=True).execute()
        rows = resp.data or []
    except Exception as e:
        st.error(f"Query failed: {e}")
        return

    tabs = st.tabs(["🎯 Predictions", "📝 Pending", "📊 Records", "📈 Performance"])

    # ============================================================
    # TAB 1: PREDICTIONS
    # ============================================================
    with tabs[0]:
        if not rows:
            st.markdown("""
            <div class="empty-state">
                <h3>No matches in database</h3>
                <p>Insert rows into <code>matches_raw</code> to get started.</p>
            </div>
            """, unsafe_allow_html=True)
            return

        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"### {len(rows)} matches loaded")
        with col_b:
            run_all = st.button("▶️ Run v4 on all", type="primary", use_container_width=True)

        if run_all:
            progress = st.progress(0)
            status = st.empty()
            for i, row in enumerate(rows):
                result = predict_v4(row)
                save_prediction(sb, row["id"], result)
                row.update(result)
                progress.progress((i + 1) / len(rows))
                status.text(f"Processed {i + 1}/{len(rows)}: {row.get('home_team')} vs {row.get('away_team')}")
            st.success(f"✅ Ran v4 on {len(rows)} matches")
            st.rerun()

        # Filters
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            filter_call = st.selectbox(
                "Filter by call",
                ["All", "Straight Win", "Double Chance", "NO BET"],
            )
        with c2:
            filter_override = st.selectbox(
                "Filter by override",
                ["All", "Override triggered", "No override"],
            )
        with c3:
            filter_date = st.text_input("Filter by date (YYYY-MM-DD)", "")

        filtered = rows
        if filter_call != "All":
            filtered = [r for r in filtered if filter_call.lower() in (r.get("call_1x2") or "").lower()]
        if filter_override == "Override triggered":
            filtered = [r for r in filtered if r.get("f1_f5_override")]
        elif filter_override == "No override":
            filtered = [r for r in filtered if not r.get("f1_f5_override")]
        if filter_date:
            filtered = [r for r in filtered if filter_date in (r.get("match_date") or "")]

        st.markdown(f"#### Showing {len(filtered)} matches")

        for row in filtered:
            render_match_card(row)

    # ============================================================
    # TAB 2: PENDING
    # ============================================================
    with tabs[1]:
        pending = [r for r in rows if r.get("actual_home_goals") is None]
        if not pending:
            st.markdown("""
            <div class="empty-state">
                <h3>No pending matches</h3>
                <p>All matches have results entered.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"### {len(pending)} pending matches")
            for row in pending:
                with st.expander(f"{row.get('home_team')} vs {row.get('away_team')} — {row.get('match_date')}"):
                    st.markdown(f"**Call:** {row.get('call_1x2', '—')} · **O/U:** {row.get('call_ou', '—')}")
                    c1, c2 = st.columns(2)
                    hg = c1.number_input("Home goals", 0, 15, 0, key=f"hg_{row['id']}")
                    ag = c2.number_input("Away goals", 0, 15, 0, key=f"ag_{row['id']}")
                    if st.button("Submit result", key=f"sub_{row['id']}"):
                        ok, msg = save_actual(sb, row["id"], hg, ag)
                        if ok:
                            st.success("Saved")
                            st.rerun()
                        else:
                            st.error(msg)

    # ============================================================
    # TAB 3: RECORDS
    # ============================================================
    with tabs[2]:
        settled = [r for r in rows if r.get("actual_home_goals") is not None]
        if not settled:
            st.markdown("""
            <div class="empty-state">
                <h3>No settled matches yet</h3>
                <p>Enter results in the Pending tab to see records.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            df = pd.DataFrame([{
                "Date": r.get("match_date"),
                "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                "Gap": r.get("total_gap"),
                "1X2 Call": r.get("call_1x2"),
                "O/U Call": r.get("call_ou"),
                "Score": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                "Correct": "✅" if r.get("is_correct_1x2") else ("⬜" if r.get("is_correct_1x2") is None else "❌"),
                "Override": "✅" if r.get("f1_f5_override") else "",
                "Collapse": "✅" if r.get("away_collapse") else "",
                "Doubt": "✅" if r.get("doubted_starter") else "",
            } for r in settled])
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ============================================================
    # TAB 4: PERFORMANCE
    # ============================================================
    with tabs[3]:
        settled = [r for r in rows if r.get("actual_home_goals") is not None]
        if not settled:
            st.markdown("""
            <div class="empty-state">
                <h3>No performance data yet</h3>
                <p>Enter results to compute accuracy.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            placed = [r for r in settled if r.get("call_1x2") != "NO BET"]
            correct = sum(1 for r in placed if r.get("is_correct_1x2"))
            override_games = [r for r in settled if r.get("f1_f5_override")]
            override_correct = sum(1 for r in override_games if r.get("is_correct_1x2"))
            collapse_games = [r for r in settled if r.get("away_collapse")]
            doubt_games = [r for r in settled if r.get("doubted_starter")]

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Settled</div>
                    <div class="metric-value">{len(settled)}</div>
                    <div class="metric-detail">Matches with results</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                acc = (correct / len(placed) * 100) if placed else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">1X2 Accuracy</div>
                    <div class="metric-value">{acc:.0f}%</div>
                    <div class="metric-detail">{correct} of {len(placed)} bets</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                oacc = (override_correct / len(override_games) * 100) if override_games else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Override Accuracy</div>
                    <div class="metric-value">{oacc:.0f}%</div>
                    <div class="metric-detail">{override_correct} of {len(override_games)} triggers</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Triggers</div>
                    <div class="metric-value">{len(collapse_games) + len(doubt_games)}</div>
                    <div class="metric-detail">Collapse: {len(collapse_games)} · Doubt: {len(doubt_games)}</div>
                </div>
                """, unsafe_allow_html=True)

            # Breakdown by call type
            st.markdown("### Accuracy by call type")
            by_call = {}
            for r in placed:
                c = r.get("call_1x2", "Unknown")
                if c not in by_call:
                    by_call[c] = {"correct": 0, "total": 0}
                by_call[c]["total"] += 1
                if r.get("is_correct_1x2"):
                    by_call[c]["correct"] += 1

            for call, stats in by_call.items():
                acc = (stats["correct"] / stats["total"] * 100) if stats["total"] else 0
                st.markdown(f"**{call}**: {stats['correct']}/{stats['total']} ({acc:.0f}%)")


main()
