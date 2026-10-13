"""
Underking — manual input prediction tool.

You enter the match data by hand.
The app predicts. The prediction is the point.
Saving is a side effect — the row is logged so you can settle later.

Flow:
    input -> decide() -> DISPLAY PREDICTION -> upsert -> show stored status
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Underking",
    page_icon="👑",
    layout="wide",
)

# ============================================================================
# DISPLAY STYLES
# ============================================================================
st.markdown("""
<style>
    .uk-bet {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
    }
    .uk-pass {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
    }
    .uk-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        font-weight: 700;
        color: #6ee7b7;
        text-transform: uppercase;
    }
    .uk-label-grey {
        font-size: 0.7rem;
        letter-spacing: 2px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
    }
    .uk-pick {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.35rem 0;
        line-height: 1.1;
    }
    .uk-pass-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: #cbd5e1;
        margin: 0.35rem 0;
    }
    .uk-detail {
        font-size: 0.95rem;
        color: #d1fae5;
        margin-top: 0.5rem;
    }
    .uk-detail-grey {
        font-size: 0.92rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    .uk-reason {
        display: inline-block;
        background: #022c22;
        color: #6ee7b7;
        padding: 3px 8px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.82rem;
        margin-right: 0.4rem;
    }
    .uk-reason-grey {
        display: inline-block;
        background: #0f172a;
        color: #94a3b8;
        padding: 3px 8px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.82rem;
        margin-right: 0.4rem;
    }
    .uk-chip {
        display: inline-block;
        background: #1e293b;
        color: #cbd5e1;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.78rem;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }
    .uk-stored {
        display: inline-block;
        background: #064e3b;
        color: #6ee7b7;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
    .uk-skipped {
        display: inline-block;
        background: #78350f;
        color: #fde68a;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# RULE
# ============================================================================

PPG_GAP_MIN = 0.0


def _pass(segment):
    return {
        "pick":        "pass",
        "direction":   None,
        "stake_class": "NONE",
        "stake":       0.0,
        "segment":     segment,
    }


def _bet(direction):
    return {
        "pick":        "bet",
        "direction":   direction,
        "stake_class": "STANDARD",
        "stake":       1.0,
        "segment":     "table_leader_is_market_underdog",
    }


def decide(row):
    """Apply the rule. Return pick/direction/stake_class/stake/segment."""

    required = [
        "home_points", "home_played",
        "away_points", "away_played",
        "home_odds", "draw_odds", "away_odds",
    ]
    for k in required:
        if row.get(k) is None:
            return _pass(f"missing_{k}")

    if row["home_played"] <= 0 or row["away_played"] <= 0:
        return _pass("no_games_played")

    if row["home_odds"] <= 1.0 or row["away_odds"] <= 1.0:
        return _pass("invalid_odds")

    home_ppg = row["home_points"] / row["home_played"]
    away_ppg = row["away_points"] / row["away_played"]
    gap = home_ppg - away_ppg

    if abs(gap) < 1e-9:
        return _pass("no_table_leader")
    if abs(gap) < PPG_GAP_MIN:
        return _pass("ppg_gap_too_small")

    table_leader = "home" if gap > 0 else "away"

    if abs(row["home_odds"] - row["away_odds"]) < 0.01:
        return _pass("no_market_favourite")

    market_team = "home" if row["home_odds"] < row["away_odds"] else "away"

    if table_leader == market_team:
        return _pass("market_and_table_agree")

    direction = "DC 1X" if table_leader == "home" else "DC X2"
    return _bet(direction)


# ============================================================================
# DISPLAY — the prediction card
# ============================================================================

def render_prediction(row, pick, stored):
    """Render the prediction prominently. This is the product."""
    home_ppg = row["home_points"] / max(row["home_played"], 1)
    away_ppg = row["away_points"] / max(row["away_played"], 1)

    ppg_home_str = f"{home_ppg:.2f}"
    ppg_away_str = f"{away_ppg:.2f}"

    # Which side is the market on?
    if abs(row["home_odds"] - row["away_odds"]) < 0.01:
        market_side = "—"
    else:
        market_side = "home" if row["home_odds"] < row["away_odds"] else "away"

    # Table leader
    if abs(home_ppg - away_ppg) < 1e-9:
        table_side = "—"
    else:
        table_side = "home" if home_ppg > away_ppg else "away"

    stored_badge = (
        '<span class="uk-stored">✓ STORED</span>'
        if stored else
        '<span class="uk-skipped">DUPLICATE · NOT STORED</span>'
    )

    if pick["pick"] == "bet":
        html = f"""
        <div class="uk-bet">
            <div class="uk-label">⭐ UNDERKING PICK</div>
            <div class="uk-pick">{pick['direction']}
                <span style="font-size:1.1rem;opacity:0.7;font-weight:600;">
                    · {pick['stake']}u ({pick['stake_class']})
                </span>
            </div>
            <div class="uk-detail">
                Reason: <span class="uk-reason">{pick['segment']}</span>
                &nbsp;·&nbsp; {stored_badge}
            </div>
            <div class="uk-detail" style="margin-top:0.7rem;">
                <span class="uk-chip">Home PPG {ppg_home_str}</span>
                <span class="uk-chip">Away PPG {ppg_away_str}</span>
                <span class="uk-chip">Table: {table_side}</span>
                <span class="uk-chip">Market: {market_side}</span>
                <span class="uk-chip">Odds {row['home_odds']} / {row['draw_odds']} / {row['away_odds']}</span>
            </div>
        </div>
        """
    else:
        html = f"""
        <div class="uk-pass">
            <div class="uk-label-grey">UNDERKING</div>
            <div class="uk-pass-text">NO BET</div>
            <div class="uk-detail-grey">
                Reason: <span class="uk-reason-grey">{pick['segment']}</span>
                &nbsp;·&nbsp; {stored_badge}
            </div>
            <div class="uk-detail-grey" style="margin-top:0.7rem;">
                <span class="uk-chip">Home PPG {ppg_home_str}</span>
                <span class="uk-chip">Away PPG {ppg_away_str}</span>
                <span class="uk-chip">Table: {table_side}</span>
                <span class="uk-chip">Market: {market_side}</span>
                <span class="uk-chip">Odds {row['home_odds']} / {row['draw_odds']} / {row['away_odds']}</span>
            </div>
        </div>
        """

    st.markdown(html, unsafe_allow_html=True)


# ============================================================================
# SUPABASE
# ============================================================================

def get_supabase():
    try:
        from supabase import create_client
        url = str(st.secrets["SUPABASE_URL"]).strip()
        key = str(st.secrets["SUPABASE_KEY"]).strip()
        return create_client(url, key), None
    except Exception as e:
        return None, str(e)


def upsert_row(sb, payload):
    """
    Upsert by (match_date, home_team, away_team).
    Returns (ok, stored_bool, row_or_error).

    - stored_bool = True  -> new row was written
    - stored_bool = False -> row already existed, prediction still valid
    """
    try:
        response = (
            sb.table("underking_log")
            .upsert(
                payload,
                on_conflict="match_date,home_team,away_team",
                ignore_duplicates=True,
            )
            .select("id")
            .execute()
        )
        stored = bool(response.data)
        return True, stored, (response.data[0] if response.data else None)
    except Exception as e:
        return False, False, str(e)


def load_rows(sb, limit=500):
    try:
        resp = (
            sb.table("underking_log")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        st.error(f"Load failed: {e}")
        return []


def load_pending(sb, limit=500):
    try:
        resp = (
            sb.table("underking_log")
            .select("*")
            .is_("home_score", "null")
            .order("match_date", desc=False)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        st.error(f"Load pending failed: {e}")
        return []


def settle_row(sb, row_id, home_score, away_score):
    if home_score > away_score:
        result = "home"
    elif away_score > home_score:
        result = "away"
    else:
        result = "draw"

    try:
        resp = (
            sb.table("underking_log")
            .select("pick, direction")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )
        if not resp.data:
            return False, f"no row with id={row_id}"
        existing = resp.data[0]
    except Exception as e:
        return False, f"read failed: {e}"

    pick = existing.get("pick")
    direction = existing.get("direction")

    if pick == "bet":
        if direction == "DC 1X":
            hit = result in ("home", "draw")
        elif direction == "DC X2":
            hit = result in ("away", "draw")
        else:
            hit = None
    else:
        hit = None

    payload = {
        "home_score": home_score,
        "away_score": away_score,
        "result":     result,
        "hit":        hit,
    }

    try:
        sb.table("underking_log").update(payload).eq("id", row_id).execute()
        return True, payload
    except Exception as e:
        return False, str(e)


# ============================================================================
# UI
# ============================================================================

def main():
    st.title("👑 Underking")
    st.caption("Enter the match. The pick appears below.")

    sb, err = get_supabase()
    if sb is None:
        st.error(f"Supabase client failed: {err}")
        return

    tab_input, tab_settle, tab_log = st.tabs(
        ["🎯 Predict", "📝 Settle", "📜 Log"]
    )

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    with tab_input:
        st.subheader("Match")

        c1, c2, c3 = st.columns(3)
        match_date = c1.date_input("Match date")
        league     = c2.text_input("League", value="Championship")
        venue      = c3.text_input("Venue", value="")

        c1, c2 = st.columns(2)
        home_team = c1.text_input("Home team")
        away_team = c2.text_input("Away team")

        st.subheader("Table state (pre-kickoff)")

        c1, c2, c3, c4 = st.columns(4)
        home_points = c1.number_input("Home points", 0, 200, 0)
        home_played = c2.number_input("Home played", 0, 60, 0)
        away_points = c3.number_input("Away points", 0, 200, 0)
        away_played = c4.number_input("Away played", 0, 60, 0)

        st.subheader("1X2 odds (pre-kickoff)")

        c1, c2, c3 = st.columns(3)
        home_odds = c1.number_input("Home", 1.01, 100.0, 2.00, step=0.01)
        draw_odds = c2.number_input("Draw", 1.01, 100.0, 3.50, step=0.01)
        away_odds = c3.number_input("Away", 1.01, 100.0, 2.00, step=0.01)

        st.subheader("Notes")
        notes = st.text_area("Optional", height=80)

        if st.button("⚙️ Generate prediction", type="primary"):
            if not home_team or not away_team:
                st.error("Home and away team are required.")
            else:
                match_data = {
                    "match_date":  str(match_date),
                    "league":      league,
                    "venue":       venue or None,
                    "home_team":   home_team.strip(),
                    "away_team":   away_team.strip(),
                    "home_points": int(home_points),
                    "home_played": int(home_played),
                    "away_points": int(away_points),
                    "away_played": int(away_played),
                    "home_odds":   float(home_odds),
                    "draw_odds":   float(draw_odds),
                    "away_odds":   float(away_odds),
                }

                # --- 1. Always compute the prediction first ---
                prediction = decide(match_data)

                # --- 2. Persist (upsert; duplicate-safe) ---
                payload = {
                    **match_data,
                    **prediction,
                    "notes":      notes or None,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                ok, stored, info = upsert_row(sb, payload)

                # --- 3. Always show the prediction, whether stored or not ---
                st.markdown("---")
                st.markdown(f"### {home_team}  vs  {away_team}")
                st.caption(
                    f"{league or '—'} · {match_date} · {venue or '—'}"
                )

                render_prediction(match_data, prediction, stored)

                if not ok:
                    st.error(f"Save failed: {info}")
                elif stored:
                    st.caption(f"Logged as new row id = {info.get('id')}")
                else:
                    st.caption(
                        "Prediction shown. Row already exists — not duplicated."
                    )

    # ------------------------------------------------------------------
    # SETTLE
    # ------------------------------------------------------------------
    with tab_settle:
        st.subheader("Enter final scores")
        st.caption(
            "Only home_score, away_score, result, and hit are written. "
            "Pre-match data is never touched."
        )

        pending = load_pending(sb)

        if not pending:
            st.success("No pending matches.")
        else:
            for r in pending:
                row_id = r["id"]
                pick = r.get("pick", "?")
                direction = r.get("direction") or "—"
                header = (
                    f"#{row_id} · {r.get('match_date','')} · "
                    f"{r.get('home_team','')} vs {r.get('away_team','')} · "
                    f"{pick.upper()} · {direction}"
                )
                with st.expander(header):
                    c1, c2 = st.columns(2)
                    c1.metric("Pick", pick.upper())
                    c2.metric("Direction", direction)

                    st.caption(
                        f"Segment: `{r.get('segment') or '—'}`  ·  "
                        f"Stake: {r.get('stake', 0)}u"
                    )

                    hg = c1.number_input(
                        "Home score", 0, 20, 0, key=f"hg_{row_id}"
                    )
                    ag = c2.number_input(
                        "Away score", 0, 20, 0, key=f"ag_{row_id}"
                    )

                    if st.button("📝 Save result", key=f"settle_{row_id}"):
                        ok, info = settle_row(sb, row_id, hg, ag)
                        if ok:
                            st.success(
                                f"Saved: {hg}-{ag} · result={info['result']} · "
                                f"hit={info['hit']}"
                            )
                            st.rerun()
                        else:
                            st.error(info)

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------
    with tab_log:
        st.subheader("All predictions")

        rows = load_rows(sb, limit=500)
        if not rows:
            st.info("Nothing logged yet.")
        else:
            df = pd.DataFrame(rows)

            keep = [
                "id", "created_at", "match_date", "league",
                "home_team", "away_team",
                "home_odds", "draw_odds", "away_odds",
                "pick", "direction", "stake", "segment",
                "home_score", "away_score", "result", "hit",
            ]
            cols = [c for c in keep if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

            settled = [r for r in rows if r.get("home_score") is not None]
            bets = [r for r in settled if r.get("pick") == "bet"]
            hits = [r for r in bets if r.get("hit") is True]

            c1, c2, c3 = st.columns(3)
            c1.metric("Settled rows", len(settled))
            c2.metric("Bets placed", len(bets))
            c3.metric(
                "Hit rate",
                f"{len(hits) / len(bets) * 100:.1f}%" if bets else "—",
            )


main()
