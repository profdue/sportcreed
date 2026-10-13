"""
Underking — manual input prediction tool.

You enter the match data by hand.
The app applies the rule and writes one row to Supabase.
After the match, you enter the final score to settle the row.

No parser. No scrapers. No factor layer. No tag engine.
Just: input → rule → save → display → settle.
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
# RULE
# ============================================================================

def decide(row):
    """
    Replace this function body with whatever rule you want.

    Returns:
      {
        "pick":        "bet" | "pass",
        "direction":   "DC 1X" | "DC X2" | None,
        "stake_class": "FULL" | "STANDARD" | "REDUCED" | "NONE",
        "stake":       float,
        "segment":     str,
      }
    """
    home_ppg = row["home_points"] / max(row["home_played"], 1)
    away_ppg = row["away_points"] / max(row["away_played"], 1)

    table_leader = "home" if home_ppg > away_ppg else "away"
    market_fav = "home" if row["home_odds"] < row["away_odds"] else "away"

    if table_leader == market_fav:
        return {
            "pick": "pass", "direction": None,
            "stake_class": "NONE", "stake": 0.0,
            "segment": "market_and_table_agree",
        }

    direction = "DC 1X" if table_leader == "home" else "DC X2"
    return {
        "pick": "bet", "direction": direction,
        "stake_class": "STANDARD", "stake": 1.0,
        "segment": "table_leader_is_market_underdog",
    }


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


def save_row(sb, payload):
    try:
        resp = sb.table("underking_log").insert(payload).execute()
        return True, resp.data[0] if resp.data else None
    except Exception as e:
        return False, str(e)


def load_rows(sb, limit=200):
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
    """Rows that have a pick but no score yet."""
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


def load_settled(sb, limit=200):
    """Rows that have a score recorded."""
    try:
        resp = (
            sb.table("underking_log")
            .select("*")
            .not_.is_("home_score", "null")
            .order("match_date", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []
    except Exception as e:
        st.error(f"Load settled failed: {e}")
        return []


def settle_row(sb, row_id, home_score, away_score):
    """
    Write the final score to an existing row.

    Only touches: home_score, away_score, result, hit.
    Never touches pre-match fields (points, odds, pick, etc.).

    If pick == 'pass', hit is left NULL — no bet was placed,
    so there is nothing to hit or miss.
    """
    # Three-way result
    if home_score > away_score:
        result = "home"
    elif away_score > home_score:
        result = "away"
    else:
        result = "draw"

    # Read the row first to compute `hit` correctly
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

    # `hit` is only meaningful if a bet was placed
    if pick == "bet":
        if direction == "DC 1X":
            hit = result in ("home", "draw")
        elif direction == "DC X2":
            hit = result in ("away", "draw")
        else:
            hit = None
    else:
        hit = None   # pass rows are neither hit nor miss

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
    st.caption("Manual input. One rule. One row.")

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

        if st.button("⚙️ Generate & Save", type="primary"):
            if not home_team or not away_team:
                st.error("Home and away team are required.")
            elif home_played == 0 or away_played == 0:
                st.error("Played counts must be > 0.")
            else:
                row = {
                    "match_date":   str(match_date),
                    "league":       league,
                    "venue":        venue or None,
                    "home_team":    home_team.strip(),
                    "away_team":    away_team.strip(),
                    "home_points":  int(home_points),
                    "home_played":  int(home_played),
                    "away_points":  int(away_points),
                    "away_played":  int(away_played),
                    "home_odds":    float(home_odds),
                    "draw_odds":    float(draw_odds),
                    "away_odds":    float(away_odds),
                }

                pick = decide(row)

                payload = {
                    **row,
                    "pick":         pick["pick"],
                    "direction":    pick["direction"],
                    "stake_class":  pick["stake_class"],
                    "stake":        pick["stake"],
                    "segment":      pick["segment"],
                    "notes":        notes or None,
                    "created_at":   datetime.now(timezone.utc).isoformat(),
                }

                if pick["pick"] == "bet":
                    st.success(
                        f"**{pick['direction']}** · {pick['stake']}u "
                        f"({pick['stake_class']}) · `{pick['segment']}`"
                    )
                else:
                    st.info(f"**PASS** · `{pick['segment']}`")

                ok, result = save_row(sb, payload)
                if ok:
                    st.caption(f"Logged row id = {result.get('id')}")
                else:
                    st.error(f"Save failed: {result}")

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
                tier_label = r.get("pick", "?").upper()
                direction = r.get("direction") or "—"
                header = (
                    f"#{row_id} · {r.get('match_date','')} · "
                    f"{r.get('home_team','')} vs {r.get('away_team','')} · "
                    f"{tier_label} · {direction}"
                )
                with st.expander(header):
                    c1, c2 = st.columns(2)
                    c1.metric("Pick", tier_label)
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
