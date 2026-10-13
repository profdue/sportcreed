"""
Underking — manual input prediction tool.

You enter the match data by hand.
The app applies the rule and writes one row to Supabase.

No parser. No scrapers. No factor layer. No tag engine.
Just: input → rule → save → display.
"""

import sys
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Underking",
    page_icon="👑",
    layout="wide",
)


# ============================================================================
# RULE — the only logic that survived the strip
# ============================================================================
# Replace this function body with whatever rule you want Underking to apply.
# Input: a dict of the fields you typed in.
# Output: a dict with pick, direction, stake, and reason.
#
# The rest of the app doesn't care what the rule does. It just stores the
# return value alongside your inputs.

def decide(row):
    """
    Placeholder rule. Swap this out for the real one.

    Expected return shape:
      {
        "pick":        "bet" | "pass",
        "direction":   "DC 1X" | "DC X2" | None,
        "stake_class": "FULL" | "STANDARD" | "REDUCED" | "NONE",
        "stake":       float,
        "segment":     str,   # short label describing why
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


def load_rows(sb, limit=100):
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

    tab_input, tab_log = st.tabs(["🎯 Predict", "📜 Log"])

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

                # Display
                if pick["pick"] == "bet":
                    st.success(
                        f"**{pick['direction']}** · {pick['stake']}u "
                        f"({pick['stake_class']}) · `{pick['segment']}`"
                    )
                else:
                    st.info(f"**PASS** · `{pick['segment']}`")

                # Save
                ok, result = save_row(sb, payload)
                if ok:
                    st.caption(f"Logged row id = {result.get('id')}")
                else:
                    st.error(f"Save failed: {result}")

    # ------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------
    with tab_log:
        st.subheader("Recent predictions")
        rows = load_rows(sb, limit=200)
        if not rows:
            st.info("Nothing logged yet.")
        else:
            df = pd.DataFrame(rows)
            keep = [
                "created_at", "match_date", "league",
                "home_team", "away_team",
                "home_odds", "draw_odds", "away_odds",
                "pick", "direction", "stake", "segment",
            ]
            cols = [c for c in keep if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

        st.caption(
            "Enter the final score directly in Supabase to settle, or add a "
            "settlement form later."
        )


main()
