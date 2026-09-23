"""
Streamlit UI for the Refined Prediction Strategy.
Lazy-loads heavy dependencies. Fails gracefully if Supabase unavailable.
"""

import traceback
from datetime import date, datetime

import pandas as pd
import streamlit as st

# ============================================================================
# PAGE CONFIG — must be the very first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="Refined Prediction Strategy",
    page_icon="⚽",
    layout="wide",
)


# ============================================================================
# SUPABASE — lazy connection, cached, no side effects at import
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    """Return a Supabase client or None if not configured."""
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except KeyError:
        return None
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        return None


TABLE_NAME = "match_predictions"


# ============================================================================
# ENGINE — lazy import so the page renders even if bs4/scipy are missing
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_engine():
    """Import the engine on first use. Cached so it only imports once."""
    try:
        from betting_engine import analyse_html, load_parsed_match, RefinedPredictor
        return {
            "analyse_html": analyse_html,
            "load_parsed_match": load_parsed_match,
            "RefinedPredictor": RefinedPredictor,
            "ok": True,
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ============================================================================
# HELPERS
# ============================================================================
def parse_match_date(date_val):
    if not date_val:
        return datetime(1900, 1, 1)
    if isinstance(date_val, (date, datetime)):
        return datetime(date_val.year, date_val.month, date_val.day)
    s = str(date_val).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime(1900, 1, 1)


def format_date_display(date_val):
    dt = parse_match_date(date_val)
    return str(date_val) if dt.year == 1900 else dt.strftime("%Y-%m-%d")


def check_match_exists(supabase, home_team, away_team, match_date):
    if supabase is None:
        return False
    try:
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else str(match_date)[:10]
        resp = (
            supabase.table(TABLE_NAME)
            .select("id")
            .eq("home_team", home_team)
            .eq("away_team", away_team)
            .eq("match_date", date_part)
            .execute()
        )
        return len(resp.data) > 0
    except Exception:
        return False


def save_bet_to_db(supabase, match, analysis, bet):
    if supabase is None:
        return None
    try:
        home_team = match.get("home_team", "Unknown")
        away_team = match.get("away_team", "Unknown")
        match_date = match.get("date", datetime.now().strftime("%Y-%m-%d"))
        dt = parse_match_date(match_date)
        date_part = dt.strftime("%Y-%m-%d") if dt.year != 1900 else match_date[:10]

        record = {
            "match_date": date_part,
            "home_team": home_team,
            "away_team": away_team,
            "league": match.get("league", "Unknown"),
            "home_odds": match.get("home_odds", 0),
            "draw_odds": match.get("draw_odds", 0),
            "away_odds": match.get("away_odds", 0),
            "model_xg_home": analysis.get("model_xg_home", 0),
            "model_xg_away": analysis.get("model_xg_away", 0),
            "model_total": analysis.get("model_total", 0),
            "market_total": analysis.get("market_total", 0),
            "shrunk_xg_home": analysis.get("shrunk_xg_home", 0),
            "shrunk_xg_away": analysis.get("shrunk_xg_away", 0),
            "shrunk_total": analysis.get("shrunk_total", 0),
            "prob_home_win": analysis.get("probabilities", {}).get("home_win", 0),
            "prob_draw": analysis.get("probabilities", {}).get("draw", 0),
            "prob_away_win": analysis.get("probabilities", {}).get("away_win", 0),
            "prob_btts_yes": analysis.get("probabilities", {}).get("btts_yes", 0),
            "prob_over_25": analysis.get("probabilities", {}).get("over_25", 0),
            "prob_under_25": analysis.get("probabilities", {}).get("under_25", 0),
            "market": bet.get("market", ""),
            "selection": bet.get("selection", ""),
            "bet_prob": bet.get("prob", 0),
            "bet_edge": bet.get("edge", 0),
            "bet_odds": bet.get("odds", 0),
            "stake": bet.get("stake", "1 unit"),
            "confidence": bet.get("confidence", "High"),
            "predicted": "NO_DRAW" if bet.get("market") == "Match Result" else "BET",
            "multi_score": analysis.get("shrunk_total", 0),
        }
        resp = supabase.table(TABLE_NAME).insert(record).execute()
        return resp.data[0]["id"] if resp.data else None
    except Exception as e:
        st.error(f"Save failed: {e}")
        return None


def get_pending(supabase):
    if supabase is None:
        return []
    try:
        resp = supabase.table(TABLE_NAME).select("*").is_("actual_result", "null").execute()
        data = resp.data or []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")))
    except Exception:
        return []


def submit_result(supabase, analysis_id, home_goals, away_goals):
    if supabase is None:
        return False
    try:
        actual_result = "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X"

        resp = supabase.table(TABLE_NAME).select("*").eq("id", analysis_id).execute()
        if not resp.data:
            return False
        rec = resp.data[0]

        market = rec.get("market", "")
        selection = rec.get("selection", "")
        is_correct = False

        if market == "Match Result":
            if "Home" in selection:
                if "−" in selection or "-" in selection:
                    is_correct = home_goals > away_goals
                elif "+" in selection:
                    is_correct = home_goals >= away_goals
            elif "Away" in selection:
                if "−" in selection or "-" in selection:
                    is_correct = away_goals > home_goals
                elif "+" in selection:
                    is_correct = away_goals >= home_goals
            elif "Draw" in selection:
                is_correct = home_goals == away_goals
        elif market == "BTTS":
            is_correct = (home_goals >= 1 and away_goals >= 1) if "Yes" in selection else (home_goals == 0 or away_goals == 0)
        elif market == "Over/Under":
            total = home_goals + away_goals
            is_correct = total > 2.5 if "Over" in selection else total < 2.5

        supabase.table(TABLE_NAME).update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_result": actual_result,
            "is_correct": is_correct,
        }).eq("id", analysis_id).execute()
        return True
    except Exception as e:
        st.error(f"Submit failed: {e}")
        return False


def get_results(supabase):
    if supabase is None:
        return []
    try:
        resp = supabase.table(TABLE_NAME).select("*").not_.is_("actual_result", "null").execute()
        data = resp.data or []
        return sorted(data, key=lambda x: parse_match_date(x.get("match_date")), reverse=True)
    except Exception:
        return []


# ============================================================================
# DISPLAY
# ============================================================================
def display_analysis(analysis: dict, match: dict):
    st.markdown(f"### {match.get('home_team', '')} vs {match.get('away_team', '')}")
    st.caption(f"{match.get('league', '')} · {match.get('date', '')}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Total xG", f"{analysis['model_total']:.2f}")
    c2.metric("Market Total", f"{analysis['market_total']:.2f}")
    c3.metric("Shrunk Total", f"{analysis['shrunk_total']:.2f}")

    st.markdown("#### Poisson Probabilities")
    p = analysis["probabilities"]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(Home)", f"{p.get('home_win', 0):.1%}")
    c2.metric("P(Draw)", f"{p.get('draw', 0):.1%}")
    c3.metric("P(Away)", f"{p.get('away_win', 0):.1%}")

    c1, c2, c3 = st.columns(3)
    c1.metric("P(BTTS Yes)", f"{p.get('btts_yes', 0):.1%}")
    c2.metric("P(Over 2.5)", f"{p.get('over_25', 0):.1%}")
    c3.metric("P(Under 2.5)", f"{p.get('under_25', 0):.1%}")

    st.markdown("#### Selected Bets")
    if not analysis["bets"]:
        st.info("No value bets found for this match.")
    else:
        for b in analysis["bets"]:
            st.success(
                f"**{b['market']}** — {b['selection']} @ {b['odds']:.2f} "
                f"| Edge: **{b['edge']:+.1%}** | {b['stake']}"
            )

    with st.expander("Skipped markets", expanded=False):
        for s in analysis["skips"]:
            st.write(f"**{s['market']}** — {s['reason']}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    st.title("⚽ Refined Prediction Strategy")

    # -- Engine availability check
    engine = load_engine()
    if not engine.get("ok"):
        st.error("⚠️ Engine failed to load.")
        st.code(engine.get("error", "Unknown error"))
        st.info("Make sure `betting_engine.py` is in the same directory and that "
                "`beautifulsoup4` and `scipy` are in `requirements.txt`.")
        return

    # -- Supabase (optional)
    supabase = get_supabase()
    if supabase is None:
        st.info("ℹ️ Supabase not configured — predictions work, but saving is disabled.")

    tabs = st.tabs(["⚽ Predict", "📝 Pending", "📊 Records"])

    # ------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### Paste Sportsgambler HTML")
        st.caption("Open a Sportsgambler preview, View Source, copy the HTML, paste below.")

        text = st.text_area("Sportsgambler HTML", height=250, key="html_input")

        if st.button("Analyze", type="primary"):
            if not text or len(text.strip()) < 200:
                st.error("Please paste a full Sportsgambler preview page.")
                return

            try:
                with st.spinner("Parsing and analysing..."):
                    parsed, match, analysis = engine["analyse_html"](text)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.code(traceback.format_exc())
                return

            display_analysis(analysis, match)

            if supabase is not None and analysis["bets"]:
                saved = 0
                for bet in analysis["bets"]:
                    if check_match_exists(supabase, match["home_team"], match["away_team"], match["date"]):
                        continue
                    if save_bet_to_db(supabase, match, analysis, bet):
                        saved += 1
                if saved:
                    st.success(f"Saved {saved} bet(s) to database.")

    # ------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Pending Bets")
        if supabase is None:
            st.info("Supabase not configured.")
        else:
            pending = get_pending(supabase)
            if not pending:
                st.info("No pending bets.")
            for a in pending:
                with st.expander(f"{a.get('match_date','')} · {a.get('home_team','')} vs {a.get('away_team','')}"):
                    st.write(f"**{a.get('market','')}**: {a.get('selection','')} @ {a.get('bet_odds',0):.2f}")
                    c1, c2 = st.columns(2)
                    hg = c1.number_input(f"{a.get('home_team','')} goals", 0, 15, 0, key=f"hg_{a['id']}")
                    ag = c2.number_input(f"{a.get('away_team','')} goals", 0, 15, 0, key=f"ag_{a['id']}")
                    if st.button("Submit", key=f"sub_{a['id']}"):
                        if submit_result(supabase, a["id"], hg, ag):
                            st.success("Saved.")
                            st.rerun()

    # ------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Performance Records")
        if supabase is None:
            st.info("Supabase not configured.")
        else:
            results = get_results(supabase)
            if not results:
                st.info("No results recorded yet.")
            else:
                total = len(results)
                wins = sum(1 for r in results if r.get("is_correct"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", total)
                c2.metric("Win Rate", f"{wins/total*100:.0f}%")
                c3.metric("Wins", wins)

                df = pd.DataFrame([{
                    "Date": r.get("match_date", ""),
                    "Match": f"{r.get('home_team','')} vs {r.get('away_team','')}",
                    "Market": r.get("market", ""),
                    "Selection": r.get("selection", ""),
                    "Edge": f"{r.get('bet_edge', 0):+.1%}",
                    "Odds": f"{r.get('bet_odds', 0):.2f}",
                    "Score": f"{r.get('actual_home_goals','')}-{r.get('actual_away_goals','')}",
                    "Result": "✅" if r.get("is_correct") else "❌",
                } for r in results])
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
