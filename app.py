"""
v4 RAW-ONLY Predictor — FULL APP
Components:
  1. Parser: HTML → matches_raw
  2. Model:  matches_raw → predictions
  3. Audit:  actual score → is_correct_1x2
  4. Display: performance dashboard
"""

import os
import re
import json
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="v4 Raw Predictor", page_icon="⚽", layout="wide")


# ============================================================================
# SUPABASE
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_supabase():
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key), {"ok": True, "url": url}
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


# ============================================================================
# PARSER: HTML → matches_raw
# ============================================================================
def parse_html(html: str) -> dict:
    """Parse Sportsgambler HTML → dict matching matches_raw schema"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    rec = {}

    # --- FIXTURE ---
    # Team names
    teams = soup.select(".t_top .t_teams .t_name strong")
    if len(teams) >= 2:
        rec["home_team"] = teams[0].get_text(strip=True)
        rec["away_team"] = teams[1].get_text(strip=True)

    # Date + kickoff
    date_el = soup.select_one(".t_top .t_date span:first-child")
    time_el = soup.select_one(".t_top .t_date .t_time")
    if date_el:
        raw = date_el.get_text(strip=True)
        # e.g. "Sun 20 Sep"
        m = re.match(r"\w+ (\d+) (\w+)", raw)
        if m:
            day, month = m.group(1), m.group(2)
            year = datetime.now().year
            try:
                dt = datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
                rec["match_date"] = dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    if time_el:
        rec["kickoff_utc"] = time_el.get_text(strip=True)

    # League
    for link in soup.select(".t_top .t_info_link"):
        text = link.get_text(strip=True)
        if text.lower() == "football":
            continue
        if re.search(r"(League|Serie|Liga|Bundesliga|Ligue|Premier|Cup)", text, re.I):
            rec["league_name"] = text
            break

    # Venue
    venue_el = soup.select_one(".t_top .t_venue")
    if venue_el:
        rec["venue"] = venue_el.get_text(strip=True)

    # --- F1: STANDINGS ---
    # Look for the league table
    for table in soup.select("table.leage-table"):
        rows = table.select("tbody tr")
        for row in rows:
            cells = row.select("td")
            if len(cells) < 8:
                continue
            team_name = cells[1].get_text(strip=True)
            if rec.get("home_team", "").lower() in team_name.lower():
                rec["home_pos"] = _int(cells[0])
                rec["home_played"] = _int(cells[2])
                rec["home_points"] = _int(cells[7])
                gd = cells[6].get_text(strip=True)
                rec["home_gd"] = _parse_gd(gd)
            elif rec.get("away_team", "").lower() in team_name.lower():
                rec["away_pos"] = _int(cells[0])
                rec["away_played"] = _int(cells[2])
                rec["away_points"] = _int(cells[7])
                gd = cells[6].get_text(strip=True)
                rec["away_gd"] = _parse_gd(gd)

    # --- F2: LAST 5 FORM ---
    rec["home_last5"] = _parse_last5(soup, "home")
    rec["away_last5"] = _parse_last5(soup, "away")

    # --- F3/F6: LAST 10 SPLITS ---
    _parse_last10_splits(soup, "home", rec)
    _parse_last10_splits(soup, "away", rec)

    # --- F4: TOP SCORER / ASSISTER / INJURIES / XI ---
    _parse_players(soup, rec)
    _parse_injuries(soup, rec)
    _parse_xi(soup, rec)

    # --- F5: H2H ---
    rec["h2h"] = _parse_h2h(soup)
    rec["h2h_home_wins"] = sum(1 for h in rec["h2h"] if h.get("winner") == "home")
    rec["h2h_away_wins"] = sum(1 for h in rec["h2h"] if h.get("winner") == "away")
    rec["h2h_draws"] = len(rec["h2h"]) - rec["h2h_home_wins"] - rec["h2h_away_wins"]

    # --- ODDS (optional) ---
    rec["odds"] = _parse_odds(soup)

    return rec


def _int(s):
    try:
        return int(str(s).strip())
    except (ValueError, TypeError):
        return None


def _parse_gd(s):
    """Parse '+5' or '-3' or '31:7'"""
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) - int(parts[1])
        except ValueError:
            return None
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return None


def _parse_last5(soup, side):
    """Parse last 5 matches from #last-matches #All"""
    out = []
    container = soup.select_one("#last-matches #All")
    if not container:
        return out
    block = container.select_one(f".teamstats-{'left' if side == 'home' else 'right'}")
    if not block:
        return out
    for item in block.select("li.team-stat-list-item")[:5]:
        date_el = item.select_one(".team-stats-date")
        teams = item.select(".team-stats-team")
        if len(teams) < 2:
            continue
        home_name = teams[0].get_text(" ", strip=True)
        away_name = teams[1].get_text(" ", strip=True)
        h_score = _extract_score(teams[0])
        a_score = _extract_score(teams[1])
        # Determine result from tracked team's perspective
        tracked_el = block.select_one(".team-stats-team-name strong")
        tracked = tracked_el.get_text(strip=True).lower() if tracked_el else ""
        is_home = tracked in home_name.lower()
        if is_home:
            result = "W" if h_score > a_score else "D" if h_score == a_score else "L"
            score_for, score_against = h_score, a_score
        else:
            result = "W" if a_score > h_score else "D" if a_score == h_score else "L"
            score_for, score_against = a_score, h_score
        out.append({
            "date": date_el.get_text(strip=True) if date_el else "",
            "opp": away_name if is_home else home_name,
            "result": result,
            "score_for": score_for,
            "score_against": score_against,
            "is_home": is_home,
        })
    return out


def _extract_score(team_el):
    score_el = team_el.select_one(".score-right")
    if not score_el:
        return 0
    try:
        return int(score_el.get_text(strip=True))
    except ValueError:
        return 0


def _parse_last10_splits(soup, side, rec):
    """Parse last 10 home/away splits from .st-table"""
    table = soup.select_one(".st-table")
    if not table:
        return
    rows = table.select("tbody tr")
    idx = 0 if side == "home" else 1
    if idx >= len(rows):
        return
    cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
    if len(cells) < 8:
        return
    # W-D-L
    m = re.match(r"(\d+)-(\d+)-(\d+)", cells[1])
    if m:
        rec[f"{side}_last10_w"] = int(m.group(1))
        rec[f"{side}_last10_d"] = int(m.group(2))
        rec[f"{side}_last10_l"] = int(m.group(3))
    rec[f"{side}_last10_avg_scored"] = _float(cells[3])
    rec[f"{side}_last10_avg_conceded"] = _float(cells[4])
    # Win pct
    if rec.get(f"{side}_last10_w") is not None:
        rec[f"{side}_last10_win_pct"] = (rec[f"{side}_last10_w"] / 10) * 100


def _float(s):
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


def _parse_players(soup, rec):
    """Parse top scorer/assister from preview text"""
    # This is approximate — depends on exact HTML structure
    # Look for "Top Scorer" or "Players to Watch"
    text = soup.get_text(" ", strip=True)
    # Top scorer pattern: "Top scorer: Name (N goals)"
    for side in ["home", "away"]:
        # Simplified: use regex on the preview text
        pass  # Requires specific HTML structure


def _parse_injuries(soup, rec):
    """Parse injuries from .inj-two-outline"""
    rec["home_injuries"] = []
    rec["away_injuries"] = []
    for outline in soup.select(".inj-two-outline"):
        header = outline.select_one(".light-header strong")
        if not header:
            continue
        header_text = header.get_text(" ", strip=True).lower()
        side = None
        if rec.get("home_team", "").lower().split()[0] in header_text:
            side = "home"
        elif rec.get("away_team", "").lower().split()[0] in header_text:
            side = "away"
        if not side:
            continue
        for row in outline.select(".inj-two-row"):
            if "inj-two-title" in row.get("class", []):
                continue
            player_el = row.select_one(".inj-two-player")
            info_el = row.select_one(".inj-two-info")
            if not player_el:
                continue
            player = player_el.get_text(strip=True)
            status = "out"
            if info_el:
                info = info_el.get_text(strip=True).lower()
                if "doubt" in info:
                    status = "doubt"
                elif "yellow" in info or "red" in info:
                    status = "suspended"
            rec[f"{side}_injuries"].append({"player": player, "status": status})


def _parse_xi(soup, rec):
    """Parse confirmed XI from .lineups-home / .lineups-away"""
    rec["home_xi"] = []
    rec["away_xi"] = []
    for side, cls in [("home", ".lineups-home"), ("away", ".lineups-away")]:
        block = soup.select_one(cls)
        if not block:
            continue
        for player in block.select(".lineups-player"):
            name_el = player.select_one(".player-name")
            if name_el:
                rec[f"{side}_xi"].append(name_el.get_text(strip=True))
    # Formation
    form_el = soup.select_one(".lineups-toggle-formation")
    if form_el:
        rec["home_formation"] = form_el.get_text(strip=True)


def _parse_h2h(soup):
    """Parse H2H from #head-to-head"""
    out = []
    container = soup.select_one("#head-to-head")
    if not container:
        return out
    for item in container.select("li.team-stat-list-item"):
        date_el = item.select_one(".team-stats-date")
        teams = item.select(".team-stats-team")
        if len(teams) < 2:
            continue
        home_name = teams[0].get_text(" ", strip=True)
        away_name = teams[1].get_text(" ", strip=True)
        h_score = _extract_score(teams[0])
        a_score = _extract_score(teams[1])
        winner = "home" if h_score > a_score else "away" if a_score > h_score else "draw"
        out.append({
            "date": date_el.get_text(strip=True) if date_el else "",
            "home": home_name,
            "away": away_name,
            "score": f"{h_score}-{a_score}",
            "winner": winner,
        })
    return out


def _parse_odds(soup):
    """Parse odds from .nlf_odds_row"""
    out = []
    for row in soup.select(".nlf_odds_row"):
        title_el = row.select_one(".nfl_odd_title")
        if not title_el:
            continue
        market = title_el.get_text(strip=True)
        for ply in row.select(".nfl_odply"):
            label_el = ply.select_one(".nfl_ply_t")
            odds_el = ply.select_one(".nfl_ply_o")
            if label_el and odds_el:
                out.append({
                    "market": market,
                    "selection": label_el.get_text(strip=True),
                    "odds": _float(odds_el.get_text(strip=True)),
                })
    return out


# ============================================================================
# v4 MODEL
# ============================================================================
def calc_f1(row):
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
    if row.get("away_away_points") == 0 and (row.get("away_away_played") or 0) >= 3:
        f1_home = min(20, f1_home + 8)
        away_collapse = True
    return f1_home, f1_away, away_collapse


def calc_f2(last5):
    if not last5:
        return 7.5
    raw = 0
    for m in last5:
        r = m.get("result", "L")
        if r == "W":
            raw += 5
            if (m.get("score_for") or 0) >= 3:
                raw += 1
        elif r == "D":
            raw += 2
        else:
            if (m.get("score_against") or 0) >= 3:
                raw -= 1
    raw = max(0, min(15, raw))
    return (raw / 15) * 25


def calc_f3(win_pct, avg_scored, avg_conceded):
    win_pts = ((win_pct or 0) / 100) * 10
    xg_edge = (avg_scored or 0) - (avg_conceded or 0)
    xg_pts = 5 if xg_edge > 0.5 else 2 if xg_edge > 0 else 0
    return min(15, win_pts + xg_pts)


def calc_f4(top_scorer, top_assister, injuries, xi):
    f4 = 15
    injured = [i.get("player", "") for i in (injuries or [])]
    doubts = [i.get("player", "") for i in (injuries or []) if i.get("status") == "doubt"]
    xi = xi or []
    if top_scorer and top_scorer in injured and top_scorer not in xi:
        f4 -= 5
    if top_assister and top_assister in injured and top_assister not in xi:
        f4 -= 4
    doubted_starter = False
    for name in doubts:
        if name in xi and (name == top_scorer or name == top_assister):
            f4 += 5
            doubted_starter = True
    return max(0, min(20, f4)), doubted_starter


def calc_f5(h2h_home, h2h_away, h2h_total):
    if not h2h_total:
        return 5.0, 5.0
    return (h2h_home / h2h_total) * 10, (h2h_away / h2h_total) * 10


def calc_f6(home_avg, away_avg):
    if (home_avg or 0) > (away_avg or 0):
        return 11, 7
    elif (home_avg or 0) < (away_avg or 0):
        return 7, 11
    return 8, 8


def apply_override(f1_home, f1_away, f5_home, f5_away):
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
    h_s = row.get("home_home_last10_avg_scored") or 1.0
    a_c = row.get("away_away_last10_avg_conceded") or 1.0
    a_s = row.get("away_away_last10_avg_scored") or 1.0
    h_c = row.get("home_home_last10_avg_conceded") or 1.0
    return (h_s + a_c + a_s + h_c) / 2


def predict_v4(row):
    f1_home, f1_away, away_collapse = calc_f1(row)
    f2_home = calc_f2(row.get("home_last5"))
    f2_away = calc_f2(row.get("away_last5"))
    f3_home = calc_f3(row.get("home_home_win_pct"),
                      row.get("home_home_last10_avg_scored"),
                      row.get("home_home_last10_avg_conceded"))
    f3_away = calc_f3(row.get("away_away_win_pct"),
                      row.get("away_away_last10_avg_scored"),
                      row.get("away_away_last10_avg_conceded"))
    f4_home, dh = calc_f4(row.get("home_top_scorer"), row.get("home_top_assister"),
                          row.get("home_injuries"), row.get("home_xi"))
    f4_away, da = calc_f4(row.get("away_top_scorer"), row.get("away_top_assister"),
                          row.get("away_injuries"), row.get("away_xi"))
    doubted = dh or da
    f5_home, f5_away = calc_f5(row.get("h2h_home_wins") or 0,
                               row.get("h2h_away_wins") or 0,
                               len(row.get("h2h") or []))
    f6_home, f6_away = calc_f6(row.get("home_last10_avg_scored"),
                               row.get("away_last10_avg_scored"))

    f5_home, f5_away, conflict, override = apply_override(f1_home, f1_away, f5_home, f5_away)

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
    if away_collapse or doubted:
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
        "home_total": round(home_total, 2), "away_total": round(away_total, 2),
        "total_gap": round(gap, 2),
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "away_collapse": away_collapse,
        "doubted_starter": doubted,
        "call_1x2": call_1x2,
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
    }


# ============================================================================
# AUDIT
# ============================================================================
def audit_match(sb, match_id, actual_home, actual_away):
    """Update row with actual score + is_correct_1x2"""
    resp = sb.table("matches_raw").select("call_1x2").eq("id", match_id).single().execute()
    row = resp.data
    call = row.get("call_1x2", "")

    if actual_home > actual_away:
        actual_1x2 = "Home"
    elif actual_home < actual_away:
        actual_1x2 = "Away"
    else:
        actual_1x2 = "Draw"

    is_correct = None
    if call == "NO BET":
        is_correct = None
    elif "Straight Win Home" in call and actual_1x2 == "Home":
        is_correct = True
    elif "Straight Win Away" in call and actual_1x2 == "Away":
        is_correct = True
    elif "Double Chance 1X" in call and actual_1x2 in ["Home", "Draw"]:
        is_correct = True
    elif "Double Chance X2" in call and actual_1x2 in ["Away", "Draw"]:
        is_correct = True
    else:
        is_correct = False

    sb.table("matches_raw").update({
        "actual_home_goals": actual_home,
        "actual_away_goals": actual_away,
        "is_correct_1x2": is_correct,
    }).eq("id", match_id).execute()

    return is_correct


# ============================================================================
# UI
# ============================================================================
def main():
    st.title("⚽ v4 Raw-Only Predictor")
    st.caption("Parse HTML → Score F1-F6 → Audit results → Track accuracy")

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase failed: {diag.get('error')}")
        return

    tabs = st.tabs(["📥 Parse HTML", "🎯 Run Model", "📝 Audit", "📊 Performance"])

    # --- TAB 1: PARSE ---
    with tabs[0]:
        st.subheader("Parse Sportsgambler HTML → matches_raw")
        html = st.text_area("Paste HTML", height=300, key="html_input")
        if st.button("💾 Parse & Save", type="primary"):
            if not html or len(html) < 200:
                st.error("Paste a full HTML preview.")
            else:
                try:
                    rec = parse_html(html)
                    # Ensure required fields
                    if not rec.get("match_date") or not rec.get("home_team"):
                        st.error("Could not extract match_date or team names.")
                    else:
                        resp = sb.table("matches_raw").upsert(
                            rec,
                            on_conflict="match_date,home_team,away_team"
                        ).execute()
                        st.success(f"Saved: {rec['home_team']} vs {rec['away_team']} on {rec['match_date']}")
                        st.json({k: v for k, v in rec.items() if k not in ["home_last5", "away_last5", "h2h", "odds"]})
                except Exception as e:
                    st.error(f"Parse failed: {e}")

    # --- TAB 2: MODEL ---
    with tabs[1]:
        st.subheader("Run v4 on all unsettled matches")
        if st.button("▶️ Run v4", type="primary"):
            try:
                resp = sb.table("matches_raw").select("*").is_("call_1x2", "null").execute()
                rows = resp.data or []
                for row in rows:
                    result = predict_v4(row)
                    sb.table("matches_raw").update(result).eq("id", row["id"]).execute()
                st.success(f"Ran v4 on {len(rows)} matches.")
            except Exception as e:
                st.error(f"Model failed: {e}")

    # --- TAB 3: AUDIT ---
    with tabs[2]:
        st.subheader("Enter actual scores")
        try:
            resp = sb.table("matches_raw").select(
                "id,match_date,home_team,away_team,call_1x2,call_ou"
            ).is_("actual_home_goals", "null").execute()
            pending = resp.data or []
        except Exception as e:
            st.error(f"Query failed: {e}")
            pending = []

        if not pending:
            st.info("No pending matches.")
        for m in pending:
            with st.expander(f"{m['match_date']} · {m['home_team']} vs {m['away_team']} · {m.get('call_1x2','?')}"):
                c1, c2 = st.columns(2)
                hg = c1.number_input("Home goals", 0, 15, 0, key=f"hg_{m['id']}")
                ag = c2.number_input("Away goals", 0, 15, 0, key=f"ag_{m['id']}")
                if st.button("Submit", key=f"sub_{m['id']}"):
                    ok = audit_match(sb, m["id"], hg, ag)
                    st.success(f"Saved. Correct: {ok}")
                    st.rerun()

    # --- TAB 4: PERFORMANCE ---
    with tabs[3]:
        st.subheader("Performance")
        try:
            resp = sb.table("matches_raw").select("*").not_.is_("actual_home_goals", "null").execute()
            settled = resp.data or []
        except Exception as e:
            st.error(f"Query failed: {e}")
            settled = []

        if not settled:
            st.info("No settled matches.")
        else:
            placed = [r for r in settled if r.get("call_1x2") != "NO BET"]
            correct = sum(1 for r in placed if r.get("is_correct_1x2"))
            override_games = [r for r in settled if r.get("f1_f5_override")]
            override_correct = sum(1 for r in override_games if r.get("is_correct_1x2"))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Settled", len(settled))
            c2.metric("Bets placed", len(placed))
            c3.metric("1X2 accuracy", f"{correct}/{len(placed)}" if placed else "—")
            c4.metric("Override accuracy", f"{override_correct}/{len(override_games)}" if override_games else "—")

            df = pd.DataFrame([{
                "Date": r.get("match_date"),
                "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                "Gap": r.get("total_gap"),
                "1X2 Call": r.get("call_1x2"),
                "Actual": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                "Correct": r.get("is_correct_1x2"),
                "Override": r.get("f1_f5_override"),
            } for r in settled])
            st.dataframe(df, use_container_width=True)


main()
