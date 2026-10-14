#!/usr/bin/env python3
"""
king.py — The Logic.

Input HTML. Extract facts. Run the logic. Show the pick.
Send to Supabase. Track pending. Settle later. See performance.

One file. Two clauses. Three tabs.

Parsing strategy: layered funnel.
  1. Structured data (tables, data attributes, JSON-LD)
  2. Semi-structured (labeled divs with fixed classes)
  3. Prose (regex on sentences)
  4. Give up and warn

Every field failure is logged. Missing data never votes silently.
"""

import re
import json
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
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;
    }
    .badge-home { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-away { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b55; }
    .badge-skip { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win  { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.75rem; padding: 4px 10px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.nothing { background: #64748b15; color: #94a3b8; border-color: #64748b40; }
    .metric-box {
        background: #1a1f2e; border-radius: 12px; padding: 16px;
        text-align: center; border: 1px solid #232a3d;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: #1a1f2e; border-radius: 10px;
        padding: 0 18px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
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


def db_delete(match_id: int):
    sb = get_supabase()
    sb.table("matches").delete().eq("match_id", match_id).execute()


# ═══════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ═══════════════════════════════════════════════════════════════════

_TEAM_PREFIXES = (
    "club ", "atlético ", "atletico ", "deportivo ", "real ",
    "fc ", "cf ", "cd ", "ca ", "ac ", "as ", "sv ", "sc ",
    "athletic ", "sporting ", "racing ", "olympique ", "stade ",
    "rc ", "rcd ", "ud ", "sd ", "ss ", "ssc ",
)

_TEAM_SUFFIXES = (
    " sco", " fc", " cf", " sc", " ac", " cd", " sv", " as",
    " 29", " 04", " 05", " 1899", " 1900", " calcio", " futbol",
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
    "single": 1, "a": 1, "an": 1,
}


def _search_term(full_name: str) -> str:
    if not full_name:
        return ""
    name = full_name.strip()
    low = name.lower()
    for prefix in _TEAM_PREFIXES:
        if low.startswith(prefix):
            name = name[len(prefix):]
            break
    low = name.lower()
    for suffix in _TEAM_SUFFIXES:
        if low.endswith(suffix):
            name = name[: -len(suffix)]
            break
    parts = name.split()
    if len(parts) >= 2:
        if len(parts[0]) <= 3:
            return parts[-1]
        return " ".join(parts[:2])
    return name


def _candidates(term):
    if not term:
        return []
    out = [term]
    first = term.split()[0]
    if first != term:
        out.append(first)
    clean = re.sub(r"\s+\d+$", "", term).strip()
    clean = re.sub(r"\s+(SCO|FC|CF|SC|AC|CD|SV|AS|SS|UD|SD)$", "",
                   clean, flags=re.IGNORECASE).strip()
    if clean and clean not in out:
        out.append(clean)
    if clean and " " in clean:
        first2 = clean.split()[0]
        if first2 not in out:
            out.append(first2)
        last = clean.split()[-1]
        if last not in out and len(last) > 2:
            out.append(last)
    return out


def _team_match(full_name, cell_text):
    if not full_name or not cell_text:
        return False
    term = _search_term(full_name).lower()
    cell = cell_text.lower()
    if not term:
        return False
    if term in cell or cell in term:
        return True
    term_words = set(term.split())
    cell_words = set(cell.split())
    return bool(term_words & cell_words)


def _word_to_int(s):
    if isinstance(s, (int, float)):
        return int(s)
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return _NUMBER_WORDS.get(s, 0)


def _int_any(text):
    if text is None:
        return None
    m = re.search(r"-?\d+", str(text))
    if m:
        return int(m.group())
    return _word_to_int(text)


def _num(s):
    if s is None:
        return None
    s = str(s).strip().lower()
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)
    return _NUMBER_WORDS.get(s)


def _team_aliases(soup, team_full, team_short):
    aliases = set()
    if team_full:
        aliases.add(team_full.strip())
        t = _search_term(team_full)
        if t:
            aliases.add(t)
    if team_short:
        aliases.add(team_short.strip())
        t = _search_term(team_short)
        if t:
            aliases.add(t)

    for name in list(aliases):
        low = name.lower()
        for prefix in _TEAM_PREFIXES:
            if low.startswith(prefix):
                aliases.add(name[len(prefix):].strip())
        for suffix in _TEAM_SUFFIXES:
            if low.endswith(suffix):
                aliases.add(name[: -len(suffix)].strip())

    for name in list(aliases):
        parts = name.split()
        if len(parts) > 1:
            aliases.add(parts[0])
            aliases.add(parts[-1])

    cleaned = set()
    for a in aliases:
        a = a.strip()
        if a and len(a) > 2 and a.lower() not in {"the", "and", "for"}:
            cleaned.add(a)
    return list(cleaned)


# ═══════════════════════════════════════════════════════════════════
# THE LOGIC — TWO CLAUSES
# ═══════════════════════════════════════════════════════════════════

def _cmp_or_nothing(a, b):
    if a is None or b is None or a == b:
        return "nothing"
    return "home" if a > b else "away"


def compute_signals(f: dict) -> dict:
    hp = f.get("home_table_pts")
    ap = f.get("away_table_pts")
    hg = f.get("home_table_gd") or 0
    ag = f.get("away_table_gd") or 0
    if hp is None or ap is None:
        table = "nothing"
    elif (hp, hg) > (ap, ag):
        table = "home"
    elif (ap, ag) > (hp, hg):
        table = "away"
    else:
        table = "nothing"

    form  = _cmp_or_nothing(f.get("home_form_ppg"),  f.get("away_form_ppg"))
    venue = _cmp_or_nothing(f.get("home_home_ppg"),  f.get("away_away_ppg"))
    h2h   = _cmp_or_nothing(f.get("h2h_home_wins"),  f.get("h2h_away_wins"))

    return {"table": table, "form": form, "venue": venue, "h2h": h2h}


def decide(signals: dict) -> dict:
    home = sum(1 for v in signals.values() if v == "home")
    away = sum(1 for v in signals.values() if v == "away")
    nothing = sum(1 for v in signals.values() if v == "nothing")
    table_leader = signals["table"]

    if home == 4 or away == 4:
        direction = "home" if home == 4 else "away"
        return {
            "direction": direction, "stake": "full", "qualifying": 1,
            "dissent_signal": None, "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if home == 3 and away == 1:
        dissent = next((k for k, v in signals.items() if v == "away"), None)
        return {
            "direction": table_leader, "stake": "half", "qualifying": 1,
            "dissent_signal": dissent, "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }
    if away == 3 and home == 1:
        dissent = next((k for k, v in signals.items() if v == "home"), None)
        return {
            "direction": table_leader, "stake": "half", "qualifying": 1,
            "dissent_signal": dissent, "split": f"{home}-{away}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if home == 3 and away == 0 and nothing == 1:
        return {
            "direction": "home", "stake": "full", "qualifying": 1,
            "dissent_signal": "nothing", "split": f"{home}-{away}-{nothing}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }
    if away == 3 and home == 0 and nothing == 1:
        return {
            "direction": "away", "stake": "full", "qualifying": 1,
            "dissent_signal": "nothing", "split": f"{home}-{away}-{nothing}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if home == 2 and away == 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": "2-2",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    if (home + away) < 2:
        return {
            "direction": "skip", "stake": "none", "qualifying": 0,
            "dissent_signal": None, "split": f"{home}-{away}-{nothing}",
            "home_count": home, "away_count": away, "nothing_count": nothing,
        }

    split = f"{home}-{away}-{nothing}" if nothing else f"{home}-{away}"
    return {
        "direction": "skip", "stake": "none", "qualifying": 0,
        "dissent_signal": None, "split": split,
        "home_count": home, "away_count": away, "nothing_count": nothing,
    }


def settle(direction, stake, home_score, away_score):
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
        hit = result in ("home_win", "draw")
    elif direction == "away":
        hit = result in ("away_win", "draw")
    else:
        hit = False

    profit = units if hit else -units
    return result, (1 if hit else 0), units, profit


# ═══════════════════════════════════════════════════════════════════
# HTML EXTRACTION — LAYERED
# ═══════════════════════════════════════════════════════════════════

def _extract_identity(soup, f):
    t_top = soup.find("div", class_="t_top")

    # --- Teams ---
    if t_top:
        names = t_top.find_all("div", class_="t_name")
        if len(names) >= 2:
            f["home_team"] = names[0].get_text(strip=True)
            f["away_team"] = names[1].get_text(strip=True)

    title = soup.find("title")
    if not f.get("home_team") and title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", title.get_text())
        if m:
            f["home_team"] = m.group(1).strip()
            f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        h1 = soup.find("h1")
        if h1:
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", h1.get_text())
            if m:
                f["home_team"] = m.group(1).strip()
                f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", og["content"])
            if m:
                f["home_team"] = m.group(1).strip()
                f["away_team"] = m.group(2).strip()

    if not f.get("home_team"):
        link = soup.find("link", rel="canonical")
        if link and link.get("href"):
            m = re.search(r"/([^/]+)-vs-([^/]+)-prediction", link["href"])
            if m:
                f["home_team"] = m.group(1).replace("-", " ").title()
                f["away_team"] = m.group(2).replace("-", " ").title()

    # --- Short names ---
    if title:
        m = re.search(r"(.+?)\s+vs\s+(.+?)\s+Prediction", title.get_text())
        if m:
            f["home_team_short"] = m.group(1).strip()
            f["away_team_short"] = m.group(2).strip()

    if not f.get("home_team_short"):
        f["home_team_short"] = _search_term(f.get("home_team", ""))
    if not f.get("away_team_short"):
        f["away_team_short"] = _search_term(f.get("away_team", ""))

    # --- League ---
    if t_top:
        info = t_top.find("div", class_="t_info")
        if info:
            links = info.find_all("a", class_="t_info_link")
            if len(links) >= 2:
                league_txt = links[1].get_text(strip=True)
                if " - " in league_txt:
                    _, league_txt = league_txt.split(" - ", 1)
                f["league"] = league_txt

    if not f.get("league"):
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                for node in data.get("@graph", [data]):
                    event = node.get("mainEntity", {})
                    if isinstance(event, dict) and "superEvent" in event:
                        f["league"] = event["superEvent"].get("name")
                        break
                if f.get("league"):
                    break
            except Exception:
                continue

    # --- Date / time ---
    if t_top:
        info = t_top.find("div", class_="t_info")
        if info:
            date = info.find("span", class_="t_date")
            if date:
                txt = date.get_text(" ", strip=True)
                m = re.search(r"(\d{1,2}\s+\w+)", txt)
                if m:
                    f["match_date"] = m.group(1)
                m = re.search(r"(\d{1,2}:\d{2})", txt)
                if m:
                    f["kickoff_time"] = m.group(1)

    if not f.get("match_date"):
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                for node in data.get("@graph", [data]):
                    event = node.get("mainEntity", {})
                    if isinstance(event, dict) and "startDate" in event:
                        f["match_date"] = event["startDate"][:10]
                        break
                if f.get("match_date"):
                    break
            except Exception:
                continue

    # --- Venue ---
    if t_top:
        info = t_top.find("div", class_="t_info")
        if info:
            venue = info.find("span", class_="t_venue")
            if venue:
                f["venue"] = venue.get_text(strip=True)

    if not f.get("venue"):
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                for node in data.get("@graph", [data]):
                    event = node.get("mainEntity", {})
                    if isinstance(event, dict):
                        loc = event.get("location", {})
                        if isinstance(loc, dict) and "name" in loc:
                            f["venue"] = loc["name"]
                            break
                if f.get("venue"):
                    break
            except Exception:
                continue


def _parse_standings(soup, f):
    home_team = f.get("home_team", "")
    away_team = f.get("away_team", "")

    candidates = (
        soup.find_all("table", class_="clt-table") +
        soup.find_all("table", class_="leage-table") +
        soup.find_all("table", class_="league-table") +
        soup.find_all("table", class_="standings-table")
    )

    seen = set()
    for table in candidates:
        for row in table.find_all("tr"):
            cells = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cells) < 4:
                continue

            team_cell = cells[1] if len(cells) > 1 else ""

            target = None
            if _team_match(home_team, team_cell):
                target = "home"
            elif _team_match(away_team, team_cell):
                target = "away"
            if not target or target in seen:
                continue
            seen.add(target)

            nums = [_int_any(c) for c in cells if _int_any(c) is not None]

            pos = nums[0] if nums else None
            pts = nums[-1] if nums else None

            gd = None
            for c in cells:
                if ":" in c:
                    try:
                        a, b = c.split(":")
                        gd = int(a) - int(b)
                        break
                    except Exception:
                        pass
            if gd is None:
                for c in reversed(cells[:-1]):
                    s = c.strip()
                    if (s.startswith("-") or s.startswith("+")) and s[1:].isdigit():
                        gd = int(s)
                        break

            f[f"{target}_table_pos"] = pos
            f[f"{target}_table_pts"] = pts
            f[f"{target}_table_gd"]  = gd

            if len(cells) >= 11:
                f[f"{target}_table_played"] = _int_any(cells[2])
                f[f"{target}_table_gf"]     = _int_any(cells[6])
                f[f"{target}_table_ga"]     = _int_any(cells[7])

        if len(seen) == 2:
            break


# --- Regex pieces ---
_WIN_WORDS  = r"(?:wins?|victor(?:y|ies)|won)"
_LOSS_WORDS = r"(?:losses?|defeats?|lost)"
_DRAW_WORDS = r"(?:draws?|drawn)"
_NUM_PAT    = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"


def _wdl_from_cell(text):
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _parse_venue_from_stable(soup):
    st = soup.find("table", class_="st-table")
    if not st:
        return None, None

    home_wdl = away_wdl = None
    for row in st.find_all("tr"):
        cells = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cells) < 2:
            continue

        label = " ".join(cells).lower()

        wdl = None
        for c in cells:
            wdl = _wdl_from_cell(c)
            if wdl:
                break
        if not wdl:
            continue

        if "home stats" in label:
            home_wdl = wdl
        elif "away stats" in label:
            away_wdl = wdl

    return home_wdl, away_wdl


def _venue_sentence(sents):
    keywords = [
        "in the last 10 home", "in the last 10 away",
        "in the past 10 home", "in the past 10 away",
        "in the previous 10 home", "in the previous 10 away",
        "in their last 10 home", "in their last 10 away",
        "in their past 10 home", "in their past 10 away",
        "last 10 home matches", "last 10 away matches",
    ]
    return [s for s in sents if any(k in s.lower() for k in keywords)]


def _parse_venue_from_sentences(sents):
    patterns = [
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS}\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS}\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
    ]
    for s in sents:
        for tmpl, mapping in patterns:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0
            total = sum(vals.values())
            if total < 5 or total > 15:
                continue
            return vals["w"], vals["d"], vals["l"]
    return None


def _parse_venue(soup, home_sents, away_sents, f):
    home_wdl, away_wdl = _parse_venue_from_stable(soup)

    if home_wdl:
        w, d, l = home_wdl
        total = w + d + l or 1
        pts = w * 3 + d
        f["home_home_wins"]   = w
        f["home_home_draws"]  = d
        f["home_home_losses"] = l
        f["home_home_pts"]    = pts
        f["home_home_ppg"]    = round(pts / total, 2)

    if away_wdl:
        w, d, l = away_wdl
        total = w + d + l or 1
        pts = w * 3 + d
        f["away_away_wins"]   = w
        f["away_away_draws"]  = d
        f["away_away_losses"] = l
        f["away_away_pts"]    = pts
        f["away_away_ppg"]    = round(pts / total, 2)

    # Prose fallback
    if "home_home_ppg" not in f and home_sents:
        result = _parse_venue_from_sentences(_venue_sentence(home_sents))
        if result:
            w, d, l = result
            total = w + d + l or 1
            pts = w * 3 + d
            f["home_home_wins"], f["home_home_draws"], f["home_home_losses"] = w, d, l
            f["home_home_pts"] = pts
            f["home_home_ppg"] = round(pts / total, 2)

    if "away_away_ppg" not in f and away_sents:
        result = _parse_venue_from_sentences(_venue_sentence(away_sents))
        if result:
            w, d, l = result
            total = w + d + l or 1
            pts = w * 3 + d
            f["away_away_wins"], f["away_away_draws"], f["away_away_losses"] = w, d, l
            f["away_away_pts"] = pts
            f["away_away_ppg"] = round(pts / total, 2)


# ═══════════════════════════════════════════════════════════════════
# FORM PARSING — the layer that had the table-pollution bug
# ═══════════════════════════════════════════════════════════════════

def _clean_polluted_sentence(s: str) -> str:
    """
    Strip table fragments that leak into prose.

    SportsGambler pages often have the standings table followed by a
    paragraph with no punctuation between them, so the whole block
    becomes one 'sentence' in the extracted body text. This function
    cuts off the leading table dump and keeps only the prose tail.
    """
    if not s:
        return s

    # "View Full Table" is the standard transition out of the standings table
    if "View Full Table" in s:
        s = s.split("View Full Table", 1)[1]

    # If we still see the standings header, cut past it
    if "G W D L GF GA GD GD P" in s:
        idx = s.rfind("G W D L GF GA GD GD P")
        s = s[idx + len("G W D L GF GA GD GD P"):]

    # Anything that still looks like a table row (numbers, short team names)
    # gets trimmed to the first real prose paragraph we can find.
    # Heuristic: find the first "<Team> – Last 10 League Games" heading
    m = re.search(r"(?:Last|Past|Previous)\s+10\s+(?:League|Liga|Serie A|Bundesliga|La Liga|Ligue 1|Primeira Liga)\s+Games", s)
    if m and m.start() > 0:
        s = s[m.start():]

    return s.strip()


def _parse_form_from_sentences(sents, required_first_word=None):
    """
    Parse overall form (W/D/L) from prose.

    Guards:
      1. Strip table-pollution from each sentence first.
      2. Exclude venue sentences (handled by the venue parser).
      3. Require the team name to appear early (first 120 chars).
      4. Require the W/D/L pattern to appear AFTER the team name.
      5. Validate total games (5-15).
      6. Score candidates and pick the best.
    """
    # --- Guard 1: de-pollute ---
    cleaned = [_clean_polluted_sentence(s) for s in sents]
    cleaned = [s for s in cleaned if s and len(s) > 20]

    # --- Guard 2: exclude venue sentences ---
    venue_sents = set(_venue_sentence(cleaned))
    filtered = [s for s in cleaned if s not in venue_sents]

    patterns = [
        # Explicit W L D variant (covers "3 victories, 5 defeats and 2 draws")
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        # W D L
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("w","d","l")),
        # W L D (no comma before and)
        (rf"{_NUM_PAT}\s+{_WIN_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("w","l","d")),
        # L W D
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_DRAW_WORDS}", ("l","w","d")),
        # L D W
        (rf"{_NUM_PAT}\s+{_LOSS_WORDS},?\s+{_NUM_PAT}\s+{_DRAW_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("l","d","w")),
        # D W L
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_WIN_WORDS},?\s+and\s+{_NUM_PAT}\s+{_LOSS_WORDS}", ("d","w","l")),
        # D L W
        (rf"{_NUM_PAT}\s+{_DRAW_WORDS},?\s+{_NUM_PAT}\s+{_LOSS_WORDS},?\s+and\s+{_NUM_PAT}\s+{_WIN_WORDS}", ("d","l","w")),
        # won X, lost Y and drawn Z
        (rf"won\s+{_NUM_PAT},?\s+lost\s+{_NUM_PAT}\s+and\s+drawn\s+{_NUM_PAT}", ("w","l","d")),
        (rf"won\s+{_NUM_PAT},?\s+drawn\s+{_NUM_PAT}\s+and\s+lost\s+{_NUM_PAT}", ("w","d","l")),
    ]

    req_low = (required_first_word or "").lower()
    candidates = []

    for s in filtered:
        s_low = s.lower()

        # --- Guard 3: team name must appear early ---
        if req_low:
            head = s_low[:120]
            if req_low not in head:
                continue

        # --- Guard 4: W/D/L pattern must come AFTER the team name ---
        if req_low:
            idx_team = s_low.find(req_low)
            m_win = re.search(rf"{_NUM_PAT}\s+{_WIN_WORDS}", s, re.IGNORECASE)
            if m_win and idx_team != -1 and m_win.start() < idx_team:
                continue

        for tmpl, mapping in patterns:
            m = re.search(tmpl, s, re.IGNORECASE)
            if not m:
                continue
            vals = {"w": 0, "d": 0, "l": 0}
            for i, key in enumerate(mapping):
                vals[key] = _num(m.group(i + 1)) or 0

            total = sum(vals.values())
            # --- Guard 5: total games sanity ---
            if total < 5 or total > 15:
                continue

            # --- Guard 6: score ---
            score = 0
            first_token = re.split(r"\s+", s_low.strip(), maxsplit=1)[0]
            if first_token == req_low:
                score += 10
            elif req_low in s_low[:40]:
                score += 5
            else:
                score += 1
            if total == 10:
                score += 3

            candidates.append((score, vals["w"], vals["d"], vals["l"], s))
            break  # one match per sentence

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, w, d, l, _ = candidates[0]
    return w, d, l


def _team_sentences(aliases, sentences):
    if not aliases:
        return []
    result = []
    seen_ids = set()
    for alias in aliases:
        if not alias:
            continue
        alias_low = alias.lower()
        pattern = re.compile(r"\b" + re.escape(alias_low) + r"\b")
        for s in sentences:
            if id(s) in seen_ids:
                continue
            if pattern.search(s.lower()):
                result.append(s)
                seen_ids.add(id(s))
    return result


def _parse_form(home_sents, away_sents, f):
    """Parse overall form (last 10) from prose. Store debug info."""
    home_short = f.get("home_team_short") or f.get("home_team", "")
    away_short = f.get("away_team_short") or f.get("away_team", "")

    home_form = _parse_form_from_sentences(home_sents, required_first_word=home_short)
    if home_form:
        w, d, l = home_form
        total = w + d + l or 1
        pts = w * 3 + d
        f["home_form_wins"], f["home_form_draws"], f["home_form_losses"] = w, d, l
        f["home_form_pts"] = pts
        f["home_form_ppg"] = round(pts / total, 2)

    away_form = _parse_form_from_sentences(away_sents, required_first_word=away_short)
    if away_form:
        w, d, l = away_form
        total = w + d + l or 1
        pts = w * 3 + d
        f["away_form_wins"], f["away_form_draws"], f["away_form_losses"] = w, d, l
        f["away_form_pts"] = pts
        f["away_form_ppg"] = round(pts / total, 2)

    # --- Debug: capture what was considered ---
    f["_debug_home_sents"] = [
        _clean_polluted_sentence(s)[:300]
        for s in home_sents
        if _clean_polluted_sentence(s)
    ]
    f["_debug_away_sents"] = [
        _clean_polluted_sentence(s)[:300]
        for s in away_sents
        if _clean_polluted_sentence(s)
    ]


def _parse_h2h(soup, home_term, away_term, f):
    rows = []

    section = soup.find("div", id="head-to-head")
    if section:
        rows = section.find_all("li", class_="team-stat-list-item")

    if not rows:
        for h in soup.find_all(["h3", "h4", "div"], string=re.compile("head to head", re.I)):
            parent = h.find_parent()
            if parent:
                candidate_rows = parent.find_all("li", class_="team-stat-list-item")
                if candidate_rows:
                    rows = candidate_rows
                    break

    if not rows:
        f["_parse_warnings"].append("H2H: no rows found")
        return

    home_wins = away_wins = draws = 0
    for row in rows:
        text = row.get_text(" ", strip=True)
        nums = re.findall(r"\b(\d+)\b", text)
        if len(nums) < 2:
            continue
        hs, aws = int(nums[-2]), int(nums[-1])

        idx_home = text.lower().find((home_term or "").lower())
        idx_away = text.lower().find((away_term or "").lower())
        home_first = idx_home != -1 and (idx_away == -1 or idx_home < idx_away)

        if hs > aws:
            if home_first: home_wins += 1
            else: away_wins += 1
        elif aws > hs:
            if home_first: away_wins += 1
            else: home_wins += 1
        else:
            draws += 1

    f["h2h_home_wins"] = home_wins
    f["h2h_away_wins"] = away_wins
    f["h2h_draws"] = draws
    f["h2h_total"] = home_wins + away_wins + draws


def extract_facts(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    f = {"_parse_warnings": []}
    body = soup.get_text(" ", strip=True)

    # --- Identity ---
    f["match_date"] = f["kickoff_time"] = None
    f["league"] = f["country"] = f["venue"] = None
    f["home_team"] = f["away_team"] = None
    f["home_team_short"] = f["away_team_short"] = None

    _extract_identity(soup, f)

    if not f.get("home_team"):
        f["_parse_warnings"].append("home_team not found")
    if not f.get("away_team"):
        f["_parse_warnings"].append("away_team not found")

    # --- Standings ---
    _parse_standings(soup, f)

    # --- Sentences ---
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]

    home_aliases = _team_aliases(soup, f.get("home_team", ""), f.get("home_team_short", ""))
    away_aliases = _team_aliases(soup, f.get("away_team", ""), f.get("away_team_short", ""))

    home_sents = _team_sentences(home_aliases, sentences)
    away_sents = _team_sentences(away_aliases, sentences)

    # --- Form ---
    _parse_form(home_sents, away_sents, f)
    if "home_form_ppg" not in f:
        f["_parse_warnings"].append(f"home form not found for {f.get('home_team')}")
    if "away_form_ppg" not in f:
        f["_parse_warnings"].append(f"away form not found for {f.get('away_team')}")

    # --- Venue ---
    _parse_venue(soup, home_sents, away_sents, f)
    if "home_home_ppg" not in f:
        f["_parse_warnings"].append("home venue form not found")
    if "away_away_ppg" not in f:
        f["_parse_warnings"].append("away venue form not found")

    # --- H2H ---
    _parse_h2h(soup,
               f.get("home_team_short") or f.get("home_team", ""),
               f.get("away_team_short") or f.get("away_team", ""),
               f)

    # --- Validation ---
    required = [
        "home_table_pts", "away_table_pts",
        "home_form_ppg",  "away_form_ppg",
        "home_home_ppg",  "away_away_ppg",
        "h2h_home_wins",  "h2h_away_wins",
    ]
    for key in required:
        if f.get(key) is None:
            f["_parse_warnings"].append(f"missing required field: {key}")

    for k in [
        "home_table_gd", "away_table_gd",
        "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_total",
    ]:
        if k not in f:
            f[k] = None

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
    league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""

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
    league = m.get("league", "") or ""
    date = m.get("match_date", "") or ""

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
    row.pop("_parse_warnings", None)
    row.pop("home_team_short", None)
    row.pop("away_team_short", None)
    row.pop("_debug_home_sents", None)
    row.pop("_debug_away_sents", None)
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
    return {k: v for k, v in row.items() if k in allowed and v is not None}


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

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠️ Parser warnings ({len(warnings)})", expanded=True):
                for w in warnings:
                    st.write(f"- {w}")

        st.markdown("### Extracted Facts")
        display_facts = {k: v for k, v in facts.items()
                         if v is not None and not k.startswith("_")}
        st.json(display_facts)

        # --- Debug expander ---
        if facts.get("_debug_home_sents") or facts.get("_debug_away_sents"):
            with st.expander("🔍 Debug: sentences considered for form", expanded=False):
                st.markdown("**Home team sentences (after table-pollution cleanup):**")
                for s in facts.get("_debug_home_sents", []):
                    st.write(f"- `{s}`")
                st.markdown("**Away team sentences (after table-pollution cleanup):**")
                for s in facts.get("_debug_away_sents", []):
                    st.write(f"- `{s}`")

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

                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"])
                    st.success("Deleted.")
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
        df["qualifying"] = pd.to_numeric(df["qualifying"], errors="coerce").fillna(0)

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
