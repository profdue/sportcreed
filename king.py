#!/usr/bin/env python3
"""
king.py — The Logic (text-dump only, header-driven, market veto, ELO soft warning)

Classifier: 2 stats — consecutive wins + consecutive unbeaten.
Filters:
  - Market veto (hard): pick side must be the moneyline favourite
  - ELO check (soft): if pick ELO < 1650 or unknown, warn but do not reject
"""

import re
import unicodedata
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
from supabase import create_client, Client

st.set_page_config(page_title="The Logic", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { font-family: 'Poppins', sans-serif; letter-spacing: -0.02em; }
    .match-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232a3d 100%);
        border-radius: 16px; padding: 22px 26px; margin-bottom: 16px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .match-card.pick-home { border-left-color: #10b981; }
    .match-card.pick-away { border-left-color: #f59e0b; }
    .match-card.skip      { border-left-color: #64748b; opacity: 0.9; }
    .match-card.win       { border-left-color: #10b981; }
    .match-card.loss      { border-left-color: #ef4444; }
    .match-title { font-size: 1.25rem; font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0; }
    .match-meta  { font-size: 0.82rem; color: #94a3b8; margin-bottom: 14px; }
    .pick-badge {
        display: inline-block; padding: 6px 14px; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em;
    }
    .badge-solid  { background: #06d6a022; color: #06d6a0; border: 1px solid #06d6a055; }
    .badge-light  { background: #38bdf822; color: #38bdf8; border: 1px solid #38bdf855; }
    .badge-notbe  { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2455; }
    .badge-nobet  { background: #64748b22; color: #94a3b8; border: 1px solid #64748b55; }
    .badge-win    { background: #10b98122; color: #10b981; border: 1px solid #10b98155; }
    .badge-loss   { background: #ef444422; color: #ef4444; border: 1px solid #ef444455; }
    .signal-row { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
    .signal-pill {
        font-size: 0.8rem; padding: 5px 11px; border-radius: 8px;
        background: #0f172a; color: #cbd5e1; border: 1px solid #1e293b;
        font-family: monospace;
    }
    .signal-pill.home { background: #10b98115; color: #34d399; border-color: #10b98140; }
    .signal-pill.away { background: #f59e0b15; color: #fbbf24; border-color: #f59e0b40; }
    .signal-pill.odds { background: #6366f115; color: #a5b4fc; border-color: #6366f140; }
    .signal-pill.elo  { background: #a855f715; color: #c084fc; border-color: #a855f740; }
    .elo-warning {
        background: #ef444422; color: #fca5a5;
        padding: 8px 12px; border-radius: 8px; margin-top: 10px;
        font-size: 0.85rem; font-weight: 600;
        border: 1px solid #ef444455;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 44px; background-color: #1a1f2e; border-radius: 10px;
        padding: 0 18px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] { background-color: #6366f1 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ELO DATABASE (Threshold: 1650)
# ═══════════════════════════════════════════════════════════════════

ELO_MINIMUM = 1650

ELO_DB = {
    # ══════════ DOMESTIC CLUBS (EUROPE) ══════════
    # England
    "arsenal": 2047, "man city": 2002, "manchester city": 2002, "liverpool": 1937,
    "brighton": 1910, "aston villa": 1899, "bournemouth": 1892, "chelsea": 1885,
    "newcastle": 1853, "man united": 1851, "manchester united": 1851, 
    "brentford": 1848, "everton": 1831, "tottenham": 1814, "leeds": 1812, 
    "leeds united": 1812, "crystal palace": 1802, "forest": 1801, 
    "nottingham forest": 1801, "fulham": 1800, "sunderland": 1727, 
    "west ham": 1748, "wolves": 1716, "southampton": 1709, "coventry": 1701, 
    "ipswich": 1698, "hull": 1640, "middlesbrough": 1663, "birmingham": 1655,
    
    # Spain
    "barcelona": 2033, "real madrid": 2001, "atletico": 1910, "atletico madrid": 1910,
    "betis": 1829, "real sociedad": 1790, "villarreal": 1789, "athletic club": 1768,
    "athletic bilbao": 1768, "celta": 1738, "valencia": 1736, "rayo vallecano": 1736,
    "sevilla": 1720, "alaves": 1709, "levante": 1709, "osasuna": 1705,
    "elche": 1705, "mallorca": 1719, "girona": 1713, "almeria": 1711,
    "castellon": 1708, "eibar": 1677, "burgos": 1665, "las palmas": 1659,
    "oviedo": 1648, "granada": 1640,

    # Italy
    "internazionale": 1949, "inter": 1949, "roma": 1865, "como": 1838, 
    "juventus": 1830, "milan": 1820, "napoli": 1818, "lazio": 1808, 
    "atalanta": 1777, "bologna": 1775, "fiorentina": 1717, "frosinone": 1701,
    "torino": 1686, "genoa": 1682, "sassuolo": 1666, "udinese": 1659, 
    "cagliari": 1659, "parma": 1645, "venezia": 1645, "palermo": 1662,
    
    # Germany
    "bayern munchen": 2046, "bayern munich": 2046, "bayern": 2046,
    "dortmund": 1899, "b. dortmund": 1899, "leverkusen": 1859, 
    "rb leipzig": 1832, "stuttgart": 1819, "freiburg": 1793, "hoffenheim": 1739,
    "frankfurt": 1731, "mainz": 1721, "werder": 1710, "werder bremen": 1710,
    "augsburg": 1704, "hamburg": 1683, "elversberg": 1681, "union berlin": 1679,
    "paderborn": 1676, "gladbach": 1663, "koln": 1651,
    
    # France
    "paris sg": 1989, "psg": 1989, "monaco": 1790, "lyon": 1784, 
    "lille": 1780, "marseille": 1780, "lens": 1779, "rennes": 1745,
    "strasbourg": 1720, "paris fc": 1716, "toulouse": 1697, "brest": 1695,
    "lorient": 1695, "nice": 1692,
    
    # Portugal
    "benfica": 1870, "sporting": 1863, "sporting cp": 1863, "porto": 1843, 
    "braga": 1710,
    
    # Netherlands
    "psv": 1782, "ajax": 1716, "feyenoord": 1716, "twente": 1689, "az": 1681,
    
    # Belgium
    "brugge": 1839, "club brugge": 1839, "st gillis": 1826, "union sg": 1826,
    
    # Scotland
    "celtic": 1725, "rangers": 1710, 

    # Russia
    "zenit": 1824, "spartak moskva": 1738, "fc krasnodar": 1736, 
    "dynamo moskva": 1702, "cska moskva": 1695,
    
    # Turkey
    "galatasaray": 1824, "fenerbahce": 1779, "amed sk": 1710, "besiktas": 1705, 
    "trabzonspor": 1700,
    
    # Greece/Cyprus
    "olympiakos": 1747, "olympiacos": 1747, "aek": 1695, "panathinaikos": 1679,
    "paok": 1658, "omonia": 1632, "larnaca": 1598, "paphos": 1576,

    # ══════════ DOMESTIC CLUBS (AMERICAS) ══════════
    # Brazil
    "flamengo": 1883, "palmeiras": 1862, "cruzeiro": 1786, "fluminense": 1769,
    "bahia": 1728, "santos": 1727, "mirassol": 1718, "vasco da gama": 1717,
    "sao paulo": 1715, "ath paranaense": 1714, "corinthians": 1713,
    "atletico mineiro": 1711, "botafogo": 1709, "bragantino": 1699,
    "internacional": 1698, "gremio": 1695, "fortaleza": 1688, "juventude": 1683,
    "criciuma": 1656, "vitoria": 1652, 

    # Argentina
    "boca juniors": 1791, "rosario central": 1762, "river plate": 1749, 
    "argentinos junio": 1747, "velez sarsfield": 1724, "ind rivadavia": 1721, 
    "estudiantes": 1715, "lanus": 1707, "gimnasia lp": 1703, "instituto": 1701, 
    "racing club": 1693, "independiente": 1692, "belgrano": 1689, "huracan": 1679, 
    "defensa y justic": 1659, "tigre": 1656, "platense": 1653, 
    "barracas central": 1653, "union de santa f": 1647,

    # South America other
    "ind del valle": 1797, "ldu quito": 1704, "macara": 1681, "uni catolica": 1674,
    "barcelona sc": 1640, "colo-colo": 1671, "uni de chile": 1658, 
    "uni catolica chile": 1658, "coquimbo": 1656, "penarol": 1626, 
    "nacional potosi": 1488, "bolivar": 1696, "always ready": 1626, 
    "the strongest": 1606,
    
    # North/Central America
    "cruz azul": 1734, "toluca": 1720, "america": 1707, "guadalajara": 1683,
    "pachuca": 1658, "uanl": 1653, "tigres uanl": 1653, "monterrey": 1636,
    "nashville": 1703, "inter miami": 1679, "vancouver": 1668, "philadelphia": 1650,
    "lafc": 1644, "los angeles fc": 1644,
    
    # ══════════ DOMESTIC CLUBS (ASIA/AFRICA) ══════════
    "vissel kobe": 1688, "sanfrecce hirosh": 1687, "fc tokyo": 1664, 
    "kashiwa reysol": 1657, "al hilal": 1745, "al-hilal": 1745, 
    "al hilal sfc": 1745, "al nassr": 1701, "al-nassr": 1701, 
    "al ahli sfc": 1645, "al-ahli": 1645, "al ahly": 1604, "pyramids": 1601,
    "mamelodi sundown": 1621, "mamelodi sundowns": 1621,

    # ══════════ NATIONAL TEAMS ══════════
    "spain": 2290, "argentina": 2175, "england": 2150, "france": 2066, "portugal": 2055,
    "brazil": 2001, "colombia": 1990, "switzerland": 1975, "netherlands": 1974, "belgium": 1971,
    "italy": 1948, "norway": 1901, "japan": 1897, "morocco": 1896, "germany": 1895,
    "mexico": 1868, "denmark": 1861, "croatia": 1852, "uruguay": 1851, "ecuador": 1849,
    "austria": 1841, "paraguay": 1838, "senegal": 1808, "greece": 1804, "australia": 1794,
    "united states": 1792, "usa": 1792, "russia": 1787, "sweden": 1774, "turkey": 1753,
    "algeria": 1744, "canada": 1739, "iran": 1738, "venezuela": 1732, "ivory coast": 1728,
    "hungary": 1725, "ukraine": 1724, "chile": 1723, "scotland": 1718, "slovenia": 1718,
    "south korea": 1717, "egypt": 1716, "nigeria": 1714, "ireland": 1711, "kosovo": 1710,
    "dr congo": 1691, "poland": 1685, "serbia": 1683, "peru": 1681, "wales": 1681,
    "slovakia": 1678, "northern ireland": 1672, "bosnia/herzegovina": 1672, "czechia": 1653,
    "czech republic": 1653, "saudi arabia": 1652, "panama": 1650,
    
    # National teams below threshold (kept for warning display)
    "uzbekistan": 1636, "mali": 1629, "georgia": 1628, "albania": 1628, "cameroon": 1627,
    "jordan": 1622, "israel": 1619, "bolivia": 1613, "iceland": 1597, "costa rica": 1586,
    "south africa": 1580, "uae": 1570, "cape verde": 1566, "haiti": 1563, "angola": 1562,
    "new zealand": 1561, "romania": 1554, "jamaica": 1543, "belarus": 1539, "iraq": 1534,
    "north macedonia": 1533, "burkina faso": 1529, "guatemala": 1524, "honduras": 1520,
    "ghana": 1514, "finland": 1514, "palestine": 1511, "curacao": 1495, "montenegro": 1494,
    "tunisia": 1485, "oman": 1469, "guinea": 1468, "gambia": 1464, "qatar": 1461,
    "syria": 1457, "suriname": 1453, "uganda": 1452, "bulgaria": 1450, "benin": 1448,
    "libya": 1442, "gabon": 1429, "estonia": 1406, "vietnam": 1400, "faroe islands": 1397,
    "equatorial guinea": 1391, "rwanda": 1387, "luxembourg": 1383, "zimbabwe": 1382,
    "zambia": 1381, "thailand": 1379, "madagascar": 1371, "niger": 1370, "kazakhstan": 1367,
    "togo": 1366, "comoros": 1364, "kenya": 1360, "guinea-bissau": 1355, "mozambique": 1353,
    "cyprus": 1352, "armenia": 1348, "trinidad and tobago": 1344, "china": 1344,
    "sudan": 1337, "malaysia": 1331, "mauritania": 1329, "indonesia": 1328,
    "guyana": 1327, "azerbaijan": 1326, "el salvador": 1324, "bahrain": 1314,
    "kyrgyzstan": 1313, "kuwait": 1311, "moldova": 1308, "sierra leone": 1303,
    "burundi": 1302, "namibia": 1301, "ethiopia": 1299, "botswana": 1295, "tajikistan": 1290,
    "lebanon": 1288, "lithuania": 1280, "tanzania": 1273, "liberia": 1270, "malta": 1258,
    "nicaragua": 1255, "cuba": 1250, "malawi": 1242, "yemen": 1242, "latvia": 1241,
    "congo": 1220, "turkmenistan": 1209, "eritrea": 1201
}

def _normalize_team_name(name: str) -> str:
    if not name:
        return ""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8').lower()
    name = re.sub(r'[^a-z0-9 ]', '', name).strip()
    return name

def _lookup_elo(team_name):
    """
    Exact match first, then word-boundary match.
    Fixes the bug where 'Mozambique' fuzzy-matched 'Club Brugge'.
    """
    if not team_name:
        return None
    name = _normalize_team_name(team_name)
    if not name:
        return None

    # Exact match
    if name in ELO_DB:
        return ELO_DB[name]

    # Word-boundary match: the DB key must appear as a whole word in the team name
    # (or vice versa), not just as a substring.
    name_words = set(name.split())
    for k, v in ELO_DB.items():
        k_words = set(k.split())
        # Any shared word of length >= 4 is a strong signal
        shared = {w for w in name_words & k_words if len(w) >= 4}
        if shared:
            return v

    return None


# ═══════════════════════════════════════════════════════════════════
# SUPABASE
# ═══════════════════════════════════════════════════════════════════

@st.cache_resource
def get_supabase() -> Client:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
    except (KeyError, TypeError):
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

def db_insert(row: dict):
    sb = get_supabase()
    res = sb.table("matches").insert(row).execute()
    st.cache_data.clear()
    return res.data[0] if res.data else None

@st.cache_data(ttl=30)
def db_pending():
    sb = get_supabase()
    return (sb.table("matches").select("*").eq("simple_qualifying", 1).is_("home_score", "null").order("extracted_at", desc=True).execute().data or [])

@st.cache_data(ttl=30)
def db_settled():
    sb = get_supabase()
    return (sb.table("matches").select("*").eq("simple_qualifying", 1).not_.is_("home_score", "null").order("settled_at", desc=True).execute().data or [])

@st.cache_data(ttl=30)
def db_study():
    sb = get_supabase()
    return (sb.table("matches").select("*").eq("simple_qualifying", 0).order("extracted_at", desc=True).execute().data or [])

def db_find_existing(row: dict):
    sb = get_supabase()
    res = (sb.table("matches").select("*").eq("home_team", row.get("home_team")).eq("away_team", row.get("away_team")).limit(1).execute())
    return res.data[0] if res.data else None

def db_settle(match_id: int, home_score: int, away_score: int, result: str, dc_hit, profit_loss):
    sb = get_supabase()
    sb.table("matches").update({
        "home_score": home_score, "away_score": away_score, "result": result,
        "dc_hit": dc_hit, "profit_loss": profit_loss, "settled_at": datetime.now(timezone.utc).isoformat(),
    }).eq("match_id", match_id).execute()
    st.cache_data.clear()

def db_delete(match_id: int):
    get_supabase().table("matches").delete().eq("match_id", match_id).execute()
    st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════
# HELPERS & ODDS PARSER
# ═══════════════════════════════════════════════════════════════════

def _to_int_or_none(v):
    if v is None: return None
    try:
        if v != v: return None
    except Exception: pass
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"): return None
    try: return int(float(s))
    except (TypeError, ValueError): return None

def _clean_line(s: str) -> str:
    return s.strip().strip("\ufeff\u200b\u200c\u200d\u00a0 ")

def _strip_lines(text: str) -> list:
    return [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]

def _next_nonblank(lines, start):
    i = start
    while i < len(lines) and not _clean_line(lines[i]): i += 1
    if i >= len(lines): return None, i
    return _clean_line(lines[i]), i + 1

_ARROWS = {"➜", "➔", "→", "->", "=>", ">", "➡", "⬈", "⬊"}
_MARKET_MAP = {"1": "1", "X": "X", "x": "X", "2": "2", "×": "X", "✕": "X"}
_DASHES = {"-", "–", "—", "N/A", "n/a"}
_ODDS_STOP_LABELS = {"general stats", "match information", "recent form", "h2h last matches", "consecutive scoring", "consecutive wins", "consecutive unbeaten"}
_OTHER_BOOKMAKERS = {"betking", "bet9ja", "betway", "sportybet", "bet365", "pinnacle", "1xbet"}

def _find_odds_header(lines):
    for i, raw in enumerate(lines):
        s = _clean_line(raw)
        if not s: continue
        low = s.lower()
        if low == "pre-match odds" or ("odds" in low and "pre" in low and "match" in low): return i + 1
    return None

def _parse_odds_block(lines):
    start = _find_odds_header(lines)
    if start is None: return {}

    out = {}
    current_book = {}

    def has_prices(d): return d.get("1") is not None or d.get("X") is not None or d.get("2") is not None

    i = start
    guard = 0
    while i < len(lines) and guard < 200:
        guard += 1
        s = _clean_line(lines[i])
        if not s:
            i += 1
            continue

        low = s.lower()
        if low in _ODDS_STOP_LABELS: break

        if low in _OTHER_BOOKMAKERS:
            if has_prices(current_book): break
            else:
                current_book = {}
                i += 1
                continue

        if s in _MARKET_MAP:
            market = _MARKET_MAP[s]
            if market in current_book:
                if has_prices(current_book): break
                else: current_book = {}

            nxt, j = _next_nonblank(lines, i + 1)
            if nxt is None:
                i += 1
                continue

            if nxt in _DASHES:
                i = j
                continue

            if nxt in _ARROWS:
                price, k = _next_nonblank(lines, j)
                i = k
            else:
                price, k = nxt, j
                i = k

            if price and price not in _DASHES:
                try: current_book[market] = float(price)
                except (TypeError, ValueError): pass
            continue
        i += 1

    if has_prices(current_book): return current_book
    return {}

def _market_favourite(odds):
    h, a = odds.get("1"), odds.get("2")
    if h is None or a is None: return None
    if abs(h - a) < 0.05: return "tie"
    return "home" if h < a else "away"


# ═══════════════════════════════════════════════════════════════════
# CLASSIFIER — ELO IS NOW A WARNING, NOT A VETO
# ═══════════════════════════════════════════════════════════════════

NOT_BEATEN_MIN_GAP = 3

def _classify_streaks(hw, aw, hu, au):
    def result(category=None, pick=None, bet=None, qualifying=0, reason="", code=None):
        return {"category": category, "pick_team": pick, "bet": bet, "qualifying": qualifying, "reason": reason, "rejection_code": code}

    home_has, away_has = hw > 0, aw > 0

    if home_has and not away_has:
        if hu >= 2 and au == 0: return result("SOLID CLEAR", "home", "1X", 1, f"Home {hw}W/{hu}U vs Away {aw}W/{au}U — SOLID (opponent 0/0).")
        if hu == hw: return result("LIGHT CLEAR HOME", "home", "1X", 1, f"Home {hw}W/{hu}U (unbeaten = wins) vs Away {aw}W/{au}U — LIGHT-HOME.")
        return result(reason=f"Home leader {hw}W/{hu}U but unbeaten != wins and SOLID not met.", code="HOME_LEADER_NO_PADDING")

    if away_has and not home_has:
        if au >= 2 and hu == 0: return result("SOLID CLEAR", "away", "X2", 1, f"Away {aw}W/{au}U vs Home {hw}W/{hu}U — SOLID (opponent 0/0).")
        return result(reason=f"Away leader {aw}W/{au}U — LIGHT-away removed; SOLID not met.", code="AWAY_LEADER_NO_SOLID")

    if not home_has and not away_has:
        gap = abs(hu - au)
        if gap < NOT_BEATEN_MIN_GAP: return result(reason=f"Both 0 wins, unbeaten gap {gap} < {NOT_BEATEN_MIN_GAP}.", code="NOT_BEATEN_GAP_LOW")
        if hu > au: return result("NOT-BEATEN SIGNAL", "home", "1X", 1, f"Both 0W; Home unbeaten {hu} vs Away {au} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")
        return result("NOT-BEATEN SIGNAL", "away", "X2", 1, f"Both 0W; Away unbeaten {au} vs Home {hu} (gap {gap} >= {NOT_BEATEN_MIN_GAP}).")

    return result(reason=f"Both teams have wins ({hw} vs {aw}) — outside the file.", code="BOTH_HAVE_WINS")


def classify_simple(home_wins, away_wins, home_unbeaten, away_unbeaten,
                    market_favourite=None, home_elo=None, away_elo=None):
    hw, aw, hu, au = _to_int_or_none(home_wins), _to_int_or_none(away_wins), _to_int_or_none(home_unbeaten), _to_int_or_none(away_unbeaten)

    def result(category=None, pick=None, bet=None, qualifying=0, reason="", code=None):
        return {"category": category, "pick_team": pick, "bet": bet, "qualifying": qualifying, "reason": reason, "rejection_code": code}

    if None in (hw, aw, hu, au):
        return result(reason="Missing one or more streak inputs.", code="MISSING")

    streak = _classify_streaks(hw, aw, hu, au)
    if not streak["qualifying"]: return streak

    pick = streak["pick_team"]

    # ── ELO CHECK (SOFT WARNING) ──
    pick_elo = home_elo if pick == "home" else away_elo
    elo_warning = ""
    if pick_elo is None:
        elo_warning = " | ⚠️ ELO UNKNOWN"
    elif pick_elo < ELO_MINIMUM:
        elo_warning = f" | ⚠️ ELO {pick_elo} < {ELO_MINIMUM}"

    # ── MARKET CHECK (HARD VETO) ──
    if market_favourite is None:
        return result(reason=f"{streak['reason']}{elo_warning} [no odds available]", code="MARKET_UNAVAILABLE")
    if market_favourite != "tie" and market_favourite != pick:
        return result(reason=f"{streak['reason']}{elo_warning} [market favourite={market_favourite}, pick={pick}]", code="MARKET_DISAGREES")

    qualified = dict(streak)
    qualified["reason"] = f"{streak['reason']}{elo_warning}"
    return qualified


def compute_simple_row(facts: dict) -> dict:
    c = classify_simple(
        facts.get("home_consecutive_wins"), facts.get("away_consecutive_wins"),
        facts.get("home_consecutive_unbeaten"), facts.get("away_consecutive_unbeaten"),
        market_favourite=facts.get("market_favourite"),
        home_elo=facts.get("home_elo"), away_elo=facts.get("away_elo")
    )
    return {
        "simple_category": c["category"], "simple_pick_team": c["pick_team"],
        "simple_bet": c["bet"], "simple_qualifying": c["qualifying"],
        "simple_reason": c["reason"], "simple_rejection_code": c["rejection_code"],
        "home_consecutive_wins": _to_int_or_none(facts.get("home_consecutive_wins")),
        "away_consecutive_wins": _to_int_or_none(facts.get("away_consecutive_wins")),
        "home_consecutive_unbeaten": _to_int_or_none(facts.get("home_consecutive_unbeaten")),
        "away_consecutive_unbeaten": _to_int_or_none(facts.get("away_consecutive_unbeaten")),
    }

def settle_simple(simple_bet, home_score, away_score):
    if home_score > away_score: result = "home_win"
    elif away_score > home_score: result = "away_win"
    else: result = "draw"
    if simple_bet == "1X": hit = result in ("home_win", "draw")
    elif simple_bet == "X2": hit = result in ("away_win", "draw")
    else: return result, None, 0.0
    profit = 1.0 if hit else -1.0
    return result, (1 if hit else 0), profit


# ═══════════════════════════════════════════════════════════════════
# TEXT-DUMP PARSER
# ═══════════════════════════════════════════════════════════════════

_CONSEC_HEADER = re.compile(r"^Consecutive\s+(scoring|wins|unbeaten)\s*$", re.I)
_RECORD_PAREN  = re.compile(r"\(Record\s+(\d+)\)", re.I)
_SCORE_LINE    = re.compile(r"^\d+\s*-\s*\d+$")
_FORM_RESULT   = re.compile(r"^[WDL]$")
_VS_HEADER     = re.compile(r"^(?:(?:pre|re|e)[-–— ]?match\s+|match\s+)?(.+?)\s+vs\.?\s+(.+?)\s*$", re.I)

_ODDS_GLYPHS = "➜➔→➡•€$£"
_STRUCTURAL_LABELS = {"match information", "recent form", "general stats", "pre-match odds", "h2h last matches", "odds", "advertisement"}
_HEADER_JUNK_PREFIXES = ("pre-match ", "pre match ", "pre–match ", "re-match ", "re match ", "e-match ", "e match ", "match ")
_STADIUM_KEYWORDS = ("stadium", "arena", "park", "ground", "field", "centre", "center", "stadion", "estadio", "estádio", "stade", "stadio", "olympic", "olympique", "coliseum", "sports complex")

def _looks_like_text_dump(text: str) -> bool:
    low = text.lower()
    return ("pre-match" in low and " vs " in low) or ("re-match" in low and " vs " in low) or ("recent form" in low and "consecutive" in low)

def _looks_like_odds_line(s: str) -> bool:
    if any(sym in s for sym in _ODDS_GLYPHS): return True
    return len(re.findall(r"\b\d+\.\d+\b", s)) >= 2

def _looks_like_person_name(s: str) -> bool:
    tokens = s.split()
    if len(tokens) < 2 or len(tokens) > 4: return False
    for tok in tokens:
        clean = tok.replace("'", "").replace("’", "").replace("-", "").replace(".", "").replace(" ", "")
        if not clean or not clean[0].isupper() or not all(c.isalpha() for c in clean): return False
    return True

def _strip_header_prefix(name: str) -> str:
    low = name.lower()
    for marker in _HEADER_JUNK_PREFIXES:
        if low.startswith(marker): return name[len(marker):].strip()
    return name

def _teams_from_header(lines):
    for ln in lines:
        s = _clean_line(ln)
        if not s or s.lower() in _STRUCTURAL_LABELS or _looks_like_odds_line(s): continue
        m = _VS_HEADER.match(s)
        if not m: continue
        home = _strip_header_prefix(_clean_line(m.group(1)))
        away = _strip_header_prefix(_clean_line(m.group(2)))
        if not home or not away or home.isdigit() or away.isdigit(): continue
        return home, away
    return None, None

def _parse_match_information(lines, f):
    for i, ln in enumerate(lines):
        if _clean_line(ln).lower() != "match information": continue
        val, j = _next_nonblank(lines, i + 1)
        if not val: return
        entries = []
        while val and val.lower() != "recent form" and len(entries) < 5:
            entries.append(val); val, j = _next_nonblank(lines, j)
        if not entries: return
        for entry in entries:
            if any(kw in entry.lower() for kw in _STADIUM_KEYWORDS):
                f["venue"] = entry; return
        if not _looks_like_person_name(entries[0]):
            f["venue"] = entries[0]
        return

def _teams_frequency_fallback(lines):
    try:
        rf_idx = next(i for i, ln in enumerate(lines) if _clean_line(ln).lower() == "recent form")
    except StopIteration: return None, None
    _, i = _next_nonblank(lines, rf_idx + 1)
    nxt, i = _next_nonblank(lines, i)
    if nxt and not _FORM_RESULT.match(nxt): nxt, i = _next_nonblank(lines, i)
    cur = nxt
    names = []
    while cur and _FORM_RESULT.match(cur):
        team, i = _next_nonblank(lines, i)
        score, i = _next_nonblank(lines, i)
        if not team or not score or not _SCORE_LINE.match(score): break
        names.append(team)
        cur, i = _next_nonblank(lines, i)
        if not cur or not _FORM_RESULT.match(cur): break
    from collections import Counter
    if not names: return None, None
    c = Counter(names)
    if len(c) < 2: return None, None
    top = [n for n, _ in c.most_common(2)]
    first_seen = {}
    for idx, nm in enumerate(names): first_seen.setdefault(nm, idx)
    top.sort(key=lambda nm: first_seen[nm])
    return top[0], top[1]

def parse_recent_form_dump(text: str) -> dict:
    f = {
        "home_team": None, "away_team": None, "venue": None, "league": None,
        "match_date": None, "kickoff_time": None, "country": None,
        "home_consecutive_scoring": None, "away_consecutive_scoring": None,
        "home_consecutive_wins": None, "away_consecutive_wins": None,
        "home_consecutive_unbeaten": None, "away_consecutive_unbeaten": None,
        "odds": {}, "market_favourite": None,
        "home_elo": None, "away_elo": None,
        "recent_form_triplets": [], "_parse_warnings": [],
    }
    lines = _strip_lines(text)
    n = len(lines)

    _parse_match_information(lines, f)

    header_home, header_away = _teams_from_header(lines)
    if header_home and header_away:
        f["home_team"], f["away_team"] = header_home, header_away
    else:
        h, a = _teams_frequency_fallback(lines)
        f["home_team"], f["away_team"] = h, a
        if h and a: f["_parse_warnings"].append("No 'Pre-match X vs Y' header — team order guessed.")

    f["home_elo"] = _lookup_elo(f["home_team"])
    f["away_elo"] = _lookup_elo(f["away_team"])

    odds = _parse_odds_block(lines)
    f["odds"] = odds
    f["market_favourite"] = _market_favourite(odds)

    for i, ln in enumerate(lines):
        m = _CONSEC_HEADER.match(_clean_line(ln))
        if not m: continue
        kind = m.group(1).lower()
        home_val, j = _next_nonblank(lines, i + 1)
        if home_val is None: continue
        while j < n and _RECORD_PAREN.search(lines[j]): j += 1
        away_val, _ = _next_nonblank(lines, j)
        hv, av = _to_int_or_none(home_val), _to_int_or_none(away_val)
        if kind == "scoring": f["home_consecutive_scoring"], f["away_consecutive_scoring"] = hv, av
        elif kind == "wins": f["home_consecutive_wins"], f["away_consecutive_wins"] = hv, av
        elif kind == "unbeaten": f["home_consecutive_unbeaten"], f["away_consecutive_unbeaten"] = hv, av

    try:
        rf_idx = next(i for i, ln in enumerate(lines) if _clean_line(ln).lower() == "recent form")
    except StopIteration: rf_idx = -1
    if rf_idx >= 0:
        _, i = _next_nonblank(lines, rf_idx + 1)
        nxt, i = _next_nonblank(lines, i)
        if nxt and not _FORM_RESULT.match(nxt): nxt, i = _next_nonblank(lines, i)
        cur = nxt
        while cur and _FORM_RESULT.match(cur) and len(f["recent_form_triplets"]) < 20:
            team, i = _next_nonblank(lines, i)
            score, i = _next_nonblank(lines, i)
            if not team or not score or not _SCORE_LINE.match(score): break
            f["recent_form_triplets"].append({"result": cur.upper(), "team": team, "score": score})
            cur, i = _next_nonblank(lines, i)
            if not cur or not _FORM_RESULT.match(cur): break

    if not f["home_team"]: f["_parse_warnings"].append("home_team not found")
    if not f["away_team"]: f["_parse_warnings"].append("away_team not found")
    if f["market_favourite"] is None: f["_parse_warnings"].append("Odds block missing or unparseable.")

    return f


# ═══════════════════════════════════════════════════════════════════
# ROW BUILDER & CARD RENDERER
# ═══════════════════════════════════════════════════════════════════

def build_row(facts: dict) -> dict:
    simple = compute_simple_row(facts)
    now_iso = datetime.now(timezone.utc).isoformat()
    odds = facts.get("odds", {})

    row = {
        "home_team": facts.get("home_team"), "away_team": facts.get("away_team"),
        "venue": facts.get("venue"), "league": facts.get("league"),
        "match_date": facts.get("match_date"), "kickoff_time": facts.get("kickoff_time"), "country": facts.get("country"),
        "home_consecutive_wins": simple["home_consecutive_wins"], "away_consecutive_wins": simple["away_consecutive_wins"],
        "home_consecutive_unbeaten": simple["home_consecutive_unbeaten"], "away_consecutive_unbeaten": simple["away_consecutive_unbeaten"],
        "simple_category": simple["simple_category"], "simple_pick_team": simple["simple_pick_team"],
        "simple_bet": simple["simple_bet"], "simple_qualifying": simple["simple_qualifying"],
        "simple_reason": simple["simple_reason"], "simple_rejection_code": simple["simple_rejection_code"],
        "odds_1": odds.get("1"), "odds_x": odds.get("X"), "odds_2": odds.get("2"),
        "market_favourite": facts.get("market_favourite"),
        "home_elo": facts.get("home_elo"), "away_elo": facts.get("away_elo"),
        "extracted_at": now_iso,
    }
    return {k: v for k, v in row.items() if v is not None}


def _compute_elo_warning_from_row(m: dict) -> str:
    pick = m.get("simple_pick_team")
    if pick == "home": pick_elo = m.get("home_elo")
    elif pick == "away": pick_elo = m.get("away_elo")
    else: return ""

    if pick_elo is None: return "⚠️ ELO UNKNOWN"
    try:
        if float(pick_elo) < ELO_MINIMUM: return f"⚠️ ELO {int(pick_elo)} < {ELO_MINIMUM}"
    except (TypeError, ValueError): return "⚠️ ELO UNKNOWN"
    return ""


def _card_html(m: dict, settled: bool = False) -> str:
    home, away = m.get("home_team") or "?", m.get("away_team") or "?"
    hs, aws = m.get("home_score"), m.get("away_score")
    dc_hit, profit = m.get("dc_hit"), m.get("profit_loss") or 0.0

    cat, bet, pick = m.get("simple_category"), m.get("simple_bet"), m.get("simple_pick_team")
    hw, aw = m.get("home_consecutive_wins"), m.get("away_consecutive_wins")
    hu, au = m.get("home_consecutive_unbeaten"), m.get("away_consecutive_unbeaten")
    helo, aelo = m.get("home_elo"), m.get("away_elo")

    cat_cls = {"SOLID CLEAR": "badge-solid", "LIGHT CLEAR HOME": "badge-light", "NOT-BEATEN SIGNAL": "badge-notbe"}.get(cat, "badge-nobet")
    pick_lbl = "HOME" if pick == "home" else ("AWAY" if pick == "away" else "")
    card_cls = "pick-home" if pick == "home" else ("pick-away" if pick == "away" else "skip")

    meta_bits = [b for b in [m.get("league"), m.get("match_date"), m.get("venue")] if b]
    meta = " · ".join(meta_bits) if meta_bits else "—"
    title = f"{home} {hs}–{aws} {away}" if (settled and hs is not None) else f"{home} vs {away}"

    if settled and dc_hit is not None:
        vcls = "badge-win" if dc_hit == 1 else "badge-loss"
        verdict_html = f'<span class="pick-badge {vcls}" style="margin-left:8px">{"WIN" if dc_hit == 1 else "LOSS"} · {profit:+.2f}u</span>'
    else:
        verdict_html = ""

    badge_text = f"{cat} · {bet} {pick_lbl}" if cat else "NO BET"

    elo_warning_html = ""
    if cat:
        warn_text = _compute_elo_warning_from_row(m)
        if warn_text: elo_warning_html = f'<div class="elo-warning">{warn_text}</div>'

    o1, ox, o2, mf = m.get("odds_1"), m.get("odds_x"), m.get("odds_2"), m.get("market_favourite")
    odds_row = ""
    if any(v is not None for v in (o1, ox, o2)):
        mf_lbl = {"home": "HOME", "away": "AWAY", "tie": "LEVEL"}.get(mf, "?")
        odds_row = (
            '<div class="signal-row">'
            f'<span class="signal-pill odds">1: {o1 if o1 is not None else "—"}</span>'
            f'<span class="signal-pill odds">X: {ox if ox is not None else "—"}</span>'
            f'<span class="signal-pill odds">2: {o2 if o2 is not None else "—"}</span>'
            f'<span class="signal-pill odds">Mkt: <b>{mf_lbl}</b></span>'
            '</div>'
        )
        
    elo_row = ""
    if helo or aelo:
        elo_row = (
            '<div class="signal-row">'
            f'<span class="signal-pill elo">Home ELO: {helo or "?"}</span>'
            f'<span class="signal-pill elo">Away ELO: {aelo or "?"}</span>'
            '</div>'
        )

    html = f"""
    <div class="match-card {card_cls}">
        <div class="match-title">{title}</div>
        <div class="match-meta">{meta}</div>
        <span class="pick-badge {cat_cls}">{badge_text}</span>{verdict_html}
        {elo_warning_html}
        <div class="signal-row">
            <span class="signal-pill home">Home W/U: {hw} / {hu}</span>
            <span class="signal-pill away">Away W/U: {aw} / {au}</span>
        </div>
        {elo_row}
        {odds_row}
    </div>
    """
    return "".join(line.strip() for line in html.split("\n"))


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab_new, tab_pending, tab_perf, tab_study, tab_lab = st.tabs(
    ["➕ New Match", "⏳ Pending", "📊 Performance", "📚 Study", "🧪 Lab"]
)

with tab_new:
    st.markdown("### Paste the match data")
    st.caption("Include `Pre-match HOME vs AWAY` header and the `Pre-match odds` block.")
    raw_in = st.text_area("Input", height=340, label_visibility="collapsed", placeholder="Pre-match Real Zaragoza vs Huesca\n...")
    run = st.button("Run the Logic", type="primary")

    if run and raw_in.strip():
        if not _looks_like_text_dump(raw_in):
            st.error("Input must contain 'Pre-match X vs Y' or 'Recent form' + 'Consecutive' sections.")
            st.stop()
        try:
            facts = parse_recent_form_dump(raw_in.strip())
            simple = compute_simple_row(facts)
        except Exception as e:
            st.error(f"Parse failed: {e}")
            st.stop()

        st.markdown("### The Pick")
        preview = {**facts, **simple}
        st.markdown(_card_html(preview, settled=False), unsafe_allow_html=True)

        if simple["simple_qualifying"]: st.success(simple["simple_reason"])
        elif simple.get("home_consecutive_wins") is not None: st.info(f"No bet [{simple['simple_rejection_code']}] — {simple['simple_reason']}")
        else: st.warning("Missing streak values — cannot classify.")

        warnings = facts.get("_parse_warnings", [])
        if warnings:
            with st.expander(f"⚠ Parser warnings ({len(warnings)})"):
                for w in warnings: st.write(f"- {w}")

        has_required = facts.get("home_team") and facts.get("away_team") and simple.get("home_consecutive_wins") is not None
        if not has_required: st.warning("Missing required fields — not saved.")
        else:
            row = build_row(facts)
            existing = db_find_existing(row)
            if existing: st.info(f"Already stored (match_id **{existing['match_id']}**, qualifying={existing.get('simple_qualifying')}).")
            else:
                try:
                    db_insert(row)
                    if simple["simple_qualifying"]: st.success("Saved to Pending.")
                    else: st.success(f"Saved to Study ({simple['simple_rejection_code']}).")
                except Exception as e: st.error(f"Insert failed: {e}")

with tab_pending:
    st.markdown("### Pending matches")
    pending = db_pending()
    if not pending: st.info("No pending matches.")
    else:
        for m in pending:
            st.markdown(_card_html(m, settled=False), unsafe_allow_html=True)
            with st.expander("Enter result"):
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"h_{m['match_id']}")
                with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"a_{m['match_id']}")
                with c3:
                    st.write(""); st.write("")
                    if st.button("Settle", key=f"s_{m['match_id']}", type="primary"):
                        result, dc_hit, profit = settle_simple(m.get("simple_bet") or "", int(hs), int(aws))
                        db_settle(m["match_id"], int(hs), int(aws), result, dc_hit, profit)
                        st.success("Settled."); st.rerun()
                if st.button("Delete match", key=f"d_{m['match_id']}"):
                    db_delete(m["match_id"]); st.success("Deleted."); st.rerun()

with tab_perf:
    st.markdown("### Performance")
    settled = db_settled()
    if not settled: st.info("No settled matches yet.")
    else:
        bets = [m for m in settled if m.get("dc_hit") is not None]
        n = len(bets)
        wins = sum(1 for m in bets if m.get("dc_hit") == 1)
        net = sum((m.get("profit_loss") or 0.0) for m in bets)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Bets", n); c2.metric("Wins", wins); c3.metric("Losses", n - wins)
        c4.metric("Hit %", f"{(wins / n * 100) if n else 0:.1f}%"); c5.metric("Net P/L", f"{net:+.2f}u")

        if bets:
            df = pd.DataFrame([{"category": m.get("simple_category"), "bet": m.get("simple_bet"), "hit": int(m["dc_hit"]), "pl": float(m.get("profit_loss") or 0.0)} for m in bets])
            out = df.groupby(["category", "bet"]).agg(n=("hit", "count"), wins=("hit", "sum"), net_pl=("pl", "sum")).reset_index()
            out["hit_pct"] = (out["wins"] / out["n"] * 100).round(1)
            st.dataframe(out[["category", "bet", "n", "wins", "hit_pct", "net_pl"]], use_container_width=True, hide_index=True)

        st.markdown("### All settled")
        for m in settled: st.markdown(_card_html(m, settled=True), unsafe_allow_html=True)

with tab_study:
    st.markdown("### 📚 Study — rejected matches")
    study = db_study()
    if not study: st.info("No study rows yet.")
    else:
        settled = [m for m in study if m.get("home_score") is not None]
        unsettled = [m for m in study if m.get("home_score") is None]

        c1, c2, c3 = st.columns(3)
        c1.metric("Study rows", len(study)); c2.metric("Settled", len(settled)); c3.metric("Pending result", len(unsettled))

        if settled:
            st.markdown("### Actual outcome counts by rejection code")
            rows = [{"rejection_code": m.get("simple_rejection_code") or "(none)", "result": m.get("result") or "(unknown)"} for m in settled]
            df = pd.DataFrame(rows)
            pivot = df.groupby(["rejection_code", "result"]).size().unstack(fill_value=0)
            for col in ("home_win", "draw", "away_win"):
                if col not in pivot.columns: pivot[col] = 0
            pivot = pivot[["home_win", "draw", "away_win"]]
            pivot["total"] = pivot.sum(axis=1)
            st.dataframe(pivot.reset_index().sort_values("total", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("### Rows")
        all_codes = sorted({m.get("simple_rejection_code") or "(none)" for m in study})
        selected = st.multiselect("Filter by rejection code", all_codes, default=all_codes, key="study_filter")

        filtered_study = [m for m in study if (m.get("simple_rejection_code") or "(none)") in selected]
        pending_rows = [m for m in filtered_study if m.get("home_score") is None]
        settled_rows = [m for m in filtered_study if m.get("home_score") is not None]

        st.markdown(f"#### ⏳ Awaiting Result ({len(pending_rows)})")
        if not pending_rows: st.caption("None pending with the current filter.")
        else:
            for m in pending_rows:
                st.markdown(_card_html(m, settled=False), unsafe_allow_html=True)
                with st.expander("Enter final score"):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1: hs = st.number_input("Home", min_value=0, step=1, key=f"sh_{m['match_id']}")
                    with c2: aws = st.number_input("Away", min_value=0, step=1, key=f"sa_{m['match_id']}")
                    with c3:
                        st.write(""); st.write("")
                        if st.button("Save result", key=f"ss_{m['match_id']}", type="primary"):
                            if hs > aws: result = "home_win"
                            elif aws > hs: result = "away_win"
                            else: result = "draw"
                            db_settle(m["match_id"], int(hs), int(aws), result, None, None)
                            st.success("Result saved."); st.rerun()

        st.markdown(f"#### ✅ Settled ({len(settled_rows)})")
        if not settled_rows: st.caption("None settled with the current filter.")
        else:
            for m in settled_rows:
                st.markdown(_card_html(m, settled=True), unsafe_allow_html=True)
                st.caption(f"Final: **{m['home_score']}–{m['away_score']}**")
                with st.expander("Correct result"):
                    c1, c2, c3 = st.columns([1, 1, 2])
                    with c1: hs = st.number_input("Home", min_value=0, step=1, value=int(m["home_score"]), key=f"ch_{m['match_id']}")
                    with c2: aws = st.number_input("Away", min_value=0, step=1, value=int(m["away_score"]), key=f"ca_{m['match_id']}")
                    with c3:
                        st.write(""); st.write("")
                        if st.button("Update", key=f"cu_{m['match_id']}"):
                            if hs > aws: result = "home_win"
                            elif aws > hs: result = "away_win"
                            else: result = "draw"
                            db_settle(m["match_id"], int(hs), int(aws), result, None, None)
                            st.success("Updated."); st.rerun()

with tab_lab:
    st.markdown("### 🧪 Lab")
    text_in = st.text_area("Text dump", height=340, label_visibility="collapsed", key="lab_input")
    if text_in.strip():
        try:
            facts = parse_recent_form_dump(text_in)
            simple = compute_simple_row(facts)
            cA, cB, cC, cD = st.columns(4)
            cA.metric("Home W / U", f"{simple.get('home_consecutive_wins')} / {simple.get('home_consecutive_unbeaten')}")
            cB.metric("Away W / U", f"{simple.get('away_consecutive_wins')} / {simple.get('away_consecutive_unbeaten')}")
            cC.metric("Category", simple.get("simple_category") or "—")
            cD.metric("Bet", simple.get("simple_bet") or "—")
            
            if simple["simple_qualifying"]: st.success(simple.get("simple_reason") or "")
            else: st.info(f"No bet [{simple.get('simple_rejection_code')}] — {simple.get('simple_reason')}")

            with st.expander("Parsed fields"):
                st.json({k: v for k, v in facts.items() if v not in (None, [], {}, "") and k != "recent_form_triplets"})
        except Exception as e:
            st.error(f"Parse failed: {e}")
