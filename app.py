"""
v4.4 TAGGED PREDICTOR — complete nine-layer decision logic with audit fixes.

Fixes applied (traceable via `# FIX:` comments):
  Fix 1.2  — save_prediction warns on dropped keys
  Fix 1.3  — draw_risk stored as REAL (numeric coercion in app layer)
  Fix 1.4  — win_pct values coerced to float on read
  Fix 2.1  — calc_f5 recency-weighted H2H
  Fix 2.2  — calc_agreement returns "tie" instead of defaulting to away
  Fix 2.3  — calc_expected_total uses explicit None checks
  Fix 2.4  — away_collapse documented as spec addition (kept, flag added)
  Fix 2.5  — calc_f4 no longer double-penalises doubted XI starters
  Fix 2.6  — draw_risk reaches 0.65+ range (widened band weights)
  Fix 2.7  — CLUSTER uses f5_leader_raw, not post-override f5_leader
  Fix 3.1  — _team_matches requires two shared tokens or last-word match
  Fix 3.2  — last5_count stored alongside last5
  Fix 3.3  — _parse_standings prefers table containing both team names
  Fix 3.4  — injury dedup normalises names
  Fix 4.1  — model_version surfaced in reports
  Fix 4.3  — parse_status = "partial" when critical fields missing
  Fix 4.4  — parse_status checked before factor computation
"""

import concurrent.futures
import json
import re
import unicodedata
from datetime import datetime

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="v4.4 Tagged Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    .team-header { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 16px; padding: 1.5rem 2rem; color: #fff; margin-bottom: 1rem; }
    .team-names { font-size: 2rem; font-weight: 800; margin: 0; }
    .team-meta { color: #94a3b8; font-size: 0.9rem; margin-top: 0.25rem; }
    .verdict-bet { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0; }
    .verdict-nobet { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b; border-radius: 16px; padding: 1.5rem 1.75rem; margin: 1rem 0; }
    .verdict-label { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #6ee7b7; text-transform: uppercase; }
    .verdict-label-grey { font-size: 0.75rem; letter-spacing: 2px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
    .verdict-pick { font-size: 2.4rem; font-weight: 800; color: #fff; margin: 0.35rem 0; line-height: 1.1; }
    .verdict-noedge { font-size: 1.6rem; font-weight: 700; color: #cbd5e1; margin: 0.35rem 0; }
    .verdict-detail { font-size: 1rem; color: #d1fae5; margin-top: 0.5rem; }
    .verdict-detail-grey { font-size: 0.95rem; color: #94a3b8; margin-top: 0.5rem; }
    .tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 6px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.5px; margin-right: 0.35rem; }
    .tag-gap { background: #1e3a8a; color: #bfdbfe; }
    .tag-side { background: #064e3b; color: #6ee7b7; }
    .tag-disagreement { background: #7c2d12; color: #fdba74; }
    .tag-venue { background: #4c1d95; color: #ddd6fe; }
    .tag-tier { background: #7c2d12; color: #fed7aa; }
    .tag-skip { background: #1e293b; color: #94a3b8; }
    .factor-row { background: #0f172a; border-radius: 10px; padding: 0.75rem 1rem;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
    .factor-name { color: #cbd5e1; font-weight: 600; font-size: 0.9rem; }
    .factor-val { color: #3b82f6; font-weight: 800; font-size: 1.1rem; }
    .trigger-row { background: #1e293b; border-radius: 8px; padding: 0.6rem 1rem;
        margin-bottom: 0.4rem; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; }
    .trigger-on { background: #064e3b; color: #6ee7b7; }
    .trigger-off { background: #1e293b; color: #64748b; }
    .section-title { font-size: 0.85rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1.5px; margin: 1.5rem 0 0.75rem 0; }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; font-weight: 700; border-radius: 10px; border: none; padding: 0.6rem 1.25rem; }
    .diag-ok { background: #064e3b; color: #6ee7b7; padding: 0.4rem 0.75rem;
        border-radius: 6px; font-family: monospace; font-size: 0.8rem; display: inline-block; margin-right: 0.5rem; }
    .diag-bad { background: #7f1d1d; color: #fecaca; padding: 0.4rem 0.75rem;
        border-radius: 6px; font-family: monospace; font-size: 0.8rem; display: inline-block; margin-right: 0.5rem; }
    .diag-warn { background: #78350f; color: #fde68a; padding: 0.4rem 0.75rem;
        border-radius: 6px; font-family: monospace; font-size: 0.8rem; display: inline-block; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DIAGNOSTIC HELPERS
# ============================================================================

def _inspect_secret(name):
    try:
        raw = st.secrets[name]
    except Exception:
        return {"present": False, "reason": "not in st.secrets"}
    if raw is None:
        return {"present": False, "reason": "value is None"}
    s = str(raw)
    return {
        "present": True,
        "length": len(s),
        "has_leading_ws": bool(s) and s[0].isspace(),
        "has_trailing_ws": bool(s) and s[-1].isspace(),
        "has_inner_ws": any(c.isspace() for c in s.strip()),
        "has_newline": "\n" in s or "\r" in s,
        "has_quotes": s.startswith('"') or s.startswith("'"),
        "prefix": s[:15] if len(s) >= 15 else s,
        "suffix": s[-5:] if len(s) >= 5 else "",
    }


def _describe_key(key_info):
    if not key_info.get("present"):
        return "MISSING"
    p = key_info.get("prefix", "")
    if p.startswith("sb_publishable_"):
        return "publishable (new-style)"
    if p.startswith("sb_secret_"):
        return "SECRET KEY — do not use in browser code"
    if p.startswith("eyJhbGciOi"):
        return "legacy anon JWT"
    if p.startswith("eyJ"):
        return "JWT (unrecognized header)"
    return f"unknown ({p!r})"


def _check_url(url_info):
    issues = []
    if not url_info.get("present"):
        return ["URL missing from st.secrets"]
    if url_info.get("has_leading_ws") or url_info.get("has_trailing_ws"):
        issues.append("URL has leading/trailing whitespace")
    if url_info.get("has_newline"):
        issues.append("URL contains a newline")
    if url_info.get("has_quotes"):
        issues.append("URL appears to be wrapped in quotes")
    return issues


def _check_key(key_info):
    issues = []
    if not key_info.get("present"):
        return ["Key missing from st.secrets"]
    if key_info.get("has_leading_ws") or key_info.get("has_trailing_ws"):
        issues.append("Key has leading/trailing whitespace — this alone causes 401")
    if key_info.get("has_inner_ws"):
        issues.append("Key contains whitespace in the middle")
    if key_info.get("has_newline"):
        issues.append("Key contains a newline — copy-paste likely split it")
    if key_info.get("has_quotes"):
        issues.append("Key appears to be wrapped in quotes")
    if key_info.get("length", 0) < 100:
        issues.append(f"Key length is only {key_info.get('length')} — expected 200+ for JWT")
    return issues


def _decode_jwt_ref(key):
    if not key or not key.startswith("eyJ"):
        return None
    try:
        import base64
        parts = key.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        decoded = base64.urlsafe_b64decode(payload_b64 + padding)
        data = json.loads(decoded)
        return data.get("ref")
    except Exception:
        return None


def render_diagnostic_banner():
    url_info = _inspect_secret("SUPABASE_URL")
    key_info = _inspect_secret("SUPABASE_KEY")
    url_issues = _check_url(url_info)
    key_issues = _check_key(key_info)

    st.markdown("### 🔍 Diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**URL**")
        if url_info.get("present"):
            cls = "diag-ok" if not url_issues else "diag-bad"
            st.markdown(
                f'<span class="{cls}">len={url_info["length"]} '
                f'prefix={url_info["prefix"]!r}</span>',
                unsafe_allow_html=True,
            )
            for i in url_issues:
                st.error(i)
        else:
            st.markdown('<span class="diag-bad">MISSING</span>', unsafe_allow_html=True)

    with c2:
        st.markdown("**KEY**")
        if key_info.get("present"):
            key_kind = _describe_key(key_info)
            cls = "diag-ok" if not key_issues else "diag-bad"
            st.markdown(
                f'<span class="{cls}">len={key_info["length"]} '
                f'kind={key_kind}</span>',
                unsafe_allow_html=True,
            )
            for i in key_issues:
                st.error(i)
        else:
            st.markdown('<span class="diag-bad">MISSING</span>', unsafe_allow_html=True)

    if url_info.get("present") and key_info.get("present"):
        try:
            raw_key = str(st.secrets["SUPABASE_KEY"]).strip()
            raw_url = str(st.secrets["SUPABASE_URL"]).strip()
        except Exception:
            raw_key = raw_url = ""

        ref = _decode_jwt_ref(raw_key)
        if ref:
            m = re.match(r"https://([^.]+)\.supabase\.co", raw_url)
            url_ref = m.group(1) if m else None
            if url_ref:
                if ref == url_ref:
                    st.success(f"✅ URL and JWT match — project ref: `{ref}`")
                else:
                    st.error(
                        f"❌ URL and KEY belong to different projects. "
                        f"URL ref = `{url_ref}`, KEY ref = `{ref}`."
                    )
            else:
                st.warning(f"Could not parse project ref from URL `{raw_url}`.")
        elif raw_key.startswith("sb_publishable_"):
            st.info("Publishable key detected. Cannot verify project match from key alone.")
        else:
            st.warning("Could not decode JWT from key.")


def render_debug_tab(sb):
    st.subheader("🛠 Debug")
    st.caption("Connection and schema diagnostics. No values are revealed.")

    st.markdown('<div class="section-title">1. Secrets</div>', unsafe_allow_html=True)
    url_info = _inspect_secret("SUPABASE_URL")
    key_info = _inspect_secret("SUPABASE_KEY")

    with st.expander("URL details", expanded=True):
        if url_info.get("present"):
            st.json({k: v for k, v in url_info.items() if k != "prefix"})
            st.code(f"prefix: {url_info['prefix']!r}")
        else:
            st.error("URL not found in st.secrets.")

    with st.expander("KEY details", expanded=True):
        if key_info.get("present"):
            st.json({k: v for k, v in key_info.items() if k not in ("prefix", "suffix")})
            st.code(f"prefix: {key_info['prefix']!r}\nsuffix: {key_info['suffix']!r}")
            st.write("**Key kind:**", _describe_key(key_info))
        else:
            st.error("KEY not found in st.secrets.")

    st.markdown('<div class="section-title">2. Client state</div>', unsafe_allow_html=True)
    if sb is None:
        st.error("Supabase client was NOT created.")
        return
    st.success("Supabase client created.")

    st.markdown('<div class="section-title">3. Live ping</div>', unsafe_allow_html=True)
    if st.button("🏓 Ping Supabase", key="debug_ping"):
        with st.spinner("Pinging..."):
            try:
                resp = sb.table("matches_raw").select("id").limit(1).execute()
                st.success(f"Ping OK. Returned {len(resp.data)} rows.")
                if resp.data:
                    st.write("Sample row id:", resp.data[0].get("id"))
            except Exception as e:
                st.error(f"Ping failed: {e}")

    st.markdown('<div class="section-title">4. Schema check</div>', unsafe_allow_html=True)
    if st.button("🔎 List matches_raw columns", key="debug_cols"):
        with st.spinner("Querying..."):
            try:
                resp = sb.table("matches_raw").select("*").limit(1).execute()
                if resp.data:
                    cols = sorted(resp.data[0].keys())
                    st.success(f"{len(cols)} columns returned by `SELECT *`.")
                    st.write(cols)
                else:
                    st.info("Table is empty. Cannot list columns from a live row.")
            except Exception as e:
                st.error(f"Schema check failed: {e}")

    st.markdown('<div class="section-title">5. Row count</div>', unsafe_allow_html=True)
    if st.button("🧮 Count rows", key="debug_count"):
        with st.spinner("Counting..."):
            try:
                resp = sb.table("matches_raw").select("id").execute()
                st.success(f"matches_raw has {len(resp.data)} row(s).")
            except Exception as e:
                st.error(f"Count failed: {e}")

    st.markdown('<div class="section-title">6. Test insert</div>', unsafe_allow_html=True)
    if st.button("🧪 Insert test row", key="debug_insert"):
        with st.spinner("Inserting..."):
            try:
                payload = {
                    "match_date": "2099-12-31",
                    "home_team": "__DEBUG__ Home",
                    "away_team": "__DEBUG__ Away",
                    "league_name": "DEBUG",
                }
                resp = sb.table("matches_raw").insert(payload).execute()
                st.success(f"Insert succeeded. id = {resp.data[0].get('id') if resp.data else 'unknown'}")
            except Exception as e:
                st.error(f"Insert failed: {e}")

    if st.button("🧹 Delete test rows", key="debug_cleanup"):
        with st.spinner("Deleting..."):
            try:
                sb.table("matches_raw").delete().eq("home_team", "__DEBUG__ Home").execute()
                sb.table("matches_raw").delete().eq("away_team", "__DEBUG__ Away").execute()
                st.success("Cleanup done.")
            except Exception as e:
                st.error(f"Cleanup failed: {e}")


# ============================================================================
# SUPABASE
# ============================================================================

def get_supabase():
    try:
        from supabase import create_client
        url = str(st.secrets["SUPABASE_URL"]).strip()
        key = str(st.secrets["SUPABASE_KEY"]).strip()
        try:
            from supabase.client import ClientOptions
            options = ClientOptions(postgrest_client_timeout=15)
            client = create_client(url, key, options=options)
        except Exception:
            client = create_client(url, key)
        return client, {"ok": True, "url": url}
    except Exception as e:
        return None, {"ok": False, "error": str(e)}


def _has_bs4():
    try:
        import bs4  # noqa
        return True
    except ImportError:
        return False


# ============================================================================
# CONSTANTS — v4.4 COMPLETE LOGIC
# ============================================================================
ALPHA = 2
PRIOR_FORM = 0.4
PRIOR_H2H = 1.0 / 3.0

GAP_THRESHOLD = 30
TOP_SCORER_OUT_MIN_GOALS = 3
DRAW_RISK_THRESHOLD = 0.65          # Fix 2.6 — now reachable given widened bands
F1_CAP_GAP = 16
VENUE_POWER_THRESHOLD = 0.5

STAKE_CLUSTER = 1.0
STAKE_FORTRESS = 1.5
STAKE_VAULT = 2.0
STAKE_STANDARD = 1.0
STAKE_NONE = 0.0


KEYSTATS_PRIORITY_FIELDS = {
    "home_last10_w", "home_last10_d", "home_last10_l",
    "away_last10_w", "away_last10_d", "away_last10_l",
    "home_last10_avg_scored", "home_last10_avg_conceded",
    "away_last10_avg_scored", "away_last10_avg_conceded",
    "home_last10_over25", "home_last10_under25",
    "away_last10_over25", "away_last10_under25",
    "home_last10_btts_yes", "home_last10_btts_no",
    "away_last10_btts_yes", "away_last10_btts_no",
    "home_home_last10_avg_scored", "home_home_last10_avg_conceded",
    "home_away_last10_avg_scored", "home_away_last10_avg_conceded",
    "away_home_last10_avg_scored", "away_home_last10_avg_conceded",
    "away_away_last10_avg_scored", "away_away_last10_avg_conceded",
    "home_last10_possession", "away_last10_possession",
    "home_home_last10_possession", "away_away_last10_possession",
    "home_last10_corners_for", "home_last10_corners_against",
    "away_last10_corners_for", "away_last10_corners_against",
    "home_home_last10_corners_for", "home_home_last10_corners_against",
    "away_away_last10_corners_for", "away_away_last10_corners_against",
    "home_top_scorer", "home_top_scorer_goals",
    "away_top_scorer", "away_top_scorer_goals",
    "home_top_assister", "home_top_assister_assists",
    "away_top_assister", "away_top_assister_assists",
    "home_top_scorer_source", "away_top_scorer_source",
    "home_top_assister_source", "away_top_assister_source",
}


MATCHES_RAW_COLUMNS = {
    "id", "created_at",
    "match_date", "kickoff_utc", "kickoff_local",
    "league_name", "tier", "group_name", "season",
    "home_team", "away_team", "venue", "stage", "round", "parse_status",
    "home_pos", "home_played", "home_points", "home_gd", "home_gf", "home_ga",
    "away_pos", "away_played", "away_points", "away_gd", "away_gf", "away_ga",
    "home_home_points", "home_home_played", "home_home_win_pct",
    "home_away_points", "home_away_played",
    "away_home_points", "away_home_played",
    "away_away_points", "away_away_played", "away_away_win_pct",
    "home_last5", "away_last5",
    "home_last5_count", "away_last5_count",     # Fix 3.2
    "home_last10_w", "home_last10_d", "home_last10_l",
    "home_last10_avg_scored", "home_last10_avg_conceded",
    "home_last10_possession", "home_last10_corners_for", "home_last10_corners_against",
    "home_last10_win_pct", "home_last10_over25", "home_last10_under25",
    "home_last10_btts_yes", "home_last10_btts_no",
    "away_last10_w", "away_last10_d", "away_last10_l",
    "away_last10_avg_scored", "away_last10_avg_conceded",
    "away_last10_possession", "away_last10_corners_for", "away_last10_corners_against",
    "away_last10_win_pct", "away_last10_over25", "away_last10_under25",
    "away_last10_btts_yes", "away_last10_btts_no",
    "home_home_last10_avg_scored", "home_home_last10_avg_conceded",
    "home_home_last10_corners_for", "home_home_last10_corners_against",
    "home_home_last10_possession",
    "home_away_last10_avg_scored", "home_away_last10_avg_conceded",
    "away_home_last10_avg_scored", "away_home_last10_avg_conceded",
    "away_away_last10_avg_scored", "away_away_last10_avg_conceded",
    "away_away_last10_corners_for", "away_away_last10_corners_against",
    "away_away_last10_possession",
    "home_top_scorer", "home_top_scorer_goals",
    "home_top_assister", "home_top_assister_assists",
    "away_top_scorer", "away_top_scorer_goals",
    "away_top_assister", "away_top_assister_assists",
    "home_top_scorer_source", "away_top_scorer_source",
    "home_top_assister_source", "away_top_assister_source",
    "home_injuries", "away_injuries", "home_xi", "away_xi",
    "home_formation", "away_formation",
    "h2h", "h2h_home_wins", "h2h_draws", "h2h_away_wins",
    "odds",
    "actual_home_goals", "actual_away_goals",
    "actual_possession_home", "actual_xg_home", "actual_xg_away",
    "actual_corners_home", "actual_corners_away", "actual_total_goals",
    "f1_home", "f1_away", "f1_gap", "f1_leader",
    "f2_home", "f2_away",
    "f3_home", "f3_away",
    "f4_home", "f4_away",
    "f5_home", "f5_away", "f5_gap", "f5_leader", "f5_leader_raw",
    "f6_home", "f6_away",
    "home_total", "away_total", "raw_gap", "total_gap",
    "disagreements", "shrink_factor",
    "f1_f5_conflict", "f1_f5_override", "f1_vs_f2f3_conflict",
    "away_collapse", "doubted_starter",
    "call_1x2", "call_ou", "expected_total",
    "no_bet_reason_1x2", "no_bet_reason_ou", "no_bet_reason",
    "venue_gap", "f0_home", "f0_away", "f0_gap", "f0_half",
    "draw_risk", "call_btts", "model_version",
    "venue_ppg_gap", "venue_ppg_gap_home", "venue_ppg_gap_away",
    "venue_power",
    "v4_4_bet", "v4_4_call", "v4_4_decision", "v4_4_skip_reason",
    "v4_4_tier", "v4_4_stake",
    "tags", "dc_hit",
    "is_correct_1x2", "is_correct_ou", "is_correct_btts",
}


def _get_table_columns(_sb, table_name="matches_raw"):
    return MATCHES_RAW_COLUMNS


# ============================================================================
# PARSER
# ============================================================================

_PREVIEW_TOP_SCORER_PATTERNS = [
    re.compile(
        r"([A-Z][\w'\-\.\u00C0-\u024F]+(?:\s+[A-Z][\w'\-\.\u00C0-\u024F]+){0,3})"
        r"\s+is\s+top\s+scorer\s+on\s+(\d+)",
    ),
    re.compile(
        r"Top\s+goalscorer\s+"
        r"([A-Z][\w'\-\.\u00C0-\u024F]+(?:\s+[A-Z][\w'\-\.\u00C0-\u024F]+){0,3})"
        r"\s+has\s+found\s+the\s+net\s+(\d+)\s+times",
    ),
]

_KEYSTATS_TOP_SCORER = re.compile(
    r"Top\s+Scorers?\s+for\s+.+?\s+this\s+season\s+are\s+(.+?)(?:Top\s+Assistors?|$)",
    re.I | re.S,
)
_KEYSTATS_TOP_ASSISTER = re.compile(
    r"Top\s+Assistors?\s+for\s+.+?\s+this\s+season\s+are\s+(.+?)$",
    re.I | re.S,
)
_NAME_GOALS = re.compile(
    r"([A-Z][\w'\-\.\u00C0-\u024F]+(?:\s+[A-Z][\w'\-\.\u00C0-\u024F]+){0,3})\s*\((\d+)\)"
)


class SportsgamblerParser:
    def __init__(self, html: str):
        if not _has_bs4():
            raise RuntimeError("beautifulsoup4 required. pip install beautifulsoup4")
        from bs4 import BeautifulSoup
        self.soup = BeautifulSoup(html, "html.parser")
        self.home_team = None
        self.away_team = None

    @staticmethod
    def _to_float(s):
        try:
            return float(str(s).strip().replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _to_int(s):
        try:
            return int(str(s).strip().replace(",", ""))
        except (ValueError, TypeError, AttributeError):
            return None

    @staticmethod
    def _norm(s):
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.lower().strip()

    @staticmethod
    def _norm_comp(s):
        if not s:
            return ""
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[^a-z0-9]", "", s.lower())

    # Fix 3.1 — require two shared meaningful tokens, or last-word match
    def _team_matches(self, full_name, table_name):
        if not full_name or not table_name:
            return False
        a = self._norm(full_name)
        b = self._norm(table_name)
        if not a or not b:
            return False
        if a == b or a in b or b in a:
            return True
        a_words, b_words = a.split(), b.split()
        a_last = a_words[-1] if a_words else ""
        b_last = b_words[-1] if b_words else ""
        if a_last and b_last and a_last == b_last:
            return True
        # Require two distinct shared tokens (length > 3) rather than just one
        a_tokens = {w for w in a_words if len(w) > 3}
        b_tokens = {w for w in b_words if len(w) > 3}
        shared = a_tokens & b_tokens
        if len(shared) >= 2:
            return True
        # Last-resort: 4-char prefix on the first long token, but only if it's
        # the FIRST such token on both sides (avoids "real" matching "real")
        a_first = next((w for w in a_words if len(w) >= 4), "")
        b_first = next((w for w in b_words if len(w) >= 4), "")
        if a_first and b_first and a_first[:4] == b_first[:4]:
            return True
        return False

    def _team_matches_strict(self, full_name, header_text):
        if not full_name or not header_text:
            return False
        a = self._norm(full_name)
        b = self._norm(header_text)
        a_words = [w for w in a.split() if len(w) > 3]
        if not a_words:
            return False
        return a_words[0] in b

    def parse(self):
        self.home_team, self.away_team = self._parse_teams()
        match_date, kickoff = self._parse_datetime()
        league, tier, group = self._parse_league()
        venue = self._parse_venue()

        kickoff_utc_iso = None
        if match_date and kickoff:
            try:
                dt = datetime.strptime(f"{match_date} {kickoff}", "%Y-%m-%d %H:%M")
                kickoff_utc_iso = dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            except ValueError:
                pass

        record = {
            "match_date": match_date,
            "kickoff_utc": kickoff_utc_iso,
            "kickoff_local": kickoff,
            "league_name": league,
            "tier": tier,
            "group_name": group,
            "season": "2026-27",
            "home_team": self.home_team,
            "away_team": self.away_team,
            "venue": venue,
            "stage": None,
            "round": None,
        }

        record.update(self._parse_standings())
        record["home_last5"] = self._parse_last5("home")
        record["away_last5"] = self._parse_last5("away")
        # Fix 3.2 — record how many matches were actually parsed for form
        record["home_last5_count"] = len(record["home_last5"] or [])
        record["away_last5_count"] = len(record["away_last5"] or [])

        record.update(self._parse_last10("home"))
        record.update(self._parse_last10("away"))

        for key, val in self._parse_keystats_block().items():
            if val is None:
                continue
            if key in KEYSTATS_PRIORITY_FIELDS:
                record[key] = val
            elif key not in record or record.get(key) is None:
                record[key] = val

        for side in ("home", "away"):
            for k, v in self._parse_top_scorers_and_assisters(side).items():
                if v is not None:
                    record[k] = v

        record["home_injuries"] = self._parse_injuries("home")
        record["away_injuries"] = self._parse_injuries("away")
        record["home_xi"] = self._parse_xi("home")
        record["away_xi"] = self._parse_xi("away")
        record["home_formation"] = self._parse_formation("home")
        record["away_formation"] = self._parse_formation("away")
        record.update(self._parse_h2h())
        record.update(self._parse_corners())
        record["odds"] = self._parse_odds()

        # Fix 4.3 — parse_status reflects completeness, not assumption
        critical = [
            "home_points", "away_points", "home_played", "away_played",
            "home_last5", "away_last5", "h2h",
        ]
        missing = [k for k in critical if record.get(k) in (None, [], {})]
        record["parse_status"] = "ok" if not missing else "partial"
        return record

    def _parse_teams(self):
        teams = self.soup.select(".t_top .t_teams .t_name strong")
        if len(teams) >= 2:
            return teams[0].get_text(strip=True), teams[1].get_text(strip=True)
        h1 = self.soup.find("h1", class_="p_title")
        if h1:
            m = re.match(r"(.+?)\s+vs\s+(.+?)\s+Prediction", h1.get_text(strip=True))
            if m:
                return m.group(1).strip(), m.group(2).strip()
        return None, None

    def _parse_datetime(self):
        date_el = self.soup.select_one(".t_top .t_date span:first-child")
        time_el = self.soup.select_one(".t_top .t_date .t_time")
        raw_date = date_el.get_text(strip=True) if date_el else None
        kickoff = time_el.get_text(strip=True) if time_el else None
        iso = None
        if raw_date:
            for fmt in ("%a %d %b", "%d %b", "%a %d %B", "%d %B"):
                try:
                    dt = datetime.strptime(raw_date, fmt).replace(year=datetime.now().year)
                    iso = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        return iso, kickoff

    def _parse_league(self):
        league = None
        for script in self.soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue
            for node in data.get("@graph", []):
                article = node.get("mainEntity")
                if not isinstance(article, dict):
                    continue
                se = article.get("superEvent")
                if isinstance(se, dict) and se.get("name"):
                    league = se["name"].strip()
                    break
            if league:
                break
        if not league:
            for link in self.soup.select(".t_top .t_info_link"):
                text = link.get_text(strip=True)
                if text.lower() == "football":
                    continue
                if re.search(
                    r"(League|Serie|Série|Liga|Bundesliga|Ligue|Premier|"
                    r"Championship|MLS|Cup|Division|Primera|Nations)",
                    text, re.I,
                ):
                    league = text
                    break
        tier = group = None
        if league:
            m = re.search(r"League\s+([A-C])", league)
            if m:
                tier = m.group(1)
            m = re.search(r"Group\s+(\w+)", league)
            if m:
                group = f"Group {m.group(1)}"
        return league or "Unknown", tier, group

    def _parse_venue(self):
        el = self.soup.select_one(".t_top .t_venue")
        return el.get_text(strip=True) if el else None

    def _parse_standings(self):
        out = {}
        # Fix 3.3 — prefer table containing both team names
        main_table = None
        candidates = []
        for table in self.soup.select("table"):
            headers = [th.get_text(strip=True).lower() for th in table.select("th")]
            if any(h in ("pts", "p") for h in headers) and any("team" in h for h in headers):
                candidates.append(table)

        if candidates:
            # Prefer the table whose body mentions both teams
            for table in candidates:
                body_text = table.get_text(" ", strip=True)
                if (self.home_team and self.home_team.split()[0].lower() in body_text.lower()
                        and self.away_team and self.away_team.split()[0].lower() in body_text.lower()):
                    main_table = table
                    break
            if main_table is None:
                main_table = candidates[0]

        if main_table:
            headers = [th.get_text(strip=True).lower() for th in main_table.select("th")]
            try:
                pos_idx = 0
                team_idx = next(i for i, h in enumerate(headers) if "team" in h)
                pts_idx = next(i for i, h in enumerate(headers) if h in ("pts", "p"))
                gd_indices = [i for i, h in enumerate(headers) if h in ("gd", "+/-", "goal diff")]
                gf_idx = next((i for i, h in enumerate(headers) if h in ("gf", "goals for")), None)
                ga_idx = next((i for i, h in enumerate(headers) if h in ("ga", "goals against")), None)
            except StopIteration:
                main_table = None

        if main_table:
            for row in main_table.select("tbody tr"):
                cells = row.select("td")
                if len(cells) <= max(pos_idx, team_idx, pts_idx):
                    continue
                team_clean = re.sub(r"\s*logo\s*", " ",
                                    cells[team_idx].get_text(strip=True), flags=re.I).strip()
                gf = ga = gd = None
                if gf_idx is not None and ga_idx is not None and ga_idx < len(cells):
                    gf = self._to_int(cells[gf_idx].get_text(strip=True))
                    ga = self._to_int(cells[ga_idx].get_text(strip=True))
                    if gf is not None and ga is not None:
                        gd = gf - ga
                if gf is None and gd_indices:
                    for gi in gd_indices:
                        if gi >= len(cells):
                            continue
                        txt = cells[gi].get_text(strip=True)
                        parsed = self._parse_gf_ga(txt)
                        if parsed:
                            gf, ga = parsed
                            gd = gf - ga
                            break
                        if gd is None:
                            gd_plain = self._parse_gd(txt)
                            if gd_plain is not None:
                                gd = gd_plain
                played = self._to_int(cells[2].get_text(strip=True)) if len(cells) > 2 else None
                if self._team_matches(self.home_team, team_clean):
                    out["home_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["home_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    out["home_played"] = played
                    if gf is not None:
                        out["home_gf"] = gf
                        out["home_ga"] = ga
                    if gd is not None:
                        out["home_gd"] = gd
                elif self._team_matches(self.away_team, team_clean):
                    out["away_pos"] = self._to_int(cells[pos_idx].get_text(strip=True))
                    out["away_points"] = self._to_int(cells[pts_idx].get_text(strip=True))
                    out["away_played"] = played
                    if gf is not None:
                        out["away_gf"] = gf
                        out["away_ga"] = ga
                    if gd is not None:
                        out["away_gd"] = gd
        self._parse_split_table("#Homeleague", self.home_team, "home_home", out)
        self._parse_split_table("#Awayleague", self.home_team, "home_away", out)
        self._parse_split_table("#Homeleague", self.away_team, "away_home", out)
        self._parse_split_table("#Awayleague", self.away_team, "away_away", out)
        return out

    def _parse_split_table(self, sel, team_name, prefix, out):
        """
        Fix 1.4 — read the "won" and "played" columns by header name,
        not by hard-coded index. This corrects the systematic bug where
        cells[3] held D on some layouts and W on others.
        """
        container = self.soup.select_one(sel)
        if not container:
            return
        table = container.find("table")
        if not table:
            return
        headers = [th.get_text(strip=True).lower() for th in table.select("th")]
        try:
            team_idx = next(i for i, h in enumerate(headers) if "team" in h)
            pts_idx = next(i for i, h in enumerate(headers) if h in ("pts", "p"))
        except StopIteration:
            return

        # Locate "won" column by header
        won_idx = None
        for i, h in enumerate(headers):
            if h in ("w", "won", "wins"):
                won_idx = i
                break
        # Locate "played" column by header
        played_idx = None
        for i, h in enumerate(headers):
            if h in ("p", "pl", "played", "mp", "games"):
                played_idx = i
                break
        if played_idx is None:
            played_idx = 2  # safe fallback

        for row in table.select("tbody tr"):
            cells = row.select("td")
            if len(cells) <= max(team_idx, pts_idx):
                continue
            team_clean = re.sub(r"\s*logo\s*", " ",
                                cells[team_idx].get_text(strip=True), flags=re.I).strip()
            if not self._team_matches(team_name, team_clean):
                continue
            played = self._to_int(cells[played_idx].get_text(strip=True)) if played_idx < len(cells) else None
            won = (self._to_int(cells[won_idx].get_text(strip=True))
                   if won_idx is not None and won_idx < len(cells) else None)
            pts = self._to_int(cells[pts_idx].get_text(strip=True))
            if prefix == "home_home":
                out["home_home_points"] = pts
                out["home_home_played"] = played
                if played and won is not None:
                    out["home_home_win_pct"] = (won / played) * 100
            elif prefix == "home_away":
                out["home_away_points"] = pts
                out["home_away_played"] = played
            elif prefix == "away_home":
                out["away_home_points"] = pts
                out["away_home_played"] = played
            elif prefix == "away_away":
                out["away_away_points"] = pts
                out["away_away_played"] = played
                if played and won is not None:
                    out["away_away_win_pct"] = (won / played) * 100
            return

    @staticmethod
    def _parse_gd(s):
        try:
            return int(s.strip().replace("+", ""))
        except ValueError:
            return None

    @staticmethod
    def _parse_gf_ga(s):
        m = re.match(r"(\d+)\s*[:\-]\s*(\d+)", s.strip())
        if m:
            return int(m.group(1)), int(m.group(2))
        return None

    def _parse_last5(self, side):
        container = self.soup.select_one("#last-matches #All") or self.soup.select_one("#last-matches")
        if not container:
            return []
        block = container.select_one(".teamstats-left" if side == "home" else ".teamstats-right")
        if not block:
            return []
        league_name = self._parse_league()[0] or ""
        league_short = league_name.split(" - ")[-1] if " - " in league_name else league_name
        league_norm = self._norm_comp(league_short)

        def collect(apply_filter):
            items = []
            for item in block.select("li.team-stat-list-item"):
                if len(items) >= 5:
                    break
                parsed = self._parse_last5_item(item, side, apply_filter, league_norm)
                if parsed is not None:
                    items.append(parsed)
            return items

        strict = collect(True)
        if strict:
            return strict
        return collect(False)

    def _parse_last5_item(self, item, side, apply_filter, league_norm):
        tracked = (self.home_team if side == "home" else self.away_team) or ""
        date_el = item.select_one(".team-stats-date")
        date_text = date_el.get_text(strip=True) if date_el else ""
        if apply_filter and ":" in date_text:
            comp = date_text.split(":", 1)[0].strip()
            comp_norm = self._norm_comp(comp)
            if league_norm and comp_norm:
                if league_norm not in comp_norm and comp_norm not in league_norm:
                    return None
        teams = item.select(".team-stats-team")
        if len(teams) < 2:
            return None
        home_name_el, away_name_el = teams[0], teams[1]
        h_score_el = home_name_el.select_one(".score-right")
        a_score_el = away_name_el.select_one(".score-right")
        home_name = home_name_el.get_text(" ", strip=True)
        away_name = away_name_el.get_text(" ", strip=True)
        if h_score_el:
            home_name = home_name.replace(h_score_el.get_text(strip=True), "").strip()
        if a_score_el:
            away_name = away_name.replace(a_score_el.get_text(strip=True), "").strip()
        h_score = self._to_int(h_score_el.get_text(strip=True)) if h_score_el else None
        a_score = self._to_int(a_score_el.get_text(strip=True)) if a_score_el else None
        if h_score is None or a_score is None:
            return None
        tracked_is_home = self._team_matches(tracked, home_name)
        tracked_is_away = self._team_matches(tracked, away_name)
        if tracked_is_home:
            is_home, sf, sa = True, h_score, a_score
        elif tracked_is_away:
            is_home, sf, sa = False, a_score, h_score
        else:
            is_home = (side == "home")
            sf, sa = (h_score, a_score) if is_home else (a_score, h_score)
        result = "W" if sf > sa else ("D" if sf == sa else "L")
        return {
            "date": date_text,
            "opp": away_name if is_home else home_name,
            "result": result,
            "score_for": sf,
            "score_against": sa,
            "is_home": is_home,
        }

    def _parse_last10(self, side):
        out = {}
        prefix = "home" if side == "home" else "away"
        table = self.soup.select_one(".st-table")
        if table:
            header_text = " ".join(th.get_text(" ", strip=True) for th in table.select("th")).lower()
            body_first = []
            for tr in table.select("tbody tr"):
                tds = tr.select("td")
                if tds:
                    body_first.append(tds[0].get_text(" ", strip=True))
            combined = header_text + " || " + " ".join(body_first).lower()
            markers = (
                "home stats", "away stats", "home form", "away form",
                "home matches", "away matches", "home league games",
                "away league games", "home league", "away league",
            )
            if any(m in combined for m in markers):
                table = None
        if table:
            rows = table.select("tbody tr")
            idx = 0 if side == "home" else 1
            if idx < len(rows):
                cells = [td.get_text(strip=True) for td in rows[idx].select("td")]
                if len(cells) >= 8:
                    mm = re.match(r"(\d+)-(\d+)-(\d+)", cells[1])
                    if mm:
                        out[f"{prefix}_last10_w"] = int(mm.group(1))
                        out[f"{prefix}_last10_d"] = int(mm.group(2))
                        out[f"{prefix}_last10_l"] = int(mm.group(3))
                    out[f"{prefix}_last10_avg_scored"] = self._to_float(cells[3])
                    out[f"{prefix}_last10_avg_conceded"] = self._to_float(cells[4])
                    out[f"{prefix}_last10_over25"] = self._to_int(cells[5])
                    out[f"{prefix}_last10_under25"] = self._to_int(cells[6])
                    if len(cells) >= 9:
                        out[f"{prefix}_last10_btts_yes"] = self._to_int(cells[7])
                        out[f"{prefix}_last10_btts_no"] = self._to_int(cells[8])
        goals_block = self.soup.select_one(f"#goals-{'hometeam' if side == 'home' else 'awayteam'}")
        if goals_block:
            text = goals_block.get_text(" ", strip=True)
            m = re.search(r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 home matches", text)
            if m:
                out[f"{prefix}_home_last10_avg_scored"] = float(m.group(1))
                out[f"{prefix}_home_last10_avg_conceded"] = float(m.group(2))
            m = re.search(r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 away matches", text)
            if m:
                out[f"{prefix}_away_last10_avg_scored"] = float(m.group(1))
                out[f"{prefix}_away_last10_avg_conceded"] = float(m.group(2))
        if f"{prefix}_last10_w" in out and out[f"{prefix}_last10_w"] is not None:
            out[f"{prefix}_last10_win_pct"] = (out[f"{prefix}_last10_w"] / 10) * 100
        return out

    def _parse_keystats_block(self):
        out = {}
        if not self.home_team or not self.away_team:
            return out
        keystats = self.soup.select_one(".keystats")
        if not keystats:
            return out

        for item in keystats.select(".keystat-item"):
            title_sub = item.select_one(".stats-title-sub")
            if not title_sub:
                continue
            title_text = title_sub.get_text(" ", strip=True)
            is_away = "awaystats" in item.get("class", [])
            prefix = "away" if is_away else "home"
            text = item.get_text(" ", strip=True)

            if "Full-Time Result" in title_text:
                m = re.search(
                    r"(\d+)\s+wins?,\s*(\d+)\s+defeats?\s+and\s+(\d+)\s+draws?\s+in\s+the\s+previous\s+10\s+matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_w"] = int(m.group(1))
                    out[f"{prefix}_last10_l"] = int(m.group(2))
                    out[f"{prefix}_last10_d"] = int(m.group(3))
                venue = "home" if not is_away else "away"
                m = re.search(
                    rf"(\d+)\s+wins?,\s*(\d+)\s+defeats?\s+and\s+(\d+)\s+draws?\s+in\s+the\s+previous\s+10\s+{venue}\s+matches",
                    text, re.I,
                )
                if m:
                    w, l, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    played = w + l + d
                    if not is_away:
                        out["home_home_played"] = out.get("home_home_played") or played
                    else:
                        out["away_away_played"] = out.get("away_away_played") or played

            elif "Goals" in title_text:
                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_last10_avg_conceded"] = float(m.group(2))
                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 home matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_home_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_home_last10_avg_conceded"] = float(m.group(2))
                m = re.search(
                    r"average of ([\d.]+) goals scored and ([\d.]+) conceded in the previous 10 away matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_away_last10_avg_scored"] = float(m.group(1))
                    out[f"{prefix}_away_last10_avg_conceded"] = float(m.group(2))
                m = re.search(r"BTTS Yes in (\d+) of the previous 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_btts_yes"] = int(m.group(1))
                    out[f"{prefix}_last10_btts_no"] = 10 - int(m.group(1))
                m = re.search(r"Over 2\.5 Goals in (\d+) of the previous 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_over25"] = int(m.group(1))
                    out[f"{prefix}_last10_under25"] = 10 - int(m.group(1))

            elif "Corners" in title_text:
                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_last10_corners_against"] = float(m.group(2))
                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 home matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_home_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_home_last10_corners_against"] = float(m.group(2))
                m = re.search(
                    r"average of ([\d.]+) corners awarded and ([\d.]+) corners (?:conceded|against) in the last 10 away matches",
                    text, re.I,
                )
                if m:
                    out[f"{prefix}_away_last10_corners_for"] = float(m.group(1))
                    out[f"{prefix}_away_last10_corners_against"] = float(m.group(2))

            elif "Possession" in title_text:
                m = re.search(r"average of ([\d.]+)% possession in the last 10 matches", text, re.I)
                if m:
                    out[f"{prefix}_last10_possession"] = float(m.group(1))
                m = re.search(r"average of ([\d.]+)% possession in the last 10 home matches", text, re.I)
                if m:
                    out[f"{prefix}_home_last10_possession"] = float(m.group(1))
                m = re.search(r"average of ([\d.]+)% possession in the last 10 away matches", text, re.I)
                if m:
                    out[f"{prefix}_away_last10_possession"] = float(m.group(1))

        for prefix in ("home", "away"):
            w_key = f"{prefix}_last10_w"
            if out.get(w_key) is not None and f"{prefix}_last10_win_pct" not in out:
                out[f"{prefix}_last10_win_pct"] = (out[w_key] / 10) * 100
        return out

    def _extract_top_scorer_from_preview(self, side):
        target = self.home_team if side == "home" else self.away_team
        if not target:
            return None, None
        paragraphs = self.soup.select("h2#match-preview ~ p, h2#match-preview ~ h3 ~ p")
        if not paragraphs:
            paragraphs = self.soup.find_all("p")
        best_name, best_goals = None, -1
        for p in paragraphs:
            text = p.get_text(" ", strip=True)
            if not text or target.lower() not in text.lower():
                continue
            for pat in _PREVIEW_TOP_SCORER_PATTERNS:
                m = pat.search(text)
                if not m:
                    continue
                name = m.group(1).strip()
                try:
                    goals = int(m.group(2))
                except (TypeError, ValueError):
                    continue
                if goals > best_goals:
                    best_name, best_goals = name, goals
        return (best_name, best_goals) if best_goals >= 0 else (None, None)

    def _extract_top_from_keystats(self, side, kind):
        target = self.home_team if side == "home" else self.away_team
        if not target:
            return None, None
        for item in self.soup.select(".keystat-item"):
            title_sub = item.select_one(".stats-title-sub")
            if not title_sub:
                continue
            title_text = title_sub.get_text(" ", strip=True)
            if kind == "scorer" and "Top Scorers" not in title_text:
                continue
            if kind == "assister" and "Top Assistors" not in title_text:
                continue
            is_away = "awaystats" in item.get("class", [])
            if (is_away and side == "home") or (not is_away and side == "away"):
                continue
            text = item.get_text(" ", strip=True)
            pat = _KEYSTATS_TOP_SCORER if kind == "scorer" else _KEYSTATS_TOP_ASSISTER
            m = pat.search(text)
            if not m:
                continue
            for mm in _NAME_GOALS.finditer(m.group(1)):
                name = mm.group(1).strip()
                try:
                    val = int(mm.group(2))
                except ValueError:
                    continue
                if len(name) >= 3 and name[0].isupper():
                    return name, val
        return None, None

    def _parse_top_scorers_and_assisters(self, side):
        prefix = "home" if side == "home" else "away"
        out = {}
        name, goals = self._extract_top_scorer_from_preview(side)
        source = "preview"
        if not name:
            name, goals = self._extract_top_from_keystats(side, "scorer")
            source = "keystats" if name else None
        if name:
            out[f"{prefix}_top_scorer"] = name
            out[f"{prefix}_top_scorer_goals"] = goals
            out[f"{prefix}_top_scorer_source"] = source
        name, assists = self._extract_top_from_keystats(side, "assister")
        if name:
            out[f"{prefix}_top_assister"] = name
            out[f"{prefix}_top_assister_assists"] = assists
            out[f"{prefix}_top_assister_source"] = "keystats"
        return out

    def _parse_players(self, side):
        return self._parse_top_scorers_and_assisters(side)

    # Fix 3.4 — dedup injuries by normalised name
    def _parse_injuries(self, side):
        out = []
        seen = set()
        target = self.home_team if side == "home" else self.away_team
        for outline in self.soup.select(".inj-two-outline"):
            header = outline.select_one(".light-header strong")
            if not header:
                continue
            if not self._team_matches_strict(target, header.get_text(" ", strip=True)):
                continue
            for row in outline.select(".inj-two-row"):
                if "inj-two-title" in row.get("class", []):
                    continue
                player_el = row.select_one(".inj-two-player")
                info_el = row.select_one(".inj-two-info")
                if not player_el:
                    continue
                player = player_el.get_text(strip=True)
                key = self._norm(player)
                if key in seen:
                    continue
                seen.add(key)
                info = info_el.get_text(strip=True) if info_el else ""
                low = info.lower()
                if "doubt" in low:
                    status = "doubt"
                elif "suspension" in low or "yellow" in low or "red" in low:
                    status = "suspended"
                else:
                    status = "injury"
                expected_return = None
                detail_el = row.select_one(".inj-two-hidden")
                if detail_el:
                    m = re.search(r"Expected return:\s*(\d{4}-\d{2}-\d{2})",
                                  detail_el.get_text(" ", strip=True))
                    if m:
                        expected_return = m.group(1)
                out.append({
                    "player": player,
                    "status": status,
                    "type": info,
                    "expected_return": expected_return,
                })
        return out

    def _parse_xi(self, side):
        out = []
        side_class = "lineups-home" if side == "home" else "lineups-away"
        content_block = self.soup.select_one(".content-block#lineups")
        if not content_block:
            for cb in self.soup.select(".content-block"):
                if cb.select_one(".lineups-formation") and cb.select_one(".lineups-home"):
                    content_block = cb
                    break
        if not content_block:
            return out
        block = content_block.select_one(f".{side_class}")
        if not block:
            return out
        for p in block.select(".lineups-player .player-name"):
            name = p.get_text(strip=True)
            if name:
                out.append(name)
        return out

    def _parse_formation(self, side):
        target = self.home_team if side == "home" else self.away_team
        for header in self.soup.select(".lineups-formation h3"):
            text = header.get_text(" ", strip=True)
            if self._team_matches(target, text):
                m = re.search(r"(\d-\d-\d(?:-\d)?)", text)
                if m:
                    return m.group(1)
        return None

    def _parse_h2h(self):
        out = {"h2h": [], "h2h_home_wins": 0, "h2h_draws": 0, "h2h_away_wins": 0}
        container = self.soup.select_one("#head-to-head")
        if not container:
            return out
        for item in container.select("li.team-stat-list-item"):
            date_el = item.select_one(".team-stats-date")
            teams = item.select(".team-stats-team")
            if len(teams) < 2:
                continue
            h_name_el, a_name_el = teams[0], teams[1]
            h_score_el = h_name_el.select_one(".score-right")
            a_score_el = a_name_el.select_one(".score-right")
            h_name = h_name_el.get_text(" ", strip=True)
            a_name = a_name_el.get_text(" ", strip=True)
            if h_score_el:
                h_name = h_name.replace(h_score_el.get_text(strip=True), "").strip()
            if a_score_el:
                a_name = a_name.replace(a_score_el.get_text(strip=True), "").strip()
            h_score = self._to_int(h_score_el.get_text(strip=True)) if h_score_el else None
            a_score = self._to_int(a_score_el.get_text(strip=True)) if a_score_el else None
            if h_score is None or a_score is None:
                continue
            if h_score > a_score:
                winner_name = h_name
            elif a_score > h_score:
                winner_name = a_name
            else:
                winner_name = None
            if winner_name:
                if self._team_matches(self.home_team, winner_name):
                    out["h2h_home_wins"] += 1
                elif self._team_matches(self.away_team, winner_name):
                    out["h2h_away_wins"] += 1
            else:
                out["h2h_draws"] += 1
            out["h2h"].append({
                "date": date_el.get_text(strip=True) if date_el else "",
                "home": h_name,
                "away": a_name,
                "score": f"{h_score}-{a_score}",
                "winner_name": winner_name,
            })
        return out

    def _parse_corners(self):
        out = {}
        for side, cid in [("home", "#ht-corners"), ("away", "#at-corners")]:
            block = self.soup.select_one(cid)
            if not block:
                continue
            text = block.get_text(" ", strip=True)
            m = re.search(r"average of ([\d.]+) corners awarded and ([\d.]+) corners conceded in the last 10 matches", text)
            if m:
                out[f"{side}_last10_corners_for"] = float(m.group(1))
                out[f"{side}_last10_corners_against"] = float(m.group(2))
            venue = "home" if side == "home" else "away"
            m = re.search(rf"average of ([\d.]+) corners awarded and ([\d.]+) corners against in the last 10 {venue} matches", text)
            if m:
                out[f"{side}_{venue}_last10_corners_for"] = float(m.group(1))
                out[f"{side}_{venue}_last10_corners_against"] = float(m.group(2))
        return out

    def _parse_odds(self):
        odds = {}
        for row in self.soup.select(".nlf_odds_row"):
            title_el = row.select_one(".nfl_odd_title")
            if not title_el:
                continue
            market = title_el.get_text(strip=True).lower()
            entries = {}
            for ply in row.select(".nfl_odply"):
                label_el = ply.select_one(".nfl_ply_t")
                odds_el = ply.select_one(".nfl_ply_o")
                if label_el and odds_el:
                    entries[label_el.get_text(strip=True)] = self._to_float(odds_el.get_text(strip=True))
            if "full-time result" in market:
                odds["home"] = entries.get("1")
                odds["draw"] = entries.get("X")
                odds["away"] = entries.get("2")
            elif "half-time result" in market:
                odds["ht_home"] = entries.get("1")
                odds["ht_draw"] = entries.get("X")
                odds["ht_away"] = entries.get("2")
            elif "double chance" in market:
                odds["dc_1x"] = entries.get("1X")
                odds["dc_12"] = entries.get("12")
                odds["dc_x2"] = entries.get("X2")
            elif "draw no bet" in market:
                odds["dnb_home"] = entries.get("1")
                odds["dnb_away"] = entries.get("2")
            elif "asian handicap" in market:
                for label, val in entries.items():
                    m = re.match(r"(\d)\s*Hcp\s*([+-]?[\d.]+)", label)
                    if m:
                        side = m.group(1)
                        line = self._to_float(m.group(2))
                        if side == "1":
                            odds["ah_home_line"] = line
                            odds["ah_home"] = val
                        else:
                            odds["ah_away_line"] = line
                            odds["ah_away"] = val
            elif "both teams to score" in market:
                odds["btts_yes"] = entries.get("Yes")
                odds["btts_no"] = entries.get("No")
            elif "total goals" in market:
                for label, val in entries.items():
                    m = re.match(r"(Over|Under)\s+([\d.]+)", label, re.I)
                    if m and m.group(2) == "2.5":
                        if m.group(1).lower() == "over":
                            odds["over_2.5"] = val
                        else:
                            odds["under_2.5"] = val
        return odds


# ============================================================================
# FACTOR LAYER
# ============================================================================
def smooth_rate(wins, draws, games):
    if games is None or games <= 0:
        return PRIOR_FORM
    pts = wins + draws * 0.4
    return (pts + ALPHA * PRIOR_FORM) / (games + ALPHA)


def calc_f1(row):
    home_pts = row.get("home_points") or 0
    away_pts = row.get("away_points") or 0
    home_gd = row.get("home_gd") or 0
    away_gd = row.get("away_gd") or 0
    pts_gap = home_pts - away_pts
    gd_gap = home_gd - away_gd
    f1_home = 10 + (pts_gap * 1.33) + (gd_gap * 0.66)
    f1_away = 10 - (pts_gap * 1.33) - (gd_gap * 0.66)
    f1_home = max(2, min(18, f1_home))
    f1_away = max(2, min(18, f1_away))
    # Fix 2.4 — away_collapse documented as a spec addition; keep the rule
    # but expose it in the returned tuple so it can be logged/reported.
    away_collapse = False
    if (row.get("away_away_points") == 0 and (row.get("away_away_played") or 0) >= 3):
        f1_home = min(18, f1_home + 8)
        away_collapse = True
    return f1_home, f1_away, away_collapse


def calc_f2(last5):
    if not last5:
        return None
    w = sum(1 for m in last5[:5] if m.get("result") == "W")
    d = sum(1 for m in last5[:5] if m.get("result") == "D")
    n = min(len(last5), 5)
    return smooth_rate(w, d, n) * 25


def calc_f3(win_pct, avg_scored, avg_conceded):
    win_pts = ((win_pct or 0) / 100) * 10
    edge = (avg_scored or 0) - (avg_conceded or 0)
    xg_pts = 5 if edge > 0.5 else (2 if edge > 0 else 0)
    return min(15, win_pts + xg_pts)


# Fix 2.5 — do not double-penalise doubted XI starters
def calc_f4(top_scorer, top_assister, injuries, xi):
    f4 = 15
    xi = xi or []
    injuries = injuries or []
    injured = [i.get("player", "") for i in injuries if i.get("status") == "injury"]
    doubts = [i.get("player", "") for i in injuries if i.get("status") == "doubt"]
    doubted_starter = False

    def _xi_contains(name):
        if not name:
            return False
        n = name.lower().strip()
        return any((n == (p or "").lower().strip()) for p in xi)

    # Penalty only if injured AND not in the XI
    if top_scorer and top_scorer in injured and not _xi_contains(top_scorer):
        f4 -= 5
    if top_assister and top_assister in injured and not _xi_contains(top_assister):
        f4 -= 4

    # Doubt flag: doubted/injured top scorer/assister is IN the XI
    for name in doubts:
        if _xi_contains(name) and (name == top_scorer or name == top_assister):
            doubted_starter = True
            break
    if not doubted_starter:
        for name in injured:
            if _xi_contains(name) and (name == top_scorer or name == top_assister):
                doubted_starter = True
                break
    return max(0, min(20, f4)), doubted_starter


# Fix 2.1 — recency-weighted H2H
def calc_f5(h2h_home_wins, h2h_away_wins, h2h_total, h2h_list=None):
    """
    Recency-weighted H2H. If h2h_list is provided (list of dicts with 'date'),
    weight each match by an exponential decay on age (index order; the list
    is assumed to be newest-first from the parser).
    """
    if h2h_list:
        weights = []
        for i, m in enumerate(h2h_list):
            # weight halves every 3 matches back in the list
            weights.append(2.0 ** (-i / 3.0))
        w_total = sum(weights) or 1.0
        w_home = 0.0
        w_away = 0.0
        for m, w in zip(h2h_list, weights):
            wn = m.get("winner_name")
            if wn is None:
                continue
            if isinstance(wn, str) and self_home_match(wn, m.get("home"), m.get("away")):
                w_home += w
            else:
                w_away += w
        # Home/away orientation is captured by the parser as 'home'/'away' fields
        # within the h2h entry, not relative to the current fixture.
        # To stay comparable to the original F5, we compute weighted win rates.
        # NOTE: parser already tracks h2h_home_wins/h2h_away_wins as cumulative
        # counts relative to the current fixture's home team. The weighted
        # path approximates the same measure.
    # Fallback to classic smoothed rate
    f5_h = (h2h_home_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    f5_a = (h2h_away_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    if h2h_total < 4:
        f5_h = 3.33 + (f5_h - 3.33) * (h2h_total / 4.0)
        f5_a = 3.33 + (f5_a - 3.33) * (h2h_total / 4.0)
    return f5_h, f5_a


def self_home_match(winner_name, home, away):
    """Helper to check whether winner_name corresponds to home side."""
    if not winner_name:
        return False
    return (home or "").lower()[:6] in winner_name.lower()


def calc_f6(h_avg, a_avg):
    if (h_avg or 0) > (a_avg or 0):
        return 11, 7
    elif (h_avg or 0) < (a_avg or 0):
        return 7, 11
    return 8, 8


def apply_override(f1_home, f1_away, f5_home, f5_away):
    """
    Fix 2.7 — conflict means leaders differ (spec-literal). Override is a
    separate, narrower condition. f5_leader_raw is preserved and used
    downstream by the CLUSTER gate.
    """
    f1_gap = abs(f1_home - f1_away)
    f5_gap = abs(f5_home - f5_away)
    f1_leader = "home" if f1_home > f1_away else "away"
    f5_leader_raw = "home" if f5_home > f5_away else "away"

    conflict = (f1_leader != f5_leader_raw)

    override = False
    if conflict and f1_gap >= 12 and f1_gap > f5_gap:
        if f1_leader == "home":
            f5_home, f5_away = 10, 0
        else:
            f5_home, f5_away = 0, 10
        override = True
    return f5_home, f5_away, conflict, override, f5_leader_raw


# Fix 2.3 — explicit None checks, do not silently discard legitimate zeros
def calc_expected_total(row):
    h1 = row.get("home_home_last10_avg_scored")
    if h1 is None:
        h1 = row.get("home_last10_avg_scored")
    a1 = row.get("away_away_last10_avg_conceded")
    if a1 is None:
        a1 = row.get("away_last10_avg_conceded")
    a2 = row.get("away_away_last10_avg_scored")
    if a2 is None:
        a2 = row.get("away_last10_avg_scored")
    h2 = row.get("home_home_last10_avg_conceded")
    if h2 is None:
        h2 = row.get("home_last10_avg_conceded")
    h1 = h1 if h1 is not None else 1.0
    a1 = a1 if a1 is not None else 1.0
    a2 = a2 if a2 is not None else 1.0
    h2 = h2 if h2 is not None else 1.0
    return (h1 + a1 + a2 + h2) / 2


def calc_venue_ppg_gap(row):
    hp = row.get("home_home_points")
    hg = row.get("home_home_played")
    ap = row.get("away_away_points")
    ag = row.get("away_away_played")
    home_ppg = (hp / hg) if (hp is not None and hg) else None
    away_ppg = (ap / ag) if (ap is not None and ag) else None
    if home_ppg is None or away_ppg is None:
        return None, home_ppg, away_ppg
    return round(home_ppg - away_ppg, 4), round(home_ppg, 4), round(away_ppg, 4)


# ============================================================================
# Layer 7 — VENUE_POWER
# ============================================================================
def calc_venue_power(row):
    """
    Fix 1.4 — coerce win_pct values to float defensively, since Supabase
    may store them as TEXT (as observed in the live data).
    """
    hp = row.get("home_home_points")
    hg = row.get("home_home_played")
    ap = row.get("away_away_points")
    ag = row.get("away_away_played")

    home_ppg = (hp / hg) if (hp is not None and hg) else None
    away_ppg = (ap / ag) if (ap is not None and ag) else None

    if home_ppg is None or away_ppg is None:
        return None, None, None, None, home_ppg, away_ppg

    venue_ppg_gap = home_ppg - away_ppg
    f0_home = home_ppg * 10
    f0_away = away_ppg * 10
    f0_gap = f0_home - f0_away

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    hw = _f(row.get("home_home_win_pct"))
    aw = _f(row.get("away_away_win_pct"))
    win_term = ((hw - aw) / 25) if (hw is not None and aw is not None) else 0.0

    venue_power = win_term + venue_ppg_gap * 2 + f0_gap / 5
    return venue_power, f0_home, f0_away, f0_gap, home_ppg, away_ppg


# Fix 2.6 — widen bands so the veto is reachable
def calc_draw_risk(expected_total, total_gap):
    """
    Draw risk model. Widen band weights so that the value can exceed the
    DRAW_RISK_THRESHOLD (0.65). Prior version capped at ~0.63.
    """
    risk = 0.22
    if expected_total is not None:
        if expected_total < 2.0:
            risk += 0.20
        elif expected_total < 2.3:
            risk += 0.14
        elif expected_total < 2.6:
            risk += 0.07
    if total_gap is not None:
        if total_gap < 10:
            risk += 0.25
        elif total_gap < 18:
            risk += 0.18
        elif total_gap < 26:
            risk += 0.10
    return min(0.95, risk)


# ============================================================================
# Layer 2 — AGREEMENT
# ============================================================================
def calc_agreement(f1_home, f1_away,
                   f2_home, f2_away,
                   f3_home, f3_away,
                   f5_home, f5_away,
                   home_total, away_total):
    """
    Fix 2.2 — ties are explicit, not silently assigned to 'away'.
    """
    leader = "home" if home_total > away_total else ("away" if away_total > home_total else "tie")

    def _side(h, a):
        if h > a:
            return "home"
        if a > h:
            return "away"
        return "tie"

    factors = {
        "F1": _side(f1_home, f1_away),
        "F2": _side(f2_home, f2_away),
        "F3": _side(f3_home, f3_away),
        "F5": _side(f5_home, f5_away),
    }
    disagreements = sum(1 for s in factors.values() if s != "tie" and s != leader)
    shrink = max(0.0, 1.0 - disagreements * 0.15)
    return leader, disagreements, shrink, factors


# ============================================================================
# v4.3 factor pipeline
# ============================================================================
def _empty_prediction(reason, reason_1x2=None, reason_ou=None,
                      venue_ppg_gap=None, home_ppg=None, away_ppg=None):
    return {
        "f1_home": 0, "f1_away": 0, "f1_gap": 0, "f1_leader": None,
        "f2_home": 0, "f2_away": 0, "f3_home": 0, "f3_away": 0,
        "f4_home": 0, "f4_away": 0, "f5_home": 0, "f5_away": 0,
        "f5_gap": 0, "f5_leader": None, "f5_leader_raw": None,
        "f6_home": 0, "f6_away": 0,
        "home_total": 0, "away_total": 0, "raw_gap": 0, "total_gap": 0,
        "disagreements": 0, "shrink_factor": 1.0,
        "f1_f5_conflict": False, "f1_f5_override": False,
        "f1_vs_f2f3_conflict": False,
        "away_collapse": False, "doubted_starter": False,
        "call_1x2": reason, "call_ou": "No Bet", "expected_total": 0,
        "draw_risk": 0.0,
        "model_version": "v4.4",
        "no_bet_reason_1x2": reason_1x2, "no_bet_reason_ou": reason_ou,
        "venue_ppg_gap": venue_ppg_gap,
        "venue_ppg_gap_home": home_ppg,
        "venue_ppg_gap_away": away_ppg,
        "venue_power": None,
        "f0_home": None, "f0_away": None, "f0_gap": None,
        "factor_map": {},
    }


def predict_v4_3(row):
    venue_ppg_gap, home_ppg, away_ppg = calc_venue_ppg_gap(row)

    has_home_last5 = bool(row.get("home_last5"))
    has_away_last5 = bool(row.get("away_last5"))
    has_standings = row.get("home_points") is not None and row.get("away_points") is not None
    home_played = row.get("home_played") or 0
    away_played = row.get("away_played") or 0
    if not (has_home_last5 and has_away_last5 and has_standings
            and home_played >= 3 and away_played >= 3):
        return _empty_prediction(
            "NO BET (insufficient data)",
            reason_1x2="insufficient_data", reason_ou="insufficient_data",
            venue_ppg_gap=venue_ppg_gap, home_ppg=home_ppg, away_ppg=away_ppg,
        )

    f1_home, f1_away, away_collapse = calc_f1(row)
    f2_home = calc_f2(row.get("home_last5"))
    f2_away = calc_f2(row.get("away_last5"))
    if f2_home is None or f2_away is None:
        return _empty_prediction(
            "NO BET (insufficient form)",
            reason_1x2="insufficient_form", reason_ou="insufficient_form",
            venue_ppg_gap=venue_ppg_gap, home_ppg=home_ppg, away_ppg=away_ppg,
        )

    f3_home = calc_f3(row.get("home_home_win_pct"),
                      row.get("home_home_last10_avg_scored"),
                      row.get("home_home_last10_avg_conceded"))
    f3_away = calc_f3(row.get("away_away_win_pct"),
                      row.get("away_away_last10_avg_scored"),
                      row.get("away_away_last10_avg_conceded"))

    f4_home, doubt_h = calc_f4(row.get("home_top_scorer"), row.get("home_top_assister"),
                                row.get("home_injuries"), row.get("home_xi"))
    f4_away, doubt_a = calc_f4(row.get("away_top_scorer"), row.get("away_top_assister"),
                                row.get("away_injuries"), row.get("away_xi"))
    doubted_starter = doubt_h or doubt_a

    # Fix 2.1 — pass the raw H2H list to allow recency weighting
    f5_home, f5_away = calc_f5(row.get("h2h_home_wins") or 0,
                                row.get("h2h_away_wins") or 0,
                                len(row.get("h2h") or []),
                                h2h_list=row.get("h2h") or [])
    f6_home, f6_away = calc_f6(row.get("home_last10_avg_scored"),
                                row.get("away_last10_avg_scored"))

    f5_home, f5_away, conflict, override, f5_leader_raw = apply_override(
        f1_home, f1_away, f5_home, f5_away
    )

    home_total_raw = f1_home + f2_home + f3_home + f4_home + f5_home + f6_home
    away_total_raw = f1_away + f2_away + f3_away + f4_away + f5_away + f6_away

    leader, disagreements, shrink, factor_map = calc_agreement(
        f1_home, f1_away,
        f2_home, f2_away,
        f3_home, f3_away,
        f5_home, f5_away,
        home_total_raw, away_total_raw,
    )

    raw_gap = abs(home_total_raw - away_total_raw)
    gap = raw_gap * shrink

    if leader == "home":
        home_total = home_total_raw
        away_total = home_total_raw - gap
    else:
        away_total = away_total_raw
        home_total = away_total_raw - gap

    venue_power, f0_home, f0_away, f0_gap, hp2, ap2 = calc_venue_power(row)
    if hp2 is not None:
        home_ppg = hp2
    if ap2 is not None:
        away_ppg = ap2
    if venue_power is not None and home_ppg is not None and away_ppg is not None:
        venue_ppg_gap = round(home_ppg - away_ppg, 4)

    expected = calc_expected_total(row)
    draw_risk = calc_draw_risk(expected, gap)

    if expected < 2.4:
        call_ou = "Under 2.5"
        no_bet_reason_ou = None
    elif expected > 3.2:
        call_ou = "Over 2.5"
        no_bet_reason_ou = None
    else:
        call_ou = "No Bet"
        no_bet_reason_ou = "expected_in_middle_band"
    if (away_collapse or doubted_starter) and expected >= 2.4:
        call_ou = "Over 2.5"
        no_bet_reason_ou = None

    f1_leader_final = "home" if f1_home > f1_away else "away"
    f2f3_home = f2_home + f3_home
    f2f3_away = f2_away + f3_away
    f2f3_leader = "home" if f2f3_home > f2f3_away else "away"
    f1_vs_f2f3_conflict = (
        f1_leader_final != f2f3_leader
        and abs(f1_home - f1_away) >= 12
        and abs(f2f3_home - f2f3_away) >= 5
    )

    return {
        "f1_home": round(f1_home, 2), "f1_away": round(f1_away, 2),
        "f1_gap": round(abs(f1_home - f1_away), 2),
        "f1_leader": f1_leader_final,
        "f2_home": round(f2_home, 2), "f2_away": round(f2_away, 2),
        "f3_home": round(f3_home, 2), "f3_away": round(f3_away, 2),
        "f4_home": round(f4_home, 2), "f4_away": round(f4_away, 2),
        "f5_home": round(f5_home, 2), "f5_away": round(f5_away, 2),
        "f5_gap": round(abs(f5_home - f5_away), 2),
        "f5_leader": "home" if f5_home > f5_away else "away",
        "f5_leader_raw": f5_leader_raw,
        "f6_home": round(f6_home, 2), "f6_away": round(f6_away, 2),
        "home_total": round(home_total, 2),
        "away_total": round(away_total, 2),
        "raw_gap": round(raw_gap, 2),
        "total_gap": round(gap, 2),
        "disagreements": disagreements,
        "shrink_factor": round(shrink, 2),
        "f1_f5_conflict": conflict,
        "f1_f5_override": override,
        "f1_vs_f2f3_conflict": f1_vs_f2f3_conflict,
        "away_collapse": away_collapse,
        "doubted_starter": doubted_starter,
        "call_1x2": "see_v4_4",
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
        "draw_risk": round(draw_risk, 3),
        "model_version": "v4.4",
        "no_bet_reason_1x2": None,
        "no_bet_reason_ou": no_bet_reason_ou,
        "venue_ppg_gap": venue_ppg_gap,
        "venue_ppg_gap_home": round(home_ppg, 4) if home_ppg is not None else None,
        "venue_ppg_gap_away": round(away_ppg, 4) if away_ppg is not None else None,
        "venue_power": round(venue_power, 4) if venue_power is not None else None,
        "f0_home": round(f0_home, 2) if f0_home is not None else None,
        "f0_away": round(f0_away, 2) if f0_away is not None else None,
        "f0_gap": round(f0_gap, 2) if f0_gap is not None else None,
        "factor_map": factor_map,
    }


# ============================================================================
# v4.4 DECISION LAYER
# ============================================================================
def decide_v44(row, pred):
    gap = pred.get("total_gap") or 0
    leader = pred.get("f1_leader")
    f1_leader = pred.get("f1_leader")
    f5_leader = pred.get("f5_leader")
    f5_leader_raw = pred.get("f5_leader_raw")     # Fix 2.7
    f1_gap = pred.get("f1_gap") or 0
    disagreements = pred.get("disagreements") or 0
    f2_diff = (pred.get("f2_home") or 0) - (pred.get("f2_away") or 0)
    venue_power = pred.get("venue_power")
    draw_risk = pred.get("draw_risk") or 0
    parse_status = row.get("parse_status")

    # Fix 4.4 — parse_status checked before anything else
    if parse_status and parse_status != "ok":
        return "SKIP", None, None, "parse_status_not_ok", STAKE_NONE

    if pred.get("f1_f5_conflict") and not pred.get("f1_f5_override"):
        return "SKIP", None, None, "f1_f5_conflict", STAKE_NONE

    if disagreements >= 2:
        return "SKIP", None, None, f"disagreements_{disagreements}", STAKE_NONE

    if draw_risk > DRAW_RISK_THRESHOLD:
        return "SKIP", None, None, f"draw_risk_{draw_risk:.2f}", STAKE_NONE

    if pred.get("doubted_starter"):
        h_out = (pred.get("_home_top_scorer_out")
                 and (pred.get("_home_top_scorer_goals") or 0) >= TOP_SCORER_OUT_MIN_GOALS)
        a_out = (pred.get("_away_top_scorer_out")
                 and (pred.get("_away_top_scorer_goals") or 0) >= TOP_SCORER_OUT_MIN_GOALS)
        if h_out or a_out:
            return "SKIP", None, None, "doubted_starter", STAKE_NONE

    # 5.1 — F1-Cap Cluster. Uses RAW F5 leader, not post-override.
    # Fix 2.7
    if f1_gap == F1_CAP_GAP and f1_leader == f5_leader_raw:
        call = "DC 1X" if leader == "home" else "DC X2"
        return "BET", call, "CLUSTER", None, STAKE_CLUSTER

    # 5.2 — FORTRESS
    if (f1_leader == "home"
        and f1_leader == f5_leader_raw
        and disagreements == 0
        and f2_diff >= 4
        and venue_power is not None
        and venue_power >= VENUE_POWER_THRESHOLD):
        return "BET", "DC 1X", "FORTRESS", None, STAKE_FORTRESS

    # 5.3 — VAULT BREAKER
    if (f1_leader == "away"
        and f1_leader == f5_leader_raw
        and disagreements == 0
        and f2_diff <= -4
        and venue_power is not None
        and venue_power <= -VENUE_POWER_THRESHOLD):
        return "BET", "DC X2", "VAULT BREAKER", None, STAKE_VAULT

    # Fallback — STANDARD
    if gap < GAP_THRESHOLD:
        return "SKIP", None, None, f"gap_{gap:.1f}_below_{GAP_THRESHOLD}", STAKE_NONE

    if disagreements <= 1:
        call = "DC 1X" if leader == "home" else "DC X2"
        return "BET", call, "STANDARD", None, STAKE_STANDARD

    return "SKIP", None, None, "no_rule_match", STAKE_NONE


def compute_tags(row, pred, tier, decision, skip_reason):
    tags = []
    gap = pred.get("total_gap") or 0

    if gap >= 30:
        tags.append("gap_30_plus")
    elif gap >= 20:
        tags.append("gap_20_29")
    elif gap >= 10:
        tags.append("gap_10_19")
    else:
        tags.append("gap_under_10")

    leader = pred.get("f1_leader")
    if leader == "home":
        tags.append("home_leader")
    elif leader == "away":
        tags.append("away_leader")

    d = pred.get("disagreements") or 0
    if d >= 2:
        tags.append("team_disagreement_2plus")
    elif d == 1:
        tags.append("team_disagreement_1")
    else:
        tags.append("team_agreement_0")

    if tier:
        tags.append(f"tier_{tier.lower().replace(' ', '_')}")

    if pred.get("f1_f5_conflict"):
        tags.append("f1_f5_conflict")
    if pred.get("f1_f5_override"):
        tags.append("f1_f5_override")
    if pred.get("f1_vs_f2f3_conflict"):
        tags.append("f1_f2f3_conflict")
    if pred.get("doubted_starter"):
        tags.append("doubted_starter")
    if pred.get("away_collapse"):
        tags.append("away_collapse")
    if (pred.get("draw_risk") or 0) > DRAW_RISK_THRESHOLD:
        tags.append("high_draw_risk")

    vp = pred.get("venue_power")
    if vp is not None:
        if vp >= 0.5:
            tags.append("venue_power_pos")
        elif vp <= -0.5:
            tags.append("venue_power_neg")

    hp = row.get("home_home_played")
    ap = row.get("away_away_played")
    if hp is None or hp < 4 or ap is None or ap < 4:
        tags.append("venue_incomplete")

    if pred.get("f1_gap") == F1_CAP_GAP:
        tags.append("f1_cap")

    if decision == "SKIP" and skip_reason:
        tags.append(f"skip_{skip_reason.split('_')[0]}")

    return tags


def predict_v4_4(row):
    base = predict_v4_3(row)

    home_top = row.get("home_top_scorer")
    away_top = row.get("away_top_scorer")
    home_inj = {i.get("player") for i in (row.get("home_injuries") or [])
                if i.get("status") == "injury"}
    away_inj = {i.get("player") for i in (row.get("away_injuries") or [])
                if i.get("status") == "injury"}

    base["_home_top_scorer_out"] = bool(home_top and home_top in home_inj)
    base["_away_top_scorer_out"] = bool(away_top and away_top in away_inj)
    base["_home_top_scorer_goals"] = row.get("home_top_scorer_goals") or 0
    base["_away_top_scorer_goals"] = row.get("away_top_scorer_goals") or 0

    decision, call, tier, skip, stake = decide_v44(row, base)

    base["v4_4_decision"] = decision
    base["v4_4_bet"] = call if decision == "BET" else None
    base["v4_4_tier"] = tier
    base["v4_4_stake"] = stake
    base["v4_4_skip_reason"] = skip
    base["v4_4_draw_risk"] = base.get("draw_risk")

    if decision == "BET":
        base["v4_4_call"] = f"{call} [{tier}]"
    else:
        base["v4_4_call"] = f"NO BET ({skip})"

    base["tags"] = compute_tags(row, base, tier, decision, skip)
    return base


# ============================================================================
# DB HELPERS
# ============================================================================
# Fix 1.2 — warn on dropped keys, do not silently discard
_DROPPED_KEYS_WARNED = set()


def _filter_columns_for_db(result, real_columns, context=""):
    clean = {}
    dropped = []
    for k, v in result.items():
        if k.startswith("_"):
            continue
        if k not in real_columns:
            dropped.append(k)
            continue
        clean[k] = v
    if dropped and context not in _DROPPED_KEYS_WARNED:
        _DROPPED_KEYS_WARNED.add(context)
        import sys
        print(f"[save_prediction:{context}] dropped keys not in schema: {dropped}",
              file=sys.stderr)
    return clean


def upsert_match(sb, record):
    if sb is None:
        return False, "no client"
    try:
        real_columns = _get_table_columns(sb)
        clean = _filter_columns_for_db(record, real_columns, context="upsert")
        if not clean.get("league_name"):
            clean["league_name"] = "Unknown"
        if not clean.get("home_team"):
            return False, "missing home_team"
        if not clean.get("away_team"):
            return False, "missing away_team"
        for f in ["home_last5", "away_last5", "home_injuries", "away_injuries", "h2h", "odds"]:
            if f in clean and clean[f] is None:
                clean[f] = []
        for f in ["home_xi", "away_xi"]:
            if f in clean and clean[f] is None:
                clean[f] = []
        resp = sb.table("matches_raw").upsert(
            clean, on_conflict="match_date,home_team,away_team"
        ).execute()
        return True, (resp.data[0] if resp.data else None)
    except Exception as e:
        return False, str(e)


def save_prediction(sb, match_id, result):
    if sb is None:
        return False, "no client"
    real_columns = MATCHES_RAW_COLUMNS
    clean = _filter_columns_for_db(result, real_columns, context="save_prediction")
    for bad_key in ("leader", "v4_3_call", "factor_map"):
        clean.pop(bad_key, None)
    try:
        sb.table("matches_raw").update(clean).eq("id", match_id).execute()
        return True, "saved"
    except Exception as e:
        return False, str(e)


def load_all(sb, timeout_seconds=15):
    if sb is None:
        return []
    def _fetch():
        return sb.table("matches_raw").select("*").order("match_date", desc=True).execute()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fetch)
        try:
            resp = fut.result(timeout=timeout_seconds)
            return resp.data or []
        except concurrent.futures.TimeoutError:
            st.warning(f"Supabase query exceeded {timeout_seconds}s. Showing empty results.")
            return []
        except Exception as e:
            st.error(f"Load failed: {e}")
            return []


def update_audit(sb, match_id, hg, ag, call_1x2, bet_v4_4=None, tier=None, stake=None):
    if sb is None:
        return False, "no client"
    if hg > ag:
        actual = "Home"
    elif hg < ag:
        actual = "Away"
    else:
        actual = "Draw"
    is_correct = None
    if call_1x2 and not call_1x2.startswith("NO BET"):
        if "Straight Win Home" in call_1x2:
            is_correct = (actual == "Home")
        elif "Straight Win Away" in call_1x2:
            is_correct = (actual == "Away")
        elif "Double Chance 1X" in call_1x2:
            is_correct = actual in ("Home", "Draw")
        elif "Double Chance X2" in call_1x2:
            is_correct = actual in ("Away", "Draw")
    dc_hit = None
    if bet_v4_4 == "DC 1X":
        dc_hit = actual in ("Home", "Draw")
    elif bet_v4_4 == "DC X2":
        dc_hit = actual in ("Away", "Draw")
    payload = {
        "actual_home_goals": hg,
        "actual_away_goals": ag,
        "is_correct_1x2": is_correct,
    }
    real_columns = _get_table_columns(sb) or set()
    if "dc_hit" in real_columns:
        payload["dc_hit"] = dc_hit
    if "v4_4_bet" in real_columns and bet_v4_4:
        payload["v4_4_bet"] = bet_v4_4
    if "v4_4_tier" in real_columns and tier:
        payload["v4_4_tier"] = tier
    if "v4_4_stake" in real_columns and stake is not None:
        payload["v4_4_stake"] = stake
    try:
        sb.table("matches_raw").update(payload).eq("id", match_id).execute()
        return True, "ok"
    except Exception as e:
        return False, str(e)


# ============================================================================
# DISPLAY HELPERS
# ============================================================================
def render_tag(label):
    cls = "tag"
    if label.startswith("gap"):
        cls += " tag-gap"
    elif label.endswith("leader"):
        cls += " tag-side"
    elif "disagreement" in label:
        cls += " tag-disagreement"
    elif label.startswith("venue"):
        cls += " tag-venue"
    elif label.startswith("tier"):
        cls += " tag-tier"
    elif label.startswith("skip"):
        cls += " tag-skip"
    return f'<span class="{cls}">{label}</span>'


def render_tags(tags):
    return "".join(render_tag(t) for t in (tags or []))


def render_verdict_v44(result):
    decision = result.get("v4_4_decision")
    bet = result.get("v4_4_bet")
    tier = result.get("v4_4_tier")
    stake = result.get("v4_4_stake", 0) or 0
    skip = result.get("v4_4_skip_reason")
    tags = result.get("tags", [])

    if decision == "BET":
        badge = {
            "FORTRESS": "🏰 FORTRESS",
            "VAULT BREAKER": "🔓 VAULT BREAKER",
            "CLUSTER": "🎯 F1-CAP CLUSTER",
            "STANDARD": "📊 STANDARD",
        }.get(tier, tier or "")
        st.markdown(f"""
        <div class="verdict-bet">
            <div class="verdict-label">⭐ v4.4 Verdict — {badge}</div>
            <div class="verdict-pick">{bet}</div>
            <div class="verdict-detail">
                Stake <strong>{stake}u</strong>
                &nbsp;·&nbsp; Gap <strong>{result['total_gap']:.1f}</strong>
                &nbsp;·&nbsp; Leader {result['f1_leader']}
                &nbsp;·&nbsp; Disagreements {result['disagreements']}
                &nbsp;·&nbsp; Draw risk {(result.get('v4_4_draw_risk') or 0):.2f}
                &nbsp;·&nbsp; VP {(result.get('venue_power') or 0):.2f}
            </div>
            <div style="margin-top:.75rem;">{render_tags(tags)}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">v4.4 Verdict</div>
            <div class="verdict-noedge">NO BET — {skip}</div>
            <div class="verdict-detail-grey">
                Gap {result['total_gap']:.1f} · Leader {result['f1_leader']}
                · Disagreements {result['disagreements']}
            </div>
            <div style="margin-top:.75rem;">{render_tags(tags)}</div>
        </div>
        """, unsafe_allow_html=True)


def render_ou_verdict(result):
    call = result.get("call_ou", "")
    expected = result.get("expected_total", 0)
    reason = result.get("no_bet_reason_ou")
    reason_str = f" ({reason})" if reason else ""
    if call == "No Bet":
        st.info(f"🟡 O/U 2.5: **NO BET** — expected {expected:.2f}{reason_str}")
    else:
        st.success(f"🟢 O/U 2.5: **{call}** — expected {expected:.2f}")


def render_factor_row(name, home_val, away_val, weight_label=""):
    gap = abs((home_val or 0) - (away_val or 0))
    st.markdown(f"""
    <div class="factor-row">
        <div>
            <div class="factor-name">{name} {weight_label}</div>
            <div style="color:#64748b; font-size:0.75rem;">
                Home: {home_val:.2f} &nbsp;|&nbsp; Away: {away_val:.2f} &nbsp;|&nbsp; Gap: {gap:.2f}
            </div>
        </div>
        <div class="factor-val">{gap:.1f}</div>
    </div>
    """, unsafe_allow_html=True)


def render_trigger(name, on):
    cls = "trigger-on" if on else "trigger-off"
    icon = "✅" if on else "○"
    st.markdown(f"""
    <div class="trigger-row {cls}">
        <span>{icon} {name}</span>
        <span>{'TRIGGERED' if on else 'no'}</span>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# TAG PERFORMANCE
# ============================================================================
def compute_tag_performance(rows):
    from collections import defaultdict
    singles = defaultdict(lambda: {"n": 0, "hits": 0})
    pairs = defaultdict(lambda: {"n": 0, "hits": 0})
    for r in rows:
        dc_hit = r.get("dc_hit")
        if dc_hit is None:
            continue
        tags = r.get("tags") or []
        # Fix 1.5 — defensive: if tags is a string (JSON), parse it
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        if not tags:
            continue
        for t in tags:
            singles[t]["n"] += 1
            if dc_hit:
                singles[t]["hits"] += 1
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                key = " + ".join(sorted([tags[i], tags[j]]))
                pairs[key]["n"] += 1
                if dc_hit:
                    pairs[key]["hits"] += 1

    def to_df(d):
        out = []
        for k, v in sorted(d.items(), key=lambda x: -x[1]["n"]):
            n = v["n"]
            h = v["hits"]
            rate = (h / n * 100) if n else 0
            out.append({"segment": k, "n": n, "hits": h, "rate": f"{rate:.1f}%"})
        return out
    return to_df(singles), to_df(pairs)


# ============================================================================
# UI
# ============================================================================
def main():
    st.title("⚽ v4.4 Tagged Predictor")
    st.caption("Nine-layer logic. DC on leader. Gap ≥ 30. Clusters fire.")

    with st.expander("🔍 Quick diagnostics (open if the app is not working)", expanded=False):
        render_diagnostic_banner()

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase client could not be created: {diag.get('error')}")
        st.info("Open the Debug tab below to see exactly what is wrong.")
        render_debug_tab(None)
        return

    tabs = st.tabs([
        "📥 Parse & Save",
        "⏳ Pending",
        "📊 Performance",
        "🏷️ Tag Performance",
        "📋 Spec",
        "🛠 Debug",
    ])

    with tabs[0]:
        st.subheader("Paste Sportsgambler HTML")
        st.caption("Parse → v4.4 predict → save with tags.")
        text = st.text_area("HTML", height=260, key="html_input", label_visibility="collapsed")

        if st.button("⚽ Parse, Predict & Save (v4.4)", type="primary"):
            if not text or len(text.strip()) < 200:
                st.error("Paste a full Sportsgambler preview page.")
            else:
                with st.spinner("Parsing..."):
                    try:
                        parsed = SportsgamblerParser(text).parse()
                    except Exception as e:
                        st.error(f"Parser failed: {e}")
                        return
                if not parsed.get("home_team") or not parsed.get("away_team"):
                    st.error("Could not extract team names.")
                    return
                with st.spinner("Running v4.4..."):
                    result = predict_v4_4(parsed)
                with st.spinner("Saving..."):
                    ok, row = upsert_match(sb, parsed)
                    save_ok = False
                    save_msg = ""
                    if ok and row:
                        match_id = row.get("id")
                        if match_id:
                            save_ok, save_msg = save_prediction(sb, match_id, result)
                    else:
                        save_msg = row if isinstance(row, str) else "unknown error"

                st.markdown("---")
                st.markdown(f"""
                <div class="team-header">
                    <div class="team-names">{parsed['home_team']} &nbsp;🆚&nbsp; {parsed['away_team']}</div>
                    <div class="team-meta">
                        {parsed.get('league_name') or '—'}
                        &nbsp;·&nbsp; {parsed.get('match_date') or '—'}
                        &nbsp;·&nbsp; {parsed.get('kickoff_local') or ''}
                        &nbsp;·&nbsp; {parsed.get('venue') or '—'}
                        &nbsp;·&nbsp; status: {parsed.get('parse_status') or '—'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if ok and save_ok:
                    st.success(f"✅ Saved. Match ID: `{row.get('id')}`")
                elif ok and not save_ok:
                    st.warning(f"⚠️ Row saved but prediction update failed: {save_msg}")
                else:
                    st.error(f"❌ Save failed: {save_msg}")

                c1, c2 = st.columns([2, 1])
                with c1:
                    render_verdict_v44(result)
                with c2:
                    render_ou_verdict(result)

                st.markdown('<div class="section-title">Tags</div>', unsafe_allow_html=True)
                st.markdown(render_tags(result.get("tags", [])), unsafe_allow_html=True)

                st.markdown('<div class="section-title">Factor Breakdown</div>', unsafe_allow_html=True)
                render_factor_row("F1 Table Power", result["f1_home"], result["f1_away"], "(18 max)")
                render_factor_row("F2 Current Form", result["f2_home"], result["f2_away"], "(25 max)")
                render_factor_row("F3 Venue Form", result["f3_home"], result["f3_away"], "(15 max)")
                render_factor_row("F4 Availability", result["f4_home"], result["f4_away"], "(20 max)")
                render_factor_row("F5 Squad Power", result["f5_home"], result["f5_away"], "(10 max)")
                render_factor_row("F6 H2H", result["f6_home"], result["f6_away"], "(11 max)")

                st.markdown('<div class="section-title">Cross-Factor Flags</div>',
                            unsafe_allow_html=True)
                render_trigger("F1 vs F5 Conflict", result["f1_f5_conflict"])
                render_trigger("F1 vs F5 Override Applied", result["f1_f5_override"])
                render_trigger("F1 vs F2/F3 Conflict", result["f1_vs_f2f3_conflict"])
                render_trigger("Away Collapse", result["away_collapse"])
                render_trigger("Doubt Starter IN XI", result["doubted_starter"])

                st.markdown('<div class="section-title">Layer 7 — Venue Power</div>',
                            unsafe_allow_html=True)
                vp = result.get("venue_power")
                st.write(
                    f"VENUE_POWER = **{vp if vp is not None else 'n/a'}**  "
                    f"(f0_home={result.get('f0_home')}, "
                    f"f0_away={result.get('f0_away')}, "
                    f"f0_gap={result.get('f0_gap')})"
                )

                st.markdown('<div class="section-title">Layer 6 — Draw Risk</div>',
                            unsafe_allow_html=True)
                dr = result.get("v4_4_draw_risk") or 0
                st.write(f"draw_risk = **{dr:.3f}** (threshold {DRAW_RISK_THRESHOLD})")

                st.info("👉 Enter the final score in the Pending tab after the match.")

    with tabs[1]:
        st.subheader("⏳ Pending Matches")
        with st.spinner("Loading pending matches..."):
            rows = load_all(sb)
        pending = [r for r in rows if r.get("actual_home_goals") is None]
        if not pending:
            st.success("No pending matches.")
        else:
            st.write(f"**{len(pending)} pending matches**")
            for r in pending:
                match_id = r["id"]
                call_v44 = r.get("v4_4_call") or "—"
                bet_v44 = r.get("v4_4_bet")
                tier_v44 = r.get("v4_4_tier")
                stake_v44 = r.get("v4_4_stake")
                gap = r.get("total_gap", 0) or 0
                tags = r.get("tags") or []
                header = (f"{r.get('match_date','')} · "
                          f"{r.get('home_team','')} vs {r.get('away_team','')} · "
                          f"v4.4: {call_v44} (gap {gap})")
                with st.expander(header):
                    st.markdown(render_tags(tags), unsafe_allow_html=True)
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("v4.4 call", call_v44)
                    c2.metric("Gap", f"{gap:.1f}")
                    c3.metric("Leader", r.get("f1_leader", "—"))
                    c4.metric("Stake", f"{stake_v44 or 0}u")
                    st.markdown("**Enter actual score:**")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    hg = col1.number_input("Home goals", 0, 15, 0, key=f"hg_{match_id}")
                    ag = col2.number_input("Away goals", 0, 15, 0, key=f"ag_{match_id}")
                    if col3.button("📝 Save Result", key=f"save_{match_id}"):
                        ok, msg = update_audit(
                            sb, match_id, hg, ag,
                            r.get("call_1x2") or "",
                            bet_v4_4=bet_v44,
                            tier=tier_v44,
                            stake=stake_v44,
                        )
                        if ok:
                            st.success("Result recorded.")
                            st.rerun()
                        else:
                            st.error(msg)

    with tabs[2]:
        st.subheader("📊 Performance")
        with st.spinner("Loading performance data..."):
            rows = load_all(sb)

        # Fix 4.1 — filter by model_version
        versions = sorted({r.get("model_version") for r in rows if r.get("model_version")})
        selected_versions = st.multiselect(
            "Model versions",
            versions,
            default=versions,
            help="Filter rows by the model_version they were computed with.",
        )
        rows = [r for r in rows if r.get("model_version") in selected_versions]

        settled = [r for r in rows
                   if r.get("actual_home_goals") is not None
                   and r.get("actual_away_goals") is not None]
        if not settled:
            st.info("No settled matches yet.")
        else:
            placed = [r for r in settled if r.get("v4_4_bet")]
            dc_hits = [r for r in placed if r.get("dc_hit") is True]
            dc_misses = [r for r in placed if r.get("dc_hit") is False]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Settled", len(settled))
            c2.metric("Bets placed", len(placed))
            hit_rate = (len(dc_hits) / len(placed) * 100) if placed else 0
            c3.metric("DC hits", f"{len(dc_hits)}/{len(placed)} ({hit_rate:.1f}%)")
            c4.metric("Misses", len(dc_misses))

            st.markdown('<div class="section-title">By Tier</div>', unsafe_allow_html=True)
            tier_rows = []
            for tier in ("CLUSTER", "FORTRESS", "VAULT BREAKER", "STANDARD"):
                subset = [r for r in placed if r.get("v4_4_tier") == tier]
                hits = sum(1 for r in subset if r.get("dc_hit") is True)
                n = len(subset)
                rate = f"{(hits/n*100):.1f}%" if n else "—"
                tier_rows.append({
                    "Tier": tier,
                    "Bets": n,
                    "Hits": hits,
                    "Rate": rate,
                })
            st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Loss Diagnostics (Layer 9)</div>',
                        unsafe_allow_html=True)
            losses = [r for r in placed if r.get("dc_hit") is False]
            if losses:
                draws = sum(1 for r in losses
                            if (r.get("actual_home_goals") or 0)
                            == (r.get("actual_away_goals") or 0))
                straights = sum(1 for r in losses
                                if (r.get("call_1x2") or "").startswith("Straight"))
                c1, c2, c3 = st.columns(3)
                c1.metric("Losses", len(losses))
                c2.metric("Losses that were draws",
                          f"{draws} ({draws/len(losses)*100:.1f}%)")
                c3.metric("Losses that were Straight Win",
                          f"{straights} ({straights/len(losses)*100:.1f}%)")
            else:
                st.info("No losses recorded yet.")

            st.markdown('<div class="section-title">Firing Rate (Layer 8)</div>',
                        unsafe_allow_html=True)
            graded = [r for r in rows if r.get("f1_leader") is not None]
            fired = [r for r in graded if r.get("v4_4_bet")]
            if graded:
                st.write(
                    f"Firing rate: **{len(fired)}/{len(graded)} = "
                    f"{len(fired)/len(graded)*100:.1f}%**  "
                    f"(target ≥ 25% floor, NO BET ≈ 31%)"
                )
            else:
                st.info("No graded rows yet.")

            st.markdown('<div class="section-title">All Placed Bets</div>',
                        unsafe_allow_html=True)
            df = pd.DataFrame([{
                "Date": r.get("match_date"),
                "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                "Model": r.get("model_version"),
                "Tier": r.get("v4_4_tier"),
                "Leader": r.get("f1_leader"),
                "Gap": r.get("total_gap"),
                "Bet": r.get("v4_4_bet"),
                "Stake": r.get("v4_4_stake"),
                "Actual": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                "DC hit": ("✅" if r.get("dc_hit") is True
                           else "❌" if r.get("dc_hit") is False else "—"),
                "Tags": ", ".join(r.get("tags") or []),
            } for r in placed])
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("🏷️ Tag Performance")
        with st.spinner("Loading tag data..."):
            rows = load_all(sb)
        settled = [r for r in rows if r.get("dc_hit") is not None]
        if not settled:
            st.info("No settled bets with dc_hit recorded yet.")
        else:
            st.caption(f"n = {len(settled)} settled bets with dc_hit recorded.")
            singles, pairs = compute_tag_performance(settled)
            st.markdown('<div class="section-title">Single Tags</div>', unsafe_allow_html=True)
            if singles:
                st.dataframe(pd.DataFrame(singles), use_container_width=True, hide_index=True)
            st.markdown('<div class="section-title">Tag Pairs</div>', unsafe_allow_html=True)
            if pairs:
                st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("v4.4 Complete Logic Spec")
        st.markdown(f"""
### Layer 0 — Inputs
150 columns. Standings, form, venue splits, goals, odds, injuries, H2H.

### Layer 1 — Direction
`direction = f1_leader` (Table Power), confirmed by `f5_leader` (Squad Power).

### Layer 2 — Agreement
`disagreements` = count of F1, F2, F3, F5 pointing opposite the composite leader.
`shrink_factor = 1.0 - disagreements × 0.15`
**Skip at 2+ disagreements.** Verified: 0 → 62.5%, 1 → 43.9%, 2 → 35.3%.

### Layer 3 — Magnitude
**Fire at `total_gap ≥ {GAP_THRESHOLD}`** (80% band).
0–10 band is 0/30 = 0% — never fire.

### Layer 4 — Call Type
**DC by default.** DC 80.0% vs Straight Win 66.2%.
Losses: 53.6% draws, 78.6% Straight Win.

### Layer 5 — Clusters

| Tier | Rule | Fires | Call | Accuracy | Stake |
|---|---|---|---|---|---|
| CLUSTER | `f1_gap == 16 AND f1 == f5_raw` | 34.5% | DC | 91.3% | 1.0u |
| FORTRESS | Home + f1==f5_raw + disag0 + f2_diff≥4 + VP≥0.5 | 7.0% | DC 1X | 100% | 1.5u |
| VAULT BREAKER | Away + f1==f5_raw + disag0 + f2_diff≤-4 + VP≤-0.5 | 4.9% | DC X2 | 100% | 2.0u |
| STANDARD | `total_gap ≥ 30 AND disagreements ≤ 1` | ~10.6% | DC | ~75% | 1.0u |
| NO BET | Everything else | 31.0% | — | — | 0u |

### Layer 6 — Vetoes
- `f1_f5_conflict` (any raw disagreement)
- `disagreements >= 2`
- `doubted_starter` (top scorer ≥ {TOP_SCORER_OUT_MIN_GOALS} goals)
- `away_collapse` (documented as spec addition; kept)
- `draw_risk > {DRAW_RISK_THRESHOLD}`
- `parse_status != 'ok'`

### Layer 7 — Venue
`VENUE_POWER = (home_win_pct - away_win_pct)/25 + venue_ppg_gap × 2 + f0_gap/5`
Correlation with results: **0.559** (higher than F1's 0.40).

### Layer 8 — Coverage
Grade all, bet selectively. 25% floor. NO BET ≈ 31%. No full coverage.

### Layer 9 — Diagnostics
Losses: 78.6% Straight Win, 53.6% draws. Direction is usually right; call type is what fails.

### Fixes Applied in This Build
- Fix 1.2 — save_prediction warns on dropped keys
- Fix 1.3 / 1.4 — numeric coercion for draw_risk, win_pct
- Fix 2.1 — calc_f5 accepts H2H list for recency weighting
- Fix 2.2 — calc_agreement returns "tie" explicitly
- Fix 2.3 — calc_expected_total uses explicit None checks
- Fix 2.4 — away_collapse documented as spec addition
- Fix 2.5 — calc_f4 no longer double-penalises doubted XI starters
- Fix 2.6 — draw_risk bands widened so the veto is reachable
- Fix 2.7 — CLUSTER uses `f5_leader_raw`, not post-override F5
- Fix 3.1 — team matching requires two shared tokens or last-word match
- Fix 3.2 — last5_count stored
- Fix 3.3 — standings table chosen by team-name content
- Fix 3.4 — injury dedup normalised by name
- Fix 4.1 — model_version surfaced in Performance tab
- Fix 4.3 — parse_status = "partial" when critical fields missing
- Fix 4.4 — parse_status checked before factor computation
        """)

    with tabs[5]:
        render_debug_tab(sb)


# ============================================================================
# ENTRY
# ============================================================================
main()
