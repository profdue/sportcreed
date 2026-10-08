"""
Sportcreed — v6.0.

TIER1 (FULL 1.5u):
  RULE_A: standard_candidate + team_agreement_0
          → DC (follow f1_leader, default X2)
  RULE_B: f1_f5_override + home_leader
          → DC 1X (forced)

TIER2 (STANDARD 0.75u) — old TIER1 remainder:
  OLD_TIER1: gap_30_plus + home_leader
  OLD_TIER1: home_leader + standard_candidate

TIER3 (REDUCED 0.5u) — old TIER2 minus the RULE_B promotion:
  OLD_TIER2: 16 retained segments (see FORMULA_TIER3)

SKIP: everything else.

Direction: derived from the segment's own required tags.
  away_leader only  → DC X2
  home_leader only  → DC 1X
  otherwise         → DC 1X if f1_leader == "home", else DC X2
  explicit override → used for RULE_B (DC 1X)

Priority: TIER1 → TIER2 → TIER3, alphabetical within tier.

Persistence: writes v5_* (compat) and v6_* columns on every save.
Reads v6_* for display.
"""

import concurrent.futures
import json
import re
import sys
import unicodedata
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sportcreed",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.25rem; max-width: 1200px; }
    .team-header { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 14px; padding: 1.1rem 1.5rem; color: #fff; margin-bottom: 1rem; }
    .team-names { font-size: 1.5rem; font-weight: 800; margin: 0; }
    .team-meta { color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem; }
    .verdict-bet { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 6px solid #10b981; border-radius: 14px; padding: 1.25rem 1.5rem; margin: 1rem 0; }
    .verdict-nobet { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 6px solid #64748b; border-radius: 14px; padding: 1.25rem 1.5rem; margin: 1rem 0; }
    .verdict-label { font-size: 0.7rem; letter-spacing: 2px; font-weight: 700; color: #6ee7b7; text-transform: uppercase; }
    .verdict-label-grey { font-size: 0.7rem; letter-spacing: 2px; font-weight: 700; color: #94a3b8; text-transform: uppercase; }
    .verdict-pick { font-size: 2rem; font-weight: 800; color: #fff; margin: 0.3rem 0; line-height: 1.1; }
    .verdict-noedge { font-size: 1.3rem; font-weight: 700; color: #cbd5e1; margin: 0.3rem 0; }
    .verdict-detail { font-size: 0.92rem; color: #d1fae5; margin-top: 0.4rem; }
    .verdict-detail-grey { font-size: 0.88rem; color: #94a3b8; margin-top: 0.4rem; }
    .tag { display: inline-block; padding: 0.12rem 0.45rem; border-radius: 5px;
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.4px; margin-right: 0.3rem; margin-bottom:0.2rem; }
    .tag-gap { background: #1e3a8a; color: #bfdbfe; }
    .tag-side { background: #064e3b; color: #6ee7b7; }
    .tag-disagreement { background: #7c2d12; color: #fdba74; }
    .tag-venue { background: #4c1d95; color: #ddd6fe; }
    .tag-tier { background: #7c2d12; color: #fed7aa; }
    .tag-skip { background: #1e293b; color: #94a3b8; }
    .tag-pair { background: #065f46; color: #a7f3d0; }
    .factor-row { background: #0f172a; border-radius: 10px; padding: 0.65rem 1rem;
        display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem; }
    .factor-name { color: #cbd5e1; font-weight: 600; font-size: 0.88rem; }
    .factor-val { color: #3b82f6; font-weight: 800; font-size: 1rem; }
    .trigger-row { background: #1e293b; border-radius: 8px; padding: 0.55rem 1rem;
        margin-bottom: 0.35rem; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; }
    .trigger-on { background: #064e3b; color: #6ee7b7; }
    .trigger-off { background: #1e293b; color: #64748b; }
    .section-title { font-size: 0.78rem; font-weight: 700; color: #64748b;
        text-transform: uppercase; letter-spacing: 1.3px; margin: 1.25rem 0 0.6rem 0; }
    .stButton button { background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; font-weight: 700; border-radius: 10px; border: none; padding: 0.55rem 1.2rem; }
    .diag-ok { background: #064e3b; color: #6ee7b7; padding: 0.35rem 0.7rem;
        border-radius: 6px; font-family: monospace; font-size: 0.78rem; display: inline-block; margin-right: 0.5rem; }
    .diag-bad { background: #7f1d1d; color: #fecaca; padding: 0.35rem 0.7rem;
        border-radius: 6px; font-family: monospace; font-size: 0.78rem; display: inline-block; margin-right: 0.5rem; }
    .diag-warn { background: #78350f; color: #fde68a; padding: 0.35rem 0.7rem;
        border-radius: 6px; font-family: monospace; font-size: 0.78rem; display: inline-block; margin-right: 0.5rem; }
    .path-badge { display: inline-block; padding: 0.18rem 0.5rem; border-radius: 7px;
        font-size: 0.68rem; font-weight: 700; margin-left: 0.4rem; }
    .path-tier1 { background: #064e3b; color: #6ee7b7; }
    .path-tier2 { background: #065f46; color: #a7f3d0; }
    .path-tier3 { background: #7c2d12; color: #fed7aa; }
    .path-skip { background: #1e293b; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# V6 MAPPINGS — the single source of truth for stored v6 values
# ============================================================================

V6_STAKE_CLASS_TO_UNITS = {
    "FULL":     1.5,
    "STANDARD": 0.75,
    "REDUCED":  0.5,
    "NONE":     0.0,
}

V6_TIER_TO_PATH = {
    "NEW_TIER1": "TIER1",
    "NEW_TIER2": "TIER2",
    "NEW_TIER3": "TIER3",
    "SKIP":      "SKIP",
}

V6_PATH_TO_TIER = {
    "TIER1": "NEW_TIER1",
    "TIER2": "NEW_TIER2",
    "TIER3": "NEW_TIER3",
    "SKIP":  "SKIP",
}

V6_PATH_TO_STAKE_CLASS = {
    "TIER1": "FULL",
    "TIER2": "STANDARD",
    "TIER3": "REDUCED",
    "SKIP":  "NONE",
}

V6_RESULT_HIT   = "HIT_RECORDED"
V6_RESULT_MISS  = "MISS_RECORDED"
V6_RESULT_HYP   = "HIT_HYPOTHETICAL_DC_1X"
V6_RESULT_NONE  = "NO_RECORDED_BET"

V6_PARSE_OK       = "PARSE_OK"
V6_PARSE_NA       = "NOT_APPLICABLE"
V6_PARSE_PENDING  = "POLICY_PENDING_PARSE_REVIEW"

V6_ASSIGN_RULE_A    = "RULE_A"
V6_ASSIGN_RULE_B    = "RULE_B"
V6_ASSIGN_OLD_T1    = "OLD_TIER1"
V6_ASSIGN_OLD_T2    = "OLD_TIER2"
V6_ASSIGN_NO_BET    = "NO_BET"

V6_FORMULA_VERSION = "v6.0"


# ============================================================================
# RUNTIME SCHEMA DISCOVERY
# ============================================================================

_LIVE_SCHEMA_CACHE = {"columns": None, "table": None}


def _probe_live_schema(sb, table="matches_raw"):
    if sb is None:
        return None
    if _LIVE_SCHEMA_CACHE["columns"] is not None and _LIVE_SCHEMA_CACHE["table"] == table:
        return _LIVE_SCHEMA_CACHE["columns"]
    try:
        resp = sb.table(table).select("*").limit(1).execute()
        if resp.data and len(resp.data) > 0:
            cols = set(resp.data[0].keys())
            _LIVE_SCHEMA_CACHE["columns"] = cols
            _LIVE_SCHEMA_CACHE["table"] = table
            return cols
        _LIVE_SCHEMA_CACHE["columns"] = None
        return None
    except Exception as e:
        print(f"[schema] probe failed: {e}", file=sys.stderr)
        return None


def reload_schema_cache():
    _LIVE_SCHEMA_CACHE["columns"] = None
    _LIVE_SCHEMA_CACHE["table"] = None


def _get_table_columns(sb, table="matches_raw"):
    live = _probe_live_schema(sb, table)
    if live is not None:
        return live
    return MATCHES_RAW_COLUMNS


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
        return "SECRET KEY"
    if p.startswith("eyJhbGciOi"):
        return "legacy anon JWT"
    if p.startswith("eyJ"):
        return "JWT"
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
        issues.append("Key has leading/trailing whitespace")
    if key_info.get("has_inner_ws"):
        issues.append("Key contains whitespace in the middle")
    if key_info.get("has_newline"):
        issues.append("Key contains a newline")
    if key_info.get("has_quotes"):
        issues.append("Key appears to be wrapped in quotes")
    if key_info.get("length", 0) < 100:
        issues.append(f"Key length is only {key_info.get('length')}")
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
        elif raw_key.startswith("sb_publishable_"):
            st.info("Publishable key detected.")


def render_debug_tab(sb):
    st.subheader("🛠 Debug")

    st.markdown('<div class="section-title">1. Secrets</div>', unsafe_allow_html=True)
    url_info = _inspect_secret("SUPABASE_URL")
    key_info = _inspect_secret("SUPABASE_KEY")

    with st.expander("URL details", expanded=False):
        if url_info.get("present"):
            st.json({k: v for k, v in url_info.items() if k != "prefix"})
            st.code(f"prefix: {url_info['prefix']!r}")
        else:
            st.error("URL not found.")

    with st.expander("KEY details", expanded=False):
        if key_info.get("present"):
            st.json({k: v for k, v in key_info.items() if k not in ("prefix", "suffix")})
            st.code(f"prefix: {key_info['prefix']!r}\nsuffix: {key_info['suffix']!r}")
            st.write("**Key kind:**", _describe_key(key_info))
        else:
            st.error("KEY not found.")

    st.markdown('<div class="section-title">2. Client state</div>', unsafe_allow_html=True)
    if sb is None:
        st.error("Supabase client was NOT created.")
        return
    st.success("Supabase client created.")

    st.markdown('<div class="section-title">3. Live schema</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Reload schema cache", key="debug_reload_schema"):
            reload_schema_cache()
            st.success("Schema cache cleared.")
    with c2:
        if st.button("🔎 Discover live schema", key="debug_probe_schema"):
            reload_schema_cache()
            live = _probe_live_schema(sb, "matches_raw")
            if live is None:
                st.warning("Could not discover live schema.")
            else:
                st.success(f"Live schema has {len(live)} columns.")
                missing = sorted(MATCHES_RAW_COLUMNS - live)
                if missing:
                    st.warning(f"{len(missing)} declared columns missing:")
                    st.code("\n".join(missing))
                else:
                    st.success("All declared columns exist.")

    st.markdown('<div class="section-title">4. Ping</div>', unsafe_allow_html=True)
    if st.button("🏓 Ping Supabase", key="debug_ping"):
        try:
            resp = sb.table("matches_raw").select("id").limit(1).execute()
            st.success(f"Ping OK. Returned {len(resp.data)} rows.")
        except Exception as e:
            st.error(f"Ping failed: {e}")

    st.markdown('<div class="section-title">5. Row count</div>', unsafe_allow_html=True)
    if st.button("🧮 Count rows", key="debug_count"):
        try:
            resp = sb.table("matches_raw").select("id").execute()
            st.success(f"matches_raw has {len(resp.data)} row(s).")
        except Exception as e:
            st.error(f"Count failed: {e}")

    st.markdown('<div class="section-title">6. v6 stored state</div>',
                unsafe_allow_html=True)
    st.caption("Row counts by v6_tier, v6_action, v6_stake_class — reads what's actually in the DB.")
    if st.button("📊 Show v6 counts", key="debug_v6_counts"):
        try:
            resp = (sb.table("matches_raw")
                    .select("v6_tier, v6_action, v6_stake_class")
                    .eq("v6_formula_version", V6_FORMULA_VERSION)
                    .execute())
            rows = resp.data or []
            if not rows:
                st.info("No v6.0 rows found.")
            else:
                from collections import Counter
                c = Counter((r.get("v6_tier"), r.get("v6_action"), r.get("v6_stake_class"))
                            for r in rows)
                df = pd.DataFrame(
                    [{"v6_tier": k[0], "v6_action": k[1], "v6_stake_class": k[2], "n": v}
                     for k, v in sorted(c.items(), key=lambda x: -x[1])]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query failed: {e}")

    st.markdown('<div class="section-title">7. Test insert / cleanup</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧪 Insert test row", key="debug_insert"):
            try:
                payload = {
                    "match_date": "2099-12-31",
                    "home_team": "__DEBUG__ Home",
                    "away_team": "__DEBUG__ Away",
                    "league_name": "DEBUG",
                }
                resp = sb.table("matches_raw").insert(payload).execute()
                st.success(f"Inserted. id = {resp.data[0].get('id') if resp.data else 'unknown'}")
            except Exception as e:
                st.error(f"Insert failed: {e}")
    with c2:
        if st.button("🧹 Delete test rows", key="debug_cleanup"):
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
# CONSTANTS
# ============================================================================
ALPHA = 2
PRIOR_FORM = 0.4
PRIOR_H2H = 1.0 / 3.0

FORTRESS_HOME_WIN_PCT_MIN = 50.0
FORTRESS_AWAY_WIN_PCT_MAX = 25.0
FORTRESS_F0_HOME_MIN = 15.0

AWAY_FORTRESS_AWAY_WIN_PCT_MIN = 50.0
AWAY_FORTRESS_HOME_WIN_PCT_MAX = 25.0
AWAY_FORTRESS_F0_AWAY_MIN = 15.0

CLUSTER_TOTAL_GAP_MAX = 10.0
CLUSTER_DRAW_RISK_MIN = 0.30
CLUSTER_POS_GAP_MAX = 3

STANDARD_POS_GAP_MIN = 5

F1_CAP_THRESHOLD = 16.0

DECISION_VERSION = "v6.0-formula"


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
    "id", "created_at", "settled_at",
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
    "home_last5_count", "away_last5_count",
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
    "call_1x2", "call_ou", "call_btts", "expected_total",
    "no_bet_reason_1x2", "no_bet_reason_ou",
    "venue_ppg_gap", "venue_ppg_gap_home", "venue_ppg_gap_away",
    "venue_power",
    "f0_home", "f0_away", "f0_gap",
    "draw_risk",
    "tags",
    "decision_version",
    "v5_decision_path", "v5_decision", "v5_bet", "v5_stake",
    "v5_combo_pair_count", "v5_combo_pairs_fired", "v5_core_fired",
    "v5_skip_reason",
    "counterfactual_call", "counterfactual_odds", "counterfactual_won",
    "actual_home_goals", "actual_away_goals",
    "actual_possession_home", "actual_xg_home", "actual_xg_away",
    "actual_corners_home", "actual_corners_away", "actual_total_goals",
    "dc_hit", "is_correct_1x2", "is_correct_ou", "is_correct_btts",
    # v6.0 columns
    "v6_formula_version",
    "v6_tier",
    "v6_assignment_reason",
    "v6_action",
    "v6_stake_class",
    "v6_parse_policy_status",
    "v6_result_basis",
}


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
            raise RuntimeError("beautifulsoup4 required.")
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
        a_tokens = {w for w in a_words if len(w) > 3}
        b_tokens = {w for w in b_words if len(w) > 3}
        shared = a_tokens & b_tokens
        if len(shared) >= 2:
            return True
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
        main_table = None
        candidates = []
        for table in self.soup.select("table"):
            headers = [th.get_text(strip=True).lower() for th in table.select("th")]
            if any(h in ("pts", "p") for h in headers) and any("team" in h for h in headers):
                candidates.append(table)

        if candidates:
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

        won_idx = None
        for i, h in enumerate(headers):
            if h in ("w", "won", "wins"):
                won_idx = i
                break
        played_idx = None
        for i, h in enumerate(headers):
            if h in ("p", "pl", "played", "mp", "games"):
                played_idx = i
                break
        if played_idx is None:
            played_idx = 2

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

    if top_scorer and top_scorer in injured and not _xi_contains(top_scorer):
        f4 -= 5
    if top_assister and top_assister in injured and not _xi_contains(top_assister):
        f4 -= 4

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


def calc_f5(h2h_home_wins, h2h_away_wins, h2h_total, h2h_list=None):
    f5_h = (h2h_home_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    f5_a = (h2h_away_wins + PRIOR_H2H) / (h2h_total + 1) * 10
    if h2h_total < 4:
        f5_h = 3.33 + (f5_h - 3.33) * (h2h_total / 4.0)
        f5_a = 3.33 + (f5_a - 3.33) * (h2h_total / 4.0)
    return f5_h, f5_a


def calc_f6(h_avg, a_avg):
    if (h_avg or 0) > (a_avg or 0):
        return 11, 7
    elif (h_avg or 0) < (a_avg or 0):
        return 7, 11
    return 8, 8


def apply_override(f1_home, f1_away, f5_home, f5_away):
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


def calc_venue_power(row):
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


def calc_draw_risk(expected_total, total_gap):
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


def calc_agreement(f1_home, f1_away,
                   f2_home, f2_away,
                   f3_home, f3_away,
                   f5_home, f5_away,
                   home_total, away_total):
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
# PREDICTION PIPELINE
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
        "model_version": DECISION_VERSION,
        "no_bet_reason_1x2": reason_1x2, "no_bet_reason_ou": reason_ou,
        "venue_ppg_gap": venue_ppg_gap,
        "venue_ppg_gap_home": home_ppg,
        "venue_ppg_gap_away": away_ppg,
        "venue_power": None,
        "f0_home": None, "f0_away": None, "f0_gap": None,
        "factor_map": {},
    }


def predict_v5(row):
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
        "call_1x2": "see_v6",
        "call_ou": call_ou,
        "expected_total": round(expected, 2),
        "draw_risk": round(draw_risk, 3),
        "model_version": DECISION_VERSION,
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
# TAGS
# ============================================================================
def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _has_tag(tags, name):
    return name in (tags or [])


def _is_f1_cap(pred):
    return (pred.get("f1_gap") or 0) >= F1_CAP_THRESHOLD


def _gap_bucket(pred):
    g = pred.get("total_gap") or 0
    if g >= 30:
        return "gap_30_plus"
    if g >= 20:
        return "gap_20_29"
    if g >= 10:
        return "gap_10_19"
    return "gap_under_10"


def check_fortress(row, pred):
    hw = _safe_float(row.get("home_home_win_pct"))
    aw = _safe_float(row.get("away_away_win_pct"))
    f0h = _safe_float(pred.get("f0_home"))
    ht = pred.get("home_total")
    at = pred.get("away_total")
    if hw is None or aw is None or f0h is None or ht is None or at is None:
        return False
    return (hw >= FORTRESS_HOME_WIN_PCT_MIN
            and aw <= FORTRESS_AWAY_WIN_PCT_MAX
            and f0h >= FORTRESS_F0_HOME_MIN
            and ht > at)


def check_away_fortress(row, pred):
    aw = _safe_float(row.get("away_away_win_pct"))
    hw = _safe_float(row.get("home_home_win_pct"))
    f0a = _safe_float(pred.get("f0_away"))
    ht = pred.get("home_total")
    at = pred.get("away_total")
    if aw is None or hw is None or f0a is None or ht is None or at is None:
        return False
    return (aw >= AWAY_FORTRESS_AWAY_WIN_PCT_MIN
            and hw <= AWAY_FORTRESS_HOME_WIN_PCT_MAX
            and f0a >= AWAY_FORTRESS_F0_AWAY_MIN
            and at > ht)


def check_cluster(row, pred):
    gap = _safe_float(pred.get("total_gap"))
    dr = _safe_float(pred.get("draw_risk"))
    hp = _safe_float(row.get("home_pos"))
    ap = _safe_float(row.get("away_pos"))
    if gap is None or dr is None or hp is None or ap is None:
        return False
    return (gap <= CLUSTER_TOTAL_GAP_MAX
            and dr >= CLUSTER_DRAW_RISK_MIN
            and abs(hp - ap) <= CLUSTER_POS_GAP_MAX)


def check_standard(row, pred):
    hp = _safe_float(row.get("home_pos"))
    ap = _safe_float(row.get("away_pos"))
    if hp is None or ap is None:
        return False
    if abs(hp - ap) < STANDARD_POS_GAP_MIN:
        return False
    if pred.get("f1_f5_conflict"):
        return False
    return True


# ============================================================================
# THE FORMULA — v6.0
# ============================================================================
# Each entry: (name, required_tags, direction_override_or_None, assignment_reason)
# direction_override: if set, the call is forced.
# assignment_reason: the string written to v6_assignment_reason when this fires.

FORMULA_TIER1 = [
    ("standard_candidate+team_agreement_0",
     frozenset({"standard_candidate", "team_agreement_0"}),
     None, V6_ASSIGN_RULE_A),
    ("f1_f5_override+home_leader",
     frozenset({"f1_f5_override", "home_leader"}),
     "DC 1X", V6_ASSIGN_RULE_B),
]

FORMULA_TIER2 = [
    ("gap_30_plus+home_leader",
     frozenset({"gap_30_plus", "home_leader"}),
     None, V6_ASSIGN_OLD_T1),
    ("home_leader+standard_candidate",
     frozenset({"home_leader", "standard_candidate"}),
     None, V6_ASSIGN_OLD_T1),
]

FORMULA_TIER3 = [
    ("away_leader+team_disagreement_2plus",
     frozenset({"away_leader", "team_disagreement_2plus"}),
     None, V6_ASSIGN_OLD_T2),
    ("away_leader+tier_core_a",
     frozenset({"away_leader", "tier_core_a"}),
     None, V6_ASSIGN_OLD_T2),
    ("away_leader+venue_power_neg",
     frozenset({"away_leader", "venue_power_neg"}),
     None, V6_ASSIGN_OLD_T2),
    ("f1_cap+team_disagreement_2plus",
     frozenset({"f1_cap", "team_disagreement_2plus"}),
     None, V6_ASSIGN_OLD_T2),
    ("f1_cap+tier_core_a",
     frozenset({"f1_cap", "tier_core_a"}),
     None, V6_ASSIGN_OLD_T2),
    ("f1_f5_conflict+venue_incomplete",
     frozenset({"f1_f5_conflict", "venue_incomplete"}),
     None, V6_ASSIGN_OLD_T2),
    ("gap_10_19+standard_candidate",
     frozenset({"gap_10_19", "standard_candidate"}),
     None, V6_ASSIGN_OLD_T2),
    ("gap_10_19+venue_incomplete",
     frozenset({"gap_10_19", "venue_incomplete"}),
     None, V6_ASSIGN_OLD_T2),
    ("gap_30_plus+tier_core_a",
     frozenset({"gap_30_plus", "tier_core_a"}),
     None, V6_ASSIGN_OLD_T2),
    ("gap_30_plus+venue_power_neg",
     frozenset({"gap_30_plus", "venue_power_neg"}),
     None, V6_ASSIGN_OLD_T2),
    ("home_leader+venue_power_pos",
     frozenset({"home_leader", "venue_power_pos"}),
     None, V6_ASSIGN_OLD_T2),
    ("team_agreement_0+tier_core_a",
     frozenset({"team_agreement_0", "tier_core_a"}),
     None, V6_ASSIGN_OLD_T2),
    ("team_agreement_0+venue_power_neg",
     frozenset({"team_agreement_0", "venue_power_neg"}),
     None, V6_ASSIGN_OLD_T2),
    ("team_disagreement_2plus+tier_f1_gap_high",
     frozenset({"team_disagreement_2plus", "tier_f1_gap_high"}),
     None, V6_ASSIGN_OLD_T2),
    ("tier_core_a+venue_incomplete",
     frozenset({"tier_core_a", "venue_incomplete"}),
     None, V6_ASSIGN_OLD_T2),
    ("tier_core_a+venue_power_neg",
     frozenset({"tier_core_a", "venue_power_neg"}),
     None, V6_ASSIGN_OLD_T2),
]


def _formula_direction(required, leader):
    has_away = "away_leader" in required
    has_home = "home_leader" in required
    if has_away and not has_home:
        return "DC X2"
    if has_home and not has_away:
        return "DC 1X"
    return "DC 1X" if leader == "home" else "DC X2"


def _resolve_call(required, override, leader):
    if override:
        return override
    return _formula_direction(required, leader)


def _formula_fired(tags, leader):
    """Return (tier, name, call, reason) for the highest-priority firing segment.
    TIER1 → TIER2 → TIER3 short-circuit."""
    tagset = set(tags or [])
    for name, required, override, reason in FORMULA_TIER1:
        if required <= tagset:
            return 1, name, _resolve_call(required, override, leader), reason
    for name, required, override, reason in FORMULA_TIER2:
        if required <= tagset:
            return 2, name, _resolve_call(required, override, leader), reason
    for name, required, override, reason in FORMULA_TIER3:
        if required <= tagset:
            return 3, name, _resolve_call(required, override, leader), reason
    return None


# ============================================================================
# DECISION LAYER — v6.0
# ============================================================================
def decide_v6(row, pred, parse_status):
    """Return a dict with v6 fields (tier, action, stake_class, etc.)."""
    empty_result = {
        "v6_formula_version": V6_FORMULA_VERSION,
        "v6_tier": "SKIP",
        "v6_assignment_reason": V6_ASSIGN_NO_BET,
        "v6_action": "NO BET",
        "v6_stake_class": "NONE",
        "v6_parse_policy_status": V6_PARSE_NA,
        "v6_result_basis": V6_RESULT_NONE,
        # legacy v5 fields (compat)
        "v5_decision_path": "SKIP",
        "v5_decision": "SKIP",
        "v5_bet": None,
        "v5_stake": 0.0,
        "v5_skip_reason": "no_segment_fired",
        "decision_version": DECISION_VERSION,
    }

    # parse gate
    if parse_status and parse_status != "ok":
        out = dict(empty_result)
        out["v5_skip_reason"] = "parse_status_not_ok"
        out["v6_parse_policy_status"] = V6_PARSE_PENDING
        return out

    # empty prediction gate
    home_total = pred.get("home_total")
    if home_total is None or home_total == 0:
        out = dict(empty_result)
        out["v5_skip_reason"] = "home_total_empty"
        return out

    tags = pred.get("tags") or []
    leader = pred.get("f1_leader")

    fired = _formula_fired(tags, leader)
    if not fired:
        return empty_result

    tier, name, call, reason = fired
    path = f"TIER{tier}"
    tier_label = V6_PATH_TO_TIER[path]
    stake_class = V6_PATH_TO_STAKE_CLASS[path]
    stake_value = V6_STAKE_CLASS_TO_UNITS[stake_class]

    return {
        "v6_formula_version": V6_FORMULA_VERSION,
        "v6_tier": tier_label,
        "v6_assignment_reason": reason,
        "v6_action": "BET",
        "v6_stake_class": stake_class,
        "v6_parse_policy_status": V6_PARSE_NA,
        "v6_result_basis": V6_RESULT_NONE,  # becomes HIT/MISS on settle
        # legacy
        "v5_decision_path": path,
        "v5_decision": "BET",
        "v5_bet": call,
        "v5_stake": stake_value,
        "v5_skip_reason": name,
        "decision_version": DECISION_VERSION,
    }


# ============================================================================
# TAGS COMPUTATION
# ============================================================================
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

    if check_cluster(row, pred):
        tags.append("tier_cluster")
    if _is_f1_cap(pred):
        tags.append("f1_cap")

    if check_fortress(row, pred):
        tags.append("fortress_candidate")
    if check_away_fortress(row, pred):
        tags.append("away_fortress_candidate")
    if check_standard(row, pred):
        tags.append("standard_candidate")

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

    if decision == "SKIP" and skip_reason:
        tags.append(f"skip_{skip_reason.split('_')[0]}")

    return tags


# ============================================================================
# FULL PREDICTION
# ============================================================================
def predict_v6_full(row):
    base = predict_v5(row)

    tags = compute_tags(row, base, None, None, None)
    base["tags"] = tags

    parse_status = row.get("parse_status")
    decision_fields = decide_v6(row, base, parse_status)

    # Recompute tags with the decision context
    final_tags = compute_tags(
        row,
        base,
        decision_fields.get("v6_tier"),
        decision_fields.get("v5_decision"),
        decision_fields.get("v5_skip_reason"),
    )
    base["tags"] = final_tags

    # Merge v6/v5 decision fields into the record
    base.update(decision_fields)

    # Counterfactual (unchanged)
    leader = base.get("f1_leader")
    if leader == "home":
        base["counterfactual_call"] = "DC 1X"
    else:
        base["counterfactual_call"] = "DC X2"

    odds = row.get("odds") or {}
    if base["counterfactual_call"] == "DC 1X":
        base["counterfactual_odds"] = odds.get("dc_1x")
    else:
        base["counterfactual_odds"] = odds.get("dc_x2")

    return base


# ============================================================================
# DB HELPERS
# ============================================================================
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
        print(f"[{context}] dropped keys not in live schema: {dropped}",
              file=sys.stderr)
    return clean


def upsert_match(sb, record):
    if sb is None:
        return False, "no client"
    try:
        real_columns = _get_table_columns(sb, "matches_raw")
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
    real_columns = _get_table_columns(sb, "matches_raw")
    clean = _filter_columns_for_db(result, real_columns, context="save_prediction")

    for bad_key in ("leader", "factor_map"):
        clean.pop(bad_key, None)

    if not clean:
        return False, "no writable keys after schema filter"

    try:
        sb.table("matches_raw").update(clean).eq("id", match_id).execute()
        return True, "saved"
    except Exception as e:
        msg = str(e)
        m = re.search(r"Could not find the '([^']+)' column", msg)
        if m:
            bad_col = m.group(1)
            reload_schema_cache()
            return False, (
                f"Live schema rejected column '{bad_col}'. "
                f"Run migration SQL, then Reload schema cache. Error: {msg}"
            )
        return False, msg


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
            st.warning(f"Supabase query exceeded {timeout_seconds}s.")
            return []
        except Exception as e:
            st.error(f"Load failed: {e}")
            return []


def update_audit(sb, match_id, hg, ag, call_1x2, bet_v5=None, counterfactual_call=None,
                 v6_bet=None):
    """Set actual scores, dc_hit, and v6_result_basis."""
    if sb is None:
        return False, "no client"

    actual = "Home" if hg > ag else ("Away" if hg < ag else "Draw")

    bet = v6_bet or bet_v5
    payload = {"actual_home_goals": hg, "actual_away_goals": ag}

    real_columns = _get_table_columns(sb, "matches_raw") or set()

    # dc_hit for the recorded bet
    dc_hit = None
    if bet == "DC 1X":
        dc_hit = actual in ("Home", "Draw")
    elif bet == "DC X2":
        dc_hit = actual in ("Away", "Draw")
    if "dc_hit" in real_columns:
        payload["dc_hit"] = dc_hit

    # v6_result_basis reflects the recorded outcome (or NO_RECORDED_BET)
    if "v6_result_basis" in real_columns:
        if bet is None:
            payload["v6_result_basis"] = V6_RESULT_NONE
        elif dc_hit is True:
            payload["v6_result_basis"] = V6_RESULT_HIT
        elif dc_hit is False:
            payload["v6_result_basis"] = V6_RESULT_MISS
        else:
            payload["v6_result_basis"] = V6_RESULT_NONE

    # counterfactual
    cf_won = None
    if counterfactual_call == "DC 1X":
        cf_won = actual in ("Home", "Draw")
    elif counterfactual_call == "DC X2":
        cf_won = actual in ("Away", "Draw")
    if "counterfactual_won" in real_columns:
        payload["counterfactual_won"] = cf_won

    # legacy is_correct_1x2
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
    if "is_correct_1x2" in real_columns:
        payload["is_correct_1x2"] = is_correct

    if "settled_at" in real_columns:
        payload["settled_at"] = datetime.utcnow().isoformat()

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
    elif label.startswith("tier") or label.startswith("cluster"):
        cls += " tag-tier"
    elif label.startswith("skip"):
        cls += " tag-skip"
    elif "+" in label:
        cls += " tag-pair"
    return f'<span class="{cls}">{label}</span>'


def render_tags(tags):
    return "".join(render_tag(t) for t in (tags or []))


def render_path_badge(path):
    cls = {
        "NEW_TIER1": "path-tier1",
        "NEW_TIER2": "path-tier2",
        "NEW_TIER3": "path-tier3",
        "SKIP": "path-skip",
    }.get(path, "path-skip")
    return f'<span class="path-badge {cls}">{path}</span>'


def _fire_summary(pred):
    tags = set(pred.get("tags") or [])
    leader = pred.get("f1_leader")
    fired = []
    for tier_list, tier_num in ((FORMULA_TIER1, 1), (FORMULA_TIER2, 2), (FORMULA_TIER3, 3)):
        for name, required, override, reason in tier_list:
            if required <= tags:
                fired.append({
                    "tier": tier_num,
                    "name": name,
                    "call": _resolve_call(required, override, leader),
                    "reason": reason,
                })
    return fired


def render_verdict_v6(result):
    """Reads v6 fields, renders the verdict card."""
    action = result.get("v6_action")
    tier = result.get("v6_tier")
    bet = result.get("v5_bet")
    stake_class = result.get("v6_stake_class")
    stake = V6_STAKE_CLASS_TO_UNITS.get(stake_class, 0.0)
    segment = result.get("v5_skip_reason")
    f1_gap = result.get("f1_gap") or 0
    leader = result.get("f1_leader") or "—"
    total_gap = result.get("total_gap") or 0

    if action == "BET":
        st.markdown(f"""
        <div class="verdict-bet">
            <div class="verdict-label">⭐ SPORTCREED PREDICTION &nbsp; {render_path_badge(tier)}</div>
            <div class="verdict-pick">{bet} &nbsp;<span style="font-size:1.1rem; opacity:0.7; font-weight:600;">· {stake}u ({stake_class})</span></div>
            <div class="verdict-detail">
                Fires on <code style="background:#022c22; color:#6ee7b7; padding:2px 6px; border-radius:4px;">{segment or '—'}</code>
                &nbsp;·&nbsp; F1 gap <strong>{f1_gap:.1f}</strong>
                &nbsp;·&nbsp; Leader <strong>{leader}</strong>
                &nbsp;·&nbsp; Total gap <strong>{total_gap:.1f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-nobet">
            <div class="verdict-label-grey">SPORTCREED PREDICTION &nbsp; {render_path_badge(tier or 'SKIP')}</div>
            <div class="verdict-noedge">NO BET — {segment}</div>
            <div class="verdict-detail-grey">
                F1 gap {f1_gap:.1f} · Leader {leader} · Total gap {total_gap:.1f}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_ou_verdict(result):
    call = result.get("call_ou", "")
    expected = result.get("expected_total", 0)
    if call == "No Bet":
        st.info(f"🟡 O/U 2.5 · **NO BET** · expected {expected:.2f}")
    else:
        st.success(f"🟢 O/U 2.5 · **{call}** · expected {expected:.2f}")


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


def render_firing_segments(pred):
    fired = _fire_summary(pred)
    if not fired:
        st.info("No segment fired. SKIP.")
        return

    winner = fired[0]
    others = fired[1:]

    tier_label = {
        1: "NEW_TIER1",
        2: "NEW_TIER2",
        3: "NEW_TIER3",
    }[winner["tier"]]

    st.markdown(f"""
    <div style="background:#064e3b; border-left:4px solid #10b981; border-radius:10px;
                padding:0.85rem 1.1rem; margin-bottom:0.6rem;">
        <div style="font-size:0.68rem; font-weight:700; letter-spacing:1.5px;
                    color:#6ee7b7; text-transform:uppercase; margin-bottom:0.2rem;">
            ⭐ WINNER · {tier_label}
        </div>
        <div style="color:#fff; font-weight:700; font-size:0.95rem;">
            {winner['name']} → {winner['call']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if others:
        with st.expander(f"Also fired ({len(others)})", expanded=False):
            for f in others:
                tl = {1: "NEW_TIER1", 2: "NEW_TIER2", 3: "NEW_TIER3"}[f["tier"]]
                render_trigger(f"{tl} · {f['name']} → {f['call']}", True)


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
    st.title("⚽ Sportcreed")

    with st.expander("🔍 Diagnostics", expanded=False):
        render_diagnostic_banner()

    sb, diag = get_supabase()
    if sb is None:
        st.error(f"Supabase client could not be created: {diag.get('error')}")
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
        text = st.text_area("Paste Sportsgambler HTML",
                            height=200, key="html_input",
                            label_visibility="collapsed")

        if st.button("⚽ Parse, Predict & Save", type="primary"):
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
                with st.spinner("Running v6.0..."):
                    result = predict_v6_full(parsed)
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
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if ok and save_ok:
                    st.success("✅ Saved.")
                elif ok and not save_ok:
                    st.warning(f"⚠️ Saved but prediction failed: {save_msg}")
                else:
                    st.error(f"❌ Save failed: {save_msg}")

                render_verdict_v6(result)
                render_firing_segments(result)
                render_ou_verdict(result)

                with st.expander("Factor breakdown", expanded=False):
                    render_factor_row("F1 Table Power", result["f1_home"], result["f1_away"], "(18)")
                    render_factor_row("F2 Current Form", result["f2_home"], result["f2_away"], "(25)")
                    render_factor_row("F3 Venue Form", result["f3_home"], result["f3_away"], "(15)")
                    render_factor_row("F4 Availability", result["f4_home"], result["f4_away"], "(20)")
                    render_factor_row("F5 Squad Power", result["f5_home"], result["f5_away"], "(10)")
                    render_factor_row("F6 H2H", result["f6_home"], result["f6_away"], "(11)")

                with st.expander("Tags", expanded=False):
                    st.markdown(render_tags(result.get("tags", [])), unsafe_allow_html=True)

                st.caption("Enter the final score in the Pending tab after the match.")

    with tabs[1]:
        st.subheader("⏳ Pending Matches")
        with st.spinner("Loading..."):
            rows = load_all(sb)
        pending = [r for r in rows
                   if r.get("actual_home_goals") is None
                   and r.get("v6_formula_version") == V6_FORMULA_VERSION]
        if not pending:
            st.success("No pending v6.0 matches.")
        else:
            for r in pending:
                match_id = r["id"]
                tier = r.get("v6_tier") or "SKIP"
                action = r.get("v6_action") or "NO BET"
                bet = r.get("v5_bet") or "—"
                stake_class = r.get("v6_stake_class") or "NONE"
                stake = V6_STAKE_CLASS_TO_UNITS.get(stake_class, 0.0)
                segment = r.get("v5_skip_reason") or "—"
                header = (f"{r.get('match_date','')} · "
                          f"{r.get('home_team','')} vs {r.get('away_team','')} · "
                          f"{tier} · {bet}")
                with st.expander(header):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Tier", tier)
                    c2.metric("Bet", bet)
                    c3.metric("Stake", f"{stake}u")
                    c4.metric("Action", action)
                    st.caption(f"Segment: `{segment}`")
                    col1, col2, col3 = st.columns([1, 1, 2])
                    hg = col1.number_input("Home", 0, 15, 0, key=f"hg_{match_id}")
                    ag = col2.number_input("Away", 0, 15, 0, key=f"ag_{match_id}")
                    if col3.button("📝 Save", key=f"save_{match_id}"):
                        ok, msg = update_audit(
                            sb, match_id, hg, ag,
                            r.get("call_1x2") or "",
                            bet_v5=r.get("v5_bet"),
                            counterfactual_call=r.get("counterfactual_call"),
                            v6_bet=r.get("v5_bet"),
                        )
                        if ok:
                            st.success("Recorded.")
                            st.rerun()
                        else:
                            st.error(msg)

    with tabs[2]:
        st.subheader("📊 Performance")
        with st.spinner("Loading..."):
            rows = load_all(sb)

        versions = sorted({r.get("v6_formula_version") for r in rows if r.get("v6_formula_version")})
        if versions:
            selected = st.multiselect("Formula versions", versions, default=versions)
            rows = [r for r in rows if r.get("v6_formula_version") in selected]

        settled = [r for r in rows
                   if r.get("actual_home_goals") is not None
                   and r.get("actual_away_goals") is not None]

        if not settled:
            st.info("No settled matches yet.")
        else:
            # By v6_tier
            path_rows = []
            for tier in ["NEW_TIER1", "NEW_TIER2", "NEW_TIER3", "SKIP"]:
                subset = [r for r in settled if r.get("v6_tier") == tier]
                scored = [r for r in subset if r.get("dc_hit") is not None]
                hits = sum(1 for r in scored if r.get("dc_hit") is True)
                n = len(scored)
                rate = f"{(hits/n*100):.1f}%" if n else "—"
                path_rows.append({"Tier": tier, "Settled": n, "Hits": hits, "Rate": rate})
            st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

            # By assignment reason
            reason_rows = []
            for reason in [V6_ASSIGN_RULE_A, V6_ASSIGN_RULE_B,
                           V6_ASSIGN_OLD_T1, V6_ASSIGN_OLD_T2,
                           V6_ASSIGN_NO_BET]:
                subset = [r for r in settled if r.get("v6_assignment_reason") == reason]
                scored = [r for r in subset if r.get("dc_hit") is not None]
                hits = sum(1 for r in scored if r.get("dc_hit") is True)
                n = len(scored)
                rate = f"{(hits/n*100):.1f}%" if n else "—"
                reason_rows.append({"Reason": reason, "Settled": n, "Hits": hits, "Rate": rate})
            st.markdown('<div class="section-title">By assignment reason</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)

            # All placed bets
            placed = [r for r in settled if r.get("v6_action") == "BET"]
            if placed:
                st.markdown('<div class="section-title">All placed bets</div>',
                            unsafe_allow_html=True)
                df = pd.DataFrame([{
                    "Date": r.get("match_date"),
                    "Match": f"{r.get('home_team')} vs {r.get('away_team')}",
                    "Version": r.get("v6_formula_version"),
                    "Tier": r.get("v6_tier"),
                    "Bet": r.get("v5_bet"),
                    "Stake": V6_STAKE_CLASS_TO_UNITS.get(r.get("v6_stake_class"), 0.0),
                    "Reason": r.get("v6_assignment_reason"),
                    "Segment": r.get("v5_skip_reason"),
                    "Actual": f"{r.get('actual_home_goals')}-{r.get('actual_away_goals')}",
                    "Hit": ("✅" if r.get("dc_hit") is True
                            else "❌" if r.get("dc_hit") is False else "—"),
                } for r in placed])
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("🏷️ Tag Performance")
        with st.spinner("Loading..."):
            rows = load_all(sb)
        settled = [r for r in rows if r.get("dc_hit") is not None]
        if not settled:
            st.info("No settled bets with dc_hit recorded yet.")
        else:
            singles, pairs = compute_tag_performance(settled)
            st.markdown("**Single tags**")
            if singles:
                st.dataframe(pd.DataFrame(singles), use_container_width=True, hide_index=True)
            st.markdown("**Tag pairs**")
            if pairs:
                st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

    with tabs[4]:
        st.subheader("Sportcreed v6.0 — The Formula")
        st.markdown("""
### NEW TIER1 — full stake (1.5u)

| Rule | Segment | Direction |
|---|---|---|
| RULE_A | `standard_candidate + team_agreement_0` | follow `f1_leader`, default DC X2 |
| RULE_B | `f1_f5_override + home_leader` | DC 1X (forced) |

### NEW TIER2 — standard stake (0.75u)

| Reason | Segment | Direction |
|---|---|---|
| OLD_TIER1 | `gap_30_plus + home_leader` | DC 1X |
| OLD_TIER1 | `home_leader + standard_candidate` | DC 1X |

### NEW TIER3 — reduced stake (0.5u)

| Reason | Segment | Direction |
|---|---|---|
| OLD_TIER2 | `away_leader + team_disagreement_2plus` | DC X2 |
| OLD_TIER2 | `away_leader + tier_core_a` | DC X2 |
| OLD_TIER2 | `away_leader + venue_power_neg` | DC X2 |
| OLD_TIER2 | `f1_cap + team_disagreement_2plus` | DC 1X |
| OLD_TIER2 | `f1_cap + tier_core_a` | DC 1X |
| OLD_TIER2 | `f1_f5_conflict + venue_incomplete` | follow `f1_leader` |
| OLD_TIER2 | `gap_10_19 + standard_candidate` | DC 1X |
| OLD_TIER2 | `gap_10_19 + venue_incomplete` | follow `f1_leader` |
| OLD_TIER2 | `gap_30_plus + tier_core_a` | DC 1X |
| OLD_TIER2 | `gap_30_plus + venue_power_neg` | follow `f1_leader` |
| OLD_TIER2 | `home_leader + venue_power_pos` | DC 1X |
| OLD_TIER2 | `team_agreement_0 + tier_core_a` | DC 1X |
| OLD_TIER2 | `team_agreement_0 + venue_power_neg` | follow `f1_leader` |
| OLD_TIER2 | `team_disagreement_2plus + tier_f1_gap_high` | follow `f1_leader` |
| OLD_TIER2 | `tier_core_a + venue_incomplete` | follow `f1_leader` |
| OLD_TIER2 | `tier_core_a + venue_power_neg` | follow `f1_leader` |

### Direction rule

Derived from the segment's own required tags:
- `away_leader` only → **DC X2**
- `home_leader` only → **DC 1X**
- Neither or both → **DC 1X** if `f1_leader == "home"`, else **DC X2**

### Priority

TIER1 → TIER2 → TIER3, alphabetical within tier.

### Stakes

- NEW_TIER1 → **FULL** (1.5u)
- NEW_TIER2 → **STANDARD** (0.75u)
- NEW_TIER3 → **REDUCED** (0.5u)
- SKIP → **NONE** (0.0u)

### Persistence

Every save writes both:
- `v6_*` columns — the authoritative v6.0 assignment
- `v5_*` columns — legacy compatibility (same decision, old labels)

On settle, `v6_result_basis` is set to `HIT_RECORDED` or `MISS_RECORDED`.
Parsing or empty-prediction gates set `v6_parse_policy_status = POLICY_PENDING_PARSE_REVIEW`.

### Backtest (as reported)

| Tier | Record | Rate |
|---|---|---|
| NEW_TIER1 | 31/31 | 100% |
| NEW_TIER2 | 39/41 | 95.1% |
| NEW_TIER3 | 48/60 | 80.0% |

Out-of-sample validation is the next milestone.
        """)

    with tabs[5]:
        render_debug_tab(sb)


# ============================================================================
# ENTRY
# ============================================================================
main()
