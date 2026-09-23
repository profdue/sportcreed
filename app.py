"""
DIAGNOSTIC VERSION — will show exactly where the app breaks.
Delete after debugging.
"""
import streamlit as st

st.set_page_config(page_title="Debug", layout="wide")

st.write("## Checkpoint 1 — Streamlit is alive")

try:
    import pandas as pd
    st.write("✅ pandas imported")
except Exception as e:
    st.error(f"❌ pandas import failed: {e}")
    st.stop()

try:
    from datetime import date, datetime
    import traceback
    st.write("✅ stdlib imported")
except Exception as e:
    st.error(f"❌ stdlib failed: {e}")
    st.stop()

try:
    from supabase import create_client
    st.write("✅ supabase imported")
except Exception as e:
    st.error(f"❌ supabase import failed: {e}")

try:
    import bs4
    st.write(f"✅ beautifulsoup4 imported (version {bs4.__version__})")
except Exception as e:
    st.error(f"❌ beautifulsoup4 import failed: {e}")

try:
    import scipy
    st.write(f"✅ scipy imported (version {scipy.__version__})")
except Exception as e:
    st.error(f"❌ scipy import failed: {e}")

try:
    import numpy as np
    st.write(f"✅ numpy imported (version {np.__version__})")
except Exception as e:
    st.error(f"❌ numpy import failed: {e}")

st.write("## Checkpoint 2 — checking secrets")

try:
    url = st.secrets.get("SUPABASE_URL", None)
    key = st.secrets.get("SUPABASE_KEY", None)
    if url:
        st.write(f"✅ SUPABASE_URL present (starts with: {url[:20]}...)")
    else:
        st.warning("⚠️ SUPABASE_URL missing")
    if key:
        st.write(f"✅ SUPABASE_KEY present (length: {len(key)})")
    else:
        st.warning("⚠️ SUPABASE_KEY missing")
except Exception as e:
    st.error(f"❌ st.secrets failed: {e}")

st.write("## Checkpoint 3 — checking betting_engine.py")

try:
    from betting_engine import analyse_html, load_parsed_match, RefinedPredictor
    st.write("✅ betting_engine imported successfully")
    st.write(f"   - analyse_html: {analyse_html}")
    st.write(f"   - RefinedPredictor: {RefinedPredictor}")
except Exception as e:
    st.error(f"❌ betting_engine import failed: {e}")
    st.code(traceback.format_exc())

st.write("## Checkpoint 4 — attempting Supabase connection")

try:
    if url and key:
        client = create_client(url, key)
        st.write("✅ Supabase client created")
    else:
        st.warning("Skipping Supabase connection — missing secrets")
except Exception as e:
    st.error(f"❌ Supabase connection failed: {e}")
    st.code(traceback.format_exc())

st.write("## ✅ All checkpoints passed — the app is healthy")
st.write("If you see this, the blank page was caused by something else in the full app.")
