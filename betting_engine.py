import streamlit as st
import re

# ============================================
# DEBUG LOGGING (stored in session state)
# ============================================

def debug_log(msg):
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []
    st.session_state.debug_messages.append(msg)

# ============================================
# PARSER CORE FUNCTIONS
# ============================================

def parse_encoded_line(line):
    """
    Parse encoded line like:
    '284726X1 - 12.0034°3.00'
    or '323533X0 - 01.4124°3.50'
    or '364322X1 - 12.3818°3.50'
    Returns dict with all essential fields or None.
    """
    debug_log(f"🔍 parse_encoded_line INPUT: '{line}'")
    if ' - ' not in line:
        debug_log("❌ No ' - ' separator found")
        return None

    left_part, right_part = line.split(' - ', 1)
    left_part = left_part.replace(" ", "")
    right_part = right_part.strip()
    debug_log(f"📊 Left: '{left_part}', Right: '{right_part}'")

    # Extract percentages, prediction, and home score
    # Pattern: 6 digits (2+2+2) + letter (1, X, 2) + digit (home score)
    pattern = r"(\d{2})(\d{2})(\d{2})([12X])(\d)"
    m = re.match(pattern, left_part)
    if not m:
        debug_log(f"❌ Left part did not match pattern: {left_part}")
        return None

    home_pct = int(m.group(1))
    draw_pct = int(m.group(2))
    away_pct = int(m.group(3))
    pred = m.group(4)
    score_home = int(m.group(5))
    debug_log(f"✅ Percentages: home={home_pct}, draw={draw_pct}, away={away_pct}, pred={pred}")
    debug_log(f"✅ Home score: {score_home}")

    # Parse right part for away score and avg goals
    # Pattern: integer_part + '.' + ... + '°' + avg_goals
    pattern2 = r"(\d+)\.\d+°\s*([\d.]+)"
    m2 = re.search(pattern2, right_part)
    if not m2:
        debug_log(f"❌ Could not parse right part: '{right_part}'")
        return None

    integer_part = m2.group(1)          # e.g., "12" or "01"
    avg_goals = float(m2.group(2))      # e.g., 3.00 or 3.50

    # Away score = FIRST digit of the integer part
    score_away = int(integer_part[0])
    debug_log(f"✅ Integer part: {integer_part}, Away score: {score_away}")
    debug_log(f"✅ Average goals: {avg_goals}")

    # Sanity check on percentages
    total = home_pct + draw_pct + away_pct
    if total < 95 or total > 105:
        debug_log(f"⚠️ Percentages sum to {total}% (unusual but continuing)")

    return {
        'home_pct': home_pct,
        'draw_pct': draw_pct,
        'away_pct': away_pct,
        'pred': pred,
        'score_home': score_home,
        'score_away': score_away,
        'avg_goals': avg_goals
    }

def parse_h2h_line(line, home_team, away_team):
    """
    Parse H2H lines like:
    'Nacional AM1 - 1 (0 - 0) Iguatu CE'
    or 'FK Ogre1 - 1 (1 - 1) Grobinas SC'
    Returns dict with scores or None.
    """
    pattern = r"(.+?)\s*(\d+)\s*-\s*(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*(.+)"
    m = re.match(pattern, line)
    if not m:
        return None
    return {
        'score_home': int(m.group(2)),
        'score_away': int(m.group(3)),
        'ht_home': int(m.group(4)),
        'ht_away': int(m.group(5))
    }

def parse_text_data(text):
    """
    Main parser for Forebet‑style match data.
    Returns dict with league and list of matches.
    """
    debug_log("=== 🔍 DEBUG: parse_text_data STARTED ===")
    lines = text.split('\n')
    debug_log(f"✅ DEBUG: Total lines: {len(lines)}")

    # Detect league from known keywords
    league = None
    for line in lines:
        if 'Brazil Serie D' in line:
            league = 'Brazil Serie D'
            break
        elif 'Czech Republic Chance Liga' in line:
            league = 'Czech Republic Chance Liga'
            break
        elif 'Latvia Virsliga' in line:
            league = 'Latvia Virsliga'
            break
        elif 'Serie D' in line:
            league = 'Serie D'
            break
        elif 'Chance Liga' in line:
            league = 'Chance Liga'
            break
        elif 'Virsliga' in line:
            league = 'Virsliga'
            break
        # Add more leagues as needed

    if league:
        debug_log(f"✅ DEBUG: League found: '{league}'")
    else:
        debug_log("❌ DEBUG: No league found – will still try to parse")

    matches = []
    match = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ---------- MATCH DETECTION (with improved validation) ----------
        # Only treat as match if:
        #  - Contains " VS "
        #  - Both sides have at least 2 characters, no quotes, no parentheses,
        #    and either contain a space or are longer than 3 characters.
        if ' VS ' in line:
            parts = line.split(' VS ', 1)
            left = parts[0].strip()
            right = parts[1].strip()
            # Check that both sides are valid (not code artifacts)
            valid = True
            for side in (left, right):
                if not side:
                    valid = False
                    break
                if "'" in side or '"' in side or '(' in side or ')' in side:
                    valid = False
                    break
                # A team name should have at least two words or be longer than 3 chars
                if not (' ' in side or len(side) > 3):
                    valid = False
                    break
            if valid:
                debug_log(f"🔍 DEBUG: Found valid VS line at index {i}: '{line}'")
                home = left
                away = right
                debug_log(f"🔍 DEBUG: Home='{home}', Away='{away}'")
                match = {
                    'home_team': home,
                    'away_team': away,
                    'league': league,
                    'home_pct': None,
                    'draw_pct': None,
                    'away_pct': None,
                    'prediction': None,
                    'score_home': None,
                    'score_away': None,
                    'avg_goals': None,
                    'date': None,
                    'h2h': []
                }
                debug_log("✅ DEBUG: match_found = True")
                continue   # skip further processing for this line

        # ---------- DATE EXTRACTION ----------
        if match:
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            if date_match:
                date_str = date_match.group(1)
                try:
                    day, month, year = date_str.split('/')
                    match['date'] = f"{year}-{month}-{day}"
                    debug_log(f"✅ DEBUG: Date found: {match['date']}")
                except:
                    pass

            # ---------- ENCODED LINE ----------
            # Look for a line with " - " and containing six digits + a letter + a digit
            if ' - ' in line and re.search(r'\d{6}[12X]\d', line.replace(' ', '')):
                debug_log(f"🔍 DEBUG: Found encoded line at index {i}: '{line}'")
                result = parse_encoded_line(line)
                if result:
                    match['home_pct'] = result['home_pct']
                    match['draw_pct'] = result['draw_pct']
                    match['away_pct'] = result['away_pct']
                    match['prediction'] = result['pred']
                    match['score_home'] = result['score_home']
                    match['score_away'] = result['score_away']
                    match['avg_goals'] = result['avg_goals']
                    debug_log("✅ DEBUG: Encoded line parsed successfully!")
                else:
                    debug_log("❌ DEBUG: parse_encoded_line FAILED")

            # ---------- H2H LINES ----------
            if match and ' - ' in line and '(' in line and ')' in line:
                if re.search(r'\d+\s*-\s*\d+\s*\(\s*\d+\s*-\s*\d+\s*\)', line):
                    debug_log(f"🔍 DEBUG: Potential H2H line at {i}: '{line}'")
                    h2h_result = parse_h2h_line(line, match['home_team'], match['away_team'])
                    if h2h_result:
                        match['h2h'].append(h2h_result)
                        debug_log(f"✅ DEBUG: Parsed H2H: {h2h_result}")

            # ---------- CHECK COMPLETENESS ----------
            if match and match['home_pct'] is not None:
                essential = ['home_pct', 'draw_pct', 'away_pct', 'prediction',
                             'score_home', 'score_away', 'avg_goals']
                missing = [f for f in essential if match.get(f) is None]
                if missing:
                    debug_log(f"❌ DEBUG: Missing essential data: {missing}")
                else:
                    debug_log("✅ DEBUG: All essential data present!")
                    matches.append(match)
                    debug_log(f"✅ DEBUG: Match stored: {match['home_team']} vs {match['away_team']}")
                    match = None   # reset for next match

            # ---------- PERIODIC DEBUG (every 50 lines) ----------
            if match and i % 50 == 0:
                debug_log("🔍 DEBUG: Checking if match is complete...")
                debug_log(f"home_pct={match.get('home_pct')}")
                debug_log(f"draw_pct={match.get('draw_pct')}")
                debug_log(f"away_pct={match.get('away_pct')}")
                debug_log(f"prediction={match.get('prediction')}")
                debug_log(f"score_home={match.get('score_home')}")
                debug_log(f"score_away={match.get('score_away')}")
                debug_log(f"avg_goals={match.get('avg_goals')}")

    debug_log("=== 🔍 DEBUG: parse_text_data COMPLETE ===")
    debug_log(f"Total matches found: {len(matches)}")
    return {'league': league, 'matches': matches}

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Forebet Parser", layout="wide")

st.title("🐛 Refined Formula V1.1 - DEBUG MODE")
st.markdown("DEBUG MODE: Shows exactly where the parser fails")
st.markdown("🔍 All parser steps are displayed - scroll up to see the full debug output")

with st.sidebar:
    st.header("📝 Paste Match Data")
    st.markdown("The debug output below will show you EXACTLY what the parser is doing")
    text_input = st.text_area("Paste Forebet data here", height=400)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📝 Input Data")
    st.text_area("Data to parse", value=text_input, height=300, key="display_text")

with col2:
    st.subheader("📊 Results")
    if st.button("🔍 Parse Data", type="primary"):
        # Clear previous debug messages
        st.session_state.debug_messages = []
        result = parse_text_data(text_input)

        # Show debug output
        with st.expander("🔍 DEBUG OUTPUT (scroll down for results)", expanded=True):
            debug_output = "\n".join(st.session_state.debug_messages)
            st.code(debug_output, language="")

        # Show results
        st.subheader("📊 Parser Results")
        if result['matches']:
            st.success(f"✅ Found {len(result['matches'])} matches!")
            for match in result['matches']:
                with st.container():
                    st.markdown(f"**{match['home_team']} vs {match['away_team']}**")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Home %", f"{match['home_pct']}%")
                    with col_b:
                        st.metric("Draw %", f"{match['draw_pct']}%")
                    with col_c:
                        st.metric("Away %", f"{match['away_pct']}%")
                    with col_d:
                        st.metric("Prediction", f"{match['prediction']}")
                    col_e, col_f, col_g = st.columns(3)
                    with col_e:
                        st.metric("Correct Score", f"{match['score_home']}-{match['score_away']}")
                    with col_f:
                        st.metric("Avg Goals", f"{match['avg_goals']}")
                    with col_g:
                        st.metric("Date", match['date'] if match['date'] else "Not found")
                    if match['h2h']:
                        st.write(f"📊 Found {len(match['h2h'])} H2H matches")
                    st.divider()
            st.subheader("📋 Parsed Match Data")
            st.json(result['matches'])
        else:
            st.error("❌ No matches found in the data.")
            st.info("Scroll up to see the DEBUG output - it will show exactly where the parser failed.")

# Display parser rules
with st.expander("📋 Parser Rules"):
    st.markdown("""
    ### Your 5 Refined Formula Rules:
    1. 🏰 **Home Fortress** - Unbeaten in last 5 home games → Back Home Win
    2. 💀 **Away Form Killer** - Lost 4 of last 6 away games → Back Home Win
    3. 🏆 **H2H Dominance** - 3 of last 4 H2Hs won → Draw is a trap
    4. 🤝 **H2H Draw Rate** - 4+ draws in last 6 H2Hs → Trust the Draw
    5. 😴 **Midweek Fatigue** - Away played 3-4 days ago → Downgrade away
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use:**
1. Copy Forebet data from their website
2. Paste it in the text area
3. Click "Parse Data"
4. Check debug output for parsing details
5. See results in the right panel
""")
