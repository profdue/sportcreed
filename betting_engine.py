import streamlit as st
import re

# ============================================
# DEBUG LOGGING
# ============================================

def debug_log(msg):
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []
    st.session_state.debug_messages.append(msg)

# ============================================
# PARSER CORE FUNCTIONS
# ============================================

def parse_encoded_line(line):
    """Extract percentages, prediction, scores, and avg goals from encoded line."""
    debug_log(f"🔍 parse_encoded_line INPUT: '{line}'")
    if ' - ' not in line:
        debug_log("❌ No ' - ' separator found")
        return None

    left_part, right_part = line.split(' - ', 1)
    left_part = left_part.replace(" ", "")
    right_part = right_part.strip()
    debug_log(f"📊 Left: '{left_part}', Right: '{right_part}'")

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

    pattern2 = r"(\d+)\.\d+°\s*([\d.]+)"
    m2 = re.search(pattern2, right_part)
    if not m2:
        debug_log(f"❌ Could not parse right part: '{right_part}'")
        return None

    integer_part = m2.group(1)
    avg_goals = float(m2.group(2))
    score_away = int(integer_part[0])
    debug_log(f"✅ Integer part: {integer_part}, Away score: {score_away}")
    debug_log(f"✅ Average goals: {avg_goals}")

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

def parse_match_line(line):
    """
    Parse a match line like:
    'Cluj1 - 1 (0 - 0) Arges Pitesti'
    or 'Nacional AM1 - 1 (0 - 0) Iguatu CE'
    Returns dict with home, away, scores, or None.
    """
    pattern = r"(.+?)\s*(\d+)\s*-\s*(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*(.+)"
    m = re.match(pattern, line)
    if not m:
        return None
    return {
        'home_team': m.group(1).strip(),
        'home_score': int(m.group(2)),
        'away_score': int(m.group(3)),
        'ht_home': int(m.group(4)),
        'ht_away': int(m.group(5)),
        'away_team': m.group(6).strip()
    }

def get_result(parsed, team):
    """Return 'W', 'D', 'L' for the given team in a parsed match."""
    if parsed['home_team'] == team:
        if parsed['home_score'] > parsed['away_score']:
            return 'W'
        elif parsed['home_score'] < parsed['away_score']:
            return 'L'
        else:
            return 'D'
    elif parsed['away_team'] == team:
        if parsed['away_score'] > parsed['home_score']:
            return 'W'
        elif parsed['away_score'] < parsed['home_score']:
            return 'L'
        else:
            return 'D'
    return None

def parse_text_data(text):
    debug_log("=== 🔍 DEBUG: parse_text_data STARTED ===")
    lines = text.split('\n')
    debug_log(f"✅ DEBUG: Total lines: {len(lines)}")

    # ---- LEAGUE DETECTION ----
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
        elif 'Sweden Superettan' in line:
            league = 'Sweden Superettan'
            break
        elif 'Romania Divizia A' in line:
            league = 'Romania Divizia A'
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
        elif 'Superettan' in line:
            league = 'Superettan'
            break
        elif 'Divizia A' in line:
            league = 'Divizia A'
            break
    if league:
        debug_log(f"✅ DEBUG: League found: '{league}'")
    else:
        debug_log("❌ DEBUG: No league found – will still try to parse")

    matches = []
    match = None
    current_abbrev = None
    current_section = None

    def store_current_match():
        """Store the current match if it has essential data and is not already stored."""
        nonlocal match, matches
        if match and match['home_pct'] is not None:
            essential = ['home_pct', 'draw_pct', 'away_pct', 'prediction', 'score_home', 'score_away', 'avg_goals']
            if all(match.get(f) is not None for f in essential):
                # Process form data: we might have _form_data dict
                if hasattr(match, '_form_data'):
                    # Map abbreviations to team names
                    abbrev_to_team = {}
                    for abbrev in match['_form_data']:
                        # Try to match abbreviation to home or away team
                        if abbrev.upper() in match['home_team'].upper():
                            team = match['home_team']
                        elif abbrev.upper() in match['away_team'].upper():
                            team = match['away_team']
                        elif match['home_team'].upper().startswith(abbrev):
                            team = match['home_team']
                        elif match['away_team'].upper().startswith(abbrev):
                            team = match['away_team']
                        else:
                            continue
                        abbrev_to_team[abbrev] = team

                    for abbrev, data in match['_form_data'].items():
                        team = abbrev_to_team.get(abbrev)
                        if team:
                            if team == match['home_team']:
                                match['home_form'] = data.get('home', [])
                            elif team == match['away_team']:
                                match['away_form'] = data.get('away', [])
                            # recent form is stored separately, assign to match['recent_form']
                            match['recent_form'] = data.get('recent', [])
                    del match['_form_data']

                # Store match
                matches.append(match)
                debug_log(f"✅ DEBUG: Match stored: {match['home_team']} vs {match['away_team']}")
                match = None  # reset for next match

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ---- DETECT MATCH START (VS line) ----
        if ' VS ' in line:
            # If there is a current match, store it before starting a new one
            if match:
                store_current_match()

            parts = line.split(' VS ', 1)
            left = parts[0].strip()
            right = parts[1].strip()
            valid = True
            for side in (left, right):
                if not side or "'" in side or '"' in side or '(' in side or ')' in side:
                    valid = False
                    break
                if not (' ' in side or len(side) > 3):
                    valid = False
                    break
            if valid:
                debug_log(f"🔍 DEBUG: Found valid VS line at index {i}: '{line}'")
                match = {
                    'home_team': left,
                    'away_team': right,
                    'league': league,
                    'home_pct': None,
                    'draw_pct': None,
                    'away_pct': None,
                    'prediction': None,
                    'score_home': None,
                    'score_away': None,
                    'avg_goals': None,
                    'date': None,
                    'h2h': [],
                    'home_form': [],
                    'away_form': [],
                    'recent_form': [],
                    'fatigue': False
                }
                debug_log("✅ DEBUG: match_found = True")
                current_abbrev = None
                current_section = None
                continue

        # ---- DATE EXTRACTION ----
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

        # ---- ENCODED LINE ----
        if match and ' - ' in line and re.search(r'\d{6}[12X]\d', line.replace(' ', '')):
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
            # We do NOT store the match here; we wait for sections to be processed

        # ---- SECTION DETECTION (form, H2H) ----
        if match:
            # Check if line is a team abbreviation (2-4 uppercase letters, possibly with spaces trimmed)
            if re.match(r'^[A-Z]{2,4}$', line):
                current_abbrev = line
                debug_log(f"🔍 DEBUG: Detected team abbreviation: {current_abbrev}")
                continue

            # Detect section headers
            if 'home matches' in line.lower():
                current_section = 'home'
                debug_log(f"🔍 DEBUG: Found home matches section for {current_abbrev}")
                continue
            elif 'away matches' in line.lower():
                current_section = 'away'
                debug_log(f"🔍 DEBUG: Found away matches section for {current_abbrev}")
                continue
            elif 'last 6 matches' in line.lower():
                current_section = 'last6'
                debug_log(f"🔍 DEBUG: Found last 6 matches section for {current_abbrev}")
                continue
            elif 'head to head' in line.lower():
                current_section = 'h2h'
                debug_log(f"🔍 DEBUG: Found H2H section")
                continue

            # ---- PARSE MATCH LINES WITHIN SECTIONS ----
            if current_section and current_abbrev:
                parsed = parse_match_line(line)
                if parsed:
                    debug_log(f"🔍 DEBUG: Parsed match line in section {current_section}: {parsed}")
                    # Determine which team this section belongs to
                    # We have the abbreviation, but we need to map to full name.
                    # We'll store the result for the team that matches the abbreviation.
                    # First, find which team in the current match corresponds to this abbreviation.
                    team = None
                    if parsed['home_team'] == match['home_team'] or parsed['home_team'] == match['away_team']:
                        team = parsed['home_team']
                    elif parsed['away_team'] == match['home_team'] or parsed['away_team'] == match['away_team']:
                        team = parsed['away_team']
                    if team:
                        result = get_result(parsed, team)
                        if result:
                            if not hasattr(match, '_form_data'):
                                match['_form_data'] = {}
                            if current_abbrev not in match['_form_data']:
                                match['_form_data'][current_abbrev] = {'home': [], 'away': [], 'recent': []}
                            # Depending on section, store in appropriate list
                            if current_section == 'home':
                                match['_form_data'][current_abbrev]['home'].append(result)
                            elif current_section == 'away':
                                match['_form_data'][current_abbrev]['away'].append(result)
                            elif current_section == 'last6':
                                match['_form_data'][current_abbrev]['recent'].append(result)
                            debug_log(f"✅ DEBUG: Added {result} to {current_abbrev} {current_section} form")
                    # For H2H, we handle separately because we need winner relative to current match
                    if current_section == 'h2h' and parsed:
                        # Determine winner relative to match's home/away
                        if parsed['home_team'] == match['home_team'] and parsed['away_team'] == match['away_team']:
                            if parsed['home_score'] > parsed['away_score']:
                                winner = match['home_team']
                            elif parsed['home_score'] < parsed['away_score']:
                                winner = match['away_team']
                            else:
                                winner = 'Draw'
                            match['h2h'].append({'winner': winner, 'home_score': parsed['home_score'], 'away_score': parsed['away_score']})
                            debug_log(f"✅ DEBUG: Added H2H winner: {winner}")
                        elif parsed['home_team'] == match['away_team'] and parsed['away_team'] == match['home_team']:
                            if parsed['home_score'] > parsed['away_score']:
                                winner = match['away_team']  # because home in parsed is away in current
                            elif parsed['home_score'] < parsed['away_score']:
                                winner = match['home_team']
                            else:
                                winner = 'Draw'
                            match['h2h'].append({'winner': winner, 'home_score': parsed['home_score'], 'away_score': parsed['away_score']})
                            debug_log(f"✅ DEBUG: Added H2H winner: {winner}")

        # ---- PERIODIC DEBUG ----
        if match and i % 50 == 0:
            debug_log("🔍 DEBUG: Checking if match is complete...")
            debug_log(f"home_pct={match.get('home_pct')}")
            debug_log(f"draw_pct={match.get('draw_pct')}")
            debug_log(f"away_pct={match.get('away_pct')}")
            debug_log(f"prediction={match.get('prediction')}")
            debug_log(f"score_home={match.get('score_home')}")
            debug_log(f"score_away={match.get('score_away')}")
            debug_log(f"avg_goals={match.get('avg_goals')}")

    # After loop, store the last match if any
    if match:
        store_current_match()

    debug_log("=== 🔍 DEBUG: parse_text_data COMPLETE ===")
    debug_log(f"Total matches found: {len(matches)}")
    return {'league': league, 'matches': matches}

# ============================================
# STREAMLIT UI (unchanged)
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
        st.session_state.debug_messages = []
        result = parse_text_data(text_input)

        with st.expander("🔍 DEBUG OUTPUT (scroll down for results)", expanded=True):
            debug_output = "\n".join(st.session_state.debug_messages)
            st.code(debug_output, language="")

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
                        # Show summary
                        wins = {'home': 0, 'away': 0, 'draw': 0}
                        for h in match['h2h']:
                            if h['winner'] == match['home_team']:
                                wins['home'] += 1
                            elif h['winner'] == match['away_team']:
                                wins['away'] += 1
                            else:
                                wins['draw'] += 1
                        st.write(f"H2H record: {match['home_team']} {wins['home']} - {wins['draw']} - {wins['away']} {match['away_team']}")
                    if match['home_form']:
                        st.write(f"🏠 Home form (last 5): {', '.join(match['home_form'][:5])}")
                    if match['away_form']:
                        st.write(f"✈️ Away form (last 5): {', '.join(match['away_form'][:5])}")
                    if match['recent_form']:
                        st.write(f"📈 Recent form (last 6): {', '.join(match['recent_form'][:6])}")
                    st.divider()
            st.subheader("📋 Parsed Match Data")
            st.json(result['matches'])
        else:
            st.error("❌ No matches found in the data.")
            st.info("Scroll up to see the DEBUG output - it will show exactly where the parser failed.")

with st.expander("📋 Parser Rules"):
    st.markdown("""
    ### Your 5 Refined Formula Rules:
    1. 🏰 **Home Fortress** - Home unbeaten in last 5 home games → Home Win (1)
    2. 💀 **Away Form Killer** - Away lost 4 of last 6 away games → Home Win (1)
    3. 🏆 **H2H Dominance** - One team won 3 of last 4 H2Hs → Back that side
    4. 🤝 **H2H Draw Rate** - 4+ draws in last 6 H2Hs → Draw (X)
    5. 😴 **Midweek Fatigue** - Away played 3-4 days ago → Home Win or Draw
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
