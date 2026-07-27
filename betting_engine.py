import streamlit as st
import re
from datetime import datetime, timedelta

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
    'Iguatu CE2 - 1 (1 - 0) Maguary PE'
    or 'Nacional AM1 - 1 (0 - 0) Iguatu CE'
    Returns dict with home, away, scores, or None.
    """
    # Pattern: team1 + score1 - score2 (ht1 - ht2) team2
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

def extract_team_abbreviation_mapping(lines):
    """
    Build a mapping from abbreviation (e.g., 'IGU') to full team name.
    We look for patterns like: a line with only uppercase letters (2-4 chars)
    followed later by a line containing the full team name (from the VS line).
    But we already have the full names from the VS line, so we can map by
    scanning for lines that contain the full name and an abbreviation nearby.
    Simplified: we assume the abbreviation is the first 3-4 letters of the team name,
    but we'll also detect explicit mappings from the data.
    """
    mapping = {}
    # We'll use a heuristic: find lines like "IGU" alone and the next lines contain the team name.
    # But it's easier: we can just use the full names from the VS line and later when we see
    # sections like "IGU home matches", we know that 'IGU' corresponds to the team whose name
    # appears in the match lines. We'll store the mapping as we parse.
    # We'll do it dynamically: when we find a section header with an abbreviation,
    # we look for the full team name in the match lines that follow (the home or away team).
    return mapping  # will be filled during parsing

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
    current_abbrev = None  # e.g., 'IGU', 'NAC'
    current_section = None # 'home', 'away', 'last6', 'h2h'

    # Helper to determine result (W/D/L) for a team in a match
    def get_result(home, away, team):
        if team == home['home_team']:
            if home['home_score'] > home['away_score']: return 'W'
            elif home['home_score'] < home['away_score']: return 'L'
            else: return 'D'
        elif team == home['away_team']:
            if home['away_score'] > home['home_score']: return 'W'
            elif home['away_score'] < home['home_score']: return 'L'
            else: return 'D'
        return None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ---- DETECT MATCH START (VS line) ----
        if ' VS ' in line:
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
                # Reset section tracking for this match
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

        # ---- SECTION DETECTION (form, H2H) ----
        # We need to know which team abbreviation corresponds to which full name.
        # We'll build a mapping as we encounter abbreviations.
        # For simplicity, we will store the mapping in a dict that persists for this match.
        if match:
            # Check if line is a team abbreviation (2-4 uppercase letters, possibly followed by spaces)
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
            if current_section and current_abbrev and match:
                # Try to parse a match line
                parsed = parse_match_line(line)
                if parsed:
                    # Determine which team this section belongs to
                    # We have the abbreviation, but we need to map to full name.
                    # We can try to find the full name by checking if the parsed home or away team
                    # matches the match's home or away team (case-insensitive partial match).
                    # For now, we'll store the match data with the abbreviation as key.
                    # Later we'll associate with the match.
                    debug_log(f"🔍 DEBUG: Parsed match line in section {current_section}: {parsed}")
                    # Store the parsed match in the appropriate list based on section
                    if current_section == 'home':
                        # We assume this section is for the team with current_abbrev.
                        # We need to determine if this team is the home or away in this parsed match.
                        # Since it's "home matches", the team is likely the home team.
                        # But to be safe, we check if either team matches the match's home/away.
                        # We'll just add the result for this team (we need to know which team we care about).
                        # We'll determine the team by matching the abbreviation to the full name.
                        # For now, we'll store the raw match and later compute form.
                        # We'll store a list of (result, date) for the team.
                        # We'll need to map abbreviation to full name.
                        # We'll create a mapping dict.
                        # Since we don't have a direct mapping, we can use the match's home/away team.
                        # For home matches, the team is the home team (parsed['home_team']).
                        # For away matches, the team is the away team.
                        # We'll assume the abbreviation corresponds to the match's home or away.
                        # We'll add the result for that team.
                        team = None
                        if parsed['home_team'] == match['home_team'] or parsed['home_team'] == match['away_team']:
                            team = parsed['home_team']
                        elif parsed['away_team'] == match['home_team'] or parsed['away_team'] == match['away_team']:
                            team = parsed['away_team']
                        if team:
                            result = get_result(parsed, None, team)  # we don't have the other team
                            # But we need to know which team we are tracking.
                            # We'll store the result with the team name.
                            # We'll just add to a list of results for the current_abbrev.
                            # We'll store in a dict keyed by abbreviation.
                            if not hasattr(match, '_form_data'):
                                match['_form_data'] = {}
                            if current_abbrev not in match['_form_data']:
                                match['_form_data'][current_abbrev] = {'home': [], 'away': [], 'recent': []}
                            match['_form_data'][current_abbrev][current_section].append(result)
                            debug_log(f"✅ DEBUG: Added {result} to {current_abbrev} {current_section} form")

                    elif current_section == 'away':
                        # similar
                        team = None
                        if parsed['home_team'] == match['home_team'] or parsed['home_team'] == match['away_team']:
                            team = parsed['home_team']
                        elif parsed['away_team'] == match['home_team'] or parsed['away_team'] == match['away_team']:
                            team = parsed['away_team']
                        if team:
                            result = get_result(parsed, None, team)
                            if not hasattr(match, '_form_data'):
                                match['_form_data'] = {}
                            if current_abbrev not in match['_form_data']:
                                match['_form_data'][current_abbrev] = {'home': [], 'away': [], 'recent': []}
                            match['_form_data'][current_abbrev][current_section].append(result)
                            debug_log(f"✅ DEBUG: Added {result} to {current_abbrev} {current_section} form")

                    elif current_section == 'last6':
                        # For recent form, we need to know which team this section belongs to.
                        # It's usually the team indicated by current_abbrev.
                        # We'll add the result for that team.
                        team = None
                        if parsed['home_team'] == match['home_team'] or parsed['home_team'] == match['away_team']:
                            team = parsed['home_team']
                        elif parsed['away_team'] == match['home_team'] or parsed['away_team'] == match['away_team']:
                            team = parsed['away_team']
                        if team:
                            result = get_result(parsed, None, team)
                            if not hasattr(match, '_form_data'):
                                match['_form_data'] = {}
                            if current_abbrev not in match['_form_data']:
                                match['_form_data'][current_abbrev] = {'home': [], 'away': [], 'recent': []}
                            match['_form_data'][current_abbrev]['recent'].append(result)
                            debug_log(f"✅ DEBUG: Added {result} to {current_abbrev} recent form")

                    elif current_section == 'h2h':
                        # H2H lines are already parsed, we need to store them and determine winner
                        # We have parsed['home_team'] and parsed['away_team'] with scores.
                        # We need to check which team is the current match's home/away.
                        # We'll store the result (which team won or draw) relative to the current match.
                        if parsed['home_team'] == match['home_team'] and parsed['away_team'] == match['away_team']:
                            # correct orientation
                            if parsed['home_score'] > parsed['away_score']:
                                winner = match['home_team']
                            elif parsed['home_score'] < parsed['away_score']:
                                winner = match['away_team']
                            else:
                                winner = 'Draw'
                            match['h2h'].append({'winner': winner, 'home_score': parsed['home_score'], 'away_score': parsed['away_score']})
                            debug_log(f"✅ DEBUG: Added H2H winner: {winner}")
                        elif parsed['home_team'] == match['away_team'] and parsed['away_team'] == match['home_team']:
                            # reversed orientation
                            if parsed['home_score'] > parsed['away_score']:
                                winner = match['away_team']  # because home in parsed is away in current
                            elif parsed['home_score'] < parsed['away_score']:
                                winner = match['home_team']
                            else:
                                winner = 'Draw'
                            match['h2h'].append({'winner': winner, 'home_score': parsed['home_score'], 'away_score': parsed['away_score']})
                            debug_log(f"✅ DEBUG: Added H2H winner: {winner}")

        # ---- CHECK COMPLETENESS ----
        if match and match['home_pct'] is not None:
            essential = ['home_pct', 'draw_pct', 'away_pct', 'prediction', 'score_home', 'score_away', 'avg_goals']
            missing = [f for f in essential if match.get(f) is None]
            if missing:
                debug_log(f"❌ DEBUG: Missing essential data: {missing}")
            else:
                # Process form data: we have _form_data dict with abbreviation keys.
                # Now we need to map abbreviations to full team names.
                # We'll build a mapping from abbreviation to full team name.
                # We can derive it from the match's home/away team names.
                # For each abbreviation in _form_data, we need to find which team it corresponds to.
                # We'll check if the abbreviation appears in the match lines we parsed.
                # But we already stored the results with the abbreviation.
                # So we can just map abbreviation to team name by looking at the parsed matches.
                # We'll do a simple mapping: if the abbreviation is in the home team name (first few letters) or away team.
                # For now, we'll just copy the lists to match['home_form'] and match['away_form'] based on which team the abbreviation refers to.
                # We'll create a mapping by scanning the parsed matches.
                # Since we have the abbreviations, we can try to match them to full names.
                # We'll use a simple approach: for each abbreviation, we look at the match's home and away teams,
                # and see if the abbreviation is a substring of either team name (case-insensitive) or if the team name starts with the abbreviation.
                # We'll assign the form data to the correct team.
                # But we might have multiple abbreviations (home and away). We'll store them separately.
                if hasattr(match, '_form_data'):
                    # Determine which abbreviation belongs to which team
                    abbrev_to_team = {}
                    for abbrev in match['_form_data']:
                        # Check if abbrev is part of home team name
                        if abbrev.upper() in match['home_team'].upper():
                            team = match['home_team']
                        elif abbrev.upper() in match['away_team'].upper():
                            team = match['away_team']
                        else:
                            # try to match by first 3 letters
                            if match['home_team'].upper().startswith(abbrev):
                                team = match['home_team']
                            elif match['away_team'].upper().startswith(abbrev):
                                team = match['away_team']
                            else:
                                continue
                        abbrev_to_team[abbrev] = team

                    # Now assign form lists
                    for abbrev, data in match['_form_data'].items():
                        team = abbrev_to_team.get(abbrev)
                        if team:
                            if team == match['home_team']:
                                match['home_form'] = data.get('home', [])  # we want home form for home team
                            elif team == match['away_team']:
                                match['away_form'] = data.get('away', [])
                            # recent form can be stored for both? We'll assign to match['recent_form'] as a dict.
                            match['recent_form'] = data.get('recent', [])
                    # Clean up temporary data
                    del match['_form_data']

                # ---- FATIGUE DETECTION ----
                # Check if away team played a match within 3 days before current date.
                if match['date'] and match['away_form']:
                    # We don't have dates for away matches, but we can look at the last 6 matches
                    # to see if any match date is close to current date.
                    # We'll need to parse dates from the last 6 matches section.
                    # Since we don't store dates in the form lists, we'll skip for now.
                    # A more robust implementation would parse dates from the match lines.
                    pass

                debug_log("✅ DEBUG: All essential data present!")
                matches.append(match)
                debug_log(f"✅ DEBUG: Match stored: {match['home_team']} vs {match['away_team']}")
                match = None   # reset

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
