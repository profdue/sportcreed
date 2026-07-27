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

    pattern2 = r"(\d+)\.\d+°\s*([\d.]+|-)"
    m2 = re.search(pattern2, right_part)
    if not m2:
        debug_log(f"❌ Could not parse right part: '{right_part}'")
        return None
    integer_part = m2.group(1)
    avg_goals_str = m2.group(2)
    if avg_goals_str == '-':
        avg_goals = None
        debug_log("⚠️ No average goals provided (set to None)")
    else:
        avg_goals = float(avg_goals_str)
        debug_log(f"✅ Average goals: {avg_goals}")
    score_away = int(integer_part[0])
    debug_log(f"✅ Integer part: {integer_part}, Away score: {score_away}")

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

def get_result(parsed, team):
    """Return 'W', 'D', 'L' for the given team in a parsed match dict."""
    if parsed['home_team'] == team:
        if parsed['home_score'] > parsed['away_score']: return 'W'
        elif parsed['home_score'] < parsed['away_score']: return 'L'
        else: return 'D'
    elif parsed['away_team'] == team:
        if parsed['away_score'] > parsed['home_score']: return 'W'
        elif parsed['away_score'] < parsed['home_score']: return 'L'
        else: return 'D'
    return None

def normalize_team_name(name):
    """
    Remove common suffixes and extra text so that team names can be compared.
    Examples: 'CSKA-Sofia Br1' -> 'CSKA-Sofia'
              'Sport Recife Br2' -> 'Sport Recife'
              'Botev Plovdiv Bg1' -> 'Botev Plovdiv'
    """
    # Remove league/competition suffixes
    name = re.sub(r'\s+(Br|Br2|Br4|Bg|Bg1|Ro1|Ro2|Se2|Ec1|Lv1|UEL|BrN|BrC|Cz1|Cz2)\s*$', '', name)
    # Remove "View all"
    name = re.sub(r'\s+View all.*$', '', name)
    # Remove "Win/Draw/Lost" and percentages
    name = re.sub(r'\s+(Win|Draw|Lost)\s+\d+%.*$', '', name)
    return name.strip()

def process_section_block(match, abbrev, section_type, lines):
    """
    Process a block of lines belonging to a section (home, away, last6, h2h).
    Join lines, extract all match patterns, and update the match object.
    """
    if not match or not lines:
        return
    block = " ".join(lines)
    debug_log(f"🔍 Processing {section_type} block for {abbrev}")

    # Regex to find match patterns: TeamA score - score (HT) TeamB
    # Allow accented characters, hyphens, parentheses in team names
    pattern = re.compile(r"([A-Za-zÀ-ÿ\s.()-]+?)\s*(\d+)\s*-\s*(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*([A-Za-zÀ-ÿ\s.()]+)")
    matches = pattern.findall(block)

    if not matches:
        debug_log(f"❌ No match patterns found in {section_type} block for {abbrev}")
        return

    for m in matches:
        home_team = m[0].strip()
        home_score = int(m[1])
        away_score = int(m[2])
        ht_home = int(m[3])
        ht_away = int(m[4])
        away_team = m[5].strip()

        parsed = {
            'home_team': home_team,
            'home_score': home_score,
            'away_score': away_score,
            'ht_home': ht_home,
            'ht_away': ht_away,
            'away_team': away_team
        }

        # Normalize all team names for comparison
        parsed_home_norm = normalize_team_name(home_team)
        parsed_away_norm = normalize_team_name(away_team)
        match_home_norm = normalize_team_name(match['home_team'])
        match_away_norm = normalize_team_name(match['away_team'])

        # Determine which team this match is about
        team = None
        if parsed_home_norm == match_home_norm or parsed_home_norm == match_away_norm:
            team = home_team
        elif parsed_away_norm == match_home_norm or parsed_away_norm == match_away_norm:
            team = away_team
        # If still not matched, try using abbreviation (if available)
        if not team and abbrev:
            if abbrev.upper() in home_team.upper():
                team = home_team
            elif abbrev.upper() in away_team.upper():
                team = away_team

        if not team:
            debug_log(f"⚠️ Could not determine which team for match: {home_team} vs {away_team}")
            continue

        result = get_result(parsed, team)
        if result:
            if section_type == 'home' and team == match['home_team']:
                match['home_form'].append(result)
                debug_log(f"✅ Added {result} to home_form for {team}")
            elif section_type == 'away' and team == match['away_team']:
                match['away_form'].append(result)
                debug_log(f"✅ Added {result} to away_form for {team}")
            elif section_type == 'last6':
                if team == match['home_team']:
                    match['home_recent'].append(result)
                elif team == match['away_team']:
                    match['away_recent'].append(result)
                debug_log(f"✅ Added {result} to recent form for {team}")
            elif section_type == 'h2h':
                # Determine winner relative to current match's home/away
                if parsed_home_norm == match_home_norm and parsed_away_norm == match_away_norm:
                    if home_score > away_score:
                        winner = match['home_team']
                    elif home_score < away_score:
                        winner = match['away_team']
                    else:
                        winner = 'Draw'
                elif parsed_home_norm == match_away_norm and parsed_away_norm == match_home_norm:
                    if home_score > away_score:
                        winner = match['away_team']
                    elif home_score < away_score:
                        winner = match['home_team']
                    else:
                        winner = 'Draw'
                else:
                    winner = None
                if winner:
                    match['h2h'].append({
                        'winner': winner,
                        'home_score': home_score,
                        'away_score': away_score
                    })
                    debug_log(f"✅ Added H2H winner: {winner}")

def store_current_match(match, matches):
    """Finalize and store the current match if it has essential data."""
    if not match:
        return
    essential = ['home_pct', 'draw_pct', 'away_pct', 'prediction', 'score_home', 'score_away']
    if all(match.get(f) is not None for f in essential):
        matches.append(match)
        debug_log(f"✅ DEBUG: Match stored: {match['home_team']} vs {match['away_team']}")
    else:
        missing = [f for f in essential if match.get(f) is None]
        debug_log(f"❌ DEBUG: Match incomplete, missing: {missing}")

def parse_text_data(text):
    debug_log("=== 🔍 DEBUG: parse_text_data STARTED ===")
    lines = text.split('\n')
    debug_log(f"✅ DEBUG: Total lines: {len(lines)}")

    # ---- LEAGUE DETECTION ----
    league = None
    league_keywords = {
        'Brazil Serie D': 'Brazil Serie D',
        'Brazil Serie B': 'Brazil Serie B',
        'Czech Republic Chance Liga': 'Czech Republic Chance Liga',
        'Latvia Virsliga': 'Latvia Virsliga',
        'Sweden Superettan': 'Sweden Superettan',
        'Romania Divizia A': 'Romania Divizia A',
        'Ecuador Serie A': 'Ecuador Serie A',
        'Bulgaria Parva Liga': 'Bulgaria Parva Liga',
        'Serie D': 'Serie D',
        'Chance Liga': 'Chance Liga',
        'Virsliga': 'Virsliga',
        'Superettan': 'Superettan',
        'Divizia A': 'Divizia A',
        'Parva Liga': 'Parva Liga'
    }
    for line in lines:
        for key, val in league_keywords.items():
            if key in line:
                league = val
                break
        if league:
            break
    if league:
        debug_log(f"✅ DEBUG: League found: '{league}'")
    else:
        debug_log("❌ DEBUG: No league found – will still try to parse")

    matches = []
    match = None
    current_abbrev = None
    current_section = None
    section_lines = []  # accumulate lines for the current section

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            # Blank line: if we were accumulating, process the block now
            if current_section and section_lines:
                process_section_block(match, current_abbrev, current_section, section_lines)
                section_lines = []
            continue

        # ---- DETECT MATCH START ----
        if ' VS ' in line:
            if match:
                store_current_match(match, matches)
                match = None
            parts = line.split(' VS ', 1)
            left = parts[0].strip()
            right = parts[1].strip()
            # Only reject if empty or contains quotes (code artifacts)
            if left and right and "'" not in left and '"' not in left and "'" not in right and '"' not in right:
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
                    'home_recent': [],
                    'away_recent': [],
                    'fatigue': False
                }
                debug_log("✅ DEBUG: match_found = True")
                current_abbrev = None
                current_section = None
                section_lines = []
                continue

        # ---- DATE EXTRACTION (first date with time) ----
        if match and match['date'] is None:
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})\s+\d{2}:\d{2}', line)
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
            # Do NOT reset section – encoded line is separate
            continue

        # ---- SECTION DETECTION ----
        if match:
            # Check for team abbreviation (2-4 uppercase letters)
            if re.match(r'^[A-Z]{2,4}$', line):
                # Process any pending section before changing abbreviation
                if current_section and section_lines:
                    process_section_block(match, current_abbrev, current_section, section_lines)
                    section_lines = []
                current_abbrev = line
                debug_log(f"🔍 DEBUG: Detected team abbreviation: {current_abbrev}")
                continue

            lower = line.lower()
            if 'home matches' in lower:
                if current_section and section_lines:
                    process_section_block(match, current_abbrev, current_section, section_lines)
                    section_lines = []
                current_section = 'home'
                debug_log(f"🔍 DEBUG: Found home matches section for {current_abbrev}")
                continue
            elif 'away matches' in lower:
                if current_section and section_lines:
                    process_section_block(match, current_abbrev, current_section, section_lines)
                    section_lines = []
                current_section = 'away'
                debug_log(f"🔍 DEBUG: Found away matches section for {current_abbrev}")
                continue
            elif 'last 6 matches' in lower:
                if current_section and section_lines:
                    process_section_block(match, current_abbrev, current_section, section_lines)
                    section_lines = []
                current_section = 'last6'
                debug_log(f"🔍 DEBUG: Found last 6 matches section for {current_abbrev}")
                continue
            elif 'head to head' in lower:
                if current_section and section_lines:
                    process_section_block(match, current_abbrev, current_section, section_lines)
                    section_lines = []
                current_section = 'h2h'
                debug_log(f"🔍 DEBUG: Found H2H section")
                continue

            # ---- ACCUMULATE LINES FOR THE CURRENT SECTION ----
            if current_section and match:
                section_lines.append(line)

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

    # Process any remaining section lines
    if current_section and section_lines and match:
        process_section_block(match, current_abbrev, current_section, section_lines)

    # Store last match if any
    if match:
        store_current_match(match, matches)

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
                        st.metric("Avg Goals", f"{match['avg_goals'] if match['avg_goals'] is not None else 'N/A'}")
                    with col_g:
                        st.metric("Date", match['date'] if match['date'] else "Not found")

                    if match['h2h']:
                        st.write(f"📊 Found {len(match['h2h'])} H2H matches")
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
                    if match['home_recent']:
                        st.write(f"📈 Recent form (last 6) - {match['home_team']}: {', '.join(match['home_recent'][:6])}")
                    if match['away_recent']:
                        st.write(f"📈 Recent form (last 6) - {match['away_team']}: {', '.join(match['away_recent'][:6])}")
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
