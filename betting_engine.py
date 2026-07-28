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

def normalize_team_name(name):
    suffixes = r'(Br|Br2|Br4|Bg|Bg1|Ro1|Ro2|Se2|Ec1|Lv1|UEL|BrN|BrC|Cz1|Cz2|ECL|Ro|BgC|Cup|Copa|Série|LvC|Ru1|Ru2)'
    name = re.sub(r'\s+' + suffixes + r'\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+View all.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(Win|Draw|Lost)\s+\d+%.*$', '', name, flags=re.IGNORECASE)
    return name.strip()

def get_result(parsed, team):
    if parsed['home_team'] == team:
        if parsed['home_score'] > parsed['away_score']: return 'W'
        elif parsed['home_score'] < parsed['away_score']: return 'L'
        else: return 'D'
    elif parsed['away_team'] == team:
        if parsed['away_score'] > parsed['home_score']: return 'W'
        elif parsed['away_score'] < parsed['home_score']: return 'L'
        else: return 'D'
    return None

def process_section_block(match, abbrev, section_type, lines):
    if not match or not lines:
        return
    block = " ".join(lines)
    debug_log(f"🔍 Processing {section_type} block for {abbrev}")

    pattern = re.compile(r"([A-Za-zÀ-ÿ\s.()-]+?)\s*(\d+)\s*-\s*(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*([A-Za-zÀ-ÿ\s.()]+)")
    matches = pattern.findall(block)

    if not matches:
        # Try again after removing dates (dd/mm or dd/mm/yyyy)
        cleaned = re.sub(r'\b\d{1,2}/\d{1,2}(?:/\d{4})?\b', '', block)
        matches = pattern.findall(cleaned)

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

        parsed_home_norm = normalize_team_name(home_team)
        parsed_away_norm = normalize_team_name(away_team)
        match_home_norm = normalize_team_name(match['home_team'])
        match_away_norm = normalize_team_name(match['away_team'])

        team = None
        if parsed_home_norm == match_home_norm or parsed_home_norm == match_away_norm:
            team = home_team
        elif parsed_away_norm == match_home_norm or parsed_away_norm == match_away_norm:
            team = away_team
        if not team:
            if parsed_home_norm and (parsed_home_norm in match_home_norm or match_home_norm in parsed_home_norm):
                team = home_team
            elif parsed_away_norm and (parsed_away_norm in match_home_norm or match_home_norm in parsed_away_norm):
                team = away_team
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
            elif section_type == 'away' and team == match['away_team']:
                match['away_form'].append(result)
            elif section_type == 'last6':
                if team == match['home_team']:
                    match['home_recent'].append(result)
                elif team == match['away_team']:
                    match['away_recent'].append(result)

def store_current_match(match, matches):
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
        'Russia Premier League': 'Russia Premier League',
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
    section_lines = []

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if current_section and section_lines:
                process_section_block(match, current_abbrev, current_section, section_lines)
                section_lines = []
            continue

        if ' VS ' in line:
            if match:
                store_current_match(match, matches)
                match = None
            parts = line.split(' VS ', 1)
            left = parts[0].strip()
            right = parts[1].strip()
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
                    'home_form': [],
                    'away_form': [],
                    'home_recent': [],
                    'away_recent': [],
                    'fatigue': False,
                    'overall_stats': {}
                }
                debug_log("✅ DEBUG: match_found = True")
                current_abbrev = None
                current_section = None
                section_lines = []
                continue

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
            continue

        if match:
            # Team abbreviation (2-4 uppercase letters) – ignore false positives
            if re.match(r'^[A-Z]{2,4}$', line):
                false_abbrevs = {'FT', 'HT', 'ALL', 'VIEW', 'WIN', 'DRAW', 'LOST', 'PTS', 'GP', 'GF', 'GA'}
                if line.upper() not in false_abbrevs:
                    if current_section and section_lines:
                        process_section_block(match, current_abbrev, current_section, section_lines)
                        section_lines = []
                    current_abbrev = line
                    debug_log(f"🔍 DEBUG: Detected team abbreviation: {current_abbrev}")
                    continue
                else:
                    debug_log(f"ℹ️ Ignoring non-team abbreviation: {line}")
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
            elif 'head to head' in lower or 'head-to-head' in lower:
                debug_log("ℹ️ Skipping H2H section (not used)")
                current_section = None
                section_lines = []
                continue

            if current_section and match:
                section_lines.append(line)

        if match and i % 50 == 0:
            debug_log("🔍 DEBUG: Checking if match is complete...")
            debug_log(f"home_pct={match.get('home_pct')}")
            debug_log(f"draw_pct={match.get('draw_pct')}")
            debug_log(f"away_pct={match.get('away_pct')}")
            debug_log(f"prediction={match.get('prediction')}")
            debug_log(f"score_home={match.get('score_home')}")
            debug_log(f"score_away={match.get('score_away')}")
            debug_log(f"avg_goals={match.get('avg_goals')}")

    if current_section and section_lines and match:
        process_section_block(match, current_abbrev, current_section, section_lines)

    if match:
        store_current_match(match, matches)

    debug_log("=== 🔍 DEBUG: parse_text_data COMPLETE ===")
    debug_log(f"Total matches found: {len(matches)}")
    return {'league': league, 'matches': matches}

# ============================================
# PREDICTION LOGIC
# ============================================

def compute_draw_score(match):
    score = 0
    home_recent = match.get('home_recent', [])
    away_recent = match.get('away_recent', [])
    home_form = match.get('home_form', [])
    away_form = match.get('away_form', [])

    if len(home_recent) >= 6:
        draw_rate_home_recent = sum(1 for r in home_recent[-6:] if r == 'D') / 6
        if draw_rate_home_recent >= 0.33:
            score += 2
    if len(away_recent) >= 6:
        draw_rate_away_recent = sum(1 for r in away_recent[-6:] if r == 'D') / 6
        if draw_rate_away_recent >= 0.33:
            score += 2

    if len(home_form) >= 5:
        draw_rate_home_form = sum(1 for r in home_form[-5:] if r == 'D') / 5
        if draw_rate_home_form >= 0.4:
            score += 1
    if len(away_form) >= 5:
        draw_rate_away_form = sum(1 for r in away_form[-5:] if r == 'D') / 5
        if draw_rate_away_form >= 0.4:
            score += 1

    # We don't have overall stats yet, so skip goal averages for now
    return score

def decide_prediction(match):
    # Rule 1: Home Fortress
    home_form = match.get('home_form', [])
    if len(home_form) >= 5 and all(r != 'L' for r in home_form[-5:]):
        return {'prediction': '1', 'confidence': 'HIGH', 'stake': 2, 'rule': 'Home Fortress'}

    # Rule 2: Away Form Killer
    away_form = match.get('away_form', [])
    if len(away_form) >= 6 and sum(1 for r in away_form[-6:] if r == 'L') >= 4:
        return {'prediction': '1', 'confidence': 'HIGH', 'stake': 2, 'rule': 'Away Form Killer'}

    # Rule 3: Draw Score
    draw_score = compute_draw_score(match)
    if draw_score >= 7:
        return {'prediction': 'X', 'confidence': 'HIGH', 'stake': 2, 'rule': f'Draw Score ({draw_score})'}

    # Default: Trust Forebet
    forebet_pred = match.get('prediction')
    if forebet_pred:
        return {'prediction': forebet_pred, 'confidence': 'LOW', 'stake': 0.25, 'rule': 'Forebet Default'}
    else:
        return {'prediction': 'X', 'confidence': 'LOW', 'stake': 0.25, 'rule': 'Fallback Draw'}

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Forebet Parser & Predictor", layout="wide")
st.title("🐛 Refined Formula V1.1 - DEBUG MODE")
st.markdown("DEBUG MODE: Shows exactly where the parser fails and the prediction logic.")

with st.sidebar:
    st.header("📝 Paste Match Data")
    text_input = st.text_area("Paste Forebet data here", height=400)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📝 Input Data")
    st.text_area("Data to parse", value=text_input, height=300, key="display_text")

with col2:
    st.subheader("📊 Results")
    if st.button("🔍 Parse & Predict", type="primary"):
        st.session_state.debug_messages = []
        result = parse_text_data(text_input)

        with st.expander("🔍 DEBUG OUTPUT", expanded=True):
            debug_output = "\n".join(st.session_state.debug_messages)
            st.code(debug_output, language="")

        st.subheader("📊 Prediction Results")
        if result['matches']:
            for match in result['matches']:
                with st.container():
                    st.markdown(f"**{match['home_team']} vs {match['away_team']}**")
                    cols = st.columns(5)
                    cols[0].metric("Home %", f"{match['home_pct']}%")
                    cols[1].metric("Draw %", f"{match['draw_pct']}%")
                    cols[2].metric("Away %", f"{match['away_pct']}%")
                    cols[3].metric("Forebet Pred", match['prediction'])
                    decision = decide_prediction(match)
                    cols[4].metric("Our Prediction", decision['prediction'],
                                   help=f"Rule: {decision['rule']}\nConfidence: {decision['confidence']}\nStake: {decision['stake']} units")

                    if match['home_form']:
                        st.write(f"🏠 Home form (last 5): {', '.join(match['home_form'][:5])}")
                    if match['away_form']:
                        st.write(f"✈️ Away form (last 5): {', '.join(match['away_form'][:5])}")
                    if match['home_recent']:
                        st.write(f"📈 Recent form (last 6) - {match['home_team']}: {', '.join(match['home_recent'][:6])}")
                    if match['away_recent']:
                        st.write(f"📈 Recent form (last 6) - {match['away_team']}: {', '.join(match['away_recent'][:6])}")
                    score = compute_draw_score(match)
                    st.write(f"🎯 Draw Score: {score}/10")
                    st.divider()
        else:
            st.error("❌ No matches found.")

with st.expander("📋 Prediction Rules"):
    st.markdown("""
    1. **Home Fortress** – Home unbeaten in last 5 home → Home Win (HIGH, 2u)
    2. **Away Form Killer** – Away lost 4 of last 6 away → Home Win (HIGH, 2u)
    3. **Draw Score** – If score ≥ 7 → Draw (HIGH, 2u)
       - Factors: draw rates, goal similarity, low scoring, BTTS
    4. **Midweek Fatigue** – (not yet implemented)
    5. **Default** – Trust Forebet (LOW, 0.25u)
    """)
