import streamlit as st
import re
import json
from datetime import datetime
from supabase import create_client, Client
from collections import defaultdict

# ============================================================================
# SUPABASE SETUP
# ============================================================================
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Supabase connection failed: {e}")
    st.stop()

TABLE_NAME = "match_predictions"

# ============================================================================
# DEBUG LOGGING (stored in session state)
# ============================================================================

def debug_log(msg):
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []
    st.session_state.debug_messages.append(msg)

# ============================================================================
# PARSER CORE FUNCTIONS
# ============================================================================

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
        elif 'Superliga' in line:
            league = 'Serbia Superliga'
            break
        elif 'Divizia A' in line:
            league = 'Romania Divizia A'
            break
        elif 'Parva Liga' in line:
            league = 'Bulgaria Parva Liga'
            break

    if league:
        debug_log(f"✅ DEBUG: League found: '{league}'")
    else:
        debug_log("⚠️ DEBUG: No league found – will still try to parse")

    matches = []
    match = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # ---------- MATCH DETECTION (with improved validation) ----------
        if ' VS ' in line:
            parts = line.split(' VS ', 1)
            left = parts[0].strip()
            right = parts[1].strip()
            valid = True
            for side in (left, right):
                if not side:
                    valid = False
                    break
                if "'" in side or '"' in side or '(' in side or ')' in side:
                    valid = False
                    break
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
                continue

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
                    match = None

            # ---------- PERIODIC DEBUG ----------
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

# ============================================================================
# YOUR 5 RULES
# ============================================================================

def check_home_fortress(home_form: list) -> tuple:
    """
    Rule 1: Home Fortress
    Home team unbeaten in last 5 home games → Back Home Win
    """
    if len(home_form) < 5:
        return False, 0, f"Only {len(home_form)} home matches available (need 5)"
    
    recent = home_form[:5]
    unbeaten = sum(1 for r in recent if r != 'L')
    if unbeaten >= 5:
        return True, unbeaten, f"Unbeaten in last {unbeaten} home games"
    return False, unbeaten, f"Only {unbeaten}/5 unbeaten"

def check_away_form_killer(away_form: list) -> tuple:
    """
    Rule 2: Away Form Killer
    Away team lost 4 of last 6 away games → Back Home Win
    """
    if len(away_form) < 6:
        return False, 0, f"Only {len(away_form)} away matches available (need 6)"
    
    recent = away_form[:6]
    losses = sum(1 for r in recent if r == 'L')
    if losses >= 4:
        return True, losses, f"Lost {losses}/6 away games"
    return False, losses, f"Only {losses}/6 losses"

def check_h2h_dominance(h2h_data: list) -> tuple:
    """
    Rule 3: H2H Dominance
    One team won 3 of last 4 H2Hs → Draw is a trap
    """
    if len(h2h_data) < 4:
        return None, 0, 0, f"Only {len(h2h_data)} H2H matches available (need 4)"
    
    recent = h2h_data[:4]
    home_wins = 0
    away_wins = 0
    draws = 0
    
    for m in recent:
        if m.get('score_home', 0) > m.get('score_away', 0):
            home_wins += 1
        elif m.get('score_away', 0) > m.get('score_home', 0):
            away_wins += 1
        else:
            draws += 1
    
    if home_wins >= 3:
        return 'home', home_wins, draws, f"Home won {home_wins}/4 H2Hs → Draw is a trap"
    elif away_wins >= 3:
        return 'away', away_wins, draws, f"Away won {away_wins}/4 H2Hs → Draw is a trap"
    return None, max(home_wins, away_wins), draws, f"No dominance (H:{home_wins}, A:{away_wins}, D:{draws})"

def check_h2h_draw_rate(h2h_data: list) -> tuple:
    """
    Rule 4: H2H Draw Rate
    4+ draws in last 6 H2Hs → Trust the Draw
    """
    if len(h2h_data) < 6:
        return False, 0, f"Only {len(h2h_data)} H2H matches available (need 6)"
    
    recent = h2h_data[:6]
    draws = sum(1 for m in recent if m.get('score_home', 0) == m.get('score_away', 0))
    if draws >= 4:
        return True, draws, f"{draws}/6 H2Hs were draws → Trust the Draw"
    return False, draws, f"Only {draws}/6 draws"

def check_midweek_fatigue(team: str, match_date: str, fixtures: list) -> tuple:
    """
    Rule 5: Midweek Fatigue
    Away played 3-4 days ago → Downgrade away
    """
    if not match_date or not fixtures:
        return False, "No fixtures data"
    
    try:
        match_date_obj = datetime.strptime(match_date, "%Y-%m-%d").date()
    except:
        return False, "Invalid match date"
    
    for fixture in fixtures:
        if fixture.get('team') == team:
            fixture_date = fixture.get('date')
            if fixture_date:
                try:
                    if isinstance(fixture_date, str):
                        fixture_date_obj = datetime.strptime(fixture_date, "%Y-%m-%d").date()
                    else:
                        fixture_date_obj = fixture_date
                    days_diff = (match_date_obj - fixture_date_obj).days
                    if 3 <= days_diff <= 4:
                        return True, f"Played {days_diff} days ago"
                except:
                    pass
    return False, "No recent midweek fixture"

def get_stake_display(stake: str) -> tuple:
    stake_map = {
        "2 units": ("2 units", "HIGH"),
        "1.5 units": ("1.5 units", "MEDIUM"),
        "1 unit": ("1 unit", "MEDIUM"),
        "0.5 units": ("0.5 units", "LOW"),
        "0.25 units": ("0.25 units", "LOW"),
    }
    return stake_map.get(stake, (stake, "LOW"))

# ============================================================================
# REFINED FORMULA DECISION - YOUR 5 RULES
# ============================================================================

def refined_formula_decision(data: dict) -> dict:
    """
    Your 5 Rules:
    1. Home Fortress
    2. Away Form Killer
    3. H2H Dominance
    4. H2H Draw Rate
    5. Midweek Fatigue
    """
    
    home_team = data.get('home_team', 'Unknown')
    away_team = data.get('away_team', 'Unknown')
    match_date = data.get('date')
    forebet_pred = data.get('prediction', 'X')
    fixtures = data.get('midweek_fixtures', [])
    
    # For now, we don't have form data in the parser
    # This will be populated later from form sections
    home_form = data.get('home_form', [])
    away_form = data.get('away_form', [])
    h2h_data = data.get('h2h', [])
    
    home_fatigued, home_fatigue_msg = check_midweek_fatigue(home_team, match_date, fixtures)
    away_fatigued, away_fatigue_msg = check_midweek_fatigue(away_team, match_date, fixtures)
    
    # Rule 1: Home Fortress
    fortress, streak, msg = check_home_fortress(home_form)
    if fortress:
        return {
            'prediction': '1',
            'rule': 'Home Fortress',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Home Fortress']
        }
    
    # Rule 2: Away Form Killer
    killer, losses, msg = check_away_form_killer(away_form)
    if killer:
        return {
            'prediction': '1',
            'rule': 'Away Form Killer',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Home Win',
            'reason': msg,
            'rules_passed': ['Away Form Killer']
        }
    
    # Rule 3: H2H Dominance
    dominant, wins, draws, msg = check_h2h_dominance(h2h_data)
    if dominant:
        pred = '1' if dominant == 'home' else '2'
        winner = 'Home' if dominant == 'home' else 'Away'
        if (dominant == 'home' and home_fatigued) or (dominant == 'away' and away_fatigued):
            return {
                'prediction': 'X',
                'rule': 'H2H Dominance + Fatigue',
                'confidence': 'MEDIUM',
                'stake': '1 unit',
                'bet': 'Draw',
                'reason': f"{msg} but {winner} team fatigued → Draw",
                'rules_passed': ['H2H Dominance', 'Midweek Fatigue']
            }
        return {
            'prediction': pred,
            'rule': 'H2H Dominance',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': f"{winner} Win",
            'reason': msg,
            'rules_passed': ['H2H Dominance']
        }
    
    # Rule 4: H2H Draw Rate
    draw_rate, draws2, msg = check_h2h_draw_rate(h2h_data)
    if draw_rate:
        return {
            'prediction': 'X',
            'rule': 'H2H Draw Rate',
            'confidence': 'HIGH',
            'stake': '2 units',
            'bet': 'Draw',
            'reason': msg,
            'rules_passed': ['H2H Draw Rate']
        }
    
    # Rule 5: Midweek Fatigue
    if away_fatigued and not home_fatigued:
        return {
            'prediction': '1',
            'rule': 'Midweek Fatigue (Away)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Home Win',
            'reason': away_fatigue_msg,
            'rules_passed': ['Midweek Fatigue']
        }
    elif home_fatigued and not away_fatigued:
        return {
            'prediction': 'X',
            'rule': 'Midweek Fatigue (Home)',
            'confidence': 'MEDIUM',
            'stake': '1 unit',
            'bet': 'Draw',
            'reason': home_fatigue_msg,
            'rules_passed': ['Midweek Fatigue']
        }
    
    # Default: Trust Forebet
    bet_text = 'Home Win' if forebet_pred == '1' else 'Draw' if forebet_pred == 'X' else 'Away Win'
    return {
        'prediction': forebet_pred,
        'rule': 'Forebet Default',
        'confidence': 'LOW',
        'stake': '0.25 units',
        'bet': bet_text,
        'reason': 'No rules triggered → Trust Forebet',
        'rules_passed': ['Forebet Default']
    }

# ============================================================================
# DATABASE HELPERS
# ============================================================================

def save_prediction_to_db(data: dict) -> str:
    try:
        existing = supabase.table(TABLE_NAME).select("id")\
            .eq("home_team", data['home_team'])\
            .eq("away_team", data['away_team'])\
            .eq("match_date", data['match_date'])\
            .execute()
        if existing.data:
            return "ALREADY_EXISTS"
        
        response = supabase.table(TABLE_NAME).insert(data).execute()
        return str(response.data[0]['id']) if response.data else None
    except Exception as e:
        st.error(f"Failed to save: {e}")
        return None

def get_all_matches():
    try:
        response = supabase.table(TABLE_NAME).select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Error fetching matches: {e}")
        return []

# ============================================================================
# DISPLAY FUNCTION
# ============================================================================

def display_refined_analysis(match_data: dict, decision: dict, league: str = "Unknown"):
    home_team = match_data.get('home_team', 'Unknown')
    away_team = match_data.get('away_team', 'Unknown')
    home_pct = match_data.get('home_pct', '?')
    draw_pct = match_data.get('draw_pct', '?')
    away_pct = match_data.get('away_pct', '?')
    
    stake_display, confidence_level = get_stake_display(decision.get('stake', '0.25 units'))
    conf_color = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'gray'}.get(decision.get('confidence', 'LOW'), 'gray')
    pred_emoji = {'1': '🏠', 'X': '🤝', '2': '✈️'}.get(decision.get('prediction', 'X'), '🎯')
    
    st.subheader(f"{pred_emoji} {home_team} vs {away_team}")
    st.caption(f"📅 {match_data.get('date', '')} | 🏆 {league}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("🏠 Home", f"{home_pct}%")
    c2.metric("🤝 Draw", f"{draw_pct}%")
    c3.metric("✈️ Away", f"{away_pct}%")
    
    st.markdown("---")
    st.markdown(f"### 🎯 {decision.get('bet', 'Unknown')}")
    st.markdown(f"**Rule:** {decision.get('rule', 'Unknown')}")
    st.markdown(f"**Confidence:** :{conf_color}[{decision.get('confidence', 'LOW')}]")
    st.markdown(f"**Stake:** {stake_display}")
    st.markdown(f"**Reason:** {decision.get('reason', '')}")
    
    rules_passed = decision.get('rules_passed', ['Forebet Default'])
    st.caption(f"📋 Rules Passed: {', '.join(rules_passed)}")
    st.caption(f"📊 Forebet Original: {match_data.get('prediction', '?')} | Avg Goals: {match_data.get('avg_goals', '?')}")
    st.markdown("---")
    
    if match_data.get('h2h'):
        with st.expander("📊 Head-to-Head History"):
            h2h_df = pd.DataFrame(match_data['h2h'])
            st.dataframe(h2h_df, use_container_width=True)

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Forebet Parser", layout="wide")

st.title("🐛 Refined Formula V1.1 - DEBUG MODE")
st.markdown("DEBUG MODE: Shows exactly where the parser fails")
st.markdown("🔍 All parser steps are displayed - scroll up to see the full debug output")

# Sidebar with paste area
with st.sidebar:
    st.header("📝 Paste Match Data")
    st.markdown("The debug output below will show you EXACTLY what the parser is doing")
    text_input = st.text_area("Paste Forebet data here", height=400)
    
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Copy Forebet data from their website
    2. Paste it in the text area
    3. Click "Parse Data"
    4. Check debug output for parsing details
    5. See results in the right panel
    """)

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Data Preview")
    st.text_area("Data to parse (first 500 chars)", 
                 value=text_input[:500] if text_input else "", 
                 height=200, 
                 key="display_text",
                 disabled=True)

with col2:
    st.subheader("📊 Results")
    if st.button("🔍 Parse & Analyze", type="primary"):
        # Clear previous debug messages
        st.session_state.debug_messages = []
        
        # Parse the data
        parsed = parse_text_data(text_input)
        matches = parsed.get('matches', [])
        league = parsed.get('league', 'Unknown')
        
        # Show debug output
        with st.expander("🔍 DEBUG OUTPUT (scroll down for results)", expanded=True):
            debug_output = "\n".join(st.session_state.debug_messages)
            st.code(debug_output, language="")
        
        # Show results
        st.subheader("📊 Parser Results")
        if matches:
            st.success(f"✅ Found {len(matches)} matches!")
            
            # Track saves
            saved_count = 0
            duplicate_count = 0
            
            for i, match in enumerate(matches, 1):
                st.markdown(f"**Match {i}: {match['home_team']} vs {match['away_team']}**")
                
                # Show match data
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
                
                # Apply refined formula
                decision = refined_formula_decision(match)
                display_refined_analysis(match, decision, league)
                
                # Auto-save to database
                h2h_json = json.dumps(match.get('h2h', []))
                
                # Calculate H2H stats
                h2h_data = match.get('h2h', [])
                h2h_dominance = None
                h2h_dominance_count = 0
                h2h_draw_count = 0
                
                if len(h2h_data) >= 4:
                    home_wins = sum(1 for m in h2h_data[:4] if m.get('score_home', 0) > m.get('score_away', 0))
                    away_wins = sum(1 for m in h2h_data[:4] if m.get('score_away', 0) > m.get('score_home', 0))
                    if home_wins >= 3:
                        h2h_dominance = 'home'
                        h2h_dominance_count = home_wins
                    elif away_wins >= 3:
                        h2h_dominance = 'away'
                        h2h_dominance_count = away_wins
                
                if len(h2h_data) >= 6:
                    h2h_draw_count = sum(1 for m in h2h_data[:6] if m.get('score_home', 0) == m.get('score_away', 0))
                
                db_data = {
                    'match_date': match.get('date', datetime.now().date()),
                    'league_name': league if league else 'Unknown',
                    'home_team': match.get('home_team', 'Unknown'),
                    'away_team': match.get('away_team', 'Unknown'),
                    'season_round': None,
                    'forebet_home_pct': match.get('home_pct', 0),
                    'forebet_draw_pct': match.get('draw_pct', 0),
                    'forebet_away_pct': match.get('away_pct', 0),
                    'forebet_prediction': match.get('prediction', 'X'),
                    'forebet_correct_score_home': match.get('score_home'),
                    'forebet_correct_score_away': match.get('score_away'),
                    'forebet_avg_goals': match.get('avg_goals', 2.5),
                    'forebet_double_chance': None,
                    'home_form': '',
                    'away_form': '',
                    'h2h_data': h2h_json,
                    'h2h_dominance': h2h_dominance,
                    'h2h_dominance_count': h2h_dominance_count,
                    'h2h_draw_count': h2h_draw_count,
                    'midweek_fatigue_home': False,
                    'midweek_fatigue_away': False,
                    'refined_prediction': decision['prediction'],
                    'refined_rule_triggered': decision['rule'],
                    'refined_confidence': decision['confidence'],
                    'refined_bet': decision['bet'],
                    'refined_stake': decision['stake'],
                    'refined_reason': decision['reason'],
                    'rules_passed': decision.get('rules_passed', [])
                }
                
                result = save_prediction_to_db(db_data)
                if result == "ALREADY_EXISTS":
                    duplicate_count += 1
                    st.warning(f"⚠️ Match {i} already exists (skipped)")
                elif result:
                    saved_count += 1
                    st.success(f"✅ Match {i} saved with ID: {result}")
                else:
                    st.error(f"❌ Failed to save match {i}")
                
                st.divider()
            
            st.info(f"📊 Summary: {saved_count} saved, {duplicate_count} duplicates skipped.")
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

# Import pandas for display
import pandas as pd
