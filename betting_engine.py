import streamlit as st
import re
from datetime import datetime

# ============================================
# FIXED PARSER FUNCTIONS
# ============================================

def parse_encoded_line(line):
    """
    Parse encoded line like:
    '284726X1 - 12.0034°3.00'
    or '323533X0 - 01.4124°3.50'
    
    Returns dict with all essential fields or None
    """
    st.debug(f"🔍 parse_encoded_line INPUT: '{line}'")
    
    # Clean and split
    if ' - ' not in line:
        st.debug("❌ No ' - ' separator found")
        return None
    
    left_part, right_part = line.split(' - ', 1)
    left_part = left_part.replace(" ", "")
    right_part = right_part.strip()
    
    st.debug(f"📊 Left: '{left_part}', Right: '{right_part}'")
    
    # Extract percentages, prediction, and home score
    # Pattern: 2 digits (home%) + 2 digits (draw%) + 2 digits (away%) + letter (prediction) + digit (home score)
    pattern = r"(\d{2})(\d{2})(\d{2})([12X])(\d)"
    m = re.match(pattern, left_part)
    
    if not m:
        st.debug(f"❌ Left part did not match pattern: {left_part}")
        return None
    
    home_pct = int(m.group(1))
    draw_pct = int(m.group(2))
    away_pct = int(m.group(3))
    pred = m.group(4)          # '1', 'X', or '2'
    score_home = int(m.group(5))   # Home score (single digit)
    
    st.debug(f"✅ Percentages: home={home_pct}, draw={draw_pct}, away={away_pct}, pred={pred}")
    st.debug(f"✅ Home score: {score_home}")
    
    # Parse right part for away score and avg goals
    # Pattern: integer_part + '.' + ... + '°' + avg_goals
    # Example: "12.0034°3.00" -> integer_part="12", avg_goals="3.00"
    pattern2 = r"(\d+)\.\d+°\s*([\d.]+)"
    m2 = re.search(pattern2, right_part)
    
    if not m2:
        st.debug(f"❌ Could not parse right part: '{right_part}'")
        return None
    
    integer_part = m2.group(1)   # e.g., "12" or "01"
    avg_goals = float(m2.group(2))  # e.g., 3.00 or 3.50
    
    # Away score = FIRST digit of the integer part
    # (Because "12" represents away score 1, "01" represents away score 0)
    score_away = int(integer_part[0])
    
    st.debug(f"✅ Integer part: {integer_part}, Away score: {score_away}")
    st.debug(f"✅ Average goals: {avg_goals}")
    
    # Validate percentages sum to ~100
    total = home_pct + draw_pct + away_pct
    if total < 95 or total > 105:
        st.debug(f"⚠️ Percentages sum to {total}% (unusual but continuing)")
    
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
    """
    # Pattern: [away_team][home_score] - [away_score] ( [ht_home] - [ht_away] ) [home_team]
    # This is a simplified version - full implementation would handle team name matching
    
    # Try to match pattern: team names with scores
    pattern = r"(.+?)\s*(\d+)\s*-\s*(\d+)\s*\(\s*(\d+)\s*-\s*(\d+)\s*\)\s*(.+)"
    m = re.match(pattern, line)
    
    if not m:
        return None
    
    # For now, just return the scores
    # The full implementation would need to validate team names match
    return {
        'score_home': int(m.group(2)),
        'score_away': int(m.group(3)),
        'ht_home': int(m.group(4)),
        'ht_away': int(m.group(5))
    }

def parse_text_data(text):
    """Main parser for Forebet text data"""
    
    st.debug("=== 🔍 DEBUG: parse_text_data STARTED ===")
    
    lines = text.split('\n')
    st.debug(f"✅ DEBUG: Total lines: {len(lines)}")
    
    # Extract league
    league = None
    for line in lines:
        if 'Brazil Serie D' in line:
            league = 'Brazil Serie D'
            break
        elif 'Czech Republic Chance Liga' in line:
            league = 'Czech Republic Chance Liga'
            break
        elif 'Serie D' in line:
            league = 'Serie D'
            break
        elif 'Chance Liga' in line:
            league = 'Chance Liga'
            break
    
    if league:
        st.debug(f"✅ DEBUG: League found: '{league}'")
    else:
        st.debug("❌ DEBUG: No league found")
    
    matches = []
    match = None
    h2h_matches = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Look for VS line
        if ' VS ' in line:
            st.debug(f"🔍 DEBUG: Found VS line at index {i}: '{line}'")
            home, away = line.split(' VS ', 1)
            home = home.strip()
            away = away.strip()
            st.debug(f"🔍 DEBUG: Home='{home}', Away='{away}'")
            
            # Start new match
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
            st.debug(f"✅ DEBUG: match_found = True")
        
        # Extract date (YYYY-MM-DD or DD/MM/YYYY)
        elif match:
            date_match = re.search(r'(\d{2}/\d{2}/\d{4})', line)
            if date_match:
                date_str = date_match.group(1)
                try:
                    # Convert DD/MM/YYYY to YYYY-MM-DD
                    day, month, year = date_str.split('/')
                    match['date'] = f"{year}-{month}-{day}"
                    st.debug(f"✅ DEBUG: Date found: {match['date']}")
                except:
                    pass
            
            # Parse encoded line
            if ' - ' in line and re.search(r'\d{6}[12X]\d', line.replace(' ', '')):
                st.debug(f"🔍 DEBUG: Found encoded line at index {i}: '{line}'")
                result = parse_encoded_line(line)
                if result:
                    match['home_pct'] = result['home_pct']
                    match['draw_pct'] = result['draw_pct']
                    match['away_pct'] = result['away_pct']
                    match['prediction'] = result['pred']
                    match['score_home'] = result['score_home']
                    match['score_away'] = result['score_away']
                    match['avg_goals'] = result['avg_goals']
                    st.debug("✅ DEBUG: Encoded line parsed successfully!")
                else:
                    st.debug("❌ DEBUG: parse_encoded_line FAILED")
            
            # Detect H2H section
            if 'Head to head' in line or 'Head-to-head' in line:
                st.debug(f"🔍 DEBUG: Found H2H section at index {i}")
                # Next lines might contain H2H data
                # This is simplified - full implementation would loop through following lines
            
            # Parse H2H matches (simplified for debug)
            if match and ' - ' in line and '(' in line and ')' in line:
                # Check if it looks like a score line
                if re.search(r'\d+\s*-\s*\d+\s*\(\s*\d+\s*-\s*\d+\s*\)', line):
                    st.debug(f"🔍 DEBUG: Potential H2H line at {i}: '{line}'")
                    h2h_result = parse_h2h_line(line, match['home_team'], match['away_team'])
                    if h2h_result:
                        match['h2h'].append(h2h_result)
                        st.debug(f"✅ DEBUG: Parsed H2H: {h2h_result}")
            
            # Check if match is complete and should be saved
            if match and match['home_pct'] is not None:
                # Check essential fields
                essential = ['home_pct', 'draw_pct', 'away_pct', 'prediction', 'score_home', 'score_away', 'avg_goals']
                missing = [f for f in essential if match.get(f) is None]
                
                if missing:
                    st.debug(f"❌ DEBUG: Missing essential data: {missing}")
                else:
                    st.debug("✅ DEBUG: All essential data present!")
                    # Save match
                    matches.append(match)
                    st.debug(f"✅ DEBUG: Match stored: {match['home_team']} vs {match['away_team']}")
                    # Start new match
                    match = None
            
            # Debug progress - show current state
            if match and i % 50 == 0:
                st.debug(f"🔍 DEBUG: Checking if match is complete...")
                st.debug(f"home_pct={match.get('home_pct')}")
                st.debug(f"draw_pct={match.get('draw_pct')}")
                st.debug(f"away_pct={match.get('away_pct')}")
                st.debug(f"prediction={match.get('prediction')}")
                st.debug(f"score_home={match.get('score_home')}")
                st.debug(f"score_away={match.get('score_away')}")
                st.debug(f"avg_goals={match.get('avg_goals')}")
    
    st.debug("=== 🔍 DEBUG: parse_text_data COMPLETE ===")
    st.debug(f"Total matches found: {len(matches)}")
    
    return {
        'league': league,
        'matches': matches
    }

# ============================================
# STREAMLIT APP
# ============================================

st.set_page_config(page_title="Forebet Parser", layout="wide")

st.title("🐛 Refined Formula V1.1 - DEBUG MODE")
st.markdown("DEBUG MODE: Shows exactly where the parser fails")

st.markdown("""
🔍 All parser steps are displayed - scroll up to see the full debug output

[🔮 Analyze](#) | [📝 Pending](#) | [📊 Records](#) | [📈 Dashboard](#)
""")

# Sidebar
with st.sidebar:
    st.header("📝 Paste Match Data")
    st.markdown("The debug output below will show you EXACTLY what the parser is doing")
    
    # Sample data text area
    sample_text = """Iguatu CE VS Nacional AM
Estádio Antônio Moreno de Melo 34°
Iguatu CE - LogoDWWDWW 
25/07/2026 20:00
X
Draw Probability47%
 
Nacional AM - LogoDLWDWW
1st place
Brazil Serie D
2nd place
1X2
 
Under/Over 2.5
 
Half Time
 
HT/FT
 
Btts
 
Handicap
 
Corners
 
Cards
 Home team
Away team 
Prob. %
1X2 Pred Correct score Avg. goals Weather conditions Coef.  Score  Live
coef.
Round 16, 1/8-finals
 Br4 
Iguatu CE
Nacional AM
25/07/2026 20:00
284726X1 - 12.0034°3.00 
 -
Head to head
All
18/07
2026
 Nacional AM1 - 1
(0 - 0)
 Iguatu CE Br4"""

    text_input = st.text_area(
        "Paste Forebet data here",
        value=sample_text,
        height=400
    )

    # Add second sample button
    if st.button("Load Second Sample (SK Lisen)"):
        second_sample = """SK Lisen VS Mlada Boleslav
Městský fotbalový stadion Srbská 24°
SK Lisen - LogoLWWLDW 
27/07/2026 17:00
X
Draw Probability35%
 
Mlada Boleslav - LogoLWLDDW
9th place
Czech Republic Chance Liga
8th place
1X2
 
Under/Over 2.5
 
Half Time
 
HT/FT
 
Btts
 
Handicap
 
Corners
 
Cards
 Home team
Away team 
Prob. %
1X2 Pred Correct score Avg. goals Weather conditions Coef.  Score  Live
coef.
Round 1, Regular Season
 Cz1 
SK Lisen
Mlada Boleslav
27/07/2026 17:00
323533X0 - 01.4124°3.50 
 -
Head to head
All
0
0%
Draw 0
0%
0
0%"""
        # Update the text area
        # Note: Streamlit doesn't allow direct manipulation of text_area value
        # So we use session_state
        st.session_state['input_text'] = second_sample
        st.rerun()

    # Initialize session state
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = sample_text

    # Use the session state value
    if 'input_text' in st.session_state and text_input != st.session_state['input_text']:
        # Only update if different
        if st.session_state['input_text']:
            # Can't directly update text_area, but we can store and use it
            # Workaround: use the session state as the default value
            text_input = st.session_state['input_text']

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Data")
    # Display the text that will be parsed
    st.text_area("Data to parse", value=text_input, height=300, key="display_text")

with col2:
    st.subheader("📊 Results")
    
    if st.button("🔍 Parse Data", type="primary"):
        # Clear previous debug
        st.session_state['debug_clear'] = True
        
        # Parse the data with debug
        with st.expander("🔍 DEBUG OUTPUT (scroll down for results)", expanded=True):
            # Redirect print to st.write for debug
            import sys
            from io import StringIO
            
            # Capture debug output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            # Parse
            result = parse_text_data(text_input)
            
            # Get debug output
            debug_output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Display debug
            st.code(debug_output, language="")
        
        # Display results
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
            
            # Show parsed match data
            st.subheader("📋 Parsed Match Data")
            st.json(result['matches'])
            
        else:
            st.error("❌ No matches found in the data.")
            st.info("Scroll up to see the DEBUG output - it will show exactly where the parser failed.")

# Show parser rules
with st.expander("📋 Parser Rules"):
    st.markdown("""
    ### Your 5 Refined Formula Rules:
    
    1. 🏰 **Home Fortress** - Unbeaten in last 5 home games → Back Home Win
    2. 💀 **Away Form Killer** - Lost 4 of last 6 away games → Back Home Win
    3. 🏆 **H2H Dominance** - 3 of last 4 H2Hs won → Draw is a trap
    4. 🤝 **H2H Draw Rate** - 4+ draws in last 6 H2Hs → Trust the Draw
    5. 😴 **Midweek Fatigue** - Away played 3-4 days ago → Downgrade away
    """)

# Instructions
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How to use:**
1. Copy Forebet data from their website
2. Paste it in the text area
3. Click "Parse Data"
4. Check debug output for parsing details
5. See results in the right panel

**Samples:**
- Click "Load Second Sample" to test with SK Lisen data
- Edit the text area to test with your own data
""")
