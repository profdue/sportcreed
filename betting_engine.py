
### Decision Rules

| Draw Probability | Decision | Stake |
|------------------|----------|-------|
| **> 32%** | DRAW | 2 units |
| **< 28%** | NOT DRAW | 1 unit |
| **28-32%** | COIN FLIP | 0.1 unit |

### Data Required (Only 5-6 numbers)

1. Home Goals Scored Avg
2. Home Goals Conceded Avg
3. Away Goals Scored Avg
4. Away Goals Conceded Avg
5. League Draw Rate (select from dropdown)
6. Draw Odds (optional)

**No team names needed. No form. No streaks. No Forebet.**
""")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analyze", "📝 Pending", "📊 Records", "📈 Dashboard"])

with tab1:
st.markdown("### 📝 Input Match Data")

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["Manual Entry", "Paste Data (Auto-Extract)"],
    horizontal=True
)

if input_method == "Manual Entry":
    st.markdown("### Enter the 5 numbers directly")
    
    col1, col2 = st.columns(2)
    with col1:
        home_scored = st.number_input("🏠 Home Goals Scored Avg", min_value=0.0, max_value=5.0, value=1.1, step=0.01)
        home_conceded = st.number_input("🏠 Home Goals Conceded Avg", min_value=0.0, max_value=5.0, value=1.35, step=0.01)
    with col2:
        away_scored = st.number_input("✈️ Away Goals Scored Avg", min_value=0.0, max_value=5.0, value=1.7, step=0.01)
        away_conceded = st.number_input("✈️ Away Goals Conceded Avg", min_value=0.0, max_value=5.0, value=0.8, step=0.01)
    
    st.markdown("### League & Odds")
    col3, col4 = st.columns(2)
    with col3:
        league = st.selectbox("📊 League", list(LEAGUE_DRAW_RATES.keys()), index=0)
        league_draw_rate = LEAGUE_DRAW_RATES[league]
    with col4:
        draw_odds = st.number_input("🎯 Draw Odds (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.01, help="Enter 0 to skip market adjustment")
    
    home_team = st.text_input("🏠 Home Team Name (optional)", value="Home Team")
    away_team = st.text_input("✈️ Away Team Name (optional)", value="Away Team")
    
    if st.button("📊 Calculate Draw Probability", type="primary"):
        data = {
            "home_team": home_team,
            "away_team": away_team,
            "home_scored_avg": home_scored,
            "home_conceded_avg": home_conceded,
            "away_scored_avg": away_scored,
            "away_conceded_avg": away_conceded,
            "date": datetime.now().strftime("%d/%m/%Y"),
            "is_finished": False,
        }
        
        analysis = analyze_match(data, league_draw_rate, draw_odds if draw_odds > 0 else None)
        
        if analysis.get("verdict") != "SKIP":
            saved_id = save_to_db(data, analysis, league, draw_odds if draw_odds > 0 else None)
            if saved_id == "ALREADY_EXISTS":
                st.warning("This match already exists in the database.")
            elif saved_id:
                st.success(f"✅ Saved to database (ID: {saved_id})")
            
            display_analysis(data, analysis, league, False)

else:
    st.markdown("### Paste the data and the app will extract the numbers")
    
    text_data = st.text_area(
        "Paste match data here",
        height=300,
        placeholder="Paste the complete data including season statistics, tables, etc."
    )
    
    if text_data:
        with st.spinner("Extracting data..."):
            matches = extract_match_data(text_data)
        
        if matches:
            st.success(f"✅ Found {len(matches)} matches")
            
            league = st.selectbox("📊 League", list(LEAGUE_DRAW_RATES.keys()), index=0)
            league_draw_rate = LEAGUE_DRAW_RATES[league]
            draw_odds = st.number_input("🎯 Draw Odds (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.01)
            
            if st.button("📊 Analyze All Matches", type="primary"):
                for match in matches:
                    if match.get("is_finished"):
                        st.info(f"⏭️ {match['home_team']} vs {match['away_team']} — Already played (FT)")
                        continue
                    
                    analysis = analyze_match(match, league_draw_rate, draw_odds if draw_odds > 0 else None)
                    
                    if analysis.get("verdict") != "SKIP":
                        saved_id = save_to_db(match, analysis, league, draw_odds if draw_odds > 0 else None)
                        if saved_id == "ALREADY_EXISTS":
                            st.warning(f"⚠️ {match['home_team']} vs {match['away_team']} — Already exists")
                        elif saved_id:
                            st.success(f"✅ {match['home_team']} vs {match['away_team']} — Saved")
                        
                        display_analysis(match, analysis, league, saved_id == "ALREADY_EXISTS")
                    else:
                        st.info(f"⏭️ {match['home_team']} vs {match['away_team']} — Skipped")
        else:
            st.warning("No matches found in the data.")

with tab2:
st.subheader("📝 Pending Matches")
pending = get_pending()
if pending:
    st.write(f"**{len(pending)} pending result(s)**")
    for a in pending:
        ht = a.get('home_team', 'Home')
        at = a.get('away_team', 'Away')
        pred = a.get('predicted_decision', '?')
        confidence = a.get('prediction_confidence', '')
        draw_prob = a.get('draw_probability', 0)
        match_date = a.get('match_date', 'Date unknown')
        
        with st.expander(f"📅 {match_date} | {ht} vs {at} | {pred} ({confidence}) — Draw: {draw_prob:.1%}"):
            st.info(f"Prediction: {pred} ({confidence})")
            st.caption(f"Draw Probability: {draw_prob:.1%}")
            c1, c2 = st.columns(2)
            with c1:
                hg = st.number_input(f"{ht} Goals", 0, 15, 0, key=f"hg_{a['id']}")
            with c2:
                ag = st.number_input(f"{at} Goals", 0, 15, 0, key=f"ag_{a['id']}")
            if st.button("✅ Submit Result", key=f"sub_{a['id']}"):
                if submit_result(a['id'], hg, ag):
                    st.success("Result submitted!")
                    st.rerun()
else:
    st.info("No pending matches.")

with tab3:
st.subheader("📊 Performance Records")
results = get_results()
display_records_table(results)

with tab4:
st.subheader("📊 Dashboard")
results = get_results()
if not results:
    st.info("No results recorded yet.")
    return

total = len(results)
correct = 0

for r in results:
    if r.get('predicted_decision') and r.get('actual_decision'):
        if r['predicted_decision'] == r['actual_decision']:
            correct += 1

overall_rate = round(correct / total * 100) if total > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{total}</div><div class="stat-label">Total</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{overall_rate}%</div><div class="stat-label">Accuracy</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{correct}</div><div class="stat-label">Correct</div></div>', unsafe_allow_html=True)

# Distribution by draw probability range
st.markdown("#### Draw Probability Distribution")
ranges = {"< 20%": 0, "20-24%": 0, "24-28%": 0, "28-32%": 0, "32-36%": 0, "> 36%": 0}
correct_by_range = {"< 20%": 0, "20-24%": 0, "24-28%": 0, "28-32%": 0, "32-36%": 0, "> 36%": 0}

for r in results:
    prob = r.get('draw_probability', 0)
    if prob < 0.20:
        key = "< 20%"
    elif prob < 0.24:
        key = "20-24%"
    elif prob < 0.28:
        key = "24-28%"
    elif prob < 0.32:
        key = "28-32%"
    elif prob < 0.36:
        key = "32-36%"
    else:
        key = "> 36%"
    
    ranges[key] += 1
    if r.get('predicted_decision') == r.get('actual_decision'):
        correct_by_range[key] += 1

df_ranges = pd.DataFrame([
    {"Range": k, "Total": v, "Correct": correct_by_range[k], "Rate": f"{round(correct_by_range[k]/v*100) if v > 0 else 0}%"}
    for k, v in ranges.items() if v > 0
])
st.dataframe(df_ranges, use_container_width=True)


if __name__ == "__main__":
main()
