# grokbet_v4.py - GROKBET LOGIC v4.0
# Based on 121+ matches | 68.12% win rate | 4-column filter system

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="GrokBet v4.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# GROKBET v4.0 - Based on 121+ Real Matches
# ============================================================================

PERFORMANCE = {
    "total_matches": 121,
    "raw_forebet_wins": 59,
    "raw_forebet_rate": 48.8,
    "v4_qualifiers": 69,
    "v4_wins": 47,
    "v4_win_rate": 68.12,
    "improvement": 19.36
}

class GrokBetV4:
    """GrokBet Logic v4.0 - 4-column filter system"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("grokbet_history.json"):
                with open("grokbet_history.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("grokbet_history.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def evaluate(self, match_data):
        """Apply v4.0 4-filter system"""
        
        pred = match_data.get('forebet_pred', '')
        prob_home = match_data.get('prob_home', 0)
        prob_draw = match_data.get('prob_draw', 0)
        prob_away = match_data.get('prob_away', 0)
        coef = match_data.get('coefficient', 0)
        avg_goals = match_data.get('avg_goals', 0)
        
        # ================================================================
        # RULE 1: Pred = 1 (Home Win)
        # ================================================================
        if pred == '1':
            # All filters must pass
            if prob_home >= 48 and coef >= 1.45 and avg_goals >= 2.5:
                return {
                    "valid": True,
                    "decision": "BET HOME WIN (1)",
                    "strength": "STRONG",
                    "expected": "68-72%",
                    "reasons": [
                        f"✅ Pred = 1 (Home Win)",
                        f"✅ Prob Home = {prob_home}% ≥ 48%",
                        f"✅ Coef. = {coef} ≥ 1.45",
                        f"✅ Avg Goals = {avg_goals} ≥ 2.5"
                    ],
                    "warnings": [],
                    "filters_passed": 4,
                    "filters_total": 4
                }
            else:
                failures = []
                if prob_home < 48:
                    failures.append(f"Prob Home {prob_home}% < 48%")
                if coef < 1.45:
                    failures.append(f"Coef. {coef} < 1.45")
                if avg_goals < 2.5:
                    failures.append(f"Avg Goals {avg_goals} < 2.5")
                return {
                    "valid": False,
                    "decision": "SKIP",
                    "strength": "INVALID",
                    "expected": None,
                    "reasons": [],
                    "warnings": failures,
                    "filters_passed": 4 - len(failures),
                    "filters_total": 4
                }
        
        # ================================================================
        # RULE 2: Pred = 2 (Away Win)
        # ================================================================
        elif pred == '2':
            # All filters must pass
            if prob_away >= 48 and coef >= 1.45 and avg_goals >= 2.5:
                return {
                    "valid": True,
                    "decision": "BET AWAY WIN (2)",
                    "strength": "STRONG",
                    "expected": "68-72%",
                    "reasons": [
                        f"✅ Pred = 2 (Away Win)",
                        f"✅ Prob Away = {prob_away}% ≥ 48%",
                        f"✅ Coef. = {coef} ≥ 1.45",
                        f"✅ Avg Goals = {avg_goals} ≥ 2.5"
                    ],
                    "warnings": [],
                    "filters_passed": 4,
                    "filters_total": 4
                }
            else:
                failures = []
                if prob_away < 48:
                    failures.append(f"Prob Away {prob_away}% < 48%")
                if coef < 1.45:
                    failures.append(f"Coef. {coef} < 1.45")
                if avg_goals < 2.5:
                    failures.append(f"Avg Goals {avg_goals} < 2.5")
                return {
                    "valid": False,
                    "decision": "SKIP",
                    "strength": "INVALID",
                    "expected": None,
                    "reasons": [],
                    "warnings": failures,
                    "filters_passed": 4 - len(failures),
                    "filters_total": 4
                }
        
        # ================================================================
        # RULE 3: Pred = X (Draw) → Flip to 12 (No Draw)
        # ================================================================
        elif pred == 'X':
            # All filters must pass
            if prob_draw <= 42 and coef >= 2.80 and avg_goals >= 2.5:
                return {
                    "valid": True,
                    "decision": "BET DOUBLE CHANCE 12 (No Draw)",
                    "strength": "STRONG",
                    "expected": "65-71%",
                    "reasons": [
                        f"✅ Pred = X (Draw) → Flipping to 12",
                        f"✅ Prob Draw = {prob_draw}% ≤ 42% (low confidence)",
                        f"✅ Draw Coef. = {coef} ≥ 2.80 (good value)",
                        f"✅ Avg Goals = {avg_goals} ≥ 2.5 (goals break draws)"
                    ],
                    "warnings": [],
                    "filters_passed": 4,
                    "filters_total": 4
                }
            else:
                failures = []
                if prob_draw > 42:
                    failures.append(f"Prob Draw {prob_draw}% > 42% (too confident in draw)")
                if coef < 2.80:
                    failures.append(f"Coef. {coef} < 2.80 (poor value)")
                if avg_goals < 2.5:
                    failures.append(f"Avg Goals {avg_goals} < 2.5")
                return {
                    "valid": False,
                    "decision": "SKIP",
                    "strength": "INVALID",
                    "expected": None,
                    "reasons": [],
                    "warnings": failures,
                    "filters_passed": 4 - len(failures),
                    "filters_total": 4
                }
        
        else:
            return {
                "valid": False,
                "decision": "SKIP",
                "strength": "INVALID",
                "expected": None,
                "reasons": [],
                "warnings": ["Invalid prediction"],
                "filters_passed": 0,
                "filters_total": 4
            }
    
    def get_stats(self):
        if not self.match_history:
            return None
        total = len(self.match_history)
        correct = sum(1 for m in self.match_history if m.get('actual_result') == 'Win')
        return {
            "total": total,
            "correct": correct,
            "win_rate": (correct / total * 100) if total > 0 else 0
        }

def main():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        color: #fbbf24;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        color: #94a3b8;
    }
    .badge {
        display: inline-block;
        background: #fbbf24;
        color: #0f172a;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
        margin-right: 0.5rem;
    }
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .result-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #334155;
        margin-top: 1rem;
    }
    .verdict-strong {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-skip {
        background: #1e293b;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stat-box {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    .section-title {
        color: #fbbf24;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .rule-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .filter-pass {
        color: #10b981;
        font-family: monospace;
    }
    .filter-fail {
        color: #ef4444;
        font-family: monospace;
    }
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    grokbet = GrokBetV4()
    stats = grokbet.get_stats()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 GrokBet v4.0</h1>
        <p>4-Filter System | 121+ Matches Analyzed | 68.12% Win Rate</p>
        <div>
            <span class="badge">📊 Raw Forebet: 48.8%</span>
            <span class="badge">🏆 v4.0: 68.12%</span>
            <span class="badge">📈 +19.36% Improvement</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown('<div class="section-title">📊 FOREBET MATCH DATA</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            # Row 1: Core inputs
            col1, col2, col3 = st.columns(3)
            with col1:
                forebet_pred = st.selectbox("Pred (Circled)", ["1", "X", "2"])
            with col2:
                prob_home = st.number_input("Prob % Home", 0.0, 100.0, 45.0, 1.0)
            with col3:
                prob_draw = st.number_input("Prob % Draw", 0.0, 100.0, 30.0, 1.0)
            
            # Row 2: More inputs
            col4, col5, col6 = st.columns(3)
            with col4:
                prob_away = st.number_input("Prob % Away", 0.0, 100.0, 25.0, 1.0)
            with col5:
                coefficient = st.number_input("Coef. (Odds)", 0.0, 10.0, 2.00, 0.05, format="%.2f")
            with col6:
                avg_goals = st.number_input("Avg Goals", 0.0, 5.0, 2.50, 0.05, format="%.2f")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze = st.button("🔍 ANALYZE WITH GROKBET v4.0", use_container_width=True, type="primary")
            
            if analyze:
                match_data = {
                    'forebet_pred': forebet_pred,
                    'prob_home': prob_home,
                    'prob_draw': prob_draw,
                    'prob_away': prob_away,
                    'coefficient': coefficient,
                    'avg_goals': avg_goals
                }
                
                result = grokbet.evaluate(match_data)
                
                if result['valid']:
                    st.markdown(f"""
                    <div class="verdict-strong">
                        <h2 style="margin: 0; color: #10b981;">✅ QUALIFIED BET</h2>
                        <p style="margin: 0.5rem 0; font-size: 1.2rem;">🎯 {result['decision']}</p>
                        <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['expected']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show filters passed
                    st.markdown("##### ✅ All 4 Filters Passed")
                    for reason in result['reasons']:
                        st.success(reason)
                    
                else:
                    st.markdown(f"""
                    <div class="verdict-skip">
                        <h2 style="margin: 0; color: #ef4444;">❌ SKIP THIS MATCH</h2>
                        <p style="margin: 0.5rem 0;">{result['decision']}</p>
                        <p style="margin: 0; color: #94a3b8;">Filters Passed: {result['filters_passed']}/{result['filters_total']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result['warnings']:
                        st.markdown("##### ❌ Failed Filters")
                        for w in result['warnings']:
                            st.error(w)
                
                # Save buttons (only if match was analyzed)
                st.markdown("---")
                col_s1, col_s2, _ = st.columns([1, 1, 2])
                with col_s1:
                    if st.button("✅ Save as WIN", use_container_width=True):
                        grokbet.save_match(match_data, "Win")
                        st.success("Saved as WIN!")
                        st.rerun()
                with col_s2:
                    if st.button("❌ Save as LOSS", use_container_width=True):
                        grokbet.save_match(match_data, "Loss")
                        st.warning("Saved as LOSS!")
                        st.rerun()
    
    with col_right:
        # Stats
        if stats:
            st.markdown('<div class="section-title">📊 YOUR PERFORMANCE</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-box">
                <div style="display: flex; justify-content: space-around;">
                    <div><span style="color: #94a3b8;">Bets</span><br><span style="font-size: 1.5rem; font-weight: bold;">{stats['total']}</span></div>
                    <div><span style="color: #94a3b8;">Wins</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{stats['correct']}</span></div>
                    <div><span style="color: #94a3b8;">Win Rate</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #fbbf24;">{stats['win_rate']:.1f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Decision Rules
        st.markdown('<div class="section-title">📋 GROKBET v4.0 RULES</div>', unsafe_allow_html=True)
        
        # Rule 1
        st.markdown("""
        <div class="rule-card">
            <strong style="color: #fbbf24;">PRED = 1 (Home Win)</strong><br>
            <span class="filter-pass">✓ Prob Home ≥ 48%</span><br>
            <span class="filter-pass">✓ Coef. ≥ 1.45</span><br>
            <span class="filter-pass">✓ Avg Goals ≥ 2.5</span><br>
            <span style="color: #94a3b8;">→ Bet HOME WIN (68-72%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Rule 2
        st.markdown("""
        <div class="rule-card">
            <strong style="color: #fbbf24;">PRED = 2 (Away Win)</strong><br>
            <span class="filter-pass">✓ Prob Away ≥ 48%</span><br>
            <span class="filter-pass">✓ Coef. ≥ 1.45</span><br>
            <span class="filter-pass">✓ Avg Goals ≥ 2.5</span><br>
            <span style="color: #94a3b8;">→ Bet AWAY WIN (68-72%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Rule 3
        st.markdown("""
        <div class="rule-card">
            <strong style="color: #fbbf24;">PRED = X (Draw) → FLIP</strong><br>
            <span class="filter-pass">✓ Prob Draw ≤ 42% (low confidence)</span><br>
            <span class="filter-pass">✓ Draw Coef. ≥ 2.80 (good value)</span><br>
            <span class="filter-pass">✓ Avg Goals ≥ 2.5</span><br>
            <span style="color: #94a3b8;">→ Bet DOUBLE CHANCE 12 (65-71%)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick Reference Table
        st.markdown('<div class="section-title">⚡ QUICK REFERENCE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rule-card">
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #334155;">
                    <th style="text-align: left; padding: 4px 0;">Pred</th>
                    <th style="text-align: left; padding: 4px 0;">Bet</th>
                    <th style="text-align: left; padding: 4px 0;">Key Filters</th>
                </tr>
                <tr>
                    <td style="padding: 4px 0;">1</td>
                    <td style="padding: 4px 0;">Home Win</td>
                    <td style="padding: 4px 0;">Prob≥48%, Coef≥1.45, xG≥2.5</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0;">2</td>
                    <td style="padding: 4px 0;">Away Win</td>
                    <td style="padding: 4px 0;">Prob≥48%, Coef≥1.45, xG≥2.5</td>
                </tr>
                <tr>
                    <td style="padding: 4px 0;">X</td>
                    <td style="padding: 4px 0;">12 (No Draw)</td>
                    <td style="padding: 4px 0;">Prob≤42%, Coef≥2.80, xG≥2.5</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance Stats
        st.markdown('<div class="section-title">📈 V4.0 PERFORMANCE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="rule-card">
            <div style="margin-bottom: 0.5rem;">📊 Total Matches Analyzed: <strong>{PERFORMANCE['total_matches']}</strong></div>
            <div style="margin-bottom: 0.5rem;">🎯 Raw Forebet: <strong>{PERFORMANCE['raw_forebet_rate']}%</strong></div>
            <div style="margin-bottom: 0.5rem;">🏆 v4.0 Qualifiers: <strong>{PERFORMANCE['v4_qualifiers']}</strong> ({PERFORMANCE['v4_qualifiers']/PERFORMANCE['total_matches']*100:.0f}% of matches)</div>
            <div style="margin-bottom: 0.5rem;">✅ v4.0 Wins: <strong>{PERFORMANCE['v4_wins']}</strong></div>
            <div style="margin-bottom: 0.5rem;">📈 v4.0 Win Rate: <strong style="color: #10b981;">{PERFORMANCE['v4_win_rate']}%</strong></div>
            <div>🚀 Improvement: <strong style="color: #fbbf24;">+{PERFORMANCE['improvement']}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Why This Works
        st.markdown('<div class="section-title">💡 WHY THIS WORKS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="rule-card">
            <div style="margin-bottom: 0.5rem;">1️⃣ <strong>High-confidence 1/2 calls</strong> (Prob≥48%) → ~57-60% raw</div>
            <div style="margin-bottom: 0.5rem;">2️⃣ <strong>Add Coef. filter</strong> (≥1.45) → avoids short odds traps</div>
            <div style="margin-bottom: 0.5rem;">3️⃣ <strong>Avg Goals ≥2.5</strong> → goals break draws, create winners</div>
            <div style="margin-bottom: 0.5rem;">4️⃣ <strong>Flip low-confidence X</strong> (Prob≤42%) → draws fail 71% of time</div>
            <div><strong>Result:</strong> 68.12% win rate across 69 qualifying bets</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recent matches
        if grokbet.match_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📜 RECENT RESULTS</div>', unsafe_allow_html=True)
            recent = grokbet.match_history[-5:]
            for m in reversed(recent):
                result_color = "#10b981" if m.get('actual_result') == 'Win' else "#ef4444"
                result_icon = "✅" if m.get('actual_result') == 'Win' else "❌"
                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">Pred {m.get('forebet_pred', '?')}</span>
                        <span style="color: {result_color};">{result_icon} {m.get('actual_result', '?')}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        Prob: {m.get('prob_home', '?')}/{m.get('prob_draw', '?')}/{m.get('prob_away', '?')} | 
                        Coef {m.get('coefficient', '?')} | xG {m.get('avg_goals', '?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **GrokBet v4.0** | 4-Filter System | 121+ matches | 68.12% win rate | +19.36% over raw Forebet | Data-driven from your screenshots")

if __name__ == "__main__":
    main()