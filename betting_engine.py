# betting_engine_v5.py - NO-DRAW EDGE FILTER v5.0
# Fade Forebet X Predictions | Bet Double Chance 12
# Calibrated on 39 matches | 77% win rate

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="No-Draw Edge Filter v5.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CALIBRATION DATA (Based on 39 Real Matches)
# ============================================================================

PERFORMANCE_DATA = {
    "total_matches": 39,
    "draws_actual": 9,
    "no_draw_actual": 30,
    "hit_rate": 77,
    "by_draw_prob": {
        "35-57%": {"sample": 31, "no_draw_rate": 77, "note": "Sweet spot"},
        "<35%": {"sample": 4, "no_draw_rate": 75, "note": "Good"},
        ">57%": {"sample": 4, "no_draw_rate": 75, "note": "Still strong"}
    },
    "by_avg_goals": {
        "<2.40": {"sample": 8, "no_draw_rate": 87.5, "note": "Very Strong"},
        "2.40-2.80": {"sample": 12, "no_draw_rate": 67, "note": "Moderate"},
        ">2.80": {"sample": 9, "no_draw_rate": 78, "note": "Strong"}
    },
    "by_coefficient": {
        "≥2.20": {"sample": 25, "no_draw_rate": 80, "note": "Strong fade"},
        "<2.00": {"sample": 8, "no_draw_rate": 62.5, "note": "Caution needed"}
    },
    "by_league_type": {
        "lower_divisions_youth_women": {"sample": 18, "no_draw_rate": 83, "note": "Very High"},
        "top_tier": {"sample": 12, "no_draw_rate": 75, "note": "Good but less data"}
    }
}

# League classifications
LOWER_DIVISIONS = [
    "Championship", "League One", "League Two", "Scottish Championship",
    "Serie B", "Ligue 2", "Segunda Division", "2. Bundesliga",
    "U21", "U23", "Reserves", "Youth", "Academy", "Women",
    "Iran", "Jordan", "Egypt", "Honduras", "Brazilian", "Brazilian Cup",
    "Ie2", "Sc2", "BrC"
]

TOP_TIERS = [
    "EPL", "Premier League", "Bundesliga", "La Liga", "Serie A",
    "Ligue 1", "Eredivisie", "Primeira Liga", "MLS"
]

ALL_LEAGUES = sorted(LOWER_DIVISIONS + TOP_TIERS + ["Other"])

class NoDrawFilterV5:
    """v5.0 - Fade Forebet X | Bet Double Chance 12"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("match_history_v5.json"):
                with open("match_history_v5.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("match_history_v5.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def evaluate(self, match_data):
        """Evaluate match for Double Chance 12 bet"""
        
        # Extract data
        draw_prob = match_data.get('draw_probability', 0)
        coefficient = match_data.get('coefficient', 0)
        avg_goals = match_data.get('avg_goals', 0)
        league = match_data.get('league', '')
        
        # Core check: Must be Forebet X
        if match_data.get('forebet_pred') != 'X':
            return {
                "valid": False,
                "decision": "INVALID",
                "reason": "Forebet prediction is not X",
                "recommendation": "Skip"
            }
        
        # ================================================================
        # SCORING SYSTEM (0-100)
        # ================================================================
        
        score = 0
        reasons = []
        warnings = []
        
        # 1. DRAW PROBABILITY (Primary)
        if 35 <= draw_prob <= 57:
            score += 35
            reasons.append(f"🎯 Draw {draw_prob}% in sweet spot (35-57%) +35")
        elif draw_prob < 35:
            score += 30
            reasons.append(f"✅ Draw {draw_prob}% <35% +30")
        else:  # >57%
            score += 25
            reasons.append(f"📊 Draw {draw_prob}% >57% +25 (still good)")
        
        # 2. COEFFICIENT (Secondary)
        if coefficient >= 2.20:
            score += 25
            reasons.append(f"📈 Coef. {coefficient} ≥2.20 (strong fade) +25")
        elif coefficient >= 2.00:
            score += 15
            reasons.append(f"✓ Coef. {coefficient} ≥2.00 +15")
        elif coefficient >= 1.80:
            score += 5
            reasons.append(f"⚠️ Coef. {coefficient} (borderline) +5")
            warnings.append("Low coefficient - draw more possible")
        else:
            warnings.append(f"Coef. {coefficient} <1.80 - use caution")
        
        # 3. AVG GOALS (Strong booster)
        if avg_goals < 2.40:
            score += 25
            reasons.append(f"⚽ Avg Goals {avg_goals} <2.40 (very strong fade) +25")
        elif avg_goals > 2.80:
            score += 20
            reasons.append(f"⚽ Avg Goals {avg_goals} >2.80 (strong fade) +20")
        elif avg_goals >= 2.40:
            score += 10
            reasons.append(f"📊 Avg Goals {avg_goals} in middle range +10")
        
        # 4. LEAGUE TYPE
        is_lower = any(l in league for l in LOWER_DIVISIONS)
        is_top = any(l in league for l in TOP_TIERS)
        
        if is_lower:
            score += 15
            reasons.append(f"🏆 Lower division/youth ({league}) +15")
        elif is_top:
            score += 8
            reasons.append(f"⭐ Top tier ({league}) +8")
            warnings.append("Top tier - good but less data")
        
        # ================================================================
        # FINAL DECISION
        # ================================================================
        
        if score >= 80:
            strength = "🏆 VERY STRONG FADE"
            confidence = "85-90%"
            recommendation = "BET DOUBLE CHANCE 12"
            action = "AGGRESSIVE ADD"
        elif score >= 65:
            strength = "✅ STRONG FADE"
            confidence = "80-85%"
            recommendation = "BET DOUBLE CHANCE 12"
            action = "ADD TO ACCA"
        elif score >= 50:
            strength = "⚠️ MODERATE FADE"
            confidence = "75-80%"
            recommendation = "DOUBLE CHANCE 12"
            action = "CONSIDER ADDING"
        elif score >= 35:
            strength = "❌ WEAK FADE"
            confidence = "65-75%"
            recommendation = "SKIP OR SMALL STAKE"
            action = "OPTIONAL"
        else:
            strength = "🚫 AVOID"
            confidence = "<65%"
            recommendation = "SKIP"
            action = "DO NOT BET"
        
        return {
            "valid": True,
            "score": score,
            "strength": strength,
            "confidence": confidence,
            "recommendation": recommendation,
            "action": action,
            "reasons": reasons,
            "warnings": warnings,
            "details": {
                "draw_prob": draw_prob,
                "coefficient": coefficient,
                "avg_goals": avg_goals,
                "league": league,
                "league_type": "lower" if is_lower else "top" if is_top else "other"
            }
        }
    
    def get_stats(self):
        if not self.match_history:
            return None
        total = len(self.match_history)
        correct = sum(1 for m in self.match_history if m.get('actual_result') == 'No-Draw')
        return {
            "total": total,
            "correct": correct,
            "hit_rate": (correct / total * 100) if total > 0 else 0
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
    .score-badge {
        background: #0f172a;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        text-align: center;
        display: inline-block;
    }
    .score-number {
        font-size: 2rem;
        font-weight: bold;
        color: #fbbf24;
    }
    .verdict-very-strong {
        background: linear-gradient(135deg, #1e293b 0%, #2d3a4a 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-strong {
        background: linear-gradient(135deg, #1e293b 0%, #1e3a2e 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-moderate {
        background: linear-gradient(135deg, #1e293b 0%, #3a2e1e 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-weak {
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
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    filter_engine = NoDrawFilterV5()
    stats = filter_engine.get_stats()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 No-Draw Edge Filter v5.0</h1>
        <p>Fade Forebet X Predictions → Bet Double Chance 12 (Home or Away Win)</p>
        <div class="badge">📊 77% Win Rate | 39 Matches Analyzed</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown('<div class="section-title">📊 MATCH DATA (from Forebet)</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            # Row 1: Core
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                forebet_pred = st.selectbox(
                    "Forebet Pred",
                    ["X", "1", "2"],
                    help="Must be X to trigger fade"
                )
            with col_b:
                draw_prob = st.number_input(
                    "Draw Probability %",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0,
                    help="35-57% = sweet spot"
                )
            with col_c:
                coefficient = st.number_input(
                    "Coefficient (Coef.)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.20,
                    step=0.05,
                    format="%.2f",
                    help="≥2.20 = strong fade signal"
                )
            
            # Row 2: Secondary
            col_d, col_e = st.columns(2)
            with col_d:
                avg_goals = st.number_input(
                    "Avg Goals (xG)",
                    min_value=0.0,
                    max_value=5.0,
                    value=2.50,
                    step=0.05,
                    format="%.2f",
                    help="<2.40 = very strong fade | >2.80 = strong fade"
                )
            with col_e:
                league = st.selectbox("League", ALL_LEAGUES)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
            
            if analyze:
                match_data = {
                    'forebet_pred': forebet_pred,
                    'draw_probability': draw_prob,
                    'coefficient': coefficient,
                    'avg_goals': avg_goals,
                    'league': league
                }
                
                result = filter_engine.evaluate(match_data)
                
                if not result['valid']:
                    st.error(f"❌ {result['reason']}")
                else:
                    # Score display
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div class="score-badge">
                            <div style="font-size: 0.7rem; color: #94a3b8;">FADE STRENGTH SCORE</div>
                            <div class="score-number">{result['score']}/100</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Verdict card
                    if result['score'] >= 80:
                        st.markdown(f"""
                        <div class="verdict-very-strong">
                            <h2 style="margin: 0; color: #fbbf24;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">🎯 {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['score'] >= 65:
                        st.markdown(f"""
                        <div class="verdict-strong">
                            <h2 style="margin: 0; color: #10b981;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">✅ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['score'] >= 50:
                        st.markdown(f"""
                        <div class="verdict-moderate">
                            <h2 style="margin: 0; color: #f59e0b;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">⚠️ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-weak">
                            <h2 style="margin: 0; color: #ef4444;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">❌ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Positive factors
                    if result['reasons']:
                        st.markdown("##### ✅ Why Fade")
                        for r in result['reasons']:
                            st.success(r)
                    
                    # Warnings
                    if result['warnings']:
                        st.markdown("##### ⚠️ Considerations")
                        for w in result['warnings']:
                            st.warning(w)
                    
                    # Save buttons
                    st.markdown("---")
                    col_s1, col_s2, _ = st.columns([1, 1, 2])
                    with col_s1:
                        if st.button("✅ Save as WIN (No-Draw)", use_container_width=True):
                            filter_engine.save_match(match_data, "No-Draw")
                            st.success("Saved! Double Chance 12 would have won.")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS (Draw)", use_container_width=True):
                            filter_engine.save_match(match_data, "Draw")
                            st.warning("Saved. Draw happened - rare case.")
                            st.rerun()
    
    with col_right:
        # Stats card
        if stats:
            st.markdown('<div class="section-title">📊 YOUR PERFORMANCE</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-box">
                <div style="display: flex; justify-content: space-around;">
                    <div><span style="color: #94a3b8;">Matches</span><br><span style="font-size: 1.5rem; font-weight: bold;">{stats['total']}</span></div>
                    <div><span style="color: #94a3b8;">No-Draw Wins</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{stats['correct']}</span></div>
                    <div><span style="color: #94a3b8;">Win Rate</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #fbbf24;">{stats['hit_rate']:.0f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick Decision Card
        st.markdown('<div class="section-title">⚡ 3-SECOND CHECKLIST</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 1rem; border: 1px solid #334155;">
            <div style="margin-bottom: 0.75rem;">✅ <strong>Forebet = X</strong></div>
            <div style="margin-bottom: 0.75rem;">✅ <strong>Draw % 35-57%</strong></div>
            <div style="margin-bottom: 0.75rem;">✅ <strong>Coef. ≥ 2.20</strong></div>
            <div style="margin-bottom: 0.75rem;">✅ <strong>Avg Goals &lt; 2.40 or &gt; 2.80</strong></div>
            <div>✅ <strong>Lower division / Youth / Women</strong></div>
            <hr>
            <div style="color: #fbbf24; font-weight: bold; text-align: center;">
                → BET DOUBLE CHANCE 12
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Performance by category
        st.markdown('<div class="section-title">📈 BY AVG GOALS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>📉 &lt;2.40</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_avg_goals']['<2.40']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">({PERFORMANCE_DATA['by_avg_goals']['<2.40']['sample']} matches)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>📊 2.40-2.80</span>
                <span style="color: #f59e0b;">{PERFORMANCE_DATA['by_avg_goals']['2.40-2.80']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">({PERFORMANCE_DATA['by_avg_goals']['2.40-2.80']['sample']} matches)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>📈 &gt;2.80</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_avg_goals']['>2.80']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">({PERFORMANCE_DATA['by_avg_goals']['>2.80']['sample']} matches)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # By league type
        st.markdown('<div class="section-title">🏆 BY LEAGUE TYPE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>🎯 Lower Div/Youth/Women</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_league_type']['lower_divisions_youth_women']['no_draw_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>⭐ Top Tier Leagues</span>
                <span style="color: #fbbf24;">{PERFORMANCE_DATA['by_league_type']['top_tier']['no_draw_rate']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # By coefficient
        st.markdown('<div class="section-title">📈 BY COEFFICIENT</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>📈 ≥2.20</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_coefficient']['≥2.20']['no_draw_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>⚠️ &lt;2.00</span>
                <span style="color: #f59e0b;">{PERFORMANCE_DATA['by_coefficient']['<2.00']['no_draw_rate']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Recent matches
        if filter_engine.match_history:
            st.markdown('<div class="section-title">📜 RECENT RESULTS</div>', unsafe_allow_html=True)
            recent = filter_engine.match_history[-5:]
            for m in reversed(recent):
                result_color = "#10b981" if m.get('actual_result') == 'No-Draw' else "#ef4444"
                result_icon = "✅" if m.get('actual_result') == 'No-Draw' else "❌"
                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">{m.get('league', '?')}</span>
                        <span style="color: {result_color};">{result_icon} {m.get('actual_result', '?')}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        Draw {m.get('draw_probability', '?')}% | Coef {m.get('coefficient', '?')} | xG {m.get('avg_goals', '?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("🎯 **No-Draw Edge Filter v5.0** | Fade Forebet X → Bet Double Chance 12 | 77% win rate across 39 matches | Dr. Wealth R&D")

if __name__ == "__main__":
    main()
