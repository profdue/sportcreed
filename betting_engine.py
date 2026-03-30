# betting_engine_v4.1.py - NO-DRAW EDGE FILTER v4.1
# Grounded Version | Fade Forebet X with MODERATE draw % only
# Realistic: 65-75% win rate | Not 85-90% nonsense

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="No-Draw Edge Filter v4.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# HONEST DATA - Based on ALL your matches (45 total)
# ============================================================================

PERFORMANCE_DATA = {
    "total_matches": 45,
    "draws_actual": 12,
    "no_draw_actual": 33,
    "hit_rate": 73.3,
    "by_draw_prob": {
        "≥45%": {
            "sample": 12,
            "no_draw_rate": 58,  # Only 58% no-draw when draw% high
            "note": "⚠️ DANGER: Draw often happens here",
            "action": "SKIP OR BET DRAW"
        },
        "37-44%": {
            "sample": 18,
            "no_draw_rate": 72,  # Sweet spot for fade
            "note": "✅ BEST ZONE for fade",
            "action": "CONSIDER DOUBLE CHANCE 12"
        },
        "≤36%": {
            "sample": 15,
            "no_draw_rate": 80,  # Small sample, looks promising
            "note": "✓ GOOD but small sample",
            "action": "CONSIDER DOUBLE CHANCE 12"
        }
    },
    "by_avg_goals": {
        "<2.0": {
            "sample": 8,
            "no_draw_rate": 62.5,
            "note": "⚠️ Low goals = draw risk"
        },
        "2.0-2.7": {
            "sample": 14,
            "no_draw_rate": 71,
            "note": "✓ Moderate zone"
        },
        ">2.7": {
            "sample": 9,
            "no_draw_rate": 78,
            "note": "✅ High goals = break draws"
        }
    },
    "by_favorite": {
        "clear_favorite_45%+": {
            "sample": 22,
            "no_draw_rate": 77,
            "note": "✅ Strong signal"
        },
        "even_match": {
            "sample": 23,
            "no_draw_rate": 70,
            "note": "✓ Still decent"
        }
    }
}

# League types (simplified)
LOWER_DIVISIONS = [
    "Championship", "League One", "League Two", "Scottish Championship",
    "Serie B", "Ligue 2", "Segunda Division", "2. Bundesliga",
    "Reserves", "Women", "Vietnam", "Indonesia", "Iran", "Jordan", 
    "Egypt", "Honduras", "Brazilian", "Ie2", "Sc2", "BrC"
]

YOUTH_LEAGUES = ["U19", "U21", "U23", "Youth", "Academy", "Primavera"]

TOP_TIERS = [
    "EPL", "Premier League", "Bundesliga", "La Liga", "Serie A",
    "Ligue 1", "Eredivisie", "Primeira Liga", "MLS"
]

ALL_LEAGUES = sorted(LOWER_DIVISIONS + YOUTH_LEAGUES + TOP_TIERS + ["Other"])

class NoDrawFilterV41:
    """v4.1 - Honest, grounded, no over-promising"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("match_history_v41.json"):
                with open("match_history_v41.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("match_history_v41.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def evaluate(self, match_data):
        """Honest evaluation based on actual data patterns"""
        
        draw_prob = match_data.get('draw_probability', 0)
        avg_goals = match_data.get('avg_goals', 0)
        home_prob = match_data.get('home_probability', 0)
        away_prob = match_data.get('away_probability', 0)
        league = match_data.get('league', '')
        
        # Core check
        if match_data.get('forebet_pred') != 'X':
            return {
                "valid": False,
                "decision": "INVALID",
                "reason": "Forebet prediction is not X",
                "recommendation": "Skip"
            }
        
        # ================================================================
        # SIMPLE POINTS SYSTEM (0-20)
        # No over-complicated scoring
        # ================================================================
        
        points = 0
        reasons = []
        warnings = []
        
        # 1. DRAW PROBABILITY (Most important)
        if 32 <= draw_prob <= 40:
            points += 4
            reasons.append(f"🎯 Draw {draw_prob}% in sweet spot (32-40%) +4")
        elif 41 <= draw_prob <= 44:
            points += 2
            reasons.append(f"📊 Draw {draw_prob}% (41-44%) +2")
        elif draw_prob >= 45:
            points -= 3
            warnings.append(f"⚠️ Draw {draw_prob}% ≥45% → draw often happens in your data. SKIP or bet DRAW. -3")
        elif draw_prob <= 31:
            points += 2
            reasons.append(f"✅ Draw {draw_prob}% ≤31% (low confidence) +2")
        
        # 2. AVG GOALS (Correct interpretation: HIGH goals = good for fade)
        if avg_goals > 0:
            if avg_goals >= 2.7:
                points += 2
                reasons.append(f"⚽ Avg Goals {avg_goals} ≥2.7 (goals break draws) +2")
            elif avg_goals < 2.0:
                points -= 2
                warnings.append(f"⚠️ Low Avg Goals {avg_goals} <2.0 → higher draw risk -2")
            elif avg_goals >= 2.0:
                points += 1
                reasons.append(f"✓ Avg Goals {avg_goals} (moderate) +1")
        
        # 3. CLEAR FAVORITE (One team has strong probability)
        max_prob = max(home_prob, away_prob)
        if max_prob >= 45:
            points += 3
            reasons.append(f"🏆 Clear favorite ({max_prob}% probability) +3")
        elif max_prob >= 40:
            points += 1
            reasons.append(f"📌 Moderate favorite ({max_prob}%) +1")
        
        # 4. LEAGUE CONTEXT (Minor factor)
        if any(l in league for l in YOUTH_LEAGUES):
            points += 0  # Youth leagues are volatile
            warnings.append("⚠️ Youth league - higher variance in your data")
        
        # ================================================================
        # HONEST DECISION (No 85-90% nonsense)
        # ================================================================
        
        if points >= 8:
            strength = "✅ GOOD FADE SIGNAL"
            expected_rate = "70-75%"
            recommendation = "Double Chance 12"
            action = "Consider for accumulator"
        elif points >= 5:
            strength = "⚠️ MODERATE SIGNAL"
            expected_rate = "65-70%"
            recommendation = "Double Chance 12 (cautious)"
            action = "Optional - lower stake"
        elif points >= 2:
            strength = "❌ WEAK SIGNAL"
            expected_rate = "55-65%"
            recommendation = "Skip or small stake"
            action = "Not recommended"
        else:
            strength = "🚫 AVOID"
            expected_rate = "<55%"
            recommendation = "SKIP"
            action = "Do not bet"
        
        # Special case: High draw % override
        if draw_prob >= 45:
            strength = "🚫 AVOID - HIGH DRAW %"
            expected_rate = "42% (draw happens 58% in your data)"
            recommendation = "BET DRAW or SKIP"
            action = "Do NOT bet Double Chance 12"
        
        return {
            "valid": True,
            "points": points,
            "strength": strength,
            "expected_rate": expected_rate,
            "recommendation": recommendation,
            "action": action,
            "reasons": reasons,
            "warnings": warnings,
            "details": {
                "draw_prob": draw_prob,
                "avg_goals": avg_goals,
                "favorite_prob": max_prob if max_prob >= 40 else None,
                "league": league
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
    .honest-badge {
        background: #ef4444;
        color: white;
    }
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .points-badge {
        background: #0f172a;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        text-align: center;
        display: inline-block;
    }
    .points-number {
        font-size: 2rem;
        font-weight: bold;
        color: #fbbf24;
    }
    .verdict-good {
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
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    hr {
        margin: 1rem 0;
        border-color: #334155;
    }
    </style>
    """, unsafe_allow_html=True)
    
    filter_engine = NoDrawFilterV41()
    stats = filter_engine.get_stats()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 No-Draw Edge Filter v4.1</h1>
        <p>Honest, Data-Driven | Fade Forebet X with MODERATE draw % only</p>
        <div class="badge">📊 73% Overall | 45 Matches</div>
        <div class="badge honest-badge" style="background: #ef4444;">⚠️ HIGH DRAW % (≥45%) = SKIP OR BET DRAW</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown('<div class="section-title">📊 MATCH DATA (from Forebet)</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                forebet_pred = st.selectbox("Forebet Pred", ["X", "1", "2"])
            with col_b:
                draw_prob = st.number_input("Draw Probability %", 0.0, 100.0, 40.0, 1.0)
            with col_c:
                avg_goals = st.number_input("Avg Goals (Coef.)", 0.0, 5.0, 2.50, 0.05, format="%.2f")
            
            col_d, col_e = st.columns(2)
            with col_d:
                home_prob = st.number_input("Home %", 0.0, 100.0, 35.0, 1.0)
            with col_e:
                away_prob = st.number_input("Away %", 0.0, 100.0, 35.0, 1.0)
            
            league = st.selectbox("League", ALL_LEAGUES)
            
            # Critical warnings
            if draw_prob >= 45:
                st.markdown(f"""
                <div class="warning-box">
                    🚨 <strong>CRITICAL: Draw probability {draw_prob}% ≥45%</strong><br>
                    In your data, when draw % is high, actual draw happens 58% of the time.<br>
                    <strong>Do NOT bet Double Chance 12 here. Bet DRAW or SKIP.</strong>
                </div>
                """, unsafe_allow_html=True)
            
            if 0 < avg_goals < 2.0:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>Low Avg Goals: {avg_goals}</strong><br>
                    Low-scoring games have higher draw risk. Your data shows only 62.5% no-draw rate.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze = st.button("🔍 ANALYZE MATCH", use_container_width=True, type="primary")
            
            if analyze:
                match_data = {
                    'forebet_pred': forebet_pred,
                    'draw_probability': draw_prob,
                    'avg_goals': avg_goals,
                    'home_probability': home_prob,
                    'away_probability': away_prob,
                    'league': league
                }
                
                result = filter_engine.evaluate(match_data)
                
                if not result['valid']:
                    st.error(f"❌ {result['reason']}")
                else:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div class="points-badge">
                            <div style="font-size: 0.7rem; color: #94a3b8;">FADE STRENGTH</div>
                            <div class="points-number">{result['points']}/12</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result['strength'].startswith("✅"):
                        st.markdown(f"""
                        <div class="verdict-good">
                            <h2 style="margin: 0; color: #10b981;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">🎯 {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Expected No-Draw Rate: {result['expected_rate']}</p>
                            <p style="margin: 0; color: #94a3b8;">{result['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['strength'].startswith("⚠️"):
                        st.markdown(f"""
                        <div class="verdict-moderate">
                            <h2 style="margin: 0; color: #f59e0b;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">⚠️ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Expected No-Draw Rate: {result['expected_rate']}</p>
                            <p style="margin: 0; color: #94a3b8;">{result['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-weak">
                            <h2 style="margin: 0; color: #ef4444;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">❌ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">{result['action']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if result['reasons']:
                        st.markdown("##### ✅ Factors")
                        for r in result['reasons']:
                            st.success(r)
                    
                    if result['warnings']:
                        st.markdown("##### ⚠️ Warnings")
                        for w in result['warnings']:
                            st.warning(w)
                    
                    st.markdown("---")
                    col_s1, col_s2, _ = st.columns([1, 1, 2])
                    with col_s1:
                        if st.button("✅ Save as WIN (No-Draw)", use_container_width=True):
                            filter_engine.save_match(match_data, "No-Draw")
                            st.success("Saved!")
                            st.rerun()
                    with col_s2:
                        if st.button("❌ Save as LOSS (Draw)", use_container_width=True):
                            filter_engine.save_match(match_data, "Draw")
                            st.warning("Saved!")
                            st.rerun()
    
    with col_right:
        if stats:
            st.markdown('<div class="section-title">📊 YOUR PERFORMANCE</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-box">
                <div style="display: flex; justify-content: space-around;">
                    <div><span style="color: #94a3b8;">Matches</span><br><span style="font-size: 1.5rem; font-weight: bold;">{stats['total']}</span></div>
                    <div><span style="color: #94a3b8;">No-Draw Wins</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{stats['correct']}</span></div>
                    <div><span style="color: #94a3b8;">Win Rate</span><br><span style="font-size: 1.5rem; font-weight: bold; color: #fbbf24;">{stats['hit_rate']:.1f}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">⚡ 3-SECOND CHECKLIST</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 1rem; border: 1px solid #334155;">
            <div style="margin-bottom: 0.5rem;">✅ <strong>Forebet = X</strong></div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Draw % ≤ 44%</strong> ← CRITICAL</div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Clear favorite (≥45%)</strong> OR <strong>Avg Goals ≥2.7</strong></div>
            <div style="color: #ef4444; margin-top: 0.5rem;">🚫 <strong>Draw % ≥45% = SKIP or BET DRAW</strong></div>
            <hr>
            <div style="color: #fbbf24; font-weight: bold; text-align: center;">
                → Double Chance 12 (No-Draw)
            </div>
            <div style="color: #94a3b8; font-size: 0.8rem; text-align: center; margin-top: 0.5rem;">
                Expected: 65-75% win rate
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📈 HONEST DATA BY DRAW %</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>📊 Draw ≥45%</span>
                <span style="color: #ef4444;">{PERFORMANCE_DATA['by_draw_prob']['≥45%']['no_draw_rate']}% no-draw</span>
                <span style="color: #94a3b8;">({PERFORMANCE_DATA['by_draw_prob']['≥45%']['sample']} matches)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>✅ Draw 37-44%</span>
                <span style="color: #fbbf24;">{PERFORMANCE_DATA['by_draw_prob']['37-44%']['no_draw_rate']}% no-draw</span>
                <span style="color: #94a3b8;">(BEST ZONE)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>✓ Draw ≤36%</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_draw_prob']['≤36%']['no_draw_rate']}% no-draw</span>
                <span style="color: #94a3b8;">(small sample)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📈 BY AVG GOALS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>⚠️ &lt;2.0</span>
                <span style="color: #f59e0b;">{PERFORMANCE_DATA['by_avg_goals']['<2.0']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">higher draw risk</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>✓ 2.0-2.7</span>
                <span style="color: #fbbf24;">{PERFORMANCE_DATA['by_avg_goals']['2.0-2.7']['no_draw_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>✅ &gt;2.7</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_avg_goals']['>2.7']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">goals break draws</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">💡 KEY INSIGHT</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <p style="margin: 0;"><strong>High draw % (≥45%) = Forebet believes in the draw</strong><br>
            Your data: only 58% no-draw when draw % high.<br>
            <strong>Do NOT fade high draw %.</strong></p>
            <p style="margin: 0.5rem 0 0 0;"><strong>Best fade spots:</strong><br>
            • Draw % 37-44% (72% no-draw)<br>
            • Clear favorite + Avg Goals ≥2.7</p>
        </div>
        """, unsafe_allow_html=True)
        
        if filter_engine.match_history:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">📜 RECENT RESULTS</div>', unsafe_allow_html=True)
            recent = filter_engine.match_history[-5:]
            for m in reversed(recent):
                result_color = "#10b981" if m.get('actual_result') == 'No-Draw' else "#ef4444"
                result_icon = "✅" if m.get('actual_result') == 'No-Draw' else "❌"
                draw_prob_display = m.get('draw_probability', '?')
                # Highlight if high draw % and was a draw
                if draw_prob_display >= 45 and m.get('actual_result') == 'Draw':
                    result_color = "#ef4444"
                    result_icon = "⚠️"
                st.markdown(f"""
                <div style="background: #0f172a; border-radius: 8px; padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #94a3b8;">{m.get('league', '?')}</span>
                        <span style="color: {result_color};">{result_icon} {m.get('actual_result', '?')}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #64748b;">
                        Draw {draw_prob_display}% | xG {m.get('avg_goals', '?')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("🎯 **No-Draw Edge Filter v4.1** | Honest, grounded version | 45 matches | 73% overall | Best zone: Draw 37-44% + favorite + high goals")

if __name__ == "__main__":
    main()
