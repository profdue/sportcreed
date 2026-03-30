# betting_engine_v5.1.py - NO-DRAW EDGE FILTER v5.1
# Updated with 45 matches | 73.3% win rate
# New: Extreme low goals penalty | Youth league adjustment

import streamlit as st
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="No-Draw Edge Filter v5.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# UPDATED CALIBRATION (45 Matches: 39 original + 6 new)
# ============================================================================

PERFORMANCE_DATA = {
    "total_matches": 45,
    "draws_actual": 12,
    "no_draw_actual": 33,
    "hit_rate": 73.3,
    "by_draw_prob": {
        "35-57%": {"sample": 37, "no_draw_rate": 73, "note": "Sweet spot - consistent"},
        "<35%": {"sample": 4, "no_draw_rate": 75, "note": "Small sample"},
        ">57%": {"sample": 4, "no_draw_rate": 75, "note": "Still strong"}
    },
    "by_coefficient": {
        "≥3.00": {"sample": 12, "no_draw_rate": 75, "note": "Very strong fade"},
        "≥2.80": {"sample": 18, "no_draw_rate": 78, "note": "Strong fade"},
        "2.20-2.79": {"sample": 15, "no_draw_rate": 73, "note": "Good"},
        "<2.00": {"sample": 8, "no_draw_rate": 62.5, "note": "Caution needed"}
    },
    "by_avg_goals": {
        "<1.50": {"sample": 1, "no_draw_rate": 0, "note": "DANGER ZONE - extreme low"},
        "1.50-2.39": {"sample": 7, "no_draw_rate": 71, "note": "Good but watch extremes"},
        "2.40-2.80": {"sample": 12, "no_draw_rate": 67, "note": "Moderate"},
        ">2.80": {"sample": 9, "no_draw_rate": 78, "note": "Strong"}
    },
    "by_league_type": {
        "youth_u19": {"sample": 7, "no_draw_rate": 57, "note": "VOLATILE - use with caution"},
        "lower_divisions": {"sample": 21, "no_draw_rate": 76, "note": "Good"},
        "top_tier": {"sample": 12, "no_draw_rate": 75, "note": "Good but less data"}
    }
}

# Updated league classifications with youth caution
LOWER_DIVISIONS = [
    "Championship", "League One", "League Two", "Scottish Championship",
    "Serie B", "Ligue 2", "Segunda Division", "2. Bundesliga",
    "Reserves", "Women", "Vietnam", "Indonesia", "Turkey 2",
    "Iran", "Jordan", "Egypt", "Honduras", "Brazilian", "Brazilian Cup",
    "Ie2", "Sc2", "BrC"
]

YOUTH_LEAGUES = ["U19", "U21", "U23", "Youth", "Academy", "Primavera"]

TOP_TIERS = [
    "EPL", "Premier League", "Bundesliga", "La Liga", "Serie A",
    "Ligue 1", "Eredivisie", "Primeira Liga", "MLS"
]

ALL_LEAGUES = sorted(LOWER_DIVISIONS + YOUTH_LEAGUES + TOP_TIERS + ["Other"])

class NoDrawFilterV51:
    """v5.1 - Updated with new insights from 45 matches"""
    
    def __init__(self):
        self.match_history = []
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists("match_history_v51.json"):
                with open("match_history_v51.json", "r") as f:
                    self.match_history = json.load(f)
        except:
            self.match_history = []
    
    def save_match(self, match_data, result):
        self.match_history.append({
            "timestamp": datetime.now().isoformat(),
            **match_data,
            "actual_result": result
        })
        with open("match_history_v51.json", "w") as f:
            json.dump(self.match_history, f, indent=2)
    
    def evaluate(self, match_data):
        """Evaluate match for Double Chance 12 bet with updated rules"""
        
        draw_prob = match_data.get('draw_probability', 0)
        coefficient = match_data.get('coefficient', 0)
        avg_goals = match_data.get('avg_goals', 0)
        league = match_data.get('league', '')
        
        if match_data.get('forebet_pred') != 'X':
            return {
                "valid": False,
                "decision": "INVALID",
                "reason": "Forebet prediction is not X",
                "recommendation": "Skip"
            }
        
        score = 0
        reasons = []
        warnings = []
        
        # ================================================================
        # 1. DRAW PROBABILITY (Primary)
        # ================================================================
        if 35 <= draw_prob <= 57:
            score += 35
            reasons.append(f"🎯 Draw {draw_prob}% in sweet spot (35-57%) +35")
        elif draw_prob < 35:
            score += 30
            reasons.append(f"✅ Draw {draw_prob}% <35% +30")
        else:  # >57%
            score += 25
            reasons.append(f"📊 Draw {draw_prob}% >57% +25")
        
        # ================================================================
        # 2. COEFFICIENT (Stronger threshold based on new data)
        # ================================================================
        if coefficient >= 3.00:
            score += 30
            reasons.append(f"📈 Coef. {coefficient} ≥3.00 (very strong fade) +30")
        elif coefficient >= 2.80:
            score += 25
            reasons.append(f"📈 Coef. {coefficient} ≥2.80 (strong fade) +25")
        elif coefficient >= 2.20:
            score += 15
            reasons.append(f"✓ Coef. {coefficient} ≥2.20 +15")
        elif coefficient >= 2.00:
            score += 8
            reasons.append(f"⚠️ Coef. {coefficient} (borderline) +8")
            warnings.append("Lower coefficient - draw more possible")
        else:
            warnings.append(f"Coef. {coefficient} <2.00 - use caution")
        
        # ================================================================
        # 3. AVG GOALS (NEW: Extreme low penalty)
        # ================================================================
        if avg_goals > 0:  # Only if provided
            if avg_goals < 1.50:
                score -= 20  # Penalty for extreme low goals
                warnings.append(f"🚨 EXTREME LOW GOALS ({avg_goals}) - high draw risk (Vietnam 0-0 case) -20")
            elif avg_goals < 2.40:
                score += 20
                reasons.append(f"⚽ Avg Goals {avg_goals} <2.40 (strong fade) +20")
            elif avg_goals > 2.80:
                score += 18
                reasons.append(f"⚽ Avg Goals {avg_goals} >2.80 (strong fade) +18")
            elif avg_goals >= 2.40:
                score += 8
                reasons.append(f"📊 Avg Goals {avg_goals} in middle range +8")
        
        # ================================================================
        # 4. LEAGUE TYPE (NEW: Youth volatility penalty)
        # ================================================================
        is_youth = any(l in league for l in YOUTH_LEAGUES)
        is_lower = any(l in league for l in LOWER_DIVISIONS)
        is_top = any(l in league for l in TOP_TIERS)
        
        if is_youth:
            score += 10
            reasons.append(f"🏆 Youth league ({league}) +10")
            warnings.append("⚠️ Youth leagues are VOLATILE - 57% win rate in your data")
        elif is_lower:
            score += 18
            reasons.append(f"🏆 Lower division ({league}) +18")
        elif is_top:
            score += 10
            reasons.append(f"⭐ Top tier ({league}) +10")
        
        # ================================================================
        # FINAL SCORE & DECISION
        # ================================================================
        
        # Cap at 100
        score = min(100, max(0, score))
        
        if score >= 85:
            strength = "🏆 VERY STRONG FADE"
            confidence = "85-90%"
            recommendation = "BET DOUBLE CHANCE 12"
            action = "AGGRESSIVE ADD"
        elif score >= 70:
            strength = "✅ STRONG FADE"
            confidence = "80-85%"
            recommendation = "BET DOUBLE CHANCE 12"
            action = "ADD TO ACCA"
        elif score >= 55:
            strength = "⚠️ MODERATE FADE"
            confidence = "75-80%"
            recommendation = "DOUBLE CHANCE 12"
            action = "CONSIDER ADDING"
        elif score >= 40:
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
                "league_type": "youth" if is_youth else "lower" if is_lower else "top" if is_top else "other"
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
    .warning-badge {
        display: inline-block;
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.7rem;
        margin-left: 0.5rem;
    }
    .input-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid #334155;
        margin-bottom: 1rem;
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
    .danger-zone {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    filter_engine = NoDrawFilterV51()
    stats = filter_engine.get_stats()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 No-Draw Edge Filter v5.1</h1>
        <p>Fade Forebet X → Bet Double Chance 12 | Updated with 45 matches</p>
        <div class="badge">📊 73.3% Win Rate | 45 Matches Analyzed</div>
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
                coefficient = st.number_input("Coefficient (Coef.)", 0.0, 5.0, 2.80, 0.05, format="%.2f")
            
            col_d, col_e = st.columns(2)
            with col_d:
                avg_goals = st.number_input("Avg Goals (xG)", 0.0, 5.0, 2.50, 0.05, format="%.2f",
                    help="<1.50 = DANGER ZONE (high draw risk)")
            with col_e:
                league = st.selectbox("League", ALL_LEAGUES)
            
            # Danger zone warning
            if avg_goals < 1.50 and avg_goals > 0:
                st.markdown('<div class="danger-zone">🚨 EXTREME LOW GOALS DETECTED: Avg Goals <1.50 significantly increases draw risk. Vietnam 0-0 case in your data. Consider skipping.</div>', unsafe_allow_html=True)
            
            if any(l in league for l in ["U19", "U21", "Youth"]):
                st.markdown('<div class="danger-zone">⚠️ YOUTH LEAGUE NOTE: Your data shows 57% win rate in youth leagues (vs 73% overall). Higher variance than other leagues.</div>', unsafe_allow_html=True)
            
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
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <div class="score-badge">
                            <div style="font-size: 0.7rem; color: #94a3b8;">FADE STRENGTH SCORE</div>
                            <div class="score-number">{result['score']}/100</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if result['score'] >= 85:
                        st.markdown(f"""
                        <div class="verdict-very-strong">
                            <h2 style="margin: 0; color: #fbbf24;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">🎯 {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['score'] >= 70:
                        st.markdown(f"""
                        <div class="verdict-strong">
                            <h2 style="margin: 0; color: #10b981;">{result['strength']}</h2>
                            <p style="margin: 0.5rem 0; font-size: 1.1rem;">✅ {result['recommendation']}</p>
                            <p style="margin: 0; color: #94a3b8;">Projected Win Rate: {result['confidence']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif result['score'] >= 55:
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
                    
                    if result['reasons']:
                        st.markdown("##### ✅ Why Fade")
                        for r in result['reasons']:
                            st.success(r)
                    
                    if result['warnings']:
                        st.markdown("##### ⚠️ Considerations")
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
        
        st.markdown('<div class="section-title">⚡ UPDATED CHECKLIST (v5.1)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 1rem; border: 1px solid #334155;">
            <div style="margin-bottom: 0.5rem;">✅ <strong>Forebet = X</strong></div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Draw % 35-57%</strong></div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Coef. ≥ 2.80</strong> (stronger = better)</div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Avg Goals NOT &lt;1.50</strong> ⚠️</div>
            <div style="margin-bottom: 0.5rem;">✅ <strong>Lower division</strong> (preferred)</div>
            <div style="color: #f59e0b; margin-top: 0.5rem;">⚠️ <strong>Youth leagues</strong> = higher variance (57%)</div>
            <hr>
            <div style="color: #fbbf24; font-weight: bold; text-align: center;">
                → BET DOUBLE CHANCE 12
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">📈 UPDATED PERFORMANCE</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>📈 Coef. ≥3.00</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_coefficient']['≥3.00']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">({PERFORMANCE_DATA['by_coefficient']['≥3.00']['sample']} matches)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>⚽ Avg Goals >2.80</span>
                <span style="color: #10b981;">{PERFORMANCE_DATA['by_avg_goals']['>2.80']['no_draw_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span>⚽ Avg Goals 1.50-2.39</span>
                <span style="color: #f59e0b;">{PERFORMANCE_DATA['by_avg_goals']['1.50-2.39']['no_draw_rate']}%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>🚨 Avg Goals &lt;1.50</span>
                <span style="color: #ef4444;">{PERFORMANCE_DATA['by_avg_goals']['<1.50']['no_draw_rate']}%</span>
                <span style="color: #94a3b8;">DANGER ZONE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">⚠️ KEY INSIGHTS (from 07/03 batch)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #1e293b; border-radius: 12px; padding: 0.75rem; border: 1px solid #334155;">
            <div style="margin-bottom: 0.5rem;">🔴 <strong>Extreme low goals (&lt;1.50)</strong> = High draw risk</div>
            <div style="margin-bottom: 0.5rem;">🟡 <strong>Youth leagues</strong> = 57% win rate (volatile)</div>
            <div>🟢 <strong>Coef. ≥3.00</strong> = Still strong at 75%</div>
        </div>
        """, unsafe_allow_html=True)
        
        if filter_engine.match_history:
            st.markdown("<br>", unsafe_allow_html=True)
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
    
    st.markdown("---")
    st.caption("🎯 **No-Draw Edge Filter v5.1** | 45 matches | 73.3% win rate | Updated: Extreme low goals penalty + Youth league variance")

if __name__ == "__main__":
    main()
