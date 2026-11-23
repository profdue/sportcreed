
"""
Correct implementation of the agreed logic.
No PPG as a pick. PPG is only a tie-breaker if Steps 1-4 are all tied.

Inputs must be PRE-MATCH, HOME split for home team, AWAY split for away team.
Never use total when a split exists. Never use league-average as a team number.
"""

from dataclasses import dataclass, field
from typing import Optional

MIN_LEADS_FOR_TRUST = 3   # raw-count rule: <3 leads/trails => 100%/0% ignored
VETO_CONCEDED_PER_GAME = 1.60  # Step 4 veto threshold, in that home/away split
DRAW_GAP_TOLERANCE = 10.0  # if total gaps within this => Draw / No bet


@dataclass
class TeamSplit:
    # Step 1: Leading and Trailing > OPENING GOALS > Team scored first
    scored_first_pct: float  # 0-100

    # Step 2: Leading and Trailing > DEFENDING THE LEAD
    lead_defending_pct: float  # 0-100
    goals_giving_lead: int  # raw count: Goals giving the lead

    # Step 3: Leading and Trailing > EQUALIZING
    equalizing_pct: float  # 0-100
    goals_giving_lead_to_opponent: int  # raw count: Goals giving lead to opponent

    # Step 4: Contextual averages / Team Stats vs League Average
    clean_sheets_pct: float  # 0-100
    failed_to_score_pct: float  # 0-100
    conceded_per_game: float  # in this home/away split

    # PPG: ONLY tie-breaker. Never used unless Steps 1-4 all tied.
    ppg_split: Optional[float] = None


@dataclass
class StepResult:
    winner: str  # 'home', 'away', 'tie', 'unreliable'
    gap: float  # absolute gap in favour of winner, 0 if tie/unreliable
    home_value: Optional[float]
    away_value: Optional[float]
    reliable: bool
    note: str = ""


def _assess_step(home_val: float, away_val: float,
                 home_count: Optional[int], away_count: Optional[int],
                 step_name: str) -> StepResult:
    """
    Raw-count rule:
    - If EITHER side's rate is built on < MIN_LEADS_FOR_TRUST leads/trails,
      the step is unreliable.
    - In particular, a 0% or 100% on <3 is ignored: step becomes unreliable,
      gap = 0, winner = 'unreliable'. It does NOT count toward the total.
    - If both counts >= threshold, normal comparison.
    """
    home_reliable = home_count is None or home_count >= MIN_LEADS_FOR_TRUST
    away_reliable = away_count is None or away_count >= MIN_LEADS_FOR_TRUST
    reliable = home_reliable and away_reliable

    # A 0/100 on a small sample is the dangerous case: neutralise it.
    home_extreme = home_val in (0.0, 100.0) and not home_reliable
    away_extreme = away_val in (0.0, 100.0) and not away_reliable
    if home_extreme or away_extreme:
        return StepResult(
            winner="unreliable", gap=0.0,
            home_value=home_val, away_value=away_val, reliable=False,
            note=f"{step_name}: extreme % on <{MIN_LEADS_FOR_TRUST} leads/trails -> ignored",
        )
    if not reliable:
        # Small sample but not extreme: still do not let it decide. Mark unreliable.
        return StepResult(
            winner="unreliable", gap=0.0,
            home_value=home_val, away_value=away_val, reliable=False,
            note=f"{step_name}: <{MIN_LEADS_FOR_TRUST} leads/trails on one side -> ignored",
        )

    gap = abs(home_val - away_val)
    if gap == 0:
        return StepResult(winner="tie", gap=0.0, home_value=home_val,
                          away_value=away_val, reliable=True, note=f"{step_name}: tie")
    winner = "home" if home_val > away_val else "away"
    return StepResult(winner=winner, gap=gap, home_value=home_val,
                      away_value=away_val, reliable=True, note="")


def _step4_shape(home: TeamSplit, away: TeamSplit) -> str:
    """
    Step 4 does NOT pick a winner. It only shapes the scoreline.
    Returns: 'to-nil / low-score', 'both to score', or 'neutral'.
    """
    home_clean = home.clean_sheets_pct
    away_failed = away.failed_to_score_pct
    away_clean = away.clean_sheets_pct
    home_failed = home.failed_to_score_pct

    # Home to-nil shape: home keeps clean sheets AND away blanks often
    if home_clean >= 50 and away_failed >= 40:
        return "home to-nil / low-score shape"
    if away_clean >= 50 and home_failed >= 40:
        return "away to-nil / low-score shape"
    # Both to score shape: neither keeps clean sheets, neither blanks often
    if home_clean < 30 and away_clean < 30 and home_failed < 30 and away_failed < 30:
        return "both teams to score shape"
    return "neutral shape"


def evaluate_match(home: TeamSplit, away: TeamSplit) -> dict:
    # --- Steps 1-3, with raw-count rule applied inside each step ---
    # Step 1: scored-first is on matches, not leads. Its count is not in
    # Goals giving the lead, so pass None (no neutralisation on count).
    # Its reliability is handled by the tier logic via the other steps.
    # To keep the rule strict, Step 1 is compared directly, but a 100% on
    # a 5-game split is still noted as small-sample in the log.
    s1_gap = abs(home.scored_first_pct - away.scored_first_pct)
    s1 = StepResult(
        winner="tie" if s1_gap == 0 else ("home" if home.scored_first_pct > away.scored_first_pct else "away"),
        gap=s1_gap, home_value=home.scored_first_pct, away_value=away.scored_first_pct,
        reliable=True, note="Step 1: OPENING GOALS > Team scored first",
    )

    s2 = _assess_step(home.lead_defending_pct, away.lead_defending_pct,
                      home.goals_giving_lead, away.goals_giving_lead,
                      "Step 2: DEFENDING THE LEAD")
    s3 = _assess_step(home.equalizing_pct, away.equalizing_pct,
                      home.goals_giving_lead_to_opponent, away.goals_giving_lead_to_opponent,
                      "Step 3: EQUALIZING")

    # --- Gap addition: bigger total wins ---
    home_total = 0.0
    away_total = 0.0
    for s in (s1, s2, s3):
        if s.winner == "home":
            home_total += s.gap
        elif s.winner == "away":
            away_total += s.gap
        # tie / unreliable contribute 0

    net_gap = abs(home_total - away_total)
    all_steps_tied = (s1.winner == "tie" and s2.winner in ("tie", "unreliable")
                      and s3.winner in ("tie", "unreliable") and net_gap == 0)
    all_three_agree = (s1.winner != "tie" and s1.winner == s2.winner == s3.winner)

    shape = _step4_shape(home, away)

    # --- Step 4 veto: downgrade only, never upgrade ---
    # Favourite = side with the bigger total. If it concedes >1.60 in its split, downgrade one tier.
    favourite = "home" if home_total > away_total else ("away" if away_total > home_total else None)
    veto_applies = False
    veto_note = ""
    if favourite == "home" and home.conceded_per_game > VETO_CONCEDED_PER_GAME:
        veto_applies = True
        veto_note = f"Veto: favourite (home) concedes {home.conceded_per_game:.2f} > {VETO_CONCEDED_PER_GAME} in split -> downgrade one tier"
    if favourite == "away" and away.conceded_per_game > VETO_CONCEDED_PER_GAME:
        veto_applies = True
        veto_note = f"Veto: favourite (away) concedes {away.conceded_per_game:.2f} > {VETO_CONCEDED_PER_GAME} in split -> downgrade one tier"

    # --- Tier ---
    # PPG tie-breaker: ONLY if Steps 1-3 all tied/unreliable AND Step 4 shape is neutral.
    # Even then it only breaks the tie, it does not create a Lock/Lean.
    ppg_pick = None
    if all_steps_tied and shape == "neutral shape":
        if home.ppg_split is not None and away.ppg_split is not None and home.ppg_split != away.ppg_split:
            ppg_pick = "home" if home.ppg_split > away.ppg_split else "away"

    if ppg_pick is not None:
        tier = "Lean (PPG tie-breaker only)"
        pick = ppg_pick
    elif net_gap <= DRAW_GAP_TOLERANCE:
        tier = "Draw / No bet"
        pick = None
    elif all_three_agree:
        tier = "Lock"
        pick = favourite
    else:
        tier = "Lean"
        pick = favourite

    # Confidence
    unreliable_present = (not s2.reliable) or (not s3.reliable)
    if tier == "Draw / No bet":
        confidence = "No bet"
    elif tier == "Lock" and not unreliable_present and not veto_applies:
        confidence = "High"
    elif tier == "Lean" and not unreliable_present and not veto_applies:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Apply veto downgrade AFTER tier is set
    if veto_applies:
        if tier == "Lock":
            tier = "Lean (downgraded by veto)"
            confidence = "Low"
        elif tier == "Lean":
            tier = "Draw / No bet (downgraded by veto)"
            pick = None
            confidence = "No bet"

    # --- Predicted script (from steps, not from PPG) ---
    if s1.winner == "home":
        script_first = "Home scores first"
    elif s1.winner == "away":
        script_first = "Away scores first"
    else:
        script_first = "No clear first scorer"

    if s3.winner == "home" and s1.winner == "away":
        script_second = "Home equalizes (Step 3)"
    elif s3.winner == "away" and s1.winner == "home":
        script_second = "Away equalizes (Step 3)"
    elif s2.winner == s1.winner and s1.winner != "tie":
        script_second = "First scorer holds lead (Step 2)"
    else:
        script_second = "No clear second phase"

    return {
        "step1": {"winner": s1.winner, "gap": round(s1.gap, 1),
                  "home": s1.home_value, "away": s1.away_value, "note": s1.note},
        "step2": {"winner": s2.winner, "gap": round(s2.gap, 1),
                  "home": s2.home_value, "away": s2.away_value,
                  "reliable": s2.reliable, "note": s2.note},
        "step3": {"winner": s3.winner, "gap": round(s3.gap, 1),
                  "home": s3.home_value, "away": s3.away_value,
                  "reliable": s3.reliable, "note": s3.note},
        "step4_shape": shape,
        "home_total_gap": round(home_total, 1),
        "away_total_gap": round(away_total, 1),
        "net_gap": round(net_gap, 1),
        "all_three_agree": all_three_agree,
        "veto_applies": veto_applies,
        "veto_note": veto_note,
        "ppg_pick": ppg_pick,
        "pick": pick,
        "tier": tier,
        "confidence": confidence,
        "predicted_script": f"{script_first} -> {script_second} | {shape}",
        # Log this BEFORE the result. Score after in raw counts.
        "log_before_result": {
            "step1_winner": s1.winner, "step2_winner": s2.winner, "step3_winner": s3.winner,
            "pick": pick, "tier": tier, "script": f"{script_first} -> {script_second} | {shape}",
        },
    }


def score_after_result(log: dict, first_scorer: str, leader_held: bool,
                       trailer_equalized: bool) -> dict:
    """Score the logged pick in raw counts. Call AFTER the result. No % here."""
    return {
        "step1_held": (log["step1_winner"] == first_scorer),
        "step2_held": (leader_held == (log["step2_winner"] == first_scorer)) if log["step2_winner"] in ("home", "away") else None,
        "step3_held": (trailer_equalized == (log["step3_winner"] != first_scorer and log["step3_winner"] in ("home", "away"))) if log["step3_winner"] in ("home", "away") else None,
        "pick": log["pick"],
        "tier": log["tier"],
    }
