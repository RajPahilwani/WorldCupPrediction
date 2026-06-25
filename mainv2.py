#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
  ADVANCED SOCCER PREDICTION MODEL  v4.0
  Fixes:
    • opponent-adjusted rolling attack/defense strengths
    • no direct attack/defense division explosion
    • weaker, time-decayed head-to-head
    • fixed mirror augmentation bug
    • more honest evaluation metrics
    • larger holdout split
    • score NLL added alongside outcome accuracy
══════════════════════════════════════════════════════════════════════════
"""

import math
import os
import warnings
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_GOALS = 9
RECENCY_HALF_LIFE_DAYS = 365.0
ENSEMBLE_CV = 5
HOLDOUT_FRACTION = 0.20
MIN_HOLDOUT_MATCHES = 30

DEFAULT_ELO = 1500.0
DEFAULT_LEAGUE_GOALS_PER_TEAM = 1.35
DEFAULT_ATTACK_STRENGTH = 1.0
DEFAULT_DEFENSE_WEAKNESS = 1.0
DEFAULT_FORM = 50.0
DEFAULT_CONSISTENCY = 1.0

STRENGTH_SMOOTHING = 3.0
H2H_HALF_LIFE_DAYS = 1825.0  # 5 years
EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = ["Team 1", "Team 2", "Goals1", "Goals2", "Elo1", "Elo2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "Stage" not in df.columns:
        df["Stage"] = "Group"

    df["Date"] = pd.to_datetime(df.get("Date", pd.NaT), errors="coerce")

    for col in ["Goals1", "Goals2", "Elo1", "Elo2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Goals1", "Goals2", "Elo1", "Elo2"]).copy()
    df["Goals1"] = df["Goals1"].astype(int)
    df["Goals2"] = df["Goals2"].astype(int)

    df = df.sort_values("Date", kind="mergesort", na_position="last").reset_index(drop=True)
    df["match_idx"] = np.arange(len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Builds features using only past data.

    Main change from the old version:
    - rolling attack/defense are opponent-adjusted and normalized
    - xG prior is multiplicative, not attack/defense division
    - H2H is time-decayed and heavily damped
    """

    def __init__(self, window: int = 8):
        self.window = window

        self._team_goals_scored: Dict[str, List[float]] = defaultdict(list)
        self._team_goals_conceded: Dict[str, List[float]] = defaultdict(list)
        self._team_adj_for: Dict[str, List[float]] = defaultdict(list)
        self._team_adj_against: Dict[str, List[float]] = defaultdict(list)
        self._team_elo_history: Dict[str, List[float]] = defaultdict(list)
        self._team_results: Dict[str, List[float]] = defaultdict(list)  # 3=W,1=D,0=L
        self._team_match_count: Dict[str, int] = defaultdict(int)
        self._team_opponent_elos: Dict[str, List[float]] = defaultdict(list)
        self._league_goal_history: List[float] = []
        # key -> list of (team_a, team_b, goals_a, goals_b, date)
        self._h2h: Dict[Tuple[str, str], List[Tuple[str, str, int, int, Optional[pd.Timestamp]]]] = defaultdict(list)

    def _weighted_mean(self, vals: List[float], n: int = None, smoothing: float = STRENGTH_SMOOTHING) -> float:
        n = n or self.window
        if not vals:
            return 0.0
        tail = np.asarray(vals[-n:], dtype=float)
        if len(tail) == 0:
            return 0.0
        return float((tail.sum() + smoothing * float(np.mean(tail))) / (len(tail) + smoothing))

    def _rolling_mean(self, vals: List[float], n: int = None) -> float:
        n = n or self.window
        if not vals:
            return 0.0
        return float(np.mean(vals[-n:]))

    def _rolling_std(self, vals: List[float], n: int = None) -> float:
        n = n or self.window
        if len(vals) < 2:
            return 0.5
        return float(np.std(vals[-n:]))

    def _weighted_form(self, results: List[float]) -> float:
        if not results:
            return DEFAULT_FORM
        r = results[-self.window:]
        weights = np.exp(np.linspace(-1.5, 0.0, len(r)))
        return float(np.average(r, weights=weights) / 3.0 * 100.0)

    def _league_avg_goals_per_team(self) -> float:
        if not self._league_goal_history:
            return DEFAULT_LEAGUE_GOALS_PER_TEAM
        return float(np.mean(self._league_goal_history[-self.window:]))

    def _team_strengths(self, team: str) -> dict:
        """
        Returns current team state built only from past matches.
        attack_strength > 1 means stronger attack than average.
        defense_weakness > 1 means weaker defense than average.
        """
        gs = self._team_goals_scored[team]
        gc = self._team_goals_conceded[team]
        adj_for = self._team_adj_for[team]
        adj_against = self._team_adj_against[team]
        elo_hist = self._team_elo_history[team]
        results = self._team_results[team]
        opp_elos = self._team_opponent_elos[team]
        mc = self._team_match_count[team]

        league_avg = self._league_avg_goals_per_team()

        raw_attack = self._weighted_mean(gs)
        raw_defense = self._weighted_mean(gc)

        attack_strength = self._weighted_mean(adj_for) / max(league_avg, 0.1) if adj_for else DEFAULT_ATTACK_STRENGTH
        defense_weakness = self._weighted_mean(adj_against) / max(league_avg, 0.1) if adj_against else DEFAULT_DEFENSE_WEAKNESS

        elo_last = elo_hist[-1] if elo_hist else DEFAULT_ELO
        elo_trend = (elo_hist[-1] - elo_hist[-5]) if len(elo_hist) >= 5 else 0.0
        elo_trend3 = (elo_hist[-1] - elo_hist[-3]) if len(elo_hist) >= 3 else 0.0
        consistency = 1.0 / (self._rolling_std(gs) + 0.5)

        clean_sheet = sum(1 for g in gc[-self.window:] if g == 0) / max(len(gc[-self.window:]), 1)
        btts = sum(1 for g1, g2 in zip(gs[-self.window:], gc[-self.window:]) if g1 > 0 and g2 > 0)
        btts /= max(len(gs[-self.window:]), 1)

        form5 = sum(results[-5:]) / (3.0 * max(len(results[-5:]), 1)) * 100.0
        form_w = self._weighted_form(results)
        opp_elo_avg = self._rolling_mean(opp_elos)

        return {
            "elo": elo_last,
            "elo_trend": elo_trend,
            "elo_trend3": elo_trend3,
            "raw_attack": raw_attack,
            "raw_defense": raw_defense,
            "attack_strength": attack_strength,
            "defense_weakness": defense_weakness,
            "defense_strength": 1.0 / max(defense_weakness, 0.25),
            "consistency": consistency,
            "clean_sheet_rate": clean_sheet,
            "form_last5": form5,
            "form_weighted": form_w,
            "btts_rate": btts,
            "match_count": mc,
            "opp_elo_avg": opp_elo_avg,
        }

    def _h2h_stats(self, t1: str, t2: str, current_date: Optional[pd.Timestamp] = None) -> Tuple[float, float, float, int]:
        """
        Returns time-decayed (w, d, l, n) from t1 perspective.
        Uses a damped prior so sparse H2H does not dominate.
        """
        key = tuple(sorted([t1, t2]))
        history = self._h2h.get(key, [])
        if not history:
            return 1 / 3, 1 / 3, 1 / 3, 0

        w = d = l = 0.0
        total_w = 0.0

        for a, b, g_a, g_b, dt in history:
            if a == t1 and b == t2:
                s1, s2 = g_a, g_b
            else:
                s1, s2 = g_b, g_a

            if current_date is not None and pd.notna(current_date) and dt is not None and pd.notna(dt):
                age_days = max((current_date - dt).days, 0)
                weight = math.exp(-age_days / H2H_HALF_LIFE_DAYS)
            else:
                weight = 1.0

            total_w += weight
            if s1 > s2:
                w += weight
            elif s1 == s2:
                d += weight
            else:
                l += weight

        # damping prior keeps signal small and stable
        prior = 1.0
        denom = total_w + 3.0 * prior
        return (w + prior) / denom, (d + prior) / denom, (l + prior) / denom, len(history)

    def _core_features(
        self,
        t1: str,
        t2: str,
        elo1: float,
        elo2: float,
        s1: dict,
        s2: dict,
        is_group: bool,
        current_date: Optional[pd.Timestamp],
    ) -> Tuple[dict, float, float]:
        league_avg = self._league_avg_goals_per_team()

        elo_diff = elo1 - elo2
        elo_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        elo_sum = elo1 + elo2
        elo_ratio = elo1 / max(elo2, 1.0)

        # momentum
        trend_diff = s1["elo_trend"] - s2["elo_trend"]
        trend3_diff = s1["elo_trend3"] - s2["elo_trend3"]

        # Stable multiplicative prior, not attack/defense division
        stage_goal_mod = 1.0 if is_group else 0.92
        base_xg1 = league_avg * s1["attack_strength"] * s2["defense_weakness"] * stage_goal_mod
        base_xg2 = league_avg * s2["attack_strength"] * s1["defense_weakness"] * stage_goal_mod

        # More grounded composite strength
        fw1 = s1["form_weighted"] / 100.0
        fw2 = s2["form_weighted"] / 100.0
        def1_strength = s1["defense_strength"]
        def2_strength = s2["defense_strength"]

        strength_t1 = (
            0.35 * elo_win_prob
            + 0.25 * min(s1["attack_strength"], 3.0) / 3.0
            + 0.15 * min(def1_strength, 3.0) / 3.0
            + 0.15 * fw1
            + 0.10 * s1["clean_sheet_rate"]
        )
        strength_t2 = (
            0.35 * (1.0 - elo_win_prob)
            + 0.25 * min(s2["attack_strength"], 3.0) / 3.0
            + 0.15 * min(def2_strength, 3.0) / 3.0
            + 0.15 * fw2
            + 0.10 * s2["clean_sheet_rate"]
        )

        h2hw, h2hd, h2hl, h2hn = self._h2h_stats(t1, t2, current_date=current_date)
        h2h_advantage = h2hw - h2hl

        feat = {
            # Core Elo
            "elo_diff": elo_diff,
            "elo_win_prob": elo_win_prob,
            "elo_sum": elo_sum,
            "elo_ratio": elo_ratio,
            "elo1": elo1,
            "elo2": elo2,
            # Base goal prior (stable multiplicative baseline)
            "elo_xg1": base_xg1,
            "elo_xg2": base_xg2,
            # Momentum
            "elo_trend_diff": trend_diff,
            "elo_trend3_diff": trend3_diff,
            "elo_trend_t1": s1["elo_trend"],
            "elo_trend_t2": s2["elo_trend"],
            # Raw rolling goals
            "raw_attack_t1": s1["raw_attack"],
            "raw_attack_t2": s2["raw_attack"],
            "raw_defense_t1": s1["raw_defense"],
            "raw_defense_t2": s2["raw_defense"],
            # Opponent-adjusted strengths
            "attack_strength_t1": s1["attack_strength"],
            "attack_strength_t2": s2["attack_strength"],
            "attack_strength_diff": s1["attack_strength"] - s2["attack_strength"],
            "defense_weakness_t1": s1["defense_weakness"],
            "defense_weakness_t2": s2["defense_weakness"],
            "defense_weakness_diff": s1["defense_weakness"] - s2["defense_weakness"],
            "defense_strength_t1": def1_strength,
            "defense_strength_t2": def2_strength,
            # Base xG matchup
            "xg_base_t1": base_xg1,
            "xg_base_t2": base_xg2,
            "xg_diff": base_xg1 - base_xg2,
            "xg_sum": base_xg1 + base_xg2,
            # Form
            "form_last5_t1": s1["form_last5"],
            "form_last5_t2": s2["form_last5"],
            "form_diff_last5": s1["form_last5"] - s2["form_last5"],
            "form_weighted_t1": fw1,
            "form_weighted_t2": fw2,
            "form_diff_weighted": fw1 - fw2,
            # Variance / consistency
            "goal_var_t1": self._rolling_std(self._team_goals_scored[t1]),
            "goal_var_t2": self._rolling_std(self._team_goals_scored[t2]),
            "concede_var_t1": self._rolling_std(self._team_goals_conceded[t1]),
            "concede_var_t2": self._rolling_std(self._team_goals_conceded[t2]),
            "consistency_t1": s1["consistency"],
            "consistency_t2": s2["consistency"],
            # Defensive record
            "clean_sheet_t1": s1["clean_sheet_rate"],
            "clean_sheet_t2": s2["clean_sheet_rate"],
            # BTTS
            "btts_t1": s1["btts_rate"],
            "btts_t2": s2["btts_rate"],
            # Composite strength
            "strength_t1": strength_t1,
            "strength_t2": strength_t2,
            "strength_diff": strength_t1 - strength_t2,
            # H2H
            "h2h_advantage": h2h_advantage,
            "h2h_win_rate_t1": h2hw,
            "h2h_draw_rate": h2hd,
            "h2h_matches": h2hn,
            # Stage
            "is_group": int(is_group),
            "is_knockout": int(not is_group),
            "stage_goal_mod": stage_goal_mod,
            # Goal pattern
            "avg_total_goals_pred": base_xg1 + base_xg2,
            "goal_ratio": base_xg1 / max(base_xg2, 0.1),
            # Opposition quality / experience
            "opp_elo_avg_t1": s1["opp_elo_avg"],
            "opp_elo_avg_t2": s2["opp_elo_avg"],
            "opp_elo_diff": s1["opp_elo_avg"] - s2["opp_elo_avg"],
            "experience_t1": min(s1["match_count"] / 20.0, 1.0),
            "experience_t2": min(s2["match_count"] / 20.0, 1.0),
        }
        return feat, base_xg1, base_xg2

    def snapshot(self, team: str) -> dict:
        return self._team_strengths(team)

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_rows = []

        for _, row in df.iterrows():
            t1 = str(row["Team 1"]).strip()
            t2 = str(row["Team 2"]).strip()
            elo1 = float(row["Elo1"])
            elo2 = float(row["Elo2"])
            g1 = int(row["Goals1"])
            g2 = int(row["Goals2"])
            stage = str(row.get("Stage", "Group")).strip().lower()
            is_group = "group" in stage or stage == "g"
            row_date = row["Date"] if "Date" in row and pd.notna(row["Date"]) else None

            # snapshot BEFORE updating
            s1 = self.snapshot(t1)
            s2 = self.snapshot(t2)

            feat, _, _ = self._core_features(
                t1=t1,
                t2=t2,
                elo1=elo1,
                elo2=elo2,
                s1=s1,
                s2=s2,
                is_group=is_group,
                current_date=row_date,
            )
            feature_rows.append(feat)

            # Update after snapshot
            league_avg = self._league_avg_goals_per_team()
            opp_def_t2 = max(s2["defense_weakness"], 0.75)
            opp_att_t2 = max(s2["attack_strength"], 0.75)
            opp_def_t1 = max(s1["defense_weakness"], 0.75)
            opp_att_t1 = max(s1["attack_strength"], 0.75)

            # opponent-adjusted updates
            adj_for_t1 = g1 / opp_def_t2
            adj_against_t1 = g2 / opp_att_t2
            adj_for_t2 = g2 / opp_def_t1
            adj_against_t2 = g1 / opp_att_t1

            self._team_goals_scored[t1].append(g1)
            self._team_goals_conceded[t1].append(g2)
            self._team_adj_for[t1].append(adj_for_t1)
            self._team_adj_against[t1].append(adj_against_t1)
            self._team_elo_history[t1].append(elo1)
            self._team_results[t1].append(3.0 if g1 > g2 else (1.0 if g1 == g2 else 0.0))
            self._team_match_count[t1] += 1
            self._team_opponent_elos[t1].append(elo2)

            self._team_goals_scored[t2].append(g2)
            self._team_goals_conceded[t2].append(g1)
            self._team_adj_for[t2].append(adj_for_t2)
            self._team_adj_against[t2].append(adj_against_t2)
            self._team_elo_history[t2].append(elo2)
            self._team_results[t2].append(3.0 if g2 > g1 else (1.0 if g1 == g2 else 0.0))
            self._team_match_count[t2] += 1
            self._team_opponent_elos[t2].append(elo1)

            self._league_goal_history.append((g1 + g2) / 2.0)

            key = tuple(sorted([t1, t2]))
            self._h2h[key].append((t1, t2, g1, g2, row_date))

        return pd.DataFrame(feature_rows, index=df.index)

    def build_lookup(self) -> dict:
        teams = set(self._team_match_count.keys())
        return {team: self.snapshot(team) for team in teams}

    def predict_features(
        self,
        t1: str,
        t2: str,
        t1_elo: float,
        t2_elo: float,
        t1_stats: dict,
        t2_stats: dict,
        is_group: bool = True,
    ) -> pd.DataFrame:
        feat, _, _ = self._core_features(
            t1=t1,
            t2=t2,
            elo1=t1_elo,
            elo2=t2_elo,
            s1=t1_stats,
            s2=t2_stats,
            is_group=is_group,
            current_date=None,
        )
        return pd.DataFrame([feat])


# ─────────────────────────────────────────────────────────────────────────────
# 3. DIXON-COLES CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def dc_tau(g1: int, g2: int, lam1: float, lam2: float, rho: float) -> float:
    if g1 == 0 and g2 == 0:
        return max(1e-9, 1 - lam1 * lam2 * rho)
    elif g1 == 1 and g2 == 0:
        return max(1e-9, 1 + lam2 * rho)
    elif g1 == 0 and g2 == 1:
        return max(1e-9, 1 + lam1 * rho)
    elif g1 == 1 and g2 == 1:
        return max(1e-9, 1 - rho)
    return 1.0


def estimate_rho(goals1: np.ndarray, goals2: np.ndarray,
                 lam1_vals: np.ndarray, lam2_vals: np.ndarray) -> float:
    def neg_ll(rho: float) -> float:
        if abs(rho) >= 0.99:
            return 1e12
        ll = 0.0
        for i in range(len(goals1)):
            g1 = int(goals1[i])
            g2 = int(goals2[i])
            if g1 <= 1 and g2 <= 1:
                tau = dc_tau(g1, g2, float(lam1_vals[i]), float(lam2_vals[i]), float(rho))
                ll += math.log(max(tau, 1e-12))
        return -ll

    result = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
    return float(result.x)


def score_log_prob(y1: int, y2: int, lam1: float, lam2: float, rho: float) -> float:
    lp = poisson.logpmf(y1, lam1) + poisson.logpmf(y2, lam2)
    if y1 <= 1 and y2 <= 1:
        lp += math.log(max(dc_tau(y1, y2, lam1, lam2, rho), 1e-12))
    return float(lp)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SoccerEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rho = 0.0
        self.w1 = np.array([0.34, 0.33, 0.33], dtype=float)
        self.w2 = np.array([0.34, 0.33, 0.33], dtype=float)
        self.feature_cols: List[str] = []

        # Side 1 models
        self.models_t1 = [
            PoissonRegressor(alpha=0.08, max_iter=5000),
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                min_samples_leaf=3,
                random_state=42,
            ),
            RandomForestRegressor(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ]
        # Side 2 models
        self.models_t2 = [
            PoissonRegressor(alpha=0.08, max_iter=5000),
            GradientBoostingRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                min_samples_leaf=3,
                random_state=42,
            ),
            RandomForestRegressor(
                n_estimators=350,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ]

    def _augment(
        self,
        X: pd.DataFrame,
        y1: np.ndarray,
        y2: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Mirror matches to reduce positional bias.

        Fixed bug from prior version:
        - win-rate columns are swapped, not replaced by draw rate
        - directional features are properly flipped
        """
        feat_cols = X.columns.tolist()

        # swap *_t1 <-> *_t2 columns
        swap_map = {}
        for col in feat_cols:
            if col.endswith("_t1"):
                partner = col[:-3] + "_t2"
                if partner in feat_cols:
                    swap_map[col] = partner

        X_rev = X.copy()
        for c1, c2 in swap_map.items():
            X_rev[c1] = X[c2].values
            X_rev[c2] = X[c1].values

        # Flip directional features
        flip_cols = [
            "elo_diff", "attack_strength_diff", "defense_weakness_diff",
            "xg_diff", "form_diff_last5", "form_diff_weighted",
            "strength_diff", "elo_trend_diff", "elo_trend3_diff",
            "h2h_advantage", "goal_ratio", "opp_elo_diff"
        ]
        for col in flip_cols:
            if col in feat_cols:
                X_rev[col] = -X[col].values

        # Reciprocal / complementary features
        if "elo_ratio" in feat_cols:
            X_rev["elo_ratio"] = 1.0 / np.clip(X["elo_ratio"].values, 0.01, None)
        if "elo_win_prob" in feat_cols:
            X_rev["elo_win_prob"] = 1.0 - X["elo_win_prob"].values

        # Swap obvious directional columns explicitly
        for a, b in [("elo1", "elo2"), ("elo_xg1", "elo_xg2")]:
            if a in feat_cols and b in feat_cols:
                X_rev[a] = X[b].values
                X_rev[b] = X[a].values

        # H2H rates should swap, not get corrupted
        if "h2h_win_rate_t1" in feat_cols and "h2h_win_rate_t2" in feat_cols:
            X_rev["h2h_win_rate_t1"] = X["h2h_win_rate_t2"].values
            X_rev["h2h_win_rate_t2"] = X["h2h_win_rate_t1"].values

        # Symmetric features remain as-is:
        # elo_sum, xg_sum, stage flags, total-goal prior, etc.

        X_all = pd.concat([X, X_rev], ignore_index=True)
        y1_all = np.concatenate([y1, y2])
        y2_all = np.concatenate([y2, y1])
        w_all = np.concatenate([w, w])
        return X_all.values, y1_all, y2_all, w_all

    @staticmethod
    def _target_for_model(y: np.ndarray, model_idx: int) -> np.ndarray:
        # PoissonRegressor uses raw counts; tree models use log1p(target)
        if model_idx == 0:
            return y
        return np.log1p(y)

    @staticmethod
    def _pred_to_count(pred: np.ndarray, model_idx: int) -> np.ndarray:
        if model_idx == 0:
            return np.clip(pred, 0, 10)
        return np.clip(np.expm1(pred), 0, 10)

    def _fit_one_side(
        self,
        models: List,
        Xs: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> None:
        for i, m in enumerate(models):
            yy = self._target_for_model(y, i)
            m.fit(Xs, yy, sample_weight=w)

    def _predict_one_side(self, models: List, Xs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        preds = []
        for i, m in enumerate(models):
            raw = m.predict(Xs)
            preds.append(self._pred_to_count(np.asarray(raw, dtype=float), i))
        pred = np.zeros(len(Xs), dtype=float)
        for i in range(len(models)):
            pred += weights[i] * preds[i]
        return np.clip(pred, 0.05, 7.5)

    def fit(
        self,
        X: pd.DataFrame,
        y1: np.ndarray,
        y2: np.ndarray,
        sample_weight: np.ndarray,
        cv_X: pd.DataFrame,
        cv_y1: np.ndarray,
        cv_y2: np.ndarray,
        cv_w: np.ndarray,
    ) -> "SoccerEnsemble":
        self.feature_cols = X.columns.tolist()
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        sample_weight = np.asarray(sample_weight, dtype=float)

        # Augment training data
        X_aug_arr, y1_aug, y2_aug, w_aug = self._augment(X, y1, y2, sample_weight)
        Xs_aug = self.scaler.fit_transform(X_aug_arr)

        self._fit_one_side(self.models_t1, Xs_aug, y1_aug, w_aug)
        self._fit_one_side(self.models_t2, Xs_aug, y2_aug, w_aug)

        # Weight calibration on original chronological data
        if len(cv_X) < 10:
            self.w1 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            self.w2 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        else:
            n_splits = min(ENSEMBLE_CV, max(2, len(cv_X) // 30))
            n_splits = min(n_splits, len(cv_X) - 1)
            if n_splits < 2:
                self.w1 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
                self.w2 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_X_vals = cv_X.values
                cv_y1 = np.asarray(cv_y1, dtype=float)
                cv_y2 = np.asarray(cv_y2, dtype=float)
                cv_w = np.asarray(cv_w, dtype=float)

                def cv_mse(base_est, target, side: int) -> float:
                    mses = []
                    for tr, val in tscv.split(cv_X_vals):
                        sc = StandardScaler()
                        Xtr = sc.fit_transform(cv_X_vals[tr])
                        Xval = sc.transform(cv_X_vals[val])
                        est = clone(base_est)

                        Xtr_df = pd.DataFrame(Xtr, columns=self.feature_cols)
                        X_cv_aug, y1_cv_aug, y2_cv_aug, w_cv_aug = self._augment(
                            Xtr_df, cv_y1[tr], cv_y2[tr], cv_w[tr]
                        )
                        X_cv_aug_scaled = sc.transform(X_cv_aug)

                        if side == 1:
                            y_target = self._target_for_model(y1_cv_aug, 0)  # placeholder; overwritten below
                        else:
                            y_target = self._target_for_model(y2_cv_aug, 0)  # placeholder; overwritten below

                        # Fit same transformation rule used in full training
                        if isinstance(est, PoissonRegressor):
                            if side == 1:
                                y_target = y1_cv_aug
                            else:
                                y_target = y2_cv_aug
                        else:
                            if side == 1:
                                y_target = np.log1p(y1_cv_aug)
                            else:
                                y_target = np.log1p(y2_cv_aug)

                        est.fit(X_cv_aug_scaled, y_target, sample_weight=w_cv_aug)
                        raw_pred = np.asarray(est.predict(Xval), dtype=float)
                        if isinstance(est, PoissonRegressor):
                            pred = np.clip(raw_pred, 0, 10)
                        else:
                            pred = np.clip(np.expm1(raw_pred), 0, 10)

                        target_vals = target[val]
                        mses.append(mean_squared_error(target_vals, pred))
                    return float(np.mean(mses)) if mses else 1e6

                mse1 = np.array([cv_mse(m, cv_y1, 1) for m in self.models_t1], dtype=float)
                mse2 = np.array([cv_mse(m, cv_y2, 2) for m in self.models_t2], dtype=float)

                inv1 = 1.0 / (mse1 + 1e-9)
                inv2 = 1.0 / (mse2 + 1e-9)
                self.w1 = inv1 / inv1.sum()
                self.w2 = inv2 / inv2.sum()

        # Estimate Dixon-Coles rho on training data
        Xs_orig = self.scaler.transform(X.values)
        lam1_tr = self._raw_predict_array(Xs_orig)[0]
        lam2_tr = self._raw_predict_array(Xs_orig)[1]
        self.rho = estimate_rho(y1, y2, lam1_tr, lam2_tr)

        return self

    def _raw_predict_array(self, Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        p1 = self._predict_one_side(self.models_t1, Xs, self.w1)
        p2 = self._predict_one_side(self.models_t2, Xs, self.w2)
        return np.clip(p1, 0.05, 7.0), np.clip(p2, 0.05, 7.0)

    def predict_xg(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_use = X.copy()
        for col in self.feature_cols:
            if col not in X_use.columns:
                X_use[col] = 0.0
        X_use = X_use[self.feature_cols]
        Xs = self.scaler.transform(X_use.values)
        return self._raw_predict_array(Xs)

    def score_matrix(self, lam1: float, lam2: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        p1 = poisson.pmf(g, lam1)
        p2 = poisson.pmf(g, lam2)
        mat = np.outer(p1, p2)

        # low-score correction
        for i in range(min(2, MAX_GOALS + 1)):
            for j in range(min(2, MAX_GOALS + 1)):
                mat[i, j] *= dc_tau(i, j, lam1, lam2, self.rho)

        total = mat.sum()
        if total > 0:
            mat /= total
        return mat

    def outcome_probs(self, mat: np.ndarray) -> Tuple[float, float, float]:
        pw = float(np.tril(mat, -1).sum())  # team1 wins: g1 > g2
        pd_ = float(np.trace(mat))
        pl = float(np.triu(mat, 1).sum())   # team2 wins: g2 > g1
        return pw, pd_, pl


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def recency_weights(df: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(df["Date"], errors="coerce")
    if dates.notna().sum() == 0:
        return np.ones(len(df), dtype=float)
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.fillna(9999).astype(float)
    w = np.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
    return np.where(np.isfinite(w), w, 1.0)


def train_pipeline(df: pd.DataFrame) -> Tuple[SoccerEnsemble, FeatureEngine, pd.DataFrame]:
    engine = FeatureEngine(window=8)
    feat_df = engine.compute_all_features(df)

    y1 = df["Goals1"].values.astype(float)
    y2 = df["Goals2"].values.astype(float)
    weights = recency_weights(df)

    model = SoccerEnsemble()
    model.fit(feat_df, y1, y2, weights, feat_df, y1, y2, weights)
    return model, engine, feat_df


def evaluate(model: SoccerEnsemble, feat_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    y1 = df["Goals1"].values.astype(int)
    y2 = df["Goals2"].values.astype(int)
    lam1, lam2 = model.predict_xg(feat_df)

    exact = []
    outcome_correct = []
    nlls = []

    for i in range(len(df)):
        mat = model.score_matrix(float(lam1[i]), float(lam2[i]))
        pw, pd_, pl = model.outcome_probs(mat)
        pred_outcome = np.argmax([pw, pd_, pl])
        true_outcome = 0 if y1[i] > y2[i] else (1 if y1[i] == y2[i] else 2)
        outcome_correct.append(int(pred_outcome == true_outcome))
        exact.append(int(y1[i] == round(float(lam1[i])) and y2[i] == round(float(lam2[i]))))
        nlls.append(-score_log_prob(y1[i], y2[i], float(lam1[i]), float(lam2[i]), model.rho))

    return {
        "rmse_t1": float(np.sqrt(mean_squared_error(y1, lam1))),
        "rmse_t2": float(np.sqrt(mean_squared_error(y2, lam2))),
        "mae_t1": float(mean_absolute_error(y1, lam1)),
        "mae_t2": float(mean_absolute_error(y2, lam2)),
        "exact_score_pct": float(np.mean(exact) * 100.0),
        "outcome_acc_pct": float(np.mean(outcome_correct) * 100.0),
        "score_nll": float(np.mean(nlls)),
        "rho": float(model.rho),
        "w1": model.w1.tolist(),
        "w2": model.w2.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(
    model: SoccerEnsemble,
    engine: FeatureEngine,
    t1: str,
    t2: str,
    is_group: bool = True,
) -> dict:
    lookup = engine.build_lookup()

    if t1 not in lookup:
        raise KeyError(f"Team not found: {t1}")
    if t2 not in lookup:
        raise KeyError(f"Team not found: {t2}")

    s1 = lookup[t1]
    s2 = lookup[t2]

    feat = engine.predict_features(t1, t2, s1["elo"], s2["elo"], s1, s2, is_group)

    for col in model.feature_cols:
        if col not in feat.columns:
            feat[col] = 0.0
    feat = feat[model.feature_cols]

    lam1_arr, lam2_arr = model.predict_xg(feat)
    lam1 = float(lam1_arr[0])
    lam2 = float(lam2_arr[0])

    mat = model.score_matrix(lam1, lam2)
    pw, pd_, pl = model.outcome_probs(mat)

    flat = np.argsort(mat.ravel())[::-1]
    top5 = []
    for idx in flat[:5]:
        g1, g2 = divmod(int(idx), mat.shape[1])
        top5.append((g1, g2, float(mat[g1, g2])))

    return {
        "team1": t1,
        "team2": t2,
        "lam1": lam1,
        "lam2": lam2,
        "top5": top5,
        "win": pw,
        "draw": pd_,
        "loss": pl,
        "matrix": mat,
        "stats1": s1,
        "stats2": s2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def hline(char="─", width=70):
    print(char * width)


def center(text, width=70):
    print(text.center(width))


def display_prediction(result: dict):
    t1, t2 = result["team1"], result["team2"]
    lam1, lam2 = result["lam1"], result["lam2"]
    pw, pd_, pl = result["win"], result["draw"], result["loss"]
    total = pw + pd_ + pl
    top5 = result["top5"]

    print()
    hline("═")
    center(f"  {t1}  vs  {t2}  ")
    hline("═")

    print(f"\n  Expected Goals:  {t1}: {lam1:.2f}   {t2}: {lam2:.2f}")
    print(
        f"  Win Probs:  {t1}: {pw / total * 100:.1f}%  |  "
        f"Draw: {pd_ / total * 100:.1f}%  |  {t2}: {pl / total * 100:.1f}%\n"
    )

    hline()
    print("  TOP PREDICTED SCORES:")
    hline()
    medals = ["🥇", "🥈", "🥉", "   ", "   "]
    for i, (g1, g2, prob) in enumerate(top5):
        outcome = f"{t1} Win" if g1 > g2 else ("Draw" if g1 == g2 else f"{t2} Win")
        bar = "█" * int(prob * 200) + "░" * max(0, 20 - int(prob * 200))
        print(f"  {medals[i]}  {g1} – {g2}  ({outcome:<16})  {prob * 100:5.2f}%  {bar}")

    print()
    hline()
    print("  OUTCOME PROBABILITIES:")
    hline()
    for label, p in [(f"{t1} Win", pw), ("Draw", pd_), (f"{t2} Win", pl)]:
        pct = p / total * 100
        bar = "█" * int(pct / 2) + "░" * max(0, 50 - int(pct / 2))
        print(f"  {label:<18} {pct:5.1f}%  {bar}")

    print()
    if pw >= pd_ and pw >= pl:
        verdict = f"{t1} to WIN  ({pw / total * 100:.1f}% confidence)"
    elif pd_ >= pw and pd_ >= pl:
        verdict = f"DRAW  ({pd_ / total * 100:.1f}% confidence)"
    else:
        verdict = f"{t2} to WIN  ({pl / total * 100:.1f}% confidence)"
    hline("═")
    center(f"  VERDICT: {verdict}  ")
    hline("═")

    print("\n  Score Probability Matrix (goals 0–5):")
    n = 6
    header = f"  {t1[:8]:<8}↓ {t2[:8]:<8}→" + "".join(f"  {g2:>4}" for g2 in range(n))
    print(header)
    mat = result["matrix"]
    for g1 in range(n):
        row = f"  {g1:>17}"
        for g2 in range(n):
            row += f"  {mat[g1, g2] * 100:4.1f}"
        if g1 == 0:
            row += "  (% chance)"
        print(row)

    s1, s2 = result["stats1"], result["stats2"]
    print()
    hline()
    print(f"  {'STAT':<28} {'':>4}{t1[:12]:>12}  {'':>0}{t2[:12]:>12}")
    hline()
    stats_to_show = [
        ("Elo Rating", "elo", ".0f"),
        ("Elo Trend (last 5)", "elo_trend", "+.1f"),
        ("Raw Goals Scored", "raw_attack", ".2f"),
        ("Raw Goals Conceded", "raw_defense", ".2f"),
        ("Attack Strength", "attack_strength", ".2f"),
        ("Def Weakness", "defense_weakness", ".2f"),
        ("Form (last 5)", "form_last5", ".1f"),
        ("Clean Sheet Rate", "clean_sheet_rate", ".1%"),
        ("BTTS Rate", "btts_rate", ".1%"),
        ("Consistency Score", "consistency", ".2f"),
    ]
    for label, key, fmt in stats_to_show:
        v1 = s1.get(key, 0)
        v2 = s2.get(key, 0)
        print(f"  {label:<28} {format(v1, fmt):>12}  {format(v2, fmt):>12}")
    hline()


def display_model_stats(metrics: dict, n_train: int, n_test: int, n_teams: int):
    print()
    hline("═")
    center("  MODEL PERFORMANCE SUMMARY  ")
    hline("═")
    print(f"  Training matches:   {n_train}")
    print(f"  Holdout matches:    {n_test}")
    print(f"  Teams in lookup:    {n_teams}")
    hline()
    print(f"  RMSE (Team 1 xG):   {metrics['rmse_t1']:.3f}")
    print(f"  RMSE (Team 2 xG):   {metrics['rmse_t2']:.3f}")
    print(f"  MAE  (Team 1 xG):   {metrics['mae_t1']:.3f}")
    print(f"  MAE  (Team 2 xG):   {metrics['mae_t2']:.3f}")
    print(f"  Score NLL:          {metrics['score_nll']:.3f}")
    print(f"  Exact score acc.:   {metrics['exact_score_pct']:.1f}%")
    print(f"  Outcome accuracy:   {metrics['outcome_acc_pct']:.1f}%")
    print(f"  Dixon-Coles ρ:      {metrics['rho']:+.4f}")
    w1 = metrics["w1"]
    w2 = metrics["w2"]
    print(f"  Ensemble weights T1: Poisson={w1[0]:.2f}  GB={w1[1]:.2f}  RF={w1[2]:.2f}")
    print(f"  Ensemble weights T2: Poisson={w2[0]:.2f}  GB={w2[1]:.2f}  RF={w2[2]:.2f}")
    hline("═")


def run_backtest(model: SoccerEnsemble, engine_trained: FeatureEngine,
                 test_df: pd.DataFrame, feat_test: pd.DataFrame):
    print()
    hline("═")
    center("  BACKTEST — Held-out Matches  ")
    hline("═")

    lam1_all, lam2_all = model.predict_xg(feat_test)
    y1 = test_df["Goals1"].values
    y2 = test_df["Goals2"].values

    correct = 0
    exact = 0
    n = len(test_df)

    print(f"  {'Match':<26} {'Actual':>7} {'Pred#1':>7} {'Prob':>7} {'OK?':>5}  xG")
    hline()

    for i, (_, row) in enumerate(test_df.iterrows()):
        t1, t2 = str(row["Team 1"])[:11], str(row["Team 2"])[:11]
        s1, s2 = int(y1[i]), int(y2[i])
        lam1, lam2 = float(lam1_all[i]), float(lam2_all[i])

        mat = model.score_matrix(lam1, lam2)
        flat = np.argsort(mat.ravel())[::-1]
        pg1, pg2 = divmod(int(flat[0]), mat.shape[1])
        prob = float(mat[pg1, pg2])

        act_out = "W" if s1 > s2 else ("D" if s1 == s2 else "L")
        pw, pd_, pl = model.outcome_probs(mat)
        pred_out = "W" if pw > pd_ and pw > pl else ("D" if pd_ >= pw and pd_ >= pl else "L")

        ok = act_out == pred_out
        ex = pg1 == s1 and pg2 == s2
        if ok:
            correct += 1
        if ex:
            exact += 1

        tick = "✓" if ok else "✗"
        print(f"  {t1:<12} v {t2:<12} {s1}–{s2}  {pg1}–{pg2}  {prob * 100:.1f}%  {tick}  {lam1:.2f}/{lam2:.2f}")

    hline()
    print(f"  Outcome: {correct}/{n} ({correct / n * 100:.0f}%)   Exact: {exact}/{n} ({exact / n * 100:.0f}%)")
    hline("═")


# ─────────────────────────────────────────────────────────────────────────────
# 8. INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_find(query: str, names: List[str]) -> Optional[str]:
    q = query.strip().lower()
    for n in names:
        if n.lower() == q:
            return n
    matches = [n for n in names if n.lower().startswith(q)]
    if len(matches) == 1:
        return matches[0]
    matches = [n for n in names if q in n.lower()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        print(f"  Did you mean: {', '.join(matches[:5])}?")
    return None


def print_teams(lookup: dict):
    names = sorted(lookup.keys())
    cols = 4
    for i in range(0, len(names), cols):
        row = names[i:i + cols]
        print("  " + "  |  ".join(f"{n:<22}" for n in row))


def interactive(model: SoccerEnsemble, engine: FeatureEngine,
                backtest_df=None, backtest_feat=None):
    lookup = engine.build_lookup()
    names = sorted(lookup.keys())

    print()
    hline("═")
    center("  COMMANDS: predict | list | backtest | quit  ")
    hline("═")

    while True:
        print()
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if cmd in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        elif cmd in ("list", "ls", "teams"):
            print_teams(lookup)

        elif cmd in ("backtest", "bt"):
            if backtest_df is not None:
                run_backtest(model, engine, backtest_df, backtest_feat)
            else:
                print("  No backtest data available.")

        elif cmd in ("predict", "p", "pred", ""):
            print("  Enter team names (or partial). Type 'list' to see all teams.")

            t1_name = None
            while not t1_name:
                t1_input = input("  Team 1: ").strip()
                if t1_input.lower() == "list":
                    print_teams(lookup)
                    continue
                t1_name = fuzzy_find(t1_input, names)
                if not t1_name:
                    print(f"  Not found: '{t1_input}'")

            t2_name = None
            while not t2_name:
                t2_input = input("  Team 2: ").strip()
                if t2_input.lower() == "list":
                    print_teams(lookup)
                    continue
                t2_name = fuzzy_find(t2_input, names)
                if not t2_name:
                    print(f"  Not found: '{t2_input}'")

            if t1_name == t2_name:
                print("  A team cannot play itself.")
                continue

            stage = input("  Stage [G=Group / K=Knockout, default G]: ").strip().upper()
            is_group = stage != "K"

            print("  Running prediction...")
            result = predict_match(model, engine, t1_name, t2_name, is_group)
            display_prediction(result)

        else:
            print(f"  Unknown command: '{cmd}'. Try: predict | list | backtest | quit")


# ─────────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    path = "trimmed.csv"
    if not os.path.exists(path):
        path = "/mnt/user-data/uploads/FINALSCORES.csv"
    if not os.path.exists(path):
        import glob
        csvs = glob.glob("/mnt/user-data/uploads/*.csv")
        if csvs:
            path = csvs[0]

    print()
    hline("═")
    center("  ADVANCED SOCCER PREDICTION MODEL  v4.0  ")
    center("  Opponent-adjusted Poisson × GB × RF × Dixon-Coles  ")
    hline("═")

    print(f"\n  Loading data from: {path}")
    df = load_and_clean(path)
    print(f"  Loaded {len(df)} matches")

    holdout_n = min(max(MIN_HOLDOUT_MATCHES, int(len(df) * HOLDOUT_FRACTION)), len(df) - 1)
    holdout_n = max(1, holdout_n)

    train_df = df.iloc[:-holdout_n].reset_index(drop=True)
    test_df = df.iloc[-holdout_n:].reset_index(drop=True)

    print(f"  Training on {len(train_df)} matches, holding out {len(test_df)}")

    print("\n  Training holdout evaluation model...")
    eval_model, eval_engine, eval_feat = train_pipeline(train_df)

    # Sequentially build test features using training state first, then evolving through test matches
    test_feat = eval_engine.compute_all_features(test_df)

    metrics = evaluate(eval_model, test_feat, test_df)
    lookup = eval_engine.build_lookup()
    display_model_stats(metrics, len(train_df), len(test_df), len(lookup))

    run_backtest(eval_model, eval_engine, test_df, test_feat)

    print("\n  Training final model on all data...")
    final_model, final_engine, _ = train_pipeline(df)
    final_lookup = final_engine.build_lookup()
    print(f"  Final model trained. {len(final_lookup)} teams available.\n")

    interactive(final_model, final_engine, backtest_df=test_df, backtest_feat=test_feat)


if __name__ == "__main__":
    main()