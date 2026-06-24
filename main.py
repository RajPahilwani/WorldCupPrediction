#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════
  FINAL SCORE PREDICTION PLATFORM  ·  v2.0
  Ensemble ML  ×  Poisson GLM  ×  Dixon-Coles Correction
══════════════════════════════════════════════════════════════════════════
  Models stacked:
    • Poisson GLM      – gold-standard for goal-count distribution
    • Gradient Boosting – captures non-linear feature interactions
    • Random Forest     – variance-reduction bagging
  Correction:
    • Dixon-Coles τ    – fixes 0-0 / 1-0 / 0-1 / 1-1 dependency bias
══════════════════════════════════════════════════════════════════════════
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Prompt
from rich.columns import Columns
from rich import box
from rich.align import Align

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "FINALSCORES.csv"
MAX_GOALS = 9       # max goals per team considered in the probability matrix
ENSEMBLE_CV = 5     # k-fold splits for weight estimation

THEME = {
    "gold":    "bold yellow",
    "silver":  "bold cyan",
    "bronze":  "bold green",
    "dim":     "dim white",
    "header":  "bold white on dark_blue",
    "win":     "bold green",
    "draw":    "bold yellow",
    "loss":    "bold red",
    "brand":   "bold cyan",
    "label":   "bold blue",
}

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_team_lookup(df: pd.DataFrame) -> dict:
    """
    Build a per-team stats dict using their MOST RECENT appearance.
    Combines Team 1 and Team 2 appearances, keyed by team name.
    """
    lookup = {}

    for _, row in df.iterrows():
        for side, team_col, elo_col, atk_col, def_col, fl5_col, fw_col in [
            ("t1", "Team 1", "Elo1", "Attack_Rating_T1", "Defense_Rating_T1",
             "Form_Last5_T1", "Form_Weighted_T1"),
            ("t2", "Team 2", "Elo2", "Attack_Rating_T2", "Defense_Rating_T2",
             "Form_Last5_T2", "Form_Weighted_T2"),
        ]:
            name = row[team_col]
            lookup[name] = {
                "elo":           float(row[elo_col]),
                "attack":        float(row[atk_col]),
                "defense":       float(row[def_col]),
                "form_last5":    float(row[fl5_col]),
                "form_weighted": float(row[fw_col]),
            }

    return lookup


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a rich asymmetric feature matrix.
    Separate predictors for each team's goal side for maximum accuracy.
    """
    f = pd.DataFrame(index=df.index)

    # ── Elo features (most predictive single signal in football) ──────────────
    f["elo_diff"]      = df["Elo1"] - df["Elo2"]
    f["elo_ratio"]     = df["Elo1"] / df["Elo2"]
    # Probability that Team 1 wins purely from Elo (Bradley-Terry / Elo formula)
    f["elo_win_prob"]  = 1.0 / (1.0 + 10.0 ** ((df["Elo2"] - df["Elo1"]) / 400.0))
    f["elo_sum"]       = df["Elo1"] + df["Elo2"]    # overall match quality

    # ── Absolute ratings ───────────────────────────────────────────────────────
    f["attack_t1"]  = df["Attack_Rating_T1"]
    f["attack_t2"]  = df["Attack_Rating_T2"]
    f["defense_t1"] = df["Defense_Rating_T1"]
    f["defense_t2"] = df["Defense_Rating_T2"]

    # ── Attack × Defense matchup (the most direct xG signal) ──────────────────
    f["xg_t1"] = df["Attack_Rating_T1"] / (df["Defense_Rating_T2"].clip(lower=0.1))
    f["xg_t2"] = df["Attack_Rating_T2"] / (df["Defense_Rating_T1"].clip(lower=0.1))
    f["xg_diff"] = f["xg_t1"] - f["xg_t2"]

    # ── Form features ──────────────────────────────────────────────────────────
    f["form_last5_t1"]    = df["Form_Last5_T1"]
    f["form_last5_t2"]    = df["Form_Last5_T2"]
    f["form_weighted_t1"] = df["Form_Weighted_T1"]
    f["form_weighted_t2"] = df["Form_Weighted_T2"]
    f["form_diff_last5"]  = df["Form_Last5_T1"] - df["Form_Last5_T2"]
    f["form_diff_weighted"] = df["Form_Weighted_T1"] - df["Form_Weighted_T2"]

    # ── Composite strength index (calibrated blend) ────────────────────────────
    f["strength_t1"] = (
        f["elo_win_prob"]       * 0.40 +
        f["xg_t1"].clip(0, 6)  * 0.15 +
        f["form_weighted_t1"] / 100.0 * 0.45
    )
    f["strength_t2"] = (
        (1 - f["elo_win_prob"]) * 0.40 +
        f["xg_t2"].clip(0, 6)  * 0.15 +
        f["form_weighted_t2"] / 100.0 * 0.45
    )
    f["strength_ratio"] = f["strength_t1"] / (f["strength_t2"].clip(lower=0.01))

    # ── Stage flag (knockout reduces open play; leads to fewer goals) ──────────
    f["is_group"] = df["Stage"].str.contains("Group", case=False, na=False).astype(int)

    return f


def make_feature_row(t1_stats: dict, t2_stats: dict, is_group: bool = True) -> pd.DataFrame:
    """Build a single-row feature DataFrame from raw team stats dicts."""
    elo1, elo2 = t1_stats["elo"], t2_stats["elo"]
    elo_win_prob = 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))

    atk1, def1 = t1_stats["attack"], t1_stats["defense"]
    atk2, def2 = t2_stats["attack"], t2_stats["defense"]

    xg_t1 = atk1 / max(def2, 0.1)
    xg_t2 = atk2 / max(def1, 0.1)

    fw1, fw2 = t1_stats["form_weighted"], t2_stats["form_weighted"]
    fl1, fl2 = t1_stats["form_last5"],    t2_stats["form_last5"]

    str1 = elo_win_prob * 0.40 + min(xg_t1, 6) * 0.15 + fw1 / 100.0 * 0.45
    str2 = (1 - elo_win_prob) * 0.40 + min(xg_t2, 6) * 0.15 + fw2 / 100.0 * 0.45

    row = {
        "elo_diff":          elo1 - elo2,
        "elo_ratio":         elo1 / elo2,
        "elo_win_prob":      elo_win_prob,
        "elo_sum":           elo1 + elo2,
        "attack_t1":         atk1,
        "attack_t2":         atk2,
        "defense_t1":        def1,
        "defense_t2":        def2,
        "xg_t1":             xg_t1,
        "xg_t2":             xg_t2,
        "xg_diff":           xg_t1 - xg_t2,
        "form_last5_t1":     fl1,
        "form_last5_t2":     fl2,
        "form_weighted_t1":  fw1,
        "form_weighted_t2":  fw2,
        "form_diff_last5":   fl1 - fl2,
        "form_diff_weighted":fw1 - fw2,
        "strength_t1":       str1,
        "strength_t2":       str2,
        "strength_ratio":    str1 / max(str2, 0.01),
        "is_group":          int(is_group),
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIXON-COLES CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def dc_tau(g1: int, g2: int, lam1: float, lam2: float, rho: float) -> float:
    """
    Dixon-Coles τ correction factor.
    Only applied to {0-0, 1-0, 0-1, 1-1} cells where Poisson is inaccurate.
    All other cells return 1.0.
    """
    if   g1 == 0 and g2 == 0:  return max(1e-6, 1 - lam1 * lam2 * rho)
    elif g1 == 1 and g2 == 0:  return max(1e-6, 1 + lam2 * rho)
    elif g1 == 0 and g2 == 1:  return max(1e-6, 1 + lam1 * rho)
    elif g1 == 1 and g2 == 1:  return max(1e-6, 1 - rho)
    return 1.0


def estimate_rho(df: pd.DataFrame, lam1_vals: np.ndarray, lam2_vals: np.ndarray) -> float:
    """
    MLE estimate of the Dixon-Coles dependency parameter ρ.
    Maximises the log-likelihood over the low-score cells.
    """
    g1 = df["Goals1"].values
    g2 = df["Goals2"].values

    def neg_log_likelihood(rho: float) -> float:
        if abs(rho) >= 0.99:
            return 1e12
        ll = 0.0
        for i in range(len(g1)):
            if g1[i] <= 1 and g2[i] <= 1:
                tau = dc_tau(g1[i], g2[i], lam1_vals[i], lam2_vals[i], rho)
                ll += np.log(max(tau, 1e-9))
        return -ll

    result = minimize_scalar(neg_log_likelihood, bounds=(-0.99, 0.99), method="bounded")
    return float(result.x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ScorePredictionEnsemble:
    """
    Three-model ensemble for expected-goals prediction.
    Weights are assigned via inverse-CV-MSE weighting.
    """

    def __init__(self):
        self.scaler = StandardScaler()

        # Estimators for Team 1 goals
        self.poisson_t1 = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb_t1      = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, min_samples_leaf=4, random_state=42
        )
        self.rf_t1      = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=4,
            random_state=42
        )

        # Estimators for Team 2 goals
        self.poisson_t2 = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb_t2      = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.04,
            subsample=0.75, min_samples_leaf=4, random_state=42
        )
        self.rf_t2      = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=4,
            random_state=42
        )

        self.w1 = np.array([1/3, 1/3, 1/3])   # ensemble weights for Team 1
        self.w2 = np.array([1/3, 1/3, 1/3])   # ensemble weights for Team 2
        self.rho = 0.0                          # Dixon-Coles parameter
        self.feature_cols: list = []

        # Stored metrics
        self.cv_metrics: dict = {}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y1: np.ndarray, y2: np.ndarray) -> "ScorePredictionEnsemble":
        self.feature_cols = X.columns.tolist()
        Xs = self.scaler.fit_transform(X)
        kf  = KFold(n_splits=ENSEMBLE_CV, shuffle=True, random_state=0)

        # ── Train all estimators ──────────────────────────────────────────────
        for est in [self.poisson_t1, self.gb_t1, self.rf_t1]:
            est.fit(Xs, y1)
        for est in [self.poisson_t2, self.gb_t2, self.rf_t2]:
            est.fit(Xs, y2)

        # ── CV weights (inverse MSE weighting) ───────────────────────────────
        def cv_mse(est, Xs, y):
            return -cross_val_score(est, Xs, y, cv=kf,
                                    scoring="neg_mean_squared_error").mean()

        mse1 = np.array([cv_mse(self.poisson_t1, Xs, y1),
                         cv_mse(self.gb_t1, Xs, y1),
                         cv_mse(self.rf_t1, Xs, y1)])
        mse2 = np.array([cv_mse(self.poisson_t2, Xs, y2),
                         cv_mse(self.gb_t2, Xs, y2),
                         cv_mse(self.rf_t2, Xs, y2)])

        inv1 = 1.0 / (mse1 + 1e-9);  self.w1 = inv1 / inv1.sum()
        inv2 = 1.0 / (mse2 + 1e-9);  self.w2 = inv2 / inv2.sum()

        # ── Estimate Dixon-Coles ρ on full-data predictions ───────────────────
        lam1, lam2 = self._raw_predict(Xs)
        df_proxy = pd.DataFrame({"Goals1": y1, "Goals2": y2})
        self.rho  = estimate_rho(df_proxy, lam1, lam2)

        # ── Store performance metrics ─────────────────────────────────────────
        pred_y1 = np.round(lam1).astype(int).clip(0)
        pred_y2 = np.round(lam2).astype(int).clip(0)
        exact    = np.mean((pred_y1 == y1) & (pred_y2 == y2))
        outcome_pred = np.sign(lam1 - lam2)
        outcome_true = np.sign(y1    - y2)
        outcome_acc  = np.mean(outcome_pred == outcome_true)

        self.cv_metrics = {
            "rmse_t1":       float(np.sqrt(mean_squared_error(y1, lam1))),
            "rmse_t2":       float(np.sqrt(mean_squared_error(y2, lam2))),
            "mae_t1":        float(mean_absolute_error(y1, lam1)),
            "mae_t2":        float(mean_absolute_error(y2, lam2)),
            "exact_score":   float(exact),
            "outcome_acc":   float(outcome_acc),
            "rho":           self.rho,
            "weights_t1":    self.w1.tolist(),
            "weights_t2":    self.w2.tolist(),
        }

        return self

    # ── Prediction helpers ─────────────────────────────────────────────────────

    def _raw_predict(self, Xs: np.ndarray):
        lam1 = (self.w1[0] * np.clip(self.poisson_t1.predict(Xs), 0, 10) +
                self.w1[1] * np.clip(self.gb_t1.predict(Xs),      0, 10) +
                self.w1[2] * np.clip(self.rf_t1.predict(Xs),      0, 10))
        lam2 = (self.w2[0] * np.clip(self.poisson_t2.predict(Xs), 0, 10) +
                self.w2[1] * np.clip(self.gb_t2.predict(Xs),      0, 10) +
                self.w2[2] * np.clip(self.rf_t2.predict(Xs),      0, 10))
        return np.clip(lam1, 0.05, 7.0), np.clip(lam2, 0.05, 7.0)

    def predict_expected_goals(self, X: pd.DataFrame):
        Xs = self.scaler.transform(X[self.feature_cols])
        return self._raw_predict(Xs)

    def score_matrix(self, lam1: float, lam2: float) -> np.ndarray:
        """
        Build a (MAX_GOALS+1 × MAX_GOALS+1) joint probability matrix
        with Dixon-Coles correction applied to low-score cells.
        """
        mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for g1 in range(MAX_GOALS + 1):
            for g2 in range(MAX_GOALS + 1):
                p  = poisson.pmf(g1, lam1) * poisson.pmf(g2, lam2)
                dc = dc_tau(g1, g2, lam1, lam2, self.rho)
                mat[g1, g2] = p * dc
        mat /= mat.sum()    # renormalise after DC correction
        return mat

    def outcomes(self, mat: np.ndarray) -> tuple:
        """Return (P_win, P_draw, P_loss) for Team 1."""
        n = mat.shape[0]
        pw = 0.0;   pd_ = 0.0;   pl = 0.0
        for g1 in range(n):
            for g2 in range(n):
                if   g1 > g2: pw += mat[g1, g2]
                elif g1 == g2: pd_ += mat[g1, g2]
                else:           pl  += mat[g1, g2]
        return pw, pd_, pl


# ─────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner():
    console.print()
    console.print(Panel(
        Align.center(
            Text.from_markup(
                "[bold cyan]⚽  FINAL SCORE PREDICTION PLATFORM  ⚽[/bold cyan]\n"
                "[dim]Ensemble ML  ×  Poisson GLM  ×  Dixon-Coles Correction[/dim]"
            )
        ),
        box=box.DOUBLE_EDGE, border_style="cyan", padding=(0, 4)
    ))


def print_model_stats(metrics: dict, n_matches: int, n_teams: int):
    t = Table(title="📊 Model Training Summary", box=box.ROUNDED,
              border_style="blue", show_header=True)
    t.add_column("Metric",         style="bold white", justify="left")
    t.add_column("Value",          style="bold yellow", justify="center")
    t.add_column("Description",    style="dim white",   justify="left")

    t.add_row("Matches trained on", str(n_matches),         "Historical records in dataset")
    t.add_row("Teams covered",      str(n_teams),           "Unique teams in lookup table")
    t.add_row("RMSE – Team 1 xG",  f"{metrics['rmse_t1']:.3f}", "Lower = more accurate goal estimate")
    t.add_row("RMSE – Team 2 xG",  f"{metrics['rmse_t2']:.3f}", "Lower = more accurate goal estimate")
    t.add_row("MAE – Team 1 xG",   f"{metrics['mae_t1']:.3f}",  "Mean absolute goal count error")
    t.add_row("MAE – Team 2 xG",   f"{metrics['mae_t2']:.3f}",  "Mean absolute goal count error")
    t.add_row("Exact score acc.",   f"{metrics['exact_score']*100:.1f}%", "% of matches with exact score right")
    t.add_row("Outcome accuracy",   f"{metrics['outcome_acc']*100:.1f}%", "Win/Draw/Loss prediction accuracy")
    t.add_row("Dixon-Coles ρ",      f"{metrics['rho']:+.4f}",   "Low-score dependency correction")
    w1 = metrics["weights_t1"]
    w2 = metrics["weights_t2"]
    t.add_row("Ensemble w (T1)",   f"P={w1[0]:.2f} GB={w1[1]:.2f} RF={w1[2]:.2f}", "Poisson / GradBoost / RandForest")
    t.add_row("Ensemble w (T2)",   f"P={w2[0]:.2f} GB={w2[1]:.2f} RF={w2[2]:.2f}", "Poisson / GradBoost / RandForest")

    console.print(t)


def display_prediction(result: dict):
    t1   = result["team1"]
    t2   = result["team2"]
    lam1 = result["lam1"]
    lam2 = result["lam2"]
    top3 = result["top3"]
    pw   = result["win"]
    pd_  = result["draw"]
    pl   = result["loss"]
    mat  = result["matrix"]

    console.print()
    console.print(Rule(f"[bold cyan]{t1}  vs  {t2}[/bold cyan]"))

    # ── Expected goals ────────────────────────────────────────────────────────
    xg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    xg_table.add_column("", style="bold white")
    xg_table.add_column("", style="bold yellow", justify="center")
    xg_table.add_column("", style="bold white")
    xg_table.add_column("", style="bold yellow", justify="center")
    xg_table.add_row(
        f"⚡ {t1} xG:", f"{lam1:.2f}",
        f"⚡ {t2} xG:", f"{lam2:.2f}"
    )
    console.print(xg_table)

    # ── Top 3 predicted scorelines ────────────────────────────────────────────
    medals = ["🥇", "🥈", "🥉"]
    medal_styles = [THEME["gold"], THEME["silver"], THEME["bronze"]]

    score_table = Table(
        title="🎯 Top 3 Predicted Final Scores",
        box=box.ROUNDED, border_style="yellow", show_header=True
    )
    score_table.add_column("Rank",        justify="center", style="bold white",  width=6)
    score_table.add_column("Score",       justify="center", style="bold yellow", width=14)
    score_table.add_column("Probability", justify="center", style="bold cyan",   width=14)
    score_table.add_column("Confidence",  justify="left",   style="dim white",   width=25)

    for i, (g1, g2, prob) in enumerate(top3):
        bar_len = int(prob * 100)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        result_label = (
            f"[green]{t1} WIN[/green]" if g1 > g2 else
            ("[yellow]DRAW[/yellow]"   if g1 == g2 else
             f"[red]{t2} WIN[/red]")
        )
        score_table.add_row(
            f"[{medal_styles[i]}]{medals[i]}[/{medal_styles[i]}]",
            f"[{medal_styles[i]}]{g1} – {g2}[/{medal_styles[i]}]  {result_label}",
            f"[bold cyan]{prob*100:.2f}%[/bold cyan]",
            f"[dim]{bar}[/dim]",
        )

    console.print(score_table)

    # ── Win / Draw / Loss probabilities ───────────────────────────────────────
    outcome_table = Table(
        title="📈 Match Outcome Probabilities",
        box=box.ROUNDED, border_style="green", show_header=True
    )
    outcome_table.add_column("Outcome",     justify="center", style="bold white",  width=20)
    outcome_table.add_column("Probability", justify="center", width=14)
    outcome_table.add_column("Likelihood",  justify="left",   width=30)

    total = pw + pd_ + pl

    def make_bar(p, color, width=25):
        filled = int(round(p / total * width))
        return f"[{color}]" + "█" * filled + "[/]" + "[dim]" + "░" * (width - filled) + "[/dim]"

    winner  = t1 if pw > pl else (t2 if pl > pw else "Draw")
    verdict = (
        f"[green]{t1}[/green]" if pw >= pd_ and pw >= pl else
        (f"[yellow]Draw[/yellow]" if pd_ >= pw and pd_ >= pl else
         f"[red]{t2}[/red]")
    )
    top_prob = max(pw, pd_, pl) / total

    outcome_table.add_row(
        f"✅ {t1} Win",
        f"[bold green]{pw/total*100:.1f}%[/bold green]",
        make_bar(pw, "green")
    )
    outcome_table.add_row(
        "🤝 Draw",
        f"[bold yellow]{pd_/total*100:.1f}%[/bold yellow]",
        make_bar(pd_, "yellow")
    )
    outcome_table.add_row(
        f"❌ {t2} Win",
        f"[bold red]{pl/total*100:.1f}%[/bold red]",
        make_bar(pl, "red")
    )

    console.print(outcome_table)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict_text = (
        f"  Model predicts {verdict} to win  ·  "
        f"Confidence: [bold cyan]{top_prob*100:.1f}%[/bold cyan]  "
    )
    console.print(Panel(
        Align.center(Text.from_markup(verdict_text)),
        box=box.HEAVY, border_style="cyan", title="[bold]VERDICT[/bold]"
    ))

    # ── Probability heatmap (text version) ───────────────────────────────────
    _print_heatmap(mat, t1, t2)


def _print_heatmap(mat: np.ndarray, t1: str, t2: str, show_max: int = 6):
    """Small ASCII probability heatmap for score grid."""
    console.print()
    console.print(Rule("[dim]Score Probability Heatmap[/dim]"))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, border_style="dim")
    t.add_column(f"{t1}↓  {t2}→", justify="center", style="bold dim", width=12)
    for g2 in range(show_max + 1):
        t.add_column(str(g2), justify="center", width=8)

    max_p = mat[:show_max+1, :show_max+1].max()

    for g1 in range(show_max + 1):
        row_vals = []
        for g2 in range(show_max + 1):
            p = mat[g1, g2]
            pct = p * 100
            # Colour intensity
            if p >= max_p * 0.8:
                style = "bold bright_yellow"
            elif p >= max_p * 0.5:
                style = "bold cyan"
            elif p >= max_p * 0.25:
                style = "white"
            else:
                style = "dim white"
            row_vals.append(f"[{style}]{pct:.1f}%[/{style}]")
        t.add_row(str(g1), *row_vals)

    console.print(t)
    console.print(f"[dim]  Yellow = highest probability  ·  Goals shown up to {show_max}[/dim]")


def print_team_list(lookup: dict):
    names = sorted(lookup.keys())
    cols  = 4
    rows  = [names[i:i+cols] for i in range(0, len(names), cols)]

    t = Table(box=box.SIMPLE, show_header=False, border_style="dim", padding=(0, 1))
    for _ in range(cols):
        t.add_column("", style="cyan")

    for row in rows:
        while len(row) < cols:
            row.append("")
        t.add_row(*row)

    console.print(Panel(t, title="[bold cyan]Available Teams[/bold cyan]",
                        border_style="cyan"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. MATCH PREDICTION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(
    model: ScorePredictionEnsemble,
    t1_stats: dict, t2_stats: dict,
    t1_name: str,   t2_name: str,
    is_group: bool = True
) -> dict:
    X    = make_feature_row(t1_stats, t2_stats, is_group)
    lam1, lam2 = model.predict_expected_goals(X)
    lam1, lam2 = float(lam1[0]), float(lam2[0])

    mat  = model.score_matrix(lam1, lam2)
    pw, pd_, pl = model.outcomes(mat)

    # Top-3 scorelines
    flat = np.argsort(mat.ravel())[::-1][:20]
    top3 = []
    for idx in flat:
        g1, g2 = divmod(int(idx), mat.shape[1])
        top3.append((g1, g2, float(mat[g1, g2])))
        if len(top3) == 3:
            break

    return {
        "team1":  t1_name,
        "team2":  t2_name,
        "lam1":   lam1,
        "lam2":   lam2,
        "top3":   top3,
        "win":    pw,
        "draw":   pd_,
        "loss":   pl,
        "matrix": mat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, model: ScorePredictionEnsemble, lookup: dict, n: int = 15):
    console.print(Rule("[bold cyan]Backtest – Last 15 Matches[/bold cyan]"))

    t = Table(box=box.ROUNDED, border_style="blue", show_header=True)
    t.add_column("Match",         style="white",       width=26)
    t.add_column("Actual",        style="bold yellow", justify="center", width=8)
    t.add_column("Predicted #1",  style="cyan",        justify="center", width=12)
    t.add_column("Prob",          style="cyan",        justify="center", width=8)
    t.add_column("Outcome ✓?",    style="white",       justify="center", width=10)
    t.add_column("xG (1 vs 2)",   style="dim white",   justify="center", width=14)

    sample = df.tail(n)
    correct_outcome = 0
    exact_scores    = 0

    for _, row in sample.iterrows():
        t1, t2 = row["Team 1"], row["Team 2"]
        s1, s2 = int(row["Goals1"]), int(row["Goals2"])

        t1s = {
            "elo":           row["Elo1"],
            "attack":        row["Attack_Rating_T1"],
            "defense":       row["Defense_Rating_T1"],
            "form_last5":    row["Form_Last5_T1"],
            "form_weighted": row["Form_Weighted_T1"],
        }
        t2s = {
            "elo":           row["Elo2"],
            "attack":        row["Attack_Rating_T2"],
            "defense":       row["Defense_Rating_T2"],
            "form_last5":    row["Form_Last5_T2"],
            "form_weighted": row["Form_Weighted_T2"],
        }
        is_group = "Group" in str(row["Stage"])
        res = predict_match(model, t1s, t2s, t1, t2, is_group)

        pg1, pg2, prob = res["top3"][0]

        actual_outcome  = "W" if s1 > s2 else ("D" if s1 == s2 else "L")
        pred_outcome    = "W" if res["win"] > res["loss"] and res["win"] > res["draw"] else (
                          "D" if res["draw"] >= res["win"] and res["draw"] >= res["loss"] else "L")

        outcome_ok = (actual_outcome == pred_outcome)
        exact_ok   = (pg1 == s1 and pg2 == s2)
        if outcome_ok:   correct_outcome += 1
        if exact_ok:     exact_scores    += 1

        tick = "[green]✓[/green]" if outcome_ok else "[red]✗[/red]"
        match_str = f"{t1[:10]} v {t2[:10]}"
        t.add_row(
            match_str,
            f"{s1}–{s2}",
            f"[{'green' if exact_ok else 'cyan'}]{pg1}–{pg2}[/{'green' if exact_ok else 'cyan'}]",
            f"{prob*100:.1f}%",
            tick,
            f"{res['lam1']:.2f} / {res['lam2']:.2f}",
        )

    console.print(t)
    console.print(
        f"  Outcome correct: [bold green]{correct_outcome}/{n} "
        f"({correct_outcome/n*100:.0f}%)[/bold green]   "
        f"Exact score: [bold yellow]{exact_scores}/{n} "
        f"({exact_scores/n*100:.0f}%)[/bold yellow]\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def fuzzy_find(query: str, lookup: dict) -> str | None:
    """Case-insensitive partial match."""
    q = query.strip().lower()
    # Exact first
    for name in lookup:
        if name.lower() == q:
            return name
    # Prefix
    matches = [n for n in lookup if n.lower().startswith(q)]
    if len(matches) == 1:
        return matches[0]
    # Contains
    matches = [n for n in lookup if q in n.lower()]
    if len(matches) == 1:
        return matches[0]
    return None


def interactive_loop(model: ScorePredictionEnsemble, lookup: dict, df: pd.DataFrame):
    console.print()
    console.print(Panel(
        "[bold white]Commands:[/bold white]\n"
        "  [cyan]predict[/cyan]   – predict a match between two teams\n"
        "  [cyan]list[/cyan]      – show all available teams\n"
        "  [cyan]backtest[/cyan]  – compare predictions vs recent historical results\n"
        "  [cyan]quit[/cyan]      – exit",
        title="[bold cyan]Help[/bold cyan]", border_style="dim", box=box.ROUNDED
    ))

    while True:
        console.print()
        cmd = Prompt.ask("[bold cyan]>[/bold cyan] Enter command", default="predict").strip().lower()

        # ── QUIT ──────────────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        # ── LIST ──────────────────────────────────────────────────────────────
        elif cmd in ("list", "teams", "ls"):
            print_team_list(lookup)

        # ── BACKTEST ──────────────────────────────────────────────────────────
        elif cmd in ("backtest", "bt", "test"):
            run_backtest(df, model, lookup)

        # ── PREDICT ───────────────────────────────────────────────────────────
        elif cmd in ("predict", "p", "pred"):
            console.print("[dim]Type a team name (or part of it). Type [bold]list[/bold] to see all teams.[/dim]")

            # ── Team 1 ────────────────────────────────────────────────────────
            while True:
                t1_input = Prompt.ask("[bold yellow]  Team 1[/bold yellow]").strip()
                if t1_input.lower() == "list":
                    print_team_list(lookup)
                    continue
                t1_name = fuzzy_find(t1_input, lookup)
                if t1_name:
                    console.print(f"  [green]✓  Found:[/green] [bold]{t1_name}[/bold]")
                    break
                console.print(f"  [red]✗  '[/red][bold]{t1_input}[/bold][red]' not found.[/red] "
                              f"Type [cyan]list[/cyan] to see available teams.")

            # ── Team 2 ────────────────────────────────────────────────────────
            while True:
                t2_input = Prompt.ask("[bold yellow]  Team 2[/bold yellow]").strip()
                if t2_input.lower() == "list":
                    print_team_list(lookup)
                    continue
                t2_name = fuzzy_find(t2_input, lookup)
                if t2_name:
                    console.print(f"  [green]✓  Found:[/green] [bold]{t2_name}[/bold]")
                    break
                console.print(f"  [red]✗  '[/red][bold]{t2_input}[/bold][red]' not found.[/red] "
                              f"Type [cyan]list[/cyan] to see available teams.")

            if t1_name == t2_name:
                console.print("[red]A team cannot play itself. Please choose different teams.[/red]")
                continue

            # ── Stage ─────────────────────────────────────────────────────────
            stage_ans = Prompt.ask(
                "  Stage type [bold](G[/bold]=Group, [bold]K[/bold]=Knockout)",
                choices=["G", "K", "g", "k"],
                default="G"
            )
            is_group = stage_ans.upper() == "G"

            # ── Run prediction ────────────────────────────────────────────────
            console.print()
            with console.status("[cyan]Running ensemble prediction…[/cyan]"):
                result = predict_match(
                    model,
                    lookup[t1_name], lookup[t2_name],
                    t1_name, t2_name,
                    is_group
                )

            display_prediction(result)

        else:
            console.print(f"[red]Unknown command:[/red] '{cmd}'. Try [cyan]predict[/cyan], [cyan]list[/cyan], [cyan]backtest[/cyan], or [cyan]quit[/cyan].")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    banner()

    # ── Load data ─────────────────────────────────────────────────────────────
    console.print()
    with console.status("[cyan]Loading dataset…[/cyan]"):
        df     = load_data(DATA_PATH)
        lookup = build_team_lookup(df)
        X      = engineer_features(df)
        y1     = df["Goals1"].values.astype(float)
        y2     = df["Goals2"].values.astype(float)

    console.print(f"  [green]✓[/green] Loaded [bold]{len(df)}[/bold] matches · [bold]{len(lookup)}[/bold] teams\n")

    # ── Train model ───────────────────────────────────────────────────────────
    with console.status(
        "[cyan]Training ensemble  (Poisson GLM + Gradient Boosting + Random Forest)…[/cyan]"
    ):
        model = ScorePredictionEnsemble()
        model.fit(X, y1, y2)

    console.print(f"  [green]✓[/green] Ensemble trained  ·  Dixon-Coles ρ = [bold]{model.rho:+.4f}[/bold]\n")

    # ── Show stats ────────────────────────────────────────────────────────────
    print_model_stats(model.cv_metrics, len(df), len(lookup))

    # ── Quick backtest ────────────────────────────────────────────────────────
    run_backtest(df, model, lookup, n=15)

    # ── Interactive ───────────────────────────────────────────────────────────
    interactive_loop(model, lookup, df)


if __name__ == "__main__":
    main()