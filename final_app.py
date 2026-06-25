#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║   SOCCER ORACLE  —  Streamlit Web App                       ║
║   Advanced Match Prediction  v4.1                           ║
║                                                              ║
║   Install:  pip install streamlit plotly scikit-learn       ║
║             scipy pandas numpy                               ║
║   Run:      streamlit run soccer_oracle_app.py               ║
╚══════════════════════════════════════════════════════════════╝
"""

# ── IMPORTS ──────────────────────────────────────────────────────────────────
import math
import hashlib
import warnings
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
# PAGE CONFIG  ← must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Soccer Oracle",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg:#080D1A; --surf:#101828; --surf2:#192236; --surf3:#1F2D42;
  --accent:#22D3EE; --purple:#A78BFA;
  --t1:#60A5FA; --t2:#FB923C;
  --win:#34D399; --draw:#FBBF24; --loss:#F87171;
  --text:#F1F5F9; --muted:#94A3B8; --border:#1E2D45;
}

/* ─ Base ─ */
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:#060A14!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] .stMarkdown p{color:var(--muted)!important;font-size:0.82rem;}
.block-container{padding-top:1.5rem!important;}

/* ─ Typography ─ */
h1,h2,h3,h4{font-family:'Bebas Neue',sans-serif!important;letter-spacing:0.07em;color:var(--text)!important;}
p,label,[data-testid="stText"]{color:var(--text);}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"]{
  background:var(--surf)!important;border-radius:10px!important;
  padding:4px!important;border:1px solid var(--border)!important;gap:2px!important;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;color:var(--muted)!important;
  border-radius:7px!important;font-weight:500!important;
  padding:0.45rem 1.2rem!important;font-family:'Inter',sans-serif!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(34,211,238,.15),rgba(167,139,250,.15))!important;
  color:var(--text)!important;border:1px solid rgba(34,211,238,.35)!important;
}

/* ─ Buttons ─ */
.stButton>button{
  background:linear-gradient(135deg,#22D3EE,#818CF8)!important;
  color:#080D1A!important;font-weight:700!important;border:none!important;
  border-radius:9px!important;padding:0.65rem 2rem!important;
  font-size:0.95rem!important;letter-spacing:0.06em!important;
  font-family:'Inter',sans-serif!important;transition:opacity .2s!important;
  width:100%;
}
.stButton>button:hover{opacity:.86!important;transform:translateY(-1px);}

/* ─ Select boxes ─ */
.stSelectbox>div>div{
  background:var(--surf2)!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--text)!important;
}
.stSelectbox label{color:var(--muted)!important;font-size:0.78rem!important;
  text-transform:uppercase;letter-spacing:.1em;}

/* ─ Radio ─ */
.stRadio>label{color:var(--muted)!important;font-size:0.78rem!important;
  text-transform:uppercase;letter-spacing:.1em;}
.stRadio [data-testid="stMarkdownContainer"] p{color:var(--text)!important;}

/* ─ File uploader ─ */
[data-testid="stFileUploader"]{
  background:var(--surf2)!important;border:2px dashed var(--border)!important;
  border-radius:10px!important;
}
[data-testid="stFileUploader"] *{color:var(--text)!important;}

/* ─ Metrics ─ */
[data-testid="stMetric"]{background:var(--surf2);border:1px solid var(--border);
  border-radius:10px;padding:.9rem 1rem;}
[data-testid="stMetricValue"]{font-family:'Bebas Neue',sans-serif!important;
  font-size:2rem!important;color:var(--text)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.72rem!important;
  text-transform:uppercase;letter-spacing:.1em;}
[data-testid="stMetricDelta"]{font-size:.8rem!important;}

/* ─ Cards ─ */
.oracle-card{background:var(--surf);border:1px solid var(--border);
  border-radius:12px;padding:1.25rem 1.4rem;margin-bottom:.6rem;}
.oracle-card-glow{background:var(--surf);border:1px solid var(--accent);
  border-radius:12px;padding:1.25rem 1.4rem;
  box-shadow:0 0 24px rgba(34,211,238,.14);margin-bottom:.6rem;}

/* ─ VS Banner ─ */
.vs-banner{display:flex;align-items:center;justify-content:space-between;
  background:var(--surf);border:1px solid var(--border);border-radius:14px;
  padding:1.4rem 2rem;margin-bottom:1.2rem;}
.vs-team-t1{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;
  color:var(--t1);letter-spacing:.06em;line-height:1;}
.vs-team-t2{font-family:'Bebas Neue',sans-serif;font-size:2.6rem;
  color:var(--t2);letter-spacing:.06em;text-align:right;line-height:1;}
.vs-center{display:flex;flex-direction:column;align-items:center;gap:.3rem;}
.vs-score{font-family:'Bebas Neue',sans-serif;font-size:3.4rem;
  color:var(--text);letter-spacing:.12em;line-height:1;}
.vs-badge{font-family:'Bebas Neue',sans-serif;font-size:1rem;
  color:var(--muted);background:var(--surf3);border-radius:6px;
  padding:.2rem .8rem;border:1px solid var(--border);}

/* ─ Outcome Probs ─ */
.outcome-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:.6rem;margin:.8rem 0;}
.outcome-cell{text-align:center;border-radius:10px;padding:1rem .5rem;border:1px solid var(--border);}
.outcome-cell.win{background:rgba(52,211,153,.12);border-color:rgba(52,211,153,.35);}
.outcome-cell.draw{background:rgba(251,191,36,.10);border-color:rgba(251,191,36,.35);}
.outcome-cell.loss{background:rgba(248,113,113,.10);border-color:rgba(248,113,113,.30);}
.outcome-pct{font-family:'Bebas Neue',sans-serif;font-size:2.4rem;line-height:1;}
.outcome-pct.win{color:var(--win);}
.outcome-pct.draw{color:var(--draw);}
.outcome-pct.loss{color:var(--loss);}
.outcome-label{font-size:.72rem;text-transform:uppercase;
  letter-spacing:.12em;color:var(--muted);margin-top:.25rem;}

/* ─ Score Grid ─ */
.score-grid{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem;}
.score-pill{display:inline-flex;align-items:center;gap:.5rem;
  background:var(--surf3);border:1px solid var(--border);
  border-radius:8px;padding:.4rem .8rem;}
.score-pill.top1{border-color:rgba(34,211,238,.5);
  background:rgba(34,211,238,.08);}
.score-num{font-family:'JetBrains Mono',monospace;font-size:1.05rem;
  font-weight:600;color:var(--text);}
.score-pct{font-size:.78rem;font-weight:600;color:var(--accent);}
.score-tag{font-size:.68rem;color:var(--muted);
  background:var(--surf2);border-radius:4px;padding:.1rem .35rem;}

/* ─ Verdict ─ */
.verdict{background:linear-gradient(135deg,rgba(34,211,238,.12),rgba(167,139,250,.12));
  border:1px solid rgba(34,211,238,.4);border-radius:12px;
  padding:1.4rem;text-align:center;margin-top:.8rem;}
.verdict-main{font-family:'Bebas Neue',sans-serif;font-size:2rem;
  letter-spacing:.1em;color:var(--accent);}
.verdict-sub{font-size:.8rem;color:var(--muted);margin-top:.25rem;}

/* ─ Stats comparison ─ */
.stat-row{display:grid;grid-template-columns:1fr 120px 1fr;
  align-items:center;gap:.5rem;padding:.5rem 0;
  border-bottom:1px solid var(--border);}
.stat-row:last-child{border-bottom:none;}
.stat-val-t1{text-align:right;font-family:'JetBrains Mono',monospace;
  font-size:.9rem;color:var(--t1);}
.stat-val-t2{text-align:left;font-family:'JetBrains Mono',monospace;
  font-size:.9rem;color:var(--t2);}
.stat-name{text-align:center;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);}

/* ─ Backtest table ─ */
.bt-row{display:grid;grid-template-columns:2fr 60px 60px 60px 55px 90px;
  gap:.3rem;align-items:center;padding:.5rem .6rem;border-radius:6px;
  margin:.2rem 0;font-size:.82rem;}
.bt-row.correct{background:rgba(52,211,153,.06);}
.bt-row.wrong{background:rgba(248,113,113,.06);}
.bt-header{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
.bt-tick{font-size:1rem;}

/* ─ Section header ─ */
.sh{display:flex;align-items:center;gap:.5rem;
  padding:.4rem 0 .6rem;border-bottom:1px solid var(--border);margin-bottom:.75rem;}
.sh-dot{width:7px;height:7px;border-radius:50%;background:var(--accent);flex-shrink:0;}
.sh-title{font-family:'Bebas Neue',sans-serif;font-size:1.15rem;
  letter-spacing:.08em;color:var(--text);}

/* ─ Scrollbar ─ */
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:var(--surf);}
::-webkit-scrollbar-thumb{background:var(--surf3);border-radius:3px;}

/* ─ Sidebar logo ─ */
.sidebar-logo{text-align:center;padding:1rem 0 .5rem;}
.sidebar-logo-text{font-family:'Bebas Neue',sans-serif;font-size:1.9rem;
  letter-spacing:.1em;color:var(--accent);}
.sidebar-logo-sub{font-size:.72rem;color:var(--muted);letter-spacing:.15em;
  text-transform:uppercase;margin-top:-.25rem;}

/* ─ DataFrames ─ */
[data-testid="stDataFrame"]{border-radius:8px!important;}
[data-testid="stDataFrame"] table{background:var(--surf)!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONSTANTS
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
STRENGTH_SMOOTHING = 3.0
H2H_HALF_LIFE_DAYS = 1825.0
EPS = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    required = ["Team 1", "Team 2", "Goals1", "Goals2", "Elo1", "Elo2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
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
# FEATURE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class FeatureEngine:
    def __init__(self, window: int = 8):
        self.window = window
        self._team_goals_scored: Dict[str, List[float]] = defaultdict(list)
        self._team_goals_conceded: Dict[str, List[float]] = defaultdict(list)
        self._team_adj_for: Dict[str, List[float]] = defaultdict(list)
        self._team_adj_against: Dict[str, List[float]] = defaultdict(list)
        self._team_elo_history: Dict[str, List[float]] = defaultdict(list)
        self._team_results: Dict[str, List[float]] = defaultdict(list)
        self._team_match_count: Dict[str, int] = defaultdict(int)
        self._team_opponent_elos: Dict[str, List[float]] = defaultdict(list)
        self._league_goal_history: List[float] = []
        self._h2h: Dict[
            Tuple[str, str],
            List[Tuple[str, str, int, int, Optional[pd.Timestamp]]],
        ] = defaultdict(list)

    def _weighted_mean(self, vals, n=None, smoothing=STRENGTH_SMOOTHING):
        n = n or self.window
        if not vals:
            return 0.0
        tail = np.asarray(vals[-n:], dtype=float)
        return float((tail.sum() + smoothing * float(np.mean(tail))) / (len(tail) + smoothing))

    def _rolling_mean(self, vals, n=None):
        n = n or self.window
        return float(np.mean(vals[-n:])) if vals else 0.0

    def _rolling_std(self, vals, n=None):
        n = n or self.window
        return float(np.std(vals[-n:])) if len(vals) >= 2 else 0.5

    def _weighted_form(self, results):
        if not results:
            return DEFAULT_FORM
        r = results[-self.window:]
        weights = np.exp(np.linspace(-1.5, 0.0, len(r)))
        return float(np.average(r, weights=weights) / 3.0 * 100.0)

    def _league_avg_goals_per_team(self):
        if not self._league_goal_history:
            return DEFAULT_LEAGUE_GOALS_PER_TEAM
        return float(np.mean(self._league_goal_history[-self.window:]))

    def _team_strengths(self, team: str) -> dict:
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
        attack_strength = (
            self._weighted_mean(adj_for) / max(league_avg, 0.1) if adj_for else DEFAULT_ATTACK_STRENGTH
        )
        defense_weakness = (
            self._weighted_mean(adj_against) / max(league_avg, 0.1) if adj_against else DEFAULT_DEFENSE_WEAKNESS
        )
        elo_last = elo_hist[-1] if elo_hist else DEFAULT_ELO
        elo_trend = (elo_hist[-1] - elo_hist[-5]) if len(elo_hist) >= 5 else 0.0
        elo_trend3 = (elo_hist[-1] - elo_hist[-3]) if len(elo_hist) >= 3 else 0.0
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
            "consistency": 1.0 / (self._rolling_std(gs) + 0.5),
            "clean_sheet_rate": clean_sheet,
            "form_last5": form5,
            "form_weighted": form_w,
            "btts_rate": btts,
            "match_count": mc,
            "opp_elo_avg": opp_elo_avg,
        }

    def _h2h_stats(self, t1, t2, current_date=None):
        key = tuple(sorted([t1, t2]))
        history = self._h2h.get(key, [])
        if not history:
            return 1 / 3, 1 / 3, 1 / 3, 0
        w = d = l = 0.0
        total_w = 0.0
        for a, b, g_a, g_b, dt in history:
            s1, s2 = (g_a, g_b) if a == t1 else (g_b, g_a)
            weight = 1.0
            if current_date is not None and pd.notna(current_date) and dt is not None and pd.notna(dt):
                age_days = max((current_date - dt).days, 0)
                weight = math.exp(-age_days / H2H_HALF_LIFE_DAYS)
            total_w += weight
            if s1 > s2:
                w += weight
            elif s1 == s2:
                d += weight
            else:
                l += weight
        prior = 1.0
        denom = total_w + 3.0 * prior
        return (w + prior) / denom, (d + prior) / denom, (l + prior) / denom, len(history)

    def _core_features(self, t1, t2, elo1, elo2, s1, s2, is_group, current_date):
        league_avg = self._league_avg_goals_per_team()
        elo_diff = elo1 - elo2
        elo_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        elo_sum = elo1 + elo2
        elo_ratio = elo1 / max(elo2, 1.0)
        trend_diff = s1["elo_trend"] - s2["elo_trend"]
        trend3_diff = s1["elo_trend3"] - s2["elo_trend3"]
        stage_goal_mod = 1.0 if is_group else 0.92
        base_xg1 = league_avg * s1["attack_strength"] * s2["defense_weakness"] * stage_goal_mod
        base_xg2 = league_avg * s2["attack_strength"] * s1["defense_weakness"] * stage_goal_mod
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
        feat = {
            "elo_diff": elo_diff, "elo_win_prob": elo_win_prob, "elo_sum": elo_sum,
            "elo_ratio": elo_ratio, "elo1": elo1, "elo2": elo2,
            "elo_xg1": base_xg1, "elo_xg2": base_xg2,
            "elo_trend_diff": trend_diff, "elo_trend3_diff": trend3_diff,
            "elo_trend_t1": s1["elo_trend"], "elo_trend_t2": s2["elo_trend"],
            "raw_attack_t1": s1["raw_attack"], "raw_attack_t2": s2["raw_attack"],
            "raw_defense_t1": s1["raw_defense"], "raw_defense_t2": s2["raw_defense"],
            "attack_strength_t1": s1["attack_strength"], "attack_strength_t2": s2["attack_strength"],
            "attack_strength_diff": s1["attack_strength"] - s2["attack_strength"],
            "defense_weakness_t1": s1["defense_weakness"], "defense_weakness_t2": s2["defense_weakness"],
            "defense_weakness_diff": s1["defense_weakness"] - s2["defense_weakness"],
            "defense_strength_t1": def1_strength, "defense_strength_t2": def2_strength,
            "xg_base_t1": base_xg1, "xg_base_t2": base_xg2,
            "xg_diff": base_xg1 - base_xg2, "xg_sum": base_xg1 + base_xg2,
            "form_last5_t1": s1["form_last5"], "form_last5_t2": s2["form_last5"],
            "form_diff_last5": s1["form_last5"] - s2["form_last5"],
            "form_weighted_t1": fw1, "form_weighted_t2": fw2, "form_diff_weighted": fw1 - fw2,
            "goal_var_t1": self._rolling_std(self._team_goals_scored[t1]),
            "goal_var_t2": self._rolling_std(self._team_goals_scored[t2]),
            "concede_var_t1": self._rolling_std(self._team_goals_conceded[t1]),
            "concede_var_t2": self._rolling_std(self._team_goals_conceded[t2]),
            "consistency_t1": s1["consistency"], "consistency_t2": s2["consistency"],
            "clean_sheet_t1": s1["clean_sheet_rate"], "clean_sheet_t2": s2["clean_sheet_rate"],
            "btts_t1": s1["btts_rate"], "btts_t2": s2["btts_rate"],
            "strength_t1": strength_t1, "strength_t2": strength_t2,
            "strength_diff": strength_t1 - strength_t2,
            "h2h_advantage": h2hw - h2hl,
            "h2h_win_rate_t1": h2hw, "h2h_draw_rate": h2hd,
            "h2h_win_rate_t2": h2hl,
            "h2h_matches": h2hn,
            "is_group": int(is_group), "is_knockout": int(not is_group),
            "stage_goal_mod": stage_goal_mod,
            "avg_total_goals_pred": base_xg1 + base_xg2,
            "goal_ratio": base_xg1 / max(base_xg2, 0.1),
            "opp_elo_avg_t1": s1["opp_elo_avg"], "opp_elo_avg_t2": s2["opp_elo_avg"],
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
            elo1, elo2 = float(row["Elo1"]), float(row["Elo2"])
            g1, g2 = int(row["Goals1"]), int(row["Goals2"])
            stage = str(row.get("Stage", "Group")).strip().lower()
            is_group = "group" in stage or stage == "g"
            row_date = row["Date"] if "Date" in row and pd.notna(row["Date"]) else None

            s1, s2 = self.snapshot(t1), self.snapshot(t2)
            feat, _, _ = self._core_features(t1, t2, elo1, elo2, s1, s2, is_group, row_date)
            feature_rows.append(feat)

            league_avg = self._league_avg_goals_per_team()
            opp_def_t2 = max(s2["defense_weakness"], 0.75)
            opp_att_t2 = max(s2["attack_strength"], 0.75)
            opp_def_t1 = max(s1["defense_weakness"], 0.75)
            opp_att_t1 = max(s1["attack_strength"], 0.75)

            for team, gs, gc, af, aa, elo_v, opp_elo, goals_for, goals_ag in [
                (t1, g1, g2, g1 / opp_def_t2, g2 / opp_att_t2, elo1, elo2, g1, g2),
                (t2, g2, g1, g2 / opp_def_t1, g1 / opp_att_t1, elo2, elo1, g2, g1),
            ]:
                self._team_goals_scored[team].append(gs)
                self._team_goals_conceded[team].append(gc)
                self._team_adj_for[team].append(af)
                self._team_adj_against[team].append(aa)
                self._team_elo_history[team].append(elo_v)
                self._team_results[team].append(
                    3.0 if goals_for > goals_ag else (1.0 if goals_for == goals_ag else 0.0)
                )
                self._team_match_count[team] += 1
                self._team_opponent_elos[team].append(opp_elo)

            self._league_goal_history.append((g1 + g2) / 2.0)
            key = tuple(sorted([t1, t2]))
            self._h2h[key].append((t1, t2, g1, g2, row_date))

        return pd.DataFrame(feature_rows, index=df.index)

    def build_lookup(self) -> dict:
        return {team: self.snapshot(team) for team in self._team_match_count}

    def predict_features(self, t1, t2, t1_elo, t2_elo, t1_stats, t2_stats, is_group=True) -> pd.DataFrame:
        feat, _, _ = self._core_features(t1, t2, t1_elo, t2_elo, t1_stats, t2_stats, is_group, None)
        return pd.DataFrame([feat])


# ─────────────────────────────────────────────────────────────────────────────
# DIXON-COLES
# ─────────────────────────────────────────────────────────────────────────────
def dc_tau(g1, g2, lam1, lam2, rho):
    if g1 == 0 and g2 == 0:
        return max(1e-9, 1 - lam1 * lam2 * rho)
    elif g1 == 1 and g2 == 0:
        return max(1e-9, 1 + lam2 * rho)
    elif g1 == 0 and g2 == 1:
        return max(1e-9, 1 + lam1 * rho)
    elif g1 == 1 and g2 == 1:
        return max(1e-9, 1 - rho)
    return 1.0


def estimate_rho(goals1, goals2, lam1_vals, lam2_vals):
    def neg_ll(rho):
        if abs(rho) >= 0.99:
            return 1e12
        ll = sum(
            math.log(max(dc_tau(int(g1), int(g2), float(l1), float(l2), rho), 1e-12))
            for g1, g2, l1, l2 in zip(goals1, goals2, lam1_vals, lam2_vals)
            if int(g1) <= 1 and int(g2) <= 1
        )
        return -ll
    return float(minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded").x)


def score_log_prob(y1, y2, lam1, lam2, rho):
    lp = poisson.logpmf(y1, lam1) + poisson.logpmf(y2, lam2)
    if y1 <= 1 and y2 <= 1:
        lp += math.log(max(dc_tau(y1, y2, lam1, lam2, rho), 1e-12))
    return float(lp)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SoccerEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rho = 0.0
        self.w1 = np.array([1/3, 1/3, 1/3])
        self.w2 = np.array([1/3, 1/3, 1/3])
        self.feature_cols: List[str] = []
        self.models_t1 = [
            PoissonRegressor(alpha=0.08, max_iter=5000),
            GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                      subsample=0.85, min_samples_leaf=3, random_state=42),
            RandomForestRegressor(n_estimators=350, max_depth=8, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1),
        ]
        self.models_t2 = [
            PoissonRegressor(alpha=0.08, max_iter=5000),
            GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                      subsample=0.85, min_samples_leaf=3, random_state=42),
            RandomForestRegressor(n_estimators=350, max_depth=8, min_samples_leaf=3,
                                  random_state=42, n_jobs=-1),
        ]

    def _augment(self, X, y1, y2, w):
        feat_cols = X.columns.tolist()
        swap_map = {c: c[:-3] + "_t2" for c in feat_cols if c.endswith("_t1") and c[:-3] + "_t2" in feat_cols}
        X_rev = X.copy()
        # Read all values from original X before writing to X_rev to avoid
        # pandas chained-assignment issues on simultaneous swap
        for c1, c2 in swap_map.items():
            v1, v2 = X[c2].values.copy(), X[c1].values.copy()
            X_rev[c1] = v1
            X_rev[c2] = v2
        for col in ["elo_diff", "attack_strength_diff", "defense_weakness_diff",
                    "xg_diff", "form_diff_last5", "form_diff_weighted",
                    "strength_diff", "elo_trend_diff", "elo_trend3_diff",
                    "h2h_advantage", "goal_ratio", "opp_elo_diff"]:
            if col in feat_cols:
                X_rev[col] = -X[col].values
        if "elo_ratio" in feat_cols:
            X_rev["elo_ratio"] = 1.0 / np.clip(X["elo_ratio"].values, 0.01, None)
        if "elo_win_prob" in feat_cols:
            X_rev["elo_win_prob"] = 1.0 - X["elo_win_prob"].values
        for a, b in [("elo1", "elo2"), ("elo_xg1", "elo_xg2")]:
            if a in feat_cols and b in feat_cols:
                va, vb = X[b].values.copy(), X[a].values.copy()
                X_rev[a] = va
                X_rev[b] = vb
        if "h2h_win_rate_t1" in feat_cols and "h2h_win_rate_t2" in feat_cols:
            v1, v2 = X["h2h_win_rate_t2"].values.copy(), X["h2h_win_rate_t1"].values.copy()
            X_rev["h2h_win_rate_t1"] = v1
            X_rev["h2h_win_rate_t2"] = v2
        X_all = pd.concat([X, X_rev], ignore_index=True)
        return X_all.values, np.concatenate([y1, y2]), np.concatenate([y2, y1]), np.concatenate([w, w])

    @staticmethod
    def _tgt(y, idx):
        return y if idx == 0 else np.log1p(y)

    @staticmethod
    def _pred(p, idx):
        return np.clip(p, 0, 10) if idx == 0 else np.clip(np.expm1(p), 0, 10)

    def _fit_side(self, models, Xs, y, w):
        for i, m in enumerate(models):
            m.fit(Xs, self._tgt(y, i), sample_weight=w)

    def _predict_side(self, models, Xs, weights):
        preds = [self._pred(np.asarray(m.predict(Xs), dtype=float), i) for i, m in enumerate(models)]
        return np.clip(sum(w * p for w, p in zip(weights, preds)), 0.05, 7.5)

    def fit(self, X, y1, y2, sample_weight, cv_X, cv_y1, cv_y2, cv_w):
        self.feature_cols = X.columns.tolist()
        y1, y2 = np.asarray(y1, float), np.asarray(y2, float)
        w = np.asarray(sample_weight, float)
        Xa, ya1, ya2, wa = self._augment(X, y1, y2, w)
        Xs = self.scaler.fit_transform(Xa)
        self._fit_side(self.models_t1, Xs, ya1, wa)
        self._fit_side(self.models_t2, Xs, ya2, wa)

        if len(cv_X) >= 10:
            n_splits = min(ENSEMBLE_CV, max(2, len(cv_X) // 30), len(cv_X) - 1)
            if n_splits >= 2:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_arr = cv_X.values
                cv_y1, cv_y2, cv_w = np.asarray(cv_y1, float), np.asarray(cv_y2, float), np.asarray(cv_w, float)

                def cv_mse(base_est, target_y, side):
                    mses = []
                    for tr, val in tscv.split(cv_arr):
                        sc2 = StandardScaler()
                        Xtr = sc2.fit_transform(cv_arr[tr])
                        Xval = sc2.transform(cv_arr[val])
                        est = clone(base_est)
                        Xtr_df = pd.DataFrame(Xtr, columns=self.feature_cols)
                        Xa2, ya12, ya22, wa2 = self._augment(Xtr_df, cv_y1[tr], cv_y2[tr], cv_w[tr])
                        Xas = sc2.transform(Xa2)
                        y_use = ya12 if side == 1 else ya22
                        if not isinstance(est, PoissonRegressor):
                            y_use = np.log1p(y_use)
                        est.fit(Xas, y_use, sample_weight=wa2)
                        raw = np.asarray(est.predict(Xval), float)
                        pred = raw if isinstance(est, PoissonRegressor) else np.clip(np.expm1(raw), 0, 10)
                        mses.append(mean_squared_error(target_y[val], pred))
                    return float(np.mean(mses)) if mses else 1e6

                m1 = np.array([cv_mse(m, cv_y1, 1) for m in self.models_t1])
                m2 = np.array([cv_mse(m, cv_y2, 2) for m in self.models_t2])
                i1, i2 = 1 / (m1 + 1e-9), 1 / (m2 + 1e-9)
                self.w1, self.w2 = i1 / i1.sum(), i2 / i2.sum()

        Xs_orig = self.scaler.transform(X.values)
        l1, l2 = self._predict_side(self.models_t1, Xs_orig, self.w1), self._predict_side(self.models_t2, Xs_orig, self.w2)
        self.rho = estimate_rho(y1, y2, l1, l2)
        return self

    def predict_xg(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X_use = X.copy()
        for col in self.feature_cols:
            if col not in X_use.columns:
                X_use[col] = 0.0
        Xs = self.scaler.transform(X_use[self.feature_cols].values)
        return (
            np.clip(self._predict_side(self.models_t1, Xs, self.w1), 0.05, 7.0),
            np.clip(self._predict_side(self.models_t2, Xs, self.w2), 0.05, 7.0),
        )

    def score_matrix(self, lam1: float, lam2: float) -> np.ndarray:
        g = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(g, lam1), poisson.pmf(g, lam2))
        for i in range(min(2, MAX_GOALS + 1)):
            for j in range(min(2, MAX_GOALS + 1)):
                mat[i, j] *= dc_tau(i, j, lam1, lam2, self.rho)
        total = mat.sum()
        if total > 0:
            mat /= total
        return mat

    def outcome_probs(self, mat: np.ndarray) -> Tuple[float, float, float]:
        return float(np.tril(mat, -1).sum()), float(np.trace(mat)), float(np.triu(mat, 1).sum())


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def recency_weights(df):
    dates = pd.to_datetime(df["Date"], errors="coerce")
    if dates.notna().sum() == 0:
        return np.ones(len(df))
    age = (dates.max() - dates).dt.days.fillna(9999).astype(float)
    w = np.exp(-age / RECENCY_HALF_LIFE_DAYS)
    return np.where(np.isfinite(w), w, 1.0)


def train_pipeline(df):
    engine = FeatureEngine(window=8)
    feat_df = engine.compute_all_features(df)
    y1, y2 = df["Goals1"].values.astype(float), df["Goals2"].values.astype(float)
    w = recency_weights(df)
    model = SoccerEnsemble()
    model.fit(feat_df, y1, y2, w, feat_df, y1, y2, w)
    return model, engine, feat_df


def evaluate(model, feat_df, df):
    y1, y2 = df["Goals1"].values.astype(int), df["Goals2"].values.astype(int)
    lam1, lam2 = model.predict_xg(feat_df)
    exact, outcome_correct, nlls = [], [], []
    for i in range(len(df)):
        mat = model.score_matrix(float(lam1[i]), float(lam2[i]))
        pw, pd_, pl = model.outcome_probs(mat)
        pred_out = np.argmax([pw, pd_, pl])
        true_out = 0 if y1[i] > y2[i] else (1 if y1[i] == y2[i] else 2)
        outcome_correct.append(int(pred_out == true_out))
        exact.append(int(y1[i] == round(float(lam1[i])) and y2[i] == round(float(lam2[i]))))
        nlls.append(-score_log_prob(y1[i], y2[i], float(lam1[i]), float(lam2[i]), model.rho))
    return {
        "rmse_t1": float(np.sqrt(mean_squared_error(y1, lam1))),
        "rmse_t2": float(np.sqrt(mean_squared_error(y2, lam2))),
        "mae_t1": float(mean_absolute_error(y1, lam1)),
        "mae_t2": float(mean_absolute_error(y2, lam2)),
        "exact_score_pct": float(np.mean(exact) * 100),
        "outcome_acc_pct": float(np.mean(outcome_correct) * 100),
        "score_nll": float(np.mean(nlls)),
        "rho": float(model.rho),
        "w1": model.w1.tolist(),
        "w2": model.w2.tolist(),
    }


def predict_match(model, engine, t1, t2, is_group=True):
    lookup = engine.build_lookup()
    s1, s2 = lookup[t1], lookup[t2]
    feat = engine.predict_features(t1, t2, s1["elo"], s2["elo"], s1, s2, is_group)
    for col in model.feature_cols:
        if col not in feat.columns:
            feat[col] = 0.0
    feat = feat[model.feature_cols]
    lam1_arr, lam2_arr = model.predict_xg(feat)
    lam1, lam2 = float(lam1_arr[0]), float(lam2_arr[0])
    mat = model.score_matrix(lam1, lam2)
    pw, pd_, pl = model.outcome_probs(mat)
    flat = np.argsort(mat.ravel())[::-1]
    top5 = [(divmod(int(i), mat.shape[1])[0], divmod(int(i), mat.shape[1])[1], float(mat.ravel()[i]))
            for i in flat[:5]]
    return {"team1": t1, "team2": t2, "lam1": lam1, "lam2": lam2,
            "top5": top5, "win": pw, "draw": pd_, "loss": pl,
            "matrix": mat, "stats1": s1, "stats2": s2}


# ─────────────────────────────────────────────────────────────────────────────
# CACHED TRAINING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cached_train(file_hash: str, _df: pd.DataFrame):
    """Train both evaluation and final models, cached by CSV hash."""
    df = _df
    holdout_n = max(1, min(max(MIN_HOLDOUT_MATCHES, int(len(df) * HOLDOUT_FRACTION)), len(df) - 1))
    train_df = df.iloc[:-holdout_n].reset_index(drop=True)
    test_df = df.iloc[-holdout_n:].reset_index(drop=True)

    eval_model, eval_engine, _ = train_pipeline(train_df)
    feat_test = eval_engine.compute_all_features(test_df)
    metrics = evaluate(eval_model, feat_test, test_df)

    final_model, final_engine, _ = train_pipeline(df)
    lookup = final_engine.build_lookup()

    return {
        "final_model": final_model,
        "final_engine": final_engine,
        "eval_model": eval_model,
        "eval_engine": eval_engine,
        "metrics": metrics,
        "test_df": test_df,
        "feat_test": feat_test,
        "teams": sorted(lookup.keys()),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_matches": len(df),
    }


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# FIX: margin removed from shared layout dict to avoid "multiple values for
# keyword argument 'margin'" when individual charts pass their own margin.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8"),
)

# Default tight margin re-used by charts that don't need a custom one
_MARGIN_DEFAULT = dict(l=10, r=10, t=30, b=10)


def score_heatmap(mat: np.ndarray, t1: str, t2: str) -> go.Figure:
    n = 7
    z = mat[:n, :n] * 100
    text = [[f"{z[i,j]:.1f}%" for j in range(n)] for i in range(n)]

    colorscale = [
        [0.0, "#101828"],
        [0.05, "#0d3349"],
        [0.2, "#0e4f6b"],
        [0.5, "#0e7490"],
        [0.8, "#22D3EE"],
        [1.0, "#bffdff"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        textfont=dict(size=10, family="JetBrains Mono"),
        colorscale=colorscale, showscale=False,
        hovertemplate=f"{t1} %{{y}} – %{{x}} {t2}: %{{z:.2f}}%<extra></extra>",
        xgap=2, ygap=2,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=_MARGIN_DEFAULT,
        xaxis=dict(
            title=dict(text=t2[:14], font=dict(color="#FB923C", size=12)),
            tickvals=list(range(n)), ticktext=list(range(n)),
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text=t1[:14], font=dict(color="#60A5FA", size=12)),
            tickvals=list(range(n)), ticktext=list(range(n)),
            autorange="reversed", tickfont=dict(size=11),
        ),
        height=310,
    )

    # Highlight diagonal (draws)
    for i in range(n):
        fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5,
                      line=dict(color="#FBBF24", width=1.2), fillcolor="rgba(0,0,0,0)")
    return fig


def radar_chart(s1: dict, s2: dict, t1: str, t2: str) -> go.Figure:
    cats = ["Attack\nStrength", "Defense\nStrength", "Form", "Clean Sheets", "BTTS Rate", "Consistency"]

    def norm(val, lo, hi):
        return max(0.0, min(1.0, (val - lo) / (hi - lo + 1e-9)))

    v1 = [
        norm(s1["attack_strength"], 0, 2.5),
        norm(s1["defense_strength"], 0, 3.0),
        norm(s1["form_weighted"], 0, 100),
        s1["clean_sheet_rate"],
        s1["btts_rate"],
        norm(s1["consistency"], 0, 3.0),
    ]
    v2 = [
        norm(s2["attack_strength"], 0, 2.5),
        norm(s2["defense_strength"], 0, 3.0),
        norm(s2["form_weighted"], 0, 100),
        s2["clean_sheet_rate"],
        s2["btts_rate"],
        norm(s2["consistency"], 0, 3.0),
    ]

    def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()

    for vals, name, color in [(v1, t1, "#60A5FA"), (v2, t2, "#FB923C")]:
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=name[:14],
                line=dict(color=color, width=2),
                fillcolor=hex_to_rgba(color, 0.12),
            )
        )

    # FIX: margin is specified here directly (not via PLOTLY_LAYOUT) to avoid
    # "multiple values for keyword argument 'margin'" TypeError.
    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=dict(l=30, r=30, t=20, b=30),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9),
                            gridcolor="#1E2D45", linecolor="#1E2D45"),
            angularaxis=dict(tickfont=dict(size=10, family="Inter"), gridcolor="#1E2D45",
                             linecolor="#1E2D45"),
        ),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        height=300,
    )
    return fig


def prob_donut(pw: float, pd_: float, pl: float, t1: str, t2: str) -> go.Figure:
    total = pw + pd_ + pl
    labels = [f"{t1[:12]} Win", "Draw", f"{t2[:12]} Win"]
    values = [pw / total * 100, pd_ / total * 100, pl / total * 100]
    colors = ["#60A5FA", "#FBBF24", "#FB923C"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#080D1A", width=3)),
        textfont=dict(size=11, family="Inter"),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=_MARGIN_DEFAULT,
        height=240,
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                    font=dict(size=10)),
    )
    return fig


def backtest_chart(test_df, feat_test, eval_model) -> go.Figure:
    y1 = test_df["Goals1"].values
    y2 = test_df["Goals2"].values
    lam1_all, lam2_all = eval_model.predict_xg(feat_test)

    correct_cum = []
    correct = 0
    for i, (_, row) in enumerate(test_df.iterrows()):
        mat = eval_model.score_matrix(float(lam1_all[i]), float(lam2_all[i]))
        pw, pd_, pl = eval_model.outcome_probs(mat)
        pred = ["W", "D", "L"][np.argmax([pw, pd_, pl])]
        act = "W" if y1[i] > y2[i] else ("D" if y1[i] == y2[i] else "L")
        if pred == act:
            correct += 1
        correct_cum.append(correct / (i + 1) * 100)

    fig = go.Figure()
    fig.add_hline(y=33.3, line=dict(dash="dash", color="#F87171", width=1),
                  annotation_text="Random (33%)", annotation_font=dict(size=9, color="#F87171"))
    fig.add_trace(go.Scatter(
        x=list(range(1, len(correct_cum) + 1)),
        y=correct_cum,
        mode="lines",
        line=dict(color="#22D3EE", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(34,211,238,0.08)",
        hovertemplate="Match %{x}: %{y:.1f}% accuracy<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        margin=_MARGIN_DEFAULT,
        xaxis=dict(title=dict(text="Matches"), gridcolor="#1E2D45", linecolor="#1E2D45"),
        yaxis=dict(title=dict(text="Cumulative Accuracy (%)"), gridcolor="#1E2D45",
                   linecolor="#1E2D45", range=[0, 105]),
        height=240,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def section(title: str, icon: str = ""):
    st.markdown(f"""
<div class="sh">
  <div class="sh-dot"></div>
  <div class="sh-title">{icon}&nbsp;{title}</div>
</div>""", unsafe_allow_html=True)


def vs_banner(t1: str, t2: str, lam1: float, lam2: float):
    st.markdown(f"""
<div class="vs-banner">
  <div class="vs-team-t1">{t1}</div>
  <div class="vs-center">
    <div class="vs-score">{lam1:.1f} – {lam2:.1f}</div>
    <div class="vs-badge">Expected Goals</div>
  </div>
  <div class="vs-team-t2">{t2}</div>
</div>""", unsafe_allow_html=True)


def outcome_cards(pw: float, pd_: float, pl: float, t1: str, t2: str):
    total = pw + pd_ + pl
    w_pct, d_pct, l_pct = pw / total * 100, pd_ / total * 100, pl / total * 100
    st.markdown(f"""
<div class="outcome-grid">
  <div class="outcome-cell win">
    <div class="outcome-pct win">{w_pct:.0f}%</div>
    <div class="outcome-label">{t1[:12]} Win</div>
  </div>
  <div class="outcome-cell draw">
    <div class="outcome-pct draw">{d_pct:.0f}%</div>
    <div class="outcome-label">Draw</div>
  </div>
  <div class="outcome-cell loss">
    <div class="outcome-pct loss">{l_pct:.0f}%</div>
    <div class="outcome-label">{t2[:12]} Win</div>
  </div>
</div>""", unsafe_allow_html=True)


def top_scores(top5: list, t1: str, t2: str):
    html = '<div class="score-grid">'
    for idx, (g1, g2, prob) in enumerate(top5):
        outcome = f"{t1[:8]} Win" if g1 > g2 else ("Draw" if g1 == g2 else f"{t2[:8]} Win")
        cls = "score-pill top1" if idx == 0 else "score-pill"
        html += f"""
<div class="{cls}">
  <span class="score-num">{g1}–{g2}</span>
  <span class="score-tag">{outcome}</span>
  <span class="score-pct">{prob*100:.1f}%</span>
</div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def verdict_box(pw: float, pd_: float, pl: float, t1: str, t2: str):
    total = pw + pd_ + pl
    if pw >= pd_ and pw >= pl:
        txt = f"{t1} TO WIN"
        conf = f"{pw/total*100:.0f}% confidence"
    elif pd_ >= pw and pd_ >= pl:
        txt = "MATCH ENDS IN A DRAW"
        conf = f"{pd_/total*100:.0f}% confidence"
    else:
        txt = f"{t2} TO WIN"
        conf = f"{pl/total*100:.0f}% confidence"
    st.markdown(f"""
<div class="verdict">
  <div class="verdict-main">⚽ {txt}</div>
  <div class="verdict-sub">{conf} based on Poisson × Gradient Boost × Random Forest ensemble</div>
</div>""", unsafe_allow_html=True)


def stats_comparison(s1: dict, s2: dict, t1: str, t2: str):
    rows = [
        ("Elo Rating",         "elo",              ".0f",  None),
        ("Elo Trend (5 gms)",  "elo_trend",        "+.1f", None),
        ("Goals Scored (avg)", "raw_attack",        ".2f",  "higher"),
        ("Goals Conceded",     "raw_defense",       ".2f",  "lower"),
        ("Attack Strength",    "attack_strength",   ".2f",  "higher"),
        ("Defense Weakness",   "defense_weakness",  ".2f",  "lower"),
        ("Form (last 5)",      "form_last5",        ".1f",  "higher"),
        ("Clean Sheet Rate",   "clean_sheet_rate",  ".1%",  "higher"),
        ("BTTS Rate",          "btts_rate",         ".1%",  None),
        ("Consistency",        "consistency",       ".2f",  "higher"),
    ]
    html = f"""
<div style="margin-top:.5rem;">
  <div class="stat-row">
    <div class="stat-val-t1" style="color:#94A3B8;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;text-align:right;">{t1[:16]}</div>
    <div class="stat-name" style="color:#94A3B8;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;">Stat</div>
    <div class="stat-val-t2" style="color:#94A3B8;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;">{t2[:16]}</div>
  </div>"""
    for label, key, fmt, better in rows:
        v1, v2 = s1.get(key, 0), s2.get(key, 0)
        c1, c2 = "#60A5FA", "#FB923C"
        if better == "higher":
            if v1 > v2:
                c1 = "#34D399"
            elif v2 > v1:
                c2 = "#34D399"
        elif better == "lower":
            if v1 < v2:
                c1 = "#34D399"
            elif v2 < v1:
                c2 = "#34D399"
        f1 = format(v1, fmt)
        f2 = format(v2, fmt)
        html += f"""
  <div class="stat-row">
    <div class="stat-val-t1" style="color:{c1};">{f1}</div>
    <div class="stat-name">{label}</div>
    <div class="stat-val-t2" style="color:{c2};">{f2}</div>
  </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────────────────────
def page_predict(state: dict):
    model = state["final_model"]
    engine = state["final_engine"]
    teams = state["teams"]

    st.markdown("### Match Prediction")

    col_t1, col_vs, col_t2 = st.columns([5, 1, 5])
    with col_t1:
        t1_default_idx = teams.index(st.session_state.get("t1_prev", teams[0])) \
            if st.session_state.get("t1_prev") in teams else 0
        t1 = st.selectbox("🔵 Team 1", teams, key="t1_sel", index=t1_default_idx)
    with col_vs:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center;font-family:\"Bebas Neue\",sans-serif;"
            "font-size:1.4rem;color:#94A3B8;padding-top:.7rem'>VS</div>",
            unsafe_allow_html=True,
        )
    with col_t2:
        default_t2 = teams[1] if len(teams) > 1 else teams[0]
        t2_default_idx = teams.index(st.session_state.get("t2_prev", default_t2)) \
            if st.session_state.get("t2_prev") in teams else 1
        t2 = st.selectbox("🟠 Team 2", teams, key="t2_sel", index=t2_default_idx)

    col_stage, col_btn = st.columns([3, 2])
    with col_stage:
        stage = st.radio("Stage", ["Group Stage", "Knockout"], horizontal=True, label_visibility="collapsed")
    with col_btn:
        predict_clicked = st.button("⚽  PREDICT MATCH", use_container_width=True)

    if predict_clicked:
        if t1 == t2:
            st.error("Select two different teams.")
            return
        st.session_state["t1_prev"] = t1
        st.session_state["t2_prev"] = t2
        is_group = (stage == "Group Stage")
        with st.spinner("Running ensemble prediction…"):
            result = predict_match(model, engine, t1, t2, is_group)
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#475569;border:1px dashed #1E2D45;border-radius:12px;margin-top:1rem;">
  <div style="font-size:2.5rem;margin-bottom:.5rem;">⚽</div>
  <div style="font-family:'Bebas Neue',sans-serif;font-size:1.3rem;letter-spacing:.1em;">
    Select two teams and click Predict
  </div>
  <div style="font-size:.85rem;margin-top:.4rem;">
    Powered by Poisson regression · Gradient Boosting · Random Forest · Dixon-Coles
  </div>
</div>""", unsafe_allow_html=True)
        return

    r = result
    vs_banner(r["team1"], r["team2"], r["lam1"], r["lam2"])

    col_left, col_right = st.columns([3, 2])
    with col_left:
        section("Outcome Probabilities", "📊")
        outcome_cards(r["win"], r["draw"], r["loss"], r["team1"], r["team2"])
        verdict_box(r["win"], r["draw"], r["loss"], r["team1"], r["team2"])
    with col_right:
        st.plotly_chart(
            prob_donut(r["win"], r["draw"], r["loss"], r["team1"], r["team2"]),
            width="stretch", config={"displayModeBar": False},
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_scores, col_heat = st.columns([2, 3])
    with col_scores:
        section("Top Predicted Scorelines", "🎯")
        top_scores(r["top5"], r["team1"], r["team2"])
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        section("Team Comparison", "📈")
        stats_comparison(r["stats1"], r["stats2"], r["team1"], r["team2"])
    with col_heat:
        section("Score Probability Matrix", "🔥")
        st.plotly_chart(
            score_heatmap(r["matrix"], r["team1"], r["team2"]),
            width="stretch", config={"displayModeBar": False},
        )
        section("Attribute Radar", "🎭")
        st.plotly_chart(
            radar_chart(r["stats1"], r["stats2"], r["team1"], r["team2"]),
            width="stretch", config={"displayModeBar": False},
        )


def page_backtest(state: dict):
    st.markdown("### Holdout Backtest")
    st.markdown(
        f"<p style='color:#94A3B8;font-size:.85rem;'>Results on the final "
        f"<b style='color:#22D3EE'>{state['n_test']}</b> matches held out from training.</p>",
        unsafe_allow_html=True,
    )

    metrics = state["metrics"]
    m = metrics

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outcome Accuracy", f"{m['outcome_acc_pct']:.1f}%")
    c2.metric("Exact Score Acc.", f"{m['exact_score_pct']:.1f}%")
    c3.metric("Score NLL", f"{m['score_nll']:.3f}")
    c4.metric("Dixon-Coles ρ", f"{m['rho']:+.4f}")

    section("Cumulative Accuracy", "📈")
    st.plotly_chart(
        backtest_chart(state["test_df"], state["feat_test"], state["eval_model"]),
        width="stretch", config={"displayModeBar": False},
    )

    section("Match Log", "📋")

    eval_model = state["eval_model"]
    test_df = state["test_df"]
    feat_test = state["feat_test"]

    y1 = test_df["Goals1"].values
    y2 = test_df["Goals2"].values
    lam1_all, lam2_all = eval_model.predict_xg(feat_test)

    rows = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        mat = eval_model.score_matrix(float(lam1_all[i]), float(lam2_all[i]))
        pw, pd_, pl = eval_model.outcome_probs(mat)
        flat = np.argsort(mat.ravel())[::-1]
        pg1, pg2 = divmod(int(flat[0]), mat.shape[1])
        prob = float(mat[pg1, pg2]) * 100

        act_out = "W" if y1[i] > y2[i] else ("D" if y1[i] == y2[i] else "L")
        pred_out = ["W", "D", "L"][np.argmax([pw, pd_, pl])]

        rows.append({
            "Match": f"{row['Team 1']} vs {row['Team 2']}",
            "Actual": f"{int(y1[i])}–{int(y2[i])}",
            "Predicted": f"{pg1}–{pg2}",
            "Prob": f"{prob:.1f}%",
            "xG": f"{lam1_all[i]:.2f}/{lam2_all[i]:.2f}",
            "✓": "✓" if act_out == pred_out else "✗",
        })

    results_df = pd.DataFrame(rows)

    def color_result(val):
        if val == "✓":
            return "background-color:rgba(52,211,153,.15);color:#34D399;font-weight:700"
        elif val == "✗":
            return "background-color:rgba(248,113,113,.1);color:#F87171;font-weight:700"
        return ""

    # .map() is the current API; .applymap() was renamed in pandas 2.1
    try:
        styled = results_df.style.map(color_result, subset=["✓"])
    except AttributeError:
        styled = results_df.style.applymap(color_result, subset=["✓"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=420)


def page_teams(state: dict):
    st.markdown("### Team Stats Browser")
    lookup = state["final_engine"].build_lookup()
    teams = sorted(lookup.keys())

    search = st.text_input("🔍 Search teams", placeholder="Start typing a team name…")
    filtered = [t for t in teams if search.lower() in t.lower()] if search else teams

    st.markdown(f"<p style='color:#94A3B8;font-size:.82rem;'>{len(filtered)} teams</p>",
                unsafe_allow_html=True)

    if not filtered:
        st.info("No teams match your search.")
        return

    sort_col, _ = st.columns([2, 5])
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            ["Elo Rating", "Attack Strength", "Form", "Clean Sheet Rate"],
            label_visibility="collapsed",
        )
    sort_map = {
        "Elo Rating": "elo", "Attack Strength": "attack_strength",
        "Form": "form_last5", "Clean Sheet Rate": "clean_sheet_rate",
    }
    sort_key = sort_map[sort_by]
    filtered = sorted(filtered, key=lambda t: lookup[t].get(sort_key, 0), reverse=True)

    cols_per_row = 3
    for i in range(0, min(len(filtered), 30), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(filtered):
                break
            team = filtered[i + j]
            s = lookup[team]
            trend_symbol = "▲" if s["elo_trend"] > 10 else ("▼" if s["elo_trend"] < -10 else "●")
            trend_color = "#34D399" if s["elo_trend"] > 10 else ("#F87171" if s["elo_trend"] < -10 else "#FBBF24")
            form_bar = "█" * int(s["form_last5"] / 20) + "░" * (5 - int(s["form_last5"] / 20))
            with col:
                st.markdown(f"""
<div class="oracle-card" style="cursor:pointer;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem;">
    <div style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem;letter-spacing:.05em;color:#F1F5F9;line-height:1.2;">{team}</div>
    <div style="font-size:.75rem;color:{trend_color};">{trend_symbol} {s['elo_trend']:+.0f}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:.3rem;font-size:.78rem;">
    <div><span style="color:#94A3B8;">ELO</span> <span style="font-family:'JetBrains Mono',monospace;color:#22D3EE;">{s['elo']:.0f}</span></div>
    <div><span style="color:#94A3B8;">ATK</span> <span style="font-family:'JetBrains Mono',monospace;color:#60A5FA;">{s['attack_strength']:.2f}</span></div>
    <div><span style="color:#94A3B8;">DEF</span> <span style="font-family:'JetBrains Mono',monospace;color:#A78BFA;">{s['defense_strength']:.2f}</span></div>
    <div><span style="color:#94A3B8;">CS%</span> <span style="font-family:'JetBrains Mono',monospace;color:#34D399;">{s['clean_sheet_rate']:.0%}</span></div>
  </div>
  <div style="margin-top:.5rem;font-size:.72rem;color:#94A3B8;">
    Form <span style="font-family:'JetBrains Mono',monospace;color:#FBBF24;font-size:.8rem;">{form_bar}</span>
  </div>
  <div style="font-size:.68rem;color:#475569;margin-top:.25rem;">{s['match_count']} matches in dataset</div>
</div>""", unsafe_allow_html=True)


def page_model(state: dict):
    st.markdown("### Model Performance")
    m = state["metrics"]

    col_a, col_b = st.columns(2)
    with col_a:
        section("Prediction Quality")
        mc1, mc2 = st.columns(2)
        mc1.metric("Outcome Accuracy", f"{m['outcome_acc_pct']:.1f}%",
                   help="% of matches where Win/Draw/Loss was correctly predicted")
        mc2.metric("Exact Score Accuracy", f"{m['exact_score_pct']:.1f}%")
        mc3, mc4 = st.columns(2)
        mc3.metric("Score NLL (lower=better)", f"{m['score_nll']:.3f}")
        mc4.metric("Dixon-Coles ρ", f"{m['rho']:+.4f}", help="Low-score correlation correction")
        mc5, mc6 = st.columns(2)
        mc5.metric("RMSE Team 1 xG", f"{m['rmse_t1']:.3f}")
        mc6.metric("RMSE Team 2 xG", f"{m['rmse_t2']:.3f}")
        mc7, mc8 = st.columns(2)
        mc7.metric("MAE Team 1 xG", f"{m['mae_t1']:.3f}")
        mc8.metric("MAE Team 2 xG", f"{m['mae_t2']:.3f}")

    with col_b:
        section("Ensemble Weights")
        w1, w2 = m["w1"], m["w2"]
        model_names = ["Poisson GLM", "Gradient Boost", "Random Forest"]
        colors = ["#22D3EE", "#A78BFA", "#FB923C"]

        for label, weights in [("Team 1 Goals", w1), ("Team 2 Goals", w2)]:
            fig = go.Figure(go.Bar(
                x=model_names, y=[w * 100 for w in weights],
                marker=dict(color=colors, line=dict(color="#080D1A", width=1.5)),
                text=[f"{w*100:.0f}%" for w in weights],
                textposition="outside",
                textfont=dict(size=11, family="JetBrains Mono"),
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            ))
            # FIX: margin specified here directly (not via PLOTLY_LAYOUT) to
            # avoid "multiple values for keyword argument 'margin'" TypeError.
            fig.update_layout(
                **PLOTLY_LAYOUT,
                margin=dict(l=10, r=10, t=35, b=10),
                title=dict(text=label, font=dict(family="Bebas Neue", size=14, color="#94A3B8")),
                yaxis=dict(
                    title=dict(text="Weight %", font=dict(size=11)),
                    gridcolor="#1E2D45", range=[0, 60],
                ),
                xaxis=dict(linecolor="#1E2D45"),
                height=200,
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("Dataset Summary")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Matches", f"{state['n_matches']:,}")
    d2.metric("Training Matches", f"{state['n_train']:,}")
    d3.metric("Holdout Matches", f"{state['n_test']:,}")

    st.markdown("""
<div class="oracle-card" style="margin-top:.75rem;">
<p style="color:#94A3B8;font-size:.83rem;line-height:1.6;margin:0;">
<b style="color:#22D3EE;">Architecture:</b> Three-model ensemble (Poisson GLM + Gradient Boosting + Random Forest) with time-series CV weight calibration.
Features include opponent-adjusted rolling attack/defense strengths, Elo-based win probability, time-decayed head-to-head records,
form metrics, clean sheet rates, and stage modifiers. Low-score predictions are corrected with a Dixon-Coles τ adjustment
estimated via maximum likelihood.
</p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar() -> Optional[pd.DataFrame]:
    with st.sidebar:
        st.markdown("""
<div class="sidebar-logo">
  <div class="sidebar-logo-text">⚽ SOCCER ORACLE</div>
  <div class="sidebar-logo-sub">Advanced Match Prediction</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("<div class='sidebar-label'>Upload Match Data (CSV)</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

        st.markdown("""
<div style='margin-top:.5rem;padding:.6rem;background:#192236;border-radius:8px;border:1px solid #1E2D45;'>
<p style='font-size:.75rem;color:#64748B;margin:0;line-height:1.5;'>
Required columns:<br>
<code style='color:#22D3EE;font-size:.72rem;'>Team 1, Team 2, Goals1, Goals2, Elo1, Elo2</code><br>
Optional: <code style='color:#94A3B8;font-size:.72rem;'>Date, Stage</code>
</p>
</div>""", unsafe_allow_html=True)

        if not uploaded:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
<div style='text-align:center;padding:1.5rem .5rem;color:#475569;'>
  <div style='font-size:2rem;'>📂</div>
  <div style='font-size:.82rem;margin-top:.4rem;'>Upload a CSV to get started</div>
</div>""", unsafe_allow_html=True)
            return None

        raw_bytes = uploaded.read()
        fh = file_hash(raw_bytes)
        try:
            df_raw = pd.read_csv(pd.io.common.BytesIO(raw_bytes))
            df = load_and_clean(df_raw)
        except Exception as e:
            st.error(f"CSV error: {e}")
            return None

        teams_count = df[["Team 1", "Team 2"]].stack().nunique()
        st.markdown(f"""
<div style='margin-top:.75rem;display:flex;gap:.5rem;flex-wrap:wrap;'>
  <div style='flex:1;min-width:70px;background:#192236;border-radius:8px;
              border:1px solid #1E2D45;padding:.5rem;text-align:center;'>
    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;color:#22D3EE;'>{len(df):,}</div>
    <div style='font-size:.65rem;color:#64748B;text-transform:uppercase;'>Matches</div>
  </div>
  <div style='flex:1;min-width:70px;background:#192236;border-radius:8px;
              border:1px solid #1E2D45;padding:.5rem;text-align:center;'>
    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;color:#A78BFA;'>{teams_count}</div>
    <div style='font-size:.65rem;color:#64748B;text-transform:uppercase;'>Teams</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.session_state["file_hash"] = fh
        st.session_state["df"] = df
        return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    inject_css()

    df = render_sidebar()

    if df is None:
        st.markdown("""
<div style='text-align:center;padding:4rem 2rem 2rem;'>
  <div style='font-family:"Bebas Neue",sans-serif;font-size:4rem;letter-spacing:.12em;
              background:linear-gradient(135deg,#22D3EE,#A78BFA);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              line-height:1;'>
    SOCCER ORACLE
  </div>
  <div style='color:#475569;font-size:1rem;margin-top:.75rem;letter-spacing:.05em;'>
    ENSEMBLE MATCH PREDICTION  ·  POISSON · GRADIENT BOOST · RANDOM FOREST · DIXON-COLES
  </div>
</div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        for col, icon, title, desc in [
            (col1, "🧠", "3-Model Ensemble",
             "Poisson GLM, Gradient Boosting, and Random Forest combined with time-series CV calibration."),
            (col2, "📐", "Dixon-Coles Correction",
             "Low-score probability adjusted via MLE-estimated τ correction for 0-0, 1-0, 0-1, 1-1 outcomes."),
            (col3, "🏹", "Opponent-Adjusted Stats",
             "Rolling attack/defense strengths normalized against opponent quality, not raw averages."),
        ]:
            col.markdown(f"""
<div class="oracle-card" style="text-align:center;padding:1.5rem;">
  <div style='font-size:2rem;margin-bottom:.5rem;'>{icon}</div>
  <div style='font-family:"Bebas Neue",sans-serif;font-size:1.1rem;letter-spacing:.06em;margin-bottom:.4rem;'>{title}</div>
  <div style='font-size:.82rem;color:#64748B;line-height:1.5;'>{desc}</div>
</div>""", unsafe_allow_html=True)
        return

    fh = st.session_state.get("file_hash", "")

    with st.spinner("🏋️ Training ensemble model — this takes ~30 seconds on first load…"):
        state = cached_train(fh, df)

    tab_predict, tab_backtest, tab_teams, tab_model = st.tabs([
        "⚽  Predict Match",
        "📊  Backtest",
        "🏆  Teams",
        "🔬  Model Stats",
    ])

    with tab_predict:
        page_predict(state)
    with tab_backtest:
        page_backtest(state)
    with tab_teams:
        page_teams(state)
    with tab_model:
        page_model(state)


if __name__ == "__main__":
    main()