"""
⚽ final score prediction platform — streamlit app
==================================================
run with:  streamlit run app.py

put finalscores.csv in the same folder, or upload it via the sidebar.
"""

# ─────────────────────────────────────────────────────────────────────────────
# imports
# ─────────────────────────────────────────────────────────────────────────────
import io
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ score predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# global constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_GOALS = 9
ENSEMBLE_CV = 5
DEFAULT_CSV = "FINALSCORES.csv"

# ─────────────────────────────────────────────────────────────────────────────
# custom css
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --bg: #08111c;
    --panel: rgba(15, 24, 38, 0.88);
    --panel-2: rgba(17, 29, 46, 0.96);
    --line: rgba(255,255,255,0.08);
    --text: #edf4ff;
    --muted: #a8bbd0;
    --accent: #61dafb;
    --accent-2: #ff8f5e;
    --success: #3ddc97;
    --warning: #ffd166;
    --danger: #ff6b6b;
}

html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text) !important;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(97, 218, 251, 0.12), transparent 30%),
        radial-gradient(circle at bottom right, rgba(255, 143, 94, 0.10), transparent 28%),
        linear-gradient(180deg, #07101a 0%, #08111c 100%);
    color: var(--text) !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08111c 0%, #0b1624 100%);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, rgba(17,29,46,0.95), rgba(10,19,31,0.92));
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 24px 28px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    line-height: 1.1;
    color: #ffffff;
}

.hero p {
    margin: 0.45rem 0 0 0;
    color: var(--muted);
    font-size: 0.98rem;
}

.subtle {
    color: var(--muted) !important;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    color: var(--text) !important;
}

.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #ffffff !important;
    margin: 0 0 0.6rem 0;
}

.small-label {
    color: var(--muted) !important;
    font-size: 0.80rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.score-card {
    border-radius: 18px;
    padding: 1rem 0.9rem;
    text-align: center;
    border: 1px solid var(--line);
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    background: linear-gradient(180deg, rgba(23,35,52,0.96), rgba(13,23,36,0.96));
    min-height: 175px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    color: #ffffff !important;
}

.score-card.first {
    border-color: rgba(97, 218, 251, 0.30);
}
.score-card.second {
    border-color: rgba(255, 143, 94, 0.30);
}
.score-card.third {
    border-color: rgba(61, 220, 151, 0.28);
}

.medal {
    font-size: 1.25rem;
    margin-bottom: 0.25rem;
    color: #ffffff;
}

.scoreline {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #ffffff !important;
    margin: 0.15rem 0 0.2rem 0;
}

.probability {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--accent) !important;
    margin: 0.1rem 0;
}

.outcome {
    margin-top: 0.3rem;
    color: var(--muted) !important;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.verdict {
    border-radius: 18px;
    padding: 16px 18px;
    font-weight: 700;
    font-size: 1.02rem;
    border: 1px solid var(--line);
    box-shadow: 0 10px 26px rgba(0,0,0,0.18);
    color: #ffffff !important;
}

.verdict.win {
    background: linear-gradient(135deg, rgba(35, 101, 61, 0.95), rgba(18, 64, 40, 0.95));
    color: #dffcef !important;
    border-color: rgba(61, 220, 151, 0.30);
}

.verdict.draw {
    background: linear-gradient(135deg, rgba(115, 90, 24, 0.95), rgba(63, 49, 10, 0.95));
    color: #fff5d6 !important;
    border-color: rgba(255, 209, 102, 0.30);
}

.verdict.loss {
    background: linear-gradient(135deg, rgba(110, 35, 35, 0.95), rgba(68, 18, 18, 0.95));
    color: #ffe3e3 !important;
    border-color: rgba(255, 107, 107, 0.30);
}

.xg-strip {
    background: linear-gradient(180deg, rgba(12,20,33,0.95), rgba(15,24,38,0.92));
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 0.8rem;
    align-items: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.14);
    color: #ffffff !important;
}

.xg-item {
    text-align: center;
}

.xg-name {
    color: var(--muted) !important;
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.xg-val {
    color: var(--accent) !important;
    font-weight: 800;
    font-size: 2rem;
    line-height: 1.05;
    margin: 0.15rem 0;
}

.xg-divider {
    color: rgba(255,255,255,0.35) !important;
    font-size: 1.2rem;
    font-weight: 700;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(18,28,44,0.95), rgba(14,23,36,0.95));
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 10px 14px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.14);
    color: #ffffff !important;
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stMetricDelta"] {
    color: #cfe9ff !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] label {
    color: var(--text);
}

[data-baseweb="tab"] {
    color: var(--muted) !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
}

button, input, textarea, select {
    color: #ffffff !important;
}

code, pre {
    background: rgba(8,17,28,0.85) !important;
    color: #f5f9ff !important;
    border: 1px solid var(--line);
}

hr {
    border-color: rgba(255,255,255,0.10);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--line);
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# helper functions
# ─────────────────────────────────────────────────────────────────────────────
def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_xg(x: float) -> str:
    return f"{x:.2f}"


def fmt_score(g1: int, g2: int) -> str:
    return f"{g1}–{g2}"


def dark_df_style(df: pd.DataFrame, highlight=None):
    styler = (
        df.style
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#111d2e"), ("color", "#ffffff"), ("border", "1px solid rgba(255,255,255,0.08)")]},
            {"selector": "tbody td", "props": [("background-color", "#0f1928"), ("color", "#edf4ff"), ("border", "1px solid rgba(255,255,255,0.06)")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
        .set_properties(**{
            "background-color": "#0f1928",
            "color": "#edf4ff",
            "border-color": "rgba(255,255,255,0.06)",
        })
    )
    if highlight is not None:
        styler = styler.apply(highlight, axis=1)
    return styler


# ─────────────────────────────────────────────────────────────────────────────
# feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)

    f["elo_diff"] = df["Elo1"] - df["Elo2"]
    f["elo_ratio"] = df["Elo1"] / df["Elo2"]
    f["elo_win_prob"] = 1.0 / (1.0 + 10.0 ** ((df["Elo2"] - df["Elo1"]) / 400.0))
    f["elo_sum"] = df["Elo1"] + df["Elo2"]

    f["attack_t1"] = df["Attack_Rating_T1"]
    f["attack_t2"] = df["Attack_Rating_T2"]
    f["defense_t1"] = df["Defense_Rating_T1"]
    f["defense_t2"] = df["Defense_Rating_T2"]

    f["xg_t1"] = df["Attack_Rating_T1"] / df["Defense_Rating_T2"].clip(lower=0.1)
    f["xg_t2"] = df["Attack_Rating_T2"] / df["Defense_Rating_T1"].clip(lower=0.1)
    f["xg_diff"] = f["xg_t1"] - f["xg_t2"]

    f["form_last5_t1"] = df["Form_Last5_T1"]
    f["form_last5_t2"] = df["Form_Last5_T2"]
    f["form_weighted_t1"] = df["Form_Weighted_T1"]
    f["form_weighted_t2"] = df["Form_Weighted_T2"]
    f["form_diff_last5"] = df["Form_Last5_T1"] - df["Form_Last5_T2"]
    f["form_diff_weighted"] = df["Form_Weighted_T1"] - df["Form_Weighted_T2"]

    f["strength_t1"] = (
        f["elo_win_prob"] * 0.40
        + f["xg_t1"].clip(0, 6) * 0.15
        + f["form_weighted_t1"] / 100.0 * 0.45
    )
    f["strength_t2"] = (
        (1 - f["elo_win_prob"]) * 0.40
        + f["xg_t2"].clip(0, 6) * 0.15
        + f["form_weighted_t2"] / 100.0 * 0.45
    )
    f["strength_ratio"] = f["strength_t1"] / f["strength_t2"].clip(lower=0.01)
    f["is_group"] = df["Stage"].astype(str).str.contains("Group", case=False, na=False).astype(int)

    return f


def make_feature_row(s1: dict, s2: dict, is_group: bool) -> pd.DataFrame:
    e1, e2 = s1["elo"], s2["elo"]
    wp = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))

    xg1 = s1["attack"] / max(s2["defense"], 0.1)
    xg2 = s2["attack"] / max(s1["defense"], 0.1)

    fw1, fw2 = s1["form_weighted"], s2["form_weighted"]
    fl1, fl2 = s1["form_last5"], s2["form_last5"]

    st1 = wp * 0.40 + min(xg1, 6) * 0.15 + fw1 / 100.0 * 0.45
    st2 = (1 - wp) * 0.40 + min(xg2, 6) * 0.15 + fw2 / 100.0 * 0.45

    return pd.DataFrame(
        [{
            "elo_diff": e1 - e2,
            "elo_ratio": e1 / e2,
            "elo_win_prob": wp,
            "elo_sum": e1 + e2,
            "attack_t1": s1["attack"],
            "attack_t2": s2["attack"],
            "defense_t1": s1["defense"],
            "defense_t2": s2["defense"],
            "xg_t1": xg1,
            "xg_t2": xg2,
            "xg_diff": xg1 - xg2,
            "form_last5_t1": fl1,
            "form_last5_t2": fl2,
            "form_weighted_t1": fw1,
            "form_weighted_t2": fw2,
            "form_diff_last5": fl1 - fl2,
            "form_diff_weighted": fw1 - fw2,
            "strength_t1": st1,
            "strength_t2": st2,
            "strength_ratio": st1 / max(st2, 0.01),
            "is_group": int(is_group),
        }]
    )


def build_team_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        for tc, ec, ac, dc, fl, fw in [
            ("Team 1", "Elo1", "Attack_Rating_T1", "Defense_Rating_T1", "Form_Last5_T1", "Form_Weighted_T1"),
            ("Team 2", "Elo2", "Attack_Rating_T2", "Defense_Rating_T2", "Form_Last5_T2", "Form_Weighted_T2"),
        ]:
            lookup[row[tc]] = {
                "elo": float(row[ec]),
                "attack": float(row[ac]),
                "defense": float(row[dc]),
                "form_last5": float(row[fl]),
                "form_weighted": float(row[fw]),
            }
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# dixon-coles correction
# ─────────────────────────────────────────────────────────────────────────────
def dc_tau(g1, g2, lam1, lam2, rho):
    if g1 == 0 and g2 == 0:
        return max(1e-6, 1 - lam1 * lam2 * rho)
    elif g1 == 1 and g2 == 0:
        return max(1e-6, 1 + lam2 * rho)
    elif g1 == 0 and g2 == 1:
        return max(1e-6, 1 + lam1 * rho)
    elif g1 == 1 and g2 == 1:
        return max(1e-6, 1 - rho)
    return 1.0


def estimate_rho(g1_arr, g2_arr, lam1_arr, lam2_arr):
    def neg_ll(rho):
        if abs(rho) >= 0.99:
            return 1e12

        ll = 0.0
        for i in range(len(g1_arr)):
            if g1_arr[i] <= 1 and g2_arr[i] <= 1:
                tau = dc_tau(g1_arr[i], g2_arr[i], lam1_arr[i], lam2_arr[i], rho)
                ll += np.log(max(tau, 1e-9))
        return -ll

    res = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
    return float(res.x)


# ─────────────────────────────────────────────────────────────────────────────
# ensemble model
# ─────────────────────────────────────────────────────────────────────────────
class ScorePredictionEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()

        gb_kw = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.75,
            min_samples_leaf=4,
            random_state=42,
        )
        rf_kw = dict(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=4,
            random_state=42,
        )

        self.p1 = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb1 = GradientBoostingRegressor(**gb_kw)
        self.rf1 = RandomForestRegressor(**rf_kw)

        self.p2 = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb2 = GradientBoostingRegressor(**gb_kw)
        self.rf2 = RandomForestRegressor(**rf_kw)

        self.w1 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        self.w2 = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
        self.rho = 0.0
        self.feature_cols = []
        self.metrics = {}

    def fit(self, X: pd.DataFrame, y1, y2):
        self.feature_cols = X.columns.tolist()
        Xs = self.scaler.fit_transform(X)
        kf = KFold(n_splits=ENSEMBLE_CV, shuffle=True, random_state=0)

        for est, y in [
            (self.p1, y1), (self.gb1, y1), (self.rf1, y1),
            (self.p2, y2), (self.gb2, y2), (self.rf2, y2),
        ]:
            est.fit(Xs, y)

        def cv_mse(estimator, y):
            return -cross_val_score(
                estimator,
                Xs,
                y,
                cv=kf,
                scoring="neg_mean_squared_error",
            ).mean()

        for w_attr, ests, y in [
            ("w1", [self.p1, self.gb1, self.rf1], y1),
            ("w2", [self.p2, self.gb2, self.rf2], y2),
        ]:
            mses = np.array([cv_mse(e, y) for e in ests], dtype=float)
            inv = 1.0 / (mses + 1e-9)
            setattr(self, w_attr, inv / inv.sum())

        lam1, lam2 = self._raw(Xs)
        self.rho = estimate_rho(y1.astype(int), y2.astype(int), lam1, lam2)

        p1r = np.round(lam1).clip(0).astype(int)
        p2r = np.round(lam2).clip(0).astype(int)

        self.metrics = {
            "rmse_t1": float(np.sqrt(mean_squared_error(y1, lam1))),
            "rmse_t2": float(np.sqrt(mean_squared_error(y2, lam2))),
            "mae_t1": float(mean_absolute_error(y1, lam1)),
            "mae_t2": float(mean_absolute_error(y2, lam2)),
            "exact_score": float(np.mean((p1r == y1.astype(int)) & (p2r == y2.astype(int)))),
            "outcome_acc": float(np.mean(np.sign(lam1 - lam2) == np.sign(y1 - y2))),
            "rho": self.rho,
            "w1": self.w1.tolist(),
            "w2": self.w2.tolist(),
        }
        return self

    def _raw(self, Xs):
        def ens(ests, weights):
            return sum(w * np.clip(e.predict(Xs), 0, 10) for e, w in zip(ests, weights))

        return (
            np.clip(ens([self.p1, self.gb1, self.rf1], self.w1), 0.05, 7.0),
            np.clip(ens([self.p2, self.gb2, self.rf2], self.w2), 0.05, 7.0),
        )

    def predict_xg(self, X: pd.DataFrame):
        Xs = self.scaler.transform(X[self.feature_cols])
        l1, l2 = self._raw(Xs)
        return float(l1[0]), float(l2[0])

    def score_matrix(self, lam1, lam2):
        mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for g1 in range(MAX_GOALS + 1):
            for g2 in range(MAX_GOALS + 1):
                mat[g1, g2] = (
                    poisson.pmf(g1, lam1)
                    * poisson.pmf(g2, lam2)
                    * dc_tau(g1, g2, lam1, lam2, self.rho)
                )
        return mat / mat.sum()

    def outcomes(self, mat):
        n = mat.shape[0]
        pw = pd_ = pl = 0.0
        for g1 in range(n):
            for g2 in range(n):
                if g1 > g2:
                    pw += mat[g1, g2]
                elif g1 == g2:
                    pd_ += mat[g1, g2]
                else:
                    pl += mat[g1, g2]
        return pw, pd_, pl


def run_prediction(model, s1, s2, name1, name2, is_group):
    X = make_feature_row(s1, s2, is_group)
    lam1, lam2 = model.predict_xg(X)
    mat = model.score_matrix(lam1, lam2)
    pw, pd_, pl = model.outcomes(mat)

    flat = np.argsort(mat.ravel())[::-1]
    top3 = []
    for idx in flat:
        g1, g2 = divmod(int(idx), mat.shape[1])
        top3.append((g1, g2, float(mat[g1, g2])))
        if len(top3) == 3:
            break

    return {
        "name1": name1,
        "name2": name2,
        "lam1": lam1,
        "lam2": lam2,
        "top3": top3,
        "win": pw,
        "draw": pd_,
        "loss": pl,
        "mat": mat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# cached data + model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


@st.cache_resource(show_spinner=False)
def train_model(csv_bytes: bytes):
    df = load_csv(csv_bytes)
    lookup = build_team_lookup(df)
    X = engineer_features(df)
    y1 = df["Goals1"].values.astype(float)
    y2 = df["Goals2"].values.astype(float)
    model = ScorePredictionEnsemble()
    model.fit(X, y1, y2)
    return model, lookup, df


# ─────────────────────────────────────────────────────────────────────────────
# plotly helpers
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e8eef7",
)

def outcome_chart(pw, pd_, pl, name1, name2):
    total = max(pw + pd_ + pl, 1e-9)
    labels = [f"{name1} win", "draw", f"{name2} win"]
    values = [pw / total * 100, pd_ / total * 100, pl / total * 100]
    colors = ["#3ddc97", "#ffd166", "#ff6b6b"]

    fig = go.Figure()
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(
            go.Bar(
                name=label,
                x=[val],
                y=[""],
                orientation="h",
                marker_color=color,
                text=f"{val:.1f}%",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=15, color="white"),
                hovertemplate=f"{label}: {val:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=105,
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def heatmap_chart(mat, name1, name2, show=7):
    sub = mat[:show, :show] * 100
    text = [[f"{sub[r, c]:.1f}%" for c in range(show)] for r in range(show)]

    fig = go.Figure(
        go.Heatmap(
            z=sub,
            x=[str(i) for i in range(show)],
            y=[str(i) for i in range(show)],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorscale=[
                [0.0, "rgba(9,24,42,0.90)"],
                [0.3, "rgba(0,88,128,0.92)"],
                [0.6, "rgba(0,156,184,0.92)"],
                [0.85, "rgba(255,160,0,0.92)"],
                [1.0, "rgba(255,220,0,1.0)"],
            ],
            showscale=False,
            hovertemplate=f"{name1} %{{y}} – {name2} %{{x}}: %{{text}}<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=390,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(
            title=f"{name2} goals",
            tickfont=dict(size=13, color="#d7e4f3"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title=f"{name1} goals",
            tickfont=dict(size=13, color="#d7e4f3"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        title=dict(
            text="score probability heatmap (%)",
            x=0.5,
            font=dict(size=15, color="#9fb4cc"),
        ),
    )
    return fig


def radar_chart(s1, s2, name1, name2):
    cats = ["elo", "attack", "defense", "form weighted", "form last 5"]

    def norm(val, lo, hi):
        return (val - lo) / (hi - lo) * 100

    elo_lo, elo_hi = 1200, 2200
    vals1 = [
        norm(s1["elo"], elo_lo, elo_hi),
        s1["attack"] / 5 * 100,
        s1["defense"] / 2 * 100,
        s1["form_weighted"] / 50 * 100,
        s1["form_last5"] / 15 * 100,
    ]
    vals2 = [
        norm(s2["elo"], elo_lo, elo_hi),
        s2["attack"] / 5 * 100,
        s2["defense"] / 2 * 100,
        s2["form_weighted"] / 50 * 100,
        s2["form_last5"] / 15 * 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals1 + [vals1[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=name1,
            line=dict(color="#61dafb", width=2.4),
            fillcolor="rgba(97,218,251,0.18)",
            opacity=0.95,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=vals2 + [vals2[0]],
            theta=cats + [cats[0]],
            fill="toself",
            name=name2,
            line=dict(color="#ff8f5e", width=2.4),
            fillcolor="rgba(255,143,94,0.18)",
            opacity=0.95,
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        polar=dict(
            bgcolor="rgba(10,20,35,0.82)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255,255,255,0.10)",
                tickfont=dict(color="#d7e4f3", size=9),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.10)",
                tickfont=dict(color="#d7e4f3", size=11),
            ),
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.12,
            font=dict(size=12, color="#d7e4f3"),
        ),
        title=dict(
            text="team attribute comparison",
            x=0.5,
            font=dict(size=14, color="#9fb4cc"),
        ),
    )
    return fig


def goals_dist_chart(lam1, lam2, name1, name2):
    goals = list(range(MAX_GOALS + 1))
    p1 = [poisson.pmf(g, lam1) * 100 for g in goals]
    p2 = [poisson.pmf(g, lam2) * 100 for g in goals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=goals,
            y=p1,
            name=name1,
            marker_color="#61dafb",
            opacity=0.88,
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=goals,
            y=p2,
            name=name2,
            marker_color="#ff8f5e",
            opacity=0.88,
            hovertemplate="%{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=285,
        margin=dict(l=10, r=10, t=50, b=10),
        barmode="group",
        xaxis=dict(
            title="goals scored",
            tickvals=goals,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#d7e4f3"),
        ),
        yaxis=dict(
            title="probability (%)",
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#d7e4f3"),
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.12,
            font=dict(size=12, color="#d7e4f3"),
        ),
        title=dict(
            text="goal scoring distribution",
            x=0.5,
            font=dict(size=14, color="#9fb4cc"),
        ),
    )
    return fig


def confidence_gauge(confidence, label="Prediction Confidence"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            title={"text": label, "font": {"size": 16, "color": "#ffffff"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#61dafb"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,107,107,0.18)"},
                    {"range": [40, 65], "color": "rgba(255,209,102,0.16)"},
                    {"range": [65, 100], "color": "rgba(61,220,151,0.16)"},
                ],
            },
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=260,
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(
            text="confidence gauge",
            x=0.5,
            font=dict(size=14, color="#9fb4cc"),
        ),
    )
    return fig


def outcome_donut(pw, pd_, pl, name1, name2):
    labels = [f"{name1} win", "draw", f"{name2} win"]
    values = [pw, pd_, pl]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            sort=False,
            textinfo="none",
            marker=dict(colors=["#3ddc97", "#ffd166", "#ff6b6b"]),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        margin=dict(l=10, r=10, t=55, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.08,
            font=dict(size=12, color="#d7e4f3"),
        ),
        annotations=[
            dict(
                text="win<br>split",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#ffffff"),
                showarrow=False,
            )
        ],
        title=dict(
            text="outcome split",
            x=0.5,
            font=dict(size=14, color="#9fb4cc"),
        ),
    )
    return fig


def score_table(topn, name1, name2):
    rows = []
    for i, (g1, g2, prob) in enumerate(topn, start=1):
        rows.append(
            {
                "#": i,
                "Score": fmt_score(g1, g2),
                "Prob": f"{prob * 100:.2f}%",
                "Winner": name1 if g1 > g2 else (name2 if g2 > g1 else "draw"),
            }
        )
    return pd.DataFrame(rows)


def best_outcome(pw, pd_, pl, team1, team2):
    outcomes = [
        ("team1", pw, team1),
        ("draw", pd_, "Draw"),
        ("team2", pl, team2),
    ]
    outcomes_sorted = sorted(outcomes, key=lambda x: x[1], reverse=True)
    top_key, top_prob, top_label = outcomes_sorted[0]
    second_prob = outcomes_sorted[1][1]
    edge = max(top_prob - second_prob, 0.0)
    return top_key, top_label, top_prob, edge


# ─────────────────────────────────────────────────────────────────────────────
# backtest helper
# ─────────────────────────────────────────────────────────────────────────────
def get_backtest_df(df, model, n=20):
    rows = []
    sample = df.tail(n)

    for _, row in sample.iterrows():
        s1 = dict(
            elo=row["Elo1"],
            attack=row["Attack_Rating_T1"],
            defense=row["Defense_Rating_T1"],
            form_last5=row["Form_Last5_T1"],
            form_weighted=row["Form_Weighted_T1"],
        )
        s2 = dict(
            elo=row["Elo2"],
            attack=row["Attack_Rating_T2"],
            defense=row["Defense_Rating_T2"],
            form_last5=row["Form_Last5_T2"],
            form_weighted=row["Form_Weighted_T2"],
        )

        r = run_prediction(
            model,
            s1,
            s2,
            row["Team 1"],
            row["Team 2"],
            "Group" in str(row["Stage"]),
        )

        g1a, g2a = int(row["Goals1"]), int(row["Goals2"])
        pg1, pg2, prob = r["top3"][0]
        actual = "W" if g1a > g2a else ("D" if g1a == g2a else "L")
        pw, pd_, pl = r["win"], r["draw"], r["loss"]
        pred = "W" if pw > pl and pw > pd_ else ("D" if pd_ >= pw and pd_ >= pl else "L")

        rows.append(
            {
                "Match": f"{row['Team 1']} vs {row['Team 2']}",
                "Stage": row["Stage"],
                "Actual Score": fmt_score(g1a, g2a),
                "Pred Score #1": fmt_score(pg1, pg2),
                "Prob": f"{prob*100:.1f}%",
                "Outcome ✓?": "✅" if actual == pred else "❌",
                "Exact ✓?": "✅" if pg1 == g1a and pg2 == g2a else "❌",
                "xG T1": fmt_xg(r["lam1"]),
                "xG T2": fmt_xg(r["lam2"]),
            }
        )

    bt = pd.DataFrame(rows)
    oc = (bt["Outcome ✓?"] == "✅").sum()
    ex = (bt["Exact ✓?"] == "✅").sum()
    return bt, oc, ex


# ─────────────────────────────────────────────────────────────────────────────
# ui helpers
# ─────────────────────────────────────────────────────────────────────────────
def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <div class="hero">
            <h1>⚽ final score prediction platform</h1>
            <p>ensemble ml · poisson glm · dixon-coles correction · scoreline ranking · backtesting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            "<h3 style='color:blue; margin-bottom:0.4rem;'>📂 Data Source</h3>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "upload a csv",
            type=["csv"],
            help="expected columns: Team 1, Team 2, Elo1, Elo2, Goals1, Goals2, Attack/Defense/Form ratings",
        )

        csv_bytes = None
        if uploaded is not None:
            csv_bytes = uploaded.read()
            st.caption("using uploaded file.")
        else:
            try:
                with open(DEFAULT_CSV, "rb") as f:
                    csv_bytes = f.read()
                st.caption(f"using `{DEFAULT_CSV}` from the working folder.")
            except FileNotFoundError:
                st.warning(f"no file uploaded and `{DEFAULT_CSV}` was not found.")
                st.stop()

        st.divider()

        with st.spinner("training ensemble model..."):
            model, lookup, df = train_model(csv_bytes)

        st.success(f"loaded {len(df)} matches · {len(lookup)} teams")

        st.divider()
        st.markdown(
            "<h3 style='color:white; margin-bottom:0.4rem;'>🏟️ Match Setup</h3>",
            unsafe_allow_html=True,
        )

        teams = sorted(lookup.keys())
        if not teams:
            st.error("no teams found in dataset.")
            st.stop()

        if len(teams) < 2:
            st.error("need at least two teams in the dataset.")
            st.stop()

        t1_default = teams.index("Brazil") if "Brazil" in teams else 0
        t2_default = teams.index("Argentina") if "Argentina" in teams else 1
        t1_default = min(t1_default, len(teams) - 1)
        t2_default = min(t2_default, len(teams) - 1)

        with st.form("prediction_form"):
            team1 = st.selectbox("team 1", teams, index=t1_default)
            team2_options = [t for t in teams if t != team1]
            if not team2_options:
                st.error("choose a different team 1; no other teams left to compare.")
                st.stop()

            team2_default = min(t2_default, len(team2_options) - 1)
            team2_default = max(team2_default, 0)
            team2 = st.selectbox("team 2", team2_options, index=team2_default)

            stage = st.radio("stage", ["Group Stage", "Knockout"], horizontal=True)
            predict_btn = st.form_submit_button("⚡ run prediction", use_container_width=False)

        is_group = stage == "Group Stage"

        st.divider()
        st.markdown("### 📊 model health")
        m = model.metrics
        st.metric("Outcome accuracy", f"{m['outcome_acc']*100:.1f}%")
        st.metric("Exact score accuracy", f"{m['exact_score']*100:.1f}%")
        st.metric("xG RMSE (T1 / T2)", f"{m['rmse_t1']:.3f} / {m['rmse_t2']:.3f}")
        st.metric("Dixon-Coles ρ", f"{m['rho']:+.4f}")

    tab_pred, tab_bt, tab_info = st.tabs(["🎯 prediction", "📋 backtest", "ℹ️ model info"])

    with tab_pred:
        if not predict_btn:
            st.markdown(
                """
                <div class="panel" style="text-align:center; padding:3rem 1rem;">
                    <div style="font-size:3.4rem; line-height:1;">⚽</div>
                    <div style="font-size:1.2rem; font-weight:700; color:#ffffff; margin-top:0.8rem;">
                        choose two teams in the sidebar and run a prediction
                    </div>
                    <div class="subtle" style="margin-top:0.45rem;">
                        you will see the most likely scorelines, win/draw/loss chances, xg, and comparison charts.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            s1 = lookup[team1]
            s2 = lookup[team2]

            with st.spinner("running ensemble prediction..."):
                result = run_prediction(model, s1, s2, team1, team2, is_group)

            top3 = result["top3"]
            pw, pd_, pl = result["win"], result["draw"], result["loss"]
            lam1, lam2 = result["lam1"], result["lam2"]

            top_key, top_label, top_prob, edge = best_outcome(pw, pd_, pl, team1, team2)
            total_xg = lam1 + lam2
            total_prob = max(pw + pd_ + pl, 1e-9)
            btts = float(np.sum(result["mat"][1:, 1:]))

            st.markdown(
                f"""
                <div class="panel" style="padding:1.1rem 1.1rem 1.0rem 1.1rem; margin-bottom:1rem;">
                    <div class="small-label">match center</div>
                    <div style="display:flex; flex-wrap:wrap; gap:0.9rem; align-items:center; justify-content:space-between;">
                        <div style="min-width:220px;">
                            <div style="font-size:1.55rem; font-weight:800; color:#fff; line-height:1.1;">{team1} <span style="color:#9fb4cc; font-size:1rem;">vs</span> {team2}</div>
                            <div class="subtle" style="margin-top:0.25rem;">{stage}</div>
                        </div>
                        <div style="display:flex; gap:0.55rem; flex-wrap:wrap;">
                            <div style="background:rgba(97,218,251,0.12); border:1px solid rgba(97,218,251,0.22); padding:0.42rem 0.7rem; border-radius:999px;">🏆 Top outcome: <b>{top_label}</b></div>
                            <div style="background:rgba(61,220,151,0.12); border:1px solid rgba(61,220,151,0.22); padding:0.42rem 0.7rem; border-radius:999px;">🎯 Most likely: <b>{fmt_score(top3[0][0], top3[0][1])}</b></div>
                            <div style="background:rgba(255,209,102,0.12); border:1px solid rgba(255,209,102,0.22); padding:0.42rem 0.7rem; border-radius:999px;">⚡ Total xG: <b>{fmt_xg(total_xg)}</b></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🏆 Top outcome", top_label, f"+{edge*100:.1f}% edge")
            k2.metric("🎯 Most likely", fmt_score(top3[0][0], top3[0][1]), fmt_pct(top_prob))
            k3.metric("⚡ Total xG", fmt_xg(total_xg))
            k4.metric("🔥 BTTS", fmt_pct(btts))

            st.write("")

            st.markdown(
                f"""
                <div class="xg-strip">
                    <div class="xg-item">
                        <div class="xg-name">{team1}</div>
                        <div class="xg-val">{fmt_xg(lam1)}</div>
                        <div class="xg-name">expected goals</div>
                    </div>
                    <div class="xg-divider">⚡</div>
                    <div class="xg-item">
                        <div class="xg-name">match type</div>
                        <div class="xg-val" style="color:#ffffff; font-size:1.1rem; line-height:1.25; padding-top:0.22rem;">{stage}</div>
                        <div class="xg-name">setup</div>
                    </div>
                    <div class="xg-divider">⚡</div>
                    <div class="xg-item">
                        <div class="xg-name">{team2}</div>
                        <div class="xg-val">{fmt_xg(lam2)}</div>
                        <div class="xg-name">expected goals</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            section_title("🧠 prediction summary")

            summary_col1, summary_col2 = st.columns([1.4, 1])
            with summary_col1:
                if top_key == "team1":
                    summary_text = (
                        f"{team1} are favored because the model leans toward a stronger attacking output and overall edge in the feature blend. "
                        f"The most likely scoreline is {fmt_score(top3[0][0], top3[0][1])}, and the model gives {team1} a {fmt_pct(pw)} chance to win."
                    )
                elif top_key == "draw":
                    summary_text = (
                        f"The model sees this as a tight match with draw value rising. "
                        f"The most likely scoreline is {fmt_score(top3[0][0], top3[0][1])}, and the draw probability is {fmt_pct(pd_)}."
                    )
                else:
                    summary_text = (
                        f"{team2} are favored by the model based on the combined strength profile and expected goals balance. "
                        f"The most likely scoreline is {fmt_score(top3[0][0], top3[0][1])}, and {team2} win probability is {fmt_pct(pl)}."
                    )

                st.markdown(
                    f"""
                    <div class="panel" style="min-height:260px;">
                        <div style="font-size:1.05rem; font-weight:800; color:#fff; margin-bottom:0.45rem;">🧠 Match Summary</div>
                        <div class="subtle" style="font-size:0.98rem; line-height:1.75;">{summary_text}</div>
                        <hr>
                        <div style="display:flex; flex-wrap:wrap; gap:0.6rem;">
                            <div style="background:rgba(97,218,251,0.10); border:1px solid rgba(97,218,251,0.18); padding:0.45rem 0.7rem; border-radius:999px;">📍 Top score: <b>{fmt_score(top3[0][0], top3[0][1])}</b></div>
                            <div style="background:rgba(61,220,151,0.10); border:1px solid rgba(61,220,151,0.18); padding:0.45rem 0.7rem; border-radius:999px;">🔒 Win/Draw/Loss sum: <b>{fmt_pct(total_prob)}</b></div>
                            <div style="background:rgba(255,143,94,0.10); border:1px solid rgba(255,143,94,0.18); padding:0.45rem 0.7rem; border-radius:999px;">⚔️ Edge: <b>{fmt_pct(edge)}</b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with summary_col2:
                st.plotly_chart(
                    confidence_gauge(max(pw, pd_, pl)),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            st.write("")
            section_title("🎯 top 3 predicted scorelines")

            cols = st.columns(3)
            medals = [("🥇", "first"), ("🥈", "second"), ("🥉", "third")]

            for i, ((emoji, cls), (g1, g2, prob)) in enumerate(zip(medals, top3)):
                outcome_label = f"{team1} win" if g1 > g2 else ("draw" if g1 == g2 else f"{team2} win")
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="score-card {cls}">
                            <div class="medal">{emoji}</div>
                            <div class="scoreline">{g1} – {g2}</div>
                            <div class="probability">{prob*100:.2f}%</div>
                            <div class="outcome">{outcome_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.write("")
            top10 = []
            flat = np.argsort(result["mat"].ravel())[::-1]
            for idx in flat[:10]:
                g1, g2 = divmod(int(idx), result["mat"].shape[1])
                top10.append((g1, g2, float(result["mat"][g1, g2])))

            left, right = st.columns([1.1, 0.9])

            with left:
                section_title("📈 win / draw / loss probabilities")
                st.plotly_chart(
                    outcome_chart(pw, pd_, pl, team1, team2),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🏆 {team1} win", fmt_pct(pw))
                c2.metric("🤝 draw", fmt_pct(pd_))
                c3.metric(f"🏆 {team2} win", fmt_pct(pl))

            with right:
                section_title("🍩 outcome split")
                st.plotly_chart(
                    outcome_donut(pw, pd_, pl, team1, team2),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            st.write("")
            c1, c2 = st.columns([3, 2])
            with c1:
                section_title("🔥 score probability heatmap")
                st.plotly_chart(
                    heatmap_chart(result["mat"], team1, team2),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            with c2:
                section_title("📊 score distribution")
                st.plotly_chart(
                    goals_dist_chart(lam1, lam2, team1, team2),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.plotly_chart(
                    radar_chart(s1, s2, team1, team2),
                    width="stretch",
                    config={"displayModeBar": False},
                )

            st.write("")
            section_title("🏅 top 10 scoreline leaderboard")
            leaderboard = score_table(top10, team1, team2)
            st.dataframe(dark_df_style(leaderboard), width="stretch", hide_index=True)

    with tab_bt:
        st.markdown("### 📋 latest matches backtest")
        if len(df) < 5:
            st.warning("not enough rows for a useful backtest.")
        else:
            bt_max = min(50, len(df))
            if bt_max <= 5:
                n_bt = bt_max
            else:
                n_bt = st.slider("number of matches to backtest", 5, bt_max, min(20, bt_max))

            with st.spinner("running backtest..."):
                bt_df, oc, ex = get_backtest_df(df, model, n=n_bt)

            c1, c2, c3 = st.columns(3)
            c1.metric("Outcome accuracy", f"{oc}/{n_bt} ({oc / n_bt * 100:.0f}%)")
            c2.metric("Exact score hits", f"{ex}/{n_bt} ({ex / n_bt * 100:.0f}%)")
            c3.metric("Matches tested", str(n_bt))

            def highlight(row):
                if row["Outcome ✓?"] == "✅":
                    color = "background-color: rgba(61,220,151,0.12); color: #edf4ff;"
                else:
                    color = "background-color: rgba(255,107,107,0.10); color: #edf4ff;"
                return [color] * len(row)

            st.dataframe(
                dark_df_style(bt_df, highlight=highlight),
                width="stretch",
                height=520,
            )

    with tab_info:
        st.markdown("### ℹ️ how the prediction engine works")
        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div class="panel">
                    <div class="section-title">model stack</div>
                    <div class="subtle">
                        three models are trained together and blended with inverse-cv-mse weighting.
                    </div>
                    <div style="margin-top:0.8rem; line-height:1.8; color:#edf4ff;">
                        <b>poisson glm</b> — count-data baseline for goals<br>
                        <b>gradient boosting</b> — nonlinear interactions<br>
                        <b>random forest</b> — variance reduction and stability
                    </div>
                    <hr>
                    <div class="section-title">dixon-coles correction</div>
                    <div class="subtle">
                        small-score cells like 0–0, 1–0, 0–1, and 1–1 are adjusted with a fitted dependency parameter ρ.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="panel">
                    <div class="section-title">feature set</div>
                    <div class="subtle">
                        the model uses 21 engineered features derived from team strength, ratings, and form.
                    </div>
                    <div style="margin-top:0.8rem; line-height:1.8; color:#edf4ff;">
                        <b>elo</b> — difference, ratio, win prob, sum<br>
                        <b>ratings</b> — attack / defense per team<br>
                        <b>xg proxy</b> — attack divided by opponent defense<br>
                        <b>form</b> — last-5 points and weighted form<br>
                        <b>composite</b> — blended strength index<br>
                        <b>stage</b> — group vs knockout flag
                    </div>
                    <hr>
                    <div class="section-title">prediction pipeline</div>
                    <div class="subtle" style="font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;">
                        features → ensemble λ₁, λ₂ → poisson grid → dixon-coles adjustment → renormalize → rank scores → sum win/draw/loss cells
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("### 📊 trained model metrics")

        m = model.metrics
        metrics_data = pd.DataFrame(
            {
                "Metric": [
                    "RMSE – T1 goals",
                    "RMSE – T2 goals",
                    "MAE – T1 goals",
                    "MAE – T2 goals",
                    "Exact score accuracy",
                    "Outcome accuracy",
                    "Dixon-Coles ρ",
                ],
                "Value": [
                    f"{m['rmse_t1']:.4f}",
                    f"{m['rmse_t2']:.4f}",
                    f"{m['mae_t1']:.4f}",
                    f"{m['mae_t2']:.4f}",
                    f"{m['exact_score']*100:.1f}%",
                    f"{m['outcome_acc']*100:.1f}%",
                    f"{m['rho']:+.4f}",
                ],
                "Notes": [
                    "goal prediction error for team 1",
                    "goal prediction error for team 2",
                    "mean absolute goal error for team 1",
                    "mean absolute goal error for team 2",
                    "training exact score hit rate",
                    "training win/draw/loss hit rate",
                    "low-score dependency fit by MLE",
                ],
            }
        )
        st.dataframe(dark_df_style(metrics_data), width="stretch", hide_index=True)

        st.write("")
        st.markdown("### ⚖️ ensemble weights")
        w1 = m["w1"]
        w2 = m["w2"]
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Poisson GLM", f"T1: {w1[0]:.3f}  |  T2: {w2[0]:.3f}")
        wc2.metric("Gradient Boosting", f"T1: {w1[1]:.3f}  |  T2: {w2[1]:.3f}")
        wc3.metric("Random Forest", f"T1: {w1[2]:.3f}  |  T2: {w2[2]:.3f}")

        st.write("")
        st.markdown("### 📂 raw dataset preview")
        preview = df.head(30).copy()
        st.dataframe(dark_df_style(preview), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()