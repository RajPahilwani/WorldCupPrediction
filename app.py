"""
⚽ Final Score Prediction Platform — Streamlit App
===================================================
Run with:  streamlit run app.py

Put FINALSCORES.csv in the same folder, or upload it via the sidebar.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Score Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MAX_GOALS   = 9
ENSEMBLE_CV = 5
DEFAULT_CSV = "FINALSCORES.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall font */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Top banner */
.banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 14px;
    padding: 26px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.banner h1 { color: #00d4ff; font-size: 2.2rem; margin: 0 0 4px 0; }
.banner p  { color: #a0c4d8; font-size: 0.95rem; margin: 0; }

/* Score medal cards */
.medal-row { display: flex; gap: 16px; margin-bottom: 20px; }
.medal-card {
    flex: 1;
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
}
.medal-card.gold   { background: linear-gradient(135deg,#7c5a00,#c9941a); border: 2px solid #f5c518; }
.medal-card.silver { background: linear-gradient(135deg,#1a2a3a,#2e4a60); border: 2px solid #00bcd4; }
.medal-card.bronze { background: linear-gradient(135deg,#1a2e1a,#2a5a2a); border: 2px solid #66bb6a; }
.medal-card .rank  { font-size: 1.8rem; margin-bottom: 4px; }
.medal-card .score { font-size: 2.4rem; font-weight: 800; color: #ffffff; letter-spacing: 2px; }
.medal-card .prob  { font-size: 1.1rem; font-weight: 600; color: #e0e0e0; margin-top: 4px; }
.medal-card .label { font-size: 0.8rem; color: #b0b0b0; margin-top: 6px; text-transform: uppercase; letter-spacing: 1px; }

/* Verdict banner */
.verdict {
    border-radius: 12px;
    padding: 20px 28px;
    margin: 20px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    font-size: 1.3rem;
    font-weight: 700;
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
}
.verdict.win  { background: linear-gradient(135deg,#0a3d0a,#1a6b1a); border: 2px solid #4caf50; color: #c8f7c8; }
.verdict.draw { background: linear-gradient(135deg,#3d3300,#6b5a00); border: 2px solid #ffc107; color: #fff3c8; }
.verdict.loss { background: linear-gradient(135deg,#3d0a0a,#6b1a1a); border: 2px solid #f44336; color: #f7c8c8; }

/* xG metric strip */
.xg-strip {
    background: #0f1923;
    border-radius: 10px;
    padding: 14px 20px;
    margin: 16px 0;
    display: flex;
    justify-content: space-around;
    border: 1px solid #1e3040;
}
.xg-item { text-align: center; }
.xg-item .xg-val  { font-size: 2rem; font-weight: 800; color: #00d4ff; }
.xg-item .xg-name { font-size: 0.8rem; color: #8899aa; text-transform: uppercase; letter-spacing: 1px; }
.xg-divider { color: #334455; font-size: 1.8rem; align-self: center; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1b2a; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════  ML ENGINE  ══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["elo_diff"]          = df["Elo1"] - df["Elo2"]
    f["elo_ratio"]         = df["Elo1"] / df["Elo2"]
    f["elo_win_prob"]      = 1.0 / (1.0 + 10.0 ** ((df["Elo2"] - df["Elo1"]) / 400.0))
    f["elo_sum"]           = df["Elo1"] + df["Elo2"]
    f["attack_t1"]         = df["Attack_Rating_T1"]
    f["attack_t2"]         = df["Attack_Rating_T2"]
    f["defense_t1"]        = df["Defense_Rating_T1"]
    f["defense_t2"]        = df["Defense_Rating_T2"]
    f["xg_t1"]             = df["Attack_Rating_T1"] / df["Defense_Rating_T2"].clip(lower=0.1)
    f["xg_t2"]             = df["Attack_Rating_T2"] / df["Defense_Rating_T1"].clip(lower=0.1)
    f["xg_diff"]           = f["xg_t1"] - f["xg_t2"]
    f["form_last5_t1"]     = df["Form_Last5_T1"]
    f["form_last5_t2"]     = df["Form_Last5_T2"]
    f["form_weighted_t1"]  = df["Form_Weighted_T1"]
    f["form_weighted_t2"]  = df["Form_Weighted_T2"]
    f["form_diff_last5"]   = df["Form_Last5_T1"]  - df["Form_Last5_T2"]
    f["form_diff_weighted"]= df["Form_Weighted_T1"] - df["Form_Weighted_T2"]
    f["strength_t1"]       = (f["elo_win_prob"] * 0.40 +
                               f["xg_t1"].clip(0, 6) * 0.15 +
                               f["form_weighted_t1"] / 100.0 * 0.45)
    f["strength_t2"]       = ((1 - f["elo_win_prob"]) * 0.40 +
                               f["xg_t2"].clip(0, 6) * 0.15 +
                               f["form_weighted_t2"] / 100.0 * 0.45)
    f["strength_ratio"]    = f["strength_t1"] / f["strength_t2"].clip(lower=0.01)
    f["is_group"]          = df["Stage"].str.contains("Group", case=False, na=False).astype(int)
    return f


def make_feature_row(s1: dict, s2: dict, is_group: bool) -> pd.DataFrame:
    e1, e2 = s1["elo"], s2["elo"]
    wp = 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))
    xg1 = s1["attack"] / max(s2["defense"], 0.1)
    xg2 = s2["attack"] / max(s1["defense"], 0.1)
    fw1, fw2 = s1["form_weighted"], s2["form_weighted"]
    fl1, fl2 = s1["form_last5"],    s2["form_last5"]
    st1 = wp * 0.40 + min(xg1, 6) * 0.15 + fw1 / 100.0 * 0.45
    st2 = (1 - wp) * 0.40 + min(xg2, 6) * 0.15 + fw2 / 100.0 * 0.45
    return pd.DataFrame([{
        "elo_diff": e1 - e2, "elo_ratio": e1 / e2, "elo_win_prob": wp, "elo_sum": e1 + e2,
        "attack_t1": s1["attack"], "attack_t2": s2["attack"],
        "defense_t1": s1["defense"], "defense_t2": s2["defense"],
        "xg_t1": xg1, "xg_t2": xg2, "xg_diff": xg1 - xg2,
        "form_last5_t1": fl1, "form_last5_t2": fl2,
        "form_weighted_t1": fw1, "form_weighted_t2": fw2,
        "form_diff_last5": fl1 - fl2, "form_diff_weighted": fw1 - fw2,
        "strength_t1": st1, "strength_t2": st2,
        "strength_ratio": st1 / max(st2, 0.01),
        "is_group": int(is_group),
    }])


def build_team_lookup(df: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in df.iterrows():
        for tc, ec, ac, dc, fl, fw in [
            ("Team 1","Elo1","Attack_Rating_T1","Defense_Rating_T1","Form_Last5_T1","Form_Weighted_T1"),
            ("Team 2","Elo2","Attack_Rating_T2","Defense_Rating_T2","Form_Last5_T2","Form_Weighted_T2"),
        ]:
            lookup[row[tc]] = {
                "elo": float(row[ec]), "attack": float(row[ac]),
                "defense": float(row[dc]), "form_last5": float(row[fl]),
                "form_weighted": float(row[fw]),
            }
    return lookup


# ── Dixon-Coles correction ────────────────────────────────────────────────────

def dc_tau(g1, g2, lam1, lam2, rho):
    if   g1 == 0 and g2 == 0: return max(1e-6, 1 - lam1 * lam2 * rho)
    elif g1 == 1 and g2 == 0: return max(1e-6, 1 + lam2 * rho)
    elif g1 == 0 and g2 == 1: return max(1e-6, 1 + lam1 * rho)
    elif g1 == 1 and g2 == 1: return max(1e-6, 1 - rho)
    return 1.0


def estimate_rho(g1_arr, g2_arr, lam1_arr, lam2_arr):
    def neg_ll(rho):
        if abs(rho) >= 0.99: return 1e12
        ll = 0.0
        for i in range(len(g1_arr)):
            if g1_arr[i] <= 1 and g2_arr[i] <= 1:
                tau = dc_tau(g1_arr[i], g2_arr[i], lam1_arr[i], lam2_arr[i], rho)
                ll += np.log(max(tau, 1e-9))
        return -ll
    res = minimize_scalar(neg_ll, bounds=(-0.99, 0.99), method="bounded")
    return float(res.x)


# ── Ensemble model ────────────────────────────────────────────────────────────

class ScorePredictionEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        gb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.04,
                     subsample=0.75, min_samples_leaf=4, random_state=42)
        rf_kw = dict(n_estimators=300, max_depth=5, min_samples_leaf=4, random_state=42)
        self.p1  = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb1 = GradientBoostingRegressor(**gb_kw)
        self.rf1 = RandomForestRegressor(**rf_kw)
        self.p2  = PoissonRegressor(alpha=0.3, max_iter=2000)
        self.gb2 = GradientBoostingRegressor(**gb_kw)
        self.rf2 = RandomForestRegressor(**rf_kw)
        self.w1 = np.array([1/3, 1/3, 1/3])
        self.w2 = np.array([1/3, 1/3, 1/3])
        self.rho = 0.0
        self.feature_cols = []
        self.metrics = {}

    def fit(self, X: pd.DataFrame, y1, y2):
        self.feature_cols = X.columns.tolist()
        Xs = self.scaler.fit_transform(X)
        kf = KFold(n_splits=ENSEMBLE_CV, shuffle=True, random_state=0)

        for est, y in [(self.p1, y1), (self.gb1, y1), (self.rf1, y1),
                       (self.p2, y2), (self.gb2, y2), (self.rf2, y2)]:
            est.fit(Xs, y)

        def cv_mse(e, y):
            return -cross_val_score(e, Xs, y, cv=kf,
                                    scoring="neg_mean_squared_error").mean()

        for w_attr, ests, y in [("w1", [self.p1, self.gb1, self.rf1], y1),
                                  ("w2", [self.p2, self.gb2, self.rf2], y2)]:
            mses = np.array([cv_mse(e, y) for e in ests])
            inv  = 1.0 / (mses + 1e-9)
            setattr(self, w_attr, inv / inv.sum())

        lam1, lam2 = self._raw(Xs)
        self.rho = estimate_rho(y1.astype(int), y2.astype(int), lam1, lam2)

        p1r = np.round(lam1).clip(0).astype(int)
        p2r = np.round(lam2).clip(0).astype(int)
        self.metrics = {
            "rmse_t1":     float(np.sqrt(mean_squared_error(y1, lam1))),
            "rmse_t2":     float(np.sqrt(mean_squared_error(y2, lam2))),
            "mae_t1":      float(mean_absolute_error(y1, lam1)),
            "mae_t2":      float(mean_absolute_error(y2, lam2)),
            "exact_score": float(np.mean((p1r == y1.astype(int)) & (p2r == y2.astype(int)))),
            "outcome_acc": float(np.mean(np.sign(lam1 - lam2) == np.sign(y1 - y2))),
            "rho":         self.rho,
            "w1":          self.w1.tolist(),
            "w2":          self.w2.tolist(),
        }
        return self

    def _raw(self, Xs):
        def ens(ests, weights):
            return sum(w * np.clip(e.predict(Xs), 0, 10)
                       for e, w in zip(ests, weights))
        return (np.clip(ens([self.p1, self.gb1, self.rf1], self.w1), 0.05, 7.0),
                np.clip(ens([self.p2, self.gb2, self.rf2], self.w2), 0.05, 7.0))

    def predict_xg(self, X: pd.DataFrame):
        Xs = self.scaler.transform(X[self.feature_cols])
        l1, l2 = self._raw(Xs)
        return float(l1[0]), float(l2[0])

    def score_matrix(self, lam1, lam2):
        mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for g1 in range(MAX_GOALS + 1):
            for g2 in range(MAX_GOALS + 1):
                mat[g1, g2] = (poisson.pmf(g1, lam1) *
                                poisson.pmf(g2, lam2) *
                                dc_tau(g1, g2, lam1, lam2, self.rho))
        return mat / mat.sum()

    def outcomes(self, mat):
        n = mat.shape[0]
        pw = pd_ = pl = 0.0
        for g1 in range(n):
            for g2 in range(n):
                if   g1 > g2:  pw  += mat[g1, g2]
                elif g1 == g2: pd_ += mat[g1, g2]
                else:           pl  += mat[g1, g2]
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
    return {"name1": name1, "name2": name2,
            "lam1": lam1, "lam2": lam2,
            "top3": top3, "win": pw, "draw": pd_, "loss": pl, "mat": mat}


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA & MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))


@st.cache_resource(show_spinner=False)
def train_model(csv_bytes: bytes):
    df     = load_csv(csv_bytes)
    lookup = build_team_lookup(df)
    X      = engineer_features(df)
    y1     = df["Goals1"].values.astype(float)
    y2     = df["Goals2"].values.astype(float)
    model  = ScorePredictionEnsemble()
    model.fit(X, y1, y2)
    return model, lookup, df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c8d8e8",
    margin=dict(l=10, r=10, t=40, b=10),
)


def outcome_chart(pw, pd_, pl, name1, name2):
    total = pw + pd_ + pl
    labels = [f"🏆 {name1} Win", "🤝 Draw", f"❌ {name2} Win"]
    values = [pw / total * 100, pd_ / total * 100, pl / total * 100]
    colors = ["#4caf50", "#ffc107", "#f44336"]

    fig = go.Figure()
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            name=label, x=[val], y=[""], orientation="h",
            marker_color=color,
            text=f"{val:.1f}%", textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=15, color="white", family="Inter"),
            hovertemplate=f"{label}: {val:.2f}%<extra></extra>",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        height=90,
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 100]),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def heatmap_chart(mat, name1, name2, show=7):
    sub = mat[:show, :show] * 100
    text = [[f"{sub[r, c]:.1f}%" for c in range(show)] for r in range(show)]

    fig = go.Figure(go.Heatmap(
        z=sub,
        x=[str(i) for i in range(show)],
        y=[str(i) for i in range(show)],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12, color="white"),
        colorscale=[
            [0.0,  "rgba(10,30,50,0.9)"],
            [0.3,  "rgba(0,80,120,0.9)"],
            [0.6,  "rgba(0,150,180,0.9)"],
            [0.85, "rgba(250,180,0,0.9)"],
            [1.0,  "rgba(255,220,0,1.0)"],
        ],
        showscale=False,
        hovertemplate=f"{name1} %{{y}} – {name2} %{{x}}: %{{text}}<extra></extra>",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        xaxis=dict(title=f"← {name2} goals →", tickfont=dict(size=13), gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title=f"← {name1} goals →", tickfont=dict(size=13), gridcolor="rgba(255,255,255,0.05)"),
        title=dict(text="Score Probability Heatmap (%)", x=0.5, font=dict(size=15, color="#8ab4cc")),
    )
    return fig


def radar_chart(s1, s2, name1, name2):
    cats = ["Elo (norm)", "Attack", "Defense", "Form (wt)", "Form L5"]

    def norm(val, lo, hi):
        return (val - lo) / (hi - lo) * 100

    elo_lo, elo_hi = 1200, 2200
    vals1 = [
        norm(s1["elo"], elo_lo, elo_hi),
        s1["attack"]       / 5  * 100,
        s1["defense"]      / 2  * 100,
        s1["form_weighted"]/ 50 * 100,
        s1["form_last5"]   / 15 * 100,
    ]
    vals2 = [
        norm(s2["elo"], elo_lo, elo_hi),
        s2["attack"]       / 5  * 100,
        s2["defense"]      / 2  * 100,
        s2["form_weighted"]/ 50 * 100,
        s2["form_last5"]   / 15 * 100,
    ]

    fig = go.Figure()
    for vals, name, color in [(vals1, name1, "#00d4ff"), (vals2, name2, "#ff6b6b")]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself", name=name,
            line=dict(color=color, width=2),
            fillcolor=color.replace("ff", "33") if "#" not in color else color + "33",
            opacity=0.8,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340,
        polar=dict(
            bgcolor="rgba(10,20,35,0.8)",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.1)",
                            tickfont=dict(color="#5a7a8a", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             tickfont=dict(color="#a0b8cc", size=11)),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12,
                    font=dict(size=13)),
        title=dict(text="Team Attribute Comparison", x=0.5,
                   font=dict(size=14, color="#8ab4cc")),
    )
    return fig


def goals_dist_chart(lam1, lam2, name1, name2):
    goals = list(range(MAX_GOALS + 1))
    p1 = [poisson.pmf(g, lam1) * 100 for g in goals]
    p2 = [poisson.pmf(g, lam2) * 100 for g in goals]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=goals, y=p1, name=name1,
                          marker_color="#00d4ff", opacity=0.85,
                          hovertemplate="%{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(x=goals, y=p2, name=name2,
                          marker_color="#ff6b6b", opacity=0.85,
                          hovertemplate="%{y:.1f}%<extra></extra>"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        barmode="group",
        xaxis=dict(title="Goals scored", tickvals=goals,
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Probability (%)", gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12,
                    font=dict(size=12)),
        title=dict(text="Goal Scoring Distribution", x=0.5,
                   font=dict(size=14, color="#8ab4cc")),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_backtest_df(df, model, n=20):
    rows = []
    sample = df.tail(n)
    for _, row in sample.iterrows():
        s1 = dict(elo=row["Elo1"], attack=row["Attack_Rating_T1"],
                  defense=row["Defense_Rating_T1"], form_last5=row["Form_Last5_T1"],
                  form_weighted=row["Form_Weighted_T1"])
        s2 = dict(elo=row["Elo2"], attack=row["Attack_Rating_T2"],
                  defense=row["Defense_Rating_T2"], form_last5=row["Form_Last5_T2"],
                  form_weighted=row["Form_Weighted_T2"])
        r = run_prediction(model, s1, s2, row["Team 1"], row["Team 2"],
                           "Group" in str(row["Stage"]))
        g1a, g2a = int(row["Goals1"]), int(row["Goals2"])
        pg1, pg2, prob = r["top3"][0]
        actual  = "W" if g1a > g2a else ("D" if g1a == g2a else "L")
        pw, pd_, pl = r["win"], r["draw"], r["loss"]
        pred    = ("W" if pw > pl and pw > pd_ else
                   ("D" if pd_ >= pw and pd_ >= pl else "L"))
        rows.append({
            "Match":         f"{row['Team 1']}  vs  {row['Team 2']}",
            "Stage":         row["Stage"],
            "Actual Score":  f"{g1a}–{g2a}",
            "Pred Score #1": f"{pg1}–{pg2}",
            "Prob":          f"{prob*100:.1f}%",
            "Outcome ✓?":    "✅" if actual == pred else "❌",
            "Exact ✓?":      "✅" if pg1 == g1a and pg2 == g2a else "❌",
            "xG T1":         f"{r['lam1']:.2f}",
            "xG T2":         f"{r['lam2']:.2f}",
        })
    bt = pd.DataFrame(rows)
    oc = (bt["Outcome ✓?"] == "✅").sum()
    ex = (bt["Exact ✓?"]   == "✅").sum()
    return bt, oc, ex


# ─────────────────────────────────────────────────────────────────────────────
# MAIN STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="banner">
      <h1>⚽ Final Score Prediction Platform</h1>
      <p>Ensemble ML · Poisson GLM · Dixon-Coles Correction · 3-Model Stack</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Data Source")
        uploaded = st.file_uploader("Upload your CSV", type=["csv"],
                                     help="Expected columns: Team 1, Team 2, Elo1, Elo2, Goals1, Goals2, Attack/Defense/Form ratings")

        # Resolve CSV bytes
        csv_bytes = None
        if uploaded is not None:
            csv_bytes = uploaded.read()
        else:
            try:
                with open(DEFAULT_CSV, "rb") as f:
                    csv_bytes = f.read()
                st.caption(f"Using `{DEFAULT_CSV}` from working directory.")
            except FileNotFoundError:
                st.warning(f"No file uploaded and `{DEFAULT_CSV}` not found. Please upload a CSV above.")
                st.stop()

        st.divider()

        # Train / load model
        with st.spinner("🧠 Training ensemble model…"):
            model, lookup, df = train_model(csv_bytes)

        st.success(f"✅ {len(df)} matches · {len(lookup)} teams loaded")

        st.markdown("---")
        st.markdown("### 🏟️ Match Setup")

        teams = sorted(lookup.keys())

        t1_default = teams.index("Brazil") if "Brazil" in teams else 0
        t2_default = teams.index("Argentina") if "Argentina" in teams else 1

        team1 = st.selectbox("🔵 Team 1", teams, index=t1_default)
        team2 = st.selectbox("🔴 Team 2",
                              [t for t in teams if t != team1],
                              index=min(t2_default, len(teams) - 2))

        stage = st.radio("🏆 Stage", ["Group Stage", "Knockout"], horizontal=True)
        is_group = (stage == "Group Stage")

        predict_btn = st.button("⚡ Run Prediction", type="primary", use_container_width=True)

        st.divider()
        m = model.metrics
        st.markdown("### 📊 Model Health")
        st.metric("Outcome Accuracy",  f"{m['outcome_acc']*100:.1f}%")
        st.metric("Exact Score Acc.",   f"{m['exact_score']*100:.1f}%")
        st.metric("xG RMSE (T1 / T2)", f"{m['rmse_t1']:.3f} / {m['rmse_t2']:.3f}")
        st.metric("Dixon-Coles ρ",      f"{m['rho']:+.4f}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_pred, tab_bt, tab_info = st.tabs(["🎯 Prediction", "📋 Backtest", "ℹ️ Model Info"])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – PREDICTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab_pred:
        if not predict_btn:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#4a6a7a;'>
              <div style='font-size:4rem;'>⚽</div>
              <div style='font-size:1.3rem; margin-top:12px;'>
                Select two teams in the sidebar and click <strong>Run Prediction</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            s1 = lookup[team1]
            s2 = lookup[team2]

            with st.spinner("Running ensemble prediction…"):
                result = run_prediction(model, s1, s2, team1, team2, is_group)

            top3  = result["top3"]
            pw, pd_, pl = result["win"], result["draw"], result["loss"]
            lam1, lam2  = result["lam1"], result["lam2"]
            total = pw + pd_ + pl

            # ── xG strip ──────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="xg-strip">
              <div class="xg-item">
                <div class="xg-name">{team1}</div>
                <div class="xg-val">{lam1:.2f}</div>
                <div class="xg-name">Expected Goals</div>
              </div>
              <div class="xg-divider">⚡</div>
              <div class="xg-item">
                <div class="xg-name">vs</div>
                <div class="xg-val" style="color:#8899aa;">–</div>
                <div class="xg-name">{stage}</div>
              </div>
              <div class="xg-divider">⚡</div>
              <div class="xg-item">
                <div class="xg-name">{team2}</div>
                <div class="xg-val">{lam2:.2f}</div>
                <div class="xg-name">Expected Goals</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Top 3 score cards ──────────────────────────────────────────────
            st.markdown("#### 🎯 Top 3 Predicted Scorelines")
            medals = [("🥇", "gold"), ("🥈", "silver"), ("🥉", "bronze")]
            cols = st.columns(3)
            for i, ((emoji, cls), (g1, g2, prob)) in enumerate(zip(medals, top3)):
                outcome_label = (f"🏆 {team1} WIN" if g1 > g2 else
                                  ("🤝 DRAW" if g1 == g2 else f"🏆 {team2} WIN"))
                with cols[i]:
                    st.markdown(f"""
                    <div class="medal-card {cls}">
                      <div class="rank">{emoji}</div>
                      <div class="score">{g1} – {g2}</div>
                      <div class="prob">{prob*100:.2f}%</div>
                      <div class="label">{outcome_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Outcome probability bar ────────────────────────────────────────
            st.markdown("#### 📈 Match Outcome Probabilities")
            st.plotly_chart(outcome_chart(pw, pd_, pl, team1, team2),
                            use_container_width=True, config={"displayModeBar": False})

            col_w, col_d, col_l = st.columns(3)
            col_w.metric(f"🏆 {team1} Win", f"{pw/total*100:.1f}%")
            col_d.metric("🤝 Draw",          f"{pd_/total*100:.1f}%")
            col_l.metric(f"❌ {team2} Win",  f"{pl/total*100:.1f}%")

            # ── Verdict ────────────────────────────────────────────────────────
            if pw >= pd_ and pw >= pl:
                v_class = "win"
                v_text  = f"🏆 {team1} are predicted to WIN  ·  {pw/total*100:.1f}% confidence"
            elif pd_ >= pw and pd_ >= pl:
                v_class = "draw"
                v_text  = f"🤝 A DRAW is the most likely outcome  ·  {pd_/total*100:.1f}% confidence"
            else:
                v_class = "loss"
                v_text  = f"🏆 {team2} are predicted to WIN  ·  {pl/total*100:.1f}% confidence"

            st.markdown(f'<div class="verdict {v_class}">{v_text}</div>',
                        unsafe_allow_html=True)

            # ── Heatmap + Goals dist side by side ─────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.plotly_chart(heatmap_chart(result["mat"], team1, team2),
                                use_container_width=True)
            with c2:
                st.plotly_chart(goals_dist_chart(lam1, lam2, team1, team2),
                                use_container_width=True)
                st.plotly_chart(radar_chart(s1, s2, team1, team2),
                                use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – BACKTEST
    # ══════════════════════════════════════════════════════════════════════════
    with tab_bt:
        st.markdown("### 📋 Last 20 Matches — Prediction vs Actual")
        n_bt = st.slider("Number of matches to backtest", 5, min(50, len(df)), 20)

        with st.spinner("Running backtest…"):
            bt_df, oc, ex = get_backtest_df(df, model, n=n_bt)

        c1, c2, c3 = st.columns(3)
        c1.metric("Outcome Accuracy",  f"{oc}/{n_bt}  ({oc/n_bt*100:.0f}%)")
        c2.metric("Exact Score Hits",  f"{ex}/{n_bt}  ({ex/n_bt*100:.0f}%)")
        c3.metric("Matches Tested",    str(n_bt))

        # Colour-code the dataframe
        def highlight(row):
            colour = ""
            if row["Outcome ✓?"] == "✅":
                colour = "background-color: rgba(76,175,80,0.15);"
            elif row["Outcome ✓?"] == "❌":
                colour = "background-color: rgba(244,67,54,0.12);"
            return [colour] * len(row)

        styled = bt_df.style.apply(highlight, axis=1)
        st.dataframe(styled, use_container_width=True, height=520)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – MODEL INFO
    # ══════════════════════════════════════════════════════════════════════════
    with tab_info:
        st.markdown("### ℹ️ How the Prediction Engine Works")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Model Stack**

Three models are trained simultaneously and combined via
*inverse-MSE weighting* (better CV models get higher weight):

| Model | Role |
|---|---|
| **Poisson GLM** | Baseline — perfect for count data like goals |
| **Gradient Boosting** | Captures non-linear feature interactions |
| **Random Forest** | Variance reduction via bagging |

**Dixon-Coles Correction**

Raw independent Poisson slightly over/under-predicts the
frequencies of 0–0, 1–0, 0–1, and 1–1 scorelines. The
correction factor τ(g₁, g₂, λ₁, λ₂, ρ) is applied to
those four cells only; ρ is estimated via MLE on your data.
""")

        with col2:
            st.markdown("""
**Feature Set (21 engineered features)**

| Feature group | Signals |
|---|---|
| **Elo** | Difference, ratio, win probability, sum |
| **Ratings** | Attack / Defense per team |
| **xG proxy** | Attack ÷ opponent Defence |
| **Form** | Last-5 points, weighted recent form |
| **Composite** | Strength index (Elo + xG + Form blend) |
| **Stage** | Group vs Knockout flag |

**Prediction pipeline**

```
features → ensemble predict λ₁, λ₂
→ Poisson PMF grid (MAX_GOALS × MAX_GOALS)
→ Dixon-Coles τ applied to {0-0,1-0,0-1,1-1}
→ renormalise matrix
→ rank scorelines  →  sum Win / Draw / Loss cells
```
""")

        st.markdown("---")
        st.markdown("### 📊 Trained Model Metrics")
        m = model.metrics
        metrics_data = {
            "Metric": ["RMSE – T1 goals", "RMSE – T2 goals",
                       "MAE – T1 goals", "MAE – T2 goals",
                       "Exact score accuracy", "Outcome accuracy",
                       "Dixon-Coles ρ"],
            "Value": [f"{m['rmse_t1']:.4f}", f"{m['rmse_t2']:.4f}",
                      f"{m['mae_t1']:.4f}", f"{m['mae_t2']:.4f}",
                      f"{m['exact_score']*100:.1f}%", f"{m['outcome_acc']*100:.1f}%",
                      f"{m['rho']:+.4f}"],
            "Notes": [
                "Goal prediction error for Team 1 (lower = better)",
                "Goal prediction error for Team 2 (lower = better)",
                "Mean absolute goal count error — T1",
                "Mean absolute goal count error — T2",
                "% of training matches where top prediction was exact",
                "% of training matches where W/D/L direction was correct",
                "Low-score dependency parameter (fitted via MLE)",
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

        w1 = m["w1"]
        w2 = m["w2"]
        st.markdown("### ⚖️ Ensemble Weights")
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Poisson GLM",      f"T1: {w1[0]:.3f}  |  T2: {w2[0]:.3f}")
        wc2.metric("Gradient Boosting", f"T1: {w1[1]:.3f}  |  T2: {w2[1]:.3f}")
        wc3.metric("Random Forest",     f"T1: {w1[2]:.3f}  |  T2: {w2[2]:.3f}")

        st.markdown("---")
        st.markdown("### 📂 Raw Dataset Preview")
        st.dataframe(df.head(30), use_container_width=True)


if __name__ == "__main__":
    main()