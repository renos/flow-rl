# ICML-ready grouped table (readable font, tight spacing, baseline alignment)
# ---------------------------------------------------------------------------
# - Baselines evaluated on SAME steps as reference ablation run
# - Grouped header with short labels, booktabs, minimal whitespace
# - Defaults to table* (two-column width) to avoid tiny text

import os, re, time, warnings
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import wandb
from wandb.apis.public import Api

# ----------------------------- CONFIG -----------------------------

# === Baselines (top) ===
BASELINE_RUNS = [
    "interpretable_rl/flow-rl/qmz5qr15",
    "interpretable_rl/flow-rl/v5zxnk7o",
    "interpretable_rl/flow-rl/5muf5m05",
    "interpretable_rl/flow-rl/rk3smltz",
    "interpretable_rl/flow-rl/6myw32xn",
]
BASELINE_NAMES = ["PPO-RNN", "PPO-FC", "E3B", "ICM", "PPO-RND"]

# === Ablations (bottom) ===
ABLATION_RUNS = [
    "interpretable_rl/flow-rl/abapycf4",
    "interpretable_rl/flow-rl/n9uu2se0",
    "interpretable_rl/flow-rl/1lvvnyt9",
    "interpretable_rl/flow-rl/8sio2ezt",
]
ABLATION_NAMES = [
    "SCALAR-Dense (Our Method)",
    "SCALAR-Sparse (Our Method)",
    "No Trajectory Analysis",
    "Shared Networks",
]

REFERENCE_METHOD_NAME = "SCALAR (Our Method)"  # for step alignment

# --- Columns (we'll render them as grouped headers with short labels) ---
# Full metric keys (fetch/aggregate on these):
METRICS = [
    "Achievements/place_table",
    "Achievements/place_furnace",
    "Achievements/make_wood_pickaxe",
    "Achievements/make_stone_pickaxe",
    "Achievements/make_iron_pickaxe",
    "Achievements/collect_diamond",
    "Achievements/wake_up",
    "Achievements/collect_drink",
    "Achievements/eat_cow",
    "episode_length",
]

# Display names if you print the dataframe preview (not the header in TeX)
DISPLAY_NAMES: Dict[str, str] = {
    "Achievements/collect_diamond": "Diamond (\\%)",
    "Achievements/place_table": "Table",
    "Achievements/place_furnace": "Furnace",
    "Achievements/make_wood_pickaxe": "Wood",
    "Achievements/make_stone_pickaxe": "Stone",
    "Achievements/make_iron_pickaxe": "Iron",
    "Achievements/wake_up": "Sleep",
    "Achievements/collect_drink": "Drink",
    "Achievements/eat_cow": "Eat",
    "episode_length": "Ep. Len.",
}

# Header grouping for the TeX table (order defines column order)
# (group title, [metric keys], [short column labels as shown in header])
HEADER_GROUPS = [
    (
        "Setup",
        [
            "Achievements/place_table",
            "Achievements/place_furnace",
        ],
        ["Table", "Furnace"],
    ),
    (
        "Pickaxes",
        [
            "Achievements/make_wood_pickaxe",
            "Achievements/make_stone_pickaxe",
            "Achievements/make_iron_pickaxe",
        ],
        ["Wood", "Stone", "Iron"],
    ),
    (
        "Goal",
        [
            "Achievements/collect_diamond",
        ],
        ["Diamond (\\%)"],
    ),
    (
        "Survival",
        [
            "Achievements/wake_up",
            "Achievements/collect_drink",
            "Achievements/eat_cow",
        ],
        ["Sleep", "Drink", "Eat"],
    ),
    (
        "Episodes",
        [
            "episode_length",
        ],
        ["Ep. Len."],
    ),
]

# Aggregation
X_AXIS = "_step"
AGG_METHOD = "last_k_mean"  # "last" | "last_k_mean" | "max"
LAST_K = 100
MAX_HISTORY_SAMPLES = None

# Output
SAVE_FOLDER = "paper_tables"
FILENAME_PREFIX = "ablations_table_icml_grouped_aligned"
CAPTION = (
    "Baselines (top) versus ablations (bottom). Baselines are aggregated using the "
    "same timesteps as the reference ablation run to ensure equal training horizon. "
    f"Values are mean over the last \\textit{{K}} points (K={LAST_K}); achievements "
    "reported as percentages when underlying metric is in [0,1]."
)
LABEL = "tab:ablations-icml-grouped-aligned"

# LaTeX styling (no resizebox; readable fonts; minimal whitespace)
TABLE_ENV_STAR = True  # True → table* (two-column width), False → table
TABLE_FONTSIZE = r"\small"  # \normalsize | \small
TABCOLSEP_PT = 3  # tighter columns
ARRAYSTRETCH = 1.05  # slightly tighter row height
ADDLINESPACE_PT = 1  # minimal extra space before group rows

# Robust W&B
RETRY_ON_ERROR = True
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# --------------------------- END CONFIG ---------------------------

api = Api()


def _robust(call, *a, **k):
    if not RETRY_ON_ERROR:
        return call(*a, **k)
    n = 0
    while True:
        try:
            return call(*a, **k)
        except Exception as e:
            n += 1
            if n >= MAX_RETRIES:
                raise e
            time.sleep(RETRY_SLEEP)


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def fetch_df(run_path: str, keys: Sequence[str]) -> Optional[pd.DataFrame]:
    try:
        run = _robust(api.run, run_path)
        need = sorted(set(list(keys) + [X_AXIS]))
        if MAX_HISTORY_SAMPLES is not None:
            df = _robust(
                run.history, keys=need, samples=MAX_HISTORY_SAMPLES, pandas=True
            )
        else:
            df = _robust(run.history, keys=need, pandas=True)
        if X_AXIS not in df.columns:
            df[X_AXIS] = np.arange(len(df))
        return df.sort_values(by=X_AXIS).reset_index(drop=True)
    except Exception as e:
        print(f"[fetch] {run_path}: {e}")
        return None


def aggregate(series: pd.Series, method: str = AGG_METHOD) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if method == "last":
        return float(s.iloc[-1])
    if method == "last_k_mean":
        return float(s.iloc[-min(max(1, LAST_K), len(s)) :].mean())
    if method == "max":
        return float(s.max())
    return float(s.iloc[-1])


def aggregate_aligned(series: pd.Series, method: str = AGG_METHOD) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if method in ("last", "last_k_mean"):
        return float(s.mean() if method == "last_k_mean" else s.iloc[-1])
    if method == "max":
        return float(s.max())
    return float(s.iloc[-1])


def maybe_percent(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    return v * 100.0 if 0.0 <= v <= 1.1 else v


def fmt(v: Optional[float], percent=False) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return r"\multicolumn{1}{c}{—}"
    if percent:
        return f"{v:.1f}"
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


def get_reference_steps() -> Tuple[np.ndarray, int]:
    try:
        ridx = ABLATION_NAMES.index(REFERENCE_METHOD_NAME)
        ref_run = ABLATION_RUNS[ridx]
    except ValueError:
        ref_run = ABLATION_RUNS[0]
        print(
            f"[warn] reference '{REFERENCE_METHOD_NAME}' not found; using {ABLATION_NAMES[0]}"
        )
    dfr = fetch_df(ref_run, [X_AXIS])
    if dfr is None or dfr.empty:
        return np.array([], dtype=int), 0
    ref_max = int(dfr[X_AXIS].max())
    if AGG_METHOD == "last":
        steps = np.array([ref_max], dtype=int)
    elif AGG_METHOD == "last_k_mean":
        k = min(max(1, LAST_K), len(dfr))
        steps = dfr[X_AXIS].astype(int).to_numpy()[-k:]
    else:
        steps = np.array([], dtype=int)
    return steps, ref_max


def values_at_steps(df: pd.DataFrame, metric: str, steps: np.ndarray) -> pd.Series:
    if metric not in df.columns or df.empty or steps.size == 0:
        return pd.Series([], dtype=float)
    sub = df[[X_AXIS, metric]].dropna().copy()
    sub[X_AXIS] = pd.to_numeric(sub[X_AXIS], errors="coerce").astype("Int64")
    sub = sub.dropna(subset=[X_AXIS]).set_index(X_AXIS)
    return sub.reindex(steps)[metric]


def build_rows(
    run_ids: List[str],
    names: List[str],
    align_to_steps: bool,
    ref_steps: np.ndarray,
    ref_max: int,
) -> List[Dict[str, str]]:
    rows = []
    for r, n in zip(run_ids, names):
        df = fetch_df(r, METRICS)
        row = {"Method": latex_escape(n)}
        if df is None or df.empty:
            for gtitle, metrics, _labels in HEADER_GROUPS:
                for m in metrics:
                    row[m] = r"\multicolumn{1}{c}{—}"
            rows.append(row)
            continue

        if AGG_METHOD == "max":  # fairness cutoff
            df = df[df[X_AXIS] <= ref_max]

        for m in METRICS:
            if m not in df.columns:
                row[m] = r"\multicolumn{1}{c}{—}"
                continue
            if align_to_steps and AGG_METHOD in ("last", "last_k_mean"):
                val = aggregate_aligned(values_at_steps(df, m, ref_steps), AGG_METHOD)
            else:
                val = aggregate(df[m])
            val = (
                maybe_percent(val)
                if m.startswith("Achievements/") and "episode" not in m
                else val
            )
            row[m] = fmt(
                val, percent=(m != "episode_length" and m.startswith("Achievements/"))
            )
        rows.append(row)
    return rows


def tex_grouped_table(
    baselines_df: pd.DataFrame, ablations_df: pd.DataFrame, caption: str, label: str
) -> str:
    assert list(baselines_df.columns) == list(ablations_df.columns)
    # Column order = Method + HEADER_GROUPS metrics in order
    col_order = ["Method"] + [m for _g, metrics, _l in HEADER_GROUPS for m in metrics]
    n_cols = 1 + sum(len(metrics) for _g, metrics, _l in HEADER_GROUPS)

    env = "table*" if TABLE_ENV_STAR else "table"
    col_spec = "@{}l" + ("r" * (n_cols - 1)) + "@{}"  # kill outer padding

    # First header row: group titles
    parts = ["Method"]
    col_ranges = []  # for cmidrule
    col_idx = 2  # 1-based; col 1 is Method
    for gtitle, metrics, labels in HEADER_GROUPS:
        span = len(metrics)
        parts.append(rf"\multicolumn{{{span}}}{{c}}{{{gtitle}}}")
        col_ranges.append((col_idx, col_idx + span - 1))
        col_idx += span
    header1 = " & ".join(parts) + r" \\"

    # cmidrules
    cmids = [rf"\cmidrule(lr){{{a}-{b}}}" for (a, b) in col_ranges]

    # Second header row: short labels
    labels = ["Method"] + [lbl for _g, metrics, lbls in HEADER_GROUPS for lbl in lbls]
    header2 = " & ".join(labels) + r" \\"

    def body(df: pd.DataFrame) -> List[str]:
        lines = []
        for _, r in df.iterrows():
            cells = [r["Method"]] + [
                r[m] for _g, metrics, _l in HEADER_GROUPS for m in metrics
            ]
            lines.append(" & ".join(cells) + r" \\")
        return lines

    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        TABLE_FONTSIZE,
        rf"\setlength{{\tabcolsep}}{{{TABCOLSEP_PT}pt}}",
        rf"\renewcommand{{\arraystretch}}{{{ARRAYSTRETCH}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header1,
        *cmids,
        header2,
        r"\midrule",
        rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Baselines}}}} \\",
        rf"\addlinespace[{ADDLINESPACE_PT}pt]",
        *body(baselines_df),
        r"\midrule",
        rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Ablations}}}} \\",
        rf"\addlinespace[{ADDLINESPACE_PT}pt]",
        *body(ablations_df),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\end{{{env}}}",
    ]
    return "\n".join(lines)


def main():
    print("Building ICML table (readable, tight, aligned)...")
    ref_steps, ref_max = get_reference_steps()
    print(f"[reference] steps={len(ref_steps)}, max_step={ref_max}")

    # Column order from HEADER_GROUPS
    cols = ["Method"] + [m for _g, metrics, _l in HEADER_GROUPS for m in metrics]

    base_rows = build_rows(BASELINE_RUNS, BASELINE_NAMES, True, ref_steps, ref_max)
    abl_rows = build_rows(ABLATION_RUNS, ABLATION_NAMES, False, ref_steps, ref_max)

    baselines_df = pd.DataFrame(base_rows, columns=cols)
    ablations_df = pd.DataFrame(abl_rows, columns=cols)

    # Preview (console)
    preview_cols = ["Method"] + [
        DISPLAY_NAMES.get(m, m.split("/")[-1]) for m in cols[1:]
    ]
    print("\n== Baselines (aligned) ==")
    print(
        baselines_df.rename(columns=dict(zip(cols[1:], preview_cols[1:]))).to_string(
            index=False
        )
    )
    print("\n== Ablations ==")
    print(
        ablations_df.rename(columns=dict(zip(cols[1:], preview_cols[1:]))).to_string(
            index=False
        )
    )

    tex = tex_grouped_table(baselines_df, ablations_df, CAPTION, LABEL)
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    out_path = os.path.join(SAVE_FOLDER, f"{FILENAME_PREFIX}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(f"\n[save] {os.path.abspath(out_path)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
