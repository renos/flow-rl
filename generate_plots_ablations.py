# Paper-ready W&B plotting script (ICML-compact, frames axis + deeper legend + smoothing)
# --------------------------------------------------------------------------------------
# - Legend moved further down; larger bottom padding so it never touches x-axis
# - X-axis converted from steps -> frames via 64*1024 multiplier
# - X-axis labeled in millions: ticks at 0, 200, 400, ..., 1000 (i.e., up to 1e9 frames)
# - Raw curve shown faintly; smoothed (moving average) curve shown on top
# - Compact typography, custom metric display names, vector (PDF) + high-DPI PNG exports

import os
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional

import wandb
from wandb.apis.public import Api
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- CONFIG -----------------------------

# W&B runs and labels (ordering defines legend order)
runs = [
    "interpretable_rl/flow-rl/abapycf4",
    "interpretable_rl/flow-rl/n9uu2se0",
    "interpretable_rl/flow-rl/1lvvnyt9",
    "interpretable_rl/flow-rl/8sio2ezt",
]
run_names = [
    "SCALAR-Dense (Our Method)",
    "SCALAR-Sparse (Our Method)",
    "No Trajectory Analysis",
    "Shared Networks",
]

# Raw W&B metric keys to plot
metrics = [
    "Achievements/collect_diamond",
    "episode_length",
]

# Custom display names for metrics
METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "episode_length": "Average Episode Length",
    "Achievements/collect_diamond": "Diamond Collected (%)",
}

# Output
SAVE_FOLDER = "paper_plots"
FILENAME_PREFIX = "ablations_plot"

# ---- ICML-compact typography preset ----
USE_TEX = False  # Set True if you have LaTeX available
FONT_FAMILY = "DejaVu Sans"
CONTEXT = "paper"  # seaborn contexts: "paper", "notebook", "talk", "poster"
FONT_SCALE = 1.05  # global multiplier (kept modest)
TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 10
LEGEND_FONTSIZE = 9
TITLE_WEIGHT = "semibold"
LABEL_WEIGHT = "medium"

# Figure sizing
FIG_WIDTH_PER_SUBPLOT = 4.2  # inches per subplot (compact)
FIG_HEIGHT = 3.2  # inches height
FIG_DPI = 600
SAVE_FORMATS = ["pdf", "png"]  # add "svg" if needed

# Plot styling
PALETTE_NAME = "colorblind"
RAW_LINE_ALPHA = 0.25  # faint raw line underneath
RAW_LINE_WIDTH = 1.2
SMOOTH_LINE_WIDTH = 2.4
MARKERS = False
MARKER_SIZE = 3.5
GRID_ALPHA = 0.25
DESPINE = True
LEGEND_OUTSIDE = True
LEGEND_NCOL = 2

# Legend positioning (moved even further down to avoid axis overlap)
# LEGEND_COMBINED_BBOX_Y = -0.18  # lower center y for combined figure
# BOTTOM_ADJUST_COMBINED = 0.36  # extra bottom margin for combined figure
# LEGEND_INDIVIDUAL_BBOX_Y = -0.30  # lower center y for individual plots
# BOTTOM_ADJUST_INDIVIDUAL = 0.42  # extra bottom margin for individual plots

LEGEND_COMBINED_BBOX_Y = -0.10  # lower center y for combined figure
BOTTOM_ADJUST_COMBINED = 0.20  # extra bottom margin for combined figure
LEGEND_INDIVIDUAL_BBOX_Y = -0.22  # lower center y for individual plots
BOTTOM_ADJUST_INDIVIDUAL = 0.32  # extra bottom margin for individual plots

# Frames axis conversion
FRAMES_PER_STEP = 64 * 1024  # multiply steps by this
FRAMES_MAX = 1_000_000_000  # show axis from 0 to 1e9 frames
XTICKS_M_INTERVAL = 200  # tick every 200M frames
X_LABEL = "Frames (M)"  # shown as millions on the axis

# Data handling
SMOOTH_WINDOW = 31  # moving-average window (points); 1 = no smoothing
DOWNSAMPLE_EVERY = 1  # 1 = no downsampling
X_AXIS = "_step"
MAX_HISTORY_SAMPLES = None
RETRY_ON_ERROR = True
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# --------------------------- END CONFIG ---------------------------

# Initialize W&B API
api = Api()


def set_paper_style():
    """Configure seaborn/matplotlib for compact, publication-quality output."""
    sns.set_theme(
        context=CONTEXT, style="whitegrid", font="DejaVu Sans", font_scale=FONT_SCALE
    )
    if USE_TEX:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
            }
        )
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.titlesize": TITLE_SIZE,
            "axes.titleweight": TITLE_WEIGHT,
            "axes.labelsize": LABEL_SIZE,
            "axes.labelweight": LABEL_WEIGHT,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "axes.linewidth": 1.0,
            "grid.alpha": GRID_ALPHA,
            "axes.grid": True,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Use manual spacing (we'll control bottom margins), so keep constrained_layout off
            "figure.constrained_layout.use": False,
        }
    )


def pretty_metric_name(raw_key: str, custom_map: Dict[str, str]) -> str:
    """Map a raw key to a nice display name with custom overrides and smart fallback."""
    if raw_key in custom_map:
        return custom_map[raw_key]
    name = raw_key.split("/")[-1]
    name = name.replace("_", " ").replace(".", " ")
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    name = name.title()
    name = (
        name.replace("Rl", "RL")
        .replace("Rnn", "RNN")
        .replace("Fc", "FC")
        .replace("Ppo", "PPO")
        .replace("Icm", "ICM")
        .replace("Rnd", "RND")
    )
    return name


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-_]+", "_", s)


def _robust_run_call(callable_fn, *args, **kwargs):
    """Retry wrapper for W&B API calls."""
    if not RETRY_ON_ERROR:
        return callable_fn(*args, **kwargs)
    attempt = 0
    while True:
        try:
            return callable_fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt >= MAX_RETRIES:
                raise e
            time.sleep(RETRY_SLEEP)


def inspect_run_metrics(run_path: str) -> Tuple[str, List[str]]:
    """Inspect available metric keys for a run. Returns (run.name, sorted metric keys)."""
    try:
        run = _robust_run_call(api.run, run_path)
        run_name = getattr(run, "name", os.path.basename(run_path))
        metric_names = []
        try:
            hk = getattr(run, "history_keys", None) or {}
            metric_names = list((hk.get("keys", {}) or {}).keys())
        except Exception:
            metric_names = []
        if not metric_names:
            sample_history = _robust_run_call(
                run.history,
                samples=(
                    200
                    if MAX_HISTORY_SAMPLES is None
                    else min(200, MAX_HISTORY_SAMPLES)
                ),
                pandas=True,
            )
            metric_names = [
                c
                for c in list(sample_history.columns)
                if c not in (X_AXIS, "_runtime", "_timestamp")
            ]
        return run_name, sorted(set(metric_names))
    except Exception as e:
        print(f"[inspect] Error for {run_path}: {e}")
        return os.path.basename(run_path), []


def fetch_run_data(
    run_path: str, metric_keys: List[str]
) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
    """Fetch history DataFrame for specified metrics (+ X_AXIS)."""
    try:
        run = _robust_run_call(api.run, run_path)
        _, available = inspect_run_metrics(run_path)
        available_set = set(available)
        existing_metrics = [k for k in metric_keys if k in available_set]

        missing = [k for k in metric_keys if k not in available_set]
        if missing:
            print(f"[warn] {run_path} missing metrics: {missing}")

        if not existing_metrics:
            print(f"[warn] No requested metrics present for {run_path}")
            return None, None

        keys_to_fetch = sorted(set(existing_metrics + [X_AXIS]))
        if MAX_HISTORY_SAMPLES is not None:
            history_df = _robust_run_call(
                run.history,
                keys=keys_to_fetch,
                samples=MAX_HISTORY_SAMPLES,
                pandas=True,
            )
        else:
            history_df = _robust_run_call(run.history, keys=keys_to_fetch, pandas=True)

        if X_AXIS not in history_df.columns:
            history_df[X_AXIS] = np.arange(len(history_df))
        history_df = history_df.dropna(how="all", subset=existing_metrics)
        history_df = history_df.sort_values(by=X_AXIS).reset_index(drop=True)
        return history_df, run
    except Exception as e:
        print(f"[fetch] Error for {run_path}: {e}")
        return None, None


def smooth_series(y: pd.Series, window: int) -> pd.Series:
    """Centered moving-average smoothing."""
    if window is None or window <= 1:
        return y
    return y.rolling(
        window=int(window), min_periods=max(1, window // 2), center=True
    ).mean()


def downsample_df(df: pd.DataFrame, every: int) -> pd.DataFrame:
    if every is None or every <= 1:
        return df
    return df.iloc[::every, :].reset_index(drop=True)


def make_palette(n: int):
    return sns.color_palette(PALETTE_NAME, n_colors=n)


def set_frames_axis(ax):
    """Configure x-axis to show frames in millions from 0 to FRAMES_MAX."""
    ax.set_xlim(0, FRAMES_MAX)
    ticks = np.arange(0, FRAMES_MAX + 0.1, XTICKS_M_INTERVAL * 1_000_000)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t/1_000_000):d}" for t in ticks])
    ax.set_xlabel(X_LABEL)


def plot_metrics(
    runs: List[str],
    run_names: List[str],
    metrics: List[str],
    metric_display_names: Dict[str, str],
    save_folder: str = SAVE_FOLDER,
    filename_prefix: str = FILENAME_PREFIX,
):
    """Create combined + per-metric figures with compact, publication-quality styling."""
    if len(runs) != len(run_names):
        raise ValueError("Length of 'runs' must match length of 'run_names'.")

    os.makedirs(save_folder, exist_ok=True)
    set_paper_style()

    # Fetch per-run data
    run_data: Dict[str, pd.DataFrame] = {}
    all_available_metrics = set()

    print("\n=== FETCHING DATA ===")
    for rpath, rname in zip(runs, run_names):
        df, _ = fetch_run_data(rpath, metrics)
        if df is None:
            print(f"[warn] Could not fetch data for: {rname} ({rpath})")
            continue
        df = downsample_df(df, DOWNSAMPLE_EVERY)
        run_data[rname] = df
        all_available_metrics.update(df.columns.tolist())

    available_metrics = [m for m in metrics if m in all_available_metrics]
    if not available_metrics:
        print("[error] No requested metrics available across runs. Exiting.")
        return

    print(f"Will plot metrics: {available_metrics}")

    # Shared palette/linestyles across runs
    palette = make_palette(len(run_data))
    color_map = {name: palette[i] for i, name in enumerate(run_data.keys())}
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_map = {
        name: linestyles[i % len(linestyles)] for i, name in enumerate(run_data.keys())
    }

    # ----------------- Combined figure -----------------
    n_metrics = len(available_metrics)
    fig_w = max(3.6, n_metrics * FIG_WIDTH_PER_SUBPLOT)
    fig_h = FIG_HEIGHT

    fig, axes = plt.subplots(1, n_metrics, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        disp_name = pretty_metric_name(metric, metric_display_names)

        for rname, df in run_data.items():
            if metric not in df.columns:
                continue
            # Convert steps -> frames
            x_frames = df[X_AXIS].to_numpy().astype(float) * FRAMES_PER_STEP
            y_raw = df[metric].to_numpy().astype(float)
            y_smooth = smooth_series(pd.Series(y_raw), SMOOTH_WINDOW).to_numpy()

            # Raw (faint, no legend)
            if MARKERS:
                ax.plot(
                    x_frames,
                    y_raw,
                    label="_nolegend_",
                    linewidth=RAW_LINE_WIDTH,
                    linestyle=linestyle_map[rname],
                    marker="o",
                    markersize=MARKER_SIZE,
                    color=color_map[rname],
                    alpha=RAW_LINE_ALPHA,
                )
            else:
                ax.plot(
                    x_frames,
                    y_raw,
                    label="_nolegend_",
                    linewidth=RAW_LINE_WIDTH,
                    linestyle=linestyle_map[rname],
                    color=color_map[rname],
                    alpha=RAW_LINE_ALPHA,
                )

            # Smoothed (legend)
            ax.plot(
                x_frames,
                y_smooth,
                label=rname,
                linewidth=SMOOTH_LINE_WIDTH,
                linestyle=linestyle_map[rname],
                color=color_map[rname],
            )

        ax.set_title(disp_name, pad=6)
        # Left-most subplot carries the y-label
        if idx == 0:
            ax.set_ylabel(disp_name)
        else:
            ax.set_ylabel("")

        set_frames_axis(ax)
        if DESPINE:
            sns.despine(ax=ax)
        ax.margins(x=0)

    # Legend (further down; more bottom padding)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles and labels:
        if LEGEND_OUTSIDE:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(LEGEND_NCOL, len(labels)),
                frameon=False,
                bbox_to_anchor=(0.5, LEGEND_COMBINED_BBOX_Y),
            )
            plt.subplots_adjust(bottom=BOTTOM_ADJUST_COMBINED)
        else:
            axes[0].legend(loc="best", frameon=False)

    # Save combined figure
    base_name = f"{filename_prefix}_combined_{n_metrics}x"
    for fmt in SAVE_FORMATS:
        path = os.path.join(save_folder, f"{base_name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", transparent=True)
        print(f"[save] {path}")
    plt.close(fig)

    # --------------- Individual figures per metric ---------------
    for metric in available_metrics:
        disp_name = pretty_metric_name(metric, metric_display_names)
        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_PER_SUBPLOT, FIG_HEIGHT))

        for rname, df in run_data.items():
            if metric not in df.columns:
                continue
            x_frames = df[X_AXIS].to_numpy().astype(float) * FRAMES_PER_STEP
            y_raw = df[metric].to_numpy().astype(float)
            y_smooth = smooth_series(pd.Series(y_raw), SMOOTH_WINDOW).to_numpy()

            # Raw (faint)
            if MARKERS:
                ax.plot(
                    x_frames,
                    y_raw,
                    label="_nolegend_",
                    linewidth=RAW_LINE_WIDTH,
                    linestyle=linestyle_map[rname],
                    marker="o",
                    markersize=MARKER_SIZE,
                    color=color_map[rname],
                    alpha=RAW_LINE_ALPHA,
                )
            else:
                ax.plot(
                    x_frames,
                    y_raw,
                    label="_nolegend_",
                    linewidth=RAW_LINE_WIDTH,
                    linestyle=linestyle_map[rname],
                    color=color_map[rname],
                    alpha=RAW_LINE_ALPHA,
                )

            # Smoothed (legend)
            ax.plot(
                x_frames,
                y_smooth,
                label=rname,
                linewidth=SMOOTH_LINE_WIDTH,
                linestyle=linestyle_map[rname],
                color=color_map[rname],
            )

        ax.set_title(disp_name, pad=6)
        ax.set_ylabel(disp_name)
        set_frames_axis(ax)
        if DESPINE:
            sns.despine(ax=ax)
        ax.margins(x=0)

        if LEGEND_OUTSIDE:
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:
                ax.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, LEGEND_INDIVIDUAL_BBOX_Y),
                    ncol=min(LEGEND_NCOL, len(labels)),
                    frameon=False,
                )
                plt.subplots_adjust(bottom=BOTTOM_ADJUST_INDIVIDUAL)
        else:
            ax.legend(loc="best", frameon=False)

        safe_metric = re.sub(r"[^\w\-_]+", "_", metric)
        base_name = f"{filename_prefix}_{safe_metric}"
        for fmt in SAVE_FORMATS:
            path = os.path.join(save_folder, f"{base_name}.{fmt}")
            fig.savefig(path, bbox_inches="tight", transparent=True)
            print(f"[save] {path}")
        plt.close(fig)


def print_metric_catalog(per_run_metrics: Dict[str, List[str]]):
    all_metrics = (
        sorted(set().union(*per_run_metrics.values())) if per_run_metrics else []
    )
    print("\n=== ALL UNIQUE METRICS ACROSS RUNS ===")
    for i, m in enumerate(all_metrics, start=1):
        print(f"{i:3d}. {m}")


def main():
    print("Fetching W&B run data and creating compact publication-ready plots...")
    print(f"Runs ({len(runs)}): {runs}")
    print(f"Run names: {run_names}")
    print(f"Metrics requested: {metrics}")
    print("-" * 60)

    print("\n=== INSPECTING AVAILABLE METRICS PER RUN ===")
    per_run_metrics: Dict[str, List[str]] = {}
    for r in runs:
        rname, mlist = inspect_run_metrics(r)
        per_run_metrics[r] = mlist
        print(f"• {r} ({rname}) -> {len(mlist)} metrics")

    all_available = set().union(*per_run_metrics.values()) if per_run_metrics else set()
    print("\n=== CHECKING REQUESTED METRICS ===")
    existing = []
    for m in metrics:
        if m in all_available:
            print(
                f"✓ Found:    {m}  ->  '{pretty_metric_name(m, METRIC_DISPLAY_NAMES)}'"
            )
            existing.append(m)
        else:
            print(f"✗ Missing:  {m}")
            tokens = [t for t in re.split(r"[/_\s]+", m.lower()) if t]
            similar = sorted(
                {k for k in all_available if any(tok in k.lower() for tok in tokens)}
            )
            if similar:
                print(f"   Similar: {similar[:12]}")

    if not existing:
        print("\nNo requested metrics were found across the provided runs.")
        print("Please revise the metric names.")
        return

    plot_metrics(
        runs,
        run_names,
        existing,
        METRIC_DISPLAY_NAMES,
        save_folder=SAVE_FOLDER,
        filename_prefix=FILENAME_PREFIX,
    )
    print("\nDone. Figures saved in:", os.path.abspath(SAVE_FOLDER))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
