# Paper-ready W&B plotting script with derived state rate metrics
# --------------------------------------------------------------------------------------
# - Calculates percentage of training frames for each skill/state
# - State rate divided by cumulative sum of state rates up to that point
# - X-axis in frames (millions), smoothing, compact ICML styling

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

# Target state for each run (diamond collection state varies by method)
# This specifies the maximum state achieved by each method
run_target_states = [
    11,  # SCALAR reaches state 11 for diamond
    11,  # wo/ Dense Rewards reaches state 8
    10,  # No Trajectory Analysis reaches state 9
    11,  # Shared Networks reaches state 7
]

# State rate metrics to fetch (raw W&B keys)
# We'll fetch up to the maximum state across all runs
MAX_STATE = max(run_target_states)
state_rates = [f"state_rate_{i}" for i in range(MAX_STATE + 1)]

METRIC_DISPLAY_NAME = "Diamond Collection Skill"
Y_AXIS_LABEL = "Frontier Training Efficiency (%)"

# Output
SAVE_FOLDER = "paper_plots"
FILENAME_PREFIX = "state_rate_percentage"

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
FIG_WIDTH = 5.0  # inches for single plot
FIG_HEIGHT = 3.5  # inches height
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

# Legend positioning
LEGEND_BBOX_Y = -0.22  # lower center y for legend
BOTTOM_ADJUST = 0.32  # extra bottom margin

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
            "figure.constrained_layout.use": False,
        }
    )


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


def calculate_state_percentage(
    df: pd.DataFrame, state_rates: List[str], target_state: int
) -> pd.Series:
    """
    Calculate percentage of training frames for a specific state.

    Formula: state_rate_N / sum(state_rate_0 to state_rate_N) * 100

    Args:
        df: DataFrame containing state rate columns
        state_rates: List of state rate column names
        target_state: Which state index to calculate percentage for

    Returns:
        Series with calculated percentages
    """
    target_col = f"state_rate_{target_state}"

    # Get columns for states 0 to target_state
    relevant_states = [f"state_rate_{i}" for i in range(target_state + 1)]

    # Filter to only include columns that exist in the dataframe
    existing_states = [col for col in relevant_states if col in df.columns]

    if target_col not in df.columns:
        print(f"[warn] Target state column {target_col} not found")
        return pd.Series([np.nan] * len(df))

    if not existing_states:
        print(f"[warn] No state rate columns found")
        return pd.Series([np.nan] * len(df))

    # Calculate sum of state rates from 0 to target_state
    cumulative_sum = df[existing_states].sum(axis=1)

    # Avoid division by zero
    cumulative_sum = cumulative_sum.replace(0, np.nan)

    # Calculate percentage
    percentage = (df[target_col] / cumulative_sum) * 100

    return percentage


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


def plot_state_percentage(
    runs: List[str],
    run_names: List[str],
    run_target_states: List[int],
    state_rates: List[str],
    metric_display_name: str,
    y_axis_label: str,
    save_folder: str = SAVE_FOLDER,
    filename_prefix: str = FILENAME_PREFIX,
):
    """Create plot showing percentage of training frames for target state (varies per run)."""
    if len(runs) != len(run_names) or len(runs) != len(run_target_states):
        raise ValueError(
            "Length of 'runs', 'run_names', and 'run_target_states' must match."
        )

    os.makedirs(save_folder, exist_ok=True)
    set_paper_style()

    # Fetch per-run data
    run_data: Dict[str, Tuple[pd.DataFrame, int]] = {}

    print("\n=== FETCHING DATA ===")
    for rpath, rname, target_state in zip(runs, run_names, run_target_states):
        df, _ = fetch_run_data(rpath, state_rates)
        if df is None:
            print(f"[warn] Could not fetch data for: {rname} ({rpath})")
            continue
        df = downsample_df(df, DOWNSAMPLE_EVERY)

        # Calculate the derived metric for this run's specific target state
        df["state_percentage"] = calculate_state_percentage(
            df, state_rates, target_state
        )

        run_data[rname] = (df, target_state)
        print(f"Fetched and processed: {rname} (target state: {target_state})")

    if not run_data:
        print("[error] No data available across runs. Exiting.")
        return

    # Shared palette/linestyles across runs
    palette = make_palette(len(run_data))
    color_map = {name: palette[i] for i, name in enumerate(run_data.keys())}
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    linestyle_map = {
        name: linestyles[i % len(linestyles)] for i, name in enumerate(run_data.keys())
    }

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    for rname, (df, target_state) in run_data.items():
        if "state_percentage" not in df.columns:
            continue

        # Convert steps -> frames
        x_frames = df[X_AXIS].to_numpy().astype(float) * FRAMES_PER_STEP
        y_raw = df["state_percentage"].to_numpy().astype(float)

        # Remove NaN values
        valid_mask = ~np.isnan(y_raw)
        x_frames = x_frames[valid_mask]
        y_raw = y_raw[valid_mask]

        if len(y_raw) == 0:
            print(f"[warn] No valid data for {rname}")
            continue

        y_smooth = smooth_series(pd.Series(y_raw), SMOOTH_WINDOW).to_numpy()

        # Update label to show target state, no don't actually
        label_with_state = f"{rname}"

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
            label=label_with_state,
            linewidth=SMOOTH_LINE_WIDTH,
            linestyle=linestyle_map[rname],
            color=color_map[rname],
        )

    ax.set_title(metric_display_name, pad=6)
    ax.set_ylabel(y_axis_label)
    set_frames_axis(ax)

    # Set y-axis to percentage scale
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))

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
                bbox_to_anchor=(0.5, LEGEND_BBOX_Y),
                ncol=min(LEGEND_NCOL, len(labels)),
                frameon=False,
            )
            plt.subplots_adjust(bottom=BOTTOM_ADJUST)
    else:
        ax.legend(loc="best", frameon=False)

    # Save figure
    safe_name = safe_filename(f"{filename_prefix}_variable_states")
    for fmt in SAVE_FORMATS:
        path = os.path.join(save_folder, f"{safe_name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", transparent=True)
        print(f"[save] {path}")
    plt.close(fig)


def main():
    print("Fetching W&B run data and creating state percentage plot...")
    print(f"Runs ({len(runs)}): {runs}")
    print(f"Run names: {run_names}")
    print(f"Target states per run: {run_target_states}")
    print(f"State rates to fetch: state_rate_0 through state_rate_{MAX_STATE}")
    print("-" * 60)

    print("\n=== INSPECTING AVAILABLE METRICS PER RUN ===")
    per_run_metrics: Dict[str, List[str]] = {}
    for i, r in enumerate(runs):
        rname, mlist = inspect_run_metrics(r)
        per_run_metrics[r] = mlist
        print(f"• {r} ({rname}) -> {len(mlist)} metrics")
        print(f"  Target state for this run: {run_target_states[i]}")

        # Check which state rates are available
        available_states = [m for m in mlist if m.startswith("state_rate_")]
        if available_states:
            print(
                f"  Available state rates: {', '.join(sorted(available_states)[:5])}..."
            )

    plot_state_percentage(
        runs,
        run_names,
        run_target_states,
        state_rates,
        METRIC_DISPLAY_NAME,
        Y_AXIS_LABEL,
        save_folder=SAVE_FOLDER,
        filename_prefix=FILENAME_PREFIX,
    )
    print(f"\nDone. Figure saved in: {os.path.abspath(SAVE_FOLDER)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
