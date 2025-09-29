#!/usr/bin/env python3
"""
Generate survival distribution histograms from local evaluation results.
Shows distributions of Energy, Food, Drink restored per episode + Episode Length.
"""

import os
import json
import glob
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- CONFIG -----------------------------

# === Method directories to read from (same as generate_table_from_local.py) ===
METHOD_DIRS = {
    "SCALAR-Dense (Our Method)": "/home/renos/flow-rl/exp_results/scalar_dense",
    "SCALAR-Sparse (Our Method)": "/home/renos/flow-rl/exp_results/scalar_sparse",
    "No Trajectory Analysis": "/home/renos/flow-rl/exp_results/scalar_no_traj_analysis",
    "Shared Networks": "/home/renos/flow-rl/exp_results/scalar_common_network",
    "PPO-RNN": "/home/renos/flow-rl/exp_results/ppo_rnn",
    "PPO-FC": "/home/renos/flow-rl/exp_results/ppo_mlp",
    "PPO-RND": "/home/renos/flow-rl/exp_results/ppo_rnd",
    "ICM": "/home/renos/flow-rl/exp_results/icm",
    "E3B": "/home/renos/flow-rl/exp_results/e3b",
}

# === Baselines (top) ===
BASELINE_METHODS = ["PPO-RNN", "PPO-FC", "PPO-RND", "ICM", "E3B"]

# === Ablations (bottom) ===
ABLATION_METHODS = [
    "SCALAR-Dense (Our Method)",
    "SCALAR-Sparse (Our Method)",
    "No Trajectory Analysis",
    "Shared Networks",
]

# Metrics to plot as histograms
METRICS_TO_PLOT = [
    "mean_energy_restored",
    "mean_food_restored", 
    "mean_drink_restored",
    "mean_episode_length"
]

METRIC_LABELS = {
    "mean_energy_restored": "Energy Restored per Episode",
    "mean_food_restored": "Food Restored per Episode",
    "mean_drink_restored": "Drink Restored per Episode", 
    "mean_episode_length": "Episode Length"
}

# Mapping from metric names to raw data keys in JSON files
METRIC_RAW_KEYS = {
    "mean_energy_restored": "all_energy_restored",
    "mean_food_restored": "all_food_restored", 
    "mean_drink_restored": "all_drink_restored",
    "mean_episode_length": "all_lengths"
}

# Output
SAVE_FOLDER = "paper_plots"
FILENAME_PREFIX = "survival_distributions"

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
FIG_WIDTH = 12  # inches width for 2x2 grid
FIG_HEIGHT = 8  # inches height for 2x2 grid
FIG_DPI = 600
SAVE_FORMATS = ["pdf", "png"]  # add "svg" if needed

# Plot styling
PALETTE_NAME = "colorblind"
HIST_ALPHA = 0.6
HIST_BINS = 30
GRID_ALPHA = 0.25
DESPINE = True

# --------------------------- END CONFIG ---------------------------


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


def make_palette(n: int):
    return sns.color_palette(PALETTE_NAME, n_colors=n)


def load_latest_result(method_dir: str) -> Dict:
    """Load the most recent result file from a method directory."""
    if not os.path.exists(method_dir):
        print(f"[warn] Directory not found: {method_dir}")
        return None

    # Find all JSON files in the directory
    json_files = glob.glob(os.path.join(method_dir, "*.json"))
    if not json_files:
        print(f"[warn] No JSON files found in: {method_dir}")
        return None

    # Get the most recent file (by modification time)
    latest_file = max(json_files, key=os.path.getmtime)

    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
        print(f"[info] Loaded: {latest_file}")
        return data
    except Exception as e:
        print(f"[error] Failed to load {latest_file}: {e}")
        return None


def plot_survival_distributions(
    method_dirs: Dict[str, str] = METHOD_DIRS,
    baseline_methods: List[str] = BASELINE_METHODS,
    ablation_methods: List[str] = ABLATION_METHODS,
    save_folder: str = SAVE_FOLDER,
    filename_prefix: str = FILENAME_PREFIX,
):
    """Plot 4 histograms showing distributions of survival metrics and episode length."""
    
    # Set paper style
    set_paper_style()
    
    # Load data from local results
    all_methods = baseline_methods + ablation_methods
    method_data = {}
    
    print("\n=== LOADING LOCAL DATA FOR HISTOGRAMS ===")
    for method_name in all_methods:
        if method_name not in method_dirs:
            print(f"[warn] Method '{method_name}' not found in method_dirs")
            continue
            
        method_dir = method_dirs[method_name]
        result = load_latest_result(method_dir)
        
        if result is not None:
            method_data[method_name] = result
    
    if not method_data:
        print("[error] No data loaded. Cannot generate histograms.")
        return
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten()
    
    # Color palette for methods
    palette = make_palette(len(method_data))
    color_map = {name: palette[i] for i, name in enumerate(method_data.keys())}
    
    # Plot each metric
    for idx, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[idx]
        
        # Get the corresponding raw data key
        raw_key = METRIC_RAW_KEYS[metric]
        
        # Find the 90th percentile across all methods for this metric to set axis limits
        all_values_for_metric = []
        for method_name, data in method_data.items():
            if raw_key in data and data[raw_key]:
                values = np.array(data[raw_key])
                all_values_for_metric.extend(values)
        
        if all_values_for_metric:
            percentile_90 = np.percentile(all_values_for_metric, 90)
        else:
            percentile_90 = 100  # fallback
        
        # Plot smoothed histogram (KDE) for each method using only top 10% of trajectories
        for method_name, data in method_data.items():
            if raw_key in data and data[raw_key]:
                values = np.array(data[raw_key])
                
                if len(values) > 0:
                    # Plot KDE (smoothed density) - no log scale
                    sns.kdeplot(
                        data=values,
                        ax=ax,
                        label=method_name,
                        color=color_map[method_name],
                        alpha=0.8,
                        linewidth=2.5
                    )
                    
                    # Add density for values above 90th percentile as a point at the end
                    above_90th = values[values > percentile_90]
                    if len(above_90th) > 0:
                        density_above_90th = len(above_90th) / len(values)
                        # Plot this density as a point at percentile_90 + some offset
                        ax.scatter(
                            [percentile_90 * 1.1], 
                            [density_above_90th], 
                            color=color_map[method_name], 
                            s=50, 
                            alpha=0.8,
                            marker='o'
                        )
        
        ax.set_xlabel(METRIC_LABELS[metric])
        ax.set_ylabel("Density")
        ax.set_title(METRIC_LABELS[metric], pad=6)
        
        # Set x-axis to show up to 90th percentile + some buffer, with final point labeled
        ax.set_xlim(0, percentile_90 * 1.2)
        
        # Add custom tick at the 90th percentile position
        current_ticks = list(ax.get_xticks())
        # Filter out ticks that are beyond our x-axis limit
        valid_ticks = [tick for tick in current_ticks if tick <= percentile_90 * 1.2]
        # Add the custom tick for the 90th percentile
        valid_ticks.append(percentile_90 * 1.1)
        
        # Create matching labels
        labels = [f"{int(x)}" for x in valid_ticks[:-1]]  # All but the last
        labels.append(f"{int(percentile_90)}+")  # Custom label for last tick
        
        ax.set_xticks(valid_ticks)
        ax.set_xticklabels(labels)
        
        if DESPINE:
            sns.despine(ax=ax)
        
        # Add grid
        ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
        
        # Only show legend on first subplot to avoid clutter
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE-1, frameon=False)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_folder, exist_ok=True)
    base_name = f"{filename_prefix}_histograms"
    for fmt in SAVE_FORMATS:
        path = os.path.join(save_folder, f"{base_name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", transparent=True, dpi=FIG_DPI)
        print(f"[save] {path}")
    
    plt.close(fig)
    print(f"\nHistogram plots saved in: {os.path.abspath(save_folder)}")


def main():
    """Main function to generate the histograms."""
    print("Generating survival distribution histograms from local evaluation results...")
    print(f"Looking for methods in: {list(METHOD_DIRS.keys())}")
    
    plot_survival_distributions()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()