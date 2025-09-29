#!/usr/bin/env python3
"""
Generate table from local evaluation results instead of wandb.
Reads JSON files from exp_results/ directory structure.
"""

import os
import json
import glob
import warnings
from typing import Dict, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------- CONFIG -----------------------------

# === Method directories to read from ===
METHOD_DIRS = {
    "SCALAR-Dense (Our Method)": "/home/renos/flow-rl/exp_results/scalar_dense",
    "SCALAR-Sparse (Our Method)": "/home/renos/flow-rl/exp_results/scalar_sparse",
    "No Trajectory Analysis": "/home/renos/flow-rl/exp_results/scalar_no_traj_analysis",
    "Shared Networks": "/home/renos/flow-rl/exp_results/scalar_common_network",
    # Add more methods here as you run evaluations:
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

REFERENCE_METHOD_NAME = "SCALAR (Our Method)"  # for step alignment

# --- Columns (we'll render them as grouped headers with short labels) ---
# Map from achievement indices to names (based on Craftax achievements)
ACHIEVEMENT_NAMES = {
    0: "collect_wood",
    1: "place_table",
    2: "eat_cow",
    3: "collect_sapling",
    4: "collect_drink",
    5: "make_wood_pickaxe",
    6: "make_wood_sword",
    7: "place_plant",
    8: "defeat_zombie",
    9: "collect_stone",
    10: "place_stone",
    11: "eat_plant",
    12: "defeat_skeleton",
    13: "make_stone_pickaxe",
    14: "make_stone_sword",
    15: "wake_up",
    16: "place_furnace",
    17: "collect_coal",
    18: "collect_iron",
    19: "collect_diamond",
    20: "make_iron_pickaxe",
    21: "make_iron_sword",
}

# Metrics we want to show in the table
METRICS = [
    "achievement_1_rate",  # place_table
    "achievement_16_rate",  # place_furnace
    "achievement_5_rate",  # make_wood_pickaxe
    "achievement_13_rate",  # make_stone_pickaxe
    "achievement_20_rate",  # make_iron_pickaxe
    "achievement_19_rate",  # collect_diamond
    "mean_energy_restored",  # energy restored per episode
    "mean_food_restored",  # food restored per episode
    "mean_drink_restored",  # drink restored per episode
    "mean_episode_length",
]

# Display names for preview
DISPLAY_NAMES: Dict[str, str] = {
    "achievement_19_rate": "Diamond (\\%)",
    "achievement_1_rate": "Table",
    "achievement_16_rate": "Furnace",
    "achievement_5_rate": "Wood",
    "achievement_13_rate": "Stone",
    "achievement_20_rate": "Iron",
    "mean_energy_restored": "Energy",
    "mean_food_restored": "Food",
    "mean_drink_restored": "Drink",
    "mean_episode_length": "Ep. Len.",
}

# Header grouping for the TeX table
HEADER_GROUPS = [
    (
        "Setup (\\%)",
        [
            "achievement_1_rate",  # place_table
            "achievement_16_rate",  # place_furnace
        ],
        ["Table", "Furnace"],
    ),
    (
        "Pickaxes (\\%)",
        [
            "achievement_5_rate",  # make_wood_pickaxe
            "achievement_13_rate",  # make_stone_pickaxe
            "achievement_20_rate",  # make_iron_pickaxe
        ],
        ["Wood", "Stone", "Iron"],
    ),
    (
        "Goal (\\%)",
        [
            "achievement_19_rate",  # collect_diamond
        ],
        ["Diamond"],
    ),
    (
        "Survival",
        [
            "mean_energy_restored",  # energy restored per episode
            "mean_food_restored",  # food restored per episode
            "mean_drink_restored",  # drink restored per episode
        ],
        ["Energy", "Food", "Drink"],
    ),
    (
        "Episodes",
        [
            "mean_episode_length",
        ],
        ["Ep. Len."],
    ),
]

# Output
SAVE_FOLDER = "paper_tables"
FILENAME_PREFIX = "local_results_table_icml_grouped"
CAPTION = (
    "Evaluation results from local experiments. "
    "Achievement rates reported as percentages; episode lengths as raw values."
)
LABEL = "tab:local-results-icml-grouped"

# LaTeX styling
TABLE_ENV_STAR = True  # True → table* (two-column width), False → table
TABLE_FONTSIZE = r"\small"  # \normalsize | \small
TABCOLSEP_PT = 3  # tighter columns
ARRAYSTRETCH = 1.05  # slightly tighter row height
ADDLINESPACE_PT = 1  # minimal extra space before group rows

# --------------------------- END CONFIG ---------------------------


def latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
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


def load_latest_result(method_dir: str) -> Optional[Dict]:
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


def extract_metrics_from_result(result: Dict) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Extract the metrics we need from a result dictionary.
    Returns:
        metrics: Dictionary of mean values
        raw_data: Dictionary of raw data arrays for statistical testing
    """
    metrics = {}
    raw_data = {}

    # Extract achievement rates (convert to percentages)
    if "achievement_completion_rates" in result:
        rates = result["achievement_completion_rates"]
        for i, rate in enumerate(rates):
            metrics[f"achievement_{i}_rate"] = (
                float(rate) * 100.0
            )  # Convert to percentage
            
    # Try to get raw achievement data for statistical testing
    if "all_final_achievements" in result:
        all_achievements = np.array(result["all_final_achievements"])
        for i in range(all_achievements.shape[1] if len(all_achievements.shape) > 1 else 0):
            achievement_values = all_achievements[:, i] * 100.0  # Convert to percentage
            raw_data[f"achievement_{i}_rate"] = achievement_values.tolist()

    # Extract episode length
    if "mean_episode_length" in result:
        metrics["mean_episode_length"] = float(result["mean_episode_length"])
        if "all_lengths" in result:
            raw_data["mean_episode_length"] = result["all_lengths"]

    # Extract restoration statistics if available
    if "mean_energy_restored" in result:
        metrics["mean_energy_restored"] = float(result["mean_energy_restored"])
        if "all_energy_restored" in result:
            raw_data["mean_energy_restored"] = result["all_energy_restored"]
            
    if "mean_food_restored" in result:
        metrics["mean_food_restored"] = float(result["mean_food_restored"])
        if "all_food_restored" in result:
            raw_data["mean_food_restored"] = result["all_food_restored"]
            
    if "mean_drink_restored" in result:
        metrics["mean_drink_restored"] = float(result["mean_drink_restored"])
        if "all_drink_restored" in result:
            raw_data["mean_drink_restored"] = result["all_drink_restored"]

    return metrics, raw_data


def extract_conditional_survival_metrics(result: Dict) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """Extract survival metrics conditioned on episode success (diamond collected).
    Returns:
        metrics: Dictionary of conditional mean values
        raw_data: Dictionary of conditional raw data arrays for statistical testing
    """
    metrics = {}
    raw_data = {}
    
    # Define success as collecting diamond (achievement 19)
    if ("all_final_achievements" in result and 
        "all_energy_restored" in result and
        "all_food_restored" in result and 
        "all_drink_restored" in result and
        "all_lengths" in result):
        
        all_achievements = np.array(result["all_final_achievements"])
        all_energy = np.array(result["all_energy_restored"])
        all_food = np.array(result["all_food_restored"]) 
        all_drink = np.array(result["all_drink_restored"])
        all_lengths = np.array(result["all_lengths"])
        
        # Success mask: episodes where diamond was collected (achievement 19)
        if all_achievements.shape[1] > 19:
            success_mask = all_achievements[:, 19] == 1  # Diamond collected
            failure_mask = ~success_mask
            
            # Calculate conditional metrics
            if np.sum(success_mask) > 0:
                metrics["energy_success"] = float(np.mean(all_energy[success_mask]))
                metrics["food_success"] = float(np.mean(all_food[success_mask]))
                metrics["drink_success"] = float(np.mean(all_drink[success_mask]))
                metrics["length_success"] = float(np.mean(all_lengths[success_mask]))
                
                raw_data["energy_success"] = all_energy[success_mask].tolist()
                raw_data["food_success"] = all_food[success_mask].tolist()
                raw_data["drink_success"] = all_drink[success_mask].tolist()
                raw_data["length_success"] = all_lengths[success_mask].tolist()
            else:
                # No successful episodes - mark as None for "—" display
                for metric in ["energy_success", "food_success", "drink_success", "length_success"]:
                    metrics[metric] = None
                    raw_data[metric] = []
            
            if np.sum(failure_mask) > 0:
                metrics["energy_failure"] = float(np.mean(all_energy[failure_mask]))
                metrics["food_failure"] = float(np.mean(all_food[failure_mask]))
                metrics["drink_failure"] = float(np.mean(all_drink[failure_mask]))
                metrics["length_failure"] = float(np.mean(all_lengths[failure_mask]))
                
                raw_data["energy_failure"] = all_energy[failure_mask].tolist()
                raw_data["food_failure"] = all_food[failure_mask].tolist()
                raw_data["drink_failure"] = all_drink[failure_mask].tolist()
                raw_data["length_failure"] = all_lengths[failure_mask].tolist()
            else:
                # All episodes successful - mark as None for "—" display
                for metric in ["energy_failure", "food_failure", "drink_failure", "length_failure"]:
                    metrics[metric] = None
                    raw_data[metric] = []
    
    return metrics, raw_data


def perform_statistical_tests(method_raw_data: Dict[str, Dict[str, List[float]]], metric: str, alpha: float = 0.01) -> List[str]:
    """
    Perform pairwise t-tests for a given metric across all methods.
    Returns list of method names that are statistically tied for the best.
    """
    # Get all methods that have data for this metric
    methods_with_data = [method for method, data in method_raw_data.items() if metric in data and data[metric]]
    
    if len(methods_with_data) < 2:
        return methods_with_data  # Not enough data for comparison
    
    # Find the method with highest mean
    means = {}
    for method in methods_with_data:
        means[method] = np.mean(method_raw_data[method][metric])
    
    best_method = max(means.keys(), key=lambda m: means[m])
    best_mean = means[best_method]
    
    # Find all methods that are not statistically significantly worse than the best
    statistically_best = [best_method]
    
    for method in methods_with_data:
        if method == best_method:
            continue
            
        # Perform t-test between best method and current method
        best_data = method_raw_data[best_method][metric]
        current_data = method_raw_data[method][metric]
        
        # Use Welch's t-test (unequal variances)
        try:
            _, p_value = stats.ttest_ind(best_data, current_data, equal_var=False)
            
            # If p-value > alpha, the difference is not significant
            if p_value > alpha:
                statistically_best.append(method)
        except:
            # If test fails, be conservative and include the method
            statistically_best.append(method)
    
    return statistically_best


def fmt(v: Optional[float], percent=False, bold=False) -> str:
    """Format a value for display in the table."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return r"\multicolumn{1}{c}{—}"
    
    if percent:
        formatted = f"{v:.1f}"
    else:
        formatted = f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"
    
    if bold:
        formatted = r"\textbf{" + formatted + "}"
    
    return formatted


def build_rows(methods: List[str]) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    """Build table rows for the specified methods."""
    rows = []
    method_raw_data = {}  # Store raw data for statistical testing

    # First pass: collect all data
    for method_name in methods:
        if method_name not in METHOD_DIRS:
            print(f"[warn] Method '{method_name}' not found in METHOD_DIRS")
            continue

        method_dir = METHOD_DIRS[method_name]
        result = load_latest_result(method_dir)

        if result is None:
            continue

        # Extract metrics and raw data
        extracted, raw_data = extract_metrics_from_result(result)
        method_raw_data[method_name] = raw_data

    # Perform statistical tests for each metric
    statistical_best = {}
    for m in METRICS:
        statistical_best[m] = perform_statistical_tests(method_raw_data, m)

    # Second pass: build rows with statistical significance
    for method_name in methods:
        if method_name not in METHOD_DIRS:
            continue

        method_dir = METHOD_DIRS[method_name]
        result = load_latest_result(method_dir)

        row = {"Method": latex_escape(method_name)}

        if result is None:
            # Fill with missing values
            for gtitle, metrics, _labels in HEADER_GROUPS:
                for m in metrics:
                    row[m] = r"\multicolumn{1}{c}{—}"
            rows.append(row)
            continue

        # Extract metrics
        extracted, _ = extract_metrics_from_result(result)

        # Fill row with extracted metrics
        for m in METRICS:
            if m in extracted:
                val = extracted[m]
                # Check if this method is statistically best for this metric
                is_best = method_name in statistical_best[m]
                # Achievement rates are already percentages, episode length is raw
                is_percent = m.startswith("achievement_") and m.endswith("_rate")
                row[m] = fmt(val, percent=is_percent, bold=is_best)
            else:
                row[m] = r"\multicolumn{1}{c}{—}"

        rows.append(row)

    return rows, statistical_best


def tex_grouped_table(
    baselines_df: pd.DataFrame, ablations_df: pd.DataFrame, caption: str, label: str
) -> str:
    """Generate LaTeX table with grouped headers."""
    # Handle case where baselines_df might be empty
    if not baselines_df.empty:
        assert list(baselines_df.columns) == list(ablations_df.columns)
        all_cols = list(baselines_df.columns)
    else:
        all_cols = list(ablations_df.columns)

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
    ]

    # Add baselines section if we have any
    if not baselines_df.empty:
        lines.extend(
            [
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Baselines}}}} \\",
                rf"\addlinespace[{ADDLINESPACE_PT}pt]",
                *body(baselines_df),
                r"\midrule",
            ]
        )

    # Add ablations section
    lines.extend(
        [
            rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Our Methods}}}} \\",
            rf"\addlinespace[{ADDLINESPACE_PT}pt]",
            *body(ablations_df),
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            rf"\end{{{env}}}",
        ]
    )

    return "\n".join(lines)


def main():
    """Main function to generate the table."""
    print("Building table from local evaluation results...")
    print(f"Looking for methods in: {list(METHOD_DIRS.keys())}")

    # Column order from HEADER_GROUPS
    cols = ["Method"] + [m for _g, metrics, _l in HEADER_GROUPS for m in metrics]

    # Build baseline and ablation rows with combined statistical testing
    all_methods = BASELINE_METHODS + ABLATION_METHODS
    all_rows, all_stats = build_rows(all_methods)
    
    # Split the rows back into baseline and ablation groups
    base_rows = all_rows[:len(BASELINE_METHODS)] if BASELINE_METHODS else []
    abl_rows = all_rows[len(BASELINE_METHODS):]

    baselines_df = (
        pd.DataFrame(base_rows, columns=cols)
        if base_rows
        else pd.DataFrame(columns=cols)
    )
    ablations_df = pd.DataFrame(abl_rows, columns=cols)

    # Preview (console)
    preview_cols = ["Method"] + [
        DISPLAY_NAMES.get(m, m.replace("_", " ").title()) for m in cols[1:]
    ]

    if not baselines_df.empty:
        print("\n== Baselines ==")
        print(
            baselines_df.rename(
                columns=dict(zip(cols[1:], preview_cols[1:]))
            ).to_string(index=False)
        )

    print("\n== Our Methods ==")
    print(
        ablations_df.rename(columns=dict(zip(cols[1:], preview_cols[1:]))).to_string(
            index=False
        )
    )
    
    # Print statistical significance summary
    print(f"\n== Statistical Significance (p < 0.01) ==")
    for m in METRICS:
        if m in all_stats and len(all_stats[m]) > 1:
            best_methods = ", ".join(all_stats[m])
            metric_display = DISPLAY_NAMES.get(m, m.replace("_", " ").title())
            print(f"{metric_display}: {best_methods}")

    # Generate LaTeX table
    tex = tex_grouped_table(baselines_df, ablations_df, CAPTION, LABEL)

    # Save to file
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    out_path = os.path.join(SAVE_FOLDER, f"{FILENAME_PREFIX}.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(f"\n[save] {os.path.abspath(out_path)}")


def generate_conditional_survival_table():
    """Generate a table showing survival metrics conditioned on episode success."""
    
    # New metrics for conditional table
    CONDITIONAL_METRICS = [
        "energy_success", "food_success", "drink_success", "length_success",
        "energy_failure", "food_failure", "drink_failure", "length_failure"
    ]
    
    # New display names for conditional metrics
    CONDITIONAL_DISPLAY_NAMES = {
        "energy_success": "Energy (Success)",
        "food_success": "Food (Success)", 
        "drink_success": "Drink (Success)",
        "length_success": "Length (Success)",
        "energy_failure": "Energy (Failure)",
        "food_failure": "Food (Failure)",
        "drink_failure": "Drink (Failure)", 
        "length_failure": "Length (Failure)"
    }
    
    # New header grouping for conditional table
    CONDITIONAL_HEADER_GROUPS = [
        (
            "Success Episodes",
            ["energy_success", "food_success", "drink_success", "length_success"],
            ["Energy", "Food", "Drink", "Length"]
        ),
        (
            "Failure Episodes", 
            ["energy_failure", "food_failure", "drink_failure", "length_failure"],
            ["Energy", "Food", "Drink", "Length"]
        )
    ]
    
    print("Building conditional survival table from local evaluation results...")
    print(f"Looking for methods in: {list(METHOD_DIRS.keys())}")
    
    # Column order from CONDITIONAL_HEADER_GROUPS
    cols = ["Method"] + [m for _g, metrics, _l in CONDITIONAL_HEADER_GROUPS for m in metrics]
    
    # Build baseline and ablation rows with combined statistical testing
    all_methods = BASELINE_METHODS + ABLATION_METHODS
    all_rows = []
    method_raw_data = {}
    
    # First pass: collect all data
    for method_name in all_methods:
        if method_name not in METHOD_DIRS:
            print(f"[warn] Method '{method_name}' not found in METHOD_DIRS")
            continue

        method_dir = METHOD_DIRS[method_name]
        result = load_latest_result(method_dir)

        if result is None:
            continue

        # Extract conditional metrics and raw data
        extracted, raw_data = extract_conditional_survival_metrics(result)
        method_raw_data[method_name] = raw_data

    # Perform statistical tests for each metric
    statistical_best = {}
    for m in CONDITIONAL_METRICS:
        statistical_best[m] = perform_statistical_tests(method_raw_data, m)

    # Second pass: build rows with statistical significance
    for method_name in all_methods:
        if method_name not in METHOD_DIRS:
            continue

        method_dir = METHOD_DIRS[method_name]
        result = load_latest_result(method_dir)

        row = {"Method": latex_escape(method_name)}

        if result is None:
            # Fill with missing values
            for gtitle, metrics, _labels in CONDITIONAL_HEADER_GROUPS:
                for m in metrics:
                    row[m] = r"\multicolumn{1}{c}{—}"
            all_rows.append(row)
            continue

        # Extract conditional metrics
        extracted, _ = extract_conditional_survival_metrics(result)

        # Fill row with extracted metrics
        for m in CONDITIONAL_METRICS:
            if m in extracted:
                val = extracted[m]
                # Check if this method is statistically best for this metric
                is_best = method_name in statistical_best[m]
                row[m] = fmt(val, percent=False, bold=is_best)
            else:
                row[m] = r"\multicolumn{1}{c}{—}"

        all_rows.append(row)
    
    # Split the rows back into baseline and ablation groups
    base_rows = all_rows[:len(BASELINE_METHODS)] if BASELINE_METHODS else []
    abl_rows = all_rows[len(BASELINE_METHODS):]

    baselines_df = (
        pd.DataFrame(base_rows, columns=cols)
        if base_rows
        else pd.DataFrame(columns=cols)
    )
    ablations_df = pd.DataFrame(abl_rows, columns=cols)

    # Preview (console)
    preview_cols = ["Method"] + [
        CONDITIONAL_DISPLAY_NAMES.get(m, m.replace("_", " ").title()) for m in cols[1:]
    ]

    if not baselines_df.empty:
        print("\n== Baselines ==")
        print(
            baselines_df.rename(
                columns=dict(zip(cols[1:], preview_cols[1:]))
            ).to_string(index=False)
        )

    print("\n== Our Methods ==")
    print(
        ablations_df.rename(columns=dict(zip(cols[1:], preview_cols[1:]))).to_string(
            index=False
        )
    )
    
    # Print statistical significance summary
    print(f"\n== Statistical Significance (p < 0.01) ==")
    for m in CONDITIONAL_METRICS:
        if m in statistical_best and len(statistical_best[m]) > 1:
            best_methods = ", ".join(statistical_best[m])
            metric_display = CONDITIONAL_DISPLAY_NAMES.get(m, m.replace("_", " ").title())
            print(f"{metric_display}: {best_methods}")

    # Generate LaTeX table with modified function for conditional headers
    def tex_conditional_grouped_table(
        baselines_df: pd.DataFrame, ablations_df: pd.DataFrame, caption: str, label: str
    ) -> str:
        """Generate LaTeX table with conditional grouped headers."""
        # Handle case where baselines_df might be empty
        if not baselines_df.empty:
            assert list(baselines_df.columns) == list(ablations_df.columns)
            all_cols = list(baselines_df.columns)
        else:
            all_cols = list(ablations_df.columns)

        # Column order = Method + CONDITIONAL_HEADER_GROUPS metrics in order
        col_order = ["Method"] + [m for _g, metrics, _l in CONDITIONAL_HEADER_GROUPS for m in metrics]
        n_cols = 1 + sum(len(metrics) for _g, metrics, _l in CONDITIONAL_HEADER_GROUPS)

        env = "table*" if TABLE_ENV_STAR else "table"
        col_spec = "@{}l" + ("r" * (n_cols - 1)) + "@{}"  # kill outer padding

        # First header row: group titles
        parts = ["Method"]
        col_ranges = []  # for cmidrule
        col_idx = 2  # 1-based; col 1 is Method
        for gtitle, metrics, labels in CONDITIONAL_HEADER_GROUPS:
            span = len(metrics)
            parts.append(rf"\multicolumn{{{span}}}{{c}}{{{gtitle}}}")
            col_ranges.append((col_idx, col_idx + span - 1))
            col_idx += span
        header1 = " & ".join(parts) + r" \\"

        # cmidrules
        cmids = [rf"\cmidrule(lr){{{a}-{b}}}" for (a, b) in col_ranges]

        # Second header row: short labels
        labels = ["Method"] + [lbl for _g, metrics, lbls in CONDITIONAL_HEADER_GROUPS for lbl in lbls]
        header2 = " & ".join(labels) + r" \\"

        def body(df: pd.DataFrame) -> List[str]:
            lines = []
            for _, r in df.iterrows():
                cells = [r["Method"]] + [
                    r[m] for _g, metrics, _l in CONDITIONAL_HEADER_GROUPS for m in metrics
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
        ]

        # Add baselines section if we have any
        if not baselines_df.empty:
            lines.extend(
                [
                    rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Baselines}}}} \\",
                    rf"\addlinespace[{ADDLINESPACE_PT}pt]",
                    *body(baselines_df),
                    r"\midrule",
                ]
            )

        # Add ablations section
        lines.extend(
            [
                rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Our Methods}}}} \\",
                rf"\addlinespace[{ADDLINESPACE_PT}pt]",
                *body(ablations_df),
                r"\bottomrule",
                r"\end{tabular}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                rf"\end{{{env}}}",
            ]
        )

        return "\n".join(lines)

    # Generate LaTeX table
    conditional_caption = (
        "Conditional survival metrics from local experiments. "
        "Values show mean metrics for episodes that succeeded (collected diamond) vs failed."
    )
    conditional_label = "tab:conditional-survival-results"
    
    tex = tex_conditional_grouped_table(baselines_df, ablations_df, conditional_caption, conditional_label)

    # Save to file
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    out_path = os.path.join(SAVE_FOLDER, f"conditional_survival_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(f"\n[save] {os.path.abspath(out_path)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--conditional":
        generate_conditional_survival_table()
    else:
        main()
