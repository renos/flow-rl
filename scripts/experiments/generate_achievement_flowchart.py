#!/usr/bin/env python3
"""
Generate dynamic achievement flow chart with completion rate coloring.
Each node is colored from red (0% completion) to green (100% completion).
"""

import os
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import wandb
from wandb.apis.public import Api

# ----------------------------- CONFIG -----------------------------

# Default run to analyze
DEFAULT_RUNS = [
    #"interpretable_rl/flow-rl/v3a0y2v0",  # Single run to analyze
    "interpretable_rl/flow-rl/0kfeorlf",
]

DEFAULT_RUN_NAMES = [
    "Flow-RL Run",
]

# Achievement metric mapping (Wandb metric names to achievement node names)
ACHIEVEMENT_MAPPING = {
    # Classic Core achievements
    "Achievements/collect_wood": "COLLECT_WOOD",
    "Achievements/place_table": "PLACE_TABLE",
    "Achievements/make_wood_pickaxe": "MAKE_WOOD_PICKAXE",
    "Achievements/collect_stone": "COLLECT_STONE",
    "Achievements/collect_coal": "COLLECT_COAL",
    "Achievements/place_furnace": "PLACE_FURNACE",
    "Achievements/make_stone_pickaxe": "MAKE_STONE_PICKAXE",
    "Achievements/collect_iron": "COLLECT_IRON",
    "Achievements/make_iron_pickaxe": "MAKE_IRON_PICKAXE",
    "Achievements/collect_diamond": "COLLECT_DIAMOND",

    # Glass & Ceramics
    "Achievements/place_kiln": "PLACE_KILN",
    "Achievements/collect_sand": "COLLECT_SAND",
    "Achievements/smelt_glass": "SMELT_GLASS",
    "Achievements/make_glass_bottle": "MAKE_GLASS_BOTTLE",
    "Achievements/make_lens": "MAKE_LENS",
    "Achievements/make_telescope": "MAKE_TELESCOPE",
    "Achievements/collect_clay": "COLLECT_CLAY",
    "Achievements/fire_brick": "FIRE_BRICK",
    "Achievements/collect_limestone": "COLLECT_LIMESTONE",
    "Achievements/make_lime": "MAKE_LIME",
    "Achievements/make_mortar": "MAKE_MORTAR",
    "Achievements/place_window": "PLACE_WINDOW",
    "Achievements/place_wall_masonry": "PLACE_WALL_MASONRY",

    # Metallurgy
    "Achievements/place_anvil": "PLACE_ANVIL",
    "Achievements/make_bellows": "MAKE_BELLOWS",
    "Achievements/smelt_iron_bar": "SMELT_IRON_BAR",
    "Achievements/make_steel_bar": "MAKE_STEEL_BAR",
    "Achievements/collect_copper": "COLLECT_COPPER",
    "Achievements/collect_tin": "COLLECT_TIN",
    "Achievements/make_bronze_bar": "MAKE_BRONZE_BAR",
    "Achievements/forge_steel_pickaxe": "FORGE_STEEL_PICKAXE",
    "Achievements/forge_bronze_pickaxe": "FORGE_BRONZE_PICKAXE",
    "Achievements/harden_edge": "HARDEN_EDGE",
    "Achievements/forge_steel_sword": "FORGE_STEEL_SWORD",
    "Achievements/forge_bronze_sword": "FORGE_BRONZE_SWORD",

    # Chemistry & Alchemy
    "Achievements/place_composter": "PLACE_COMPOSTER",
    "Achievements/make_tar": "MAKE_TAR",
    "Achievements/make_fertilizer": "MAKE_FERTILIZER",
    "Achievements/place_alchemy_bench": "PLACE_ALCHEMY_BENCH",
    "Achievements/brew_tonic": "BREW_TONIC",
    "Achievements/brew_stone_skin": "BREW_STONE_SKIN",
    "Achievements/make_flux": "MAKE_FLUX",
    "Achievements/place_plant": "PLACE_PLANT",
}

# Aggregation settings
X_AXIS = "_step"
AGG_METHOD = "last_k_mean"  # "last" | "last_k_mean" | "max"
LAST_K = 100
MAX_HISTORY_SAMPLES = None

# Output settings
OUTPUT_DIR = "achievement_flowcharts"
BASE_DIAGRAM_PATH = "Craftax/craftax/fabrax/achievement_diagram.mmd"

# Robust W&B settings
RETRY_ON_ERROR = True
MAX_RETRIES = 3
RETRY_SLEEP = 2.0

# ----------------------------- END CONFIG -----------------------------

api = Api()


def _robust(call, *a, **k):
    """Robust API call with retries."""
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


def fetch_df(run_path: str, keys: List[str]) -> Optional[pd.DataFrame]:
    """Fetch dataframe from W&B run."""
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
    """Aggregate series using specified method."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    if method == "last":
        return float(s.iloc[-1])
    if method == "last_k_mean":
        return float(s.iloc[-min(max(1, LAST_K), len(s)):].mean())
    if method == "max":
        return float(s.max())
    return float(s.iloc[-1])


def get_achievement_rates(run_paths: List[str]) -> Dict[str, float]:
    """Get achievement completion rates for given runs."""
    print(f"Fetching achievement data from {len(run_paths)} runs...")

    achievement_metrics = list(ACHIEVEMENT_MAPPING.keys())
    all_rates = {}

    # Collect data from all runs
    run_data = []
    for run_path in run_paths:
        df = fetch_df(run_path, achievement_metrics)
        if df is not None:
            run_data.append(df)
        else:
            print(f"Warning: Could not fetch data from {run_path}")

    if not run_data:
        print("Error: No data could be fetched from any runs")
        return {}

    # Calculate completion rates for each achievement
    for metric, achievement_name in ACHIEVEMENT_MAPPING.items():
        rates = []
        for df in run_data:
            if metric in df.columns:
                rate = aggregate(df[metric])
                if rate is not None:
                    # Convert to percentage (0-100 scale)
                    rate_pct = rate * 100.0 if 0.0 <= rate <= 1.1 else rate
                    rates.append(rate_pct)

        if rates:
            # Average across all runs
            avg_rate = np.mean(rates)
            all_rates[achievement_name] = max(0.0, min(100.0, avg_rate))
        else:
            all_rates[achievement_name] = 0.0

    return all_rates


def completion_rate_to_color(rate: float) -> str:
    """Convert completion rate (0-100) to color (red to green)."""
    # Ensure rate is in [0, 100] range
    rate = max(0.0, min(100.0, rate))

    # Convert to 0-1 scale
    normalized = rate / 100.0

    # Interpolate from red (0) to green (1)
    # Red: #ff0000, Green: #00ff00
    red_component = int(255 * (1 - normalized))
    green_component = int(255 * normalized)
    blue_component = 0

    return f"#{red_component:02x}{green_component:02x}{blue_component:02x}"


def parse_mermaid_diagram(file_path: str) -> Tuple[List[str], Dict[str, str]]:
    """Parse existing mermaid diagram to extract structure and node info."""
    print(f"Parsing base diagram from: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    structure_lines = []
    node_info = {}

    # Extract the main structure and node definitions
    in_flowchart = False
    for line in lines:
        stripped = line.strip()

        if stripped.startswith('flowchart'):
            in_flowchart = True
            structure_lines.append(line)
            continue

        if not in_flowchart:
            structure_lines.append(line)
            continue

        # Skip classDef and node styling lines (we'll replace them)
        if (stripped.startswith('classDef') or
            stripped.endswith(':::classic') or
            stripped.endswith(':::met') or
            stripped.endswith(':::glass') or
            stripped.endswith(':::chem') or
            stripped.endswith(':::station')):
            continue

        # Extract node definitions and connections
        if '-->' in stripped or '-.->.' in stripped:
            structure_lines.append(line)
        elif '[' in stripped and ']' in stripped:
            # Extract node name and description
            match = re.search(r'(\w+)\["([^"]+)"\]', stripped)
            if match:
                node_name = match.group(1)
                description = match.group(2)
                node_info[node_name] = description
            structure_lines.append(line)
        else:
            structure_lines.append(line)

    return structure_lines, node_info


def generate_dynamic_flowchart(
    run_paths: List[str],
    run_names: List[str],
    output_filename: Optional[str] = None,
    base_diagram_path: str = BASE_DIAGRAM_PATH
) -> str:
    """Generate achievement flowchart with completion-rate-based coloring."""

    print("=== Generating Dynamic Achievement Flowchart ===")

    # Get completion rates
    completion_rates = get_achievement_rates(run_paths)
    print(f"Retrieved completion rates for {len(completion_rates)} achievements")

    # Parse base diagram
    structure_lines, node_info = parse_mermaid_diagram(base_diagram_path)

    # Generate output content
    output_lines = []
    found_flowchart = False

    for line in structure_lines:
        stripped = line.strip()

        # Add header with run info
        if stripped.startswith('flowchart') and not found_flowchart:
            found_flowchart = True
            output_lines.append("---")
            output_lines.append("title: Achievement Completion Rates")
            output_lines.append(f"subtitle: Averaged across {len(run_names)} runs")
            output_lines.append("config:")
            output_lines.append("  layout: elk")
            output_lines.append("---")
            output_lines.append(line)
            continue

        output_lines.append(line)

    # Add dynamic node coloring
    output_lines.append("")
    output_lines.append("    %% Dynamic completion rate coloring (red=0% to green=100%)")

    for achievement_name, rate in completion_rates.items():
        color = completion_rate_to_color(rate)
        # Add the node styling with completion rate info
        output_lines.append(f"    {achievement_name}:::rate{int(rate)}")
        output_lines.append(f"    classDef rate{int(rate)} fill:{color},stroke:#333,stroke-width:2px,color:#fff")

    # Add legend/info
    output_lines.append("")
    output_lines.append("    %% Achievement Completion Rates:")
    for achievement_name, rate in sorted(completion_rates.items()):
        output_lines.append(f"    %% {achievement_name}: {rate:.1f}%")

    output_lines.append("")
    output_lines.append(f"    %% Generated from runs: {', '.join(run_names)}")

    # Save to file
    if output_filename is None:
        # Generate filename based on run names
        safe_names = [re.sub(r'[^\w\-_]', '_', name) for name in run_names]
        output_filename = f"achievement_rates_{'_'.join(safe_names[:2])}.mmd"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n[save] Generated flowchart: {os.path.abspath(output_path)}")

    # Print completion rate summary
    print(f"\n=== Achievement Completion Rate Summary ===")
    for achievement_name, rate in sorted(completion_rates.items(), key=lambda x: -x[1]):
        color_hex = completion_rate_to_color(rate)
        print(f"{achievement_name:25} {rate:6.1f}% {color_hex}")

    return output_path


def generate_comparison_flowcharts(
    run_groups: List[Tuple[List[str], List[str], str]],
    base_diagram_path: str = BASE_DIAGRAM_PATH
) -> List[str]:
    """Generate multiple flowcharts for comparison between different run groups.

    Args:
        run_groups: List of (run_paths, run_names, output_suffix) tuples
        base_diagram_path: Path to base mermaid diagram

    Returns:
        List of generated file paths
    """
    output_paths = []

    for run_paths, run_names, suffix in run_groups:
        output_filename = f"achievement_rates_{suffix}.mmd"
        path = generate_dynamic_flowchart(
            run_paths, run_names, output_filename, base_diagram_path
        )
        output_paths.append(path)

    return output_paths


def main():
    """Generate achievement flowchart for the single specified run."""
    print(f"Generating achievement flowchart for run: {DEFAULT_RUNS[0]}")

    # Generate flowchart for the single run
    output_path = generate_dynamic_flowchart(
        DEFAULT_RUNS,
        DEFAULT_RUN_NAMES,
        output_filename="achievement_rates_v3a0y2v0.mmd"
    )

    print(f"✅ Flowchart generated successfully!")
    print(f"📁 File saved at: {output_path}")
    print(f"🎨 Nodes are colored from red (0% completion) to green (100% completion)")

    return output_path


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()