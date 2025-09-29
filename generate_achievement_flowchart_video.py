#!/usr/bin/env python3
"""
Generate a video of achievement flowchart showing completion rates changing over training time.
Each frame shows the flowchart with nodes colored by completion rates at that timestep.
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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter
import subprocess
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ----------------------------- CONFIG -----------------------------

# Default run to analyze
DEFAULT_RUNS = [
    #"interpretable_rl/flow-rl/0kfeorlf",  # Single run to analyze
    "interpretable_rl/flow-rl/ny62debx",
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

# Achievement groupings for layout
ACHIEVEMENT_GROUPS = {
    "Classic Core": [
        "COLLECT_WOOD", "PLACE_TABLE", "MAKE_WOOD_PICKAXE", "COLLECT_STONE",
        "COLLECT_COAL", "PLACE_FURNACE", "MAKE_STONE_PICKAXE", "COLLECT_IRON",
        "MAKE_IRON_PICKAXE", "COLLECT_DIAMOND"
    ],
    "Glass & Ceramics": [
        "PLACE_KILN", "COLLECT_SAND", "SMELT_GLASS", "MAKE_GLASS_BOTTLE",
        "MAKE_LENS", "MAKE_TELESCOPE", "COLLECT_CLAY", "FIRE_BRICK",
        "COLLECT_LIMESTONE", "MAKE_LIME", "MAKE_MORTAR", "PLACE_WINDOW",
        "PLACE_WALL_MASONRY"
    ],
    "Metallurgy": [
        "PLACE_ANVIL", "MAKE_BELLOWS", "SMELT_IRON_BAR", "MAKE_STEEL_BAR",
        "COLLECT_COPPER", "COLLECT_TIN", "MAKE_BRONZE_BAR", "FORGE_STEEL_PICKAXE",
        "FORGE_BRONZE_PICKAXE", "HARDEN_EDGE", "FORGE_STEEL_SWORD", "FORGE_BRONZE_SWORD"
    ],
    "Chemistry & Alchemy": [
        "PLACE_COMPOSTER", "MAKE_TAR", "MAKE_FERTILIZER", "PLACE_ALCHEMY_BENCH",
        "BREW_TONIC", "BREW_STONE_SKIN", "MAKE_FLUX", "PLACE_PLANT"
    ]
}

# Achievement dependencies from the Mermaid diagram
ACHIEVEMENT_DEPENDENCIES = {
    "PLACE_TABLE": ["COLLECT_WOOD"],
    "MAKE_WOOD_PICKAXE": ["PLACE_TABLE"],
    "COLLECT_STONE": ["MAKE_WOOD_PICKAXE"],
    "COLLECT_COAL": ["MAKE_WOOD_PICKAXE"],
    "PLACE_FURNACE": ["COLLECT_STONE"],
    "MAKE_STONE_PICKAXE": ["COLLECT_STONE"],
    "COLLECT_IRON": ["MAKE_STONE_PICKAXE"],
    "MAKE_IRON_PICKAXE": ["COLLECT_IRON"],
    "COLLECT_DIAMOND": ["MAKE_IRON_PICKAXE"],
    "SMELT_GLASS": ["COLLECT_SAND", "COLLECT_COAL", "PLACE_KILN"],
    "MAKE_GLASS_BOTTLE": ["SMELT_GLASS", "PLACE_KILN"],
    "MAKE_LENS": ["SMELT_GLASS", "PLACE_TABLE"],
    "MAKE_TELESCOPE": ["MAKE_LENS", "COLLECT_WOOD", "PLACE_TABLE"],
    "FIRE_BRICK": ["COLLECT_CLAY", "COLLECT_COAL", "PLACE_KILN"],
    "MAKE_LIME": ["COLLECT_LIMESTONE", "COLLECT_COAL", "PLACE_KILN"],
    "MAKE_MORTAR": ["MAKE_LIME", "COLLECT_SAND", "PLACE_TABLE"],
    "PLACE_WINDOW": ["SMELT_GLASS"],
    "PLACE_WALL_MASONRY": ["MAKE_MORTAR"],
    "SMELT_IRON_BAR": ["COLLECT_IRON", "COLLECT_COAL", "PLACE_FURNACE"],
    "MAKE_STEEL_BAR": ["SMELT_IRON_BAR", "MAKE_BELLOWS"],
    "MAKE_BRONZE_BAR": ["COLLECT_COPPER", "COLLECT_TIN", "COLLECT_COAL", "PLACE_FURNACE"],
    "FORGE_STEEL_PICKAXE": ["MAKE_STEEL_BAR", "PLACE_TABLE", "PLACE_ANVIL"],
    "FORGE_BRONZE_PICKAXE": ["MAKE_BRONZE_BAR", "PLACE_TABLE", "PLACE_ANVIL"],
    "HARDEN_EDGE": ["FORGE_STEEL_PICKAXE", "FORGE_BRONZE_PICKAXE"],
    "FORGE_STEEL_SWORD": ["MAKE_STEEL_BAR", "PLACE_ANVIL"],
    "FORGE_BRONZE_SWORD": ["MAKE_BRONZE_BAR", "PLACE_ANVIL"],
    "MAKE_TAR": ["COLLECT_WOOD", "PLACE_KILN"],
    "MAKE_FERTILIZER": ["PLACE_COMPOSTER", "MAKE_TAR", "PLACE_PLANT"],
    "BREW_TONIC": ["PLACE_ALCHEMY_BENCH", "MAKE_GLASS_BOTTLE", "PLACE_PLANT"],
    "BREW_STONE_SKIN": ["PLACE_ALCHEMY_BENCH", "MAKE_TAR", "MAKE_GLASS_BOTTLE"],
    "MAKE_FLUX": ["MAKE_LIME", "MAKE_TAR", "PLACE_ALCHEMY_BENCH"],
}

# Video settings
X_AXIS = "_step"
VIDEO_FPS = 10
VIDEO_DURATION_SECONDS = 30  # Target video duration
OUTPUT_DIR = "achievement_videos"
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
        df = _robust(run.history, keys=need, pandas=True)
        if X_AXIS not in df.columns:
            df[X_AXIS] = np.arange(len(df))
        return df.sort_values(by=X_AXIS).reset_index(drop=True)
    except Exception as e:
        print(f"[fetch] {run_path}: {e}")
        return None


def completion_rate_to_color(rate: float) -> Tuple[float, float, float]:
    """Convert completion rate (0-100) to RGB color (red to green)."""
    rate = max(0.0, min(100.0, rate))
    normalized = rate / 100.0

    # Red to green gradient
    red = 1.0 - normalized
    green = normalized
    blue = 0.0

    return (red, green, blue)


def completion_rate_to_hex_color(rate: float) -> str:
    """Convert completion rate (0-100) to hex color (red to green)."""
    rate = max(0.0, min(100.0, rate))
    normalized = rate / 100.0

    # Red to green gradient
    red_component = int(255 * (1.0 - normalized))
    green_component = int(255 * normalized)
    blue_component = 0

    return f"#{red_component:02x}{green_component:02x}{blue_component:02x}"


def generate_mermaid_diagram(current_rates: Dict[str, float], timestep: int) -> str:
    """Generate Mermaid diagram with current completion rates."""

    # Read the base diagram
    with open(BASE_DIAGRAM_PATH, 'r', encoding='utf-8') as f:
        base_content = f.read()

    lines = base_content.split('\n')
    output_lines = []

    # Keep everything until we reach the styling section
    for line in lines:
        stripped = line.strip()

        # Skip existing classDef and node styling lines
        if (stripped.startswith('classDef') or
            stripped.endswith(':::classic') or
            stripped.endswith(':::met') or
            stripped.endswith(':::glass') or
            stripped.endswith(':::chem') or
            stripped.endswith(':::station')):
            continue

        output_lines.append(line)

        # If we just added the last connection, break to add our custom styling
        if 'MAKE_FLUX -.-> PLACE_FURNACE' in line:
            break

    # Add dynamic node coloring based on completion rates
    output_lines.append("")
    output_lines.append("    %% Dynamic completion rate coloring")

    for achievement_name, rate in current_rates.items():
        color = completion_rate_to_hex_color(rate)
        class_name = f"rate{achievement_name}"
        output_lines.append(f"    {achievement_name}:::{class_name}")
        output_lines.append(f"    classDef {class_name} fill:{color},stroke:#333,stroke-width:2px,color:#fff")

    # Add timestep info as title
    output_lines.insert(1, f"title: Achievement Progress - Step {timestep}")

    return '\n'.join(output_lines)


def mermaid_to_image(mermaid_content: str, output_path: str) -> bool:
    """Convert Mermaid diagram to PNG image using mermaid-cli."""
    try:
        # Write mermaid content to temp file
        temp_mmd = output_path.replace('.png', '.mmd')
        with open(temp_mmd, 'w', encoding='utf-8') as f:
            f.write(mermaid_content)

        # Use mermaid-cli to convert to PNG with even dimensions for H.264 compatibility
        result = subprocess.run([
            'mmdc', '-i', temp_mmd, '-o', output_path,
            '--width', '1920', '--height', '1080',
            '--backgroundColor', 'white',
            '--scale', '1'
        ], capture_output=True, text=True, timeout=30)

        # Clean up temp file
        os.remove(temp_mmd)

        if result.returncode == 0:
            return True
        else:
            print(f"Mermaid CLI error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("Mermaid CLI timed out")
        return False
    except Exception as e:
        print(f"Error converting Mermaid to image: {e}")
        return False


def get_achievement_timeseries(run_paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Get achievement completion rates over time for given runs."""
    print(f"Fetching achievement timeseries from {len(run_paths)} runs...")

    achievement_metrics = list(ACHIEVEMENT_MAPPING.keys())

    # Collect data from all runs
    all_data = []
    for run_path in run_paths:
        df = fetch_df(run_path, achievement_metrics)
        if df is not None:
            df['run_path'] = run_path
            all_data.append(df)
        else:
            print(f"Warning: Could not fetch data from {run_path}")

    if not all_data:
        print("Error: No data could be fetched from any runs")
        return pd.DataFrame(), []

    # Combine all runs
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert achievement values to percentages
    for metric in achievement_metrics:
        if metric in combined_df.columns:
            combined_df[metric] = combined_df[metric]  # Convert to percentage
            combined_df[metric] = combined_df[metric].fillna(0.0)  # Fill NaN with 0

    # Get timesteps
    timesteps = sorted(combined_df[X_AXIS].unique())

    print(f"Found {len(timesteps)} timesteps over {len(combined_df)} total records")

    return combined_df, timesteps


def create_achievement_layout() -> Dict[str, Tuple[float, float]]:
    """Create a layout for achievement nodes."""
    positions = {}

    # Group positions
    group_positions = {
        "Classic Core": (0.2, 0.7),
        "Glass & Ceramics": (0.8, 0.7),
        "Metallurgy": (0.2, 0.3),
        "Chemistry & Alchemy": (0.8, 0.3)
    }

    # Layout achievements within each group
    for group_name, achievements in ACHIEVEMENT_GROUPS.items():
        base_x, base_y = group_positions[group_name]

        # Arrange in grid within group
        grid_size = int(np.ceil(np.sqrt(len(achievements))))
        for i, achievement in enumerate(achievements):
            row = i // grid_size
            col = i % grid_size

            x = base_x + (col - grid_size/2) * 0.05
            y = base_y + (row - grid_size/2) * 0.08

            positions[achievement] = (x, y)

    return positions


def create_frame(df: pd.DataFrame, timestep: int, positions: Dict[str, Tuple[float, float]],
                frame_num: int, total_frames: int) -> plt.Figure:
    """Create a single frame of the achievement flowchart."""

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Filter data for this timestep
    timestep_data = df[df[X_AXIS] <= timestep]
    if timestep_data.empty:
        timestep_data = df[df[X_AXIS] == df[X_AXIS].min()]

    # Get latest values for each achievement
    current_rates = {}
    for metric, achievement_name in ACHIEVEMENT_MAPPING.items():
        if metric in timestep_data.columns:
            latest_value = timestep_data[metric].iloc[-1] if not timestep_data[metric].empty else 0.0
            current_rates[achievement_name] = latest_value
        else:
            current_rates[achievement_name] = 0.0

    # Draw group backgrounds
    group_colors = {
        "Classic Core": (0.9, 0.9, 0.9, 0.3),
        "Glass & Ceramics": (1.0, 0.9, 0.6, 0.3),
        "Metallurgy": (0.9, 0.6, 0.9, 0.3),
        "Chemistry & Alchemy": (0.6, 0.9, 1.0, 0.3)
    }

    for group_name, achievements in ACHIEVEMENT_GROUPS.items():
        if achievements:
            xs = [positions[ach][0] for ach in achievements if ach in positions]
            ys = [positions[ach][1] for ach in achievements if ach in positions]
            if xs and ys:
                min_x, max_x = min(xs) - 0.08, max(xs) + 0.08
                min_y, max_y = min(ys) - 0.05, max(ys) + 0.05

                rect = patches.Rectangle(
                    (min_x, min_y), max_x - min_x, max_y - min_y,
                    facecolor=group_colors.get(group_name, (0.8, 0.8, 0.8, 0.3)),
                    edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)

                # Add group label
                ax.text(min_x + 0.01, max_y - 0.01, group_name,
                       fontsize=12, fontweight='bold', va='top')

    # Draw dependency arrows first (so they appear behind nodes)
    for achievement, dependencies in ACHIEVEMENT_DEPENDENCIES.items():
        if achievement in positions:
            target_x, target_y = positions[achievement]
            for dep in dependencies:
                if dep in positions:
                    source_x, source_y = positions[dep]

                    # Calculate arrow direction and offset to avoid overlapping with circles
                    dx = target_x - source_x
                    dy = target_y - source_y
                    length = np.sqrt(dx**2 + dy**2)

                    if length > 0:
                        # Normalize direction
                        dx_norm = dx / length
                        dy_norm = dy / length

                        # Offset start and end points to avoid circle overlap
                        start_x = source_x + dx_norm * 0.03
                        start_y = source_y + dy_norm * 0.03
                        end_x = target_x - dx_norm * 0.03
                        end_y = target_y - dy_norm * 0.03

                        # Draw arrow
                        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.7))

    # Draw achievement nodes
    for achievement_name, (x, y) in positions.items():
        rate = current_rates.get(achievement_name, 0.0)
        color = completion_rate_to_color(rate)

        # Draw circle for achievement
        circle = patches.Circle((x, y), 0.025, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)

        # Add achievement label
        clean_name = achievement_name.replace('_', ' ').title()
        ax.text(x, y - 0.04, clean_name, ha='center', va='top', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Add percentage text
        ax.text(x, y, f"{rate:.0f}%", ha='center', va='center', fontsize=6,
               fontweight='bold', color='white' if rate < 50 else 'black')

    # Add title and progress info
    ax.text(0.5, 0.95, f"Achievement Progress - Step {timestep:,}",
           ha='center', va='top', fontsize=16, fontweight='bold')

    ax.text(0.5, 0.92, f"Frame {frame_num}/{total_frames}",
           ha='center', va='top', fontsize=12)

    # Add color legend
    legend_x = 0.02
    legend_y = 0.15
    ax.text(legend_x, legend_y + 0.08, "Completion Rate:", fontsize=10, fontweight='bold')

    for i, rate in enumerate([0, 25, 50, 75, 100]):
        color = completion_rate_to_color(rate)
        circle = patches.Circle((legend_x + 0.02, legend_y - i * 0.015), 0.008,
                              facecolor=color, edgecolor='black')
        ax.add_patch(circle)
        ax.text(legend_x + 0.035, legend_y - i * 0.015, f"{rate}%",
               va='center', fontsize=8)

    plt.tight_layout()
    return fig


def create_frame_mermaid(df: pd.DataFrame, timestep: int, frame_num: int, total_frames: int, temp_dir: str) -> str:
    """Create a single frame using Mermaid diagram converted to image."""

    # Filter data for this timestep
    timestep_data = df[df[X_AXIS] <= timestep]
    if timestep_data.empty:
        timestep_data = df[df[X_AXIS] == df[X_AXIS].min()]

    # Get latest values for each achievement
    current_rates = {}
    for metric, achievement_name in ACHIEVEMENT_MAPPING.items():
        if metric in timestep_data.columns:
            latest_value = timestep_data[metric].iloc[-1] if not timestep_data[metric].empty else 0.0
            rate_pct = latest_value  # Already in percentage format
            current_rates[achievement_name] = max(0.0, min(100.0, rate_pct))
        else:
            current_rates[achievement_name] = 0.0

    # Generate Mermaid diagram with current rates
    mermaid_content = generate_mermaid_diagram(current_rates, timestep)

    # Create frame image path
    frame_path = os.path.join(temp_dir, f"frame_{frame_num:04d}.png")

    # Convert to image
    success = mermaid_to_image(mermaid_content, frame_path)

    if success:
        print(f"Generated frame {frame_num}/{total_frames} for timestep {timestep}")
        return frame_path
    else:
        print(f"Failed to generate frame {frame_num} for timestep {timestep}")
        return ""


def create_frame_task(args) -> tuple:
    """Worker function for parallel frame generation."""
    df, timestep, frame_num, total_frames, temp_dir = args
    frame_path = create_frame_mermaid(df, timestep, frame_num, total_frames, temp_dir)
    return frame_num, frame_path


def generate_achievement_video_mermaid(
    run_paths: List[str],
    run_names: List[str],
    output_filename: Optional[str] = None,
    max_frames: int = 30,
    max_workers: int = 16,
    run_id: Optional[str] = None
) -> str:
    """Generate achievement progress video using Mermaid diagrams with multithreading."""

    print("=== Generating Achievement Progress Video with Mermaid ===")

    # Get timeseries data
    df, timesteps = get_achievement_timeseries(run_paths)
    if df.empty:
        print("Error: No data available")
        return ""

    # Sample timesteps for video frames
    if len(timesteps) > max_frames:
        step_size = len(timesteps) // max_frames
        sampled_timesteps = timesteps[::step_size][:max_frames]
    else:
        sampled_timesteps = timesteps

    print(f"Creating {len(sampled_timesteps)} frames from {len(timesteps)} timesteps using {max_workers} threads")

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp(prefix="achievement_video_")
    frame_results = {}

    try:
        # Prepare tasks for parallel processing
        tasks = []
        for i, timestep in enumerate(sampled_timesteps):
            tasks.append((df, timestep, i + 1, len(sampled_timesteps), temp_dir))

        # Generate frames in parallel
        completed_frames = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(create_frame_task, task): task for task in tasks}

            # Process completed tasks
            for future in as_completed(future_to_task):
                try:
                    frame_num, frame_path = future.result()
                    if frame_path:
                        frame_results[frame_num] = frame_path
                        completed_frames += 1
                        if completed_frames % 10 == 0:  # Progress update every 10 frames
                            print(f"Completed {completed_frames}/{len(sampled_timesteps)} frames")
                    else:
                        print(f"Failed to generate frame {frame_num}")
                except Exception as e:
                    task = future_to_task[future]
                    print(f"Frame generation failed for task {task}: {e}")

        if not frame_results:
            print("Error: No frames were generated successfully")
            return ""

        print(f"Successfully generated {len(frame_results)} frames")

        # Create video from frames using ffmpeg
        if output_filename is None:
            safe_names = [re.sub(r'[^\w\-_]', '_', name) for name in run_names]
            output_filename = f"achievement_progress_mermaid_{'_'.join(safe_names[:2])}.mp4"

        # Create results directory structure based on run_id
        if run_id:
            results_dir = os.path.join("results", run_id)
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, output_filename)
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, output_filename)

        # FFmpeg command with padding filter to ensure even dimensions
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-framerate', str(VIDEO_FPS),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Ensure even dimensions
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_path
        ]

        print(f"Creating video with {len(frame_results)} frames at {VIDEO_FPS} FPS...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print(f"✅ Video created successfully: {os.path.abspath(output_path)}")
            return output_path
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return ""

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)


def generate_achievement_video(
    run_paths: List[str],
    run_names: List[str],
    output_filename: Optional[str] = None,
    max_frames: int = None,
    run_id: Optional[str] = None
) -> str:
    """Generate achievement flowchart video showing progress over time."""

    print("=== Generating Achievement Progress Video ===")

    # Get timeseries data
    df, timesteps = get_achievement_timeseries(run_paths)

    if df.empty:
        print("No data available for video generation")
        return ""

    # Determine frame sampling
    if max_frames is None:
        max_frames = min(len(timesteps), VIDEO_FPS * VIDEO_DURATION_SECONDS)

    # Sample timesteps for video frames
    if len(timesteps) > max_frames:
        step_size = len(timesteps) // max_frames
        sampled_timesteps = timesteps[::step_size][:max_frames]
    else:
        sampled_timesteps = timesteps

    print(f"Creating video with {len(sampled_timesteps)} frames from {len(timesteps)} timesteps")

    # Create layout
    positions = create_achievement_layout()

    # Generate frames
    frames = []
    temp_dir = tempfile.mkdtemp()

    try:
        for i, timestep in enumerate(sampled_timesteps):
            print(f"Generating frame {i+1}/{len(sampled_timesteps)} (step {timestep})")

            fig = create_frame(df, timestep, positions, i+1, len(sampled_timesteps))

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            frames.append(frame_path)
            plt.close(fig)

        # Create video using ffmpeg
        if output_filename is None:
            safe_names = [re.sub(r'[^\w\-_]', '_', name) for name in run_names]
            output_filename = f"achievement_progress_{'_'.join(safe_names[:2])}.mp4"

        # Create results directory structure based on run_id
        if run_id:
            results_dir = os.path.join("results", run_id)
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, output_filename)
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Use ffmpeg to create video
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(VIDEO_FPS),
            '-i', os.path.join(temp_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path
        ]

        print(f"Creating video with ffmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Video created successfully: {os.path.abspath(output_path)}")

            # Print summary statistics
            final_rates = {}
            final_timestep = sampled_timesteps[-1]
            final_data = df[df[X_AXIS] <= final_timestep]

            for metric, achievement_name in ACHIEVEMENT_MAPPING.items():
                if metric in final_data.columns and not final_data[metric].empty:
                    final_rates[achievement_name] = final_data[metric].iloc[-1]

            print(f"\n=== Final Achievement Rates (Step {final_timestep}) ===")
            for achievement_name, rate in sorted(final_rates.items(), key=lambda x: -x[1]):
                print(f"{achievement_name:25} {rate:6.1f}%")

        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return ""

    finally:
        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir)

    return output_path


def main(run_id: Optional[str] = None):
    """Generate achievement progress video for the specified run."""
    print(f"Generating achievement progress video for run: {DEFAULT_RUNS[0]}")

    # Extract run ID from run path if not provided
    if run_id is None and DEFAULT_RUNS:
        run_id = DEFAULT_RUNS[0].split('/')[-1]  # Extract last part of run path

    output_path = generate_achievement_video_mermaid(
        DEFAULT_RUNS,
        DEFAULT_RUN_NAMES,
        output_filename="achievement_progress_mermaid.mp4",
        max_frames=300,  # 30 second video at 10 FPS
        run_id=run_id
    )

    if output_path:
        print(f"🎬 Video generation completed!")
        print(f"📁 Video saved at: {output_path}")
        print(f"🎥 You can now watch the achievement progression over training time!")

    return output_path


if __name__ == "__main__":
    import argparse
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        description="Generate achievement progress video for Flow-RL runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for organizing output (will extract from DEFAULT_RUNS if not provided)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to generate for the video"
    )

    parser.add_argument(
        "--run-path",
        type=str,
        help="Wandb run path (e.g., 'interpretable_rl/flow-rl/abc123'). Overrides DEFAULT_RUNS if provided"
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="Flow-RL Run",
        help="Display name for the run"
    )

    parser.add_argument(
        "--output-filename",
        type=str,
        help="Custom output filename (defaults to 'achievement_progress_mermaid.mp4')"
    )

    args = parser.parse_args()

    # Use provided run path or fall back to default
    run_paths = [args.run_path] if args.run_path else DEFAULT_RUNS
    run_names = [args.run_name] if args.run_path else DEFAULT_RUN_NAMES

    # Extract run ID from run path if not provided
    run_id = args.run_id
    if run_id is None and run_paths:
        run_id = run_paths[0].split('/')[-1]  # Extract last part of run path

    print(f"Generating achievement progress video...")
    print(f"Run path(s): {run_paths}")
    print(f"Run ID: {run_id}")
    print(f"Max frames: {args.max_frames}")

    output_path = generate_achievement_video_mermaid(
        run_paths,
        run_names,
        output_filename=args.output_filename or "achievement_progress_mermaid.mp4",
        max_frames=args.max_frames,
        run_id=run_id
    )

    if output_path:
        print(f"🎬 Video generation completed!")
        print(f"📁 Video saved at: {output_path}")
        print(f"🎥 You can now watch the achievement progression over training time!")
    else:
        print("❌ Video generation failed!")
        exit(1)