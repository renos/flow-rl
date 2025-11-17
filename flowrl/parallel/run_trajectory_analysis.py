"""
Standalone script for running trajectory analysis on a specific GPU.

This script is called by the scheduler to analyze trajectories after initial training.
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from flowrl.llm.flow import Flow
from flowrl.parallel.training_processor import load_global_checkpoint
from flowrl.utils.test import gen_trajectories_for_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--run_folder", type=str, required=True)
    parser.add_argument("--expert_idx", type=int, required=True)
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--frame_gen_envs", type=int, default=32)
    parser.add_argument("--frame_gen_max_frames", type=int, default=1000)
    parser.add_argument("--num_trajectories", type=int, default=10, help="Number of successful trajectories to collect")
    args = parser.parse_args()

    # Load skill data
    with open(args.skill_data_path, "rb") as f:
        skill_data = pickle.load(f)

    # Minimal Flow setup
    class MockArgs:
        def __init__(self, env_name, base_dir):
            self.env_name = env_name
            self.graph_path = base_dir
            self.llm_name = None
            self.current_i = 0
            self.previous_i = None

    flow_graph = Flow(MockArgs(args.env_name, args.base_dir))

    # Load existing skills from global checkpoint so prompts have correct context
    try:
        global_ckpt = load_global_checkpoint(Path(args.base_dir))
        skills_from_ckpt = global_ckpt.get("skills", {}) or {}
        flow_graph.skills = skills_from_ckpt
        flow_graph.db["skills"] = skills_from_ckpt
        # Build skills_without_code view (include metrics if present)
        skills_wo = {}
        for key, val in skills_from_ckpt.items():
            swc = val.get("skill_with_consumption", {}) if isinstance(val, dict) else {}
            entry = dict(swc) if isinstance(swc, dict) else {}
            metrics = val.get("metrics") if isinstance(val, dict) else None
            if metrics:
                entry["metrics"] = metrics
            skills_wo[key] = entry
        flow_graph.db["skills_without_code"] = skills_wo
        # Refresh frontier summary when applicable
        if getattr(flow_graph, "use_frontier_based_prompts", False):
            flow_graph.db["frontier_summary"] = flow_graph.generate_frontier_summary_from_skills()
    except Exception:
        # Continue even if checkpoint is missing; analysis can still run
        pass

    # Run trajectory analysis
    run_folder = Path(args.run_folder)
    policies_path = run_folder / f"{args.expert_idx}_policies"
    exec_plan = skill_data.get("execution_plan", []) or []
    goal_state = len(exec_plan) - 1 if len(exec_plan) > 0 else 0

    # gen_frames_hierarchical expects goal_state to be the actual target state
    # (after all execution_plan steps complete)
    actual_goal_state = goal_state + 1

    print(f"Collecting trajectories for analysis...")
    print(f"  Policies: {policies_path}")
    print(f"  Execution plan length: {len(exec_plan)}")
    print(f"  Target goal state: {actual_goal_state}")
    print(f"  Num envs: {args.frame_gen_envs}")
    print(f"  Max frames: {args.frame_gen_max_frames}")
    print(f"  Num trajectories: {args.num_trajectories}")

    trajectories = gen_trajectories_for_analysis(
        policy_path=policies_path,
        max_num_frames=args.frame_gen_max_frames,
        goal_state=actual_goal_state,
        num_envs=args.frame_gen_envs,
        num_trajectories=args.num_trajectories,
    )

    # Prepare Flow context
    flow_graph.db["current"] = flow_graph.db.get("current", {})
    flow_graph.db["current"]["skill_with_consumption"] = skill_data.get("skill_with_consumption", {})

    print(f"Running trajectory explanation on {len(trajectories)} successful trajectories...")
    flow_graph.explain_trajectories(trajectories, goal_state)

    # Save trajectory debugging data to readable text file
    trajectory_debug_path = Path(args.output_path).parent / "trajectory_debug.txt"
    example_trajectories = flow_graph.db.get("example_trajectories", [])
    with open(trajectory_debug_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAJECTORY ANALYSIS DEBUG DATA\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Skill Name: {skill_data.get('skill_with_consumption', {}).get('skill_name', 'Unknown')}\n")
        f.write(f"Goal State: {goal_state}\n")
        f.write(f"Execution Plan Length: {len(exec_plan)}\n")
        f.write(f"Number of Environments: {args.frame_gen_envs}\n")
        f.write(f"Max Frames: {args.frame_gen_max_frames}\n")
        f.write(f"Number of Trajectories Collected: {len(trajectories)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("EXECUTION PLAN:\n")
        f.write("-" * 80 + "\n")
        for i, step in enumerate(exec_plan):
            f.write(f"{i}: {step}\n")
        f.write("\n")

        # Write all trajectories
        for traj_idx, example_trajectory in enumerate(example_trajectories):
            f.write("=" * 80 + "\n")
            f.write(f"TRAJECTORY #{traj_idx + 1} (Analyzed):\n")
            f.write("=" * 80 + "\n")
            if example_trajectory:
                for item in example_trajectory:
                    f.write(f"{item}\n")
            else:
                f.write("No trajectory data (empty list)\n")
            f.write("\n")

        # Write shapes for first trajectory only
        if len(trajectories) > 0:
            first_traj = trajectories[0]
            f.write("-" * 80 + "\n")
            f.write("ENV STATES SHAPE & INFO (First Trajectory):\n")
            f.write("-" * 80 + "\n")
            env_states = first_traj['env_states']
            f.write(f"Type: {type(env_states)}\n")
            if hasattr(env_states, 'shape'):
                f.write(f"Shape: {env_states.shape}\n")
            if hasattr(env_states, 'dtype'):
                f.write(f"Dtype: {env_states.dtype}\n")
            f.write("\n")

            f.write("-" * 80 + "\n")
            f.write("ACTIONS SHAPE & INFO (First Trajectory):\n")
            f.write("-" * 80 + "\n")
            actions = first_traj['actions']
            f.write(f"Type: {type(actions)}\n")
            if hasattr(actions, 'shape'):
                f.write(f"Shape: {actions.shape}\n")
            if hasattr(actions, 'dtype'):
                f.write(f"Dtype: {actions.dtype}\n")
            f.write("\n")

    print(f"Trajectory debug data saved to: {trajectory_debug_path}")

    # Apply updates to skill data
    skill_updates = flow_graph.db.get("current", {}).get("skill_update_results", {})
    if skill_updates:
        print(f"Applying skill updates:")
        swc = skill_data.setdefault("skill_with_consumption", {})
        if "updated_requirements" in skill_updates:
            print(f"  - Updated requirements: {skill_updates['updated_requirements']}")
            swc["requirements"] = skill_updates["updated_requirements"]
        if "updated_consumption" in skill_updates:
            print(f"  - Updated consumption: {skill_updates['updated_consumption']}")
            swc["consumption"] = skill_updates["updated_consumption"]
        if "updated_gain" in skill_updates:
            print(f"  - Updated gain: {skill_updates['updated_gain']}")
            swc["gain"] = skill_updates["updated_gain"]

    # Save result
    with open(args.output_path, "wb") as f:
        pickle.dump(skill_data, f)

    print(f"Trajectory analysis complete. Output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
