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
from flowrl.utils.test import gen_frames_hierarchical


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

    print(f"Generating frames for trajectory analysis...")
    print(f"  Policies: {policies_path}")
    print(f"  Execution plan length: {len(exec_plan)}")
    print(f"  Target goal state: {actual_goal_state}")
    print(f"  Num envs: {args.frame_gen_envs}")
    print(f"  Max frames: {args.frame_gen_max_frames}")

    frames, states, env_states, actions = gen_frames_hierarchical(
        policy_path=policies_path,
        max_num_frames=args.frame_gen_max_frames,
        goal_state=actual_goal_state,
        num_envs=args.frame_gen_envs,
    )

    # Prepare Flow context
    flow_graph.db["current"] = flow_graph.db.get("current", {})
    flow_graph.db["current"]["skill_with_consumption"] = skill_data.get("skill_with_consumption", {})

    print(f"Running trajectory explanation...")
    flow_graph.explain_trajectory(env_states, actions, goal_state)

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
