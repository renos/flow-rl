"""
Bottom-up skill learning with parallel training.

This version uses the parallel training infrastructure to train multiple
independent skills simultaneously using tmux orchestration.
"""

import argparse
import os
from flowrl.llm.flow import Flow
import sys
from pathlib import Path
import json
import time
import threading
import numpy as np
from flowrl.utils.test import gen_frames_hierarchical, render_video

# Parallel training components
from flowrl.parallel.scheduler import SkillScheduler
from flowrl.parallel.training_setup import prepare_training_run
from flowrl.parallel.training_processor import process_completed_training, load_global_checkpoint


def get_skill_dependencies(skill_name: str, all_skills: dict, training_skills: dict) -> list:
    """
    Determine which skills the given skill depends on.

    Uses the skill's requirements to find which skills produce those items.

    Args:
        skill_name: Name of skill to analyze
        all_skills: Dict of all known skills (completed + training)
        training_skills: Dict of currently training skills

    Returns:
        List of skill names this skill depends on
    """
    if skill_name not in all_skills:
        return []

    skill_data = all_skills[skill_name]
    requirements = skill_data.get("skill_with_consumption", {}).get("requirements", {})

    dependencies = []

    # For each item this skill requires
    for req_item in requirements.keys():
        # Find which skill produces this item
        for other_skill_name, other_skill_data in all_skills.items():
            if other_skill_name == skill_name:
                continue

            gain = other_skill_data.get("skill_with_consumption", {}).get("gain", {})
            if req_item in gain:
                dependencies.append(other_skill_name)
                break

    # Remove duplicates
    return list(set(dependencies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # Exploration
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    # Flow params
    parser.add_argument("--graph-path", type=str, default="$PROJECT_DIR/exp/bottom_up_parallel/")
    parser.add_argument("--success_state_rate", type=float, default=0.8)
    parser.add_argument(
        "--llm_name",
        type=str,
        default=None,
        help='LLM model to use (e.g., "gpt-5", "gpt-4o-mini"). If not specified, uses default (gpt-5).'
    )
    parser.add_argument("--max_nodes", type=int, default=40, help="Maximum number of skills to generate")
    parser.add_argument("--trajectory_analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame_gen_envs", type=int, default=32)
    parser.add_argument("--frame_gen_max_frames", type=int, default=1000)
    parser.add_argument("--skill_timesteps_budget", type=lambda x: int(float(x)), default=10_000_000,
                        help="Combined budget (timesteps) per skill across Phase A+B")
    parser.add_argument("--target_skill", type=str, required=True,
                        help="Target skill name from the knowledgebase to train towards")

    # Iteration tracking (compatibility with Flow)
    parser.add_argument("--current_i", type=int, default=0,
                        help="Generation index for Flow graph artifacts")
    parser.add_argument("--previous_i", type=int, default=None,
                        help="Optional: load previous iteration checkpoint (unused in parallel mode)")

    # Parallel training params
    parser.add_argument("--max_parallel_skills", type=int, default=3,
                       help="Maximum number of skills to train in parallel")
    parser.add_argument("--scheduler_poll_interval", type=int, default=30,
                       help="Seconds between scheduler status checks")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum retries for failed skills")
    parser.add_argument("--gpu_ids", type=str, default=None,
                       help="Comma-separated GPU IDs to use (one skill per GPU). If omitted, uses current CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--conda_env", type=str, default="jax",
                        help="Name of the conda environment to activate inside tmux before training (e.g., 'jax').")

    # PPO Flow variant selection
    parser.add_argument("--ppo_flow_variant", type=str, default="ppo_flow", choices=["ppo_flow", "ppo_flow_w_reset"],
                       help="Which PPO flow variant to use for training")

    # Parameters specific to ppo_flow_w_reset
    parser.add_argument("--latest_reset_prob", type=float, default=0.0,
                       help="Probability of resetting to latest state (only for ppo_flow_w_reset)")
    parser.add_argument("--progressive_reset_curriculum", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable progressive reset curriculum (only for ppo_flow_w_reset)")
    parser.add_argument("--progressive_reset_threshold", type=float, default=0.2,
                       help="Threshold for advancing in progressive curriculum (only for ppo_flow_w_reset)")
    parser.add_argument("--per_skill_balance", action=argparse.BooleanOptionalAction, default=True,
                       help="Balance resets per skill (only for ppo_flow_w_reset)")
    parser.add_argument("--per_skill_balance_threshold", type=int, default=64,
                       help="Threshold for per-skill balancing (only for ppo_flow_w_reset)")
    parser.add_argument("--per_skill_balance_cap", type=float, default=4.0,
                       help="Cap for per-skill balancing (only for ppo_flow_w_reset)")
    parser.add_argument("--success_rate_ema_alpha", type=float, default=0.15,
                       help="EMA alpha for success rate tracking (only for ppo_flow_w_reset)")

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    args.graph_path = args.graph_path.replace("$PROJECT_DIR", str(Path(__file__).parent.parent))
    graph_path = Path(args.graph_path)
    graph_path.mkdir(parents=True, exist_ok=True)

    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    # Initialize Flow graph
    flow_graph = Flow(args)

    # Load previously completed skills from scheduler state (for restart support)
    scheduler_state_file = graph_path / "scheduler_state.json"
    if scheduler_state_file.exists():
        print(f"Loading completed skills from scheduler state...")
        with open(scheduler_state_file, 'r') as f:
            scheduler_state = json.load(f)
        loaded_count = 0
        for skill_id, skill_info in scheduler_state.get("skills", {}).items():
            if skill_info.get("status") == "completed":
                skill_name = skill_info.get("skill_name")
                skill_data = skill_info.get("skill_data")
                if skill_name and skill_data:
                    flow_graph.skills[skill_name] = skill_data
                    loaded_count += 1
        if loaded_count > 0:
            print(f"  ✓ Loaded {loaded_count} completed skills into Flow graph")

    # Initialize scheduler
    # Determine GPU list: CLI arg > env CUDA_VISIBLE_DEVICES > fallback ["0"]
    if args.gpu_ids is not None:
        gpu_list = [g.strip() for g in args.gpu_ids.split(',') if g.strip() != '']
    else:
        env_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
        if env_cuda and env_cuda.strip():
            gpu_list = [g.strip() for g in env_cuda.split(',') if g.strip() != '']
        else:
            gpu_list = ["0"]

    scheduler = SkillScheduler(
        base_dir=graph_path,
        max_parallel=args.max_parallel_skills,
        poll_interval=args.scheduler_poll_interval,
        max_retries=args.max_retries,
        gpu_ids=gpu_list,
        conda_env=args.conda_env,
        skill_budget_timesteps=args.skill_timesteps_budget,
        frame_gen_envs=args.frame_gen_envs,
        frame_gen_max_frames=args.frame_gen_max_frames,
        env_name=args.env_name,
        ppo_flow_variant=args.ppo_flow_variant,
        latest_reset_prob=args.latest_reset_prob,
        progressive_reset_curriculum=args.progressive_reset_curriculum,
        progressive_reset_threshold=args.progressive_reset_threshold,
        per_skill_balance=args.per_skill_balance,
        per_skill_balance_threshold=args.per_skill_balance_threshold,
        per_skill_balance_cap=args.per_skill_balance_cap,
        success_rate_ema_alpha=args.success_rate_ema_alpha,
    )

    # Set up callbacks
    def on_skill_complete_callback(skill_name: str):
        """Called when a skill completes training."""
        print(f"\n{'='*70}")
        print(f"Skill '{skill_name}' completed training")
        print(f"{'='*70}\n")

        # Move from training_skills to skills
        flow_graph.on_skill_complete(skill_name)

        # Attach metrics from scheduler processing results to Flow.db for prompts
        info = scheduler.get_skill_info(skill_name)
        if info and info.get("processing_results"):
            metrics = info["processing_results"].get("metrics", {})
            if metrics:
                try:
                    flow_graph.db.setdefault("skills", {})
                    flow_graph.db["skills"].setdefault(skill_name, {})
                    flow_graph.db["skills"][skill_name].setdefault("metrics", {})
                    flow_graph.db["skills"][skill_name]["metrics"].update(metrics)
                except Exception:
                    # Best-effort attachment; make debugging explicit
                    raise

        # TODO: Generate frames and video for completed skill
        # This would require access to the skill's expert index and policies

    def analyze_trajectories_callback(skill_name: str, skill_data: dict, run_folder: Path, expert_idx: int, base_dir: Path, frame_gen_envs: int, frame_gen_max_frames: int):
        # Collect trajectories from saved policies
        policies_path = run_folder / f"{expert_idx}_policies"
        # Build goal_state from execution_plan length
        exec_plan = skill_data.get("execution_plan", []) or []
        goal_state = len(exec_plan) - 1 if len(exec_plan) > 0 else 0
        frames, states, env_states, actions = gen_frames_hierarchical(
            policy_path=policies_path,
            max_num_frames=frame_gen_max_frames,
            goal_state=goal_state + 1,
            num_envs=frame_gen_envs,
        )

        # Prepare Flow's current context minimally for analysis
        flow_graph.db["current"] = flow_graph.db.get("current", {})
        flow_graph.db["current"]["skill_with_consumption"] = skill_data.get("skill_with_consumption", {})

        # Run trajectory explanation to produce updated_requirements/consumption/gain
        flow_graph.explain_trajectory(env_states, actions, goal_state)

        # Apply updates directly to this skill definition (no need to persist in Flow.skills)
        skill_updates = flow_graph.db.get("current", {}).get("skill_update_results", {})
        if skill_updates:
            swc = skill_data.setdefault("skill_with_consumption", {})
            if "updated_requirements" in skill_updates:
                swc["requirements"] = skill_updates["updated_requirements"]
            if "updated_consumption" in skill_updates:
                swc["consumption"] = skill_updates["updated_consumption"]
            if "updated_gain" in skill_updates:
                swc["gain"] = skill_updates["updated_gain"]

        # Rebuild dependency graph and updated execution plan for this skill

        # Sync flow_graph.skills with latest state from scheduler before dependency resolution
        scheduler_state_file = base_dir / "scheduler_state.json"
        if scheduler_state_file.exists():
            with open(scheduler_state_file, 'r') as f:
                scheduler_state = json.load(f)
            for skill_id, skill_info in scheduler_state.get("skills", {}).items():
                if skill_info.get("status") == "completed":
                    skill_name_from_scheduler = skill_info.get("skill_name")
                    skill_data_from_scheduler = skill_info.get("skill_data")
                    if skill_name_from_scheduler and skill_data_from_scheduler:
                        # Update with latest skill data (includes trajectory analysis updates)
                        flow_graph.skills[skill_name_from_scheduler] = skill_data_from_scheduler

        # Temporarily ensure skill exists in flow_graph.skills for planning only
        temp_already = skill_name in flow_graph.skills
        prev_entry = flow_graph.skills.get(skill_name)
        flow_graph.skills[skill_name] = skill_data
        flow_graph.build_skill_dependency_graph(skill_name)
        execution_plan = []
        for s_name, count in getattr(flow_graph, 'execution_order', []):
            if s_name == skill_name:
                funcs = skill_data.get("functions", [])
            else:
                funcs = flow_graph.skills.get(s_name, {}).get("functions", [])
            execution_plan.append({
                "skill_name": s_name,
                "count": count,
                "functions": funcs,
            })
        skill_data["execution_plan"] = execution_plan
        if temp_already:
            flow_graph.skills[skill_name] = prev_entry
        else:
            del flow_graph.skills[skill_name]

        return skill_data

    # Wrap prepare_training_run to inject env_name for correct codegen imports
    def prepare_training_run_with_env(**kwargs):
        return prepare_training_run(env_name=args.env_name, **kwargs)

    scheduler.set_callbacks(
        on_skill_complete=on_skill_complete_callback,
        prepare_training_run=prepare_training_run_with_env,
        process_completed_training=process_completed_training,
        analyze_trajectories=analyze_trajectories_callback,
    )

    # Start scheduler in background thread
    def run_scheduler():
        """Run scheduler with current completed skills."""
        # Build completed_skills dict with expert indices
        completed_skills_dict = {}

        # Load from global checkpoint if exists
        global_ckpt = load_global_checkpoint(graph_path)
        for skill_name, skill_info in global_ckpt.get("skills", {}).items():
            completed_skills_dict[skill_name] = {
                "expert_idx": skill_info["expert_idx"],
                "path": skill_info.get("path", f"skills/{skill_info['expert_idx']}_{skill_name}/")
            }

        # Keep exceptions visible instead of silently dying in the background thread
        try:
            scheduler.run(completed_skills_dict)
        except Exception:
            import traceback
            print("\n[Scheduler Thread] Fatal error:")
            traceback.print_exc()
            raise

    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

    # Event-driven generation loop
    total_skills_generated = 0
    frontier_blocked = False
    last_completed_count = 0

    print(f"\n{'='*70}")
    print(f"Starting Parallel Bottom-Up Skill Learning")
    print(f"  Target skill: {args.target_skill}")
    print(f"  Max skills: {args.max_nodes}")
    print(f"  Max parallel: {min(args.max_parallel_skills, len(gpu_list))} (GPUs: {','.join(gpu_list)})")
    print(f"  Conda env: {args.conda_env}")
    print(f"  Base directory: {graph_path}")
    print(f"{'='*70}\n")

    while total_skills_generated < args.max_nodes or not scheduler._is_complete():
        # Capacity check: only propose when there is actual free capacity (running + waiting < max)
        running_cnt = len(scheduler.state["currently_running"])
        waiting_cnt = sum(1 for s in scheduler.state["skills"].values() if s["status"] == "waiting")
        free_slots = max(args.max_parallel_skills - (running_cnt + waiting_cnt), 0)

        # Try to generate if not blocked and have free capacity
        if free_slots > 0 and not frontier_blocked:
            currently_training = scheduler.get_running_skill_names()

            print(f"\n{'─'*70}")
            print(f"Attempting to generate skill #{total_skills_generated + 1}...")
            print(f"  Capacity: running={running_cnt}, waiting={waiting_cnt}, free={free_slots}/{args.max_parallel_skills}")
            print(f"  Currently training: {currently_training if currently_training else 'none'}")
            print(f"{'─'*70}")

            # Generate next skill from knowledgebase based on target skill
            if True:
                skill_name, skill_data = flow_graph.select_next_skill_from_knowledgebase(args.target_skill)

                # Check if all skills are completed or training
                if skill_name is None:
                    print(f"  → All skills for target '{args.target_skill}' are completed or in training")
                    break

                # Note: skill is now in flow_graph.training_skills

                # Build full execution plan like sequential path (temporary include current skill)

                # Sync flow_graph.skills with latest state from scheduler
                # This ensures dependency resolver sees updated skills from trajectory analysis
                scheduler_state_file = graph_path / "scheduler_state.json"
                if scheduler_state_file.exists():
                    with open(scheduler_state_file, 'r') as f:
                        scheduler_state = json.load(f)
                    for skill_id, skill_info in scheduler_state.get("skills", {}).items():
                        if skill_info.get("status") == "completed":
                            skill_name_from_scheduler = skill_info.get("skill_name")
                            skill_data_from_scheduler = skill_info.get("skill_data")
                            if skill_name_from_scheduler and skill_data_from_scheduler:
                                # Update with latest skill data (includes trajectory analysis updates)
                                flow_graph.skills[skill_name_from_scheduler] = skill_data_from_scheduler

                temp_already = skill_name in flow_graph.skills
                prev_entry = flow_graph.skills.get(skill_name)
                flow_graph.skills[skill_name] = skill_data
                if True:
                    flow_graph.build_skill_dependency_graph(skill_name)
                    execution_plan = []
                    for s_name, count in getattr(flow_graph, 'execution_order', []):
                        if s_name == skill_name:
                            funcs = skill_data.get("functions", [])
                        else:
                            funcs = flow_graph.skills.get(s_name, {}).get("functions", [])
                        execution_plan.append({
                            "skill_name": s_name,
                            "count": count,
                            "functions": funcs,
                        })
                    skill_data["execution_plan"] = execution_plan
                if True:
                    # Restore skills dict to avoid marking current skill as completed
                    if temp_already:
                        flow_graph.skills[skill_name] = prev_entry
                    else:
                        del flow_graph.skills[skill_name]

                # Determine dependencies (check against ALL skills: completed + training)
                all_skills = {**flow_graph.skills, **flow_graph.training_skills}
                dependencies = get_skill_dependencies(skill_name, all_skills, flow_graph.training_skills)

                # FRONTIER CHECK: Does this skill depend on currently training skills?
                frontier_blocked = flow_graph.check_frontier_blocked(skill_name, skill_data)

                if frontier_blocked:
                    # Depends on training skills - frontier blocked
                    training_deps = [d for d in dependencies if d in flow_graph.training_skills]
                    print(f"  → Generated: {skill_name}")
                    print(f"  → Dependencies: {dependencies}")
                    print(f"  → FRONTIER BLOCKED: Depends on training skills {training_deps}")
                    print(f"  → Pausing generation until skills complete...")

                    # Add to scheduler (will launch when deps satisfied)
                    scheduler.add_skill(skill_name, skill_data, dependencies)
                    total_skills_generated += 1

                    # Advance Flow's generation index to avoid overwriting artifacts
                    try:
                        flow_graph.current_i += 1
                    except Exception:
                        pass
                else:
                    # Independent or depends only on completed skills
                    print(f"  → Generated: {skill_name}")
                    print(f"  → Dependencies: {dependencies if dependencies else 'none'}")
                    print(f"  → Frontier OPEN")

                    # Add to scheduler queue
                    scheduler.add_skill(skill_name, skill_data, dependencies)
                    total_skills_generated += 1

                    # Advance Flow's generation index to avoid overwriting artifacts
                    try:
                        flow_graph.current_i += 1
                    except Exception:
                        pass

                    # Do not generate more than current free capacity; wait for scheduler to update

            # except Exception as e:
            #     print(f"Error generating skill: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     break

        # Monitor for completions
        completed_count = sum(1 for s in scheduler.state["skills"].values()
                            if s["status"] == "completed")

        if completed_count > last_completed_count:
            last_completed_count = completed_count
            print(f"\n{'='*70}")
            print(f"Progress: {completed_count}/{total_skills_generated} skills completed")
            print(f"{'='*70}")

            # Frontier may be unblocked now
            if frontier_blocked:
                frontier_blocked = False
                print(f"  → Frontier unblocked - resuming generation")

        # Check if we're done - either max_nodes reached or target skill completed
        if total_skills_generated >= args.max_nodes and scheduler._is_complete():
            print(f"\n{'='*70}")
            print(f"Max skills ({args.max_nodes}) reached and all training complete!")
            print(f"{'='*70}\n")
            break

        # Check if target skill is completed
        if args.target_skill in flow_graph.skills and scheduler._is_complete():
            print(f"\n{'='*70}")
            print(f"Target skill '{args.target_skill}' completed successfully!")
            print(f"{'='*70}\n")
            break

        # Wait before next poll
        time.sleep(args.scheduler_poll_interval)

    # Signal scheduler to stop and wait for thread to finish
    print("\nWaiting for all skills to complete...")
    try:
        scheduler.stop()
    except Exception:
        pass
    scheduler_thread.join()

    print("\n" + "="*70)
    print("Parallel Bottom-Up Training Complete!")
    print(f"  Total skills: {total_skills_generated}")
    print(f"  Completed: {sum(1 for s in scheduler.state['skills'].values() if s['status'] == 'completed')}")
    print(f"  Failed: {sum(1 for s in scheduler.state['skills'].values() if s['status'] == 'failed')}")
    print(f"  Global checkpoint: {graph_path}/checkpoints/global_latest.pkl")
    print(f"  Scheduler state: {graph_path}/scheduler_state.json")
    print("="*70 + "\n")
