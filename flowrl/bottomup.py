import argparse
from flowrl.llm.flow import Flow
import sys
from flowrl.ppo_flow import run_ppo
from flowrl.utils.test import gen_frames_hierarchical, render_video
import numpy as np
import shutil
from pathlib import Path
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e9
    )  # Allow scientific notation
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
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ICM
    parser.add_argument("--icm_reward_coeff", type=float, default=1.0)
    parser.add_argument("--train_icm", action="store_true")
    parser.add_argument("--icm_lr", type=float, default=3e-4)
    parser.add_argument("--icm_forward_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_inverse_loss_coef", type=float, default=1.0)
    parser.add_argument("--icm_layer_size", type=int, default=256)
    parser.add_argument("--icm_latent_size", type=int, default=32)
    # E3B
    parser.add_argument("--e3b_reward_coeff", type=float, default=1.0)
    parser.add_argument("--use_e3b", action="store_true")
    parser.add_argument("--e3b_lambda", type=float, default=0.1)

    # Flow Params
    # path for rl, changes every iteration
    parser.add_argument("--module_path", type=str, default=None)
    # path for flow graph generation
    parser.add_argument(
        "--graph-path",
        type=str,
        default="$PROJECT_DIR/exp/bottom_up/",
    )
    # what sucdess rate to achieve before optimizing next node
    parser.add_argument("--success_state_rate", type=float, default=0.8)
    parser.add_argument(
        "--current_i",
        type=int,
        default=0,
        help="Previous node index to load from",
    )
    parser.add_argument(
        "--previous_i",
        type=int,
        default=None,
        help="Previous node index to load from",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=40,
        help="Maximum number of nodes to generate",
    )

    parser.add_argument(
        "--trajectory_analysis", action=argparse.BooleanOptionalAction, default=True
    )

    args, rest_args = parser.parse_known_args(sys.argv[1:])

    args.graph_path = args.graph_path.replace(
        "$PROJECT_DIR", str(Path(__file__).parent.parent)
    )
    graph_path = Path(args.graph_path)

    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.use_e3b:
        assert args.train_icm
        assert args.icm_reward_coeff == 0
    if args.seed is None:
        args.seed = np.random.randint(2**31)

    flow_graph = Flow(args)

    while flow_graph.current_i < args.max_nodes:
        print(f"Current iteration: {flow_graph.current_i}")
        print(f"Previous iteration: {flow_graph.previous_i}")

        # Clean up any stale files from previous runs of this iteration
        current_iter_dir = Path(args.graph_path) / str(flow_graph.current_i)
        if current_iter_dir.exists():
            print(
                f"Cleaning up existing directory for iteration {flow_graph.current_i}"
            )
            shutil.rmtree(current_iter_dir)

        # Create checkpoint BEFORE any skill generation - this represents the clean state
        # loaded from previous_i, ready to start current_i fresh
        checkpoint_name = f"before_skill_{flow_graph.current_i}"
        flow_graph.create_checkpoint(checkpoint_name)

        # Generate the next skill
        skill_name, skill_data = flow_graph.next_skill()
        flow_graph.add_skill(skill_name, skill_data)

        # Build dependency graph for this skill and write code
        flow_graph.build_skill_dependency_graph(skill_name)
        flow_graph.write_code()

        args.module_path = str(
            Path(args.graph_path)
            / str(flow_graph.current_i)
            / f"{flow_graph.current_i}.py"
        )
        args.prev_module_path = (
            str(
                Path(args.graph_path)
                / str(flow_graph.previous_i)
                / f"{flow_graph.previous_i}.py"
            )
            if flow_graph.previous_i is not None
            else None
        )
        if args.trajectory_analysis:
            args.success_state_rate = 0.05
        else:
            args.success_state_rate = 0.8
        out, success = run_ppo(args, training_state_i=len(flow_graph.execution_order))

        if not success:
            print(
                f"Skill {skill_name} at iteration {flow_graph.current_i} failed to train"
            )
            print("Restoring from checkpoint...")

            # Clean up the failed iteration directory before restoring
            current_iter_dir = Path(args.graph_path) / str(flow_graph.current_i)
            if current_iter_dir.exists():
                print(f"Cleaning up failed iteration directory: {current_iter_dir}")
                shutil.rmtree(current_iter_dir)

            flow_graph.restore_checkpoint(checkpoint_name)
            continue

        # Generate frames and test the skill
        frames, states, env_states, actions = gen_frames_hierarchical(
            policy_path=Path(args.graph_path)
            / str(flow_graph.current_i)
            / f"{flow_graph.current_i}_policies",
            max_num_frames=2000,
            goal_state=len(flow_graph.execution_order),
        )

        if args.trajectory_analysis:
            # Analyze trajectory and update knowledge base
            flow_graph.explain_trajectory(
                env_states, actions, len(flow_graph.execution_order) - 1
            )

            # Rebuild dependency graph since skill requirements may have changed
            flow_graph.build_skill_dependency_graph(skill_name)
            flow_graph.write_code()

            # Update module paths - previous is now the same as current since skill was partially trained
            args.prev_module_path = args.module_path

            # Test final success rate
            args.success_state_rate = 0.8
            out, success = run_ppo(
                args, training_state_i=len(flow_graph.execution_order)
            )

            if not success:
                print(
                    f"Skill {skill_name} at iteration {flow_graph.current_i} failed final training"
                )
                print("Restoring from checkpoint...")

                # Clean up the failed iteration directory before restoring
                current_iter_dir = Path(args.graph_path) / str(flow_graph.current_i)
                if current_iter_dir.exists():
                    print(f"Cleaning up failed iteration directory: {current_iter_dir}")
                    shutil.rmtree(current_iter_dir)

                flow_graph.restore_checkpoint(checkpoint_name)
                continue

        flow_graph.update_db(out, skill_name, skill_data)

        # Create and save successful checkpoint to disk, then clean up
        success_checkpoint_name = (
            f"successful_skill_{flow_graph.current_i}_{skill_name}"
        )
        flow_graph.create_checkpoint(success_checkpoint_name)
        flow_graph.save_checkpoint_to_disk(success_checkpoint_name)
        flow_graph.cleanup_checkpoint(
            success_checkpoint_name
        )  # Remove from memory after saving to disk
        flow_graph.cleanup_checkpoint(
            checkpoint_name
        )  # Clean up the original checkpoint

        # Save video and advance to next iteration
        video_path = Path(args.graph_path) / str(flow_graph.current_i) / "video.mp4"
        render_video(frames, states, video_path=video_path)

        # Save skill data to file
        skill_data_path = (
            Path(args.graph_path) / str(flow_graph.current_i) / "skills.json"
        )
        with open(skill_data_path, "w") as f:
            import json

            json.dump({"data": flow_graph.skills}, f, indent=2)

        flow_graph.previous_i = flow_graph.current_i
        flow_graph.current_i += 1
