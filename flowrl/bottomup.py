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
        "--total_timesteps", type=lambda x: int(float(x)), default=1e7
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
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
        default=20,
        help="Maximum number of nodes to generate",
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
        print(f"Current node: {flow_graph.current_i}")
        print(f"Previous node: {flow_graph.previous_i}")
        current_i_save = flow_graph.current_i
        previoius_i_save = flow_graph.previous_i

        # Generate the next node
        next_node_code = flow_graph.next_node()
        flow_graph.add_node(next_node_code)

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
        args.success_state_rate = 0.05
        out, success = run_ppo(args, training_state_i=flow_graph.current_i + 1)

        if not success:
            print(f"Node {flow_graph.current_i} failed to train")
            flow_graph.current_i = current_i_save
            continue
        code_string = "\n".join(next_node_code)

        frames, states, env_states, actions = gen_frames_hierarchical(
            policy_path=Path(args.graph_path)
            / str(flow_graph.current_i)
            / f"{flow_graph.current_i}_policies",
            max_num_frames=2000,
            goal_state=flow_graph.current_i + 1,
        )
        missing_resources = flow_graph.explain_trajectory(env_states, actions)

        if len(missing_resources) > 0:
            flow_graph.precollect_resources(missing_resources)
            flow_graph.write_code()
            args.module_path = str(
                Path(args.graph_path)
                / str(flow_graph.current_i)
                / f"{flow_graph.current_i}.py"
            )
            # load from previous checkpoint since we're essencially continuining trainig with new nodes but same skills
            args.prev_module_path = args.module_path
        args.success_state_rate = 0.8
        out, success = run_ppo(args, training_state_i=flow_graph.current_i + 1)

        if not success:
            print(f"Node {flow_graph.current_i} failed to train")
            flow_graph.current_i = current_i_save
            continue

        flow_graph.update_db(out, code_string)

        video_path = Path(args.graph_path) / str(flow_graph.current_i) / "video.mp4"
        render_video(frames, states, video_path=video_path)

        flow_graph.previous_i = flow_graph.current_i
        flow_graph.current_i += 1
