"""
Training run setup for parallel skill learning.

This module handles:
1. Building global→local expert index remapping
2. Creating training_runs/ folder structure
3. Loading dependency expert networks
4. Building initial seed checkpoint (Orbax) from dependency experts
5. Generating .py modules with remapping logic
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManager,
    CheckpointManagerOptions,
)
from flax.training import orbax_utils
import yaml
import re


def build_remapping(
    dependency_skill_names: List[str],
    new_skill_global_expert_idx: int,
    completed_skills: Dict[str, Dict]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build global→local and local→global expert index mappings.

    Args:
        dependency_skill_names: List of skill names this skill depends on
        new_skill_global_expert_idx: Global expert index for the new skill
        completed_skills: Dict of completed skills with expert_idx metadata

    Returns:
        (global_to_local, local_to_global) mappings

    Example:
        >>> deps = ["Collect_Wood", "Collect_Stone"]
        >>> completed = {
        ...     "Collect_Wood": {"expert_idx": 0},
        ...     "Collect_Stone": {"expert_idx": 2}
        ... }
        >>> g2l, l2g = build_remapping(deps, 7, completed)
        >>> g2l
        {0: 0, 2: 1, 7: 2}
        >>> l2g
        {0: 0, 1: 2, 2: 7}
    """
    # Collect global expert indices from dependencies
    global_experts = []

    for dep_skill_name in dependency_skill_names:
        if dep_skill_name not in completed_skills:
            raise ValueError(f"Dependency skill '{dep_skill_name}' not found in completed_skills")

        dep_expert_idx = completed_skills[dep_skill_name]["expert_idx"]
        global_experts.append((dep_expert_idx, dep_skill_name))

    # Add the new skill's expert
    global_experts.append((new_skill_global_expert_idx, "NEW"))

    # Sort by global index for consistent ordering
    global_experts.sort(key=lambda x: x[0])

    # Build mappings
    global_to_local = {}
    local_to_global = {}

    for local_idx, (global_idx, _) in enumerate(global_experts):
        global_to_local[global_idx] = local_idx
        local_to_global[local_idx] = global_idx

    return global_to_local, local_to_global


def load_expert_params(expert_path: Path) -> Dict:
    """
    Load expert network parameters from skills/ folder.

    Args:
        expert_path: Path to expert_X_policy/ folder

    Returns:
        Expert parameters dict
    """
    params_file = expert_path / "params.pkl"

    if not params_file.exists():
        raise FileNotFoundError(f"Expert params not found at {params_file}")

    with open(params_file, 'rb') as f:
        data = pickle.load(f)

    # Backward-compat: allow old format {"params": ..., "metadata": ...}
    if isinstance(data, dict) and "params" in data and any(k in data for k in ("metadata",)):
        return data["params"]
    # New format: normalized dict with keys like 'actor_network', 'critic_network'
    return data


def load_expert_metadata(expert_path: Path) -> Dict:
    """
    Load expert metadata (skill_name, total_frames, etc.).

    Args:
        expert_path: Path to expert_X_policy/ folder

    Returns:
        Metadata dict with keys: skill_name, global_expert_idx, total_frames
    """
    # Prefer external metadata.json
    meta_json = expert_path / "metadata.json"
    if meta_json.exists():
        import json
        with open(meta_json, 'r') as f:
            return json.load(f)

    # Backward-compat: older format stored metadata within params.pkl
    params_file = expert_path / "params.pkl"
    if params_file.exists():
        with open(params_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "metadata" in data:
            return data["metadata"]
    return {}


def prepare_training_run(
    skill_name: str,
    skill_data: Dict,
    global_expert_idx: int,
    dependency_skill_names: List[str],
    completed_skills: Dict[str, Dict],
    base_dir: Path,
    initialize_expert_fn: Optional[callable] = None,
    env_name: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Prepare a training run with remapped expert indices.

    This function:
    1. Builds global→local remapping for efficient MoE
    2. Creates training_runs/{skill}/ folder structure
    3. Loads dependency expert networks from skills/ folders
    4. Creates MoE checkpoint with remapped (contiguous) indices
    5. Generates .py module with injected remapping logic

    Args:
        skill_name: Name of the skill to train
        skill_data: Skill data from Flow.next_skill() (includes code, skill_with_consumption)
        global_expert_idx: Global expert index assigned to this skill
        dependency_skill_names: List of dependency skill names
        completed_skills: Dict of completed skills with expert_idx and paths
        base_dir: Base experiment directory (e.g., exp/bottom_up/)
        initialize_expert_fn: Optional function to initialize new expert params
        env_name: Optional string identifying environment (e.g., 'Craftax-Symbolic-v1' or 'Fabrax-Symbolic-v1')

    Returns:
        (run_folder, module_path, policies_folder) paths

Example:
        >>> run_folder, module_path, policies_folder = prepare_training_run(
        ...     skill_name="Make_Pickaxe",
        ...     skill_data={"code": "...", "skill_with_consumption": {...}},
        ...     global_expert_idx=2,
        ...     dependency_skill_names=["Collect_Wood", "Collect_Stone"],
        ...     completed_skills={
        ...         "Collect_Wood": {"expert_idx": 0, "path": "skills/0_Collect_Wood/"},
        ...         "Collect_Stone": {"expert_idx": 1, "path": "skills/1_Collect_Stone/"}
        ...     },
        ...     base_dir=Path("exp/bottom_up/"),
        ...     env_name="Craftax-Symbolic-v1",
        ... )
        >>> # Creates: training_runs/skill_2_Make_Pickaxe/
        >>> #   - 2.py (with remapping: {0→0, 1→1, 2→2})
        >>> #   - 2_policies/checkpoint_0.pkl (MoE with 3 experts)
    """
    base_dir = Path(base_dir)

    # 1. Build remapping
    # Only include skills that are actually referenced in the execution plan
    # (not abstract graph dependencies that aren't used in training)
    exec_plan = skill_data.get("execution_plan", []) or []
    plan_skill_names = [
        step.get("skill_name") for step in exec_plan
        if step.get("skill_name") and step.get("skill_name") != skill_name
    ]
    # Only include ones we know about in completed_skills
    plan_existing = [s for s in plan_skill_names if s in completed_skills]
    # Use only skills that appear in the execution plan - these are the actual MoE experts needed
    required_skill_names = list(dict.fromkeys(plan_existing))

    global_to_local, local_to_global = build_remapping(
        required_skill_names,
        global_expert_idx,
        completed_skills
    )

    # 2. Create training_runs folder structure
    # Remove special characters that cause issues with shell commands
    safe_skill_name = skill_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    run_folder = base_dir / "training_runs" / f"skill_{global_expert_idx}_{safe_skill_name}"
    run_folder.mkdir(parents=True, exist_ok=True)

    policies_folder = run_folder / f"{global_expert_idx}_policies"
    policies_folder.mkdir(parents=True, exist_ok=True)

    # 3. Load dependency experts and build MoE with REMAPPED indices
    moe_params = {}
    initial_frame_counts = {}

    # Get sorted list of (local_idx, global_idx) pairs
    local_global_pairs = sorted(local_to_global.items())

    for local_idx, global_idx in local_global_pairs:
        if global_idx == global_expert_idx:
            # Check if this expert already exists (continuation training)
            safe_skill = skill_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
            existing_expert_path = base_dir / "skills" / f"{global_idx}_{safe_skill}" / f"expert_{global_idx}_policy"

            if existing_expert_path.exists():
                # Continuation training - load existing expert and frame count
                print(f"  [Continuation] Loading existing expert {global_idx} from {existing_expert_path}")
                expert_params = load_expert_params(existing_expert_path)
                expert_metadata = load_expert_metadata(existing_expert_path)
                moe_params[f"expert_{local_idx}"] = expert_params
                initial_frame_counts[global_idx] = expert_metadata.get("total_frames", 0)
                print(f"  [Continuation] Loaded initial frames: {initial_frame_counts[global_idx]:,}")
            else:
                # New expert - will be initialized by training code or use provided function
                if initialize_expert_fn is not None:
                    moe_params[f"expert_{local_idx}"] = initialize_expert_fn()
                else:
                    # Placeholder - training code will initialize
                    moe_params[f"expert_{local_idx}"] = None
                initial_frame_counts[global_idx] = 0
        else:
            # Load from completed skills
            # Find which skill owns this expert
            owning_skill = None
            for skill, skill_info in completed_skills.items():
                if skill_info["expert_idx"] == global_idx:
                    owning_skill = skill
                    break

            if owning_skill is None:
                raise ValueError(
                    f"Expert {global_idx} needed for dependency but not found in completed_skills"
                )

            # Get skill path
            if "path" in completed_skills[owning_skill]:
                skill_folder = base_dir / completed_skills[owning_skill]["path"]
            else:
                # Construct path from skill name and expert index
                from flowrl.parallel.training_processor import sanitize_skill_name
                safe_dep_name = sanitize_skill_name(owning_skill)
                skill_folder = base_dir / "skills" / f"{global_idx}_{safe_dep_name}"

            expert_path = skill_folder / f"expert_{global_idx}_policy"

            # Load expert params and metadata
            expert_params = load_expert_params(expert_path)
            expert_metadata = load_expert_metadata(expert_path)

            moe_params[f"expert_{local_idx}"] = expert_params
            initial_frame_counts[global_idx] = expert_metadata.get("total_frames", 0)

    # 4. Save initial remap sidecar (for post-processing)
    # Write remapping sidecar for post-processing and (future) seeding
    remap_sidecar = {
        "global_to_local": global_to_local,
        "local_to_global": local_to_global,
        "initial_frame_counts": initial_frame_counts,
        "skill_name": skill_name,
        "global_expert_idx": global_expert_idx,
        "dependencies": required_skill_names,
        "skill_with_consumption": skill_data.get("skill_with_consumption", {}),
    }
    import json
    with open(policies_folder / "remap.json", 'w') as f:
        json.dump(remap_sidecar, f, indent=2)

    # 4b. Optionally create an initial seed checkpoint (Orbax) if there are dependencies
    if required_skill_names:
        # Build a seed params tree matching ppo_flow expectations: {'params': {'params': {...}}}
        seed_params_root: Dict = {"params": {"params": {}}}
        params_map = seed_params_root["params"]["params"]

        # For each dependency, load normalized expert params and place at correct local index
        for dep_name in required_skill_names:
            if dep_name not in completed_skills:
                raise ValueError(f"Dependency '{dep_name}' missing from completed_skills for seeding")
            dep_global = completed_skills[dep_name]["expert_idx"]
            local_idx = global_to_local.get(dep_global)
            if local_idx is None:
                raise ValueError(f"No local index for dependency expert {dep_global}")

            # Load normalized expert params
            if "path" in completed_skills[dep_name]:
                skill_folder = base_dir / completed_skills[dep_name]["path"]
            else:
                safe_dep_name = dep_name.replace(' ', '_').replace('/', '_')
                skill_folder = base_dir / "skills" / f"{dep_global}_{safe_dep_name}"
            expert_path = skill_folder / f"expert_{dep_global}_policy"
            normalized = load_expert_params(expert_path)

            # Support both normalized format {'actor_network':..., 'critic_network':...}
            # and legacy format with task-indexed keys (actor_networks_0, critic_networks_0)
            actor_sub = None
            critic_sub = None
            if isinstance(normalized, dict):
                actor_sub = normalized.get("actor_network")
                critic_sub = normalized.get("critic_network")
                if actor_sub is None and critic_sub is None:
                    # attempt legacy keys
                    # find any key that startswith actor_networks_ and critic_networks_
                    for k, v in normalized.items():
                        if isinstance(k, str) and k.startswith("actor_networks_"):
                            actor_sub = v
                        elif isinstance(k, str) and k.startswith("critic_networks_"):
                            critic_sub = v

            if actor_sub is None or critic_sub is None:
                raise ValueError(f"Expert params for {dep_name} not in expected format for seeding")

            params_map[f"actor_networks_{local_idx}"] = actor_sub
            params_map[f"critic_networks_{local_idx}"] = critic_sub

        # Save the seed via Orbax to policies/0
        seed_checkpointer = PyTreeCheckpointer()
        options = CheckpointManagerOptions(max_to_keep=1, create=True)
        policies_dir = policies_folder / "policies"
        policies_dir.mkdir(parents=True, exist_ok=True)
        manager = CheckpointManager(str(policies_dir), seed_checkpointer, options)
        save_args = orbax_utils.save_args_from_target(seed_params_root)
        manager.save(0, seed_params_root, save_kwargs={"save_args": save_args})

        # Write a minimal config.yaml so ppo_flow.restore_model() can open it
        cfg_path = policies_folder / "config.yaml"
        seed_cfg = {
            "total_timesteps": 0,
            "trained_timesteps": 0,
            "note": "seed checkpoint generated by training_setup.prepare_training_run()"
        }
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(seed_cfg, f)
        print("SAVED SEEDED CHECKPOINT")
        

    # 5. Generate .py module with remapping
    # Provide dependency→expert mapping to module generator
    dep_expert_map = {dep: completed_skills[dep]["expert_idx"] for dep in dependency_skill_names}

    # Augment mapping with any skills referenced in the execution_plan that already exist
    exec_plan = skill_data.get("execution_plan", []) or []
    for step in exec_plan:
        s_name = step.get("skill_name")
        if s_name and s_name != skill_name and s_name in completed_skills:
            dep_expert_map.setdefault(s_name, completed_skills[s_name]["expert_idx"])

    skill_data_for_module = dict(skill_data)
    skill_data_for_module["dependency_expert_map"] = dep_expert_map

    resolved_env_name = env_name or 'Craftax-Symbolic-v1'
    module_content = generate_module_with_remapping(
        skill_name,
        skill_data_for_module,
        global_to_local,
        local_to_global,
        global_expert_idx,
        env_name=resolved_env_name
    )

    module_path = run_folder / f"{global_expert_idx}.py"
    with open(module_path, 'w') as f:
        f.write(module_content)

    return run_folder, module_path, policies_folder


def generate_module_with_remapping(
    skill_name: str,
    skill_data: Dict,
    global_to_local: Dict[int, int],
    local_to_global: Dict[int, int],
    global_expert_idx: int,
    env_name: str = "Craftax-Symbolic-v1"
) -> str:
    """
    Generate .py module with injected remapping logic.

    Takes LLM-generated code and injects:
    1. GLOBAL_TO_LOCAL and LOCAL_TO_GLOBAL mappings
    2. Wrapper for task_X_network_number() functions to return local indices

    Args:
        skill_name: Name of the skill
        skill_data: Skill data containing LLM-generated code
        global_to_local: Global→local expert index mapping
        local_to_global: Local→global expert index mapping
        global_expert_idx: This skill's global expert index

    Returns:
        Complete module content as string
    """
    # Build header + imports
    if "Fabrax" in env_name:
        import_lines = (
            "from craftax.fabrax.constants import *\n"
            "from craftax.fabrax.envs.craftax_state import Inventory\n"
        )
    else:
        import_lines = (
            "from craftax.craftax.constants import *\n"
            "from craftax.craftax.craftax_state import Inventory\n"
        )

    header = (
        f"# Auto-generated training module for: {skill_name}\n"
        f"# Global expert index: {global_expert_idx}\n"
        f"# This module uses REMAPPED (local) expert indices for efficient MoE training\n\n"
        f"# === REMAPPING METADATA ===\n"
        f"GLOBAL_TO_LOCAL = {global_to_local}\n"
        f"LOCAL_TO_GLOBAL = {local_to_global}\n\n"
        f"{import_lines}"
        f"import jax\n\n"
    )

    exec_plan = skill_data.get("execution_plan", [])
    if not exec_plan:
        raise ValueError("execution_plan missing from skill_data; call Flow.build_skill_dependency_graph() and attach plan before scheduling.")

    # Map skill_name -> global expert idx
    mapping_by_skill: Dict[str, int] = {}
    mapping_by_skill[skill_name] = global_expert_idx
    dep_map = skill_data.get("dependency_expert_map", {})
    for k, v in dep_map.items():
        mapping_by_skill[k] = int(v)

    parts: List[str] = [header]

    for idx, step in enumerate(exec_plan):
        s_name = step.get("skill_name")
        n_count = int(step.get("count", 1))
        functions = step.get("functions", [])
        if not functions or len(functions) < 2:
            raise ValueError(f"Missing functions for step {idx} ({s_name}) in execution_plan")

        # Rename functions and patch default n
        func_is_done = re.sub(r"def\s+task_is_done\(", f"def task_{idx}_is_done(", functions[0])
        func_is_done = re.sub(r"(\w+,\s*n)\):", rf"\1={n_count}):", func_is_done)
        func_reward = re.sub(r"def\s+task_reward\(", f"def task_{idx}_reward(", functions[1])

        # Determine global index for this step and create remapped network_number
        global_idx = mapping_by_skill.get(s_name, global_expert_idx if s_name == skill_name else None)
        if global_idx is None:
            raise ValueError(f"Missing global expert idx for dependency '{s_name}'. Provide dependency_expert_map in skill_data.")
        func_netnum = (
            f"def task_{idx}_network_number():\n"
            f"    return GLOBAL_TO_LOCAL[{int(global_idx)}]\n"
        )

        parts.append(func_is_done + "\n")
        parts.append(func_reward + "\n")
        parts.append(func_netnum + "\n")

    return "".join(parts)
