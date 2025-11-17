"""
Post-training processing for parallel skill learning.

This module handles:
1. Loading final checkpoints from training_runs/
2. Extracting remapping metadata
3. Calculating updated frame counts per expert
4. Applying frame-count heuristic for conflicts
5. Saving individual experts to skills/ folders
6. Updating global checkpoint metadata
7. Archiving and cleaning up training_runs/
"""

import pickle
import shutil
import fcntl
import time
import re
from pathlib import Path
from typing import Dict, Optional
import json

from flowrl.utils.test import load_policy_params


def sanitize_skill_name(skill_name: str) -> str:
    """
    Sanitize skill name for use in file paths.

    Args:
        skill_name: Raw skill name

    Returns:
        Sanitized name safe for file paths (alphanumeric and underscores only)
    """
    # First replace spaces with underscores
    sanitized = skill_name.replace(' ', '_')
    # Then remove all non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '', sanitized)
    return sanitized


def save_expert_to_skills(
    global_expert_idx: int,
    skill_name: str,
    expert_params: Dict,
    total_frames: int,
    base_dir: Path
) -> Path:
    """
    Save individual expert network to skills/ folder.

    Args:
        global_expert_idx: Global expert index
        skill_name: Name of the skill
        expert_params: Expert network parameters
        total_frames: Total frames trained for this expert
        base_dir: Base experiment directory

    Returns:
        Path to saved expert folder
    """
    base_dir = Path(base_dir)
    safe_skill_name = sanitize_skill_name(skill_name)

    # Create skill folder
    skill_folder = base_dir / "skills" / f"{global_expert_idx}_{safe_skill_name}"
    skill_folder.mkdir(parents=True, exist_ok=True)

    # Create expert policy folder
    expert_folder = skill_folder / f"expert_{global_expert_idx}_policy"
    expert_folder.mkdir(parents=True, exist_ok=True)

    # Normalize expert params to expert-only format: actor/ and critic/ subtrees (no MoE indexing)
    # expert_params currently may be a dict with top-level keys like actor_networks_i / critic_networks_i
    actor_sub = None
    critic_sub = None
    for k, v in expert_params.items():
        if isinstance(k, str) and k.startswith("actor_networks_"):
            actor_sub = v
        elif isinstance(k, str) and k.startswith("critic_networks_"):
            critic_sub = v
    normalized = {}
    if actor_sub is not None:
        normalized["actor_network"] = actor_sub
    if critic_sub is not None:
        normalized["critic_network"] = critic_sub

    # Save normalized params only
    params_path = expert_folder / "params.pkl"
    with open(params_path, 'wb') as f:
        pickle.dump(normalized, f)

    # Save metadata separately
    meta = {
        "skill_name": skill_name,
        "global_expert_idx": global_expert_idx,
        "total_frames": total_frames,
    }
    import json, datetime
    with open(expert_folder / "metadata.json", 'w') as f:
        meta["updated_at"] = datetime.datetime.now().isoformat()
        json.dump(meta, f, indent=2)

    return expert_folder


def load_global_checkpoint(base_dir: Path) -> Dict:
    """
    Load global checkpoint, or create new one if doesn't exist.

    Args:
        base_dir: Base experiment directory

    Returns:
        Global checkpoint dict
    """
    base_dir = Path(base_dir)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_ckpt_path = checkpoint_dir / "global_latest.pkl"

    if global_ckpt_path.exists():
        with open(global_ckpt_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Create new global checkpoint
        return {
            "skills": {},
            "db": {},
        }


def save_global_checkpoint(checkpoint: Dict, base_dir: Path):
    """
    Save global checkpoint.

    Args:
        checkpoint: Global checkpoint dict
        base_dir: Base experiment directory
    """
    base_dir = Path(base_dir)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    global_ckpt_path = checkpoint_dir / "global_latest.pkl"

    with open(global_ckpt_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def process_completed_training(
    run_folder: Path,
    skill_name: str,
    global_expert_idx: int,
    base_dir: Path,
    total_timesteps: Optional[int] = None,
    use_file_lock: bool = True
) -> Dict:
    """
    Process completed training run with frame-count heuristic.

    This function:
    1. Loads final training checkpoint with remapped experts
    2. Extracts remapping metadata
    3. Calculates new frame counts for each expert
    4. Compares with global checkpoint (frame-count heuristic)
    5. Updates or keeps existing experts based on frame counts
    6. Saves individual experts to skills/ folders
    7. Updates global checkpoint metadata
    8. Archives training artifacts to skills/ folder
    9. Cleans up training_runs/ folder

    Args:
        run_folder: Path to training_runs/skill_X_Name/
        skill_name: Name of the completed skill
        global_expert_idx: This skill's global expert index
        base_dir: Base experiment directory
        total_timesteps: Total timesteps trained (if None, extracted from checkpoint)
        use_file_lock: Use file locking for concurrent processing

    Returns:
        Dict with processing results: {
            "skill_name": str,
            "expert_updates": {global_idx: {"action": "updated"|"kept", "frames": int}},
            "new_expert": {"global_idx": int, "frames": int}
        }

    Example:
        >>> results = process_completed_training(
        ...     run_folder=Path("training_runs/skill_2_Make_Pickaxe/"),
        ...     skill_name="Make_Pickaxe",
        ...     global_expert_idx=2,
        ...     base_dir=Path("exp/bottom_up/"),
        ...     total_timesteps=100_000_000
        ... )
        >>> # Updates experts 0, 1 if they have more frames than existing
        >>> # Saves new expert 2
    """
    run_folder = Path(run_folder)
    base_dir = Path(base_dir)

    if use_file_lock:
        return _process_with_lock(
            run_folder, skill_name, global_expert_idx, base_dir, total_timesteps
        )
    else:
        return _process_completed_training_impl(
            run_folder, skill_name, global_expert_idx, base_dir, total_timesteps
        )


def _process_with_lock(
    run_folder: Path,
    skill_name: str,
    global_expert_idx: int,
    base_dir: Path,
    total_timesteps: Optional[int]
) -> Dict:
    """
    Process with file locking to handle concurrent completions.
    """
    base_dir = Path(base_dir)
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lock_path = checkpoint_dir / "global.lock"
    max_retries = 20
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            with open(lock_path, 'w') as lock_file:
                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Got lock - perform processing
                results = _process_completed_training_impl(
                    run_folder, skill_name, global_expert_idx, base_dir, total_timesteps
                )

                # Lock is automatically released when file is closed
                return results

        except IOError:
            # Lock held by another process
            if attempt < max_retries - 1:
                print(f"  [Lock] Waiting for global checkpoint lock (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                raise Exception(
                    f"Failed to acquire global checkpoint lock after {max_retries} attempts. "
                    f"Another process may be stuck."
                )


def _process_completed_training_impl(
    run_folder: Path,
    skill_name: str,
    global_expert_idx: int,
    base_dir: Path,
    total_timesteps: Optional[int]
) -> Dict:
    """
    Implementation of training processing logic.
    """
    run_folder = Path(run_folder)
    base_dir = Path(base_dir)

    print(f"\n{'='*70}")
    print(f"Processing completed skill: {skill_name} (expert {global_expert_idx})")
    print(f"{'='*70}")

    # 1. Resolve policies folder and load final TrainState via Orbax
    policies_folder = run_folder / f"{global_expert_idx}_policies"
    try:
        if not policies_folder.exists():
            raise FileNotFoundError(f"Policies folder not found: {policies_folder}")

        print(f"Loading final TrainState from: {policies_folder}/policies")
        train_state = load_policy_params(str(policies_folder))
        # Extract params map (handle nested {'params': {'params': {...}}})
        params_root = getattr(train_state, 'params', None) or train_state.get('params')
        if params_root is None:
            raise ValueError("Restored TrainState has no 'params' field")
        params = params_root.get('params') if isinstance(params_root, dict) and 'params' in params_root else params_root
    except Exception as e:
        print(f"[postproc] Error loading final TrainState: {e}")
        breakpoint()
        raise

    # 2. Load remapping sidecar
    remap_path = policies_folder / "remap.json"
    try:
        if not remap_path.exists():
            raise FileNotFoundError(f"remap.json not found at {remap_path}; cannot map local↔global experts")
        with open(remap_path, 'r') as f:
            remap = json.load(f)
        local_to_global = {int(k): int(v) for k, v in remap["local_to_global"].items()}
        initial_frame_counts = {int(k): int(v) for k, v in remap.get("initial_frame_counts", {}).items()}
    except Exception as e:
        print(f"[postproc] Error loading remap.json: {e}")
        breakpoint()
        raise

    # 3. Get total timesteps trained from config.yaml (and auxiliary metrics)
    if total_timesteps is None:
        try:
            import yaml
            config_path = policies_folder / "config.yaml"
            if not config_path.exists():
                print("  Warning: config.yaml not found; using 0 timesteps for frame accounting")
                total_timesteps = 0
                aux_metrics = {}
            else:
                with open(config_path, 'r') as f:
                    cfg = yaml.load(f, Loader=yaml.FullLoader)
                # Values may be nested under {'value': ...}
                def get_val(v):
                    return v.get('value') if isinstance(v, dict) and 'value' in v else v
                # Prefer TRAINED_TIMESTEPS, then fallback to TOTAL_TIMESTEPS; accept either case
                candidates = [
                    'TRAINED_TIMESTEPS', 'trained_timesteps',
                    'TOTAL_TIMESTEPS', 'total_timesteps'
                ]
                found = None
                for key in candidates:
                    if key in cfg and get_val(cfg.get(key)) is not None:
                        try:
                            found = int(get_val(cfg.get(key)))
                            break
                        except Exception:
                            continue
                total_timesteps = int(found) if found is not None else 0
                print(f"Total timesteps accounted for merging: {total_timesteps:,}")

                # Extract auxiliary metrics if present
                aux_metrics = {}
                for k in [
                    'FRAMES_PER_SUCCESS', 'frames_per_success',
                    'MEAN_EPISODE_LENGTH', 'mean_episode_length',
                    'SUCCESS_RATE', 'success_rate'
                ]:
                    if k in cfg and get_val(cfg.get(k)) is not None:
                        try:
                            aux_metrics[k.upper()] = float(get_val(cfg.get(k)))
                        except Exception:
                            pass
        except Exception as e:
            print(f"[postproc] Error reading config.yaml: {e}")
            breakpoint()
            raise

    print(f"Total timesteps trained: {total_timesteps:,}")

    # 4. Load global checkpoint
    global_ckpt = load_global_checkpoint(base_dir)

    # 5. Process each expert with frame-count heuristic
    expert_updates = {}

    # Helper: extract per-task subtree (actor/critic) by local index
    def extract_subtree_by_local(params_tree, local_idx: int) -> Dict:
        try:
            # params is a nested dict-like; we select keys starting with actor_networks_{i} and critic_networks_{i}
            subtree = {}
            for top_key, top_val in params_tree.items():
                if isinstance(top_key, str) and (top_key.startswith(f"actor_networks_{local_idx}") or top_key.startswith(f"critic_networks_{local_idx}")):
                    subtree[top_key] = top_val
            return subtree
        except Exception as e:
            print(f"[postproc] Error extracting subtree for local {local_idx}: {e}")
            breakpoint()
            raise

    for local_idx, global_idx in sorted(local_to_global.items()):
        # Extract per-task params
        try:
            expert_param_subtree = extract_subtree_by_local(params, local_idx)
        except Exception:
            # extract_subtree_by_local already breakpoints; re-raise
            raise
        if not expert_param_subtree:
            print(f"  Warning: no params found for local task {local_idx}; skipping")
            continue

        # Calculate new total frames for this expert
        initial_frames = initial_frame_counts.get(global_idx, 0)
        new_total_frames = initial_frames + total_timesteps

        # Find which skill owns this expert
        owning_skill = None
        for skill, skill_data in global_ckpt["skills"].items():
            if skill_data.get("expert_idx") == global_idx:
                owning_skill = skill
                break

        try:
            if owning_skill is not None:
                # Expert exists - apply frame-count heuristic
                existing_frames = global_ckpt["skills"][owning_skill].get("total_frames", 0)

                if new_total_frames > existing_frames:
                    # This version has more training → UPDATE
                    print(f"  ✓ Expert {global_idx} ({owning_skill}): "
                          f"UPDATING ({new_total_frames:,} > {existing_frames:,} frames)")

                    save_expert_to_skills(global_idx, owning_skill, expert_param_subtree, new_total_frames, base_dir)
                    global_ckpt["skills"][owning_skill]["total_frames"] = new_total_frames

                    expert_updates[global_idx] = {"action": "updated", "frames": new_total_frames}
                else:
                    # Keep existing version
                    print(f"  → Expert {global_idx} ({owning_skill}): "
                          f"keeping existing ({existing_frames:,} ≥ {new_total_frames:,} frames)")

                    expert_updates[global_idx] = {"action": "kept", "frames": existing_frames}
            else:
                # New expert for this skill
                print(f"  ✓ Expert {global_idx} ({skill_name}): NEW ({new_total_frames:,} frames)")

                save_expert_to_skills(global_idx, skill_name, expert_param_subtree, new_total_frames, base_dir)

                # Add to global checkpoint
                safe_skill_name = skill_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                global_ckpt["skills"][skill_name] = {
                    "expert_idx": global_expert_idx,
                    "skill_with_consumption": remap.get("skill_with_consumption", {}),
                    "path": f"skills/{global_expert_idx}_{safe_skill_name}/",
                    "total_frames": new_total_frames,
                }

                expert_updates[global_idx] = {"action": "new", "frames": new_total_frames}
        except Exception as e:
            print(f"[postproc] Error updating expert {global_idx}: {e}")
            breakpoint()
            raise

    # 6. Save updated global checkpoint
    # Attach auxiliary metrics to the owning skill, if any
    owning_skill_record = global_ckpt["skills"].get(skill_name)
    if owning_skill_record is not None and aux_metrics:
        metrics = owning_skill_record.get("metrics", {})
        if 'FRAMES_PER_SUCCESS' in aux_metrics:
            metrics['frames_per_success'] = aux_metrics['FRAMES_PER_SUCCESS']
        if 'MEAN_EPISODE_LENGTH' in aux_metrics:
            metrics['mean_episode_length'] = aux_metrics['MEAN_EPISODE_LENGTH']
        if 'SUCCESS_RATE' in aux_metrics:
            metrics['success_rate'] = aux_metrics['SUCCESS_RATE']
        owning_skill_record["metrics"] = metrics

    save_global_checkpoint(global_ckpt, base_dir)
    print(f"  Global checkpoint updated")

    # 7. Archive training artifacts to skills/ folder
    safe_skill_name = skill_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    skill_folder = base_dir / "skills" / f"{global_expert_idx}_{safe_skill_name}"
    skill_folder.mkdir(parents=True, exist_ok=True)

    # Copy module file
    module_file = run_folder / f"{global_expert_idx}.py"
    if module_file.exists():
        shutil.copy(module_file, skill_folder / f"{global_expert_idx}.py")

    # Copy training log
    log_file = run_folder / "training.log"
    if log_file.exists():
        shutil.copy(log_file, skill_folder / "training.log")

    # Copy trajectory analysis debug files
    trajectory_debug_file = run_folder / "trajectory_debug.txt"
    if trajectory_debug_file.exists():
        shutil.copy(trajectory_debug_file, skill_folder / "trajectory_debug.txt")

    analysis_output_file = run_folder / "analysis_output.pkl"
    if analysis_output_file.exists():
        shutil.copy(analysis_output_file, skill_folder / "analysis_output.pkl")

    # Copy any videos
    for video_file in run_folder.glob("*.mp4"):
        shutil.copy(video_file, skill_folder / video_file.name)

    # Write metrics sidecar if available
    metrics = {}
    if aux_metrics:
        if 'FRAMES_PER_SUCCESS' in aux_metrics:
            metrics['frames_per_success'] = aux_metrics['FRAMES_PER_SUCCESS']
        if 'MEAN_EPISODE_LENGTH' in aux_metrics:
            metrics['mean_episode_length'] = aux_metrics['MEAN_EPISODE_LENGTH']
        if 'SUCCESS_RATE' in aux_metrics:
            metrics['success_rate'] = aux_metrics['SUCCESS_RATE']
    if metrics:
        with open(skill_folder / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"  Archived artifacts to {skill_folder.relative_to(base_dir)}")

    # 8. Clean up training_runs folder
    # Also copy policies folder to skill folder for potential continuation runs
    policies_src = run_folder / f"{global_expert_idx}_policies"
    policies_dst = skill_folder / f"{global_expert_idx}_policies"
    if policies_src.exists():
        # If destination exists, remove it to avoid stale mix
        if policies_dst.exists():
            shutil.rmtree(policies_dst)
        shutil.copytree(policies_src, policies_dst)

    try:
        shutil.rmtree(run_folder)
        print(f"  Cleaned up {run_folder.relative_to(base_dir)}")
    except Exception as e:
        print(f"  Warning: Failed to clean up {run_folder}: {e}")

    print(f"{'='*70}\n")

    # Return processing results
    results = {
        "skill_name": skill_name,
        "global_expert_idx": global_expert_idx,
        "total_timesteps": total_timesteps,
        "expert_updates": expert_updates,
        "metrics": {
            "frames_per_success": aux_metrics.get('FRAMES_PER_SUCCESS'),
            "mean_episode_length": aux_metrics.get('MEAN_EPISODE_LENGTH'),
            "success_rate": aux_metrics.get('SUCCESS_RATE'),
        },
    }

    return results
