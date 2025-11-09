# Parallel Checkpointing Plan

This document summarizes what needs to be implemented so parallel training uses and produces checkpoints that are compatible with `ppo_flow`, and so experts are correctly seeded and updated across skills.

## Goals

- Seed each parallel training run with the best-available parameters for all required experts (dependencies + new expert) in the exact format `ppo_flow` expects.
- After training, extract each expert’s updated parameters from the MoE and update the corresponding expert in `skills/` if it has been trained on more total frames (frame-count heuristic).
- Preserve a single source of truth for expert→frames metadata to enable consistent “keep vs update” decisions.

## Current Behavior (and Gaps)

- `ppo_flow` loads a Flax TrainState from a previous module via `--prev_module_path`, using Orbax under `<prev>_policies/policies/` with a `config.yaml` in `<prev>_policies/`.
- `ppo_flow` does not read the custom `checkpoint_0.pkl` and does not persist any “remapping_metadata”. It saves a new Orbax TrainState in `<module_dir>/<stem>_policies/policies/` with a `config.yaml` alongside.
- Our current parallel code writes `checkpoint_0.pkl` (custom pickle) with remapping plus “expert_params”, but this is not used by `ppo_flow` and is therefore ineffective for seeding.
- Post‑processing expects remapping metadata in the training checkpoint, which `ppo_flow` does not write.

## Requirements

1) Initial seeding must provide a valid Orbax TrainState that `ppo_flow` can restore when passed `--prev_module_path <run_folder>/<E>.py`.
2) The per-task (local) experts in the TrainState must correspond to the contiguous local indices defined by the remapping (dependencies first + new expert).
3) After training, we must extract each local expert’s parameters from the saved TrainState, map local→global via the remapping, and update the persisted expert under `skills/` if the new total frames exceed the stored total (frame-count heuristic).
4) We must maintain and update expert metadata (skill_name, global_expert_idx, total_frames) to support consistent comparisons and future seeding.

## Design Overview

We will align both directions (seeding and extraction) to `ppo_flow`’s single supported format: Orbax TrainState saved under `<module_dir>/<stem>_policies/policies/` with a `config.yaml` in `<module_dir>/<stem>_policies/`.

### A. Initial Seed Checkpoint (Before Training)

- Inputs:
  - `execution_plan` (ordered steps with counts and validated function strings).
  - `dependency_skill_names` and their `expert_idx` (from global checkpoint or scheduler state).
  - `global_to_local` and `local_to_global` remapping.
  - Existing per-expert params stored under `skills/<global_idx>_<SkillName>/expert_<global_idx}_policy/`.

- Steps:
  1. Instantiate `ActorCriticMoE(num_tasks = len(dependencies) + 1)` using the same shapes/activation as `ppo_flow`.
  2. Initialize a fresh TrainState to get the parameter tree structure.
  3. For each dependency global expert:
     - Load that expert’s saved (per-expert) Flax subtree from `skills/…/expert_<global_idx>_policy/params.pkl`.
     - Determine its local index via `GLOBAL_TO_LOCAL[global_idx]`.
     - Overwrite the corresponding per-task subtree in the TrainState (actor + critic) with the loaded expert subtree.
  4. Leave the new expert (the current skill’s global index) at its random initialization or apply a light init policy if desired.
  5. Save this TrainState using Orbax to `<run_folder>/<E>_policies/policies/0` and write a minimal `config.yaml` into `<run_folder>/<E>_policies/` (only the fields required by `restore_model()` and downstream).
  6. Launch `ppo_flow` with `--module_path <run_folder>/<E>.py` and `--prev_module_path <run_folder>/<E>.py`, plus `--success_state` and other training args.

Notes:
- `ppo_flow`’s `restore_model()` reads `config.yaml` from `<stem>_policies/` and uses `load_policy_params()` to pick the largest numeric checkpoint in `policies/`. Using `0` for the seed is fine.
- We will not rely on the custom `checkpoint_0.pkl` going forward. The seed must be a proper Orbax TrainState.

### B. Post‑Training Expert Extraction (After Training)

- Inputs:
  - The saved TrainState from `<run_folder>/<E>_policies/policies/<step>`.
  - Remapping used for this run (`local_to_global`, `initial_frame_counts`). The source of truth can be the run’s metadata written at launch (e.g., `skills/<E>_<Skill>/metadata.json`) or a lightweight sidecar in `<run_folder>/<E>_policies/`.
  - `TOTAL_TIMESTEPS` used for this run (or measured trained steps) for frame accounting.

- Steps:
  1. Load the final TrainState via Orbax.
  2. For each local task index:
     - Extract that per-task subtree (actor + critic) from the TrainState params.
     - Map local→global via `local_to_global`.
     - Compute `new_total_frames = initial_frame_counts[global_idx] + TOTAL_TIMESTEPS` (or actual measured frames if available).
     - Compare to the existing expert’s frames in `skills/<global_idx>_<Skill>/expert_<global_idx>_policy/params.pkl` metadata.
     - If `new_total_frames` > existing, overwrite that expert’s params with the extracted subtree and update its metadata (`total_frames`). Otherwise keep existing.
  3. Update the global checkpoint metadata accordingly (expert frames, owning skill, path).
  4. Archive the module `<E>.py` and any run artifacts into `skills/<E>_<Skill>/` for provenance.

### C. Per‑Expert Storage Format

- Each expert folder `skills/<global_idx>_<SkillName>/expert_<global_idx>_policy/params.pkl` should contain:
  - A Flax params subtree compatible with a single task index of `ActorCriticMoE` (i.e., the per-task actor and critic trees only, not all tasks).
  - Metadata: `{ "skill_name", "global_expert_idx", "total_frames" }`.

This makes it possible to:
- Seed a new run by copying per‑expert subtrees into a freshly initialized MoE TrainState at the correct local indices.
- Update a single expert after training by extracting that subtree from the MoE TrainState and writing it back to the expert folder.

## Interfaces to Update

1) `flowrl/parallel/training_setup.py`
   - Build seed TrainState (Orbax) at `<run_folder>/<E>_policies/policies/0` using dependency experts.
   - Write minimal `config.yaml` in `<run_folder>/<E>_policies/` for `restore_model()`.
   - Provide remapping and `initial_frame_counts` in a small JSON sidecar (e.g., `<run_folder>/<E>_policies/remap.json`) and/or also in `skills/<E>_<Skill>/metadata.json` for post‑processing.

2) `flowrl/parallel/scheduler.py`
   - Pass `--prev_module_path <run_folder>/<E>.py` and `--success_state <len(execution_plan)>` in the tmux command.
   - Ensure total timesteps, num_envs, success rate, wandb flags match the orchestrator config.

3) `flowrl/parallel/training_processor.py`
   - Load final TrainState from `<run_folder>/<E>_policies/policies/` via Orbax.
   - Read remapping + initial_frame_counts (from sidecar or skill metadata).
   - Extract per‑task subtrees, apply frame‑count heuristic, and update per‑expert folders under `skills/`.
   - Update the global checkpoint metadata and archive artifacts.

## Failure Handling & Logging

- Surface errors when seed TrainState creation fails (missing expert params, shape mismatch, etc.).
- Log the exact tmux command (already done) and training logs under `skills/<E>_<Skill>/training.log`.
- If post‑processing cannot find a final checkpoint, raise a clear error indicating the policies folder and expected files.

## Testing Plan

- Unit tests:
  - Seed builder: construct MoE TrainState with N tasks and verify per‑task params match loaded expert subtrees in local indices.
  - Extractor: given a saved MoE TrainState and remapping, verify extracted subtrees and frame heuristics update the correct experts.
  - Config and restore: ensure `restore_model()` can load the seed checkpoint produced by the seed builder (with `config.yaml`).

- Integration tests:
  - End‑to‑end parallel run with 1–2 dependencies + 1 new expert; verify that:
    - PPO restores from the seed checkpoint and runs.
    - Experts are updated in `skills/` based on frames.
    - Global checkpoint reflects updated frames and paths.

## Open Questions / Decisions

- Do we want to initialize the new expert with a specific strategy (e.g., copy from closest dependency) or keep random init?
- Where to store the definitive remapping metadata for a run: in the skill’s metadata.json, in `<E>_policies/remap.json`, or both?
- Whether to record actual trained steps per expert (if available) rather than using TOTAL_TIMESTEPS uniformly.

