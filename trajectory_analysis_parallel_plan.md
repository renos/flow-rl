**Trajectory Analysis in Parallel Training — Plan**

**Objectives**
- Reintroduce trajectory analysis into the parallel pipeline without counting its frames as “training”.
- Use a two‑phase scheme per skill: initial low threshold to get any successes, analyze trajectories to refine spec, then final training to production threshold.
- Avoid premature/incorrect preconditions by leveraging environment distribution (e.g., resource density varies by region/floor).

**Two‑Phase Flow**
- Phase A: Initial viability
  - Train to low bar success: `success_state_rate ≈ 0.01`.
  - Save policy (Orbax in `<run>/<E>_policies/policies/<STEP>` + `config.yaml` with `TRAINED_TIMESTEPS`).
  - Do not process experts into the global checkpoint yet.
- Trajectory analysis
  - Rollout with saved policies via `gen_frames_hierarchical(policy_path=…_policies)` to collect trajectories for the current skill’s execution graph.
  - Call `Flow.explain_trajectory(env_states, actions, goal_state)` to propose spec updates (requirements/consumption/gain) and KB updates.
  - Rebuild skill dependency graph and rewrite module with remapping.
- Phase B: Final training
  - Resume PPO with `--prev_module_path` to seed from Phase A checkpoint.
  - For now, train for a fixed budget of frames (e.g., `TOTAL_TIMESTEPS = 10_000_000`) regardless of success threshold; then post‑process experts and archive artifacts.

**Combined Budget (Phase A + Phase B)**
- Use a single per‑skill budget `SKILL_TIMESTEPS_BUDGET` (e.g., `10M`).
- Phase A consumes budget until sufficient successful trajectories are observed, then stops immediately to run trajectory analysis.
  - Trigger options (configurable):
    - `success_state_rate ≥ 0.01` AND at least `MIN_SUCCESS_EPISODES` (e.g., 8–32) completed successes, or
    - `SUCCESS_TRANSITIONS ≥ T_MIN` (count of `penultimate→goal` transitions),
    - OR a hard cap `PHASE_A_MAX_UPDATES` to avoid stalling.
- Phase B consumes the remaining budget after analysis using `--prev_module_path` to resume from Phase A checkpoint.
- Trajectory collection frames are not counted toward the budget (evaluation‑only rollouts).

**Parallel Integration (where/how)**
- Scheduler states (flowrl/parallel/scheduler.py)
  - Add `phase` per skill: `initial → analyzing → final → completed`.
  - Use `--success_state_rate=0.01` for initial, configured final threshold (e.g., 0.8) for final.
  - After initial success: run analysis on scheduler thread; regenerate module and relaunch Phase B with `--prev_module_path`.
  - Post‑process only after final pass.
- Training setup (flowrl/parallel/training_setup.py)
  - No change to dependency seeding; Phase B seeding uses `prev_module_path` provided by scheduler.
- Post‑processing (flowrl/parallel/training_processor.py)
  - Unchanged; it relies on `TRAINED_TIMESTEPS` and remap to account only training frames. Trajectory frames are not included.

**Success Threshold Strategy**
- Initial pass (bootstrap): fixed low threshold `p_init = 0.01` (1%). Goal: collect any successful episodes quickly to inform spec.
- Final pass (production): default `p_final = 0.8` (80%) for short‑horizon skills; adapt for long‑horizon/low‑base‑rate skills.
  - Option A: per‑category defaults (collection, crafting, placement: 0.8; multi‑step deep skills: 0.5–0.7; very sparse goals: 0.2–0.4).
  - Option B: statistical criterion using Wilson score lower bound on M evaluation episodes (e.g., LB ≥ target) to avoid flukes.
  - Option C: transition‑based success from PPO metrics (`transition_success_rate`), require both `reached_state` and transition success above thresholds.

**Avoiding Premature Preconditions**
- Problem: Environment distribution is heterogeneous (e.g., more diamonds at “second floor”/deeper layers). If we lock preconditions to the first floor, training could stall.
- Approach:
  - During trajectory analysis, estimate contextual predicates from successful episodes: region/floor/depth proxies, proximity to target blocks, tool tiers present.
  - Maintain “ASSUMPTION vs VERIFIED” flags (existing prompts) and only promote to VERIFIED when the trajectory evidences necessity.
  - If initial training yields very low transition success and trajectories cluster in a different context (e.g., deeper strata), expand preconditions to match observed context before Phase B.
  - Prefer minimal enabling sets: add verified tool tiers (e.g., `pickaxe ≥ iron`) and enabling achievements; avoid over‑constraining (don’t require exact coordinates/floors unless consistently observed as necessary).

**Precondition Refinement Heuristics**
- Tools/tiered items: infer minimal required tier from deltas in trajectory (e.g., stone/iron pickaxe before diamond mining).
- Access predicates: infer enabling actions (e.g., place torches, craft ladders) if repeatedly present before success.
- Region proxies: use observation stats (e.g., nearest block types, densities) as soft predicates; if env exposes a floor/depth signal, optionally encode as a precondition after repeated corroboration.
- Stop rules: only lock a new precondition if: (1) present in ≥K successful trajectories and (2) removing it in a what‑if analysis predicts failure (LLM‑guided or rule‑based).

**Metrics and Decision Rules**
- PPO metrics already exposed (flowrl/ppo_flow.py):
  - `info["reached_state"][SUCCESS_STATE_INDEX]` (episode success rate)
  - `transition_success_rate` from `penultimate → goal` vs `penultimate → fail`
  - `state_rates`, episode counts
- Promotion from Phase A to B: `success_state_rate ≥ 0.01` OR time cap reached.
- Phase A: stop early on success as soon as triggers are met (see above), conserving budget for Phase B.
- Phase B: train with remaining budget (no threshold gating for completion).
- Plateau guard: if success plateaus and trajectories indicate missing preconditions, trigger another refine cycle before giving up.

**Cost Metric: Frames per Successful Execution**
- Definition: `frames_per_success = mean_episode_length / success_rate`.
  - `mean_episode_length`: average of `returned_episode_lengths` over completed episodes.
  - `success_rate`: `info["reached_state"][SUCCESS_STATE_INDEX]` measured over the same window.
- Implementation notes:
  - Compute in `ppo_flow` during training; persist into `config.yaml` as `FRAMES_PER_SUCCESS` and `MEAN_EPISODE_LENGTH` alongside `TRAINED_TIMESTEPS`.
  - In post‑processing, read and store under the skill’s entry in the global checkpoint (e.g., `skills[skill_name].frames_per_success`).
  - Expose to the LLM: merge into `db["skills"][name]["metrics"]["frames_per_success"]` so prompts can trade off frontier value vs. execution cost (e.g., expensive preconditions).

**Data and Artifacts**
- Phase A policy: `<run>/<E>_policies/` (Orbax) + `config.yaml` with `TRAINED_TIMESTEPS`.
- Trajectory logs and videos remain in `<run>/` and are archived into `skills/<E>_<Name>/` at final post‑processing.
- Remapping sidecars: `remap.json` continues to enumerate all experts referenced in the execution plan (deps + current), ensuring `task_*_network_number()` maps via `GLOBAL_TO_LOCAL`.

**Open Questions**
- Per‑skill thresholds: do we want a config mapping skill categories → final thresholds, or keep a single global default with overrides per skill?
- Environment signals: do we have explicit “floor/depth” features to promote to hard preconditions, or should we remain with observation‑based proxies (e.g., block density/proximity)?
- Evaluation passes: add a quick evaluation-only rollout after Phase B to compute a Wilson LB on success for gating?

**Implementation Tasks**
- Scheduler (phaseful):
  - Track `phase` per skill; wire `--success_state_rate` based on phase.
  - Maintain and pass a single `SKILL_TIMESTEPS_BUDGET` across phases; track remaining budget.
  - After Phase A success: call `gen_frames_hierarchical()` then `Flow.explain_trajectory()`; rebuild graph and rewrite module; relaunch Phase B with `--prev_module_path` and `TOTAL_TIMESTEPS = remaining_budget`.
  - Ensure training frames only come from PPO; trajectory collection isn’t counted.
- Command wiring:
  - Pass through `--total_timesteps`, `--num_envs`, wandb args, `--success_state`, per‑phase `--success_state_rate`, and Phase A early‑stop triggers (`--min_success_episodes`, `--phase_a_max_updates`).
- Persistence and recovery:
  - Persist `phase` and derived artifacts in `scheduler_state.json` so runs can resume mid‑phase after restarts.
- Post‑processing:
  - Keep current logic; only run after final success to update experts and global checkpoint; retain run folders on failure.

**Next Steps**
- Decide per‑skill threshold policy (global default vs category map + Wilson LB use).
- Implement scheduler phase flow and command wiring.
- Add a small e2e test: skill with dependencies → Phase A (1%), refine → Phase B (final), verify experts updated and remap consistent.
