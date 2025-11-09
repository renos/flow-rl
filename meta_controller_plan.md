# Meta Controller Plan

## Objective
- Add a meta-controller on top of existing experts that selects which skill to execute next, runs it until completion, and iterates.
- Use the meta-controller to satisfy the preconditions of the currently training skill; once satisfied, execute and learn that skill.

## Scope
- Keep existing experts/skills intact; introduce a controller that sequences them.
- Provide a clear definition and implementation of skill completion and precondition satisfaction in terms of observable state deltas (including the ambiguous `n`).
- Extend PPO training to hierarchical control: controller policy (discrete over skills) + skill policies (unchanged PPO or gated execution).

## Design Overview
- Hierarchical policy:
  - Meta-controller π_c(a_c|s) chooses next skill id `k` (discrete over skill library).
  - Selected skill π_k(a|s) acts at each env step until completion/termination criteria fire or a max steps cap is reached.
  - Control returns to meta-controller for the next skill selection.
- Skill interface additions:
  - `preconditions: List[Callable[[State], bool]]` — predicates that must be true for the skill to be “ready”.
  - `completion: Callable[[TrajectoryWindow], bool]` — termination function defined on short state/action windows or deltas.
  - Optional: `timeout_steps: int`, `min_duration_steps: int` to avoid flapping.
- Completion/termination functions:
  - Define via differences in tracked features (including `n`) across a sliding window: Δf = f_t − f_{t−w}.
  - Support composite logic (AND/OR), thresholds, hysteresis, and cooldowns to stabilize.
- Training coordination goal:
  - Meta-controller’s reward encourages achieving the preconditions of the target skill under training.
  - Once all preconditions are satisfied, hand control to the target skill for its learning phase.

## Completion & Preconditions
- Feature set:
  - `n` and related counters/metrics relevant to the domain.
  - Key task features (positions, distances, resource counts, booleans for “event occurred”).
- Delta-based detectors:
  - `delta_ge(f, thr, w)`: completion if f_t − f_{t−w} ≥ thr.
  - `relative_drop(f, pct, w)`: completion if (f_{t−w} − f_t)/max(ε, f_{t−w}) ≥ pct.
  - `stability(f, tol, w)`: completion if |f_t − f_{t−w}| ≤ tol for w steps (useful for “held position/steady state”).
  - `event_reached(b)`: completion if boolean event flips from 0→1 within window.
- Composite logic and guards:
  - Support `AllOf([...])`, `AnyOf([...])`, with `min_duration_steps` and `timeout_steps`.
  - Add hysteresis: completion needs to persist for `persist_k` steps before terminating to avoid jitter.
- Preconditions:
  - Same predicate system; preconditions operate on current state or short history.
  - Example: “resource n ≥ N0” AND “distance_to_target ≤ d0”.

## Rollouts & Buffers
- Segment episodes into skill segments:
  - Each segment stores: skill id, start/end step, termination reason (complete/timeout/abort), precondition status at entry.
  - Maintain per-segment masks for credit assignment to π_c (only at selection/termination boundaries) and to π_k (every env step within segment).
- Termination signal propagation:
  - Add `done_segment` markers in the buffer; carry skill id as part of observation or info for the policy.
- Logging:
  - Log per-skill success rate, average segment length, termination reasons, precondition satisfaction latency.

## Rewards
- Meta-controller reward shaping:
  - Primary: sparse +1 when all preconditions of the target skill become satisfied; 0 otherwise.
  - Optional dense shaping: negative step penalty until satisfaction; positive partial credit when individual preconditions flip true.
  - Entropy bonus to encourage exploring different skill sequences.
- Skill reward:
  - Keep existing PPO skill rewards unchanged. Optionally add bonus when its own completion triggers to reinforce crisp termination.

## Training Loop (High-Level)
1) Select current training skill `T` for this phase/epoch (curriculum or schedule).
2) Reset env; while not episode done:
   - While preconditions(T) are not all true and episode not done:
     - Meta-controller selects next skill `k = π_c(s)`.
     - Execute π_k for steps until `completion_k` OR `timeout_k`.
     - Give meta-controller rewards based on progress toward preconditions(T).
   - When preconditions(T) satisfied:
     - Execute π_T for steps per normal PPO until its termination criterion or episode end.
     - Log success/metrics, then either continue to new target skill or end episode based on schedule.
3) Update:
   - Update π_c from its segmented trajectories.
   - Update π_k from their within-segment trajectories (only those executed).

## Implementation Plan
1) API additions
   - Add `SkillSpec` in `flowrl/utils/skills.py` with fields: `name`, `preconditions`, `completion`, `timeout_steps`, `min_duration_steps`.
   - Add predicate/completion combinators in `flowrl/utils/terminations.py` (delta-based and event-based).
2) Controller policy
   - New PPO head/classifier over skill ids. Expose as `ControllerPolicy`.
   - Observation augmentation to include current precondition statuses and target-skill id.
3) Buffers
   - Extend rollout buffer to store segment metadata and skill ids; add `done_segment` markers.
4) Trainer
   - Introduce `flowrl/ppo_meta.py` variant that orchestrates controller+skills, backed by existing PPO implementations for each skill.
   - CLI flags: `--use-meta`, `--target-skill`, `--controller-update-every`, `--segment-timeout`, `--min-segment-len`.
5) Scheduling
   - Curriculum utility in `flowrl/utils/curriculum.py` to rotate/advance target skills based on success rates.
6) Logging
   - Per-skill success rates, termination reasons, precondition latencies; integrate with W&B if enabled.
7) Tests
   - `test_meta_controller.py`: unit tests for termination combinators; integration test with seeded env to validate segmenting and controller updates.

## Minimal Pseudocode Sketch
```
state = env.reset()
while not env_done:
  if not preconds_target(state):
    k = controller.select_skill(state, target_id)
    seg = start_segment(k)
    for t in range(max_steps):
      action = skills[k].act(state)
      next_state, r, done, info = env.step(action)
      buf.add_step(state, action, r, done, skill_id=k)
      state = next_state
      if completed[k](window(state_history)) or t+1 >= timeout[k]:
        end_segment(seg, reason)
        controller_buf.add_transition(seg_summary)
        break
      if done: break
  else:
    # Train target skill
    for t in range(max_target_steps):
      action = skill_T.act(state)
      state, r, done, info = env.step(action)
      target_buf.add_step(...)
      if completed_T(window(...)) or done: break

# PPO updates: controller_buf -> π_c; skill buffers -> respective π_k
```

## Open Questions / Assumptions
- Clarify `n`: treat as one of the tracked features; completion/preconditions operate on deltas (Δn) or thresholds on `n` itself.
- Abort semantics: if a skill makes preconditions harder (regression), introduce penalty or early abort.
- Skill library: mapping from expert names to skill ids and their specs; ensure stable ordering.
- Exploration vs exploitation at controller level: start with higher entropy; anneal.

## Milestones
1) Termination/precondition library implemented and tested on synthetic signals.
2) Rollout buffer segmentation with metrics and logs.
3) Controller policy wired with simple two-skill toy env; verify learning curves.
4) Integrate with existing experts; add `ppo_meta.py` and CLI flags.
5) Curriculum for target-skill scheduling; W&B dashboards for success rates and latencies.

