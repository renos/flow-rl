# Staircase Dungeon Environment

## Slide 1: Environment Overview

**Goal**: Navigate through 30 floors by learning which staircase type is correct on each floor

**Environment Setup**:
- **Grid**: 1x100 corridor (configurable)
- **Floors**: 30 sequential floors (0-29)
- **Staircases**: Two types per floor (Type 2 and Type 3)
  - Placed at opposite ends (x=0 and x=99)
  - Random left/right placement (50/50) each episode
- **Agent**: Spawns randomly in middle section (x ∈ [1, 98])
- **Actions**: 5 discrete actions (UP, DOWN, LEFT, RIGHT, NOOP)

---

## Slide 2: Core Mechanics

**Staircase Pattern (Fixed Across Episodes)**:
```
Floor 0: Type 2 correct (90% → next floor, 10% → death)
Floor 1: Type 2 correct
Floor 2: Type 3 correct ← FIRST SWITCH
Floor 3: Type 2 correct
...
```

**Rewards**:
- **+1**: Successfully progress to next floor
- **-1**: Die (wrong staircase or 10% bad luck)
- **+10**: Complete all 30 floors (win)

**Key Challenge**: Agent must learn the TYPE pattern (Type 2 vs Type 3), not spatial patterns (left vs right), since spatial placement is randomized each episode.

---

## Slide 3: Learning Challenge

**Why This Is Hard**:

1. **Partial Observability**: Agent sees current floor grid + floor number
   - Can identify staircase types (2 vs 3) in observation
   - Must learn which type is correct through trial and error

2. **Credit Assignment**:
   - Long corridor (50+ steps to reach staircase)
   - Sparse rewards (only at staircases)
   - Stochastic outcomes (90% success rate)

3. **Generalization Required**:
   - Cannot rely on spatial cues (randomized placement)
   - Must learn type-based policy
   - Pattern switches at floor 2 break naive policies

4. **Exploration Problem**:
   - Compound probability: p(reach floor N) = 0.9^N
   - Floor 5 reached only ~59% of episodes
   - Later floors rarely visited without curriculum

---

## Slide 4: Key Metrics

**Absolute Metrics**:
- `floor_reached/floor_i`: Probability of reaching floor i in any episode
- `max_floor_reached`: Highest floor reached across all environments

**Conditional Metrics** (Most Important):
- `floor_conditional/p_floor_i+1_given_i`: Transition probability p(floor i+1 | floor i)
- Directly measures agent's success rate on each floor
- Expected: ~0.9 for all floors (if agent learns correctly)
- Observed: Drops sharply at floor 2 (first pattern switch)

**Training Metrics**:
- `charts/episodic_return`: Average episode return
- `env/win_rate`: Fraction reaching floor 30
- `floor_completion/floor_i`: Rate of successful floor completions

---

## Slide 5: Current Results & Open Questions

**Observed Behavior**:
- ✅ Floor 0→1: p(floor 1 | floor 0) ≈ 0.9 (learning works!)
- ✅ Floor 1→2: p(floor 2 | floor 1) ≈ 0.8 (still learning)
- ❌ Floor 2→3: p(floor 3 | floor 2) ≈ 0.4 (sharp drop!)

**Root Cause Analysis**:
1. ✅ **Not spatial bias**: Type placement is truly random (50/50 left/right)
2. ✅ **Not spawn bias**: Random spawning provides diverse training signal
3. ⚠️ **Hypothesis**: Insufficient experience at floor 2
   - Floors 0-1 both need Type 2 → Agent learns "Type 2 = good"
   - Floor 2 needs Type 3 → Conflicts with learned policy
   - Agent rarely reaches floor 2 → Slow unlearning

**Proposed Solutions**:
1. **Progressive Reset Curriculum**: Reset to frontier floor (implemented in `ppo_flow_staircase.py`)
2. **Increased Exploration**: Higher entropy coefficient (ENT_COEF: 0.05)
3. **More Environments**: 4096 parallel envs for more floor 2 experiences
4. **Mixture of Experts**: One policy per floor (isolates learning per floor)

---

## Appendix: Technical Details

**Environment Files**:
- `staircase_env.py`: Core environment implementation
- `ppo_staircase.py`: Standard PPO training
- `ppo_flow_staircase.py`: MoE + progressive curriculum

**Key Parameters**:
- Success probability: 0.9 (correct staircase)
- Corridor length: 100 tiles
- Max timesteps: 2000 per episode
- Training: 1B timesteps, 4096 envs, 64 steps per rollout

**Correct Staircase Pattern** (first 10 floors):
```
[True, True, False, True, True, True, True, True, False, True, ...]
True  = Type 2 correct
False = Type 3 correct
```
