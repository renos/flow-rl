# Parallel Training Quick Start Guide

## Overview

The parallel training system allows multiple skills to train simultaneously, dramatically reducing wall-clock time for bottom-up skill learning.

## Installation & Setup

No additional dependencies needed beyond the base Flow-RL requirements.

```bash
# Ensure you have tmux installed (required for parallel sessions)
# macOS:
brew install tmux

# Linux:
sudo apt-get install tmux
```

## Running Parallel Training

### Basic Usage

```bash
python -m flowrl.bottomup_parallel \
  --env_name Craftax-Symbolic-v1 \
  --graph-path exp/parallel_test/ \
  --max_nodes 20 \
  --max_parallel_skills 3 \
  --scheduler_poll_interval 30 \
  --total_timesteps 1e8 \
  --use_wandb
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_parallel_skills` | 3 | Number of skills to train simultaneously |
| `--scheduler_poll_interval` | 30 | Seconds between scheduler checks |
| `--max_retries` | 2 | Retry attempts for failed skills |
| `--max_nodes` | 40 | Total skills to generate/train |
| `--llm_name` | None (gpt-5) | LLM model to use (e.g., "gpt-4o-mini" for cheaper testing) |

### Command-Line Examples

**Small test (3 skills, 2 parallel, cheap LLM):**
```bash
python -m flowrl.bottomup_parallel \
  --max_nodes 3 \
  --max_parallel_skills 2 \
  --total_timesteps 5e7 \
  --llm_name gpt-4o-mini \
  --no-use_wandb
```

**Production run (20 skills, 4 parallel):**
```bash
python -m flowrl.bottomup_parallel \
  --max_nodes 20 \
  --max_parallel_skills 4 \
  --total_timesteps 2e8 \
  --scheduler_poll_interval 60 \
  --use_wandb \
  --wandb_project flow_rl_parallel
```

## Monitoring

### View Scheduler State

```bash
# Pretty-print current state
cat exp/parallel_test/scheduler_state.json | jq

# Watch status summary
watch -n 5 'cat exp/parallel_test/scheduler_state.json | jq ".skills | group_by(.status) | map({status: .[0].status, count: length})"'
```

### List Active Training Sessions

```bash
# List all flowrl tmux sessions
tmux list-sessions | grep flowrl_

# Example output:
# flowrl_e0_Collect_Wood: 1 windows (created Tue Oct  7 14:23:45 2025)
# flowrl_e1_Collect_Stone: 1 windows (created Tue Oct  7 14:23:50 2025)
# flowrl_e2_Make_Pickaxe: 1 windows (created Tue Oct  7 14:45:12 2025)
```

### Attach to a Training Session

```bash
# Attach to see live training output
tmux attach -t flowrl_e0_Collect_Wood

# Detach from session (while keeping it running)
# Press: Ctrl-B then D
```

### View Training Logs

```bash
# Tail a specific skill's log
tail -f exp/parallel_test/training_runs/skill_0_Collect_Wood/training.log

# View completed skill's log
cat exp/parallel_test/skills/0_Collect_Wood/training.log
```

## Understanding the Workflow

### 1. Event-Driven Generation

Unlike sequential training, parallel training uses an event-driven approach:

```
1. Generate skill A (no deps) → Launch immediately
2. Generate skill B (no deps) → Launch immediately (parallel with A)
3. Generate skill C (depends on A, B) → Wait in queue
4. Skill A completes → Check if C can launch (still waiting for B)
5. Skill B completes → Launch skill C (deps satisfied)
```

### 2. Frontier Blocking

The system automatically detects when to pause generation:

```python
# Frontier OPEN: Keep generating
Generated: Collect_Wood → No dependencies → Add to queue
Generated: Collect_Stone → No dependencies → Add to queue
Generated: Eat_Food → No dependencies → Add to queue

# Frontier BLOCKED: Pause generation
Generated: Make_Pickaxe → Depends on [Collect_Wood, Collect_Stone]
  → Both are training → FRONTIER BLOCKED → Wait for completion
```

### 3. Folder Structure

```
exp/parallel_test/
├── skills/                              # Permanent storage
│   ├── 0_Collect_Wood/
│   │   ├── expert_0_policy/             # Individual expert network
│   │   │   └── params.pkl               # (50M frames)
│   │   ├── 0.py                         # Generated module (archived)
│   │   └── training.log                 # Training logs
│   └── 1_Collect_Stone/
│       └── expert_1_policy/
│           └── params.pkl               # (40M frames)
│
├── training_runs/                       # Temporary (active training only)
│   └── skill_2_Make_Pickaxe/
│       ├── 2.py                         # Module with remapping
│       ├── 2_policies/                  # Full MoE (local indices)
│       │   └── checkpoint_*.pkl
│       └── training.log
│
├── checkpoints/
│   ├── global_latest.pkl                # Metadata only (no network params)
│   └── global.lock                      # For concurrent processing
│
└── scheduler_state.json                 # Scheduler state
```

## Troubleshooting

### Issue: Skills stuck in "waiting" state

**Check dependencies:**
```bash
cat exp/parallel_test/scheduler_state.json | jq '.skills[] | select(.status=="waiting") | {name: .skill_name, deps: .dependencies}'
```

**Solution:** Ensure dependency skills have completed successfully.

### Issue: Tmux session doesn't exist but skill shows "running"

**Diagnose:**
```bash
# Check if session really exists
tmux has-session -t flowrl_e0_Collect_Wood
echo $?  # 0 = exists, 1 = doesn't exist
```

**Solution:** Kill the scheduler process and restart. The scheduler will resume from saved state.

### Issue: "Failed to acquire global checkpoint lock"

**Cause:** Multiple skills completed simultaneously and one got stuck holding the lock.

**Solution:**
```bash
# Remove stale lock file
rm exp/parallel_test/checkpoints/global.lock

# Restart training
```

### Issue: Out of memory during training

**Cause:** Too many parallel skills or skills with many dependencies.

**Solutions:**
1. Reduce `--max_parallel_skills`
2. Reduce `--num_envs` (default 1024)
3. Limit skill dependencies (modify prompts)

### Issue: Training fails with remapping errors

**Check module:**
```bash
# View generated module to verify remapping
cat exp/parallel_test/training_runs/skill_2_Make_Pickaxe/2.py | head -20
```

**Should see:**
```python
# Auto-generated training module for: Make_Pickaxe
# Global expert index: 2
GLOBAL_TO_LOCAL = {0: 0, 1: 1, 2: 2}
LOCAL_TO_GLOBAL = {0: 0, 1: 1, 2: 2}
```

## Performance Expectations

### Speedup

**Sequential (baseline):**
- 3 skills @ 15 min each = 45 minutes total

**Parallel (3 slots):**
- 3 skills @ 15 min each = ~15 minutes total (3x speedup)

**Realistic (varied durations):**
- Skills: [10min, 15min, 180min] sequential = 205min
- Parallel: [10min, 15min, 180min] = ~180min (1.15x speedup)

→ Speedup depends on skill independence and duration variance

### Resource Usage

**Memory:**
- Each training session: 4-6 GB GPU memory
- MoE with 3 experts: ~6-8 GB
- Limit parallel sessions based on GPU capacity

**Disk Space:**
- Each skill: ~500 MB (checkpoints + logs)
- 20 skills: ~10 GB total

## Comparison: Sequential vs Parallel

| Aspect | Sequential | Parallel |
|--------|-----------|----------|
| **Wall Time** | 100% | 30-60% (2-3x faster) |
| **GPU Utilization** | ~33% | ~85-95% |
| **Memory Usage** | Low (1 skill at a time) | Higher (N skills) |
| **Complexity** | Simple | More complex |
| **Failure Isolation** | One failure blocks all | Failures don't block |
| **Debugging** | Easier (sequential logs) | Harder (concurrent logs) |

## Advanced Usage

### Custom Dependency Resolution

Modify `get_skill_dependencies()` in `bottomup_parallel.py`:

```python
def get_skill_dependencies(skill_name: str, all_skills: dict, training_skills: dict) -> list:
    # Custom logic here
    # Example: Add manual dependency overrides
    manual_deps = {
        "Make_Diamond_Pickaxe": ["Mine_Diamond", "Make_Pickaxe"]
    }
    if skill_name in manual_deps:
        return manual_deps[skill_name]

    # Default: use requirement-based detection
    # ...
```

### Integration with Existing Checkpoints

To continue from a sequential training run:

```bash
# 1. Copy sequential checkpoint to parallel structure
mkdir -p exp/parallel_test/checkpoints
cp exp/bottom_up/checkpoint.pkl exp/parallel_test/checkpoints/global_latest.pkl

# 2. Extract individual experts (if needed)
# TODO: Write migration script

# 3. Resume parallel training
python -m flowrl.bottomup_parallel --graph-path exp/parallel_test/
```

## Testing

### Run Unit Tests

```bash
# All parallel training tests
python -m pytest tests/parallel/ -v

# Specific test
python -m pytest tests/parallel/test_integration.py -v
```

### Manual Integration Test

```bash
# Generate 3 skills with 2 parallel
python -m flowrl.bottomup_parallel \
  --max_nodes 3 \
  --max_parallel_skills 2 \
  --total_timesteps 1e7 \
  --no-use_wandb \
  --graph-path exp/test_parallel/

# Verify outputs
ls -la exp/test_parallel/skills/
cat exp/test_parallel/scheduler_state.json | jq
```

## LLM Configuration

You can specify which LLM to use for skill generation:

```bash
# Use cheaper model for testing (10-20x cheaper)
python -m flowrl.bottomup_parallel \
  --llm_name gpt-4o-mini \
  --max_nodes 3

# Use default model (GPT-5, highest quality)
python -m flowrl.bottomup_parallel \
  --max_nodes 10
```

See `LLM_CONFIGURATION.md` for full details on:
- Supported models
- Cost comparisons
- When to use each model
- Troubleshooting

## Next Steps

1. **Start small:** Test with 3-5 skills and 2 parallel sessions
2. **Use cheap LLM:** Add `--llm_name gpt-4o-mini` for initial testing
3. **Monitor closely:** Watch tmux sessions and logs
4. **Tune parameters:** Adjust `max_parallel_skills` based on GPU memory
5. **Scale up:** Gradually increase to production workloads with better LLM

## Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review logs in `training_runs/` and `skills/` folders
3. Inspect `scheduler_state.json` for stuck states
4. Review test suite: `tests/parallel/test_integration.py`

## Summary

**Parallel training is recommended when:**
- ✅ You have GPU memory for 2+ concurrent sessions
- ✅ Skills have diverse dependencies (enables parallelism)
- ✅ You need faster wall-clock time
- ✅ You're comfortable with tmux and concurrent debugging

**Stick with sequential training when:**
- ❌ Limited GPU memory (< 12 GB)
- ❌ Skills are mostly sequential (few independent skills)
- ❌ Debugging a new setup
- ❌ Simpler infrastructure is preferred
