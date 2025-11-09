#!/bin/bash
# Parallel training initialization test for Craftax
#
# This script runs parallel skill training with:
# - Craftax-Symbolic-v1 environment
# - GPUs 0-7
# - gpt-5-nano for cost-effective testing
# - 3-5 skills with 2-3 parallel sessions

set -e  # Exit on error

echo "========================================"
echo "Craftax Parallel Training Init Test"
echo "========================================"
echo ""

# Configuration
GRAPH_PATH="/home/renos/flow-rl/exp/craftax_init_parallel_test"
LLM_MODEL="gpt-5"
GPU_IDS="0,1,2,3,4,5,6,7"
MAX_SKILLS="${MAX_SKILLS:-25}"
PARALLEL_SESSIONS="${PARALLEL_SESSIONS:-5}"
TIMESTEPS="${TIMESTEPS:-1e8}"  # 100M timesteps per skill
POLL_INTERVAL=30
CONDA_ENV="${CONDA_ENV:-jax}"

echo "Configuration:"
echo "  Graph path: $GRAPH_PATH"
echo "  LLM model: $LLM_MODEL"
echo "  GPUs: $GPU_IDS"
echo "  Max skills: $MAX_SKILLS"
echo "  Parallel sessions: $PARALLEL_SESSIONS"
echo "  Timesteps per skill: $TIMESTEPS"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "  Conda env: $CONDA_ENV"
echo ""

# Create directory if it doesn't exist
mkdir -p "$GRAPH_PATH"

echo "Starting parallel training..."
echo ""

# Activate conda environment for the parent orchestrator process
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "[warn] conda not found on PATH; ensure env '$CONDA_ENV' is active." >&2
  fi
fi

# Run parallel training
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m flowrl.bottomup_parallel \
  --env_name Craftax-Symbolic-v1 \
  --graph-path "$GRAPH_PATH" \
  --llm_name "$LLM_MODEL" \
  --max_nodes "$MAX_SKILLS" \
  --max_parallel_skills "$PARALLEL_SESSIONS" \
  --total_timesteps "$TIMESTEPS" \
  --scheduler_poll_interval "$POLL_INTERVAL" \
  --max_retries 2 \
  --num_envs 1024 \
  --use_wandb \
  --wandb_project flow_rl_craftax_parallel \
  --seed 42 \
  --conda_env "$CONDA_ENV"

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""

# Show results
echo "Results:"
echo ""

if [ -f "$GRAPH_PATH/scheduler_state.json" ]; then
    echo "Scheduler State:"
    cat "$GRAPH_PATH/scheduler_state.json" | jq '.skills | group_by(.status) | map({status: .[0].status, count: length})'
    echo ""
fi

if [ -d "$GRAPH_PATH/skills" ]; then
    echo "Completed Skills:"
    ls -1 "$GRAPH_PATH/skills/" 2>/dev/null || echo "  No skills completed yet"
    echo ""
fi

if [ -f "$GRAPH_PATH/checkpoints/global_latest.pkl" ]; then
    echo "Global checkpoint exists: ✓"
else
    echo "Global checkpoint: Not found"
fi

echo ""
echo "View full state:"
echo "  cat $GRAPH_PATH/scheduler_state.json | jq"
echo ""
echo "Monitor tmux sessions:"
echo "  tmux list-sessions | grep flowrl_"
echo ""
echo "Attach to a session:"
echo "  tmux attach -t flowrl_e0_<skill_name>"
echo ""
