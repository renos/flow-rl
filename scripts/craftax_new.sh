#!/usr/bin/env bash
# Craftax parallel training (single-session) with target skill preset
# Usage: run from repo root
#   ./craftax_new [extra bottomup_parallel args]
# or
#   bash craftax_new [extra args]

set -euo pipefail

echo "========================================"
echo "Craftax New Parallel Training"
echo "========================================"
echo ""

# Resolve repo root (this script lives at repo root)
# Repo root is the parent of the scripts directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configuration (override via env vars)
GRAPH_PATH="${GRAPH_PATH:-${REPO_ROOT}/exp/craftax_new}"
LLM_MODEL="${LLM_MODEL:-gpt-5}"
GPU_IDS="${GPU_IDS:-0}"
MAX_SKILLS="${MAX_SKILLS:-40}"
PARALLEL_SESSIONS="${PARALLEL_SESSIONS:-1}"
# Per-skill budget as requested (100M); also pass total_timesteps for PPO settings
SKILL_BUDGET="${SKILL_BUDGET:-100000000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
TARGET_SKILL="${TARGET_SKILL:-Descend to Dungeon (Floor 1)}"
# Default to jax2 as discussed; override with CONDA_ENV if needed
CONDA_ENV="${CONDA_ENV:-jax2}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-1024}"

echo "Configuration:"
echo "  Graph path:         $GRAPH_PATH"
echo "  LLM model:          $LLM_MODEL"
echo "  GPUs:               $GPU_IDS"
echo "  Max skills:         $MAX_SKILLS"
echo "  Parallel sessions:  $PARALLEL_SESSIONS"
echo "  Skill budget:       $SKILL_BUDGET"
echo "  Total timesteps:    $TOTAL_TIMESTEPS"
echo "  Poll interval:      ${POLL_INTERVAL}s"
echo "  Target skill:       $TARGET_SKILL"
echo "  Conda env:          $CONDA_ENV"
echo "  Num envs:           $NUM_ENVS"
echo ""

# Create directory if it doesn't exist
mkdir -p "$GRAPH_PATH"

# Activate conda environment for the parent orchestrator process (optional)
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "[warn] conda not found on PATH; ensure env '$CONDA_ENV' is active." >&2
  fi
fi

echo "Starting parallel training..."
echo ""

# Ensure potential conflicting library paths are not inherited by launched jobs
unset LD_LIBRARY_PATH || true

# Run parallel training (append any extra args passed to this script)
CUDA_VISIBLE_DEVICES="$GPU_IDS" python -m flowrl.bottomup_parallel \
  --env_name "Craftax-Symbolic-v1" \
  --target_skill "$TARGET_SKILL" \
  --graph-path "$GRAPH_PATH" \
  --llm_name "$LLM_MODEL" \
  --max_nodes "$MAX_SKILLS" \
  --max_parallel_skills "$PARALLEL_SESSIONS" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --skill_timesteps_budget "$SKILL_BUDGET" \
  --scheduler_poll_interval "$POLL_INTERVAL" \
  --max_retries 2 \
  --num_envs "$NUM_ENVS" \
  --use_wandb \
  --wandb_project flow_rl_craftax_parallel \
  --seed "$SEED" \
  --conda_env "$CONDA_ENV" \
  "$@"

echo ""
echo "========================================"
echo "Training Launched"
echo "========================================"
echo ""

echo "Monitor tmux sessions:"
echo "  tmux list-sessions | grep flowrl_"
echo "Attach to a session:"
echo "  tmux attach -t flowrl_e0_<skill_name>"
echo ""
