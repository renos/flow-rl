#!/usr/bin/env bash
# Craftax parallel training with state reset curriculum
# Usage: run from repo root
#   ./scripts/craftax_bottomup_wstatereset.sh [extra bottomup_parallel args]
# or
#   bash scripts/craftax_bottomup_wstatereset.sh [extra args]

set -euo pipefail

echo "========================================"
echo "Craftax Parallel Training (w/ State Reset)"
echo "========================================"
echo ""

# Resolve repo root (this script lives in scripts/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configuration (override via env vars)
GRAPH_PATH="${GRAPH_PATH:-${REPO_ROOT}/exp/craftax_bottomup_reset}"
LLM_MODEL="${LLM_MODEL:-gpt-5}"
GPU_IDS="${GPU_IDS:-0}"
MAX_SKILLS="${MAX_SKILLS:-40}"
PARALLEL_SESSIONS="${PARALLEL_SESSIONS:-1}"
# Per-skill budget as requested (100M); also pass total_timesteps for PPO settings
SKILL_BUDGET="${SKILL_BUDGET:-100000000}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
TARGET_SKILL="${TARGET_SKILL:-Collect Diamond}"
# Default to jax2 as discussed; override with CONDA_ENV if needed
CONDA_ENV="${CONDA_ENV:-jax2}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-1024}"

# PPO Flow with Reset specific parameters (from run_craftax_bow_and_sword.sh)
USE_OPTIMISTIC_RESETS="${USE_OPTIMISTIC_RESETS:-true}"
OPTIMISTIC_RESET_RATIO="${OPTIMISTIC_RESET_RATIO:-16}"
# Progressive curriculum settings
LATEST_RESET_PROB="${LATEST_RESET_PROB:-0.99}"
PROGRESSIVE_RESET_CURRICULUM="${PROGRESSIVE_RESET_CURRICULUM:-true}"
PROGRESSIVE_RESET_THRESHOLD="${PROGRESSIVE_RESET_THRESHOLD:-0.3}"
SUCCESS_RATE_EMA_ALPHA="${SUCCESS_RATE_EMA_ALPHA:-0.10}"
# Per-skill balancing
PER_SKILL_BALANCE="${PER_SKILL_BALANCE:-true}"
PER_SKILL_BALANCE_THRESHOLD="${PER_SKILL_BALANCE_THRESHOLD:-64}"
PER_SKILL_BALANCE_CAP="${PER_SKILL_BALANCE_CAP:-10}"
# GAE settings
GAE_LAMBDA="${GAE_LAMBDA:-0.8}"

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
echo "State Reset Parameters:"
echo "  Use optimistic resets:       $USE_OPTIMISTIC_RESETS"
echo "  Optimistic reset ratio:      $OPTIMISTIC_RESET_RATIO"
echo "  Latest reset prob:           $LATEST_RESET_PROB"
echo "  Progressive curriculum:      $PROGRESSIVE_RESET_CURRICULUM"
echo "  Progressive threshold:       $PROGRESSIVE_RESET_THRESHOLD"
echo "  Success rate EMA alpha:      $SUCCESS_RATE_EMA_ALPHA"
echo "  Per-skill balance:           $PER_SKILL_BALANCE"
echo "  Per-skill balance threshold: $PER_SKILL_BALANCE_THRESHOLD"
echo "  Per-skill balance cap:       $PER_SKILL_BALANCE_CAP"
echo "  GAE lambda:                  $GAE_LAMBDA"
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

echo "Starting parallel training with state reset curriculum..."
echo ""

# Ensure potential conflicting library paths are not inherited by launched jobs
unset LD_LIBRARY_PATH || true

# Build optimistic resets argument
OPTIMISTIC_RESETS_ARG=""
if [ "$USE_OPTIMISTIC_RESETS" = "true" ]; then
  OPTIMISTIC_RESETS_ARG="--use_optimistic_resets"
fi

# Build progressive curriculum argument
PROGRESSIVE_CURRICULUM_ARG=""
if [ "$PROGRESSIVE_RESET_CURRICULUM" = "true" ]; then
  PROGRESSIVE_CURRICULUM_ARG="--progressive_reset_curriculum"
fi

# Build per-skill balance argument
PER_SKILL_BALANCE_ARG=""
if [ "$PER_SKILL_BALANCE" = "true" ]; then
  PER_SKILL_BALANCE_ARG="--per_skill_balance"
fi

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
  --wandb_project flow_rl_craftax_parallel_reset \
  --seed "$SEED" \
  --conda_env "$CONDA_ENV" \
  --ppo_flow_variant ppo_flow_w_reset \
  $OPTIMISTIC_RESETS_ARG \
  --optimistic_reset_ratio "$OPTIMISTIC_RESET_RATIO" \
  --latest_reset_prob "$LATEST_RESET_PROB" \
  $PROGRESSIVE_CURRICULUM_ARG \
  --progressive_reset_threshold "$PROGRESSIVE_RESET_THRESHOLD" \
  $PER_SKILL_BALANCE_ARG \
  --per_skill_balance_threshold "$PER_SKILL_BALANCE_THRESHOLD" \
  --per_skill_balance_cap "$PER_SKILL_BALANCE_CAP" \
  --success_rate_ema_alpha "$SUCCESS_RATE_EMA_ALPHA" \
  --gae_lambda "$GAE_LAMBDA" \
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
