#!/bin/bash
# Simple runner: PPO-RNN on Craftax for 1B timesteps, saves checkpoint

set -euo pipefail

echo "========================================"
echo "Run: PPO-RNN on Craftax (1B steps)"
echo "========================================"

# Configurable params (override via env vars)
ENV_NAME=${ENV_NAME:-Craftax-Symbolic-v1}
WANDB_PROJECT=${WANDB_PROJECT:-flow_rl_ppo_rnn_1b}
WANDB_ENTITY=${WANDB_ENTITY:-}
NUM_ENVS=${NUM_ENVS:-1024}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1e9}
LR=${LR:-2e-4}
NUM_STEPS=${NUM_STEPS:-64}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
LAYER_SIZE=${LAYER_SIZE:-512}
SEED=${SEED:-42}
USE_OPTIMISTIC_RESETS=${USE_OPTIMISTIC_RESETS:-true}
OPTIMISTIC_RESET_RATIO=${OPTIMISTIC_RESET_RATIO:-16}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_rnn_craftax_1b}
GPU_ID=${GPU_ID:-0}

echo "Configuration:"
echo "  ENV_NAME:             $ENV_NAME"
echo "  WANDB_PROJECT:        $WANDB_PROJECT"
echo "  WANDB_ENTITY:         ${WANDB_ENTITY:-<none>}"
echo "  NUM_ENVS:             $NUM_ENVS"
echo "  TOTAL_TIMESTEPS:      $TOTAL_TIMESTEPS"
echo "  LR:                   $LR"
echo "  NUM_STEPS:            $NUM_STEPS"
echo "  UPDATE_EPOCHS:        $UPDATE_EPOCHS"
echo "  NUM_MINIBATCHES:      $NUM_MINIBATCHES"
echo "  LAYER_SIZE:           $LAYER_SIZE"
echo "  SEED:                 $SEED"
echo "  USE_OPTIMISTIC_RESETS:$USE_OPTIMISTIC_RESETS"
echo "  OPTIMISTIC_RESET_RATIO:$OPTIMISTIC_RESET_RATIO"
echo "  EXPERIMENT_NAME:      $EXPERIMENT_NAME"
echo "  GPU_ID:               $GPU_ID"
echo

# Important: run as a file (not module) so ppo_rnn's imports resolve
env CUDA_VISIBLE_DEVICES="$GPU_ID" \
python flowrl/ppo_rnn.py \
  --env_name "$ENV_NAME" \
  --num_envs "$NUM_ENVS" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --lr "$LR" \
  --num_steps "$NUM_STEPS" \
  --update_epochs "$UPDATE_EPOCHS" \
  --num_minibatches "$NUM_MINIBATCHES" \
  --layer_size "$LAYER_SIZE" \
  --seed "$SEED" \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  $( [ "$USE_OPTIMISTIC_RESETS" = "true" ] && echo "--use_optimistic_resets" ) \
  --optimistic_reset_ratio "$OPTIMISTIC_RESET_RATIO" \
  --save_policy \
  --experiment_name "$EXPERIMENT_NAME"

echo
echo "Done. Policy checkpoint saved under: exp/$EXPERIMENT_NAME/policies"
echo

