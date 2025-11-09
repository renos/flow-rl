#!/bin/bash
# Run PPO Flow on a three-node descend plan: craft up to iron sword, then descend

set -euo pipefail

echo "========================================"
echo "Run: Three-Node Descend (wintrinsics, GAE=0.8, p=0.99)"
echo "========================================"

# Configurable params (override via env vars)
ENV_NAME=${ENV_NAME:-Craftax-Symbolic-v1}
MODULE_PATH=${MODULE_PATH:-$(pwd)/exp/three_node_descend/1.py}
WANDB_PROJECT=${WANDB_PROJECT:-flow_rl_three_node_descend_wintr}
WANDB_ENTITY=${WANDB_ENTITY:-}
NUM_ENVS=${NUM_ENVS:-1024}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1e10}
LR=${LR:-2e-4}
NUM_STEPS=${NUM_STEPS:-64}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
LAYER_SIZE=${LAYER_SIZE:-512}
SEED=${SEED:-42}
USE_OPTIMISTIC_RESETS=${USE_OPTIMISTIC_RESETS:-true}
OPTIMISTIC_RESET_RATIO=${OPTIMISTIC_RESET_RATIO:-16}
# Goal state index for descend node (task_9); adjust as needed by your scheduler
SUCCESS_STATE=${SUCCESS_STATE:-10}
GPU_ID=${GPU_ID:-3}

echo "Configuration:"
echo "  ENV_NAME:             $ENV_NAME"
echo "  MODULE_PATH:          $MODULE_PATH"
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
echo "  SUCCESS_STATE:        $SUCCESS_STATE"
echo "  GPU_ID:               $GPU_ID"
echo

# Run PPO Flow
env CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m flowrl.ppo_flow_w_reset \
  --env_name "$ENV_NAME" \
  --module_path "$MODULE_PATH" \
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
  --success_state "$SUCCESS_STATE" \
  --latest_reset_prob 0.99 \
  --per_skill_balance \
  --per_skill_balance_threshold 64 \
  --per_skill_balance_cap 10 \
  --gae_lambda 0.8

echo
echo "Done. Policies will be saved next to the module:"
echo "  $(dirname "$MODULE_PATH")/$(basename "${MODULE_PATH%.py}")_policies/policies"
echo

