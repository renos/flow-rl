#!/bin/bash
# Run PPO Flow on the premade Craftax baseline with GAElambda=0.8 on GPU 6

set -euo pipefail

echo "========================================"
echo "Run: Premade Craftax PPO Baseline (GPU 6, GAE=0.8)"
echo "========================================"

# Configurable params (override via env vars)
ENV_NAME=${ENV_NAME:-Craftax-Symbolic-v1}
MODULE_PATH=${MODULE_PATH:-$(pwd)/exp/premade_craftax/1.py}
WANDB_PROJECT=${WANDB_PROJECT:-flow_rl_premade_craftax_baseline}
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
SUCCESS_STATE=${SUCCESS_STATE:-12}
GPU_ID=${GPU_ID:-6}
GAE_LAMBDA=${GAE_LAMBDA:-0.8}

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
echo "  GPU_ID:              $GPU_ID"
echo "  GAE_LAMBDA:           $GAE_LAMBDA"
echo

# Run PPO Flow
env CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m flowrl.ppo_flow \
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
  --gae_lambda "$GAE_LAMBDA" \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  $( [ "$USE_OPTIMISTIC_RESETS" = "true" ] && echo "--use_optimistic_resets" ) \
  --optimistic_reset_ratio "$OPTIMISTIC_RESET_RATIO" \
  --success_state "$SUCCESS_STATE"

echo
echo "Done. Policies will be saved next to the module:"
echo "  $(dirname "$MODULE_PATH")/$(basename "${MODULE_PATH%.py}")_policies/policies"
echo
