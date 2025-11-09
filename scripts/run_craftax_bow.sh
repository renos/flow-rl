#!/bin/bash
# Run PPO Flow on craftax_bow with progressive curriculum: collect wood → craft pickaxe → craft arrows → descend → open chest → kill monsters

set -euo pipefail

echo "========================================"
echo "Run: Craftax Bow (Progressive Curriculum, GAE=0.95, p=0.99)"
echo "========================================"

# Configurable params (override via env vars)
ENV_NAME=${ENV_NAME:-Craftax-Symbolic-v1}
MODULE_PATH=${MODULE_PATH:-$(pwd)/exp/craftax_bow/1.py}
WANDB_PROJECT=${WANDB_PROJECT:-flow_rl_craftax_bow}
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
# Goal state index for final task (task_6: kill monsters); 7 states total (0-6, success at 7)
SUCCESS_STATE=${SUCCESS_STATE:-7}
# Progressive curriculum settings
PROGRESSIVE_RESET_THRESHOLD=${PROGRESSIVE_RESET_THRESHOLD:-0.2}
SUCCESS_RATE_EMA_ALPHA=${SUCCESS_RATE_EMA_ALPHA:-0.15}
GPU_ID=${GPU_ID:-2}

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
echo "  PROGRESSIVE_THRESHOLD:$PROGRESSIVE_RESET_THRESHOLD"
echo "  EMA_ALPHA:            $SUCCESS_RATE_EMA_ALPHA"
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
  --gae_lambda 0.95 \
  --progressive_reset_curriculum \
  --progressive_reset_threshold "$PROGRESSIVE_RESET_THRESHOLD" \
  --success_rate_ema_alpha "$SUCCESS_RATE_EMA_ALPHA"

echo
echo "Done. Policies will be saved next to the module:"
echo "  $(dirname "$MODULE_PATH")/$(basename "${MODULE_PATH%.py}")_policies/policies"
echo
