#!/bin/bash
# Run PPO Flow with Transformer-XL on a single-node experiment: descend from Floor 0 to Floor 1

set -euo pipefail

echo "========================================"
echo "Run: One-Node Descend Floor (PPO Flow with Transformer-XL)"
echo "========================================"

# Configurable params (override via env vars)
ENV_NAME=${ENV_NAME:-Craftax-Symbolic-v1}
MODULE_PATH=${MODULE_PATH:-$(pwd)/exp/two_node_descend_intr/1.py}
WANDB_PROJECT=${WANDB_PROJECT:-flow_rl_two_node_descend_wintr_txrl}
WANDB_ENTITY=${WANDB_ENTITY:-}
NUM_ENVS=${NUM_ENVS:-1024}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1e10}
LR=${LR:-2e-4}
NUM_STEPS=${NUM_STEPS:-64}
UPDATE_EPOCHS=${UPDATE_EPOCHS:-4}
NUM_MINIBATCHES=${NUM_MINIBATCHES:-8}
SEED=${SEED:-42}
USE_OPTIMISTIC_RESETS=${USE_OPTIMISTIC_RESETS:-true}
OPTIMISTIC_RESET_RATIO=${OPTIMISTIC_RESET_RATIO:-16}
# Goal state after completing task_0 (descend to Floor 1)
SUCCESS_STATE=${SUCCESS_STATE:-5}
GPU_ID=${GPU_ID:-2}
GAE_LAMBDA=${GAE_LAMBDA:-0.95}

# TransformerXL-specific hyperparameters (from transformerXL_PPO_JAX defaults)
EMBED_SIZE=${EMBED_SIZE:-256}
HIDDEN_LAYERS=${HIDDEN_LAYERS:-256}
NUM_HEADS=${NUM_HEADS:-8}
QKV_FEATURES=${QKV_FEATURES:-256}
NUM_LAYERS=${NUM_LAYERS:-2}
WINDOW_MEM=${WINDOW_MEM:-128}
WINDOW_GRAD=${WINDOW_GRAD:-64}
GATING=${GATING:-false}
GATING_BIAS=${GATING_BIAS:-0.0}

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
echo "  SEED:                 $SEED"
echo "  USE_OPTIMISTIC_RESETS:$USE_OPTIMISTIC_RESETS"
echo "  OPTIMISTIC_RESET_RATIO:$OPTIMISTIC_RESET_RATIO"
echo "  SUCCESS_STATE:        $SUCCESS_STATE"
echo "  GPU_ID:               $GPU_ID"
echo "  GAE_LAMBDA:           $GAE_LAMBDA"
echo ""
echo "TransformerXL Configuration:"
echo "  EMBED_SIZE:           $EMBED_SIZE"
echo "  HIDDEN_LAYERS:        $HIDDEN_LAYERS"
echo "  NUM_HEADS:            $NUM_HEADS"
echo "  QKV_FEATURES:         $QKV_FEATURES"
echo "  NUM_LAYERS:           $NUM_LAYERS"
echo "  WINDOW_MEM:           $WINDOW_MEM"
echo "  WINDOW_GRAD:          $WINDOW_GRAD"
echo "  GATING:               $GATING"
echo "  GATING_BIAS:          $GATING_BIAS"
echo

# Run PPO Flow with Transformer-XL
env CUDA_VISIBLE_DEVICES="$GPU_ID" \
python -m flowrl.ppo_flow_txrl \
  --env_name "$ENV_NAME" \
  --module_path "$MODULE_PATH" \
  --num_envs "$NUM_ENVS" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --lr "$LR" \
  --num_steps "$NUM_STEPS" \
  --update_epochs "$UPDATE_EPOCHS" \
  --num_minibatches "$NUM_MINIBATCHES" \
  --seed "$SEED" \
  --gae_lambda "$GAE_LAMBDA" \
  --embed_size "$EMBED_SIZE" \
  --hidden_layers "$HIDDEN_LAYERS" \
  --num_heads "$NUM_HEADS" \
  --qkv_features "$QKV_FEATURES" \
  --num_layers "$NUM_LAYERS" \
  --window_mem "$WINDOW_MEM" \
  --window_grad "$WINDOW_GRAD" \
  $( [ "$GATING" = "true" ] && echo "--gating" ) \
  --gating_bias "$GATING_BIAS" \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  $( [ "$USE_OPTIMISTIC_RESETS" = "true" ] && echo "--use_optimistic_resets" ) \
  --optimistic_reset_ratio "$OPTIMISTIC_RESET_RATIO" \
  --success_state "$SUCCESS_STATE" \
  --latest_reset_prob 0.8 \
  --per_skill_balance \
  --per_skill_balance_threshold 64 \
  --per_skill_balance_cap 10

echo
echo "Done. Policies will be saved next to the module:"
echo "  $(dirname "$MODULE_PATH")/$(basename "${MODULE_PATH%.py}")_policies/policies"
echo
