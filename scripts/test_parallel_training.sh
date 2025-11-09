#!/bin/bash
# Test script for parallel training with minimal configuration
# This runs a small test with 3 skills and 2 parallel sessions
#
# Usage:
#   ./scripts/test_parallel_training.sh                    # Uses gpt-4o-mini (default)
#   LLM_MODEL=gpt-5 ./scripts/test_parallel_training.sh    # Uses gpt-5
#   LLM_MODEL=gpt-4o ./scripts/test_parallel_training.sh   # Uses gpt-4o

set -e  # Exit on error

echo "========================================"
echo "Parallel Training Test Script"
echo "========================================"
echo ""

# Configuration
TEST_DIR="exp/test_parallel_$(date +%Y%m%d_%H%M%S)"
MAX_SKILLS=3
PARALLEL_SESSIONS=2
TIMESTEPS=5e7  # 50M timesteps (relatively quick)
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"  # Default to cheap model, override with LLM_MODEL env var

echo "Configuration:"
echo "  Test directory: $TEST_DIR"
echo "  Max skills: $MAX_SKILLS"
echo "  Parallel sessions: $PARALLEL_SESSIONS"
echo "  Timesteps per skill: $TIMESTEPS"
echo "  LLM model: $LLM_MODEL"
echo ""

# Create test directory
mkdir -p "$TEST_DIR"

echo "Starting parallel training..."
echo ""

# Run parallel training
python -m flowrl.bottomup_parallel \
  --graph-path "$TEST_DIR" \
  --max_nodes "$MAX_SKILLS" \
  --max_parallel_skills "$PARALLEL_SESSIONS" \
  --total_timesteps "$TIMESTEPS" \
  --scheduler_poll_interval 10 \
  --llm_name "$LLM_MODEL" \
  --no-use_wandb \
  --env_name Craftax-Symbolic-v1

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo ""

# Show results
echo "Scheduler State:"
cat "$TEST_DIR/scheduler_state.json" | jq '.skills | group_by(.status) | map({status: .[0].status, count: length})'
echo ""

echo "Completed Skills:"
ls -1 "$TEST_DIR/skills/"
echo ""

echo "Global Checkpoint:"
if [ -f "$TEST_DIR/checkpoints/global_latest.pkl" ]; then
    echo "  ✓ Global checkpoint exists"
    python -c "import pickle; ckpt = pickle.load(open('$TEST_DIR/checkpoints/global_latest.pkl', 'rb')); print(f'  Skills: {len(ckpt.get(\"skills\", {}))}'); [print(f'    - {name} (expert {info[\"expert_idx\"]}, {info[\"total_frames\"]:,} frames)') for name, info in ckpt.get('skills', {}).items()]"
else
    echo "  ✗ Global checkpoint not found"
fi
echo ""

echo "Test directory: $TEST_DIR"
echo ""
echo "To view scheduler state:"
echo "  cat $TEST_DIR/scheduler_state.json | jq"
echo ""
echo "To inspect a skill:"
echo "  ls -la $TEST_DIR/skills/0_*/"
echo ""
