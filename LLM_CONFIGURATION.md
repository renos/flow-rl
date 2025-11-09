# LLM Configuration Guide

## Overview

Flow-RL now supports configurable LLM models for skill generation. By default, the system uses GPT-5, but you can specify cheaper or different models for testing and experimentation.

## Command-Line Usage

### Basic Usage

```bash
# Use default model (GPT-5)
python -m flowrl.bottomup --env_name Craftax-Symbolic-v1

# Use GPT-4o-mini (cheaper for testing)
python -m flowrl.bottomup --env_name Craftax-Symbolic-v1 --llm_name gpt-4o-mini

# Use GPT-4o
python -m flowrl.bottomup --env_name Craftax-Symbolic-v1 --llm_name gpt-4o

# Parallel training with custom LLM
python -m flowrl.bottomup_parallel --llm_name gpt-4o-mini --max_nodes 5
```

## Supported Models

The exact model names depend on your OpenAI API configuration. Common options:

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `gpt-5` | Latest GPT-5 (default) | Production, best quality |
| `gpt-4o` | GPT-4 Optimized | Good balance of quality/cost |
| `gpt-4o-mini` | Smaller, cheaper GPT-4 | Testing, rapid iteration |
| `gpt-4-turbo` | GPT-4 Turbo | Fast, high quality |
| `o1-preview` | Reasoning model | Complex planning tasks |
| `o1-mini` | Cheaper reasoning model | Testing reasoning |

**Note:** The actual model name strings depend on your API provider. Check your OpenAI or custom API documentation for exact names.

## Examples

### Testing with Cheap Model

```bash
# Quick test with 3 skills using cheaper model
python -m flowrl.bottomup_parallel \
  --llm_name gpt-4o-mini \
  --max_nodes 3 \
  --max_parallel_skills 2 \
  --total_timesteps 5e7 \
  --no-use_wandb
```

### Production Run with GPT-5

```bash
# Production run with default high-quality model
python -m flowrl.bottomup \
  --max_nodes 20 \
  --total_timesteps 2e8 \
  --use_wandb
```

### Mixed Approach

You can also start with a cheaper model for initial testing, then switch to a better model:

```bash
# Phase 1: Test with cheap model (3 skills)
python -m flowrl.bottomup \
  --llm_name gpt-4o-mini \
  --max_nodes 3 \
  --graph-path exp/test_run/

# Phase 2: Continue with better model (if checkpoint loading works)
python -m flowrl.bottomup \
  --llm_name gpt-5 \
  --max_nodes 10 \
  --graph-path exp/test_run/ \
  --current_i 3
```

## Cost Considerations

**Approximate costs per skill generation** (varies by skill complexity):

| Model | Cost per Skill | 10 Skills | Notes |
|-------|----------------|-----------|-------|
| gpt-5 | ~$0.50-1.00 | ~$5-10 | Highest quality |
| gpt-4o | ~$0.20-0.40 | ~$2-4 | Good quality |
| gpt-4o-mini | ~$0.05-0.10 | ~$0.50-1 | Fast testing |

*Note: Actual costs depend on prompt size, skill complexity, and number of retries.*

## Implementation Details

### Default Behavior

If `--llm_name` is not specified:
- Uses GPT-5 by default (via agentkit)
- Falls back to custom llm_api if available

### How It Works

1. **Command-line arg:** `--llm_name gpt-4o-mini`
2. **Flow class:** Reads `args.llm_name`
3. **Generate graph:** Passes to `generate_graph(llm_name=...)`
4. **LLM module:** Creates API function with specified model

### Code Location

**Updated files:**
- `flowrl/llm/craftax/llm.py` - Accepts `llm_name` parameter
- `flowrl/llm/craftax_classic/llm.py` - Accepts `llm_name` parameter
- `flowrl/llm/fabrax/llm.py` - Accepts `llm_name` parameter
- `flowrl/llm/flow.py` - Passes `llm_name` to generators
- `flowrl/bottomup.py` - Adds `--llm_name` argument
- `flowrl/bottomup_parallel.py` - Adds `--llm_name` argument

## Custom LLM Providers

If you're using a custom LLM API (not OpenAI):

1. The system first tries to import custom `llm_api` module
2. Falls back to `agentkit.llm_api` if not available
3. Your custom provider should support the same model name strings

**Example custom API integration:**

```python
# In your custom llm_api module
def get_query(model_name):
    """Return query function for specified model."""
    return lambda prompt, max_gen: your_api_call(model_name, prompt, max_gen)
```

## Troubleshooting

### Invalid Model Name

**Error:** `InvalidRequestError: The model 'xyz' does not exist`

**Solution:** Check that the model name matches your API provider's naming:
```bash
# OpenAI models typically use hyphens
--llm_name gpt-4o-mini  # ✓ Correct
--llm_name gpt4omini    # ✗ Wrong
```

### API Rate Limits

**Error:** `RateLimitError: Rate limit exceeded`

**Solution:** Use a cheaper model with higher rate limits:
```bash
--llm_name gpt-4o-mini  # Higher rate limits than gpt-5
```

### Quality Issues with Cheaper Models

**Symptom:** Skills fail validation more often with cheaper models

**Solution:**
1. Use cheaper models for early testing only
2. Switch to `gpt-4o` or `gpt-5` for production
3. Increase `--max_nodes` to generate more skills (some will fail)

## Best Practices

### Development Workflow

1. **Initial testing:** Use `gpt-4o-mini` to verify setup
   ```bash
   --llm_name gpt-4o-mini --max_nodes 2
   ```

2. **Iteration:** Use `gpt-4o` for moderate quality/cost
   ```bash
   --llm_name gpt-4o --max_nodes 10
   ```

3. **Production:** Use `gpt-5` for best results
   ```bash
   --llm_name gpt-5 --max_nodes 40
   ```

### Cost Optimization

- Use `gpt-4o-mini` for debugging and setup
- Use `gpt-4o` for most experiments
- Reserve `gpt-5` for final production runs
- Enable `--no-use_wandb` during testing to reduce overhead

### Quality vs. Cost

**When to use cheaper models:**
- ✓ Testing setup and configuration
- ✓ Debugging training pipeline
- ✓ Prototyping new prompts
- ✓ Rapid iteration on small skill counts

**When to use expensive models:**
- ✓ Final production runs
- ✓ Complex environments (Craftax, Fabrax)
- ✓ Skills with many dependencies
- ✓ When skill quality is critical

## Examples by Use Case

### Quick Test (Cheapest)
```bash
python -m flowrl.bottomup_parallel \
  --llm_name gpt-4o-mini \
  --max_nodes 3 \
  --total_timesteps 5e7 \
  --no-use_wandb
```

### Development (Balanced)
```bash
python -m flowrl.bottomup \
  --llm_name gpt-4o \
  --max_nodes 10 \
  --total_timesteps 1e8
```

### Production (Best Quality)
```bash
python -m flowrl.bottomup_parallel \
  --llm_name gpt-5 \
  --max_nodes 40 \
  --max_parallel_skills 4 \
  --total_timesteps 2e8 \
  --use_wandb
```

## Summary

- **Default:** GPT-5 (highest quality)
- **Testing:** Use `--llm_name gpt-4o-mini` (10-20x cheaper)
- **Production:** Use `--llm_name gpt-5` or omit for default
- **Works with:** Both `bottomup.py` and `bottomup_parallel.py`
- **No code changes needed:** Just add command-line flag
