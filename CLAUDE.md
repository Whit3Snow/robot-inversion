# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the paper "Task Reconstruction and Extrapolation for π₀ using Text Latent". It's built on top of the [openpi](https://github.com/Physical-Intelligence/openpi) framework and focuses on robotic policy learning with text latent representations for task reconstruction and extrapolation.

## Development Environment Setup

This project requires **two separate conda environments**:

### Primary Environment (text-latent)
```bash
conda create -n text-latent python=3.11 -y
conda activate text-latent
pip install uv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

### LIBERO Environment (for benchmarking)
```bash
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/modified_libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/modified_libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
```

## Common Commands

### Code Quality
- **Lint**: `uv run ruff check --fix` (includes auto-fix)
- **Format**: `uv run ruff format`
- **Type Check**: No explicit type checker configured; ruff handles basic type checks
- **Tests**: `uv run pytest` (runs tests in `src/`, `scripts/`, and `packages/`)
- **Pre-commit**: `pre-commit run --all-files`

### Running Experiments

**Basic two-terminal setup for experiments:**

Terminal 1 (text-latent environment):
```bash
python scripts/serve_policy.py
```

Terminal 2 (LIBERO environment):
```bash
python examples/libero/main.py
```

### Key Scripts
- `scripts/serve_policy.py` - Policy server for model serving
- `scripts/text_latent.py` - Text latent identification (1 hour on RTX4090)
- `scripts/action_inversion.py` - Action inversion functionality
- `scripts/serve_inversion_experiment.py` - Inversion experiment server
- `scripts/run_inversion_client.py` - Inversion client with visualization
- `examples/libero/main.py` - LIBERO benchmark execution

## Architecture Overview

### Core Components

- **`src/openpi/models/pi0.py`** - Main Pi0 model implementation with attention mechanisms and text latent integration
- **`src/openpi/models/gemma.py`** - Gemma language model integration 
- **`src/openpi/policies/`** - Policy implementations for different environments
- **`src/openpi/serving/`** - WebSocket-based policy serving infrastructure

### Key Model Classes
- `Pi0` - Main policy model with text latent capabilities
- `Gemma`/`GemmaFast` - Language model backends
- Policy servers support ALOHA, DROID, and LIBERO environments

### Data Flow
1. **Policy Server** (`serve_policy.py`) loads trained checkpoints and serves models via WebSocket
2. **Client Scripts** connect to policy server for inference
3. **LIBERO Integration** provides standardized robotic task evaluation
4. **Text Latent Pipeline** processes demonstrations to extract task representations

## Environment-Specific Notes

### LIBERO Modifications
- Contact threshold relaxed from 0.03 to 0.1 for better task completion detection
- Stove tasks allow placement anywhere on stove (not just cooking region)
- BDDL files located in `third_party/libero/libero/libero/bddl_files/`

### Data Organization
- Experiment data expected in `exp_data/pi0/*.pkl`
- Videos saved to `pi0-text-latent/data/libero/`
- Pre-computed text latents available from HuggingFace: `Shady0057/pi0-text-latent`

## Inversion & Reconstruction Experiments

### Server-Client Architecture
The repository supports advanced inversion experiments through a server-client setup:

**Server Setup** (text-latent environment):
```bash
CUDA_VISIBLE_DEVICES=2 uv run python scripts/serve_inversion_experiment.py \
--config_name pi0_libero --port 8000 --verbose --method editing
```

**Client Setup** (LIBERO environment):
```bash
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/modified_libero
CUDA_VISIBLE_DEVICES=2 python scripts/run_inversion_client.py \
--port 8000 --task_suite_name libero_goal --task_id 9 \
--prompt "put the wine bottle on the rack" --closed_loop --replan_steps 5 --draw_traj
```

### Experiment Methods

#### Inversion Method (`--method inversion`)
- **Action Generation**: Sample actions using Pi0 model for given observation/prompt
- **Action Inversion**: Convert generated actions back to noise using rectified flow inversion
- **Reconstruction**: Regenerate actions from recovered noise to test fidelity
- **Metrics**: Computes MSE between original and reconstructed actions in both 32D (model space) and 7D (Libero action space)

#### Editing Method (`--method editing`)
- **Experimental feature** that attempts prompt-based action editing during reconstruction
- Modifies prompts for first 60 timesteps (e.g., "put the cream cheese in the bowl")
- Currently unstable and doesn't work reliably due to fundamental limitations described below

### Key Components

**`InversionExperimentPolicy` class**:
- Handles action sampling, inversion, and reconstruction pipeline
- Works with 32D model actions internally, converts to 7D for Libero control
- Supports prompt modification for editing experiments

**`run_inversion_client.py`**:
- Closed-loop execution with periodic replanning every `replan_steps`
- Trajectory visualization with `--draw_traj` flag
- Video recording capabilities for experiment analysis
- Support for various LIBERO task suites (libero_goal, libero_spatial, etc.)

### Workflow
1. **Observation Processing**: Client creates Libero environment observation (state + RGB images)
2. **Action Planning**: Server generates action sequence using Pi0 model
3. **Inversion**: Server inverts actions to recover noise representation
4. **Reconstruction**: Server reconstructs actions from noise (with optional prompt editing)
5. **Execution**: Client executes actions in environment with replanning
6. **Analysis**: MSE metrics computed between original and reconstructed action plans

## Known Issues and Limitations

### Out-of-Distribution (OOD) Prompt Problem

The Pi0 model exhibits significant performance degradation when handling prompts not seen during training, even when all individual components are familiar:

**Example Problem Case:**
- Training prompt: `"put the orange juice on the plate"` ✓ (works well)
- OOD prompt: `"put the orange juice on the stove"` ✗ (fails despite all words being seen)

**Root Cause Analysis:**
- All key terms (`put`, `on`, `orange juice`, `stove`) exist individually in training data
- Model appears to overfit to specific prompt-object-destination combinations rather than learning compositional understanding
- VLM (Gemma) spatial attention analysis reveals:
  - **Good attention**: Robot body, destination locations (plate, stove)
  - **Poor attention**: Target objects (orange juice, cream cheese, etc.)

### Inversion Editing Limitations

**Original Approach (pi0-text-latent):** Latent interpolation for OOD generalization
**Attempted Solution:** Inversion editing with prompt modification

**Why Inversion Editing Failed:**
1. **Low Prompt Influence**: Prompts have limited impact on action generation compared to visual features
2. **Text-Latent Dominance**: Hidden layer text features (text-latent) have stronger influence than surface prompt modifications
3. **Spatial Attention Issues**: Fundamental VLM attention problems aren't solved by prompt engineering
4. **Overfitting to Training Prompts**: Model relies too heavily on memorized prompt-action mappings rather than compositional understanding

**Technical Details:**
- Inverting original actions to noise and reconstructing with modified OOD prompts shows minimal guidance effect
- Text-latent modifications at hidden layer level are more impactful than surface prompt changes
- Need to preserve original action characteristics while providing editing guidance creates conflicting objectives

### Recommended Research Directions

1. **Improve Spatial Attention**: Focus on VLM attention mechanisms to better attend to target objects
2. **Compositional Learning**: Develop training strategies that encourage compositional understanding over prompt memorization
3. **Text-Latent Interpolation**: Continue developing the original latent interpolation approach for OOD generalization
4. **Data Augmentation**: Increase diversity of prompt-object-destination combinations in training data

## Development Tools

- **UV Package Manager**: Used for dependency management and Python execution
- **Ruff**: Linting and formatting (configured in `pyproject.toml`)
- **JAX/Flax**: Primary ML framework with NNX for neural networks
- **Pre-commit**: Automated code quality checks on commit