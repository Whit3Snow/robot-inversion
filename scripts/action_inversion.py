#!/usr/bin/env python3
"""
Action inversion and reconstruction experiment for Pi0 model.

This script:
1. Generates actions using Pi0 model for a given prompt in Libero environment
2. Inverts the actions to recover initial noise using rectified flow inversion
3. Reconstructs actions from the recovered noise
4. Computes reconstruction loss
"""

import jax
import jax.numpy as jnp
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import Dict, Any

from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.models import model as _model
from openpi import transforms as _transforms

import math


def compute_reconstruction_loss(original_actions: jax.Array, reconstructed_actions: jax.Array) -> float:
    """Compute MSE loss between original and reconstructed actions."""
    mse_loss = jnp.mean((original_actions - reconstructed_actions) ** 2)
    return float(mse_loss)


def run_libero_prompt_experiment(
    config_name: str = "pi0_libero",
    prompt: str = "pick up the red block",
    task_suite_name: str = "libero_goal",
    task_id: int = 0,
    num_steps: int = 10,
    save_results: bool = True,
    output_dir: str = "action_inversion_results"
) -> Dict[str, Any]:
    """
    Run the full inversion experiment pipeline with real Libero observation.
    
    Args:
        config_name: model config name
        prompt: text prompt for action generation (if None, uses task description)
        task_suite_name: Libero task suite name (e.g., "libero_goal", "libero_spatial", "libero_object")
        task_id: task ID within the suite
        num_steps: rectified flow integration steps
        save_results: whether to save results to disk
        output_dir: directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    print(f"Loading model config: {config_name}")
    cfg = config.get_config(config_name)
    
    # Download checkpoint
    checkpoint_path_map = {
        "pi0_fast_droid": "s3://openpi-assets/checkpoints/pi0_fast_droid",
        "pi0_droid": "s3://openpi-assets/checkpoints/pi0_droid", 
        "pi0_base": "s3://openpi-assets/checkpoints/pi0_base",
        "pi0_fast_base": "s3://openpi-assets/checkpoints/pi0_fast_base",
        "pi0_libero": "s3://openpi-assets/checkpoints/pi0_libero",
    }
    
    if config_name not in checkpoint_path_map:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(checkpoint_path_map.keys())}")
    
    checkpoint_dir = download.maybe_download(checkpoint_path_map[config_name])
    print(f"Checkpoint downloaded to: {checkpoint_dir}")
    
    # Create policy
    policy = policy_config.create_trained_policy(cfg, checkpoint_dir)
    print(f"Policy created successfully")
    
    # Create real Libero observation
    print(f"Creating Libero observation from task suite: {task_suite_name}, task_id: {task_id}")
    observation = create_libero_observation(
        task_suite_name=task_suite_name,
        task_id=task_id,
        prompt=prompt,
        resize_size=224,
        seed=42,
        state_pad_dim=policy._model.action_dim,
    )
    
    print(f"Generating actions for prompt: '{prompt}'")
    print(f"Observation state shape: {observation.state.shape}")
    print(f"Observation images shape: {observation.images['base_0_rgb'].shape}")
    
    # Step 1: Generate original actions
    rng = jax.random.PRNGKey(42)
    rng_sample, _ = jax.random.split(rng)

    original_actions, original_layer_output = policy._sample_actions(
        rng_sample, observation, num_steps=num_steps
    )
    print(f"Original actions shape: {original_actions.shape}")
    
    # Step 2: Invert actions to noise
    print("Inverting actions to noise...")
    inverted_noise, invert_layer_output = policy._invert_actions(
        observation, original_actions, num_steps=num_steps
    )
    print(f"Inverted noise shape: {inverted_noise.shape}")
    
    # Step 3: Reconstruct actions from inverted noise
    print("Reconstructing actions from inverted noise...")
    # We need to manually set the initial noise in sample_actions
    # Since sample_actions generates its own noise, we'll implement custom reconstruction
    reconstructed_actions, reconstruct_layer_output = policy._reconstruct_from_noise(
        observation, inverted_noise, num_steps=num_steps
    )
    print(f"Reconstructed actions shape: {reconstructed_actions.shape}")
    
    # Step 4: Compute reconstruction loss
    reconstruction_loss = compute_reconstruction_loss(original_actions, reconstructed_actions)
    print(f"Reconstruction MSE loss: {reconstruction_loss:.6f}")
    
    # Compile results
    results = {
        "config_name": config_name,
        "prompt": prompt,
        "task_suite_name": task_suite_name,
        "task_id": task_id,
        "num_steps": num_steps,
        "original_actions": np.array(original_actions),
        "inverted_noise": np.array(inverted_noise),
        "reconstructed_actions": np.array(reconstructed_actions),
        "reconstruction_loss": reconstruction_loss,
        "action_shape": original_actions.shape,
        "action_horizon": policy._model.action_horizon,
        "action_dim": policy._model.action_dim,
        "observation_state": np.array(observation.state),
    }
    
    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_file = output_path / f"inversion_experiment_{config_name}_{task_suite_name}_task{task_id}_{num_steps}steps.pkl"
        with open(results_file, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to: {results_file}")
    
    return results


def create_libero_observation(
    task_suite_name: str = "libero_goal",
    task_id: int = 0,
    prompt: str | None = None,
    resize_size: int = 224,
    seed: int = 42,
    state_pad_dim: int = 32,
) -> _model.Observation:
    """Create a real observation from Libero environment."""
    # Lazy import Libero and client image tools to avoid requiring them on the server.
    import sys as _sys
    import os as _os
    from openpi_client import image_tools  # lightweight, available in both envs
    _sys.path.append(_os.path.join(_os.path.dirname(__file__), "..", "third_party", "modified_libero"))
    from libero.libero import benchmark  # type: ignore
    from libero.libero import get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv  # type: ignore

    # Initialize Libero task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    # Get specific task
    task = task_suite.get_task(task_id)
    task_description = task.language
    
    if prompt is None:
        prompt = task_description
    
    print(f"Creating Libero observation for task: {task_description}")
    
    # Initialize environment
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": 256, 
        "camera_widths": 256
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    
    # Reset environment to get initial observation
    obs = env.reset()
    
    # Wait a few steps for objects to stabilize (like in main.py)
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    
    # Process images (rotate 180 degrees like in main.py)
    original_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    original_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    # Resize and convert to proper format
    img_uint8 = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(original_img, resize_size, resize_size)
    )
    wrist_uint8 = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(original_wrist_img, resize_size, resize_size)
    )
    # Convert to [-1, 1] float32 expected by the model
    img = (img_uint8.astype(jnp.float32) / 255.0) * 2.0 - 1.0
    wrist_img = (wrist_uint8.astype(jnp.float32) / 255.0) * 2.0 - 1.0
    
    # Create state vector (position + orientation + gripper)
    state = jnp.concatenate([
        obs["robot0_eef_pos"],
        _quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(jnp.float32)
    # Pad state to model action_dim (default 32 for Pi0)
    state = jnp.asarray(_transforms.pad_to_dim(np.asarray(state), state_pad_dim))
    
    # Create batch dimension
    batch_size = 1
    dummy_mask = jnp.ones((batch_size,), dtype=jnp.bool_)
    
    # Create observation
    observation = _model.Observation(
        images={
            "base_0_rgb": img[None, ...],  # Add batch dimension
            "left_wrist_0_rgb": wrist_img[None, ...],
            "right_wrist_0_rgb": wrist_img[None, ...],  # Use same wrist image for both
        },
        image_masks={
            "base_0_rgb": dummy_mask,
            "left_wrist_0_rgb": dummy_mask,
            "right_wrist_0_rgb": dummy_mask,
        },
        state=state[None, ...],  # Add batch dimension
        prompt=[prompt],
    )
    
    # Clean up
    env.close()
    
    return observation


def _quat2axisangle(quat):
    """
    Convert quaternion to axis-angle representation.
    Copied from robosuite.
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pi0 action inversion and reconstruction experiment (Libero)")
    parser.add_argument("--config_name", type=str, default="pi0_libero",
                        help="Model config name (e.g., pi0_base, pi0_libero)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Optional prompt to override the Libero task description")
    parser.add_argument("--task_suite_name", type=str, default="libero_goal",
                        help="Libero task suite name (e.g., libero_goal, libero_spatial, libero_object)")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID within the suite")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of rectified flow steps")
    parser.add_argument("--no_save", action="store_true", help="Do not save results to disk")
    parser.add_argument("--output_dir", type=str, default="action_inversion_results", help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prompt = args.prompt
    if prompt is None:
        print("No prompt provided; will use Libero task description.")
    results = run_libero_prompt_experiment(
        config_name=args.config_name,
        prompt=prompt if prompt is not None else None,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        num_steps=args.num_steps,
        save_results=(not args.no_save),
        output_dir=args.output_dir,
    )
    print(f"Done. Reconstruction MSE: {results['reconstruction_loss']:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Action inversion and reconstruction experiment")
    parser.add_argument("--config", default="pi0_libero", help="Model config name")
    parser.add_argument("--prompt", default=None, help="Text prompt (if None, uses task description)")
    parser.add_argument("--task-suite", default="libero_goal", help="Libero task suite name")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID within the suite")
    parser.add_argument("--steps", type=int, default=10, help="Number of rectified flow steps")
    parser.add_argument("--output-dir", default="action_inversion_results", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    results = run_libero_prompt_experiment(
        config_name=args.config,
        prompt=args.prompt,
        task_suite_name=args.task_suite,
        task_id=args.task_id,
        num_steps=args.steps,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Config: {results['config_name']}")
    print(f"Task Suite: {results['task_suite_name']}")
    print(f"Task ID: {results['task_id']}")
    print(f"Prompt: '{results['prompt']}'")
    print(f"Steps: {results['num_steps']}")
    print(f"Action shape: {results['action_shape']}")
    print(f"Reconstruction MSE loss: {results['reconstruction_loss']:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
