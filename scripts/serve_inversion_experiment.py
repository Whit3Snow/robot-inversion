import dataclasses
import logging
import socket
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.serving import websocket_policy_server
from openpi.models import model as _model


@dataclasses.dataclass
class Args:
    # Model to serve and checkpoint location
    config_name: str = "pi0_libero"  # e.g., pi0_base, pi0_libero
    checkpoint_dir: str | None = None  # if None, use defaults based on config_name

    method: str = "inversion"  # "inversion" or "editing"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001

    # Inversion settings
    num_steps: int = 10
    verbose: bool = False


DEFAULT_CHECKPOINT: Dict[str, str] = {
    "pi0_fast_droid": "s3://openpi-assets/checkpoints/pi0_fast_droid",
    "pi0_droid": "s3://openpi-assets/checkpoints/pi0_droid",
    "pi0_base": "s3://openpi-assets/checkpoints/pi0_base",
    "pi0_fast_base": "s3://openpi-assets/checkpoints/pi0_fast_base",
    "pi0_libero": "s3://openpi-assets/checkpoints/pi0_libero",
}


class InversionExperimentPolicy:
    """A lightweight policy that performs action sampling, inversion, and reconstruction.

    Expects Libero-style observation dict from the client:
      - "observation/state": np.ndarray [8]
      - "observation/image": np.ndarray [H, W, 3] (uint8 or float)
      - "observation/wrist_image": np.ndarray [H, W, 3]
      - "prompt": str
    """

    def __init__(self, config_name: str, checkpoint_dir: str, *, num_steps: int, verbose: bool = False) -> None:
        self._cfg = _config.get_config(config_name)
        self._policy = _policy_config.create_trained_policy(self._cfg, checkpoint_dir)
        self._model = self._policy._model
        self._num_steps = num_steps
        self._rng = jax.random.key(0)
        self._verbose = verbose

    def _modify_observation_prompt(self, observation: _model.Observation, new_prompt: str) -> _model.Observation:
        """Create a new observation with a different prompt, keeping all other data the same."""
        # Find the tokenizer in the transform chain
        tokenizer = None
        for transform in self._policy._input_transform.transforms:
            if hasattr(transform, 'tokenizer'):
                tokenizer = transform.tokenizer
                break
        
        if tokenizer is None:
            raise ValueError("Could not find tokenizer in input transform chain")
        
        # Tokenize the new prompt
        new_tokens, new_mask = tokenizer.tokenize(new_prompt)
        
        # Create new observation with updated prompt tokens
        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=jnp.array(new_tokens)[None, ...],
            tokenized_prompt_mask=jnp.array(new_mask)[None, ...],
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    def infer(self, obs: Dict) -> Dict:  # type: ignore[override]
        # Use the same input transform chain as the trained policy
        inputs = self._policy._input_transform(obs)
        inputs_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs_batched)

        if self._verbose:
            print(f"🔍 Input observation shapes after transform:")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape}")

        # Sample original actions using the model directly
        self._rng, sample_rng = jax.random.split(self._rng)
        raw_original_actions, _ = self._policy._sample_actions(
            sample_rng, observation, num_steps=self._num_steps
        )


        if self._verbose:
            print(f"🔍 Raw model output shape: {raw_original_actions.shape}")
            print(f"🔍 Model action_dim: {self._model.action_dim}, action_horizon: {self._model.action_horizon}")

        # Apply the policy's output transform chain to obtain Libero 7D, unnormalized actions
        out_dict = {"state": inputs_batched["state"], "actions": raw_original_actions}

        # Unbatch and convert to numpy (mimic Policy.infer)
        out_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), out_dict)
        out_np = self._policy._output_transform(out_np)


        original_actions_7d = out_np["actions"]  # shape [horizon, 7]


        if self._verbose:
            print(f"✅ Applied output transforms (Unnormalize + LiberoOutputs)")
            print(f"🔍 Final actions shape: {original_actions_7d.shape}")

        # For inversion experiment, we need to work with the raw model actions
        # Invert to noise and reconstruct using the RAW 32D actions
        inverted_noise, _ = self._policy._invert_actions(
            observation, raw_original_actions, num_steps=self._num_steps
        )
        
        # 🔥 Reconstruct from noise with potentially different prompt
        # Example: Use a different prompt for reconstruction
        # reconstruction_observation = self._modify_observation_prompt(observation, "put the object in the drawer")
        reconstruction_observation = observation  # Use original prompt by default
        
        raw_reconstructed_actions, _ = self._policy._reconstruct_from_noise(
            reconstruction_observation, inverted_noise, num_steps=self._num_steps
        )

        # Apply output transforms to reconstructed actions to get comparable 7D commands
        recon_out_dict = {"state": inputs_batched["state"], "actions": raw_reconstructed_actions}
        recon_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), recon_out_dict)
        recon_np = self._policy._output_transform(recon_np)
        reconstructed_actions_7d = recon_np["actions"]  # [horizon, 7]

        # Calculate MSE on both 32D (model space) and 7D (Libero action space)
        mse_32 = jnp.mean((raw_original_actions - raw_reconstructed_actions) ** 2)
        mse_7 = jnp.mean((original_actions_7d - reconstructed_actions_7d) ** 2)

        # Convert to numpy
        original_actions_7_np = np.asarray(original_actions_7d)
        reconstructed_actions_7_np = np.asarray(reconstructed_actions_7d)
        original_actions_32_np = np.asarray(raw_original_actions[0])
        reconstructed_actions_32_np = np.asarray(raw_reconstructed_actions[0])
        inverted_noise_np = np.asarray(inverted_noise[0])

        if self._verbose:
            print(f"🔍 Final action shapes - Original 7D: {original_actions_7_np.shape}, Reconstructed 7D: {reconstructed_actions_7_np.shape}")
            print(f"🔍 MSE 7D: {float(mse_7.item()):.6f}, MSE 32D: {float(mse_32.item()):.6f}")

        return {
            # 7D actions for Libero control
            "actions": original_actions_7_np,
            "original_actions": original_actions_7_np,
            "reconstructed_actions": reconstructed_actions_7_np,
            # Raw 32D actions and noise for analysis
            "original_actions_32": original_actions_32_np,
            "reconstructed_actions_32": reconstructed_actions_32_np,
            "inverted_noise": inverted_noise_np,
            # Metrics
            "reconstruction_loss": float(mse_7.item()),
            "reconstruction_loss_32d": float(mse_32.item()),
            # Shapes
            "action_horizon": int(original_actions_7_np.shape[0]),
            "action_dim": int(original_actions_7_np.shape[1]),
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "config_name": str(self._cfg),
            "action_dim": int(self._model.action_dim),
            "action_horizon": int(self._model.action_horizon),
            "num_steps": int(self._num_steps),
        }



class EditingExperimentPolicy:
    """A lightweight policy that performs action sampling, inversion, and reconstruction.

    Expects Libero-style observation dict from the client:
      - "observation/state": np.ndarray [8]
      - "observation/image": np.ndarray [H, W, 3] (uint8 or float)
      - "observation/wrist_image": np.ndarray [H, W, 3]
      - "prompt": str
    """

    def __init__(self, config_name: str, checkpoint_dir: str, *, num_steps: int, verbose: bool = False) -> None:
        self._cfg = _config.get_config(config_name)
        self._policy = _policy_config.create_trained_policy(self._cfg, checkpoint_dir)
        self._model = self._policy._model
        self._num_steps = num_steps
        self._rng = jax.random.key(0)
        self._verbose = verbose
        self.timestep = 0

    def _modify_observation_prompt(self, observation: _model.Observation, new_prompt: str) -> _model.Observation:
        """Create a new observation with a different prompt, keeping all other data the same."""
        # Find the tokenizer in the transform chain
        tokenizer = None


        for transform in self._policy._input_transform.transforms:
            if hasattr(transform, 'tokenizer'):
                tokenizer = transform.tokenizer
                break
        
        if tokenizer is None:
            raise ValueError("Could not find tokenizer in input transform chain")
        
        # Tokenize the new prompt
        new_tokens, new_mask = tokenizer.tokenize(new_prompt)
        
        # Create new observation with updated prompt tokens
        return _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=jnp.array(new_tokens)[None, ...],
            tokenized_prompt_mask=jnp.array(new_mask)[None, ...],
        )

    def infer(self, obs: Dict) -> Dict:  # type: ignore[override]
        # Use the same input transform chain as the trained policy

        inputs = self._policy._input_transform(obs)
        inputs_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs_batched)

        if self._verbose:
            print(f"🔍 Input observation shapes after transform:")
            for key, value in inputs.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape}")

        # Sample original actions using the model directly
        self._rng, sample_rng = jax.random.split(self._rng)
        raw_original_actions, _ = self._policy._sample_actions(
            sample_rng, observation, num_steps=self._num_steps
        )


        if self._verbose:
            print(f"🔍 Raw model output shape: {raw_original_actions.shape}")
            print(f"🔍 Model action_dim: {self._model.action_dim}, action_horizon: {self._model.action_horizon}")

        # Apply the policy's output transform chain to obtain Libero 7D, unnormalized actions
        out_dict = {"state": inputs_batched["state"], "actions": raw_original_actions}

        # Unbatch and convert to numpy (mimic Policy.infer)
        out_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), out_dict)
        out_np = self._policy._output_transform(out_np)


        original_actions_7d = out_np["actions"]  # shape [horizon, 7]


        if self._verbose:
            print(f"✅ Applied output transforms (Unnormalize + LiberoOutputs)")
            print(f"🔍 Final actions shape: {original_actions_7d.shape}")

        # For inversion experiment, we need to work with the raw model actions
        # Invert to noise and reconstruct using the RAW 32D actions
        inverted_noise, _ = self._policy._invert_actions(
            observation, raw_original_actions, num_steps=self._num_steps
        )

        # 🔥 Reconstruct from noise with potentially different prompt
        # Example: Use a different prompt for reconstruction
        # reconstruction_observation = self._modify_observation_prompt(observation, "put the object in the drawer")
        reconstruction_observation = observation  # Use original prompt by default
        
        #todo: editing from inverted noise with prompt
        if self.timestep < 60: #! Edit first replan_steps * 10 actions
            # new_prompt = "put the cream cheese on the rack"
            new_prompt = "put the cream cheese in the bowl"

            editing_observation = self._modify_observation_prompt(reconstruction_observation, new_prompt)

            '''
            editing_observation 계산하기 - reconstruction_observation 에서 prompt 부분만 변경
            '''
            #todo: recon code 검토
            '''
            1. inverted_noise를 random_noise로 변경
            2. modify 검토, breakpoint
            '''

            raw_reconstructed_actions, _ = self._policy._reconstruct_from_noise(
                editing_observation, inverted_noise, num_steps=self._num_steps
            )
        else:
            raw_reconstructed_actions, _ = self._policy._reconstruct_from_noise(
                reconstruction_observation, inverted_noise, num_steps=self._num_steps
            )

        # Apply output transforms to reconstructed actions to get comparable 7D commands
        recon_out_dict = {"state": inputs_batched["state"], "actions": raw_reconstructed_actions}
        recon_np = jax.tree.map(lambda x: np.asarray(x[0, ...]), recon_out_dict)
        recon_np = self._policy._output_transform(recon_np)
        reconstructed_actions_7d = recon_np["actions"]  # [horizon, 7]

        # Calculate MSE on both 32D (model space) and 7D (Libero action space)
        mse_32 = jnp.mean((raw_original_actions - raw_reconstructed_actions) ** 2)
        mse_7 = jnp.mean((original_actions_7d - reconstructed_actions_7d) ** 2)

        # Convert to numpy
        original_actions_7_np = np.asarray(original_actions_7d)
        reconstructed_actions_7_np = np.asarray(reconstructed_actions_7d)
        original_actions_32_np = np.asarray(raw_original_actions[0])
        reconstructed_actions_32_np = np.asarray(raw_reconstructed_actions[0])
        inverted_noise_np = np.asarray(inverted_noise[0])

        if self._verbose:
            print(f"🔍 Final action shapes - Original 7D: {original_actions_7_np.shape}, Reconstructed 7D: {reconstructed_actions_7_np.shape}")
            print(f"🔍 MSE 7D: {float(mse_7.item()):.6f}, MSE 32D: {float(mse_32.item()):.6f}")

        self.timestep += 1

        return {
            # 7D actions for Libero control
            "actions": original_actions_7_np,
            "original_actions": original_actions_7_np,
            "reconstructed_actions": reconstructed_actions_7_np,
            # Raw 32D actions and noise for analysis
            "original_actions_32": original_actions_32_np,
            "reconstructed_actions_32": reconstructed_actions_32_np,
            "inverted_noise": inverted_noise_np,
            # Metrics
            "reconstruction_loss": float(mse_7.item()),
            "reconstruction_loss_32d": float(mse_32.item()),
            # Shapes
            "action_horizon": int(original_actions_7_np.shape[0]),
            "action_dim": int(original_actions_7_np.shape[1]),
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "config_name": str(self._cfg),
            "action_dim": int(self._model.action_dim),
            "action_horizon": int(self._model.action_horizon),
            "num_steps": int(self._num_steps),
        }


def main(args: Args) -> None:
    ckpt = args.checkpoint_dir or DEFAULT_CHECKPOINT.get(args.config_name)
    if ckpt is None:
        raise ValueError(f"No checkpoint mapping for config_name={args.config_name} and none provided")

    if args.method == "inversion":
        policy = InversionExperimentPolicy(args.config_name, ckpt, num_steps=args.num_steps, verbose=args.verbose)
    elif args.method == "editing":
        policy = EditingExperimentPolicy(args.config_name, ckpt, num_steps=args.num_steps, verbose=args.verbose)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating inversion server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
