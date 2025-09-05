import copy
import logging
import os.path
import pathlib
import pickle
from collections.abc import Sequence
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy


def load_hidden_states(file_path, num_valid_token, offset):
    print("Using: {}".format(os.path.basename(file_path)))
    with open(file_path, "rb") as f:
        _hidden_states_2 = pickle.load(f)["hidden_states_avg"]
    _hidden_states_2 = _hidden_states_2[:, offset: offset + num_valid_token]
    _hidden_states_2 = _hidden_states_2.astype(jnp.bfloat16)
    return _hidden_states_2


def write_async(executor, dset, data):
    executor.submit(dset.append, data[np.newaxis, ...])


BasePolicy: TypeAlias = _base_policy.BasePolicy


def orthogonalize(v, u):
    """
    v - u for orthogonalization
    """
    proj = (jnp.sum(u * v, axis=-1, keepdims=True) /
            jnp.sum(u * u, axis=-1, keepdims=True)) * u
    return v - proj


def cosine_interp(a, b, alpha):
    alpha2 = (1 - jnp.cos(alpha * jnp.pi)) / 2
    return (1 - alpha2) * a + alpha2 * b


def slerp(a, b, alpha, eps=1e-8):
    a_norm = a / (jnp.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b_norm = b / (jnp.linalg.norm(b, axis=-1, keepdims=True) + eps)
    dot = jnp.sum(a_norm * b_norm, axis=-1, keepdims=True)
    dot = jnp.clip(dot, -1.0, 1.0)

    omega = jnp.arccos(dot)
    sin_omega = jnp.sin(omega)

    factor_a = jnp.sin((1 - alpha) * omega) / (sin_omega + eps)
    factor_b = jnp.sin(alpha * omega) / (sin_omega + eps)

    return factor_a * a + factor_b * b


def switch_interp(a, b, alpha):
    if alpha < 0.5:
        return a
    else:
        return b


def linear_interp(a, b, alpha):
    return (1 - alpha) * a + alpha * b


class Policy(BasePolicy):
    def __init__(
            self,
            model: _model.BaseModel,
            *,
            rng: at.KeyArrayLike | None = None,
            transforms: Sequence[_transforms.DataTransformFn] = (),
            output_transforms: Sequence[_transforms.DataTransformFn] = (),
            sample_kwargs: dict[str, Any] | None = None,
            metadata: dict[str, Any] | None = None,
    ):
        # Expose model for advanced use (e.g., inversion experiments)
        self._model = model
        # self._sample_actions = model.sample_actions # stop jit for debugging
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.activation_dataset = None
        self.mask_dataset = None
        self.step = 0

        self._num_valid_token = 9
        self._prefix_length = 816
        self._model_name = "pi0"
        self.offset = 256 * 3
        # auto-filled parameters
        self._alpha = None
        self._this_episode_hidden_states_to_use = None
        self._hidden_states_1 = None
        self._hidden_states_2 = None

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        prompt = copy.deepcopy(obs["prompt"])
        if obs["done"]:
            self.step = 0

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        # mask prompt exp
        if obs["mask_prompt_method"] is not None:
            if obs["mask_prompt_method"] in ["blank", "blank_with_text_latent"]:
                inputs["tokenized_prompt"][:inputs["tokenized_prompt_mask"].sum() - 1] = 2  # using BOS
            elif obs["mask_prompt_method"] == "mask":
                inputs["tokenized_prompt_mask"][...] = 0.
            else:
                raise ValueError("Unknown mask_prompt_method: {}".format(obs["mask_prompt_method"]))
            if obs["done"]:
                # load the stat only when reset
                if obs["mask_prompt_method"] == "blank_with_text_latent":
                    to_use = obs["task_hidden_states_mapping"][prompt.replace(" ", "_")]
                    hidden_states_mapping_file = obs["hidden_states_mapping_file"]
                    _hidden_states_1 = load_hidden_states(
                        hidden_states_mapping_file[to_use[0]].format(self._model_name),
                        self._num_valid_token, self.offset)

                    # Interpolating both text embedding and text latent without removing another task information
                    self._hidden_states_1 = jnp.zeros((18, self._prefix_length, 2048), dtype=jnp.bfloat16)
                    self._hidden_states_1 = self._hidden_states_1.at[:,self.offset: self.offset + self._num_valid_token].add(_hidden_states_1)
                    self._hidden_states_1 = self._hidden_states_1.at[0].set(0.)  # remove text embedding avoid prompt leaking!
                print("Mask Prompt Method: {}".format(obs["mask_prompt_method"]))
                print("Tokenized Prompt 0-15: {}".format(inputs["tokenized_prompt"][:15]))
                print("Prompt Mask 0-15: {}".format(inputs["tokenized_prompt_mask"][:15]))

            # set sample kwargs every step
            if obs["mask_prompt_method"] == "blank_with_text_latent":
                self._sample_kwargs["hidden_states_to_add"] = self._hidden_states_1

        # logitlens exp
        if obs.get("prompt_to_use", False):
            assert obs["mask_prompt_method"] is None
            if obs["done"]:
                print("Using prompt: {}".format(obs["prompt_to_use"]))
                print("decoding ret: {}".format(obs["prompt_to_use_str"]))
            inputs["tokenized_prompt"][:len(obs["prompt_to_use"])] = obs["prompt_to_use"]

        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        # TEI and TLI exp
        if obs["layer_to_intervene"] is not None:
            assert obs["layer_to_intervene"] in ["all"] + [i for i in range(18)]
            if obs["done"]:
                print("Switching hidden states from {} to {}".format(self._this_episode_hidden_states_to_use, prompt))
                print("Tokens: {}".format(inputs["tokenized_prompt"][0][:15]))
                print("layers to intervene: {}".format(obs["layer_to_intervene"]))
                to_use = obs["task_hidden_states_mapping"][prompt.replace(" ", "_")]
                hidden_states_mapping_file = obs["hidden_states_mapping_file"]

                # load target 1
                _hidden_states_1 = load_hidden_states(hidden_states_mapping_file[to_use[0]].format(self._model_name),
                                                      self._num_valid_token, self.offset)

                # load target 2
                _hidden_states_2 = load_hidden_states(hidden_states_mapping_file[to_use[1]].format(self._model_name),
                                                      self._num_valid_token, self.offset)

                # Interpolating both text embedding and text latent without removing another task information
                self._hidden_states_1 = jnp.zeros((18, self._prefix_length, 2048), dtype=jnp.bfloat16)
                self._hidden_states_2 = jnp.zeros((18, self._prefix_length, 2048), dtype=jnp.bfloat16)

                self._hidden_states_1 = self._hidden_states_1.at[:,
                                        self.offset: self.offset + self._num_valid_token].add(_hidden_states_1)
                self._hidden_states_2 = self._hidden_states_2.at[:,
                                        self.offset: self.offset + self._num_valid_token].add(_hidden_states_2)

                # Remove another task information for only text latent!
                self._hidden_states_1 = self._hidden_states_1.at[1:,
                                        self.offset: self.offset + self._num_valid_token].add(-_hidden_states_2[1:])
                self._hidden_states_2 = self._hidden_states_2.at[1:,
                                        self.offset: self.offset + self._num_valid_token].add(-_hidden_states_1[1:])
                self._alpha = [alpha for alpha in jnp.linspace(0, 1, to_use[2] + 2, dtype=jnp.bfloat16)]
                self._this_episode_hidden_states_to_use = prompt
            assert self._this_episode_hidden_states_to_use == prompt

            alpha = self._alpha[self.step] if self.step < len(self._alpha) else self._alpha[-1]
            interpolated_rep = linear_interp(self._hidden_states_1, self._hidden_states_2, alpha)

            if obs["layer_to_intervene"] == "all":
                hidden_states_to_add = interpolated_rep
                if not obs["use_TEI_and_TLI"]:
                    # set first layer to 0, line 533 in gemma.py knows that the text embedding should not be overwritten
                    hidden_states_to_add = hidden_states_to_add.at[0].set(0.)
            else:
                hidden_states_to_add = jnp.zeros_like(interpolated_rep, dtype=jnp.bfloat16)
                hidden_states_to_add = hidden_states_to_add.at[obs["layer_to_intervene"]].set(interpolated_rep[obs["layer_to_intervene"]])
            self._sample_kwargs["hidden_states_to_add"] = hidden_states_to_add

        self._rng, sample_rng = jax.random.split(self._rng)
        action, layer_output = self._sample_actions(sample_rng,
                                                    _model.Observation.from_dict(inputs),
                                                    **self._sample_kwargs)
        self._sample_kwargs = {}  # clean every step
        outputs = {
            "state": inputs["state"],
            "actions": action}

        self.step += 1

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
