# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""
import dataclasses
from collections.abc import Sequence
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "dummy_300m_1_layer":
        return Config(
            width=1024,
            depth=1,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256)
    if variant == "dummy_2b_1_layer":
        return Config(
            width=2048,
            depth=1,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "dummy_300m_5_layer":
        return Config(
            width=1024,
            depth=5,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256)
    if variant == "dummy_2b_5_layer":
        return Config(
            width=2048,
            depth=5,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, norm_scale=None):
        dtype = x.dtype  # original dtype, could be half-precision
        if norm_scale is None:
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            scale = (1 + scale) * jnp.reciprocal(
                jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-06))
        else:
            scale = norm_scale
        normed_inputs = x * scale
        return normed_inputs.astype(dtype), scale  # NOTE: return in original dtype, precision is important!


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, attention=None, attn_out_to_add=None, head_to_ablate=None):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype, dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        if attention is None:
            q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
            logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

            if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
                raise ValueError(
                    f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
                )

            # big_neg = jnp.finfo(logits.dtype).min
            big_neg = -2.3819763e38  # See gemma/modules.py
            masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

            probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)
        else:
            # print("Using attention")
            probs = attention

        head_output_encoder = None
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # original implementation
        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                after_wo = out_einsum("BTNH,NHD->BTND", encoded[:, start:end])

                if head_to_ablate is not None:
                    assert i == 0
                    after_wo = after_wo.at[..., -48:, head_to_ablate[0], :].set(head_to_ablate[1])
                if i == 0:
                    head_output_encoder = after_wo[:, -48:]

                out.append(after_wo.sum(-2))  # sum over all heads
                start = end
                if i == 0 and attn_out_to_add is not None:
                    out[0] += attn_out_to_add
            else:
                out.append(None)

        return out, (k, v), (probs, head_output_encoder)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: Sequence[Config]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, decode, deterministic=True, attention=None,
                 pre_attn_norm_scale=None, pre_mlp_norm_scale=None, apply_mlp=True,
                 mlp_activation_overwrite=None, attn_out_to_add=None, head_to_ablate=None):  # noqa: FBT002
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        output_pre_attn_norm_scale = pre_attn_norm_scale
        output_pre_mlp_norm_scale = pre_mlp_norm_scale
        for i, x in enumerate(xs):
            if x is not None:
                x, output_pre_attn_norm_scale = RMSNorm(name=_name("pre_attention_norm", i))(x,
                                                                                             pre_attn_norm_scale)  # noqa: PLW2901
            pre_attn.append(x)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache, attention_output = attn(pre_attn, positions, attn_mask, kv_cache, attention=attention,
                                                     attn_out_to_add=attn_out_to_add, head_to_ablate=head_to_ablate)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)
        post_attn_embedding = jax.tree.map(lambda x: x.copy() if x is not None else None, xs)

        out = []
        mlp_activations = None
        if apply_mlp:
            for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
                if x is not None:
                    x, output_pre_mlp_norm_scale = RMSNorm(name=_name("pre_ffw_norm", i))(x,
                                                                                          pre_mlp_norm_scale)  # noqa: PLW2901
                    x, mlp_activations = lora.FeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=_name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x, mlp_activation_overwrite)
                out.append(x)

            out = sharding.activation_sharding_constraint(out)

            out = jax.tree.map(lambda x: drop(x, deterministic), out)

            xs = jax.tree.map(lambda x, y: x + y, xs, out)
            xs = sharding.activation_sharding_constraint(xs)

        return xs, post_attn, kv_cache, mlp_activations, attention_output, output_pre_attn_norm_scale, output_pre_mlp_norm_scale, post_attn_embedding


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        self.layers = [
            Block(
                configs=self.configs,
                dropout=self.dropout,
                dropout_bdims=self.dropout_bdims,
                name=f"layers_{idx}"
            )
            for idx in range(self.configs[0].depth)
        ]
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def unembed(self, x: at.Float[at.Array, "b t d"]) -> at.Float[at.Array, "b t v"]:
        return self.embedder.decode(x).astype(self.embed_dtype)

    def get_layer_output(self, layer_index,
                         embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
                         positions: at.Int[at.Array, "b t"],
                         mask: at.Bool[at.Array, "b t s"],
                         *,
                         kv_cache: KVCache | None = None,
                         attention=None,
                         pre_attn_norm_scales=None,
                         pre_mlp_norm_scales=None,
                         deterministic: bool = True,
                         apply_mlp: bool = True,
                         mlp_activation: dict | None = None,
                         attn_out_to_add=None,
                         head_to_ablate=None):

        this_layer_cache = [c[layer_index] for c in kv_cache] if kv_cache is not None else None
        this_layer_attention = attention[layer_index] if attention is not None else None
        this_layer_pre_attn_scale = pre_attn_norm_scales[layer_index] if pre_attn_norm_scales is not None else None
        this_layer_pre_mlp_scale = pre_mlp_norm_scales[layer_index] if pre_mlp_norm_scales is not None else None
        mlp_activation = mlp_activation.get(layer_index, None) if mlp_activation is not None else None

        ret = self.layers[layer_index](embedded,
                                       this_layer_cache,
                                       positions,
                                       mask,
                                       deterministic,
                                       attention=this_layer_attention,
                                       pre_attn_norm_scale=this_layer_pre_attn_scale,
                                       pre_mlp_norm_scale=this_layer_pre_mlp_scale,
                                       apply_mlp=apply_mlp,
                                       mlp_activation_overwrite=mlp_activation,
                                       attn_out_to_add=attn_out_to_add,
                                       head_to_ablate=head_to_ablate)
        return ret

    @at.typecheck
    def __call__(
            self,
            # list of token arrays, one for each expert, or None if that expert should not be run
            embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
            positions: at.Int[at.Array, "b t"],
            mask: at.Bool[at.Array, "b t s"],
            *,
            kv_cache: KVCache | None = None,
            attention=None,
            pre_attn_norm_scales=None,
            pre_mlp_norm_scales=None,
            final_norm_scales=None,
            deterministic: bool = True,
            apply_mlp: bool = True,
            mlp_activation: dict | None = None,
            layer_indices=None,
            hidden_states_to_add=None,
            attn_out_to_add=None,
            layer_head_to_ablate=None,
    ) -> tuple[
        Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache, tuple[at.Float[at.Array, "l b _t _d"] | None, at.Array,
        at.Array, at.Array | None, list[at.Array | None], tuple[
            at.Array | None, at.Array | None], at.Array, at.Array, at.Array | None]]:

        # check
        assert attention is None or attention.shape[0] == len(self.layers)
        assert pre_attn_norm_scales is None or len(pre_attn_norm_scales) == len(self.layers)
        assert pre_mlp_norm_scales is None or len(pre_mlp_norm_scales) == len(self.layers)
        final_norm_scales = final_norm_scales or [None] * len(self.final_norms)

        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]

        if layer_indices is None:
            apply_final_layer_norm = True
            layer_indices = range(len(self.layers))
        else:
            apply_final_layer_norm = False

        # to collect
        kv_cache_0 = []
        kv_cache_1 = []
        all_activations = []
        all_attentions = []
        output_pre_attn_scales = []
        output_pre_mlp_scales = []
        all_post_attn_1 = []
        all_post_attn_2 = []
        layer_hidden_states = []
        post_attn_embeddings = []
        text_representation = []

        def true_branch(operands):
            hidden_states_to_add, embedded = operands  # Unpack operands
            return embedded[0].at[:, 256 *3:].set(hidden_states_to_add[:, 256*3:])

        # Define the function for the False case
        def false_branch(operands):
            hidden_states_to_add, embedded = operands  # Unpack operands
            return embedded[0].at[...].add(hidden_states_to_add)

        for index in layer_indices:
            if layer_head_to_ablate is not None:
                head_to_ablate = layer_head_to_ablate[index]
                if head_to_ablate[0] is None:
                    head_to_ablate = None
            else:
                head_to_ablate = None
            this_layer_attn_to_add = attn_out_to_add[index:index + 1] if attn_out_to_add is not None else None

            if hidden_states_to_add is not None:
                embedded[0] = jax.lax.cond(
                    index == 0 and jnp.any(jnp.abs(hidden_states_to_add[0]) > 1e-3),
                    true_branch,
                    false_branch,
                    (hidden_states_to_add[index:index + 1], embedded)  # Operands packed in a tuple
                )
            layer_hidden_states.append(embedded[1] if embedded[0] is None else embedded[0])

            (embedded, post_attn, inter_kv_cache, activations, inter_attention,
             pre_attn_scale, pre_mlp_scale, post_attn_embed) = self.get_layer_output(index,
                                                                                     embedded,
                                                                                     positions,
                                                                                     mask,
                                                                                     kv_cache=kv_cache,
                                                                                     deterministic=deterministic,
                                                                                     attention=attention,
                                                                                     pre_attn_norm_scales=pre_attn_norm_scales,
                                                                                     pre_mlp_norm_scales=pre_mlp_norm_scales,
                                                                                     apply_mlp=apply_mlp,
                                                                                     mlp_activation=mlp_activation,
                                                                                     attn_out_to_add=this_layer_attn_to_add,
                                                                                     head_to_ablate=head_to_ablate)

            all_post_attn_1.append(post_attn[0])
            all_post_attn_2.append(post_attn[1])
            kv_cache_0.append(inter_kv_cache[0])
            kv_cache_1.append(inter_kv_cache[1])
            all_activations.append(activations)
            all_attentions.append(inter_attention[0])
            text_representation.append(inter_attention[1])
            output_pre_attn_scales.append(pre_attn_scale)
            output_pre_mlp_scales.append(pre_mlp_scale)

            post_attn_embeddings.append(post_attn_embed[1] if post_attn_embed[0] is None else post_attn_embed[0])

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        kv_cache_0 = jnp.stack(kv_cache_0, axis=0)
        kv_cache_1 = jnp.stack(kv_cache_1, axis=0)
        if None not in all_activations:
            mlp_activations = jnp.stack(all_activations, axis=0)
            output_pre_mlp_scales = jnp.stack(output_pre_mlp_scales, axis=0)
        else:
            mlp_activations = None
            output_pre_mlp_scales = None
        attention = jnp.stack(all_attentions, axis=0)
        if text_representation[0] is not None:
            text_representation = jnp.stack(text_representation, axis=0)
        else:
            text_representation = None
        output_pre_attn_scales = jnp.stack(output_pre_attn_scales, axis=0)
        if None not in all_post_attn_1:
            all_post_attn_1 = jnp.stack(all_post_attn_1, axis=0)
        else:
            all_post_attn_1 = None
        if None not in all_post_attn_2:
            all_post_attn_2 = jnp.stack(all_post_attn_2, axis=0)
        else:
            all_post_attn_2 = None

        if apply_final_layer_norm:
            output = []
            output_norm = []
            for f, e, n_s in zip(self.final_norms, embedded, final_norm_scales, strict=True):
                if e is not None:
                    e, norm_scale = f(e, n_s)
                    output_norm.append(norm_scale)
                    output.append(e)
                else:
                    output.append(None)
                    output_norm.append(None)
        else:
            output = embedded
            output_norm = [None, None]

        layer_hidden_states = jnp.stack(layer_hidden_states, axis=0)
        post_attn_embeddings = jnp.stack(post_attn_embeddings, axis=0)

        # collect attn information
        attn_output = (mlp_activations, attention, output_pre_attn_scales, output_pre_mlp_scales,
                       output_norm, (all_post_attn_1, all_post_attn_2), layer_hidden_states, post_attn_embeddings,
                       text_representation)
        return output, (kv_cache_0, kv_cache_1), attn_output

    def init(self):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


class Analysis:
    @staticmethod
    def get_mlp_activation(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[0]))]
        return attn_output[0][jnp.asarray(layer_index)] if attn_output[0] is not None else None

    @staticmethod
    def get_attention(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[1]))]
        return attn_output[1][jnp.asarray(layer_index)]

    @staticmethod
    def get_pre_attn_norm_scales(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[2]))]
        return attn_output[2][jnp.asarray(layer_index)]

    @staticmethod
    def get_pre_mlp_norm_scales(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[3]))]
        return attn_output[3][jnp.asarray(layer_index)] if attn_output[3] is not None else None

    @staticmethod
    def get_final_norm_scales(attn_output):
        return attn_output[4]

    @staticmethod
    def get_post_attn_value(attn_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(attn_output[5][0]))]
        return [x[jnp.asarray(layer_index)] if x is not None else None for x in attn_output[5]]

    @staticmethod
    def get_neuron_memory(nnx_model, module_index, layer_index, neuron_index=None):
        layer = getattr(nnx_model, f"layers_{layer_index}")
        module_name = "mlp" if module_index == 0 else "mlp_1"
        memory = layer[module_name]["linear"]
        if neuron_index is not None:
            return memory[neuron_index]
        else:
            return memory

    @staticmethod
    def get_hidden_states(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[6]))]
        return layer_output[6][jnp.asarray(layer_index)]

    @staticmethod
    def get_post_attn_embedding(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[7]))]
        return layer_output[7][jnp.asarray(layer_index)]

    @staticmethod
    def get_text_representation(layer_output, layer_index=None):
        layer_index = layer_index or [i for i in range(len(layer_output[8]))]
        return layer_output[8][jnp.asarray(layer_index)]