"""The Privacy Filter network in MLX.

Token ids go in, one row of class logits per token comes out:

    logits = unembedding(norm(blocks(embedding(tokens))))

Each of the blocks is banded bidirectional attention followed by a
mixture-of-experts feed-forward layer, both pre-normalized and residual.
docs/architecture.md describes every step and where it was read from.

The module tree uses the checkpoint's tensor names (`block.3.attn.qkv.weight`,
`block.3.mlp.swiglu.bias`, ...), so the checkpoint loads without a name map.

Precision follows the reference's CPU path. The residual stream stays in the
dtype of the weights. Normalization, attention scores after the dot product,
the softmax, the router and the expert MLPs run in float32.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from .config import ModelConfig, rope_blend_range

LN2 = math.log(2.0)


class RMSNorm(nn.Module):
    """`x * rsqrt(mean(x^2) + eps) * scale` over the last axis, computed in float32."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.scale = mx.ones((size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        t = x.astype(mx.float32)
        t = t * mx.rsqrt(mx.mean(mx.square(t), axis=-1, keepdims=True) + self.eps)
        return (t * self.scale.astype(mx.float32)).astype(x.dtype)


@lru_cache(maxsize=4)
def rope_tables(config: ModelConfig, num_positions: int) -> tuple[mx.array, mx.array]:
    """Rotary cosine and sine tables: float32, shape [num_positions, head_dim / 2].

    Rotation pair `i` turns by `position * inv_freq[i]`. YaRN keeps the fast
    pairs at their original frequency, slows the slow pairs by the scaling
    factor, blends in between, and scales both tables by a constant.

    The table is built on the CPU stream, where the arithmetic is IEEE-exact,
    so it is the same whichever device runs the model.
    """
    half = config.head_dim // 2
    with mx.stream(mx.cpu):
        # MLX's float32 `power` is one unit in the last place off for about a
        # third of these values, and an error in a frequency grows with
        # position. Compute them in double precision and round once.
        exponents = (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim).tolist()
        freq = mx.array([config.rope_theta**exponent for exponent in exponents], dtype=mx.float32)

        if config.rope_scaling_factor > 1.0:
            low, high = rope_blend_range(
                config.head_dim,
                config.initial_context_length,
                config.rope_theta,
                config.rope_ntk_alpha,
                config.rope_ntk_beta,
            )
            ramp = (mx.arange(half, dtype=mx.float32) - low) / (high - low)
            keep = 1 - mx.clip(ramp, 0, 1)
            inv_freq = (1.0 / (config.rope_scaling_factor * freq)) * (1 - keep) + (1.0 / freq) * keep
            magnitude = 0.1 * math.log(config.rope_scaling_factor) + 1.0
        else:
            inv_freq = 1.0 / freq
            magnitude = 1.0

        angle = mx.arange(num_positions, dtype=mx.float32)[:, None] * inv_freq[None, :]
        cos, sin = mx.cos(angle) * magnitude, mx.sin(angle) * magnitude
        mx.eval(cos, sin)
    return cos, sin


def rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply the rotary tables to `x` [batch, tokens, heads, head_dim].

    Pairs are interleaved: (x[2i], x[2i+1]) is rotated by the angle of pair i.
    """
    cos = cos[:, None, :].astype(x.dtype)
    sin = sin[:, None, :].astype(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = mx.stack([even * cos - odd * sin, odd * cos + even * sin], axis=-1)
    return rotated.reshape(x.shape)


def scale(x: mx.array, factor: float) -> mx.array:
    """`x * factor`, multiplied in float32 and rounded once to the dtype of `x`.

    This is what PyTorch does with a bfloat16 tensor and a Python number. MLX
    would round the number to bfloat16 first, which changes it in the fourth
    digit and with it the last bit of a few percent of the products.
    """
    return (x.astype(mx.float32) * factor).astype(x.dtype)


class Attention(nn.Module):
    """Banded bidirectional grouped-query attention with a learned sink logit.

    Token t attends to tokens t-left .. t+right. Several query heads share each
    key/value head. A per-head scalar, the sink, joins every softmax as one more
    logit and is then dropped, so a head can attend to nothing.
    """

    def __init__(self, config: ModelConfig, block_size: int):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.qkv = nn.Linear(config.hidden_size, config.qkv_size)
        self.sinks = mx.zeros((config.num_attention_heads,))
        self.out = nn.Linear(config.attention_width, config.hidden_size)

        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.left = config.left_context
        self.right = config.right_context
        self.qk_scale = 1 / math.sqrt(math.sqrt(config.head_dim))
        self.block_size = block_size

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, T, _ = x.shape
        H, G, d = self.heads, self.kv_heads, self.head_dim
        M = H // G

        qkv = self.qkv(self.norm(x))
        q = qkv[..., : H * d].reshape(B, T, H, d)
        k = qkv[..., H * d : (H + G) * d].reshape(B, T, G, d)
        v = qkv[..., (H + G) * d :].reshape(B, T, G, d)

        q = scale(rotate(q, cos, sin), self.qk_scale)
        k = scale(rotate(k, cos, sin), self.qk_scale)

        q = q.reshape(B, T, G, M, d).transpose(0, 2, 3, 1, 4)
        k = k.transpose(0, 2, 1, 3)[:, :, None]
        v = v.transpose(0, 2, 1, 3)[:, :, None]
        sink = (self.sinks.astype(mx.float32) * LN2).reshape(1, G, M, 1, 1)  # stored in log2 units

        attended = []
        for start in range(0, T, self.block_size):
            stop = min(start + self.block_size, T)
            first, last = max(0, start - self.left), min(T, stop + self.right)

            scores = q[:, :, :, start:stop] @ k[:, :, :, first:last].swapaxes(-1, -2)
            scores = scores.astype(mx.float32)
            offset = mx.arange(first, last)[None, :] - mx.arange(start, stop)[:, None]
            in_band = mx.logical_and(offset >= -self.left, offset <= self.right)
            scores = mx.where(in_band, scores, -mx.inf)
            scores = mx.concatenate([scores, mx.broadcast_to(sink, (B, G, M, stop - start, 1))], axis=-1)

            weights = mx.softmax(scores, axis=-1)[..., :-1]
            values = v[:, :, :, first:last]
            attended.append(weights.astype(values.dtype) @ values)

        attended = attended[0] if len(attended) == 1 else mx.concatenate(attended, axis=3)
        attended = attended.transpose(0, 3, 1, 2, 4).reshape(B, T, H * d)
        return x + self.out(attended)


def swiglu(u: mx.array, limit: float, alpha: float) -> mx.array:
    """Clamped SwiGLU. The first half of `u` is the gate, the second half is linear."""
    gate, linear = mx.split(u, 2, axis=-1)
    gate = mx.minimum(gate, limit)
    linear = mx.clip(linear, -limit, limit)
    return gate * mx.sigmoid(alpha * gate) * (linear + 1)


class ExpertLinear(nn.Module):
    """One linear layer per expert. `weight` is [experts, in, out]: y = x @ weight[e] + bias[e]."""

    def __init__(self, experts: int, inputs: int, outputs: int):
        super().__init__()
        self.weight = mx.zeros((experts, inputs, outputs))
        self.bias = mx.zeros((experts, outputs))

    def __call__(self, x: mx.array, experts: mx.array) -> mx.array:
        """Apply to each float32 row of `x` [n, in] the expert named in `experts` [n], which must be sorted.

        The weights are upcast for this use. `gather_mm` multiplies every row
        by its own expert's matrix, so a row's result does not depend on the
        rows around it. Because the indices are sorted, MLX can take a run of
        rows that share an expert in one step.
        """
        weight, bias = self.weight.astype(mx.float32), self.bias.astype(mx.float32)
        product = mx.gather_mm(x[:, None, :], weight, rhs_indices=experts, sorted_indices=True)
        return product[:, 0, :] + bias[experts]


class MixtureOfExperts(nn.Module):
    """Feed-forward layer in which each token runs its top-k of many expert MLPs.

    A router scores every expert for every token. The k best run, and their
    outputs are mixed with a softmax over those k scores only.
    """

    def __init__(self, config: ModelConfig, chunk_size: int):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        self.swiglu = ExpertLinear(config.num_experts, config.hidden_size, 2 * config.intermediate_size)
        self.out = ExpertLinear(config.num_experts, config.intermediate_size, config.hidden_size)

        self.top_k = config.experts_per_token
        self.limit = config.swiglu_limit
        self.alpha = config.swiglu_alpha
        self.chunk_size = chunk_size

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x).reshape(-1, x.shape[-1]).astype(mx.float32)

        router = h @ self.gate.weight.astype(mx.float32).T + self.gate.bias.astype(mx.float32)
        chosen = mx.argsort(-router, axis=-1)[:, : self.top_k]
        mix = mx.softmax(mx.take_along_axis(router, chosen, axis=-1), axis=-1)

        steps = []
        for start in range(0, h.shape[0], self.chunk_size):
            stop = start + self.chunk_size
            outputs = self._experts(h[start:stop], chosen[start:stop])
            steps.append((outputs * mix[start:stop, :, None]).sum(axis=1))
            mx.eval(steps[-1])
        y = steps[0] if len(steps) == 1 else mx.concatenate(steps, axis=0)
        return x + y.astype(x.dtype).reshape(x.shape)

    def _experts(self, h: mx.array, chosen: mx.array) -> mx.array:
        n, k = chosen.shape
        expert_of_pair = chosen.reshape(-1)
        order = mx.argsort(expert_of_pair)
        experts = expert_of_pair[order]
        hidden = swiglu(self.swiglu(h[order // k], experts), self.limit, self.alpha)
        outputs = self.out(hidden, experts)[mx.argsort(order)]
        return outputs.reshape(n, k, -1)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, attention_block: int, expert_chunk: int):
        super().__init__()
        self.attn = Attention(config, attention_block)
        self.mlp = MixtureOfExperts(config, expert_chunk)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        return self.mlp(self.attn(x, cos, sin))


class Transformer(nn.Module):
    """The whole network.

    `attention_block` (queries per attention step) and `expert_chunk` (tokens
    whose experts run per step) bound working memory. `expert_chunk` does not
    change the result by a bit; `attention_block` changes at most the rounding
    of float32 sums.
    """

    def __init__(self, config: ModelConfig, *, attention_block: int = 128, expert_chunk: int = 16384):
        super().__init__()
        if attention_block < 1:
            raise ValueError(f"attention_block must be at least 1 (got {attention_block})")
        if expert_chunk < 1:
            raise ValueError(f"expert_chunk must be at least 1 (got {expert_chunk})")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.block = [Block(config, attention_block, expert_chunk) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        """Token ids [batch, tokens] -> logits [batch, tokens, num_labels] in the weights' dtype.

        Every position is a real token: there is no padding mask. Positions
        count from 0 within the sequence.
        """
        if tokens.ndim != 2:
            raise ValueError(f"expected a batch of token ids with shape [batch, tokens], got shape {tokens.shape}")
        if tokens.shape[1] == 0:
            raise ValueError("expected at least one token per sequence")
        lowest, highest = tokens.min().item(), tokens.max().item()
        if lowest < 0 or highest >= self.config.vocab_size:
            raise ValueError(
                f"token ids must be inside the vocabulary 0..{self.config.vocab_size - 1} "
                f"(got values from {lowest} to {highest})"
            )

        cos, sin = rope_tables(self.config, tokens.shape[1])
        x = self.embedding(tokens)
        for block in self.block:
            x = block(x, cos, sin)
        return self.unembedding(self.norm(x))
