"""The MLX model against an independent NumPy implementation and its own invariants.

`reference_forward` below is written straight from docs/architecture.md with
plain loops over tokens, heads and experts in float64. It shares no code and
no structure with `opf_mlx.model` (which uses blocked matmuls and groups tokens
by expert), so agreement between the two is evidence about the math rather than
about one implementation agreeing with itself.
"""

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from helpers import random_weights
from opf_mlx.config import ModelConfig
from opf_mlx.model import MixtureOfExperts, Transformer, rope_tables, swiglu
from opf_mlx.weights import expected_tensors

# --------------------------------------------------------------------------
# Independent reference
# --------------------------------------------------------------------------


def ref_rope(cfg: ModelConfig, num_positions: int) -> tuple[np.ndarray, np.ndarray]:
    half = cfg.head_dim // 2
    i = np.arange(half, dtype=np.float64)
    freq = cfg.rope_theta ** (2 * i / cfg.head_dim)
    if cfg.rope_scaling_factor > 1.0:
        log_theta = math.log(cfg.rope_theta)
        low = half * math.log(cfg.initial_context_length / (cfg.rope_ntk_beta * 2 * math.pi)) / log_theta
        high = half * math.log(cfg.initial_context_length / (cfg.rope_ntk_alpha * 2 * math.pi)) / log_theta
        mask = 1 - np.clip((i - low) / (high - low), 0, 1)
        inv = mask / freq + (1 - mask) / (cfg.rope_scaling_factor * freq)
        magnitude = 0.1 * math.log(cfg.rope_scaling_factor) + 1
    else:
        inv, magnitude = 1 / freq, 1.0
    angle = np.arange(num_positions, dtype=np.float64)[:, None] * inv[None, :]
    return magnitude * np.cos(angle), magnitude * np.sin(angle)


def ref_rotate(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """x: [T, heads, head_dim]; rotate interleaved pairs (2i, 2i+1) by position."""
    out = np.empty_like(x)
    for t in range(x.shape[0]):
        for i in range(x.shape[2] // 2):
            a, b = x[t, :, 2 * i], x[t, :, 2 * i + 1]
            out[t, :, 2 * i] = a * cos[t, i] - b * sin[t, i]
            out[t, :, 2 * i + 1] = b * cos[t, i] + a * sin[t, i]
    return out


def ref_rmsnorm(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5) * scale


def ref_softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def reference_forward(cfg: ModelConfig, w: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    T, d, H, G = len(tokens), cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
    inner = cfg.intermediate_size
    cos, sin = ref_rope(cfg, T)
    x = w["embedding.weight"][tokens]

    for n in range(cfg.num_hidden_layers):
        p = f"block.{n}.attn"
        h = ref_rmsnorm(x, w[f"{p}.norm.scale"])
        qkv = h @ w[f"{p}.qkv.weight"].T + w[f"{p}.qkv.bias"]
        q = qkv[:, : H * d].reshape(T, H, d)
        k = qkv[:, H * d : (H + G) * d].reshape(T, G, d)
        v = qkv[:, (H + G) * d :].reshape(T, G, d)
        q = ref_rotate(q, cos, sin) * d**-0.25
        k = ref_rotate(k, cos, sin) * d**-0.25
        attended = np.zeros((T, H, d))
        for t in range(T):
            lo, hi = max(0, t - cfg.left_context), min(T, t + cfg.right_context + 1)
            for head in range(H):
                group = head // (H // G)
                scores = [q[t, head] @ k[j, group] for j in range(lo, hi)]
                scores.append(w[f"{p}.sinks"][head] * math.log(2.0))  # the sink: a logit with no value
                prob = ref_softmax(np.array(scores))
                for offset, j in enumerate(range(lo, hi)):
                    attended[t, head] += prob[offset] * v[j, group]
        x = x + attended.reshape(T, H * d) @ w[f"{p}.out.weight"].T + w[f"{p}.out.bias"]

        p = f"block.{n}.mlp"
        h = ref_rmsnorm(x, w[f"{p}.norm.scale"])
        router = h @ w[f"{p}.gate.weight"].T + w[f"{p}.gate.bias"]
        update = np.zeros_like(x)
        for t in range(T):
            chosen = np.argsort(-router[t])[: cfg.experts_per_token]
            mix = ref_softmax(router[t, chosen])
            for expert, weight in zip(chosen, mix):
                u = h[t] @ w[f"{p}.swiglu.weight"][expert] + w[f"{p}.swiglu.bias"][expert]
                gate = np.minimum(u[:inner], cfg.swiglu_limit)
                linear = np.clip(u[inner:], -cfg.swiglu_limit, cfg.swiglu_limit)
                act = gate / (1 + np.exp(-1.702 * gate)) * (linear + 1)
                update[t] += weight * (act @ w[f"{p}.out.weight"][expert] + w[f"{p}.out.bias"][expert])
        x = x + update

    return ref_rmsnorm(x, w["norm.scale"]) @ w["unembedding.weight"].T


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    from conftest import TINY_CONFIG

    return ModelConfig.from_dict(TINY_CONFIG)


@pytest.fixture(scope="module")
def stored(cfg) -> dict[str, mx.array]:
    """Random weights in the checkpoint's storage dtypes (bfloat16 stream)."""
    return random_weights(cfg, seed=7)


def as_float32(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {name: array.astype(mx.float32) for name, array in weights.items()}


def as_numpy(weights: dict[str, mx.array]) -> dict[str, np.ndarray]:
    return {name: np.array(array.astype(mx.float32)).astype(np.float64) for name, array in weights.items()}


def build(cfg: ModelConfig, weights: dict[str, mx.array], **options) -> Transformer:
    model = Transformer(cfg, **options)
    model.load_weights(list(weights.items()), strict=True)
    return model


def run(model: Transformer, tokens: list[int]) -> np.ndarray:
    return np.array(model(mx.array([tokens])).astype(mx.float32))[0]


TOKENS = [3, 141, 59, 26, 53, 58, 97, 93, 23, 84, 62, 64, 33, 83, 27, 95, 2, 88, 41, 97, 16, 93, 99]
DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_logits_match_the_independent_numpy_implementation(cfg, stored):
    # 23 tokens against a band of 3 left / 2 right: most pairs are masked.
    expected = reference_forward(cfg, as_numpy(stored), TOKENS)

    logits = run(build(cfg, as_float32(stored)), TOKENS)

    assert logits.shape == (23, 33)
    np.testing.assert_allclose(logits, expected, rtol=2e-4, atol=2e-4)


def test_checkpoint_precision_stays_close_to_exact_arithmetic(cfg, stored):
    expected = reference_forward(cfg, as_numpy(stored), TOKENS)

    logits = run(build(cfg, stored), TOKENS)

    # bfloat16 keeps 8 significant bits and rounds after every step. Single
    # logits can be off by tenths (measured: up to 1.4 on a scale of 6 across
    # seeds), but the error is a few percent of the signal on average. The
    # sharp check of this path is parity with the PyTorch reference, one
    # sublayer at a time: tests/test_golden.py and tests/test_parity.py.
    assert np.isfinite(logits).all()
    assert np.sqrt(np.mean((logits - expected) ** 2)) < 0.1 * np.sqrt(np.mean(expected**2))


def test_logits_come_out_in_the_residual_stream_precision(cfg, stored):
    tokens = mx.array([TOKENS])

    assert build(cfg, stored)(tokens).dtype == mx.bfloat16
    assert build(cfg, as_float32(stored))(tokens).dtype == mx.float32


def test_a_token_only_affects_logits_within_the_receptive_field(cfg, stored):
    model = build(cfg, as_float32(stored))
    rng = np.random.default_rng(3)
    tokens = [int(t) for t in rng.integers(0, 500, size=40)]
    changed = list(tokens)
    changed[20] = tokens[20] + 1

    delta = np.abs(run(model, changed) - run(model, tokens)).max(axis=1)

    # Token t reads tokens t-3..t+2, so token 20 is read by tokens 18..23, and
    # after two layers by tokens 16..26.
    assert delta[20] > 1e-3
    assert delta[16] > 1e-6 and delta[26] > 1e-6
    assert delta[:16].max() < 1e-5
    assert delta[27:].max() < 1e-5


def test_results_do_not_depend_on_how_the_work_is_chunked(cfg, stored):
    weights = as_float32(stored)
    whole = run(build(cfg, weights), TOKENS)

    chunked = run(build(cfg, weights, attention_block=4, expert_chunk=5), TOKENS)

    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision", ["bfloat16", "float32"])
def test_expert_steps_of_any_size_give_the_same_bits(cfg, stored, device, precision):
    weights = stored if precision == "bfloat16" else as_float32(stored)

    with mx.stream(DEVICES[device]):
        whole = run(build(cfg, weights), TOKENS)
        stepped = {size: run(build(cfg, weights, expert_chunk=size), TOKENS) for size in (1, 2, 5)}

    for size, logits in stepped.items():
        assert np.array_equal(logits, whole), f"{size} tokens per step"


@pytest.mark.parametrize("device", DEVICES)
def test_a_tokens_experts_give_it_the_same_bits_whatever_else_is_routed_with_it(released_config, device):
    # The reference applies experts to one token at a time. Grouping tokens by
    # expert must not let the other tokens into a token's arithmetic, or its
    # result would depend on the rest of the input and on `expert_chunk`. A
    # matrix product of one row and the same row inside a product of many can
    # differ in the last bit, so this is checked at the released layer sizes,
    # where the kernels are the ones that run in practice.
    config = ModelConfig.from_dict({**released_config, "num_experts": 8})
    rng = np.random.default_rng(0)
    parameters = tree_map(
        lambda p: mx.array(rng.standard_normal(p.shape).astype(np.float32) * 0.05),
        MixtureOfExperts(config, 1).parameters(),
    )
    x = mx.array(rng.standard_normal((1, 48, config.hidden_size)).astype(np.float32))

    def expert_layer(tokens_per_step: int) -> MixtureOfExperts:
        layer = MixtureOfExperts(config, tokens_per_step)
        layer.update(parameters)
        return layer

    with mx.stream(DEVICES[device]):
        together = np.array(expert_layer(48)(x))  # about 24 tokens share each expert
        one_by_one = np.array(expert_layer(1)(x))  # every token is alone with each of its experts

    assert not np.array_equal(together, np.array(x))  # the layer did something
    assert np.array_equal(one_by_one, together)


def test_the_experts_work_without_a_copy_of_their_weights_for_every_token(released_config):
    # One expert's first matrix is 3.3 MB at the released sizes. Picking the
    # matrices out per token (`weight[experts]`) gives the same numbers and
    # needs that much for each of a token's four experts: 3 GiB for the 256
    # tokens here, 200 GiB for one step of the default size.
    config = ModelConfig.from_dict({**released_config, "num_experts": 8})
    rng = np.random.default_rng(0)
    layer = MixtureOfExperts(config, 256)
    layer.update(
        tree_map(lambda p: mx.array(rng.standard_normal(p.shape).astype(np.float32) * 0.05), layer.parameters())
    )
    x = mx.array(rng.standard_normal((1, 256, config.hidden_size)).astype(np.float32))
    mx.eval(layer.parameters(), x)

    held = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(layer(x))

    assert mx.get_peak_memory() - held < 256 * 2**20


def test_sequences_in_a_batch_do_not_interact(cfg, stored):
    model = build(cfg, as_float32(stored))
    first, second = TOKENS[:12], TOKENS[11:]

    batch = np.array(model(mx.array([first, second])))

    np.testing.assert_allclose(batch[0], run(model, first), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(batch[1], run(model, second), rtol=1e-5, atol=1e-5)


def test_a_dominant_sink_silences_attention(cfg, stored):
    weights = as_float32(stored)
    for n in range(cfg.num_hidden_layers):
        weights[f"block.{n}.attn.sinks"] = mx.full((cfg.num_attention_heads,), 200.0)
    model = build(cfg, weights)
    x = mx.array(np.random.default_rng(5).standard_normal((1, 9, cfg.hidden_size)).astype(np.float32))
    cos, sin = rope_tables(cfg, 9)

    out = model.block[0].attn(x, cos, sin)

    # With all probability on the sink, nothing is read from any token and the
    # block adds only its output bias.
    np.testing.assert_allclose(np.array(out), np.array(x + weights["block.0.attn.out.bias"]), atol=1e-5)


def test_with_uniform_scores_each_token_keeps_its_band_share_against_the_sink(cfg, stored):
    # Zero query/key projections make every score 0, so a token that can see n
    # tokens gives each of them weight 1 / (n + 2^sink): the sink is stored in
    # log2 units. Every value is 1, so the block adds n / (n + 2^sink).
    weights = as_float32(stored)
    bias = np.zeros(80, dtype=np.float32)
    bias[64:] = 1.0  # the value part of the fused q/k/v projection (48 + 16 + 16)
    weights["block.0.attn.qkv.weight"] = mx.zeros((80, 16))
    weights["block.0.attn.qkv.bias"] = mx.array(bias)
    weights["block.0.attn.sinks"] = mx.ones((6,))  # 2^1 = 2
    weights["block.0.attn.out.weight"] = mx.array(np.eye(16, 48, dtype=np.float32))
    weights["block.0.attn.out.bias"] = mx.zeros((16,))
    model = build(cfg, weights)
    x = mx.array(np.random.default_rng(11).standard_normal((1, 9, 16)).astype(np.float32))

    added = np.array(model.block[0].attn(x, *rope_tables(cfg, 9)) - x)[0]

    # The band is 3 left / 2 right, so tokens 0..8 of 9 see 3, 4, 5, 6, 6, 6, 6, 5, 4 tokens.
    share = np.array([3 / 5, 4 / 6, 5 / 7, 6 / 8, 6 / 8, 6 / 8, 6 / 8, 5 / 7, 4 / 6])
    np.testing.assert_allclose(added, np.repeat(share[:, None], 16, axis=1), atol=1e-5)


def test_the_query_and_key_scale_is_applied_in_float32_and_rounded_once(stored):
    # PyTorch multiplies a bfloat16 tensor by a Python number in float32 and
    # rounds the product once. Rounding the number first gives another scale:
    # for a head size of 8 it is 8^-0.25 = 0.59460354 in float32 but 0.59375 in
    # bfloat16. One token whose query and key are both 1248 tells the rules apart:
    #
    #   query, key  1248 * 0.59460354 = 742.07 -> 744    (bfloat16 steps of 4)
    #               1248 * 0.59375    = 741.00 -> 740
    #   score       744 * 744 = 553536 -> 552960         (steps of 4096)
    #               744 * 740 = 550560 -> 548864         (one of the two scaled the other way)
    #               742.07 * 742.07 = 550661             (neither rounded after scaling)
    #   sink        794624 * ln 2 = 550791
    #
    # Done right, the token beats the sink by 2169 and keeps all of the weight.
    # Done any other way it loses by 130 or more and keeps none.
    from conftest import TINY_CONFIG

    config = ModelConfig.from_dict({**TINY_CONFIG, "rope_scaling_factor": 1.0})  # position 0 is left as it is
    bias = np.zeros(80, dtype=np.float32)
    bias[0:48:8] = 1248.0  # the first coordinate of each of the 6 queries,
    bias[48:64:8] = 1248.0  # of the 2 keys
    bias[64:80:8] = 1.0  # and of the 2 values
    weights = dict(stored)
    weights["block.0.attn.qkv.weight"] = mx.zeros((80, 16), dtype=mx.bfloat16)
    weights["block.0.attn.qkv.bias"] = mx.array(bias).astype(mx.bfloat16)
    weights["block.0.attn.sinks"] = mx.full((6,), 794624.0).astype(stored["block.0.attn.sinks"].dtype)
    weights["block.0.attn.out.weight"] = mx.array(np.eye(16, 48, dtype=np.float32)).astype(mx.bfloat16)
    weights["block.0.attn.out.bias"] = mx.zeros((16,), dtype=mx.bfloat16)
    model = build(config, weights)
    x = mx.zeros((1, 1, 16), dtype=mx.bfloat16)

    added = np.array(model.block[0].attn(x, *rope_tables(config, 1)).astype(mx.float32))[0, 0]

    # Residual dimension j receives coordinate j of the attention output, so
    # dimensions 0 and 8 hold the first coordinates of heads 0 and 1.
    assert added.tolist() == [1.0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0]


def test_token_ids_must_be_a_batch(cfg, stored):
    with pytest.raises(ValueError, match="batch"):
        build(cfg, stored)(mx.array(TOKENS))


@pytest.mark.parametrize("option", ["attention_block", "expert_chunk"])
def test_chunk_sizes_must_be_positive(cfg, option):
    with pytest.raises(ValueError, match=option):
        Transformer(cfg, **{option: 0})


def test_empty_sequence_is_rejected(cfg, stored):
    with pytest.raises(ValueError, match="at least one token"):
        build(cfg, stored)(mx.zeros((1, 0), dtype=mx.int32))


@pytest.mark.parametrize("bad", [-1, 200064])
def test_token_ids_outside_the_vocabulary_are_rejected(cfg, stored, bad):
    # Out-of-range gathers are undefined in MLX, so they must never reach the embedding.
    with pytest.raises(ValueError, match="vocabulary"):
        build(cfg, stored)(mx.array([[5, bad, 7]]))


def test_rope_tables_stay_accurate_far_from_the_origin(released_config):
    config = ModelConfig.from_dict(released_config)

    cos, sin = rope_tables(config, 131072)

    # Pair 0 turns exactly one radian per token, so the expected values need no
    # float32 angle rounding. Trigonometry with poor range reduction fails here.
    for position in (100_000, 131_071):
        assert cos[position, 0].item() == pytest.approx(1.3465736 * math.cos(position), abs=1e-6)
        assert sin[position, 0].item() == pytest.approx(1.3465736 * math.sin(position), abs=1e-6)


@pytest.mark.parametrize("fixture_name", ["tiny_config", "released_config"])
def test_module_tree_is_exactly_the_checkpoint_schema(fixture_name, request):
    config = ModelConfig.from_dict(request.getfixturevalue(fixture_name))

    parameters = {name: tuple(array.shape) for name, array in tree_flatten(Transformer(config).parameters())}

    assert parameters == {name: spec.shape for name, spec in expected_tensors(config).items()}


def test_rope_tables_follow_the_yarn_formula(released_config):
    config = ModelConfig.from_dict(released_config)

    cos, sin = rope_tables(config, 300)
    want_cos, want_sin = ref_rope(config, 300)

    assert cos.dtype == mx.float32 and cos.shape == (300, 32)
    np.testing.assert_allclose(np.array(cos), want_cos, atol=1e-4)
    np.testing.assert_allclose(np.array(sin), want_sin, atol=1e-4)
    # Magnitude 0.1 * ln(32) + 1 at position 0; pair 0 turns one radian per token.
    assert np.array(cos)[0, 0] == pytest.approx(1.3465736, abs=1e-6)
    assert np.array(sin)[1, 0] == pytest.approx(1.3465736 * math.sin(1.0), abs=1e-6)
    # Pair 31 is slowed by the scaling factor: 1 / (32 * 150000^(62/64)).
    assert np.array(sin)[299, 31] == pytest.approx(1.3465736 * math.sin(299 / (32 * 150000 ** (62 / 64))), abs=1e-7)


def test_swiglu_uses_contiguous_halves_clamps_and_adds_one():
    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-z))

    # Gate half is the first three entries, linear half the last three.
    u = mx.array([[8.0, 1.0, -9.0, 0.5, -9.0, 2.0]])

    out = np.array(swiglu(u, limit=7.0, alpha=1.702))[0]

    # Gate 8.0 clamps to 7.0. Linear -9.0 clamps to -7.0. A gate of -9.0 is
    # not clamped: only the upper side of the gate is limited.
    assert out[0] == pytest.approx(7.0 * sigmoid(1.702 * 7.0) * (0.5 + 1), rel=1e-6)
    assert out[1] == pytest.approx(1.0 * sigmoid(1.702) * (-7.0 + 1), rel=1e-6)
    assert out[2] == pytest.approx(-9.0 * sigmoid(1.702 * -9.0) * (2.0 + 1), rel=1e-4)
