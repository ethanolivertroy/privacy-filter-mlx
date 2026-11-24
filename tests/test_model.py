import math

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map

from helpers import random_weights
from opf_mlx.config import ModelConfig
from opf_mlx.model import MixtureOfExperts, Transformer, rope_tables, swiglu
from opf_mlx.weights import expected_tensors


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
                scores.append(w[f"{p}.sinks"][head] * math.log(2.0))
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


@pytest.fixture(scope="module")
def cfg() -> ModelConfig:
    from conftest import TINY_CONFIG

    return ModelConfig.from_dict(TINY_CONFIG)


@pytest.fixture(scope="module")
def stored(cfg) -> dict[str, mx.array]:
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


def test_logits_match_the_independent_numpy_implementation(cfg, stored):
    expected = reference_forward(cfg, as_numpy(stored), TOKENS)

    logits = run(build(cfg, as_float32(stored)), TOKENS)

    assert logits.shape == (23, 33)
    np.testing.assert_allclose(logits, expected, rtol=2e-4, atol=2e-4)


def test_checkpoint_precision_stays_close_to_exact_arithmetic(cfg, stored):
    expected = reference_forward(cfg, as_numpy(stored), TOKENS)

    logits = run(build(cfg, stored), TOKENS)

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
        together = np.array(expert_layer(48)(x))
        one_by_one = np.array(expert_layer(1)(x))

    assert not np.array_equal(together, np.array(x))
    assert np.array_equal(one_by_one, together)


def test_the_experts_work_without_a_copy_of_their_weights_for_every_token(released_config):
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

    np.testing.assert_allclose(np.array(out), np.array(x + weights["block.0.attn.out.bias"]), atol=1e-5)


def test_with_uniform_scores_each_token_keeps_its_band_share_against_the_sink(cfg, stored):
    weights = as_float32(stored)
    bias = np.zeros(80, dtype=np.float32)
    bias[64:] = 1.0
    weights["block.0.attn.qkv.weight"] = mx.zeros((80, 16))
    weights["block.0.attn.qkv.bias"] = mx.array(bias)
    weights["block.0.attn.sinks"] = mx.ones((6,))
    weights["block.0.attn.out.weight"] = mx.array(np.eye(16, 48, dtype=np.float32))
    weights["block.0.attn.out.bias"] = mx.zeros((16,))
    model = build(cfg, weights)
    x = mx.array(np.random.default_rng(11).standard_normal((1, 9, 16)).astype(np.float32))

    added = np.array(model.block[0].attn(x, *rope_tables(cfg, 9)) - x)[0]

    share = np.array([3 / 5, 4 / 6, 5 / 7, 6 / 8, 6 / 8, 6 / 8, 6 / 8, 5 / 7, 4 / 6])
    np.testing.assert_allclose(added, np.repeat(share[:, None], 16, axis=1), atol=1e-5)


def test_the_query_and_key_scale_is_applied_in_float32_and_rounded_once(stored):
    from conftest import TINY_CONFIG

    config = ModelConfig.from_dict({**TINY_CONFIG, "rope_scaling_factor": 1.0})
    head_dim = config.head_dim
    q_end = config.num_attention_heads * head_dim
    k_end = q_end + config.num_key_value_heads * head_dim
    v_end = config.qkv_size

    qk_scale = head_dim**-0.25
    probe_value = 1248.0
    score_if_f32_scaled = float((mx.array(probe_value, mx.float32) * qk_scale).astype(mx.bfloat16).item()) ** 2
    score_if_bf16_scaled = float((mx.array(probe_value, mx.bfloat16) * mx.array(qk_scale, mx.bfloat16)).item()) ** 2
    assert score_if_f32_scaled > score_if_bf16_scaled
    threshold_sink_log2 = ((score_if_f32_scaled + score_if_bf16_scaled) / 2) / math.log(2.0)

    bias = np.zeros(v_end, dtype=np.float32)
    bias[0:q_end:head_dim] = probe_value
    bias[q_end:k_end:head_dim] = probe_value
    bias[k_end:v_end:head_dim] = 1.0
    weights = dict(stored)
    weights["block.0.attn.qkv.weight"] = mx.zeros((v_end, config.hidden_size), dtype=mx.bfloat16)
    weights["block.0.attn.qkv.bias"] = mx.array(bias).astype(mx.bfloat16)
    weights["block.0.attn.sinks"] = mx.full((config.num_attention_heads,), threshold_sink_log2).astype(
        stored["block.0.attn.sinks"].dtype
    )
    weights["block.0.attn.out.weight"] = mx.array(np.eye(config.hidden_size, q_end, dtype=np.float32)).astype(mx.bfloat16)
    weights["block.0.attn.out.bias"] = mx.zeros((config.hidden_size,), dtype=mx.bfloat16)
    model = build(config, weights)
    x = mx.zeros((1, 1, config.hidden_size), dtype=mx.bfloat16)

    added = np.array(model.block[0].attn(x, *rope_tables(config, 1)).astype(mx.float32))[0, 0]

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
    with pytest.raises(ValueError, match="vocabulary"):
        build(cfg, stored)(mx.array([[5, bad, 7]]))


def test_rope_tables_stay_accurate_far_from_the_origin(released_config):
    config = ModelConfig.from_dict(released_config)

    cos, sin = rope_tables(config, 131072)

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
    assert np.array(cos)[0, 0] == pytest.approx(1.3465736, abs=1e-6)
    assert np.array(sin)[1, 0] == pytest.approx(1.3465736 * math.sin(1.0), abs=1e-6)
    assert np.array(sin)[299, 31] == pytest.approx(1.3465736 * math.sin(299 / (32 * 150000 ** (62 / 64))), abs=1e-7)


def test_swiglu_uses_contiguous_halves_clamps_and_adds_one():
    def sigmoid(z: float) -> float:
        return 1 / (1 + math.exp(-z))

    u = mx.array([[8.0, 1.0, -9.0, 0.5, -9.0, 2.0]])

    out = np.array(swiglu(u, limit=7.0, alpha=1.702))[0]

    assert out[0] == pytest.approx(7.0 * sigmoid(1.702 * 7.0) * (0.5 + 1), rel=1e-6)
    assert out[1] == pytest.approx(1.0 * sigmoid(1.702) * (-7.0 + 1), rel=1e-6)
    assert out[2] == pytest.approx(-9.0 * sigmoid(1.702 * -9.0) * (2.0 + 1), rel=1e-4)
