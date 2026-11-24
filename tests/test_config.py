import pytest

from opf_mlx.config import ConfigError, ModelConfig


def test_released_config_is_accepted_and_derives_head_grouping(released_config):
    cfg = ModelConfig.from_dict(released_config)

    assert (cfg.num_hidden_layers, cfg.hidden_size, cfg.num_labels) == (8, 640, 33)
    assert (cfg.left_context, cfg.right_context) == (128, 128)
    assert cfg.queries_per_kv_head == 7
    assert cfg.qkv_size == 1152
    assert cfg.swiglu_limit == 7.0


def test_context_length_defaults_to_the_checkpoints_default_n_ctx(released_config):
    assert ModelConfig.from_dict(released_config).default_n_ctx == 128000


def test_context_length_falls_back_to_initial_context_length(released_config):
    del released_config["default_n_ctx"]

    assert ModelConfig.from_dict(released_config).default_n_ctx == 4096


@pytest.mark.parametrize("bad", [0, -5, True, "4096", 12.5])
def test_non_positive_or_non_integer_default_n_ctx_is_rejected(released_config, bad):
    released_config["default_n_ctx"] = bad

    with pytest.raises(ConfigError, match="default_n_ctx"):
        ModelConfig.from_dict(released_config)


def test_all_missing_keys_are_reported_together(released_config):
    del released_config["num_experts"]
    del released_config["rope_theta"]

    with pytest.raises(ConfigError) as excinfo:
        ModelConfig.from_dict(released_config)

    assert "num_experts" in str(excinfo.value)
    assert "rope_theta" in str(excinfo.value)


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("model_type", "gpt_oss", "model_type"),
        ("bidirectional_context", False, "bidirectional_context"),
        ("sliding_window", 128, "sliding_window"),
        ("num_attention_heads", 13, "num_attention_heads"),
        ("head_dim", 63, "head_dim"),
        ("experts_per_token", 129, "experts_per_token"),
        ("param_dtype", "float16", "param_dtype"),
        ("inference_contract_version", 2, "inference_contract_version"),
        ("encoding", "", "encoding"),
        ("num_hidden_layers", 0, "num_hidden_layers"),
        ("hidden_size", 640.5, "hidden_size"),
    ],
)
def test_incompatible_values_are_rejected(released_config, key, value, message):
    released_config[key] = value

    with pytest.raises(ConfigError, match=message):
        ModelConfig.from_dict(released_config)


def test_packed_activation_layout_is_rejected_rather_than_misread(released_config):
    released_config["packed_geglu"] = True

    with pytest.raises(ConfigError, match="packed_geglu"):
        ModelConfig.from_dict(released_config)


@pytest.mark.parametrize(
    "alias, expected",
    [("bfloat16", "bfloat16"), ("bf16", "bfloat16"), ("float32", "float32"), ("fp32", "float32"), ("BF16", "bfloat16")],
)
def test_param_dtype_aliases_are_normalized(released_config, alias, expected):
    released_config["param_dtype"] = alias

    assert ModelConfig.from_dict(released_config).param_dtype == expected


def test_rope_blend_region_must_lie_inside_the_frequency_range(released_config):
    released_config["initial_context_length"] = 16

    with pytest.raises(ConfigError, match="rope"):
        ModelConfig.from_dict(released_config)


def test_from_file_reads_json(tmp_path, released_config):
    import json

    path = tmp_path / "config.json"
    path.write_text(json.dumps(released_config), encoding="utf-8")

    assert ModelConfig.from_file(path).vocab_size == 200064


def test_from_file_rejects_non_object_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ConfigError, match="JSON object"):
        ModelConfig.from_file(path)
