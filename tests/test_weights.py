"""Checkpoint tensor inventory, validation and loading."""

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from helpers import random_weights, write_checkpoint
from opf_mlx.config import ModelConfig
from opf_mlx.weights import WeightsError, expected_tensors, load_weights, read_safetensors_header


@pytest.fixture
def cfg(tiny_config) -> ModelConfig:
    return ModelConfig.from_dict(tiny_config)


@pytest.fixture
def weights(cfg) -> dict[str, mx.array]:
    return random_weights(cfg, seed=1)


def test_schema_for_the_released_config_matches_the_checkpoint_inventory(released_config):
    schema = expected_tensors(ModelConfig.from_dict(released_config))

    # 107 tensors and 1,399,486,832 parameters are what the header of the
    # released original/model.safetensors contains.
    assert len(schema) == 107
    assert sum(int(np.prod(spec.shape)) for spec in schema.values()) == 1_399_486_832
    assert schema["embedding.weight"].shape == (200064, 640)
    assert schema["block.3.attn.qkv.weight"].shape == (1152, 640)
    assert schema["block.3.attn.qkv.bias"].shape == (1152,)
    assert schema["block.0.attn.sinks"].shape == (14,)
    assert schema["block.5.attn.out.weight"].shape == (640, 896)
    assert schema["block.7.mlp.gate.weight"].shape == (128, 640)
    assert schema["block.7.mlp.swiglu.weight"].shape == (128, 640, 1280)
    assert schema["block.7.mlp.swiglu.bias"].shape == (128, 1280)
    assert schema["block.7.mlp.out.weight"].shape == (128, 640, 640)
    assert schema["block.7.mlp.out.bias"].shape == (128, 640)
    assert schema["norm.scale"].shape == (640,)
    assert schema["unembedding.weight"].shape == (33, 640)
    assert "unembedding.bias" not in schema


def test_stream_tensors_must_be_stored_at_the_configured_precision(released_config):
    schema = expected_tensors(ModelConfig.from_dict(released_config))

    assert schema["embedding.weight"].dtypes == ("bfloat16",)
    assert schema["block.0.mlp.swiglu.weight"].dtypes == ("bfloat16",)
    # The reference holds these two kinds in float32 parameters, so either
    # storage dtype loads without loss. The released file stores the sinks as
    # F32 and the norm scales as BF16.
    assert set(schema["block.0.attn.sinks"].dtypes) == {"float32", "bfloat16"}
    assert set(schema["block.0.attn.norm.scale"].dtypes) == {"float32", "bfloat16"}


def test_valid_checkpoint_loads_every_tensor_as_stored(tmp_path, tiny_config, cfg, weights):
    write_checkpoint(tmp_path, tiny_config, weights)

    loaded = load_weights(tmp_path, cfg)

    assert set(loaded) == set(weights)
    assert loaded["embedding.weight"].dtype == mx.bfloat16
    assert loaded["block.0.attn.sinks"].dtype == mx.float32
    for name in ("block.1.mlp.out.weight", "block.0.attn.sinks", "norm.scale"):
        assert loaded[name].shape == weights[name].shape
        assert mx.array_equal(loaded[name], weights[name])


def test_missing_tensor_fails_and_names_it(tmp_path, tiny_config, cfg, weights):
    del weights["block.1.mlp.gate.bias"]
    write_checkpoint(tmp_path, tiny_config, weights)

    with pytest.raises(WeightsError, match=r"missing.*block\.1\.mlp\.gate\.bias"):
        load_weights(tmp_path, cfg)


def test_unexpected_tensor_fails_and_names_it(tmp_path, tiny_config, cfg, weights):
    weights["unembedding.bias"] = mx.zeros((33,), dtype=mx.bfloat16)
    write_checkpoint(tmp_path, tiny_config, weights)

    with pytest.raises(WeightsError, match=r"unexpected.*unembedding\.bias"):
        load_weights(tmp_path, cfg)


def test_wrong_shape_fails_and_shows_both_shapes(tmp_path, tiny_config, cfg, weights):
    weights["block.0.attn.qkv.weight"] = weights["block.0.attn.qkv.weight"].T
    write_checkpoint(tmp_path, tiny_config, weights)

    with pytest.raises(WeightsError, match=r"block\.0\.attn\.qkv\.weight.*\(80, 16\).*\(16, 80\)"):
        load_weights(tmp_path, cfg)


def test_wrong_dtype_fails_instead_of_converting(tmp_path, tiny_config, cfg, weights):
    weights["block.0.mlp.out.weight"] = weights["block.0.mlp.out.weight"].astype(mx.float32)
    write_checkpoint(tmp_path, tiny_config, weights)

    with pytest.raises(WeightsError, match=r"block\.0\.mlp\.out\.weight.*bfloat16.*float32"):
        load_weights(tmp_path, cfg)


def test_every_problem_is_reported_in_one_error(tmp_path, tiny_config, cfg, weights):
    del weights["norm.scale"]
    weights["surprise"] = mx.zeros((2,))
    weights["unembedding.weight"] = weights["unembedding.weight"].astype(mx.float16)
    write_checkpoint(tmp_path, tiny_config, weights)

    with pytest.raises(WeightsError) as excinfo:
        load_weights(tmp_path, cfg)

    message = str(excinfo.value)
    assert "norm.scale" in message and "surprise" in message and "unembedding.weight" in message


def test_sinks_and_norm_scales_may_be_stored_in_either_lossless_dtype(tmp_path, tiny_config, cfg, weights):
    weights["block.0.attn.sinks"] = weights["block.0.attn.sinks"].astype(mx.bfloat16)
    weights["norm.scale"] = weights["norm.scale"].astype(mx.float32)
    write_checkpoint(tmp_path, tiny_config, weights)

    loaded = load_weights(tmp_path, cfg)

    assert loaded["block.0.attn.sinks"].dtype == mx.bfloat16
    assert loaded["norm.scale"].dtype == mx.float32


def test_float32_option_upcasts_every_tensor_without_changing_values(tmp_path, tiny_config, cfg, weights):
    write_checkpoint(tmp_path, tiny_config, weights)

    loaded = load_weights(tmp_path, cfg, dtype="float32")

    assert all(array.dtype == mx.float32 for array in loaded.values())
    name = "block.0.mlp.swiglu.weight"
    assert mx.array_equal(loaded[name], weights[name].astype(mx.float32))


def test_downcasting_a_float32_checkpoint_is_refused(tmp_path, tiny_config):
    tiny_config["param_dtype"] = "float32"
    cfg32 = ModelConfig.from_dict(tiny_config)
    write_checkpoint(tmp_path, tiny_config, random_weights(cfg32))

    with pytest.raises(ValueError, match="bfloat16"):
        load_weights(tmp_path, cfg32, dtype="bfloat16")


def test_tensors_may_be_split_across_several_files(tmp_path, tiny_config, cfg, weights):
    write_checkpoint(tmp_path, tiny_config, {k: v for k, v in weights.items() if k.startswith("block.")})
    mx.save_safetensors(str(tmp_path / "rest.safetensors"), {k: v for k, v in weights.items() if not k.startswith("block.")})

    assert set(load_weights(tmp_path, cfg)) == set(weights)


def test_tensor_present_in_two_files_is_rejected(tmp_path, tiny_config, cfg, weights):
    write_checkpoint(tmp_path, tiny_config, weights)
    mx.save_safetensors(str(tmp_path / "dup.safetensors"), {"norm.scale": weights["norm.scale"]})

    with pytest.raises(WeightsError, match=r"norm\.scale.*more than one file"):
        load_weights(tmp_path, cfg)


def test_directory_without_weights_is_rejected(tmp_path, cfg):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        load_weights(tmp_path, cfg)


def test_truncated_file_is_detected_from_the_header_alone(tmp_path, tiny_config, weights):
    path = write_checkpoint(tmp_path, tiny_config, weights) / "model.safetensors"
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 1000])

    with pytest.raises(WeightsError, match="truncated|size"):
        read_safetensors_header(path)


def test_file_that_is_not_safetensors_is_rejected(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", 10_000_000) + b"not json")

    with pytest.raises(WeightsError):
        read_safetensors_header(path)


def test_header_whose_byte_ranges_disagree_with_the_shapes_is_rejected(tmp_path):
    header = {"w": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 20]}}  # needs 24 bytes
    blob = json.dumps(header).encode()
    path = tmp_path / "model.safetensors"
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(20))

    with pytest.raises(WeightsError, match="w"):
        read_safetensors_header(path)


def test_header_reports_dtype_shape_and_size(tmp_path):
    path = tmp_path / "x.safetensors"
    mx.save_safetensors(str(path), {"a": mx.zeros((3, 5), dtype=mx.bfloat16), "b": mx.zeros((7,), dtype=mx.float32)})

    header = read_safetensors_header(path)

    assert (header["a"].dtype, header["a"].shape, header["a"].nbytes) == ("bfloat16", (3, 5), 30)
    assert (header["b"].dtype, header["b"].shape, header["b"].nbytes) == ("float32", (7,), 28)
