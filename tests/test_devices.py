"""Device selection: what `--device` resolves to, and how a missing MLX install is reported.

Needs neither weights nor a GPU: the availability checks are replaced, so every ordering is
exercised on any machine, CI's Linux runners included.
"""

from __future__ import annotations

import pytest
import torch

from strands_decider import cli, infer


@pytest.mark.parametrize(
    ("cuda", "mlx", "mps", "expected"),
    [
        (True, True, True, "cuda"),
        (False, True, True, "mps"),  # mlx is opt-in: an installed extra does not change the default
        (False, True, False, "cpu"),
        (False, False, False, "cpu"),
    ],
)
def test_auto_device_is_cuda_then_mps_then_cpu_and_never_mlx(monkeypatch, cuda, mlx, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(infer, "mlx_available", lambda: mlx)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert cli._auto_device() == expected


def test_mlx_is_unavailable_off_apple_silicon(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert not infer.mlx_available()


@pytest.mark.parametrize(
    ("found", "expected"),
    [
        ({"mlx.core", "mlx_lm"}, True),
        ({"mlx_lm"}, False),  # an `mlx` namespace directory left behind without mlx.core
        (None, False),  # no `mlx` at all: finding `mlx.core` raises
    ],
    ids=["installed", "namespace-leftover", "absent"],
)
def test_mlx_is_available_only_with_mlx_core(monkeypatch, found, expected):
    import importlib.util

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")

    def find_spec(name, *args, **kwargs):
        if found is None:
            raise ModuleNotFoundError(f"No module named {name.split('.')[0]!r}")
        return object() if name in found else None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert infer.mlx_available() is expected


def test_device_mlx_without_the_extra_names_it_before_loading_anything(monkeypatch):
    monkeypatch.setattr(infer, "mlx_available", lambda: False)

    def refuse(*args, **kwargs):
        raise AssertionError("nothing should load")

    monkeypatch.setattr(infer.StrandsDeciderModel, "load", refuse)
    with pytest.raises(RuntimeError, match=r"strands-decider\[mlx\]"):
        infer.load_engine("any/checkpoint", device="mlx")


def test_the_server_builds_the_mlx_engine_for_device_mlx(monkeypatch):
    from fastapi.testclient import TestClient

    from strands_decider import server

    class _Engine:
        cfg = type("Cfg", (), {"model_name": "hobson", "device": "mlx", "use_prefix_cache": True})()
        model = type(
            "Model",
            (),
            {
                "config": type(
                    "Config", (), {"base_model": "stub", "temperature": 1.0, "max_length": 512}
                )()
            },
        )()

    calls = []
    monkeypatch.setattr(
        server,
        "load_mlx",
        lambda checkpoint, config: calls.append((checkpoint, config)) or _Engine(),
    )
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("torch load")
    )
    # No warm-up: the stub engine only answers /health.
    app = server.create_app("org/hobson", device="mlx", max_batch=7, warmup=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["device"] == "mlx"
    [(checkpoint, config)] = calls
    assert checkpoint == "org/hobson"
    assert (config.device, config.use_prefix_cache, config.model_name) == ("mlx", True, "hobson")
    assert config.max_batch == 7  # the shared config reaches MLX


def test_neuron_is_never_auto_selected(monkeypatch):
    from strands_decider import neuron_engine

    monkeypatch.setattr(neuron_engine, "neuron_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert cli._auto_device() == "cpu"


@pytest.mark.parametrize(
    ("system", "device_file", "found", "expected"),
    [
        ("Linux", True, {"libtorch_neuronx_lite", "nki", "nkilib"}, True),
        ("Linux", True, {"libtorch_neuronx_lite", "nki"}, False),  # nkilib: the container only
        ("Linux", False, {"libtorch_neuronx_lite", "nki", "nkilib"}, False),  # no Trainium
        ("Darwin", True, {"libtorch_neuronx_lite", "nki", "nkilib"}, False),
    ],
    ids=["installed", "no-nkilib", "no-device", "not-linux"],
)
def test_neuron_is_available_only_with_a_device_and_the_sdk(
    monkeypatch, system, device_file, found, expected
):
    import importlib.util
    import os

    from strands_decider import neuron_engine

    monkeypatch.setattr("platform.system", lambda: system)
    real_exists = os.path.exists
    monkeypatch.setattr(
        os.path, "exists", lambda p: device_file if p == "/dev/neuron0" else real_exists(p)
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: object() if name in found else None
    )
    assert neuron_engine.neuron_available() is expected


def test_device_neuron_without_the_sdk_names_the_install_before_loading_anything(monkeypatch):
    from strands_decider import neuron_engine

    monkeypatch.setattr(neuron_engine, "neuron_available", lambda: False)
    monkeypatch.setattr(
        infer.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("nothing should load")
    )
    with pytest.raises(RuntimeError, match=r"strands-decider\[neuron\].*pip\.repos\.neuron"):
        infer.load_engine("any/checkpoint", device="neuron")


def test_the_server_builds_the_neuron_engine_for_device_neuron(monkeypatch):
    from fastapi.testclient import TestClient

    from strands_decider import server

    class _Engine:
        cfg = type(
            "Cfg", (), {"model_name": "decider", "device": "neuron", "use_prefix_cache": True}
        )()
        model = type(
            "Model",
            (),
            {
                "config": type(
                    "Config", (), {"base_model": "stub", "temperature": 1.0, "max_length": 512}
                )()
            },
        )()

    calls = []
    monkeypatch.setattr(
        server,
        "load_neuron",
        lambda checkpoint, config: calls.append((checkpoint, config)) or _Engine(),
    )
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", lambda *a, **k: pytest.fail("torch load")
    )
    app = server.create_app("org/decider", device="neuron", max_batch=5, warmup=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["device"] == "neuron"
    [(checkpoint, config)] = calls
    assert (checkpoint, config.device, config.max_batch) == ("org/decider", "neuron", 5)


@pytest.mark.parametrize("device", ["neuron", "mlx"])
def test_vision_is_refused_off_torch_devices(device):
    from strands_decider import server

    with pytest.raises(ValueError, match=f"not {device}"):
        server.create_app("org/decider", device=device, vision=True, warmup=False)
