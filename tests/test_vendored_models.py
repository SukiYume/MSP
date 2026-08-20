import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

MUSICNET_MODELS = (
    Path(__file__).resolve().parents[1] / "src" / "radiosonify" / "models" / "musicnet"
)
HIFIGAN_MODELS = Path(__file__).resolve().parents[1] / "src" / "radiosonify" / "models" / "hifigan"


def _definitions(filename):
    tree = ast.parse((MUSICNET_MODELS / filename).read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_vendored_runtime_excludes_training_export_and_profiling_dead_code():
    assert {
        "export_layer_weights",
        "export_embed_weights",
        "export_final_weights",
    }.isdisjoint(_definitions("wavenet.py"))
    assert {"ZDiscriminator", "cross_entropy_loss"}.isdisjoint(_definitions("wavenet_models.py"))
    assert "timeit" not in _definitions("utils.py")
    assert {"reset", "softmax_and_sample"}.isdisjoint(_definitions("wavenet_generator.py"))


def test_vendored_inference_policy_is_documented():
    root_policy = (MUSICNET_MODELS.parent / "VENDORED.md").read_text(encoding="utf-8")
    policy = (MUSICNET_MODELS / "VENDORED.md").read_text(encoding="utf-8")
    assert "Ruff" in root_policy
    assert "forward-shape" in root_policy
    assert "license" in root_policy
    assert "inference" in policy
    assert "state-dict" in policy
    assert "LICENSE" in policy


def test_musicnet_carries_the_actual_noncommercial_license_text():
    license_text = (MUSICNET_MODELS / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("Attribution-NonCommercial 4.0 International")
    assert "for NonCommercial purposes only" in license_text
    assert "Section 8 -- Interpretation" in license_text


def test_wavenet_forward_preserves_batch_time_and_emits_256_classes():
    torch = pytest.importorskip("torch", reason="requires the musicnet optional dependency")
    from radiosonify.models.musicnet.wavenet import WaveNet

    model = WaveNet(
        SimpleNamespace(
            blocks=1,
            layers=2,
            kernel_size=2,
            skip_channels=4,
            residual_channels=4,
            latent_d=2,
        )
    )
    samples = torch.full((2, 32), 128, dtype=torch.long)
    conditioning = torch.zeros((2, 2, 1))

    with torch.no_grad():
        logits = model(samples, conditioning)

    assert logits.shape == (2, 256, 32)


def test_queued_conv_rejects_unsupported_wrapped_modules():
    tree = ast.parse((MUSICNET_MODELS / "wavenet_generator.py").read_text(encoding="utf-8"))
    queued_init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "__init__"
        and any(
            isinstance(parent, ast.ClassDef) and parent.name == "QueuedConv1d"
            for parent in ast.walk(tree)
            if node in getattr(parent, "body", ())
        )
    )

    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "TypeError"
        for node in ast.walk(queued_init)
    )


def test_hifigan_generator_uses_explicit_final_channels_without_extra_reinitialization():
    source = (HIFIGAN_MODELS / "generator.py").read_text(encoding="utf-8")
    definitions = _definitions_for_path(HIFIGAN_MODELS / "generator.py")

    assert "init_weights" not in definitions
    assert "final_channels" in source
    assert "Conv1d(final_channels, 1" in source


def test_hifigan_generator_postpones_pep604_annotations_for_python39():
    source = (HIFIGAN_MODELS / "generator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr) for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _definitions_for_path(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
