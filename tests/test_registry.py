import inspect

import pytest

from radiosonify.inputs import DataType
from radiosonify.registry import (
    available_methods,
    available_postprocessors,
    default_method,
    resolve_method,
    resolve_postprocessor,
)


def test_profile_and_dynamic_spectrum_get_different_method_sets():
    profile_methods = {item.name for item in available_methods("profile")}
    spectrum_methods = {item.name for item in available_methods("dynamic_spectrum")}

    assert profile_methods == {"profile", "amplitude"}
    assert spectrum_methods == {"profile", "amplitude", "erb", "griffinlim", "hifigan"}
    assert {item.name for item in available_methods("image")} == spectrum_methods
    assert {item.name for item in available_methods("layered_matrix")} == {"spatial_erb"}


def test_defaults_use_transparent_profile_and_full_spectrum_dynamic_method():
    assert default_method(DataType.PROFILE) == "amplitude"
    assert default_method(DataType.MATRIX) == "erb"
    assert default_method(DataType.LAYERED_MATRIX) == "spatial_erb"


def test_method_specs_expose_effective_parameter_names():
    amplitude = resolve_method("amplitude", "profile")
    erb = resolve_method("erb", "matrix")
    griffinlim = resolve_method("griffinlim", "dynamic_spectrum")
    hifigan = resolve_method("hifigan", "dynamic_spectrum")

    assert amplitude.parameters == (
        "sr",
        "freq",
        "compression",
        "harmonics",
        "harmonic_decay",
    )
    assert amplitude.defaults["sr"] == 48_000
    assert amplitude.defaults["compression"] == 0
    assert amplitude.defaults["harmonics"] == 4
    assert amplitude.default_repeat == 5
    assert erb.defaults["frequency_scale"] == "mel"
    assert erb.defaults["min_freq"] == 100
    assert erb.defaults["max_freq"] == 2_000
    assert erb.defaults["n_bands"] is None
    assert erb.defaults["gamma"] == 4
    assert erb.defaults["timbre"] == "sine"
    assert erb.defaults["mapping_level_db"] == 0
    assert erb.defaults["ambient_level_db"] == -30
    assert erb.defaults["voice_params"] is None
    assert erb.defaults["event_voice"] == "none"
    assert erb.defaults["event_params"] is None
    assert erb.defaults["rms_limit_dbfs"] == -20
    assert "foreground_threshold" not in erb.parameters
    assert "max_polyphony" not in erb.parameters
    assert erb.output_peak is None
    assert griffinlim.defaults["n_iter"] == 64
    assert griffinlim.defaults["preemphasis"] == 0
    # 所有改动数据的旋钮都已迁出方法参数，只留下声音映射本身的设置。
    assert "freq_rebin" not in griffinlim.parameters
    assert "time_rebin" not in griffinlim.parameters
    assert "n_mels" not in griffinlim.parameters
    assert hifigan.parameters == ()
    assert griffinlim.default_time_rebin == "auto"
    assert griffinlim.allow_frame_upsampling is True
    assert griffinlim.repeat_frame_overlap == 1
    assert griffinlim.input_feature_bins is None
    assert griffinlim.resolve_feature_geometry({"n_fft": 16}) == (9, 9)
    assert hifigan.default_time_rebin == "auto"
    assert hifigan.allow_frame_upsampling is False
    # The checkpoint's fixed 80-bin encoding belongs to the HiFi-GAN adapter,
    # so the registry keeps the shared-preprocessing width free.
    assert hifigan.input_feature_bins is None
    assert hifigan.resolve_frame_geometry({}) == (22_050, 256)
    assert hifigan.output_peak is None


def test_resolve_method_accepts_public_alias_and_rejects_mismatch():
    assert resolve_method("profile_to_wave", "profile").name == "profile"
    with pytest.raises(ValueError, match="does not accept profile"):
        resolve_method("hifigan", "profile")


def test_registry_is_the_single_source_for_runners_and_capabilities():
    profile = resolve_method("profile", "profile")
    griffinlim = resolve_method("griffinlim", "dynamic_spectrum")
    musicnet = resolve_postprocessor("musicnet")
    rave = resolve_postprocessor("rave")

    assert profile.synthesizes_duration is True
    assert griffinlim.synthesizes_duration is False
    assert profile.output_channels == 1
    assert resolve_method("spatial_erb", "layered_matrix").output_channels == 2
    assert musicnet.max_input_channels == 1
    assert musicnet.preflight_name == "_preflight_musicnet"
    assert profile.load_runner().__name__ == "profile_to_wave"
    assert griffinlim.load_runner().__name__ == "griffinlim"
    assert musicnet.load_runner().__name__ == "musicnet"
    assert available_postprocessors() == (musicnet, rave)


def test_perceptual_runner_defaults_match_the_registry():
    """Duplicated public signatures must stay aligned with unified API provenance."""
    for method, data_type in (("erb", "matrix"), ("spatial_erb", "layered_matrix")):
        spec = resolve_method(method, data_type)
        signature = inspect.signature(spec.load_runner())
        assert {name: signature.parameters[name].default for name in spec.defaults} == dict(
            spec.defaults
        )


def test_unknown_postprocessor_lists_available_choices():
    with pytest.raises(ValueError, match="available: musicnet"):
        resolve_postprocessor("unknown")


def test_grouped_perceptual_defaults_reuse_the_shared_configuration_source():
    from radiosonify._perceptual_config import EVENT_DEFAULTS, VOICE_DEFAULTS

    for name in ("erb", "spatial_erb"):
        spec = resolve_method(name, "matrix" if name == "erb" else "layered_matrix")
        assert spec.grouped_defaults is not None
        # Identity, so a new advanced setting reaches the registry, the CLI and
        # the synthesis engine from one definition.
        assert spec.grouped_defaults["voice_params"] is VOICE_DEFAULTS
        assert spec.grouped_defaults["event_params"] is EVENT_DEFAULTS
        assert set(spec.grouped_defaults) <= set(spec.parameters)

    # Methods without grouped extensions register nothing to expand.
    assert resolve_method("griffinlim", "matrix").grouped_defaults is None
