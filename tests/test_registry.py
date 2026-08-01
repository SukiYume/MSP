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
    assert spectrum_methods == {"profile", "amplitude", "griffinlim", "hifigan"}


def test_defaults_use_transparent_profile_and_full_spectrum_dynamic_method():
    assert default_method(DataType.PROFILE) == "amplitude"
    assert default_method(DataType.DYNAMIC_SPECTRUM) == "griffinlim"


def test_method_specs_expose_effective_parameter_names():
    amplitude = resolve_method("amplitude", "profile")
    griffinlim = resolve_method("griffinlim", "dynamic_spectrum")
    hifigan = resolve_method("hifigan", "dynamic_spectrum")

    assert amplitude.parameters == (
        "sr",
        "freq",
        "compression",
        "time_downsample",
    )
    assert amplitude.defaults["sr"] == 48_000
    assert amplitude.defaults["compression"] == 99
    assert griffinlim.defaults["n_iter"] == 64
    assert griffinlim.defaults["preemphasis"] == 0
    assert "freq_rebin" in griffinlim.parameters
    assert "n_mels" not in griffinlim.parameters
    assert "time_smoothing" in hifigan.parameters


def test_resolve_method_accepts_public_alias_and_rejects_mismatch():
    assert resolve_method("profile_to_wave", "profile").name == "profile"
    with pytest.raises(ValueError, match="does not accept profile"):
        resolve_method("hifigan", "profile")


def test_registry_is_the_single_source_for_runners_and_capabilities():
    profile = resolve_method("profile", "profile")
    griffinlim = resolve_method("griffinlim", "dynamic_spectrum")
    musicnet = resolve_postprocessor("musicnet")

    assert profile.supports_repeat is True
    assert profile.synthesizes_duration is True
    assert griffinlim.supports_repeat is False
    assert griffinlim.synthesizes_duration is False
    assert profile.load_runner().__name__ == "profile_to_wave"
    assert griffinlim.load_runner().__name__ == "griffinlim"
    assert musicnet.load_runner().__name__ == "musicnet"
    assert available_postprocessors() == (musicnet,)


def test_unknown_postprocessor_lists_available_choices():
    with pytest.raises(ValueError, match="available: musicnet"):
        resolve_postprocessor("unknown")
