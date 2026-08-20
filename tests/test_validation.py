import numpy as np
import pytest

from radiosonify.validation import _immutable_array, _merge_settings


def test_strided_input_is_frozen_in_an_immutable_buffer():
    source = np.arange(24.0).reshape(2, 3, 4)
    transposed = np.transpose(source, (1, 2, 0))
    frozen = _immutable_array(transposed, dtype=np.float64)
    np.testing.assert_array_equal(frozen, transposed)
    assert frozen.flags.writeable is False
    root = frozen
    while isinstance(root, np.ndarray):
        root = root.base
    assert isinstance(root, bytes)
    with pytest.raises(ValueError, match="WRITEABLE"):
        frozen.setflags(write=True)
    source[0, 0, 0] = 999.0
    assert frozen[0, 0, 0] == 0.0


def test_immutable_array_rejects_object_dtype():
    with pytest.raises(TypeError, match="object arrays"):
        _immutable_array(np.array([object()], dtype=object))


def test_merge_settings_applies_defaults_without_mutating_input():
    defaults = {"alpha": 1, "beta": 2}
    supplied = {"beta": 9}
    assert _merge_settings(defaults, supplied, field_name="f", unknown_label="u") == {
        "alpha": 1,
        "beta": 9,
    }
    assert supplied == {"beta": 9}


def test_merge_settings_reports_container_keys_and_unknown_values():
    defaults = {"alpha": 1, "beta": 2}
    with pytest.raises(ValueError, match="settings must be a mapping or None"):
        _merge_settings(defaults, [("alpha", 1)], field_name="settings", unknown_label="u")
    with pytest.raises(ValueError, match="settings keys must be strings"):
        _merge_settings(defaults, {1: "x"}, field_name="settings", unknown_label="u")
    with pytest.raises(ValueError, match="bad key: gamma; allowed: alpha, beta"):
        _merge_settings(
            defaults,
            {"gamma": 1},
            field_name="settings",
            unknown_label="bad key",
        )
