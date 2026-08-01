import numpy as np
import pytest

from radiosonify.inputs import DataType, SonificationInput, infer_data_type, parse_data_type


def test_infers_the_two_scientific_input_types():
    assert infer_data_type(np.ones(16)) is DataType.PROFILE
    assert infer_data_type(np.ones((16, 32))) is DataType.DYNAMIC_SPECTRUM


@pytest.mark.parametrize("alias", ["dynamic_spectrum", "dynamic-spectrum", "spectrogram"])
def test_dynamic_spectrum_aliases_are_canonical(alias):
    assert parse_data_type(alias) is DataType.DYNAMIC_SPECTRUM


def test_input_stores_float_data_duration_and_name():
    source = SonificationInput([1, 2, 3], duration=0.25, name="  candidate  ")

    assert source.data.dtype == np.float64
    assert source.duration == pytest.approx(0.25)
    assert source.data_type is DataType.PROFILE
    assert source.name == "candidate"


def test_input_copies_and_freezes_the_scientific_array():
    original = np.arange(4, dtype=np.float64)
    source = SonificationInput(original, duration=1)
    original[0] = 99

    assert source.data[0] == 0
    with pytest.raises(ValueError, match="read-only"):
        source.data[0] = 5


def test_input_equality_and_hashing_use_identity_without_array_errors():
    first = SonificationInput(np.arange(4), duration=1)
    second = SonificationInput(np.arange(4), duration=1)

    assert first == first
    assert first != second
    assert {first: "first", second: "second"}[first] == "first"


@pytest.mark.parametrize(
    ("data", "data_type", "message"),
    [
        (np.ones((2, 3)), "profile", "requires a 1D"),
        (np.ones(3), "dynamic_spectrum", "requires a 2D"),
        (np.ones((2, 3, 4)), None, "1D profiles and 2D"),
        (np.array([1.0, np.nan]), None, "finite"),
        (np.array([1 + 2j, 3 + 4j]), None, "complex"),
    ],
)
def test_input_rejects_invalid_shape_type_and_values(data, data_type, message):
    with pytest.raises(ValueError, match=message):
        SonificationInput(data, duration=1, data_type=data_type)


@pytest.mark.parametrize("duration", [0, -1, np.inf, True])
def test_input_rejects_invalid_physical_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        SonificationInput(np.ones(4), duration=duration)
