import numpy as np
import pytest

from radiosonify.inputs import DataType, SonificationInput, infer_data_type, parse_data_type


def test_infers_profiles_matrices_and_layered_matrices():
    assert infer_data_type(np.ones(16)) is DataType.PROFILE
    assert infer_data_type(np.ones((16, 32))) is DataType.MATRIX
    assert infer_data_type(np.ones((4, 16, 32))) is DataType.LAYERED_MATRIX


@pytest.mark.parametrize("alias", ["dynamic_spectrum", "dynamic-spectrum", "spectrogram"])
def test_dynamic_spectrum_aliases_are_canonical(alias):
    assert parse_data_type(alias) is DataType.MATRIX


@pytest.mark.parametrize("alias", ["matrix", "image", "array-2d"])
def test_generic_matrix_aliases_are_canonical(alias):
    assert parse_data_type(alias) is DataType.MATRIX


@pytest.mark.parametrize("alias", ["layered_matrix", "image-stack", "cube", "iquv"])
def test_layered_matrix_aliases_are_canonical(alias):
    assert parse_data_type(alias) is DataType.LAYERED_MATRIX


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
    with pytest.raises(ValueError):
        source.data.setflags(write=True)


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
        (np.ones((2, 3, 4, 5)), None, "1D profiles, 2D matrices, and 3D"),
        (np.array([1.0, np.inf]), None, "infinite"),
        (np.array([1 + 2j, 3 + 4j]), None, "complex"),
    ],
)
def test_input_rejects_invalid_shape_type_and_values(data, data_type, message):
    with pytest.raises(ValueError, match=message):
        SonificationInput(data, duration=1, data_type=data_type)


def test_input_accepts_nan_and_defers_the_decision_to_preprocessing():
    """掩通道的 NaN 是真实科学输入；接受与否由 preprocess 的 nan_policy 决定。"""
    source = SonificationInput(np.array([1.0, np.nan, 3.0]), duration=1)

    assert np.isnan(source.data[1])


@pytest.mark.parametrize(
    ("shape", "kwargs", "expected"),
    [
        ((4, 6), {"time_axis": 1}, (6, 4)),
        ((4, 6), {"time_axis": 0}, (4, 6)),
        ((4, 6), {}, (4, 6)),
        ((2, 5, 7), {}, (2, 5, 7)),
        ((5, 7, 2), {"layer_axis": 2}, (2, 5, 7)),
        ((5, 2, 7), {"layer_axis": 1}, (2, 5, 7)),
        ((7, 2, 5), {"layer_axis": 1, "time_axis": 2}, (2, 5, 7)),
    ],
)
def test_input_transposes_into_the_standard_layout(shape, kwargs, expected):
    """轴语义在输入阶段解决，预处理和方法之后只见标准布局。"""
    source = SonificationInput(np.zeros(shape), duration=1, **kwargs)

    assert source.data.shape == expected


def test_input_records_original_shape_and_resolved_source_axes():
    source = SonificationInput(
        np.zeros((7, 2, 5)),
        duration=1,
        layer_axis=1,
        time_axis=2,
    )

    assert source.input_shape == (7, 2, 5)
    assert source.source_time_axis == 2
    assert source.source_layer_axis == 1
    assert source.data.shape == (2, 5, 7)
    assert source.time_axis == 1
    assert source.layer_axis == 0


@pytest.mark.parametrize(
    ("shape", "kwargs", "message"),
    [
        ((8,), {"time_axis": 1}, "do not apply to 1D"),
        ((8,), {"time_axis": False}, "do not apply to 1D"),
        ((8,), {"layer_axis": 0}, "do not apply to 1D"),
        ((4, 6), {"layer_axis": 1}, "only applies to 3D"),
        ((4, 6), {"time_axis": 2}, "time_axis must be between"),
        ((2, 5, 7), {"layer_axis": 1, "time_axis": 1}, "must refer to different axes"),
    ],
)
def test_input_rejects_inconsistent_axis_declarations(shape, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SonificationInput(np.zeros(shape), duration=1, **kwargs)


@pytest.mark.parametrize("duration", [0, -1, np.inf, True])
def test_input_rejects_invalid_physical_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        SonificationInput(np.ones(4), duration=duration)
