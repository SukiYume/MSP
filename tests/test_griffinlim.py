import importlib
import inspect

import numpy as np
import pytest

from radiosonify.griffinlim import griffinlim

griffinlim_module = importlib.import_module("radiosonify.griffinlim")
validation_module = importlib.import_module("radiosonify.validation")

LEGACY_GRIFFIN_TIME_BINS = 128
LEGACY_GRIFFIN_FREQ_BINS = 512


class TestGriffinLim:
    def test_default_iteration_count_avoids_the_convergence_plateau(self):
        parameters = inspect.signature(griffinlim).parameters
        assert parameters["n_iter"].default == 64
        assert parameters["preemphasis"].default == 0

    def test_rejects_bad_output_path_before_spectrogram_preparation(self, monkeypatch):
        monkeypatch.setattr(
            griffinlim_module,
            "_prepare_spectrogram",
            lambda *args, **kwargs: pytest.fail("spectrogram preparation must not run"),
        )

        with pytest.raises(ValueError, match=".wav"):
            griffinlim(np.ones((8, 8)), output="bad.flac")

    def test_preprocessing_scans_input_finiteness_once(self, monkeypatch):
        calls = []
        original_isfinite = validation_module.np.isfinite

        def tracked_isfinite(value):
            calls.append(np.asarray(value).shape)
            return original_isfinite(value)

        monkeypatch.setattr(validation_module.np, "isfinite", tracked_isfinite)
        prepared = griffinlim_module._prepare_spectrogram(
            np.arange(256, dtype=np.float64).reshape(16, 16) / 255,
            n_fft=32,
        )

        assert prepared.shape == (16, 16)
        assert calls == [(16, 16)]

    def test_returns_tuple(self):
        rng = np.random.default_rng(42)
        spec = rng.random((LEGACY_GRIFFIN_TIME_BINS, LEGACY_GRIFFIN_FREQ_BINS))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
        )
        assert isinstance(audio, np.ndarray)
        assert sr == 48000

    def test_output_is_1d(self):
        rng = np.random.default_rng(42)
        spec = rng.random((LEGACY_GRIFFIN_TIME_BINS, LEGACY_GRIFFIN_FREQ_BINS))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
        )
        assert audio.ndim == 1
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 1e-6
        assert np.max(np.abs(audio)) <= 0.95 + 1e-6

    def test_rejects_1d(self):
        with pytest.raises(ValueError):
            griffinlim(np.ones(100), sr=48000)

    def test_auto_rebin_freq(self):
        rng = np.random.default_rng(42)
        spec = rng.random((LEGACY_GRIFFIN_TIME_BINS, LEGACY_GRIFFIN_FREQ_BINS))
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
        )
        assert audio.ndim == 1

    def test_saves_to_file(self, tmp_path):
        rng = np.random.default_rng(42)
        spec = rng.random((LEGACY_GRIFFIN_TIME_BINS, LEGACY_GRIFFIN_FREQ_BINS))
        out = tmp_path / "out.wav"
        audio, sr = griffinlim(
            spec,
            sr=48000,
            n_iter=10,
            output=str(out),
        )
        assert out.exists()

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_iter": 0}, "n_iter"),
            ({"frame_length": 0}, "frame_length"),
            ({"preemphasis": 1.0}, "preemphasis"),
            ({"max_db": 10, "ref_db": 20}, "ref_db"),
            ({"n_fft": 64, "frame_length": 0.04}, "longer than n_fft"),
            ({"frame_length": "0.04"}, "frame_length"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            griffinlim(np.ones((16, 16)), **kwargs)

    def test_rejects_non_finite_input(self):
        data = np.ones((16, 16))
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            griffinlim(data)

    def test_rebinning_knobs_left_the_method_signature(self):
        """时间/频率重分箱只属于统一预处理，方法层不再接受这些参数。"""
        parameters = inspect.signature(griffinlim).parameters

        assert "time_rebin" not in parameters
        assert "freq_rebin" not in parameters
        assert "n_mels" not in parameters

    def test_frame_geometry_matches_the_synthesis_hop(self):
        """注册表用它把目标时长换算成输入帧数，必须与实际合成的跳距一致。"""
        sample_rate, hop_length = griffinlim_module._frame_geometry(
            {"sr": 48_000, "frame_length": 0.04}
        )

        assert sample_rate == 48_000
        assert hop_length == int(48_000 * 0.04) // 4

    def test_keeps_quiet_edges_that_encode_event_time(self, monkeypatch):
        native = np.zeros(40, dtype=np.float64)
        native[12:18] = 1
        monkeypatch.setattr(
            griffinlim_module,
            "_griffin_lim",
            lambda *args, **kwargs: native.copy(),
        )

        audio, _ = griffinlim(
            np.arange(64, dtype=np.float64).reshape(8, 8) / 63,
            sr=100,
            n_iter=1,
            n_fft=16,
            frame_length=0.08,
            preemphasis=0,
        )

        assert len(audio) == len(native)
        assert np.all(audio[:12] == 0)
        assert np.all(audio[18:] == 0)

    def test_rejects_data_that_skipped_shared_preprocessing(self):
        with pytest.raises(ValueError, match="preprocess"):
            griffinlim(np.arange(16.0).reshape(4, 4))
