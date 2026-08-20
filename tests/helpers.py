import numpy as np


def dominant_frequency(audio: np.ndarray, sr: int) -> float:
    """Return the strongest positive-frequency FFT bin for test assertions."""
    frequencies = np.fft.rfftfreq(len(audio), d=1 / sr)
    return float(frequencies[np.argmax(np.abs(np.fft.rfft(audio)))])
