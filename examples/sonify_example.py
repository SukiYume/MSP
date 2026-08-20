"""Run the unified API on a bundled two-dimensional-array example.

The downloaded array contains no time-coordinate metadata, so the listening
duration below is an explicit example calibration rather than an inferred
physical value. Replace it with the measured span of real observations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import radiosonify as rs

EXAMPLE_DURATION_SECONDS = 1.0
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "radiosonify" / "unified_output.wav"


def sonify_example(output: str | Path = DEFAULT_OUTPUT) -> rs.SonificationResult:
    """Download the example array, sonify it, and return full provenance."""
    source = rs.SonificationInput(
        rs.load_example("raw_burst"),
        duration=EXAMPLE_DURATION_SECONDS,
        name="bundled-raw-burst",
    )
    return rs.sonify(
        source,
        method="auto",
        output=output,
    )


if __name__ == "__main__":
    result = sonify_example()
    print(f"Wrote {result.output_path} ({result.output_duration:.3f} s at {result.sample_rate} Hz)")
