"""Convert OGG files to a concatenated int16 memmap with an offsets index."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SAMPLE_RATE = 32_000


def _get_num_frames(audio_bytes: bytes) -> int:
    """Return frame count from file header without decoding."""
    return sf.info(io.BytesIO(audio_bytes)).frames


def _decode_to_int16(audio_bytes: bytes) -> np.ndarray:
    """Decode OGG bytes to a mono int16 array."""
    audio, _ = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return (audio * 32767).astype(np.int16)


def oggs_to_memmap(
    named_oggs: list[tuple[str, bytes]],
    output_path: Path,
    *,
    min_duration_s: float | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> pd.DataFrame:
    """Convert a list of named OGG files to a concatenated int16 memmap.

    Parameters
    ----------
    named_oggs:
        List of ``(filename, ogg_bytes)`` pairs. Compatible with
        ``TrainClips.clips`` and ``Soundscapes.recordings``.
    output_path:
        Path to write the ``.npy`` memmap file.
    min_duration_s:
        If set, files shorter than this are excluded from the output.
    sample_rate:
        Used to convert ``min_duration_s`` to a frame count.

    Returns
    -------
    pd.DataFrame
        Columns: ``filename``, ``offset_start``, ``offset_end`` (int64).
        One row per file written. Files dropped by the duration filter are absent.
    """
    if not named_oggs:
        mm = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.int16, shape=(0,)
        )
        mm.flush()
        return pd.DataFrame(columns=["filename", "offset_start", "offset_end"])

    # Pass 1 — count frames from headers (no decoding)
    with_counts = [(fn, b, _get_num_frames(b)) for fn, b in named_oggs]

    # Apply minimum duration filter
    if min_duration_s is not None:
        min_frames = int(min_duration_s * sample_rate)
        with_counts = [(fn, b, fc) for fn, b, fc in with_counts if fc >= min_frames]

    if not with_counts:
        mm = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.int16, shape=(0,)
        )
        mm.flush()
        return pd.DataFrame(columns=["filename", "offset_start", "offset_end"])

    # Allocate memmap — small per-file padding guards against header/decoder discrepancies
    total = sum(fc for _, _, fc in with_counts)
    mm = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.int16, shape=(total + len(with_counts) * 10,)
    )

    # Pass 2 — decode and write
    rows = []
    pos = 0
    for fn, ogg_bytes, _ in with_counts:
        arr = _decode_to_int16(ogg_bytes)
        mm[pos : pos + len(arr)] = arr
        rows.append({"filename": fn, "offset_start": pos, "offset_end": pos + len(arr)})
        pos += len(arr)

    mm.flush()

    offsets = pd.DataFrame(rows)
    offsets["offset_start"] = offsets["offset_start"].astype(np.int64)
    offsets["offset_end"] = offsets["offset_end"].astype(np.int64)
    return offsets
