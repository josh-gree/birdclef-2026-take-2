"""Build labelled sample-level index DataFrames from memmap offsets."""

import pandas as pd

from birdclef_2026_take_2.preparation.memmap import SAMPLE_RATE


def _parse_time(hms: str) -> int:
    """Parse HH:MM:SS string to integer seconds."""
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def build_train_index(
    offsets: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join file-level offsets with clip metadata.

    Parameters
    ----------
    offsets:
        DataFrame with columns ``filename``, ``offset_start``, ``offset_end``
        as returned by :func:`oggs_to_memmap`.
    metadata:
        DataFrame with a ``filename`` column and any additional columns
        (e.g. ``train_clips.metadata``).

    Returns
    -------
    pd.DataFrame
        One row per clip with all metadata columns alongside
        ``offset_start`` / ``offset_end``.
    """
    return offsets.merge(metadata, on="filename", how="inner")


def build_soundscape_index(
    offsets: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> pd.DataFrame:
    """Build a window-level index for soundscape recordings.

    Joins file-level offsets with per-window labels, then recomputes
    ``offset_start`` / ``offset_end`` at sample resolution for each 5-second
    window.  Unlabelled recordings (absent from ``labels``) drop out via the
    inner join.

    Parameters
    ----------
    offsets:
        DataFrame with columns ``filename``, ``offset_start``, ``offset_end``
        as returned by :func:`oggs_to_memmap`.
    labels:
        DataFrame with columns ``filename``, ``start``, ``end`` (HH:MM:SS),
        ``primary_label`` (e.g. ``soundscapes.labels``).
    sample_rate:
        Sample rate used to convert window times to sample offsets.

    Returns
    -------
    pd.DataFrame
        Columns: ``filename``, ``offset_start``, ``offset_end``,
        ``primary_label``.  One row per labelled window; offsets are
        sample-level within the concatenated memmap.
    """
    merged = offsets.merge(labels, on="filename", how="inner")
    file_offset = merged["offset_start"]
    merged = merged.copy()
    merged["offset_start"] = file_offset + merged["start"].map(_parse_time) * sample_rate
    merged["offset_end"] = file_offset + merged["end"].map(_parse_time) * sample_rate
    return merged[["filename", "offset_start", "offset_end", "primary_label"]].reset_index(drop=True)
