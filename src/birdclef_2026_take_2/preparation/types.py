"""Canonical in-memory representations of the dataset, regardless of source."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrainClips:
    """Single-label training clips with their metadata.

    Attributes
    ----------
    clips:
        List of ``(filename, ogg_bytes)`` pairs. ``filename`` matches the
        ``filename`` column in ``metadata`` and the ``train_audio/`` zip path.
    metadata:
        DataFrame with the same columns as the real ``train.csv``.
    """

    clips: list[tuple[str, bytes]]
    metadata: pd.DataFrame


@dataclass
class Soundscapes:
    """Soundscape recordings with their windowed labels.

    Attributes
    ----------
    recordings:
        List of ``(filename, ogg_bytes)`` pairs. ``filename`` matches the
        ``filename`` column in ``labels`` and the ``train_soundscapes/`` zip path.
    labels:
        DataFrame with columns ``filename``, ``start``, ``end``,
        ``primary_label`` — the content of ``train_soundscapes_labels.csv``.
        One row per 5-second labeled window. Unlabelled recordings have no
        rows here but still appear in ``recordings``.
    """

    recordings: list[tuple[str, bytes]]
    labels: pd.DataFrame
