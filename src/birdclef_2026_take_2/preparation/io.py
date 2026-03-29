"""Load competition zip files into canonical in-memory representations."""

import io
import zipfile
from pathlib import Path

import pandas as pd

from birdclef_2026_take_2.preparation.types import Soundscapes, TrainClips


def read_train_clips_from_zip(zip_path: Path) -> TrainClips:
    """Load train audio clips and metadata from the competition zip.

    Parameters
    ----------
    zip_path:
        Path to the competition zip file.

    Returns
    -------
    TrainClips
        ``clips`` contains all ``train_audio/*.ogg`` entries as
        ``(filename, bytes)`` pairs where ``filename`` matches the
        ``filename`` column in ``train.csv``.
        ``metadata`` is the full ``train.csv`` DataFrame.
    """
    with zipfile.ZipFile(zip_path) as zf:
        metadata = pd.read_csv(io.BytesIO(zf.read("train.csv")))
        clips = [
            (name.removeprefix("train_audio/"), zf.read(name))
            for name in zf.namelist()
            if name.startswith("train_audio/") and name.endswith(".ogg")
        ]
    return TrainClips(clips=clips, metadata=metadata)


def read_soundscapes_from_zip(zip_path: Path) -> Soundscapes:
    """Load soundscape recordings and labels from the competition zip.

    All soundscape OGG files are included in ``recordings``, but only
    the labelled subset has rows in ``labels`` — matching the real
    competition where most soundscapes are unlabelled.

    Parameters
    ----------
    zip_path:
        Path to the competition zip file.

    Returns
    -------
    Soundscapes
        ``recordings`` contains all ``train_soundscapes/*.ogg`` entries as
        ``(filename, bytes)`` pairs where ``filename`` matches the
        ``filename`` column in ``train_soundscapes_labels.csv``.
        ``labels`` is the full ``train_soundscapes_labels.csv`` DataFrame.
    """
    with zipfile.ZipFile(zip_path) as zf:
        labels = pd.read_csv(io.BytesIO(zf.read("train_soundscapes_labels.csv")))
        recordings = [
            (name.removeprefix("train_soundscapes/"), zf.read(name))
            for name in zf.namelist()
            if name.startswith("train_soundscapes/") and name.endswith(".ogg")
        ]
    return Soundscapes(recordings=recordings, labels=labels)
