"""End-to-end data preparation pipeline: zip → memmap + parquet indexes."""

from dataclasses import dataclass
from pathlib import Path

from birdclef_2026_take_2.preparation.index import build_soundscape_index, build_train_index
from birdclef_2026_take_2.preparation.io import read_soundscapes_from_zip, read_train_clips_from_zip
from birdclef_2026_take_2.preparation.memmap import oggs_to_memmap


@dataclass
class PreparedDataset:
    """Paths to the outputs produced by :func:`prepare_dataset`.

    Attributes
    ----------
    train_memmap:
        Concatenated int16 memmap of all single-label training clips.
    train_index:
        Parquet index with one row per clip: ``filename``, ``offset_start``,
        ``offset_end``, and all ``train.csv`` metadata columns.
    soundscapes_memmap:
        Concatenated int16 memmap of all soundscape recordings.
    soundscapes_index:
        Parquet index with one row per labelled 5-second window:
        ``filename``, ``offset_start``, ``offset_end``, ``primary_label``.
    """

    train_memmap: Path
    train_index: Path
    soundscapes_memmap: Path
    soundscapes_index: Path


def prepare_dataset(zip_path: Path, output_dir: Path) -> PreparedDataset:
    """Convert a competition zip into memmaps and parquet indexes.

    Reads all audio and metadata from ``zip_path``, writes two ``.npy``
    memmaps (one for single-label clips, one for soundscapes) and two
    parquet index files into ``output_dir``.

    Parameters
    ----------
    zip_path:
        Path to the competition zip file (real or synthetic).
    output_dir:
        Directory in which to write all outputs.  Created if absent.

    Returns
    -------
    PreparedDataset
        Paths to the four output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- single-label clips ---
    train_clips = read_train_clips_from_zip(zip_path)
    train_memmap_path = output_dir / "train.npy"
    train_offsets = oggs_to_memmap(train_clips.clips, train_memmap_path)
    train_index = build_train_index(train_offsets, train_clips.metadata)
    train_index_path = output_dir / "train_index.parquet"
    train_index.to_parquet(train_index_path, index=False)

    # --- soundscapes ---
    soundscapes = read_soundscapes_from_zip(zip_path)
    soundscapes_memmap_path = output_dir / "soundscapes.npy"
    soundscape_offsets = oggs_to_memmap(soundscapes.recordings, soundscapes_memmap_path)
    soundscapes_index = build_soundscape_index(soundscape_offsets, soundscapes.labels)
    soundscapes_index_path = output_dir / "soundscapes_index.parquet"
    soundscapes_index.to_parquet(soundscapes_index_path, index=False)

    return PreparedDataset(
        train_memmap=train_memmap_path,
        train_index=train_index_path,
        soundscapes_memmap=soundscapes_memmap_path,
        soundscapes_index=soundscapes_index_path,
    )
