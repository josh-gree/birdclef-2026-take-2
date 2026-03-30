"""End-to-end data preparation pipeline: zip → memmap + parquet indexes."""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

from birdclef_2026_take_2.preparation.index import build_soundscape_index, build_train_index
from birdclef_2026_take_2.preparation.io import read_soundscapes_from_zip, read_train_clips_from_zip
from birdclef_2026_take_2.preparation.memmap import oggs_to_memmap

log = logging.getLogger(__name__)


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
    taxonomy: Path


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
    log.info("Output directory: %s", output_dir)

    # --- single-label clips ---
    log.info("Reading train clips from %s …", zip_path)
    train_clips = read_train_clips_from_zip(zip_path)
    log.info("Loaded %d train clips (%d metadata rows)", len(train_clips.clips), len(train_clips.metadata))

    train_memmap_path = output_dir / "train.npy"
    log.info("Building train memmap → %s …", train_memmap_path)
    train_offsets = oggs_to_memmap(train_clips.clips, train_memmap_path)
    log.info("Train memmap written: %d files, %d rows in offsets index", len(train_clips.clips), len(train_offsets))

    train_index = build_train_index(train_offsets, train_clips.metadata)
    train_index_path = output_dir / "train_index.parquet"
    train_index.to_parquet(train_index_path, index=False)
    log.info("Train index written → %s (%d rows)", train_index_path, len(train_index))

    # --- soundscapes ---
    log.info("Reading soundscapes from %s …", zip_path)
    soundscapes = read_soundscapes_from_zip(zip_path)
    log.info(
        "Loaded %d soundscape recordings (%d label rows)",
        len(soundscapes.recordings),
        len(soundscapes.labels),
    )

    soundscapes_memmap_path = output_dir / "soundscapes.npy"
    log.info("Building soundscapes memmap → %s …", soundscapes_memmap_path)
    soundscape_offsets = oggs_to_memmap(soundscapes.recordings, soundscapes_memmap_path)
    log.info(
        "Soundscapes memmap written: %d files, %d rows in offsets index",
        len(soundscapes.recordings),
        len(soundscape_offsets),
    )

    soundscapes_index = build_soundscape_index(soundscape_offsets, soundscapes.labels)
    soundscapes_index_path = output_dir / "soundscapes_index.parquet"
    soundscapes_index.to_parquet(soundscapes_index_path, index=False)
    log.info("Soundscapes index written → %s (%d rows)", soundscapes_index_path, len(soundscapes_index))

    taxonomy_path = output_dir / "taxonomy.csv"
    with zipfile.ZipFile(zip_path) as zf:
        taxonomy_path.write_bytes(zf.read("taxonomy.csv"))
    log.info("Taxonomy written → %s", taxonomy_path)

    return PreparedDataset(
        train_memmap=train_memmap_path,
        train_index=train_index_path,
        soundscapes_memmap=soundscapes_memmap_path,
        soundscapes_index=soundscapes_index_path,
        taxonomy=taxonomy_path,
    )
