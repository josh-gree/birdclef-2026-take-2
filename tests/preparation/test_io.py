"""Tests for zip → TrainClips / Soundscapes loading."""

from pathlib import Path

from birdclef_2026_take_2.preparation.io import (
    read_soundscapes_from_zip,
    read_train_clips_from_zip,
)
from birdclef_2026_take_2.preparation.types import Soundscapes, TrainClips

SPECIES = ["spc1", "spc2", "spc3"]


# ---------------------------------------------------------------------------
# read_train_clips_from_zip
# ---------------------------------------------------------------------------


def test_read_train_clips_returns_dataclass(dataset_zip: Path):
    result = read_train_clips_from_zip(dataset_zip)
    assert isinstance(result, TrainClips)


def test_read_train_clips_clip_count(dataset_zip: Path):
    result = read_train_clips_from_zip(dataset_zip)
    # defaults: 3 species × 3 clips
    assert len(result.clips) == len(SPECIES) * 3


def test_read_train_clips_metadata_row_count(dataset_zip: Path):
    result = read_train_clips_from_zip(dataset_zip)
    assert len(result.metadata) == len(SPECIES) * 3


def test_read_train_clips_filenames_consistent(dataset_zip: Path):
    result = read_train_clips_from_zip(dataset_zip)
    clip_filenames = {fn for fn, _ in result.clips}
    meta_filenames = set(result.metadata["filename"])
    assert clip_filenames == meta_filenames


def test_read_train_clips_metadata_has_primary_label(dataset_zip: Path):
    result = read_train_clips_from_zip(dataset_zip)
    assert "primary_label" in result.metadata.columns
    assert set(result.metadata["primary_label"]) == set(SPECIES)


# ---------------------------------------------------------------------------
# read_soundscapes_from_zip
# ---------------------------------------------------------------------------


def test_read_soundscapes_returns_dataclass(dataset_zip: Path):
    result = read_soundscapes_from_zip(dataset_zip)
    assert isinstance(result, Soundscapes)


def test_read_soundscapes_recording_count(dataset_zip: Path):
    result = read_soundscapes_from_zip(dataset_zip)
    # defaults: 2 labelled + 4 unlabelled
    assert len(result.recordings) == 6


def test_read_soundscapes_labels_row_count(dataset_zip: Path):
    result = read_soundscapes_from_zip(dataset_zip)
    # defaults: 2 labelled × 12 windows
    assert len(result.labels) == 2 * 12


def test_read_soundscapes_unlabelled_not_in_labels(dataset_zip: Path):
    result = read_soundscapes_from_zip(dataset_zip)
    recording_fns = {fn for fn, _ in result.recordings}
    labelled_fns = set(result.labels["filename"])
    assert len(recording_fns - labelled_fns) == 4   # unlabelled
    assert len(labelled_fns) == 2                   # labelled


def test_read_soundscapes_labels_has_required_columns(dataset_zip: Path):
    result = read_soundscapes_from_zip(dataset_zip)
    assert {"filename", "start", "end", "primary_label"} <= set(result.labels.columns)
