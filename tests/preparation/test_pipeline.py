import numpy as np
import pandas as pd
import pytest

from birdclef_2026_take_2.preparation.pipeline import PreparedDataset, prepare_dataset
from birdclef_2026_take_2.preparation.synth import SAMPLE_RATE

# make_dataset_zip defaults: 3 species * 3 clips/species = 9 train clips,
# 2 labelled soundscapes * 12 windows = 24 soundscape index rows.
EXPECTED_TRAIN_ROWS = 9
EXPECTED_SOUNDSCAPE_ROWS = 24


@pytest.fixture(scope="module")
def prepared(dataset_zip, tmp_path_factory):
    out = tmp_path_factory.mktemp("prepared")
    return prepare_dataset(dataset_zip, out)


def test_returns_prepared_dataset(prepared):
    assert isinstance(prepared, PreparedDataset)


def test_all_output_files_exist(prepared):
    assert prepared.train_memmap.exists()
    assert prepared.train_index.exists()
    assert prepared.soundscapes_memmap.exists()
    assert prepared.soundscapes_index.exists()


def test_train_memmap_is_int16(prepared):
    arr = np.load(prepared.train_memmap, mmap_mode="r")
    assert arr.dtype == np.int16
    assert arr.ndim == 1
    assert len(arr) > 0


def test_soundscapes_memmap_is_int16(prepared):
    arr = np.load(prepared.soundscapes_memmap, mmap_mode="r")
    assert arr.dtype == np.int16
    assert arr.ndim == 1
    assert len(arr) > 0


def test_train_index_columns(prepared):
    df = pd.read_parquet(prepared.train_index)
    assert {"filename", "offset_start", "offset_end", "primary_label"}.issubset(df.columns)


def test_train_index_row_count(prepared):
    df = pd.read_parquet(prepared.train_index)
    assert len(df) == EXPECTED_TRAIN_ROWS


def test_train_index_offsets_valid(prepared):
    df = pd.read_parquet(prepared.train_index)
    assert (df["offset_start"] >= 0).all()
    assert (df["offset_end"] > df["offset_start"]).all()


def test_soundscapes_index_columns(prepared):
    df = pd.read_parquet(prepared.soundscapes_index)
    assert set(df.columns) == {"filename", "offset_start", "offset_end", "primary_label"}


def test_soundscapes_index_row_count(prepared):
    df = pd.read_parquet(prepared.soundscapes_index)
    assert len(df) == EXPECTED_SOUNDSCAPE_ROWS


def test_soundscapes_index_window_length(prepared):
    df = pd.read_parquet(prepared.soundscapes_index)
    assert ((df["offset_end"] - df["offset_start"]) == 5 * SAMPLE_RATE).all()


def test_soundscapes_index_offsets_valid(prepared):
    df = pd.read_parquet(prepared.soundscapes_index)
    assert (df["offset_start"] >= 0).all()
    assert (df["offset_end"] > df["offset_start"]).all()


def test_output_dir_created_if_absent(dataset_zip, tmp_path):
    out = tmp_path / "nested" / "output"
    assert not out.exists()
    prepare_dataset(dataset_zip, out)
    assert out.exists()
