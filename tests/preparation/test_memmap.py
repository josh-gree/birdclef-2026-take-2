"""Tests for oggs_to_memmap."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from birdclef_2026_take_2.preparation.memmap import SAMPLE_RATE, oggs_to_memmap
from birdclef_2026_take_2.preparation.synth import make_ogg_bytes


@pytest.fixture(scope="module")
def memmap_result(train_clips, tmp_path_factory):
    path = tmp_path_factory.mktemp("memmap") / "audio.npy"
    offsets = oggs_to_memmap(train_clips.clips, path)
    return path, offsets


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_output_file_is_int16(memmap_result):
    path, _ = memmap_result
    mm = np.load(path, mmap_mode="r")
    assert mm.dtype == np.int16


def test_offsets_dataframe_columns(memmap_result):
    _, offsets = memmap_result
    assert list(offsets.columns) == ["filename", "offset_start", "offset_end"]


def test_offsets_row_count(memmap_result, train_clips):
    _, offsets = memmap_result
    assert len(offsets) == len(train_clips.clips)


def test_offsets_filenames_match_input(memmap_result, train_clips):
    _, offsets = memmap_result
    assert set(offsets["filename"]) == {fn for fn, _ in train_clips.clips}


def test_offsets_are_int64(memmap_result):
    _, offsets = memmap_result
    assert offsets["offset_start"].dtype == np.int64
    assert offsets["offset_end"].dtype == np.int64


# ---------------------------------------------------------------------------
# Offset correctness
# ---------------------------------------------------------------------------


def test_offsets_non_overlapping_and_ordered(memmap_result):
    _, offsets = memmap_result
    for i in range(len(offsets) - 1):
        assert offsets.iloc[i]["offset_end"] == offsets.iloc[i + 1]["offset_start"]


def test_total_length_matches_last_offset(memmap_result):
    path, offsets = memmap_result
    mm = np.load(path, mmap_mode="r")
    assert offsets["offset_end"].iloc[-1] <= len(mm)


def test_all_clips_have_positive_length(memmap_result):
    _, offsets = memmap_result
    assert (offsets["offset_end"] > offsets["offset_start"]).all()


# ---------------------------------------------------------------------------
# Round-trip: slice recovers original audio
# ---------------------------------------------------------------------------


def test_slice_recovers_original_audio(memmap_result, train_clips):
    path, offsets = memmap_result
    mm = np.load(path, mmap_mode="r")

    # Check the first clip
    fn, ogg_bytes = train_clips.clips[0]
    row = offsets[offsets["filename"] == fn].iloc[0]
    stored = mm[row["offset_start"] : row["offset_end"]]

    expected, _ = sf.read(io.BytesIO(ogg_bytes), dtype="float32")
    expected_int16 = (expected * 32767).astype(np.int16)

    np.testing.assert_array_equal(stored, expected_int16)


# ---------------------------------------------------------------------------
# min_duration_s filtering
# ---------------------------------------------------------------------------


def test_min_duration_filters_short_clips(tmp_path):
    long_ogg = make_ogg_bytes(SAMPLE_RATE * 10)   # 10 s
    short_ogg = make_ogg_bytes(SAMPLE_RATE * 2)   # 2 s

    named = [("long.ogg", long_ogg), ("short.ogg", short_ogg)]
    offsets = oggs_to_memmap(named, tmp_path / "audio.npy", min_duration_s=5.0)

    assert len(offsets) == 1
    assert offsets.iloc[0]["filename"] == "long.ogg"


def test_min_duration_all_filtered_returns_empty(tmp_path):
    short_ogg = make_ogg_bytes(SAMPLE_RATE * 2)
    named = [("a.ogg", short_ogg), ("b.ogg", short_ogg)]
    offsets = oggs_to_memmap(named, tmp_path / "audio.npy", min_duration_s=5.0)

    assert len(offsets) == 0
    assert list(offsets.columns) == ["filename", "offset_start", "offset_end"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_dataframe(tmp_path):
    offsets = oggs_to_memmap([], tmp_path / "audio.npy")
    assert len(offsets) == 0
    assert list(offsets.columns) == ["filename", "offset_start", "offset_end"]


def test_empty_input_writes_zero_length_memmap(tmp_path):
    path = tmp_path / "audio.npy"
    oggs_to_memmap([], path)
    mm = np.load(path, mmap_mode="r")
    assert len(mm) == 0
    assert mm.dtype == np.int16
