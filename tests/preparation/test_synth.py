import ast
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from birdclef_2026_take_2.preparation.synth import (
    SAMPLE_RATE,
    Soundscapes,
    TrainClips,
    _SOUNDSCAPE_SITES,
    make_dataset_zip,
    make_ogg_bytes,
    make_soundscapes,
    make_train_clips,
)

SPECIES = ["spc1", "spc2", "spc3"]

TRAIN_CSV_COLUMNS = [
    "primary_label", "secondary_labels", "type", "latitude", "longitude",
    "scientific_name", "common_name", "class_name", "inat_taxon_id",
    "author", "license", "rating", "url", "filename", "collection",
]


# ---------------------------------------------------------------------------
# make_ogg_bytes
# ---------------------------------------------------------------------------


def test_make_ogg_bytes_roundtrip():
    n_samples = SAMPLE_RATE * 7  # 7 seconds
    ogg = make_ogg_bytes(n_samples)
    audio, sr = sf.read(io.BytesIO(ogg))
    assert sr == SAMPLE_RATE
    assert audio.ndim == 1
    assert len(audio) == n_samples


def test_make_ogg_bytes_reproducible():
    # OGG container headers are non-deterministic (stream serial), so compare
    # decoded audio rather than raw bytes.
    ogg_a = make_ogg_bytes(32000, rng=np.random.default_rng(0))
    ogg_b = make_ogg_bytes(32000, rng=np.random.default_rng(0))
    audio_a, _ = sf.read(io.BytesIO(ogg_a))
    audio_b, _ = sf.read(io.BytesIO(ogg_b))
    np.testing.assert_array_equal(audio_a, audio_b)


# ---------------------------------------------------------------------------
# make_train_clips
# ---------------------------------------------------------------------------


def test_make_train_clips_returns_dataclass(train_clips):
    assert isinstance(train_clips, TrainClips)


def test_make_train_clips_clip_count():
    result = make_train_clips(SPECIES, n_clips_per_species=4)
    assert len(result.clips) == len(SPECIES) * 4


def test_make_train_clips_metadata_columns(train_clips):
    assert list(train_clips.metadata.columns) == TRAIN_CSV_COLUMNS


def test_make_train_clips_metadata_row_count(train_clips):
    assert len(train_clips.metadata) == len(SPECIES) * 3


def test_make_train_clips_primary_labels(train_clips):
    assert set(train_clips.metadata["primary_label"]) == set(SPECIES)


def test_make_train_clips_filenames_consistent(train_clips):
    clip_filenames = {fn for fn, _ in train_clips.clips}
    meta_filenames = set(train_clips.metadata["filename"])
    assert clip_filenames == meta_filenames


def test_make_train_clips_filename_prefixes(many_train_clips):
    for fn, _ in many_train_clips.clips:
        sp = fn.split("/")[0]
        stem = fn.split("/")[1]
        assert stem.startswith("XC") or stem.startswith("iNat"), fn
        assert fn.endswith(".ogg")
        assert sp in SPECIES


def test_make_train_clips_collection_matches_filename(many_train_clips):
    for _, row in many_train_clips.metadata.iterrows():
        stem = row["filename"].split("/")[1]
        if stem.startswith("XC"):
            assert row["collection"] == "XC"
        else:
            assert row["collection"] == "iNat"


def test_make_train_clips_xc_has_type(many_train_clips):
    for _, row in many_train_clips.metadata.iterrows():
        if row["collection"] == "XC":
            assert row["type"] != "[]"
        else:
            assert row["type"] == "[]"


def test_make_train_clips_rating_values(many_train_clips):
    valid_ratings = {0.0, 3.0, 3.5, 4.0, 4.5, 5.0}
    assert set(many_train_clips.metadata["rating"]).issubset(valid_ratings)


def test_make_train_clips_species_classes():
    classes = {"spc1": "Aves", "spc2": "Amphibia", "spc3": "Insecta"}
    result = make_train_clips(SPECIES, species_classes=classes)
    for _, row in result.metadata.iterrows():
        assert row["class_name"] == classes[row["primary_label"]]


def test_make_train_clips_secondary_labels_always_present(train_clips):
    for _, row in train_clips.metadata.iterrows():
        assert row["secondary_labels"] != "[]", f"expected secondary labels for {row['filename']}"


def test_make_train_clips_secondary_labels_reference_known_species(train_clips):
    for _, row in train_clips.metadata.iterrows():
        labels = ast.literal_eval(row["secondary_labels"])
        assert isinstance(labels, list)
        assert all(lbl in SPECIES for lbl in labels)
        assert row["primary_label"] not in labels


def test_make_train_clips_ogg_valid():
    result = make_train_clips(["spc1"], n_clips_per_species=1, clip_duration_s=6.0)
    _, ogg = result.clips[0]
    audio, sr = sf.read(io.BytesIO(ogg))
    assert sr == SAMPLE_RATE
    assert len(audio) == int(6.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# make_soundscapes
# ---------------------------------------------------------------------------


def test_make_soundscapes_returns_dataclass(soundscapes_few):
    assert isinstance(soundscapes_few, Soundscapes)


def test_make_soundscapes_recording_count():
    result = make_soundscapes(SPECIES, n_labelled=2, n_unlabelled=3)
    assert len(result.recordings) == 5


def test_make_soundscapes_default_windows_per_file(soundscapes_few):
    assert len(soundscapes_few.labels) == 12


def test_make_soundscapes_labels_columns(soundscapes_few):
    assert {"filename", "start", "end", "primary_label"} <= set(soundscapes_few.labels.columns)


def test_make_soundscapes_unlabelled_not_in_labels():
    result = make_soundscapes(SPECIES, n_labelled=2, n_unlabelled=3)
    labelled_fns = set(result.labels["filename"])
    all_fns = {fn for fn, _ in result.recordings}
    assert len(all_fns - labelled_fns) == 3
    assert len(labelled_fns) == 2


def test_make_soundscapes_labelled_subset_has_all_windows(soundscapes_few):
    for fn in set(soundscapes_few.labels["filename"]):
        assert len(soundscapes_few.labels[soundscapes_few.labels["filename"] == fn]) == 12


def test_make_soundscapes_filename_format(soundscapes_many):
    pattern = re.compile(r"^BC2026_Train_\d{4}_S\d{2}_\d{8}_\d{6}\.ogg$")
    for fn, _ in soundscapes_many.recordings:
        assert pattern.match(fn), fn


def test_make_soundscapes_site_codes(soundscapes_many):
    sites_used = {fn.split("_")[3] for fn, _ in soundscapes_many.recordings}
    assert sites_used <= set(_SOUNDSCAPE_SITES)


def test_make_soundscapes_time_format(soundscapes_few):
    for val in soundscapes_few.labels["start"]:
        parts = val.split(":")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


def test_make_soundscapes_primary_label_is_semicolon_separated(soundscapes_few):
    for val in soundscapes_few.labels["primary_label"]:
        labels = val.split(";")
        assert len(labels) >= 1
        assert all(lbl in SPECIES for lbl in labels)


def test_make_soundscapes_windows_are_5s(soundscapes_few):
    def to_s(t: str) -> int:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    for _, row in soundscapes_few.labels.iterrows():
        assert to_s(row["end"]) - to_s(row["start"]) == 5


def test_make_soundscapes_ogg_valid():
    result = make_soundscapes(SPECIES, n_labelled=1, n_unlabelled=0, soundscape_duration_s=30.0)
    _, ogg = result.recordings[0]
    audio, sr = sf.read(io.BytesIO(ogg))
    assert sr == SAMPLE_RATE
    assert len(audio) == int(30.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# make_dataset_zip
# ---------------------------------------------------------------------------


def test_dataset_zip_exists(dataset_zip: Path):
    assert dataset_zip.exists()
    assert zipfile.is_zipfile(dataset_zip)


def test_dataset_zip_structure(dataset_zip: Path):
    with zipfile.ZipFile(dataset_zip) as zf:
        names = set(zf.namelist())

    assert "train.csv" in names
    assert "train_soundscapes_labels.csv" in names
    assert "taxonomy.csv" in names

    for sp in SPECIES:
        assert any(n.startswith(f"train_audio/{sp}/") and n.endswith(".ogg") for n in names)

    assert any(n.startswith("train_soundscapes/BC2026_Train_") and n.endswith(".ogg") for n in names)


def test_dataset_zip_train_csv(dataset_zip: Path):
    with zipfile.ZipFile(dataset_zip) as zf:
        df = pd.read_csv(io.BytesIO(zf.read("train.csv")))
    assert set(TRAIN_CSV_COLUMNS) <= set(df.columns)
    assert len(df) == len(SPECIES) * 3


def test_dataset_zip_soundscapes_labels(dataset_zip: Path):
    with zipfile.ZipFile(dataset_zip) as zf:
        names = set(zf.namelist())
        df = pd.read_csv(io.BytesIO(zf.read("train_soundscapes_labels.csv")))

    assert {"filename", "start", "end", "primary_label"} <= set(df.columns)
    assert len(df) == 2 * 12  # n_labelled_soundscapes default × windows_per_soundscape default

    soundscape_files = {n.removeprefix("train_soundscapes/") for n in names if n.startswith("train_soundscapes/")}
    labelled_files = set(df["filename"].unique())
    assert len(soundscape_files - labelled_files) == 4   # n_unlabelled_soundscapes default
    assert len(labelled_files) == 2                      # n_labelled_soundscapes default


def test_dataset_zip_taxonomy(dataset_zip: Path):
    with zipfile.ZipFile(dataset_zip) as zf:
        df = pd.read_csv(io.BytesIO(zf.read("taxonomy.csv")))
    assert set(df["primary_label"]) == set(SPECIES)
    assert "class_name" in df.columns
    assert set(df["class_name"]) <= {"Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"}
