import pytest

from birdclef_2026_take_2.preparation.index import build_soundscape_index, build_train_index
from birdclef_2026_take_2.preparation.memmap import oggs_to_memmap
from birdclef_2026_take_2.preparation.synth import SAMPLE_RATE


@pytest.fixture(scope="module")
def train_offsets(train_clips, tmp_path_factory):
    path = tmp_path_factory.mktemp("train_memmap") / "train.npy"
    return oggs_to_memmap(train_clips.clips, path)


@pytest.fixture(scope="module")
def soundscape_offsets(soundscapes_many, tmp_path_factory):
    path = tmp_path_factory.mktemp("soundscape_memmap") / "soundscapes.npy"
    return oggs_to_memmap(soundscapes_many.recordings, path)


# --- TrainClips index ---


def test_train_index_row_count(train_offsets, train_clips):
    index = build_train_index(train_offsets, train_clips.metadata)
    assert len(index) == len(train_clips.clips)


def test_train_index_columns(train_offsets, train_clips):
    index = build_train_index(train_offsets, train_clips.metadata)
    expected = {"offset_start", "offset_end"} | set(train_clips.metadata.columns)
    assert expected.issubset(set(index.columns))


def test_train_index_offsets_preserved(train_offsets, train_clips):
    index = build_train_index(train_offsets, train_clips.metadata)
    idx = index.set_index("filename")
    src = train_offsets.set_index("filename")
    for fn in src.index:
        assert idx.loc[fn, "offset_start"] == src.loc[fn, "offset_start"]
        assert idx.loc[fn, "offset_end"] == src.loc[fn, "offset_end"]


def test_train_index_primary_labels(train_offsets, train_clips):
    index = build_train_index(train_offsets, train_clips.metadata)
    assert index["primary_label"].notna().all()
    assert (index["primary_label"] != "").all()


# --- Soundscape index ---


def test_soundscape_index_row_count(soundscape_offsets, soundscapes_many):
    index = build_soundscape_index(soundscape_offsets, soundscapes_many.labels)
    assert len(index) == len(soundscapes_many.labels)


def test_soundscape_index_columns(soundscape_offsets, soundscapes_many):
    index = build_soundscape_index(soundscape_offsets, soundscapes_many.labels)
    assert set(index.columns) == {"filename", "offset_start", "offset_end", "primary_label"}


def test_soundscape_index_window_length_in_samples(soundscape_offsets, soundscapes_many):
    index = build_soundscape_index(soundscape_offsets, soundscapes_many.labels)
    lengths = index["offset_end"] - index["offset_start"]
    assert (lengths == 5 * SAMPLE_RATE).all()


def test_soundscape_index_window_offsets_within_file(soundscape_offsets, soundscapes_many):
    index = build_soundscape_index(soundscape_offsets, soundscapes_many.labels)
    file_bounds = soundscape_offsets.set_index("filename")
    for _, row in index.iterrows():
        fb = file_bounds.loc[row["filename"]]
        assert row["offset_start"] >= fb["offset_start"]
        assert row["offset_end"] <= fb["offset_end"]


def test_soundscape_index_unlabelled_files_absent(soundscape_offsets, soundscapes_many):
    index = build_soundscape_index(soundscape_offsets, soundscapes_many.labels)
    labelled = set(soundscapes_many.labels["filename"])
    all_recordings = {fn for fn, _ in soundscapes_many.recordings}
    unlabelled = all_recordings - labelled
    assert len(unlabelled) > 0, "fixture must contain unlabelled recordings"
    assert unlabelled.isdisjoint(set(index["filename"]))
