"""Tests for TrainClipDataset."""

import numpy as np
import pytest

from birdclef_2026_take_2.dataset import (
    WINDOW_SAMPLES,
    ConsecutiveWindows,
    MiddleWindow,
    RandomWindow,
    TrainClipDataset,
)
from birdclef_2026_take_2.preparation.pipeline import prepare_dataset


@pytest.fixture(scope="module")
def prepared(dataset_zip, tmp_path_factory):
    out = tmp_path_factory.mktemp("prepared")
    return prepare_dataset(dataset_zip, out)


@pytest.fixture(scope="module")
def consecutive_ds(prepared):
    return TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        ConsecutiveWindows(),
        seed=0,
    )


@pytest.fixture(scope="module")
def random_ds(prepared):
    return TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        RandomWindow(),
        seed=0,
    )


@pytest.fixture(scope="module")
def middle_ds(prepared):
    return TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        MiddleWindow(),
        seed=0,
    )


# --- ConsecutiveWindows ---

def test_consecutive_len(prepared, consecutive_ds):
    import pandas as pd
    index = pd.read_parquet(prepared.train_index)
    index = index[(index["offset_end"] - index["offset_start"]) >= WINDOW_SAMPLES]
    expected = sum(
        (row.offset_end - row.offset_start) // WINDOW_SAMPLES
        for row in index.itertuples()
    )
    assert len(consecutive_ds) == expected


def test_consecutive_audio_shape(consecutive_ds):
    for idx in range(len(consecutive_ds)):
        item = consecutive_ds[idx]
        assert item["audio"].shape == (WINDOW_SAMPLES,)
        assert item["audio"].dtype == np.int16


def test_consecutive_labels_valid(prepared, consecutive_ds):
    import pandas as pd
    taxonomy = pd.read_csv(prepared.taxonomy).sort_values("primary_label").reset_index(drop=True)
    n_species = len(taxonomy)
    for idx in range(len(consecutive_ds)):
        label = consecutive_ds[idx]["label"]
        assert 0 <= label < n_species



# --- RandomWindow ---

def test_random_len(random_ds, prepared):
    import pandas as pd
    index = pd.read_parquet(prepared.train_index)
    expected = len(index[(index["offset_end"] - index["offset_start"]) >= WINDOW_SAMPLES])
    assert len(random_ds) == expected


def test_random_audio_shape(random_ds):
    for idx in range(len(random_ds)):
        item = random_ds[idx]
        assert item["audio"].shape == (WINDOW_SAMPLES,)
        assert item["audio"].dtype == np.int16


def test_random_reproducible_same_epoch(prepared):
    """Same dataset, same epoch → same window for each clip."""
    ds = TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        RandomWindow(),
        seed=42,
    )
    for idx in range(len(ds)):
        a = ds[idx]["audio"].copy()
        b = ds[idx]["audio"].copy()
        np.testing.assert_array_equal(a, b)



def test_random_epoch0_reproducible(prepared):
    """Two datasets with the same seed at epoch 0 produce identical outputs."""
    def make():
        return TrainClipDataset(
            prepared.train_memmap,
            prepared.train_index,
            prepared.taxonomy,
            RandomWindow(),
            seed=7,
        )
    ds1, ds2 = make(), make()
    for idx in range(len(ds1)):
        np.testing.assert_array_equal(ds1[idx]["audio"], ds2[idx]["audio"])


# --- MiddleWindow ---

def test_middle_audio_shape(middle_ds):
    for idx in range(len(middle_ds)):
        item = middle_ds[idx]
        assert item["audio"].shape == (WINDOW_SAMPLES,)
        assert item["audio"].dtype == np.int16


def test_middle_deterministic(prepared):
    """MiddleWindow output is independent of epoch."""
    ds = TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        MiddleWindow(),
        seed=0,
    )
    ds.set_epoch(0)
    audio0 = [ds[i]["audio"].copy() for i in range(len(ds))]
    ds.set_epoch(5)
    audio5 = [ds[i]["audio"].copy() for i in range(len(ds))]
    for a, b in zip(audio0, audio5):
        np.testing.assert_array_equal(a, b)


# --- Short clip filtering ---

def test_short_clips_dropped(prepared):
    """Clips shorter than WINDOW_SAMPLES must not appear in the dataset index."""
    import pandas as pd
    ds = TrainClipDataset(
        prepared.train_memmap,
        prepared.train_index,
        prepared.taxonomy,
        MiddleWindow(),
    )
    clip_lens = ds._index["offset_end"] - ds._index["offset_start"]
    assert (clip_lens >= WINDOW_SAMPLES).all()


# --- Label checks ---

def test_label_is_int(middle_ds):
    for idx in range(len(middle_ds)):
        assert isinstance(middle_ds[idx]["label"], int)


def test_label_range(prepared, middle_ds):
    import pandas as pd
    n_species = len(pd.read_csv(prepared.taxonomy))
    for idx in range(len(middle_ds)):
        assert 0 <= middle_ds[idx]["label"] < n_species
