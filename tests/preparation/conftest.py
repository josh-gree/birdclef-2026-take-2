import numpy as np
import pytest

from birdclef_2026_take_2.preparation.synth import make_soundscapes, make_train_clips
from birdclef_2026_take_2.preparation.types import Soundscapes, TrainClips

SPECIES = ["spc1", "spc2", "spc3"]


@pytest.fixture(scope="module")
def train_clips() -> TrainClips:
    return make_train_clips(SPECIES, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def many_train_clips() -> TrainClips:
    return make_train_clips(SPECIES, n_clips_per_species=20, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def soundscapes_few() -> Soundscapes:
    return make_soundscapes(SPECIES, n_labelled=1, n_unlabelled=0, rng=np.random.default_rng(0))


@pytest.fixture(scope="module")
def soundscapes_many() -> Soundscapes:
    return make_soundscapes(SPECIES, n_labelled=5, n_unlabelled=4, rng=np.random.default_rng(0))
