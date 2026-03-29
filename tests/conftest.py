import pytest

from birdclef_2026_take_2.preparation.synth import make_dataset_zip


@pytest.fixture(scope="session")
def default_species() -> list[str]:
    return ["spc1", "spc2", "spc3"]


@pytest.fixture(scope="session")
def dataset_zip(tmp_path_factory, default_species: list[str]):
    path = tmp_path_factory.mktemp("data")
    return make_dataset_zip(path, species=default_species, seed=42)
