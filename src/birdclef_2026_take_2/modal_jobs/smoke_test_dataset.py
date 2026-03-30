"""Throwaway smoke test: exercise TrainClipDataset against the real processed data."""

import modal

app = modal.App("birdclef-2026-smoke-test-dataset")

processed_volume = modal.Volume.from_name("birdclef-2026-processed", create_if_missing=False)

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas", "pyarrow", "torch")
    .add_local_python_source("birdclef_2026_take_2")
)


@app.function(
    image=image,
    volumes={"/processed": processed_volume},
    timeout=600,
)
def smoke_test():
    import logging
    from pathlib import Path

    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    from birdclef_2026_take_2.dataset import (
        WINDOW_SAMPLES,
        ConsecutiveWindows,
        MiddleWindow,
        RandomWindow,
        TrainClipDataset,
    )

    processed = Path("/processed")
    memmap_path = processed / "train.npy"
    index_path = processed / "train_index.parquet"
    taxonomy_path = processed / "taxonomy.csv"

    for strategy, name in [
        (ConsecutiveWindows(), "ConsecutiveWindows"),
        (RandomWindow(), "RandomWindow"),
        (MiddleWindow(), "MiddleWindow"),
    ]:
        log.info("--- %s ---", name)
        ds = TrainClipDataset(memmap_path, index_path, taxonomy_path, strategy)
        log.info("len: %d", len(ds))

        # sample a handful of items spread across the dataset
        indices = np.linspace(0, len(ds) - 1, num=5, dtype=int).tolist()
        for idx in indices:
            item = ds[idx]
            assert item["audio"].shape == (WINDOW_SAMPLES,), f"bad shape at idx {idx}"
            assert item["audio"].dtype == np.int16, f"bad dtype at idx {idx}"
            assert isinstance(item["label"], int), f"label not int at idx {idx}"
            log.info("  idx=%d label=%d audio[0]=%d", idx, item["label"], item["audio"][0])

    log.info("All checks passed.")


@app.local_entrypoint()
def main():
    smoke_test.remote()
