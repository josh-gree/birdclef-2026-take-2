"""Compute a UMAP of raw mel spectrograms and save a plot."""

import modal

app = modal.App("birdclef-2026-spectrogram-umap")

processed_volume = modal.Volume.from_name("birdclef-2026-processed", create_if_missing=False)
artifacts_volume = modal.Volume.from_name("birdclef-2026-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy", "pandas", "pyarrow", "torch", "nnaudio",
        "scikit-learn", "umap-learn", "matplotlib",
    )
    .add_local_python_source("birdclef_2026_take_2")
)


@app.function(
    image=image,
    volumes={
        "/processed": processed_volume,
        "/artifacts": artifacts_volume,
    },
    timeout=1800,
    cpu=8,
)
def compute_umap(n_samples: int = 2000, pca_components: int = 50):
    import logging
    import shutil
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.decomposition import PCA
    from umap import UMAP

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    from birdclef_2026_take_2.dataset import MiddleWindow, TrainClipDataset
    from birdclef_2026_take_2.transforms import build_spectrogram_pipeline

    log.info("Copying train.npy to /tmp...")
    shutil.copy2("/processed/train.npy", "/tmp/train.npy")
    log.info("Done.")

    ds = TrainClipDataset(
        memmap_path=Path("/tmp/train.npy"),
        index_path=Path("/processed/train_index.parquet"),
        taxonomy_path=Path("/processed/taxonomy.csv"),
        window_strategy=MiddleWindow(),
    )

    taxonomy = pd.read_csv("/processed/taxonomy.csv").sort_values("primary_label").reset_index(drop=True)
    label_to_name = {i: row.primary_label for i, row in taxonomy.iterrows()}

    n_samples = min(n_samples, len(ds))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=n_samples, replace=False)

    log.info("Building spectrogram pipeline...")
    pipeline = build_spectrogram_pipeline()
    pipeline.eval()

    log.info("Extracting spectrograms for %d samples...", n_samples)
    spectrograms = []
    labels = []
    with torch.no_grad():
        batch_size = 64
        for start in range(0, n_samples, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_audio = torch.tensor(
                np.stack([ds[int(i)]["audio"] for i in batch_indices])
            )
            specs = pipeline(batch_audio)  # (B, 1, 256, 256)
            spectrograms.append(specs.numpy().reshape(len(batch_indices), -1))
            labels.extend([ds[int(i)]["label"] for i in batch_indices])
            if (start // batch_size) % 5 == 0:
                log.info("  %d / %d", start + len(batch_indices), n_samples)

    X = np.concatenate(spectrograms, axis=0)  # (n_samples, 65536)
    y = np.array(labels)
    log.info("Spectrogram matrix: %s", X.shape)

    log.info("Running PCA to %d components...", pca_components)
    pca = PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X)
    log.info("Explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)

    log.info("Running UMAP...")
    reducer = UMAP(n_components=2, random_state=42, verbose=True)
    embedding = reducer.fit_transform(X_pca)

    log.info("Plotting...")
    n_classes = len(taxonomy)
    cmap = plt.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(14, 10))
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=y, cmap=cmap, s=6, alpha=0.7, linewidths=0,
    )
    ax.set_title(f"UMAP of mel spectrograms — {n_samples} samples, {n_classes} species", fontsize=13)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.axis("off")

    out_path = Path("/artifacts/spectrogram_umap.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info("Saved plot to %s", out_path)
    artifacts_volume.commit()


@app.local_entrypoint()
def main():
    compute_umap.remote()
