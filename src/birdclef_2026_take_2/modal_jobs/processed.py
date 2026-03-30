import modal

app = modal.App("birdclef-2026-processed")

raw_volume = modal.Volume.from_name("birdclef-2026-raw", create_if_missing=False)
processed_volume = modal.Volume.from_name("birdclef-2026-processed", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("numpy", "pandas", "pyarrow", "soundfile")
    .add_local_python_source("birdclef_2026_take_2")
)


@app.function(
    image=image,
    volumes={
        "/raw": raw_volume,
        "/processed": processed_volume,
    },
    timeout=3600,
)
def process_data():
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from birdclef_2026_take_2.preparation.pipeline import prepare_dataset

    prepare_dataset(
        zip_path=Path("/raw/birdclef-2026.zip"),
        output_dir=Path("/processed"),
    )
    processed_volume.commit()


@app.local_entrypoint()
def main():
    process_data.remote()
