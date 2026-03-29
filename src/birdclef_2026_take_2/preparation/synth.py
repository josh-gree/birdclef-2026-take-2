"""Synthetic dataset generation for local testing without real competition data."""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from birdclef_2026_take_2.preparation.types import Soundscapes, TrainClips

SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 5 * SAMPLE_RATE  # 160 000

# Real competition constants mirrored in synthetic data
_SOUNDSCAPE_SITES = ["S03", "S08", "S09", "S13", "S15", "S18", "S19", "S22", "S23"]
_XC_CALL_TYPES = ["call", "song", "alarm call", "flight call"]
_RATINGS = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0]
# 36% of real clips are unrated (0.0); remainder spread across 3–5
_RATING_WEIGHTS = [0.36, 0.10, 0.10, 0.15, 0.15, 0.14]


def make_ogg_bytes(
    n_samples: int,
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> bytes:
    """Return bytes of a mono OGG/Vorbis file containing white noise.

    Parameters
    ----------
    n_samples:
        Number of audio frames to generate.
    sample_rate:
        Sample rate in Hz.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    bytes
        Raw OGG/Vorbis file contents.
    """
    if rng is None:
        rng = np.random.default_rng()
    audio = rng.uniform(-0.5, 0.5, size=n_samples).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="OGG", subtype="VORBIS")
    return buf.getvalue()


def _fmt_time(seconds: int) -> str:
    """Format integer seconds as HH:MM:SS."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _fmt_hhmmss(total_seconds: int) -> str:
    """Format integer seconds as HHMMSS (no colons, for filenames)."""
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}{m:02d}{s:02d}"



def make_train_clips(
    species: list[str],
    *,
    n_clips_per_species: int = 3,
    clip_duration_s: float = 10.0,
    species_classes: dict[str, str] | None = None,
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> TrainClips:
    """Generate synthetic single-label training clips.

    Mirrors the real ``train.csv`` column set and filename conventions:

    - 65% of clips use XenoCanto filenames (``XC{id}.ogg``, collection ``XC``),
      35% use iNaturalist (``iNat{id}.ogg``, collection ``iNat``).
    - XC clips get a ``type`` call-type list; iNat clips get ``[]``.
    - Every clip with more than one species available gets 1–2 secondary labels.
    - Ratings follow the real distribution: 36% unrated (0.0), rest spread 3–5.

    Parameters
    ----------
    species:
        List of species codes to generate clips for.
    n_clips_per_species:
        Number of clips to generate per species.
    clip_duration_s:
        Duration of each clip in seconds.
    species_classes:
        Optional mapping of species code → ``class_name``
        (e.g. ``{"spc1": "Aves", "spc2": "Amphibia"}``).
        Defaults to ``"Aves"`` for all species.
    sample_rate:
        Sample rate in Hz.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    TrainClips
        Clips and matching metadata DataFrame.
    """
    if rng is None:
        rng = np.random.default_rng()
    if species_classes is None:
        species_classes = {}

    clip_samples = int(clip_duration_s * sample_rate)
    clips: list[tuple[str, bytes]] = []
    rows: list[dict] = []

    clip_id = 1
    for sp in species:
        class_name = species_classes.get(sp, "Aves")
        for _ in range(n_clips_per_species):
            is_xc = rng.random() < 0.65
            if is_xc:
                filename = f"{sp}/XC{clip_id:07d}.ogg"
                collection = "XC"
                n_types = int(rng.integers(1, 3))
                call_types = rng.choice(_XC_CALL_TYPES, size=n_types, replace=False).tolist()
                clip_type = str(call_types)
            else:
                filename = f"{sp}/iNat{clip_id:07d}.ogg"
                collection = "iNat"
                clip_type = "[]"

            # Always assign secondary labels from other species when possible
            if len(species) > 1:
                others = [s for s in species if s != sp]
                n_secondary = int(rng.integers(1, min(3, len(others)) + 1))
                secondary = rng.choice(others, size=n_secondary, replace=False).tolist()
                secondary_labels = str(secondary)
            else:
                secondary_labels = "[]"

            rating = float(rng.choice(_RATINGS, p=_RATING_WEIGHTS))

            ogg = make_ogg_bytes(clip_samples, sample_rate=sample_rate, rng=rng)
            clips.append((filename, ogg))
            rows.append({
                "primary_label": sp,
                "secondary_labels": secondary_labels,
                "type": clip_type,
                "latitude": round(rng.uniform(-90, 90), 4),
                "longitude": round(rng.uniform(-180, 180), 4),
                "scientific_name": f"Synthus {sp}",
                "common_name": f"Synth Bird {sp}",
                "class_name": class_name,
                "inat_taxon_id": sp if not is_xc else "",
                "author": "synth",
                "license": "cc-by-nc",
                "rating": rating,
                "url": f"https://example.com/{sp}/{clip_id}",
                "filename": filename,
                "collection": collection,
            })
            clip_id += 1

    return TrainClips(clips=clips, metadata=pd.DataFrame(rows))


def make_soundscapes(
    species: list[str],
    *,
    n_labelled: int = 2,
    n_unlabelled: int = 4,
    soundscape_duration_s: float = 60.0,
    windows_per_file: int = 12,
    sample_rate: int = SAMPLE_RATE,
    rng: np.random.Generator | None = None,
) -> Soundscapes:
    """Generate synthetic soundscape recordings with windowed labels.

    Mirrors the real competition where most soundscapes are unlabelled:
    all recordings appear in the zip under ``train_soundscapes/``, but only
    the labelled subset has rows in ``train_soundscapes_labels.csv``.

    Filename pattern: ``BC2026_Train_{N:04d}_{site}_{YYYYMMDD}_{HHMMSS}.ogg``

    Site codes are drawn from the 9 sites that have expert annotations in the
    real dataset: S03, S08, S09, S13, S15, S18, S19, S22, S23.

    Parameters
    ----------
    species:
        List of species codes to assign as window labels.
    n_labelled:
        Number of soundscape recordings that have ground-truth labels.
    n_unlabelled:
        Number of soundscape recordings with no labels (audio only).
    soundscape_duration_s:
        Duration of each recording in seconds. Real soundscapes are 60 s.
    windows_per_file:
        Number of labelled 5-second windows per labelled recording. Real
        labelled soundscapes have 12 windows (covering the full 60 s).
    sample_rate:
        Sample rate in Hz.
    rng:
        Optional random generator for reproducibility.

    Returns
    -------
    Soundscapes
        All recordings (labelled + unlabelled) and a labels DataFrame that
        covers only the labelled subset.
    """
    if rng is None:
        rng = np.random.default_rng()

    soundscape_samples = int(soundscape_duration_s * sample_rate)
    recordings: list[tuple[str, bytes]] = []
    rows: list[dict] = []

    base_date = "20250101"
    base_time_s = 3 * 3600  # 03:00:00

    n_total = n_labelled + n_unlabelled
    for i in range(n_total):
        site = _SOUNDSCAPE_SITES[i % len(_SOUNDSCAPE_SITES)]
        time_s = base_time_s + i * 3600
        filename = f"BC2026_Train_{i + 1:04d}_{site}_{base_date}_{_fmt_hhmmss(time_s)}.ogg"

        ogg = make_ogg_bytes(soundscape_samples, sample_rate=sample_rate, rng=rng)
        recordings.append((filename, ogg))

        if i < n_labelled:
            for w in range(windows_per_file):
                start_s = w * 5
                end_s = start_s + 5
                n_labels = int(rng.integers(1, min(3, len(species)) + 1))
                window_species = rng.choice(species, size=n_labels, replace=False).tolist()
                rows.append({
                    "filename": filename,
                    "start": _fmt_time(start_s),
                    "end": _fmt_time(end_s),
                    "primary_label": ";".join(window_species),
                })

    return Soundscapes(recordings=recordings, labels=pd.DataFrame(rows))


def make_dataset_zip(
    path: Path,
    *,
    species: list[str] | None = None,
    species_classes: dict[str, str] | None = None,
    n_clips_per_species: int = 3,
    clip_duration_s: float = 10.0,
    n_labelled_soundscapes: int = 2,
    n_unlabelled_soundscapes: int = 4,
    soundscape_duration_s: float = 60.0,
    windows_per_soundscape: int = 12,
    sample_rate: int = SAMPLE_RATE,
    seed: int | None = None,
) -> Path:
    """Build a complete synthetic dataset zip mirroring the competition layout.

    The zip contains:
        train.csv
        train_audio/{species}/{XC or iNat id}.ogg
        train_soundscapes_labels.csv
        train_soundscapes/BC2026_Train_NNNN_SXX_YYYYMMDD_HHMMSS.ogg
        taxonomy.csv

    Parameters
    ----------
    path:
        Directory in which to write ``synth_dataset.zip``.
    species:
        List of species codes. Defaults to ``["spc1", "spc2", "spc3"]``.
    species_classes:
        Optional mapping of species code → ``class_name``. If omitted, cycles
        through ``["Aves", "Amphibia", "Insecta"]`` across the species list.
    n_clips_per_species:
        Number of single-label clips to generate per species.
    clip_duration_s:
        Duration of each single-label clip in seconds.
    n_labelled_soundscapes:
        Number of soundscape recordings with ground-truth labels.
    n_unlabelled_soundscapes:
        Number of soundscape recordings with no labels (audio only).
    soundscape_duration_s:
        Duration of each soundscape recording in seconds.
    windows_per_soundscape:
        Number of labelled 5-second windows per soundscape file. Defaults to
        12, matching the real competition's full-60s labelled files.
    sample_rate:
        Sample rate in Hz used for all audio.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    Path
        Path to the created zip file.
    """
    if species is None:
        species = ["spc1", "spc2", "spc3"]

    if species_classes is None:
        _class_cycle = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
        species_classes = {sp: _class_cycle[i % len(_class_cycle)] for i, sp in enumerate(species)}

    rng = np.random.default_rng(seed)
    zip_path = path / "synth_dataset.zip"

    train_clips = make_train_clips(
        species,
        n_clips_per_species=n_clips_per_species,
        clip_duration_s=clip_duration_s,
        species_classes=species_classes,
        sample_rate=sample_rate,
        rng=rng,
    )
    soundscapes = make_soundscapes(
        species,
        n_labelled=n_labelled_soundscapes,
        n_unlabelled=n_unlabelled_soundscapes,
        soundscape_duration_s=soundscape_duration_s,
        windows_per_file=windows_per_soundscape,
        sample_rate=sample_rate,
        rng=rng,
    )
    taxonomy_df = pd.DataFrame([
        {
            "primary_label": sp,
            "common_name": f"Synth Bird {sp}",
            "scientific_name": f"Synthus {sp}",
            "class_name": species_classes[sp],
        }
        for sp in species
    ])

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for filename, ogg in train_clips.clips:
            zf.writestr(f"train_audio/{filename}", ogg)
        zf.writestr("train.csv", train_clips.metadata.to_csv(index=False))

        for filename, ogg in soundscapes.recordings:
            zf.writestr(f"train_soundscapes/{filename}", ogg)
        zf.writestr("train_soundscapes_labels.csv", soundscapes.labels.to_csv(index=False))

        zf.writestr("taxonomy.csv", taxonomy_df.to_csv(index=False))

    return zip_path
