"""PyTorch Dataset for single-label training clips."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 5 * SAMPLE_RATE  # 160_000


@dataclass
class ConsecutiveWindows:
    """Expand each clip into non-overlapping 5-second windows (drop remainder)."""


@dataclass
class RandomWindow:
    """Sample one random 5-second window per clip per epoch."""


@dataclass
class MiddleWindow:
    """Take the centre 5-second window from each clip."""


WindowStrategy = ConsecutiveWindows | RandomWindow | MiddleWindow


class TrainClipDataset(Dataset):
    """Dataset of 5-second windows sliced from a memmap of training clips.

    Parameters
    ----------
    memmap_path:
        Path to the int16 ``.npy`` memmap produced by :func:`prepare_dataset`.
    index_path:
        Path to the ``train_index.parquet`` file produced by :func:`prepare_dataset`.
    taxonomy_path:
        Path to ``taxonomy.csv`` (columns: ``primary_label``, ``common_name``,
        ``scientific_name``, ``class_name``).  Integer labels are assigned by
        sorting ``primary_label`` alphabetically and using the row position.
    window_strategy:
        One of :class:`ConsecutiveWindows`, :class:`RandomWindow`, or
        :class:`MiddleWindow`.
    seed:
        Base seed for :class:`RandomWindow` reproducibility.
    """

    def __init__(
        self,
        memmap_path: Path,
        index_path: Path,
        taxonomy_path: Path,
        window_strategy: WindowStrategy,
        seed: int = 0,
    ) -> None:
        self._memmap = np.load(memmap_path, mmap_mode="r")

        index = pd.read_parquet(index_path)
        index = index[(index["offset_end"] - index["offset_start"]) >= WINDOW_SAMPLES]

        taxonomy = pd.read_csv(taxonomy_path).sort_values("primary_label").reset_index(drop=True)
        label_map: dict[str, int] = {row.primary_label: i for i, row in taxonomy.iterrows()}

        index = index[index["primary_label"].isin(label_map)].reset_index(drop=True)

        self._index = index
        self._label_map = label_map
        self._strategy = window_strategy
        self._seed = seed
        self._epoch = 0

        match window_strategy:
            case ConsecutiveWindows():
                # Explicit flat index: one (clip_idx, window_within_clip) entry per window.
                # __getitem__ does a direct list lookup — no binary search needed.
                self._window_index: list[tuple[int, int]] = [
                    (clip_idx, w)
                    for clip_idx, row in index.iterrows()
                    for w in range((row.offset_end - row.offset_start) // WINDOW_SAMPLES)
                ]

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for :class:`RandomWindow` reproducibility."""
        self._epoch = epoch

    def __len__(self) -> int:
        match self._strategy:
            case ConsecutiveWindows():
                return len(self._window_index)
            case RandomWindow() | MiddleWindow():
                return len(self._index)
            case _:
                raise TypeError(f"Unknown window strategy: {type(self._strategy)}")

    def __getitem__(self, idx: int) -> dict:
        match self._strategy:
            case ConsecutiveWindows():
                clip_idx, window_within_clip = self._window_index[idx]
                # offset_start is the clip's position in the flat memmap.
                # window_within_clip steps through the clip in 160,000-sample
                # increments, so window 0 starts at offset_start, window 1 at
                # offset_start + 160,000, window 2 at offset_start + 320,000, etc.
                sample_start = int(self._index["offset_start"].iat[clip_idx]) + window_within_clip * WINDOW_SAMPLES
            case RandomWindow():
                clip_idx = idx
                clip_len = int(self._index["offset_end"].iat[clip_idx]) - int(self._index["offset_start"].iat[clip_idx])
                max_offset = clip_len - WINDOW_SAMPLES
                # Seed the RNG from (base_seed, epoch, clip_idx) so that:
                #   - the same clip in the same epoch always returns the same window
                #     (deterministic, safe to call __getitem__ multiple times)
                #   - different epochs produce different windows for the same clip
                #     (call set_epoch(epoch) before each epoch to advance this)
                #   - two datasets constructed with the same seed are identical
                # numpy accepts a sequence as a seed, hashing the tuple into a
                # single internal state — no manual seed arithmetic required.
                rng = np.random.default_rng([self._seed, self._epoch, clip_idx])
                window_offset = int(rng.integers(0, max_offset + 1))
                sample_start = int(self._index["offset_start"].iat[clip_idx]) + window_offset
            case MiddleWindow():
                clip_idx = idx
                clip_len = int(self._index["offset_end"].iat[clip_idx]) - int(self._index["offset_start"].iat[clip_idx])
                window_offset = (clip_len - WINDOW_SAMPLES) // 2
                sample_start = int(self._index["offset_start"].iat[clip_idx]) + window_offset
            case _:
                raise TypeError(f"Unknown window strategy: {type(self._strategy)}")

        audio = self._memmap[sample_start : sample_start + WINDOW_SAMPLES].astype(np.float32) / 32768.0
        label = self._label_map[self._index["primary_label"].iat[clip_idx]]
        return {"audio": audio, "label": label}
