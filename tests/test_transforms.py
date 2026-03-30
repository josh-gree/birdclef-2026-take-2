"""Tests for the waveform-to-spectrogram pipeline."""

import torch
import pytest

from birdclef_2026_take_2.transforms import (
    AmplitudeToDB,
    MelSpectrogram,
    PerSampleMinMaxNorm,
    Resize,
    build_spectrogram_pipeline,
)

BATCH = 4
SAMPLES = 160_000  # 5s at 32kHz


@pytest.fixture
def waveforms():
    return torch.randn(BATCH, SAMPLES)


def test_mel_spectrogram_output_shape(waveforms):
    mel = MelSpectrogram(n_mels=128)
    out = mel(waveforms)
    assert out.shape[0] == BATCH
    assert out.shape[1] == 128


def test_mel_spectrogram_non_negative(waveforms):
    out = MelSpectrogram()(waveforms)
    assert out.min().item() >= 0.0


def test_mel_spectrogram_weights_frozen(waveforms):
    mel = MelSpectrogram()
    for p in mel.parameters():
        assert not p.requires_grad


def test_amplitude_to_db_output_range(waveforms):
    mel = MelSpectrogram()
    spec = mel(waveforms)
    db = AmplitudeToDB(top_db=80.0)(spec)
    max_db = db.amax(dim=(-2, -1))
    min_db = db.amin(dim=(-2, -1))
    assert (max_db - min_db).max().item() <= 80.0 + 1e-3


def test_per_sample_norm_range(waveforms):
    mel = MelSpectrogram()
    spec = AmplitudeToDB()(mel(waveforms))
    normed = PerSampleMinMaxNorm()(spec)
    assert normed.min().item() >= 0.0 - 1e-5
    assert normed.max().item() <= 1.0 + 1e-5


def test_resize_output_shape(waveforms):
    mel = MelSpectrogram()
    spec = mel(waveforms)
    out = Resize(224, 224)(spec)
    assert out.shape == (BATCH, 1, 224, 224)


def test_resize_arbitrary_size(waveforms):
    mel = MelSpectrogram()
    spec = mel(waveforms)
    out = Resize(128, 312)(spec)
    assert out.shape == (BATCH, 1, 128, 312)


def test_pipeline_output_shape(waveforms):
    pipeline = build_spectrogram_pipeline(height=224, width=224)
    out = pipeline(waveforms)
    assert out.shape == (BATCH, 1, 224, 224)


def test_pipeline_output_range(waveforms):
    pipeline = build_spectrogram_pipeline(height=224, width=224)
    out = pipeline(waveforms)
    assert out.min().item() >= 0.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


def test_pipeline_output_dtype(waveforms):
    pipeline = build_spectrogram_pipeline(height=224, width=224)
    out = pipeline(waveforms)
    assert out.dtype == torch.float32


def test_pipeline_no_grad(waveforms):
    pipeline = build_spectrogram_pipeline()
    for p in pipeline.parameters():
        assert not p.requires_grad
