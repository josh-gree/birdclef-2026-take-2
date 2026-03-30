"""GPU-compatible waveform-to-spectrogram pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio.features import MelSpectrogram as _MelSpectrogram


class MelSpectrogram(nn.Module):
    """Compute a mel spectrogram from a batch of waveforms using nnAudio.

    Weights are frozen (non-trainable).

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input waveforms in Hz.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between STFT frames.
    n_mels : int
        Number of mel filterbank bins.
    fmin : float
        Lowest frequency of the mel filterbank in Hz.
    fmax : float
        Highest frequency of the mel filterbank in Hz.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
    ):
        super().__init__()
        self.mel = _MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            trainable_mel=False,
            trainable_STFT=False,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveforms : torch.Tensor
            Shape ``(batch, samples)``, float32 in ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 power spectrogram.
        """
        return self.mel(waveforms)


class AmplitudeToDB(nn.Module):
    """Convert a power spectrogram to decibel scale and clamp the dynamic range.

    Applies ``10 * log10(x)``, then floors values at ``max - top_db`` so the
    output is always in ``[-top_db, 0]`` dB relative to the loudest frame.

    Parameters
    ----------
    top_db : float
        Dynamic range to retain in dB.
    """

    def __init__(self, top_db: float = 80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 power spectrogram.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 in ``[-top_db, 0]`` dB.
        """
        x_db = 10.0 * torch.log10(x.clamp(min=1e-9))
        max_db = x_db.amax(dim=(-2, -1), keepdim=True)
        return x_db.clamp(min=max_db - self.top_db)


class PerSampleMinMaxNorm(nn.Module):
    """Normalise each spectrogram independently to ``[0, 1]``.

    Parameters
    ----------
    eps : float
        Small constant added to the denominator to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_mels, time)``, float32 in ``[0, 1]``.
        """
        min_ = x.amin(dim=(-2, -1), keepdim=True)
        max_ = x.amax(dim=(-2, -1), keepdim=True)
        return (x - min_) / (max_ - min_ + self.eps)


class Resize(nn.Module):
    """Resize a ``(batch, n_mels, time)`` spectrogram to a fixed spatial size.

    Adds a channel dimension, producing ``(batch, 1, height, width)`` output.

    Parameters
    ----------
    height : int
        Target height in pixels.
    width : int
        Target width in pixels.
    """

    def __init__(self, height: int, width: int):
        super().__init__()
        self.size = (height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, n_mels, time)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1, height, width)``, float32.
        """
        x = x.unsqueeze(1)
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


def build_spectrogram_pipeline(
    sample_rate: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 20.0,
    fmax: float = 16000.0,
    top_db: float = 80.0,
    height: int = 256,
    width: int = 256,
) -> nn.Sequential:
    """Build the waveform-to-image pipeline.

    Converts a batch of float32 waveforms to normalised single-channel
    spectrogram images ready for a backbone with ``in_chans=1``.

    Pipeline: MelSpectrogram → AmplitudeToDB → PerSampleMinMaxNorm → Resize

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input waveforms in Hz.
    n_fft : int
        FFT size.
    hop_length : int
        Hop length between STFT frames.
    n_mels : int
        Number of mel filterbank bins.
    fmin : float
        Lowest frequency of the mel filterbank in Hz.
    fmax : float
        Highest frequency of the mel filterbank in Hz.
    top_db : float
        Dynamic range retained by ``AmplitudeToDB``.
    height : int
        Target image height in pixels.
    width : int
        Target image width in pixels.

    Returns
    -------
    nn.Sequential
        ``(batch, samples) -> (batch, 1, height, width)``
    """
    return nn.Sequential(
        MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        ),
        AmplitudeToDB(top_db=top_db),
        PerSampleMinMaxNorm(),
        Resize(height, width),
    )
