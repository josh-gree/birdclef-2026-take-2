import torch
import torch.nn as nn


PERCH_EMBEDDING_DIM = 1536
PERCH_SPATIAL_TIME = 16
PERCH_SPATIAL_FREQ = 4


class GeMFreqPooling(nn.Module):
    """Generalised Mean Pooling over the frequency axis.

    Collapses the freq dim of a (B, T, F, C) tensor, leaving time intact.
    p=1 → average pool, p→∞ → max pool. Initialised to p=3 (soft-max).
    """

    def __init__(self, init_p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(init_p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F, C)
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=2).pow(1.0 / p)
        # returns (B, T, C)


class _PerchBase(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        import os
        import glob as _glob
        print(f"[PerchBase] torch version: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        print(f"[PerchBase] torch cuda available: {torch.cuda.is_available()}")
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        cublas_libs = _glob.glob(os.path.join(torch_lib, "libcublas*.so*"))
        print(f"[PerchBase] cublas libs in torch/lib: {cublas_libs}")

        import onnxruntime as ort
        print(f"[PerchBase] onnxruntime version: {ort.__version__}")
        print(f"[PerchBase] available providers: {ort.get_available_providers()}")
        print(f"[PerchBase] calling preload_dlls()...")
        ort.preload_dlls()
        print(f"[PerchBase] available providers after preload: {ort.get_available_providers()}")
        ort.set_default_logger_severity(3)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[PerchBase] active providers: {self._session.get_providers()}")
        self._input_name = self._session.get_inputs()[0].name


class PerchSpatialAttention(_PerchBase):
    def __init__(self, onnx_path: str, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__(onnx_path)

        output_names = [o.name for o in self._session.get_outputs()]
        assert "spatial_embedding" in output_names, f"Expected 'spatial_embedding' in {output_names}"

        self.freq_pool = GeMFreqPooling()

        # Shared per-timestep classification head.
        # nn.Linear broadcasts over leading dims, so [B, 16, 1536] -> [B, 16, num_classes].
        self.timestep_head = nn.Sequential(
            nn.Linear(PERCH_EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Attention network: [B, 16, 1536] -> [B, 16, 1] scores, then softmax over time.
        self.attn_net = nn.Sequential(
            nn.Linear(PERCH_EMBEDDING_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio_np = waveforms.detach().cpu().numpy()
        spatial = self._session.run(["spatial_embedding"], {self._input_name: audio_np})[0]
        spatial = torch.from_numpy(spatial).to(waveforms.device)  # [B, T=16, F=4, C=1536]
        time_feats = self.freq_pool(spatial)    # [B, 16, 1536]

        step_logits = self.timestep_head(time_feats)   # [B, 16, num_classes]

        attn_scores = self.attn_net(time_feats)         # [B, 16, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        logits = (attn_weights * step_logits).sum(dim=1)  # [B, num_classes]

        return logits
