import torch
import torch.nn as nn


PERCH_EMBEDDING_DIM = 1536


class PerchMLP(nn.Module):
    def __init__(self, onnx_path: str, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        import os
        import torch
        # Expose PyTorch's bundled CUDA libraries so ORT can find them
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        os.environ["LD_LIBRARY_PATH"] = torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)

        output_names = [o.name for o in self._session.get_outputs()]
        assert "embedding" in output_names, f"Expected 'embedding' in {output_names}"
        self._input_name = self._session.get_inputs()[0].name

        self.head = nn.Sequential(
            nn.Linear(PERCH_EMBEDDING_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio_np = waveforms.detach().cpu().numpy()
        embeddings = self._session.run(["embedding"], {self._input_name: audio_np})[0]
        return self.head(torch.from_numpy(embeddings).to(waveforms.device))
