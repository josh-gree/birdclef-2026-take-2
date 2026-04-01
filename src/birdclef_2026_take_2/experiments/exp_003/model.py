import torch
import torch.nn as nn


PERCH_EMBEDDING_DIM = 1536
PERCH_LABEL_DIM = 14795


class PerchMLP(nn.Module):
    def __init__(self, onnx_path: str, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3, use_label_head: bool = False):
        super().__init__()
        import os
        import glob
        print(f"[PerchMLP] torch version: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        print(f"[PerchMLP] torch cuda available: {torch.cuda.is_available()}")
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        cublas_libs = glob.glob(os.path.join(torch_lib, "libcublas*.so*"))
        print(f"[PerchMLP] cublas libs in torch/lib: {cublas_libs}")

        import onnxruntime as ort
        print(f"[PerchMLP] onnxruntime version: {ort.__version__}")
        print(f"[PerchMLP] available providers: {ort.get_available_providers()}")
        print(f"[PerchMLP] calling preload_dlls()...")
        ort.preload_dlls()
        print(f"[PerchMLP] available providers after preload: {ort.get_available_providers()}")
        ort.set_default_logger_severity(3)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[PerchMLP] active providers: {self._session.get_providers()}")

        output_names = [o.name for o in self._session.get_outputs()]
        assert "embedding" in output_names, f"Expected 'embedding' in {output_names}"
        self._input_name = self._session.get_inputs()[0].name
        self._use_label_head = use_label_head

        input_dim = PERCH_LABEL_DIM if use_label_head else PERCH_EMBEDDING_DIM
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        audio_np = waveforms.detach().cpu().numpy()
        output_key = "label" if self._use_label_head else "embedding"
        features = self._session.run([output_key], {self._input_name: audio_np})[0]
        return self.head(torch.from_numpy(features).to(waveforms.device))
