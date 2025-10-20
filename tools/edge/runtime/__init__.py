"""Runtime adapters used by the deployment scripts."""

from .onnxrt import OnnxRuntimeEmulator
from .tflite import TFLiteEmulator

__all__ = ["OnnxRuntimeEmulator", "TFLiteEmulator"]
