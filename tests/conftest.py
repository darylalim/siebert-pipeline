"""Patch model-loading before streamlit_app is imported at collection time.

Module-level patches (not fixtures) are required because streamlit_app.py
executes load_model() at import time, which would otherwise attempt to
download model weights and connect to the network.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

_model_patch = patch(
    "mlx_transformers.models.RobertaForSequenceClassification",
    new_callable=lambda: lambda *a, **kw: MagicMock(),
)
_model_patch.start()

_download_patch = patch(
    "huggingface_hub.snapshot_download",
    return_value=str(Path(__file__).parent),
)
_download_patch.start()

_torch_load_patch = patch("torch.load", return_value={})
_torch_load_patch.start()

_save_patch = patch("safetensors.torch.save_file")
_save_patch.start()
