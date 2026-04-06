"""Patch model-loading before streamlit_app is imported at collection time.

Module-level patches (not fixtures) are required because streamlit_app.py
executes load_model() at import time, which would otherwise attempt to
download model weights.
"""

from unittest.mock import MagicMock, patch

_model_patch = patch(
    "mlx_transformers.models.RobertaForSequenceClassification",
    new_callable=lambda: lambda *a, **kw: MagicMock(),
)
_model_patch.start()
