# Copyright © 2026 Weslyn Cory Whitehead Jr. / AsAManThinks Research.
# Licensed under the AAMT Community License 1.0 (LICENSE.md); patents pending, see PATENTS.md.
"""The .aamt file format, in Python: read model files (`aamt.model`) and open them in Hugging Face transformers (`aamt.hf`)."""

from .model import AamtModel, Packed, TensorEntry, dequantize, parse_blob

__all__ = ["AamtModel", "Packed", "TensorEntry", "dequantize", "parse_blob"]
__version__ = "0.1.0"
