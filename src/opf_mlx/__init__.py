"""OpenAI Privacy Filter, implemented from scratch in MLX for Apple Silicon."""

from .precision import request_full_precision

# Must happen before the first GPU matmul of the process. See opf_mlx.precision.
request_full_precision()

from .pipeline import PrivacyFilter, PrivacyFilterWarning, RedactionResult  # noqa: E402
from .spans import DetectedSpan  # noqa: E402

__all__ = ["DetectedSpan", "PrivacyFilter", "PrivacyFilterWarning", "RedactionResult"]
