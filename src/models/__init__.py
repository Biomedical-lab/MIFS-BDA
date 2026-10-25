from .encoders import ImageEncoder, TextEncoder
from .fusion import (
    ImageOnly,
    TextOnly,
    EarlyFusion,
    DeepFusion,
    LMF,
    LateFusion,
    CrossAttn,
    GAF,
    HybridFusion,
)

MODEL_REGISTRY = {
    "ImageOnly": ImageOnly,
    "TextOnly": TextOnly,
    "EarlyFusion": EarlyFusion,
    "DeepFusion": DeepFusion,
    "LMF": LMF,
    "LateFusion": LateFusion,
    "CrossAttn": CrossAttn,
    "GAF": GAF,
    "HybridFusion": HybridFusion,
}

def get_model(name, vocab_size, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](vocab_size, **kwargs)

__all__ = [
    "ImageEncoder",
    "TextEncoder",
    "ImageOnly",
    "TextOnly",
    "EarlyFusion",
    "DeepFusion",
    "LMF",
    "LateFusion",
    "CrossAttn",
    "GAF",
    "HybridFusion",
    "MODEL_REGISTRY",
    "get_model",
]
