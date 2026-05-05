# utils/registry.py
from typing import Callable, Dict, Any

class Registry:
    def __init__(self, name: str):
        self._name = name
        self._dict: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def _wrap(obj: Callable[..., Any]):
            key = name.lower()
            if key in self._dict:
                raise KeyError(f"{self._name} '{name}' already registered.")
            self._dict[key] = obj
            return obj
        return _wrap

    def get(self, name: str):
        key = (name or "").lower()
        if key not in self._dict:
            raise KeyError(f"{self._name} '{name}' not found. Available: {list(self._dict.keys())}")
        return self._dict[key]

    def names(self):
        return sorted(list(self._dict.keys()))

MODEL_REGISTRY   = Registry("Model")
DATASET_REGISTRY = Registry("Dataset")
AUGMENT_REGISTRY = Registry("Augment")

def register_model(name: str):
    return MODEL_REGISTRY.register(name)

