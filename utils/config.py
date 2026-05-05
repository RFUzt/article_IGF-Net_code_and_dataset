# utils/config.py

import yaml
import copy
from typing import Any, Dict


def load_config(path: str):

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults: Dict[str, Any] = {
        "seed": 42,
        "device": "cuda",
        "output_dir": "runs/default",

        "model": {
            "name": "unet",
            "in_channels": 3,
            "num_classes": 1,
            "apply_sigmoid": True,
            "params": {}
        },

        "data": {
            "train": {
                "dataset": "npyseg",
                "params": {}
            },
            "val": {
                "dataset": "npyseg",
                "params": {}
            },
            "test": {
                "dataset": "npyseg",
                "params": {}
            },
            "infer": {
                "dataset": "npyseg",
                "params": {}
            },
        },

        "optim": {
            "name": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0
        },

        "sched": {
            "name": None
        },

        "train": {
            "epochs": 100,
            "batch_size": 4,
            "num_workers": 4,
            "loss": "bce"
        },

        "val": {
            "batch_size": 4,
            "num_workers": 4,
            "checkpoint": ""
        },

        "test": {
            "batch_size": 4,
            "num_workers": 4,
            "checkpoint": ""
        },

        "infer": {
            "threshold": 0.5,
            "checkpoint": "",
            "save_dir": "runs/infer"
        }
    }

    def merge(a: Any, b: Any):

        if isinstance(a, dict) and isinstance(b, dict):
            z = copy.deepcopy(a)
            for k, v in b.items():
                z[k] = merge(a.get(k), v)
            return z

        return b if b is not None else a

    return merge(defaults, cfg or {})

