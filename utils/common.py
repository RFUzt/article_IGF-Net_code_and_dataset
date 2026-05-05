# utils/common.py

import os
import torch
import random
import numpy as np
from typing import Dict, Any

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(state: Dict[str, Any], is_best: bool, out_dir: str, filename: str = "last.pth"):
    ensure_dir(out_dir)
    fpath = os.path.join(out_dir, filename)
    torch.save(state, fpath)
    if is_best:
        best_path = os.path.join(out_dir, "best.pth")
        torch.save(state, best_path)
