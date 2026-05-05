# infer.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging

import cv2
import torch
import numpy as np

from utils.config import load_config
from utils.common import ensure_dir
from utils.registry import MODEL_REGISTRY
from engine import build_dataset, make_loader
import models


def setup_logger(save_dir: str, filename="infer.log"):
    ensure_dir(save_dir)
    logger = logging.getLogger("infer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    fh = logging.FileHandler(os.path.join(save_dir, filename), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def build_model(mcfg: dict):
    ctor = MODEL_REGISTRY.get(mcfg["name"])
    model = ctor(
        in_channels=mcfg.get("in_channels", 3),
        num_classes=mcfg.get("num_classes", 1),
        apply_sigmoid=mcfg.get("apply_sigmoid", True),
        **(mcfg.get("params") or {})
    )
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="YAML 配置文件路径")
    ap.add_argument("--ckpt", "-k", default=None, help="权重路径（可省略，YAML infer.checkpoint 兜底）")
    ap.add_argument("--save-dir", "-o", default=None, help="输出目录（可省略，YAML infer.save_dir 兜底）")
    ap.add_argument("--suffix", default=None, help="输出图像后缀（可省略，YAML infer.suffix 兜底）")
    ap.add_argument("--threshold", type=float, default=None, help="二分类阈值（可省略，YAML infer.threshold 兜底）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu")
    infer_cfg = cfg.get("infer", {}) if isinstance(cfg, dict) else {}

    ckpt = args.ckpt or infer_cfg.get("checkpoint")
    save_dir = args.save_dir or infer_cfg.get("save_dir")
    suffix = args.suffix or infer_cfg.get("suffix", ".png")
    threshold = args.threshold if args.threshold is not None else infer_cfg.get("threshold", 0.5)

    if not ckpt or not save_dir:
        raise ValueError("必须指定 checkpoint 与 save_dir；可在命令行传入，或在 YAML 的 infer 段中提供。")

    args.ckpt = ckpt
    args.save_dir = save_dir
    args.suffix = suffix

    logger = setup_logger(args.save_dir)
    logger.info("==== Inference start ====")

    model = build_model(cfg["model"]).to(device)
    ckpt_obj = torch.load(args.ckpt, map_location="cpu")
    state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(f"Loaded weights from: {args.ckpt}")
    if missing:    logger.info(f"Missing keys: {missing}")
    if unexpected: logger.info(f"Unexpected keys: {unexpected}")
    model.eval()

    dcfg_test = cfg["data"]["test"]
    ds_te = build_dataset(dcfg_test, is_train=False)
    te_loader = make_loader(
        ds_te,
        batch_size=cfg["val"]["batch_size"],
        num_workers=cfg["val"]["num_workers"],
        shuffle=False
    )

    ensure_dir(args.save_dir)
    num_classes = cfg["model"]["num_classes"]
    apply_sigmoid = cfg["model"].get("apply_sigmoid", True)

    total, saved = 0, 0
    for batch in te_loader:
        if len(batch) != 2:
            raise RuntimeError("Test dataset must return (image, stem) when mask_dir is None")
        x, stems = batch
        x = x.to(device)

        y = model(x)

        if num_classes == 1:
            probs = torch.sigmoid(y) if not apply_sigmoid else y
            pred01 = (probs >= threshold).float()
            pred01 = pred01[:, 0].cpu().numpy()
            for i, stem in enumerate(stems):
                out_path = os.path.join(args.save_dir, stem + args.suffix)
                out_img = (pred01[i] * 255).astype(np.uint8)
                cv2.imwrite(out_path, out_img)
                saved += 1
        else:
            cls_map = y.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for i, stem in enumerate(stems):
                out_path = os.path.join(args.save_dir, stem + args.suffix)
                cv2.imwrite(out_path, cls_map[i])
                saved += 1
        total += x.size(0)

    logger.info(f"Inference done. Batches={len(te_loader)}, images={total}, saved={saved}")
    logger.info("==== Inference end ====")


if __name__ == "__main__":
    main()
