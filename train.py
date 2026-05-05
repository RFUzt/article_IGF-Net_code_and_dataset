# train.py
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import logging
import argparse

import datetime as dt

from pprint import pformat
from utils.config import load_config
from utils.registry import MODEL_REGISTRY
from utils.common import set_seed, ensure_dir, save_checkpoint

import models
from engine import (
    build_dataset, make_loader, make_optimizer, make_scheduler,
    train_one_epoch, evaluate, get_current_lrs
)


def setup_logger(out_dir: str, filename: str = "train.log"):

    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, filename)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logger initialized. Log file -> {log_path}")
    return logger


def build_model(mcfg: dict):

    name = mcfg["name"]
    ctor = MODEL_REGISTRY.get(name)
    model = ctor(
        in_channels=mcfg.get("in_channels", 3),
        num_classes=mcfg.get("num_classes", 1),
        apply_sigmoid=mcfg.get("apply_sigmoid", True),
        **(mcfg.get("params") or {})
    )
    return model


def main(cfg_path: str):

    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_dir = cfg["output_dir"]
    ensure_dir(out_dir)

    logger = setup_logger(out_dir)
    logger.info("#" * 80)
    logger.info("Start training")
    logger.info(f"Time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Torch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.get_device_name(0)}")
    logger.info("#" * 80)

    try:
        logger.info("Full config:")
        logger.info(json.dumps(cfg, ensure_ascii=False, indent=2))
    except Exception:
        logger.info("Full config (fallback pformat):")
        logger.info(pformat(cfg))

    model = build_model(cfg["model"]).to(device)
    logger.info(f"Model built from registry: {cfg['model']['name']}")
    logger.info(f"Model structure:\n{model}")

    ds_tr = build_dataset(cfg["data"]["train"], is_train=True)
    ds_va = build_dataset(cfg["data"]["val"],   is_train=False)
    tr_loader = make_loader(
        ds_tr,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        shuffle=True
    )
    va_loader = make_loader(
        ds_va,
        batch_size=cfg["val"]["batch_size"],
        num_workers=cfg["val"]["num_workers"],
        shuffle=False
    )

    logger.info(f"Train dataset size: {len(ds_tr)}")
    logger.info(f"Val   dataset size: {len(ds_va)}")

    optimizer = make_optimizer(model.parameters(), cfg["optim"])
    scheduler = make_scheduler(optimizer, cfg["sched"], cfg["train"]["epochs"])
    model.optimizer = optimizer

    best_iou = -1.0
    best_epoch = -1
    num_classes = cfg["model"]["num_classes"]
    threshold = cfg["infer"].get("threshold", 0.5)
    loss_name = cfg["train"]["loss"]

    logger.info("#" * 80)
    logger.info("Begin epochs")
    logger.info("#" * 80)

    logger.info("EPOCH | lr | train_loss | val_iou | val_dice | val_precision | val_recall | best_iou")

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        lrs = get_current_lrs(optimizer)
        lr_str = ",".join([f"{v:.6f}" for v in lrs])

        tr_loss = train_one_epoch(
            model, tr_loader, device,
            num_classes=num_classes,
            loss_name=loss_name,
            threshold=threshold
        )

        if scheduler:
            scheduler.step()

        val_metrics = evaluate(
            model, va_loader, device,
            num_classes=num_classes,
            threshold=threshold,
            loss_name=loss_name
        )

        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou = val_metrics["iou"]
            best_epoch = epoch

        save_checkpoint(
            {"model": model.state_dict(), "epoch": epoch, "best_iou": best_iou},
            is_best=is_best,
            out_dir=out_dir,
            filename="last.pth"
        )

        logger.info(
            f"{epoch:>5d} | {lr_str} | {tr_loss:.6f} | "
            f"{val_metrics['iou']:.6f} | {val_metrics['dice']:.6f} | "
            f"{val_metrics['precision']:.6f} | {val_metrics['recall']:.6f} | "
            f"{best_iou:.6f}"
        )

    logger.info("#" * 80)
    logger.info(f"Training finished. Best IoU={best_iou:.6f} at epoch {best_epoch}.")
    logger.info("#" * 80)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", required=True, help="路径到训练 YAML 配置")
    args = ap.parse_args()
    main(args.config)
