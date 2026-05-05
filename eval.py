import os
import sys
import glob
import logging
import yaml
import argparse
from typing import Optional

import cv2
import torch
import numpy as np

try:
    from utils.metrics import compute_metrics
except Exception:
    from utils.metrics import compute_metrics
try:
    from data.dataset import _read_mask
except Exception:
    from data.dataset import _read_mask


def load_yaml_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML config file.')
    return parser.parse_args()


args = parse_args()


config = load_yaml_config(args.config)

PRED_DIR    = config['eval']['pred_dir']
MASK_DIR    = config['eval']['mask_dir']
SUFFIX      = config['eval']['suffix']
LOG_PATH    = config['eval']['log_path']

NUM_CLASSES = config['eval']['num_classes']
THRESHOLD   = config['eval']['threshold']
EXPECT_LOGITS = config['eval']['expect_logits']

EPS = 1e-6


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("eval_preds")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def _binarize_pred_img(img: np.ndarray):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if (img > 1).any():
        return (img > 127).astype(np.uint8)
    else:
        return (img > 0).astype(np.uint8)


def _find_mask_by_stem(mask_dir: str, stem: str):
    for suf in [".png", ".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".bmp"]:
        cand = os.path.join(mask_dir, stem + suf)
        if os.path.exists(cand):
            return cand
    return None


def _probs_to_logits(p: torch.Tensor, eps: float = EPS):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


def main():
    log_path = LOG_PATH or os.path.join(PRED_DIR, "eval.txt")
    logger = setup_logger(log_path)

    suffix = SUFFIX if SUFFIX.startswith(".") else ("." + SUFFIX)

    logger.info("==== Evaluation start ====")
    logger.info(f"pred-dir: {PRED_DIR}")
    logger.info(f"mask-dir: {MASK_DIR}")
    logger.info(f"suffix:   {suffix}")
    logger.info(f"num_classes: {NUM_CLASSES}")
    logger.info(f"use_logits: {EXPECT_LOGITS}, threshold: {THRESHOLD}")

    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*" + suffix)))
    if len(pred_files) == 0:
        logger.info("No prediction files found. Check PRED_DIR and SUFFIX.")
        logger.info("==== Evaluation end ====")
        return

    per_img = []
    agg_iou = agg_dice = agg_prec = agg_rec = 0.0
    counted = 0

    for pf in pred_files:
        stem = os.path.splitext(os.path.basename(pf))[0]
        gt_path = _find_mask_by_stem(MASK_DIR, stem)
        if gt_path is None:
            logger.info(f"[Skip] GT not found for: {stem}")
            continue

        pred_img = cv2.imread(pf, cv2.IMREAD_GRAYSCALE)
        if pred_img is None:
            logger.info(f"[Skip] Cannot read pred: {pf}")
            continue
        pred01 = _binarize_pred_img(pred_img)

        gt01 = _read_mask(gt_path)

        h = min(pred01.shape[0], gt01.shape[0])
        w = min(pred01.shape[1], gt01.shape[1])
        pred01 = pred01[:h, :w]
        gt01   = gt01[:h, :w]

        pred_prob_t = torch.from_numpy(pred01[None, None, ...]).to(torch.float32)
        gt_t        = torch.from_numpy(gt01[None, ...]).to(torch.long)

        if EXPECT_LOGITS:
            out_t = _probs_to_logits(pred_prob_t)
        else:
            out_t = pred_prob_t

        m = compute_metrics(
            outputs=out_t,
            targets=gt_t,
            num_classes=NUM_CLASSES,
            threshold=THRESHOLD,
            use_logits=EXPECT_LOGITS,
            ignore_index=None
        )

        miou  = float(m["mean"]["miou_excl_ignored"])
        mdice = float(m["mean"]["mdice_excl_ignored"])
        mprec = float(m["mean"]["precision_macro"])
        mrec  = float(m["mean"]["recall_macro"])

        per_img.append((stem, miou, mdice, mprec, mrec))
        agg_iou  += miou
        agg_dice += mdice
        agg_prec += mprec
        agg_rec  += mrec
        counted  += 1

    if counted == 0:
        logger.info("No valid pairs to evaluate.")
        logger.info("==== Evaluation end ====")
        return

    per_img_sorted = sorted(per_img, key=lambda x: x[1], reverse=True)
    logger.info("---- Per-image metrics (sorted by IoU desc; mean.* fields) ----")
    logger.info("stem\tmIoU\tmDice\tPrecision(macro)\tRecall(macro)")
    for row in per_img_sorted:
        logger.info("{:s}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(*row))

    miou  = agg_iou / counted
    mdice = agg_dice / counted
    mprec = agg_prec / counted
    mrec  = agg_rec / counted

    logger.info("---- Summary ----")
    logger.info(f"Counted images: {counted}")
    logger.info(f"mIoU (mean.*): {miou:.6f}")
    logger.info(f"mDice(mean.*): {mdice:.6f}")
    logger.info(f"Precision(macro): {mprec:.6f}")
    logger.info(f"Recall(macro):    {mrec:.6f}")
    logger.info("==== Evaluation end ====")


if __name__ == "__main__":
    main()

