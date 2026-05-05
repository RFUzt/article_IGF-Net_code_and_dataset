# engine.py

import torch
from tqdm import tqdm
from typing import List
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import data.dataset
from utils.registry import DATASET_REGISTRY
from utils.metrics import compute_metrics


def build_dataset(dcfg: dict, is_train: bool):

    name = dcfg.get("name", "npyseg")
    params = dcfg.get("params", {})
    cls = DATASET_REGISTRY.get(name)
    return cls(is_train=is_train, **params)


def make_loader(dataset, batch_size: int, num_workers: int, shuffle: bool = False):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      shuffle=shuffle,
                      pin_memory=True,
                      drop_last=False)


def make_optimizer(params, ocfg: dict):

    name = (ocfg.get("name") or "adam").lower()
    lr = ocfg["lr"]
    wd = ocfg.get("weight_decay", 0.0)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    return Adam(params, lr=lr, weight_decay=wd)


def make_scheduler(optim, scfg: dict, epochs: int):

    if scfg is None:
        return None
    name = (scfg.get("name") or "").lower()
    if name == "steplr":
        step_size = scfg.get("step_size", 30)
        gamma = scfg.get("gamma", 0.1)
        return StepLR(optim, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return CosineAnnealingLR(optim, T_max=epochs)
    return None


def _build_criterion(loss_name: str, num_classes: int, model_has_sigmoid: bool):

    ln = (loss_name or "").lower()
    if num_classes == 1:
        if ln in ("bce", "bceloss"):
            return torch.nn.BCELoss(), False
        if ln in ("bce_logits", "bcewithlogits"):
            if model_has_sigmoid:
                print("[Warn] BCEWithLogitsLoss 与模型内 sigmoid 叠加，建议将 apply_sigmoid=false。")
            return torch.nn.BCEWithLogitsLoss(), True
        raise ValueError(f"Unknown binary loss {loss_name}")
    else:

        if ln in ("ce", "crossentropy", "crossentropyloss"):
            if model_has_sigmoid:
                print("[Warn] 多类 CE 建议关闭模型内 sigmoid (apply_sigmoid=false)。")
            return torch.nn.CrossEntropyLoss(), True
        raise ValueError(f"Unknown multiclass loss {loss_name}. Use 'ce'.")


def get_current_lrs(optimizer):
    return [pg.get("lr", 0.0) for pg in optimizer.param_groups]


def train_one_epoch(model, loader, device, num_classes, loss_name, threshold):

    model.train()
    criterion, expect_logits = _build_criterion(loss_name, num_classes, getattr(model, "apply_sigmoid", False))
    optimizer = model.optimizer

    running = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            raise RuntimeError("Train/Val/Test dataset must return (image, mask, stem)")
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)

        if num_classes == 1:
            y_float = y.float().unsqueeze(1)
            loss = criterion(out, y_float)
        else:
            if y.dim() != 3:
                y = y.argmax(dim=1)
            loss = criterion(out, y.long())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += loss.item()

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, num_classes, threshold, loss_name):

    model.eval()

    criterion, expect_logits = _build_criterion(loss_name, num_classes, getattr(model, "apply_sigmoid", False))

    sum_iou = 0.0
    sum_dice = 0.0
    sum_prec = 0.0
    sum_rec = 0.0
    n = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            raise RuntimeError("Eval dataset must return (image, mask, stem)")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(x)

        if num_classes == 1:
            m = compute_metrics(
                out, y,
                num_classes=1,
                threshold=threshold,
                use_logits=expect_logits,
                ignore_index=None
            )
        else:
            if y.dim() != 3:
                y = y.argmax(dim=1)
            m = compute_metrics(
                out, y,
                num_classes=num_classes,
                threshold=threshold,
                use_logits=True,
                ignore_index=None
            )

        miou = m["mean"]["miou_excl_ignored"]
        mdice = m["mean"]["mdice_excl_ignored"]
        precision_macro = m["mean"]["precision_macro"]
        recall_macro = m["mean"]["recall_macro"]

        sum_iou += float(miou)
        sum_dice += float(mdice)
        sum_prec += float(precision_macro)
        sum_rec += float(recall_macro)
        n += 1

    if n == 0:
        return {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "iou":        sum_iou / n,
        "dice":       sum_dice / n,
        "precision":  sum_prec / n,
        "recall":     sum_rec / n
    }


