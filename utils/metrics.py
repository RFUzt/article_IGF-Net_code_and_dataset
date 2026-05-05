# utils/metrics.py


from typing import Dict, List, Optional, Sequence, Union
import torch
import torch.nn.functional as F
import numpy as np


def _ensure_tensor(x: torch.Tensor):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x


def _to_probs(outputs: torch.Tensor, num_classes: int, use_logits: bool = True) :

    if use_logits:
        if num_classes == 1:
            probs = torch.sigmoid(outputs)
        else:
            probs_full = F.softmax(outputs, dim=1)

            probs = probs_full
    else:

        probs = outputs
        if num_classes == 1 and probs.shape[1] != 1:
            probs = probs[:, :1]

    return probs


def _to_onehot(targets: torch.Tensor, num_classes: int):

    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]

    if num_classes == 1:
        tgt = (targets > 0.5).float().unsqueeze(1)
        return tgt
    else:
        if targets.dtype != torch.long:
            targets = targets.long()
        n, h, w = targets.shape
        onehot = torch.zeros((n, num_classes, h, w), dtype=torch.float32, device=targets.device)
        return onehot.scatter_(1, targets.unsqueeze(1), 1.0)


def _mask_ignore(targets_oh: torch.Tensor, ignore_index: Optional[Union[int, Sequence[int]]]):

    if ignore_index is None:
        return torch.zeros(targets_oh.shape[0], 1, targets_oh.shape[-2], targets_oh.shape[-1],
                           dtype=torch.bool, device=targets_oh.device)
    if isinstance(ignore_index, int):
        ignore_index = [ignore_index]

    if targets_oh.shape[1] == 1:
        return torch.zeros_like(targets_oh, dtype=torch.bool)

    ig = torch.zeros_like(targets_oh[:, :1], dtype=torch.bool)  # (N,1,H,W)
    for cls in ignore_index:
        ig = ig | (targets_oh[:, cls:cls+1] > 0.5)
    return ig


def _threshold_or_argmax(probs: torch.Tensor,
                         num_classes: int,
                         threshold: float = 0.5):
    if num_classes == 1:
        pred_bin = (probs >= threshold).float()
        return pred_bin
    else:
        argmax = probs.argmax(dim=1, keepdim=True)
        onehot = torch.zeros_like(probs).scatter_(1, argmax, 1.)
        return onehot


def _per_class_confusion(pred_oh: torch.Tensor,
                         tgt_oh: torch.Tensor,
                         ignore_mask: torch.Tensor):

    keep = (~ignore_mask).float()
    pred_oh = pred_oh * keep
    tgt_oh  = tgt_oh  * keep

    tp = (pred_oh * tgt_oh).sum(dim=(0, 2, 3))
    fp = (pred_oh * (1 - tgt_oh)).sum(dim=(0, 2, 3))
    fn = ((1 - pred_oh) * tgt_oh).sum(dim=(0, 2, 3))
    tn = ((1 - pred_oh) * (1 - tgt_oh)).sum(dim=(0, 2, 3))
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-7):
    return numer / (denom + eps)


def compute_metrics(outputs: torch.Tensor,
                    targets: torch.Tensor,
                    num_classes: int,
                    threshold: float = 0.5,
                    use_logits: bool = True,
                    ignore_index: Optional[Union[int, Sequence[int]]] = None,
                    class_names: Optional[List[str]] = None
                    ):
    outputs = _ensure_tensor(outputs)
    targets = _ensure_tensor(targets)

    probs = _to_probs(outputs, num_classes=num_classes, use_logits=use_logits)
    pred_oh = _threshold_or_argmax(probs, num_classes=num_classes, threshold=threshold)

    tgt_oh = _to_onehot(targets, num_classes=num_classes)
    if num_classes == 1 and pred_oh.shape[1] != 1:
        pred_oh = pred_oh[:, :1]
    if num_classes == 1 and tgt_oh.shape[1] != 1:
        tgt_oh = tgt_oh[:, :1]

    ignore_mask = _mask_ignore(tgt_oh, ignore_index=ignore_index)

    stat = _per_class_confusion(pred_oh, tgt_oh, ignore_mask)
    tp, fp, tn, fn = stat["tp"], stat["fp"], stat["tn"], stat["fn"]

    iou   = _safe_div(tp, (tp + fp + fn))
    dice  = _safe_div(2 * tp, (2 * tp + fp + fn))
    prec  = _safe_div(tp, (tp + fp))
    rec   = _safe_div(tp, (tp + fn))
    acc_c = _safe_div(tp + tn, (tp + tn + fp + fn))

    overall_acc = _safe_div((tp + tn).sum(), (tp + tn + fp + fn).sum()).item()

    C = iou.numel()
    ignore_set = set() if ignore_index is None else set(ignore_index if isinstance(ignore_index, (list, tuple, set)) else [ignore_index])

    all_idx = list(range(C))
    keep_idx = [i for i in all_idx if i not in ignore_set]
    if len(keep_idx) == 0:
        keep_idx = all_idx

    iou_np  = iou.detach().cpu().numpy().tolist()
    dice_np = dice.detach().cpu().numpy().tolist()
    pr_np   = prec.detach().cpu().numpy().tolist()
    rc_np   = rec.detach().cpu().numpy().tolist()
    ac_np   = acc_c.detach().cpu().numpy().tolist()

    miou_incl = float(iou.mean().item())
    mdice_incl = float(dice.mean().item())
    miou_excl = float(iou[keep_idx].mean().item())
    mdice_excl = float(dice[keep_idx].mean().item())

    precision_macro = float(prec[keep_idx].mean().item())
    recall_macro    = float(rec[keep_idx].mean().item())

    f1_each = _safe_div(2 * prec * rec, (prec + rec))
    f1_macro = float(f1_each[keep_idx].mean().item())

    result = {
        "per_class": {
            "names": class_names if class_names is not None else [f"class_{i}" for i in range(C)],
            "iou": iou_np,
            "dice": dice_np,
            "prec": pr_np,
            "recall": rc_np,
            "acc": ac_np
        },
        "mean": {
            "miou_incl_ignored": miou_incl,
            "miou_excl_ignored": miou_excl,
            "mdice_incl_ignored": mdice_incl,
            "mdice_excl_ignored": mdice_excl,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "accuracy": overall_acc
        }
    }
    return result
