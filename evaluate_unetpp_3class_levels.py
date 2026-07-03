"""
Evaluate 3-class U-Net++ models from Training_unetpp_3class_levels.py.

Classes: 0=background, 1=Neg, 2=Pos.

Reports per-class Dice/recall/specificity (one-vs-rest) for Neg and Pos, plus mean foreground Dice.
Optional Otsu tissue mask (--tissue-mask otsu): metrics only on tissue pixels; outside -> class 0.
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from Training_unetpp_3class_levels import (
    BACKGROUND_CLASS,
    NEG_CLASS,
    NUM_CLASSES,
    POS_CLASS,
    PairDataset,
    build_unetpp,
    class_balanced_subset,
    collect_level_samples,
    parse_levels,
    print_all_args,
    sample_to_target_count,
    set_seed,
    split_dataset,
)
from tissue_mask import mask_predictions_outside_tissue

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOT = _SCRIPT_DIR / "Data" / "Kopiga_DB"
_DEFAULT_MODELS_ROOT = _SCRIPT_DIR / "results_unetpp_3class_levels"

VIS_SCALE = {BACKGROUND_CLASS: 0, NEG_CLASS: 127, POS_CLASS: 255}


def effective_models_root(models_root: Path, run_name: str) -> Path:
    rn = (run_name or "").strip()
    return models_root / rn if rn else models_root


def find_existing_checkpoints(models_root: Path) -> Dict[int, Path]:
    found: Dict[int, Path] = {}
    if not models_root.is_dir():
        return found
    for sub in models_root.iterdir():
        if not sub.is_dir():
            continue
        m = re.match(r"unetpp3_level(\d+)$", sub.name)
        if not m:
            continue
        p = sub / "best_model.pth"
        if p.is_file():
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def resolve_checkpoint(effective_root: Path, level: int, run_name: str) -> Optional[Path]:
    sub = effective_root / f"unetpp3_level{level}"
    p0 = sub / "best_model.pth"
    if p0.is_file():
        return p0
    rn = (run_name or "").strip()
    if rn:
        p1 = sub / f"{rn}_level{level}_best_model.pth"
        if p1.is_file():
            return p1
    return None


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _fmt_metric(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _class_metrics_one_vs_rest(
    pred: np.ndarray, gt: np.ndarray, class_id: int
) -> Tuple[Optional[float], Optional[float], Optional[float], int, int, int, int]:
    tp = int(np.sum((pred == class_id) & (gt == class_id)))
    tn = int(np.sum((pred != class_id) & (gt != class_id)))
    fp = int(np.sum((pred == class_id) & (gt != class_id)))
    fn = int(np.sum((pred != class_id) & (gt == class_id)))
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return dice, recall, specificity, tp, tn, fp, fn


def _to_uint8_image(image_t: torch.Tensor) -> np.ndarray:
    arr = image_t.detach().cpu().numpy().astype(np.float32)
    c = arr.shape[0]
    if c == 1:
        mean = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
        std = np.array([0.5], dtype=np.float32).reshape(1, 1, 1)
        arr = (arr * std) + mean
        arr = np.clip(arr, 0.0, 1.0)
        gray = (arr[0] * 255.0).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr * std) + mean
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0).astype(np.uint8)


def _class_map_to_vis(class_map: np.ndarray) -> np.ndarray:
    out = np.zeros(class_map.shape, dtype=np.uint8)
    for cls, val in VIS_SCALE.items():
        out[class_map == cls] = val
    return out


def _make_colored_overlay(image_u8: np.ndarray, class_map: np.ndarray, alpha: int = 110) -> np.ndarray:
    base = Image.fromarray(image_u8, mode="RGB").convert("RGBA")
    overlay = np.zeros((class_map.shape[0], class_map.shape[1], 4), dtype=np.uint8)
    overlay[class_map == NEG_CLASS] = [0, 0, 255, alpha]
    overlay[class_map == POS_CLASS] = [255, 0, 0, alpha]
    composed = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA")).convert("RGB")
    return np.array(composed)


def _write_single_prediction_pngs(
    out_dir: Path,
    basename: str,
    image_t_row: torch.Tensor,
    gt_class_map: np.ndarray,
    pred_class_map: np.ndarray,
    *,
    rgb_basename_suffix: str,
    save_overlays: bool,
) -> None:
    img_u8 = _to_uint8_image(image_t_row)
    gt_u8 = _class_map_to_vis(gt_class_map)
    pred_u8 = _class_map_to_vis(pred_class_map)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(out_dir / f"{basename}_{rgb_basename_suffix}.png")
    Image.fromarray(gt_u8, mode="L").save(out_dir / f"{basename}_mask_gt.png")
    Image.fromarray(pred_u8, mode="L").save(out_dir / f"{basename}_mask_pred.png")
    if save_overlays:
        Image.fromarray(_make_colored_overlay(img_u8, pred_class_map), mode="RGB").save(
            out_dir / f"{basename}_overlay_pred_on_image.png"
        )


def evaluate_test_split(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    test_samples,
    preview_dir: Optional[Path],
    num_preview: int,
    export_all_dir: Optional[Path],
    export_all_include_overlays: bool,
    tissue_mask_method: str,
    otsu_min_size: int,
) -> Dict[str, object]:
    model.eval()
    stats = {
        "neg": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "pos": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
        "pixel_correct": 0,
        "pixel_total": 0,
    }
    preview_written = 0
    sample_offset = 0
    if preview_dir is not None and num_preview > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
    if export_all_dir is not None:
        export_all_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="test", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            if isinstance(out, list):
                out = out[0]
            pred = torch.argmax(out, dim=1)
            pred_np = pred.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            bs = pred_np.shape[0]

            for i in range(bs):
                pred_map = pred_np[i]
                gt_map = y_np[i]
                tissue = None
                if tissue_mask_method == "otsu":
                    img_u8 = _to_uint8_image(x[i].detach().cpu())
                    pred_map, gt_map, tissue = mask_predictions_outside_tissue(
                        pred_map, gt_map, img_u8, min_size=otsu_min_size
                    )

                if tissue is not None and not np.any(tissue):
                    continue
                    p = pred_map[tissue].ravel()
                    t = gt_map[tissue].ravel()
                else:
                    p = pred_map.ravel()
                    t = gt_map.ravel()

                stats["pixel_correct"] += int(np.sum(p == t))
                stats["pixel_total"] += int(p.size)

                for key, cls in (("neg", NEG_CLASS), ("pos", POS_CLASS)):
                    d, r, s, tp, tn, fp, fn = _class_metrics_one_vs_rest(p, t, cls)
                    stats[key]["tp"] += tp
                    stats[key]["tn"] += tn
                    stats[key]["fp"] += fp
                    stats[key]["fn"] += fn

                sample = test_samples[sample_offset + i]
                stem = sample.image_path.stem
                if export_all_dir is not None:
                    _write_single_prediction_pngs(
                        export_all_dir,
                        stem,
                        x[i].detach().cpu(),
                        gt_map,
                        pred_map,
                        rgb_basename_suffix="input",
                        save_overlays=export_all_include_overlays,
                    )
                if preview_dir is not None and num_preview > 0 and preview_written < num_preview:
                    prefix = f"{preview_written + 1:02d}_{stem}"
                    _write_single_prediction_pngs(
                        preview_dir,
                        prefix,
                        x[i].detach().cpu(),
                        gt_map,
                        pred_map,
                        rgb_basename_suffix="image",
                        save_overlays=True,
                    )
                    preview_written += 1

            sample_offset += bs

    def _finalize(prefix: str, class_id: int) -> Dict[str, Optional[float]]:
        s = stats[prefix]
        tp, tn, fp, fn = s["tp"], s["tn"], s["fp"], s["fn"]
        return {
            "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
            "recall": _safe_div(tp, tp + fn),
            "specificity": _safe_div(tn, tn + fp),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    neg_m = _finalize("neg", NEG_CLASS)
    pos_m = _finalize("pos", POS_CLASS)
    dice_vals = [m["dice"] for m in (neg_m, pos_m) if m["dice"] is not None]
    mean_fg_dice = float(np.mean(dice_vals)) if dice_vals else None
    accuracy = _safe_div(stats["pixel_correct"], stats["pixel_total"])

    return {
        "neg": neg_m,
        "pos": pos_m,
        "mean_fg_dice": mean_fg_dice,
        "accuracy": accuracy,
        "pixel_total": stats["pixel_total"],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3-class U-Net++ Neg/Pos/background models.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--models-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--eval-from", choices=("train_split", "test_folders"), default="test_folders")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--pos-dir", type=str, default="Pos")
    parser.add_argument("--neg-dir", type=str, default="Neg")
    parser.add_argument("--train-images-dir", type=str, default="Train_images")
    parser.add_argument("--train-masks-dir", type=str, default="Train_masks")
    parser.add_argument("--test-images-dir", type=str, default="Test_images")
    parser.add_argument("--test-masks-dir", type=str, default="Test_masks")
    parser.add_argument("--balance-level", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--in-channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-mask-examples", type=int, default=5)
    parser.add_argument("--save-all-test-outputs", action="store_true")
    parser.add_argument("--save-all-include-overlays", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--tissue-mask", choices=("none", "otsu"), default="otsu")
    parser.add_argument("--otsu-min-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    if not args.save_all_test_outputs and args.save_all_include_overlays:
        raise ValueError("--save-all-include-overlays requires --save-all-test-outputs")

    selected_levels = parse_levels(args.levels)
    balance_level = args.balance_level
    if args.eval_from == "train_split" and balance_level is None:
        balance_level = max(selected_levels)

    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else _DEFAULT_DATA_ROOT
    models_root = Path(args.models_root) if args.models_root else _DEFAULT_MODELS_ROOT
    if args.in_channels == 1 and args.models_root is None:
        gray_root = _SCRIPT_DIR / "results_unetpp_3class_levels_gray"
        if gray_root.is_dir():
            models_root = gray_root
    effective_models = effective_models_root(models_root, args.run_name)
    run_name = args.run_name.strip()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_all_args(
        args,
        extra={
            "resolved_data_root": str(data_root.resolve()),
            "resolved_models_root": str(models_root.resolve()),
            "effective_checkpoints_dir": str(effective_models.resolve()),
            "selected_levels": selected_levels,
            "balance_level": balance_level,
            "device": str(device),
            "num_classes": NUM_CLASSES,
        },
    )

    if args.eval_from == "train_split":
        levels_to_collect = sorted(set(selected_levels + [balance_level]))
        all_samples = {
            lvl: collect_level_samples(
                data_root, lvl, args.pos_dir, args.neg_dir, args.train_images_dir, args.train_masks_dir
            )
            for lvl in levels_to_collect
        }
        ref_balanced = class_balanced_subset(all_samples[balance_level], seed=args.seed + balance_level)
        target_count = len(ref_balanced)
        if target_count == 0:
            raise ValueError(f"level{balance_level} has no class-balanced train samples.")

    pin = torch.cuda.is_available() and args.num_workers > 0
    rows: List[Dict] = []

    for lvl in selected_levels:
        ckpt_path = resolve_checkpoint(effective_models, lvl, run_name)
        if ckpt_path is None or not ckpt_path.is_file():
            expected = effective_models / f"unetpp3_level{lvl}" / "best_model.pth"
            if args.skip_missing:
                print(f"SKIP: Missing checkpoint for level {lvl}: {expected}")
                continue
            raise FileNotFoundError(f"Missing checkpoint: {expected}")

        if args.eval_from == "train_split":
            samples = sample_to_target_count(all_samples[lvl], target_count, seed=args.seed + lvl)
            ds = PairDataset(
                samples, target_size=(args.image_size, args.image_size), augment=False, in_channels=args.in_channels
            )
            split_seed = args.seed + lvl
            _, _, test_ds = split_dataset(ds, seed=split_seed)
            subset_indices = getattr(test_ds, "indices", list(range(len(test_ds))))
            test_samples = [ds.samples[i] for i in subset_indices]
            split_seed_out = split_seed
        else:
            try:
                test_samples = collect_level_samples(
                    data_root, lvl, args.pos_dir, args.neg_dir, args.test_images_dir, args.test_masks_dir
                )
            except FileNotFoundError as e:
                if args.skip_missing:
                    print(f"SKIP level{lvl}: {e}")
                    continue
                raise
            if not test_samples:
                msg = f"No test PNGs in Pos/Neg level{lvl}"
                if args.skip_missing:
                    print(f"SKIP: {msg}")
                    continue
                raise ValueError(msg)
            ds = PairDataset(
                test_samples, target_size=(args.image_size, args.image_size), augment=False, in_channels=args.in_channels
            )
            split_seed_out = ""

        test_loader = DataLoader(
            test_ds if args.eval_from == "train_split" else ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
        )
        test_ds_len = len(test_ds if args.eval_from == "train_split" else ds)

        model = build_unetpp(in_channels=args.in_channels, out_channels=NUM_CLASSES).to(device)
        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=True)

        examples_subdir = "examples_test_folders" if args.eval_from == "test_folders" else "test_mask_examples"
        preview_dir = effective_models / f"unetpp3_level{lvl}" / examples_subdir if args.num_mask_examples > 0 else None
        export_all_dir = effective_models / f"unetpp3_level{lvl}" / "test_export_all" if args.save_all_test_outputs else None

        metrics = evaluate_test_split(
            model=model,
            test_loader=test_loader,
            device=device,
            test_samples=test_samples,
            preview_dir=preview_dir,
            num_preview=args.num_mask_examples,
            export_all_dir=export_all_dir,
            export_all_include_overlays=args.save_all_include_overlays,
            tissue_mask_method=args.tissue_mask,
            otsu_min_size=args.otsu_min_size,
        )

        neg_m = metrics["neg"]
        pos_m = metrics["pos"]
        print(
            f"[level{lvl}] test n={test_ds_len} tiles | Acc={_fmt_metric(metrics['accuracy'])} "
            f"MeanFgDice={_fmt_metric(metrics['mean_fg_dice'])} | "
            f"Neg Dice={_fmt_metric(neg_m['dice'])} Pos Dice={_fmt_metric(pos_m['dice'])}"
        )
        rows.append(
            {
                "level": lvl,
                "test_size": test_ds_len,
                "split_seed": split_seed_out,
                "accuracy": metrics["accuracy"],
                "mean_fg_dice": metrics["mean_fg_dice"],
                "neg_dice": neg_m["dice"],
                "neg_recall": neg_m["recall"],
                "neg_specificity": neg_m["specificity"],
                "pos_dice": pos_m["dice"],
                "pos_recall": pos_m["recall"],
                "pos_specificity": pos_m["specificity"],
                "neg_tp": neg_m["tp"],
                "neg_tn": neg_m["tn"],
                "neg_fp": neg_m["fp"],
                "neg_fn": neg_m["fn"],
                "pos_tp": pos_m["tp"],
                "pos_tn": pos_m["tn"],
                "pos_fp": pos_m["fp"],
                "pos_fn": pos_m["fn"],
            }
        )

    if not rows:
        print("No levels evaluated.", file=sys.stderr)
        sys.exit(1)

    def _mean_key(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    suffix = "_test_folders" if args.eval_from == "test_folders" else "_train_split"
    tissue_tag = "" if args.tissue_mask == "none" else f"_{args.tissue_mask}"
    rn_tag = f"_{run_name}" if run_name else ""
    out_csv = effective_models / f"eval_metrics_3class{rn_tag}{suffix}{tissue_tag}.csv"

    fields = [
        "level", "test_size", "split_seed", "accuracy", "mean_fg_dice",
        "neg_dice", "neg_recall", "neg_specificity",
        "pos_dice", "pos_recall", "pos_specificity",
        "neg_tp", "neg_tn", "neg_fp", "neg_fn",
        "pos_tp", "pos_tn", "pos_fp", "pos_fn",
    ]
    float_fields = [
        "accuracy", "mean_fg_dice", "neg_dice", "neg_recall", "neg_specificity",
        "pos_dice", "pos_recall", "pos_specificity",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **{k: r[k] for k in fields if k not in float_fields},
                    **{k: "" if r[k] is None else f"{r[k]:.8f}" for k in float_fields},
                }
            )
        w.writerow(
            {
                "level": "ALL_DEFINED_MEAN",
                "test_size": sum(r["test_size"] for r in rows),
                "split_seed": "",
                **{k: "" if _mean_key(k) is None else f"{_mean_key(k):.8f}" for k in float_fields},
                **{k: sum(r[k] for r in rows) for k in fields if k.endswith("_tp") or k.endswith("_tn") or k.endswith("_fp") or k.endswith("_fn")},
            }
        )

    print(
        f"Final mean | Acc={_fmt_metric(_mean_key('accuracy'))} "
        f"MeanFgDice={_fmt_metric(_mean_key('mean_fg_dice'))} "
        f"NegDice={_fmt_metric(_mean_key('neg_dice'))} PosDice={_fmt_metric(_mean_key('pos_dice'))}"
    )
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
