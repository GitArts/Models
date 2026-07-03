"""
Evaluate trained U-Net++ models from Training_unetpp_levels.py.

Three data sources (--eval-from):

- **train_split** (default): same Train_images/Train_masks, balancing, 90/5/5 split, and
  per-level seed as training, so the "test" set matches the script-side test fold.

- **test_folders**: all held-out tiles under levelN/Test_images and levelN/Test_masks.

- **both**: run train_split and test_folders in one invocation (two CSV files).

Metrics (binary foreground = class 1): Recall, Specificity, Dice (micro over all pixels).

By default each tile is masked with Otsu tissue detection before metrics: predictions and GT
outside tissue are cleared, and TP/TN/FP/FN are accumulated only on tissue pixels.
Use --tissue-mask segmenter for SlideSegmenter (NN; better when pen markings are present).
Use --tissue-mask none to evaluate on all pixels (legacy behaviour).

Optional: --save-all-test-outputs writes every tile's model input, GT mask, and predicted mask
under unetpp_levelN/test_export_all/.
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from Training_unetpp_levels import (
    PairDataset,
    PairSample,
    build_unetpp,
    collect_level_samples,
    parse_levels,
    sample_balanced,
    set_seed,
    split_dataset,
)
from tissue_mask import (
    add_tissue_mask_cli_args,
    format_tissue_mask_args,
    mask_predictions_outside_tissue,
    validate_tissue_mask_args,
)

# Defaults relative to this file so cwd does not matter.
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA_ROOT = _SCRIPT_DIR.parent / "Results" / "Dysplasia"
_DEFAULT_MODELS_ROOT = _SCRIPT_DIR / "results_unetpp_levels"


def effective_models_root(models_root: Path, run_name: str) -> Path:
    rn = (run_name or "").strip()
    return models_root / rn if rn else models_root


def find_existing_checkpoints(models_root: Path) -> Dict[int, Path]:
    """Map level -> best_model.pth path for folders unetpp_level{N} under models_root."""
    found: Dict[int, Path] = {}
    if not models_root.is_dir():
        return found
    for sub in models_root.iterdir():
        if not sub.is_dir():
            continue
        m = re.match(r"unetpp_level(\d+)$", sub.name)
        if not m:
            continue
        p = sub / "best_model.pth"
        if p.is_file():
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def resolve_checkpoint(effective_root: Path, level: int, run_name: str) -> Optional[Path]:
    """Prefer best_model.pth; then {RUN}_level{N}_best_model.pth if run_name is set."""
    sub = effective_root / f"unetpp_level{level}"
    p0 = sub / "best_model.pth"
    if p0.is_file():
        return p0
    rn = (run_name or "").strip()
    if rn:
        p1 = sub / f"{rn}_level{level}_best_model.pth"
        if p1.is_file():
            return p1
    return None


def _safe_div(num: int, den: int) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def _fmt_metric(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.6f}"


def _to_uint8_image(image_t: torch.Tensor) -> np.ndarray:
    """Undo PairDataset normalization; return HxWx3 uint8 RGB for overlays."""
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


def _make_pred_overlay(image_u8: np.ndarray, pred_mask_u8: np.ndarray, alpha: int = 110) -> np.ndarray:
    """Create RGB overlay: predicted mask in red over image."""
    base = Image.fromarray(image_u8, mode="RGB").convert("RGBA")
    overlay_arr = np.zeros((pred_mask_u8.shape[0], pred_mask_u8.shape[1], 4), dtype=np.uint8)
    pred_on = pred_mask_u8 > 0
    overlay_arr[pred_on] = [255, 0, 0, alpha]
    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(composed)


def _make_gt_pred_comparison(gt_mask_u8: np.ndarray, pred_mask_u8: np.ndarray) -> np.ndarray:
    """
    Create RGB comparison map:
    - GT only   -> green
    - Pred only -> red
    - Both      -> yellow
    - Neither   -> black
    """
    gt_on = gt_mask_u8 > 0
    pred_on = pred_mask_u8 > 0
    both = gt_on & pred_on
    gt_only = gt_on & ~pred_on
    pred_only = pred_on & ~gt_on

    out = np.zeros((gt_mask_u8.shape[0], gt_mask_u8.shape[1], 3), dtype=np.uint8)
    out[gt_only] = [0, 255, 0]
    out[pred_only] = [255, 0, 0]
    out[both] = [255, 255, 0]
    return out


def _write_single_prediction_pngs(
    out_dir: Path,
    basename: str,
    image_t_row: torch.Tensor,
    gt_class_map: np.ndarray,
    pred_class_map: np.ndarray,
    *,
    rgb_basename_suffix: str = "input",
    save_overlays: bool,
) -> None:
    """Save one tile: RGB (as seen by model), GT mask, pred mask; optionally overlay + comparison."""
    img_u8 = _to_uint8_image(image_t_row)
    gt_u8 = (gt_class_map.astype(np.uint8) * 255)
    pred_u8 = (pred_class_map.astype(np.uint8) * 255)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(out_dir / f"{basename}_{rgb_basename_suffix}.png")
    Image.fromarray(gt_u8, mode="L").save(out_dir / f"{basename}_mask_gt.png")
    Image.fromarray(pred_u8, mode="L").save(out_dir / f"{basename}_mask_pred.png")
    if save_overlays:
        pred_overlay_u8 = _make_pred_overlay(img_u8, pred_u8)
        gt_pred_cmp_u8 = _make_gt_pred_comparison(gt_u8, pred_u8)
        Image.fromarray(pred_overlay_u8, mode="RGB").save(out_dir / f"{basename}_overlay_pred_on_image.png")
        Image.fromarray(gt_pred_cmp_u8, mode="RGB").save(out_dir / f"{basename}_compare_gt_pred.png")


def _accumulate_confusion(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    tissue: Optional[np.ndarray],
) -> Tuple[int, int, int, int]:
    """TP/TN/FP/FN on tissue pixels only when tissue mask is provided."""
    if tissue is not None:
        if not np.any(tissue):
            return 0, 0, 0, 0
        p = pred_map[tissue].ravel()
        t = gt_map[tissue].ravel()
    else:
        p = pred_map.ravel()
        t = gt_map.ravel()
    tp = int(np.sum((p == 1) & (t == 1)))
    tn = int(np.sum((p == 0) & (t == 0)))
    fp = int(np.sum((p == 1) & (t == 0)))
    fn = int(np.sum((p == 0) & (t == 1)))
    return tp, tn, fp, fn


def _dice_strict(tp: int, fp: int, fn: int) -> Optional[float]:
    """Standard Dice; undefined when TP, FP, FN are all zero (empty GT and prediction)."""
    den = (2 * tp) + fp + fn
    if den == 0:
        return None
    return (2 * tp) / den


def _dice_empty_one(tp: int, fp: int, fn: int) -> float:
    """Dice with empty-empty tiles scored as 1.0."""
    den = (2 * tp) + fp + fn
    if den == 0:
        return 1.0
    return (2 * tp) / den


def _mean_defined(values: List[Optional[float]]) -> Optional[float]:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


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
    segmenter_min_size: int,
    segmenter_model_folder: str,
    segmenter_device: Optional[str],
) -> Dict[str, object]:
    """Return metrics dict with micro + mean-per-tile Dice (strict and empty=1)."""
    model.eval()
    tp = tn = fp = fn = 0
    tile_dice_strict: List[Optional[float]] = []
    tile_dice_empty_one: List[float] = []
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
                if tissue_mask_method != "none":
                    img_u8 = _to_uint8_image(x[i].detach().cpu())
                    pred_map, gt_map, tissue = mask_predictions_outside_tissue(
                        pred_map,
                        gt_map,
                        img_u8,
                        method=tissue_mask_method,
                        min_size=otsu_min_size,
                        segmenter_min_size=segmenter_min_size,
                        segmenter_model_folder=segmenter_model_folder,
                        segmenter_device=segmenter_device,
                    )

                tile_tp, tile_tn, tile_fp, tile_fn = _accumulate_confusion(pred_map, gt_map, tissue)
                tp += tile_tp
                tn += tile_tn
                fp += tile_fp
                fn += tile_fn
                tile_dice_strict.append(_dice_strict(tile_tp, tile_fp, tile_fn))
                tile_dice_empty_one.append(_dice_empty_one(tile_tp, tile_fp, tile_fn))

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

    dice_undefined_tiles = sum(1 for d in tile_dice_strict if d is None)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    return {
        "recall": recall,
        "specificity": specificity,
        "dice_micro": _dice_strict(tp, fp, fn),
        "dice_micro_empty1": _dice_empty_one(tp, fp, fn),
        "dice_mean_excl_empty": _mean_defined(tile_dice_strict),
        "dice_mean_empty1": _mean_defined(tile_dice_empty_one),
        "dice_undefined_tiles": dice_undefined_tiles,
        "tile_count": len(tile_dice_strict),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _undefined_dice_summary(undefined: int, total: int) -> str:
    if total == 0:
        return "Strict Dice undefined: 0 out of 0 tiles"
    pct = 100.0 * undefined / total
    return f"Strict Dice undefined: {undefined} out of {total} tiles ({pct:.2f}%)"


def resolve_eval_modes(eval_from: str) -> List[str]:
    if eval_from == "both":
        return ["train_split", "test_folders"]
    return [eval_from]


def train_split_test_samples(
    all_samples: Dict[int, List[PairSample]],
    lvl: int,
    target_count: int,
    balance_level: int,
    seed: int,
    image_size: int,
    in_channels: int,
) -> List[PairSample]:
    samples = sample_balanced(
        samples=all_samples[lvl],
        target_count=target_count,
        seed=seed + lvl,
        enforce_positive_priority=(lvl != balance_level),
    )
    ds = PairDataset(
        samples,
        target_size=(image_size, image_size),
        augment=False,
        in_channels=in_channels,
    )
    _, _, test_ds = split_dataset(ds, seed=seed + lvl)
    subset_indices = getattr(test_ds, "indices", list(range(len(test_ds))))
    return [ds.samples[i] for i in subset_indices]


def test_folder_samples(
    data_root: Path,
    lvl: int,
    test_images_dir: str,
    test_masks_dir: str,
) -> List[PairSample]:
    return collect_level_samples(
        data_root,
        lvl,
        images_subdir=test_images_dir,
        masks_subdir=test_masks_dir,
    )


def eval_mode_suffix(eval_mode: str) -> str:
    if eval_mode == "test_folders":
        return "_test_folders"
    return "_train_split"


def eval_mode_output_dirs(eval_mode: str) -> Tuple[str, str]:
    if eval_mode == "test_folders":
        return "examples_test_folders", "test_export_all"
    return "test_mask_examples", "test_export_all_train_split"


def build_test_loader(
    samples: List[PairSample],
    image_size: int,
    in_channels: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[PairSample], int]:
    ds = PairDataset(
        samples,
        target_size=(image_size, image_size),
        augment=False,
        in_channels=in_channels,
    )
    pin = torch.cuda.is_available() and num_workers > 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return loader, ds.samples, len(ds)


def write_metrics_csv(
    rows: List[Dict],
    out_csv: Path,
    *,
    final_recall: Optional[float],
    final_spec: Optional[float],
    final_dice_micro: Optional[float],
    final_dice_micro_empty1: Optional[float],
    final_dice_mean_excl_empty: Optional[float],
    final_dice_mean_empty1: Optional[float],
) -> None:
    tp_sum = int(sum(r["tp"] for r in rows))
    tn_sum = int(sum(r["tn"] for r in rows))
    fp_sum = int(sum(r["fp"] for r in rows))
    fn_sum = int(sum(r["fn"] for r in rows))
    fields = [
        "level",
        "test_size",
        "split_seed",
        "recall",
        "specificity",
        "dice_micro",
        "dice_micro_empty1",
        "dice_mean_excl_empty",
        "dice_mean_empty1",
        "dice_undefined_tiles",
        "tile_count",
        "tp",
        "tn",
        "fp",
        "fn",
    ]
    float_fields = [
        "recall",
        "specificity",
        "dice_micro",
        "dice_micro_empty1",
        "dice_mean_excl_empty",
        "dice_mean_empty1",
    ]

    def _fmt(v: Optional[float]) -> str:
        return "" if v is None else f"{v:.8f}"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "level": r["level"],
                    "test_size": r["test_size"],
                    "split_seed": r["split_seed"],
                    **{k: _fmt(r[k]) for k in float_fields},
                    "dice_undefined_tiles": r["dice_undefined_tiles"],
                    "tile_count": r["tile_count"],
                    "tp": r["tp"],
                    "tn": r["tn"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                }
            )
        w.writerow(
            {
                "level": "ALL_DEFINED_MEAN",
                "test_size": sum(r["test_size"] for r in rows),
                "split_seed": "",
                "recall": _fmt(final_recall),
                "specificity": _fmt(final_spec),
                "dice_micro": _fmt(final_dice_micro),
                "dice_micro_empty1": _fmt(final_dice_micro_empty1),
                "dice_mean_excl_empty": _fmt(final_dice_mean_excl_empty),
                "dice_mean_empty1": _fmt(final_dice_mean_empty1),
                "dice_undefined_tiles": sum(r["dice_undefined_tiles"] for r in rows),
                "tile_count": sum(r["tile_count"] for r in rows),
                "tp": tp_sum,
                "tn": tn_sum,
                "fp": fp_sum,
                "fn": fn_sum,
            }
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate U-Net++ level models (train-split test fold or held-out Test_images/Test_masks)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help=f"Dataset root with levelN folders. Default: {_DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=None,
        help=f"Training --out-root (folder containing unetpp_level*). Default: {_DEFAULT_MODELS_ROOT}",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Same as training --run-name (checkpoints under models-root/run-name/).",
    )
    parser.add_argument(
        "--eval-from",
        choices=("train_split", "test_folders", "both"),
        default="train_split",
        help="train_split: 5%% test fold from Train_*. test_folders: all Test_* tiles. "
        "both: train_split and test_folders (two CSV files).",
    )
    parser.add_argument("--levels", type=str, default="0,1,2,3,4,5")
    parser.add_argument(
        "--balance-level",
        type=int,
        default=None,
        help="For train_split only: reference level count (default: max of --levels). Ignored for test_folders.",
    )
    parser.add_argument(
        "--test-images-dir",
        type=str,
        default="Test_images",
        help="For test_folders: image subdir under levelN.",
    )
    parser.add_argument(
        "--test-masks-dir",
        type=str,
        default="Test_masks",
        help="For test_folders: mask subdir under levelN.",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--in-channels",
        type=int,
        choices=(1, 3),
        default=3,
        help="Must match training (1 = grayscale PairDataset).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Same base seed as training.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu; default: auto")
    parser.add_argument(
        "--num-mask-examples",
        type=int,
        default=5,
        help="How many preview examples (first N tiles) to save under examples_test_folders/ or test_mask_examples/.",
    )
    parser.add_argument(
        "--save-all-test-outputs",
        action="store_true",
        help="Save every analysed test tile: {stem}_input.png, {stem}_mask_gt.png, {stem}_mask_pred.png under test_export_all/.",
    )
    parser.add_argument(
        "--save-all-include-overlays",
        action="store_true",
        help="With --save-all-test-outputs, also save overlay and GT-vs-pred comparison for each tile (more disk).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip levels whose best_model.pth is missing instead of failing.",
    )
    add_tissue_mask_cli_args(parser)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}")
    validate_tissue_mask_args(args)

    eval_modes = resolve_eval_modes(args.eval_from)
    selected_levels = parse_levels(args.levels)
    balance_level = args.balance_level
    needs_train_samples = "train_split" in eval_modes
    if needs_train_samples:
        if balance_level is None:
            balance_level = max(selected_levels)
        if balance_level < 0:
            raise ValueError(f"--balance-level must be non-negative, got {balance_level}")

    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else _DEFAULT_DATA_ROOT
    models_root = Path(args.models_root) if args.models_root else _DEFAULT_MODELS_ROOT
    effective_models = effective_models_root(models_root, args.run_name)
    run_name = args.run_name.strip()

    print(f"data-root: {data_root.resolve()}")
    print(f"models-root: {models_root.resolve()}")
    print(f"eval-from: {args.eval_from} -> modes: {eval_modes}")
    if run_name:
        print(f"effective checkpoints dir: {effective_models.resolve()}")

    existing_ckpt = find_existing_checkpoints(effective_models)
    if existing_ckpt:
        print(f"Checkpoints found for levels: {list(existing_ckpt.keys())}")
    else:
        print(
            "No best_model.pth found under the effective models directory. "
            "Train with Training_unetpp_levels.py or pass --models-root / --run-name."
        )

    if needs_train_samples:
        assert balance_level is not None
        levels_to_collect = sorted(set(selected_levels + [balance_level]))
        all_samples = {lvl: collect_level_samples(data_root, lvl) for lvl in levels_to_collect}
        target_count = len(all_samples[balance_level])
        if target_count == 0:
            raise ValueError(f"level{balance_level} has zero pairs in Train_images/Train_masks.")
    else:
        all_samples = {}
        target_count = 0

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"in_channels (must match checkpoints): {args.in_channels}")
    print(f"tissue-mask: {format_tissue_mask_args(args)}")
    if needs_train_samples:
        print(f"Balance target count (level{balance_level}): {target_count}")

    if args.num_mask_examples < 0:
        raise ValueError(f"--num-mask-examples must be >= 0, got {args.num_mask_examples}")
    if not args.save_all_test_outputs and args.save_all_include_overlays:
        raise ValueError("--save-all-include-overlays requires --save-all-test-outputs")

    tissue_tag = "" if args.tissue_mask == "none" else f"_{args.tissue_mask}"
    rn_tag = f"_{run_name}" if run_name else ""
    any_rows = False
    total_dice_undefined = 0
    total_tile_count = 0

    for eval_mode in eval_modes:
        print(f"\n=== Evaluation mode: {eval_mode} ===")
        rows: List[Dict] = []
        for lvl in selected_levels:
            ckpt_path = resolve_checkpoint(effective_models, lvl, run_name)
            if ckpt_path is None or not ckpt_path.is_file():
                if lvl in existing_ckpt:
                    ckpt_path = existing_ckpt[lvl]
                else:
                    expected = effective_models / f"unetpp_level{lvl}" / "best_model.pth"
                    msg = f"Missing checkpoint for level {lvl}: {expected}"
                    if args.skip_missing:
                        print(f"SKIP: {msg}")
                        continue
                    raise FileNotFoundError(
                        f"{msg}\n"
                        f"Looked under: {effective_models.resolve()}\n"
                        f"Use --models-root (training --out-root) and --run-name if you used one. "
                        f"Found checkpoints only for: {list(existing_ckpt.keys())}"
                    )

            if eval_mode == "train_split":
                test_samples = train_split_test_samples(
                    all_samples,
                    lvl,
                    target_count,
                    balance_level,
                    args.seed,
                    args.image_size,
                    args.in_channels,
                )
                split_seed_out = args.seed + lvl
            elif eval_mode == "test_folders":
                try:
                    test_samples = test_folder_samples(
                        data_root, lvl, args.test_images_dir, args.test_masks_dir
                    )
                except FileNotFoundError as e:
                    if args.skip_missing:
                        print(f"SKIP level{lvl}: {e}")
                        continue
                    raise
                if not test_samples:
                    msg = f"No paired PNGs in level{lvl} {args.test_images_dir}/ {args.test_masks_dir}"
                    if args.skip_missing:
                        print(f"SKIP: {msg}")
                        continue
                    raise ValueError(msg)
                split_seed_out = ""
            else:
                raise ValueError(f"Unknown evaluation mode: {eval_mode}")

            test_loader, test_samples, test_ds_len = build_test_loader(
                test_samples,
                args.image_size,
                args.in_channels,
                args.batch_size,
                args.num_workers,
            )

            model = build_unetpp(in_channels=args.in_channels).to(device)
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=True)

            examples_subdir, export_subdir = eval_mode_output_dirs(eval_mode)
            preview_dir = (
                effective_models / f"unetpp_level{lvl}" / examples_subdir if args.num_mask_examples > 0 else None
            )
            export_all_dir = (
                effective_models / f"unetpp_level{lvl}" / export_subdir if args.save_all_test_outputs else None
            )
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
                segmenter_min_size=args.segmenter_min_size,
                segmenter_model_folder=args.segmenter_model_folder,
                segmenter_device=args.segmenter_device,
            )
            print(
                f"[{eval_mode} level{lvl}] test n={test_ds_len} | Recall={_fmt_metric(metrics['recall'])} "
                f"Specificity={_fmt_metric(metrics['specificity'])} | "
                f"DiceMicro={_fmt_metric(metrics['dice_micro'])} "
                f"DiceMicroEmpty1={_fmt_metric(metrics['dice_micro_empty1'])} | "
                f"DiceMeanExclEmpty={_fmt_metric(metrics['dice_mean_excl_empty'])} "
                f"DiceMeanEmpty1={_fmt_metric(metrics['dice_mean_empty1'])} | "
                f"{_undefined_dice_summary(metrics['dice_undefined_tiles'], metrics['tile_count'])} "
                f"(TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']})"
            )
            if args.num_mask_examples > 0:
                print(f"[{eval_mode} level{lvl}] Saved up to {args.num_mask_examples} preview examples in: {preview_dir}")
            if args.save_all_test_outputs:
                print(f"[{eval_mode} level{lvl}] Saved all {test_ds_len} test tiles to: {export_all_dir}")
            rows.append(
                {
                    "level": lvl,
                    "test_size": test_ds_len,
                    "split_seed": split_seed_out,
                    "recall": metrics["recall"],
                    "specificity": metrics["specificity"],
                    "dice_micro": metrics["dice_micro"],
                    "dice_micro_empty1": metrics["dice_micro_empty1"],
                    "dice_mean_excl_empty": metrics["dice_mean_excl_empty"],
                    "dice_mean_empty1": metrics["dice_mean_empty1"],
                    "dice_undefined_tiles": metrics["dice_undefined_tiles"],
                    "tile_count": metrics["tile_count"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                }
            )

        if not rows:
            print(f"No levels evaluated for {eval_mode} (missing checkpoints or all skipped).", file=sys.stderr)
            continue

        any_rows = True
        mode_undefined = sum(r["dice_undefined_tiles"] for r in rows)
        mode_tiles = sum(r["tile_count"] for r in rows)
        total_dice_undefined += mode_undefined
        total_tile_count += mode_tiles
        def _mean_key(key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(vals)) if vals else None

        final_recall = _mean_key("recall")
        final_spec = _mean_key("specificity")
        final_dice_micro = _mean_key("dice_micro")
        final_dice_micro_empty1 = _mean_key("dice_micro_empty1")
        final_dice_mean_excl_empty = _mean_key("dice_mean_excl_empty")
        final_dice_mean_empty1 = _mean_key("dice_mean_empty1")

        suffix = eval_mode_suffix(eval_mode)
        out_csv = effective_models / f"eval_metrics_all_levels{rn_tag}{suffix}{tissue_tag}.csv"
        write_metrics_csv(
            rows,
            out_csv,
            final_recall=final_recall,
            final_spec=final_spec,
            final_dice_micro=final_dice_micro,
            final_dice_micro_empty1=final_dice_micro_empty1,
            final_dice_mean_excl_empty=final_dice_mean_excl_empty,
            final_dice_mean_empty1=final_dice_mean_empty1,
        )
        print(
            f"Final [{eval_mode}] (mean across levels) | "
            f"Recall={_fmt_metric(final_recall)} Specificity={_fmt_metric(final_spec)} | "
            f"DiceMicro={_fmt_metric(final_dice_micro)} DiceMicroEmpty1={_fmt_metric(final_dice_micro_empty1)} | "
            f"DiceMeanExclEmpty={_fmt_metric(final_dice_mean_excl_empty)} "
            f"DiceMeanEmpty1={_fmt_metric(final_dice_mean_empty1)} | "
            f"{_undefined_dice_summary(mode_undefined, mode_tiles)}"
        )
        print(f"Saved: {out_csv}")

    if not any_rows:
        sys.exit(1)

    print(_undefined_dice_summary(total_dice_undefined, total_tile_count))


if __name__ == "__main__":
    main()
