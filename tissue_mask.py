"""
Tissue detection helpers for slide / tile pipelines.

Otsu: default for slides without pen markers (dark ink is indistinguishable from tissue).
SlideSegmenter: use when pen markings are present (--tissue-mask segmenter).
Requires: pip install git+https://github.com/RTLucassen/slidesegmenter
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple, Union

import numpy as np
from skimage import filters, morphology
from skimage.color import rgb2gray

import inspect

ArrayLike = Union[np.ndarray, "np.typing.NDArray"]

_USE_SKIMAGE_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_objects).parameters

_segmenter_cache: Dict[str, object] = {}


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components with area < min_size (skimage 0.26+ uses max_size)."""
    if _USE_SKIMAGE_MAX_SIZE:
        return morphology.remove_small_objects(mask, max_size=max(0, min_size - 1))
    return morphology.remove_small_objects(mask, min_size=min_size)


def _remove_small_holes(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Fill holes with area < min_size (skimage 0.26+ uses max_size)."""
    if _USE_SKIMAGE_MAX_SIZE:
        return morphology.remove_small_holes(mask, max_size=max(0, min_size - 1))
    return morphology.remove_small_holes(mask, area_threshold=min_size)


def tissue_mask_otsu(img: ArrayLike, min_size: int = 200) -> np.ndarray:
    """
    Binary tissue mask via Otsu threshold on grayscale + morphology.

    Returns:
        bool array (H, W); True = tissue.
    """
    if img.ndim == 3:
        gray = rgb2gray(img)
    else:
        gray = img.astype(np.float64)
        if gray.max() > 1.0:
            gray = gray / 255.0
    threshold = filters.threshold_otsu(gray)
    tissue = gray < threshold
    tissue = _remove_small_objects(tissue, min_size)
    tissue = _remove_small_holes(tissue, min_size)
    return tissue.astype(bool)


def _resolve_segmenter_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_slide_segmenter(device: str, model_folder: str):
    key = f"{device}:{model_folder}"
    if key in _segmenter_cache:
        return _segmenter_cache[key]
    from slidesegmenter import SlideSegmenter

    segmenter = SlideSegmenter(
        tissue_segmentation=True,
        pen_marking_segmentation=False,
        separate_cross_sections=False,
        device=device,
        model_folder=model_folder,
    )
    _segmenter_cache[key] = segmenter
    return segmenter


def require_slidesegmenter() -> None:
    """Raise ImportError with install hint if slidesegmenter is missing."""
    try:
        import slidesegmenter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SlideSegmenter requires slidesegmenter. Install from GitHub (not PyPI):\n"
            "  pip install git+https://github.com/RTLucassen/slidesegmenter"
        ) from exc


def tissue_mask_segmenter(
    img: ArrayLike,
    *,
    device: Optional[str] = None,
    min_size: int = 40,
    model_folder: str = "latest",
) -> np.ndarray:
    """
    Binary tissue mask via SlideSegmenter (NN). Returns bool (H, W); True = tissue.
    """
    require_slidesegmenter()
    dev = _resolve_segmenter_device(device)
    segmenter = _get_slide_segmenter(dev, model_folder)
    img_float = img.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0
    segmentation = segmenter.segment(img_float)
    tissue_mask = np.squeeze(segmentation["tissue"]).astype(bool)
    tissue_mask = _remove_small_objects(tissue_mask, min_size)
    tissue_mask = _remove_small_holes(tissue_mask, min_size)
    return tissue_mask


def compute_tissue_mask(
    image_u8: np.ndarray,
    *,
    method: str = "otsu",
    min_size: int = 200,
    segmenter_min_size: int = 40,
    segmenter_model_folder: str = "latest",
    segmenter_device: Optional[str] = None,
) -> np.ndarray:
    if method == "otsu":
        return tissue_mask_otsu(image_u8, min_size=min_size)
    if method == "segmenter":
        return tissue_mask_segmenter(
            image_u8,
            device=segmenter_device,
            min_size=segmenter_min_size,
            model_folder=segmenter_model_folder,
        )
    raise ValueError(f"Unknown tissue mask method: {method!r}. Use otsu or segmenter.")


def mask_predictions_outside_tissue(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    image_u8: np.ndarray,
    *,
    method: str = "otsu",
    min_size: int = 200,
    segmenter_min_size: int = 40,
    segmenter_model_folder: str = "latest",
    segmenter_device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zero pred and GT outside tissue; return (pred, gt, tissue_mask).

    method: "otsu" or "segmenter".
    Metrics should be computed only on pixels where tissue_mask is True.
    """
    tissue = compute_tissue_mask(
        image_u8,
        method=method,
        min_size=min_size,
        segmenter_min_size=segmenter_min_size,
        segmenter_model_folder=segmenter_model_folder,
        segmenter_device=segmenter_device,
    )
    pred_out = pred_map.copy()
    gt_out = gt_map.copy()
    pred_out[~tissue] = 0
    gt_out[~tissue] = 0
    return pred_out, gt_out, tissue


def add_tissue_mask_cli_args(parser: argparse.ArgumentParser, *, default: str = "otsu") -> None:
    parser.add_argument(
        "--tissue-mask",
        choices=("none", "otsu", "segmenter"),
        default=default,
        help="Per-tile tissue mask before metrics. otsu (default): Otsu + morphology. "
        "segmenter: SlideSegmenter NN (better when pen markings present). "
        "none: evaluate on all pixels.",
    )
    parser.add_argument(
        "--otsu-min-size",
        type=int,
        default=200,
        help="Min object/hole size (px) for Otsu morphology when --tissue-mask=otsu.",
    )
    parser.add_argument(
        "--segmenter-min-size",
        type=int,
        default=40,
        help="Min object/hole size (px) for segmenter morphology when --tissue-mask=segmenter.",
    )
    parser.add_argument(
        "--segmenter-model-folder",
        type=str,
        default="latest",
        help="SlideSegmenter model folder name when --tissue-mask=segmenter.",
    )
    parser.add_argument(
        "--segmenter-device",
        type=str,
        default=None,
        help="Device for SlideSegmenter (cuda/cpu). Default: auto.",
    )


def validate_tissue_mask_args(args) -> None:
    if args.otsu_min_size < 1:
        raise ValueError(f"--otsu-min-size must be >= 1, got {args.otsu_min_size}")
    if args.segmenter_min_size < 1:
        raise ValueError(f"--segmenter-min-size must be >= 1, got {args.segmenter_min_size}")
    if args.tissue_mask == "segmenter":
        require_slidesegmenter()


def tissue_mask_kwargs_from_args(args) -> dict:
    return {
        "method": args.tissue_mask,
        "min_size": args.otsu_min_size,
        "segmenter_min_size": args.segmenter_min_size,
        "segmenter_model_folder": args.segmenter_model_folder,
        "segmenter_device": args.segmenter_device,
    }


def format_tissue_mask_args(args) -> str:
    if args.tissue_mask == "none":
        return "none"
    if args.tissue_mask == "otsu":
        return f"otsu (min_size={args.otsu_min_size})"
    return (
        f"segmenter (min_size={args.segmenter_min_size}, "
        f"model_folder={args.segmenter_model_folder}, device={args.segmenter_device or 'auto'})"
    )
