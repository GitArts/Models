This guide covers **5 trained model families** (**31 U-Net++ checkpoints** in total). Training always saves **one network per pyramid level** (`unetpp_level{N}/best_model.pth` for binary runs, `unetpp3_level{N}/best_model.pth` for the 3-class run). Levels are the QuPath export downsampling steps **1–6** (except IM, which uses **level 3 only**).

| # | Model family | Run name / results folder | Levels | Checkpoints | Task |
|---|--------------|---------------------------|--------|-------------|------|
| 1 | **Dysplasia** | `Dysplasia` | 1–6 | **6** | Binary segmentation (CE) |
| 2 | **IM_only_level_3** | `IM_only_level_3` | 3 | **1** | Binary segmentation (Dice) |
| 3 | **HID_model** | `HID_model` | 1–6 | **6** | Binary segmentation |
| 4 | **Pos / Neg** | `results_unetpp_posneg_levels_dicece/Pos` + `.../Neg` | 1–6 each | **12** | Two binary models (Pos and Neg, Dice+CE) |
| 5 | **3-class PosNeg** | `results_unetpp_3class_PosNeg_levels_dicece` | 1–6 | **6** | One model: background / Neg / Pos |
| | | | **Total** | **31** | |

Sections **1–5** below match this table. Eval script **#1–3** use `evaluate_unetpp_levels.py`; **#4** uses `evaluate_unetpp_posneg_levels.py` (run twice for Pos and Neg); **#5** uses `evaluate_unetpp_3class_levels.py`.

## Files you need

| File | Why |
|------|-----|
| `evaluate_unetpp_levels.py` | Binary U-Net++ (Dysplasia, IM level3, HID_model, …) |
| `evaluate_unetpp_posneg_levels.py` | Separate Pos / Neg binary models |
| `evaluate_unetpp_3class_levels.py` | Single 3-class model (bg / Neg / Pos) |
| `Training_unetpp_*.py` | Imported by eval (same folder) |
| **`tissue_mask.py`** | **Required** — see below |

Copy `tissue_mask.py` next to the eval script if you only sync individual files.

## Eval Dysplasia 
python evaluate_unetpp_levels.py \
  --data-root ./Dysplasia \
  --models-root ./results_unetpp_levels_dicece \
  --run-name YOUR_RUN_NAME \
  --levels 1,2,3,4,5,6 \
  --eval-from both \
  --save-all-test-outputs

## Eval IM only level3
python evaluate_unetpp_levels.py \
  --data-root Data \
  --models-root results_unetpp_levels_dice \
  --run-name IM_only_level_3 \
  --in-channels 3 \
  --levels 3 \
  --eval-from both \
  --tissue-mask otsu


## Eval 3 class dysplasia 
python evaluate_unetpp_3class_levels.py \
  --data-root Data/Kopiga_DB \
  --models-root results_unetpp_3class_levels_dicece \
  --run-name Dys_3class \
  --in-channels 3 \
  --levels 1,2,3,4,5,6 \
  --eval-from test_folders

## Eval 2 class dysplasia Pos only class
python evaluate_unetpp_posneg_levels.py \
  --target-class Pos \
  --data-root ~/AIDA/Data/Kopiga_DB \
  --models-root ~/AIDA/results_unetpp_posneg_levels_dicece/Pos \
  --run-name Pos_train \
  --eval-from test_folders \
  --levels 1,2,3,4,5,6 \
  --loss dicece \
  --seed 42

## Eval 2 class dysplasia only Neg class
python evaluate_unetpp_posneg_levels.py \
  --target-class Neg \
  --data-root ~/AIDA/Data/Kopiga_DB \
  --models-root ~/AIDA/results_unetpp_posneg_levels_dicece/Neg \
  --run-name Neg_train \
  --eval-from test_folders \
  --levels 1,2,3,4,5,6 \
  --loss dicece \
  --seed 42

---

## Flag descrition

Data & checkpoints

	--data-root — Dataset root with levelN/ folders (Train_*, Test_*). Default: Results/Dysplasia.
	--models-root — Training output root with unetpp_levelN/. Default: results_unetpp_levels.
	--run-name — Training run name; checkpoints under models-root/run-name/. Default: empty.

What to evaluate

	--eval-from — train_split (5% test from Train_*), test_folders (all Test_*), or both. Default: train_split.
	--levels — Comma-separated levels, e.g. 1,2,3,4,5,6. Default: 0,1,2,3,4,5.
	--balance-level — Reference level for sample count in train_split only. Default: max of --levels.
	--test-images-dir — Test image subfolder name. Default: Test_images.
	--test-masks-dir — Test mask subfolder name. Default: Test_masks.

Model & inference

	--image-size — Tile size in pixels. Default: 512.
	--in-channels — 1 (grayscale) or 3 (RGB). Default: 3.
	--batch-size — Batch size. Default: 1.
	--seed — Random seed (must match training for train_split). Default: 42.
	--device — cuda or cpu. Default: auto.
	--num-workers — DataLoader workers. Default: 0.

Tissue mask

	--tissue-mask — otsu, segmenter, or none. Default: otsu.
	--otsu-min-size — Otsu morphology min size (px). Default: 200.
	--segmenter-min-size — Segmenter morphology min size (px). Default: 40.
	--segmenter-model-folder — SlideSegmenter model folder. Default: latest.
	--segmenter-device — Segmenter device. Default: auto.	

Outputs & previews

	--num-mask-examples — Number of preview tiles to save. Default: 5 (0 = off).
	--save-all-test-outputs — Save all test tiles (input, GT mask, pred mask).
	--save-all-include-overlays — Also save overlay and GT vs pred images (needs --save-all-test-outputs).
	--skip-missing — Skip levels without checkpoint instead of failing.

---

## Why `tissue_mask.py`?

At eval time, by default (`--tissue-mask otsu`):

Histology tiles include large areas of empty glass and background that are not part of the tissue region of interest.
During **training**, models see full tiles and learn from the masks as exported;
During **evaluation**, we want metrics that reflect segmentation quality **on tissue only**,
in line with how predictions are used on whole slides.
The module `tissue_mask.py` provides that step: for each tile it builds a binary tissue mask from the RGB image,
then clears both the model prediction and the ground-truth mask outside that region before TP/TN/FP/FN and Dice are accumulated.


The default method is **Otsu** (`--tissue-mask otsu`). The tile is converted to grayscale, thresholded with Otsu’s method,
and cleaned with morphological operations (small objects and holes removed).
This works well when slides have no pen markings, because dark ink can look like tissue under a simple threshold.
When **pen markings** are present, use **`--tissue-mask segmenter`**, which runs the SlideSegmenter neural network to
separate tissue from background and ink; install it with `pip install git+https://github.com/RTLucassen/slidesegmenter`.
To score **every pixel** in the tile (including glass), pass **`--tissue-mask none`**.


| `--tissue-mask` | When to use |
|-----------------|-------------|
| `otsu` (default) | No pen markings on slides |
| `segmenter` | Pen ink present — needs `pip install git+https://github.com/RTLucassen/slidesegmenter` |
| `none` | All pixels (old behaviour) |

---

## Dice columns in CSV

- **dice_micro** — all pixels pooled
- **dice_micro_empty1** — same; empty-empty globally → 1.0
- **dice_mean_excl_empty** — mean per-tile Dice, skip empty-empty tiles
- **dice_mean_empty1** — mean per-tile Dice, empty-empty tile → 1.0
