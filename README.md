

# ======= Test Dysplasia =========
python evaluate_unetpp_levels.py \
  --data-root ./Dysplasia \
  --models-root ./results_unetpp_levels_dicece \
  --run-name YOUR_RUN_NAME \
  --levels 1,2,3,4,5,6 \
  --eval-from both \
  --save-all-test-outputs

# ======= Test IM only level3 =========
python evaluate_unetpp_levels.py \
  --data-root Data \
  --models-root results_unetpp_levels_dice \
  --run-name IM_only_level_3 \
  --in-channels 3 \
  --levels 3 \
  --eval-from both \
  --tissue-mask otsu


# ======= Test 3 class dysplasia =========
python evaluate_unetpp_3class_levels.py \
  --data-root Data/Kopiga_DB \
  --models-root results_unetpp_3class_levels_dicece \
  --run-name Dys_3class \
  --in-channels 3 \
  --levels 1,2,3,4,5,6 \
  --eval-from test_folders


python evaluate_unetpp_posneg_levels.py \
  --target-class Pos \
  --data-root ~/AIDA/Data/Kopiga_DB \
  --models-root ~/AIDA/results_unetpp_posneg_levels_dicece/Pos \
  --run-name Pos_train \
  --eval-from test_folders \
  --levels 1,2,3,4,5,6 \
  --loss dicece \
  --seed 42

python evaluate_unetpp_posneg_levels.py \
  --target-class Neg \
  --data-root ~/AIDA/Data/Kopiga_DB \
  --models-root ~/AIDA/results_unetpp_posneg_levels_dicece/Neg \
  --run-name Neg_train \
  --eval-from test_folders \
  --levels 1,2,3,4,5,6 \
  --loss dicece \
  --seed 42


# Flag descrition

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
