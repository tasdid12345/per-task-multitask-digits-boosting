Digits multi-task comparison: shared MLP baseline vs per-task class-balanced LightGBM.

Setup
- Data: scikit-learn bundled digits (load_digits), no download.
- Tasks: is_even, is_gt_4, is_prime derived from the digit label.
- One stratified 70/30 split (seed 42); the test set is used only for the final scores.
- Seeds 42, 43, 44. Hyper-parameters chosen by 5-fold stratified CV on the training split only.

Models
- Baseline: sklearn MLPClassifier, one shared hidden layer, on standardised pixels (unchanged).
- Improved: per-task LightGBM (class_weight="balanced") on standardised pixels plus 30 PCA components
  (PCA fit on training pixels). LightGBM grid: num_leaves, learning_rate, n_estimators, colsample_bytree.

Run
    pip install -r requirements.txt
    python analysis.py

Outputs: results.json and figures/per_task_macro_f1.png.

Caveats
- One train/test split; seed variation only. The test set is small, so per-task gaps of a few points may be noise.
- The improved model was changed after a run in which it trailed the baseline; the new version has not been
  verified by a successful CI run in this submission.

## Results (latest run)

On the scikit-learn digits test set, a standard hard-parameter-sharing MLP reached 0.983 macro-F1 over three binary tasks, versus 0.984 for class-balanced LightGBM on pixels plus PCA: a negligible, mixed gain, since is-greater-than-4 fell from 0.981 to 0.975.

| Metric | Value |
|---|---|
| baseline_score | 0.9832 |
| improved_score | 0.9836 |
| improvement | 0.0004267 |
| baseline_score_std | 0.001521 |
| improved_score_std | 0.001059 |
| baseline_worst_task_f1 | 0.9809 |
| improved_worst_task_f1 | 0.9753 |
| n_tasks_improved_lower | 1 |
| n_tasks_improved_higher | 2 |
| baseline_f1_is_even | 0.9815 |
| baseline_f1_is_gt_4 | 0.9809 |
| baseline_f1_is_prime | 0.9872 |
| improved_f1_is_even | 0.987 |
| improved_f1_is_gt_4 | 0.9753 |
| improved_f1_is_prime | 0.9885 |

Run on 2026-10-19. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Only one train/test split is used, so split-to-split variance is not measured; seeds capture model randomness only. The three tasks are deterministic functions of one label and therefore strongly related; the test set is small, so per-task gaps of a few points may be noise. Both models get the same 5-fold CV folds per seed; the baseline grid has 4 configurations and the LightGBM grid has 24 per task. The LightGBM model receives the same standardised pixels the baseline uses, plus PCA components. PCA dimension is fixed, not tuned. The paper's architecture is not specified in the provided abstract, so this is not a replication of its method. load_digits is bundled, so no network download or retry logic is needed.

## Reference

This project is motivated by the following paper:

> D Kollias, V Sharmanska, S Zafeiriou (2024). *Distribution matching for multi-task learning of classification tasks: a large-scale study on faces & beyond*. Proceedings of the AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/28061

Paper author: Prof. Dimitrios Kollias, Queen Mary University of London (d.kollias@qmul.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
