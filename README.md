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

## Reference

This project is motivated by the following paper:

> D Kollias, V Sharmanska, S Zafeiriou (2024). *Distribution matching for multi-task learning of classification tasks: a large-scale study on faces & beyond*. Proceedings of the AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/28061

Paper author: Prof. Dimitrios Kollias, Queen Mary University of London (d.kollias@qmul.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
