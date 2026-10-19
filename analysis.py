"""Baseline: shared-hidden-layer MLP (sklearn MLPClassifier, sigmoid multi-label outputs).
Improved: per-task class-balanced LightGBM on richer features (standardised pixels + PCA components),
with a wider train-only CV grid that includes feature subsampling. The test set is used once, at the end.
Tasks are derived from scikit-learn's bundled digits data."""
import itertools
import json
import os
import sys

import lightgbm as lgb
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]  # both models are randomised -> run each with 3 seeds
N_PCA = 30            # fixed, not tuned on test
TASKS = ["is_even", "is_gt_4", "is_prime"]
SOURCE = "https://scikit-learn.org/stable/datasets/toy_dataset.html"
MLP_GRID = [dict(hidden_layer_sizes=(h,), alpha=a) for h, a in itertools.product([64, 128], [1e-4, 1e-2])]
LGB_GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, colsample_bytree=cs)
            for nl, lr, ne, cs in itertools.product([7, 15, 31], [0.05, 0.1], [200, 400], [0.3, 0.7])]


def macro_f1_per_task(y_true, y_pred):
    """Macro-F1 for each of the binary tasks (one column per task)."""
    return np.array([f1_score(y_true[:, t], y_pred[:, t], average="macro") for t in range(y_true.shape[1])])


def cv_task_scores(fit_predict, X, Y, strata, seed):
    """5-fold stratified CV on the TRAINING split only; returns mean per-task macro-F1."""
    scores = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
        scores.append(macro_f1_per_task(Y[va], fit_predict(X[tr], Y[tr], X[va])))
    return np.mean(scores, axis=0)


def mlp_fit_predict(params, seed):
    """Shared MLP: one network predicts all tasks jointly (sigmoid outputs, threshold 0.5)."""
    def fp(Xa, Ya, Xb):
        return MLPClassifier(max_iter=500, random_state=seed, **params).fit(Xa, Ya).predict(Xb)
    return fp


def lgb_fit_predict(params, seed):
    """Separate class-balanced LightGBM per task on the shared feature matrix."""
    def fp(Xa, Ya, Xb):
        out = np.zeros((len(Xb), Ya.shape[1]), dtype=int)
        for t in range(Ya.shape[1]):
            m = lgb.LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1,
                                   **params)
            out[:, t] = m.fit(Xa, Ya[:, t]).predict(Xb)
        return out
    return fp


def main():
    data = load_digits()  # bundled with scikit-learn: no download needed
    X, digit = data.data, data.target  # 64 pixel features, digit label 0-9
    Y = np.column_stack([digit % 2 == 0, digit > 4, np.isin(digit, [2, 3, 5, 7])]).astype(int)
    print(f"Loaded {len(X)} images; tasks: {TASKS}")

    # One stratified 70/30 split (stratified on the digit label), fixed seed
    Xtr, Xte, Ytr, Yte, dtr, _ = train_test_split(X, Y, digit, test_size=0.3, stratify=digit, random_state=SEED)

    # Standardise pixels and fit the PCA on training data only
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler.transform(Xtr), scaler.transform(Xte)
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(Ztr)
    Ptr, Pte = pca.transform(Ztr), pca.transform(Zte)
    # Improved features: standardised pixels + PCA components (both fit on training data only)
    Ftr, Fte = np.hstack([Ztr, Ptr]), np.hstack([Zte, Pte])
    print(f"Train {len(Xtr)} / test {len(Xte)}; PCA explains {pca.explained_variance_ratio_.sum():.3f} variance")

    base_runs, impr_runs = [], []
    for seed in SEEDS:
        # ---- Baseline: shared MLP; hidden size / alpha chosen by train-only CV (unchanged) ----
        cv = [cv_task_scores(mlp_fit_predict(p, seed), Ztr, Ytr, dtr, seed).mean() for p in MLP_GRID]
        best = MLP_GRID[int(np.argmax(cv))]
        pred = mlp_fit_predict(best, seed)(Ztr, Ytr, Zte)
        base_runs.append(macro_f1_per_task(Yte, pred))
        print(f"seed {seed} baseline: best={best} test F1 per task={np.round(base_runs[-1], 4)}")

        # ---- Improved: per-task balanced LightGBM on pixels + PCA; per-task params chosen by train-only CV ----
        cv = np.array([cv_task_scores(lgb_fit_predict(p, seed), Ftr, Ytr, dtr, seed) for p in LGB_GRID])
        pred, chosen = np.zeros_like(Yte), []
        for t in range(len(TASKS)):
            p = LGB_GRID[int(np.argmax(cv[:, t]))]
            chosen.append(p)
            pred[:, t] = lgb_fit_predict(p, seed)(Ftr, Ytr[:, [t]], Fte)[:, 0]
        impr_runs.append(macro_f1_per_task(Yte, pred))
        print(f"seed {seed} improved: best={chosen} test F1 per task={np.round(impr_runs[-1], 4)}")

    B, I = np.array(base_runs), np.array(impr_runs)  # shape: seeds x tasks
    b_task, i_task = B.mean(0), I.mean(0)            # per-task, averaged over seeds
    b_seed, i_seed = B.mean(1), I.mean(1)            # main metric per seed (mean over tasks)
    b_score, i_score = float(b_seed.mean()), float(i_seed.mean())
    if not np.all(np.isfinite([b_score, i_score])):
        sys.exit("Non-finite metrics; cannot produce meaningful results.")
    n_worse = int(np.sum(i_task < b_task))
    n_better = int(np.sum(i_task > b_task))

    # Chart: per-task test macro-F1, baseline vs improved
    os.makedirs("figures", exist_ok=True)
    x, w = np.arange(len(TASKS)), 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - w / 2, b_task, w, label="baseline: shared MLP")
    plt.bar(x + w / 2, i_task, w, label="improved: pixels+PCA balanced LightGBM")
    plt.xticks(x, TASKS)
    plt.ylabel("test macro-F1 (mean over seeds)")
    plt.ylim(max(0.0, min(b_task.min(), i_task.min()) - 0.05), 1.0)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/per_task_macro_f1.png", dpi=120)
    plt.close()

    results = {
        "question": "Does a per-task class-balanced LightGBM on a shared PCA representation beat a standard "
                    "shared-hidden-layer MLP on every related classification task, not just on average, "
                    "using scikit-learn's digits data?",
        "method": "Derive three binary tasks (is-even, is-greater-than-4, is-prime) from the digit label. "
                  "One stratified 70/30 split. Compare a shared-hidden-layer MLP with per-task class-balanced "
                  "LightGBM on standardised pixels plus PCA components. Hyper-parameters chosen by 5-fold CV on "
                  "train only; test used once. Seeds 42, 43, 44.",
        "dataset": {
            "name": "scikit-learn digits (load_digits)",
            "source_url": SOURCE,
            "n_rows": int(len(X)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 averaged over 3 binary tasks (is_even, is_gt_4, is_prime)",
        "higher_is_better": True,
        "baseline": "Standard hard-parameter-sharing MLP: one shared hidden layer, sigmoid multi-label outputs "
                    "(sklearn MLPClassifier) on standardised pixels; hidden size and alpha chosen by train-only CV.",
        "improved": "Per-task class-balanced LightGBM on standardised pixels concatenated with 30 PCA components "
                    "(fit on training pixels); num_leaves, learning_rate, n_estimators and colsample_bytree "
                    "chosen per task by train-only 5-fold CV over a 24-configuration grid.",
        "metrics": {
            "baseline_score": b_score,
            "improved_score": i_score,
            "improvement": i_score - b_score,
            "baseline_score_std": float(b_seed.std()),
            "improved_score_std": float(i_seed.std()),
            "baseline_worst_task_f1": float(b_task.min()),
            "improved_worst_task_f1": float(i_task.min()),
            "n_tasks_improved_lower": n_worse,
            "n_tasks_improved_higher": n_better,
            **{f"baseline_f1_{t}": float(v) for t, v in zip(TASKS, b_task)},
            **{f"improved_f1_{t}": float(v) for t, v in zip(TASKS, i_task)},
        },
        "notes": "Only one train/test split is used, so split-to-split variance is not measured; seeds capture "
                 "model randomness only. The three tasks are deterministic functions of one label and therefore "
                 "strongly related; the test set is small, so per-task gaps of a few points may be noise. "
                 "Both models get the same 5-fold CV folds per seed; the baseline grid has 4 configurations and "
                 "the LightGBM grid has 24 per task. The LightGBM model receives the same standardised pixels "
                 "the baseline uses, plus PCA components. PCA dimension is fixed, not tuned. The paper's "
                 "architecture is not specified in the provided abstract, so this is not a replication of its "
                 "method. load_digits is bundled, so no network download or retry logic is needed.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline macro-F1 {b_score:.4f} vs improved {i_score:.4f}; "
          f"tasks where improved is lower: {n_worse}. Wrote results.json and figures/.")


if __name__ == "__main__":
    main()
