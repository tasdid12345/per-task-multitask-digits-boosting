# Per-task LightGBM on shared PCA vs shared-hidden-layer MLP for related digit tasks

This experiment is motivated by "Distribution matching for multi-task learning of classification tasks: a large-scale study on faces & beyond" (AAAI, 2024), which studies how well multi-task classifiers handle several related classification tasks at once. A common baseline for such settings is hard parameter sharing, where one shared hidden layer feeds all task outputs. This repository tests a simpler alternative on a small, fully reproducible problem.

**Question:** Does a per-task class-balanced LightGBM on a shared PCA representation beat a standard shared-hidden-layer MLP on every related classification task, not just on average?

**Dataset:** scikit-learn's bundled digits dataset (`load_digits`), with three binary tasks derived from the digit label: is-even, is-greater-than-4, and is-prime. This is not the paper's own dataset. A single stratified 70/30 split with a fixed seed is used.

**Baseline:** a standard hard-parameter-sharing MLP with one shared hidden layer and sigmoid multi-label outputs (scikit-learn `MLPClassifier`), trained on standardised pixels.

**Improved model:** a shared PCA representation fit on training pixels, with one class-balanced LightGBM per task. Hyper-parameters are chosen by cross-validation on the training set only.

**How to run**

```
pip install -r requirements.txt
python analysis.py
```

**Results:** results are written to `results.json`, and a per-task chart is saved to `figures/`.

## Reference

This project is motivated by the following paper:

> D Kollias, V Sharmanska, S Zafeiriou (2024). *Distribution matching for multi-task learning of classification tasks: a large-scale study on faces & beyond*. Proceedings of the AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/28061

Paper author: Prof. Dimitrios Kollias, Queen Mary University of London (d.kollias@qmul.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
