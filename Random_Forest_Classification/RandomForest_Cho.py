"""
Random Forest on Cho dataset:
Assignment-aligned protocol:
- Create train/test split from Cho.
- Hyperparameter tuning uses K-fold CV on training set only K=3.
- Repeat full experiment t=3 times with different random seeds.

Dependencies i needed to install:
  pip install numpy scikit-learn

Run from repository root:
  python Random_Forest_Classification/RandomForest_Cho.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


def detect_repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "cho.txt").is_file():
        return cwd.resolve()
    if (cwd.parent / "cho.txt").is_file():
        return cwd.parent.resolve()
    script = Path(__file__).resolve()
    cand = script.parent.parent
    if (cand / "cho.txt").is_file():
        return cand.resolve()
    raise FileNotFoundError(
        "Could not find cho.txt. Run from repo root or keep script under "
        "Random_Forest_Classification/."
    )


def load_cho(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # cho.txt format: [sample_id] [label] [feature_1] ... [feature_d]
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Unexpected cho.txt format. Need id, label, and features.")
    y = data[:, 1].astype(int)
    X = data[:, 2:].astype(np.float32)
    return X, y


def multiclass_auc(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    if len(classes) == 2:
        pos_class = classes[1]
        y_bin = (y_true == pos_class).astype(int)
        pos_idx = int(np.where(classes == pos_class)[0][0])
        return float(roc_auc_score(y_bin, proba[:, pos_idx]))
    y_onehot = label_binarize(y_true, classes=classes)
    return float(
        roc_auc_score(
            y_onehot,
            proba,
            average="macro",
            multi_class="ovr",
        )
    )


def build_search(k: int, seed: int) -> GridSearchCV:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", "passthrough"),
            (
                "rf",
                RandomForestClassifier(
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Includes "no PCA" and optional PCA branch.
    param_grid: list[dict[str, Any]] = [
        {
            "pca": ["passthrough"],
            "rf__n_estimators": [200, 500],
            "rf__max_features": ["sqrt", None],
            "rf__max_depth": [None, 20],
            "rf__min_samples_leaf": [1, 2],
        },
        {
            "pca": [PCA(random_state=seed)],
            "pca__n_components": [0.95, 0.99],
            "rf__n_estimators": [200, 500],
            "rf__max_features": ["sqrt", None],
            "rf__max_depth": [None, 20],
            "rf__min_samples_leaf": [1, 2],
        },
    ]

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def parse_args(argv: list[str]) -> tuple[int, int, float]:
    k, t, test_size = 3, 3, 0.2
    for arg in argv:
        if arg.startswith("--k="):
            k = int(arg.split("=", 1)[1])
        elif arg.startswith("--t="):
            t = int(arg.split("=", 1)[1])
        elif arg.startswith("--test-size="):
            test_size = float(arg.split("=", 1)[1])
    if k < 2:
        raise ValueError("k must be >= 2.")
    if t < 1:
        raise ValueError("t must be >= 1.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test-size must be in (0, 1).")
    return k, t, test_size


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    k, t, test_size = parse_args(argv)
    root = detect_repo_root()
    data_path = root / "cho.txt"

    X, y = load_cho(data_path)
    classes = np.unique(y)
    if len(classes) < 2:
        print("Need at least 2 classes for classification.", file=sys.stderr)
        return 1

    print("=== Random Forest on Cho (small dataset protocol) ===")
    print(f"Dataset: {data_path}")
    print(f"Samples={X.shape[0]}, Features={X.shape[1]}, Classes={len(classes)}")
    print(f"K-fold CV for tuning on train only: K={k}")
    print(f"Repeated experiments: t={t}")
    print(f"Train/Test split each run: {(1 - test_size):.0%}/{test_size:.0%}\n")

    accs: list[float] = []
    f1s: list[float] = []
    aucs: list[float] = []

    for run in range(t):
        seed = 347 + run
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )

        search = build_search(k, seed)
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))
        auc = multiclass_auc(y_test, y_proba, classes)

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        print(
            f"[run {run + 1}/{t}] seed={seed} "
            f"best_cv_acc={search.best_score_:.4f} "
            f"test_acc={acc:.4f} test_f1_macro={f1:.4f} test_auc={auc:.4f}"
        )
        print(f"  best_params: {search.best_params_}")

    def mean_std(xs: list[float]) -> tuple[float, float]:
        arr = np.asarray(xs, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0)

    acc_m, acc_s = mean_std(accs)
    f1_m, f1_s = mean_std(f1s)
    auc_m, auc_s = mean_std(aucs)

    print("\n=== Final test results over t runs (mean ± std) ===")
    print(f"Accuracy: {acc_m:.4f} ± {acc_s:.4f}")
    print(f"F1-macro: {f1_m:.4f} ± {f1_s:.4f}")
    print(f"AUC:      {auc_m:.4f} ± {auc_s:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
