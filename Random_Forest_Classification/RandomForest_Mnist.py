"""
Random Forest on MNIST (large dataset) per course workflow:

- Hyperparameter tuning uses 80% of the official *training* set for learning and
  20% as a held-out *validation* set. The official MNIST *test* set is never used
  during tuning.
- After selecting hyperparameters, the final model is refit on **all** official
  training images, then evaluated on the official test split.
- Metrics on the test set: Accuracy, macro F1, and multi-class AUC-ROC (OvR,
  macro-averaged), consistent with common multi-class AUC practice.

Dependencies: pip install numpy scikit-learn

Run from repository root:
  python Random_Forest_Classification/RandomForest_Mnist.py
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def read_idx3_ubyte_images(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Not an IDX3-ubyte image file: {path}")
        buf = f.read(n * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8, count=n * rows * cols)
        return data.reshape(n, rows * cols)


def read_idx1_ubyte_labels(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Not an IDX1-ubyte label file: {path}")
        buf = f.read(n)
        return np.frombuffer(buf, dtype=np.uint8, count=n)


def detect_repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "training_data_mnist").is_dir():
        return cwd.resolve()
    if (cwd.parent / "training_data_mnist").is_dir():
        return cwd.parent.resolve()
    script = Path(__file__).resolve()
    cand = script.parent.parent
    if (cand / "training_data_mnist").is_dir():
        return cand.resolve()
    raise FileNotFoundError(
        "Could not find training_data_mnist. Run from the repository root "
        "(folder that contains training_data_mnist and testing_data_mnist), "
        "or keep this script under Random_Forest_Classification/."
    )


@dataclass(frozen=True)
class TrialResult:
    params: dict[str, Any]
    val_accuracy: float
    val_f1_macro: float
    val_auc_macro_ovr: float


def _fit_eval_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int | None,
    max_features: str | float | int,
    min_samples_leaf: int,
    pca_components: int | None,
    random_state: int,
) -> TrialResult:
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
        Xt = pca.fit_transform(X_train)
        Xv = pca.transform(X_val)
        fitted_pca = pca
    else:
        Xt, Xv = X_train, X_val
        fitted_pca = None

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
    )
    clf.fit(Xt, y_train)
    pred = clf.predict(Xv)
    proba = clf.predict_proba(Xv)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred, average="macro", labels=np.arange(10))
    auc = roc_auc_score(
        y_val,
        proba,
        multi_class="ovr",
        average="macro",
        labels=np.arange(10),
    )
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "pca_components": pca_components,
    }
    return TrialResult(params, acc, f1, auc)


def _refit_full_and_predict_test(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    best: TrialResult,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, PCA | None]:
    pca_components = best.params["pca_components"]
    if pca_components is not None:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_t = pca.fit_transform(X_train_full)
        X_test_t = pca.transform(X_test)
    else:
        pca = None
        X_train_t, X_test_t = X_train_full, X_test

    clf = RandomForestClassifier(
        n_estimators=best.params["n_estimators"],
        max_depth=best.params["max_depth"],
        max_features=best.params["max_features"],
        min_samples_leaf=best.params["min_samples_leaf"],
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train_t, y_train_full)
    pred = clf.predict(X_test_t)
    proba = clf.predict_proba(X_test_t)
    return pred, proba, pca


def _format_params(p: dict[str, Any]) -> str:
    parts = [f"{k}={repr(v)}" for k, v in p.items()]
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    random_state = 347
    val_fraction = 0.20
    for arg in argv:
        if arg.startswith("--random-state="):
            random_state = int(arg.split("=", 1)[1])
        if arg.startswith("--val-fraction="):
            val_fraction = float(arg.split("=", 1)[1])

    root = detect_repo_root()
    train_x = root / "training_data_mnist" / "train-images.idx3-ubyte"
    train_y = root / "training_data_mnist" / "train-labels.idx1-ubyte"
    test_x = root / "testing_data_mnist" / "t10k-images.idx3-ubyte"
    test_y = root / "testing_data_mnist" / "t10k-labels.idx1-ubyte"

    for p in (train_x, train_y, test_x, test_y):
        if not p.is_file():
            print(f"Missing file: {p}", file=sys.stderr)
            return 1

    print(f"Loading MNIST from {root} ...")
    X_train_full = read_idx3_ubyte_images(train_x).astype(np.float32, copy=False)
    y_train_full = read_idx1_ubyte_labels(train_y)
    X_test = read_idx3_ubyte_images(test_x).astype(np.float32, copy=False)
    y_test = read_idx1_ubyte_labels(test_y)

    if X_train_full.shape[0] != y_train_full.shape[0] or X_test.shape[0] != y_test.shape[0]:
        print("Image/label row counts do not match.", file=sys.stderr)
        return 1

    n_feat = X_train_full.shape[1]
    print("\n=== Experimental configuration (MNIST = large dataset) ===")
    print(f"Classifier: sklearn.ensemble.RandomForestClassifier")
    print(f"Tuning split: stratified {100 * (1 - val_fraction):.0f}% train / {100 * val_fraction:.0f}% validation")
    print(f"Selection metric: validation accuracy (tie-break: macro F1, then macro OvR AUC)")
    print(f"Final refit: all official training samples (n={X_train_full.shape[0]})")
    print(f"Test evaluation: official MNIST test (n={X_test.shape[0]})")
    print(f"Features per image: {n_feat} (optional PCA candidates in search space)")
    print(f"random_state={random_state}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_fraction,
        stratify=y_train_full,
        random_state=random_state,
    )

    search_space: Iterable[dict[str, Any]] = (
        dict(
            n_estimators=n_est,
            max_depth=depth,
            max_features=mfeat,
            min_samples_leaf=msl,
            pca_components=pca_n,
        )
        for n_est, depth, mfeat, msl, pca_n in product(
            (300, 500),
            (None, 30),
            ("sqrt", 64),
            (1,),
            (None, 100),
        )
    )

    trials: list[TrialResult] = []
    print("\n=== Hyperparameter search (validation only; test untouched) ===")
    for i, kw in enumerate(search_space, start=1):
        print(f"[trial {i}] {_format_params(kw)}")
        trials.append(
            _fit_eval_val(
                X_tr,
                y_tr,
                X_val,
                y_val,
                random_state=random_state,
                **kw,
            )
        )
        tr = trials[-1]
        print(
            f"         val_acc={tr.val_accuracy:.4f}  val_f1_macro={tr.val_f1_macro:.4f}  "
            f"val_auc_macro_ovr={tr.val_auc_macro_ovr:.4f}"
        )

    best = max(
        trials,
        key=lambda t: (t.val_accuracy, t.val_f1_macro, t.val_auc_macro_ovr),
    )
    print("\n=== Best validation configuration ===")
    print(_format_params(best.params))
    print(
        f"val_acc={best.val_accuracy:.4f}  val_f1_macro={best.val_f1_macro:.4f}  "
        f"val_auc_macro_ovr={best.val_auc_macro_ovr:.4f}"
    )

    print("\n=== Final fit on full training + test metrics ===")
    pred, proba, _pca = _refit_full_and_predict_test(
        X_train_full,
        y_train_full,
        X_test,
        y_test,
        best,
        random_state=random_state,
    )

    acc = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro", labels=np.arange(10))
    f1_weighted = f1_score(y_test, pred, average="weighted", labels=np.arange(10))
    auc_macro_ovr = roc_auc_score(
        y_test,
        proba,
        multi_class="ovr",
        average="macro",
        labels=np.arange(10),
    )

    print(f"Test accuracy:           {acc:.4f}")
    print(f"Test F1 (macro):         {f1_macro:.4f}")
    print(f"Test F1 (weighted):      {f1_weighted:.4f}")
    print(
        "Test AUC-ROC (macro, OvR): "
        f"{auc_macro_ovr:.4f}  [multi_class='ovr', average='macro']"
    )
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, pred, labels=np.arange(10)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
