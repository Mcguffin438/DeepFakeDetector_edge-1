"""Tune, evaluate, and export the LightGBM model using York audio features.

The input CSV must contain the 26 model features and a LABEL column where
FAKE=0 and REAL=1. The fixed 30% holdout mirrors the source notebook split.
The existing ONNX model is replaced only if the tuned model scores higher
on that holdout.
"""

import argparse
import os
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split


FEATURES = [
    "chroma_stft",
    "rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "rolloff",
    "zero_crossing_rate",
    *(f"mfcc{i}" for i in range(1, 21)),
]
FAKE_CLASS = 0
RANDOM_SEED = 0
TEST_SIZE = 0.30


def score_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "fake_recall": recall_score(labels, predictions, pos_label=FAKE_CLASS, zero_division=0),
        "real_recall": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "confusion_matrix": matrix,
    }


def predict_onnx(model_path: Path, features: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    probabilities = session.run(
        ["probabilities"],
        {input_name: features.astype(np.float32, copy=False)},
    )[0]
    if probabilities.shape != (len(features), 2):
        raise ValueError(f"Expected two-class probabilities, received {probabilities.shape}")
    return probabilities.argmax(axis=1).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("audio_features_stringremoved.csv"),
        help="York feature CSV with 26 model columns and LABEL",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            Path(__file__).parents[2]
            / "app"
            / "src"
            / "main"
            / "assets"
            / "models"
            / "lgbmv2.onnx"
        ),
        help="Existing LightGBM ONNX model; replaced only after a holdout improvement",
    )
    parser.add_argument("--iterations", type=int, default=32)
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be positive")
    data = pd.read_csv(args.dataset)
    missing = sorted(set([*FEATURES, "LABEL"]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[FEATURES].isna().any().any() or data["LABEL"].isna().any():
        raise ValueError("Dataset contains missing feature values or labels")

    labels = data["LABEL"].to_numpy(dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected labels FAKE=0 and REAL=1")
    features = data[FEATURES].to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Dataset contains non-finite feature values")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    print(
        f"Dataset: {len(labels)} rows; train={len(y_train)}, holdout={len(y_test)}; "
        f"features={len(FEATURES)}; split=random seed {RANDOM_SEED}"
    )

    baseline = score_predictions(y_test, predict_onnx(args.model, x_test))
    print(
        f"Current ONNX holdout: accuracy={baseline['accuracy']:.4%}, "
        f"balanced_accuracy={baseline['balanced_accuracy']:.4%}, "
        f"fake_recall={baseline['fake_recall']:.4%}, "
        f"real_recall={baseline['real_recall']:.4%}"
    )
    print(
        "Current confusion matrix [true fake, true real] x [pred fake, pred real]:\n"
        f"{baseline['confusion_matrix']}"
    )

    estimator = lgb.LGBMClassifier(
        objective="binary",
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=max(1, min(os.cpu_count() or 1, 8)),
    )
    parameters = {
        "n_estimators": [100, 180, 280, 400, 550, 700],
        "learning_rate": loguniform(0.015, 0.15),
        "num_leaves": randint(7, 48),
        "max_depth": [-1, 4, 6, 8, 10],
        "min_child_samples": randint(10, 81),
        "subsample": uniform(0.65, 0.35),
        "subsample_freq": [1],
        "colsample_bytree": uniform(0.65, 0.35),
        "reg_lambda": loguniform(1e-3, 10.0),
    }
    search = RandomizedSearchCV(
        estimator,
        parameters,
        n_iter=args.iterations,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        refit=True,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=1,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    candidate_predictions = search.best_estimator_.predict(x_test)
    candidate = score_predictions(y_test, candidate_predictions)
    print(f"Selected hyperparameters: {search.best_params_}")
    print(
        f"Tuned holdout: accuracy={candidate['accuracy']:.4%}, "
        f"balanced_accuracy={candidate['balanced_accuracy']:.4%}, "
        f"fake_recall={candidate['fake_recall']:.4%}, "
        f"real_recall={candidate['real_recall']:.4%}"
    )
    print(
        "Tuned confusion matrix [true fake, true real] x [pred fake, pred real]:\n"
        f"{candidate['confusion_matrix']}"
    )
    print(f"Holdout accuracy change: {(candidate['accuracy'] - baseline['accuracy']):+.4%}")

    if candidate["accuracy"] <= baseline["accuracy"]:
        print("Not exporting: tuned model did not improve holdout accuracy.")
        return

    final_model = lgb.LGBMClassifier(**search.best_params_)
    final_model.set_params(
        objective="binary",
        random_state=RANDOM_SEED,
        verbosity=-1,
        n_jobs=max(1, min(os.cpu_count() or 1, 8)),
    )
    final_model.fit(features, labels)
    onnx_model = convert_lightgbm(
        final_model,
        initial_types=[("float_input", FloatTensorType([None, len(FEATURES)]))],
        target_opset=14,
        zipmap=False,
    )
    onnx.checker.check_model(onnx_model)

    args.model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{args.model.stem}-",
        suffix=".onnx",
        dir=args.model.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(onnx_model, temporary_path)
        converted_predictions = predict_onnx(temporary_path, x_test)
        full_model_predictions = final_model.predict(x_test)
        parity = accuracy_score(full_model_predictions, converted_predictions)
        if parity != 1.0:
            raise RuntimeError(f"ONNX export prediction parity failed: {parity:.4%}")
        os.replace(temporary_path, args.model)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"Exported full-data model to {args.model}; ONNX predictions exactly "
        "matched the tuned LightGBM estimator on the holdout."
    )


if __name__ == "__main__":
    main()
