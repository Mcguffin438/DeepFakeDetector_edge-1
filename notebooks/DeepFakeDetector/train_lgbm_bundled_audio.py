"""Exploratory LightGBM retraining on the 30 bundled Android audio samples.

The split is stratified by source category so each provider has separate
training and held-out clips. The held-out clips are never used to fit the
exported model. Results are exploratory only because there are just ten clips
per source category.
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
import soundfile as sf
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split


SAMPLE_RATE = 16_000
FFT_SIZE = 780
HOP_LENGTH = 195
NUM_MELS = 64
FEATURE_COUNT = 26
FAKE_CLASS = 0
SEED = 42
CATEGORIES = {
    "york_real": 1,
    "amazon_polly_fake": 0,
    "elevenlabs_fake": 0,
}


def mel_filter_bank() -> np.ndarray:
    f32 = np.float32
    mel_min = f32(2595.0) * f32(np.log10(f32(1.0) + f32(80.0) / f32(700.0)))
    mel_max = f32(2595.0) * f32(
        np.log10(f32(1.0) + f32(8000.0) / f32(700.0))
    )
    mel_points = np.array(
        [f32(mel_min + f32((mel_max - mel_min) * f32(i) / f32(NUM_MELS + 1)))
         for i in range(NUM_MELS + 2)],
        dtype=np.float32,
    )
    hz_points = np.array(
        [
            f32(f32(700.0) * (f32(f32(10.0) ** f32(mel / f32(2595.0))) - f32(1.0)))
            for mel in mel_points
        ],
        dtype=np.float32,
    )
    bins = np.array(
        [
            min(int(f32(f32(FFT_SIZE // 2 + 1) * hz / f32(SAMPLE_RATE / 2))), FFT_SIZE // 2)
            for hz in hz_points
        ]
    )
    bank = np.zeros((NUM_MELS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for mel_index in range(NUM_MELS):
        left, center, right = bins[mel_index : mel_index + 3]
        if center > left:
            indexes = np.arange(left, center)
            bank[mel_index, indexes] = (indexes - left) / (center - left)
        if right > center:
            indexes = np.arange(center, right + 1)
            bank[mel_index, indexes] = (right - indexes) / (right - center)
    return bank


def extract_android_features(path: Path, bank: np.ndarray) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {sample_rate}")

    mono = np.zeros(audio.shape[0], dtype=np.float32)
    for channel in range(audio.shape[1]):
        mono += audio[:, channel].astype(np.float32) / np.float32(32767.0)
    mono /= np.float32(audio.shape[1])
    mono = mono[: SAMPLE_RATE * 6]
    waveform = np.pad(mono, (0, max(0, FFT_SIZE - mono.size)))

    frame_count = (waveform.size - FFT_SIZE) // HOP_LENGTH + 1
    frames = np.stack(
        [
            waveform[index * HOP_LENGTH : index * HOP_LENGTH + FFT_SIZE]
            for index in range(frame_count)
        ]
    )
    window = np.array(
        [
            np.float32(0.54 - 0.46 * np.cos(2.0 * np.pi * i / (FFT_SIZE - 1)))
            for i in range(FFT_SIZE)
        ],
        dtype=np.float32,
    )
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)

    mel = np.asarray(spectrum @ bank.T, dtype=np.float32).T
    db = np.asarray(20.0 * np.log10(np.maximum(mel, np.float32(1e-10))), dtype=np.float32)
    db = np.maximum(db, np.max(db) - np.float32(80.0))
    dct = np.array(
        [
            [np.cos(np.pi * coefficient * (2 * index + 1) / (2 * NUM_MELS))
             for index in range(NUM_MELS)]
            for coefficient in range(20)
        ],
        dtype=np.float64,
    )
    mfcc = np.asarray(dct @ db.astype(np.float64), dtype=np.float32)

    values = np.zeros(25, dtype=np.float64)
    chroma = np.zeros(12, dtype=np.float32)
    frequencies = np.arange(FFT_SIZE // 2 + 1, dtype=np.float64) * SAMPLE_RATE / FFT_SIZE
    for frame_index, magnitude in enumerate(spectrum):
        start = frame_index * HOP_LENGTH
        frame = waveform[start : min(start + FFT_SIZE, waveform.size)]
        square_sum = np.sum(frame.astype(np.float64) ** 2)
        values[0] += np.sqrt(square_sum / max(1, frame.size))
        crossings = np.count_nonzero((frame[1:] >= 0.0) != (frame[:-1] >= 0.0))
        values[4] += crossings / max(1, frame.size - 1)

        energy = magnitude.astype(np.float64) ** 2
        energy_sum = np.sum(energy)
        if energy_sum > 0.0:
            centroid = np.sum(frequencies * energy) / energy_sum
            values[1] += centroid
            values[2] += np.sqrt(np.sum(((frequencies - centroid) ** 2) * energy) / energy_sum)
            rolloff_index = np.searchsorted(np.cumsum(energy), energy_sum * 0.85)
            values[3] += frequencies[min(rolloff_index, frequencies.size - 1)]

        pitches = np.zeros(12, dtype=np.float32)
        midi_notes = 69.0 + 12.0 * np.log2(frequencies[1:] / 440.0)
        pitch_classes = np.floor(midi_notes + 0.5).astype(np.int64) % 12
        np.add.at(pitches, pitch_classes, magnitude[1:])
        max_pitch = np.max(pitches)
        if max_pitch > 0:
            chroma += pitches / max_pitch
        values[5:] += mfcc[:, frame_index]

    values = (values / frame_count).astype(np.float32)
    chroma_mean = np.float32(np.mean(chroma, dtype=np.float64)) / np.float32(frame_count)
    return np.concatenate(([chroma_mean], values))


def onnx_fake_probabilities(path: Path, features: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features.astype(np.float32, copy=False)})
    for output in outputs:
        if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[1] == 2:
            return output[:, FAKE_CLASS]
    raise ValueError("Could not find a two-class probability output in the ONNX model")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).parents[2] / "app" / "src" / "main" / "assets" / "samples" / "audio" / "dataset",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parents[2] / "app" / "src" / "main" / "assets" / "models" / "lgbmv2.onnx",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    bank = mel_filter_bank()
    rows = []
    for category, label in CATEGORIES.items():
        for path in sorted((args.samples / category).glob("*.wav")):
            rows.append((path, category, label, extract_android_features(path, bank)))
    if len(rows) != 30:
        raise ValueError(f"Expected 30 bundled WAVs, found {len(rows)}")

    features = np.stack([row[3] for row in rows]).astype(np.float32)
    labels = np.array([row[2] for row in rows], dtype=np.int64)
    categories = np.array([row[1] for row in rows])
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values")

    indexes = np.arange(len(rows))
    train_indexes, test_indexes = train_test_split(
        indexes,
        test_size=0.30,
        random_state=args.seed,
        stratify=categories,
    )
    x_train, y_train = features[train_indexes], labels[train_indexes]
    x_test, y_test = features[test_indexes], labels[test_indexes]
    test_categories = categories[test_indexes]

    print(
        f"Android-matched audio features: {len(rows)} clips; "
        f"train={len(train_indexes)}, held-out={len(test_indexes)}; "
        f"input width={features.shape[1]}"
    )
    print("Held-out clips (kept out of model fitting):")
    for index in test_indexes:
        print(f"  {rows[index][1]}/{rows[index][0].name}")

    baseline_probabilities = onnx_fake_probabilities(args.model, x_test)
    baseline_predictions = (baseline_probabilities > 0.5).astype(np.int64)
    print(
        f"Current ONNX held-out: accuracy={accuracy_score(y_test, baseline_predictions):.3%}, "
        f"balanced_accuracy={balanced_accuracy_score(y_test, baseline_predictions):.3%}"
    )

    configurations = [
        {"n_estimators": trees, "num_leaves": leaves, "max_depth": depth, "min_child_samples": child}
        for trees in (25, 60, 120)
        for leaves, depth, child in ((3, 3, 1), (5, 3, 2), (7, -1, 3))
    ]
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    cv_results = []
    for params in configurations:
        fold_scores = []
        for fit_indexes, validation_indexes in folds.split(x_train, categories[train_indexes]):
            estimator = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.05,
                reg_lambda=1.0,
                class_weight="balanced",
                random_state=args.seed,
                verbosity=-1,
                n_jobs=max(1, min(os.cpu_count() or 1, 4)),
                **params,
            )
            estimator.fit(x_train[fit_indexes], y_train[fit_indexes])
            fold_scores.append(
                balanced_accuracy_score(
                    y_train[validation_indexes],
                    estimator.predict(x_train[validation_indexes]),
                )
            )
        cv_results.append((float(np.mean(fold_scores)), params))
    cv_results.sort(key=lambda result: result[0], reverse=True)
    best_cv_score, best_params = cv_results[0]
    print(f"Selected parameters by training-only CV (balanced accuracy={best_cv_score:.3%}): {best_params}")

    candidate = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=args.seed,
        verbosity=-1,
        n_jobs=max(1, min(os.cpu_count() or 1, 4)),
        **best_params,
    )
    candidate.fit(x_train, y_train)
    candidate_predictions = candidate.predict(x_test)
    candidate_probabilities = candidate.predict_proba(x_test)[:, FAKE_CLASS]
    baseline_balanced = balanced_accuracy_score(y_test, baseline_predictions)
    candidate_balanced = balanced_accuracy_score(y_test, candidate_predictions)
    print(
        f"Candidate held-out: accuracy={accuracy_score(y_test, candidate_predictions):.3%}, "
        f"balanced_accuracy={candidate_balanced:.3%}, "
        f"confusion[true fake/real, predicted fake/real]=\n"
        f"{confusion_matrix(y_test, candidate_predictions, labels=[0, 1])}"
    )
    for category in sorted(np.unique(test_categories)):
        mask = test_categories == category
        print(
            f"  {category}: {accuracy_score(y_test[mask], candidate_predictions[mask]):.3%} "
            f"({mask.sum()} held-out clips)"
        )

    if candidate_balanced <= baseline_balanced:
        print("Not exporting: candidate did not improve held-out balanced accuracy.")
        return

    converted = convert_lightgbm(
        candidate,
        initial_types=[("float_input", FloatTensorType([None, FEATURE_COUNT]))],
        target_opset=14,
        zipmap=False,
    )
    onnx.checker.check_model(converted)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{args.model.stem}-",
        suffix=".onnx",
        dir=args.model.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        onnx.save_model(converted, temporary_path)
        exported_probabilities = onnx_fake_probabilities(temporary_path, x_test)
        if not np.allclose(exported_probabilities, candidate_probabilities, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Exported ONNX probabilities differ from the selected LightGBM estimator")
        os.replace(temporary_path, args.model)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(
        f"Exported exploratory model trained on {len(train_indexes)} clips to {args.model}; "
        "held-out clips were excluded from fitting."
    )


if __name__ == "__main__":
    main()
