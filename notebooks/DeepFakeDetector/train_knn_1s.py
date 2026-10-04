"""Train and export the one-second KNN used by the Android app.

Expected dataset layout:
    dataset-root/
        real/*.flac
        fake/*.flac

Dataset: garystafford/deepfake-audio-detection (CC BY 4.0).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
import soundfile as sf
from scipy.signal import resample_poly
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


SAMPLE_RATE = 16_000
FFT_SIZE = 780
HOP_LENGTH = 195
FEATURE_COUNT = 25


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def make_mel_filterbank() -> np.ndarray:
    mel_min = hz_to_mel(80.0)
    mel_max = hz_to_mel(8000.0)
    mel_points = [
        mel_min + (mel_max - mel_min) * index / 65
        for index in range(66)
    ]
    bin_indices = [
        min(int(391 * mel_to_hz(value) / 8000.0), 390)
        for value in mel_points
    ]
    filterbank = np.zeros((64, 391), dtype=np.float32)
    for mel_index in range(64):
        left, center, right = bin_indices[mel_index : mel_index + 3]
        for bin_index in range(left, right + 1):
            if bin_index < 391:
                if bin_index < center and center > left:
                    filterbank[mel_index, bin_index] = (
                        np.float32(bin_index - left) / np.float32(center - left)
                    )
                elif bin_index >= center and right > center:
                    filterbank[mel_index, bin_index] = (
                        np.float32(right - bin_index) / np.float32(right - center)
                    )
    return filterbank


MEL_FILTERBANK = make_mel_filterbank()
DCT_MATRIX = np.cos(
    np.pi
    * np.arange(64)[:, None]
    * (2 * np.arange(64)[None, :] + 1)
    / (2 * 64)
).astype(np.float32)
HAMMING_WINDOW = (
    0.54
    - 0.46
    * np.cos(2 * np.pi * np.arange(FFT_SIZE) / (FFT_SIZE - 1))
).astype(np.float32)
FREQUENCIES = np.arange(FFT_SIZE // 2 + 1, dtype=np.float64) * (
    SAMPLE_RATE / FFT_SIZE
)


def extract_features(pcm: np.ndarray) -> np.ndarray:
    waveform = pcm.astype(np.float32) / np.float32(32767.0)
    if waveform.size < FFT_SIZE:
        waveform = np.pad(waveform, (0, FFT_SIZE - waveform.size))

    frames = np.lib.stride_tricks.sliding_window_view(
        waveform, FFT_SIZE
    )[::HOP_LENGTH]
    spectrum = np.abs(
        np.fft.rfft(frames * HAMMING_WINDOW, n=FFT_SIZE, axis=1)
    ).astype(np.float32)
    mel = spectrum @ MEL_FILTERBANK.T
    db = (20.0 * np.log10(np.maximum(mel, np.float32(1e-10)))).astype(np.float32)
    db = np.maximum(db, np.max(db) - np.float32(80.0))
    mfcc = DCT_MATRIX @ db.T

    stats_frames = np.stack(
        [waveform[start : start + FFT_SIZE]
         for start in np.arange(len(frames)) * HOP_LENGTH]
    )
    rms = np.sqrt(np.mean(stats_frames.astype(np.float64) ** 2, axis=1))
    zero_crossing = np.count_nonzero(
        (stats_frames[:, 1:] >= 0) != (stats_frames[:, :-1] >= 0), axis=1
    ) / (FFT_SIZE - 1)

    power = spectrum.astype(np.float64) ** 2
    energy = power.sum(axis=1)
    centroid = np.divide(
        power @ FREQUENCIES,
        energy,
        out=np.zeros_like(energy),
        where=energy > 0,
    )
    bandwidth = np.sqrt(
        np.divide(
            (
                power
                * (FREQUENCIES[None, :] - centroid[:, None]) ** 2
            ).sum(axis=1),
            energy,
            out=np.zeros_like(energy),
            where=energy > 0,
        )
    )
    rolloff_bins = np.argmax(
        np.cumsum(power, axis=1) >= energy[:, None] * 0.85,
        axis=1,
    )
    rolloff = np.where(energy > 0, FREQUENCIES[rolloff_bins], 0.0)

    features = np.empty(FEATURE_COUNT, dtype=np.float32)
    features[:5] = (
        rms.mean(),
        centroid.mean(),
        bandwidth.mean(),
        rolloff.mean(),
        zero_crossing.mean(),
    )
    features[5:] = mfcc[:20].mean(axis=1)
    return features


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    labels = []
    groups = []
    for folder, label in (("real", 0), ("fake", 1)):
        files = sorted((root / folder).glob("*.flac"))
        if not files:
            raise FileNotFoundError(f"No FLAC files found in {root / folder}")
        for path in files:
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sample_rate != SAMPLE_RATE:
                divisor = math.gcd(sample_rate, SAMPLE_RATE)
                mono = resample_poly(
                    mono,
                    SAMPLE_RATE // divisor,
                    sample_rate // divisor,
                )
            pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
            if pcm.size < SAMPLE_RATE:
                continue

            features.append(extract_features(pcm[:SAMPLE_RATE]))
            labels.append(label)
            # Keep clips derived from the same source voice/video in one fold.
            groups.append("_".join(path.stem.split("_")[:2]))

    if len(set(labels)) != 2:
        raise ValueError("Dataset must contain both real and fake clips")
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def make_pipeline(scaler_name: str, metric: str, neighbors: int, weights: str):
    scaler = RobustScaler() if scaler_name == "robust" else StandardScaler()
    return Pipeline(
        [
            ("scaler", scaler),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=neighbors,
                    metric=metric,
                    weights=weights,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def select_parameters(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[str, str, int, str, float, float]:
    folds = GroupKFold(n_splits=5)
    candidates = [
        (scaler, metric, neighbors, weights)
        for scaler in ("robust", "standard")
        for metric in ("euclidean", "cosine", "manhattan")
        for neighbors in (5, 11, 15)
        for weights in ("uniform", "distance")
    ]
    best = None
    for config in candidates:
        predictions = np.zeros_like(labels)
        for train, validation in folds.split(features, labels, groups):
            model = make_pipeline(*config).fit(features[train], labels[train])
            predictions[validation] = model.predict(features[validation])
        score = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, zero_division=0)
        if best is None or (score, recall) > (best[4], best[5]):
            best = (*config, score, recall)
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing real/ and fake/ subdirectories of FLAC files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/src/main/assets/models/knn_modelv2.onnx"),
    )
    args = parser.parse_args()

    features, labels, groups = load_dataset(args.dataset_root)
    scaler, metric, neighbors, weights, cv_accuracy, cv_recall = select_parameters(
        features, labels, groups
    )
    print(
        f"Selected {scaler}/{metric}/k={neighbors}/{weights}; "
        f"5-fold grouped CV accuracy={cv_accuracy:.4f}, "
        f"fake recall={cv_recall:.4f}"
    )

    model = make_pipeline(scaler, metric, neighbors, weights).fit(features, labels)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, FEATURE_COUNT]))],
        options={id(model.named_steps["knn"]): {"zipmap": False}},
        target_opset=14,
    )
    onnx.checker.check_model(onnx_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(onnx_model, args.output)
    print(f"Exported model trained on {len(labels)} one-second clips to {args.output}")


if __name__ == "__main__":
    main()
