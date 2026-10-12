"""Train and export experimental MobileNetV3/V5-style audio classifiers.

The model consumes one-second, 16 kHz PCM windows converted to the app's
three-channel log-Mel/MFCC/LFCC representation. Clips from a common source
recording are kept in the same grouped holdout fold.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SAMPLE_RATE = 16_000
FFT_SIZE = 780
HOP_LENGTH = 195
FEATURE_BINS = 64
WINDOW_SAMPLES = SAMPLE_RATE
SEED = 42
CATEGORIES = {
    "york_real": 0,
    "amazon_polly_fake": 1,
    "elevenlabs_fake": 1,
}


def make_mel_filter_bank() -> np.ndarray:
    mel_min = np.float32(2595.0 * math.log10(1.0 + 80.0 / 700.0))
    mel_max = np.float32(2595.0 * math.log10(1.0 + 8000.0 / 700.0))
    mel_points = np.array(
        [
            mel_min + np.float32((mel_max - mel_min) * np.float32(i) / np.float32(FEATURE_BINS + 1))
            for i in range(FEATURE_BINS + 2)
        ],
        dtype=np.float32,
    )
    hz_points = np.array(
        [np.float32(700.0) * (np.float32(10.0) ** np.float32(mel / np.float32(2595.0)) - np.float32(1.0))
         for mel in mel_points],
        dtype=np.float32,
    )
    bins = np.array(
        [
            min(int(np.float32((FFT_SIZE // 2 + 1) * hz / np.float32(SAMPLE_RATE / 2))), FFT_SIZE // 2)
            for hz in hz_points
        ]
    )
    bank = np.zeros((FEATURE_BINS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for index in range(FEATURE_BINS):
        left, center, right = bins[index : index + 3]
        for bin_index in range(left, right + 1):
            if bin_index < FFT_SIZE // 2 + 1:
                if bin_index < center and center > left:
                    bank[index, bin_index] = np.float32(bin_index - left) / np.float32(center - left)
                elif bin_index >= center and right > center:
                    bank[index, bin_index] = np.float32(right - bin_index) / np.float32(right - center)
    return bank


MEL_FILTER_BANK = make_mel_filter_bank()
HAMMING_WINDOW = (
    0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(FFT_SIZE) / (FFT_SIZE - 1))
).astype(np.float32)
DCT_MATRIX = np.array(
    [
        [math.cos(math.pi * coefficient * (2 * index + 1) / (2 * FEATURE_BINS))
         for index in range(FEATURE_BINS)]
        for coefficient in range(FEATURE_BINS)
    ],
    dtype=np.float32,
)


def to_db(spectrogram: np.ndarray) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(spectrogram, np.float32(1e-10)))
    return np.maximum(db, np.max(db) - np.float32(80.0)).astype(np.float32)


def extract_features(pcm: np.ndarray) -> np.ndarray:
    waveform = pcm.astype(np.float32) / np.float32(32767.0)
    if waveform.size < FFT_SIZE:
        waveform = np.pad(waveform, (0, FFT_SIZE - waveform.size))
    frames = np.lib.stride_tricks.sliding_window_view(waveform, FFT_SIZE)[::HOP_LENGTH]
    spectrum = np.abs(np.fft.rfft(frames * HAMMING_WINDOW, axis=1)).astype(np.float32)

    mel = (spectrum @ MEL_FILTER_BANK.T).T
    mel_db = to_db(mel)
    mfcc = DCT_MATRIX @ mel_db

    frequency_bins = spectrum.shape[1]
    frequency_step = frequency_bins / FEATURE_BINS
    linear = np.zeros((FEATURE_BINS, spectrum.shape[0]), dtype=np.float32)
    for filter_index in range(FEATURE_BINS):
        start = int(filter_index * frequency_step)
        end = min(int((filter_index + 1) * frequency_step), frequency_bins - 1)
        band = spectrum[:, start : end + 1]
        linear[filter_index] = band.mean(axis=1) if end > start else band[:, 0]
    lfcc = DCT_MATRIX @ to_db(linear)
    return np.stack((mel_db, mfcc, lfcc)).astype(np.float32)


def get_source_group(path: Path) -> str:
    match = re.search(r"(yt_\d{4}|(?:po|el)_\d{4})", path.stem)
    if not match:
        raise ValueError(f"Cannot identify source recording for {path.name}")
    return match.group(1)


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = []
    labels = []
    groups = []
    for category, label in CATEGORIES.items():
        files = sorted((root / category).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No WAV clips found in {root / category}")
        for path in files:
            audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if source_rate != SAMPLE_RATE:
                divisor = math.gcd(source_rate, SAMPLE_RATE)
                mono = resample_poly(
                    mono,
                    SAMPLE_RATE // divisor,
                    source_rate // divisor,
                )
            pcm = np.rint(np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
            if not pcm.size:
                continue
            group = f"{category}:{get_source_group(path)}"
            for start in range(0, pcm.size, WINDOW_SAMPLES):
                window = pcm[start : start + WINDOW_SAMPLES]
                if window.size < WINDOW_SAMPLES:
                    window = np.pad(window, (0, WINDOW_SAMPLES - window.size))
                examples.append(extract_features(window))
                labels.append(label)
                groups.append(group)
    if not examples or set(labels) != {0, 1}:
        raise ValueError("The WAV dataset must contain both real and fake audio")
    features = np.stack(examples).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced non-finite values")
    return features, np.asarray(labels, dtype=np.int64), np.asarray(groups)


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduced_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, reduced_channels, 1)
        self.expand = nn.Conv2d(reduced_channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = torch.relu(self.reduce(scale))
        scale = nn.functional.hardsigmoid(self.expand(scale))
        return x * scale


class MobileNetV3Block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        expanded_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        activation: str,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        if expanded_channels != input_channels:
            layers.extend(
                [
                    nn.Conv2d(input_channels, expanded_channels, 1, bias=False),
                    nn.BatchNorm2d(expanded_channels),
                    nn.Hardswish() if activation == "HS" else nn.ReLU(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    expanded_channels,
                    expanded_channels,
                    kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    groups=expanded_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(expanded_channels),
            ]
        )
        if use_se:
            layers.append(
                SqueezeExcitation(
                    expanded_channels,
                    make_divisible(expanded_channels / 4),
                )
            )
        layers.extend(
            [
                nn.Hardswish() if activation == "HS" else nn.ReLU(inplace=True),
                nn.Conv2d(expanded_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and input_channels == output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        return x + result if self.residual else result


class MobileNetV3SmallAudio(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("feature_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("feature_std", std.view(1, 3, 1, 1).clamp_min(1e-4))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )
        settings = [
            (16, 16, 3, 2, True, "RE"),
            (72, 24, 3, 2, False, "RE"),
            (88, 24, 3, 1, False, "RE"),
            (96, 40, 5, 2, True, "HS"),
            (240, 40, 5, 1, True, "HS"),
            (240, 40, 5, 1, True, "HS"),
            (120, 48, 5, 1, True, "HS"),
            (144, 48, 5, 1, True, "HS"),
            (288, 96, 5, 2, True, "HS"),
            (576, 96, 5, 1, True, "HS"),
            (576, 96, 5, 1, True, "HS"),
        ]
        blocks = []
        input_channels = 16
        for expanded, output, kernel, stride, use_se, activation in settings:
            blocks.append(
                MobileNetV3Block(
                    input_channels,
                    expanded,
                    output,
                    kernel,
                    stride,
                    use_se,
                    activation,
                )
            )
            input_channels = output
        self.features = nn.Sequential(*blocks)
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
        x = self.stem(x)
        x = self.features(x)
        x = self.final_conv(x)
        return torch.softmax(self.classifier(self.pool(x).flatten(1)), dim=1)


class GhostModule(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, activation: bool = True):
        super().__init__()
        primary_channels = math.ceil(output_channels / 2)
        cheap_channels = output_channels - primary_channels
        self.primary = nn.Sequential(
            nn.Conv2d(input_channels, primary_channels, 1, bias=False),
            nn.BatchNorm2d(primary_channels),
            nn.Hardswish() if activation else nn.Identity(),
        )
        self.cheap = (
            nn.Sequential(
                nn.Conv2d(
                    primary_channels,
                    cheap_channels,
                    3,
                    padding=1,
                    groups=primary_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(cheap_channels),
                nn.Hardswish() if activation else nn.Identity(),
            )
            if cheap_channels
            else None
        )
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        if self.cheap is None:
            return primary
        return torch.cat((primary, self.cheap(primary)), dim=1)[:, : self.output_channels]


class MobileNetV5Block(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        expansion: int,
        stride: int,
        use_se: bool,
    ):
        super().__init__()
        expanded_channels = make_divisible(input_channels * expansion)
        layers: list[nn.Module] = [GhostModule(input_channels, expanded_channels)]
        if stride > 1:
            layers.extend(
                [
                    nn.Conv2d(
                        expanded_channels,
                        expanded_channels,
                        5,
                        stride=stride,
                        padding=2,
                        groups=expanded_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(expanded_channels),
                    nn.Hardswish(),
                ]
            )
        if use_se:
            layers.append(
                SqueezeExcitation(
                    expanded_channels,
                    make_divisible(expanded_channels / 4),
                )
            )
        layers.append(GhostModule(expanded_channels, output_channels, activation=False))
        self.block = nn.Sequential(*layers)
        self.shortcut = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    input_channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=input_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(input_channels),
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.hardswish(self.block(x) + self.shortcut(x))


class ExperimentalMobileNetV5Audio(nn.Module):
    """Project-specific Ghost-bottleneck audio model, not an official MobileNetV5."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("feature_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("feature_std", std.view(1, 3, 1, 1).clamp_min(1e-4))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.Hardswish(),
        )
        settings = [
            (24, 24, 1, 1, False),
            (24, 32, 2, 2, False),
            (32, 32, 2, 1, False),
            (32, 48, 3, 2, True),
            (48, 48, 3, 1, True),
            (48, 72, 4, 2, True),
            (72, 72, 4, 1, True),
            (72, 96, 4, 2, True),
            (96, 96, 4, 1, True),
        ]
        blocks = []
        for input_channels, output_channels, expansion, stride, use_se in settings:
            blocks.append(
                MobileNetV5Block(
                    input_channels,
                    output_channels,
                    expansion,
                    stride,
                    use_se,
                )
            )
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv2d(96, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(384, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
        x = self.stem(x)
        x = self.features(x)
        return torch.softmax(self.head(x), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).parents[2]
    parser.add_argument(
        "--samples",
        type=Path,
        default=repo_root / "app" / "src" / "main" / "assets" / "samples" / "audio" / "dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--architecture",
        choices=("v3", "v5"),
        default="v3",
        help="Experimental model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            repo_root
            / "app"
            / "src"
            / "main"
            / "assets"
            / "models"
            / f"mobilenet{args.architecture}_audio.onnx"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    features, labels, groups = load_dataset(args.samples)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_indices, validation_indices = next(splitter.split(features, labels, groups))
    if set(groups[train_indices]) & set(groups[validation_indices]):
        raise RuntimeError("Source recordings leaked across training and validation sets")

    train_x = torch.from_numpy(features[train_indices])
    train_y = torch.from_numpy(labels[train_indices])
    validation_x = torch.from_numpy(features[validation_indices])
    validation_y = labels[validation_indices]
    mean = train_x.mean(dim=(0, 2, 3))
    std = train_x.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-4)
    model_type = MobileNetV3SmallAudio if args.architecture == "v3" else ExperimentalMobileNetV5Audio
    model = model_type(mean, std)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_function = nn.NLLLoss()
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            probabilities = model(batch_x)
            loss = loss_function(torch.log(probabilities.clamp_min(1e-8)), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_probabilities = model(validation_x)
            validation_loss = loss_function(
                torch.log(validation_probabilities.clamp_min(1e-8)),
                torch.from_numpy(validation_y),
            ).item()
        print(f"epoch={epoch + 1} grouped_validation_loss={validation_loss:.5f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= 5:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probabilities = model(validation_x)[:, 1].numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    print(
        f"grouped_holdout={len(validation_indices)}/{len(labels)} "
        f"accuracy={accuracy_score(validation_y, predictions):.3%} "
        f"balanced_accuracy={balanced_accuracy_score(validation_y, predictions):.3%}"
    )
    print("Confusion matrix [real, fake]:")
    print(confusion_matrix(validation_y, predictions, labels=[0, 1]))
    print("This small bundled dataset supports only experimental, non-clinical evaluation.")

    full_x = torch.from_numpy(features)
    full_y = torch.from_numpy(labels)
    full_mean = full_x.mean(dim=(0, 2, 3))
    full_std = full_x.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-4)
    final_model = model_type(full_mean, full_std)
    full_loader = DataLoader(
        TensorDataset(full_x, full_y),
        batch_size=args.batch_size,
        shuffle=True,
    )
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=0.001,
        weight_decay=0.01,
    )
    for _ in range(best_epoch):
        final_model.train()
        for batch_x, batch_y in full_loader:
            final_optimizer.zero_grad(set_to_none=True)
            probabilities = final_model(batch_x)
            loss = loss_function(torch.log(probabilities.clamp_min(1e-8)), batch_y)
            loss.backward()
            final_optimizer.step()
    final_model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, FEATURE_BINS, WINDOW_SAMPLES // HOP_LENGTH), dtype=torch.float32)
    torch.onnx.export(
        final_model,
        dummy,
        args.output,
        input_names=["audio_features"],
        output_names=["class_probabilities"],
        dynamic_axes={
            "audio_features": {0: "batch", 3: "time"},
            "class_probabilities": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.output))
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    test_probabilities = session.run(
        ["class_probabilities"],
        {"audio_features": features[validation_indices[:1]]},
    )[0]
    with torch.no_grad():
        expected_probabilities = final_model(validation_x[:1]).numpy()[0]
    if not np.allclose(test_probabilities[0], expected_probabilities, atol=1e-4):
        raise RuntimeError("Exported ONNX probabilities do not match the trained PyTorch model")
    print(f"Exported verified experimental MobileNet{args.architecture.upper()} ONNX model to {args.output}")


if __name__ == "__main__":
    main()
