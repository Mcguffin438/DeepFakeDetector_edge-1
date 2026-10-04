"""Expand the attributed app audio set to 100 clips per source category."""

import csv
import io
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ROOT = Path(__file__).parents[2]
AUDIO_ROOT = ROOT / "app" / "src" / "main" / "assets" / "samples" / "audio"
DATASET_ROOT = AUDIO_ROOT / "dataset"
MANIFEST_PATH = DATASET_ROOT / "MANIFEST.csv"
REVISION = "fcf5344bb7f82b54b6b932291326d29750ef1e82"
REPOSITORY = "garystafford/deepfake-audio-detection"
TARGET_RATE = 16_000
TARGET_COUNT = 100
CATEGORIES = {
    "york_real": ("real", "real", r"^real/yt_\d{4}"),
    "amazon_polly_fake": ("fake", "fake", r"^fake/po_\d{4}"),
    "elevenlabs_fake": ("fake", "fake", r"^fake/el_\d{4}"),
}


def fetch_json(url: str) -> list[dict[str, object]]:
    request = urllib.request.Request(url, headers={"User-Agent": "DeepFakeDetectorEdge/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        import json

        data = json.load(response)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list from dataset API: {url}")
    return data


def source_group(source_path: str) -> str:
    match = re.search(r"(yt_\d{4}|(?:po|el)_\d{4})", source_path)
    if not match:
        raise ValueError(f"Cannot determine source recording group from {source_path}")
    return match.group(1)


def choose_paths(
    available: list[str],
    existing_paths: list[str],
    group_pattern: str,
) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for source_path in sorted(available):
        if re.match(group_pattern, source_path):
            groups[source_group(source_path)].append(source_path)

    selected = list(dict.fromkeys(path for path in existing_paths if path in available))
    if len(selected) > TARGET_COUNT:
        raise ValueError(f"Existing category already has {len(selected)} clips")
    selected_set = set(selected)
    for paths in groups.values():
        paths[:] = [path for path in paths if path not in selected_set]

    group_names = sorted(groups)
    while len(selected) < TARGET_COUNT:
        added_in_round = False
        for group in group_names:
            if groups[group] and len(selected) < TARGET_COUNT:
                selected.append(groups[group].pop(0))
                added_in_round = True
        if not added_in_round:
            raise ValueError(f"Only {len(selected)} source clips are available")
    return selected


def convert_clip(source_path: str, destination: Path) -> None:
    url = (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{quote(source_path, safe='/')}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "DeepFakeDetectorEdge/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        encoded_audio = response.read()
    audio, sample_rate = sf.read(io.BytesIO(encoded_audio), dtype="float32", always_2d=True)
    if audio.size == 0 or sample_rate <= 0:
        raise ValueError(f"Empty or invalid source audio: {source_path}")
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != TARGET_RATE:
        divisor = np.gcd(sample_rate, TARGET_RATE)
        mono = resample_poly(
            mono,
            TARGET_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, mono, TARGET_RATE, subtype="PCM_16")


def main() -> None:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as manifest_file:
        original_rows = list(csv.DictReader(manifest_file))

    tree_url = (
        f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}"
    )
    source_files = {
        "real": fetch_json(f"{tree_url}/real"),
        "fake": fetch_json(f"{tree_url}/fake"),
    }
    existing_by_category = {
        category: [
            row["source_dataset_path"]
            for row in original_rows
            if row["category"] == category
        ]
        for category in CATEGORIES
    }
    output_rows = []
    for category, (source_directory, label, group_pattern) in CATEGORIES.items():
        available = [
            str(entry["path"])
            for entry in source_files[source_directory]
            if entry.get("type") == "file"
            and str(entry.get("path", "")).endswith(".flac")
            and re.match(group_pattern, str(entry.get("path", "")))
        ]
        selected = choose_paths(
            available,
            existing_by_category[category],
            group_pattern,
        )
        existing_by_path = {
            row["source_dataset_path"]: row
            for row in original_rows
            if row["category"] == category
        }
        for index, source_path in enumerate(selected, start=1):
            previous = existing_by_path.get(source_path)
            if previous:
                output_rows.append(previous)
                continue
            source_name = Path(source_path).stem
            wav_name = f"{category}_{index:03d}_{source_name}.wav"
            asset_path = f"dataset/{category}/{wav_name}"
            destination = AUDIO_ROOT / asset_path
            if not destination.exists():
                convert_clip(source_path, destination)
            output_rows.append(
                {
                    "asset_path": asset_path,
                    "category": category,
                    "reference_label": label,
                    "source_dataset_path": source_path,
                }
            )
        print(f"{category}: prepared {len(selected)} clips")

    output_rows.sort(key=lambda row: (row["category"], row["asset_path"]))
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "asset_path",
                "category",
                "reference_label",
                "source_dataset_path",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} manifest rows to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
