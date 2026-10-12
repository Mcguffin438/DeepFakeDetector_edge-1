# RADD - Real-time Audio Deepfake Detection

A real-time Android deepfake-audio detection app that runs KNN, LightGBM, MobileNetV3, and an experimental MobileNetV5-style classifier locally with ONNX Runtime. It displays separate fake/real scores, per-model inference times, and whether predictions agree.

![KNN training and KNN, LightGBM, MobileNetV3, and MobileNetV5-style inference workflow](images/deepfake_edge_workflow.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Key Technical Achievements](#key-technical-achievements)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Core Components](#core-components)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

## Overview

This project implements a complete pipeline for edge-deployed deepfake detection, from feature extraction to real-time inference.

**Feature Extraction Process:**
- 25 KNN features per live audio chunk: RMS, spectral centroid, bandwidth, rolloff, zero-crossing rate, and 20 MFCC means
- 26 LightGBM inputs: an Android-estimated chroma feature followed by the same 25 features
- MobileNetV3 and experimental MobileNetV5-style inputs: a three-channel log-Mel/MFCC/LFCC spectrogram
- Audio is processed at 16 kHz; call monitoring uses chunks from the built-in microphone and is intended for speakerphone use
- Live monitoring collects one-second chunks and makes a new prediction once per second
- Each model receives features in its respective training order; convolutional model scores are averaged over one-second windows for longer uploaded audio

**Model Deployment:**
- KNN ONNX input shape: `[batch, 25]`; RobustScaler is embedded in the graph
- LightGBM ONNX input shape: `[batch, 26]`; fake is class 0, while the KNN model's fake class is class 1
- MobileNet ONNX input shape: `[batch, 3, 64, time]`; class order is real, fake
- Each ONNX model outputs class probabilities; each score and ONNX inference time is shown separately
- ONNX Runtime uses available execution providers with fallback; CPU inference is supported
- Release builds include ONNX Runtime with QNN and try the Snapdragon GPU backend first, then QNN HTP, NNAPI, and CPU as available. Debug builds use the standard Android ONNX Runtime so they can run on x86_64 emulators. Android emulators do not emulate Snapdragon hardware; QNN GPU acceleration must be verified on a physical compatible Snapdragon device

**Edge Integration:**
- Android foreground service with automatic call detection
- Real-time visual overlay, separate fake/real scores, per-model inference timings, and model agreement status
- 100% local processing - no network connectivity required
- Privacy-first design with in-memory audio processing

## Features

- **Audio Features**: 25-dimensional KNN, 26-dimensional LightGBM, and three-channel spectrogram inputs for MobileNet
- **Edge Deployment**: ONNX Runtime with CPU fallback
- **Real-Time Alerts**: Visual overlay with confidence-based threat detection
- **Auto-Activation**: Foreground service monitors calls automatically
- **Privacy-First**: 100% local processing, no data transmission
- **Models**: KNN, LightGBM, MobileNetV3, and experimental MobileNetV5-style ONNX models in `app/src/main/assets/models/`

## Key Technical Achievements

**Model and evaluation:**
- Audio processing: 25 KNN and 26 LightGBM input features per chunk
- The bundled MobileNetV3 and project-specific MobileNetV5-style models consume the app's three-channel 64-bin log-Mel/MFCC/LFCC features over one-second windows
- MobileNetV3 grouped holdout on the bundled audio windows: 94.07% accuracy and 93.89% balanced accuracy (253 held-out windows)
- Experimental MobileNetV5-style grouped holdout on the same windows: 86.96% accuracy and 86.34% balanced accuracy (253 held-out windows)
- These exploratory scores come from a small bundle of 300 source clips, with windows grouped by source recording within their dataset category. They are not independent, generator-held-out, or live-call/device benchmarks and should not be interpreted as production accuracy
- The bundled KNN is trained on one-second clips from Gary Stafford's CC BY 4.0 dataset, with RobustScaler embedded in the ONNX graph
- Nested, source-grouped five-fold evaluation on that dataset: 81.94% accuracy and 85.85% fake recall; this is not a live-call benchmark
- Evaluation groups keep clips from the same source recording/voice together, but do not hold out entire synthetic generator platforms
- The bundled `lgbmv2.onnx` was retrained using Android-matched features from 300 CC BY 4.0 clips: 100 Gary Stafford real, 100 Amazon Polly synthetic, and 100 ElevenLabs synthetic
- The training split keeps source recordings/voice IDs grouped: 239 clips for training and 61 source-grouped clips held out for evaluation. The selected LightGBM achieved 63.93% accuracy and 61.19% balanced accuracy on that holdout, compared with 22.95% and 23.16% for the previous model using the same feature port/split
- Held-out per-category accuracy was 11/21 (52.38%) Gary Stafford real, 13/14 (92.86%) Amazon Polly synthetic, and 15/26 (57.69%) ElevenLabs synthetic. This indicates limited generalization, particularly to unseen ElevenLabs voices and Gary recordings; it is not a production or live-call accuracy claim
- The holdout comparison is also used as a promotion gate, so it is not an untouched final test. After evaluation, the exported model is refit on all 300 clips; the 61-clip scores describe the pre-refit candidate, not an independent test of the final all-data artifact
- Training clips span 14 Gary Stafford recordings, 30 Polly voice IDs, and 28 ElevenLabs voice IDs. The samples are full 2.5–13 second dataset segments; the app feature extractor uses up to six seconds, not one-second windows
- One-second live predictions have not been validated for real phone calls; Android call capture and acoustics may reduce reliability

**Architecture Highlights:**
- KNN model input: `[batch, 25]`; fake class index 1
- KNN configuration: 15 neighbors, Euclidean distance, distance weighting
- LightGBM model input: `[batch, 26]`; fake class index 0
- MobileNetV3 and MobileNetV5-style input: `[batch, 3, 64, time]`; channels are log-Mel, MFCC, and LFCC; output class order is real then fake
- Alert system: separate fake/real probabilities and inference times per model, with experimental warnings

## Prerequisites

- Android device (API 24+)
- Android Studio for development
- Required call/audio permissions and microphone access
- Speakerphone is intended for call-audio capture; actual capture is device- and Android-version-dependent

## Installation

### Build and Install
```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Grant Permissions
Required permissions:
- Phone state access, Audio recording, System overlay
- Foreground service (microphone + phone call types)

## Usage

### Real-Time Detection
1. **Auto-Activation**: Service starts automatically during phone calls
2. **Visual Alerts**: Real-time overlay shows detection status:
   - 🔍 **Yellow**: Analyzing audio chunks
   - ✅ **Green**: All four models flag the chunk as real
   - ⚠️ **Yellow**: Models disagree or the KNN result is below its alert threshold
   - ⚠️ **Red**: All four models flag fake and KNN confidence exceeds the overlay's 70% threshold
3. **Privacy**: All processing occurs locally, no data transmission

### Bundled Demo Audio
Use **Upload real sample** or **Upload fake sample** in the Tools section to analyze a selected audio file, or choose **Browse 300 attributed dataset clips** to select from the bundled dataset. All selected audio is evaluated by KNN, LightGBM, MobileNetV3, and the experimental MobileNetV5-style model. Reference labels come from the dataset; predictions may not match them, and this small sample set is not an accuracy test. Clips are from [Gary Stafford's Deepfake Audio Detection Dataset v4](https://huggingface.co/datasets/garystafford/deepfake-audio-detection), revision `fcf5344bb7f82b54b6b932291326d29750ef1e82`, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The dataset contains 1,866 clips total: 933 real and 933 synthetic. Its synthetic audio is attributed to these text-to-speech platforms:

| Synthetic audio source | Clips |
|---|---:|
| Amazon Polly | 209 |
| ElevenLabs | 173 |
| Hexgrad Kokoro | 68 |
| Hume AI | 116 |
| Luvvoice | 156 |
| Speechify | 211 |
| **Total synthetic clips** | **933** |

The existing `demo_fake.wav` comes from `fake/el_0001_part_001.flac`; `demo_real.wav` comes from `real/yt_0000_part_001.flac`. The 30 additional clips are 16 kHz mono 16-bit PCM WAV conversions from the three categories listed above; all source FLAC paths are recorded in `app/src/main/assets/samples/audio/dataset/MANIFEST.csv`. The samples exercise the app's audio-loading and inference flow; they are not a validation set, and detector predictions may not match their reference labels. Clip-level attribution and conversion details are in `app/src/main/assets/samples/audio/ATTRIBUTION.txt`.

### Example Detection Flow
```kotlin
// The service extracts model-specific features and reports separate scores and inference times.
val result = service.analyzeRawAudio(audioData, SAMPLE_RATE)
val fakeProbability = result.fakeConfidence
val lightgbmFakeProbability = result.lgbmFakeConfidence
val mobileNetV3FakeProbability = result.mobileNetFakeConfidence
val mobileNetV5FakeProbability = result.mobileNetV5FakeConfidence
```

## Core Components

### Audio Processing
- **AudioProcessor**: KNN/LightGBM feature extraction plus log-Mel/MFCC/LFCC features for MobileNet
- **RealTimeAudioDetectionService**: ONNX Runtime inference, call-audio monitoring, and provider fallback
- **PhoneStateReceiver**: Automatic call detection and service activation
- **MainActivity**: WAV/RIFF and M4A file analysis with per-model inference results

### User Interface
- **OverlayView**: Real-time visual alerts with confidence-based threat levels
- **MainActivity**: App configuration and monitoring controls

### Model Integration
- **ONNX Runtime**: Loads bundled KNN, LightGBM, MobileNetV3, and MobileNetV5-style ONNX models
- **Feature Pipeline**: 16 kHz call audio → one-second windows → per-model features → separate ONNX inference → fake/real probabilities and timings

## Project Structure

```
DeepFakeDetector_edge/
├── app/src/main/
│   ├── java/com/example/realtimeaudiodetect/
│   │   ├── AudioProcessor.kt           # Audio feature extraction
│   │   ├── RealTimeAudioDetectionService.kt # Edge deployment service
│   │   ├── OverlayView.kt             # Real-time visual alerts
│   │   └── PhoneStateReceiver.kt      # Call detection
│   ├── assets/models/                 # KNN, LightGBM, and MobileNet ONNX models
│   ├── assets/samples/audio/          # Attributed real/fake demo WAV files
│   └── AndroidManifest.xml           # Permissions and services
├── notebooks/                         # Training notebooks
└── README.md
```

## Development

### Model Requirements
- **KNN Model**: `app/src/main/assets/models/knn_modelv2.onnx`
- **KNN Input Shape**: `[batch, 25]`
- **KNN Input Features**: RMS, spectral centroid, bandwidth, rolloff, zero-crossing rate, followed by 20 MFCC means
- **LightGBM Model**: `app/src/main/assets/models/lgbmv2.onnx`, input shape `[batch, 26]`
- **LightGBM Input Features**: estimated chroma_stft followed by the 25 KNN features; this Android chroma is approximate
- **MobileNetV3 Model**: `app/src/main/assets/models/mobilenetv3_audio.onnx`
- **MobileNetV5-style Model**: `app/src/main/assets/models/mobilenetv5_audio.onnx`; this is a project-specific experimental Ghost-bottleneck model, not an official standardized MobileNetV5 architecture
- **MobileNet Input Features**: `[batch, 3, 64, time]` log-Mel, MFCC, and LFCC features; output class order is real then fake
- **Audio Format**: 16 kHz mono or stereo PCM; live call inference uses one-second windows
- **Outputs**: Predicted class, class probabilities, and per-model ONNX inference duration

### Retraining the KNN
The current model was retrained from the one-second feature pipeline in `notebooks/DeepFakeDetector/train_knn_1s.py`. Download the [Deepfake Audio Detection Dataset v4](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) into a directory with `real/` and `fake/` FLAC subdirectories, then run:

```bash
python notebooks/DeepFakeDetector/train_knn_1s.py --dataset-root path/to/dataset
```

The script selects KNN settings with source-grouped cross-validation, fits the selected RobustScaler/KNN pipeline on all one-second clips, and exports the ONNX model to the app assets by default. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution is recorded in `app/src/main/assets/samples/audio/ATTRIBUTION.txt`.

### Retraining the LightGBM
Install the dependencies from `requirements.txt`, then run:

```bash
python notebooks/DeepFakeDetector/train_lgbm_york.py
```

This tunes LightGBM with five-fold stratified cross-validation on the training portion of `audio_features_stringremoved.csv`, compares it to the bundled ONNX model on the fixed 30% random holdout, and replaces `lgbmv2.onnx` only if holdout accuracy improves. The source CSV lacks source IDs and audio-duration metadata, so its holdout score must not be interpreted as one-second or live-call accuracy.

The app includes 100 clips per source category (300 total). To refresh/recreate the attributed bundle from the pinned dataset revision, run:

```bash
python notebooks/DeepFakeDetector/prepare_bundled_audio_dataset.py
```

The script downloads from the pinned dataset revision, converts to mono 16 kHz PCM16 WAV, preserves existing samples, and records each original FLAC path in `dataset/MANIFEST.csv`.

To train and export the LightGBM model on the expanded sample set, run:

```bash
python notebooks/DeepFakeDetector/train_lgbm_bundled_audio.py
```

This ports the app's 26 input features, uses source-grouped cross-validation for parameter selection, and evaluates on a source-grouped holdout before refitting the final ONNX model on all 300 clips. The bundled files are 2.5–13 second segments, analyzed as full clips (up to six seconds), not one-second windows. The dataset is small and source-limited; holdout metrics are exploratory and should not be treated as real-world accuracy.

### Training the MobileNet Audio Models
Install the Python dependencies in `requirements.txt`, then train either model on the bundled 300 attributed WAV clips:

```bash
python notebooks/DeepFakeDetector/train_mobilenetv3_bundled_audio.py --architecture v3
python notebooks/DeepFakeDetector/train_mobilenetv3_bundled_audio.py --architecture v5
```

The script extracts one-second, 16 kHz log-Mel/MFCC/LFCC feature windows, uses a source-grouped validation holdout, reports exploratory metrics, refits on all available windows, exports ONNX, and checks the exported output against PyTorch. The V5 option is a custom experimental Ghost-bottleneck architecture and is not an official MobileNetV5 release. These 300 source clips are limited in diversity; the reported holdout metrics do not establish performance on unseen synthesis systems or live calls.

### Adding New Features
1. Keep feature extraction in `AudioProcessor.kt` aligned with each model’s training feature order and scaling.
2. Update model input/output handling in `RealTimeAudioDetectionService.kt` if the ONNX interface changes.
3. Update the displayed probabilities in `MainActivity.kt` and `OverlayView.kt` as needed.

### Key Dependencies
- ONNX Runtime Android
- Foreground service framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

All courses.

* [Part 1: Audio deepfake fraud detection system](https://thehyperplane.substack.com/p/audio-deepfake-fraud-detection-system?r=5l0jbv)  
* [Part 2: Training a model to detect deepfake audio](https://thehyperplane.substack.com/p/training-a-model-to-detect-deepfake?r=5l0jbv)
* [RealTimeAudioDetect on the Edge: Building Smarter Detection Where It Counts](https://thehyperplane.substack.com/p/training-a-model-to-detect-deepfake?r=5l0jbv)
* [Beyond the Cloud: Why the Future of AI Is on the Edge](https://thehyperplane.substack.com/p/beyond-the-cloud-why-the-future-of?r=5l0jbv)
