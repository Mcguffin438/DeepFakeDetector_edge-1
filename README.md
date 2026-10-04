# RealTimeAudioDetect - Real-time Deepfake Audio Detection

A real-time Android deepfake-audio detection app that runs a KNN classifier locally with ONNX Runtime. The app extracts 25 audio features and feeds them to the bundled `knn_modelv2.onnx` model, which includes its RobustScaler.

![KNN deepfake audio detection workflow](images/deepfake_edge_workflow.svg)

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
- 25 features per audio chunk: RMS, spectral centroid, bandwidth, rolloff, zero-crossing rate, and 20 MFCC means
- Audio is processed at 16 kHz; call monitoring uses chunks from the built-in microphone and is intended for speakerphone use
- Features are passed to the KNN model in the training feature order

**Model Training & Deployment:**
- ONNX input shape: `[batch, 25]`
- KNN classifier with embedded RobustScaler, 3 neighbors, cosine distance, and uniform weights
- ONNX model outputs the class label and class probabilities
- ONNX Runtime uses available execution providers with fallback; CPU inference is supported

**Edge Integration:**
- Android foreground service with automatic call detection
- Real-time visual overlay and fake/real probability scores
- 100% local processing - no network connectivity required
- Privacy-first design with in-memory audio processing

## Features

- **Audio Features**: 25-dimensional feature vector for the bundled KNN
- **Edge Deployment**: ONNX Runtime with CPU fallback
- **Real-Time Alerts**: Visual overlay with confidence-based threat detection
- **Auto-Activation**: Foreground service monitors calls automatically
- **Privacy-First**: 100% local processing, no data transmission
- **Model**: `app/src/main/assets/models/knn_modelv2.onnx`

## Key Technical Achievements

**Model and evaluation:**
- Audio processing: 25 KNN input features per chunk
- Model deployment: ONNX with the scaler embedded in the model
- The 96.03% holdout score is from one fixed dataset split, not an on-device or live-call benchmark

**Architecture Highlights:**
- KNN model input: `[batch, 25]`
- KNN configuration: 3 neighbors, cosine distance, uniform weights
- Alert system: fake/real probability scores with experimental warnings

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
   - ✅ **Green**: Authentic speech detected
   - ⚠️ **Red**: High-confidence deepfake (>90%)
3. **Privacy**: All processing occurs locally, no data transmission

### Bundled Demo Audio
Use **Test real sample** or **Test fake sample** in the Tools section to run the included speech clips through the KNN model. The reference labels identify the dataset classes; predictions may not match them and are not an accuracy test. These two clips are small examples from [Gary Stafford's Deepfake Audio Detection Dataset v4](https://huggingface.co/datasets/garystafford/deepfake-audio-detection), not the full dataset. The source dataset contains 1,866 clips total: 933 real and 933 synthetic. Its synthetic audio is attributed to these text-to-speech platforms:

| Synthetic audio source | Clips |
|---|---:|
| Amazon Polly | 209 |
| ElevenLabs | 173 |
| Hexgrad Kokoro | 68 |
| Hume AI | 116 |
| Luvvoice | 156 |
| Speechify | 211 |
| **Total synthetic clips** | **933** |

The app bundles only one synthetic example: `demo_fake.wav`, sourced from `fake/el_0001_part_001.flac` (ElevenLabs, based on the dataset filename prefix). It is distributed as 16 kHz, mono, 16-bit PCM WAV and is about 3.9 seconds long. The bundled real example, `demo_real.wav`, comes from `real/yt_0000_part_001.flac`; it was downmixed from stereo and resampled from 44.1 kHz to 16 kHz mono PCM WAV. The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See `app/src/main/assets/samples/audio/ATTRIBUTION.txt` for clip-level attribution. The samples exercise the app's audio-loading and inference flow; they are not a validation set, and detector predictions may not match the reference labels.

### Example Detection Flow
```kotlin
// The service extracts the model's 25 features, creates a [1, 25] ONNX tensor,
// and reads class probabilities through ONNX Runtime.
val result = service.analyzeRawAudio(audioData, SAMPLE_RATE)
val fakeProbability = result.fakeConfidence
```

## Core Components

### Audio Processing
- **AudioProcessor**: KNN feature extraction (five spectral/time-domain statistics plus 20 MFCC means)
- **RealTimeAudioDetectionService**: ONNX Runtime inference, call-audio monitoring, and provider fallback
- **PhoneStateReceiver**: Automatic call detection and service activation

### User Interface
- **OverlayView**: Real-time visual alerts with confidence-based threat levels
- **MainActivity**: App configuration and monitoring controls

### Model Integration
- **ONNX Runtime**: Loads the bundled `knn_modelv2.onnx`, whose graph includes RobustScaler and KNN
- **Feature Pipeline**: 16 kHz call audio → 25 features → ONNX KNN inference → fake/real probabilities

## Project Structure

```
DeepFakeDetector_edge/
├── app/src/main/
│   ├── java/com/example/realtimeaudiodetect/
│   │   ├── AudioProcessor.kt           # KNN audio feature extraction
│   │   ├── RealTimeAudioDetectionService.kt # Edge deployment service
│   │   ├── OverlayView.kt             # Real-time visual alerts
│   │   └── PhoneStateReceiver.kt      # Call detection
│   ├── assets/models/                 # ONNX models
│   ├── assets/samples/audio/          # Attributed real/fake demo WAV files
│   └── AndroidManifest.xml           # Permissions and services
├── notebooks/                         # Training notebooks
└── README.md
```

## Development

### Model Requirements
- **Model**: `app/src/main/assets/models/knn_modelv2.onnx`
- **Input Shape**: `[batch, 25]`
- **Input Features**: RMS, spectral centroid, bandwidth, rolloff, zero-crossing rate, followed by 20 MFCC means
- **Audio Format**: 16 kHz mono or stereo PCM; inference uses a six-second feature window
- **Outputs**: Predicted class and class probabilities

### Adding New Features
1. Keep feature extraction in `AudioProcessor.kt` aligned with the model’s training feature order and scaling.
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
