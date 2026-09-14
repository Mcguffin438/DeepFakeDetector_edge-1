# Implementation Plan - Create Unit Test Suite

Create a comprehensive unit test suite (`AudioProcessorTest.kt`) for the core audio processing logic (`AudioProcessor.kt`) and a Python test script (`test_knn_detector.py`) for the Python ML pipeline to ensure all code works reliably.

## User Review Required

> [!IMPORTANT]
> This will add:
> 1. An Android local unit test (`AudioProcessorTest.kt`) under `app/src/test/java/com/example/realtimeaudiodetect/` to test feature extraction and multi-channel shape generation.
> 2. A Python test script (`notebooks/test_knn_detector.py`) to verify embedding extraction, KNN model fitting, and inference.

## Open Questions

- None. The tests use JUnit4 for Android and standard Python assertions for the ML pipeline.

## Proposed Changes

### [Test Suite Implementation]

#### [NEW] [AudioProcessorTest.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/test/java/com/example/realtimeaudiodetect/AudioProcessorTest.kt)
- Test `AudioProcessor` feature extraction, sample rate handling, and feature pooling.

#### [NEW] [test_knn_detector.py](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/notebooks/test_knn_detector.py)
- Test PyTorch feature extractor and KNN deepfake classifier.

## Verification Plan

### Automated Tests
- Run Gradle unit tests (`gradle_build("app:testDebugUnitTest")`) to verify Android unit tests.
- Run Python test script (`python notebooks/test_knn_detector.py`).
