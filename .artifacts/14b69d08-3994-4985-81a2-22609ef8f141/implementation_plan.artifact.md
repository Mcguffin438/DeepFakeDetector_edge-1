# Implementation Plan - Refactor Application Code for Issues & Performance

Refactor core Android application source files (`AudioProcessor.kt`, `RealTimeAudioDetectionService.kt`, `MainActivity.kt`, `OverlayView.kt`, `PhoneStateReceiver.kt`) to fix potential performance bottlenecks, optimize memory allocation in audio processing loops, enhance coroutine resource management, and clean up warnings.

## User Review Required

> [!IMPORTANT]
> This refactoring optimizes audio processing buffer reuse in `AudioProcessor.kt`, strengthens ONNX/ML resource management in `RealTimeAudioDetectionService.kt`, and improves UI animation lifecycle handling in `OverlayView.kt` and `MainActivity.kt`.

## Open Questions

- None. The changes focus entirely on robustness, memory efficiency, and clean code practices.

## Proposed Changes

### [Core Audio & ML Components]

#### [MODIFY] [AudioProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/AudioProcessor.kt)
- Optimize `computeDFT` and STFT loops to minimize object allocations during real-time streaming feature extraction.
- Improve error handling and array slicing boundaries.

#### [MODIFY] [RealTimeAudioDetectionService.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/RealTimeAudioDetectionService.kt)
- Optimize audio buffer streaming loop in `processAudioStream` to prevent GC pressure.
- Ensure robust cleanup of ONNX session and environment resources.

#### [MODIFY] [OverlayView.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/OverlayView.kt)
- Clean up animator cancellation and prevent memory leaks on detach.

#### [MODIFY] [MainActivity.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/MainActivity.kt)
- Ensure safe service binding unregistration and lifecycle safety in coroutine scopes.

## Verification Plan

### Automated Tests
- Run Gradle build (`gradle_build("app:assembleDebug")`) to verify clean compilation across all refactored files.
