# Comprehensive Code Review & Improvements Plan

Following a complete review of all source files in the project (`MainActivity.kt`, `RealTimeAudioDetectionService.kt`, `AudioProcessor.kt`, `OverlayView.kt`, `PhoneStateReceiver.kt`, `AndroidManifest.xml`, and build configurations), several architectural, robustness, and performance improvements have been identified.

## User Review Required

> [!IMPORTANT]
> **ONNX Environment Management**: In ONNX Runtime Java, `OrtEnvironment.getEnvironment()` returns a process-wide singleton environment. Calling `.close()` on it in `RealTimeAudioDetectionService.onDestroy()` can close the shared environment prematurely if other components or service restarts occur. We recommend omitting `ortEnv?.close()` or properly managing it.

> [!NOTE]
> **Audio Recording Sources (`VOICE_DOWNLINK` / `VOICE_CALL`)**: On Android 9 (API 28) and above, recording audio from `VOICE_DOWNLINK` or `VOICE_CALL` is restricted by Android system security to system/dialer apps. While the service iterates through fallback sources (`VOICE_COMMUNICATION`, `MIC`), attempting restricted sources will generate security/IllegalArgument warnings in Logcat.

---

## Proposed Changes

### [Service & Lifecycle] RealTimeAudioDetectionService.kt
- **ONNX Environment Cleanup**: Prevent closing the singleton `OrtEnvironment` in `onDestroy()` to avoid process-level crashes on service restart.
- **Audio Recording Fallback Logging**: Optimize audio source fallback sequence to prioritize `VOICE_COMMUNICATION` and `MIC` on modern Android versions where `VOICE_DOWNLINK` / `VOICE_CALL` are restricted.

### [Audio Processing] AudioProcessor.kt
- **Performance Optimization**: Add caching for pre-computed trig tables in `computeDFT` or document performance characteristics for audio frame feature generation.

### [UI & Overlay] OverlayView.kt & MainActivity.kt
- **Accessibility & Lint Warnings**: Address minor lint warnings (e.g., string literals in `setText`, API version checks).

---

## Verification Plan

### Automated Tests
- Run Gradle build (`app:assembleDebug`) to ensure all code compiles cleanly.

### Manual Verification
- Deploy app to a test device/emulator.
- Verify service start/stop, audio feature extraction, inference execution, and UI overlay rendering.
