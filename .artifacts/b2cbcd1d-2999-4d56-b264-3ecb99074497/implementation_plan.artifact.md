# Implementation Plan - MainActivity & Service Refactoring

This plan addresses technical debt in `MainActivity.kt` and `DeepfakeDetectionService.kt` to improve robustness, API compatibility, and performance.

## User Review Required

> [!IMPORTANT]
> **API 34 (Android 14) Requirements**: To monitor calls in the background, we must use the `phoneCall` foreground service type. This requires the user to grant the `FOREGROUND_SERVICE_PHONE_CALL` permission, which is a "special" permission on Android 14. I will implement the logic to handle this, but it may require an extra step in the UI for users on latest Android versions.

## Proposed Changes

### [Component] Audio & ML Infrastructure

#### [MODIFY] [DeepfakeDetectionService.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/DeepfakeDetectionService.kt)
- Add a companion `isRunning` and `modelLoadingState` using `StateFlow` so the Activity can observe them without "busy-waiting" or using deprecated APIs.

#### [MODIFY] [AudioProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/AudioProcessor.kt)
- (Small addition) Add a utility method to robustly find the `data` chunk in a WAV byte array instead of assuming a 44-byte header.

---

### [Component] UI & Lifecycle

#### [MODIFY] [MainActivity.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/MainActivity.kt)
- **Modernize Permissions**:
    - Conditional request for `FOREGROUND_SERVICE_PHONE_CALL` (API 34+).
    - Add `POST_NOTIFICATIONS` (API 33+) to ensures the foreground notification is visible.
- **Reactive UI**:
    - Observe `DeepfakeDetectionService.isRunning` flow to update the "Start/Stop" button state.
    - Replace `while(attempts < 100)` loops with `modelLoadingState.first { it == Loaded }`.
- **KTX Enhancements**:
    - Use `SharedPreferences.edit { ... }`.
    - Use `String.toUri()` and `ViewBinding` (if already present, otherwise stick to `findViewById` but clean up).
- **Hardcoded Strings**: Move all status messages to `res/values/strings.xml`.

#### [MODIFY] [AndroidManifest.xml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/AndroidManifest.xml)
- Add missing permission declarations for API 33/34.

## Verification Plan

### Automated Tests
- Create `WavParserTest.kt` to verify the new WAV header parsing logic with various valid and invalid headers.

### Manual Verification
- **Permission Flow**: Verify that on Android 14, the app correctly asks for the phone call foreground permission.
- **Service Lifecycle**: Start and stop the service; verify the UI updates instantly via the `StateFlow`.
- **Audio Analysis**: Analyze a local WAV file; verify it processes correctly.
