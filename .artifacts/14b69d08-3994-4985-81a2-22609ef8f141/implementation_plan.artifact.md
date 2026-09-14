# Implementation Plan - Refactor MainActivity.kt

Refactor `MainActivity.kt` to address potential memory and reliability issues:
1. **Memory Safety in Audio File Reading**: Replace full-file `inputStream.readBytes()` with a safe buffered/chunked PCM sample reader to prevent `OutOfMemoryError` on large audio files.
2. **Reliable Service State Detection**: Avoid using the deprecated and restricted `activityManager.getRunningServices()` API by tracking service status through binding state or explicit flags.
3. **Robust Error Handling**: Add proper error logging (`Timber.e`) and UI state reset in `try-catch` blocks during audio file analysis.

## User Review Required

> [!IMPORTANT]
> This refactor hardens `MainActivity.kt` against Out-Of-Memory exceptions when picking large audio files and replaces deprecated API calls for service status checks.

## Open Questions

- None.

## Proposed Changes

### [MainActivity Refactoring]

#### [MODIFY] [MainActivity.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/MainActivity.kt)
- Refactor `readAudioFile` to read PCM samples bufferedly/safely.
- Refactor `isServiceRunning` to rely on robust service binding and state tracking rather than deprecated `getRunningServices()`.
- Populate empty catch blocks with proper `Timber.e` logging and UI recovery.

## Verification Plan

### Automated Tests
- Run Gradle build (`gradle_build("app:assembleDebug")`) to verify clean compilation.
