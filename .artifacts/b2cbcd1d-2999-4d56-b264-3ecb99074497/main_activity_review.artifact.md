# MainActivity.kt Technical Review

I have reviewed [MainActivity.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/MainActivity.kt) for potential issues, API compatibility, and best practices.

## Critical Issues

### 1. API Compatibility (Foreground Service)
- **Line 47**: `Manifest.permission.FOREGROUND_SERVICE_PHONE_CALL` requires API 34. The project's `minSdk` is 24. This will cause crashes on older devices if not handled conditionally.
- **Line 333**: `startForegroundService` requires API 26+. While the app's `targetSdk` is 35, calling this on devices below API 26 without a check will fail.

### 2. Deprecated Service Check
- **Line 299**: `getRunningServices(Integer.MAX_VALUE)` is deprecated and unreliable on modern Android versions for security reasons.
- **Recommendation**: Use a `StateFlow` or a shared `AtomicBoolean` within the `DeepfakeDetectionService` to track its running state, or rely on the `ServiceConnection` state.

### 3. Inefficient "Busy Wait" for Service/Model
- **Lines 398 & 463**: The code uses manual `delay(100)` loops with an `attempts` counter to wait for the service to bind or the model to load.
- **Impact**: This blocks the execution flow and is prone to race conditions or unnecessary delays.
- **Recommendation**: Use `CompletableDeferred` or a callback-based approach to resume execution exactly when the service is ready.

### 4. Fragile Audio File Reading
- **Line 650**: `readAudioFile` manually skips 44 bytes for WAV headers.
- **Risk**: Not all WAV files have exactly a 44-byte header (e.g., those with metadata chunks). This could lead to "pops" or misaligned audio samples.

## UI & Idiomatic Kotlin Improvements

| Line | Description |
| :--- | :--- |
| 59 | Use `SwitchCompat` or `MaterialSwitch` instead of the legacy `Switch`. |
| 242+ | Multiple hardcoded strings should be moved to `strings.xml` for localization support. |
| 169 | Use KTX `sharedPreferences.edit { ... }` for cleaner syntax. |
| 744 | Notification settings API calls need `Build.VERSION_CODES.O` checks. |

## Suggested Improvements

1.  **Permission Guards**: Wrap API-specific permissions in `if (Build.VERSION.SDK_INT >= ...)` blocks.
2.  **Service Lifecycle**: Implement a more robust service tracking mechanism using `LiveData` or `Flow`.
3.  **WAV Parsing**: Use a basic WAV header parser or a library to correctly identify the `data` chunk offset.
4.  **Resource Handling**: Move all UI text to `strings.xml`.

---

> [!IMPORTANT]
> **Foreground Service Type**: Ensure that the `AndroidManifest.xml` correctly declares `android:foregroundServiceType="phoneCall"` to match the code's request.

Would you like me to prepare a plan to fix these issues in `MainActivity.kt`?
