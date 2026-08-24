# PhoneStateReceiver.kt Technical Review

I have reviewed [PhoneStateReceiver.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/PhoneStateReceiver.kt) for reliability, API compatibility, and call flow logic.

## Critical Issues

### 1. API 26+ Dependency
- **Line 97 & 112**: The receiver uses `context.startForegroundService(intent)`. This method was introduced in API 26.
- **Risk**: Since the project's `minSdk` is 24, this will cause a **crash** on Android 7.0/7.1 devices (API 24/25) during any call event.
- **Fix**: Use `ContextCompat.startForegroundService(context, intent)` which handles the version check internally (falling back to `startService` on older APIs).

### 2. Deprecated `ACTION_NEW_OUTGOING_CALL`
- **Line 30**: The receiver still relies on `ACTION_NEW_OUTGOING_CALL`. While it still functions for now, it is officially deprecated in favor of more modern APIs like `CallRedirectionService`.
- **Note**: It also requires the `PROCESS_OUTGOING_CALLS` permission, which is not currently in the `MainActivity`'s `REQUIRED_PERMISSIONS` list.

### 3. Static State Fragility
- **Lines 17-19**: `lastState`, `isIncoming`, and `callerNumber` are stored in a `companion object`.
- **Risk**: If the Android system kills the app process while a call is active (e.g., due to memory pressure), these values will reset to defaults when the process is recreated for the "Call Ended" event. This could lead to incorrect state logic.

## Logic & Best Practices

### 1. Missing "Auto-Start" Preference Check
- The receiver starts the `DeepfakeDetectionService` immediately upon call detection.
- **Recommendation**: It should check the `auto_start` preference (saved in `MainActivity`) before launching the service to respect the user's settings.

### 2. Redundant Service Starts
- `ACTION_PHONE_STATE_CHANGED` can fire multiple times with the same state.
- **Recommendation**: The check `currentState != lastState` (Line 54) is good, but ensuring the service is only started once per call session is safer.

## Suggested Improvements

| Line | Description |
| :--- | :--- |
| 52 | Refactor the `when` block to use `currentState` as the subject for better readability. |
| 58 | Use named arguments for better clarity: `prepareForCall(..., incoming = true)`. |
| 40 | `EXTRA_INCOMING_NUMBER` requires the `READ_CALL_LOG` permission on Android 9+. Ensure this is clearly communicated to the user. |

---

Would you like me to include the fixes for `PhoneStateReceiver.kt` into the project-wide refactoring plan?
