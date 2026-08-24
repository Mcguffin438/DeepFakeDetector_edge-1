# Fix PyTorch Audio Dependency Resolution Issue

The project is failing to sync because it tries to resolve `org.pytorch:pytorch_android_torchaudio:2.1.0`, which does not exist as an official Gradle artifact. PyTorch Mobile does not provide a separate `torchaudio` library for Android; instead, audio features are expected to be handled within the core PyTorch runtime (TorchScript) or via custom preprocessing.

Research shows that the project already implements its own audio feature extraction in `AudioProcessor.kt` and uses the core `pytorch_android` library for inference in `DeepfakeDetectionService.kt`.

## Proposed Changes

### Build Configuration

#### [MODIFY] [libs.versions.toml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/gradle/libs.versions.toml)
- Remove `pytorch-android-torchaudio` from the `[libraries]` section.
- Remove `pytorch-android-torchaudio` from the `pytorch` bundle in the `[bundles]` section.

## Verification Plan

### Automated Tests
- Run Gradle sync to verify that the project now resolves all dependencies correctly.
- Run `./gradlew :app:assembleDebug` to ensure the project builds successfully.

### Manual Verification
- Verify that the app still works as expected (since `torchaudio` was not actually being used by the code).
