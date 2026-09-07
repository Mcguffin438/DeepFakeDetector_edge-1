# Convert PyTorch to Scikit-Learn ONNX (KNN Model)

This plan outlines the steps to migrate the deepfake detection model from PyTorch to the `knn_modelv2.onnx` (Scikit-Learn) model found in assets.

## User Review Required

> [!IMPORTANT]
> The model `knn_modelv2.onnx` will be used as the primary inference engine.

> [!WARNING]
> Scikit-Learn models (like KNN) usually expect a fixed-size 2D input `[1, n_features]`. The current feature extraction produces a variable-length sequence `[3, 64, T]`. I will implement a global average pooling or flattening mechanism to ensure the input matches the model's expected size.

## Proposed Changes

### [Deepfake Detection Service]

#### [MODIFY] [DeepfakeDetectionService.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/DeepfakeDetectionService.kt)
- Fully remove PyTorch imports and commented-out code.
- Standardize ONNX Runtime usage (OrtSession, OrtEnvironment).
- Update `getModelFile` to use `knn_modelv2.onnx`.
- Refine `processAudioChunk` to handle the output of a Scikit-Learn model (which may be a label index or a probability array).

### [Audio Processor]

#### [MODIFY] [AudioProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/AudioProcessor.kt)
- Add a pooling function (e.g., Global Average Pooling) to convert `[3, 64, T]` features into a fixed-size vector `[3 * 64]` if the model expects it.
- Ensure the feature extraction matches the training pipeline used for the KNN model.

### [Build Configuration]

#### [MODIFY] [build.gradle.kts](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/build.gradle.kts)
- Remove PyTorch dependencies (`libs.bundles.pytorch`).
- Keep `onnxruntime-android`.

#### [MODIFY] [libs.versions.toml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/gradle/libs.versions.toml)
- Remove PyTorch entries.

## Verification Plan

### Automated Tests
- Build the project to ensure no PyTorch references remain.
- Verify `AudioProcessor` output shape matches the expected input of `knn_modelv2.onnx`.

### Manual Verification
- Deploy to a device.
- Check logs for successful ONNX session creation with `knn_modelv2.onnx`.
- Test inference with a sample audio file.
