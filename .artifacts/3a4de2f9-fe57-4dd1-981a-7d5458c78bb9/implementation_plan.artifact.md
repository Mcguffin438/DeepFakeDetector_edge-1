# Convert PyTorch to Scikit-Learn ONNX

This plan outlines the steps to migrate the deepfake detection model from PyTorch to an ONNX-exported Scikit-Learn model.

## User Review Required

> [!IMPORTANT]
> This change assumes the new model will be named `deepfake_detector.onnx` and placed in `app/src/main/assets/models/`.

> [!WARNING]
> Scikit-Learn models via ONNX often require a specific input shape (usually 1D or 2D). I will adapt the code to handle a flattened feature vector, but the exact input name (e.g., `"input"`, `"float_input"`) and output format (label vs. probability) may need adjustment once the model is available.

## Proposed Changes

### [Deepfake Detection Service]

#### [MODIFY] [DeepfakeDetectionService.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/DeepfakeDetectionService.kt)
- Replace PyTorch `Module` and `Tensor` with ONNX Runtime `OrtSession` and `OnnxTensor`.
- Update `loadDeepfakeModel` to initialize the `OrtEnvironment` and create an `OrtSession`.
- Update inference logic in `processAudioChunk` and `analyzeRawAudio` to use ONNX Runtime.
- Update `generateCallSummary` and UI updates if output format changes.

### [Audio Processor]

#### [MODIFY] [AudioProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/AudioProcessor.kt)
- Ensure feature extraction remains compatible with the new model's requirements.
- Add a helper to flatten or pool features if the Scikit-Learn model expects a fixed-size vector.

### [Build Configuration]

#### [MODIFY] [build.gradle.kts](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/build.gradle.kts)
- Remove PyTorch dependencies.
- Ensure ONNX Runtime dependencies are active.

#### [MODIFY] [libs.versions.toml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/gradle/libs.versions.toml)
- Cleanup PyTorch versions and libraries.

## Verification Plan

### Automated Tests
- Build the project to ensure all PyTorch references are removed and ONNX Runtime is correctly integrated.
- (If model is provided) Run unit tests for `AudioProcessor` to verify feature shapes.

### Manual Verification
- Deploy the app to a device/emulator.
- Check logs for "✅ ONNX model loaded successfully!".
- Verify that inference triggers when a call is detected (requires the `.onnx` file).
