# Implementation Plan - Integrate PyTorch ExecuTorch

This plan outlines the integration of **PyTorch ExecuTorch** into the Android application. ExecuTorch is the next-generation on-device inference engine from PyTorch, optimized for edge devices.

> [!IMPORTANT]
> **Model Format Compatibility**: ExecuTorch uses the `.pte` model format. The current `knn_modelv2.onnx` file is an ONNX model and cannot be run directly by the ExecuTorch runtime without conversion to the ExecuTorch-compatible ATen dialect and exported as a `.pte` file. This plan sets up the infrastructure to support ExecuTorch inference once a `.pte` model is provided.

## User Review Required

> [!WARNING]
> **ExecuTorch Stability**: ExecuTorch is currently in a beta/early-access phase. The API and dependency coordinates may shift. We are using the most recent stable-release coordinates (`0.4.0`/`0.5.0` series).

## Proposed Changes

### [Dependencies]
#### [MODIFY] [libs.versions.toml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/gradle/libs.versions.toml)
- Add ExecuTorch version and library definitions.
- Coordinate: `org.pytorch:executorch:0.4.0` (Core runtime).

#### [MODIFY] [build.gradle.kts](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/build.gradle.kts)
- Add ExecuTorch dependency to the `dependencies` block.

### [Inference Layer]
#### [NEW] [ExecuTorchProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/ExecuTorchProcessor.kt)
- Create a dedicated processor for ExecuTorch inference.
- Methods for loading the `.pte` model from assets and executing forward passes.

### [Service Layer]
#### [MODIFY] [RealTimeAudioDetectionService.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/realtimeaudiodetect/RealTimeAudioDetectionService.kt)
- Integrate `ExecuTorchProcessor` into the service lifecycle.
- Add support for loading `knn_modelv2.pte` (as a future-proof path for the ONNX model's ExecuTorch equivalent).
- Provide a fallback/toggle mechanism between ONNX Runtime and ExecuTorch.

## Verification Plan

### Automated Tests
- Run Gradle sync to verify dependency resolution.
- Run `app:assembleDebug` to ensure compilation success.

### Manual Verification
- Deploy to device.
- Verify through logs that the service attempts to initialize ExecuTorch if a `.pte` model is detected in the assets.
