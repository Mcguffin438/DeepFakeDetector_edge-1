# Implementation Plan - Add PyTorch Vulkan Dependencies to Android Application

Add PyTorch Mobile Vulkan GPU backend dependency (`org.pytorch:pytorch_android_vulkan:2.1.0`) and TorchVision (`org.pytorch:pytorch_android_torchvision:2.1.0`) to `gradle/libs.versions.toml` and `app/build.gradle.kts` to enable hardware-accelerated Vulkan inference for PyTorch `.pt` models in the Android app.

## User Review Required

> [!IMPORTANT]
> This will integrate PyTorch Mobile Vulkan GPU acceleration (`org.pytorch:pytorch_android_vulkan:2.1.0`) into the Android app dependencies, allowing models like `deepfake_detector.pt` to run on GPU via Vulkan.

## Open Questions

- None. PyTorch Android Vulkan is the standard backend for mobile GPU inference.

## Proposed Changes

### [Android App Dependencies]

#### [MODIFY] [libs.versions.toml](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/gradle/libs.versions.toml)
- Add `pytorch = "2.1.0"` version entry.
- Add `pytorch-android-vulkan` and `pytorch-android-torchvision` library definitions.

#### [MODIFY] [build.gradle.kts](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/build.gradle.kts)
- Add `pytorch-android-vulkan` and `pytorch-android-torchvision` dependencies.

## Verification Plan

### Automated Tests
- Run Gradle build (`gradle_build("app:assembleDebug")`) to verify dependency resolution and compilation success.
