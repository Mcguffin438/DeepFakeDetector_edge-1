# Implementation Plan - Convert KNN Audio Detector to PyTorch TorchScript (.pt)

Convert the PyTorch KNN Audio Deepfake detector from `notebooks/knn_audio_df_detector.py` into a compiled PyTorch TorchScript `.pt` model file (`knn_audio_detector.pt`) suitable for loading and running in Android via PyTorch Mobile.

## User Review Required

> [!IMPORTANT]
> This creates an end-to-end wrapper module (`AudioDeepfakeDetectorEndToEnd`) combining feature extraction and the fitted k-NN classifier, compiles it via `torch.jit.script` or `torch.jit.trace`, and saves it as `app/src/main/assets/models/knn_audio_detector.pt`.

## Open Questions

- None. TorchScript will enable direct execution of the model inside Android Java/Kotlin using PyTorch Mobile (`LiteModuleLoader`).

## Proposed Changes

### [KNN Audio Detector PyTorch Export]

#### [MODIFY] [knn_audio_df_detector.py](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/notebooks/knn_audio_df_detector.py)
- Add a unified `AudioDeepfakeDetectorEndToEnd` module that bundles `AudioFeatureExtractor` and `PyTorchKNNAudioDeepfakeClassifier`.
- Add `export_to_torchscript()` function to trace/script the fitted model and save it to `app/src/main/assets/models/knn_audio_detector.pt`.
- Update `__main__` block to execute training fitting, inference testing, and `.pt` export.

## Verification Plan

### Automated Tests
- Run `python notebooks/knn_audio_df_detector.py` to verify model building, fitting, TorchScript compilation, and `.pt` file generation.
