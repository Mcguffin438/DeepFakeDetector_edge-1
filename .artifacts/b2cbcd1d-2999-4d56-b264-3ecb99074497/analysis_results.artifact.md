# AudioProcessor.kt Technical Review

I have reviewed [AudioProcessor.kt](file:///C:/Users/dusty/Desktop/DeepFakeDetector_edge/app/src/main/java/com/example/deepfakeguard/AudioProcessor.kt) for performance, correctness, and idiomatic Kotlin usage.

## Critical Issues

### 1. Severe Performance Bottleneck ($O(N^2)$ DFT)
The `computeDFT` function (Line 50) implements a naive Discrete Fourier Transform.
- **Complexity**: For $N=780$, it performs $\approx 600,000$ multiplications/additions per frame.
- **Frame Count**: A 6-second clip at 16kHz results in $\approx 492$ frames.
- **Total Ops**: $\approx 300,000,000$ operations per analysis. On many mobile devices, this will cause frame drops or trigger the 30s timeout in the Service.
- **Trig Recomputation**: `cosTable` and `sinTable` are re-initialized for *every* frame (Line 54).

### 2. High Memory Pressure & GC Thrashing
- **Object Allocation**: `computeDFT` creates a new `Complex` object (Line 51) for every frequency bin.
- **Scale**: $\approx 380,000$ objects are created and discarded during a single 6-second analysis. This will lead to frequent Garbage Collection pauses and high battery drain.

### 3. Fragile Stereo/Mono Heuristic
The mono conversion logic (Line 104) assumes stereo if the array size is even:
```kotlin
val monoAudio = if (audioData.size % 2 == 0) { ... }
```
- **Risk**: A mono recording with an even number of samples will be incorrectly averaged with adjacent samples, corrupting the high frequencies (aliasing/filtering effect).

### 4. Mel Filter Bank Correctness
- **Normalization**: The `createMelFilterBank` (Line 383) creates triangular filters but doesn't apply area-normalization (Slaney normalization).
- **Impact**: If your Python model used `librosa` or `torchaudio` with default settings, it likely expects normalized filters. Unnormalized filters will cause the model to see different energy distributions than it was trained on.

## Code Quality Issues (Static Analysis)

| Line | Severity | Description |
| :--- | :--- | :--- |
| 14 | Warning | `MultiChannelFeaturesResult` uses `FloatArray` in a `data class` without overriding `equals/hashCode`. This leads to identity-based equality instead of content-based. |
| 159 | Warning | Hardcoded `0 * NUM_FEATURE_BINS` is redundant and can be simplified. |
| 197 | Warning | `separateStereoChannels` is unused and should be removed. |
| 51 | Warning | Local variable `N` violates Kotlin naming conventions (should be `n`). |
| 312 | Warning | Redundant `Companion` qualifier in method call. |

## Suggested Improvements

1.  **Optimize DFT**:
    - Pre-compute `cosTable` and `sinTable` once as class members.
    - Use primitive `DoubleArray` (interleaved real/imag) instead of `Complex` objects.
2.  **Explicit Channel Handling**: Pass `numChannels` to `generateMultiChannelFeatures` instead of guessing.
3.  **Vectorize Loops**: Use standard `for` loops instead of `forEachIndexed` in inner processing loops to avoid iterator allocations.
4.  **Align with Training**: Verify if the Python training used `top_db` clipping or a fixed floor. The current implementation (Line 354) uses a global max, which is correct for `librosa.power_to_db`.

---

> [!TIP]
> **Radix-2 FFT**: If you can change the `FFT_SIZE` from 780 to 1024 (by zero-padding), we can use a standard FFT algorithm which would be **~100x faster**.

Would you like me to implement these fixes based on the plan I proposed earlier?
