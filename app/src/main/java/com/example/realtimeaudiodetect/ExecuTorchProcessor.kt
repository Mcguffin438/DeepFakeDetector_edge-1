package com.example.realtimeaudiodetect

import android.content.Context
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.IValue
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream

/**
 * Handles PyTorch model loading and inference (PyTorch Mobile / ExecuTorch runtime).
 * Optimized for edge deployment on Android.
 * Note: Requires model file (e.g. knn_modelv2.pte or .pt) in assets/models/.
 */
class ExecuTorchProcessor(private val context: Context) {

    private var module: Module? = null
    private var isLoaded = false

    /**
     * Loads the model from assets.
     * @param modelName Name of the model file in assets/models/
     * @return Boolean indicating success
     */
    @Synchronized
    fun loadModel(modelName: String): Boolean {
        return try {
            val modelFile = getModelFile(modelName)
            if (!modelFile.exists() || modelFile.length() == 0L) {
                Timber.e("Model file not found or empty: ${modelFile.absolutePath}")
                return false
            }

            // Clear previous module reference
            module?.destroy()
            module = null
            isLoaded = false

            // Module.load() takes the absolute path to the model file
            module = Module.load(modelFile.absolutePath)
            isLoaded = true
            Timber.i("✅ Model loaded successfully: $modelName")
            true
        } catch (e: Exception) {
            Timber.e(e, "❌ Failed to load model: $modelName")
            module = null
            isLoaded = false
            false
        }
    }

    /**
     * Runs inference on the provided audio features.
     * @param features Flattened float array of features
     * @param shape Shape of the input tensor (e.g., longArrayOf(1, 3, 64, time))
     * @return Prediction probability (0.0 to 1.0) where > 0.5 usually indicates 'Fake'
     */
    fun predict(
        features: FloatArray,
        shape: LongArray
    ): Float {
        if (!isReady()) {
            Timber.w("Model not loaded. Skipping prediction.")
            return 0.5f
        }

        return try {
            // Create input tensor from the raw feature array
            val inputTensor = Tensor.fromBlob(features, shape)
            
            // Forward pass takes IValue and returns IValue
            val outputIValue = module?.forward(IValue.from(inputTensor))
            val outputTensor = outputIValue?.toTensor()
            val data = outputTensor?.dataAsFloatArray
            
            if (data != null && data.isNotEmpty()) {
                data[0]
            } else {
                Timber.e("Model forward pass returned null or empty results")
                0.5f
            }
        } catch (e: Exception) {
            Timber.e(e, "Inference failed: ${e.message}")
            0.5f
        }
    }

    /**
     * Checks if the model is initialized and ready.
     */
    fun isReady(): Boolean = isLoaded && module != null

    /**
     * Clears the module reference.
     */
    fun close() {
        module?.destroy()
        module = null
        isLoaded = false
    }

    /**
     * Internal helper to model copy from assets to internal storage if needed.
     */
    private fun getModelFile(modelName: String): File {
        val internalFile = File(context.filesDir, modelName)
        if (!internalFile.exists() || internalFile.length() == 0L) {
            try {
                context.assets.open("models/$modelName").use { input ->
                    FileOutputStream(internalFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Timber.i("✅ Model copied from assets to internal storage: ${internalFile.absolutePath}")
            } catch (e: Exception) {
                Timber.e(e, "❌ Failed to copy model from assets: $modelName")
            }
        }
        return internalFile
    }
}
