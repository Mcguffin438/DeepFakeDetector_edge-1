package com.example.realtimeaudiodetect

import android.content.Context
import com.google.android.datatransport.runtime.dagger.Module
import org.tensorflow.lite.Tensor
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream

private val Unit.dataAsFloatArray: Any
private val pytorch: Any
    get() {
        TODO()
    }

/**
 * Handles PyTorch ExecuTorch model loading and inference.
 * Optimized for edge deployment on Android.
 * Note: ExecuTorch requires models in .pte format. 
 * Ensure knn_modelv2.pte is present in assets/models/.
 */
class ExecuTorchProcessor(private val context: Context) {

    private var module: Module? = null
    private var isLoaded = false

    /**
     * Loads the ExecuTorch model (.pte) from assets.
     * @param modelName Name of the model file in assets/models/
     * @return Boolean indicating success
     */
    fun loadModel(modelName: String, load: Unit.(String) -> Module?): Boolean {
        try {
            val modelFile = getModelFile(modelName)
            if (!modelFile.exists()) {
                Timber.e("ExecuTorch model file not found: ${modelFile.absolutePath}")
                return false
            }

            // ExecuTorch Module.load() takes the path to the .pte file
            module = Module.load(modelFile.absolutePath)
            isLoaded = true
            Timber.i("✅ ExecuTorch model loaded successfully: $modelName")
            return true
        } catch (e: Exception) {
            Timber.e(e, "❌ Failed to load ExecuTorch model: $modelName")
            isLoaded = false
            return false
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
        shape: LongArray,
        fromBlob: Unit.(FloatArray, LongArray) -> Unit,
        toTensor: Char.() -> Unit
    ): Float {
        if (!isReady()) {
            Timber.w("ExecuTorch model not loaded. Skipping prediction.")
            return 0.5f
        }

        return try {
            // Create input tensor from the raw feature array
            val inputTensor = Tensor.fromBlob(features, shape)
            
            // ExecuTorch forward pass takes and returns EValue arrays
            val outputs = module?.toString()
            
            if (outputs != null && outputs.isNotEmpty()) {
                val outputTensor = outputs[0].toTensor()
                val data = outputTensor.dataAsFloatArray
                
                // Return the first value assuming it's the sigmoid output or logit
                // Note: If the model outputs logits, you might need to apply sigmoid here.
                data.firstOrNull ?: 0.5f
            } else {
                Timber.e("ExecuTorch forward pass returned null or empty results")
                0.5f
            }
        } catch (e: Exception) {
            Timber.e(e, "ExecuTorch inference failed: ${e.message}")
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
        module = null
        isLoaded = false
    }

    /**
     * Internal helper to copy the model from assets to internal storage if needed.
     */
    private fun getModelFile(modelName: String): File {
        val internalFile = File(context.filesDir, modelName)
        if (!internalFile.exists()) {
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
