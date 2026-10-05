/**
 * Foreground service for real-time deepfake detection during calls
 * Captures audio, runs ONNX inference, displays overlay results
 */
package com.example.realtimeaudiodetect

import android.Manifest
import android.app.*
import android.content.Intent
import android.media.*
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.view.WindowManager
import androidx.annotation.RequiresPermission
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import timber.log.Timber
import java.io.File
import java.io.IOException
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import androidx.core.app.ActivityCompat
import java.nio.FloatBuffer
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean

class RealTimeAudioDetectionService : Service() {
    
    companion object {
        const val ACTION_PREPARE = "ACTION_PREPARE"
        const val ACTION_START_DETECTION = "ACTION_START_DETECTION"
        const val ACTION_STOP_DETECTION = "ACTION_STOP_DETECTION"
        const val EXTRA_PHONE_NUMBER = "EXTRA_PHONE_NUMBER"
        const val EXTRA_IS_INCOMING = "EXTRA_IS_INCOMING"
        
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "deepfake_detection_channel"
        private const val DEEPFAKE_ALERT_COOLDOWN_MS = 10_000L
        private const val DEEPFAKE_ALERT_DURATION_MS = 350
        const val DEEPFAKE_ALERT_CONFIDENCE_THRESHOLD = 0.8f
        
        // Audio processing setup
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_STEREO
        private const val CHANNEL_COUNT = 2
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_FACTOR = 4
        
        private const val AUDIO_CHUNK_DURATION_MS = 1000
        private const val OVERLAP_DURATION_MS = 0
        private const val KNN_FEATURE_COUNT = 25
        private const val LGBM_FEATURE_COUNT = 26
        const val MODEL_WARNING =
            "Experimental: both models' one-second live predictions are unvalidated; Android features may differ from training."

        @Volatile
        var isActive = false
            private set
    }
    
    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val isStopping = AtomicBoolean(false)
    
    // Audio processing
    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)
    private var audioProcessingJob: Job? = null
    
    // Audio state backup for restoration
    private var originalAudioMode: Int = AudioManager.MODE_NORMAL
    private var originalSpeakerState: Boolean = false
    private var currentAudioSource: Int = MediaRecorder.AudioSource.DEFAULT
    private var originalCommunicationDevice: AudioDeviceInfo? = null
    private var isSpeakerphoneRequested = false
    private var audioStateCaptured = false
    private var speakerphoneChanged = false
    private var audioFocusRequest: AudioFocusRequest? = null
    
    // ONNX ML inference
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var lgbmSession: OrtSession? = null
    private var isModelLoaded = AtomicBoolean(false)
    private val audioProcessor = AudioProcessor()
    private var pendingStartIntent: Intent? = null
    private var modelLoadJob: Job? = null
    private var deepfakeAlertTone: ToneGenerator? = null
    private var lastDeepfakeAlertTimeMs = Long.MIN_VALUE
    
    // Call state
    private var currentPhoneNumber: String? = null
    private var isIncomingCall = false
    
    // Results tracking
    private val detectionResults = java.util.Collections.synchronizedList(mutableListOf<DetectionResult>())
    @Volatile
    private var detectionResultListener: DetectionResultListener? = null

    fun setDetectionResultListener(listener: DetectionResultListener?) {
        detectionResultListener = listener
    }
    
    // UI overlay
    private var overlayView: OverlayView? = null
    private var windowManager: WindowManager? = null
    
    inner class LocalBinder : Binder() {
        fun getService(): RealTimeAudioDetectionService = this@RealTimeAudioDetectionService
        fun setDetectionResultListener(listener: DetectionResultListener?) {
            this@RealTimeAudioDetectionService.setDetectionResultListener(listener)
        }
    }

    fun interface DetectionResultListener {
        fun onDetectionResult(result: DetectionResult)
    }

    data class DetectionResult(
        val timestamp: Long,
        val isFake: Boolean,
        val confidence: Float,
        val lgbmFakeConfidence: Float? = null,
        val audioChunkId: Int = -1,
        val processingTimeMs: Long = 0L,
        val warning: String? = MODEL_WARNING
    )
    
    data class AudioAnalysisResult(
        val isFake: Boolean,
        val confidence: Float,
        val fakeConfidence: Float,
        val realConfidence: Float,
        val processingTimeMs: Long,
        val audioLengthMs: Long,
        val lgbmFakeConfidence: Float? = null,
        val lgbmRealConfidence: Float? = null,
        val error: String? = null,
        val warning: String? = MODEL_WARNING
    )
    
    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        initializeWindowManager()
        Timber.d("RealTimeAudioDetectionService created")
    }
    
    override fun onBind(intent: Intent?): IBinder = binder
    
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action ?: return START_NOT_STICKY
        
        when (action) {
            ACTION_PREPARE -> {
                isActive = true
                handlePrepare(intent)
            }
            ACTION_START_DETECTION -> {
                isActive = true
                handleStartDetection(intent)
            }
            ACTION_STOP_DETECTION -> {
                isActive = false
                detectionResultListener = null
                handleStopDetection()
            }
            else -> {
                Timber.w("Ignoring unknown service action: $action")
                return START_NOT_STICKY
            }
        }
        
        return START_STICKY
    }
    
    private fun handlePrepare(intent: Intent) {
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)
        
        startForeground(NOTIFICATION_ID, createNotification("Preparing for call monitoring..."))
        if (!isModelLoaded.get() && modelLoadJob?.isActive != true) {
            modelLoadJob = serviceScope.launch(Dispatchers.IO) {
                loadDeepfakeModel()
            }
        }
    }
    
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun handleStartDetection(intent: Intent) {
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)

        if (!isModelLoaded.get() || ortSession == null || lgbmSession == null) {
            pendingStartIntent = intent
            handlePrepare(intent)
            return
        }

        startDetectionSession()
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startDetectionSession() {
        if (isStopping.get()) return
        pendingStartIntent = null
        startForeground(NOTIFICATION_ID, createNotification("Monitoring speakerphone audio - experimental KNN + LightGBM"))
        showOverlay()
        startAudioMonitoring()
    }
    
    private fun handleStopDetection() {
        if (!isStopping.compareAndSet(false, true)) return
        pendingStartIntent = null
        val loadingJob = modelLoadJob
        loadingJob?.cancel()
        val recordingJob = audioProcessingJob
        stopAudioMonitoring()
        serviceScope.launch {
            loadingJob?.join()
            recordingJob?.join()
            hideOverlay()
            generateCallSummary()
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }
    
    private fun loadDeepfakeModel() {
        try {
            val knnModelFile = getModelFile("knn_modelv2.onnx")
            val lgbmModelFile = getModelFile("lgbmv2.onnx")
            if (!knnModelFile.exists() || !lgbmModelFile.exists()) {
                Timber.e("KNN or LightGBM ONNX model file is missing")
                isModelLoaded.set(false)
                return
            }

            ortEnv = OrtEnvironment.getEnvironment()
            val loadedKnnSession = createSessionWithProviderFallbacks(knnModelFile)
            val loadedLgbmSession = try {
                createSessionWithProviderFallbacks(lgbmModelFile)
            } catch (e: Exception) {
                loadedKnnSession.close()
                throw e
            }
            if (isStopping.get()) {
                loadedKnnSession.close()
                loadedLgbmSession.close()
                return
            }
            ortSession = loadedKnnSession
            lgbmSession = loadedLgbmSession
            isModelLoaded.set(true)

            val queuedIntent = pendingStartIntent
            if (queuedIntent != null) {
                pendingStartIntent = null
                serviceScope.launch(Dispatchers.Main) {
                    if (ActivityCompat.checkSelfPermission(
                            this as Context,
                            Manifest.permission.RECORD_AUDIO
                        ) != PackageManager.PERMISSION_GRANTED
                    ) {
                        // TODO: Consider calling
                        //    ActivityCompat#requestPermissions
                        // here to request the missing permissions, and then overriding
                        //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                        //                                          int[] grantResults)
                        // to handle the case where the user grants the permission. See the documentation
                        // for ActivityCompat#requestPermissions for more details.
                        return@launch
                    }
                    if (!isStopping.get()) startDetectionSession()
                }
            }

            Timber.i("ONNX KNN and LightGBM models loaded successfully")

            serviceScope.launch(Dispatchers.Main) {
                if (isStopping.get()) return@launch
                val notification = createNotification("KNN + LightGBM models loaded - experimental results")
                val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }

        } catch (e: Exception) {
            Timber.e(e, "Failed to load ONNX models")
            ortSession?.close()
            lgbmSession?.close()
            ortSession = null
            lgbmSession = null
            isModelLoaded.set(false)
            serviceScope.launch(Dispatchers.Main) {
                if (isStopping.get()) return@launch
                val notification = createNotification("Model loading failed - Check logs")
                val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
        }
    }

    private fun createSessionWithProviderFallbacks(modelFile: File): OrtSession {
        val env = ortEnv ?: throw IllegalStateException("ONNX Runtime environment is not initialized")
        val availableProviders = OrtEnvironment.getAvailableProviders()
        Timber.i("ONNX Runtime available providers: ${availableProviders.joinToString { it.name }}")
        val acceleratedProviders = listOf<Triple<String, OrtProvider, (OrtSession.SessionOptions) -> Unit>>(
            Triple("QNN GPU", OrtProvider.QNN) { options ->
                options.addQnn(mapOf("backend_type" to "gpu"))
            },
            Triple("QNN HTP", OrtProvider.QNN) { options ->
                options.addQnn(mapOf("backend_type" to "htp"))
            },
            Triple("NNAPI", OrtProvider.NNAPI) { options -> options.addNnapi() }
        )
        val providers = acceleratedProviders
            .filter { (_, provider, _) -> provider in availableProviders }
            .map { (name, _, configure) -> name to configure } +
            ("CPU" to { _: OrtSession.SessionOptions -> })
        if (OrtProvider.QNN !in availableProviders) {
            Timber.i("QNN is unavailable in this runtime/device; Snapdragon GPU acceleration cannot be used")
        }
        var lastError: Exception? = null

        for ((providerName, configureProvider) in providers) {
            val options = OrtSession.SessionOptions()
            try {
                configureProvider(options)
                val session = env.createSession(modelFile.absolutePath, options)
                Timber.i(
                    "ONNX session created with $providerName execution provider; " +
                        "unsupported operators may execute on CPU"
                )
                return session
            } catch (e: Exception) {
                lastError = e
                Timber.w(e, "$providerName provider unavailable for this model; trying next provider")
            } finally {
                try {
                    options.close()
                } catch (e: Exception) {
                    Timber.w(e, "Failed to release $providerName session options")
                }
            }
        }

        throw lastError ?: IllegalStateException("Unable to create ONNX Runtime session")
    }
    
    private fun getModelFile(modelName: String): File {
        val assetsModelFile = File(filesDir, modelName)
        val temporaryFile = File(filesDir, "$modelName.tmp")
        try {
            assets.open("models/$modelName").use { input ->
                FileOutputStream(temporaryFile).use { output -> input.copyTo(output) }
            }
            if (assetsModelFile.exists() && !assetsModelFile.delete()) {
                throw IOException("Unable to replace cached model: ${assetsModelFile.absolutePath}")
            }
            if (!temporaryFile.renameTo(assetsModelFile)) {
                throw IOException("Unable to move copied model into place")
            }
            Timber.i("Model copied to: ${assetsModelFile.absolutePath}")
        } catch (e: Exception) {
            temporaryFile.delete()
            throw IOException("Failed to copy ONNX model from assets: $modelName", e)
        }
        return assetsModelFile
    }
    
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startAudioMonitoring() {
        if (isRecording.get()) return
        
        try {
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                throw SecurityException("Microphone permission has not been granted")
            }

            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            originalAudioMode = audioManager.mode
            originalSpeakerState = isSpeakerphoneActive()
            audioStateCaptured = true
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            if (!originalSpeakerState) {
                setSpeakerphoneOn(true)
                speakerphoneChanged = true
            }

            val minBufferBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBufferBytes <= 0) {
                throw IllegalStateException("Unsupported audio format or sample rate (AudioRecord error $minBufferBytes)")
            }
            val bufferSizeBytes = minBufferBytes * BUFFER_SIZE_FACTOR
            val audioSources = arrayOf(
                MediaRecorder.AudioSource.MIC,
                MediaRecorder.AudioSource.VOICE_COMMUNICATION
            )
            
            for (source in audioSources) {
                var recorder: AudioRecord? = null
                try {
                    recorder = AudioRecord(source, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSizeBytes)
                    if (recorder.state == AudioRecord.STATE_INITIALIZED) {
                        audioRecord = recorder
                        currentAudioSource = source
                        break
                    }
                    Timber.w("AudioRecord source $source did not initialize")
                } catch (e: Exception) {
                    Timber.w(e, "Unable to initialize AudioRecord source $source")
                } finally {
                    if (audioRecord !== recorder) recorder?.release()
                }
            }
            
            if (audioRecord == null) {
                throw IllegalStateException("No supported audio source could initialize")
            }
            
            val recorder = audioRecord ?: throw IllegalStateException("AudioRecord was not initialized")
            recorder.startRecording()
            if (recorder.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                throw IllegalStateException("AudioRecord did not enter recording state")
            }
            isRecording.set(true)
            
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val focusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE)
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .build()
                audioFocusRequest = focusRequest
                if (audioManager.requestAudioFocus(focusRequest) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                    Timber.w("Audio focus was not granted")
                }
            }
            
            audioProcessingJob = serviceScope.launch(Dispatchers.IO) {
                processAudioStream()
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to start audio monitoring")
            stopAudioMonitoring()
            val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(
                NOTIFICATION_ID,
                createNotification("Audio capture failed - Check microphone access")
            )
        }
    }
    
    private fun stopAudioMonitoring() {
        isRecording.set(false)
        audioProcessingJob?.cancel()
        audioProcessingJob = null
        val recorder = audioRecord
        audioRecord = null
        if (recorder != null) {
            try {
                if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop()
            } catch (e: Exception) {
                Timber.w(e, "Failed to stop AudioRecord")
            }
            try {
                recorder.release()
            } catch (e: Exception) {
                Timber.w(e, "Failed to release AudioRecord")
            }
        }
        
        if (!audioStateCaptured) return
        val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
        try {
            audioManager.mode = originalAudioMode
        } catch (e: Exception) {
            Timber.w(e, "Failed to restore audio mode")
        }
        if (speakerphoneChanged) {
            try {
                setSpeakerphoneOn(false)
            } catch (e: Exception) {
                Timber.w(e, "Failed to restore communication device")
            }
        }
        speakerphoneChanged = false
        audioStateCaptured = false
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            audioFocusRequest?.let {
                try {
                    audioManager.abandonAudioFocusRequest(it)
                } catch (e: Exception) {
                    Timber.w(e, "Failed to abandon audio focus")
                }
            }
            audioFocusRequest = null
        }
    }
    
    private suspend fun processAudioStream() {
        val chunkSamples = (SAMPLE_RATE * AUDIO_CHUNK_DURATION_MS * CHANNEL_COUNT) / 1000
        val overlapSamples = (SAMPLE_RATE * OVERLAP_DURATION_MS * CHANNEL_COUNT) / 1000
        val minBufferBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        if (minBufferBytes <= 0) {
            Timber.e("Unable to determine AudioRecord read buffer size: $minBufferBytes")
            return
        }
        
        val audioBuffer = ShortArray((minBufferBytes + 1) / 2)
        val chunkQueue = ShortArray(chunkSamples)
        val inferenceQueue = Channel<Pair<Int, ShortArray>>(capacity = 2)
        val inferenceJob = serviceScope.launch(Dispatchers.IO) {
            for ((chunkId, chunk) in inferenceQueue) {
                processAudioChunk(chunk, chunkId)
            }
        }
        var queueSize = 0
        var chunkId = 0
        
        try {
            while (isRecording.get() && currentCoroutineContext().isActive) {
                try {
                    val samplesRead = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0
                    if (samplesRead < 0) {
                        Timber.e("AudioRecord read failed with error $samplesRead")
                        break
                    }
                    if (samplesRead > 0) {
                        for (i in 0 until samplesRead) {
                            if (queueSize < chunkSamples) {
                                chunkQueue[queueSize++] = audioBuffer[i]
                            } else {
                                val chunkToProcess = chunkId++ to chunkQueue.copyOf()
                                if (inferenceQueue.trySend(chunkToProcess).isFailure) {
                                    Timber.w("Dropping audio chunk ${chunkToProcess.first}; inference is behind")
                                }

                                if (overlapSamples > 0) {
                                    System.arraycopy(
                                        chunkQueue,
                                        chunkSamples - overlapSamples,
                                        chunkQueue,
                                        0,
                                        overlapSamples
                                    )
                                    queueSize = overlapSamples
                                } else {
                                    queueSize = 0
                                }
                                chunkQueue[queueSize++] = audioBuffer[i]
                            }
                        }
                    }
                    delay(10)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Timber.e(e, "Audio stream processing failed")
                    break
                }
            }
        } finally {
            inferenceQueue.close()
            inferenceJob.cancelAndJoin()
            if (isRecording.get()) {
                withContext(Dispatchers.Main) {
                    Timber.w("Audio stream ended unexpectedly; stopping capture")
                    stopAudioMonitoring()
                }
            }
        }
    }
    
    private fun buildOnnxInputTensor(
        features: FloatArray,
        session: OrtSession,
        expectedFeatureCount: Int,
        modelName: String
    ): OnnxTensor {
        val inputInfo = session.inputInfo
        if (inputInfo.size != 1) {
            throw IllegalStateException("$modelName expects one model input, found ${inputInfo.size}")
        }
        val tensorInfo = inputInfo.values.first().info as? TensorInfo
            ?: throw IllegalStateException("$modelName input is not a tensor")
        val declaredShape = tensorInfo.shape
        val env = ortEnv ?: throw IllegalStateException("ONNX Runtime environment is not initialized")
        require(declaredShape.size == 2) {
            "$modelName input must have rank 2, found ${declaredShape.size}"
        }
        require(declaredShape[0] <= 0 || declaredShape[0] == 1L) {
            "$modelName only supports batch size 1; model declares ${declaredShape[0]}"
        }
        require(declaredShape[1] == expectedFeatureCount.toLong()) {
            "$modelName expects $expectedFeatureCount features; model declares ${declaredShape[1]}"
        }
        require(tensorInfo.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
            "$modelName requires float input, found ${tensorInfo.type}"
        }
        require(features.size == expectedFeatureCount) {
            "$modelName feature extractor produced ${features.size} values; expected $expectedFeatureCount"
        }

        return OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(features),
            longArrayOf(1L, expectedFeatureCount.toLong())
        )
    }

    private fun extractFakeProbability(results: OrtSession.Result): Float {
        return extractClassProbability(results, fakeClassIndex = 1)
    }

    private fun extractLgbmFakeProbability(results: OrtSession.Result): Float {
        // The LightGBM notebook encodes FAKE as class 0 and REAL as class 1.
        return extractClassProbability(results, fakeClassIndex = 0)
    }

    private fun extractClassProbability(results: OrtSession.Result, fakeClassIndex: Int): Float {
        val probabilityOutput = results.get("probabilities").orElse(null)
            ?: results.get("output_probability").orElseThrow {
                IllegalStateException("Model probability output is missing")
            }
        val probability = when (val value = probabilityOutput.value) {
            is List<*> -> {
                val classProbabilities = value.firstOrNull() as? Map<*, *>
                    ?: throw IllegalStateException("Probability output has an unsupported map format")
                (classProbabilities[fakeClassIndex.toLong()] as? Number)?.toFloat()
                    ?: (classProbabilities[fakeClassIndex] as? Number)?.toFloat()
                    ?: (classProbabilities[fakeClassIndex.toString()] as? Number)?.toFloat()
                    ?: throw IllegalStateException("Probability output has no fake class ($fakeClassIndex)")
            }
            is Map<*, *> -> {
                (value[fakeClassIndex.toLong()] as? Number)?.toFloat()
                    ?: (value[fakeClassIndex] as? Number)?.toFloat()
                    ?: (value[fakeClassIndex.toString()] as? Number)?.toFloat()
                    ?: throw IllegalStateException("Probability output has no fake class ($fakeClassIndex)")
            }
            is Array<*> -> {
                require(value.size == 1) {
                    "Probability output must have batch size 1, found ${value.size}"
                }
                val classProbabilities = value[0] as? FloatArray
                    ?: throw IllegalStateException("Probability output batch has an unsupported array format")
                selectFakeClassProbability(classProbabilities, fakeClassIndex)
            }
            is FloatArray -> selectFakeClassProbability(value, fakeClassIndex)
            is OnnxTensor -> {
                val tensorInfo = probabilityOutput.info as? TensorInfo
                    ?: throw IllegalStateException("Probability output is not a tensor")
                require(tensorInfo.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
                    "Probability output tensor is not float"
                }
                val buffer = value.floatBuffer
                val values = FloatArray(buffer.remaining()).also(buffer::get)
                selectFakeClassProbability(values, fakeClassIndex)
            }
            else -> throw IllegalStateException("Probability output has unsupported type ${value::class.java.name}")
        }
        require(probability.isFinite() && probability in 0f..1f) {
            "Model returned invalid fake probability: $probability"
        }
        return probability
    }

    private fun selectFakeClassProbability(probabilities: FloatArray, fakeClassIndex: Int): Float {
        return when (probabilities.size) {
            1 -> probabilities[0]
            2 -> probabilities[fakeClassIndex]
            else -> throw IllegalStateException("Probability output has ${probabilities.size} values")
        }
    }

    private suspend fun processAudioChunk(audioData: ShortArray, chunkId: Int) {
        val startTime = android.os.SystemClock.elapsedRealtime()
        try {
            if (!isModelLoaded.get()) throw IllegalStateException("ONNX model is not loaded")

            val featuresResult = audioProcessor.generateKnnFeatures(audioData, SAMPLE_RATE, CHANNEL_COUNT)
            if (featuresResult.error != null) throw IllegalStateException(featuresResult.error)

            val knn = ortSession ?: throw IllegalStateException("KNN ONNX session is not loaded")
            val lgbm = lgbmSession ?: throw IllegalStateException("LightGBM ONNX session is not loaded")
            val knnInputName = knn.inputNames.firstOrNull() ?: throw IllegalStateException("KNN has no input")
            val knnFakeProb = buildOnnxInputTensor(
                featuresResult.features,
                knn,
                KNN_FEATURE_COUNT,
                "KNN"
            ).use { input ->
                knn.run(mapOf(knnInputName to input)).use(::extractFakeProbability)
            }
            val lgbmFeatures = floatArrayOf(featuresResult.chromaStft, *featuresResult.features)
            val lgbmInputName = lgbm.inputNames.firstOrNull() ?: throw IllegalStateException("LightGBM has no input")
            val lgbmFakeProb = buildOnnxInputTensor(
                lgbmFeatures,
                lgbm,
                LGBM_FEATURE_COUNT,
                "LightGBM"
            ).use { input ->
                lgbm.run(mapOf(lgbmInputName to input)).use(::extractLgbmFakeProbability)
            }

            val isFake = knnFakeProb > 0.5f
            val result = DetectionResult(
                timestamp = System.currentTimeMillis(),
                isFake = isFake,
                confidence = knnFakeProb,
                lgbmFakeConfidence = lgbmFakeProb,
                audioChunkId = chunkId,
                processingTimeMs = android.os.SystemClock.elapsedRealtime() - startTime
            )

            detectionResults.add(result)
            withContext(Dispatchers.Main) { updateOverlay(result) }

            if (isFake && knnFakeProb > 0.7f) {
                Timber.w("KNN reports high fake probability: ${knnFakeProb * 100}%")
            }
            if (lgbmFakeProb > 0.7f) {
                Timber.w("LightGBM reports high fake probability: ${lgbmFakeProb * 100}%")
            }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Timber.e(e, "Error in processAudioChunk")
        }
    }

    suspend fun analyzeRawAudio(
        audioData: ShortArray,
        sampleRate: Int = SAMPLE_RATE,
        audioLengthMs: Long = -1L,
        channelCount: Int = 1
    ): AudioAnalysisResult {
        val startTime = android.os.SystemClock.elapsedRealtime()
        return withContext(Dispatchers.IO) {
            try {
                if (!isModelLoaded.get()) throw IllegalStateException("Model not loaded")

                val featuresResult = audioProcessor.generateKnnFeatures(audioData, sampleRate, channelCount)
                if (featuresResult.error != null) throw IllegalStateException(featuresResult.error)

                val knn = ortSession ?: throw IllegalStateException("KNN ONNX session is not loaded")
                val lgbm = lgbmSession ?: throw IllegalStateException("LightGBM ONNX session is not loaded")
                val knnInputName = knn.inputNames.firstOrNull() ?: throw IllegalStateException("KNN has no input")
                val knnFakeProb = buildOnnxInputTensor(
                    featuresResult.features,
                    knn,
                    KNN_FEATURE_COUNT,
                    "KNN"
                ).use { input ->
                    knn.run(mapOf(knnInputName to input)).use(::extractFakeProbability)
                }
                val lgbmFeatures = floatArrayOf(featuresResult.chromaStft, *featuresResult.features)
                val lgbmInputName = lgbm.inputNames.firstOrNull() ?: throw IllegalStateException("LightGBM has no input")
                val lgbmFakeProb = buildOnnxInputTensor(
                    lgbmFeatures,
                    lgbm,
                    LGBM_FEATURE_COUNT,
                    "LightGBM"
                ).use { input ->
                    lgbm.run(mapOf(lgbmInputName to input)).use(::extractLgbmFakeProbability)
                }

                val isFake = knnFakeProb > 0.5f
                val realProb = 1f - knnFakeProb

                AudioAnalysisResult(
                    isFake = isFake,
                    confidence = if (isFake) knnFakeProb else realProb,
                    fakeConfidence = knnFakeProb,
                    realConfidence = realProb,
                    processingTimeMs = android.os.SystemClock.elapsedRealtime() - startTime,
                    audioLengthMs = audioLengthMs,
                    lgbmFakeConfidence = lgbmFakeProb,
                    lgbmRealConfidence = 1f - lgbmFakeProb,
                    warning = MODEL_WARNING
                )
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Timber.e(e, "Failed to analyze audio")
                AudioAnalysisResult(
                    isFake = false,
                    confidence = 0f,
                    fakeConfidence = 0f,
                    realConfidence = 0f,
                    processingTimeMs = android.os.SystemClock.elapsedRealtime() - startTime,
                    audioLengthMs = audioLengthMs,
                    error = e.message ?: "Audio analysis failed",
                    warning = null
                )
            }
        }
    }

    suspend fun analyzeRawAudio(
        audioBytes: ByteArray,
        sampleRate: Int = SAMPLE_RATE,
        audioLengthMs: Long = -1L,
        channelCount: Int = 1
    ): AudioAnalysisResult {
        if (audioBytes.size % 2 != 0) {
            return AudioAnalysisResult(
                isFake = false,
                confidence = 0f,
                fakeConfidence = 0f,
                realConfidence = 0f,
                processingTimeMs = 0L,
                audioLengthMs = audioLengthMs,
                error = "PCM audio data must contain complete 16-bit samples"
            )
        }
        val audioData = ShortArray(audioBytes.size / 2)
        for (i in audioData.indices) {
            val low = audioBytes[i * 2].toInt() and 0xFF
            val high = audioBytes[i * 2 + 1].toInt() and 0xFF
            audioData[i] = ((high shl 8) or low).toShort()
        }
        return analyzeRawAudio(audioData, sampleRate, audioLengthMs, channelCount)
    }
    
    fun isReadyForAnalysis(): Boolean =
        isModelLoaded.get() && ortSession != null && lgbmSession != null
    
    fun getAudioSourceInfo(): String {
        return if (isRecording.get() && audioRecord != null) "Recording active" else "Not recording"
    }
    
    fun getCallRecordingInfo(): String = "Monitoring active"
    
    private fun isSpeakerphoneActive(): Boolean {
        val am = getSystemService(AUDIO_SERVICE) as? AudioManager ?: return false
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            am.communicationDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER
        } else {
            @Suppress("DEPRECATION") am.isSpeakerphoneOn
        }
    }
    
    private fun setSpeakerphoneOn(enabled: Boolean) {
        val am = getSystemService(AUDIO_SERVICE) as? AudioManager ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (enabled) {
                if (!isSpeakerphoneRequested) {
                    originalCommunicationDevice = am.communicationDevice
                    isSpeakerphoneRequested = true
                }
                val speaker = am.availableCommunicationDevices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                if (speaker == null || !am.setCommunicationDevice(speaker)) {
                    Timber.w("Built-in speaker is not available as a communication device")
                }
            } else {
                if (!isSpeakerphoneRequested) return
                isSpeakerphoneRequested = false
                val savedDevice = originalCommunicationDevice
                if (savedDevice != null && am.availableCommunicationDevices.contains(savedDevice)) {
                    am.setCommunicationDevice(savedDevice)
                } else {
                    am.clearCommunicationDevice()
                }
                originalCommunicationDevice = null
            }
        } else {
            @Suppress("DEPRECATION")
            am.isSpeakerphoneOn = enabled
        }
    }
    
    private fun showOverlay() {
        if (overlayView != null) return
        try {
            val view = OverlayView(this)
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                PixelFormat.TRANSLUCENT
            )
            val manager = windowManager ?: throw IllegalStateException("Window manager is unavailable")
            manager.addView(view, params)
            overlayView = view
        } catch (e: SecurityException) {
            Timber.e(e, "Overlay permission is unavailable")
        } catch (e: WindowManager.BadTokenException) {
            Timber.e(e, "Unable to attach detection overlay")
        } catch (e: IllegalArgumentException) {
            Timber.e(e, "Unable to attach detection overlay")
        } catch (e: IllegalStateException) {
            Timber.e(e, "Unable to attach detection overlay")
        }
    }
    
    private fun hideOverlay() {
        val view = overlayView ?: return
        overlayView = null
        try {
            windowManager?.removeView(view)
        } catch (e: IllegalArgumentException) {
            Timber.w(e, "Detection overlay was already detached")
        }
    }
    
    private fun updateOverlay(result: DetectionResult) {
        overlayView?.updateDetectionResult(result)
        detectionResultListener?.onDetectionResult(result)
        if (result.isFake && result.confidence > DEEPFAKE_ALERT_CONFIDENCE_THRESHOLD) {
            playDeepfakeAlert()
        }
    }

    private fun playDeepfakeAlert() {
        val now = android.os.SystemClock.elapsedRealtime()
        if (lastDeepfakeAlertTimeMs != Long.MIN_VALUE &&
            now - lastDeepfakeAlertTimeMs < DEEPFAKE_ALERT_COOLDOWN_MS
        ) {
            return
        }

        val tone = deepfakeAlertTone ?: try {
            ToneGenerator(AudioManager.STREAM_NOTIFICATION, 80).also { deepfakeAlertTone = it }
        } catch (e: RuntimeException) {
            Timber.e(e, "Unable to initialize the deepfake audio warning")
            return
        }
        if (tone.startTone(ToneGenerator.TONE_PROP_BEEP2, DEEPFAKE_ALERT_DURATION_MS)) {
            lastDeepfakeAlertTimeMs = now
            Timber.w("Audible warning played for a live-call chunk flagged as fake")
        } else {
            Timber.w("Unable to play the deepfake audio warning")
        }
    }
    
    private fun generateCallSummary() {
        val results = synchronized(detectionResults) {
            detectionResults.toList().also { detectionResults.clear() }
        }
        if (results.isEmpty()) return
        val total = results.size
        val fakes = results.count { it.isFake }
        Timber.i("Call Summary - Total: $total, Fake: $fakes")
    }
    
    private fun initializeWindowManager() {
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Deepfake Detection", NotificationManager.IMPORTANCE_LOW)
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }
    
    private fun createNotification(contentText: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("RealTimeAudioDetect")
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setSilent(true)
            .build()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        isActive = false
        stopAudioMonitoring()
        hideOverlay()
        deepfakeAlertTone?.release()
        deepfakeAlertTone = null
        try {
            ortSession?.close()
            lgbmSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            Timber.e(e, "Error closing ONNX runtime resources")
        }
        serviceScope.cancel()
    }
}
