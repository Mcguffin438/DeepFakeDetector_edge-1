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
        
        // Audio processing setup
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_STEREO
        private const val CHANNEL_COUNT = 2
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_FACTOR = 4
        
        private const val AUDIO_CHUNK_DURATION_MS = 1000
        private const val OVERLAP_DURATION_MS = 0
        private const val KNN_FEATURE_COUNT = 25
        const val MODEL_WARNING =
            "Experimental: one-second live predictions are unvalidated and Android features may differ from training."

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
    private var isModelLoaded = AtomicBoolean(false)
    private val audioProcessor = AudioProcessor()
    private var pendingStartIntent: Intent? = null
    private var modelLoadJob: Job? = null
    
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

        if (!isModelLoaded.get() || ortSession == null) {
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
        startForeground(NOTIFICATION_ID, createNotification("Monitoring speakerphone audio - experimental KNN"))
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
            val modelFile = getModelFile()
            if (!modelFile.exists()) {
                Timber.e("Model file not found: ${modelFile.absolutePath}")
                isModelLoaded.set(false)
                return
            }

            ortEnv = OrtEnvironment.getEnvironment()
            val loadedSession = createSessionWithProviderFallbacks(modelFile)
            if (isStopping.get()) {
                loadedSession.close()
                return
            }
            ortSession = loadedSession
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

            Timber.i("✅ ONNX model loaded successfully: ${modelFile.name}")

            serviceScope.launch(Dispatchers.Main) {
                if (isStopping.get()) return@launch
                val notification = createNotification("Model loaded - Experimental KNN results")
                val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }

        } catch (e: Exception) {
            Timber.e(e, "❌ Failed to load ONNX model")
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
        val providers = listOf<Pair<String, (OrtSession.SessionOptions) -> Unit>>(
            "QNN GPU" to { options -> options.addQnn(mapOf("backend_type" to "gpu")) },
            "QNN HTP" to { options -> options.addQnn(mapOf("backend_type" to "htp")) },
            "NNAPI" to { options -> options.addNnapi() },
            "CPU" to { _ -> }
        )
        var lastError: Exception? = null

        for ((providerName, configureProvider) in providers) {
            val options = OrtSession.SessionOptions()
            try {
                configureProvider(options)
                val session = env.createSession(modelFile.absolutePath, options)
                Timber.i(
                    "ONNX session created with $providerName provider enabled; " +
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
    
    private fun getModelFile(): File {
        val assetsModelFile = File(filesDir, "knn_modelv2.onnx")
        if (!assetsModelFile.exists() || assetsModelFile.length() == 0L) {
            val temporaryFile = File(filesDir, "knn_modelv2.onnx.tmp")
            try {
                assets.open("models/knn_modelv2.onnx").use { input ->
                    FileOutputStream(temporaryFile).use { output -> input.copyTo(output) }
                }
                if (!temporaryFile.renameTo(assetsModelFile)) {
                    throw IOException("Unable to move copied model into place")
                }
                Timber.i("Model copied to: ${assetsModelFile.absolutePath}")
            } catch (e: Exception) {
                temporaryFile.delete()
                throw IOException("Failed to copy ONNX model from assets", e)
            }
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
    
    private fun buildOnnxInputTensor(features: FloatArray): OnnxTensor {
        val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
        val inputInfo = session.inputInfo
        if (inputInfo.size != 1) {
            throw IllegalStateException("Expected one model input, found ${inputInfo.size}")
        }
        val tensorInfo = inputInfo.values.first().info as? TensorInfo
            ?: throw IllegalStateException("Model input is not a tensor")
        val declaredShape = tensorInfo.shape
        val env = ortEnv ?: throw IllegalStateException("ONNX Runtime environment is not initialized")
        require(declaredShape.size == 2) {
            "Expected the exported KNN model input to have rank 2, found ${declaredShape.size}"
        }
        require(declaredShape[0] <= 0 || declaredShape[0] == 1L) {
            "Only batch size 1 is supported; model declares ${declaredShape[0]}"
        }
        require(declaredShape[1] == KNN_FEATURE_COUNT.toLong()) {
            "KNN model expects $KNN_FEATURE_COUNT features; model declares ${declaredShape[1]}"
        }
        require(tensorInfo.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
            "KNN model requires float input, found ${tensorInfo.type}"
        }
        require(features.size == KNN_FEATURE_COUNT) {
            "Feature extractor produced ${features.size} values; expected $KNN_FEATURE_COUNT"
        }

        return OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(features),
            longArrayOf(1L, KNN_FEATURE_COUNT.toLong())
        )
    }

    private fun extractFakeProbability(results: OrtSession.Result): Float {
        val probabilityOutput = results.get("probabilities").orElse(null)
            ?: results.get("output_probability").orElseThrow {
                IllegalStateException("Model probability output is missing")
            }
        val probability = when (val value = probabilityOutput.value) {
            is List<*> -> {
                val classProbabilities = value.firstOrNull() as? Map<*, *>
                    ?: throw IllegalStateException("Probability output has an unsupported map format")
                (classProbabilities[1L] as? Number)?.toFloat()
                    ?: (classProbabilities[1] as? Number)?.toFloat()
                    ?: (classProbabilities["1"] as? Number)?.toFloat()
                    ?: throw IllegalStateException("Probability output has no deepfake class (label 1)")
            }
            is Map<*, *> -> {
                (value[1L] as? Number)?.toFloat()
                    ?: (value[1] as? Number)?.toFloat()
                    ?: (value["1"] as? Number)?.toFloat()
                    ?: throw IllegalStateException("Probability output has no deepfake class (label 1)")
            }
            is OnnxTensor -> {
                val tensorInfo = probabilityOutput.info as? TensorInfo
                    ?: throw IllegalStateException("Probability output is not a tensor")
                require(tensorInfo.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
                    "Probability output tensor is not float"
                }
                val buffer = value.floatBuffer
                val values = FloatArray(buffer.remaining()).also(buffer::get)
                when (values.size) {
                    1 -> values[0]
                    2 -> values[1]
                    else -> throw IllegalStateException("Probability output has ${values.size} values")
                }
            }
            else -> throw IllegalStateException("Probability output has unsupported type ${value::class.java.name}")
        }
        require(probability.isFinite() && probability in 0f..1f) {
            "Model returned invalid fake probability: $probability"
        }
        return probability
    }

    private suspend fun processAudioChunk(audioData: ShortArray, chunkId: Int) {
        val startTime = android.os.SystemClock.elapsedRealtime()
        try {
            if (!isModelLoaded.get()) throw IllegalStateException("ONNX model is not loaded")

            val featuresResult = audioProcessor.generateKnnFeatures(audioData, SAMPLE_RATE, CHANNEL_COUNT)
            if (featuresResult.error != null) throw IllegalStateException(featuresResult.error)

            val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
            val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("No model input")
            val fakeProb = buildOnnxInputTensor(featuresResult.features).use { input ->
                session.run(mapOf(inputName to input)).use(::extractFakeProbability)
            }

            val isFake = fakeProb > 0.5f
            val result = DetectionResult(
                timestamp = System.currentTimeMillis(),
                isFake = isFake,
                confidence = fakeProb,
                audioChunkId = chunkId,
                processingTimeMs = android.os.SystemClock.elapsedRealtime() - startTime
            )

            detectionResults.add(result)
            withContext(Dispatchers.Main) { updateOverlay(result) }

            if (isFake && fakeProb > 0.7f) {
                Timber.w("HIGH CONFIDENCE DEEPFAKE DETECTED: ${fakeProb * 100}%")
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

                val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
                val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("No model input")
                val fakeProb = buildOnnxInputTensor(featuresResult.features).use { input ->
                    session.run(mapOf(inputName to input)).use(::extractFakeProbability)
                }

                val isFake = fakeProb > 0.5f
                val realProb = 1f - fakeProb

                AudioAnalysisResult(
                    isFake = isFake,
                    confidence = if (isFake) fakeProb else realProb,
                    fakeConfidence = fakeProb,
                    realConfidence = realProb,
                    processingTimeMs = android.os.SystemClock.elapsedRealtime() - startTime,
                    audioLengthMs = audioLengthMs,
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
    
    fun isReadyForAnalysis(): Boolean = isModelLoaded.get() && ortSession != null
    
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
        try {
            ortSession?.close()
            ortEnv?.close()
        } catch (e: Exception) {
            Timber.e(e, "Error closing ONNX runtime resources")
        }
        serviceScope.cancel()
    }
}
