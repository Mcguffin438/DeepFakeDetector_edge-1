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
        
        private const val AUDIO_CHUNK_DURATION_MS = 6000   // 6s chunks
        private const val OVERLAP_DURATION_MS = 500        // 0.5s overlap

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
    
    // UI overlay
    private var overlayView: OverlayView? = null
    private var windowManager: WindowManager? = null
    
    inner class LocalBinder : Binder() {
        fun getService(): RealTimeAudioDetectionService = this@RealTimeAudioDetectionService
    }
    
    data class DetectionResult(
        val timestamp: Long,
        val isFake: Boolean,
        val confidence: Float,
        val audioChunkId: Int = -1,
        val processingTimeMs: Long = 0L
    )
    
    data class AudioAnalysisResult(
        val isFake: Boolean,
        val confidence: Float,
        val fakeConfidence: Float,
        val realConfidence: Float,
        val processingTimeMs: Long,
        val audioLengthMs: Long,
        val error: String? = null
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
        startForeground(NOTIFICATION_ID, createNotification("Monitoring call for deepfakes..."))
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
                val notification = createNotification("Model loaded - Ready to analyze audio")
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
            val minBufferBytes = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
            if (minBufferBytes <= 0) {
                throw IllegalStateException("Unsupported audio format or sample rate (AudioRecord error $minBufferBytes)")
            }
            val bufferSizeBytes = minBufferBytes * BUFFER_SIZE_FACTOR
            val audioSources = arrayOf(
                MediaRecorder.AudioSource.VOICE_DOWNLINK,
                MediaRecorder.AudioSource.VOICE_CALL,
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                MediaRecorder.AudioSource.MIC
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
            
            audioRecord?.startRecording()
            if (audioRecord?.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
                throw IllegalStateException("AudioRecord did not enter recording state")
            }
            isRecording.set(true)
            
            val audioManager = getSystemService(AUDIO_SERVICE) as AudioManager
            originalAudioMode = audioManager.mode
            originalSpeakerState = isSpeakerphoneActive()
            audioStateCaptured = true
            
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
            
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            if (!originalSpeakerState) {
                setSpeakerphoneOn(true)
                speakerphoneChanged = true
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
                                val chunkToProcess = chunkQueue.copyOf()
                                processAudioChunk(chunkToProcess, chunkId++)

                                val keepCount = chunkSamples - overlapSamples
                                System.arraycopy(chunkQueue, overlapSamples, chunkQueue, 0, keepCount)
                                queueSize = keepCount
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
            if (isRecording.get()) {
                withContext(Dispatchers.Main) {
                    Timber.w("Audio stream ended unexpectedly; stopping capture")
                    stopAudioMonitoring()
                }
            }
        }
    }
    
    private fun buildOnnxInputTensor(featuresResult: MultiChannelFeaturesResult): OnnxTensor {
        val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
        val inputInfo = session.inputInfo
        if (inputInfo.size != 1) {
            throw IllegalStateException("Expected one model input, found ${inputInfo.size}")
        }
        val tensorInfo = inputInfo.values.first().info as? TensorInfo
            ?: throw IllegalStateException("Model input is not a tensor")
        val declaredShape = tensorInfo.shape
        val env = ortEnv ?: throw IllegalStateException("ONNX Runtime environment is not initialized")

        val inputShape: LongArray
        val inputFeatures: FloatArray
        when (declaredShape.size) {
            1 -> {
                val featureSize = if (declaredShape[0] > 0) declaredShape[0] else {
                    audioProcessor.poolFeatures(featuresResult).size.toLong()
                }
                require(featureSize in 1..Int.MAX_VALUE.toLong()) { "Invalid model feature dimension: $featureSize" }
                inputShape = longArrayOf(featureSize)
                inputFeatures = audioProcessor.prepareModelInputVector(featuresResult, featureSize.toInt())
            }
            2 -> {
                val batchSize = declaredShape[0]
                require(batchSize <= 0 || batchSize == 1L) {
                    "Only batch size 1 is supported, model requires $batchSize"
                }
                val featureSize = if (declaredShape[1] > 0) declaredShape[1] else {
                    audioProcessor.poolFeatures(featuresResult).size.toLong()
                }
                require(featureSize in 1..Int.MAX_VALUE.toLong()) { "Invalid model feature dimension: $featureSize" }
                inputShape = longArrayOf(1L, featureSize)
                inputFeatures = audioProcessor.prepareModelInputVector(featuresResult, featureSize.toInt())
            }
            4 -> {
                val batchSize = declaredShape[0]
                val channels = declaredShape[1]
                val bins = declaredShape[2]
                require(batchSize <= 0 || batchSize == 1L) {
                    "Only batch size 1 is supported, model requires $batchSize"
                }
                require(channels <= 0 || channels == featuresResult.shape[0].toLong()) {
                    "Model expects $channels feature channels, received ${featuresResult.shape[0]}"
                }
                require(bins <= 0 || bins == featuresResult.shape[1].toLong()) {
                    "Model expects $bins feature bins, received ${featuresResult.shape[1]}"
                }
                val timeSteps = if (declaredShape[3] > 0) declaredShape[3] else featuresResult.timeSteps.toLong()
                require(timeSteps in 1..Int.MAX_VALUE.toLong()) { "Invalid model time dimension: $timeSteps" }
                inputShape = longArrayOf(1L, featuresResult.shape[0].toLong(), featuresResult.shape[1].toLong(), timeSteps)
                inputFeatures = resizeFeatureTimeAxis(featuresResult, timeSteps.toInt())
            }
            else -> throw IllegalStateException(
                "Unsupported ONNX input rank ${declaredShape.size}; supported ranks are 1, 2, and 4"
            )
        }

        return OnnxTensor.createTensor(env, FloatBuffer.wrap(inputFeatures), inputShape)
    }

    private fun resizeFeatureTimeAxis(
        featuresResult: MultiChannelFeaturesResult,
        targetTimeSteps: Int
    ): FloatArray {
        val channels = featuresResult.shape[0]
        val bins = featuresResult.shape[1]
        val sourceTimeSteps = featuresResult.shape[2]
        require(sourceTimeSteps > 0) { "Model features have no time steps" }
        val resized = FloatArray(channels * bins * targetTimeSteps)
        for (channel in 0 until channels) {
            for (bin in 0 until bins) {
                val sourceOffset = (channel * bins + bin) * sourceTimeSteps
                val targetOffset = (channel * bins + bin) * targetTimeSteps
                for (targetTime in 0 until targetTimeSteps) {
                    val sourceTime = minOf(
                        (targetTime.toLong() * sourceTimeSteps / targetTimeSteps).toInt(),
                        sourceTimeSteps - 1
                    )
                    resized[targetOffset + targetTime] = featuresResult.features[sourceOffset + sourceTime]
                }
            }
        }
        return resized
    }

    private fun extractFakeProbability(results: OrtSession.Result): Float {
        for (index in 0 until results.size()) {
            val value = results[index].value
            val probability = when (value) {
                is List<*> -> {
                    val probabilities = value.firstOrNull() as? Map<*, *>
                    (probabilities?.get(1L) as? Number)?.toFloat()
                        ?: (probabilities?.get(1) as? Number)?.toFloat()
                        ?: (probabilities?.get("1") as? Number)?.toFloat()
                }
                is OnnxTensor -> if ((value.getInfo() as? TensorInfo)?.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
                    val tensorValues = value.floatBuffer.let { buffer ->
                        FloatArray(buffer.remaining()).also(buffer::get)
                    }
                    when (tensorValues.size) {
                        1 -> tensorValues[0]
                        2 -> tensorValues[1]
                        else -> null
                    }
                } else null
                is FloatArray -> when (value.size) {
                    1 -> value[0]
                    2 -> value[1]
                    else -> null
                }
                is Number -> value.toFloat()
                else -> null
            }
            if (probability != null) {
                require(probability.isFinite() && probability in 0f..1f) {
                    "Model returned invalid fake probability: $probability"
                }
                return probability
            }
        }
        throw IllegalStateException("Model outputs did not contain a supported fake probability")
    }

    private suspend fun processAudioChunk(audioData: ShortArray, chunkId: Int) {
        val startTime = android.os.SystemClock.elapsedRealtime()
        try {
            if (!isModelLoaded.get()) throw IllegalStateException("ONNX model is not loaded")

            val featuresResult = audioProcessor.generateMultiChannelFeatures(audioData, SAMPLE_RATE)
            if (featuresResult.error != null) throw IllegalStateException(featuresResult.error)

            val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
            val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("No model input")
            val fakeProb = buildOnnxInputTensor(featuresResult).use { input ->
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
        audioLengthMs: Long = -1L
    ): AudioAnalysisResult {
        val startTime = android.os.SystemClock.elapsedRealtime()
        return withContext(Dispatchers.IO) {
            try {
                if (!isModelLoaded.get()) throw IllegalStateException("Model not loaded")

                val featuresResult = audioProcessor.generateMultiChannelFeatures(audioData, sampleRate)
                if (featuresResult.error != null) throw IllegalStateException(featuresResult.error)

                val session = ortSession ?: throw IllegalStateException("ONNX session is not loaded")
                val inputName = session.inputNames.firstOrNull() ?: throw IllegalStateException("No model input")
                val fakeProb = buildOnnxInputTensor(featuresResult).use { input ->
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
                    audioLengthMs = audioLengthMs
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
                    error = e.message ?: "Audio analysis failed"
                )
            }
        }
    }

    suspend fun analyzeRawAudio(audioBytes: ByteArray, sampleRate: Int = SAMPLE_RATE, audioLengthMs: Long = -1L): AudioAnalysisResult {
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
        return analyzeRawAudio(audioData, sampleRate, audioLengthMs)
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
