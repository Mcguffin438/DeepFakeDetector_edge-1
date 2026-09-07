/**
 * Foreground service for real-time deepfake detection during calls
 * Captures audio, runs ONNX inference, displays overlay results
 */
package com.example.realtimeaudiodetect

import android.Manifest
import android.app.*
import android.content.Context
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
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.io.FileOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.*

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
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        private const val BUFFER_SIZE_FACTOR = 4
        
        private const val AUDIO_CHUNK_DURATION_MS = 6000   // 6s chunks
        private const val OVERLAP_DURATION_MS = 500        // 0.5s overlap
    }
    
    private val binder = LocalBinder()
    private val serviceScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    
    // Audio processing
    private var audioRecord: AudioRecord? = null
    private val isRecording = AtomicBoolean(false)
    private var audioProcessingJob: Job? = null
    
    // Audio state backup for restoration
    private var originalAudioMode: Int = AudioManager.MODE_NORMAL
    private var originalSpeakerState: Boolean = false
    private var currentAudioSource: Int = MediaRecorder.AudioSource.DEFAULT
    private var originalCommunicationDevice: AudioDeviceInfo? = null
    
    // ONNX ML inference
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    private var isModelLoaded = AtomicBoolean(false)
    private val audioProcessor = AudioProcessor()
    
    // Call state
    private var currentPhoneNumber: String? = null
    private var isIncomingCall = false
    
    // Results tracking
    private var detectionResults = mutableListOf<DetectionResult>()
    private var lastDetectionTime = 0L
    
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
            ACTION_PREPARE -> handlePrepare(intent)
            ACTION_START_DETECTION -> handleStartDetection(intent)
            ACTION_STOP_DETECTION -> handleStopDetection()
        }
        
        return START_STICKY
    }
    
    private fun handlePrepare(intent: Intent) {
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)
        
        startForeground(NOTIFICATION_ID, createNotification("Preparing for call monitoring..."))
        serviceScope.launch(Dispatchers.IO) { loadDeepfakeModel() }
    }
    
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun handleStartDetection(intent: Intent) {
        if (!isModelLoaded.get()) {
            handlePrepare(intent)
            return
        }
        
        currentPhoneNumber = intent.getStringExtra(EXTRA_PHONE_NUMBER)
        isIncomingCall = intent.getBooleanExtra(EXTRA_IS_INCOMING, false)
        
        startForeground(NOTIFICATION_ID, createNotification("Monitoring call for deepfakes..."))
        showOverlay()
        startAudioMonitoring()
    }
    
    private fun handleStopDetection() {
        stopAudioMonitoring()
        hideOverlay()
        generateCallSummary()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
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
            val sessionOptions = OrtSession.SessionOptions()
            ortSession = ortEnv!!.createSession(modelFile.absolutePath, sessionOptions)
            isModelLoaded.set(true)
            
            Timber.i("✅ ONNX model loaded successfully: ${modelFile.name}")
            
            serviceScope.launch(Dispatchers.Main) {
                val notification = createNotification("Model loaded - Ready to analyze audio")
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
            
        } catch (e: Exception) {
            Timber.e(e, "❌ Failed to load ONNX model")
            isModelLoaded.set(false)
            serviceScope.launch(Dispatchers.Main) {
                val notification = createNotification("Model loading failed - Check logs")
                val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
                notificationManager.notify(NOTIFICATION_ID, notification)
            }
        }
    }
    
    private fun getModelFile(): File {
        val assetsModelFile = File(filesDir, "knn_modelv2.onnx")
        if (!assetsModelFile.exists()) {
            try {
                assets.open("models/knn_modelv2.onnx").use { input ->
                    FileOutputStream(assetsModelFile).use { output ->
                        input.copyTo(output)
                    }
                }
                Timber.i("✅ Model copied to: ${assetsModelFile.absolutePath}")
            } catch (e: Exception) {
                Timber.e(e, "❌ Failed to copy model from assets")
            }
        }
        return assetsModelFile
    }
    
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startAudioMonitoring() {
        if (isRecording.get()) return
        
        try {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT) * BUFFER_SIZE_FACTOR
            val audioSources = arrayOf(
                MediaRecorder.AudioSource.VOICE_DOWNLINK,
                MediaRecorder.AudioSource.VOICE_CALL,
                MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                MediaRecorder.AudioSource.MIC
            )
            
            for (source in audioSources) {
                val ar = AudioRecord(source, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, bufferSize)
                if (ar.state == AudioRecord.STATE_INITIALIZED) {
                    audioRecord = ar
                    currentAudioSource = source
                    break
                }
                ar.release()
            }
            
            if (audioRecord == null) {
                Timber.e("Failed to initialize AudioRecord")
                return
            }
            
            audioRecord?.startRecording()
            isRecording.set(true)
            
            val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
            originalAudioMode = audioManager.mode
            originalSpeakerState = isSpeakerphoneActive()
            
            audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
            if (!originalSpeakerState) setSpeakerphoneOn(true)
            
            audioProcessingJob = serviceScope.launch(Dispatchers.IO) {
                processAudioStream()
            }
        } catch (e: Exception) {
            Timber.e(e, "Failed to start audio monitoring")
        }
    }
    
    private fun stopAudioMonitoring() {
        isRecording.set(false)
        audioProcessingJob?.cancel()
        audioRecord?.apply {
            try { stop(); release() } catch (e: Exception) { Timber.e(e) }
        }
        audioRecord = null
        
        val audioManager = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        audioManager.mode = originalAudioMode
        setSpeakerphoneOn(originalSpeakerState)
    }
    
    private suspend fun processAudioStream() {
        val chunkSamples = (SAMPLE_RATE * AUDIO_CHUNK_DURATION_MS) / 1000
        val overlapSamples = (SAMPLE_RATE * OVERLAP_DURATION_MS) / 1000
        val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
        
        val audioBuffer = ShortArray(bufferSize)
        val audioChunk = mutableListOf<Short>()
        var chunkId = 0
        
        while (isRecording.get() && currentCoroutineContext().isActive) {
            try {
                val bytesRead = audioRecord?.read(audioBuffer, 0, bufferSize) ?: 0
                if (bytesRead > 0) {
                    for (i in 0 until bytesRead) audioChunk.add(audioBuffer[i])
                    if (audioChunk.size >= chunkSamples) {
                        val chunkToProcess = audioChunk.take(chunkSamples).toShortArray()
                        processAudioChunk(chunkToProcess, chunkId++)
                        val samplesToRemove = chunkSamples - overlapSamples
                        repeat(samplesToRemove) { if (audioChunk.isNotEmpty()) audioChunk.removeAt(0) }
                    }
                }
                delay(10)
            } catch (e: Exception) {
                Timber.e(e)
                break
            }
        }
    }
    
    private suspend fun processAudioChunk(audioData: ShortArray, chunkId: Int) {
        try {
            if (!isModelLoaded.get() || ortSession == null) return
            
            val featuresResult = audioProcessor.generateMultiChannelFeatures(audioData, SAMPLE_RATE)
            if (featuresResult.error != null) return
            
            val inputName = ortSession?.inputNames?.iterator()?.next() ?: return
            val fb = FloatBuffer.wrap(featuresResult.features)
            val onnxInput = OnnxTensor.createTensor(ortEnv, fb, longArrayOf(1, featuresResult.features.size.toLong()))
            
            val results = ortSession!!.run(mapOf(inputName to onnxInput))
            
            val fakeProb = try {
                if (results.size() > 1) {
                    @Suppress("UNCHECKED_CAST")
                    val probsMap = results[1].value as? List<Map<Long, Float>>
                    probsMap?.firstOrNull()?.get(1L) ?: 0.5f
                } else {
                    val outputValue = results[0].value as? FloatArray
                    outputValue?.firstOrNull() ?: 0.5f
                }
            } catch (e: Exception) {
                0.5f
            } finally {
                results.close()
                onnxInput.close()
            }

            val isFake = fakeProb > 0.5f
            val result = DetectionResult(System.currentTimeMillis(), isFake, fakeProb, chunkId)
            
            detectionResults.add(result)
            withContext(Dispatchers.Main) { updateOverlay(result) }
            
            if (isFake && fakeProb > 0.7f) {
                Timber.w("HIGH CONFIDENCE DEEPFAKE DETECTED: ${fakeProb * 100}%")
            }
            lastDetectionTime = System.currentTimeMillis()
        } catch (e: Exception) {
            Timber.e(e, "Error in processAudioChunk")
        }
    }
    
    suspend fun analyzeRawAudio(
        audioData: ShortArray, 
        sampleRate: Int = SAMPLE_RATE,
        audioLengthMs: Long = -1L
    ): AudioAnalysisResult {
        val startTime = System.currentTimeMillis()
        return withContext(Dispatchers.IO) {
            try {
                if (!isModelLoaded.get() || ortSession == null) {
                    throw Exception("Model not loaded")
                }
            
                val featuresResult = audioProcessor.generateMultiChannelFeatures(audioData, sampleRate)
                if (featuresResult.error != null) throw Exception(featuresResult.error)
            
                val inputName = ortSession?.inputNames?.iterator()?.next() ?: throw Exception("No input")
                val fb = FloatBuffer.wrap(featuresResult.features)
                val onnxInput = OnnxTensor.createTensor(ortEnv, fb, longArrayOf(1, featuresResult.features.size.toLong()))

                val results = ortSession!!.run(mapOf(inputName to onnxInput))
                val fakeProb = if (results.size() > 1) {
                    @Suppress("UNCHECKED_CAST")
                    val map = results[1].value as? List<Map<Long, Float>>
                    map?.firstOrNull()?.get(1L) ?: 0.5f
                } else {
                    (results[0].value as? FloatArray)?.firstOrNull() ?: 0.5f
                }
                results.close()
                onnxInput.close()

                val isFake = fakeProb > 0.5f
                val realProb = 1f - fakeProb
                
                AudioAnalysisResult(
                    isFake = isFake,
                    confidence = if (isFake) fakeProb else realProb,
                    fakeConfidence = fakeProb,
                    realConfidence = realProb,
                    processingTimeMs = System.currentTimeMillis() - startTime,
                    audioLengthMs = audioLengthMs
                )
            } catch (e: Exception) {
                Timber.e(e)
                AudioAnalysisResult(false, 0f, 0f, 1f, 0, audioLengthMs, e.message)
            }
        }
    }

    suspend fun analyzeRawAudio(audioBytes: ByteArray, sampleRate: Int = SAMPLE_RATE, audioLengthMs: Long = -1L): AudioAnalysisResult {
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
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            am.communicationDevice?.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER
        } else {
            @Suppress("DEPRECATION") am.isSpeakerphoneOn
        }
    }
    
    private fun setSpeakerphoneOn(enabled: Boolean) {
        val am = getSystemService(Context.AUDIO_SERVICE) as AudioManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (enabled) {
                originalCommunicationDevice = am.communicationDevice
                val speaker = am.availableCommunicationDevices.find { it.type == AudioDeviceInfo.TYPE_BUILTIN_SPEAKER }
                if (speaker != null) am.setCommunicationDevice(speaker)
            } else {
                originalCommunicationDevice?.let { am.setCommunicationDevice(it) } ?: am.clearCommunicationDevice()
            }
        } else {
            @Suppress("DEPRECATION") am.isSpeakerphoneOn = enabled
        }
    }
    
    private fun showOverlay() {
        if (overlayView != null) return
        try {
            overlayView = OverlayView(this)
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.WRAP_CONTENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                android.graphics.PixelFormat.TRANSLUCENT
            )
            windowManager?.addView(overlayView, params)
        } catch (e: Exception) { Timber.e(e) }
    }
    
    private fun hideOverlay() {
        overlayView?.let { try { windowManager?.removeView(it); overlayView = null } catch (e: Exception) { Timber.e(e) } }
    }
    
    private fun updateOverlay(result: DetectionResult) {
        overlayView?.updateDetectionResult(result)
    }
    
    private fun generateCallSummary() {
        if (detectionResults.isEmpty()) return
        val total = detectionResults.size
        val fakes = detectionResults.count { it.isFake }
        Timber.i("Call Summary - Total: $total, Fake: $fakes")
        detectionResults.clear()
    }
    
    private fun initializeWindowManager() {
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
    }
    
    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Deepfake Detection", NotificationManager.IMPORTANCE_LOW)
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
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
        stopAudioMonitoring()
        hideOverlay()
        serviceScope.cancel()
    }
}
