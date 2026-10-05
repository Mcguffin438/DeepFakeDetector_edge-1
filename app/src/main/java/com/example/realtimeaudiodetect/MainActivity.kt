/**
 * Main UI controller for RealTimeAudioDetect
 * Handles permissions, service lifecycle, and file analysis API
 */
package com.example.realtimeaudiodetect

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.ComponentName
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.provider.Settings
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import timber.log.Timber
import androidx.core.content.edit
import androidx.core.net.toUri
import java.io.BufferedInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

class MainActivity : AppCompatActivity(), ServiceConnection {
    
    companion object {
        private val REQUIRED_PERMISSIONS: Array<String>
            get() {
                val list = mutableListOf(
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.READ_PHONE_STATE,
                    Manifest.permission.READ_CALL_LOG
                )
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    list.add(Manifest.permission.POST_NOTIFICATIONS)
                }
                return list.toTypedArray()
            }
    }
    
    // Core UI components
    private lateinit var tvStatus: TextView
    private lateinit var tvServiceStatus: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var btnToggleService: MaterialButton
    private lateinit var btnPermissions: Button
    private lateinit var btnSettings: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var switchAutoStart: SwitchMaterial
    
    // New UI components
    private lateinit var tvMainStatusLabel: TextView
    private lateinit var tvMainStatusDesc: TextView
    private lateinit var viewShieldPulse: View
    
    // File analysis UI
    private lateinit var btnSelectAudioFile: Button
    private lateinit var btnAnalyzeFile: Button
    private lateinit var btnTestRealSample: MaterialButton
    private lateinit var btnTestFakeSample: MaterialButton
    private lateinit var btnBrowseDatasetSamples: MaterialButton
    private lateinit var tvSelectedFile: TextView
    private lateinit var layoutAnalysisResults: LinearLayout
    private lateinit var tvAnalysisResult: TextView
    private lateinit var tvConfidenceScores: TextView
    private lateinit var tvDetectionTiming: TextView
    
    // Service state
    private var deepfakeService: RealTimeAudioDetectionService? = null
    private var isServiceBound = false
    private var isBindingToService = false
    private var selectedAudioUri: Uri? = null
    private val liveProcessingTimesMs = ArrayDeque<Long>()

    private data class AudioFileData(
        val samples: ShortArray,
        val sampleRate: Int,
        val channelCount: Int
    )

    private data class BundledDemoSample(
        val assetPath: String,
        val title: String,
        val referenceLabel: String
    )
    
    // Animation state
    private var pulseAnimator: AnimatorSet? = null
    
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        updatePermissionStatus()
        if (hasAllPermissions()) checkOverlayPermission()
    }
    
    private val overlayPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { 
        updatePermissionStatus() 
    }
    
    private val audioFilePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let {
            selectedAudioUri = it
            updateSelectedFileUI()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        if (BuildConfig.DEBUG) {
            Timber.plant(Timber.DebugTree())
        }
        
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupClickListeners()
        updatePermissionStatus()
        
        Timber.i("MainActivity created")
    }
    
    override fun onResume() {
        super.onResume()
        updateServiceStatus()
        updatePermissionStatus()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        stopPulseAnimation()
        deepfakeService?.setDetectionResultListener(null)
        if (isServiceBound || isBindingToService) {
            unbindService(this)
            isServiceBound = false
            isBindingToService = false
        }
    }
    
    private fun initializeViews() {
        tvStatus = findViewById(R.id.tvStatus)
        tvServiceStatus = findViewById(R.id.tvServiceStatus)
        tvModelStatus = findViewById(R.id.tvModelStatus)
        btnToggleService = findViewById(R.id.btnToggleService)
        btnPermissions = findViewById(R.id.btnPermissions)
        btnSettings = findViewById(R.id.btnSettings)
        progressBar = findViewById(R.id.progressBar)
        switchAutoStart = findViewById(R.id.switchAutoStart)
        
        tvMainStatusLabel = findViewById(R.id.tvMainStatusLabel)
        tvMainStatusDesc = findViewById(R.id.tvMainStatusDesc)
        viewShieldPulse = findViewById(R.id.viewShieldPulse)
        
        btnSelectAudioFile = findViewById(R.id.btnSelectAudioFile)
        btnAnalyzeFile = findViewById(R.id.btnAnalyzeFile)
        btnTestRealSample = findViewById(R.id.btnTestRealSample)
        btnTestFakeSample = findViewById(R.id.btnTestFakeSample)
        btnBrowseDatasetSamples = findViewById(R.id.btnBrowseDatasetSamples)
        tvSelectedFile = findViewById(R.id.tvSelectedFile)
        layoutAnalysisResults = findViewById(R.id.layoutAnalysisResults)
        tvAnalysisResult = findViewById(R.id.tvAnalysisResult)
        tvConfidenceScores = findViewById(R.id.tvConfidenceScores)
        tvDetectionTiming = findViewById(R.id.tvDetectionTiming)
        
        val prefs = getSharedPreferences("realtime_audio_detect", MODE_PRIVATE)
        switchAutoStart.isChecked = prefs.getBoolean("auto_start", false)
    }
    
    private fun setupClickListeners() {
        btnToggleService.setOnClickListener {
            toggleService()
        }
        
        btnPermissions.setOnClickListener {
            requestPermissions()
        }
        
        btnSettings.setOnClickListener {
            showSettingsDialog()
        }
        
        switchAutoStart.setOnCheckedChangeListener { _, isChecked ->
            getSharedPreferences("realtime_audio_detect", MODE_PRIVATE)
                .edit { putBoolean("auto_start", isChecked) }
        }
        
        btnSelectAudioFile.setOnClickListener {
            selectAudioFile()
        }
        
        btnAnalyzeFile.setOnClickListener {
            analyzeSelectedAudioFile()
        }

        btnTestRealSample.setOnClickListener {
            analyzeBundledSample("samples/audio/demo_real.wav", "Gary Stafford demo clip", "real")
        }

        btnTestFakeSample.setOnClickListener {
            analyzeBundledSample("samples/audio/demo_fake.wav", "ElevenLabs demo clip", "fake")
        }

        btnBrowseDatasetSamples.setOnClickListener {
            showDatasetSamplePicker()
        }
    }
    
    private fun hasAllPermissions(): Boolean {
        return REQUIRED_PERMISSIONS.all { permission ->
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED
        }
    }
    
    private fun hasOverlayPermission(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            Settings.canDrawOverlays(this)
        } else {
            true
        }
    }
    
    private fun requestPermissions() {
        val missingPermissions = REQUIRED_PERMISSIONS.filter { permission ->
            ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED
        }
        
        if (missingPermissions.isNotEmpty()) {
            AlertDialog.Builder(this)
                .setTitle("Permissions Required")
                .setMessage("RealTimeAudioDetect needs these permissions to monitor phone calls and detect deepfakes in real-time:\n\n" +
                        "• Phone permissions: To detect when calls start/end\n" +
                        "• Audio recording: To analyze call audio\n" +
                        "• Notifications: To display service status")
                .setPositiveButton("Grant") { _, _ ->
                    permissionLauncher.launch(missingPermissions.toTypedArray())
                }
                .setNegativeButton("Cancel", null)
                .show()
        } else {
            checkOverlayPermission()
        }
    }
    
    private fun checkOverlayPermission() {
        if (!hasOverlayPermission()) {
            AlertDialog.Builder(this)
                .setTitle("Overlay Permission Required")
                .setMessage("RealTimeAudioDetect needs permission to display detection results over other apps during calls.")
                .setPositiveButton("Grant") { _, _ ->
                    val intent = Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION).apply {
                        data = "package:$packageName".toUri()
                    }
                    overlayPermissionLauncher.launch(intent)
                }
                .setNegativeButton("Skip", null)
                .show()
        }
    }
    
    private fun updatePermissionStatus() {
        val hasPermissions = hasAllPermissions()
        val hasOverlay = hasOverlayPermission()
        
        when {
            hasPermissions && hasOverlay -> {
                tvStatus.text = "Granted"
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.success_color))
                btnPermissions.visibility = View.GONE
            }
            hasPermissions -> {
                tvStatus.text = "Overlay Needed"
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.warning_color))
                btnPermissions.visibility = View.VISIBLE
                btnPermissions.text = "Grant Overlay"
            }
            else -> {
                tvStatus.text = "Required"
                tvStatus.setTextColor(ContextCompat.getColor(this, R.color.error_color))
                btnPermissions.visibility = View.VISIBLE
                btnPermissions.text = "Grant Permissions"
            }
        }
    }
    
    private fun updateServiceStatus() {
        lifecycleScope.launch {
            val isRunning = isServiceRunning()
            
            if (isRunning) {
                tvServiceStatus.text = "Active"
                tvServiceStatus.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.success_color))
                
                tvMainStatusLabel.text = "System Protected"
                tvMainStatusLabel.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.shield_active))
                tvMainStatusDesc.text = "Monitoring active for all calls"
                
                btnToggleService.backgroundTintList = ColorStateList.valueOf(
                    ContextCompat.getColor(this@MainActivity, R.color.shield_active)
                )
                startPulseAnimation(R.color.shield_active)
                
                if (!isServiceBound) {
                    bindToService()
                } else {
                    deepfakeService?.let { service ->
                        if (service.isReadyForAnalysis()) {
                            tvModelStatus.text = "Loaded"
                            tvModelStatus.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.success_color))
                        }
                    }
                }
            } else {
                tvServiceStatus.text = "Inactive"
                tvServiceStatus.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.error_color))
                
                tvMainStatusLabel.text = "System Unprotected"
                tvMainStatusLabel.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.shield_inactive))
                tvMainStatusDesc.text = "Tap to enable real-time monitoring"
                
                btnToggleService.backgroundTintList = ColorStateList.valueOf(
                    ContextCompat.getColor(this@MainActivity, R.color.shield_inactive)
                )
                stopPulseAnimation()
                
                tvModelStatus.text = "Waiting"
                tvModelStatus.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.secondary_text))
            }
        }
    }
    
    private fun isServiceRunning(): Boolean {
        return RealTimeAudioDetectionService.isActive
    }
    
    private fun toggleService() {
        if (isServiceRunning()) {
            stopMonitoringService()
        } else {
            startMonitoringService()
        }
    }
    
    private fun startMonitoringService() {
        if (!hasAllPermissions()) {
            requestPermissions()
            return
        }
        
        val intent = Intent(this, RealTimeAudioDetectionService::class.java).apply {
            action = RealTimeAudioDetectionService.ACTION_PREPARE
        }
        
        try {
            ContextCompat.startForegroundService(this, intent)
            bindToService()
            
            // Visual feedback for starting
            tvMainStatusLabel.text = "Starting..."
            btnToggleService.backgroundTintList = ColorStateList.valueOf(
                ContextCompat.getColor(this, R.color.shield_loading)
            )
            
            lifecycleScope.launch {
                kotlinx.coroutines.delay(1000)
                updateServiceStatus()
            }
            
        } catch (e: Exception) {
            Timber.e(e, "Failed to start service")
            Toast.makeText(this, "Failed to start service: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }
    
    private fun stopMonitoringService() {
        val intent = Intent(this, RealTimeAudioDetectionService::class.java).apply {
            action = RealTimeAudioDetectionService.ACTION_STOP_DETECTION
        }
        startService(intent)
        
        if (isServiceBound || isBindingToService) {
            unbindService(this)
            isServiceBound = false
            isBindingToService = false
            deepfakeService = null
        }

        lifecycleScope.launch {
            kotlinx.coroutines.delay(100)
            updateServiceStatus()
        }
    }
    
    private fun bindToService() {
        if (isServiceBound || isBindingToService) return
        val intent = Intent(this, RealTimeAudioDetectionService::class.java)
        isBindingToService = bindService(intent, this, BIND_AUTO_CREATE)
        if (!isBindingToService) {
            Timber.e("Failed to bind to detection service")
        }
    }

    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
        val binder = service as? RealTimeAudioDetectionService.LocalBinder
        if (binder == null) {
            unbindService(this)
            isBindingToService = false
            Timber.e("Unexpected binder connected to detection service")
            return
        }
        deepfakeService = binder.getService()
        binder.setDetectionResultListener(
            RealTimeAudioDetectionService.DetectionResultListener(::displayLiveDetectionResult)
        )
        isServiceBound = true
        isBindingToService = false
        updateServiceStatus()
        Timber.d("Service connected")
    }
    
    override fun onServiceDisconnected(name: ComponentName?) {
        deepfakeService?.setDetectionResultListener(null)
        deepfakeService = null
        isServiceBound = false
        Timber.d("Service disconnected")
    }

    private fun displayLiveDetectionResult(result: RealTimeAudioDetectionService.DetectionResult) {
        val knnFakePercent = (result.confidence * 100).toInt()
        val knnRealPercent = ((1f - result.confidence) * 100).toInt()
        liveProcessingTimesMs.addLast(result.processingTimeMs)
        if (liveProcessingTimesMs.size > 20) liveProcessingTimesMs.removeFirst()
        layoutAnalysisResults.visibility = View.VISIBLE
        val lgbmFake = result.lgbmFakeConfidence
        val lgbmIsFake = lgbmFake?.let { it > 0.5f }
        tvAnalysisResult.text = when {
            lgbmIsFake == null ->
                if (result.isFake) "LIVE CALL: KNN FLAGS FAKE" else "LIVE CALL: KNN FLAGS REAL"
            result.isFake && lgbmIsFake -> "LIVE CALL: BOTH MODELS FLAG FAKE"
            !result.isFake && !lgbmIsFake -> "LIVE CALL: BOTH MODELS FLAG REAL"
            else -> "LIVE CALL: MODELS DISAGREE"
        }
        tvAnalysisResult.setTextColor(
            ContextCompat.getColor(
                this,
                if (result.isFake || lgbmIsFake == true) R.color.error_color else R.color.success_color
            )
        )
        val lgbmScoreLine = lgbmFake?.let {
            val fakePercent = (it * 100).toInt()
            "LightGBM: fake $fakePercent% | real ${100 - fakePercent}%\n"
        }.orEmpty()
        tvConfidenceScores.text =
            "KNN: fake $knnFakePercent% | real $knnRealPercent%\n$lgbmScoreLine${result.warning.orEmpty()}"
        val averageProcessingTime = liveProcessingTimesMs.average()
        val maximumProcessingTime = liveProcessingTimesMs.maxOrNull() ?: result.processingTimeMs
        tvDetectionTiming.text =
            "Latest analysis: ${formatProcessingTime(result.processingTimeMs)} (chunk ${result.audioChunkId})\n" +
                "Last ${liveProcessingTimesMs.size} chunks: avg ${formatProcessingTime(averageProcessingTime)}, " +
                "max ${formatProcessingTime(maximumProcessingTime)}\n" +
                "Timing starts after each 1-second audio chunk is collected."
    }

    private fun formatProcessingTime(timeMs: Number): String {
        val milliseconds = timeMs.toDouble()
        return if (milliseconds < 1000.0) {
            String.format(Locale.US, "%.1f ms", milliseconds)
        } else {
            String.format(Locale.US, "%.2f s", milliseconds / 1000.0)
        }
    }
    
    private fun startPulseAnimation(colorRes: Int) {
        if (pulseAnimator != null) return
        
        viewShieldPulse.visibility = View.VISIBLE
        viewShieldPulse.backgroundTintList = ColorStateList.valueOf(
            ContextCompat.getColor(this, colorRes)
        )
        
        val scaleX = ObjectAnimator.ofFloat(viewShieldPulse, View.SCALE_X, 1f, 1.5f)
        val scaleY = ObjectAnimator.ofFloat(viewShieldPulse, View.SCALE_Y, 1f, 1.5f)
        val alpha = ObjectAnimator.ofFloat(viewShieldPulse, View.ALPHA, 0.4f, 0f)
        
        pulseAnimator = AnimatorSet().apply {
            playTogether(scaleX, scaleY, alpha)
            duration = 2000
            interpolator = AccelerateDecelerateInterpolator()
            addListener(object : android.animation.AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: android.animation.Animator) {
                    if (pulseAnimator != null) start()
                }
            })
            start()
        }
    }
    
    private fun stopPulseAnimation() {
        pulseAnimator?.cancel()
        pulseAnimator = null
        viewShieldPulse.visibility = View.INVISIBLE
    }

    fun analyzeRawAudio(
        audioData: ShortArray,
        sampleRate: Int = 16000,
        audioLengthMs: Long = -1L,
        channelCount: Int = 1,
        callback: (RealTimeAudioDetectionService.AudioAnalysisResult) -> Unit
    ) {
        lifecycleScope.launch {
            if (!hasAllPermissions()) {
                callback(audioAnalysisError("Required permissions have not been granted"))
                return@launch
            }

            if (deepfakeService == null) {
                startMonitoringService()
            }

            if (!waitUntil(timeoutMs = 5_000L) { deepfakeService != null }) {
                callback(audioAnalysisError("Could not connect to the detection service"))
                return@launch
            }

            val service = deepfakeService
            if (service == null) {
                callback(audioAnalysisError("Detection service disconnected before analysis"))
                return@launch
            }

            if (!waitUntil(timeoutMs = 10_000L) {
                    deepfakeService === service && service.isReadyForAnalysis()
                }) {
                callback(audioAnalysisError("Detection model did not become ready"))
                return@launch
            }

            callback(service.analyzeRawAudio(audioData, sampleRate, audioLengthMs, channelCount))
        }
    }

    private suspend fun waitUntil(timeoutMs: Long, condition: () -> Boolean): Boolean {
        val deadline = android.os.SystemClock.elapsedRealtime() + timeoutMs
        while (android.os.SystemClock.elapsedRealtime() < deadline) {
            if (condition()) return true
            kotlinx.coroutines.delay(100)
        }
        return condition()
    }

    private fun audioAnalysisError(message: String) =
        RealTimeAudioDetectionService.AudioAnalysisResult(
            isFake = false,
            confidence = 0f,
            fakeConfidence = 0f,
            realConfidence = 1f,
            processingTimeMs = 0L,
            audioLengthMs = -1L,
            error = message,
            warning = null
        )
    
    private fun selectAudioFile() {
        try {
            audioFilePickerLauncher.launch("audio/*")
        } catch (e: Exception) {
            Timber.e(e, "Error launching file picker")
        }
    }
    
    private fun updateSelectedFileUI() {
        selectedAudioUri?.let { uri ->
            tvSelectedFile.text = getFileName(uri)
            btnAnalyzeFile.isEnabled = true
            layoutAnalysisResults.visibility = View.GONE
        }
    }
    
    private fun getFileName(uri: Uri): String {
        return try {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                val nameIndex = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (nameIndex >= 0 && cursor.moveToFirst()) cursor.getString(nameIndex) else null
            } ?: "Unknown file"
        } catch (e: Exception) {
            Timber.e(e, "Error getting file name")
            "Audio file"
        }
    }
    
    private fun analyzeSelectedAudioFile() {
        selectedAudioUri?.let { uri ->
            lifecycleScope.launch {
                try {
                    btnAnalyzeFile.isEnabled = false
                    progressBar.visibility = View.VISIBLE
                    
                    val audioFile = readAudioFile(uri)
                    if (audioFile != null) {
                        analyzeRawAudio(
                            audioFile.samples,
                            audioFile.sampleRate,
                            channelCount = audioFile.channelCount
                        ) { result ->
                            lifecycleScope.launch {
                                displayAnalysisResult(result)
                                btnAnalyzeFile.isEnabled = true
                                progressBar.visibility = View.GONE
                            }
                        }
                    } else {
                        btnAnalyzeFile.isEnabled = true
                        progressBar.visibility = View.GONE
                    }
                } catch (e: Exception) {
                    Timber.e(e, "Error analyzing selected audio file")
                    Toast.makeText(this@MainActivity, "Analysis error: ${e.message}", Toast.LENGTH_LONG).show()
                    btnAnalyzeFile.isEnabled = true
                    progressBar.visibility = View.GONE
                }
            }
        }
    }

    private fun showDatasetSamplePicker() {
        val categories = listOf(
            Triple("Gary Stafford dataset - real speech", "samples/audio/dataset/york_real", "real"),
            Triple("Amazon Polly - synthetic speech", "samples/audio/dataset/amazon_polly_fake", "fake"),
            Triple("ElevenLabs - synthetic speech", "samples/audio/dataset/elevenlabs_fake", "fake")
        )
        val samples = categories.flatMap { (category, assetDirectory, referenceLabel) ->
            assets.list(assetDirectory).orEmpty()
                .filter { it.endsWith(".wav", ignoreCase = true) }
                .sorted()
                .map { fileName ->
                    BundledDemoSample(
                        assetPath = "$assetDirectory/$fileName",
                        title = "$category - ${fileName.substringBeforeLast('.')}",
                        referenceLabel = referenceLabel
                    )
                }
        }
        if (samples.isEmpty()) {
            Toast.makeText(this, "No bundled dataset clips are available", Toast.LENGTH_LONG).show()
            return
        }

        AlertDialog.Builder(this)
            .setTitle("Select an attributed dataset clip")
            .setItems(samples.map(BundledDemoSample::title).toTypedArray()) { _, index ->
                val selected = samples[index]
                analyzeBundledSample(selected.assetPath, selected.title, selected.referenceLabel)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun analyzeBundledSample(assetPath: String, title: String, expectedLabel: String) {
        lifecycleScope.launch {
            btnTestRealSample.isEnabled = false
            btnTestFakeSample.isEnabled = false
            btnBrowseDatasetSamples.isEnabled = false
            btnAnalyzeFile.isEnabled = false
            progressBar.visibility = View.VISIBLE
            tvSelectedFile.text = "$title ($expectedLabel reference label)"
            layoutAnalysisResults.visibility = View.GONE

            try {
                val audioFile = withContext(Dispatchers.IO) {
                    assets.open(assetPath).use { input ->
                        parsePcm16Wav(input.readBytes())
                    }
                }
                analyzeRawAudio(
                    audioFile.samples,
                    audioFile.sampleRate,
                    channelCount = audioFile.channelCount
                ) { result ->
                    displayAnalysisResult(result)
                    tvConfidenceScores.append("\nDemo reference label: $expectedLabel")
                    progressBar.visibility = View.GONE
                    btnTestRealSample.isEnabled = true
                    btnTestFakeSample.isEnabled = true
                    btnBrowseDatasetSamples.isEnabled = true
                    btnAnalyzeFile.isEnabled = selectedAudioUri != null
                }
            } catch (e: Exception) {
                Timber.e(e, "Error analyzing bundled $expectedLabel demo sample")
                Toast.makeText(
                    this@MainActivity,
                    "Demo sample error: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
                progressBar.visibility = View.GONE
                btnTestRealSample.isEnabled = true
                btnTestFakeSample.isEnabled = true
                btnBrowseDatasetSamples.isEnabled = true
                btnAnalyzeFile.isEnabled = selectedAudioUri != null
            }
        }
    }
    
    private suspend fun readAudioFile(uri: Uri): AudioFileData? {
        return withContext(Dispatchers.IO) {
            try {
                contentResolver.openInputStream(uri)?.use { inputStream ->
                    val bufferedStream = BufferedInputStream(inputStream)
                    parsePcm16Wav(bufferedStream.readBytes())
                } ?: throw IOException("Unable to open selected audio file")
            } catch (e: Exception) {
                Timber.e(e, "Error reading audio file")
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Unsupported or invalid WAV file: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
                null
            }
        }
    }

    private fun parsePcm16Wav(bytes: ByteArray): AudioFileData {
        if (bytes.size < 12 ||
            !bytes.copyOfRange(0, 4).contentEquals("RIFF".toByteArray(Charsets.US_ASCII)) ||
            !bytes.copyOfRange(8, 12).contentEquals("WAVE".toByteArray(Charsets.US_ASCII))
        ) {
            throw IOException("Expected a RIFF/WAVE audio file")
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        var offset = 12
        var format: Int? = null
        var channels: Int? = null
        var sampleRate: Int? = null
        var bitsPerSample: Int? = null
        var dataOffset: Int? = null
        var dataSize: Int? = null

        while (offset <= bytes.size - 8) {
            val chunkSize = buffer.getInt(offset + 4).toLong() and 0xFFFF_FFFFL
            val chunkStart = offset + 8
            if (chunkSize > bytes.size - chunkStart) {
                throw IOException("WAV chunk extends beyond end of file")
            }
            val chunkLength = chunkSize.toInt()
            val chunkEnd = chunkStart + chunkLength

            when {
                bytes.copyOfRange(offset, offset + 4)
                    .contentEquals("fmt ".toByteArray(Charsets.US_ASCII)) -> {
                    if (chunkLength < 16) throw IOException("Invalid WAV format chunk")
                    format = buffer.getShort(chunkStart).toInt() and 0xFFFF
                    channels = buffer.getShort(chunkStart + 2).toInt() and 0xFFFF
                    sampleRate = buffer.getInt(chunkStart + 4)
                    bitsPerSample = buffer.getShort(chunkStart + 14).toInt() and 0xFFFF
                }
                bytes.copyOfRange(offset, offset + 4)
                    .contentEquals("data".toByteArray(Charsets.US_ASCII)) -> {
                    dataOffset = chunkStart
                    dataSize = chunkLength
                }
            }

            offset = chunkEnd + (chunkLength and 1)
        }

        val channelCount = channels ?: throw IOException("WAV format chunk is missing")
        val rate = sampleRate ?: throw IOException("WAV sample rate is missing")
        if (format != 1 || bitsPerSample != 16 || channelCount !in 1..2) {
            throw IOException("Only 16-bit PCM mono or stereo WAV files are supported")
        }
        if (rate <= 0) throw IOException("Invalid WAV sample rate")

        val start = dataOffset ?: throw IOException("WAV data chunk is missing")
        val size = dataSize ?: throw IOException("WAV data chunk is missing")
        val blockAlign = channelCount * 2
        if (size == 0 || size % blockAlign != 0) {
            throw IOException("WAV audio data is empty or incomplete")
        }

        val frameCount = size / blockAlign
        val samples = ShortArray(frameCount * channelCount) { index -> buffer.getShort(start + index * 2) }
        return AudioFileData(samples, rate, channelCount)
    }
    
    private fun displayAnalysisResult(result: RealTimeAudioDetectionService.AudioAnalysisResult) {
        layoutAnalysisResults.visibility = View.VISIBLE
        tvDetectionTiming.text =
            "Analysis processing time: ${formatProcessingTime(result.processingTimeMs)}\n" +
                "Includes feature extraction and both model inferences; excludes audio loading."
        if (result.error != null) {
            tvAnalysisResult.text = "Error"
            tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
            tvConfidenceScores.text = result.error
        } else {
            if (result.isFake) {
                tvAnalysisResult.text = if (result.lgbmFakeConfidence != null) {
                    if (result.lgbmFakeConfidence > 0.5f) "BOTH MODELS FLAG FAKE" else "MODELS DISAGREE"
                } else {
                    "KNN FLAGS FAKE"
                }
                tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
            } else {
                tvAnalysisResult.text = if (result.lgbmFakeConfidence != null) {
                    if (result.lgbmFakeConfidence > 0.5f) "MODELS DISAGREE" else "BOTH MODELS FLAG REAL"
                } else {
                    "KNN FLAGS REAL"
                }
                val anyModelFlagsFake = result.lgbmFakeConfidence?.let { it > 0.5f } == true
                tvAnalysisResult.setTextColor(
                    ContextCompat.getColor(
                        this,
                        if (anyModelFlagsFake) R.color.error_color else R.color.success_color
                    )
                )
            }
            val fakePercent = (result.fakeConfidence * 100).toInt()
            val realPercent = (result.realConfidence * 100).toInt()
            val warning = result.warning?.let { "\n$it" }.orEmpty()
            val lgbmScoreLine = result.lgbmFakeConfidence?.let {
                val lgbmFakePercent = (it * 100).toInt()
                "LightGBM: fake $lgbmFakePercent% | real ${100 - lgbmFakePercent}%\n"
            }.orEmpty()
            tvConfidenceScores.text =
                "KNN: fake $fakePercent% | real $realPercent%\n$lgbmScoreLine$warning"
        }
    }
    
    private fun showSettingsDialog() {
        val options = arrayOf("Inference Test", "Notification Settings", "About")
        AlertDialog.Builder(this)
            .setTitle("Settings")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> runInferenceTest()
                    1 -> showNotificationSettings()
                    2 -> showAbout()
                }
            }
            .show()
    }
    
    private fun runInferenceTest() {
        if (deepfakeService?.isReadyForAnalysis() != true) {
            Toast.makeText(this, "Model not ready", Toast.LENGTH_SHORT).show()
            return
        }
        
        lifecycleScope.launch {
            progressBar.visibility = View.VISIBLE
            val dummyData = ShortArray(16000) { (it % 100).toShort() }
            analyzeRawAudio(dummyData) { result ->
                lifecycleScope.launch {
                    progressBar.visibility = View.GONE
                    displayAnalysisResult(result)
                }
            }
        }
    }
    
    private fun showNotificationSettings() {
        val intent = Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS).apply {
            putExtra(Settings.EXTRA_APP_PACKAGE, packageName)
        }
        startActivity(intent)
    }
    
    private fun showAbout() {
        AlertDialog.Builder(this)
            .setTitle("About RealTimeAudioDetect")
            .setMessage("RealTimeAudioDetect v1.0\n\nReal-time deepfake audio detection for calls.\nBuilt with ONNX Runtime.")
            .setPositiveButton("OK", null)
            .show()
    }
}
