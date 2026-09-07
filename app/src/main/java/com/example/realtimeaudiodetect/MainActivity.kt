/**
 * Main UI controller for RealTimeAudioDetect
 * Handles permissions, service lifecycle, and file analysis API
 */
package com.example.realtimeaudiodetect

import android.Manifest
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.content.ComponentName
import android.content.Context
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
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlinx.coroutines.launch
import timber.log.Timber
import androidx.core.content.edit

class MainActivity : AppCompatActivity(), ServiceConnection {
    
    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.READ_CALL_LOG,
            Manifest.permission.MODIFY_AUDIO_SETTINGS,
            Manifest.permission.FOREGROUND_SERVICE,
            Manifest.permission.FOREGROUND_SERVICE_PHONE_CALL
        )
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
    private lateinit var tvSelectedFile: TextView
    private lateinit var layoutAnalysisResults: LinearLayout
    private lateinit var tvAnalysisResult: TextView
    private lateinit var tvConfidenceScores: TextView
    
    // Service state
    private var deepfakeService: RealTimeAudioDetectionService? = null
    private var isServiceBound = false
    private var selectedAudioUri: Uri? = null
    
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
    
    @RequiresApi(Build.VERSION_CODES.O)
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
        if (isServiceBound) {
            unbindService(this)
            isServiceBound = false
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
        tvSelectedFile = findViewById(R.id.tvSelectedFile)
        layoutAnalysisResults = findViewById(R.id.layoutAnalysisResults)
        tvAnalysisResult = findViewById(R.id.tvAnalysisResult)
        tvConfidenceScores = findViewById(R.id.tvConfidenceScores)
        
        val prefs = getSharedPreferences("realtime_audio_detect", MODE_PRIVATE)
        switchAutoStart.isChecked = prefs.getBoolean("auto_start", false)
    }
    
    @RequiresApi(Build.VERSION_CODES.O)
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
                        "• Foreground service: To run in background")
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
                        data = Uri.parse("package:$packageName")
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
        val activityManager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        @Suppress("DEPRECATION")
        val services = activityManager.getRunningServices(Integer.MAX_VALUE)
        
        return services.any { serviceInfo ->
            serviceInfo.service.className == RealTimeAudioDetectionService::class.java.name
        }
    }
    
    @RequiresApi(Build.VERSION_CODES.O)
    private fun toggleService() {
        if (isServiceRunning()) {
            stopMonitoringService()
        } else {
            startMonitoringService()
        }
    }
    
    @RequiresApi(Build.VERSION_CODES.O)
    private fun startMonitoringService() {
        if (!hasAllPermissions()) {
            requestPermissions()
            return
        }
        
        val intent = Intent(this, RealTimeAudioDetectionService::class.java).apply {
            action = RealTimeAudioDetectionService.ACTION_PREPARE
        }
        
        try {
            startForegroundService(intent)
            
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
        val intent = Intent(this, RealTimeAudioDetectionService::class.java)
        stopService(intent)
        
        if (isServiceBound) {
            unbindService(this)
            isServiceBound = false
            deepfakeService = null
        }
        
        updateServiceStatus()
    }
    
    private fun bindToService() {
        val intent = Intent(this, RealTimeAudioDetectionService::class.java)
        bindService(intent, this, BIND_AUTO_CREATE)
    }
    
    override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
        val binder = service as RealTimeAudioDetectionService.LocalBinder
        deepfakeService = binder.getService()
        isServiceBound = true
        updateServiceStatus()
        Timber.d("Service connected")
    }
    
    override fun onServiceDisconnected(name: ComponentName?) {
        deepfakeService = null
        isServiceBound = false
        Timber.d("Service disconnected")
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

    @RequiresApi(Build.VERSION_CODES.O)
    fun analyzeRawAudio(
        audioData: ShortArray,
        sampleRate: Int = 16000,
        audioLengthMs: Long = -1L,
        callback: (RealTimeAudioDetectionService.AudioAnalysisResult) -> Unit
    ) {
        lifecycleScope.launch {
            if (!isServiceRunning()) {
                startMonitoringService()
                var attempts = 0
                while (!isServiceBound && attempts < 50) {
                    kotlinx.coroutines.delay(100)
                    attempts++
                }
            }
            
            if (isServiceBound && deepfakeService != null) {
                var modelAttempts = 0
                while (!deepfakeService!!.isReadyForAnalysis() && modelAttempts < 100) {
                    kotlinx.coroutines.delay(100)
                    modelAttempts++
                }
                
                val result = deepfakeService!!.analyzeRawAudio(audioData, sampleRate, audioLengthMs)
                callback(result)
            }
        }
    }
    
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
                cursor.moveToFirst()
                cursor.getString(nameIndex)
            } ?: "Unknown file"
        } catch (e: Exception) {
            "Audio file"
        }
    }
    
    @RequiresApi(Build.VERSION_CODES.O)
    private fun analyzeSelectedAudioFile() {
        selectedAudioUri?.let { uri ->
            lifecycleScope.launch {
                try {
                    btnAnalyzeFile.isEnabled = false
                    progressBar.visibility = View.VISIBLE
                    
                    val audioData = readAudioFile(uri)
                    if (audioData != null) {
                        analyzeRawAudio(audioData) { result ->
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
                    btnAnalyzeFile.isEnabled = true
                    progressBar.visibility = View.GONE
                }
            }
        }
    }
    
    private suspend fun readAudioFile(uri: Uri): ShortArray? {
        return try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val bytes = inputStream.readBytes()
                val dataStartIndex = if (bytes.size > 44 && 
                    bytes.sliceArray(0..3).contentEquals("RIFF".toByteArray())) 44 else 0
                
                val audioBytes = bytes.sliceArray(dataStartIndex until bytes.size)
                val samples = ShortArray(audioBytes.size / 2)
                for (i in samples.indices) {
                    val low = audioBytes[i * 2].toInt() and 0xFF
                    val high = audioBytes[i * 2 + 1].toInt() and 0xFF
                    samples[i] = ((high shl 8) or low).toShort()
                }
                samples
            }
        } catch (e: Exception) {
            null
        }
    }
    
    private fun displayAnalysisResult(result: RealTimeAudioDetectionService.AudioAnalysisResult) {
        layoutAnalysisResults.visibility = View.VISIBLE
        if (result.error != null) {
            tvAnalysisResult.text = "Error"
            tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
            tvConfidenceScores.text = result.error
        } else {
            if (result.isFake) {
                tvAnalysisResult.text = "DEEPFAKE DETECTED"
                tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.error_color))
            } else {
                tvAnalysisResult.text = "Real Audio"
                tvAnalysisResult.setTextColor(ContextCompat.getColor(this, R.color.success_color))
            }
            val fakePercent = (result.fakeConfidence * 100).toInt()
            val realPercent = (result.realConfidence * 100).toInt()
            tvConfidenceScores.text = "Real: $realPercent% | Fake: $fakePercent%"
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
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                analyzeRawAudio(dummyData) { result ->
                    lifecycleScope.launch {
                        progressBar.visibility = View.GONE
                        displayAnalysisResult(result)
                    }
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
