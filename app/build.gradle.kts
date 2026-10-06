import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace   = "com.example.realtimeaudiodetect"
    compileSdk  = 37

    defaultConfig {
        applicationId     = "com.example.realtimeaudiodetect"
        minSdk            = 24
        targetSdk         = 35  // Android 15
        versionCode       = 1
        versionName       = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        vectorDrawables.useSupportLibrary = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        buildConfig = true
    }

    androidResources {
        noCompress += setOf("pt", "ptl", "so", "onnx")  // Keep PyTorch and ONNX files uncompressed
    }

    // Native runtime packaging config
    packaging {
        jniLibs.useLegacyPackaging = false
        jniLibs.pickFirsts += "lib/*/libc++_shared.so"
        resources.excludes += setOf(
            "/META-INF/{AL2.0,LGPL2.1}",
            "/META-INF/INDEX.LIST",
            "/META-INF/DEPENDENCIES"
        )
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

dependencies {
    // Core Android
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.fragment.ktx)
    implementation(libs.firebase.sessions)
    implementation(libs.litert.api)

    // UI Components
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    
    // Async processing
    implementation(libs.bundles.coroutines)
    
    // Permissions
    implementation(libs.androidx.activity.ktx)
    
    // Logging
    implementation(libs.timber)

    // Testing
    testImplementation(libs.junit)
    testImplementation(libs.mockito.core)
    testImplementation(libs.kotlinx.coroutines.test)

    // ONNX Support
    implementation(libs.onnxruntime.android)
    implementation(libs.kotlindl.onnx) {
        exclude(group = "com.microsoft.onnxruntime", module = "onnxruntime-android")
    }

    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
}
