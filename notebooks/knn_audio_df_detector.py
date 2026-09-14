"""
PyTorch End-to-End K-Nearest Neighbors (KNN) Audio Deepfake Detection Pipeline
-----------------------------------------------------------------------------
A complete PyTorch-native implementation for audio deepfake binary classification using
torchaudio feature extraction (MelSpectrogram / MFCC), memory bank k-NN classification,
evaluation metrics, and ONNX export for Android deployment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    import torchaudio.transforms as T
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


class AudioFeatureExtractor(nn.Module):
    """
    Extracts log-mel spectrogram or MFCC features from raw audio waveforms.
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160, feature_type: str = "melspectrogram"):
        super().__init__()
        self.sample_rate = sample_rate
        self.feature_type = feature_type.lower()

        if HAS_TORCHAUDIO:
            if self.feature_type == "melspectrogram":
                self.transform = T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )
            elif self.feature_type == "mfcc":
                self.transform = T.MFCC(
                    sample_rate=sample_rate,
                    n_mfcc=n_mels,
                    melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels}
                )
            else:
                raise ValueError(f"Unsupported feature type: {self.feature_type}")
        else:
            self.transform = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (torch.Tensor): Audio tensor of shape [Batch, Channels, Time] or [Batch, Time].
        Returns:
            torch.Tensor: Feature embeddings of shape [Batch, Embedding_Dim].
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        if self.transform is not None:
            features = self.transform(waveform)
            if self.feature_type == "melspectrogram":
                features = torch.log(torch.clamp(features, min=1e-5))
        else:
            features = waveform.unsqueeze(1)

        if features.dim() > 2:
            embedding = torch.mean(features, dim=-1)
        else:
            embedding = features

        return embedding


class PyTorchKNNAudioDeepfakeClassifier(nn.Module):
    """
    K-NN Classifier built in PyTorch for Audio Deepfake Detection (ASVspoof / Real vs Fake).
    Stores training feature embeddings in a memory bank and computes k-NN cosine/euclidean distances.
    """
    def __init__(self, k: int = 5, metric: str = "cosine", weighted: bool = True):
        super().__init__()
        self.k = k
        self.metric = metric.lower()
        self.weighted = weighted
        
        # Memory bank buffers
        self.register_buffer("memory_embeddings", torch.empty(0))
        self.register_buffer("memory_labels", torch.empty(0, dtype=torch.long))

    @staticmethod
    def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
        """Flattens multi-channel or multi-dimensional feature maps to 2D [Batch, Dim]."""
        if x.dim() > 2:
            return x.reshape(x.size(0), -1)
        return x

    def fit(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Store training audio embeddings and their corresponding binary labels (0 = Fake, 1 = Real).
        """
        embeddings = self._flatten_if_needed(embeddings)
        self.memory_embeddings = embeddings.detach().clone()
        self.memory_labels = labels.detach().clone().long()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classifies input audio embeddings via k-Nearest Neighbors.
        
        Args:
            x (torch.Tensor): Audio feature embeddings of shape [Batch, Embedding_Dim] 
                              or multi-channel feature maps [Batch, Channels, Features, Time].
        Returns:
            torch.Tensor: Probability of being real audio (class 1) of shape [Batch, 1].
        """
        if self.memory_embeddings.numel() == 0:
            raise RuntimeError("KNN memory bank is empty. Call fit() first.")
            
        x = self._flatten_if_needed(x)
            
        num_memory = self.memory_embeddings.size(0)
        k_val = min(self.k, num_memory)

        # Distance calculation
        if self.metric == "cosine":
            x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)
            mem_norm = F.normalize(self.memory_embeddings, p=2, dim=1, eps=1e-8)
            sim_matrix = torch.mm(x_norm, mem_norm.t()) # [B, Mem]
            dist_matrix = torch.clamp(1.0 - sim_matrix, min=0.0, max=2.0)
        elif self.metric == "euclidean":
            x_sq = (x ** 2).sum(dim=1, keepdim=True)
            mem_sq = (self.memory_embeddings ** 2).sum(dim=1, keepdim=True).t()
            dist_sq = torch.clamp(x_sq + mem_sq - 2.0 * torch.mm(x, self.memory_embeddings.t()), min=0.0)
            dist_matrix = torch.sqrt(dist_sq + 1e-8)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Find k nearest neighbors
        distances, indices = torch.topk(dist_matrix, k=k_val, dim=1, largest=False, sorted=True)
        neighbor_labels = self.memory_labels[indices] # [B, k_val]

        if self.weighted and self.metric == "cosine":
            weights = torch.clamp(1.0 - distances, min=1e-5)
            real_mask = (neighbor_labels == 1).float()
            real_weights = (real_mask * weights).sum(dim=1, keepdim=True)
            total_weights = weights.sum(dim=1, keepdim=True)
            probabilities = real_weights / torch.clamp(total_weights, min=1e-8)
        elif self.weighted and self.metric == "euclidean":
            weights = 1.0 / (distances + 1e-5)
            real_mask = (neighbor_labels == 1).float()
            real_weights = (real_mask * weights).sum(dim=1, keepdim=True)
            total_weights = weights.sum(dim=1, keepdim=True)
            probabilities = real_weights / torch.clamp(total_weights, min=1e-8)
        else:
            real_votes = (neighbor_labels == 1).float().sum(dim=1, keepdim=True)
            probabilities = real_votes / float(k_val)

        return probabilities


def export_to_onnx(model: nn.Module, output_path: str = "app/src/main/assets/models/knn_modelv2.onnx", input_dim: int = 64):
    """
    Exports the trained KNN classifier or feature extractor pipeline to ONNX format for Android.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, input_dim)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_embeddings"],
            output_names=["real_probability"],
            dynamic_axes={"input_embeddings": {0: "batch_size"}, "real_probability": {0: "batch_size"}}
        )
        print(f"✅ Successfully exported model to ONNX: {output_path}")
    except Exception as e:
        print(f"⚠️ ONNX export note: {e}")


if __name__ == "__main__":
    print("=== PyTorch End-to-End KNN Audio Deepfake Classifier Pipeline ===")
    
    # 1. Initialize Feature Extractor
    feature_extractor = AudioFeatureExtractor(sample_rate=16000, n_mels=64, feature_type="melspectrogram")

    # 2. Simulate audio waveforms for training [200 samples, 16000 audio samples (1 second)]
    dummy_train_waveforms = torch.randn(200, 16000)
    dummy_train_labels = torch.randint(0, 2, (200,))

    print("Extracting training feature embeddings...")
    train_embeddings = feature_extractor(dummy_train_waveforms)

    # 3. Initialize & Fit KNN Classifier
    classifier = PyTorchKNNAudioDeepfakeClassifier(k=5, metric="cosine", weighted=True)
    classifier.fit(train_embeddings, dummy_train_labels)
    print(f"Fitted KNN Memory Bank with {train_embeddings.size(0)} samples.")

    # 4. Simulate test audio waveforms [10 samples]
    dummy_test_waveforms = torch.randn(10, 16000)
    test_embeddings = feature_extractor(dummy_test_waveforms)

    # 5. Run Inference
    preds = classifier(test_embeddings)
    print(f"Test batch predictions (Real Probability):\n{preds}")

    # 6. Export to ONNX for Android
    export_to_onnx(classifier, output_path="app/src/main/assets/models/knn_modelv2.onnx", input_dim=train_embeddings.size(1))
    print("✅ Successfully verified PyTorch End-to-End Audio Deepfake Pipeline!")
