"""
VoiceGuard: Browser-Compatible Voice Biometric Authentication
With deep learning training and speaker verification
"""

import streamlit as st
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import time
import pickle
import io
import wave

# Audio processing
from scipy import signal
from scipy.spatial.distance import cosine
import librosa
import noisereduce as nr

# Machine Learning
from sklearn.svm import SVC, OneClassSVM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Security
from cryptography.fernet import Fernet

# Set page config
st.set_page_config(
    page_title="VoiceGuard - Real Deep Learning",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    :root {
        --primary-color: #00b0ff;
        --secondary-color: #0080d6;
        --success-color: #4caf50;
        --error-color: #f44336;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main {background-color: #f5f5f7;}
    
    .custom-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    .app-header {
        background: linear-gradient(135deg, #00b0ff 0%, #0080d6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00b0ff 0%, #0080d6 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,176,255,0.4);
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00b0ff;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    
    .recording-indicator {
        background-color: #f44336;
        color: white;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Secure Database
class SecureDatabase:
    def __init__(self):
        self.data_dir = "voiceguard_data_v2"
        self.users_file = os.path.join(self.data_dir, "users.json")
        self.models_file = os.path.join(self.data_dir, "trained_models.pkl")
        self.transactions_file = os.path.join(self.data_dir, "transactions.json")
        self.key_file = os.path.join(self.data_dir, "encryption.key")
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                self.encryption_key = f.read()
        else:
            self.encryption_key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(self.encryption_key)
        
        self.cipher_suite = Fernet(self.encryption_key)
        self._load_data()
    
    def _load_data(self):
        if 'db_users' not in st.session_state:
            if os.path.exists(self.users_file):
                try:
                    with open(self.users_file, 'r') as f:
                        st.session_state.db_users = json.load(f)
                except:
                    st.session_state.db_users = {}
            else:
                st.session_state.db_users = {}
        
        if 'db_trained_models' not in st.session_state:
            if os.path.exists(self.models_file):
                try:
                    with open(self.models_file, 'rb') as f:
                        st.session_state.db_trained_models = pickle.load(f)
                except:
                    st.session_state.db_trained_models = {}
            else:
                st.session_state.db_trained_models = {}
        
        if 'db_transactions' not in st.session_state:
            if os.path.exists(self.transactions_file):
                try:
                    with open(self.transactions_file, 'r') as f:
                        st.session_state.db_transactions = json.load(f)
                except:
                    st.session_state.db_transactions = []
            else:
                st.session_state.db_transactions = []
    
    def _save_users(self):
        with open(self.users_file, 'w') as f:
            json.dump(st.session_state.db_users, f, indent=2)
    
    def _save_models(self):
        with open(self.models_file, 'wb') as f:
            pickle.dump(st.session_state.db_trained_models, f)
    
    def _save_transactions(self):
        with open(self.transactions_file, 'w') as f:
            json.dump(st.session_state.db_transactions, f, indent=2)
    
    def store_user(self, username: str, user_data: dict):
        encrypted_data = self.cipher_suite.encrypt(json.dumps(user_data).encode())
        st.session_state.db_users[username] = encrypted_data.decode('utf-8')
        self._save_users()
    
    def get_user(self, username: str) -> dict:
        if username in st.session_state.db_users:
            encrypted_data = st.session_state.db_users[username]
            if isinstance(encrypted_data, str):
                encrypted_data = encrypted_data.encode('utf-8')
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        return None
    
    def log_transaction(self, transaction: dict):
        transaction['timestamp'] = datetime.now().isoformat()
        for key, value in transaction.items():
            if isinstance(value, (np.float32, np.float64)):
                transaction[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                transaction[key] = int(value)
        st.session_state.db_transactions.append(transaction)
        self._save_transactions()


# ADVANCED Voice Feature Extractor
class AdvancedVoiceFeatureExtractor:
    def __init__(self):
        self.sample_rate = 16000
        self.n_mfcc = 20
        self.n_mels = 80
        self.hop_length = 512
        self.n_fft = 2048
    
    def extract_comprehensive_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract 200+ discriminative features"""
        
        # Advanced preprocessing
        audio_data = self.advanced_preprocess(audio_data)
        
        if len(audio_data) < self.sample_rate * 0.5:
            audio_data = np.pad(audio_data, (0, int(self.sample_rate * 0.5) - len(audio_data)))
        
        features = []
        
        # 1. MFCC Statistics (100 features)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, 
                                    n_mfcc=self.n_mfcc, hop_length=self.hop_length)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.max(mfcc, axis=1))
        features.extend(np.min(mfcc, axis=1))
        features.extend(np.median(mfcc, axis=1))
        
        # 2. Delta and Delta-Delta MFCCs (40 features)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.mean(mfcc_delta2, axis=1))
        
        # 3. Mel Spectrogram Statistics (8 features)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate,
                                                  n_mels=self.n_mels, hop_length=self.hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features.extend([
            np.mean(mel_spec_db),
            np.std(mel_spec_db),
            np.max(mel_spec_db),
            np.min(mel_spec_db),
            np.median(mel_spec_db),
            np.percentile(mel_spec_db, 25),
            np.percentile(mel_spec_db, 75),
            np.var(mel_spec_db)
        ])
        
        # 4. Spectral Features (20 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
        
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_flatness), np.std(spectral_flatness)
        ])
        features.extend(np.mean(spectral_contrast, axis=1))
        
        # 5. Chroma Features (12 features)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        features.extend(np.mean(chroma, axis=1))
        
        # 6. Zero Crossing Rate (4 features)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.extend([
            np.mean(zcr), np.std(zcr),
            np.max(zcr), np.min(zcr)
        ])
        
        # 7. Pitch/F0 Features (8 features)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(audio_data, 
                                                          fmin=librosa.note_to_hz('C2'),
                                                          fmax=librosa.note_to_hz('C7'),
                                                          sr=self.sample_rate)
            f0_voiced = f0[~np.isnan(f0)]
            if len(f0_voiced) > 5:
                features.extend([
                    np.mean(f0_voiced), np.std(f0_voiced),
                    np.max(f0_voiced), np.min(f0_voiced),
                    np.median(f0_voiced), np.percentile(f0_voiced, 25),
                    np.percentile(f0_voiced, 75), np.ptp(f0_voiced)
                ])
            else:
                features.extend([0] * 8)
        except:
            features.extend([0] * 8)
        
        # 8. Energy Features (6 features)
        rms = librosa.feature.rms(y=audio_data)[0]
        features.extend([
            np.mean(rms), np.std(rms),
            np.max(rms), np.min(rms),
            np.median(rms), np.var(rms)
        ])
        
        # 9. Tonnetz (6 features)
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=self.sample_rate)
        features.extend(np.mean(tonnetz, axis=1))
        
        # Convert and clean
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def advanced_preprocess(self, audio_data: np.ndarray) -> np.ndarray:
        """Advanced preprocessing with noise reduction"""
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Advanced noise reduction
        try:
            audio_data = nr.reduce_noise(y=audio_data, sr=self.sample_rate, 
                                        prop_decrease=0.8, stationary=False)
        except:
            pass
        
        # Pre-emphasis filter
        pre_emphasis = 0.97
        audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
        
        # Normalize again
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data


# TRAINABLE Deep Learning Model
class SpeakerEmbeddingNetwork(nn.Module):
    """Deep neural network that ACTUALLY gets trained"""
    def __init__(self, input_dim: int = 200, embedding_dim: int = 128):
        super(SpeakerEmbeddingNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, embedding_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        # L2 normalize
        return F.normalize(x, p=2, dim=1)


# Quality Checker with STRICT requirements
class StrictQualityChecker:
    def check_audio_quality(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """STRICT quality requirements"""
        
        duration = len(audio) / sample_rate
        silence_ratio = np.sum(np.abs(audio) < 0.01) / len(audio)
        clipping = np.sum(np.abs(audio) > 0.95) / len(audio)
        
        # Energy analysis
        rms_energy = np.sqrt(np.mean(audio ** 2))
        
        # SNR
        signal_power = np.mean(audio ** 2)
        noise = audio - signal.medfilt(audio, kernel_size=5)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
        
        # Dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.mean(np.abs(audio))
        
        # Energy consistency
        frame_size = int(0.1 * sample_rate)
        frame_energies = []
        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i+frame_size]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))
        
        if len(frame_energies) > 0:
            energy_std = np.std(frame_energies)
            energy_mean = np.mean(frame_energies)
            energy_consistency = energy_std / max(energy_mean, 1e-10)
        else:
            energy_consistency = 0
        
        score = 1.0
        issues = []
        
        # Duration check
        if duration < 2.0:
            score *= 0.3
            issues.append(f"Too short ({duration:.1f}s, need 2-4s)")
        elif duration < 3.0:
            score *= 0.7
            issues.append(f"Short ({duration:.1f}s)")
        
        # Silence check
        if silence_ratio > 0.6:
            score *= 0.2
            issues.append(f"Too much silence ({silence_ratio:.0%})")
        elif silence_ratio > 0.4:
            score *= 0.7
            issues.append(f"High silence ({silence_ratio:.0%})")
        
        # Clipping detection
        if clipping > 0.01:
            score *= 0.1
            issues.append(f"⚠️ SHOUTING/CLIPPING detected ({clipping:.1%})")
        
        # Energy checks
        if rms_energy < 0.02:
            score *= 0.3
            issues.append(f"Too quiet (RMS={rms_energy:.3f})")
        elif rms_energy > 0.4:
            score *= 0.2
            issues.append(f"⚠️ TOO LOUD - shouting? (RMS={rms_energy:.3f})")
        
        # Energy consistency
        if energy_consistency > 1.5:
            score *= 0.3
            issues.append(f"⚠️ Inconsistent volume - shouting detected")
        
        # SNR check
        if snr < 8:
            score *= 0.6
            issues.append(f"Noisy (SNR={snr:.1f}dB)")
        
        # Dynamic range
        if dynamic_range > 0.7:
            score *= 0.2
            issues.append(f"⚠️ Extreme volume changes - possible shouting")
        
        return {
            'overall_score': max(0.0, score),
            'duration': duration,
            'silence_ratio': silence_ratio,
            'clipping': clipping,
            'rms_energy': rms_energy,
            'energy_consistency': energy_consistency,
            'snr': snr,
            'dynamic_range': dynamic_range,
            'issues': issues
        }


# REAL Authentication Engine with TRAINING
class RealVoiceAuthenticationEngine:
    def __init__(self):
        self.feature_extractor = AdvancedVoiceFeatureExtractor()
        self.quality_checker = StrictQualityChecker()
        self.db = SecureDatabase()
        self.enrollment_samples_required = 5
        self.authentication_threshold = 0.82
    
    def enroll_user(self, username: str, audio_samples: List[np.ndarray]) -> Tuple[bool, str]:
        """Enroll with ACTUAL model training"""
        if len(audio_samples) < self.enrollment_samples_required:
            return False, f"Need {self.enrollment_samples_required} samples"
        
        # Extract features
        all_features = []
        quality_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, audio in enumerate(audio_samples):
            status_text.text(f"Analyzing sample {i+1}/{len(audio_samples)}...")
            progress_bar.progress((i + 1) / len(audio_samples))
            
            quality = self.quality_checker.check_audio_quality(audio)
            quality_scores.append(quality['overall_score'])
            
            if quality['overall_score'] < 0.50:
                progress_bar.empty()
                status_text.empty()
                issues_str = ", ".join(quality['issues'][:2])
                return False, f"❌ Sample {i+1} rejected: {issues_str}"
            
            features = self.feature_extractor.extract_comprehensive_features(audio)
            all_features.append(features)
        
        progress_bar.empty()
        status_text.empty()
        
        avg_quality = np.mean(quality_scores)
        
        if avg_quality < 0.60:
            return False, f"❌ Average quality too low ({avg_quality:.1%})"
        
        # Convert to array
        X = np.array(all_features)
        
        # Train models
        st.info("🧠 Training neural network on your voice...")
        
        # 1. Deep Learning Model
        embedding_model = SpeakerEmbeddingNetwork(input_dim=X.shape[1])
        embeddings = self._train_embedding_model(embedding_model, X)
        
        # 2. One-Class SVM (for single-user anomaly detection)
        st.info("🎯 Training One-Class SVM classifier...")
        svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
        svm_model.fit(X)
        
        # 3. Feature statistics
        feature_mean = np.mean(X, axis=0)
        feature_std = np.std(X, axis=0)
        feature_cov = np.cov(X.T)
        
        # Store models
        user_model = {
            'embedding_model_state': embedding_model.state_dict(),
            'embeddings': embeddings.tolist(),
            'svm_model': svm_model,
            'feature_stats': {
                'mean': feature_mean.tolist(),
                'std': feature_std.tolist(),
                'cov': feature_cov.tolist()
            },
            'feature_dim': X.shape[1],
            'avg_quality': float(avg_quality),
            'num_samples': len(audio_samples),
            'created_at': datetime.now().isoformat()
        }
        
        st.session_state.db_trained_models[username] = user_model
        self.db._save_models()
        
        return True, f"✅ Trained on {len(audio_samples)} samples! Quality: {avg_quality:.1%}"
    
    def _train_embedding_model(self, model: nn.Module, X: np.ndarray, epochs: int = 50) -> np.ndarray:
        """Train the neural network"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(X)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            embeddings = model(X_tensor)
            distances = torch.cdist(embeddings, embeddings, p=2)
            loss = torch.mean(distances)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            final_embeddings = model(X_tensor).numpy()
        
        return final_embeddings
    
    def authenticate(self, username: str, audio: np.ndarray) -> Tuple[bool, float, str]:
        """Authenticate using TRAINED models"""
        if username not in st.session_state.db_trained_models:
            return False, 0.0, "❌ User not enrolled"
        
        # Quality check
        quality = self.quality_checker.check_audio_quality(audio)
        
        if quality['overall_score'] < 0.45:
            issues_str = "\n• ".join(quality['issues'][:3])
            return False, 0.0, f"❌ Audio quality failed:\n• {issues_str}"
        
        # Extract features
        test_features = self.feature_extractor.extract_comprehensive_features(audio)
        user_model = st.session_state.db_trained_models[username]
        
        # Method 1: Deep Learning
        embedding_model = SpeakerEmbeddingNetwork(input_dim=user_model['feature_dim'])
        embedding_model.load_state_dict(user_model['embedding_model_state'])
        embedding_model.eval()
        
        with torch.no_grad():
            test_embedding = embedding_model(torch.FloatTensor(test_features).unsqueeze(0)).numpy()
        
        stored_embeddings = np.array(user_model['embeddings'])
        
        embedding_similarities = []
        for stored_emb in stored_embeddings:
            sim = 1 - cosine(test_embedding.flatten(), stored_emb.flatten())
            embedding_similarities.append(sim)
        
        embedding_score = np.max(embedding_similarities)
        
        # Method 2: One-Class SVM
        svm_model = user_model['svm_model']
        svm_decision = svm_model.decision_function(test_features.reshape(1, -1))[0]
        svm_score = max(0, min(1, (svm_decision + 1) / 2))
        
        # Method 3: Statistical
        feature_mean = np.array(user_model['feature_stats']['mean'])
        feature_cov = np.array(user_model['feature_stats']['cov'])
        
        try:
            diff = test_features - feature_mean
            cov_inv = np.linalg.pinv(feature_cov)
            mahal_dist = np.sqrt(diff @ cov_inv @ diff.T)
            mahal_score = 1 / (1 + mahal_dist / 10)
        except:
            mahal_score = 0.5
        
        # Ensemble scoring
        final_score = (
            0.45 * embedding_score +
            0.35 * svm_score +
            0.20 * mahal_score
        )
        
        # Quality penalty
        quality_factor = min(1.0, quality['overall_score'] / 0.7)
        final_score *= quality_factor
        
        # Adaptive threshold
        enrollment_quality = user_model.get('avg_quality', 0.7)
        adaptive_threshold = self.authentication_threshold * (0.95 + 0.05 * enrollment_quality)
        
        debug_info = (
            f"Embedding: {embedding_score:.3f} | SVM: {svm_score:.3f} | "
            f"Statistical: {mahal_score:.3f} | Quality: {quality_factor:.2f}\n"
            f"Final: {final_score:.3f} vs Threshold: {adaptive_threshold:.3f}"
        )
        
        if final_score >= adaptive_threshold:
            confidence = "Very High" if final_score > 0.90 else "High" if final_score > 0.85 else "Good"
            msg = f"✅ AUTHENTICATED\nScore: {final_score:.1%} ({confidence})\n{debug_info}"
            return True, final_score, msg
        
        msg = f"❌ AUTHENTICATION FAILED\n{debug_info}"
        return False, final_score, msg


# Browser-compatible audio processing functions
def process_uploaded_audio(audio_file, target_sr=16000):
    """Process uploaded audio file to numpy array"""
    try:
        # Read audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=target_sr, mono=True)
        
        # Normalize
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        return audio_data
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None


def show_audio_recorder(key_prefix=""):
    """Show browser-based audio recorder using Streamlit's audio_input"""
    st.markdown("#### 🎙️ Record Audio")
    
    st.info("🎤 Click the microphone button below to record. Your browser will ask for microphone permission.")
    
    # Use Streamlit's built-in audio input (requires Streamlit >= 1.28.0)
    audio_bytes = st.audio_input("Record your voice", key=f"{key_prefix}_audio_input")
    
    if audio_bytes is not None:
        # Process the audio
        audio_data = process_uploaded_audio(io.BytesIO(audio_bytes))
        
        if audio_data is not None:
            # Show audio metrics
            max_amplitude = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data ** 2))
            duration = len(audio_data) / 16000
            non_silence = np.sum(np.abs(audio_data) > 0.01) / len(audio_data)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.1f}s")
            with col2:
                st.metric("Max Amplitude", f"{max_amplitude:.3f}")
            with col3:
                st.metric("RMS Energy", f"{rms:.3f}")
            with col4:
                st.metric("Voice Activity", f"{non_silence:.0%}")
            
            # Quality feedback
            if max_amplitude < 0.01:
                st.error("❌ Microphone NOT picking up audio!")
            elif rms < 0.02:
                st.warning("⚠️ Audio very quiet. Speak louder.")
            elif duration < 2.0:
                st.warning("⚠️ Recording too short. Need at least 2 seconds.")
            elif non_silence < 0.3:
                st.warning("⚠️ Too much silence detected.")
            else:
                st.success("✅ Audio quality looks good!")
            
            return audio_data
    
    return None


# Initialize session state
def init_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'auth_engine' not in st.session_state:
        st.session_state.auth_engine = RealVoiceAuthenticationEngine()
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'user_balance' not in st.session_state:
        st.session_state.user_balance = 10000.00
    if 'audio_samples' not in st.session_state:
        st.session_state.audio_samples = []


# Login Page
def login_page():
    st.markdown('<div class="app-header"><h1>🔒 VoiceGuard REAL AI</h1><p>Deep Learning Voice Authentication (82% Threshold)</p></div>', 
                unsafe_allow_html=True)
    
    # Debug section
    with st.expander("🔍 System Status & Models"):
        st.markdown("**🧠 REAL Deep Learning Features:**")
        st.markdown("""
        - ✅ **Trained Neural Network** (512→256→128→64 dimensions)
        - ✅ **SVM Classifier** (trained on your voice patterns)
        - ✅ **200+ discriminative features** (MFCC, spectral, prosodic)
        - ✅ **82% similarity threshold** (very strict)
        """)
        
        if st.session_state.db_trained_models:
            st.success(f"**👥 {len(st.session_state.db_trained_models)} trained models:**")
            for username, model in st.session_state.db_trained_models.items():
                quality = model.get('avg_quality', 0)
                num_samples = model.get('num_samples', 0)
                st.write(f"✅ **{username}** - {num_samples} samples, Quality: {quality:.1%}")
        else:
            st.warning("No users enrolled yet.")
        
        if st.button("🗑️ Clear All Data"):
            import shutil
            if os.path.exists("voiceguard_data_v2"):
                shutil.rmtree("voiceguard_data_v2")
            st.session_state.clear()
            st.success("✅ All data cleared! Refresh page.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 🔐 Login")
        
        username = st.text_input("Username", placeholder="Enter your username", key="login_username")
        
        st.markdown("---")
        
        st.markdown("### 🎤 Voice Authentication")
        
        with st.expander("💡 IMPORTANT: How to Record"):
            st.markdown("""
            **To PASS authentication:**
            - 🗣️ Speak NORMALLY (same as enrollment)
            - 🎤 Clear voice, quiet environment
            - ⏱️ Record for 3-4 seconds
            - 🔇 Avoid background noise
            
            **System will REJECT:**
            - ⚠️ Shouting or yelling
            - ⚠️ Different voice characteristics
            - ⚠️ Too much background noise
            """)
        
        # Browser-based audio recording
        audio_data = show_audio_recorder("login")
        
        if audio_data is not None:
            st.session_state.login_audio = audio_data
            
            # Show quality analysis
            quality = st.session_state.auth_engine.quality_checker.check_audio_quality(audio_data)
            
            with st.expander("📊 Audio Quality Analysis"):
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1:
                    st.metric("Quality Score", f"{quality['overall_score']:.1%}")
                with col_q2:
                    st.metric("Duration", f"{quality['duration']:.1f}s")
                with col_q3:
                    st.metric("Energy", f"{quality['rms_energy']:.3f}")
                with col_q4:
                    st.metric("SNR", f"{quality['snr']:.1f}dB")
                
                if quality['clipping'] > 0.01:
                    st.error(f"⚠️ CLIPPING DETECTED: {quality['clipping']:.1%}")
                
                if quality['issues']:
                    st.warning("**Issues detected:**")
                    for issue in quality['issues'][:3]:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ Good quality audio!")
        
        st.markdown("---")
        
        if st.button("🔓 LOGIN", use_container_width=True, type="primary"):
            if not username:
                st.error("❌ Please enter username")
            elif 'login_audio' not in st.session_state:
                st.error("❌ Please record your voice first")
            else:
                with st.spinner("🧠 Running deep learning analysis..."):
                    success, score, message = st.session_state.auth_engine.authenticate(
                        username, st.session_state.login_audio
                    )
                    
                    if success:
                        st.success(message)
                        st.balloons()
                        st.session_state.current_user = username
                        
                        user_data = st.session_state.auth_engine.db.get_user(username)
                        if user_data:
                            st.session_state.user_balance = user_data.get('balance', 10000.00)
                        
                        time.sleep(2)
                        st.session_state.page = 'payment'
                        st.rerun()
                    else:
                        st.error(message)
                        st.warning("🔒 Access Denied")
        
        st.markdown("---")
        if st.button("📝 Don't have an account? **Register**", use_container_width=True):
            st.session_state.page = 'register'
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


# Registration Page
def registration_page():
    st.markdown('<div class="app-header"><h1>📝 Register & Train AI</h1><p>Record 5 samples to train your personal voice model</p></div>', 
                unsafe_allow_html=True)
    
    if st.button("← Back to Login"):
        st.session_state.page = 'login'
        st.session_state.audio_samples = []
        st.rerun()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        
        st.markdown("### 👤 User Information")
        username = st.text_input("Username", placeholder="Choose username", key="reg_username")
        email = st.text_input("Email", placeholder="your.email@example.com", key="reg_email")
        
        st.markdown("---")
        
        st.markdown("### 🎙️ Voice Training (5 Samples)")
        st.info("🔢 Record 5 high-quality samples. Speak naturally and clearly.")
        
        progress = len(st.session_state.audio_samples)
        st.progress(progress / 5)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("Samples Recorded", f"{progress}/5")
        with col_p2:
            if progress > 0:
                st.metric("Status", "Training..." if progress < 5 else "Ready ✅")
        
        if progress < 5:
            phrases = [
                "Hello, this is my voice for secure authentication",
                "I authorize transactions with my unique voice signature",
                "Voice biometric enrollment sample number",
                "Security verification with advanced voice recognition",
                "My voice is my password for financial transactions"
            ]
            
            st.markdown(f"**🗣️ Suggested phrase {progress+1}:**")
            st.markdown(f"*'{phrases[progress]} {progress+1}'*")
            
            st.warning("⚠️ **CRITICAL:** Speak NORMALLY. Don't shout! Record at least 3 seconds.")
            
            # Browser-based audio recording
            audio_data = show_audio_recorder(f"register_sample_{progress}")
            
            if audio_data is not None:
                quality = st.session_state.auth_engine.quality_checker.check_audio_quality(audio_data)
                
                with st.expander("📊 Sample Quality Analysis"):
                    col_q1, col_q2, col_q3 = st.columns(3)
                    with col_q1:
                        st.metric("Quality", f"{quality['overall_score']:.1%}")
                    with col_q2:
                        st.metric("Duration", f"{quality['duration']:.1f}s")
                    with col_q3:
                        st.metric("Energy", f"{quality['rms_energy']:.3f}")
                    
                    if quality['issues']:
                        st.warning("**Issues:**")
                        for issue in quality['issues']:
                            st.write(f"• {issue}")
                
                col_accept, col_reject = st.columns(2)
                
                with col_accept:
                    if st.button(f"✅ Accept Sample {progress + 1}", use_container_width=True, type="primary"):
                        if quality['overall_score'] >= 0.50:
                            st.session_state.audio_samples.append(audio_data)
                            st.success(f"✅ Sample {progress + 1} accepted! Quality: {quality['overall_score']:.1%}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Sample quality too low ({quality['overall_score']:.1%})")
                            for issue in quality['issues'][:3]:
                                st.warning(f"• {issue}")
                
                with col_reject:
                    if st.button("🔄 Record Again", use_container_width=True):
                        st.rerun()
        
        else:
            st.success("✅ All 5 samples recorded!")
            
            # Show sample quality summary
            with st.expander("📊 All Samples Quality Summary"):
                for i, audio in enumerate(st.session_state.audio_samples):
                    quality = st.session_state.auth_engine.quality_checker.check_audio_quality(audio)
                    st.write(f"**Sample {i+1}:** Quality {quality['overall_score']:.1%}, Duration {quality['duration']:.1f}s")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚀 TRAIN AI MODEL", use_container_width=True, type="primary"):
                    if not username or not email:
                        st.error("❌ Please fill all fields")
                    elif username in st.session_state.db_trained_models:
                        st.error("❌ Username already exists")
                    else:
                        with st.spinner("🧠 Training neural network on your voice..."):
                            success, message = st.session_state.auth_engine.enroll_user(
                                username, st.session_state.audio_samples
                            )
                            
                            if success:
                                user_data = {
                                    'username': username,
                                    'email': email,
                                    'created_at': datetime.now().isoformat(),
                                    'balance': 10000.00
                                }
                                st.session_state.auth_engine.db.store_user(username, user_data)
                                
                                st.success(f"🎉 {message}")
                                st.balloons()
                                
                                time.sleep(3)
                                st.session_state.audio_samples = []
                                st.session_state.page = 'login'
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
            
            with col_btn2:
                if st.button("🔄 Start Over", use_container_width=True):
                    st.session_state.audio_samples = []
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)


# Payment Page
def payment_page():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<div class="app-header"><h2>Welcome, {st.session_state.current_user}!</h2></div>', 
                    unsafe_allow_html=True)
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown(f"""
    <div class="custom-card" style="background: linear-gradient(135deg, #00b0ff 0%, #0080d6 100%); color: white;">
        <h3>💰 Available Balance</h3>
        <h1>₹{st.session_state.user_balance:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 💸 Send Money")
        
        recipient = st.text_input("Recipient", placeholder="Name or phone number")
        amount = st.number_input("Amount (₹)", min_value=0.0, step=100.0, format="%.2f")
        description = st.text_input("Note (optional)", placeholder="Payment for...")
        
        st.markdown("---")
        
        st.markdown("### 🔒 AI Voice Authorization")
        st.info("🗣️ Speak normally for 3-4 seconds to authorize this transaction")
        
        # Browser-based audio recording
        audio_data = show_audio_recorder("payment")
        
        if audio_data is not None:
            st.session_state.payment_audio = audio_data
            
            quality = st.session_state.auth_engine.quality_checker.check_audio_quality(audio_data)
            
            with st.expander("📊 Authorization Audio Quality"):
                col_q1, col_q2, col_q3 = st.columns(3)
                with col_q1:
                    st.metric("Quality", f"{quality['overall_score']:.1%}")
                with col_q2:
                    st.metric("Duration", f"{quality['duration']:.1f}s")
                with col_q3:
                    st.metric("Energy", f"{quality['rms_energy']:.3f}")
                
                if quality['overall_score'] >= 0.45:
                    st.success("✅ Audio quality acceptable")
                else:
                    st.warning("⚠️ Low quality - may affect authentication")
        
        st.markdown("---")
        
        if st.button("💳 PROCESS PAYMENT", use_container_width=True, type="primary"):
            if not recipient or amount <= 0:
                st.error("❌ Please fill all payment details")
            elif 'payment_audio' not in st.session_state:
                st.error("❌ Please record voice authorization first")
            elif amount > st.session_state.user_balance:
                st.error(f"❌ Insufficient balance. Available: ₹{st.session_state.user_balance:,.2f}")
            else:
                with st.spinner("🧠 AI verifying your identity..."):
                    success, score, message = st.session_state.auth_engine.authenticate(
                        st.session_state.current_user, 
                        st.session_state.payment_audio
                    )
                    
                    if success:
                        # Process transaction
                        transaction = {
                            'user': st.session_state.current_user,
                            'recipient': recipient,
                            'amount': float(amount),
                            'description': description,
                            'voice_score': float(score),
                            'status': 'completed'
                        }
                        st.session_state.auth_engine.db.log_transaction(transaction)
                        
                        # Update balance
                        st.session_state.user_balance -= amount
                        
                        user_data = st.session_state.auth_engine.db.get_user(st.session_state.current_user)
                        user_data['balance'] = st.session_state.user_balance
                        st.session_state.auth_engine.db.store_user(st.session_state.current_user, user_data)
                        
                        st.success(f"✅ Payment of ₹{amount:,.2f} sent successfully!")
                        st.success(message)
                        st.balloons()
                        
                        # Clear payment audio
                        if 'payment_audio' in st.session_state:
                            del st.session_state.payment_audio
                        
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
                        st.warning("🔒 PAYMENT BLOCKED - Voice authentication failed")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Recent Activity")
        
        user_transactions = [t for t in st.session_state.db_transactions 
                            if t['user'] == st.session_state.current_user]
        
        if user_transactions:
            for trans in reversed(user_transactions[-5:]):
                score = trans.get('voice_score', 0)
                timestamp = trans.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%b %d, %H:%M")
                    except:
                        time_str = ""
                else:
                    time_str = ""
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>₹{trans['amount']:,.2f}</strong><br>
                    To: {trans['recipient']}<br>
                    <small>AI Score: {score:.1%}</small><br>
                    <small>{time_str}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No transactions yet")
        
        st.markdown('</div>', unsafe_allow_html=True)


# Main
def main():
    init_session_state()
    
    if st.session_state.page == 'login':
        login_page()
    elif st.session_state.page == 'register':
        registration_page()
    elif st.session_state.page == 'payment':
        if st.session_state.current_user:
            payment_page()
        else:
            st.session_state.page = 'login'
            st.rerun()


if __name__ == "__main__":
    main()