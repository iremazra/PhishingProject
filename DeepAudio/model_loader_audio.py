#model_loader_audio
import os
import joblib
import numpy as np
import librosa
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    def __init__(
        self,
        model_path: str = r'Deepaudio\xgb_voice_model.pkl',
        scaler_path: str = r'Deepaudio\scaler.pkl' ):
    
        self.model_path = model_path
        self.scaler_path=scaler_path
        self.model = None
        self.scaler=None
        self.load_model()
        self.load_scaler()

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            logger.info("✅ Audio model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load audio model: {str(e)}")
            raise

    def load_scaler(self):
        #Load the scaler for feature normalization
        if self.scaler_path and os.path.exists(self.scaler_path):
            try:
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✅ Scaler loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load scaler: {str(e)}")
                raise
        else:
            logger.warning("⚠️ Scaler path not provided or file does not exist.")


    def extract_features(self,file_path: str) -> np.ndarray:
        """
        Extract exactly 47 features:
        - 20 MFCC means
        - 20 MFCC stds
        - 1 Chroma mean
        - 1 Chroma std
        - 1 Spectral contrast mean
        - 1 Spectral contrast std
        - 1 ZCR mean
        - 1 ZCR std
        - 1 RMS mean
        """
        try:
            y, sr = librosa.load(file_path, sr=16000)
            feats = []
    
            # 1) MFCC mean + std (20 each)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            feats.extend(np.mean(mfcc, axis=1))
            feats.extend(np.std(mfcc, axis=1))
    
            # 2) Chroma mean + std
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            feats.append(np.mean(chroma))
            feats.append(np.std(chroma))
    
            # 3) Spectral contrast mean + std
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            feats.append(np.mean(spec_contrast))
            feats.append(np.std(spec_contrast))
    
            # 4) Zero Crossing Rate mean + std
            zcr = librosa.feature.zero_crossing_rate(y=y)
            feats.append(np.mean(zcr))
            feats.append(np.std(zcr))
    
            # 5) RMS energy mean
            rms = librosa.feature.rms(y=y)
            feats.append(np.mean(rms))
    
            features = np.array(feats, dtype=np.float32)
            if features.shape[0] != 47:
                raise ValueError(f"Expected 47 features, got {features.shape[0]}")
            return features

        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            return np.zeros(47, dtype=np.float32)

    def predict(self, file_path: str) -> dict:
        #Make prediction on audio file
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        features = self.extract_features(file_path)
        if features is None:
            return {"error": "Feature extraction failed"}

        try:
            features = features.reshape(1, -1)

            if self.scaler:
                features=self.scaler.transform(features)
            prediction = self.model.predict(features)[0]
            result = {
                "prediction": "Real" if prediction == 0 else "Fake",
                "confidence": None
            }
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                result["confidence"] = float(proba[prediction])
                
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}