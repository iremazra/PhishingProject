
# DeepImage/model_loader_img.py
import tensorflow as tf
import logging

import numpy as np
from pathlib import Path
import os
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    def __init__(self):
        try:
            base_dir = os.path.dirname(__file__)
            model_path = os.path.join(base_dir, 'deepfake_model.keras')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Deepfake modeli yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"❌ Model yüklenemedi: {e}")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            return {"Prediction": "error", "Error": "Model yüklenemedi"}

        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

            img = Image.open(image_path).convert('RGB')
            img = img.resize((150, 150))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = self.model.predict(img)

            if prediction.shape[-1] == 1:
                score = float(prediction[0][0])  # Skor sayıya dönüştürülür
                
                label = "Deepfake" if score < 0.5 else "Real"
                possibilities = {"Real": score, "Deepfake": 1 - score}#hata?
                return {"Prediction": label, "Score": score, "Possibilities": possibilities}
            else:
                return {"Prediction": "Unknown", "Score": 0.0, "Error": "Beklenmeyen çıktı boyutu"}

        except Exception as e:
            logger.error(f"❌ Tahmin hatası: {e}")
            return {"Prediction": "error", "Error": str(e)}

