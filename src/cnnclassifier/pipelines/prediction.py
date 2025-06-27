import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path
from src.cnnclassifier.utils.common import load_json
from tensorflow.keras.applications.vgg16 import preprocess_input

CLASS_NAMES = {v: k.split('_')[0] for k, v in load_json("classification_order.json").items()}

class PredictionPipeline:
    def __init__(self,filename:str) -> None:
        self.filename = filename
        self.model = load_model(os.path.join("artifacts", "training", "model.keras"))
    
    def _prepare_image(self) -> np.ndarray:
        """Prepares a (1, 224, 224, 3) tensor preprocessed with VGG16 strategy"""
        img = image.load_img(self.filename, target_size=(224, 224))   
        x = image.img_to_array(img)                                   
        x = np.expand_dims(x, axis=0)                                 
        x = preprocess_input(x)                                       
        return x

    def predict(self) -> dict:
        x = self._prepare_image()
        preds = self.model.predict(x)                                 
        idx = int(np.argmax(preds[0]))
        label = CLASS_NAMES.get(idx, f"class_{idx}")
        return {
            "class": label,
            "probs": preds[0].tolist()
        }
        

