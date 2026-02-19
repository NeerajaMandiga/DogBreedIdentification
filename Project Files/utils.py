import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'dogbreed.h5')
CLASS_JSON = os.path.join(os.path.dirname(__file__), 'class_names.json')


def load_trained_model():
    """Load model and class mapping from disk. Returns (model, class_map) or (None, None) if not found."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    class_map = None
    if os.path.exists(CLASS_JSON):
        with open(CLASS_JSON, 'r') as f:
            class_map = json.load(f)
            # ensure keys are ints
            class_map = {int(k): v for k, v in class_map.items()}
    return model, class_map


def predict_image(img_path, model, class_map, img_size=(128, 128)):
    """Load image, preprocess, predict and return (label, confidence).

    label: string class name
    confidence: float between 0 and 1
    """
    if model is None or class_map is None:
        raise RuntimeError('Model or class map not loaded')

    img = image.load_img(img_path, target_size=img_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx])
    label = class_map.get(idx, str(idx))
    return label, conf
