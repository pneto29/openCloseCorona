from PIL import Image
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = "bottle_classifier_model_resnet50.h5"
IMAGE_PATH = "/home/polycarpo/Documentos/OpenOrClose/dataset_openclose/1715279783513.jpg"

# Carregar modelo
model = tf.keras.models.load_model(MODEL_PATH)

# Pré-processar imagem
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Ajustar tamanho
    image = np.array(image) / 255.0  # Normalizar
    image = np.expand_dims(image, axis=0)  # Adicionar batch
    return image

# Classificar imagem
image = preprocess_image(IMAGE_PATH)
predictions = model.predict(image)
confidence = float(np.max(predictions))
label = "open" if np.argmax(predictions) == 1 else "closed"

print(f"Label: {label}, Confidence: {confidence}")
