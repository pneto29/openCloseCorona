from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf  # Para carregar o modelo .h5
import io
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import glob
# Configuração de Logs
logging.basicConfig(level=logging.INFO)

# Inicializa o Flask
app = Flask(__name__)

# Carrega o modelo .h5
MODEL_PATH = "bottle_classifier_model_resnet50.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Modelo carregado com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar o modelo: {e}")
    model = None

# Pré-processamento da imagem
def preprocess_image(image):
    """Pré-processa a imagem para o modelo Keras."""
    try:
        target_size = (224, 224)  # Ajuste conforme a entrada do modelo
        image = image.resize(target_size)  # Redimensiona
        logging.info("Imagem redimensionada com sucesso.")
        image = np.array(image) / 255.0  # Normaliza para [0, 1]
        logging.info("Imagem normalizada com sucesso.")
        image = np.expand_dims(image, axis=0)  # Adiciona a dimensão do batch
        logging.info("Pré-processamento concluído com sucesso.")
        return image
    except Exception as e:
        logging.error(f"Erro no pré-processamento da imagem: {e}")
        raise e

# Classificação da imagem
def classify_image(image):
    """Classifica a imagem e retorna a label e o score de confiança."""
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        logging.info("Predição realizada com sucesso.")
        confidence = float(np.max(predictions))  # Confiança máxima
        label_index = int(np.argmax(predictions))  # Índice da label
        label = "open" if label_index == 1 else "closed"
        return label, confidence
    except Exception as e:
        logging.error(f"Erro na classificação da imagem: {e}")
        raise e

# Endpoint da API
@app.route('/classify', methods=['POST'])
def classify():
    """Endpoint que recebe múltiplas imagens e retorna as classificações."""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({"error": "No files provided"}), 400

    results = []
    for file in files:
        try:
            # Lê e converte a imagem
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            logging.info("Imagem carregada com sucesso.")
            label, confidence = classify_image(image)

            # Adiciona o resultado à lista
            results.append({
                "filename": file.filename,
                "Bottle_Status": label,
                "Confidence": round(confidence, 4)
            })
        except Exception as e:
            logging.error(f"Erro ao processar a imagem {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": f"Unable to process image: {e}"
            })

    # Retorna a lista de resultados
    return jsonify(results), 200

def classify_images_in_directory(directory_path):
    """Classifica todas as imagens em um diretório."""
    results = []
    try:
        image_paths = glob.glob(os.path.join(directory_path, "*.jpg"))  # Ajuste para outros formatos, se necessário
        for image_path in image_paths:
            try:
                with open(image_path, "rb") as image_file:
                    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
                    label, confidence = classify_image(image)

                    results.append({
                        "filename": os.path.basename(image_path),
                        "Bottle_Status": label,
                        "Confidence": round(confidence, 4)
                    })
            except Exception as e:
                logging.error(f"Erro ao processar a imagem {image_path}: {e}")
                results.append({
                    "filename": os.path.basename(image_path),
                    "error": f"Unable to process image: {e}"
                })
        logging.info("Classificação concluída para todas as imagens no diretório.")
    except Exception as e:
        logging.error(f"Erro ao processar o diretório: {e}")
        results.append({"error": f"Erro ao processar o diretório: {e}"})

    return results

if __name__ == '__main__':
    import sys
    if 'test' in sys.argv:
        directory_path = "sample/"  # Substitua pelo caminho do diretório contendo suas imagens
        results = classify_images_in_directory(directory_path)
        print(results)
    else:
        app.run(debug=True, port=5001)
