import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configurações
image_folder_open = "aberto"  # Caminho para a pasta com imagens abertas
image_folder_closed = "fechado"  # Caminho para a pasta com imagens fechadas
csv_path = "image_status.csv"  # Caminho para o arquivo CSV
output_model_path = "bottle_classifier_model_resnet50.h5"  # Caminho para salvar o modelo

# Configurar para uso da GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU configurada para uso.")
    except RuntimeError as e:
        print(f"Erro ao configurar a GPU: {e}")
else:
    print("Nenhuma GPU disponível. Rodando na CPU.")

# Carregar CSV
image_status_data = pd.read_csv(csv_path)

# Filtrar imagens e status
image_status_data = image_status_data[image_status_data['Status'].isin(['Open', 'Closed'])]

# Selecionar 2000 amostras de cada classe para balancear os dados
open_images = image_status_data[image_status_data['Status'] == 'Open'].sample(2000, random_state=42)
closed_images = image_status_data[image_status_data['Status'] == 'Closed'].sample(2000, random_state=42)

# Concatenar e embaralhar as amostras
balanced_data = pd.concat([open_images, closed_images]).sample(frac=1, random_state=42).reset_index(drop=True)

# Função para carregar imagens e rótulos
def load_images_and_labels(data, image_folder_open, image_folder_closed):
    """
    Carrega as imagens e seus rótulos (0 para 'Open', 1 para 'Closed') com base na coluna 'Status' do CSV.
    """
    images = []
    labels = []
    for _, row in data.iterrows():
        img_path = os.path.join(
            image_folder_open if row['Status'] == 'Open' else image_folder_closed, row['Image Name']
        )
        if os.path.exists(img_path):
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(1 if row['Status'] == 'Closed' else 0)
    return np.array(images), np.array(labels)

# Carregar imagens e rótulos
X, y = load_images_and_labels(balanced_data, image_folder_open, image_folder_closed)

# Dividir os dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar data augmentation no conjunto de treino
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Carregar o modelo ResNet50 pré-treinado
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas inferiores e descongelar as últimas camadas
for layer in base_model.layers[:-10]:  # Congelar todas menos as 10 últimas
    layer.trainable = False

# Adicionar camadas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Saída binária para "Open" (0) ou "Closed" (1)

# Criar o modelo completo
model = Model(inputs=base_model.input, outputs=output)

# Compilar o modelo com um otimizador ajustado
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks para ajustar a taxa de aprendizado dinamicamente
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Treinar o modelo
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_val, y_val),
    epochs=100,  # 50 épocas
    callbacks=[lr_scheduler]
)

# Salvar o modelo treinado
model.save(output_model_path)

# Função para classificar novas imagens
def classify_bottle(image_path, model_path):
    """
    Classifica uma garrafa como 'Open' ou 'Closed' com base no modelo treinado.
    """
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão batch
    prediction = model.predict(img_array)
    status = "Closed" if prediction[0][0] > 0.5 else "Open"
    confidence = prediction[0][0] if status == "Closed" else 1 - prediction[0][0]
    return status, confidence

# Exemplo de uso:
# status, confidence = classify_bottle("path_to_test_image.jpg", output_model_path)
# print(f"Status: {status}, Confiança: {confidence:.2f}")
