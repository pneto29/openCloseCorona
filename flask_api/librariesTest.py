import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from sklearn import __version__ as sklearn_version
from flask import __version__ as flask_version
from PIL import Image
import requests
import os

# Coletar versões das bibliotecas
versions = {
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "tensorflow": tf.__version__,
    "scikit-learn": sklearn_version,
    "flask": flask_version,
    "Pillow": Image.__version__,
    "requests": requests.__version__
}

# Exibir versões no terminal
print("Versões das bibliotecas instaladas:")
for lib, version in versions.items():
    print(f"{lib}: {version}")

# Criar arquivo requirements.txt
requirements_content = "\n".join([f"{lib}=={version}" for lib, version in versions.items()])

with open("requirements.txt", "w") as f:
    f.write(requirements_content)

print("\nArquivo 'requirements.txt' criado com sucesso!")
print("Conteúdo do arquivo:")
print(requirements_content)
