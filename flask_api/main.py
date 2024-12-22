import os
import pandas as pd

# Definir os caminhos das pastas
open_folder = 'aberto'
closed_folder = 'fechado'

# Obter os nomes das imagens em cada pasta
open_images = [(img, 'Open') for img in os.listdir(open_folder) if img.endswith(('png', 'jpg', 'jpeg'))]
closed_images = [(img, 'Closed') for img in os.listdir(closed_folder) if img.endswith(('png', 'jpg', 'jpeg'))]

# Combinar os dados de ambas as pastas
data = open_images + closed_images

# Criar um DataFrame
df = pd.DataFrame(data, columns=['Image Name', 'Status'])

# Salvar como CSV
df.to_csv('image_status.csv', index=False)

print("CSV criado com sucesso!")
