# 1. Usar uma imagem base com Python 3.9
FROM python:3.9-slim

# 2. Definir o diretório de trabalho no contêiner
WORKDIR /app

# 3. Copiar os arquivos da aplicação para o contêiner
COPY . /app

# 4. Instalar dependências do sistema
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean

# 5. Instalar dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Expor a porta que a aplicação usará
EXPOSE 5000

# 7. Comando para rodar a aplicação
CMD ["python", "app.py"]



