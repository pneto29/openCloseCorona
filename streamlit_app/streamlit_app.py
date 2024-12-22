import streamlit as st
import requests
from PIL import Image
import io

# Configuração da interface
st.title("Classificador de Garrafas - Interface com a API Flask")

st.write("Envie uma ou mais imagens de garrafas para verificar o status (aberta ou fechada).")

# Upload de imagens
uploaded_files = st.file_uploader("Carregue suas imagens aqui", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("Imagens carregadas:")
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

# Botão para enviar as imagens para a API
if st.button("Classificar"):
    st.write("Enviando imagens para a API...")
    results = []

    try:
        # Prepara as imagens para envio
        files = [('files', (file.name, file.getvalue(), "image/jpeg")) for file in uploaded_files]
        response = requests.post("http://127.0.0.1:5001/classify", files=files)

        if response.status_code == 200:
            results = response.json()
            st.success("Classificação concluída!")
        else:
            st.error(f"Erro: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Erro ao se conectar à API: {e}")

    # Mostra os resultados
    if results:
        for result in results:
            st.write(f"**Arquivo:** {result.get('filename')}")
            if 'error' in result:
                st.error(f"Erro: {result['error']}")
            else:
                st.write(f"- **Status da Garrafa:** {result['Bottle_Status']}")
                st.write(f"- **Confiança:** {result['Confidence']:.4f}")
