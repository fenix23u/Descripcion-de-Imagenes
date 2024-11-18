import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import torch

# Verificar si PyTorch está instalado correctamente
if not torch.cuda.is_available():
    st.warning("Cargando la Aplicacion, espere porfavor.")
else:
    st.success("Se detectó GPU, la aplicación usará la aceleración por hardware.")

# Cargar el modelo preentrenado BLIP (versión grande)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Cargar el modelo de traducción de inglés a español
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")

# Configuración de la interfaz de usuario con Streamlit
st.title("Descripcion de imagenes")
st.write("Sube una imagen y el modelo te dará una descripción detallada de lo que contiene en español.")

# Subir imagen
uploaded_image = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Abrir y mostrar la imagen
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Preprocesar la imagen para el modelo BLIP
    inputs = processor(images=image, return_tensors="pt")

    # Generar la descripción de la imagen con mayor longitud
    out = model.generate(**inputs, max_length=100)  # Ajustar max_length según sea necesario
    description = processor.decode(out[0], skip_special_tokens=True)

    # Traducir la descripción del inglés al español
    translated = translation_model.generate(**translation_tokenizer(description, return_tensors="pt"))
    spanish_description = translation_tokenizer.decode(translated[0], skip_special_tokens=True)

    # Mostrar la descripción en español
    st.write("Descripción detallada de la imagen:")
    st.write(spanish_description)
