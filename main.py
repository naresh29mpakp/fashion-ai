import streamlit as st
import numpy as np
import PIL.Image
import io
import requests
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

# Load API key
load_dotenv()
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

def open_image(img_path):
    """Opens an image from a local file or URL."""
    try:
        if img_path.startswith("http"):  # URL-based image
            response = requests.get(img_path, stream=True)
            img = PIL.Image.open(io.BytesIO(response.content))
        elif os.path.exists(img_path):  # Local file-based image
            img = PIL.Image.open(img_path)
        else:
            raise ValueError(f"Invalid image path: {img_path}")
        return img
    except Exception as e:
        st.error(f"Error loading image: {img_path} - {e}")
        return None

st.title("👗 AI Fashion Styling Assistant")
st.write("Answer a few quick questions, upload an image, or enter a query to get personalized fashion recommendations!")

# Personalization Questions
st.subheader("🎨 Personalize Your Style")
style_vibe = st.selectbox("What is your usual outfit vibe?", ["Trendy & Modern", "Casual & Comfortable", "Classic & Elegant", "Edgy & Bold"])
occasion = st.selectbox("What’s the occasion for this outfit?", ["Everyday Wear", "Work/Business", "Party/Night Out", "Date Night"])
color_palette = st.selectbox("What is your go-to color palette?", ["Neutral (Black, White, Beige, Gray)", "Soft & Pastel (Pink, Lavender, Light Blue)", "Bright & Playful (Red, Yellow, Green)", "Dark & Mysterious (Navy, Burgundy, Deep Green)"])

# Additional Personalization Questions
st.subheader("✨ Refine Your Look")
preferred_fit = st.selectbox("What type of fit do you prefer?", ["Slim Fit", "Regular Fit", "Oversized", "Tailored"])
fabric_preference = st.selectbox("What fabric do you feel most comfortable in?", ["Cotton", "Denim", "Silk", "Leather", "Linen"])
accessory_preference = st.selectbox("Do you like to accessorize with?", ["Minimal Jewelry", "Bold Statement Pieces", "Scarves & Hats", "No Accessories"])

uploaded_file = st.file_uploader("📸 Upload an image to retrieve similar fashion items:", type=["jpg", "jpeg", "png"])
query = st.text_input("Or, enter your styling query:")

if st.button("🔍 Generate Styling Ideas / Retrieve Images"):
    chroma_client = chromadb.PersistentClient(path="Vector_database")
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)
    
    retrieved_images = []
    
    if uploaded_file is not None:
        uploaded_image = np.array(PIL.Image.open(uploaded_file))
        retrieved_imgs = image_vdb.query(query_images=[uploaded_image], include=['uris'], n_results=10)
        image_paths = retrieved_imgs.get('uris', [])[0] if 'uris' in retrieved_imgs else []
        
        st.subheader("📷 Retrieved Similar Images:")
        for i, img_path in enumerate(image_paths[:10]):
            img = open_image(img_path)
            if img:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
                retrieved_images.append(img)
    
    if query:
        query_results = image_vdb.query(query_texts=[query], n_results=10, include=['uris'])
        query_image_paths = query_results.get('uris', [])[0] if 'uris' in query_results else []
        
        st.subheader("📷 Recommended Images Based on Your Query:")
        for i, img_path in enumerate(query_image_paths[:10]):
            img = open_image(img_path)
            if img:
                st.image(img, caption=f"Image {i+1}", use_column_width=True)
                retrieved_images.append(img)
    
    # Generate Fashion Recommendations
    if retrieved_images:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        prompt = (f"You are a fashion stylist. The user is looking for outfit ideas. Their preferences are: \n"
                  f"- **Style Vibe:** {style_vibe}\n"
                  f"- **Occasion:** {occasion}\n"
                  f"- **Color Palette:** {color_palette}\n"
                  f"- **Preferred Fit:** {preferred_fit}\n"
                  f"- **Fabric Preference:** {fabric_preference}\n"
                  f"- **Accessory Preference:** {accessory_preference}\n"
                  "Analyze the retrieved images and suggest a complete outfit, including accessories and footwear."
                  "Explain why the outfit suits their style and occasion.")
        try:
            response = model.generate_content([prompt] + retrieved_images)
            st.subheader("👗 AI Fashion Styling Recommendations:")
            st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred while generating styling recommendations: {e}")
