import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os

# Import Models
from models.clawgan import ClawGenerator
from models.infragan import UNetGenerator
from models.basic import BasicGenerator

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256  # Size used during training

st.set_page_config(page_title="RGB to Infrared Converter", layout="wide")
st.title("RGB to Infrared Image Translation")

# --- Sidebar for Model Selection ---
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose Inference Model",
    ("ClawGAN", "InfraGAN", "Basic Model")
)

# --- Helper Functions ---
@st.cache_resource
def load_model(model_name):
    """
    Loads the specific model architecture and weights.
    Expects weights to be in a 'checkpoints' folder.
    """
    try:
        if model_name == "ClawGAN":
            model = ClawGenerator(in_channels=3, out_channels=1, base_filters=32)
            # Replace with actual path to your clawgan.pth
            ckpt_path = "checkpoints/clawgan.pth" 
            
        elif model_name == "InfraGAN":
            model = UNetGenerator(input_nc=3, output_nc=1, base_f=256)
            ckpt_path = "checkpoints/infragan.pth"
            
        elif model_name == "Basic Model":
            model = BasicGenerator(input_nc=3, output_nc=1, base_f=64)
            ckpt_path = "checkpoints/basic.pth"
        
        # Load weights if available
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=DEVICE)
            # Handle dictionary keys if wrapped in 'G', 'model', etc.
            if 'G_AB' in state_dict:
                model.load_state_dict(state_dict['G_AB'])
            elif 'netG' in state_dict:
                model.load_state_dict(state_dict['netG'])
            elif 'G' in state_dict:
                model.load_state_dict(state_dict['G'])
            else:
                model.load_state_dict(state_dict)
            print(f"Loaded weights for {model_name}")
        else:
            st.warning(f"Checkpoint for {model_name} not found at {ckpt_path}. Using random weights for demo.")

        model.to(DEVICE)
        model.eval()
        return model
    
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def preprocess_image(image):
    """Resizes and normalizes the image for the model"""
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0)

def postprocess_output(tensor):
    """Denormalizes and converts output tensor to PIL Image"""
    tensor = tensor.detach().cpu()
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor * 0.5) + 0.5
    tensor = tensor.clamp(0, 1)
    
    # If 1 channel (grayscale), remove batch dimension
    if tensor.shape[1] == 1:
        tensor = tensor.squeeze(0) # 1xHxW
    
    to_pil = T.ToPILImage()
    return to_pil(tensor)

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload an RGB Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Input
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.subheader("Input (RGB)")
        st.image(image, use_column_width=True)

    # 2. Run Inference
    if st.button("Convert to Infrared"):
        with st.spinner(f"Running inference using {model_choice}..."):
            # Load Model
            model = load_model(model_choice)
            
            if model:
                # Preprocess
                input_tensor = preprocess_image(image).to(DEVICE)
                
                # Predict
                with torch.no_grad():
                    output_tensor = model(input_tensor)
                
                # Postprocess
                result_image = postprocess_output(output_tensor)
                
                # Display Output
                with col2:
                    st.subheader(f"Output ({model_choice})")
                    st.image(result_image, use_column_width=True)

else:
    st.info("Please upload an image to start.")

st.markdown("---")
st.caption(f"Running on {DEVICE}")