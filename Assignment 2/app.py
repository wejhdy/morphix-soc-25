import streamlit as st
import numpy as np
import torch
import PIL.Image
import os
import sys

# Add stylegan2 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'stylegan2'))
import dnnlib
import legacy

# Import utility functions
from utils.stylegan2_utils import load_generator, random_wplus, generate_image
from utils.latent_directions_utils import load_directions, get_default_directions

# --- CONFIG ---
WPLUS_SHAPE = (1, 18, 512)  # FFHQ StyleGAN2 W+ shape

# --- CACHING ---
@st.cache_resource
def cached_load_directions():
    return load_directions(get_default_directions())

# --- STREAMLIT UI ---
st.set_page_config(page_title="Latent Editor", layout="wide")
st.title("Real-Time Latent Editing Interface")

G, device = load_generator()
directions = cached_load_directions()

# Session state for base latent
if 'base_wplus' not in st.session_state:
    st.session_state['base_wplus'] = random_wplus(G)
if 'sliders' not in st.session_state:
    st.session_state['sliders'] = {'smile': 0.0, 'age': 0.0, 'gender': 0.0, 'yaw': 0.0, 'pitch': 0.0}

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### Controls")
    smile = st.slider('Smile', -3.0, 3.0, float(st.session_state['sliders']['smile']), 0.01)
    age = st.slider('Age', -3.0, 3.0, float(st.session_state['sliders']['age']), 0.01)
    gender = st.slider('Gender', -3.0, 3.0, float(st.session_state['sliders']['gender']), 0.01)
    yaw = st.slider('Yaw', -3.0, 3.0, float(st.session_state['sliders']['yaw']), 0.01)
    pitch = st.slider('Pitch', -3.0, 3.0, float(st.session_state['sliders']['pitch']), 0.01)
    st.session_state['sliders'] = {'smile': smile, 'age': age, 'gender': gender, 'yaw': yaw, 'pitch': pitch}

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('Random Face'):
            st.session_state['base_wplus'] = random_wplus(G)
            st.session_state['sliders'] = {'smile': 0.0, 'age': 0.0, 'gender': 0.0, 'yaw': 0.0, 'pitch': 0.0}
            st.rerun()
    with c2:
        if st.button('Reset'):
            st.session_state['sliders'] = {'smile': 0.0, 'age': 0.0, 'gender': 0.0, 'yaw': 0.0, 'pitch': 0.0}
            st.rerun()
    with c3:
        save_flag = st.button('Save')

# Compute edited latent
base_wplus = st.session_state['base_wplus'].copy()  # (18, 512)
edited_wplus = base_wplus.copy()
for name, alpha in st.session_state['sliders'].items():
    edited_wplus += alpha * directions[name]  # both (18, 512)

# Generate image
image = generate_image(G, edited_wplus)

with col1:
    st.image(image, caption="Edited Face", use_container_width=True)

if 'save_count' not in st.session_state:
    st.session_state['save_count'] = 0
if 'last_saved' not in st.session_state:
    st.session_state['last_saved'] = None

if 'save_flag' in locals() and save_flag:
    save_dir = 'saved_images'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'face_{st.session_state["save_count"]:03d}.png')
    image.save(save_path)
    st.session_state['save_count'] += 1
    st.session_state['last_saved'] = save_path
    st.success(f"Image saved to {save_path}")

if st.session_state['last_saved']:
    st.info(f"Last saved: {st.session_state['last_saved']}")
