import streamlit as st
import numpy as np
import torch
import PIL.Image
import os
import sys
import time
import io
import copy
from pathlib import Path
from collections import deque

# Add stylegan2 to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "stylegan2"))
import dnnlib
import legacy

# Import utility functions
from utils.stylegan2_utils import (
    load_generator,
    random_wplus,
    generate_image,
    stylemix_wplus,
    stylemix_grid,
    generate_image_custom,
)
from utils.latent_directions_utils import load_directions, get_default_directions

# --- CONFIG ---
WPLUS_SHAPE = (1, 18, 512)  # FFHQ StyleGAN2 W+ shape
MAX_HISTORY_SIZE = 20  # Maximum number of states to keep in history

def downscale_for_display(img, size=768):
    """Downscale PIL image to given size for faster UI display."""
    return img.resize((size, size), PIL.Image.LANCZOS)

# --- UNDO/REDO FUNCTIONS ---
def save_state_to_history():
    """Save current state to history for undo/redo functionality."""
    if "history" not in st.session_state:
        st.session_state["history"] = deque(maxlen=MAX_HISTORY_SIZE)
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = -1
    
    # Create a deep copy of the current state
    current_state = {
        "mode": st.session_state.get("mode", "🎨 Edit"),
        "sliders": copy.deepcopy(st.session_state.get("sliders", {})),
        "base_wplus": st.session_state.get("base_wplus", None),
        "base_wplus1": st.session_state.get("base_wplus1", None),
        "base_wplus2": st.session_state.get("base_wplus2", None),
        "stylemix_layers": copy.deepcopy(st.session_state.get("stylemix_layers", [])),
        "truncation_psi": st.session_state.get("truncation_psi", 1.0),
        "mixed_wplus": st.session_state.get("mixed_wplus", None),
    }
    
    # Remove states after current index (if we're not at the end)
    while len(st.session_state["history"]) > st.session_state["current_index"] + 1:
        st.session_state["history"].pop()
    
    # Add new state
    st.session_state["history"].append(current_state)
    st.session_state["current_index"] = len(st.session_state["history"]) - 1

def can_undo():
    """Check if undo is possible."""
    return "current_index" in st.session_state and st.session_state["current_index"] > 0

def can_redo():
    """Check if redo is possible."""
    return ("current_index" in st.session_state and 
            "history" in st.session_state and 
            st.session_state["current_index"] < len(st.session_state["history"]) - 1)

def undo():
    """Revert to previous state."""
    if can_undo():
        st.session_state["current_index"] -= 1
        restore_state(st.session_state["history"][st.session_state["current_index"]])

def redo():
    """Restore to next state."""
    if can_redo():
        st.session_state["current_index"] += 1
        restore_state(st.session_state["history"][st.session_state["current_index"]])

def restore_state(state):
    """Restore the application to a saved state."""
    st.session_state["mode"] = state["mode"]
    st.session_state["sliders"] = state["sliders"]
    st.session_state["base_wplus"] = state["base_wplus"]
    st.session_state["base_wplus1"] = state["base_wplus1"]
    st.session_state["base_wplus2"] = state["base_wplus2"]
    st.session_state["stylemix_layers"] = state["stylemix_layers"]
    st.session_state["truncation_psi"] = state["truncation_psi"]
    if state["mixed_wplus"] is not None:
        st.session_state["mixed_wplus"] = state["mixed_wplus"]
    elif "mixed_wplus" in st.session_state:
        del st.session_state["mixed_wplus"]

# --- CACHING ---
@st.cache_resource
def cached_load_directions():
    try:
        return load_directions(get_default_directions())
    except Exception as e:
        st.error(f"Failed to load latent directions: {e}")
        return {}

@st.cache_resource
def cached_load_generator():
    try:
        return load_generator()
    except Exception as e:
        st.error(f"Failed to load StyleGAN2 model: {e}")
        return None, None

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Latent Editor",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎨 Latent Editor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-Time StyleGAN2 Face Editing Interface</p>', unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app allows you to edit StyleGAN2-generated faces in real-time using latent space manipulation.
    
    **Features:**
    - 🎨 Edit facial attributes
    - 🔄 Mix styles between faces
    - 💾 Download high-resolution images
    - ⚡ Real-time preview
    """)
    
    st.markdown("### 🔧 System Info")
    if torch.cuda.is_available():
        st.markdown(f"<span class='status-success'>✅ GPU: {torch.cuda.get_device_name()}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-error'>⚠️ CPU Mode (slower)</span>", unsafe_allow_html=True)
    
    st.markdown(f"**PyTorch Version:** {torch.__version__}")
    st.markdown(f"**CUDA Available:** {torch.cuda.is_available()}")

# Loading states
with st.spinner("Loading StyleGAN2 model..."):
    G, device = cached_load_generator()

if G is None:
    st.error("❌ Failed to load StyleGAN2 model. Please check your installation.")
    st.stop()

with st.spinner("Loading latent directions..."):
    directions = cached_load_directions()

if not directions:
    st.error("❌ Failed to load latent directions. Please check your installation.")
    st.stop()

st.success("✅ All models loaded successfully!")

# --- Mode Selector ---
st.markdown("---")

# Mode selector with undo/redo buttons
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    mode = st.radio(
        "**Select Mode:**",
        ["🎨 Edit", "🔄 Style Mix"],
        horizontal=True,
        help="Choose between editing individual faces or mixing styles between two faces"
    )

with col2:
    if st.button("↶ Undo", disabled=not can_undo(), help="Revert to previous state"):
        undo()
        st.rerun()

with col3:
    if st.button("↷ Redo", disabled=not can_redo(), help="Restore to next state"):
        redo()
        st.rerun()



# Save current mode to session state for undo/redo
st.session_state["mode"] = mode

# Session state initialization
if "base_wplus" not in st.session_state:
    st.session_state["base_wplus"] = random_wplus(G)
if "sliders" not in st.session_state:
    st.session_state["sliders"] = {"smile": 0.0, "age": 0.0, "gender": 0.0, "yaw": 0.0}
if "save_count" not in st.session_state:
    st.session_state["save_count"] = 0

# Session state for two base latents for style mixing
if "base_wplus1" not in st.session_state:
    st.session_state["base_wplus1"] = random_wplus(G)
if "base_wplus2" not in st.session_state:
    st.session_state["base_wplus2"] = random_wplus(G)

# Initialize undo/redo history with initial state
if "history" not in st.session_state:
    save_state_to_history()

if mode == "🎨 Edit":
    colA, colC = st.columns([2, 1])
    
    with colC:
        st.markdown("### 🎛️ Controls")
        
        # Attribute sliders with better descriptions
        current_sliders = st.session_state["sliders"].copy()
        
        st.markdown("**Facial Attributes:**")
        smile = st.slider(
            "😊 Smile",
            -3.0, 3.0, float(current_sliders["smile"]), 0.01,
            help="Adjust the smile intensity. Positive values create bigger smiles."
        )
        age = st.slider(
            "👴 Age",
            -3.0, 3.0, float(current_sliders["age"]), 0.01,
            help="Adjust the apparent age. Positive values make the face appear older."
        )
        gender = st.slider(
            "👨 Gender",
            -3.0, 3.0, float(current_sliders["gender"]), 0.01,
            help="Adjust gender characteristics. Positive values make the face more masculine."
        )
        yaw = st.slider(
            "🔄 Yaw (Head Pose)",
            -3.0, 3.0, float(current_sliders["yaw"]), 0.01,
            help="Adjust head pose. Positive values turn the head rightward."
        )
        
        # Check if any slider values changed
        old_sliders = st.session_state.get("sliders", {})
        new_sliders = {
            "smile": smile,
            "age": age,
            "gender": gender,
            "yaw": yaw,
        }
        
        # Save state for undo/redo when sliders change
        if new_sliders != old_sliders:
            st.session_state["sliders"] = new_sliders
            save_state_to_history()
        else:
            st.session_state["sliders"] = new_sliders

        # Action buttons
        st.markdown("**Actions:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎲 New Image", help="Generate a completely new random face"):
                with st.spinner("Generating new face..."):
                    new_wplus = random_wplus(G)
                    st.session_state["base_wplus"] = new_wplus
                    st.session_state["base_wplus1"] = new_wplus  # Sync style mixing vector
                    st.session_state["sliders"] = {
                        "smile": 0.0,
                        "age": 0.0,
                        "gender": 0.0,
                        "yaw": 0.0,
                    }
                    # Save state for undo/redo
                    save_state_to_history()
                    st.rerun()

        with c2:
            if st.button("♻️ Reset Image", help="Reset all sliders to zero"):
                st.session_state["sliders"] = {
                    "smile": 0.0,
                    "age": 0.0,
                    "gender": 0.0,
                    "yaw": 0.0,
                }
                # Save state for undo/redo
                save_state_to_history()
                st.rerun()
        
        with c3:
            # Generate the full-resolution image for download
            base_wplus = st.session_state["base_wplus"].copy()
            edited_wplus = base_wplus.copy()
            for name, alpha in st.session_state["sliders"].items():
                if name in directions:
                    edited_wplus += alpha * directions[name]

            full_res_image = generate_image(G, edited_wplus)
            
            # Create download button for full-resolution image
            img_buffer = io.BytesIO()
            full_res_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="💾 Save Image",
                data=img_buffer.getvalue(),
                file_name=f"latent_edit_{st.session_state.get('save_count', 0):03d}.png",
                mime="image/png",
                help="Download the image in full 1024x1024 resolution"
            )

    # Generate and display the edited image
    base_wplus = st.session_state["base_wplus"].copy()
    edited_wplus = base_wplus.copy()
    for name, alpha in st.session_state["sliders"].items():
        if name in directions:
            edited_wplus += alpha * directions[name]

    with st.spinner("Generating image..."):
        image = generate_image(G, edited_wplus)

    with colA:
        st.markdown("### 🖼️ Generated Face")
        st.image(downscale_for_display(image), use_container_width=True, caption="Real-time edited face")

elif mode == "🔄 Style Mix":
    colA, colB, colC = st.columns(3)
    
    with colC:
        st.markdown("### 🎛️ Controls")
        
        # Face generation buttons
        st.markdown("**Generate Faces:**")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button("🎲 Face 1", help="Generate a new random face for position 1"):
                with st.spinner("Generating new face..."):
                    new_wplus = random_wplus(G)
                    st.session_state["base_wplus1"] = new_wplus
                    # Save state for undo/redo
                    save_state_to_history()
                    st.rerun()
        
        with btn_col2:
            if st.button("🎲 Face 2", help="Generate a new random face for position 2"):
                with st.spinner("Generating new face..."):
                    new_wplus2 = random_wplus(G)
                    st.session_state["base_wplus2"] = new_wplus2
                    # Save state for undo/redo
                    save_state_to_history()
                    st.rerun()
        
        with btn_col3:
            # Generate full-resolution mixed image for download
            if "mixed_wplus" in st.session_state:
                with st.spinner("Preparing download..."):
                    full_res_mixed = generate_image_custom(
                        G,
                        st.session_state["mixed_wplus"],
                        truncation_psi=st.session_state["truncation_psi"],
                        noise_mode='const'
                    )
                    
                    # Create download button for full-resolution mixed image
                    img_buffer = io.BytesIO()
                    full_res_mixed.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    st.download_button(
                        label="💾 Save",
                        data=img_buffer.getvalue(),
                        file_name=f"stylemix_{st.session_state.get('save_count', 0):03d}.png",
                        mime="image/png",
                        help="Download the style mixed image in full 1024x1024 resolution"
                    )



        # Style mixing controls
        st.markdown("**Style Mixing:**")
        st.markdown("**Layer Selection**")
        layer_indices = list(range(18))
        default_layers = list(range(6))
        if "stylemix_layers" not in st.session_state:
            st.session_state["stylemix_layers"] = default_layers
        
        selected_layers = st.multiselect(
            "Select layers to mix (0-17)",
            layer_indices,
            default=st.session_state["stylemix_layers"],
            help="Choose which layers to transfer from Face 2 to Face 1. Lower layers control pose, higher layers control style details."
        )
        
        # Save state for undo/redo when layer selection changes
        if selected_layers != st.session_state["stylemix_layers"]:
            st.session_state["stylemix_layers"] = selected_layers
            save_state_to_history()
        else:
            st.session_state["stylemix_layers"] = selected_layers

        # Quick region presets
        st.markdown("**Quick Presets:**")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎯 Pose (0-5)", help="Mix only pose-related layers"):
                st.session_state["stylemix_layers"] = list(range(0, 6))
                # Save state for undo/redo
                save_state_to_history()
                st.rerun()
        with c2:
            if st.button("👤 Face (6-11)", help="Mix only facial structure layers"):
                st.session_state["stylemix_layers"] = list(range(6, 12))
                # Save state for undo/redo
                save_state_to_history()
                st.rerun()
        with c3:
            if st.button("✨ Style (12-17)", help="Mix only style and detail layers"):
                st.session_state["stylemix_layers"] = list(range(12, 18))
                # Save state for undo/redo
                save_state_to_history()
                st.rerun()

        # Truncation psi
        if "truncation_psi" not in st.session_state:
            st.session_state["truncation_psi"] = 1.0
        
        st.markdown("**Quality Control:**")
        truncation_psi = st.slider(
            "🎚️ Truncation Psi",
            0.3, 1.0, float(st.session_state["truncation_psi"]), 0.01,
            help="Lower values create more average faces, higher values add variety and detail."
        )
        
        # Save state for undo/redo when truncation psi changes
        if truncation_psi != st.session_state["truncation_psi"]:
            st.session_state["truncation_psi"] = truncation_psi
            save_state_to_history()
        else:
            st.session_state["truncation_psi"] = truncation_psi

        st.markdown(
            f"<div>Selected layers: <b>{st.session_state['stylemix_layers']}</b></div></br>",
            unsafe_allow_html=True
        )



        if st.button("🔄 Style Mix!", help="Generate the mixed face"):
            with st.spinner("Mixing styles..."):
                mixed_wplus = stylemix_wplus(
                    st.session_state["base_wplus1"],
                    st.session_state["base_wplus2"],
                    st.session_state["stylemix_layers"]
                )
                st.session_state["mixed_wplus"] = mixed_wplus
                # Save state for undo/redo
                save_state_to_history()
                st.rerun()

    # Display the two source faces
    with colA:
        st.markdown("### 🖼️ Face 1 (Base)")
        with st.spinner("Generating Face 1..."):
            img1 = generate_image(G, st.session_state["base_wplus1"])
        st.image(downscale_for_display(img1), use_container_width=True)
        
        # Show the mixed result below Face 1 with smaller size
        if "mixed_wplus" in st.session_state:
            st.markdown("### 🎨 Mixed Result")
            with st.spinner("Generating mixed image..."):
                mixed_img = generate_image_custom(
                    G,
                    st.session_state["mixed_wplus"],
                    truncation_psi=st.session_state["truncation_psi"],
                    noise_mode='const'
                )
            # Display smaller mixed image
            st.image(
                downscale_for_display(mixed_img, size=384),  # Reduced size
                caption=f"Style Mixed (layers={st.session_state['stylemix_layers']}, psi={st.session_state['truncation_psi']:.2f})",
                use_container_width=False,  # Don't use full container width
            )
    
    with colB:
        st.markdown("### 🖼️ Face 2 (Style Source)")
        with st.spinner("Generating Face 2..."):
            img2 = generate_image(G, st.session_state["base_wplus2"])
        st.image(downscale_for_display(img2), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p>🎨 Latent Editor - Real-Time StyleGAN2 Face Editing Interface</p>
    <p>Built with ❤️ using Streamlit and PyTorch</p>
</div>
""", unsafe_allow_html=True)
