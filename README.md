# Latent Editor: Real-Time StyleGAN2 Face Editing Interface

A powerful web-based interface for real-time editing of StyleGAN2-generated faces using latent space manipulation. This application allows you to interactively modify facial attributes like smile, age, gender, and pose while maintaining high-quality, photorealistic results.

![Latent Editor Demo](https://img.shields.io/badge/Status-Ready%20for%20Release-green)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red)

## ✨ Features

### 🎨 **Edit Mode**
- **Real-time face editing** with intuitive sliders
- **Multiple attributes**: Smile, Age, Gender, Yaw (head pose)
- **Instant preview** with high-quality 1024x1024 resolution
- **Random face generation** with one click
- **Reset controls** to return to original face
- **Download functionality** for edited images

### 🔄 **Style Mix Mode**
- **Layer-based style mixing** between two faces
- **Predefined layer presets**: Pose (layers 0-5), Face (layers 6-11), Style (layers 12-17)
- **Custom layer selection** for fine-grained control
- **Truncation psi control** for quality vs. variety trade-off
- **Real-time preview** of mixed results

### 🚀 **Performance Optimizations**
- **GPU acceleration** with CUDA support
- **Cached model loading** for faster startup
- **Optimized image display** with smart downscaling
- **Session state management** for smooth interactions

## 🛠️ Installation

### Prerequisites
- **Python 3.7+**
- **CUDA-compatible GPU** (recommended for optimal performance)
- **NVIDIA drivers** and CUDA toolkit

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd latent_editor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the StyleGAN2 model**
   ```bash
   # Create the stylegan2 directory
   mkdir -p stylegan2
   
   # Download the FFHQ model (this will be done automatically on first run)
   # The model will be cached in ~/.cache/dnnlib/
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 Usage Guide

### Edit Mode
1. **Generate a new face**: Click "New Face" to create a random face
2. **Adjust attributes**: Use the sliders to modify:
   - **Smile**: Increase for a bigger smile
   - **Age**: Increase for an older appearance
   - **Gender**: Increase for more masculine features
   - **Yaw**: Increase for rightward head pose
3. **Reset**: Click "Reset ♻️" to return to original values
4. **Save**: Click "Save 💾" to download the full-resolution image

### Style Mix Mode
1. **Generate two faces**: Use "New Face 1" and "New Face 2" buttons
2. **Select layers**: Choose which layers to mix from Face 2 into Face 1
3. **Use presets**: Quick buttons for common layer combinations:
   - **Pose (0-5)**: Head position and orientation
   - **Face (6-11)**: Facial structure and features
   - **Style (12-17)**: Hair, skin texture, and fine details
4. **Adjust truncation**: Lower values create more average faces, higher values add variety
5. **Mix styles**: Click "Style Mix!" to generate the combined result
6. **Save**: Download the mixed image

## 🏗️ Architecture

### Core Components
- **`app.py`**: Main Streamlit application
- **`utils/stylegan2_utils.py`**: StyleGAN2 model utilities
- **`utils/latent_directions_utils.py`**: Latent direction loading
- **`stylegan2/`**: StyleGAN2-ADA PyTorch implementation
- **`latent_directions/`**: Pre-computed latent directions

### Key Features
- **W+ latent space**: Uses extended latent space for better control
- **Direction vectors**: Pre-computed semantic directions for attribute editing
- **Layer mixing**: Fine-grained control over which style layers to transfer
- **Real-time generation**: Instant feedback for all modifications

## 🔧 Technical Details

### Model Information
- **Base Model**: FFHQ StyleGAN2-ADA (1024x1024)
- **Latent Space**: W+ space (18 layers × 512 dimensions)
- **Resolution**: 1024×1024 pixels
- **Format**: PNG with full quality

### Performance
- **GPU Memory**: ~8GB recommended
- **Startup Time**: ~10-30 seconds (model loading)
- **Generation Time**: ~100-200ms per image
- **Display Resolution**: 768×768 (optimized for web)

## 🎯 Use Cases

### Creative Applications
- **Character design** for games and animation
- **Portrait photography** concept exploration
- **Fashion and beauty** visualization
- **Research and education** in generative AI

### Research Applications
- **Latent space exploration** and analysis
- **StyleGAN2 research** and experimentation
- **Facial attribute manipulation** studies
- **GAN interpretability** research

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **StyleGAN2-ADA**: Original research by NVIDIA
- **Streamlit**: Web framework for data science
- **PyTorch**: Deep learning framework
- **FFHQ Dataset**: High-quality face dataset

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/your-repo/issues) page
2. Create a new issue with detailed information
3. Include your system specifications and error messages

---

**Made with ❤️ for the AI research community** 