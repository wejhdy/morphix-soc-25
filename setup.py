#!/usr/bin/env python3
"""
Setup script for Latent Editor
A real-time StyleGAN2 face editing interface
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="latent-editor",
    version="1.0.0",
    author="Latent Editor Team",
    author_email="contact@latent-editor.com",
    description="Real-time StyleGAN2 face editing interface",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/latent-editor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "latent-editor=app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.py"],
    },
    keywords="stylegan2, gan, face-editing, latent-space, streamlit, pytorch",
    project_urls={
        "Bug Reports": "https://github.com/your-username/latent-editor/issues",
        "Source": "https://github.com/your-username/latent-editor",
        "Documentation": "https://github.com/your-username/latent-editor#readme",
    },
) 