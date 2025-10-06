# File: setup.py
#!/usr/bin/env python3

"""
FE-AI: Federated Explainable AI System for Scalable Gait-Based Neurological Disease Detection
"""

from setuptools import setup, find_packages
import os

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fe-ai-gait-detection",
    version="2.3.1",
    author="FE-AI Development Team",
    author_email="dev@fe-ai-system.com",
    description="Federated Explainable AI System for Scalable Gait-Based Neurological Disease Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-organization/FE-AI-Gait-Disease-Detection",
    project_urls={
        "Bug Tracker": "https://github.com/your-organization/FE-AI-Gait-Disease-Detection/issues",
        "Documentation": "https://fe-ai-docs.readthedocs.io",
        "Source Code": "https://github.com/your-organization/FE-AI-Gait-Disease-Detection",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fe-ai=main:main",
            "fe-ai-server=scripts.start_fl_server:main",
            "fe-ai-train=scripts.train_models:main",
            "fe-ai-init=scripts.init_database:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    zip_safe=False,
    keywords=[
        "federated learning",
        "explainable ai",
        "gait analysis",
        "neurological diseases",
        "medical ai",
        "parkinson disease",
        "machine learning",
        "deep learning",
        "healthcare",
    ],
)