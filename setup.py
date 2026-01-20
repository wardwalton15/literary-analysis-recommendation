"""Setup script for Literary Analysis Platform."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="literary-analysis",
    version="0.1.0",
    author="Literary Analysis Project",
    description="NLP-powered platform for analyzing classic literature with ML-based recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/literary-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "literary-analysis=src.cli:main",
        ],
    },
)
