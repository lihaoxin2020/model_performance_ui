from setuptools import setup, find_packages

setup(
    name="model-performance-ui",
    version="1.0.0",
    description="Interactive dashboard for analyzing model performance",
    author="AI Model Evaluator",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.24.0",
        "plotly>=5.14.0",
        "pandas>=1.5.0",
        "numpy>=1.22.0",
        "seaborn>=0.12.0",
        "datasets>=2.12.0",
        "transformers>=4.28.0",
        "matplotlib>=3.7.0"
    ],
    python_requires=">=3.8",
) 