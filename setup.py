from setuptools import setup, find_packages

setup(
    name="fine_tuning_pipeline",
    version="0.1.0",
    description="A simple data pipeline for fine-tuning language models",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0"
    ],
    python_requires=">=3.8",
)