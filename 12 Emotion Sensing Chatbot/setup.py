from setuptools import setup, find_packages

setup(
    name="mental-health-chatbot",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "transformers>=4.31.0",
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
)