# setup.py
from setuptools import setup, find_packages

setup(
    name="heatseek",
    version="0.1.0",
    author="Mani Amani",
    packages=find_packages(),
    install_requires=[
        "roboflow>=1.0",
        "ultralytics>=8.0",
        "opencv-python",
        "torch",
        "tqdm",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "heatseek= heatseek.cli:main",
        ],
    },
)
