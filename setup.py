from setuptools import setup, find_packages

setup(
    name="flood_detection_trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "segmentation-models-pytorch",
        "rasterio",
        "numpy",
        "tqdm",
    ],
)