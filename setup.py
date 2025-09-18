from setuptools import setup, find_packages


setup(
    name="pkl-diffusion-denoising",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    author="Your Name",
    description="PKL-Guided Diffusion for Microscopy Denoising",
    python_requires=">=3.8",
)


