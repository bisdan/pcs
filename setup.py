from setuptools import setup, find_packages

setup(
    name='pcs',
    version='0.0.1',
    packages=find_packages(exclude=("examples", "pcs/blender_addon")),
    description="Generation and augmentation of synthetic datasets of photorealistic crystallization images.",
    python_requires=">=3.8",
    install_requires=[
        'pyfastnoisesimd>=0.4.2',
        'opencv-python>=4.5.4',
        'scipy'
    ]
)
