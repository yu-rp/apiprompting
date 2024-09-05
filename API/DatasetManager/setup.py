from setuptools import setup, find_packages

setup(
    name='DatasetManager',
    version='1.0',
    packages=find_packages(),
    description='A package supports the consistency of dataset processing across different modules',
    install_requires=[
        "pillow",
        "pandas",
        "numpy",
        "torch",
        "datasets"
    ],
)
