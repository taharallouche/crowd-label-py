from setuptools import setup, find_packages

setup(
    name="size_matters",
    version="0.1.0",
    description="A python package for the aggregation of collectively annotated data using the size-matters principle",
    url="https://github.com/taharallouche/Truth_Tracking-via-AV",
    author="Tahar Allouche",
    author_email="tahar.allouche@dauphine.eu",
    license="BSD 2-clause",
    packages=find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.1",
        "scipy>=1.10.0",
        "pandas>=1.4.1",
        "matplotlib>=3.6.3",
        "scikit-learn>=1.2.1",
        "ray>=2.2.0",
        "tornado>=6.2",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
    ],
)
