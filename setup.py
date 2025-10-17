from setuptools import setup, find_packages

setup(
    name="jamgpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    extras_require={
        "test": ["pytest", "torch"],
    },
    python_requires=">=3.7",
)
