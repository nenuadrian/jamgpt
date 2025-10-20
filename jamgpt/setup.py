from setuptools import setup, find_packages

setup(
    name="jamgpt",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "tiktoken",
    ],
    extras_require={
        "test": ["pytest", "torch"],
    },
    python_requires=">=3.7",
)
