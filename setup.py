from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="my_approch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A project implementing CNN and DQN models for tabular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_approch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train-cnn=src.training.train_cnn:main",
            "train-dqn=src.training.train_dqn:main",
            "train-hybrid=src.training.train_hybrid:main",
        ],
    },
) 