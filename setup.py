"""
KumoRFM安装配置
"""

from setuptools import setup, find_packages
import os

# 读取README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 版本信息
VERSION = "0.1.0"

setup(
    name="kumo-rfm",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    description="A Foundation Model for In-Context Learning on Relational Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/kumo_rfm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "distributed": [
            "ray>=1.13.0",
            "horovod>=0.22.0",
        ],
        "optimization": [
            "optuna>=2.10.0",
            "hyperopt>=0.2.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "kumorfm-train=kumo_rfm.cli.train:main",
            "kumorfm-predict=kumo_rfm.cli.predict:main",
            "kumorfm-evaluate=kumo_rfm.cli.evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kumo_rfm": [
            "config/*.yaml",
            "data/examples/*.csv",
        ],
    },
    zip_safe=False,
)