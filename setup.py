from setuptools import find_packages, setup

setup(
    name="hack_digital_transformation",
    version="0.1.0",
    description="Разработка полноценного MVP продукта для определения местоположения по фотографиям с собственным сайтом для работы с пользователем",
    author="LizardAPN",
    author_email="piskunov-05@bk.ru",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "jupyter>=1.0.0",
            "boto3>=1.28.0",
            "awscli>=1.29.0",
        ],
        "production": [
            "gunicorn>=20.0.0",
            "uvicorn>=0.20.0",
            "fastapi>=0.85.0",
        ],
    },
)
