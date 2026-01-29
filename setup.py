from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="my_first_package_8TyUNLepm1b",
    version="0.0.4",
    packages=find_packages(),
    install_requires=["pandas", "kagglehub"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="My First Module",
    author="Sumner MacArthur",
    author_email="scmacarthur@icloud.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
