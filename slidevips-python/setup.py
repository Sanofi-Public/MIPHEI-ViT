import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="slidevips",
    version="0.1.0",
    author="Guillaume Balezo",
    author_email="guillaume.balezo@gmail.com",
    description="A simple tool for reading and processing slide images using pyvips",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guillaume-Balezo/slidevips-python",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        'dev': [
            'unittest'
            # Other dev dependencies like pylint, flake8 etc.
        ]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)