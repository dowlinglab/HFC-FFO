from setuptools import setup, find_packages

#####################################
VERSION = "0.0.0"
ISRELEASED = False
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev0"
#####################################

requirements = [
    "numpy",
    "pandas",
    "unyt",
    "seaborn",
    "gpflow",
    "fffit",
    "matplotlib",
    "signac",
    "signac-flow",
]

setup(
    name="hfcs",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    author="Ryan S. DeFever",
    author_email="rdefever@nd.edu",
    url="https://github.com/rsdefever/hfcs-fffit",
    install_requires=requirements,
    python_requires=">=3.6, <4",
)
