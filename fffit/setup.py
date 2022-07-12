from setuptools import setup, find_packages

#####################################
VERSION = "0.0.0"
ISRELEASED = False
if ISRELEASED:
    __version__ = VERSION
else:
    __version__ = VERSION + ".dev0"
#####################################

setup(
    name="fffit",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    author="Ryan S. DeFever",
    author_email="rdefever@nd.edu",
    url="https://github.com/rsdefever/fffit",
    python_requires=">=3.6, <4",
)
