#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "predstorm",
    version = "0.1",
    author = "Helio4Cast Group, Graz Austria",
    author_email = "christian.moestl@oeaw.ac.at",
    description = "Python3 package and scripts for space weather prediction research.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/helioforecast/Predstorm",
    download_url = "https://github.com/helioforecast/Predstorm/archive/v_01.tar.gz",
    packages = setuptools.find_packages(),
    install_requires=[
        "numpy",
        "heliosat",
      ],
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
