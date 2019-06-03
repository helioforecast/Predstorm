#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="predstorm",
    version="0.1",
    author="IWF Helio Group",
    author_email="christian.moestl@oeaw.ac.at",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IWF-helio/PREDSTORM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
