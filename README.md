# ChimpStackr
[![chimpstackr](https://snapcraft.io/chimpstackr/badge.svg)](https://snapcraft.io/chimpstackr) ![GitHub all releases](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/total) ![GitHub repo size](https://img.shields.io/github/repo-size/noah-peeters/ChimpStackr) ![GitHub issues](https://img.shields.io/github/issues-raw/noah-peeters/ChimpStackr) ![GitHub pull requests](https://img.shields.io/github/issues-pr-raw/noah-peeters/ChimpStackr) ![GitHub](https://img.shields.io/github/license/noah-peeters/ChimpStackr) ![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/noah-peeters/ChimpStackr/latest)

Open source multi-platform program for focus stacking many images. 
ChimpStackr is a program that can be used for focus stacking many images.
Focus stacking is often a necessity when working with high-magnification pictures.. 
## Table of contents
* [Installation](#installation)
* [Gallery](#gallery)
* [Build from source](#build-from-source-code)
* [Sources](#sources)
## Installation
Installation instructions per supported platform.
### Windows
* Download the latest release, and run the ".exe" file
### Linux
* Snapcraft: https://snapcraft.io/chimpstackr

## Gallery
Work in progress...

## Build from source code
* Clone repository
* Install python requirements:  ``pip install -r requirements.txt``
* Run file ``src/run.py`` to start the program
### Local build  (snapcraft)
* Run build: ```snapcraft```
* Install local snap: ```snap install *.snap --dangerous --devmode```

## Sources
* Focus stacking algorithm (slightly adapted) implemented from:
Wang, W., & Chang, F. (2011b). A Multi-focus Image Fusion Method Based on Laplacian Pyramid. _Journal of Computers_, _6_(12).
