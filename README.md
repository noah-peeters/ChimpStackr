# ChimpStackr
![GitHub all releases](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/total) ![GitHub release (latest by date)](https://img.shields.io/github/downloads/noah-peeters/ChimpStackr/latest/total) ![GitHub repo size](https://img.shields.io/github/repo-size/noah-peeters/ChimpStackr) ![GitHub](https://img.shields.io/github/license/noah-peeters/ChimpStackr) ![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/noah-peeters/ChimpStackr/latest)

<p align="center">
  <img src="https://user-images.githubusercontent.com/17707805/196983883-84ec7174-74d3-4833-b9f6-16b84e19280d.png" width="300"/>
</p>

Open source multi-platform program for focus stacking many images.

View the [wiki](https://github.com/noah-peeters/ChimpStackr/wiki/Basic-usage) for instructions.

Focus stacking is often a necessity when working with high-magnification pictures...
Chimpstackr implements the laplacian pyramid fusion algorithm (see: [Sources](#sources))

## Gallery
Following image stacks have been taken by [me](https://github.com/noah-peeters). Each image has been stacked using ChimpStackr and post-processed using [darktable](https://www.darktable.org/). Each image stack contains around 150 individual images. Every image in the stack was taken at around 4x magnification on a (slightly) wobbly rig. This should be a good test for both the stacking algorithm, and the alignment algorithm.

The post processing consists of these steps:
* A minor cropping of the edges (to remove 'artifacts' created by alignment algorithm)
* Sharpening
* Contrast, saturation improvements

![Bij_TranslationAlignment](https://user-images.githubusercontent.com/17707805/196990942-413ea35c-2abb-4bce-9807-3f3d6b3de3c5.jpg)
![Edited](https://user-images.githubusercontent.com/17707805/196991117-dc4f1c76-cc87-4ef1-92ee-9a7484c7ff07.jpg)
![Bewerkt](https://user-images.githubusercontent.com/17707805/196996295-9fb6c365-ef10-4ef5-b451-1a7269156e53.jpg)

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

* DFT image alignment algorithm adapted from: https://github.com/matejak/imreg_dft
