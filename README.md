# Learning a Near-Field Illumination Prior for Inverse Rendering

## Installation

All experiments are run in a conda environment with Python 3.10.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Introduction

The goal of this project is to learn a near-field illumination prior for inverse rendering. The near-field illumination prior is a function that takes an image as input and outputs the illumination in the scene in which the image is captured. The near-field illumination prior is learned from a dataset of images of indoor scenes called [OpenRooms](https://vilab-ucsd.github.io/ucsd-openrooms/) under different lighting conditions.

## Progress

The simplest implementation of the prior can be a neural network that takes an image as input and outputs the illumination in the scene. The neural network can be trained using supervised learning on the OpenRooms dataset. However, we found that if the target of the optimization is the position of the light source, which is a 3-vector, a simple MLP is not expressive enough to capture the complex relationship between the image and the light source position. Therefore, one possible solution is to use a more expressive model, such as a convolutional neural network (CNN) or a transformer, to learn the near-field illumination prior.

We hypothesize that the key to learning the near-field illumination prior is over-parameterization. We believe that the near-field illumination prior is a highly non-linear function of the image, and a highly non-linear model is needed to capture this relationship. Replacing the illumination representation with a light volume representation can be a way to increase the expressiveness of the model.

Inspired by [Ray Diffusion](https://jasonyzhang.com/RayDiffusion/), we are now exploring the possibility of predicting the point light sources in the scene by predicting many rays that are emitted from the light sources. We are able to convert between ray bundles and light source position. By adding more properties to the ray bundles, such as the color of the light, we can potentially predict the light sources in the scene, which is the next step of our research. Note that this is still a method that can serve as a baseline for the near-field illumination prior.