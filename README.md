# Light Ray Diffusion

## TODO

- [x] Use the camera position as world space origin.
- [x] (5/11) Render ground truth for the first 10 scenes. Generate 3 random light sources and render for each scene.
- [x] (5/13) [Fixed] The origin of coordinate system of the rays is the camera position, but the coordinate system is not camera space.
    - [x] Validate the correctness of the rays.
        - Rays indeed pass the light source. But after conversion to a point using least squares, the point deviates from the light source by 0.2-2.0.
        - [x] (5/15) Validate the correctness of the rays-to-point conversion: Turned out to be correct. The deviation is due to the filled zero values in the rays where ray doesn't intersect with anything in the scene.
- [x] Vectorize computation in ray regression.
- [x] Check the correctness of the moments of the rays.
- [ ] Render ground truth such that the moments are "smooth" and "continuous" for the first light of the first scene.
- [ ] Train and test on the first 10 scenes. The split is 80% training and 20% testing.
    - [ ] Randomly use 80% of images and rays in each scene for training.
    - [ ] Render ground truth for more random light sources and train on 80% of the light sources. This is to test performance on unseen light sources but seen scenes.
- [ ] Generate ground truth using depth maps and 3D light source bounding box ground truth in OpenRooms.
- [ ] Add ray diffusion model.

## Installation

All experiments are run in a conda environment with Python 3.10 and CUDA 11.8. To install the dependencies, run the following commands:

```bash
conda create -n lrd python=3.10
conda activate lrd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Introduction

The original goal of this project is to learn a near-field illumination prior for inverse rendering. The near-field illumination prior is a function that takes an image as input and outputs the illumination in the scene in which the image is captured. The near-field illumination prior is learned from a dataset of images of indoor scenes called [OpenRooms](https://vilab-ucsd.github.io/ucsd-openrooms/) under different lighting conditions.

However, we found that the near-field illumination prior is a highly non-linear function of the image, and a simple neural network is not expressive enough to capture the complex relationship between the image and the illumination in the scene. Therefore, we are exploring the possibility of predicting the light sources in the scene by predicting many rays that are emitted from the light sources. We are able to convert between ray bundles and light source position. By adding more properties to the ray bundles, such as the color of the light, we can potentially predict the light sources in the scene, which is the next step of our research.

## Progress

The simplest implementation of the prior can be a neural network that takes an image as input and outputs the illumination in the scene. The neural network can be trained using supervised learning on the OpenRooms dataset. However, we found that if the target of the optimization is the position of the light source, which is a 3-vector, a simple MLP is not expressive enough to capture the complex relationship between the image and the light source position. Therefore, one possible solution is to use a more expressive model, such as a convolutional neural network (CNN) or a transformer, to learn the near-field illumination prior.

We hypothesize that the key to learning the near-field illumination prior is over-parameterization. We believe that the near-field illumination prior is a highly non-linear function of the image, and a highly non-linear model is needed to capture this relationship. Replacing the illumination representation with a light volume representation can be a way to increase the expressiveness of the model.

Inspired by [Ray Diffusion](https://jasonyzhang.com/RayDiffusion/), we are now exploring the possibility of predicting the point light sources in the scene by predicting many rays that are emitted from the light sources. We are able to convert between ray bundles and light source position. By adding more properties to the ray bundles, such as the color of the light, we can potentially predict the light sources in the scene, which is the next step of our research. Note that this is still a method that can serve as a baseline for the near-field illumination prior.