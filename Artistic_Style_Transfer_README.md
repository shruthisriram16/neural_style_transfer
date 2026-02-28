# Artistic Style Transfer using PyTorch

## Project Overview

This project implements Neural Style Transfer, a deep learning technique
that combines the content of one image with the artistic style of
another image through transfer learning. The goal is to generate a new image that preserves the
structure and objects from the content image while adopting textures,
colors, and brush patterns from the style image.

The implementation is based on an optimization approach using a
pretrained convolutional neural network.

------------------------------------------------------------------------

## Core Idea

Neural Style Transfer works by defining two types of losses:

Content Loss\
Style Loss

The generated image is initialized randomly (or as a copy of the content
image) and updated iteratively to minimize the total loss.

Total Loss = Content Loss + Style Loss

------------------------------------------------------------------------

## Model Architecture

A pretrained VGG19 convolutional neural network is used as a fixed
feature extractor.

The network is not trained from scratch. Instead, its intermediate
convolutional layers are used to extract feature representations of
images.

Certain layers are selected for:

Content representation\
Style representation

The generated image is optimized using gradient descent while keeping
the VGG19 weights frozen.

------------------------------------------------------------------------

## Content Representation

Content is captured from higher convolutional layers of VGG19.

These deeper layers encode high-level structural information such as:

Object shapes\
Spatial layout\
Major edges

Content loss is calculated as the mean squared error between:

Feature maps of the content image\
Feature maps of the generated image

This ensures the generated image maintains the structure of the original
content image.

------------------------------------------------------------------------

## Style Representation

Style is captured using multiple convolutional layers from different
depths of the network.

Instead of directly comparing feature maps, style is represented using a
Gram Matrix.

### What is a Gram Matrix

The Gram Matrix measures correlations between different feature maps in
a convolutional layer.

If a layer produces N feature maps, the Gram Matrix is an N x N matrix
where:

Each value represents how strongly two feature maps are correlated
across spatial locations.

If F is a feature map matrix reshaped to size (number_of_filters x
spatial_pixels),

Gram Matrix G = F multiplied by F transpose

This captures:

Texture patterns\
Color distributions\
Repetitive structures

Because correlations represent texture information rather than spatial
arrangement, Gram matrices effectively encode artistic style.

------------------------------------------------------------------------

## Style Loss

Style loss is computed as the mean squared error between:

Gram matrices of the style image\
Gram matrices of the generated image

This is calculated across multiple layers to capture both:

Low-level textures (edges, colors)\
High-level patterns (complex brush strokes)

------------------------------------------------------------------------

## Optimization Process

Steps followed in the project:

1.  Load content and style images\
2.  Resize and normalize images\
3.  Pass images through VGG19\
4.  Extract feature maps\
5.  Compute content loss\
6.  Compute style loss using Gram matrices\
7.  Combine losses\
8.  Optimize the generated image using gradient descent

The process continues until the total loss converges.

------------------------------------------------------------------------

## Tech Stack

Python\
PyTorch\
Torchvision\
PIL\
Matplotlib

------------------------------------------------------------------------

## Key Learnings

Understanding feature representations in convolutional neural networks\
Difference between spatial structure and texture representation\
Role of Gram Matrix in capturing style\
Balancing content and style weights\
Optimization-based image generation

------------------------------------------------------------------------

## Possible Improvements

Implement Fast Style Transfer using feedforward networks\
Experiment with different style weights\
Use different pretrained backbones\
Add a web interface using Flask or FastAPI
