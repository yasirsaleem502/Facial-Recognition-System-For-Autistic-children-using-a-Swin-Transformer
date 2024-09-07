
# Facial Recognition System for Autistic Children Using Swin Transformer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Deep Learning](https://img.shields.io/badge/Method-Deep%20Learning-blue)
![Swin Transformer](https://img.shields.io/badge/Model-Swin%20Transformer-green)
![Image Processing](https://img.shields.io/badge/Method-Image%20Processing-brightgreen)
![Computer Vision](https://img.shields.io/badge/Field-Computer%20Vision-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

This project aims to develop a **Facial Recognition System for Autistic Children** using the state-of-the-art **Swin Transformer** model. The system is designed to help recognize facial expressions and emotions of autistic children, assisting caregivers and educators in better understanding their emotional states and reactions.The goal is to develop a facial recognition model that can accurately identify and classify images of children's faces as either autistic or non-autistic based on facial features.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Swin Transformer Architecture](#swin-transformer-architecture)
- [Model Results](#model-results)
- [Contact](#contact)

## Introduction

Facial recognition for autistic children can be challenging due to atypical expressions and unique behavioral patterns. This project leverages the **Swin Transformer**, a hierarchical vision transformer, to accurately capture and interpret these unique expressions, providing a robust solution for emotional recognition in real-time applications.

## Features

- **Accurate Recognition**: Utilizes Swin Transformer for high accuracy in facial recognition
- **Adaptable to Various Environments**: Robust performance across different lighting and background conditions.
- **Customizable and Extensible**: Easily customizable model architecture for specific needs.

## Dataset

The Autistic Children Facial Dataset is a specialized collection of images intended for research and development in the field of autism detection through facial recognition technology. This dataset comprises photographs and images specifically focusing on the facial features of children diagnosed with autism spectrum disorder (ASD). The dataset is curated to include images of both autistic and neurotypical (non-autistic) children, providing a comparative basis for analysis.
- [Link to Dataset] https://www.kaggle.com/datasets/imrankhan77/autistic-children-facial-data-set

## Swin Transformer Architecture

The Swin Transformer (Shifted Window Transformer) is a hierarchical vision transformer model designed for image recognition tasks. It addresses some limitations of conventional transformers, particularly in handling high-resolution images and achieving computational efficiency.

1. ## Overall Structure
   
Hierarchical Design: The Swin Transformer is built with a hierarchical approach, similar to convolutional neural networks (CNNs). It processes images in stages with progressively reduced resolution and increasing feature dimensions.

Patch Partitioning: The input image is initially divided into non-overlapping patches (similar to tokens in NLP). Each patch is flattened and linearly embedded into a feature vector.

Patch Merging: As the model progresses through different stages, patches are merged, reducing spatial resolution while increasing the depth of feature channels.

2. ## Key Components
Patch Splitting and Embedding:

The input image is split into fixed-size patches (e.g., 4x4 or 7x7).
Each patch is flattened into a vector, which is then linearly embedded into a higher-dimensional space.
Shifted Window Multi-Head Self-Attention (SW-MSA):

Window-Based Self-Attention: Instead of applying self-attention globally, Swin Transformer restricts it to local windows. This reduces computational complexity significantly.
Shifted Windows: In alternate layers, the windows are shifted by a certain number of pixels. This mechanism allows cross-window connections, enhancing the model’s ability to capture long-range dependencies while maintaining efficiency.
Multi-Head Self-Attention (MSA):

Within each window, the model applies standard multi-head self-attention, allowing it to learn complex relationships within local patches.
Patch Merging Layers:

After a certain number of transformer blocks, adjacent patches are merged, reducing the spatial dimension by half and doubling the channel dimension.
This operation enables the model to construct a hierarchical feature representation, similar to the down-sampling layers in CNNs.
MLP (Multi-Layer Perceptron) Block:

After each attention operation, features pass through an MLP block, which consists of two fully connected layers with a GELU activation in between.
Normalization Layers:

Layer normalization is applied before each MSA and MLP block, ensuring stable training.
Residual Connections:

Skip connections are used around each SW-MSA and MLP block to improve gradient flow and enable deeper architectures.

3. ## Stage-wise Representation
The Swin Transformer is divided into multiple stages, each consisting of a series of Swin Transformer blocks:

Stage 1: High resolution, small window size, shallow depth. It captures fine details.

Stage 2: Intermediate resolution and window size, with greater depth. It captures mid-level features.

Stage 3: Lower resolution, larger window size, deeper. It captures higher-level features.

Stage 4: Lowest resolution with the largest window size. It captures the most abstract features suitable for final classification or detection tasks.

4. ## Output
The final output of the Swin Transformer can be used in various downstream tasks, such as image classification, object detection, and segmentation.
The hierarchical nature of the model allows it to output features at multiple scales, making it adaptable for multi-scale tasks.


## Model Results

- Test Accuracy: 88.00% (264/300)

- Test Loss: 0.3321 


![image](https://github.com/user-attachments/assets/b8e2f59f-cbe5-4392-8bf2-2f8b087b272f)


## Contact

- **Author**: Muhammad Yasir Saleem
- **Email**: myasirsaleem94@gmail.com
- **LinkedIn**:https://www.linkedin.com/in/muhammad-yasir-saleem/

Feel free to reach out if you have any questions or need further information.

