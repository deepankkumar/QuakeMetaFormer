# Quakemetaformer

Quakemetaformer is an extension built on top of [Metaformer](https://github.com/dqshuai/MetaFormer) that implements the transformer-based architecture for earthquake damage assessment. This repository contains the code and modifications implimented for the paper "Multiclass Post-Earthquake Building Assessment Integrating Optical and SAR Satellite Imagery, Ground Motion, and Soil Data with Transformers".

## Overview

Quakemetaformer integrates both satellite imagery and complementary metadata (e.g., ground motion, soil data, SAR-derived indices) into the framework to achieve improved performance in multiclass post-earthquake building damage assessment.

## Installation

For installation of the required libraries and dependencies, please refer to the main branch of the original Metaformer repository. You can find the detailed installation instructions there, which include:

#### python module
* install `Pytorch and torchvision`
```
pip install torch==1.5.1 torchvision==0.6.1
```
* install `timm`
```
pip install timm==0.4.5
```
* install `Apex`
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* install other requirements
```
pip install opencv-python==4.5.1.48 yacs==0.1.8
```
