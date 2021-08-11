# Overview 
🚀 This collection is designed to cover classification task in 
[**Supervisely**](https://supervise.ly/). Before using these apps 
we recommend to try end-to-end working demo (retail labeling use case) - data and explanations are provided.

# Table of Contents

1. [About Supervisely](#-about-supervisely)
2. [Prerequisites](#Prerequisites)
3. [Apps Collection for Classification](#-apps-collection-for-classification)
    - [Demo data and synthetic data generation](#demo-data-and-synthetic-data)
    - [Neural Networks](#neural-networks)
    - [Integration into labeling tool](#integration-into-labeling-tool)
    - [Auxiliary apps](#auxiliary-apps)
8. [For developers](#For-developers)
9. [Contact & Questions & Suggestions](#contact--questions--suggestions)

# 🔥 About Supervisely

You can think of [Supervisely](https://supervise.ly/) as an Operating System available via Web Browser to help you solve Computer Vision tasks. The idea is to unify all the relevant tools that may be needed to make the development process as smooth and fast as possible. 

More concretely, Supervisely includes the following functionality:
 - Data labeling for images, videos, 3D point cloud and volumetric medical images (dicom)
 - Data visualization and quality control
 - State-Of-The-Art Deep Learning models for segmentation, detection, classification and other tasks
 - Interactive tools for model performance analysis
 - Specialized Deep Learning models to speed up data labeling (aka AI-assisted labeling)
 - Synthetic data generation tools
 - Instruments to make it easier to collaborate for data scientists, data labelers, domain experts and software engineers

One challenge is to make it possible for everyone to train and apply SOTA Deep Learning models directly from the Web Browser. To address it, we introduce an open sourced Supervisely Agent. All you need to do is to execute a single command on your machine with the GPU that installs the Agent. After that, you keep working in the browser and all the GPU related computations will be performed on the connected machine(s).


# Prerequisites
You should connect computer with GPU to your Supervisely account. If you already have Supervisely Agent running on your computer, you can skip this step.

 Several tools have to be installed on your computer:

- Nvidia drives + [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Docker](https://docs.docker.com/engine/install/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

Once your computer is ready just add agent to your team and execute automatically generated running command in terminal. Watch how-to video:

<a data-key="sly-embeded-video-link" href="https://youtu.be/aDqQiYycqyk" data-video-code="aDqQiYycqyk">
    <img src="https://i.imgur.com/X9NTc5X.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="width:50%;">
</a>


# 🎉 Apps Collection for classification

To learn more about how to use every app, please go to app's readme page (links are provided). Just add the apps to your team to start using them.

<img src="https://i.imgur.com/xc8Q2dt.png"/>

Collection consists of the following apps: 

## Demo data and synthetic data
- [Snacks catalog](https://ecosystem.supervise.ly/projects/snacks-catalog) - catalog of 83 
  snack items. All products are labeled and tagged. Will be used to generate synthetic training 
  dataset for object classification model  

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/snacks-catalog" src="https://i.imgur.com/Jc6wZSJ.png" width="350px"/>

- [Grocery store shelves](https://ecosystem.supervise.ly/projects/grocery-store-shelves) - images 
  with products on shelves, will be used to test classification model on real data

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/grocery-store-shelves" src="https://i.imgur.com/Mqqqs4c.png" width="350px"/>

- [Synthetic retail products](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fyolov5%252Fsupervisely%252Ftrain) - 
  app generates synthetic images for product classification from only a few labeled examples 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/synthetic-retail-products" src="https://i.imgur.com/vkPOCER.png" width="350px"/>

## Neural networks

[OpenMMLab](https://openmmlab.com/) is building great deep leaning toolboxes for different kind of tasks in computer vision. 
In this collections we completely integrated [MMClassification](https://github.com/open-mmlab/mmclassification) toolbox into Supervisely.

- [Train MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fmmclassification%252Fsupervisely%252Ftrain) - 
  start training on your custom data. Just run app from the context menu of your project, 
  choose tags of interest, train/val splits, configure training metaparameters and 
  augmentations, and monitor training metrics in realtime. App automatically validates 
  training data and perform augmentations on the fly. All training artifacts including 
  model weights will be saved to Team Files and can be easily downloaded. Conigs for 
  MMClassification also saved and can be used to train models outside Supervisely (advanced usage) 
  
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/train" src="https://i.imgur.com/mXG6njU.png" width="350px"/>
    
  The following models are available:

|         Model         | Params(M) | Flops(G) | Top-1 (%) | Top-5 (%) |
|:---------------------:|:---------:|:--------:|:---------:|:---------:|
| VGG-11 | 132.86 | 7.63 | 68.75 | 88.87 |
| VGG-13 | 133.05 | 11.34 | 70.02 | 89.46 | 
| VGG-16 | 138.36 | 15.5 | 71.62 | 90.49 | 
| VGG-19 | 143.67 | 19.67 | 72.41 | 90.80 | 
| VGG-11-BN | 132.87 | 7.64 | 70.75 | 90.12 | 
| VGG-13-BN | 133.05 | 11.36 | 72.15 | 90.71 | 
| VGG-16-BN | 138.37 | 15.53 | 73.72 | 91.68 | 
| VGG-19-BN | 143.68 | 19.7 | 74.70 | 92.24 | 
| ResNet-18             | 11.69     | 1.82     | 70.07 | 89.44 | 
| ResNet-34             | 21.8      | 3.68     | 73.85 | 91.53 | 
| ResNet-50             | 25.56     | 4.12     | 76.55 | 93.15 | 
| ResNet-101            | 44.55     | 7.85     | 78.18 | 94.03 | 
| ResNet-152            | 60.19     | 11.58    | 78.63 | 94.16 | 
| ResNeSt-50*           | 27.48     | 5.41     | 81.13 | 95.59 | 
| ResNeSt-101*          | 48.28     | 10.27    | 82.32 | 96.24 | 
| ResNeSt-200*          | 70.2      | 17.53    | 82.41 | 96.22 | 
| ResNeSt-269*          | 110.93    | 22.58    | 82.70 | 96.28 | 
| ResNetV1D-50          | 25.58     | 4.36     | 77.54  | 93.57 | 
| ResNetV1D-101         | 44.57     | 8.09     | 78.93 | 94.48 | 
| ResNetV1D-152         | 60.21     | 11.82    | 79.41 | 94.7 | 
| ResNeXt-32x4d-50      | 25.03     | 4.27     | 77.90 | 93.66 | 
| ResNeXt-32x4d-101     | 44.18     | 8.03     | 78.71  | 94.12 | 
| ResNeXt-32x8d-101     | 88.79     | 16.5     | 79.23 | 94.58 | 
| ResNeXt-32x4d-152     | 59.95     | 11.8     | 78.93 | 94.41 | 
| SE-ResNet-50          | 28.09     | 4.13     | 77.74 | 93.84 | 
| SE-ResNet-101         | 49.33     | 7.86     | 78.26 | 94.07 | 
| ShuffleNetV1 1.0x (group=3)   | 1.87      | 0.146    | 68.13 | 87.81 | 
| ShuffleNetV2 1.0x     | 2.28      | 0.149    | 69.55 | 88.92 | 
| MobileNet V2          | 3.5       | 0.319    | 71.86 | 90.42 | 
| ViT-B/16*             | 86.86     | 33.03    | 84.20 | 97.18 | 
| ViT-B/32*             | 88.3      | 8.56     | 81.73 | 96.13 | 
| ViT-L/16*             | 304.72    | 116.68   | 85.08 | 97.38 | 
| ViT-L/32*             | 306.63    | 29.66    | 81.52 | 96.06 | 
| Swin-Transformer tiny |   28.29   |   4.36   | 81.18 | 95.61 | 
| Swin-Transformer small|   49.61   |   8.52   | 83.02 | 96.29 | 
| Swin-Transformer base |   87.77   |  15.14   | 83.36 | 96.44 | 

- [Serve MMClassification](https://ecosystem.supervise.ly/apps/supervisely-ecosystem%252Fmmclassification%252Fsupervisely%252Fserve) - 
   serve model as Rest API service. You can run custom model weights trained in Supervisely. 
   Thus other apps from Ecosystem can get predictions from the deployed model. Also developers 
   can send inference requiests in a few lines of python code or use source code as an example 
   how to load model and apply it to image and how to interpret model predictions. 
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/mmclassification/supervisely/serve" src="https://i.imgur.com/CU8XHdQ.png" width="350px"/>

## Integration into labeling tool

- [AI assisted classification](https://ecosystem.supervise.ly/apps/ai-assisted-classification) - 
  app connects to deployed classification model, shows model info and classes (with image examples),
  allows to get top N predictions from model for image / object in real time, also can be used to 
  review attached tags to perform quality assurance. It significantly speeds up labeling time especially
  when labelers wotk with large number of classes and can be also used for pre labeling.
   
    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/ai-assisted-classification" src="https://i.imgur.com/bKqVnZQ.png" width="350px"/>

## Auxiliary apps

- [Tags co-occurrence matrix](https://ecosystem.supervise.ly/apps/tags-co-occurrence-matrix) - 
  helps to explore tags on images and find some collisions in data before you train image 
  classification model  

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/tags-co-occurrence-matrix" src="https://i.imgur.com/YAWNGSt.png" width="350px"/>

- [Unpack key-value tags](https://ecosystem.supervise.ly/apps/unpack-key-value-tags) - 
  if you label images with key-value tag (single tag with multiple possible values) you can convert this tags to tags 
  without value (for example, `fruit`:`lemon` to a unique tag `fruit_lemon` and then use created project to train 
  classification model)  

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/unpack-key-value-tags" src="https://i.imgur.com/OB90OrK.png" width="350px"/>

- [Copy image tags to objects](https://ecosystem.supervise.ly/apps/copy-image-tags-to-objects) - 
  may be helpful when you tagged images and then labeled 
  objects on them and would like to assign to image tag to all objects on image 
  (for example, for generating synthetic data for retail) 

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/tags-co-occurrence-matrix" src="https://i.imgur.com/MBuF2sm.png" width="350px"/>

- [Visual tagging](https://ecosystem.supervise.ly/apps/visual-tagging) - manually assign tags 
  using image examples for visual matching, app helps labelers to 
  navigate in large and/or complex classes to avoid mistakes when classes are similar    

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/visual-tagging" src="https://i.imgur.com/HnQVi32.png" width="350px"/>

- [Tags to image URLs](https://ecosystem.supervise.ly/apps/tags-to-image-urls) - creates mapping 
  between tag name and all images with this tags and saves results to a `json` file. If you 
  trained classification model on synthetic data we recommend in training artifacts 
  directory backup original file `./info/tag2urls.json` by renaming it and replace it 
  with `json` file generated by this application. It will help labelers to visually compare 
  model predictions with images / objects by using nice-looking images instead of 
  synthetically generated.   

    <img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/tags-to-image-urls" src="https://i.imgur.com/30GDLMh.png" width="350px"/>

# For Developers
- you can use sources of [Serve MMClassification app](https://github.com/supervisely-ecosystem/mmclassification/tree/master/supervisely/serve) as example of how to prepare weights, initialize model and apply it to an image and how to correctly interpret predictions
- NN apps are based on the original [MMClassification](https://github.com/open-mmlab/mmclassification). Official updates will be synchronized with from time to time or by request.

# Contact & Questions & Suggestions

- for technical support please leave issues, questions or suggestions in our [repo](https://github.com/supervisely-ecosystem/mmclassification). Our team will try to help.
- also we can chat in slack channel [![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack) 
- if you are interested in Supervisely Enterprise Edition (EE) please send us a [request](https://supervise.ly/enterprise/?demo) or email Yuri Borisov at [sales@supervise.ly](sales@supervise.ly)
