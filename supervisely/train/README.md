<div align="center" markdown>
<img src="https://user-images.githubusercontent.com/106374579/186635069-20af9595-af3d-4765-9517-c2df3c29fc1a.png"/>


# Train MMClassification

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#Updates">Updates</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmclassification)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/mmclassification/supervisely/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/mmclassification/supervisely/train.png)](https://supervise.ly)

</div>

# Overview

Train models from [MMClassification](https://github.com/open-mmlab/mmclassification) toolbox on your custom data (Supervisely format is supported). 
- Configure Train / Validation splits, model architecture and training hyperparameters
- Visualize and validate training data 
- App automatically generates training py configs in MMClassification format
- Run on any computer with GPU (agent) connected to your team 
- Monitor progress, metrics, logs and other visualizations withing a single dashboard

Watch [how-to video](https://youtu.be/R9sbH3biCmQ) for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/R9sbH3biCmQ" data-video-code="R9sbH3biCmQ">
    <img src="https://i.imgur.com/O47n1S1.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

# Updates

### v1.2.0

By default, classification model trains on the tagged images. There are cases, when user need to use tagged objects as training examples. To cover this scenario, we added additional mode to training dashboard. Now user can run training on images or objects crops. If user selects `training on objects` mode, then the additional settings with preview will be available. It means that user dont't need to run [Crop objects on images](https://ecosystem.supervise.ly/apps/crop-objects-on-images) app before training and prepare temporary project with objects crops, now it will be done automatically in training dashboard. 

Here is the UI screenshot with settings and preview if `training on objects` mode is selected:

<img src="https://github.com/supervisely-ecosystem/mmclassification/releases/download/v0.0.1/train-objects-mode.png" style="width:150%;"/>

Other features, like saving image examples for every class for trained classification model also supports new mode, technically it is achieved by saving images with other training artifacts (like checkpoints and metrics) in resulting directory in `Team Files`. [Serve MMClassification](https://app.supervise.ly/ecosystem/apps/supervisely-ecosystem%252Fmmclassification%252Fsupervisely%252Fserve) app can correctly use them with other inference applications from ecosystem: 

<img src="https://github.com/supervisely-ecosystem/mmclassification/releases/download/v0.0.1/train-resulting-dir.png">

# How to Run
1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it ([how-to video](https://youtu.be/aDqQiYycqyk))
3. Run app from context menu of project with tagged images
<img src="https://i.imgur.com/qz7IsXF.png"/>
4. Open Training Dashboard (app UI) and follow instructions provided in the video above


# How To Use
1. App downloads input project from Supervisely Instance to the local directory
2. Define train / validation splits
   - Randomly
   <img src="https://i.imgur.com/mwcos1I.png"/>
     
   - Based on image tags (for example "train" and "val", you can assign them yourself)
   <img src="https://i.imgur.com/X9mnuRK.png"/>
   
   - Based on datasets (if you want to use some datasets for training (for example "ds0", "ds1", "ds3") and 
     other datasets for validation (for example "ds_val"), it is completely up to you)
     <img src="https://i.imgur.com/Y956BvC.png"/>
   
3. Preview all available tags with corresponding image examples. Select training tags (model will be trained to predict them).
   <img src="https://i.imgur.com/g7eY0AC.png"/>

4. App validates data consistency and correctness and produces report.
   <img src="https://i.imgur.com/AHExs93.png"/>

5. Select how to augment data. All augmentations performed on the fly during training. 
   - use one of the predefined pipelines
   <img src="https://i.imgur.com/tJpY1uc.png"/>
   - or use custom augs.  To create custom augmentation pipeline use app 
   "[ImgAug Studio](https://ecosystem.supervise.ly/apps/imgaug-studio)" from Supervisely Ecosystem. This app allows to 
   export pipeline in several formats. To use custom augs just provide the path to JSON config in team files.
     <a data-key="sly-embeded-video-link" href="https://youtu.be/ZkZ7krcKq1c" data-video-code="ZkZ7krcKq1c">
         <img src="https://i.imgur.com/HFEhrdl.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
     </a>
   - preview selected augs on the random image from project
     <img src="https://i.imgur.com/TwCJnmv.png"/>
     
6. Select model and how weights should be initialized
   - pretrained on imagenet
   <img src="https://i.imgur.com/LppcO7C.png"/>
   - custom weights, provide path to the weights file in team files
   <a data-key="sly-embeded-video-link" href="https://youtu.be/XU9vCwHh9_g" data-video-code="XU9vCwHh9_g">
     <img src="https://i.imgur.com/1YHXLty.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
   </a>
     
7. Configure training hyperparameters
   <img src="https://i.imgur.com/IW5ywEo.png"/>

8. App generates py configs for MMClassification toolbox automatically. Just press `Generate` button and move forward. 
   You can modify configuration manually if you are advanced user and understand MMToolBox.
   <img src="https://i.imgur.com/a87AR7A.png"/>

9. Start training
   <img src="https://i.imgur.com/NsPUbyF.png"/>
   
10. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory 
    is generated in UI after training. 
   
   Save path is the following: ```"/mmclassification/<task id>_<input project name>/```

   For example: ```/mmclassification/5886_synthetic products v2/```
   
   Structure is the following:
   ```
   . 
   ├── checkpoints
   │   ├── 20210701_113427.log
   │   ├── 20210701_113427.log.json
   │   ├── best_accuracy_top-1_epoch_44.pth
   │   ├── epoch_48.pth
   │   ├── epoch_49.pth
   │   ├── epoch_50.pth
   │   └── latest.pth
   ├── configs
   │   ├── augs_config.json
   │   ├── augs_preview.py
   │   ├── dataset_config.py
   │   ├── model_config.py
   │   ├── runtime_config.py
   │   ├── schedule_config.py
   │   └── train_config.py
   ├── info
   │   ├── gt_labels.json
   │   ├── tag2urls.json
   │   └── ui_state.json
   └── open_app.lnk
   ```
- `checkpoints` directory contains MMClassification logs and weights
- `configs` directory contains all configs that app generated for MMClassification toolbox, they may be useful
for advanced user who would like ot export models and use them outside Supervisely
- info directory contains basic information about training
   - `gt_labels.json` - mapping between class names and their indices, this file allows to understand NN predictions. For examples:
   ```json
  {
      "cat": 0,
      "dog": 1,
      "bird": 2,
      "frog": 3
  }
   ```
  - `tag2urls.json` - for every tag some image examples were saved, this file is used when the model is integrated into labeling interface
   ```json
  {
     "cat": [
          "http://supervisely.private/a/b/c/01.jpg",
          "http://supervisely.private/a/b/c/02.jpg"
     ],
     "dog": [
         "http://supervisely.private/d/d/d/01.jpg",
         "http://supervisely.private/d/d/d/02.jpg"
     ],
     "bird": [
         "http://supervisely.private/c/c/c/01.jpg",
         "http://supervisely.private/c/c/c/02.jpg"
     ],
     "frog": [
         "http://supervisely.private/c/c/c/01.jpg",
         "http://supervisely.private/c/c/c/02.jpg"
     ]
  }
   ```
  - `ui_state.json` file with all values defined in UI
   ```json
   {
      "...": "...",
      "epochs": 50,
      "gpusId": "0",
      "imgSize": 256,
      "batchSizePerGPU": 64,
      "workersPerGPU": 3,
      "valInterval": 1,
      "metricsPeriod": 10,
      "checkpointInterval": 1,
      "maxKeepCkptsEnabled": true,
      "maxKeepCkpts": 3,
      "saveLast": true,
      "saveBest": true,
      "optimizer": "SGD",
      "lr": 0.001,
      "momentum": 0.9,
      "weightDecay": 0.0001,
      "....": "..."
   }
  ```   

- `open_app.lnk` - link to finished session, you can open finished training dashboard at any time and check all settings and visualizations   
   <img src="https://i.imgur.com/BVtNo7E.png"/>
  
