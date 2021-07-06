<div align="center" markdown>
<img src="https://i.imgur.com/UvRU16x.png"/>

# Train MMClassification

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/mmclassification/supervisely/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/mmclassification)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/train&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/serve&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/mmclassification/supervisely/train&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview

Train models from MMClassification toolbox on your custom data (Supervisely format is supported). 
- Configure Train / Validation splits, model architecture and training hyperparameters
- Visualize and validate training data 
- App automatically generates training py configs in MMClassification format
- Run on any computer with GPU (agent) connected to your team 
- Monitor progress, metrics, logs and other visualizations withing a single dashboard

Watch [how-to video](https://youtu.be/R9sbH3biCmQ) for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/R9sbH3biCmQ" data-video-code="R9sbH3biCmQ">
    <img src="https://i.imgur.com/O47n1S1.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>


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
   
3. Preview all available tags in project with corresponding image examples. Select training tags (model will be trained to predict them).
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
     
6. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory is generated in UI after training. 
   
   Save path is the following: ```"/mmclassification/<task id>_<input project name>/```

   For example: ```/mmclassification/5886_synthetic products v2/```
   
6. Also in this directory you can find file `open_app.lnk`. It is a link to the finished UI session. It can be opened at any time to 
   get more details about training: options, hyperparameters, metrics and so on.

   <img src="https://i.imgur.com/BVtNo7E.png"/>
   
   - go to `Team Files`
   - open directory with training artifacts
   - right click on file `open_app.lnk`
   - open







# Screenshot

<img src="https://i.imgur.com/eiROUgb.png"/>
