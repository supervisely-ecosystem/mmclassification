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


#@TODOs: 
- short video



# How to Run

1. Add app to your team from Ecosystem
2. Be sure that you connected computer with GPU to your team by running Supervisely Agent on it 
3. Run app from context menu of project with tagged images
4. Open Training Dashboard (app UI) and follow instructions provided in the video above
5. All training artifacts (metrics, visualizations, weights, ...) are uploaded to Team Files. Link to the directory is generated in UI after training. 
   
   Save path is the following: ```"/mmclassification/<task id>_<input project name>/```

   For example: ```/mmclassification/5886_synthetic products v2/```
   
6. Also in this directory you can find file `open_app.lnk`. It is a link to the finished UI session. It can be opened at any time to 
   get more details about training: options, hyperparameters, metrics and so on.

   <img src="https://i.imgur.com/BVtNo7E.png"/>
   
   - go to `Team Files`
   - open directory with training artifacts
   - right click on file `open_app.lnk`
   - open

# How To Use

Watch short video for more details:

<a data-key="sly-embeded-video-link" href="https://youtu.be/e47rWdgK-_M" data-video-code="e47rWdgK-_M">
    <img src="https://i.imgur.com/sJdEEkN.png" alt="SLY_EMBEDED_VIDEO_LINK"  style="max-width:100%;">
</a>

- Step1. App downloads input project
- Step2. App downloads input project








# Screenshot

<img src="https://i.imgur.com/eiROUgb.png"/>
