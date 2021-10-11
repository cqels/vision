# VisionKG: Vision Knowledge Graph
Official Repository of [VisionKG](https://vision.semkg.org/) by

Anh Le-Tuan, Trung-Kien Tran, Manh Nguyen-Duc, Jicheng Yuan, Manfred Hauswirth and Danh Le-Phuoc. 
## About The Project
VisionKG is an RDF-based knowledge and built upon the FAIR principles. It provides a fatastic way to interlink and integrate data across different sources and spaces (e.g. MSCOCO, Visual_Genome, KITTI, ImageNet and so on) and brings a novel way to organize your data, explore the interpretability and explainability of models. By a few lines of SPARQL, you could query your desired number of images, objects from various built-in datasets and get their annotations via our Web API and build your models in a data-centric way.


<p align="center" width="100%">
<img src="./resources/visionkg.jpg" width="800"/>
</p>

<p align="center" width="80%">
The Overview of VisionKG
</p>

## Demo for VisionKG:

[![Watch the video](https://i.imgur.com/vKb2F1B.png)](https://user-images.githubusercontent.com/87916250/136798334-8f61a296-1494-481a-88b3-edbaf74174a0.mp4)

As of August 2021, VisionKG already consists of 67 million triples, 239 thousand images, and nearly one million annotations. 
For the pre-trained models, besides the yolo series, now it also supports other one- or two-stage architectures such as EfficientDet, Faster-RCNN, and so on.

## Features

-   Query images / anotations across multi data sources using SPARQL
-   Online preview of the queried results
-   Graph-based exploration across visual label spaces
-   Interlinke and align labels under different labels spaces under shared semantic understanding 
-   Building training pipelines with mixed datasets
-   Cross-dataset validation and testing
-   Explore the interpretability and explainability of models

[Explore more about VisionKG →](https://vision.semkg.org/)

## Quick-View

VisionKG can also be integrated into many famous toolboxes. 
For that, we also provides three pipelines for image recognition and obejct detection based on VisionKG and other toolboxes.

### Object Detection:

VisionKG_meet_MMdetection: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cqels/vision/blob/main/tutorials/tutorials_detection_mmdetection.ipynb)

### Image Recognition:

VisionKG_meet_timm: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cqels/vision/blob/main/tutorials/tutorials_classification_timm.ipynb)

VisionKG_meet_MMclassification: [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cqels/vision/blob/main/tutorials/tutorials_classification_mmclassification.ipynb)

## Acknowledgements

* [MMdetection](https://github.com/open-mmlab/mmdetection)
* [MMclassification](https://github.com/open-mmlab/mmclassification)
* [timm](https://github.com/rwightman/pytorch-image-models)

## Citation

If you use VisionKG in your research, please cite our work.

```
@inproceedings{Anh:2021,
  author    = {Anh, Le-Tuan and Manh, Nguyen-Duc and Jicheng, Yuan and 
               Trung Kien-Tran and
               Manfred, Hauswirth and Danh, Le-Phuoc
               },
  title     = {VisionKG: Towards A Unified Vision Knowledge Graph},
  booktitle = {Proceedings of the {ISWC} 2021 Posters {\&} Demonstrations Track},
  series    = {{CEUR} Workshop Proceedings},
  year      = {2021},
  }
```
