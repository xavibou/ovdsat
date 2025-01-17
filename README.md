# Exploring Robust Features for Few-Shot Object Detection in Satellite Imagery

[Xavier Bou](https://xavibou.github.io/), [Gabriele Facciolo](http://gfacciol.github.io/), [Rafael Grompone](https://scholar.google.fr/citations?user=GLovf4UAAAAJ&hl=en), [Jean-Michel Morel](https://sites.google.com/site/jeanmichelmorelcmlaenscachan/), [Thibaud Ehret](https://tehret.github.io)

Centre Borelli, ENS Paris-Saclay

---

[![arXiv](https://img.shields.io/badge/paper-arxiv-brightgreen)]()
[![Google Drive](https://img.shields.io/badge/files-Google_Drive-blueviolet)](https://drive.google.com/drive/folders/1g3JhJivPlmpCfggAAJoiZPJDOIBeJR5J?usp=sharing)
[![Project](https://img.shields.io/badge/project%20web-github.io-red)]()

This repository is the official implementation of the paper [Exploring Robust Features for Few-Shot Object Detection in Satellite Imagery](https://arxiv.org/abs/2403.05381).

---

## **News**
🎉 **Our Paper Has Been Accepted to the EarthVision Workshop at CVPR24!** 🌍  
🚀 **Initialized and trained prototypes have been added to Google Drive directory**  

---

The goal of this paper is to perform object detection in satellite imagery with only a few examples, thus enabling users to specify any object class with minimal annotation. To this end, we explore recent methods and ideas from open-vocabulary detection for the remote sensing domain. We develop a few-shot object detector based on a traditional two-stage architecture, where the classification block is replaced by a prototype-based classifier. A large-scale pre-trained model is used to build class-reference embeddings or prototypes, which are compared to region proposal contents for label prediction. In addition, we propose to fine-tune prototypes on available training images to boost performance and learn differences between similar classes, such as aircraft types. We perform extensive evaluations on two remote sensing datasets containing challenging and rare objects. Moreover, we study the performance of both visual and image-text features, namely DINOv2 and CLIP, including two CLIP models specifically tailored for remote sensing applications. Results indicate that visual features are largely superior to vision-language models, as the latter lack the necessary domain-specific vocabulary. Lastly, the developed detector outperforms fully supervised and few-shot methods evaluated on the SIMD and DIOR datasets, despite minimal training parameters.

![Alt text](./assets/teaser_plot_v3.png)

## Contents

1. [Overview](#Overview)
1. [Requirements](#Requirements)
1. [Data preparation](#Data-preparation)
1. [Create prototypes](#Create-prototypes)
1. [Fine-tune prototypes](#Fine-tune-prototypes)
1. [Evaluate](#Evaluate)
1. [Citation](#Citation)
1. [License and Acknowledgement](#License-and-Acknowledgement)

### Overview
![Alt text](./assets/detector_inference_v1.png)

Detect your desired objects in optical remote sensing data via a few simple steps:
1. Prepare the data with N labelled examples per category (we provide examples for N={5, 10, 30})
1. Create class-reference prototypes and background prototypes
1. Fine-tune class-reference embeddings
1. Detect objects via RPN and the learned embeddings!

### Requirements:
Create a conda environment and install the required packages as follows. You might need to adapt versions of packages depending on your hardware:
```Shell
  conda create -n ovdsat python=3.9 -y
  conda activate ovdsat
  pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
  python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  pip install opencv-python albumentations transformers
```

### Data preparation and weights
To set up the data and pre-trained weights, download the contents of the following [Google Drive folder](https://drive.google.com/drive/folders/1g3JhJivPlmpCfggAAJoiZPJDOIBeJR5J?usp=sharing). We provide the same splits and labels we use in our article for the SIMD dataset (N = {5, 10, 30}). Add the data/ and weights/ directories into the project directory. The data path should follow the structure below for each dataset, e.g. simd, dior or your own:
```plaintext
data/
│
├── simd/
│   ├── train_coco_subset_N5.json
│   ├── train_coco_subset_N10.json
│   ├── train_coco_subset_N30.json
│   ├── train_coco_finetune_val.json
│   ├── val_coco.json
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── dior/
│   ├── train_coco_subset_N5.json
│   ├── train_coco_subset_N10.json
│   ├── train_coco_subset_N30.json
│   └── ...
...
```
Notice that the train_coco_finetune_val.json corresponds to a small subset of training data from the used dataset that is used as validation set during training, so that no real validation data is used as such.

#### Weights

We pre-trained a FasterRCNN model on DOTA for the RPN using the code from [DroneDetectron2](https://github.com/akhilpm/DroneDetectron2). The pre-trained checkpoints can be found in the [Google Drive directory](https://drive.google.com/drive/folders/1g3JhJivPlmpCfggAAJoiZPJDOIBeJR5J?usp=sharing). If you plan to use any of the Remote Sensing CLIP models tested in the paper, download the pre-trained weights ([RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP/tree/main) and [GeoRSClip](https://huggingface.co/Zilun/GeoRSCLIP)) and add them to the weights/ directory.

### Create prototypes
To generate the class-reference and background prototypes using [DINOv2](https://github.com/facebookresearch/dinov2) features, run the following command:
```Shell
bash scripts/init_prototypes.sh
```
**Important:** Add the path to your data and the in the DATA_DIR variable in the bash files. You can adapt the used datasets, value of N as well. If you are running other data or the files/paths differ from ours, you can adapt the contents of the bash file to your own structure.

### Fine-tune prototypes
Train the pre-initialised class-reference prototypes on the available data:
```Shell
bash scripts/train_prototypes_bbox.sh
```

### Evaluate
Evaluate the learned prototypes on unsen data:
```Shell
bash scripts/eval_detection.sh
```

### Citation
If you found our work useful, please cite it as follows:
```bibtex
@article{Bou:2024,
  title={Exploring Robust Features for Few-Shot Object Detection in Satellite Imagery},
  author={Bou, Xavier and Facciolo, Gabriele and von Gioi, Rafael Grompone and Morel, Jean-Michel and Ehret, Thibaud},
  journal={arXiv preprint arXiv:2403.05381},
  year={2024}
}
```

### License and Acknowledgement

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE) - see the [LICENSE](LICENSE) file for details.