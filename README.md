# Dade: Delay-adaptive Detector for Streaming Perception

[Arxiv]
arXiv
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.11558)
## Benchmark
|Model  | sAP<br>0.5:0.95 | sAP50 |sAP75| weights | COCO pretrained weights |
| ------        |:---:     |:---:  | :---: | :----: | :----: |
|[DaDe-l](./cfgs/l_s50_onex_dade_tal_filp.py)    |36.7     |57.9 | 37.3 |[github](https://github.com/yancie-yjr/StreamYOLO/releases/download/0.1.0rc/l_s50_one_x.pth) |[github](https://github.com/yancie-yjr/StreamYOLO/releases/download/0.1.0rc/yolox_l.pth) |

## Dataset Preparation
This implementation is built upon StreamYOLO

Download Argoverse-1.1 full dataset and annotation from HERE and unzip it.

The folder structure should be organized as follows before our processing.
'''shell
dade
├── exps
├── tools
├── yolox
├── data
│   ├── Argoverse-1.1
│   │   ├── annotations
│   │       ├── tracking
│   │           ├── train
│   │           ├── val
│   │           ├── test
│   ├── Argoverse-HD
│   │   ├── annotations
│   │       ├── test-meta.json
│   │       ├── train.json
│   │       ├── val.json
'''

## Environment Setup
'''
# basic python libraries
conda create --name dade python=3.7

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install yolox==0.3
git clone [git@github.com:yancie-yjr/StreamYOLO.git](https://github.com/danjos95/DADE.git)

cd DaDe

# add local path to PYTHONPATH and add this line to ~/.bashrc or ~/.zshrc (change the file accordingly)
ADDPATH=$(pwd)
echo export PYTHONPATH=$PYTHONPATH:$ADDPATH >> ~/.bashrc
source ~/.bashrc

# Installing `mmcv` for the official sAP evaluation:
# Please replace `{cu_version}` and ``{torch_version}`` with the versions you are currently using.
# You will get import or runtime errors if the versions are incorrect.
pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
'''

## Train
Step1. Build symbolic link to Argoverse-HD dataset.
'''
cd <StreamYOLO_HOME>
ln -s /path/to/your/Argoverse-1.1 ./data/Argoverse-1.1
ln -s /path/to/your/Argoverse-HD ./data/Argoverse-HD
'''
Step2. Train model on Argoverse:
'''
python tools/train.py -f cfgs/l_s50_onex_dade_tal_filp.py -d 8 -b 32 -c [/path/to/your/coco_pretrained_path] -o --fp16
'''
-d: number of gpu devices.
-b: total batch size, the recommended number for -b is num-gpu * 8.
--fp16: mixed precision training.
-c: model checkpoint path.
## Online Evaluation
We modify the online evaluation from sAP
'''
cd sAP/dade
. dade_l_streamyolo.sh
'''
