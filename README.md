# Continual LayerNorm Tuning
Official Repository of "[On the Effectiveness of LayerNorm Tuning for Continual Learning in Vision Transformers](https://arxiv.org/abs/2308.09610)" (Visual Continual Learning Workshop ICCV 2023). \
This repository is based on https://github.com/JH-LEE-KR/dualprompt-pytorch.

## Installation
Create and activate a conda environment with Python 3.8:
```
$ conda create -n cln python=3.8
$ conda activate cln
```
Install requirements:
```
$ pip install -r requirements.txt
```

## Datasets and Pre-trained weights
Both are automatically downloaded at training time.

## Training
Both algorithm variants can be trained by simply running the corresponding script. \
To train the Two-Stage variant run:
```
$ ./train_two.sh
```
To train the Single-Stage variant run:
```
$ ./train_single.sh
```
The code supports wandb. Activate it by adding `--wandb` to the bash script. In `engine.py`, change `entity` according to your wandb id.
> Be sure to log into wandb before running with the `--wandb` flag.

## Disclaimer
The code should support DDP, however, we did not test it as it is unnecessary to run on multiple GPUs.
That being said, the DDP support is inherited from https://github.com/JH-LEE-KR/dualprompt-pytorch.

The code does not store trained weights, thus a proper code must be written to store and load weights.

For any questions, please get in touch with us at thomas.demin@unitn.it or open an issue.

## Citation
```
@inproceedings{de2023effectiveness,
  title={On the Effectiveness of LayerNorm Tuning for Continual Learning in Vision Transformers},
  author={De Min, Thomas and Mancini, Massimiliano and Alahari, Karteek and Alameda-Pineda, Xavier and Ricci, Elisa},
  booktitle={2023 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages={3577--3586},
  year={2023},
  organization={IEEE}
}
```
