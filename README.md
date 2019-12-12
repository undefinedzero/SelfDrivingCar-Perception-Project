# ROB535 Self Driving Car Final Project Perception Part
## Team 15 🚗

The code is based on [Dectectron2](https://github.com/facebookresearch/detectron2).

## What's inside?
We used `colab` to implement algorithms at first. We tested the performance of pre-trained models given by Detectron2 in `Task1&2.ipynb` for both Task 1 and Task 2.

`retrain.py` is for retraining.

`inference.py` is for inference.

## How to use?
For `Task1&2.ipynb`, you need to have a jupyter notebook environment.

For `retrain.py` and `inference.py`, you need to install [Dectectron2](https://github.com/facebookresearch/detectron2) first following the instructions in `Task1&2.ipynb`. Then use the command line:
```
python retrain.py
python inference.py
```