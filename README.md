
# Super Interpolation using Neural Networks

This project aims to beat bicubic interpolation for super resolution using state-of-the-art neural network architectures and techniques.
So far it (successfully) implements [EDSR](https://arxiv.org/abs/1707.02921) model. Pipeline supports patch-based training, full-image evaluation, inference, pretraining and ensembling.

Training, evaluation and inference are ran by meta-scripts `train.py`, `eval.py` and `inference.py` that are in the root directory. We maintain this structure for fast prototyping and changing of architectures.

## Dataset Structure
We used following datasets:
- DIV2K bicubic LR train (only dataset used for training)
- DIV2K bicubic LR valid
- Urban100
- BSD100
- Set5
- Set14

Your dataset directories should follow this format:

```
data/
├── DIV2K-train/
│   ├── X2/
│   │   ├── LR/
│   │   └── HR/
│   └── ...
├── DIV2K-val/
│   ├── X2/
│   │   ├── LR/
│   │   └── HR/
│   └── ...
├── Urban100/
│   ├── X2/
│   │   ├── LR/
│   │   └── HR/
│   └── ...
└── ...
```
Augmented datasets (if generated) maintain the same folder format.

## Training

To train a model:

```bash
python train.py EDSR \
  --train_dir data/DIV2K-train \
  --val_dir data/DIV2K-val \
  --scale 2 \
  --run_dir runs
  ...
```

### Additional Options:

- `--pretrain path/to/checkpoint.pt` – load weights but reset tail (useful for transferring to new scale)
- `--resume path/to/checkpoint.pt` – resume full training

### Author Hyperparameters (from original EDSR paper):

- **Baseline Model**:
  - `n_blocks=16`, `n_features=64`, `res_scale=1.0`
- **Upscale Model (EDSR Large)**:
  - `n_blocks=32`, `n_features=256`, `res_scale=0.1`

We use preprocessed random augmentations (crop, flip, rotate) saved to disk for reproducibility.

## Evaluation
Evaluate model performance on full-image or patch-based validation:

```bash
python eval.py EDSR \
  --checkpoint runs/.../checkpoints/best.pt \
  --config runs/.../config.json \
  --data_dir data/Set14 \
  --use_ensemble
```

## Inference

```bash
python inference.py EDSR \
  --checkpoint runs/.../checkpoints/best.pt \
  --config runs/.../config.json \
  --img path/to/input.png \
  --out output.png \
  --use_ensemble
```

## Citation To Original Authors

```
@misc{lim2017enhanceddeepresidualnetworks,
      title={Enhanced Deep Residual Networks for Single Image Super-Resolution}, 
      author={Bee Lim and Sanghyun Son and Heewon Kim and Seungjun Nah and Kyoung Mu Lee},
      year={2017},
      eprint={1707.02921},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1707.02921}, 
}
```