# QSFusion

Code for **QSFusion: Quality-semantic collaborative network for infrared and visible image fusion**.

## Structure

```text
QSFusion/
  train.py                  Training entry
  test.py                   Testing/inference entry
  options.py                Command-line options
  datasets.py               Paired image dataset loader
  modules/                  Network modules
  utils/                    Utility functions
  weights/                  Checkpoint
  requirements.txt          Dependencies
```

## Modules

```text
QSFusion                      Overall network
ASGModule                     Adaptive Semantic Guidance
VisibleStreamEncoder          Visible stream encoder
InfraredStreamEncoder         Infrared stream encoder
GAEBlock                      GAE block
MDTA                          Multi-DConv head transposed self-attention
CFFN                          Convolutional feed-forward network
AWVAModule                    Attention Weighted Value Aggregation
MCIModule                     Multi-Scale Cross-Dimensional Interaction
SharedSemanticEncoder         Semantic consistency branch
ReferenceFreeQualityEstimator Quality-guided branch
QSFusionLoss                  Training loss
```

## Environment

Python 3.10 is recommended.

```bash
pip install -r requirements.txt
```

## Dataset

Training and testing data should be paired by file order:

```text
dataset_root/
  train/
    vi/            # or Visible/
    ir/            # or Infrared/
  test/
    vi/
    ir/
```

For standalone testing:

```text
custom_test_data/
  vi/
    1.png
  ir/
    1.png
```

## Training

```bash
python train.py --dataset_root ./dataset/train_data --output_dir ./experiments/qsfusion
```

The best checkpoint is saved to:

```text
experiments/qsfusion_*/qsfusion_best.pth
```

## Testing

```bash
python test.py --ckpt_path ./weights/qsfusion_final.pth --vi_path ./path/to/vi --ir_path ./path/to/ir --out_path ./outputs/custom
```

## Checkpoint

```text
weights/qsfusion_final.pth
```

## License

This project is released under the [MIT License](https://github.com/Data-XiaoYu/QSFusion/blob/main/LICENSE).
