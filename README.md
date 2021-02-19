# RAGA
Relation-aware Graph Attention Networks for Global Entity Alignment

## Datasets
Please download the datasets [here](https://drive.google.com/drive/folders/1SN3JAV3clMMUPQ0M6LTJQ4GZ8JFLTy0s?usp=sharing) and extract them into root directory.

## Environment

```
apex
pytorch
torch_geometric
```

## Running

For local alignment, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang zh_en
```

For local and global alignment, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang zh_en --stable_test
```
