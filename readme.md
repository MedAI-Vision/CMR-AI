# Video-based artificial intelligence for screening and diagnosis of cardiovascular diseases based on CMR: a clinical-based and proof-of-concept study.
Code used for paper: Video-based artificial intelligence for screening and diagnosis of cardiovascular diseases based on CMR: a clinical-based and proof-of-concept study.

## Introduction
We proposed video-based swin transformer (VST) as our model backbone.

## Usage
### Installation
1. This project uses Pytorch and Python3, the environment preparation please refer to [requirements.txt](https://github.com/MedAI-Vision/CMR-AI/requirements.txt) or [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer).
   
2. Download pretrained model from [VST for CMR](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth), and put it into `./checkpoints` like this:
```
│   ├── VST
│   	├── checkpoints
│   		└── swin_base_patch244_window877_kinetics600_22k.pth
│   	├── ...
```

### Data Preparation
Process data into `.nii.gz` format for all 3 modalities. 

For SAX cine, the input sample contains 3 views (eg. slice_up.nii.gz, slice_mid.nii.gz and slice_down.nii.gz), and 4CH cine and LGE both contains 1 views (eg. 4ch.nii.gz and lge.nii.gz respectively).

The annotation format is `[middle slice path of SAX cine] [4CH cine path] [LGE path]* [frame number of cine] [frame number of LGE]* [class label]` (*means optional and an example ann file is [ann_example.txt](https://github.com/MedAI-Vision/CMR-AI/ann_exsample.txt)).

### Config setting
Set or update the dataset path and other configurations in config file `config.py` .

### Training
To train a VST model for single modality CMR dataset, run:

```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```

For example, to train a VST model with 4 GPUs, run:

`CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../VST/tools/dist_train.sh config.py 4`

### Testing
To test a VST model for single or fusion modality CMR dataset, run:

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE>

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM>
```

For example, to test a VST model with 4 GPUs, run:

`CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../VST/tools/dist_test.sh config.py epoch_xx.pth 4`
