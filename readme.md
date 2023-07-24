# Automated CMR Interpretation

This repo is the official implementation of "Video-based artificial intelligence for screening and diagnosis of cardiovascular diseases based on CMR: a clinical-based and proof-of-concept study".

## Introduction

We developed and validated computerized Cardiac Magnetic Resonance (CMR) interpretation for **screening** and **diagnosis** of 11 cardiovascular diseases (CVDs). Herein, we have assigned numerical designations to these CVDs:

```python
{'HCM': 0, 'DCM': 1, 'CAD': 2, 'ARVC': 3, 'PAH': 4, 'Myocarditis': 5, 'RCM': 6, 'Ebstein’s Anomaly': 7, 'HHD': 8, 'CAM': 9, 'LVNC': 10}
```

Besides, we employed a total of 3 modalities of CMR, namely, short-axis (SAX) Cine, four-chamber (4CH) Cine, and SAX late gadolinium enhancement (LGE). Video Swin Transformer (VST) is our main model for the CMR interpretation, and the other is CNN-LSTM. For more details, please see the accompanying paper.

## Usage

### Installation

1. This project is implemented for Python 3,  and depends on [PyTorch](https://pytorch.org). Please refer to [requirements.txt](https://github.com/MedAI-Vision/CMR-AI/requirements.txt) in our repo or [install.md](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/install.md)  (recommended) from the official repo of VST to do environment preparation.
2. Download pretrained model from [link](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth) and run:

```shell
cd CMR-AI
mkdir ./checkpoints
mv ~/Downloads/swin_base_patch244_window877_kinetics600_22k.pth ./checkpoints/
```

### Data Preparation

#### Preprocess data

The inputs of VST are from three modalities with different views and frames (see table below). All data should be converted from DICOM files to NIFTI files at first. 

| Modalities | Views | Frames |
| ---------- | ----- | ------ |
| SAX Cine   | 3     | 25     |
| 4CH Cine   | 1     | 25     |
| SAX LGE    | 9     | 1      |

Moreover, all regions of interest (ROIs), which indicate the boundaries of hearts on CMR, should be processed into PKL file. After preprocessing, a typical size of each view is `224*224`. For example, the size of SAX Cine is `3*25*224*224`.

#### Make annotations

The "annotations" file indicates the correspondence between patient identifiers, addresses of modality files, and their corresponding classifications, facilitating the smooth conduction of experiments. Every line of the file is a record, and follows this format: `[middle slice path of SAX cine] [4CH cine path] <optional: LGE path> [frame number of cine] <optional: view number of LGE> [class label]`. Here are some examples of annotation files:

```
# For Screening (0, 1 indicate normal and anomaly)
/.../SAX_data/.../id1.nii.gz /.../4CH_data/.../id1.nii.gz /.../LGE_data/.../id1.nii.gz 25 9 0
/.../SAX_data/.../id1.nii.gz /.../4CH_data/.../id2.nii.gz /.../LGE_data/.../id2.nii.gz 25 9 1
...
# For Dignosis (0~10 indicate different CVDs)
/.../SAX_data/.../id2.nii.gz /.../4CH_data/.../id2.nii.gz 25 2
/.../SAX_data/.../id4.nii.gz /.../4CH_data/.../id4.nii.gz 25 8
...
```

### Config Setting

Set or update the dataset path , mask file path, annotations file path, checkpoint path, model selection, data augmentation methods and other configurations in config files in directory `./configs/`. We have given 4 examples in that directory. For instance, in `config_sax_4ch_lge_fusion_diagnosis.py`, we choose ['SwinTransformer3D'](https://github.com/MedAI-Vision/CMR-AI-Origin/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/models/backbones/swin_transformer.py#L492) as model backbone, ['fusion_ConcatHead'](https://github.com/MedAI-Vision/CMR-AI-Origin/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/models/heads/fusion_head.py#L14) as fusion MLP layer, ['RawframeDataset'](https://github.com/MedAI-Vision/CMR-AI-Origin/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/datasets/rawframe_dataset.py#L12) as the dataset and ['NIIDecodeV2'](https://github.com/MedAI-Vision/CMR-AI-Origin/blob/d9fbebbf5755270110a55bb5f453a505e1aaa464/mmaction/datasets/pipelines/loading.py#L1452) as NIFTI file decoder.

### Training

To train a VST model for single modality CMR dataset, run:

```
# single-gpu training
python tools/train.py <CONFIG_FILE>

# multi-gpu training
bash tools/dist_train.sh <CONFIG_FILE> <GPU_NUM>
```

For example, to train a VST model with 4 GPUs, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../VST/tools/dist_train.sh config.py 4
```


When training fusion model, first process all the single modality models with code `./tools/Convert_model.ipynb`, which removes the MLP layer of VST, then set the fusion mode to True in config file, also add the single modality model paths(after removed MLPs). Finally, run the same command to train the fusion model.


### Testing

To test a VST model for single or fusion modality CMR dataset, run:

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE>

# multi-gpu testing
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <GPU_NUM>
```

For example, to test a VST model with 4 GPUs, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash .../VST/tools/dist_test.sh config.py epoch_xx.pth 4
```

### Evaluation

The final evaluation and analysis are performed with `./evaluation/`.

We calculate AUROC, f1-score, specificity (sensitivity==0.9) and sensitivity (specificity==0.9) as well as their 95% CI by `./evaluation/Bootstrap_sensitivity_specificity.py` to evaluate our models. Besides, we also plot ROC curves via `./evaluation/Multi_ROC.py`.

## Usage of CNN-LSTM

The data preparation remains unchanged. 

Move to CNN-LSTM directory: `cd ./CNN-LSTM`

To train a CNN-LSTM model with 4GPUs, run:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py `

To test a CNN-LSTM model with 4GPUs, run:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py `

