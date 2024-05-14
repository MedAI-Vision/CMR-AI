# Automated CMR Interpretation

This repo is the official implementation of "Screening and diagnosis of cardiovascular disease using artificial intelligence-enabled cardiac magnetic resonance imaging". For more details, see the accompanying paper,

> [**Screening and diagnosis of cardiovascular disease using artificial intelligence-enabled cardiac magnetic resonance imaging**](https://www.nature.com/articles/s41591-024-02971-2)<br/>
  Yan-Ran Joyce Wang, Kai Yang, Yi Wen, Pengcheng Wang, et al. <b>Nature Medicine</b>, May 13, 2024. https://doi.org/10.1038/s41591-024-02971-2

## Introduction

We developed and validated computerized cardiac magnetic resonance (CMR) interpretation for **screening** and **diagnosis** covering 11 types of cardiovascular diseases (CVDs). Herein, we assigned numerical designations to the CVD classes:

```python
{'HCM': 0, 'DCM': 1, 'CAD': 2, 'ARVC': 3, 'PAH': 4, 'Myocarditis': 5, 'RCM': 6, 'Ebstein’s Anomaly': 7, 'HHD': 8, 'CAM': 9, 'LVNC': 10}
```

Three CMR modalities, i.e., short-axis (SAX) cine, four-chamber (4CH) cine, and SAX late gadolinium enhancement (LGE) were used. Video Swin Transformer (VST) is our model bonebone for CMR interpretation. We also included CNN-LSTM for modeling CMR sequences. For more details, please see the manuscript.

## Usage

### Installation

1. This project is implemented for Python 3 and depends on [PyTorch](https://pytorch.org). Please refer to [requirements.txt](https://github.com/MedAI-Vision/CMR-AI/blob/main/requirements.txt) in our repo or [install.md](https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/docs/install.md)  (recommended) from the official repo of VST for environment preparation.
2. Download the pretrained model from [link](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics600_22k.pth) and run:

```shell
cd CMR-AI
mkdir ./checkpoints
mv ~/Downloads/swin_base_patch244_window877_kinetics600_22k.pth ./checkpoints/
```

### Data Preparation

#### Preprocessing Data

The inputs for VST comprised three modalities with different views and frames (see table below). 

| Modalities | #Views | #Frames Per View|
| ---------- | ----- | ------ |
| SAX cine   | 3     | 25     |
| 4CH cine   | 1     | 25     |
| SAX LGE    | 9     | 1      |

Input imaging data needs to be converted from DICOM to NIFTI format. In addition, extracted heart region of interest (ROI) for each CMR sequence, i.e., the bounding box that covers the heart region, needs to be saved into the PKL mask file. After preprocessing, a typical size of each view is `224*224`, and the input size of SAX cine is `3*25*224*224`.

#### Setting Annotations 

The "annotation" file shows patient IDs, path of input imaging files, and their ground-truth classification labels. Each line in the annotation file is a record for one patient following the pattern: 

`[path of SAX cine] [path of 4CH cine] <optional: path of SAX LGE> [number of frames for cine sequence] <optional: number of views for SAX LGE> [ground-truth label]`. Below shows the examples in annotation files:

```
# For Screening (0 and 1 indicate normal and abnormal, respectively)
/.../SAX_cine_data/.../id1.nii.gz /.../4CH_cine_data/.../id1.nii.gz 25 0
/.../SAX_cine_data/.../id2.nii.gz /.../4CH_cine_data/.../id2.nii.gz 25 1
...
# For Diagnosis (0~10 indicate different CVDs)
/.../SAX_cine_data/.../id3.nii.gz /.../4CH_cine_data/.../id3.nii.gz /.../SAX_LGE_data/.../id3.nii.gz 25 9 2
/.../SAX_cine_data/.../id4.nii.gz /.../4CH_cine_data/.../id4.nii.gz /.../SAX_LGE_data/.../id4.nii.gz 25 9 5
...
```

### Setting Configuration

The configuration file defines the dataset path, mask file path, annotation file path, checkpoint path, model selection, single modality training/multi-modality finetuning, and data augmentation methods. We show four configuration file examples under directory `./configs/`. For instance, in `config_sax_4ch_lge_fusion_diagnosis.py`, we set ['SwinTransformer3D'](https://github.com/MedAI-Vision/CMR-AI/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/models/backbones/swin_transformer.py#L492) as model backbone, ['fusion_ConcatHead'](https://github.com/MedAI-Vision/CMR-AI/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/models/heads/fusion_head.py#L14) as fusion MLP layer, ['RawframeDataset'](https://github.com/MedAI-Vision/CMR-AI/blob/cc10778b207dc2330e7b760ade2027868ce54e25/mmaction/datasets/rawframe_dataset.py#L12) as the dataset and ['NIIDecodeV2'](https://github.com/MedAI-Vision/CMR-AI/blob/d9fbebbf5755270110a55bb5f453a505e1aaa464/mmaction/datasets/pipelines/loading.py#L1452) as NIFTI file decoder.

### Training

To train a model, run:

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

1. train single modality VST-based model (for SAX cine, 4CH cine, and SAX LGE [optional; only for diagnosis], respectively). 
Example configuration files: [screening using 4CH cine](https://github.com/MedAI-Vision/CMR-AI/blob/main/configs/config_sax_screening.py) and [diagnosis using 4CH cine](https://github.com/MedAI-Vision/CMR-AI/blob/main/configs/config_4ch_diagnosis.py).
2. extract the parameters of the VST backbone for each single modality model using `./tools/Convert_model.ipynb` and apply them for the initialization of the fusion model.
3. update configuration file:
* set type='fusion_model' 
* set num_mod = 2 or 3; # 2 for screening (SAX cine and 4CH cine); 3 for diagnosis (SAX cine, 4CH cine, and SAX LGE)
* set fusion=True
* set sax_weight, ch_weight, and lge_weight [optional; only for diagnosis] to be the path of each trained single modality model.
Example configuration files: [fusion for screening](https://github.com/MedAI-Vision/CMR-AI/blob/main/configs/config_sax_4ch_fusion_screening.py) and [fusion for diagnosis](https://github.com/MedAI-Vision/CMR-AI/blob/main/configs/config_sax_4ch_lge_fusion_diagnosis.py). 
4. run the same command as above with the updated configuration file to finetune the fusion model.

### Testing

To test a model, run:

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

We calculate AUROC, f1-score, specificity (sensitivity==0.9) and sensitivity (specificity==0.9) as well as their 95% CI by `./evaluation/Bootstrap_sensitivity_specificity.py` for model evaluation. Besides, we plot ROC curves using `./evaluation/Multi_ROC.py`.

## Usage of CNN-LSTM

The data preparation remains unchanged.

Set to CNN-LSTM directory: `cd ./CNN-LSTM`. Pretrained model is in the `frame` directory.

To train a CNN-LSTM model with 4GPUs, run:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py `

To test a CNN-LSTM model with 4GPUs, run:
`CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 test.py `

