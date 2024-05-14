import os.path as osp
import torch
import numpy as np
import copy

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class NIIDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """
    def __init__(self,
                ann_file,
                pipeline,
                data_prefix=None,   
                test_mode=False,
                multi_class=False,
                num_classes=None,
                start_index=0,
                modality='RGB',
                sample_by_class=False,
                power=0,
                dynamic_length=False,
                fusion=False,
                type:list=None):
        super().__init__(ann_file=ann_file,
                        pipeline=pipeline,
                        data_prefix=data_prefix,
                        test_mode=test_mode,
                        multi_class=multi_class,
                        num_classes=num_classes,
                        start_index=start_index,
                        modality=modality,
                        sample_by_class=sample_by_class,
                        power=power,
                        dynamic_length=dynamic_length)
        self.fusion=fusion
        self.type=type
        assert self.type is not None,'Type of dataset must be specified as a list whose components are allowed to be "sax","4ch","lge".'
    # def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
    #     super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            anns=fin.readlines()
        fin.close()
        if self.fusion:
            for i in range(len(anns)):
                single_info=anns[i].split('\n')[0].split(' ')
                data_path=[]
                data_info={}
                idx = 0
                for j in range(len(self.type)):
                    data_path.append(single_info[idx])
                    idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    data_info['offset'] = int(single_info[idx])
                    idx += 1
                if 'sax' in self.type or '4ch' in self.type:
                    data_info['total_frames_cine'] = int(single_info[-3])
                    idx += 1
                if 'lge' in self.type:
                    data_info['total_frames_lge'] = int(single_info[-2])
                    idx += 1
                if self.multi_class:
                    assert self.num_classes is not None
                    data_info['label']=[int(x) for x in single_info[idx:]]
                else:
                    data_info['label']=int(single_info[-1])
                data_info['data_path']=data_path
                video_infos.append(data_info)
        else:
            for i in range(len(anns)):
                idx=0
                single_info=anns[i].split('\n')[0].split(' ')
                data_info={}
                data_info['data_path']=single_info[idx]
                idx+=1
                if self.with_offset:
                    # idx for offset and total_frames
                    data_info['offset'] = int(single_info[idx])
                    data_info['total_frames'] = int(single_info[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    data_info['total_frames'] = int(single_info[idx])
                    idx += 1
                if self.multi_class:
                    assert self.num_classes is not None
                    data_info['label']=[int(x) for x in single_info[idx:]]
                else:
                    data_info['label']=int(single_info[-1])
                
                video_infos.append(data_info)
        return video_infos
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        results['fusion'] = self.fusion
        if self.fusion:
            results['type'] = self.type

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['start_index'] = self.start_index
        results['fusion'] = self.fusion
        if self.fusion:
            results['type'] = self.type

        # prepare tensor in getitem
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)