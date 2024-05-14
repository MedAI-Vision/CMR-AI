import copy
import os.path as osp
import warnings

import torch

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class RawframeDataset(BaseDataset):
    """Rawframe dataset for action recognition.

    The dataset loads raw frames and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    the directory to frames of a video, total frames of the video and
    the label of a video, which are split with a whitespace.
    Example of a annotation file:

    .. code-block:: txt

        some/directory-1 163 1
        some/directory-2 122 1
        some/directory-3 258 2
        some/directory-4 234 2
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a multi-class annotation file:


    .. code-block:: txt

        some/directory-1 163 1 3 5
        some/directory-2 122 1 2
        some/directory-3 258 2
        some/directory-4 234 2 4 6 8
        some/directory-5 295 3
        some/directory-6 121 3

    Example of a with_offset annotation file (clips from long videos), each
    line indicates the directory to frames of a video, the index of the start
    frame, total frames of the video clip and the label of a video clip, which
    are split with a whitespace.


    .. code-block:: txt

        some/directory-1 12 163 3
        some/directory-2 213 122 4
        some/directory-3 100 258 5
        some/directory-4 98 234 2
        some/directory-5 0 295 3
        some/directory-6 50 121 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        filename_tmpl (str): Template for each filename.
            Default: 'img_{:05}.jpg'.
        with_offset (bool): Determines whether the offset information is in
            ann_file. Default: False.
        multi_class (bool): Determines whether it is a multi-class
            recognition dataset. Default: False.
        num_classes (int | None): Number of classes in the dataset.
            Default: None.
        modality (str): Modality of data. Support 'RGB', 'Flow'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 filename_tmpl='img_{:05}.jpg',
                 with_offset=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0.,
                 dynamic_length=False,
                 fusion=False,
                 type1=None,
                 type2=None,
                 type3=None):
        self.filename_tmpl = filename_tmpl
        self.with_offset = with_offset
        self.fusion = fusion
        if fusion:
            self.type = list()
            if type1:
                self.type.append(type1)
            if type2:
                self.type.append(type2)
            if type3:
                self.type.append(type3)
        super().__init__(
            ann_file,
            pipeline,
            data_prefix,
            test_mode,
            multi_class,
            num_classes,
            start_index,
            modality,
            sample_by_class=sample_by_class,
            power=power,
            dynamic_length=dynamic_length)

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            if self.fusion:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    all_dir = []
                    ann_error=0
                    for i in range(len(self.type)):
                        frame_dir = line_split[idx]
                        if self.data_prefix is not None and osp.exists(self.data_prefix):
                            frame_dir = osp.join(self.data_prefix, frame_dir)
                        if not osp.exists(frame_dir):
                            warnings.warn(f"{frame_dir} NOT FOUND! This sample will be excluded.")
                            ann_error=1
                        all_dir.append(frame_dir)
                        idx += 1
                    if ann_error==1:
                        continue
                    video_info['frame_dir'] = all_dir
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        idx += 1
                        
                    if 'sax' in self.type or '4ch' in self.type:
                        video_info['total_frames_cine'] = int(line_split[idx])
                        idx += 1
                    if 'lge' in self.type:
                        video_info['total_frames_lge'] = int(line_split[idx])
                        idx += 1

                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
#                        assert len(label) == 1
                        video_info['label'] = label
                    video_infos.append(video_info)

            else:
                for line in fin:
                    line_split = line.strip().split()
                    video_info = {}
                    idx = 0
                    # idx for frame_dir
                    frame_dir = line_split[idx]
                    if self.data_prefix is not None and osp.exists(self.data_prefix):
                        frame_dir = osp.join(self.data_prefix, frame_dir)
                    if not osp.exists(frame_dir):
                        warnings.warn(f"{frame_dir} NOT FOUND! This sample will be excluded.")
                        continue
                    video_info['frame_dir'] = frame_dir
                    idx += 1
                    if self.with_offset:
                        # idx for offset and total_frames
                        video_info['offset'] = int(line_split[idx])
                        video_info['total_frames'] = int(line_split[idx + 1])
                        idx += 2
                    else:
                        # idx for total_frames
                        video_info['total_frames'] = int(line_split[idx])
                        idx += 1
                    # idx for label[s]
                    label = [int(x) for x in line_split[idx:]]
                    assert label, f'missing label in line: {line}'
                    if self.multi_class:
                        assert self.num_classes is not None
                        video_info['label'] = label
                    else:
                        assert len(label) == 1
                        video_info['label'] = label[0]
                    video_infos.append(video_info)
                
        return video_infos

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
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
        results['filename_tmpl'] = self.filename_tmpl
        results['modality'] = self.modality
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
