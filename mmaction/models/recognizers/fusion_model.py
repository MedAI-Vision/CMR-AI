import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class fusion_model(BaseRecognizer):

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        if self.fusion:
            for i in range(len(self.weight)):
                imgs[i] = imgs[i].reshape((-1, ) + imgs[i].shape[2:])
        else:
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()
        if len(labels[0]) > 1:
            cls_label = labels[0][0]
        else:
            cls_label = labels

        x = self.extract_feat(imgs)
        
        cls_score = self.cls_head(x, labels)
        gt_labels = cls_label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs, labels):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        if self.fusion:
            batches = imgs[0].shape[0]
            num_segs = imgs[0].shape[1]
            for i in range(len(self.weight)):
                imgs[i] = imgs[i].reshape((-1, ) + imgs[i].shape[2:])
        else:
            batches = imgs.shape[0]
            num_segs = imgs.shape[1]
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        feat = self.extract_feat(imgs)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat, tuple):
                feat = [avg_pool(x) for x in feat]
                # concat them
                feat = torch.cat(feat, axis=1)
            else:
                feat = avg_pool(feat)
            # squeeze dimensions
            feat = feat.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat, labels)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs, labels):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs, labels).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
        
