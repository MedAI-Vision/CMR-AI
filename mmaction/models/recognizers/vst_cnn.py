import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class vst_cnn(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()

        x, x_cnn = self.extract_feat(imgs)
        
        cls_score = self.cls_head(x, x_cnn)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats1 = []
            feats2 = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x1, x2 = self.extract_feat(batch_imgs)

                feats1.append(x1)
                feats2.append(x2)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats1[0], tuple):
                len_tuple = len(feats1[0])
                feat1 = [
                    torch.cat([x[i] for x in feats1]) for i in range(len_tuple)
                ]
                feat1 = tuple(feat1)
            else:
                feat1 = torch.cat(feats1)
                
            if isinstance(feats2[0], tuple):
                len_tuple = len(feats2[0])
                feat2 = [
                    torch.cat([x[i] for x in feats2]) for i in range(len_tuple)
                ]
                feat2 = tuple(feat2)
            else:
                feat2 = torch.cat(feats2)
        else:
            feat1, feat2 = self.extract_feat(imgs)

        if self.feature_extraction:
            # perform spatio-temporal pooling
            avg_pool = nn.AdaptiveAvgPool3d(1)
            if isinstance(feat1, tuple):
                feat1 = [avg_pool(x) for x in feat1]
                # concat them
                feat1 = torch.cat(feat1, axis=1)
            else:
                feat1 = avg_pool(feat1)
            if isinstance(feat2, tuple):
                feat2 = [avg_pool(x) for x in feat2]
                # concat them
                feat2 = torch.cat(feat2, axis=1)
            else:
                feat2 = avg_pool(feat2)
            # squeeze dimensions
            feat1 = feat1.reshape((batches, num_segs, -1))
            feat2 = feat2.reshape((batches, num_segs, -1))
            # temporal average pooling
            feat1 = feat1.mean(axis=1)
            feat2 = feat2.mean(axis=1)
            return feat1, feat2

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat1, feat2)
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

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

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
