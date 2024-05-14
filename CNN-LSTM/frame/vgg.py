#!/usr/bin/env python
"""
vgg.py
    - VGG11/13/16/19 and VGG11_bn/VGG13_bn/VGG16_bn/VGG19_bn in torchvision
    - load the features layers weights only (without the classifier)
"""
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

from torchvision.models.vgg import model_urls, make_layers


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def extract_feature_state_dict(pretrained_state_dict, model):
    model_state_dict = model.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    return pretrained_state_dict


def vgg11(pretrained=False, requires_grad=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg11'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg11_bn(pretrained=False, requires_grad=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg11_bn'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg13(pretrained=False, requires_grad=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg13'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg13_bn(pretrained=False, requires_grad=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg13_bn'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg16(pretrained=False, requires_grad=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg16'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg16_bn(pretrained=False, requires_grad=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg19(pretrained=False, requires_grad=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg19'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model


def vgg19_bn(pretrained=False, requires_grad=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['vgg19_bn'])
        pretrained_state_dict = extract_feature_state_dict(pretrained_state_dict, model)
        model.load_state_dict(pretrained_state_dict)
        model.requires_grad = requires_grad
    return model
