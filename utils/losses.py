import torch.nn as nn
import torch
from . import base
from . import functional as _CF
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


class TverskyLoss(base.Loss):

    def __init__(self, alpha=0.7, smooth=1., reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_true = y_true.type_as(y_pred).view_as(y_pred)
        tp = torch.sum(y_pred * y_true, dim=(1, 2, 3))
        fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
        fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        if self.reduction == 'mean':
            loss = loss.mean()
        return 1 - loss


class MFocalLoss(base.Loss):

    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 reduction='mean',
                 weight=None, ):
        super(MFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight, requires_grad=False)
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        '''
        args: y_pred: tensor of shape (N, 1, H, W)
        args: y_true: tensor of shape (N, H, W)
        '''

        pred = y_pred
        true = y_true.type_as(y_pred).view_as(y_pred).clone().detach().requires_grad_(False)

        pt = (1 - pred) * true + pred * (1 - true)
        pt = pt.clamp(min=0.1, max=0.9).clone().detach().requires_grad_(False)

        focal_weight = pt.pow(self.gamma)

        loss = F.binary_cross_entropy(pred, true, reduction='none') * focal_weight
        loss = loss.mean(dim=(1, 2, 3))

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

class PTLoss(base.Loss):

    def __init__(self, alpha=0.7, smooth=1, reduction='mean'):
        super(PTLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.smooth = smooth

        self.pool1 = nn.AdaptiveMaxPool2d(output_size=(2, 2))
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=(4, 4))
        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(8, 8))

        self.pool1_avg = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.pool2_avg = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.pool3_avg = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    def forward(self, y_pred, y_true):
        y_true = y_true.type_as(y_pred).view_as(y_pred)

        b, c, h, w = y_true.size()

        tp_m = y_pred * y_true
        fn_m = y_true * (1 - y_pred)
        fp_m = (1 - y_true) * y_pred

        tp = torch.sum(tp_m, dim=(1, 2, 3))
        fn = torch.sum(fn_m, dim=(1, 2, 3))
        fp = torch.sum(fp_m, dim=(1, 2, 3))
        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        loss1 = 1 - loss.mean()

        # 128*128
        tp = self.pool1_avg(tp_m).reshape(b, -1)
        fn = self.pool1_avg(fn_m).reshape(b, -1)
        fp = self.pool1_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        mask = self.pool1(y_true).reshape(b, -1).clone().detach().requires_grad_(False)

        loss2 = 1 - (torch.sum(loss * mask) + 1) / (torch.sum(mask) + 1)

        # 64*64
        tp = self.pool2_avg(tp_m).reshape(b, -1)
        fn = self.pool2_avg(fn_m).reshape(b, -1)
        fp = self.pool2_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        mask = self.pool2(y_true).reshape(b, -1).clone().detach().requires_grad_(False)

        loss3 = 1 - (torch.sum(loss * mask) + 1) / (torch.sum(mask) + 1)

        # 32*32
        tp = self.pool3_avg(tp_m).reshape(b, -1)
        fn = self.pool3_avg(fn_m).reshape(b, -1)
        fp = self.pool3_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        mask = self.pool3(y_true).reshape(b, -1).clone().detach().requires_grad_(False)

        loss4 = 1 - (torch.sum(loss * mask) + 1) / (torch.sum(mask) + 1)

        return loss1 + loss2 + loss3 + loss4


class PTLoss2(base.Loss):
    
    """
    The second version of PT loss.
    """

    def __init__(self, alpha=0.7, smooth=1.e-4, reduction='mean'):
        super(PTLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.smooth = smooth

        self.pool1 = nn.AdaptiveMaxPool2d(output_size=(2, 2))
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=(4, 4))
        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(8, 8))

        self.pool1_avg = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.pool2_avg = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.pool3_avg = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    def forward(self, y_pred, y_true):
        y_true = y_true.type_as(y_pred).view_as(y_pred).clone().detach().requires_grad_(False)
        mask_pred = torch.zeros_like(y_pred)
        mask_pred[y_pred.ge(0.5)] = 1

        b, c, h, w = y_true.size()

        tp_m = y_pred * y_true
        fn_m = y_true * (1 - y_pred)
        fp_m = (1 - y_true) * y_pred

        tp = torch.sum(tp_m, dim=(1, 2, 3))
        fn = torch.sum(fn_m, dim=(1, 2, 3))
        fp = torch.sum(fp_m, dim=(1, 2, 3))
        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        loss1 = 1 - loss.mean()

        # 128*128
        tp = self.pool1_avg(tp_m).reshape(b, -1)
        fn = self.pool1_avg(fn_m).reshape(b, -1)
        fp = self.pool1_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        mask = self.pool1(y_true).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool) | \
               self.pool1(mask_pred).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool)
        mask = mask.to(torch.int)

        loss2 = 1 - (torch.sum(loss * mask) + self.smooth) / (torch.sum(mask) + self.smooth)

        # 64*64
        tp = self.pool2_avg(tp_m).reshape(b, -1)
        fn = self.pool2_avg(fn_m).reshape(b, -1)
        fp = self.pool2_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)

        mask = self.pool2(y_true).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool) | \
               self.pool2(mask_pred).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool)
        mask = mask.to(torch.int)
        loss3 = 1 - (torch.sum(loss * mask) + self.smooth) / (torch.sum(mask) + self.smooth)

        # 32*32
        tp = self.pool3_avg(tp_m).reshape(b, -1)
        fn = self.pool3_avg(fn_m).reshape(b, -1)
        fp = self.pool3_avg(fp_m).reshape(b, -1)

        loss = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        mask = self.pool3(y_true).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool) | \
               self.pool3(mask_pred).reshape(b, -1).clone().detach().requires_grad_(False).type(torch.bool)
        mask = mask.to(torch.int)
        loss4 = 1 - (torch.sum(loss * mask) + self.smooth) / (torch.sum(mask) + self.smooth)

        return loss1 + loss2 + loss3 + loss4

class FocalLossWithTverskyLoss(base.Loss):

    def __init__(self):
        super(FocalLossWithTverskyLoss, self).__init__()
        self.focal_loss = MFocalLoss()
        self.tversky_loss = TverskyLoss()
        self.lamda = 1

    def forward(self, y_pred, y_true):
        loss1 = self.tversky_loss(y_pred, y_true)
        loss2 = self.focal_loss(y_pred, y_true)
        return loss1 + self.lamda * loss2


class PTMFLoss(base.Loss):

    def __init__(self):
        super(PTMFLoss, self).__init__()
        self.focal_loss = MFocalLoss()
        self.pdl_loss = PTLoss()
        self.lamda = 1

    def forward(self, y_pred, y_true):
        loss1 = self.pdl_loss(y_pred, y_true)
        loss2 = self.focal_loss(y_pred, y_true)
        return loss1 + self.lamda * loss2