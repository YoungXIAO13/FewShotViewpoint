import torch.nn as nn
import torch
CE = nn.CrossEntropyLoss().cuda()
Huber = nn.SmoothL1Loss().cuda()


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SmoothCELoss(nn.Module):
    def __init__(self, range, classes, smooth=0.):
        super(SmoothCELoss, self).__init__()
        self.__range__ = range
        self.__smooth__ = smooth
        self.__SmoothLoss__ = LabelSmoothingLoss(classes, smoothing=smooth, dim=-1)

    def forward(self, pred, target):
        binSize = self.__range__ // pred.size(1)
        trueLabel = target // binSize
        return self.__SmoothLoss__(pred, trueLabel)


def cross_entropy_loss(pred, target, range):
    binSize = range // pred.size(1)
    trueLabel = target // binSize
    return CE(pred, trueLabel)


class CELoss(nn.Module):
    def __init__(self, range):
        super(CELoss, self).__init__()
        self.__range__ = range

    def forward(self, pred, target):
        return cross_entropy_loss(pred, target, self.__range__)


def delta_loss(pred_azi, pred_ele, pred_rol, target, bin):
    # compute the ground truth delta value according to angle value and bin size
    target_delta = ((target % bin) / bin) - 0.5

    # compute the delta prediction in the ground truth bin
    target_label = (target // bin).long()
    delta_azi = pred_azi[torch.arange(pred_azi.size(0)), target_label[:, 0]].tanh() / 2
    delta_ele = pred_ele[torch.arange(pred_ele.size(0)), target_label[:, 1]].tanh() / 2
    delta_rol = pred_rol[torch.arange(pred_rol.size(0)), target_label[:, 2]].tanh() / 2
    pred_delta = torch.cat((delta_azi.unsqueeze(1), delta_ele.unsqueeze(1), delta_rol.unsqueeze(1)), 1)

    return Huber(5. * pred_delta, 5. * target_delta)


class DeltaLoss(nn.Module):
    def __init__(self, bin):
        super(DeltaLoss, self).__init__()
        self.__bin__ = bin

    def forward(self, pred_azi, pred_ele, pred_rol, target):
        return delta_loss(pred_azi, pred_ele, pred_rol, target, self.__bin__)

