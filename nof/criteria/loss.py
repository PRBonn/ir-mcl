"""The functions for loss of NOF training
"""

from torch import nn


class NOFLoss(nn.Module):
    def __init__(self):
        super(NOFLoss, self).__init__()
        self.loss = None

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]
        loss = self.loss(pred, target)
        return loss


class NOFMSELoss(NOFLoss):
    """
    MSELoss for predicted scan with real scan
    """

    def __init__(self):
        super(NOFMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')


class NOFL1Loss(NOFLoss):
    """
    L1Loss for predicted scan with real scan
    """

    def __init__(self):
        super(NOFL1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')


class NOFSmoothL1Loss(NOFLoss):
    """
    SmoothL1Loss for predicted scan with real scan
    """

    def __init__(self):
        super(NOFSmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='mean')
