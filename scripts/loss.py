import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss


class LossSpaceNet7(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True, batch=True)
        self.ce = FocalLoss(gamma=2.0, to_onehot_y=False)
        
    def _loss(self, p, y):
        return self.dice(p, y) + self.ce(p, y)
    
    def forward(self, p, y):
        return self._loss(p, y)