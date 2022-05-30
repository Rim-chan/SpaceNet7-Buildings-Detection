import torch
from torchmetrics import Metric

class DiceSpaceNet7(Metric):
    def __init__(self, n_class):
        super().__init__(dist_sync_on_step=False)
        self.n_class = n_class
        self.add_state("steps", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("dice", default=torch.zeros(self.n_class), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.zeros(1), dist_reduce_fx="sum")
        
    def update(self, p, y, loss):
        self.steps += 1
        self.dice += self.compute_stats_spacenet7(p, y)
        self.loss += loss
        
    def compute(self):                        
        return 100 * self.dice / self.steps, self.loss / self.steps
    
    def compute_stats_spacenet7(self, p, y):
        scores = torch.zeros(self.n_class, device=p.device, dtype=torch.float32)
        p = (torch.sigmoid(p) > 0.5).int()
        
        for i in range(self.n_class):
            p_i, y_i = p[:, i], y[:, i]
            if (y_i != 1).all():
                scores[i - 1] = 1 if (p_i != 1).all() else 0
                continue
            tp, fn, fp = self.get_stats(p_i, y_i, 1)
            denom = (2 * tp + fn + fp).to(torch.float)
            score_cls = (2 * tp).to(torch.float) / denom if torch.is_nonzero(denom) else 0
            scores[i - 1] = score_cls
        return scores
    
    @staticmethod
    def get_stats(p, y, class_idx):
        tp = torch.logical_and(p == class_idx, y == class_idx).sum()
        fn = torch.logical_and(p != class_idx, y == class_idx).sum()
        fp = torch.logical_and(p == class_idx, y != class_idx).sum()
        return tp, fn, fp