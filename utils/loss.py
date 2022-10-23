import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def build_loss_compute(model, field, opt, train=True):
    """
    nn.Mudule의 서브클래스를 래핑하는 로스계산 서브클래스를 반환 
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    compute = DensenetLossCompute(
            criterion,
            device=device
    )

    compute.to(device)

    return compute


class  LossComputeBase(nn.Module):
    """
    loss 기본 클래스. 
    """
    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    def _compute_loss(self, batch, output, target, **kargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """

        return NotImplementedError

    
    def __call__(self,
                batch, 
                output,
                normalization=1.0):

        loss, stats = self._compute_loss(batch)
        return loss/float(normalization), stats

class DensenetLossCompute(LossComputeBase):
    def __init__(self, criterion, generator):
        super(DensenetLossCompute, self).__init__(criterion, generator)

    def _compute_loss(self, batch , logits, targets, device, **kargs):
        # src: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854
        competition_weights = {
                '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
                '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
            }
        weights =  targets * competition_weights['+'] + (1 - targets) * competition_weights['-']
        L = torch.zeros(targets.shape, device=device)

        w = weights
        y = targets
        p = logits
        eps = 1e-8

        for i  in range(L.shape[0]):
            for j in range(L.shape[1]):
                L[i,j]= -w[i, j] * (
                y[i, j] * math.log(p[i, j] + eps) +
                (1 - y[i, j]) * math.log(1 - p[i, j] + eps))
        Exams_Loss = torch.div(torch.sum(L, dim=1), torch.sum(w, dim=1))

        return Exams_Loss