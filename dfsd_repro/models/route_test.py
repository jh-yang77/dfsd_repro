import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RouteDICE(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super().__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, 1, 1))
        self.p, self.info, self.masked_w = p, info, None

    def calculate_mask_weight(self):
        weight = self.weight.reshape(self.out_features, self.in_features)
        self.contrib = np.asarray(self.info)[None, :] * weight.detach().cpu().numpy()
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.as_tensor(self.contrib > self.thresh, device=weight.device, dtype=weight.dtype)
        self.masked_w = weight * mask

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        return F.linear(input, self.masked_w, self.bias)
