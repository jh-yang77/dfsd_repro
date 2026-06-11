import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        # self.contrib = np.abs(self.contrib)# 根据矩阵的尺寸调整图形的大小
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()
        # sns.heatmap(self.contrib, ax=axes[0], cmap='coolwarm', cbar=True, annot=False)
        # axes[0].set_title('Matrix 1')
        # axes[0].set_xlabel('Column')
        # axes[0].set_ylabel('Row')
        #
        # # 绘制第二个热图
        # sns.heatmap(self.weight.data.cpu().numpy(), ax=axes[1], cmap='coolwarm', cbar=True, annot=False)
        # axes[1].set_title('Matrix 2')
        # axes[1].set_xlabel('Column')
        # axes[1].set_ylabel('Row')
        #
        # # 绘制第三个热图
        # sns.heatmap(self.masked_w.data.cpu().numpy(), ax=axes[2], cmap='coolwarm', cbar=True, annot=False)
        # axes[2].set_title('Matrix 3')
        # axes[2].set_xlabel('Column')
        # axes[2].set_ylabel('Row')
        #
        # # 自动调整布局
        # plt.tight_layout()
        # plt.show()
        # print(1)
    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out

