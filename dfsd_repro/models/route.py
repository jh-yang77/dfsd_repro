import torch
import torch.nn as nn
import numpy as np


class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        # a = self.info[None, :]
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()
        info_std = np.sqrt(np.load("cache/CIFAR-10_densenet_feat_stat.npy"))

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        data=self.contrib
        data_train = np.load("cache/CIFAR-10_train_densenet_in.npy")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = plt.cm.cividis(data[i, j])
                #color = tuple([min(1, c * 1.5) for c in color])# 根据数据值确定颜色
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))
        plt.xlabel('Feature Dim')
        plt.ylabel('Class')
        plt.xticks([])
        plt.tight_layout()
        plt.savefig('output/{feature_dim}_hotmap.pdf')


        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(0, data.shape[0])
        plt.show()

        palette = plt.get_cmap('Set1')
        color_index = 1
        color = palette(color_index)
        x = [i for i in range(self.contrib.shape[1])]

        for j in range(0,10):
            plt.plot(x, np.sort(self.contrib[j, :]),color=color,linewidth=3.5)
            plt.fill_between(x, contrib_up[j, np.argsort(self.contrib[j, :])], contrib_down[j, np.argsort(self.contrib[j, :])],color=color, alpha=0.2)
            plt.show()
            plt.savefig('output/CF10_feature_c{}.pdf'.format(j))
        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)
        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()
        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out

