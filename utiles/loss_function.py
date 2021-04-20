import torch
from torch import nn


class DiceLoss:

    def __init__(self):
        pass

    def __call__(self, input, target):
        input = input.reshape(input.size()[0], -1)
        target = target.reshape(target.size()[0], -1).float()
        a = torch.sum(input * target, 1)
        b = torch.sum(input * input, 1)
        c = torch.sum(target * target, 1)
        d = (2 * a) / (b + c+1e-6)
        return 1 - d


class FocalLoss:

    def __init__(self, gamma=2, alpha=0.25, epsilon=1e-19):
        """

        :param gamma: gamma>0减少易分类样本的损失。使得更关注于困难的、错分的样本。越大越关注于困难样本的学习
        :param alpha:调节正负样本比例
        :param r:数值稳定系数。
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, p, target):
        p_min = p.min()
        p_max = p.max()
        if p_min < 0 or p_max > 1:
            raise ValueError('The range of predicted values should be [0, 1]')
        p = p.reshape(-1, 1)
        target = target.reshape(-1, 1)
        loss = -self.alpha * (1 - p) ** self.gamma * (target * torch.log(p + self.epsilon)) - \
               (1 - self.alpha) * p ** self.gamma * ((1 - target) * torch.log(1 - p + self.epsilon))
        return loss.mean()