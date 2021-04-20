import os

import torch
from torch.utils.data import Dataset, DataLoader
from models.net import U2NET
from utiles.dataset import MeterDataset
from utiles.loss_function import DiceLoss, FocalLoss


class Trainer(object):

    def __init__(self):
        self.net = U2NET(3, 2)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.loss_function = FocalLoss(alpha=0.75)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_set_train = MeterDataset(mode='train')
        if not os.path.exists('weight/net.pt'):
            self.net.load_state_dict(torch.load('weight/net.pt', map_location='cpu'))
        # self.data_set_val = MeterDataset(mode='val')
        self.net.to(self.device)

    def __call__(self):
        epoch_num = 100000
        batch_size_train = 5
        batch_size_val = 1
        ite_num = 0
        data_loader_train = DataLoader(self.data_set_train, batch_size_train, True, num_workers=2)
        # data_loader_val = DataLoader(self.data_set_val, batch_size_val, False, num_workers=2)
        # loss_sum = 0
        # running_tar_loss = 0
        save_frq = 1000
        model_dir = 'weight/net.pt'
        for epoch in range(epoch_num):
            '--------train---------'
            self.net.train()
            loss_sum = 0
            running_tar_loss = 0
            for i, (images, masks) in enumerate(data_loader_train):
                ite_num += 1
                images = images.to(self.device)
                masks = masks.to(self.device)
                d0, d1, d2, d3, d4, d5, d6 = self.net(images)

                loss, loss0 = self.calculate_loss(d0, d1, d2, d3, d4, d5, d6, masks)
                self.optimizer.zero_grad()
                # print(loss)
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()
                running_tar_loss += loss0.item()
                del d0, d1, d2, d3, d4, d5, d6, loss
                print(
                    f'epoch:{epoch}; batch:{i + 1}; train loss:{loss_sum / (i + 1)}; tar:{running_tar_loss / (i + 1)}')
                if ite_num % save_frq == 0:
                    torch.save(self.net.state_dict(), model_dir)
            # '---------val----------'
            # self.net.eval()
            # for i,(images,masks) in enumerate(data_loader_val):
            #     images = images.to(self.device)
            #     masks = masks.to(self.device)
            #     d0, d1, d2, d3, d4, d5, d6 = self.net(images)

    def calculate_loss(self, d0, d1, d2, d3, d4, d5, d6, labels):
        loss0 = self.loss_function(d0, labels)
        loss1 = self.loss_function(d1, labels)
        loss2 = self.loss_function(d2, labels)
        loss3 = self.loss_function(d3, labels)
        loss4 = self.loss_function(d4, labels)
        loss5 = self.loss_function(d5, labels)
        loss6 = self.loss_function(d6, labels)
        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss, loss0


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
