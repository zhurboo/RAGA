import torch
import torch.nn as nn
import torch.nn.functional as F


class L1_Loss(nn.Module):
    def __init__(self, gamma=3):
        super(L1_Loss, self).__init__()
        self.gamma = gamma
        
    def dis(self, x, y):
        return torch.sum(torch.abs(x-y), dim=-1)
    
    def forward(self, x1, x2, train_set, train_batch):
        x1_train, x2_train = x1[train_set[:, 0]], x2[train_set[:, 1]]
        x1_neg1 = x1[train_batch[0].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        x1_neg2 = x2[train_batch[1].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg1 = x2[train_batch[2].view(-1)].reshape(-1, train_set.size(0), x2.size(1))
        x2_neg2 = x1[train_batch[3].view(-1)].reshape(-1, train_set.size(0), x1.size(1))
        
        dis_x1_x2 = self.dis(x1_train, x2_train)
        loss11 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg1)))
        loss12 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x1_train, x1_neg2)))
        loss21 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg1)))
        loss22 = torch.mean(F.relu(self.gamma+dis_x1_x2-self.dis(x2_train, x2_neg2)))
        loss = (loss11+loss12+loss21+loss22)/4
        return loss
