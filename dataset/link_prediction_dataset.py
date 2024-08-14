import openhgnn as hg
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import numpy as np

# define the dataset for link prediction

class DBLPDataset(Dataset):
    def __init__(self, features_list, e_feat, pos_edges, neg_edges, device):
        self.features_list = features_list
        self.e_feat = e_feat
        self.pos_edges = pos_edges
        self.neg_edges = neg_edges
        self.device = device
        self.labels = torch.FloatTensor(
            np.concatenate([np.ones(len(pos_edges[0])), 
                            np.zeros(len(neg_edges[0]))])).to(device)
        
        self.left = np.concatenate([pos_edges[0], neg_edges[0]])
        self.right = np.concatenate([pos_edges[1], neg_edges[1]])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        left = self.left[idx]
        right = self.right[idx]
        label = self.labels[idx]

# using example
"""
train_dataset = DBLPDataset(features_list, e_feat, 
                            (train_pos_head_full, train_pos_tail_full), 
                            (train_neg_head_full, train_neg_tail_full), device)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

for epoch in range(args.epoch):
    net.train()
    for step, (left, right, labels) in enumerate(tqdm(train_loader)):
        t_start = time.time()
        left = left.to(device)
        right = right.to(device)
        labels = labels.to(device)
        
        with autocast_mode.autocast():
            logits = net(features_list, e_feat, left, right, mid, is_train=True)
            logp = F.sigmoid(logits)
            labels = labels.half()
        train_loss = loss_func(logp, labels)
        optimizer.zero_grad()

        # autograd
        scale_loss = scaler.scale(train_loss) / accum_steps
        scale_loss.backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        t_end = time.time()
        # print training info
        # print('Epoch {:05d}, Step{:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, step, train_loss.item(), t_end-t_start))
"""