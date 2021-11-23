import numpy as np
import torch
import sys
import os
sys.path.append("..")
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader

import random

class myDataset(Dataset):
    def __init__(self, scp_path_wakeup, scp_path_tongyong, lip_train_mean, lip_train_var):
        self.scp_path_wakeup = scp_path_wakeup
        self.scp_path_tongyong = scp_path_tongyong
        self.lip_train_mean = lip_train_mean
        self.lip_train_var = lip_train_var
        
        with open(self.scp_path_tongyong) as f:
            lines = f.readlines()
        self.files_scp_tongyong = [line.strip() for line in lines]

        with open(self.scp_path_wakeup) as f:
            lines = f.readlines()
        self.files_scp_wakeup = [line.strip() for line in lines]

        self.files_scp = self.files_scp_wakeup + self.files_scp_tongyong


    def __getitem__(self, idx):
        cur_idx = idx
        video_path = self.files_scp[cur_idx]

        lip_feature = np.squeeze(np.load(video_path), axis=3)
        T = lip_feature.shape[0]
        lip_feature = (lip_feature[:T,:,:]-np.tile(self.lip_train_mean,(T,1,1)))/np.sqrt(np.tile(self.lip_train_var,(T,1,1))+1e-6)
        
        if cur_idx < len(self.files_scp_wakeup):
            data_label = 1.0
        else:
            data_label = 0.0

        return lip_feature, data_label

    def __len__(self):
        return len(self.files_scp)

def myCollateFn(sample_batch):
    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)
    data_feature = [torch.from_numpy(x[0]) for x in sample_batch]
    data_label = torch.tensor([x[1] for x in sample_batch]).unsqueeze(-1)
    data_length = [x.shape[0]//1 for x in data_feature]
    data_feature = pad_sequence(data_feature, batch_first=False, padding_value=0.0)
    return data_feature, data_label, data_length

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = myCollateFn
