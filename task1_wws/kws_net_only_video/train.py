# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import copy
import random
from tqdm import tqdm
sys.path.append("..")
from tools import utils
from model.video_kwsnet import KWS_Net
from reader.data_reader_video import myDataLoader, myDataset

def main(args):
    # Compile and configure all the model parameters.
    model_path = args.project
    log_dir = args.logdir
    logger = utils.get_logger(log_dir + '/' + args.project)
    seed_torch(args.seed)

    # load mean and var
    lip_train_mean_var = np.load('scp_dir/train_mean_var_lip.npz')
    lip_train_mean = lip_train_mean_var['_mean']
    lip_train_var = lip_train_mean_var['_var']

    # file path
    file_train_positive_path = 'scp_dir/positive_train.scp'
    file_train_negative_path = 'scp_dir/negative_train.scp'
    file_dev_positive_path = 'scp_dir/positive_dev.scp'
    file_dev_negative_path = 'scp_dir/negative_dev.scp'
    
    # define the dataloader
    print("loading the dataset ...")
    dataset_train = myDataset(file_train_positive_path, file_train_negative_path, lip_train_mean, lip_train_var)
    dataset_dev = myDataset(file_dev_positive_path, file_dev_negative_path, lip_train_mean, lip_train_var)
    dataloader_train = myDataLoader(dataset=dataset_train,
                            batch_size=args.minibatchsize_train,
                            shuffle=True,
                            num_workers=args.train_num_workers)
    dataloader_dev = myDataLoader(dataset=dataset_dev,
                            batch_size=args.minibatchsize_dev,
                            shuffle=False,
                            num_workers=args.dev_num_workers)
    print("- done.")
    all_file = len(dataloader_train)
    all_file_dev = len(dataloader_dev)
    print("- {} training samples, {} dev samples".format(len(dataset_train), len(dataset_dev)))
    print("- {} training batch, {} dev batch".format(len(dataloader_train), len(dataloader_dev)))
    
    # define the model
    nnet = KWS_Net(args=args)
    nnet = nnet.cuda()

    pretrained_dict = torch.load("./model/pretrained/lipreading_LRW.pt", map_location=torch.device("cpu"))
    model_dict = nnet.lip_encoder.state_dict()
    new_model_dict = dict()
    for k, v in model_dict.items():
        if ('video_frontend.' + k) in pretrained_dict.keys() and v.size() == pretrained_dict['video_frontend.' + k].size():
            new_model_dict[k] = pretrained_dict['video_frontend.' + k]
    new_model_dict = {k: v for k, v in new_model_dict.items()
                    if k in model_dict.keys()
                    and v.size() == model_dict[k].size()}
    nnet.lip_encoder.load_state_dict(new_model_dict)

    # training setups
    optimizer = optim.Adam(nnet.parameters(), lr=args.lr)
    BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0)) 

    for iter_ in range(args.end_iter):
        start_time = time.time()
        running_loss = 0.0
        nnet.train()
        for video_feature, data_label_torch, current_frame in dataloader_train:
            video_feature = video_feature.cuda()
            data_label_torch = data_label_torch.cuda()
            outputs = nnet(video_feature, current_frame)
            loss = BCE_loss(outputs, data_label_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        logger.info("Iteration:{0}, loss = {1:.6f} ".format(iter_, running_loss / all_file))

        # eval
        nnet.eval()
        pre_list, pre_list_d, label_list = [], [], []
        with torch.no_grad():
            running_loss_dev = 0.0
            pre_sum = 0.0
            for video_feature_dev, data_label_torch_dev, current_frame_dev in dataloader_dev:
                video_feature_dev = video_feature_dev.cuda()
                data_label_torch_dev = data_label_torch_dev.cuda()
                outputs_dev = nnet(video_feature_dev, current_frame_dev)
                loss_dev = BCE_loss(outputs_dev, data_label_torch_dev)
                
                running_loss_dev += loss_dev.item()
                outputs_dev_np = (torch.ceil(torch.sigmoid(outputs_dev)-0.5)).data.cpu().numpy()
                pre_list.extend(outputs_dev_np[:,0])
                label_list.extend(list(data_label_torch_dev.data.cpu().numpy()))
                
            TP, FP, TN, FN = utils.cal_indicator(pre_list, label_list)

            FAR = FP / (FP + TN)
            FRR = 1 - TP / (TP + FN)

            logger.info("Middle video Epoch:%d, Train loss=%.4f, Valid loss=%.4f, FAR=%.4f, FRR:%.4f" % (iter_,
                        running_loss / all_file,running_loss_dev /all_file_dev, FAR,FRR))

        torch.save(nnet.state_dict(), os.path.join(model_path, "{}_{}.model".format(args.model_name, iter_)))
        nnet.train()

        end_time = time.time()
        logger.info("Time used for each epoch training: {} seconds.".format(end_time - start_time))
        logger.info("*" * 50)

def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--minibatchsize_train", default=16, type=int)
    parser.add_argument("--minibatchsize_dev", default=1, type=int)
    parser.add_argument("--input_dim", default=256, type=int)
    parser.add_argument("--hidden_sizes", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--lstm_num_layers", default=1, type=int)
    parser.add_argument("--seed", default=617, type=int)
    parser.add_argument("--project", default='video_model', type=str)
    parser.add_argument("--logdir", default='./log/', type=str)
    parser.add_argument("--model_name", default='kws_video', type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--train_num_workers", default=8, type=int, help="number of training workers")
    parser.add_argument("--dev_num_workers", default=1, type=int, help="number of validation workers")
    args = parser.parse_args()

    # run main
    main(args)