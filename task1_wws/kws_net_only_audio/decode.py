# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import random
from tqdm import tqdm
sys.path.append("..")
from tools import utils
from model.audio_kwsnet import KWS_Net 
from reader.data_reader_audio import myDataLoader, myDataset

def test(args):
    # Compile and configure all the model parameters.
    trained_model = args.load_model
    result_dir = args.result_dir

    checkdir(result_dir)
    
    # load mean and var
    fb40_train_mean_var = np.load('scp_dir/train_mean_var_fb40_.npz')
    fb40_train_mean = fb40_train_mean_var['_mean']
    fb40_train_var = fb40_train_mean_var['_var']

    # file path
    file_positive_path = 'scp_dir/positive_dev_middle.scp'
    file_negative_path = 'scp_dir/negative_dev_middle.scp'
    
    # define the dataloader
    print(" " * 50)
    print("*" * 50)
    print("loading the dataset ...")
    dataset = myDataset(file_positive_path, file_negative_path, fb40_train_mean, fb40_train_var)
    dataloader = myDataLoader(dataset=dataset,
                            batch_size=args.minibatchsize,
                            shuffle=False,
                            num_workers=args.num_workers)
    print("- done.")
    print("- {} samples".format(len(dataset)))
    
    # define and load model
    nnet = KWS_Net(args=args)
    nnet = nnet.cuda()
    trained_kws_model =  torch.load(trained_model)
    nnet.load_state_dict(trained_kws_model)

    start_time = time.time()

    # audio modal
    nnet.eval()
    pre_list, pre_list_d, label_list = [], [], []
    with torch.no_grad():
        for feature, data_label, current_frame in dataloader:
            feature = feature.cuda()
            data_label = data_label.cuda()
            outputs = nnet(feature, current_frame)

            output_np = (torch.ceil(torch.sigmoid(outputs)-0.5)).data.cpu().numpy()
            pre_list_d.append((torch.sigmoid(outputs)).data.cpu().numpy())
            pre_list.extend(output_np[:,0])
            label_list.extend(list(data_label.data.cpu().numpy()))
        
        TP, FP, TN, FN = utils.cal_indicator(pre_list, label_list)

        FAR = FP / (FP + TN)
        FRR = 1 - TP / (TP + FN)

        print("Middle audio results: FAR=%.4f, FRR:%.4f" % (FAR,FRR))

        end_time = time.time()
        print("Time used {} seconds.".format(end_time - start_time))
        
        # save
        np.save(result_dir + 'audio_result', np.array(pre_list_d))
        np.save(result_dir + 'audio_label', np.array(label_list))
        
        print("*" * 50)
        print(" " * 50)

def checkdir(_dir):
    try:
        os.makedirs(_dir)
    except OSError:
        pass


if __name__=="__main__":
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--minibatchsize", default=1, type=int)
    parser.add_argument("--input_dim", default=256, type=int)
    parser.add_argument("--hidden_sizes", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--lstm_num_layers", default=1, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--num_workers", default=1, type=int, help="number of validation workers")
    parser.add_argument("--load_model", default='./trained_model/KWS_Lite_Net_1ch_0.model', type=str, help="load audio trained model")
    parser.add_argument("--result_dir", default='../kws_net_fusion/result/', type=str, help="result directory")
    args = parser.parse_args()
    
    
    # run main
    test(args)