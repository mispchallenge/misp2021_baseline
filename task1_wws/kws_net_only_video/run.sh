#!/usr/bin/env bash
#
# This recipe is for misp2021 task1 key word spotting, it judge
# whether keyword is in the given evaluation utterance or not
#
# Copyright  2021  USTC (Author: Hengshun Zhou, Zhaoxu Nian)
# Apache 2.0

set -e

python_path=  # e.g. ./python/bin/
data_root=  # e.g. ./misp2021/
project=video_model
logdir=log
# collect train and dev corpus
echo "collect train and dev corpus"
${python_path}python create_train_scp.py --data_root $data_root

# calculate mean and var of training set
echo "calculate mean and var of training set"
if [ ! -e "scp_dir/train_mean_var_lip.npz" ]; then
  ${python_path}python get_mean_var_lip.py
fi

# start training procedure
echo "start training procedure"
mkdir -p $project
mkdir -p $logdir
${python_path}python train.py --end_iter 15 --project $project --logdir $logdir

echo "decode"
${python_path}python decode.py --load_model ./trained_model/KWS_Lite_Net_1ch_0.model  # choose a model to decode