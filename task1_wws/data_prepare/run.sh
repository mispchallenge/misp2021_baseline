#!/usr/bin/env bash
#
# This recipe is for misp2021 task1 key word spotting, it judge
# whether keyword is in the given evaluation utterance or not
# 
# Copyright  2021  USTC (Author: Zhaoxu Nian)
# Apache 2.0

set -e

python_path=  # e.g. ./python/bin/
data_root=  # e.g. ./misp2021/

# add_reverb
echo "start adding reverb"
${python_path}python add_reverb/create_scp.py --data_root $data_root --scp_path add_reverb/positive.scp
${python_path}python add_reverb/add_reverb.py --scp_path add_reverb/positive.scp

# add_noise
echo "start adding middle noise "
${python_path}python add_noise/process_noise.py --data_root $data_root
chmod 754 add_noise/AddNoise_MultiOutput
chmod 754 add_noise/bin/AddNoise_MultiOutput
mkdir -p add_noise/scp
mkdir -p add_noise/Config
mkdir -p add_noise/log
${python_path}python add_noise/process_data.py --data_root $data_root --scp_path add_noise/scp/positive.scp
perl add_noise/Add_noise_MultipleOutput.pl
rm add_noise/*/*[0-9]
rm -r positive*
${python_path}python add_noise/process_data.py --data_root $data_root --mode raw2wav

# get lip feature
echo "get lip feature"
${python_path}python get_lip_feature.py --data_root $data_root
