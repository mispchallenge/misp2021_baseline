#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Hang Chen)
# Apache 2.0

# extract region of interest (roi) in the video, store as npz file, item name is "data"

set -e
# configs for 'chain'
python_path=
stage=0
nj=15
gpu_nj=4
# End configuration section.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if [ $# != 3 ]; then
  echo "Usage: $0 <data-set> <roi-json-dir> <audio-dir>"
  echo " $0 data/train_far /path/roi data/train_far_sp_hires"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

data_set=$1
roi_json_dir=$2
audio_dir=$3
video_dir="$data_set"_video
av_dir="$audio_dir"_av
segment_video_dir="$video_dir"/segments_data
extractor_dir=extractor
visual_embedding_dir="$video_dir"/visual_embedding



###########################################################################
# concatenate audio-visual embedding
###########################################################################
if [ $stage -le 3 ]; then
  if [ ! -e $touch $av_dir/.done ]; then
    mkdir -p $av_dir/data
    mkdir -p $av_dir/log
      
    cat "$data_set"_sp_hires/segments > $av_dir/segments
    cat "$data_set"_sp_hires/spk2utt > $av_dir/spk2utt
    cat "$data_set"_sp_hires/utt2spk > $av_dir/utt2spk
    cat "$data_set"_sp_hires/text > $av_dir/text
    cat "$data_set"_sp_hires/wav.scp > $av_dir/wav.scp
    cat $video_dir/mp4.scp > $av_dir/mp4.scp 

    ${python_path}python local/concatenate_feature_sp.py --ji 0 --nj 1 $audio_dir $video_dir $av_dir/data

    cat $av_dir/data/raw_av_embedding.*.scp | sort -k 1 | uniq > $av_dir/feats.scp
    steps/compute_cmvn_stats.sh $av_dir || exit 1;
    utils/fix_data_dir.sh $av_dir || exit 1;
    echo 'concatenate done'
    touch $av_dir/.done
  fi
fi
