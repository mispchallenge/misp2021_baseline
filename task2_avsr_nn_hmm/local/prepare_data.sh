#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Hang Chen)
# Apache 2.0

# transform misp data to kaldi format
python_path=
nj=
echo "$0 $@"
. utils/parse_options.sh
. ./path.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: $0 <corpus-data-dir> <enhancement-data-dir> <data-set> <dict-dir> <store-dir>"
  echo " $0 /path/misp /path/misp_WPE train data/local/dict data/train_far"
  exit 1;
fi

data_root=$1
enhancement_root=$2
data_type=$3
dict_dir=$4
store_dir=$5

# wav.scp segments text_sentence utt2spk
echo "prepare wav.scp segments text_sentence utt2spk"
${python_path}python local/prepare_far_data.py -nj $nj $enhancement_root/audio $data_root/video $data_root/transcription $data_type $store_dir
cat $store_dir/temp/wav.scp | sort -k 1 | uniq > $store_dir/wav.scp
cat $store_dir/temp/mp4.scp | sort -k 1 | uniq > $store_dir/mp4.scp
cat $store_dir/temp/segments | sort -k 1 | uniq > $store_dir/segments
cat $store_dir/temp/utt2spk | sort -k 1 | uniq > $store_dir/utt2spk
cat $store_dir/temp/text_sentence | sort -k 1 | uniq > $store_dir/text_sentence
echo "prepare done"

# text jieba's vocab format requires word count(frequency), set to 99
echo "word segmentation"
awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $dict_dir/word_seg_vocab.txt
${python_path}python local/word_segmentation.py $dict_dir/word_seg_vocab.txt $store_dir/text_sentence > $store_dir/text
echo "segmentation done"

# spk2utt
utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt
echo "local/prepare_data.sh succeeded"
exit 0