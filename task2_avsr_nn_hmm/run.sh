#!/usr/bin/env bash
#
# This recipe is for misp2021 task2 (ASR only), it recognise
# a given evaluation utterance given ground truth
# diarization information
#
# Copyright  2021  USTC (Author: Hang Chen)
# Apache 2.0
#

# Begin configuration section.
nj=15
stage=-1
nnet_stage=0
oovSymbol="<UNK>"
boost_sil=1.0 # note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
numLeavesTri1=4000
numGaussTri1=32000
numLeavesTri2=7000
numGaussTri2=56000
numLeavesMLLT=10000
numGaussMLLT=80000
numLeavesSAT=12000
numGaussSAT=96000
# End configuration section

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh
source ./bashrc

set -e

# path settings
beamformit_path=      # e.g. /path/BeamformIt-master
python_path=          # e.g. /path/python/bin
misp2021_corpus=      # e.g. /path/misp2021
enhancement_dir=${misp2021_corpus}_WPE
dict_dir=data/local/dict
data_roi=data/local/roi

##########################################################################
# wpe+beamformit
##########################################################################

# use nara-wpe and beamformit to enhance multichannel misp data
# notice:make sure you install nara-wpe and beamformit and you need to compile BeamformIt with the kaldi script install_beamformit.sh 
if [ $stage -le -1 ]; then
  for x in dev train ; do
    if [[ ! -f ${enhancement_dir}/audio/$x.done ]]; then
      local/enhancement.sh --stage 0 --python_path $python_path --beamformit_path $beamformit_path \
        $misp2021_corpus/audio/$x ${enhancement_dir}/audio/$x  || exit 1;
      touch ${enhancement_dir}/audio/$x.done
    fi
  done
fi


###########################################################################
# prepare dict
###########################################################################

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 0 ]; then
  if [[ ! -f $dict_dir/.done ]]; then
    local/prepare_dict.sh --python_path $python_path $dict_dir || exit 1;
    touch $dict_dir/.done
  fi
fi

###########################################################################
# prepare data
###########################################################################

if [ $stage -le 1 ]; then
  for x in dev train ; do
    if [[ ! -f data/${x}_far/.done ]]; then
      local/prepare_data.sh --python_path $python_path --nj 1 ${misp2021_corpus} ${enhancement_dir} \
        $x $dict_dir data/${x}_far || exit 1;
      touch data/${x}_far/.done
    fi
  done
fi
dev_nj=$(wc -l data/dev_far/spk2utt | awk '{print $1}' || exit 1;)
if [ $dev_nj -ge $nj ]; then
    dev_nj=$nj
fi

###########################################################################
# prepare language module
###########################################################################

# L
if [ $stage -le 2 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    $dict_dir "$oovSymbol" data/local/lang data/lang  || exit 1;
fi

# arpa LM
if [ $stage -le 3 ]; then
  if [[ ! -f data/srilm/lm.gz || data/srilm/lm.gz -ot data/train_far/text ]]; then
    local/train_lms_srilm.sh --train-text data/train_far/text --dev-text data/dev_far/text \
      --oov-symbol "$oovSymbol" data/ data/srilm
  fi
fi

# prepare lang_test
if [ $stage -le 4 ]; then
  if [[ ! -f data/lang/G.fst || data/lang/G.fst -ot data/srilm/lm.gz ]]; then
    utils/format_lm.sh data/lang data/srilm/lm.gz data/local/dict/lexicon.txt data/lang_test
  fi
fi

mkdir -p exp

###########################################################################
# feature extraction
###########################################################################
if [ $stage -le 5 ]; then
  mfccdir=mfcc
  for x in dev_far train_far ; do
    if [ ! -f data/$x/mfcc.done ]; then
      steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $nj data/$x exp/make_mfcc/$x $mfccdir
      utils/fix_data_dir.sh data/$x
      steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
      utils/fix_data_dir.sh data/$x
      touch data/$x/mfcc.done
    fi
  done

  # subset the training data for fast startup
  for x in 50 100; do
    utils/subset_data_dir.sh data/train_far ${x}000 data/train_far_${x}k
  done
fi

###########################################################################
# mono phone train
###########################################################################
if [ $stage -le 6 ]; then
  if [ ! -f exp/mono/mono.train.done ]; then
    steps/train_mono.sh --boost-silence $boost_sil --nj $nj --cmd "$train_cmd" data/train_far_50k data/lang exp/mono || exit 1;
    touch exp/mono/mono.train.done
  fi
  
  # make graph
  if [ ! -f exp/mono/mono.mkgraph.done ]; then
    utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
    touch exp/mono/mono.mkgraph.done
  fi

  # decoding
  if [ ! -f exp/mono/mono.decode.done ]; then
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj exp/mono/graph data/dev_far exp/mono/decode_dev_far || exit 1;
    touch exp/mono/mono.decode.done
  fi
fi

###########################################################################
# tr1 100k delta+delta-delta
###########################################################################
if [ $stage -le 7 ]; then
  # alignment
  if [ ! -f exp/mono_ali/mono.align.done ]; then
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_far_100k data/lang exp/mono exp/mono_ali || exit 1;
    touch exp/mono_ali/mono.align.done
  fi

  # training
  if [ ! -f exp/tri1/tri1.train.done ]; then
    steps/train_deltas.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 data/train_far_100k data/lang exp/mono_ali exp/tri1 || exit 1;
    touch exp/tri1/tri1.train.done
  fi

  # make graph
  if [ ! -f exp/tri1/tri1.mkgraph.done ]; then
    utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
    touch exp/tri1/tri1.mkgraph.done
  fi

  # decoding
  if [ ! -f exp/tri1/tri1.decode.done ]; then
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri1/graph data/dev_far exp/tri1/decode_dev_far || exit 1;
    touch exp/tri1/tri1.decode.done
  fi
fi


###########################################################################
# tri2 all delta+delta-delta
###########################################################################
if [ $stage -le 8 ]; then
  # alignment
  if [ ! -f exp/tri1_ali/tri1.align.done ]; then
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_far data/lang exp/tri1 exp/tri1_ali || exit 1;
    touch exp/tri1_ali/tri1.align.done
  fi

  # training
  if [ ! -f exp/tri2/tri2.train.done ]; then
    steps/train_deltas.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 data/train_far data/lang exp/tri1_ali exp/tri2 || exit 1;
    touch exp/tri2/tri2.train.done
  fi

  # make graph
  if [ ! -f exp/tri2/tri2.mkgraph.done ]; then
    utils/mkgraph.sh data/lang_test exp/tri2 exp/tri2/graph || exit 1;
    touch exp/tri2/tri2.mkgraph.done
  fi

  # decoding
  if [ ! -f exp/tri2/tri2.decode.done ]; then
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri2/graph data/dev_far exp/tri2/decode_dev_far || exit 1;
    touch exp/tri2/tri2.decode.done
  fi
fi


###########################################################################
# tri3 all lda+mllt
###########################################################################
if [ $stage -le 9 ]; then
  # alignment
  if [ ! -f exp/tri2_ali/tri2.align.done ]; then
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_far data/lang exp/tri2 exp/tri2_ali || exit 1;
    touch exp/tri2_ali/tri2.align.done
  fi

  # training
  if [ ! -f exp/tri3/tri3.train.done ]; then
    steps/train_lda_mllt.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesMLLT $numGaussMLLT data/train_far data/lang exp/tri2_ali exp/tri3 || exit 1;
    touch exp/tri3/tri3.train.done
  fi

  # make graph
  if [ ! -f exp/tri3/tri3.mkgraph.done ]; then
    utils/mkgraph.sh data/lang_test exp/tri3 exp/tri3/graph || exit 1;
    touch exp/tri3/tri3.mkgraph.done
  fi

  # decoding
  if [ ! -f exp/tri3/tri3.decode.done ]; then
    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri3/graph data/dev_far exp/tri3/decode_dev_far || exit 1;
    touch exp/tri3/tri3.decode.done
  fi
fi


###########################################################################
# tri4 all sat
###########################################################################
if [ $stage -le 10 ]; then
  # alignment
  if [ ! -f exp/tri3_ali/tri3.align.done ]; then
    steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj 10 data/train_far data/lang exp/tri3 exp/tri3_ali || exit 1;
    touch exp/tri3_ali/tri3.align.done
  fi

  # tri4
  # training
  if [ ! -f exp/tri4/tri4.train.done ]; then
    steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train_far data/lang exp/tri3_ali exp/tri4|| exit 1;
    touch exp/tri4/tri4.train.done
  fi

  # make graph
  if [ ! -f exp/tri4/tri4.mkgraph.done ]; then
    utils/mkgraph.sh data/lang_test exp/tri4 exp/tri4/graph || exit 1;
    touch exp/tri4/tri4.mkgraph.done
  fi

  # decoding
  if [ ! -f exp/tri4/tri4.decode.done ]; then
    steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri4/graph data/dev_far exp/tri4/decode_dev_far || exit 1;
    touch exp/tri4/tri4.decode.done
  fi
fi


###########################################################################
# tri5 audio-only tdnnf with sp
###########################################################################
if [ $stage -le 11 ]; then
  # chain TDNN
  local/chain/run_tdnn_1b.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
    --nnet3-affix _train_far
fi


###########################################################################
# tri6 all audio-only tdnnf without sp
###########################################################################
if [ $stage -le 12 ]; then
  # chain TDNN without sp
  local/chain/run_tdnn_1b_no_sp.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
    --nnet3-affix _train_far
fi


###########################################################################
# tri7 all audio-visual tdnnf withoutsp
###########################################################################
if [ $stage -le 13 ]; then
  # extract visual ROI, store as npz (item: data); extract visual embedding; concatenate visual embedding and mfcc
  for x in dev_far train_far ; do
    local/extract_far_video_roi.sh --python_path $python_path --nj ${nj} data/${x} $data_roi/${x} data/${x}_hires || exit 1;
  done
  # chain audio-visual TDNN
  local/chain/run_tdnn_1b_av.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
    --nnet3-affix _train_far
fi


###########################################################################
# tri8 all audio-visual tdnnf withsp
###########################################################################
if [ $stage -le 14 ]; then
  # extract visual ROI, store as npz (item: data); extract visual embedding; concatenate visual embedding and mfcc
  local/extract_far_video_roi_sp.sh --python_path $python_path --nj ${nj} data/train_far $data_roi data/train_far_sp_hires
  # chain audio-visual TDNN
  local/chain/run_tdnn_1b_sp_av.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
    --nnet3-affix _train_far
fi

###########################################################################
# show result
###########################################################################
if [ $stage -le 15 ]; then
  # getting results (see RESULTS file)
  for x in exp/*/decode_dev_far exp/*/*/decode_dev_far exp/*/*/*/decode_dev_far; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
fi
