# MISP2021 Challenge Task2 NN-HMM Based AVSR Baseline

- **Data preparation**

  - **speech enhancement**

  We provide two baseline speech enhancement front-ends, Weighted Prediction Error(WPE) dereverberation and  weighted delay-and-sum(DAS) beamforming, to reduce reverberations and noises of speech signals. These two algorithms are implemented with open-source toolkits, [nara_wpe](https://github.com/fgnt/nara_wpe) and [BeamformIt](https://github.com/xanguera/BeamformIt), respectively.

  - **prepare data and language directory for kaldi**

  For training, development, and test sets, we prepare data directories and the lexicon in the format expected by  [kaldi](http://kaldi-asr.org/doc/data_prep.html) respectively. Note that we choose [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) raw resource and convert it to kaldi lexicon format.

- **Language model**

  We segment MISP speech transcription for language model training by applying [DaCiDian](https://github.com/aishell-foundation/DaCiDian.git) as dict and [Jieba](https://github.com/fxsjy/jieba) open-source toolkit. For the language model, we choose a maximum entropy-based 3-gram model, which achieves the best perplexity, from n-gram(n=2,3,4) models trained on MISP speech transcripts with different smoothing algorithms and parameters sets. And the selected 3-gram model has 516600 unigrams, 432247 bigrams, and 915962 trigrams respectively.  Note that the temporary and final language models are stored in /data/srilm.

- **Acoustic model**

  The acoustic model of the ASR system is built largely following the Kaldi [CHIME6](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1) recipes which mainly contain two stages: GMM-HMM state model and TDNN deep learning model.

  - **GMM-HMM**

    For features extraction, we extract 13-dimensional MFCC features plus 3-dimensional pitches. As a start point for triphone models, a monophone model is trained on a subset of 50k utterances.  Then a small triphone model and a larger triphone model are consecutively trained using delta features on a subset of 100k utterances and the whole dataset respectively. In the third triphone model training process, an MLLT-based global transform is estimated iteratively on the top of LDA feature to extract independent speaker features. For the fourth triphone model, feature space maximum likelihood linear regression (fMLLR) with speaker adaptive training (SAT) is applied in the training.

  - **NN-HMM**

    Based on the tied-triphone state alignments from GMM, TDNN is configured and trained to replace GMM. Here two data augmentation technologies, speed-perturbation and volume-perturbation are applied on signal level. The input features are 40-dimensional high-resolution MFCC features with cepstral normalization. Note that for each frame-wise input, a 100-dimensional i-vector is also attached, whose extractor was trained on the expanded corpus. An advanced time-delayed neural network (TDNN) baseline using lattice-free maximum mutual information (LF-MMI) training and other strategies is adopted in the system, and you can consult the [paper](https://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) and the [document](https://kaldi-asr.org/doc/chain.html) for more details.

- **Audio-Visual Speech Recognition**

  Based on the  NN-HMM hybrid ASR system mentioned above, we try to use lipreading to enhance the system. To get visual embeddings, we firstly crop mouth ROIs from video streams, then use the [lipreading TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)  to extract 512-dimensional features. Still using TDNN, we simply concentrate two modals embeddings(512+40+100) in the training stage. And the results preliminary show that the video information does help enhance ASR and it still has a very big upgrade space. Here comes the results.

## Results

| Models         | CCER of Dev |
| -------------- | ----------- |
| Chain-TDNN-A   | 65.89       |
| Chain-TDNN-A*  | 63.03       |
| Chain-TDNN-AV  | 65.87       |
| Chain-TDNN-AV* | 61.80       |

We denote Chain-TDNN-A as audio-only ASR and Chain-TDNN-AV as AVSR, and the symbol * indicates the model is trained with augmentation strategy.


## Quick start

- **Setting Local System Jobs**

```
# Setting local system jobs (local CPU - no external clusters)
export train_cmd=run.pl
export decode_cmd=run.pl
```

- **Download Coordinate Information Used to Crop ROI**

```
dropbox url:https://www.dropbox.com/s/m3a14wvukfkjqmr/misp2021_task2_roi.zip?dl=0
verification code: jTNEJkp83c

Baidu clouddisk url：https://pan.baidu.com/s/1B2Ocj9Gqyi297kZ_ubu-9w 
verification code：1p1j 

MD5：50e2a2ea10a436fd389d6df1f9881831
```

- **Setting  Paths**

```
--- path.sh ---
# Defining Kaldi root directory
export KALDI_ROOT=
# Setting paths to useful tools
export PATH=
# Enable SRILM
. $KALDI_ROOT/tools/env.sh
# Variable needed for proper data sorting
export LC_ALL=C

--- run.sh ---
# Defining corpus directory
misp2021_corpus=
# Defining path to beamforIt executable file
bearmformit_path = 
# Defining path to python interpreter
python_path = 
# the directory to host coordinate information used to crop ROI 
data_roi =
# dictionary directory 
dict_dir= 
```

- **Run Training**

```
./run.sh 
# options:
		--stage      -1  change the number to start from different training stages
		--nnet_stage -10 the number controls tdnn training stages including preprocessing and postprocessing
```

- **Other Tips**

Here some naming rules for directories produced during the four TDNN models training processing

| Models         | Data for Training          | Data for Alignment   | Model Directories   |
| -------------- | -------------------------- | -------------------- | ------------------- |
| Chain-TDNN-A   | data/train_far_hires       | data/train_far       | exp/train_far       |
| Chain-TDNN-A*  | data/train_far_sp_hires    | data/train_far_sp    | exp/train_far_sp    |
| Chain-TDNN-AV  | data/train_far_hires_av    | data/train_far_av    | exp/train_far_av    |
| Chain-TDNN-AV* | data/train_far_sp_hires_av | data/train_far_sp_av | exp/train_far_av_sp |

## Requirments

- **Kaldi**

- **Python Packages:**

  numpy

  tqdm

  [jieba](https://github.com/fxsjy/jieba)

- **Other Tools:**

  [nara_wpe](https://github.com/fgnt/nara_wpe)

  [Beamformit](https://github.com/xanguera/BeamformIt)

  SRILM

  [Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)

## Citation

If you find this code useful in your research, please consider to cite the following papers:

```bibtex

```

## License

It is noted that the code can only be used for comparative or benchmarking purposes. Users can only use code supplied under a [License](./LICENSE) for non-commercial purposes.

