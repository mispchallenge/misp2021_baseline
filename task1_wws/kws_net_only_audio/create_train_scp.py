import os
import glob
import codecs
import argparse
import scipy.io.wavfile as wf

def find_wav(data_root, scp_dir, scp_name='wpe'):
    all_wav_paths = []
    for i in data_root:
        all_wav_paths += glob.glob(i)
    sorted_wav_paths = sorted(all_wav_paths)
    lines = ['' for _ in range(1)]
    for wav_idx in range(len(sorted_wav_paths)):
        line = sorted_wav_paths[wav_idx]
        if 'negative_train' in scp_name:
            fs, data = wf.read(line)
            if len(data) > 12000:
                if len(data) > 48001:
                    data = data[(len(data)-48000)//2:(len(data)+48000)//2]
                    line = line.replace('.wav', '_limit3s.wav')
                    wf.write(line, fs, data)
        line += '\n'
        lines[0] += line

    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir, exist_ok=True)

    with codecs.open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as handle:
        handle.write(lines[0][:-1])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='./data/', type=str)
    args = parser.parse_args()  
    scp_name = ['positive_train', 'negative_train', 'positive_dev', 'negative_dev']
    data_root = args.data_root
    SNR_list = [-15,-10,-5,0,5,10,15]
    positive_train = [data_root+'positive/audio/train/*/*_1_*_*.wav']
    negative_train = [data_root+'negative/audio/train/*/*_1_*_*.wav']
    for snr in SNR_list:
        positive_train.append(data_root+'positive_addnoise/SNR_{}db/*/*/*/*_1mic.wav'.format(str(snr)))
        negative_train.append(data_root+'negative_addnoise/SNR_{}db/*/*/*/*_1mic.wav'.format(str(snr)))
    positive_dev = [data_root+'positive/audio/dev/*/*_1_*_*.wav']
    negative_dev = [data_root+'negative/audio/dev/*/*_1_*_*.wav']
    positive_dev_middle = [data_root+'positive/audio/dev/middle/*_1_*_*.wav']
    positive_dev_far = [data_root+'positive/audio/dev/far/*_1_*_*.wav']
    negative_dev_middle = [data_root+'negative/audio/dev/middle/*_1_*_*.wav']
    negative_dev_far = [data_root+'negative/audio/dev/far/*_1_*_*.wav']
    print('*************')
    find_wav(positive_train, 'scp_dir', 'positive_train')
    find_wav(negative_train, 'scp_dir', 'negative_train')
    find_wav(positive_dev_middle, 'scp_dir', 'positive_dev_middle')
    find_wav(positive_dev_far, 'scp_dir', 'positive_dev_far')
    find_wav(negative_dev_middle, 'scp_dir', 'negative_dev_middle')
    find_wav(negative_dev_far, 'scp_dir', 'negative_dev_far')
    print('*************')
