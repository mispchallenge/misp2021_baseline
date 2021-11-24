import os
import glob
import argparse
import soundfile as sf
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='/yrfs2/cv1/hangchen2/data/', type=str)
parser.add_argument("--scp_path", default='add_noise/middleNoise.scp', type=str)
args = parser.parse_args()  
data_root = args.data_root
wavs_add = (data_root + 'noise/audio/train/middle/*.wav')
wavs = glob.glob(wavs_add)
scpfile = args.scp_path
f = open(scpfile, 'w')
wavs.sort()
tmp = '_'.join(wavs[0].split('_')[:-1])
dir_ = tmp.replace('/audio/train/middle/', '/processedmiddle/').split('/')
os.system('mkdir -p {}'.format('/'.join(dir_[:-1])))
data, fs = sf.read(wavs[0])
line = ''
for wav in wavs:
    head = '_'.join(wav.split('_')[:-1])
    if head == tmp:
        data_, _ = sf.read(wav)
        data = np.append(data, data_)
    else:
        print(tmp)
        tmp = tmp.replace('/audio/train/middle/', '/processedmiddle/')
        sf.write(tmp+'.wav', data, fs)
        os.system('sox {} -b 16 -e signed-integer -c 1 -r 16000 -t raw {}'.format(tmp+'.wav', tmp+'.raw'))
        line = line + tmp+'.raw\n'
        data, _ = sf.read(wav)
        tmp = '_'.join(wav.split('_')[:-1])
f.write(line[:-1])
f.close()