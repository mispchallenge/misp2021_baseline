import os
import glob
import argparse
import soundfile as sf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default='/yrfs2/cv1/hangchen2/data/', type=str)
parser.add_argument("--scp_path", default='add_noise/scp/positive.scp', type=str)
parser.add_argument("--mode", default='wav2raw', type=str, help='wav2raw or raw2wav')
args = parser.parse_args()  
data_root = args.data_root
scpfile = args.scp_path
mode = args.mode
if mode == 'wav2raw':
    poss = glob.glob(data_root + 'positive_reverb/audio/train/*/*.wav')
    print(scpfile)
    p = open(scpfile, 'w')
    poss.sort()
    line = ''
    for wav in poss:
        raw = wav.replace('.wav', '.raw')
        os.system('sox {} -b 16 -e signed-integer -c 1 -r 16000 -t raw {}'.format(wav, raw))
        line += raw
        line += '\n'
    p.write(line[:-1])
    p.close()
elif mode == 'raw2wav':
    poss_add=(data_root + 'positive_addnoise/*/audio/train/*/*.raw')
    poss=glob.glob(poss_add)
    for raw in poss:
        os.system('sox -b 16 -e signed-integer -c 1 -r 16000 -t raw {} {}'.format(raw, raw.replace('.raw', '.wav')))
else:
    print('unknown mode')