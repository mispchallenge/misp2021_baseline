import os
import glob
import codecs
import argparse
import numpy as np

def find_npy(data_root, scp_dir, scp_name='positive'):
    all_wav_paths = []
    for i in data_root:
        all_wav_paths += glob.glob(i)
    sorted_wav_paths = sorted(all_wav_paths)
    print(len(sorted_wav_paths))
    lines = ['' for _ in range(1)]
    for wav_idx in range(len(sorted_wav_paths)):
        line = sorted_wav_paths[wav_idx]
        if 'negative_train' in scp_name in scp_name:
            data = np.load(line)
            if data.shape[0] > 18:
                if len(data) > 75:
                    data = data[(len(data)-75)//2:(len(data)+75)//2]
                    line = line.replace('.npy', '_limit3s.npy')
                    np.save(line, data)
        line += '\n'
        lines[0] += line


    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir, exist_ok=True)

    with codecs.open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as handle:
        handle.write(lines[0][:-1])
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='../../MISP2021_AVWWS/', type=str)
    args = parser.parse_args()
    scp_name = ['positive_train', 'negative_train', 'positive_dev', 'negative_dev']
    data_root = args.data_root
    positive_train = [data_root+'positive/video/train/middle/*.npy']
    negative_train = [data_root+'negative/video/train/middle/*.npy']
    positive_dev = [data_root+'positive/video/dev/middle/*[0-9].npy']
    negative_dev = [data_root+'negative/video/dev/middle/*[0-9].npy']
    find_npy(positive_train, 'scp_dir', 'positive_train')
    find_npy(negative_train, 'scp_dir', 'negative_train')
    find_npy(positive_dev, 'scp_dir', 'positive_dev')
    find_npy(negative_dev, 'scp_dir', 'negative_dev')
    print('*************')
    print('*************')