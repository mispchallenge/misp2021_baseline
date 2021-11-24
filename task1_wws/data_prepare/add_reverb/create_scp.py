import os
import glob
import argparse
if __name__ == '__main__':
    print("collect near wavs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/yrfs2/cv1/hangchen2/data/', type=str)
    parser.add_argument("--scp_path", default='add_reverb/positive.scp', type=str)
    args = parser.parse_args()  
    data_root = args.data_root
    scp_path = args.scp_path
    data_dir = data_root + '/positive/audio/train/near'
    print(data_dir)
    f = open(scp_path, 'w')
    flag, n = '', 0
    for root, dirs, files in os.walk(data_dir):
        files.sort()
        for file in files:
            f.write(root +'/'+ file + '\n') 
    f.close()      
            