# -*- coding:utf-8 -*-
import pyroomacoustics as pra
import numpy as np
import soundfile as sf
import os
import random
import argparse
def msec2time(msec):
    ms = msec%1000
    temp = msec//1000
    s = temp%60
    temp = temp//60
    m = temp%60
    temp = temp//60
    h = temp
    return "%02d:%02d:%02d.%03d"%(h,m,s,ms)

def get_duration(_fname):
	with contextlib.closing(wave.open(_fname,'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
	return duration

def read_romminfo(_rpath):
	tmp_dic = {}
	with open(_rpath) as fh:
		line = fh.readline()
		while line:
			tmp = line.split('\n')[0].split(' ')
			tmp_dic[tmp[0]] = tmp[1]
			line = fh.readline()
	return tmp_dic
parser = argparse.ArgumentParser()
parser.add_argument("--scp_path", default='add_reverb/positive.scp', type=str)
args = parser.parse_args()  
# read room information
room_info = read_romminfo('add_reverb/all_room_info.txt')

trainset_near_path = args.scp_path
with open(trainset_near_path) as f:
	lines = f.readlines()
trainset_near = [line.strip() for line in lines]
rt60_tgt = 0.4

for it in trainset_near:
	
	it = it.replace('\n', '')
	room_dim = []
	room_info_tmp = it.split('/')[-1].split('_')[0][1:]
	room_dim.append(float(room_info[room_info_tmp][0:3])/100)
	room_dim.append(float(room_info[room_info_tmp][4:7])/100)
	room_dim.append(float(room_info[room_info_tmp][8:11])/100)
	print(room_dim)
	mic_locs = np.c_[
    [room_dim[0]/2+0.05, (room_dim[1]-0.5)/2, 0.80],  # mic 1
    [room_dim[0]/2-0.05, (room_dim[1]-0.5)/2, 0.80],
]
	e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
	room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
	audio, fs = sf.read(it)
	wavlen = len(audio)
	# print('wav len'+str(len(audio))+'*'*50)
	speaker_location1 = random.uniform(-0.2,0.2)
	speaker_location2 = random.uniform(0.05,0.25)
	room.add_source([room_dim[0]/2+speaker_location1, speaker_location2, 1.00], signal=audio, delay=0.00)

	room.add_microphone_array(mic_locs)
	room.compute_rir()
	room.simulate()
	# print('processed len'+str(room.mic_array.signals.shape)+'*'*50)
	out_path = '/'.join(it.split('/')[:-1]).replace('tive/', 'tive_reverb/').replace('near', 'middle')
	if not os.path.exists(out_path):
		os.makedirs(out_path)
	for i in range(2):
		s = it.split('/')[-1].replace('.wav', '_{}mic.wav'.format(i))
		out_audio = out_path + '/' + s
		sf.write(out_audio, room.mic_array.signals[i, :wavlen] , fs)

















