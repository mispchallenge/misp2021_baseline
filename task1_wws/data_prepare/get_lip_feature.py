import numpy as np  
import cv2
import os
import argparse
import time
from os import path
from tqdm import tqdm


def checkdir(_dir):
	try:
		os.makedirs(_dir)
	except:
		pass

def get_file(_dir, _sig):
	L = []
	for root, dirs, files in os.walk(_dir): 
		for file in files: 
			if os.path.splitext(file)[1] == _sig: 
				L.append(os.path.join(root, file)) 
	return L

def extract_lip_feature(_video_file):
	_video_file = tqdm(_video_file)
	for index, file in enumerate(_video_file):
		video_file = file.replace('_lip_roi.npy', '.mp4')
		if path.exists(video_file):
			lip_roi = np.load(file)
			cap = cv2.VideoCapture(video_file)
			i = 0
			lips = []
			while True:
				ret, image = cap.read()
				if ret == 0:
					break
				lip = image[lip_roi[i][1]:lip_roi[i][3], lip_roi[i][0]:lip_roi[i][2]]
				lip = cv2.resize(lip, (96,96))
				lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
				lips.append(lip[:,:,np.newaxis])
				i += 1
			cap.release()
			
			lips_file = np.reshape(np.array(lips),(len(lips),96,96,1))
			lips_save = file.split('_lip_roi')[0]
			np.save(lips_save, lips_file)
		
	return 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_root', default='./MISP2021_AVWWS/', help='input lip roi dir')

	args = parser.parse_args()
	
	input_lip_roi = args.data_root
	
	t0 = time.time()
	lip_file = get_file(input_lip_roi, '.npy')
	
	
	extract_lip_feature(lip_file)
	
	tn = time.time()
	print('total time: %f' %(tn-t0))






















