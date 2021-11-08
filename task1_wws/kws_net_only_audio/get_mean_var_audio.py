import numpy as np 
from scipy.io import wavfile
from network_feature_extract import FilterBank
import torch
from tqdm import tqdm
def readfile(_path):
	tmp_list = []
	with open(_path) as fh:
		line = fh.readline()
		while line:
			tmp_list.append(line.split('\n')[0])
			line = fh.readline()
	return tmp_list

def get_mean_dev_audio(trainlist):
	mean_np_list=[]
	var_np_list=[]
	length_list = []
	i = 0
	feaExt = FilterBank()
	for it in trainlist:
		it = it.replace('\n', '')
		_, data = wavfile.read(it)
		# print(it)
		# print(data.shape)
		mel_spec, _ = feaExt(torch.from_numpy(data))
		mel_spec = mel_spec.numpy().T
		mean_np_list.append(np.mean(mel_spec,0))
		length_list.append(mel_spec.shape[0])
		i = i + 1
		if i%2000 == 0:
			print('mean process i :',i)
	print('calculate mean')
	length_list_np = np.reshape(np.array(length_list),(-1,1))
	mean = np.sum(np.multiply(np.array(mean_np_list),length_list_np)/np.sum(np.array(length_list)),0)
	print(mean.shape)
	print('calculate mean done')
	i = 0
	for it in trainlist:
		it = it.replace('\n', '')
		_, data = wavfile.read(it)
		mel_spec, _ = feaExt(torch.from_numpy(data))
		mel_spec = mel_spec.numpy().T
		cu_npvar = np.mean(np.square(mel_spec-mean),0)
		var_np_list.append(cu_npvar)
		i = i + 1
		if i%2000 == 0:
			print('var process i:', i)
	var = np.sum(np.multiply(np.array(var_np_list),length_list_np)/np.sum(np.array(length_list)),0)
	print('mean shape', mean.shape)
	print('var shape:', var.shape)
	
	return (mean,var)

def fbank_cmvn(trainlist):
	extractor = FilterBank()
	fbank_mean, fbank_std, frames_count = 0., 0., 0.
	for idx in tqdm(range(len(trainlist))):
		# fs,data = wavfile.read(trainlist[idx])
		it = trainlist[idx].replace('\n', '')
		_, data = wavfile.read(it)
		data = torch.from_numpy(data)
		data = torch.unsqueeze(data, 0)
		mixture_fbank, frame_num = extractor(data, len(data))
		# mixture_fbank = torch.unsqueeze(mixture_fbank, 1)
		print(mixture_fbank.shape, frame_num)
		for sample_idx in range(mixture_fbank.size(0)):
			avail_fbank = mixture_fbank[sample_idx, :, :frame_num[sample_idx]]
			updated_count = frames_count + frame_num[sample_idx]
			fbank_mean = fbank_mean * (frames_count / updated_count) + avail_fbank.sum(dim=1) / updated_count
			fbank_std = fbank_std * (frames_count / updated_count) + (avail_fbank ** 2).sum(dim=1) / updated_count
			frames_count = updated_count
	fbank_std = torch.sqrt(fbank_std - fbank_mean ** 2)
	return frames_count, fbank_mean.numpy(), fbank_std.numpy()

trainset_positive = readfile('scp_dir/positive_train.scp')
trainset_negative = readfile('scp_dir/negative_train.scp')

trainset =  trainset_negative+trainset_positive

# _, fb40_mean, fb40_var=fbank_cmvn(trainset[:600])
fb40_mean, fb40_var=get_mean_dev_audio(trainset)
fb40_mean = fb40_mean.astype('float32')
fb40_var = fb40_var.astype('float32')
np.savez('scp_dir/train_mean_var_fb40_.npz',_mean=fb40_mean,_var=fb40_var)