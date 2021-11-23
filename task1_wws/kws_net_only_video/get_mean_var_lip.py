import numpy as np 
import os

def get_file(file_dir,sig):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == sig:
               L.append(os.path.join(root , file))
    return L

def get_mean_dev_lip(trainlist):
	mean_np_list=[]
	var_np_list=[]
	length_list = []
	i = 0
	for it in trainlist:
		cu_np=np.reshape(np.load(it),(-1,96*96))
		if (np.load(it)).shape[0] > 3:
			mean_np_list.append(np.mean(cu_np,0))
			length_list.append(cu_np.shape[0])
			i = i + 1
			if i%500 == 0:
				print('mean process i :',i)
	length_list_np = np.reshape(np.array(length_list),(-1,1))
	mean = np.sum(np.multiply(np.array(mean_np_list),length_list_np)/np.sum(np.array(length_list)),0)
	i = 0
	for it in trainlist:
		if (np.load(it)).shape[0] > 3:
			cu_npvar = np.mean(np.square(np.reshape(np.load(it),(-1,96*96))-mean),0)
			var_np_list.append(cu_npvar)
			i = i + 1
			if i%500 == 0:
				print('var process i:', i)
	var = np.sum(np.multiply(np.array(var_np_list),length_list_np)/np.sum(np.array(length_list)),0)

	print('mean shape', mean.shape)
	print('var shape:', var.shape)
	
	return (mean,var)

trainset_positive = get_file('/yrfs2/cv1/hangchen2/hszhou2/Project_KWS_2021_MISP_Challenge/final_data_configuration_version5/MISP2021_AVWWS_Roi/z_upload_dropbox/zip_lip_npy/MISP2021_AVWWS/positive/video/train/middle/','.npy')
print('positive file num %d' %len(trainset_positive))
trainset_negative = get_file('/yrfs2/cv1/hangchen2/hszhou2/Project_KWS_2021_MISP_Challenge/final_data_configuration_version5/MISP2021_AVWWS_Roi/z_upload_dropbox/zip_lip_npy/MISP2021_AVWWS/negative/video/train/middle/','.npy')
print('negative file num %d' %len(trainset_negative))

trainset = trainset_positive + trainset_negative


_mean, _var=get_mean_dev_lip(trainset)
# _mean, _var=get_mean_dev_lip(trainset[0:10])
_mean = _mean.astype('float32')
_var = _var.astype('float32')
_mean = np.reshape(_mean,(96,96))
_var = np.reshape(_var,(96,96))
print('_mean shape', _mean.shape)
print('_var shape:', _var.shape)
np.savez('./train_mean_var_lip.npz',_mean=_mean,_var=_var)