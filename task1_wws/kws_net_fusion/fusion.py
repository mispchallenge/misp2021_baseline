import numpy as np


def cal_indicator(pre,label):
	TP = 0.0
	FN = 0.0
	TN = 0.0
	FP = 0.0
	for i, it in enumerate(pre):
		if it == 1.0 and label[i] == 1.0:
			TP += 1.0
		elif it == 1.0 and label[i] == -0.0:
			FP += 1
		elif it == -0.0 and label[i] == 1.0:
			FN += 1
		elif it == -0.0 and label[i] == -0.0:
			TN += 1.0
	return TP, FP, TN, FN

def cal_score(i, pre):
	TP, FP, TN, FN = cal_indicator(pre, label)
	FAR, FRR = FP/(FP+TN), 1-TP/(TP+FN)
	score = FAR + FRR
	if score < dic[i][1]+ dic[i][2]:
		dic[i] = [threshold, FAR, FRR]
	

dic = {'audio':[0, 1, 1], 'video':[0, 1, 1], 'fusion':[0, 1, 1]}

label = list(np.load('result/video_label.npy'))
label = [i[0] for i in label]


audio_mid_score = np.load('result/audio_result.npy')
video_mid_score = np.load('result/video_result.npy')

N = 1000
for threshold in range(N):
	threshold /= N
	audio_mid = list(np.ceil(audio_mid_score - threshold))
	audio_mid = [i[0] for i in audio_mid]
	video_mid = list(np.ceil(video_mid_score - threshold))
	video_mid = [i[0] for i in video_mid]
	fusion_mid_mean = list(np.ceil((audio_mid_score + video_mid_score)/2 - threshold))
	fusion_mid_mean = [i[0][0] for i in fusion_mid_mean]
	for i in dic:
		if i == 'audio':
			cal_score(i, audio_mid)
		elif i == 'video':
			cal_score(i, video_mid)
		elif i == 'fusion':
			cal_score(i, fusion_mid_mean)


print(' '*20)
print('*'*20)


print("For the audio, midfield result: threshold = %.3f, FAR=%.4f, FRR:%.4f, Score:%4f" %(dic['audio'][0], dic['audio'][1], dic['audio'][2], dic['audio'][1]+dic['audio'][2]))
print("For the video, midfield result: threshold = %.3f, FAR=%.4f, FRR:%.4f, Score:%4f" %(dic['video'][0], dic['video'][1], dic['video'][2], dic['video'][1]+dic['video'][2]))
print("After fusion, midfield result: threshold = %.3f, FAR=%.4f, FRR:%.4f, Score:%4f" %(dic['fusion'][0], dic['fusion'][1], dic['fusion'][2], dic['fusion'][1]+dic['fusion'][2]))

print('*'*20)
print(' '*20)






