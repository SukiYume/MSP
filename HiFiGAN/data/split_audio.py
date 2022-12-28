import os, shutil
import numpy as np
from pydub import AudioSegment
from sklearn.model_selection import train_test_split

file_list = [i for i in os.listdir('./zxc/') if i.endswith('wav')]

for i in file_list:
    audio           = AudioSegment.from_file('./zxc/' + i, 'wav')
    audio_time      = len(audio) #获取待切割音频的时长，单位是毫秒
    seg_length      = np.round(np.random.rand()*10+1, 1)

    cut_parameters  = np.arange(seg_length, audio_time/1000, seg_length)
    start_time      = int(0) #开始时间设为0

    for t in cut_parameters:
        stop_time   = int(t * 1000)  # pydub以毫秒为单位工作
        audio_chunk = audio[start_time: stop_time] #音频切割按开始时间到结束时间切割
        audio_chunk.export('qwe/{}-{:0>4d}.wav'.format(i.split('.')[0], int(t / seg_length)), format='wav')
        start_time = stop_time - 0  #开始时间变为结束时间前4s---------也就是叠加上一段音频末尾的4s
        
file_list = [i for i in os.listdir('./qwe/') if i.endswith('wav')]

string = ''
for i in range(len(file_list)):
    shutil.copy('./qwe/{}'.format(file_list[i]), './wer/{:0>5d}.wav'.format(i))
    string += '{:0>5d}'.format(i) + '|' + file_list[i] + '\n'
    
data_list = string.strip().split('\n')
train, val = train_test_split(data_list, test_size=0.01, shuffle=True)

with open('train_data.txt', 'w') as f:
    f.write('\n'.join(train))
    
with open('val_data.txt', 'w') as f:
    f.write('\n'.join(val))