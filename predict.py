import train
import music21
import os
import numpy as np
from numpy import argmax
# 분류 DNN 모델 구현 ########################
from keras import layers, models
file_name=os.getcwd()+'/sample.mid'
midi_obj = music21.converter.parse(file_name)
mel_data = train.create_mel_data_each_file(midi_obj)


class RNN(models.Sequential):
    def __init__(self, Time_step, Filter_num, Nout):
        super().__init__()

        self.add(layers.LSTM(Filter_num,input_shape=(Time_step,3),return_sequences=True))
        self.add(layers.BatchNormalization())
        self.add(layers.LSTM(Filter_num, input_shape=(Time_step, 3), return_sequences=True))
        self.add(layers.BatchNormalization())
        self.add(layers.LSTM(Filter_num))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

mel_arr = []
for i, mel_data_i in enumerate(mel_data):
    for key, value in sorted(mel_data_i.items()):
        mel_arr.append(mel_data_i[key])
print(mel_arr)
curve_seq_list=train.create_curve_seq(mel_arr)
print(curve_seq_list)
arr=curve_seq_list

arr=np.array([arr])
print(arr)
arr = train.pad_sequences(arr,padding='post',maxlen=1000,dtype=np.float)
print(arr)
arr= np.reshape(arr,(arr.shape[0],-1,3))
for i in range(len(arr)):
    arr[i] = train.normalize(arr[i], axis=0, norm='max')
print(arr.shape)
# 2. 모델 불러오기

from keras.models import load_model
model= RNN(1000,64,5)
model.load_weights('model.h5')
model.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
# model evaluation
score = model.predict(arr,verbose=0)

print(score)
print('Classisc 일 확률 : ',score[0][0])
print('Country 일 확률 : ',score[0][1])
print('Dance 일 확률 : ',score[0][2])
print('Jazz 일 확률 : ',score[0][3])
print('Rock 일 확률 : ',score[0][4])