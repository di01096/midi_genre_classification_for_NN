import train
import os
import numpy as np
# 분류 DNN 모델 구현 ########################
genre=os.listdir(os.getcwd()+'/mu/')
predic_dir='/predict/'
arr1,_=train.get_data_set_of_XY(d=predic_dir,predict=True)


from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, instrument_num,Time_step, Nout):
        super().__init__()
        self.add(layers.Dense(128,input_shape=(instrument_num,Time_step,2),activation='relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(32, activation='relu'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.2))
        self.add(layers.Flatten())
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
# 2. 모델 불러오기
model = DNN(arr1.shape[1],arr1.shape[2],  len(genre))
model.load_weights('model.h5')
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
x=os.listdir(os.getcwd()+predic_dir)
for i,arr in enumerate(arr1):
# model evaluation
    arr=np.reshape(arr,(1,arr.shape[0],arr.shape[1],arr.shape[2]))
    score = model.predict(arr,verbose=0)
    print(x[i])
    print(score)
    print(genre[0]+' 일 확률 : ',score[0][0])
    print(genre[1]+' 일 확률 : ',score[0][1])
    print(genre[2]+' 일 확률 : ',score[0][2])
    print(genre[3]+' 일 확률 : ',score[0][3])
    print(genre[4]+' 일 확률 : ',score[0][4])
