
from keras.utils import np_utils
import music21
import os
import pickle
import numpy as np
from sklearn.preprocessing import normalize
##############################################3
# 멜로디 데이터를 파일로부터 가져오는 함수
###########################################
def create_mel_data_each_file(midi_obj):
    c = midi_obj.flat.getElementsByClass(music21.instrument.Instrument) # 1곡내의 악기 데이터만 따로 추출해서 저장해둠.
    mel_data = []

    for i, m in enumerate(midi_obj): #시퀀스데이터를 트랙별로 나눔. 1개씩 볼 수있도록 for문 사용
        inst = None                 # 악기값을 넣는 변수
        mel_data.append(dict())     # 멜로디 데이터를 넣는 변수
        for n in m:                # 1개의 트랙을 다시 1개의 노트씩 보도록 나눔.
            if (n in c):           #  미리 저장해둔 악기데이터에서 나온 값과 일치하는 악기데이터를 만나면,
                inst = n.midiProgram # 뒤로 나오는 노트들은 이 악기를 사용하므로 변수에 악기ID를 넣음.


                        ##################################
            if (type(n) == music21.stream.Voice): # 시퀀스데이터 내에도 시퀀스 데이터가 있어 이 데이터도 풀어줌.
                for x in n:
                    if (x in c): # 이후 나오는 것은 뒤에서 설명한 것과, 악기값을 가져오는 것과 동일.
                        inst = x.midiProgram
                    if (type(x) == music21.chord.Chord):
                        if x.offset not in mel_data[i]:
                            mel_data[i][x.offset] = dict()
                        mel_data[i][x.offset]['offset'] = x.offset
                        for p in x.pitches:
                            mel_data[i][x.offset]['note'] = p.midi
                        mel_data[i][x.offset]['instrument'] = inst
                    if (type(x) == music21.note.Note):
                        if x.offset not in mel_data[i]:
                            mel_data[i][x.offset] = dict()
                        mel_data[i][x.offset]['offset'] = x.offset
                        prev_p = 0
                        for p in x.pitches:
                            if prev_p < p.midi:
                                mel_data[i][x.offset]['note'] = p.midi
                            prev_p = p.midi
                        mel_data[i][x.offset]['instrument'] = inst

                        ###############################
            if (type(n) == music21.chord.Chord): #Chord라는 데이터가 나온다면,
                if n.offset not in mel_data[i]: # 그값의 offset을 가져옴.
                    mel_data[i][n.offset] = dict() #offest값을 dict에 저장. // offset은 노래가 시작하고 언제 쳐질지를 결정.(시간)
                mel_data[i][n.offset]['offset'] = n.offset
                for p in n.pitches:
                    mel_data[i][n.offset]['note'] = p.midi #같은 offset내에 note값 즉, 음정값을 넣음.
                mel_data[i][n.offset]['instrument'] = inst #같은 offset내에 악기값을 넣음.
            if (type(n) == music21.note.Note):   #Note라는 데이터가 나온다면,
                if n.offset not in mel_data[i]:   # offset값을 가져옴.
                    mel_data[i][n.offset] = dict()  #이또한 저장.
                mel_data[i][n.offset]['offset'] = n.offset
                prev_p = 0
                for p in n.pitches:
                    if prev_p < p.midi:
                        mel_data[i][n.offset]['note'] = p.midi #음정값을 가져와 저장.
                    prev_p = p.midi
                mel_data[i][n.offset]['instrument'] = inst  #악기값도 저장.
    print('data length : ',sum([len(a) for a in mel_data]))   # 전체 노트즉, 음정들의 총 갯수를 적어서 보여줌.
    return mel_data           #음정과 offset과 악기정보가 담긴 1개의 곡의 데이터를 리턴.

def get_midi_set(data_path,path_name,predict=False):
    file_list = [] #파일리스트를 초기화
    for file_name in os.listdir(os.path.expanduser(data_path)): #폴더위치를 가져와 1개씩 나눔.
        if file_name.endswith('.mid') or file_name.endswith('.midi'): #폴더내에 midi나 mid 라는 확장자를 찾으면,
            file_list.append(data_path + file_name)            # 파일리스트에 midi나 mid의 전체 이름과 위치를 넣음.
    mel_arr_list = []
    for file_name in file_list:
        print(file_name + ' \n get data')  #파일이름을 보여줌.
        midi_obj = music21.converter.parse(file_name)  #미디파일을 시퀀스 데이터로 치환.
        mel_data = create_mel_data_each_file(midi_obj) #파일을 데이터화 해서 가져옴.
        print('finished..')     #끝

        mel_arr = []
        for i,mel_data_i in enumerate(mel_data):
            for key, value in sorted(mel_data_i.items()):
                mel_arr.append(mel_data_i[key])    # dict 데이터를 간결화. offset(dict의 이름과 dict내의 값)이 2번 들어가 있는걸 빼주는 형식.
        mel_arr_list.append(mel_arr)
    print('data set saving..') #데이터를 세이브함.
    if predict:                                         # 만약 predict 상황에 사용되면,
        preprocessed_dir = "./predict_preprocessed_data/" #predict 전용 폴더가 있는지 확인하여,
        if not os.path.exists(preprocessed_dir):         # 없다면,
            os.makedirs(preprocessed_dir)               #  전용 폴더를 만들어줌.
    else :                                            # predict 상황이 아닌경우, 즉, train하는 중일경우,
        preprocessed_dir = "./preprocessed_data/"      # preprocessed 폴더가 있는지 확인하여,
        if not os.path.exists(preprocessed_dir):       # 없다면,
            os.makedirs(preprocessed_dir)              # preprocessed 폴더를 만들어줌.

    with open(preprocessed_dir + path_name+"_mel_arr_list.p", "wb") as fp:  # 데이터 저장용 파일의 이름을 지정.
        pickle.dump(mel_arr_list, fp)                                      # 데이터를 저장.
    print('save finished..')
    return mel_arr_list

#####################################
# 데이터 전처리중 미분값을 만드는 함수
#####################################

def create_curve_seq(mel_arr):
    curve_seq = []
    if(len(mel_arr)>1):                        #멜로디 데이터가 1개보다 많을 경우만,
        for idx in range(1, len(mel_arr)):     #멜로디 데이터를 indexing하여 쭉 나열,
            if len(mel_arr[idx-1])==3 and len(mel_arr[idx])==3 :      # 만약, 악기정보, offset정보, 음정정보가 1개라도 없으면 못들어가도록함.
                curr_p_diff =  mel_arr[idx]['note']-mel_arr[idx-1]['note']     # 음정사이의 미분값을 저장.
                curr_t_diff = mel_arr[idx]['offset'] - mel_arr[idx-1]['offset'] #offset 즉, 음정과 음정사이의 시간차를 저장.
                if mel_arr[idx-1]['instrument']==None: #만약 악기데이터가 없는 노트일 경우,
                    x=0                                # 0으로 악기값을 저장.
                else:                                  # 악기데이터가 있다면,
                    x=mel_arr[idx-1]['instrument']+1   # 악기ID에 +1을 하여 저장.
                curve_seq.append([curr_p_diff, curr_t_diff,x]) # 음정미분/시간미분/악기 값을 list로 넣음.
    return curve_seq                                          # 전체곡이 끝나면 1개의 2차원 리스트(곡에 들어있는 유효한 노트수,3)를 리턴.
############################################
# training에 사용될 X,Y배열을 가져오는 함수
#############################################

def get_data_set_of_XY(d='/mu/',predict=False): #d= 기본적 위치주소 , predict는 predict할 경우 사용.
    org_path=os.getcwd()+d                 # d 값에 현재위치를 더한 전체위치로 만듬.
    if not(predict):                       # training한다면,
        paths=os.listdir(org_path)         # 기본적 위치주소내의 폴더 즉, 장르폴더를 가져옴.
        for i,_ in enumerate(paths):       # 장르폴더를 1개씩 살펴봄.
            paths[i]=str(org_path)+str(paths[i])+'/'   # 장르폴더의 전체위치를 저장. ex) 'C:/..../mu/classics/'
        preprocessed_dir = "/preprocessed_data/"       # preprocessed 폴더이름을 미리 지정.

    else:                                  # predict 상황일 경우,
        preprocessed_dir = "/predict_preprocessed_data/" # predict용 전처리 폴더이름을 미리 지정.
        paths=[org_path]                   # d값자체를 미리 지정. 즉, 장르폴더를 따로 가져오지 않음.
    class_num = len(paths)                # 장르의 갯수를 가져옴.
    Y_set=[]                             # 정답지 배열을 초기화
    arr=[]                               # 데이터를 불러오는데에 쓰일 배열을 초기화
    name=os.listdir(org_path)            # 장르이름을 가져옴.
    if not (os.path.isdir(os.getcwd() + preprocessed_dir)):   # 만약, preprocessed 폴더가 없다면,
        print('getting midi file set..')
        for i, p in enumerate(paths):                         # 각 장르폴더별로 들어가서
            n = name[i]
            arr.append(get_midi_set(p, n,predict))            # 그 안의 midi파일을 가져와 데이터화해서 arr에 저장.
            if not(predict):                                  # training상황 일경우
                Y_set.append(np_utils.to_categorical(i, class_num) * len(arr[i]))  # 장르폴더안의 midi파일 전체갯수만큼 정답지 생성.
    else:                                                     # preprocessed 폴더가 있다면,
        print('load midi file set from dir(preprocessed_data)...')
        for i,file_name in enumerate(os.listdir(os.getcwd()+preprocessed_dir)):  #
            if file_name.endswith('.p') or file_name.endswith('.p'):            # 미리 저장되있는 데이터를 가져옴.
                with open(os.getcwd() + preprocessed_dir + file_name, "rb") as fp:
                    arr.append(pickle.load(fp))                                # 데이터를 배열로 저장.
                    if not(predict) :                                           # training 상황일 경우,
                        for _ in arr[i]:
                            Y_set.append(np_utils.to_categorical(i,class_num))  # 장르의 midi파일 갯수만큼 정답지 생성.
    print('finished..')
    curve_seq_list = []
    arr2=[]
    for ar in arr:              # 배열이 dict형태이고 미분값이 아니기에 이를 위의 함수에 넣고, 미분값과 악기, 리스트형으로 반환
        for mel_arr in ar:
            curve_seq_list.append(create_curve_seq(mel_arr))
            arr2.append(mel_arr)   # 여기서 미분값뿐이기에 현재의 시간위치를 알수있는 offset데이터를 위해 따로 저장.
    print('second preprocessing..')
    del arr # 메모리 활용을 위해 arr는 사용되지 않아서 free 시킴.
    b = len(dir(music21.instrument))+1  # 악기갯수+유효하지않는 악기 =143을 미리 b라는 변수로 저장.
    time_num=500*4                      # 사용될 offset의 최대값과 4분의 1박자만큼을 가지도록 time_num 변수를 저장.
    arr1=np.zeros((len(curve_seq_list),b,time_num,2),dtype=np.float)
         # 사용될 배열(전체곡, 악기갯수, 사용될 시간축,데이터크기)을 전체 zero로 만들어 이후 여기에 값을 넣음.
    for i,x in enumerate(curve_seq_list):
        for j,y in enumerate(x):
            u=int(arr2[i][j]['offset']*4) # 사용될 시간축에 1/4박자가 최대이므로 4를 곱해 정수로 치환시켜 u값에 저장.
            if u<time_num and y[2]!=0 and y[1]!=0 : # 시간축내에서 0의 값의 시간미분값을 가지지않는 데이터라면,
                arr1[i][y[2]][u] = y[:1]           # 음정미분,시간미분 데이터를  현재곡/악기ID/시간축위치 에 저장.
    del u,arr2  # 메모리를 위해서 사용되지않는 변수를 free
    arr1=np.array(arr1)
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            arr1[i][j] = normalize(arr1[i][j], axis=0)    # 배열을 노말라이즈시킴.
    print(arr1.shape)
    print('input_shape =',arr1.shape)                     # input 배열을 확인.

    if not(predict) :                                     # training 상황일경우,
        Y_set = np.array(Y_set)
        Y_set=np.reshape(Y_set,(-1,class_num))            # 정답지를 다시 확인.
        print('output_shape =',Y_set.shape)
    print('prepprocessing finished..')
    return arr1,Y_set                                     # input과 정답지 즉, output을 반환

############################
# 모델의 정확성과 loss를 알기 위한 그래프를 그려주는 함수
############################3

import matplotlib.pyplot as plt
def plot_acc(history, title=None):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'], loc=0)


def plot_loss(history, title=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'], loc=0)

##########################################3
# 만들어진 모델
##############################################3

from keras import layers, models
class NN(models.Sequential):
    def __init__(self, data_dim, Nout):
        super().__init__()
        self.add(layers.Conv2D(8,
                               kernel_size=(3,3),
                               activation='relu',
                               padding='same',
                               input_shape=(data_dim[1],data_dim[2],data_dim[3])))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())
        self.add(layers.Conv2D(16,
                               kernel_size=(3,3),
                               activation='relu',
                               padding='same'))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())
        self.add(layers.Conv2D(32,
                               kernel_size=(3,3),
                               activation='relu',
                               padding='same'))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())
        self.add(layers.Flatten())
        self.add(layers.Dense(64,
                               activation='relu'))
        self.add(layers.Dropout(0.2))
        self.add(layers.BatchNormalization())
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

####################################
# 메인 함수
######################################
def main():
    X_train, Y_train= get_data_set_of_XY() # training데이터를 가져옴.
    number_of_class = Y_train.shape[1]    # 장르갯수를 가져옴.
    Nout = number_of_class
    model = NN(X_train.shape, Nout)       # 모델을 가져옴.
    model.summary()                       # 전체모델의 모양을 확인.
    history=model.fit(X_train, Y_train, epochs=20, batch_size=20, validation_split=0.2) # 모델을 피팅. val_set은 training에서 20%만큼으로 사용.
    model.save_weights("model.h5") # predict을 위한 weight값을 저장.
    print("Saved model to disk")
    plot_loss(history)              # loss값을 그래프화
    plt.savefig('train_loss.png')  # 그래프 저장.
    plt.clf()
    plot_acc(history)               # acc값을 그래프화
    plt.savefig('train_acc.png')    # 그래프 저장.
    print("Saved graph")
# Run code
if __name__ == '__main__':
    main()
