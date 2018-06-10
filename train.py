
from keras.utils import np_utils
import music21
import os
import pickle
import numpy as np
from sklearn.preprocessing import normalize
from keras.preprocessing.sequence import pad_sequences

def create_mel_data_each_file(midi_obj):
    c = midi_obj.flat.getElementsByClass(music21.instrument.Instrument)
    mel_data = []

    for i, m in enumerate(midi_obj):
        inst = None
        mel_data.append(dict())
        for n in m:
            if (n in c):
                inst = n.midiProgram

            if (type(n) == music21.stream.Voice):
                for x in n:
                    if (x in c):
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
            if (type(n) == music21.chord.Chord):
                if n.offset not in mel_data[i]:
                    mel_data[i][n.offset] = dict()
                mel_data[i][n.offset]['offset'] = n.offset
                for p in n.pitches:
                    mel_data[i][n.offset]['note'] = p.midi
                mel_data[i][n.offset]['instrument'] = inst
            if (type(n) == music21.note.Note):
                if n.offset not in mel_data[i]:
                    mel_data[i][n.offset] = dict()
                mel_data[i][n.offset]['offset'] = n.offset
                prev_p = 0
                for p in n.pitches:
                    if prev_p < p.midi:
                        mel_data[i][n.offset]['note'] = p.midi
                    prev_p = p.midi
                mel_data[i][n.offset]['instrument'] = inst
    return mel_data

def get_midi_set(data_path,path_name):
    file_list = []
    for file_name in os.listdir(os.path.expanduser(data_path)):
        if file_name.endswith('.mid') or file_name.endswith('.midi'):
            file_list.append(data_path + file_name)
    mel_arr_list = []
    for file_name in file_list:
        print(file_name + 'get data')
        midi_obj = music21.converter.parse(file_name)
        mel_data = create_mel_data_each_file(midi_obj)
        print('finished..')

        mel_arr = []
        for i,mel_data_i in enumerate(mel_data):
            for key, value in sorted(mel_data_i.items()):
                mel_arr.append(mel_data_i[key])
        mel_arr_list.append(mel_arr)
    print('data set saving..')
    preprocessed_dir = "./preprocessed_data/"
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    with open(preprocessed_dir + path_name+"_mel_arr_list.p", "wb") as fp:
        pickle.dump(mel_arr_list, fp)
    print('save finished..')
    return mel_arr_list
def create_curve_seq(mel_arr):
    curve_seq = []
    if(len(mel_arr)>1):
        for idx in range(1, len(mel_arr)):
            curr_p_diff = mel_arr[idx]['note'] - mel_arr[idx-1]['note']
            curr_t_diff = mel_arr[idx]['offset'] - mel_arr[idx-1]['offset']
            if mel_arr[idx]['instrument']==None:
                x=0
            else:
                x=mel_arr[idx]['instrument']+1
            curve_seq.append([curr_p_diff, curr_t_diff,x])
    return curve_seq
def get_data_set_of_XY():
    org_path=os.getcwd()+'/mu/'
    preprocessed_dir = "/preprocessed_data/"
    paths=os.listdir(org_path)
    class_num=len(paths)
    for i,_ in enumerate(paths):
        paths[i]=str(org_path)+str(paths[i])+'/'
    Y_set=[]
    arr=[]
    name=os.listdir(org_path)
    if not(os.path.isdir(os.getcwd()+preprocessed_dir)):
        print('getting midi file set..')
        for i,p in enumerate(paths):
            n=name[i]
            arr.append(get_midi_set(p,n))
            Y_set.append(np_utils.to_categorical(i,class_num) * len(arr[i]))

    else:
        print('getting midi file set from dir(preprocessed_data)')
        for i,file_name in enumerate(os.listdir(os.getcwd()+preprocessed_dir)):
            if file_name.endswith('.p') or file_name.endswith('.p'):
                with open(os.getcwd() + preprocessed_dir + file_name, "rb") as fp:
                    arr.append(pickle.load(fp))
                    for _ in arr[i]:
                        Y_set.append(np_utils.to_categorical(i,class_num))
    curve_seq_list = []
    for ar in arr:
        for mel_arr in ar:
            curve_seq_list.append(create_curve_seq(mel_arr))
    print('preprocessing..')
    arr=curve_seq_list

    arr=np.array(arr)
    arr = pad_sequences(arr, padding='post',maxlen=1000,dtype=np.float)
    arr= np.reshape(arr,(arr.shape[0],-1,3))
    for i in range(len(arr)):
        arr[i] = normalize(arr[i], axis=0, norm='max')


    Y_set=np.array(Y_set)
    Y_set=np.reshape(Y_set,(-1,5))
    print('prepprocessing finished..')
    return arr,Y_set

# =============================================

# 분류 DNN 모델 구현 ########################
from keras import layers, models


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





# =============================================
# 분류 DNN 학습 및 테스팅 ####################
def main():
    Filter_num = 64
    number_of_class = 5
    Nout = number_of_class

    X_train, Y_train= get_data_set_of_XY()
    model = RNN(X_train.shape[1], Filter_num, Nout)
    model.fit(X_train, Y_train, epochs=50, batch_size=1, validation_split=0.2)
    model.save('model_1.h5')
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

# Run code
if __name__ == '__main__':
    main()
