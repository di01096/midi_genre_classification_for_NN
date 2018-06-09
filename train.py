
from keras.utils import np_utils
import music21
import os
import pickle
import numpy as np

def create_mel_data_each_file(midi_obj):
    mel_data = dict()

    #print(len(midi_obj.flat.getElementsByClass(music21.chord.Chord)))
    for n in midi_obj.flat.getElementsByClass(music21.chord.Chord):
        if n.offset not in mel_data:
            mel_data[n.offset] = dict()
        mel_data[n.offset]['offset'] = n.offset
        for p in n.pitches:
            mel_data[n.offset]['note'] = p.midi

    #print(len(midi_obj.flat.getElementsByClass(music21.note.Note)))
    for n in midi_obj.flat.getElementsByClass(music21.note.Note):
        if n.offset not in mel_data:
            mel_data[n.offset] = dict()
        mel_data[n.offset]['offset'] = n.offset
        prev_p = 0
        for p in n.pitches:
            if prev_p < p.midi:
                mel_data[n.offset]['note'] = p.midi
            prev_p = p.midi
    print(len(mel_data))

    return mel_data

def get_midi_set(data_path,path_name):
    file_list = []
    for file_name in os.listdir(os.path.expanduser(data_path)):
        if file_name.endswith('.mid') or file_name.endswith('.midi'):
            file_list.append(data_path + file_name)
    print(file_list)
    mel_arr_list = []
    for file_name in file_list:
        print(file_name)
        midi_obj = music21.converter.parse(file_name)
        mel_data = create_mel_data_each_file(midi_obj)

        mel_arr = []
        for key, value in sorted(mel_data.items()):
            mel_arr.append(mel_data[key])

        mel_arr_list.append(mel_arr)
    preprocessed_dir = "./preprocessed_data/"
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    with open(preprocessed_dir + path_name+"_mel_arr_list.p", "wb") as fp:
        pickle.dump(mel_arr_list, fp)
    return mel_arr_list

def create_curve_seq(mel_arr):
    curve_seq = []
    if(len(mel_arr)>1):
        for idx in range(1, len(mel_arr)):
            curr_p_diff = mel_arr[idx]['note'] - mel_arr[idx-1]['note']
            curr_t_diff = mel_arr[idx]['offset'] - mel_arr[idx-1]['offset']
            curve_seq.append([curr_p_diff, curr_t_diff])
    return curve_seq

from keras.preprocessing.sequence import pad_sequences
def get_data_set_of_XY():
    org_path=os.getcwd()+'/mu/'
    preprocessed_dir = "/preprocessed_data/"
    paths=os.listdir(org_path)
    class_num=len(paths)
    for i,_ in enumerate(paths):
        paths[i]=str(org_path)+str(paths[i])+'/'
    print(paths)
    Y_set=[]
    arr=[]
    name=os.listdir(org_path)
    if not(os.path.isdir(os.getcwd()+preprocessed_dir)):
        for i,p in enumerate(paths):
            n=name[i]
            arr.append(get_midi_set(p,n))
            Y_set.append(np_utils.to_categorical(i,class_num))

    else:
        for i,file_name in enumerate(os.listdir(os.getcwd()+preprocessed_dir)):
            if file_name.endswith('.p') or file_name.endswith('.p'):
                with open(os.getcwd() + preprocessed_dir + file_name, "rb") as fp:
                    arr.append(pickle.load(fp))
                    for _ in arr[i]:
                        Y_set.append(np_utils.to_categorical(i,class_num))
    print(np.array(Y_set).shape)
    curve_seq_list = []
    for i,ar in enumerate(arr):
        for mel_arr in ar:
            curve_seq_list.append(create_curve_seq(mel_arr))

    arr=curve_seq_list
    print(np.array(arr).shape)
    for i in range(len(arr)):
        arr[i]=np.array(arr[i])
        arr[i]=np.reshape(arr[i],(-1,1,2))
        print(arr[i].shape)

    arr=np.array(arr)
    arr = pad_sequences(arr, padding='post')
    arr= np.reshape(arr,(arr.shape[0],-1,2))
    print(arr.shape)
    Y_set=np.array(Y_set)
    Y_set=np.reshape(Y_set,(-1,5))

    return arr,Y_set

# =============================================

# 분류 DNN 모델 구현 ########################
from keras import layers, models


class RNN(models.Sequential):
    def __init__(self, Time_step, Filter_num, Nout):
        super().__init__()

        self.add(layers.LSTM(Filter_num,input_shape=(Time_step,2),return_sequences=True))
        self.add(layers.BatchNormalization())
        self.add(layers.LSTM(Filter_num))
        self.add(layers.BatchNormalization())
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

        model_json = self.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)



# =============================================
# 분류 DNN 학습 및 테스팅 ####################
def main():
    Filter_num = 64
    number_of_class = 5
    Nout = number_of_class

    X_train, Y_train= get_data_set_of_XY()
    print(Y_train)
    model = RNN(X_train.shape[1], Filter_num, Nout)
    history=model.fit(X_train, Y_train, epochs=10, batch_size=1, validation_split=0.2)
    model.save_weights("model.h5")
    print("Saved model to disk")

# Run code
if __name__ == '__main__':
    main()