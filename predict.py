import train
import os
import numpy as np
# 데이터 가져오기########################
genre=os.listdir(os.getcwd()+'/mu/')
predic_dir='/predict/'
arr1,_=train.get_data_set_of_XY(d=predic_dir,predict=True)


# 2. 모델과 웨이트 불러오기
model =train.NN(arr1.shape,  len(genre))
model.load_weights('model.h5')
model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
x=os.listdir(os.getcwd()+predic_dir)
for i,arr in enumerate(arr1):
# 3. 프레딕트
    arr=np.reshape(arr,(1,arr.shape[0],arr.shape[1],arr.shape[2]))
    score = model.predict(arr,verbose=0)
    print('\n')
    print(x[i])
    print(score)
    for i,ge in enumerate(genre):
        print(ge+' 일 확률 : ',score[0][i])
