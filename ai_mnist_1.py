# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf 


# (참조) https://blog.naver.com/vft1500/221793591386
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



#1. 데이터셋 준비하기
#아마존에서 이미지 다운로드
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 하나의 데이터를 샘플이라고 하면
# 위의 변수들은 각각 60000x784, 60000x1, 10000x784, 10000x1의 크기로 각각이 1차원 배열로 되어 있다.
# 1) reshape을 해서 2차원 배열로 바꾸겠다.
# 2) 0~255의 정수값을 astype('float32')/255.0 으로 0~1사이의 값으로 바꾸겠다.(float32로 정규화)
#   사이즈는 더 커짐
#   딥러닝에 넣을 때, 정규화해서 넣어야 더 잘된다고 함
X_train = X_train.reshape(60000,784).astype('float32') / 255.0
X_test  = X_test.reshape(10000,784).astype('float32') / 255.0

# 라벨 값이 one-hot-encoding  으로 바뀜
# ex) 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     7 -> [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
#     입력 1바치트 -> 출력 10바이트
print(Y_train[0])
Y_train = np_utils.to_categorical(Y_train)
Y_test  = np_utils.to_categorical(Y_test)
print(Y_train[0])


#2. 모델 구성하기
model = Sequential()
model.add(Dense(units=64, input_dim=28*28, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
#소프트맥스 : 출력의 합이 1이다
'''
Categorical Cross Entropy
0.1    0
0.2    0
0.7    1 
'''

#3. 모델 엮기 : '네트워크가 학습할 준비가 되었습니다.'라는 의미
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# batch_size : 몇문항 풀고 업데이트 할 거냐?
# ex) 문제가 100개고, batch_size가 10이면 10번의 업데이트
# epoch : 전제 문제를 반복 횟수
# 300000(training 갯수-60000 x epoch-5) / 32번 네트워크 갱신한다.
hist = model.fit(X_train,Y_train, epochs=5, batch_size=32)

# 
# 데이터셋 : 훈련셋(모의고사 1~4회), 검증셋(5회), 시험셋(ex. 작년 시험문제) 
# 매 epoch마다 검증셋으로 검증한다.
