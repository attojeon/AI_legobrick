'''
    matplotlib 3.3 최신버전은 에러가 발생함.
    (venv)$ pip install matplotlib==1.5.3 으로 하위 버전 설치하여 에러 회피함.
    updated by ato@learnsteam 2020.10.14
'''
import cv2
import os 
import random
from PIL import Image
import glob, sys, numpy as np

import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Convolution2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras

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



##########################################
# 1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory = './alphabet/Training',
    target_size = (32,32),
    batch_size = 32,
    class_mode = 'categorical'
)

test_generator = test_datagen.flow_from_directory(
    directory = './alphabet/Testing',
    target_size = (32,32),
    batch_size = 32,
    class_mode = 'categorical'
)

def modeljob_start():
    # 2
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (32,32,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))


    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 26, activation = 'softmax'))


    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model.summary()

    # 3
    model.fit_generator(train_generator,
                            steps_per_epoch = 16,
                            epochs = 50,
                            validation_data = test_generator,
                            validation_steps = 16)

    model.save("./my_h5_model.h5")

# 4
def get_result(result):
    if result[0][0] == 1:
        return('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')


def model_test():
    model = keras.models.load_model("my_h5_model.h5")

    alphabet_list = glob.glob('./alphabet/Testing/**/*.png', recursive=True)

    while True:
        if input(">>> Q:끝내기, Enter:인공지능 테스트하기\n   계속하려면 엔터를 누르시오.").upper() == 'Q':
            break

        # ./alphabet/Testing\i\28.png
        filename = random.choice(alphabet_list)
        # img_label = filename.split('\\')[-2:-1][0]
        # filename = r'.\alphabet\Testing\e\25.png'

        test_image = image.load_img(filename, target_size = (32,32))
        ans_image = test_image.copy()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        result = get_result(result)
        print ('>>> 인공지능 분석결과 알파벳 : {}'.format(result))

        plt.imshow(ans_image)
        plt.show()
#################
# model loading and testing 
model_test()