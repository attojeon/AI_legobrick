'''
image_to_numpy
'''
import cv2
import os 
from PIL import Image
import glob, sys, numpy as np
from sklearn.model_selection import train_test_split

imageArray = []

image_w = 302
image_h = 403

X = []
Y = []

sub_folder = './binalized'
categories = os.listdir(sub_folder)
print(categories)
for idx, cat in enumerate(categories):
    label = [ 0 for i in range(len(categories))]
    label[idx] = 1

    for f in glob.glob(os.path.join(sub_folder, cat) + "\\*.jpg"):
        img = Image.open(f)
        data = np.asarray(img)
        X.append(data)
        Y.append(label)
X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./binary_image_data.npy", xy)
print(X)