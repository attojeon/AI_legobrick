import cv2
import numpy as np
import os

# wxh: 302x403
SrcImageInfos = [
    ['1x2', r'D:/D_SRC/Projects/Git_Python_201910-/AI_legobrick/resized/1x2'],
    ['1x4', r'D:/D_SRC/Projects/Git_Python_201910-/AI_legobrick/resized/1x4'],
    ['2x4', r'D:/D_SRC/Projects/Git_Python_201910-/AI_legobrick/resized/2x4'] 
]
DstFolder = r'D:/D_SRC/Projects/Git_Python_201910-/AI_legobrick/binalized/'


for label, fullpath in SrcImageInfos:
    print(label)
    img_list = os.listdir(fullpath)
    for img_filename in img_list:
        img = cv2.imread(os.path.join(fullpath, img_filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(302, 403), interpolation=cv2.INTER_AREA)
        ret, dst = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
        dst_path = os.path.join(DstFolder, label)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        dst_file = os.path.join(dst_path, img_filename)
        cv2.imwrite(dst_file, img)



