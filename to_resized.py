
'''
    ■ 목적
        - origin 하위 디렉토리에 각 사진들의 사이즈를 10%로 작게한다.
        - reseized 디렉토리의 같은 라벨의 서브디렉토리에 저장한다.
    ■ 주의사항 
        - 디렉토리 이름에 한글이 사용되면 파이썬이 라이브러리가 제대로 작동하지 않음.
        - 인공지능 엔진
            - 초 5학년이상의 나이라면 사용할 수 있는 수준
            - https://teachablemachine.withgoogle.com/  
'''

import os
import cv2

source_d = '.\\origin'
target_d = '.\\resized'
# labels = ['1x2', '1x4', '2x4']
# labels = ['2x4-angle', '3-beam', '3x5-angle', '5-beam', '7-beam', '9-beam', '11-beam', '13-beam', '15-beam', 't-beam']
labels = ['15-beam']
source_files = []

# 원본파일 리스트 만들기
for label in labels:
    t_dir = os.path.join(source_d, label)
    for (path, dirs, files) in os.walk(t_dir):
        for file in files:
            source_files.append([label, os.path.normpath( os.path.join(os.getcwd(), path, file))])


# 원본파일 읽어와서 리사이징 후 저장하기
total = len(source_files)
count = 0
for src_item in source_files:
    label = src_item[0]
    src_filepath = src_item[1]
    filename = os.path.split(src_filepath)[1]
    # print("label:{}, image:{}".format(label, filename))

    # print(src_filepath)
    img = cv2.imread(src_filepath, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1)
    dst_file = os.path.normpath( os.path.join(os.getcwd(), target_d, label, filename) )
    cv2.imwrite(dst_file, resized_img)

    count += 1
    print("{} / {} converted !".format(count, total))
