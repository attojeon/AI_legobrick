# AI_legobrick
레고브릭을 구분할 수 있는 인공지능 개발 프로젝트

## 실행 파일
1. ai_alphabet.py : 알파벳 문자를 구분할 수 있는 인공지능 표준 코드형, 아래의 유틸 코드 사용하지 않고 작동함.
2. to_resized.py, to_binalize.py, img_to_numpy.py : util 코드들

## 실행방법
- 가상환경(3.6) 실행
  - 윈도우 : > env/bin/activate.bat 
  - 리눅스 : $ source ./env/bin/activate 
  - pip install -r requirements ( mathplotlib==1.53 버전임.)
- ./alphabet/Testing/, ./alphabet/Training/ 두 폴더 아래에 알파벳 폴더 안에 이미지들이 위치하도록 세팅한다.
- ai_alphabet.py의 main() 함수를 적절히 수정하여 실행한다.
- my_h5_model.h5 모델을 사용할 수 있음.
