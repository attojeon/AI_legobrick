import glob
import random
from matplotlib import pyplot as plt

alphabet_list = glob.glob('./alphabet/Testing/**/*.png', recursive=True)
# ./alphabet/Testing\i\28.png
img_file = random.choice(alphabet_list)
img_label = img_file.split('\\')[-2:-1][0]
print(img_file)
print(img_label)