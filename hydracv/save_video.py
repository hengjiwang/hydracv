import sys, os, multiprocessing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '.')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
import glob

# Save video
FILENAME = "238705_2"
FPS = 10
NUMX = 600
NUMY = 650

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('/home/hengji/Data/hydracv/hydracv/data/frames/'+ FILENAME + '.avi', fourcc, FPS, (NUMX, NUMY))

# for iframe in tqdm(range(251)):
#     frame = cv2.imread('/home/hengji/Data/hydracv/hydracv/data/frames/' + FILENAME + '/img' + str(iframe) + '.png')
#     videoWriter.write(frame)
for img in glob.glob("/home/hengji/Data/hydracv/hydracv/data/frames/" + FILENAME + "/*.png"):
    frame = cv2.imread(img)
    videoWriter.write(frame[500:1150, 500:1100])
videoWriter.release()
cv2.destroyAllWindows()