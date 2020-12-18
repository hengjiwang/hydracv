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
FILENAME = "spikes_658838"
FPS = 40
NUMX = 1280
NUMY = 660

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('E:/hydra/hydra/visualizer/data/'+ FILENAME + '.avi', fourcc, FPS, (NUMX, NUMY))

for iframe in tqdm(range(251)):
    frame = cv2.imread('E:/hydra/hydra/visualizer/data/spikes_658838/img' + str(iframe) + '.jpg')
    videoWriter.write(frame)
# for img in glob.glob("./data/frames/spikes_658838/*.jpg"):
#     frame = cv2.imread(img)
#     videoWriter.write(frame)
videoWriter.release()
cv2.destroyAllWindows()