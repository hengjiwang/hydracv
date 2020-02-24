import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from vlib import save_pattern, save_curve, plot_frame, plot_frames
import pandas as pd
import cv2, tqdm, sys, os

def save_video(target):
    
    file_to_save = './save/animations/'+target+'/movie/movie.avi'
    frames_loc = './save/animations/'+target + '/frames/'
    
    frames = os.listdir(frames_loc)
    total_num = len(frames)

    fps = 200

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(file_to_save,fourcc,fps,(936,762))

    for i in tqdm.tqdm(range(total_num)):
        frame = cv2.imread(frames_loc+'img'+str(i)+'.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    df = pd.read_json('config.json')
    target = df.Name.values[0]
    save_video(target)
