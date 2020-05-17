import pandas as pd
from cv2 import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

def load_midpoints(filename, reverse=False):
    "Load midpoints"
    data = pd.read_csv(filename).values

    midpoints = []

    cap = cv2.VideoCapture("/home/hengji/Documents/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi")
    ret, frame = cap.read()
    ny, nx, _ = frame.shape

    for row in data:

        midpoint = []

        for p in row:

            if type(p) == str:

                pair = p.strip('(|)').split(', ')
                if not reverse:
                    pt = (float(pair[0]), float(pair[1]))
                else:
                    pt = (float(pair[1]), ny - float(pair[0]))
                midpoint.append(pt)

        midpoints.append(midpoint)


    return midpoints

def play(file_video, file_midpts_sklt, file_midpts_bisc, file_midpts_bisc_corrected):
    "Play"

    midpts_sklt = load_midpoints(file_midpts_sklt, reverse=True)
    midpts_bisc = load_midpoints(file_midpts_bisc)
    midpts_bisc_crr = load_midpoints(file_midpts_bisc_corrected)

    cap = cv2.VideoCapture(file_video)
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ny, nx, _ = frame.shape

    DIR = './results/'+ "Control-EGCaMP_exp1_a1_30x10fps/" +'/frames/'
    nframes = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    pbar = tqdm(total=nframes)

    plt.figure(figsize=(10, 10))
    iframe = 0

    while ret:
        
        plt.clf()
        plt.imshow(frame)

        midpt_bisc = midpts_bisc[iframe]
        midpt_bisc_crr = midpts_bisc_crr[iframe]
        midpt_sklt = midpts_sklt[iframe]

        midpt_bisc_x = [pt[0] for pt in midpt_bisc]
        midpt_bisc_y = [pt[1] for pt in midpt_bisc]
        midpt_bisc_crr_x = [pt[0] for pt in midpt_bisc_crr]
        midpt_bisc_crr_y = [pt[1] for pt in midpt_bisc_crr]
        midpt_bisc_y = [pt[1] for pt in midpt_bisc]
        midpt_sklt_x = [pt[0] for pt in midpt_sklt]
        midpt_sklt_y = [pt[1] for pt in midpt_sklt]

        # for pt in midpt_bisc:
        #     plt.plot(pt[0], pt[1], 'r.')
        
        # for pt in midpt_sklt:
        #     plt.plot(pt[0], pt[1], 'b.')

        plt.plot(midpt_bisc_crr_x, midpt_bisc_crr_y, 'g', markersize=2, linewidth=2)
        plt.plot(midpt_bisc_x, midpt_bisc_y, 'r', markersize=2, linewidth=2)
        plt.plot(midpt_sklt_x, midpt_sklt_y, 'b.', markersize=2)

        plt.xlim(0, nx)
        plt.ylim(0, ny)
        plt.pause(0.001)

        plt.savefig('./results/'+ "Control-EGCaMP_exp1_a1_30x10fps" +'/overlay/frames/img' + str(iframe) + '.jpg') # , orientation='landscape')

        ret, frame = cap.read()
        iframe += 1

        pbar.update()

def save_video(outname):
    "Count the number of frames"

    # Directory name
    DIR = './results/'+ outname +'/overlay/frames/'

    # Number of frames
    nframes = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    # Frame size
    frame_size = cv2.imread(DIR + 'img0.jpg').shape
    nx, ny = frame_size[0], frame_size[1]

    # Convert the frames into video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('./results/' + outname + '/overlay/video/movie.avi', fourcc, 40, (ny, nx))

    for iframe in tqdm(range(nframes)):
        frame = cv2.imread(DIR + 'img' + str(iframe) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":

    play("/home/hengji/Documents/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi",
         "./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints/midpoints_skeleton.csv",
         "./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints/midpoints_bisection.csv",
         "./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints/midpoints_bisection_corrected.csv")
    save_video("Control-EGCaMP_exp1_a1_30x10fps")