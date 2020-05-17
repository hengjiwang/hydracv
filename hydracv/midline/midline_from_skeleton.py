from cv2 import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

def extract_midline(filename):
    "Load the skeleton video"

    midpoints = []

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ny, nx, _ = frame.shape

    # plt.figure()
    iframe = 0

    DIR = './results/'+ "Control-EGCaMP_exp1_a1_30x10fps/" +'/frames/'
    nframes = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    pbar = tqdm(total=nframes)
    while ret:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        midpoint = []
        for i in range(ny):
            for j in range(nx):
                if frame[i, j] == 0:
                    midpoint.append((i, j))


        midpoints.append(midpoint)

        # plt.imshow(frame, cmap='binary')

        # plt.xlim(0, nx)
        # plt.ylim(0, ny)
        # plt.pause(0.001)
        ret, frame = cap.read()
        iframe += 1

        pbar.update()

    cv2.destroyAllWindows()
    pbar.close()
    return midpoints

if __name__ == "__main__":
    midpoints = extract_midline("/home/hengji/Documents/hydracv/hydracv/midline/results/Control-EGCaMP_exp1_a1_30x10fps/video/skeleton.avi")
    df = pd.DataFrame(midpoints)
    df.to_csv("./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints_skeleton.csv", index=False)