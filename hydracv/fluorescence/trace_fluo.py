import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import sys, os
import pandas as pd

def trace(video, ROI=[], display=True):
    '''
    :inputs:
        video: original video
        ROI: area of interest in video (exluding any external objects)
        display: boolean specifying whether you want to visualize results
  
    :return: an array of the integration of fluorescence for each frame of the video
    :rtype: list
    '''
    
    intensities_ = []
    cap = cv2.VideoCapture(video)
    iframe = 0
    firstf = True

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
   
        if firstf:
            # Create a mask that keeps only pixels w/in ROI
            ROI_mask = np.ones((frame.shape[0], frame.shape[1]))
            poly = np.array(ROI, dtype=np.int32)
            cv2.fillConvexPoly(ROI_mask,poly,0)
            firstf = False

        if ret:         
            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = np.ma.masked_array(frame, mask=ROI_mask)
            intensity = np.sum(frame)

            intensities_.append(intensity)

            # Display the resulting frame
            if display:
                plt.imshow(frame)
                plt.show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        iframe += 1

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return intensities_

def plot(fluorescence_, fps=20, save=True, filename=None):
    fig = plt.figure()
    x = np.linspace(0, len(fluorescence_)/fps, len(fluorescence_))
    plt.plot(x, fluorescence_, 'r-')
    plt.xlabel('time/s')
    plt.ylabel('fluorescence')
    plt.show()
    if save:
        try:
            fig.savefig('figures/' + filename + '.png')
        except FileNotFoundError:
            os.makedirs('figures/')
            fig.savefig('figures/' + filename + '.png')

        try:
            df = pd.DataFrame(fluorescence_)
            df.to_csv('data/total_fluo_' + filename + '.csv', index=False)
        except FileNotFoundError:
            os.makedirs('data/')
            df = pd.DataFrame(fluorescence_)
            df.to_csv('data/total_fluo_' + filename + '.csv', index=False)
    return


if __name__ == "__main__":
    videoname = sys.argv[1] # 'videos/NGCaMP6_50X_20fps_M1-004.avi'
    intensities = trace(videoname)
    filename = videoname.split('/')[-1].strip('.avi')
    plot(intensities, int(sys.argv[2]), True, filename)
