import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import sys, os
import pandas as pd

def trace(video, contours=[], display=True):
    '''
    :inputs:
        video: original video
        contours: list of contours for each frame of video (output from find_midline_midpoints.load_contour)
        display: boolean specifying whether you want to visualize results
  
    :return: an array of the integration of fluorescence for each frame of the video
    :rtype: list
    '''
    
    intensities_ = []
    cap = cv2.VideoCapture(video)
    firstf = True
    iframe = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            if firstf:
                size = len(frame) * len(frame[0])
                firstf = False
                
            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if contours:
                # There may be fewer contours than total number of frames in video
                if iframe == len(contours):
                    break
                    
                curr_contour = contours[iframe]
                # Create a mask that keeps only pixels w/in hydra contour
                contour_mask = np.ones(frame.shape)
                poly = np.array(curr_contour, dtype=np.int32)
                cv2.fillConvexPoly(contour_mask,poly,0)
 
                frame = np.ma.masked_array(frame, mask=contour_mask)
    
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
