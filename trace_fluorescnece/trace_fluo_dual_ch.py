import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os
import pandas as pd

def trace(video):
    # Trace the fluorescnece of John's GCaMP and RCaMP videos

    intensities_red = []
    intensities_green = []
    cap = cv2.VideoCapture(video)
    firstf = True
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if firstf:
                size = len(frame) * len(frame[0])
                firstf = False
                
            # Our operations on the frame come here

            frame_red = frame[:,:,2]
            frame_green = frame[:,:,1]
            intensity_red = np.sum(frame_red)
            intensity_green = np.sum(frame_green)
            # cv2.imshow("Red 2", frame_red)
            # cv2.imshow("Green 2", frame_green)

            intensities_red.append(intensity_red)
            intensities_green.append(intensity_green)


            # # Display the resulting frame
            # if display:
            #     cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return intensities_red, intensities_green


if __name__ == "__main__":
    # videoname = sys.argv[1] # 'videos/NGCaMP6_50X_20fps_M1-004.avi'

    image_formats = ['.mov']
    # for (root, dirs, files) in os.walk('/media/hengji/DATA/Data/Documents/szymanski_videos/'): 
    root = "" 
    for filename in ['/home/hengji/Downloads/john-long-cycles.mov']:
        if os.path.splitext(filename)[1] in image_formats:

            intensities_red, intensities_green = trace(os.path.join(root,filename))
            plt.figure()
            plt.plot(intensities_red, 'r')
            plt.plot(intensities_green, 'g')
            plt.xlabel('frames')
            plt.ylabel('Fluorescence[a.u.]')
            plt.savefig(os.path.join(root,os.path.splitext(filename)[0])+'.png')
            plt.show()

