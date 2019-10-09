import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys, os

def load_video(video, display=True):
    '''
    :return: an array of frames_ from the video
    :rtype: numpy.array
    '''

    frames_ = []
    cap = cv2.VideoCapture(video)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames_.append(frame)

            # Display the resulting frame
            if display:
                cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    frames_ = np.array(frames_)

    return frames_

def trace(frames_):
    '''
    :return: integrate the fluorescence intensities in the frames
    :rtype: numpy.array
    '''
    intensity = np.sum(np.sum(frames, axis = 1), axis = 1)
    return intensity

def plot(fluorescence_, fps=20, save=True, filename=None):
    fig = plt.figure()
    x = np.linspace(0, len(fluorescence_)/fps, len(fluorescence_))
    plt.plot(x, fluorescence, 'r-')
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
            np.savetxt('data/' + filename, fluorescence_)
        except FileNotFoundError:
            os.makedirs('data/')
            np.savetxt('data/' + filename, fluorescence_)


if __name__ == "__main__":
    videoname = sys.argv[1] # 'videos/NGCaMP6_50X_20fps_M1-004.avi'
    frames = load_video(videoname, True)
    fluorescence = trace(frames)
    filename = videoname.split('/')[-1].strip('.avi')
    plot(fluorescence, 20, True, filename)
    