import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from cv2 import cv2
from tqdm import tqdm
from scipy.signal import savgol_filter

class Classifier:
    """A classifier for hydra behaviors"""
    def __init__(self):
        self.midpoints = defaultdict(lambda:[])
        self.lengths = []
        self.nframes = None
        self.npoints = None
        self.videopath = None
        self.behaviors = []
        self.slopes = []
        self.fps = None

    def set_midpoints(self, filename, winlen_len=31):
        "Add the coordinates of midpoints from filename"
        self.midpoints = defaultdict(lambda:[])
        self.lengths = []
        data = pd.read_csv(filename).values
        self.nframes = len(data)
        self.npoints = len(data[0]) // 2

        for i in range(self.nframes):
            length = 0
            for j in range(self.npoints):
                self.midpoints[j].append((data[i][2*j], data[i][2*j+1]))
                if j > 0:
                    length += np.sqrt((data[i][2*j] - data[i][2*j-2])**2 + (data[i][2*j+1] - data[i][2*j-1])**2)
            self.lengths.append(length)

        # Normalize lengths
        minlen = min(self.lengths)
        maxlen = max(self.lengths)

        for i in range(self.nframes):
            self.lengths[i] = (self.lengths[i] - minlen) / (maxlen - minlen)

        # Smooth the slopes
        self.lengths = savgol_filter(self.lengths, winlen_len, 3)

    
    def set_videopath(self, videopath):
        "Set the videopath"
        self.videopath = videopath

        # Calculate FPS
        video = cv2.VideoCapture(videopath)

        (major_ver, _, _) = (cv2.__version__).split('.')

        if int(major_ver) < 3 :
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else :
            self.fps = video.get(cv2.CAP_PROP_FPS)

        video.release()

    def _classify_cb_el(self, winlen_slp, lo_slp_thres, hi_slp_thres, lo_len_thres, hi_len_thres, cb_no_elong):
        "Classify hydra is in contraction/elongation for each frame"
        self.slopes = [0]
        self.behaviors = []

        # Calculate the slope
        for i in range(1, self.nframes):
            self.slopes.append(self.lengths[i] - self.lengths[i-1])

        # # Smooth the slopes
        self.slopes = savgol_filter(self.slopes, winlen_slp, 3)

        # Predict the behaviors
        for i in range(0, self.nframes):

            slope = self.slopes[i]

            if cb_no_elong and self.lengths[i] < lo_len_thres:
                if slope > 0.05:
                    self.behaviors.append(['Elongation'])
                else:
                    self.behaviors.append(['Contraction'])
                    

            if slope < lo_slp_thres:
                self.behaviors.append(['Contraction'])
            elif slope > hi_slp_thres:
                self.behaviors.append(['Elongation'])
            else:
                if self.lengths[i] < lo_len_thres:
                    self.behaviors.append(['Contraction'])
                elif self.lengths[i] > hi_len_thres:
                    self.behaviors.append(['Elongation'])
                else:
                    self.behaviors.append(['Rest'])

    def _classify_bend(self, theta_mfh_thres, theta_qfm_thres):
        "Classify whether hydra is bending or not"
        for i in range(0, self.nframes):
            
            # Key points
            head = self.midpoints[self.npoints-1][i]
            foot = self.midpoints[0][i]
            mid = self.midpoints[self.npoints//2][i]
            qua = self.midpoints[self.npoints//4][i]

            # Key distances
            d_mf = np.sqrt((mid[0] - foot[0])**2 + (mid[1] - foot[1])**2)
            d_fh = np.sqrt((head[0] - foot[0])**2 + (head[1] - foot[1])**2)
            d_mh = np.sqrt((head[0] - mid[0])**2 + (head[1] - mid[1])**2)

            # Angle mid-foot-head
            theta_mfh = np.arccos((d_mf**2 + d_fh**2 - d_mh**2) / (2 * d_mf * d_fh))
            if theta_mfh > theta_mfh_thres:
                self.behaviors[i].append('Bending')
            else:
                d_qf = np.sqrt((qua[0] - foot[0])**2 + (qua[1] - foot[1])**2)
                d_qm = np.sqrt((qua[0] - mid[0])**2 + (qua[1] - mid[1])**2)
                theta_qfm = np.arccos((d_qf**2 + d_mf**2 - d_qm**2) / (2 * d_qf * d_mf))
                if theta_qfm > theta_qfm_thres:
                    self.behaviors[i].append('Bending')
                else:
                    self.behaviors[i].append('')

        


    def classify(self, winlen_slp=31, lo_slp_thres=0, hi_slp_thres=0,
                 lo_len_thres=0.2, hi_len_thres=0.7, theta_mfh_thres=0.2,
                 theta_qfm_thres=0.2, cb_no_elong=True):
        "Run the classification"

        # Classify contraction and elongation
        self._classify_cb_el(winlen_slp, lo_slp_thres, hi_slp_thres, lo_len_thres, hi_len_thres, cb_no_elong)

        # Classify bending or not
        self._classify_bend(theta_mfh_thres, theta_qfm_thres)


    def play(self, save=True, outname="Control-EGCaMP_exp1_a1_30x10fps", fps=20):
        "Play the results"

        plt.figure(figsize=(10, 10))

        cap = cv2.VideoCapture(self.videopath)
        ret, frame = cap.read()
        ny, nx, _ = frame.shape

        pbar = tqdm(total=self.nframes)

        iframe = 0
        while(ret):
            plt.clf()
            # Show frame
            plt.imshow(frame)
            # Show tracked points
            for j in range(self.npoints):
                plt.plot(self.midpoints[j][iframe][0], self.midpoints[j][iframe][1], 'r.')

            # Display the behavior
            plt.text(int(5/6*ny), int(1/6*nx), self.behaviors[iframe][0], color='white', fontsize=20)
            plt.text(int(5/6*ny), int(2/9*nx), self.behaviors[iframe][1], color='white', fontsize=20)
            # Save frame
            if save:
                plt.savefig('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg') # , orientation='landscape')
            plt.pause(0.001)
            ret, frame = cap.read()
            iframe += 1
            pbar.update()
        
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        # Save video
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            videoWriter = cv2.VideoWriter('./results/' + outname + '/video/video.avi', fourcc, fps, (1000, 1000))

            for iframe in tqdm(range(self.nframes)):
                frame = cv2.imread('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg')
                videoWriter.write(frame)
            videoWriter.release()
            cv2.destroyAllWindows()
    
    def plot_behavior_periods(self):
        "Plot the behavior periods"
        plt.figure(figsize=(20, 5))
        for iframe in tqdm(range(self.nframes)):
            if self.behaviors[iframe][0] == 'Elongation':
                y = 1
                color = 'r'
            else:
                y = 3
                color = 'g'
            plt.hlines(y, iframe/self.fps, (iframe+1)/self.fps, colors=color, linewidth=90)
            if self.behaviors[iframe][1] == 'Bending':
                plt.hlines(2, iframe/self.fps, (iframe+1)/self.fps, colors='b', linewidth=90)
        plt.yticks([1, 2, 3], ['Elongation', 'Bending', 'Contraction'])
        plt.ylim(0.5, 3.5)
        plt.xlim(0, 1500)
        plt.xlabel('time (s)')
        # plt.savefig('./results/Periods_'+FILENAME.strip('.csv')+'.png')
        plt.show()

    def plot_slopes_and_lengths(self):
        "Plot slopes and lengths"
        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.lengths, 'b', label='length')
        ax.set_ylabel('length (a.u.)')
        ax2 = ax.twinx()
        ax2.plot(self.slopes, 'r', label='slope')
        ax2.hlines(0, 0, self.nframes)
        ax2.set_ylabel('slope')
        plt.legend(['slope', 'length'])
        plt.show()




if __name__ == "__main__":
    classifier = Classifier()
    classifier.set_midpoints("/home/hengji/Documents/hydracv/hydracv/classifier/data/Control-EGCaMP_exp1_a1_30x10fps_midpoints.csv")
    classifier.set_videopath("/home/hengji/Documents/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi")
    classifier.classify(winlen_slp=21,
                        lo_slp_thres=0,
                        hi_slp_thres=0,
                        lo_len_thres=0.06,
                        hi_len_thres=0.7,
                        theta_mfh_thres=0.2,
                        theta_qfm_thres=0.2,
                        cb_no_elong=False)
    classifier.play()
        


        



