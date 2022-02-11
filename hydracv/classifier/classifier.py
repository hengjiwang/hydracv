import sys, os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '../..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from cv2 import cv2
from tqdm import tqdm
from scipy.signal import savgol_filter
from midline.spline_fit import fit
import hydracv.utils.utils as utils

class Classifier:
    """A classifier for hydra behaviors"""
    def __init__(self):
        self.midpoints = defaultdict(lambda:[])
        self.lengths = []
        self.nframes = None
        self.npoints = None
        self.videopath = None
        self.behaviors = []
        self.cb_bounds = []
        self.rp_bounds = []
        self.slopes = []
        self.fps = None
        self.curv = []

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

        # Set curvature
        if len(self.curv) == 0:
            print('Calculating curvature...')
            _, self.curv = fit(filename, display=False)
            print('Curvature set.')
            self.curv = savgol_filter(self.curv, winlen_len, 3)

    
    def set_videopath(self, videopath):
        "Set the videopath"
        self.videopath = videopath

        # Calculate FPS
        video = cv2.VideoCapture(videopath)

        (major_ver, _, _) = (cv2.__version__).split('.')

        if int(major_ver) < 3:
            self.fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            self.fps = video.get(cv2.CAP_PROP_FPS)

        video.release()

    def _classify_cb_el(self, winlen_slp, lo_slp_thres, hi_slp_thres, lo_len_thres, hi_len_thres, cb_no_elong, elong_no_cb):
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

            # If don't want elongation during CB
            if cb_no_elong and self.lengths[i] < lo_len_thres:
                if slope > 0.001:
                    self.behaviors.append(['Elongation'])
                else:
                    self.behaviors.append(['Contraction'])
                continue

            # If don't want contraction during elongated
            if elong_no_cb and self.lengths[i] > hi_len_thres:
                if slope < -0.002:
                    self.behaviors.append(['Contraction'])
                else:
                    self.behaviors.append(['Elongation'])
                continue

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

    def _classify_bend(self, curv_thres):
        "Classify whether hydra is bending or not"
        for i in range(0, self.nframes):
            if self.curv[i] > curv_thres:
                self.behaviors[i].append('Bending')
            else:
                self.behaviors[i].append('')

        


    def classify_behavior(self, winlen_slp=31, lo_slp_thres=0, hi_slp_thres=0,
                 lo_len_thres=0.2, hi_len_thres=0.7, curv_thres=0.005,
                 cb_no_elong=True, elong_no_cb=True):
        "Run the classification"

        # Classify contraction and elongation
        self._classify_cb_el(winlen_slp, lo_slp_thres, hi_slp_thres, lo_len_thres, hi_len_thres, cb_no_elong, elong_no_cb)

        # Classify bending or not
        self._classify_bend(curv_thres)

    def classify_neural_activity(self, cb_combine=30, min_cb_size=10):
        """
        'cb_combine': combine two contraction periods into one CB event if they are fewer than 'cb_combine' frames apart
        'min_cb_size': cb events should not be shorter than this value
        """
        if self.behaviors == []:
            raise Exception("Must run 'classify_behavior' prior to calling this method.")
        
        contraction_events = utils.contraction_events(self.behaviors)
    
        start_prev = contraction_events[0]
        end_prev = contraction_events[1]
        prev = contraction_events[0]
        i = 2
        combined_contractions = []
        while i < len(contraction_events)-1:
            start_curr = contraction_events[i]
            end_curr = contraction_events[i+1]
            if start_curr - end_prev < cb_combine:
                pass
                # do not update previous start, since it remains the same
            else:
                combined_contractions.extend([start_prev,end_prev])
                start_prev = start_curr
            end_prev = end_curr
            i += 2
        combined_contractions.extend([start_prev, end_prev])

        # take another pass through events and this time elimante ones that are two brief
        it = iter(combined_contractions)
        for x in it:
            start = x
            end = next(it)
            if end - start > min_cb_size:
                self.cb_bounds.extend([start,end])

    def play(self, save=True, outname="Control-EGCaMP_exp1_a1_30x10fps", fps=200):
        "Play the results"

        cap = cv2.VideoCapture(self.videopath)
        ret, frame = cap.read()
        ny, nx, _ = frame.shape

        DPI = 100
        plt.figure(figsize=(nx/DPI, ny/DPI), dpi=DPI)

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

            # Adjust margin
            plt.axis('off')

            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.xlim(0, nx)
            plt.ylim(0, ny)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)

            # Save frame
            if save:
                # print('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg')
                plt.savefig('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg', dpi=DPI) # , orientation='landscape')
            plt.pause(0.0001)
            ret, frame = cap.read()
            iframe += 1
            pbar.update()
        
        pbar.close()
        cap.release()
        cv2.destroyAllWindows()

        # Save video
        if save:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            videoWriter = cv2.VideoWriter('./results/' + outname + '/video/video.avi', fourcc, fps, (nx, ny))

            for iframe in tqdm(range(self.nframes)):
                frame = cv2.imread('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg')
                videoWriter.write(frame)
            videoWriter.release()
            cv2.destroyAllWindows()
    
    def plot_behavior_periods(self, savepath=""):
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
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def plot_cb_periods(self,ax):
        it = iter(self.cb_bounds)
        for x in it:
            ax.axvspan(x, next(it), color='orange', alpha=0.3)

    def plot_slopes_and_lengths(self):
        "Plot slopes and lengths"

        x = np.arange(0, self.nframes/self.fps, 1/self.fps)

        fig = plt.figure(figsize=(20, 5))
        ax = fig.add_subplot(1, 1, 1)
        ln1 = ax.plot(x, self.lengths, 'b', label='length')
        ax.set_ylabel('length (a.u.)')
        ax2 = ax.twinx()
        ln2 = ax2.plot(x, self.slopes, 'r', label='slope')
        ax2.hlines(self.hi_slp_thres, 0, self.nframes/self.fps, linestyles='--')
        ax2.hlines(self.lo_slp_thres, 0, self.nframes/self.fps, linestyles='--')
        ax2.set_ylabel('slope')
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs)
        plt.show()




if __name__ == "__main__":
    classifier = Classifier()
    classifier.set_midpoints("D:/hydracv/hydracv/midline/results/Control-EGCaMP_exp1_a1_30x10fps/midpoints/midpoints_bisection_corrected.csv")
    classifier.set_videopath("D:/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi")
    classifier.classify(winlen_slp=51,
                        lo_slp_thres=-0.0005,
                        hi_slp_thres=0.0005,
                        lo_len_thres=0.2,
                        hi_len_thres=0.5,
                        curv_thres=0.004,
                        cb_no_elong=True,
                        elong_no_cb=True)
    classifier.play()
        


        



