import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import sys, csv, os
import cv2
from tqdm import tqdm
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '../..')))

import hydracv.utils.utils as utils
from hydracv.midline.find_midline_midpoints import extract_midline


def sep_fluo(file_icy, file_dlc, videopath, display=False, start=0, end=-1):
    "Separate fluorescence to four quarters"

    trace1, trace2, trace3, trace4 = [], [], [], []
    fluos = []

    # Load contours and markers
    contours, markers = utils.load_contours_markers(file_icy, file_dlc, scale=(1,1))
    markers = markers.values

    # Loop over frames
    num_frames = len(contours)
    end = num_frames if end == -1 else end
    cap = cv2.VideoCapture(videopath)
    if display:
        plt.figure(figsize=(10,10))
    for iframe in tqdm(range(start, end)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, iframe-1)
        ret, frame = cap.read()

        marker = markers[iframe]
        contour = contours[iframe]

        # Divide contour and extract midpoints
        midpoints, contour_half_1, contour_half_2 = extract_midline(contour, marker, nseg=40)
        midpoints = [(midpoints[i], midpoints[i+1]) for i in range(0, len(midpoints), 2)]
        hyp = midpoints.pop()
        midpoints = np.array(midpoints)
        contour = np.array(contour)

        # Extract the polypoints
        midcontour1, index1 = utils.midpoint_of(contour_half_1)
        midcontour2, index2 = utils.midpoint_of(contour_half_2)
        midmid = ((midcontour1[0] + midcontour2[0])/2, (midcontour1[1] + midcontour2[1])/2)
        dist_ref = utils.length_segment([hyp, midmid])
        for index in range(len(midpoints)):
            if utils.length_segment([hyp, midpoints[index]]) <= dist_ref:
                break

        poly1_ = np.array(list(midpoints[:index]) + [midmid] + [midcontour1] + list(contour_half_1[:index1+1])[::-1])
        poly2_ = np.array(list(midpoints[:index]) + [midmid] + [midcontour2] + list(contour_half_2[:index2+1])[::-1])
        poly3_ = np.array([midmid] + list(midpoints[index:]) + list(contour_half_1[index1+1:][::-1]) + [midcontour1])
        poly4_ = np.array([midmid] + list(midpoints[index:]) + list(contour_half_2[index2+1:][::-1]) + [midcontour2])

        poly1 = np.array([poly1_], dtype = np.int32)
        poly2 = np.array([poly2_], dtype = np.int32)
        poly3 = np.array([poly3_], dtype = np.int32)
        poly4 = np.array([poly4_], dtype = np.int32)

        # Handle frame and calculate fluorescence
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fluo1, fluo2, fluo3, fluo4 = 0, 0, 0, 0
        cframe1, cframe2, cframe3, cframe4 = np.zeros_like(frame), np.zeros_like(frame), np.zeros_like(frame), np.zeros_like(frame)
        cv2.fillPoly(cframe1, poly1, 1)
        cv2.fillPoly(cframe2, poly2, 1)
        cv2.fillPoly(cframe3, poly3, 1)
        cv2.fillPoly(cframe4, poly4, 1)
        cframe1, cframe2, cframe3, cframe4 = cframe1 * frame, cframe2 * frame, cframe3 * frame, cframe4 * frame
        fluo1, fluo2, fluo3, fluo4 = np.sum(cframe1), np.sum(cframe2), np.sum(cframe3), np.sum(cframe4)
        trace1.append(fluo1)
        trace2.append(fluo2)
        trace3.append(fluo3)
        trace4.append(fluo4)
        fluos.append(np.sum(frame))

        # Plot
        if display:
            plt.clf()

            plt.imshow(frame, cmap='gray')

            # plt.plot(contour_half_1[:,0], contour_half_1[:,1], 'g.')
            # plt.plot(contour_half_2[:,0], contour_half_2[:,1], 'g.')
            # plt.plot(midpoints[::2], midpoints[1::2], 'r.')
            # plt.plot(midpoints[:,0], midpoints[:,1], 'r.')
            # plt.plot([midcontour1[0], midcontour2[0]], [midcontour1[1], midcontour2[1]], 'purple', linewidth=3)
            # plt.plot(midcontour2[0], midcontour2[1], 'k.')
            # plt.plot(midmid[0], midmid[1], 'k.')
            # plt.plot(midpoints[index][0], midpoints[index][1], 'g.')

            plt.fill(poly1_[:,0], poly1_[:,1], alpha = 0.5, color='g')
            plt.fill(poly2_[:,0], poly2_[:,1], alpha = 0.5, color='b')
            plt.fill(poly3_[:,0], poly3_[:,1], alpha = 0.5, color='r')
            plt.fill(poly4_[:,0], poly4_[:,1], alpha = 0.5, color='y')

            plt.xlim(0, len(frame[0]))
            plt.ylim(0, len(frame))
            plt.pause(0.0001)

    plt.show()
    # Plot intensities
    fig = plt.figure()
    plt.plot(trace1)
    plt.plot(trace2)
    plt.plot(trace3)
    plt.plot(trace4)
    plt.plot([trace1[i] + trace2[i] + trace3[i] + trace4[i] for i in range(len(trace1))], 'k', linewidth=4)
    pickle.dump(fig, open("../data/figures/sep_fluo.fig.pickle", 'wb'))
    plt.show()
    plt.figure()
    plt.plot(fluos, 'g', linewidth=4)
    plt.show()


if __name__ == '__main__':
    sep_fluo("../data/contour/Pre_Bisect_40x_4fps_ex3.xml",
             "../data/marker/Pre_Bisect_40x_4fps_ex3DeepCut_resnet50_Hydra2Nov15shuffle1_197000.csv",
             "../data/videos/NGCaMP/Pre_Bisect_40x_4fps_ex3.avi",
             display=True,
             start=300,
             end=-1)
