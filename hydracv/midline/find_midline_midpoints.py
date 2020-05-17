import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import xml.etree.ElementTree as ET
from cv2 import cv2
from tqdm import tqdm

def load_contour(filename):
    "Load and reformat contour"

    file_format = filename.split('.')[-1]
    if file_format == 'pkl':
        with open(filename, 'rb') as pickle_file:
            contours = pickle.load(pickle_file)
        # Reformat data
        for iframe in range(len(contours)):
            pts = [(pt[0][0], pt[0][1]) for pt in contours[iframe][0]]
            contours[iframe] = pts

    elif file_format == 'xml':
        root = ET.parse(filename).getroot()
        rois = root.find('rois').findall('roi')
        contours = []
        for roi in rois[1:]:
            points = roi.find('points').findall('point')
            contour = []
            for point in points:
                pos_x = float(point.find('pos_x').text)
                pos_y = float(point.find('pos_y').text)
                contour.append((pos_x, pos_y))
            contours.append(contour)

    return contours

def intp_seq(seq, nintp):
    "Interpolate sequence"
    seq_new = []
    for j in range(1, len(seq)):
        x_prev = seq[j-1][0]
        y_prev = seq[j-1][1]
        x_next = seq[j][0]
        y_next = seq[j][1]
        xintp = np.linspace(x_prev, x_next, nintp, endpoint=False)
        yintp = np.linspace(y_prev, y_next, nintp, endpoint=False)
        for k in range(len(xintp)):
            seq_new.append((xintp[k], yintp[k]))
    
    seq_new.append(seq[-1])
    return seq_new


def load_marker(filename):
    "Load tracked points"
    df = pd.read_csv(filename)
    df.columns = ['scorer', 'hypostome_x', 'hypostome_y', 'hypostome_likelihood',
                  'armpit1_x', 'armpit1_y', 'armpit1_likelihood',
                  'armpit2_x', 'armpit2_y', 'armpit2_likelihood',
                  'peduncle_x', 'peduncle_y', 'peduncle_likelihood']
    df = df.drop(index=[0, 1]).drop(columns='scorer').reset_index(drop=True)
    df = df.astype(float)
    return df  

def locate_point(marker, contour):
    "Locate the corresponding index of the marker on contour"

    index = 0
    mindist = np.inf
    for j in range(len(contour)):
        dist = (marker[0] - contour[j][0])**2 + (marker[1] - contour[j][1])**2
        if dist < mindist:
            mindist = dist
            index = j

    return index

def length_segment(seg):
    "Returns length of segment seg"
    length = 0
    for j in range(len(seg)-1):
        p1, p2 = seg[j], seg[j+1]
        length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    return length

def find_midline(file_contour, file_marker, file_video, nseg=40):
    "Find midline"

    # Load files
    contours = load_contour(file_contour)
    markers = load_marker(file_marker).values

    midpoints_all = []

    # Align data
    # nframes = min(len(contours), len(markers))
    # contours = contours[:nframes]
    # markers = markers[:nframes].values

    # plt.figure()

    # Loop over frames
    # for iframe in range(nframes):

    # cap = cv2.VideoCapture(file_video)
    # ret, frame = cap.read()
    # ny, nx, _ = frame.shape

    iframe = 0
    # while(ret):

    for iframe in tqdm(range(len(contours))):

        # plt.clf()

        # plt.imshow(frame)

        # Extract contour and marker
        contour = contours[iframe]
        marker_mat = markers[iframe]

        # Reformat marker
        marker = defaultdict(tuple)
        marker['hypostome'] = (marker_mat[0], marker_mat[1])
        marker['armpit1'] = (marker_mat[3], marker_mat[4])
        marker['armpit2'] = (marker_mat[6], marker_mat[7])
        marker['peduncle'] = (marker_mat[9], marker_mat[10])

        # Locate the peduncle on contour
        ind_ped = locate_point(marker['peduncle'], contour)

        # Reindex the contour -- start from the peduncle
        contour = contour[ind_ped:] + contour[:ind_ped]

        # Locate other markers
        ind_ped = 0
        ind_arp1 = locate_point(marker['armpit1'], contour)
        ind_arp2 = locate_point(marker['armpit2'], contour)
        ind_hyp = locate_point(marker['hypostome'], contour)

        # Separate contour to two parts
        ind_arp1, ind_arp2 = min(ind_arp1, ind_arp2), max(ind_arp1, ind_arp2)
        contour_half_1 = contour[:ind_arp1]
        contour_half_2 = [contour[0]] + contour[ind_arp2:][::-1]

        # Interpolate contours
        # contour_half_1 = intp_seq(contour_half_1, 5)
        # contour_half_2 = intp_seq(contour_half_2, 5)

        # for j, pt in enumerate(contour_half_1):
        #     plt.text(pt[0], pt[1], str(j), color='g', fontsize=5)

        # for j, pt in enumerate(contour_half_2):
        #     plt.text(pt[0], pt[1], str(j), color='g', fontsize=5)

        # for pt in contour_half_1:
        #     plt.plot(pt[0], pt[1], 'g.', markersize=5)
        # for pt in contour_half_2:
        #     plt.plot(pt[0], pt[1], 'g.', markersize=5)

        # plt.plot(contour[ind_arp1][0], contour[ind_arp1][1], 'y.', markersize=20)
        # plt.plot(contour[ind_arp2][0], contour[ind_arp2][1], 'y.', markersize=20)
        # plt.plot(contour[ind_hyp][0], contour[ind_hyp][1], 'y.', markersize=20)
        # plt.plot(contour[ind_ped][0], contour[ind_ped][1], 'y.', markersize=20)

        # plt.plot(marker['hypostome'][0], marker['hypostome'][1], 'b.', markersize=20)
        # plt.plot(marker['armpit1'][0], marker['armpit1'][1], 'b.', markersize=20)
        # plt.plot(marker['armpit2'][0], marker['armpit2'][1], 'b.', markersize=20)
        # plt.plot(marker['peduncle'][0], marker['peduncle'][1], 'b.', markersize=20)


        # Find the midpoints
        midpoints = []
        len_contour_1 = length_segment(contour_half_1)
        len_contour_2 = length_segment(contour_half_2)
        ind_seg_pt1 = 0
        ind_seg_pt2 = 0
        cum_len_1 = 0
        cum_len_2 = 0
        for j in range(1, nseg):
            
            # Locate the segment points
            while cum_len_1 < j/nseg * len_contour_1:
                cum_len_1 += length_segment(contour_half_1[ind_seg_pt1:ind_seg_pt1+2])
                ind_seg_pt1 += 1

            while cum_len_2 < j/nseg * len_contour_2:
                cum_len_2 += length_segment(contour_half_2[ind_seg_pt2:ind_seg_pt2+2])
                ind_seg_pt2 += 1

            seg_pt_1 = contour_half_1[ind_seg_pt1]
            seg_pt_2 = contour_half_2[ind_seg_pt2]
            # plt.plot([seg_pt_1[0], seg_pt_2[0]], [seg_pt_1[1], seg_pt_2[1]], 'purple')
            midpoint = ((seg_pt_1[0] + seg_pt_2[0]) // 2, (seg_pt_1[1] + seg_pt_2[1]) // 2)
            midpoints.append(midpoint)
            # plt.plot(midpoint[0], midpoint[1], 'r.')
        
        # plt.xlim(0, 500)
        # plt.ylim(0, 500)
        # plt.pause(0.001)

        # ret, frame = cap.read()
        # iframe += 1

        midpoints_all.append(midpoints)


    return midpoints_all

if __name__ == "__main__":
    midpoints = find_midline("../data/contour/Control-EGCaMP_exp1_a1_30x10fps_5%.xml",
                             "../data/marker/Control-EGCaMP_exp1_a1_30x10fps_5%_001DLC_resnet50_EGCaMPFeb14shuffle1_576000.csv",
                             "/home/hengji/Documents/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi")

    df = pd.DataFrame(midpoints)
    df.to_csv("./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints/midpoints_bisection.csv", index=False)