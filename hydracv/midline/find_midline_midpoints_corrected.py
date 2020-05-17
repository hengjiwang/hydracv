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

def angle(x, y, z):
    return np.arccos((x**2 + y**2 - z**2) / (2 * x * y))

def find_midpoints(seg1, seg2, midpoints, nseg):
    "Find the midpoints of seg1 and seg2"
    len_contour_1 = length_segment(seg1)
    len_contour_2 = length_segment(seg2)
    ind_seg_pt1 = 0
    ind_seg_pt2 = 0
    cum_len_1 = 0
    cum_len_2 = 0
    for j in range(1, nseg):
        
        # Locate the segment points
        while cum_len_1 < j/nseg * len_contour_1:
            cum_len_1 += length_segment(seg1[ind_seg_pt1:ind_seg_pt1+2])
            ind_seg_pt1 += 1

        while cum_len_2 < j/nseg * len_contour_2:
            cum_len_2 += length_segment(seg2[ind_seg_pt2:ind_seg_pt2+2])
            ind_seg_pt2 += 1

        if len(seg2) == 0:
            input()

        seg_pt_1 = seg1[ind_seg_pt1]
        seg_pt_2 = seg2[ind_seg_pt2]
        midpoint = ((seg_pt_1[0] + seg_pt_2[0]) // 2, (seg_pt_1[1] + seg_pt_2[1]) // 2)
        midpoints.append(midpoint)

def find_midline(file_contour, file_marker, file_behaviors, nseg=40):
    "Find midline"

    # Load files
    contours = load_contour(file_contour)
    markers = load_marker(file_marker).values
    
    behaviors = pd.read_csv(file_behaviors).values

    midpoints_all = []

    iframe = 0
    plt.figure()

    for iframe in tqdm(range(len(contours))):

        # Extract contour and marker
        contour = contours[iframe]
        marker_mat = markers[iframe]
        behavior = behaviors[iframe]

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

        # Locate middle segmenting points

        len_half_1 = length_segment(contour_half_1)
        len_half_2 = length_segment(contour_half_2)

        # ind_mid_1 = len(contour_half_1) // 2
        ind_mid_1 = 0
        cum_len_1 = 0
        while cum_len_1 < len_half_1 / 2:
            cum_len_1 += length_segment(contour_half_1[ind_mid_1:ind_mid_1+2])
            ind_mid_1 += 1

        pt_mid_1 = contour_half_1[ind_mid_1]
        ind_mid_2 = locate_point(pt_mid_1, contour_half_2)
        pt_mid_2 = contour_half_2[ind_mid_2]

        # criteria = angle(length_segment([pt_mid_2, pt_mid_1]),
        #                  length_segment([pt_mid_1, marker['peduncle']]),
        #                  length_segment([pt_mid_2, marker['peduncle']])) < np.pi/15

        # print(behavior)
        
        criteria = (behavior[2] == 'Bending') and ind_mid_2 > 6

        if not criteria:
            ind_mid_2 = 0
            cum_len_2 = 0
            while cum_len_2 < len_half_2 / 2:
                cum_len_2 += length_segment(contour_half_2[ind_mid_2:ind_mid_2+2])
                ind_mid_2 += 1

        # ind_mid_2 = max(ind_mid_2_, ind_mid_2)



        # Separate half contours based on the middle points
        contour_half_11 = contour_half_1[:ind_mid_1]
        contour_half_12 = contour_half_1[ind_mid_1:]
        contour_half_21 = contour_half_2[:ind_mid_2]
        contour_half_22 = contour_half_2[ind_mid_2:]

        # if iframe > 1800:

        #     plt.clf()

        #     for pt in contour_half_11:
        #         plt.plot(pt[0], pt[1], 'g.', markersize=5)
        #     for pt in contour_half_12:
        #         plt.plot(pt[0], pt[1], 'k.', markersize=5)

        #     for pt in contour_half_21:
        #         plt.plot(pt[0], pt[1], 'b.', markersize=5)
        #     for pt in contour_half_22:
        #         plt.plot(pt[0], pt[1], 'k.', markersize=5)

        #     # plt.plot(pt_mid_1[0], pt_mid_1[1], 'ro')
        #     plt.plot(pt_mid_2[0], pt_mid_2[1], 'ro')
        #     # plt.plot(contour_half_2[ind_mid_2_][0], contour_half_2[ind_mid_2_][1], 'yo')

        #     plt.xlim(0, 500)
        #     plt.ylim(0, 500)
        #     plt.pause(0.00001)

        # Find the midpoints
        midpoints = []
        find_midpoints(contour_half_11, contour_half_21, midpoints, nseg//2)
        find_midpoints(contour_half_12, contour_half_22, midpoints, nseg//2)

        midpoints_all.append(midpoints)


    return midpoints_all

if __name__ == "__main__":
    midpoints = find_midline("../data/contour/Control-EGCaMP_exp1_a1_30x10fps_5%.xml",
                             "../data/marker/Control-EGCaMP_exp1_a1_30x10fps_5%_001DLC_resnet50_EGCaMPFeb14shuffle1_576000.csv",
                             '/home/hengji/Documents/hydracv/hydracv/classifier/results/Control-EGCaMP_exp1_a1_30x10fps/behaviors.csv')

    df = pd.DataFrame(midpoints)
    df.to_csv("./results/" + "Control-EGCaMP_exp1_a1_30x10fps/" + "midpoints/midpoints_bisection_corrected.csv", index=False)