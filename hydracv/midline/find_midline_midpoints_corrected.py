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
        try:
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
        except:
            root = ET.parse(filename).getroot()
            rois = root.findall('roi')
            contours = []
            for i in range(len(rois)):
                contours.append(0)

            for roi in rois:
                id = int(roi.find('t').text)
                points = roi.find('points').findall('point')
                contour = []
                for point in points:
                    pos_x = float(point.find('pos_x').text)
                    pos_y = float(point.find('pos_y').text)
                    contour.append((pos_x, pos_y))
                try:
                    contours[id] = contour
                except:
                    print(id)

    return contours

def load_marker(filename):
    "Load tracked points"
    df = pd.read_csv(filename)
    df.columns = ['scorer', 'hypostome_x', 'hypostome_y', 'hypostome_likelihood',
                  'armpit1_x', 'armpit1_y', 'armpit1_likelihood',
                  'armpit2_x', 'armpit2_y', 'armpit2_likelihood',
                  'peduncle_x', 'peduncle_y', 'peduncle_likelihood']
    df = df.drop(index=[0, 1]).drop(columns='scorer').reset_index(drop=True)
    df = df.astype(float)
    return df.values

def load_midpoints(filename):
    "Load the midpoints"
    midpoints = []
    data = pd.read_csv(filename).values
    nframes = len(data)
    npoints = len(data[0]) // 2

    for i in range(nframes):
        midpoints.append([])
        for j in range(npoints):
            midpoints[-1].append((data[i][2*j], data[i][2*j+1]))

    return np.array(midpoints)

def angle(x, y, z):
    lenxy = length_segment([x, y])
    lenxz = length_segment([x, z])
    lenyz = length_segment([y, z])

    ag = np.arccos((lenxy**2 + lenyz**2 - lenxz**2) / (2 * lenxy * lenyz))

    # print("Angle: " + str(ag))

    return ag

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

def locate_point_with_angle_constr(marker, contour, angle_diff_thres):
    "Locate the corresponding index of the marker on contour"

    index = 0
    mindist = np.inf
    for j in range(1, len(contour)-1):
        dist = (marker[0] - contour[j][0])**2 + (marker[1] - contour[j][1])**2
        if dist < mindist:
            ag = angle(marker, contour[j-1], contour[j+1])
            if np.pi/2 - angle_diff_thres < ag < np.pi/2 + angle_diff_thres:
                # print("Final Angle: " + str(ag))
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

def middle_point(pt1, pt2):
    "Returns the midpoint of pt1 and pt2"
    return ((pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2)


def find_midpoints(seg1, seg2, midpoints, nseg, ax=None):
    "Find the midpoints of seg1 and seg2"
    len_contour_1 = length_segment(seg1)
    len_contour_2 = length_segment(seg2)
    ind_seg_pt1 = 0
    ind_seg_pt2 = 0
    cum_len_1 = 0
    cum_len_2 = 0
    for j in range(1, nseg):

        # Locate the segment points
        while cum_len_1 <= j/nseg * len_contour_1:
            cum_len_1 += length_segment(seg1[ind_seg_pt1:ind_seg_pt1+2])
            ind_seg_pt1 += 1

        while cum_len_2 <= j/nseg * len_contour_2:
            cum_len_2 += length_segment(seg2[ind_seg_pt2:ind_seg_pt2+2])
            ind_seg_pt2 += 1

        # if len(seg2) == 0:
        #     input()

        seg_pt_1 = seg1[ind_seg_pt1]
        seg_pt_2 = seg2[ind_seg_pt2]

        midpoint = ((seg_pt_1[0] + seg_pt_2[0]) // 2, (seg_pt_1[1] + seg_pt_2[1]) // 2)

        if ax:
            # ax.plot([seg_pt_1[0], seg_pt_2[0]], [seg_pt_1[1], seg_pt_2[1]], 'r-')
            ax.plot(midpoint[0], midpoint[1], 'r.')

        midpoints.append(midpoint[0])
        midpoints.append(midpoint[1])

def extract_lengths(midpoints, normalize=True):
    "Extract lengths from midpoints"
    midline_lens = []
    for midline in midpoints:
        midline_lens.append(length_segment(midline))

    maxlen = max(midline_lens)
    minlen = min(midline_lens)

    if normalize:
        res = [(x - minlen) / (maxlen - minlen) for x in midline_lens]
    else:
        res = midline_lens

    return res

def find_midline(file_contour, file_marker, file_midpoints, nseg=40, display=False):
    "Find midline"

    # Load files
    contours = load_contour(file_contour)
    markers = load_marker(file_marker)
    midpoints_orig = load_midpoints(file_midpoints)

    missed_contour = list(range(12096, 12145))
    markers = [x for i,x in enumerate(markers) if i not in missed_contour]

    # markers = markers[:]
    # contours = contours[7:]

    print(len(contours), len(markers))

    midlens = extract_lengths(midpoints_orig)

    midpoints_all = []

    # iframe = 0

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = None

    for iframe in tqdm(range(len(contours))):

        if display:
            ax.clear()

        # Extract contour and marker
        contour = contours[iframe]
        marker_mat = markers[iframe]
        midpoint_orig = midpoints_orig[iframe]

        midlen = midlens[iframe]

        if display:
            ax.plot(midpoint_orig[:, 0], midpoint_orig[:, 1], 'k.')

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

        # Locate middle segment point on one side
        len_half_1 = length_segment(contour_half_1)
        len_half_2 = length_segment(contour_half_2)

        ind_mid_1 = 0
        cum_len_1 = 0
        while cum_len_1 < len_half_1 / 2:
            cum_len_1 += length_segment(contour_half_1[ind_mid_1:ind_mid_1+2])
            ind_mid_1 += 1

        pt_mid_1 = contour_half_1[ind_mid_1]

        # Locate the middle segment point on the other side

        ind_mid_2 = locate_point(pt_mid_1, contour_half_2)

        criteria = midlen > 0.3

        if not criteria:
            ind_mid_2 = 0
            cum_len_2 = 0
            while cum_len_2 < len_half_2 / 2:
                cum_len_2 += length_segment(contour_half_2[ind_mid_2:ind_mid_2+2])
                ind_mid_2 += 1

        if ind_mid_2 == 0:
            cum_len_2 = 0
            while cum_len_2 < len_half_2 / 2:
                cum_len_2 += length_segment(contour_half_2[ind_mid_2:ind_mid_2+2])
                ind_mid_2 += 1

        pt_mid_2 = contour_half_2[ind_mid_2]

        # Separate half contours based on the middle points
        contour_half_11 = np.array(contour_half_1[:ind_mid_1])
        contour_half_12 = np.array(contour_half_1[ind_mid_1:])
        contour_half_21 = np.array(contour_half_2[:ind_mid_2])
        contour_half_22 = np.array(contour_half_2[ind_mid_2:])

        # Find the midpoints
        midpoints = []
        find_midpoints(contour_half_11, contour_half_21, midpoints, nseg//2, ax)
        find_midpoints(contour_half_12, contour_half_22, midpoints, nseg//2, ax)

        midpoints = np.array(midpoints)

        midpoints_all.append(midpoints)

        # ax.plot([pt_mid_1[0], pt_mid_2[0]], [pt_mid_1[1], pt_mid_2[1]], 'k')

        mid_mid_pt = middle_point(pt_mid_1, pt_mid_2)

        # ax.plot(mid_mid_pt[0], mid_mid_pt[1], 'k', marker='.', markersize=10)

        if display:
            ax.plot(contour_half_11[:,0], contour_half_11[:,1], 'g.', markersize=5)
            ax.plot(contour_half_12[:,0], contour_half_12[:,1], 'k.', markersize=5)
            ax.plot(contour_half_21[:,0], contour_half_21[:,1], 'b.', markersize=5)
            ax.plot(contour_half_22[:,0], contour_half_22[:,1], 'k.', markersize=5)



        # plt.plot(pt_mid_1[0], pt_mid_1[1], 'ro')
        # plt.plot(pt_mid_2[0], pt_mid_2[1], 'ro')
        # plt.plot(contour_half_2[ind_mid_2_][0], contour_half_2[ind_mid_2_][1], 'yo')

        if display:
            ax.set_xlim(0, 500)
            ax.set_ylim(0, 500)
            plt.pause(0.00001)


    return midpoints_all

if __name__ == "__main__":
    # FILENAME = "Control-EGCaMP_exp1_a1_30x10fps"
    # midpoints = find_midline("../data/contour/Control-EGCaMP_exp1_a1_30x10fps_5%.xml",
    #                          "../data/marker/Control-EGCaMP_exp1_a1_30x10fps_5%_001DLC_resnet50_EGCaMPFeb14shuffle1_576000.csv",
    #                          "./results/Control-EGCaMP_exp1_a1_30x10fps/midpoints/midpoints_bisection.csv")

    # FILENAME = "0hr_Control_ngcampmov_30x4fps_50%intensity_exp3_a3"

    # midpoints = find_midline("../data/contour/" + FILENAME + ".xml",
    #                          "../data/marker/0hr_Control_ngcampmov_30x4fps_50%intensity_exp3_a3_enhanced_editDLC_resnet50_TTypectrlFeb26shuffle1_524000.csv",
    #                          "./results/" + FILENAME + "/midpoints/midpoints_bisection.csv")

    # df = pd.DataFrame(midpoints)
    # df.to_csv("./results/" + FILENAME + "/midpoints/midpoints_bisection_corrected.csv", index=False)

    # FILENAME = "0hr_Cafree+EGTA_ngcampmov_30x4fps_50%intensity_exp2_a1"

    # midpoints = find_midline("../data/contour/" + FILENAME + ".xml",
    #                          "../data/marker/0hr_Cafree+EGTA_ngcampmov_30x4fps_50%intensity_exp2_a1_001DLC_resnet50_CaFree+EGTAFeb24shuffle1_596000.csv",
    #                          "./results/" + FILENAME + "/midpoints/midpoints_bisection.csv")

    # df = pd.DataFrame(midpoints)
    # df.to_csv("./results/" + FILENAME + "/midpoints/midpoints_bisection_corrected.csv", index=False)

    # FILENAME = "Pre_Bisect_40x_4fps_ex4"

    # midpoints = find_midline("../data/contour/Pre_Bisect_40x_4fps_ex4_ROIs.xml",
    #                          "../data/marker/Pre_Bisect_40x_4fps_ex4DeepCut_resnet50_Hydra2Nov17shuffle1_1030000.csv",
    #                          "./results/" + FILENAME + "/midpoints/midpoints_bisection.csv")

    # df = pd.DataFrame(midpoints)
    # df.to_csv("./results/" + FILENAME + "/midpoints/midpoints_bisection_corrected.csv", index=False)

    FILENAME = "Pre_Bisect_40x_4fps_ex3"

    midpoints = find_midline("../data/contour/" + FILENAME + ".xml",
                             "../data/marker/Pre_Bisect_40x_4fps_ex3DeepCut_resnet50_Hydra2Nov15shuffle1_197000.csv",
                             "../data/midpoints/" + FILENAME + "_midpoints.csv",
                             display=True)

    df = pd.DataFrame(midpoints)
    df.to_csv("./results/" + FILENAME + "/midpoints/midpoints_bisection_corrected.csv", index=False)