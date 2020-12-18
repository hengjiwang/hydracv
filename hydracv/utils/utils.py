import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))

import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.signal
import utils.disp as disp
import xml.etree.ElementTree as ET

def find_peaks(seq, height=0.1, wlen=100, prominence=0.025, min_cb_interval=10, realign=True, start=0, end=-1):
    "Find peaks of seq"
    orig_seq = seq
    seq = normalize(seq)
    peaks, _ = scipy.signal.find_peaks(seq, height=height, wlen=wlen, prominence=prominence)

    fig = plt.figure(figsize=(20, 3))
    ax = fig.add_subplot(1, 1, 1)
    disp.add_fluorescence(ax, range(len(orig_seq)), orig_seq)
    disp.add_peaks(ax, peaks, orig_seq, 1)

    if end == -1:
        end = len(seq)
    ax.set_xlim(start, end)

    plt.show()

    return np.array([x for x in peaks if start < x < end])

def cluster_peaks(peaks, min_cb_interval, realign=True):
    """Separate peaks into different clusters based on min_cb_interval(in frame numbers)"""
    clusters = [[]]

    # Clustering peaks
    for j in range(len(peaks)-1):
        pk = peaks[j]
        pk_nxt = peaks[j+1]
        clusters[-1].append(pk)
        if pk_nxt - pk < min_cb_interval:
            pass
        else:
            clusters.append([])

    clusters[-1].append(peaks[-1])

    # Subtracting offsets
    indices_to_keep = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        if len(cluster) >= 3:
            indices_to_keep.append(i)
        if realign:
            offset = cluster[0]
            for j in range(len(cluster)):
                cluster[j] -= offset

    return np.array(clusters)[indices_to_keep]

def reformat_periods(periods, unit='sec'):
    "Reformat periods extracted from manual labeling data"
    res = []
    for j in range(len(periods)):

        time_str_pair = periods[j].split('-')

        if len(time_str_pair) < 2:
            continue

        res.append((reformat_time(time_str_pair[0], unit), reformat_time(time_str_pair[1], unit)))

    return res

def reformat_time(time_string, unit='sec'):
    "Convert the time string to real time with given unit"
    time_pair = [int(x) for x in time_string.split(':')]

    if unit == 'sec':

        time_num = time_pair[0]*60 + time_pair[1]

    elif unit == 'min':

        time_num = time_pair[0] + time_pair[1]/60

    return time_num

def length_of_model(coordspath, totaltime=300, loc='x', display=True):
    "Extract the length trace of model from the coordinates of sidepoints"
    # Get number of points and time steps
    count = 0
    ntime = 0
    with open(coordspath, 'r') as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break
            if count == 7:
                npoints = len(line.split())
            if count >= 6:
                ntime += 1

    ntime = ntime // 3

    # Reformat coordinates
    mat = np.zeros((ntime, npoints, 3))

    count = 0
    itime = -1
    with open(coordspath, 'r') as fp:
        while True:
            count += 1
            line = fp.readline()
            if not line:
                break

            if count < 6:
                pass
            elif count % 3 == 0:
                itime += 1
                coords = line.split()[1:]
                coords = [float(x) for x in coords]
                mat[itime, :, 0] = coords
            elif count % 3 == 1:
                coords = line.split()
                coords = [float(x) for x in coords]
                mat[itime, :, 1] = coords
            else:
                coords = line.split()
                coords = [float(x) for x in coords]
                mat[itime, :, 2] = coords

    # Change the unit from m to mm
    mat *= 1000

    # Divide positive and negative points
    mat_pos = np.zeros((ntime, npoints//2, 3))
    mat_neg = np.zeros((ntime, npoints//2, 3))

    ipos = 0
    ineg = 0

    division = 1 if loc == 'x' else 0 if loc == 'y' else None

    for j in range(len(mat[0])):
        if mat[division][j][0] < 0:
            mat_neg[:, ineg, :] = mat[:, j, :]
            ineg += 1
        else:
            mat_pos[:, ipos, :] = mat[:, j, :]
            ipos += 1

    # Sort the points from bottom to top
    z_original = mat_pos[0, :, 2]
    argsort = np.argsort(z_original)
    mat_pos_sorted = mat_pos[:, argsort, :]

    z_original = mat_neg[0, :, 2]
    argsort = np.argsort(z_original)
    mat_neg_sorted = mat_neg[:, argsort, :]

    # Get the middle points
    mat_mid = (mat_pos_sorted + mat_neg_sorted) / 2

    # Calculate the length
    lengths = []

    for j in range(len(mat_mid)):
        points = mat_mid[j]
        length = 0
        for k in range(len(points)-1):
            diffvec = points[k+1] - points[k]
            dist = LA.norm(diffvec)
            length += dist
        lengths.append(length)

    if display:
        plt.figure(figsize=(int(totaltime/30),3))
        plt.plot(np.arange(0, totaltime+0.1, 0.1), lengths, color='b', linewidth=2)
        plt.xlabel('time(s)')
        plt.ylabel('length(mm)')
        plt.show()

    return lengths

def normalize(seq):
    "Normalize a sequence"
    minval = min(seq)
    maxval = max(seq)
    return [(x - minval) / (maxval - minval) for x in seq]

def filter_abnormal(seq, size=50, thres=0.2):
    "Filter out abnormal points in seq"

    res = []
    seq_smooth = scipy.ndimage.filters.median_filter(seq, size=size)

    maxval = max(seq)
    minval = min(seq)

    for j in range(len(seq)):
        val = seq[j]
        val_smooth = seq_smooth[j]
        if abs(val - val_smooth) > thres * (maxval - minval):
            res.append(val_smooth)
        else:
            res.append(val)

    return res

def load_contour_j(file_icy):
    root = ET.parse(file_icy).getroot()
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

def load_contour(file_icy):
    # Extract coordinates from file_icy
    root = ET.parse(file_icy).getroot()
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

def load_tracked_points(file_dlc):
    # Extract coordinates from file_dlc
    df = pd.read_csv(file_dlc)
    df.columns = ['scorer', 'hypostome_x', 'hypostome_y', 'hypostome_likelihood',
              'armpit1_x', 'armpit1_y', 'armpit1_likelihood',
              'armpit2_x', 'armpit2_y', 'armpit2_likelihood',
              'peduncle_x', 'peduncle_y', 'peduncle_likelihood']
    df = df.drop(index = [0,1]).drop(columns='scorer').reset_index(drop = True)
    df = df.astype(float)
    return df

def load_contours_markers(file_icy, file_dlc, drop = True, threshold = 0.5, scale = (2, 2)):
    '''
    :param file_icy: .xml file of hydra contour generated by ICY
    :param file_dlc: .csv file of tracked points generated by DeepLabCut
    :param drop: bool; True if to drop bad frames, otherwise do nothing
    :param threshold: float; threshold for drop
    :return: coordinates of contour, coordinates of tracked points, index of bad frames
    :rtype: list, pandas.core.frame.DataFrame, numpy.array
    '''
    try:
        contours = load_contour(file_icy)
    except:
        print('loading')
        contours = load_contour_j(file_icy)
    df = load_tracked_points(file_dlc)

    # Scale coordinates
    df[['hypostome_x', 'peduncle_x', 'armpit1_x', 'armpit2_x']] /= scale[0]
    df[['hypostome_y', 'peduncle_y', 'armpit1_y', 'armpit2_y']] /= scale[1]

    # if drop: inds_bad = drop_bad_frames(df, threshold)

    return contours, df

def length_segment(seg):
    "Returns length of segment seg"
    length = 0
    for j in range(len(seg)-1):
        p1, p2 = seg[j], seg[j+1]
        length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    return length

def interpolate(start, end, nintp):
    res = [start]
    step_x = (end[0] - start[0]) / (nintp - 1)
    step_y = (end[1] - start[1]) / (nintp - 1)
    curr = start
    for i in range(nintp-1):
        curr = (curr[0]+step_x, curr[1]+step_y)
        res.append(curr)
    return res

def midpoint_of(segment):
    length = length_segment(segment)
    next_len = 0
    for j in range(1, len(segment)):
        prev = segment[j-1]
        next = segment[j]

        next_len += length_segment([prev, next])
        if next_len > length / 2:
            index = j-1
            break
        else:
            curr_len = next_len

    pts = interpolate(prev, next, 20)
    for j in range(1, len(pts)):
        curr_len += length_segment([pts[j-1], pts[j]])
        if curr_len >= length / 2:
            return pts[j], index

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