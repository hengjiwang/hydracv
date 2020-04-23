import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os
from tqdm import tqdm

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

def load_data(file_icy, file_dlc, drop = True, threshold = 0.5, scale = (2, 2)):
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

    if drop: inds_bad = drop_bad_frames(df, threshold)

    return contours, df, inds_bad

def drop_bad_frames(df, threshold):
    '''
    Drop frames of which any likelihood is lower than threshold
    :return: index of dropped frames
    '''
    inds = df.index.values
    inds_good = df[df['hypostome_likelihood'] > threshold].index.values
    inds_good = np.intersect1d(inds_good, df[df['armpit1_likelihood'] > threshold].index.values, assume_unique = True)
    inds_good = np.intersect1d(inds_good, df[df['armpit2_likelihood'] > threshold].index.values, assume_unique = True)
    inds_good = np.intersect1d(inds_good, df[df['peduncle_likelihood'] > threshold].index.values, assume_unique = True)

    inds_bad = np.setdiff1d(inds, inds_good)
    df.iloc[inds_bad] = None

    return inds_bad

def armpit_line(armpit1, armpit2):

    try:
        slope = (armpit2[1] - armpit1[1])/(armpit2[0] - armpit1[0])
        const = (slope*armpit1[0]*-1)  + armpit1[1]
    except ZeroDivisionError:
        slope = np.inf
        const = np.inf

    return slope, const

def same_side_of_line(slope, const, hypostome, pt):
    if hypostome == None or slope == np.inf:
        return 1
    s1 = np.sign(hypostome[1] - slope*hypostome[0] - const)
    s2 = np.sign(pt[1] - slope*pt[0] - const)
    if(s1 == s2):
        return 0
    return 1

def locate_point(marker, contour, hypostome = None, slope = None, const = None):
    '''
    Locate the corresponding index of the marker on contour
    :return: int; index of marker
    '''
    index = 0
    mindist = np.inf
    for j in range(len(contour)):
        dist = (marker[0] - contour[j][0])**2 + (marker[1] - contour[j][1])**2
        s = same_side_of_line(slope, const, hypostome, contour[j])
        if dist < mindist and s:
            mindist = dist
            index = j

    return index

def divide_contour(markers, contour):
    '''
    Divide contour based on the relative positions of markers
    :return: (list, list); two segments from the contour
    '''
    # Find the corresponding positions of the tracked points on contour
    ind_ped = locate_point((markers['peduncle_x'], markers['peduncle_y']), contour)

    try:
        hypostome = ( markers['hypostome_x'], markers['hypostome_y'] )
    except:
        hypostome = None
    slope, const = armpit_line((markers['armpit1_x'], markers['armpit1_y']), (markers['armpit2_x'], markers['armpit2_y']))
    ind_arm1 = locate_point((markers['armpit1_x'], markers['armpit1_y']), contour, hypostome, slope, const)
    ind_arm2 = locate_point((markers['armpit2_x'], markers['armpit2_y']), contour, hypostome, slope, const)


    # Divide the contour into two segments joint at ind_ped
    # marker_hypostome = (markers['hypostome_x'], markers['hypostome_y'])

    marker_hypostome = ((markers['armpit1_x']+markers['armpit2_x'])/2, (markers['armpit1_y']+markers['armpit2_y'])/2)


    if ind_arm1 < ind_arm2 < ind_ped:
        # seg1 = [marker_hypostome] + contour[ind_arm2:ind_ped+1]
        seg1 = contour[ind_arm2:ind_ped+1]
        seg2 = contour[ind_ped:] + contour[0:ind_arm1+1] #+ [marker_hypostome]
        seg2.reverse()
    elif ind_arm1 < ind_ped < ind_arm2:
        seg1 = contour[ind_ped:ind_arm2+1] #+ [marker_hypostome]
        seg1.reverse()
        # seg2 = [marker_hypostome] + contour[ind_arm1:ind_ped+1]
        seg2 = contour[ind_arm1:ind_ped+1]
    elif ind_arm2 < ind_arm1 < ind_ped:
        # seg2 = [marker_hypostome] + contour[ind_arm1:ind_ped+1]
        seg2 = contour[ind_arm1:ind_ped+1]
        seg1 = contour[ind_ped:] + contour[0:ind_arm2+1] #+ [marker_hypostome]
        seg1.reverse()
    elif ind_arm2 < ind_ped < ind_arm1:
        # seg1 = [marker_hypostome] + contour[ind_arm2:ind_ped+1]
        seg1 = contour[ind_arm2:ind_ped+1]
        seg2 = contour[ind_ped:ind_arm1+1] #+ [marker_hypostome]
        seg2.reverse()
    elif ind_ped < ind_arm1 < ind_arm2:
        seg2 = contour[ind_ped:ind_arm1+1] #+ [marker_hypostome]
        seg2.reverse()
        # seg1 = [marker_hypostome] + contour[ind_arm2:] + contour[0:ind_ped+1]
        seg1 = contour[ind_arm2:] + contour[0:ind_ped+1]
    else:
        seg1 = contour[ind_ped:ind_arm2+1] #+ [marker_hypostome]
        seg1.reverse()
        # seg2 = [marker_hypostome] + contour[ind_arm1:] + contour[0:ind_ped+1]
        seg2 = contour[ind_arm1:] + contour[0:ind_ped+1]

    return seg1, seg2

def length_segment(seg):
    '''
    :return: the path-length of seg
    :rtype: float
    '''
    length = 0
    for j in range(len(seg)-1):
        p1, p2 = seg[j], seg[j+1]
        length += np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    return length

def find_midindex(seg):
    '''
    :return: The midpoint index of seg
    :rtype: int
    '''
    length_total = length_segment(seg)
    length = 0
    index = 0

    while length < length_total/2:
        length += length_segment(seg[index:index+2])
        index += 1

    return index

def find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [], depth = 0):
    '''
    :return: middle points and corresponding side points on the segments
    :rtype: list(tuple), list((tuple, tuple))
    '''

    if depth <= max_depth:
        midind1 = find_midindex(seg1)
        midind2 = find_midindex(seg2)

        sidepoint1 = seg1[midind1]
        sidepoint2 = seg2[midind2]
        sidepoints.append((sidepoint1, sidepoint2))

        midpoint = ((sidepoint1[0] + sidepoint2[0]) / 2, (sidepoint1[1] + sidepoint2[1]) / 2)
        midpoints.append(midpoint)

        seg11 = seg1[0:midind1+1]
        seg12 = seg1[midind1:]
        seg21 = seg2[0:midind2+1]
        seg22 = seg2[midind2:]

        midpoints, sidepoints = find_midline(seg11, seg21, max_depth, midpoints, sidepoints, depth+1)
        midpoints, sidepoints = find_midline(seg12, seg22, max_depth, midpoints, sidepoints, depth+1)

    return midpoints, sidepoints

def closest_pt_array(pt, midpoints, sidepoints, re_midpoints=[], re_sidepoints=[]):

    if(len(midpoints) == 0):
        return re_midpoints, re_sidepoints

    if(len(midpoints) == 1):
        re_midpoints.append(midpoints[0])
        re_sidepoints.append(sidepoints[0])
        return re_midpoints, re_sidepoints

    dist_from_pt = [length_segment([p, pt]) for p in midpoints]
    idx = dist_from_pt.index(min(dist_from_pt))
    re_midpoints.append(midpoints[idx])
    re_sidepoints.append(sidepoints[idx])

    pt = midpoints.pop(idx)
    _ = sidepoints.pop(idx)

    return closest_pt_array(pt, midpoints, sidepoints, re_midpoints, re_sidepoints)

def main(file_icy, file_dlc, max_depth, scale, display=False):
    # Load data
    contours, df, _ = load_data(file_icy, file_dlc, scale=scale)

    dropped_frames = []
    midpoints_all = []

    # Presettings
    lengths = []
    num_frames = len(contours)
    plt.figure(figsize=(20,20))

    # Loop over all frames
    for iframe in tqdm(range(num_frames)):

        markers = df.iloc[iframe]
        contour = contours[iframe]

        # Pass dropped frames
        if np.isnan(markers[0]):
            lengths.append(lengths[-1])
            dropped_frames.append(iframe)
            midpoints_all.append(midpoints_all[-1])
            continue

        # Get midpoints
        try:
            seg1, seg2 = divide_contour(markers, contour)
            midpoints, sidepoints = find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [])
        except:
            lengths.append(lengths[-1])
            dropped_frames.append(iframe)
            midpoints_all.append(midpoints_all[-1])
            continue

        # Sort midpoints based on the distances with the peduncle point
        ind_ped = locate_point((markers['peduncle_x'], markers['peduncle_y']), contour)
        ped_point = (contour[ind_ped][0], contour[ind_ped][1])
        hyp_point = ((markers['armpit1_x']+markers['armpit2_x'])/2, (markers['armpit1_y']+markers['armpit2_y'])/2)
        hypostome_point = (markers['hypostome_x'], markers['hypostome_y'])

        # print(len(midpoints), len(sidepoints))
        re_midpoints=[]
        re_sidepoints=[]
        midpoints, sidepoints = closest_pt_array(ped_point, midpoints, sidepoints, re_midpoints, re_sidepoints)
        # print(len(midpoints), len(sidepoints))
        # dist_with_ped = [length_segment([p, ped_point]) for p in midpoints]
        # indexs = np.argsort(dist_with_ped)
        # midpoints = np.array(midpoints)[indexs]
        # np.append(midpoints, hyp_point)
        # sidepoints = np.array(sidepoints)[indexs]

        midpoints_all.append(midpoints)

        # Append length of midline
        lengths.append(length_segment(midpoints))

        # Extract coordinates lists
        contour_x = [p[0] for p in contour]
        contour_y = [p[1] for p in contour]

        seg1_x = [p[0] for p in seg1]
        seg1_y = [p[1] for p in seg1]

        mid_x = [p[0] for p in midpoints]
        mid_y = [p[1] for p in midpoints]

        # Draw
        # print(iframe)
        if(display):
            matplotlib.use('Qt5Agg')
            plt.clf()
            plt.scatter(contour_x, contour_y, color = '', marker = 'o', edgecolors= 'g')
            plt.scatter(hypostome_point[0], hypostome_point[1], color='k', marker='o')
            plt.plot(mid_x, mid_y, 'r.-')
            plt.plot([markers['armpit1_x'],hyp_point[0]], [markers['armpit1_y'],hyp_point[1]], 'go-')
            plt.plot([markers['armpit2_x'],hyp_point[0]], [markers['armpit2_y'],hyp_point[1]], 'go-')
            plt.plot([hyp_point[0], mid_x[-1]], [hyp_point[1], mid_y[-1]], 'r-')
            plt.plot([ped_point[0], mid_x[0]], [ped_point[1], mid_y[0]], 'r-')
            plt.plot(markers['armpit1_x'], markers['armpit1_y'], 'bo')
            plt.plot(markers['armpit2_x'], markers['armpit2_y'], 'bo')
            plt.plot(hyp_point[0],hyp_point[1], color='orange', marker='o')
            plt.plot(ped_point[0],ped_point[1], color= 'purple', marker = 'o')
            plt.xlim(0, 600)
            plt.ylim(0, 600)
            plt.pause(0.0001)

        # if(iframe > 1600):
        #     input('Press Enter')


    return lengths, dropped_frames, midpoints_all


if __name__ == '__main__':

    df = pd.read_json('config.json')
    lengths, dropped_frames, midpoints_all = main(df.IcyFilePath.values[0],
                df.DeeplabcutFilePath.values[0],
                df.MaxDepth.values[0],
                scale=(df.ScaleX.values[0], df.ScaleY.values[0]), display=True)

    fig = plt.figure()
    plt.plot(lengths)
    plt.show()
    identifier = df.IcyFilePath.values[0].split('/')[-1].strip('.xml')
    try:
        fig.savefig('output/lengths_' + identifier + '.png')
    except FileNotFoundError:
        os.makedirs('output/')
        fig.savefig('output/lengths_' + identifier + '.png')
    df = pd.DataFrame(lengths)
    df.to_csv('output/lengths_' + identifier + '.csv', index=False)
    df = pd.DataFrame(dropped_frames)
    df.to_csv('output/dropped_frames_' + identifier + '.csv', index=False)

    mpt_df = []
    for mpt in midpoints_all:
        flat_list = []
        for x,y in mpt:
            flat_list.append(x)
            flat_list.append(y)
        mpt_df.append(flat_list)

    df = pd.DataFrame(mpt_df)
    df.to_csv('/home/shashank/Desktop/midpoints.csv', index = False)