import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os, re, cv2

from midline import *

def load_tracked_points(file_dlc):
    df = pd.read_csv(file_dlc)
    names = df.iloc[0].values
    df.columns = names
    df = df.drop(index = [0]).reset_index(drop=True)

    for j in range(1, len(df.columns)):
        df.columns.values[j] += '_' + df.iloc[0].values[j]
        df.columns.values[j] = df.columns.values[j].lower()

    df = df.drop(index=[0]).reset_index(drop=True)
    df = df.astype(float)

    return df

def load_data(file_icy, file_dlc, drop=True, threshold = 0.5, scale=(2, 2)):
    contours = load_contour(file_icy)
    df = load_tracked_points(file_dlc)

    # Rescale coordinates to match ICY and DeepLabCut results
    for col in df.columns:
        if re.search('_x', col):
            df[col] /= scale[0]
        elif re.search('_y', col):
            df[col] /= scale[1]

    if drop: inds_bad = drop_bad_frames(df, threshold)

    return contours, df, inds_bad

def drop_bad_frames(df, threshold):

    likelihoods = []
    for col in df.columns:
        if re.search('likelihood', col):
            likelihoods.append(col)

    inds = df.index.values
    inds_good = df[df[likelihoods[0]] > threshold].index.values
    for l in likelihoods[1:]:
        inds_good = np.intersect1d(inds_good, df[df[l] > threshold].index.values, assume_unique = True)
    inds_bad = np.setdiff1d(inds, inds_good)
    df.iloc[inds_bad] = None

    return inds_bad

def divide_contour(markers, contour):

    ind_hyp = locate_point((markers['hypostome_x'], markers['hypostome_y']), contour)
    ind_ped = locate_point((markers['peduncle_x'], markers['peduncle_y']), contour)

    if ind_hyp < ind_ped:
        seg1 = contour[ind_hyp:ind_ped+1]
        seg2 = contour[ind_ped:] + contour[0:ind_hyp+1]
        seg2.reverse()
    else:
        seg1 = contour[ind_ped:ind_hyp]
        seg2 = contour[ind_hyp:] + contour[0:ind_ped+1]
        seg1.reverse()

    return seg1, seg2

def sort_midpoints(markers, midpoints, sidepoints, hyp_point, ped_point):
    # Sort midpoints based on the distances with the peduncle point
    dist_with_ped = [length_segment([p, ped_point]) for p in midpoints]
    indexs = np.argsort(dist_with_ped)
    midpoints = np.array(midpoints)[indexs]
    np.append(midpoints, hyp_point)
    sidepoints = np.array(sidepoints)[indexs]
    return midpoints, sidepoints

def draw(contour, midpoints, hyp_point, ped_point):
    # Extract coordinates lists
    contour_x = [p[0] for p in contour]
    contour_y = [p[1] for p in contour]
    mid_x = [p[0] for p in midpoints]
    mid_y = [p[1] for p in midpoints]

    # Draw
    plt.clf()
    plt.scatter(contour_x, contour_y, color = '', marker = 'o', edgecolors= 'g')
    plt.plot(mid_x, mid_y, 'r.-')
    plt.plot([hyp_point[0], mid_x[-1]], [hyp_point[1], mid_y[-1]], 'r-')
    plt.plot([ped_point[0], mid_x[0]], [ped_point[1], mid_y[0]], 'r-')
    plt.plot(hyp_point[0],hyp_point[1], color='orange', marker='o')
    plt.plot(ped_point[0],ped_point[1], color= 'purple', marker = 'o')
    plt.xlim(0, 1000)
    plt.ylim(0, 500)
    plt.pause(0.0001)


def run(file_icy, file_dlc, max_depth, scale):

    contours, df, _ = load_data(file_icy, file_dlc, scale=scale)

    # Presettings
    lengths = []
    num_frames = len(contours)
    plt.figure(figsize=(20,20))

    # Loop over all frames
    for iframe in range(num_frames):

        markers = df.iloc[iframe]
        contour = contours[iframe]

        # Pass dropped frames
        if np.isnan(markers[0]):
            continue

        # Get midpoints
        seg1, seg2 = divide_contour(markers, contour)
        midpoints, sidepoints = find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [])
        
        # Sort midpoints and sidepoints
        ped_point = (markers['peduncle_x'], markers['peduncle_y'])
        hyp_point = (markers['hypostome_x'],markers['hypostome_y'])
        midpoints, sidepoints = sort_midpoints(markers, midpoints, 
            sidepoints, hyp_point, ped_point)

        # Append length of midline
        lengths.append(length_segment(midpoints))

        # Draw
        draw(contour, midpoints, hyp_point, ped_point)
        

    return lengths


if __name__ == "__main__":
    # lengths = main('../data/hy78clip1_R2.xml', '../data/hy78clip1DeepCut_resnet50_clip1Mar24shuffle1_124000.csv', max_depth = 5)
    lengths = run(sys.argv[1], sys.argv[2], max_depth = int(sys.argv[3]), scale=(int(sys.argv[4]), int(sys.argv[5])) )
    fig = plt.figure()
    plt.plot(lengths)
    plt.show()
    identifier = sys.argv[1].split('/')[-1].strip('.xml')
    try:
        fig.savefig('output/lengths_' + identifier + '.png')
    except FileNotFoundError:
        os.makedirs('output/')
        fig.savefig('output/lengths_' + identifier + '.png')
    df = pd.DataFrame(lengths)
    df.to_csv('output/lengths_' + identifier + '.csv', index=False)




    