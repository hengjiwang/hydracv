import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os, re, cv2
import sys,os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from find_midline.midline_ped import *
from tqdm import tqdm

def draw(contour, midpoints, hyp_point, ped_point, spots, map_points, seg1_points, seg2_points, frame, scale, project_name, iframe):
    # Extract coordinates lists
    contour_x = [p[0] for p in contour]
    contour_y = [p[1] for p in contour]
    mid_x = [p[0] for p in midpoints]
    mid_y = [p[1] for p in midpoints]

    # Draw
    plt.clf()
    plt.imshow(frame)
    plt.scatter(contour_x, contour_y, color = '', marker = 'o', edgecolors= 'g')
    plt.plot(mid_x, mid_y, 'r-')
    plt.plot([hyp_point[0], mid_x[-1]], [hyp_point[1], mid_y[-1]], 'r-')
    plt.plot([ped_point[0], mid_x[0]], [ped_point[1], mid_y[0]], 'r-')
    plt.plot(hyp_point[0],hyp_point[1], color='purple', marker='o')
    plt.plot(ped_point[0],ped_point[1], color= 'purple', marker = 'o')
    for j in range(len(spots)):
        # plt.plot([spots[j][0], mappoints[])
        plt.plot([seg1_points[j][0], seg2_points[j][0]],
        [seg1_points[j][1], seg2_points[j][1]], 'b-')
        plt.scatter(spots[j][0], spots[j][1], color="orange")

    plt.xlim(0, 544)
    plt.ylim(0, 440)
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.0001)

    try:
        plt.savefig('save/animations/' + project_name + '/frames/img' + str(iframe) + '.jpg', bbox_inches='tight')
    except FileNotFoundError:
        os.makedirs('save/animations/' + project_name + '/frames/')
        os.makedirs('save/animations/' + project_name + '/movie/')
        plt.savefig('save/animations/' + project_name + '/frames/img' + str(iframe) + '.jpg', bbox_inches='tight')

def interpolate(points, num):
    # Interpolate points

    new_midpoints = []

    for i in range(len(points)-1):
        this_point = points[i]
        next_point = points[i+1]

        x = np.linspace(this_point[0], next_point[0], num)
        y = np.linspace(this_point[1], next_point[1], num)

        for j in range(len(x)-1):
            new_midpoints.append([x[j], y[j]])

    new_midpoints.append(points[-1])

    return new_midpoints

def slope(p1, p2):
    # Compute the slope of (p1, p2)
    return (p2[1] - p1[1])/(p2[0] - p1[0])

def add_normal_lines():
    pass

def run(file_icy, file_dlc, max_depth, scale, videopath, interpolate_midline_num, interpolate_contour_num, project_name):

    contours, df, _ = load_data(file_icy, file_dlc, scale=scale)

    # Presettings
    lengths = []
    num_frames = len(contours)
    plt.figure(figsize=(20,20))
    cap = cv2.VideoCapture(videopath)

    # Loop over all frames
    for iframe in tqdm(range(num_frames)):

        markers = df.iloc[iframe]
        contour = contours[iframe]
        ret, frame = cap.read()

        contour = interpolate(contour, interpolate_contour_num)

        # Pass dropped frames
        if np.isnan(markers[0]):
            continue

        # Get midpoints
        seg1, seg2 = divide_contour(markers, contour)
        midpoints, sidepoints = find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [])

        # Tracked points
        xs = []
        ys = []
        for col in markers.keys():
            if re.search('_x', col) and not re.search('peduncle_x', col) and not re.search('hypostome_x', col):
                xs.append(markers[col])
            elif re.search('_y', col) and not re.search('peduncle_y', col) and not re.search('hypostome_y', col):
                ys.append(markers[col])

        spots = [(xs[j], ys[j]) for j in range(len(xs))]
        ped_point = (markers['peduncle_x'], markers['peduncle_y'])
        hyp_point = (markers['hypostome_x'],markers['hypostome_y'])

        # Sort midpoints and sidepoints
        midpoints, sidepoints = sort_midpoints(markers, midpoints,
            sidepoints, hyp_point, ped_point)

        # Append length of midline
        length = length_segment(midpoints)
        lengths.append(length)

        # Interpolate midpoints
        midpoints = interpolate(midpoints, interpolate_midline_num)

        # Find mapped points of spots on midline
        map_points = []
        seg1_points = []
        seg2_points = []
        for spot in spots:
            map_index = locate_point(spot, midpoints)
            map_point = midpoints[map_index]
            map_points.append(map_point)
            sl = slope(spot, map_point)
            min_p1 = None
            min_p2 = None
            min_diff1 = np.Infinity
            min_diff2 = np.Infinity
            for p in seg1:
                sl2 = slope(p, map_point)
                diff = np.abs(sl - sl2)
                if diff < min_diff1 and length_segment([p, map_point]) < 60:
                    min_diff1 = diff
                    min_p1 = p
            seg1_points.append(min_p1)
            for p in seg2:
                sl2 = slope(p, map_point)
                diff = np.abs(sl - sl2)
                if diff < min_diff2 and length_segment([p, map_point]) < 60:
                    min_diff2 = diff
                    min_p2 = p
            seg2_points.append(min_p2)

        # Draw
        draw(contour, midpoints, hyp_point, ped_point, spots, map_points, seg1_points, seg2_points, frame, scale, project_name, iframe)

    return lengths

if __name__ == "__main__":
    df = pd.read_json('config.json')
    lengths = run(df.IcyFilePath.values[0],
                df.DeeplabcutFilePath.values[0],
                df.MaxDepth.values[0],
                (df.ScaleX.values[0], df.ScaleY.values[0]),
                df.VideoPath.values[0],
                df.InterpolateMidlineNum.values[0],
                df.InterpolateContourNum.values[0],
                df.Name.values[0])
    fig = plt.figure()
    plt.plot(lengths)
    plt.show()
    # identifier = sys.argv[1].split('/')[-1].strip('.xml')
    # try:
    #     fig.savefig('output/lengths_' + identifier + '.png')
    # except FileNotFoundError:
    #     os.makedirs('output/')
    #     fig.savefig('output/lengths_' + identifier + '.png')
    # df = pd.DataFrame(lengths)
    # df.to_csv('output/lengths_' + identifier + '.csv', index=False)
