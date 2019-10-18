import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os
from shapely.geometry import Polygon, Point, MultiPoint
import cv2
from multiprocessing import Pool
from functools import partial


def load_data(file_icy, file_dlc, drop = True, threshold = 0.5, scale = (2, 2)):
    '''
    :param file_icy: .xml file of hydra contour generated by ICY
    :param file_dlc: .csv file of tracked points generated by DeepLabCut
    :param drop: bool; True if to drop bad frames, otherwise do nothing
    :param threshold: float; threshold for drop
    :return: coordinates of contour, coordinates of tracked points, index of bad frames
    :rtype: list, pandas.core.frame.DataFrame, numpy.array
    '''

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

    # Extract coordinates from file_dlc
    df = pd.read_csv(file_dlc)
    df.columns = ['scorer', 'hypostome_x', 'hypostome_y', 'hypostome_likelihood',
              'armpit1_x', 'armpit1_y', 'armpit1_likelihood',
              'armpit2_x', 'armpit2_y', 'armpit2_likelihood',
              'peduncle_x', 'peduncle_y', 'peduncle_likelihood']
    df = df.drop(index = [0,1]).drop(columns='scorer').reset_index(drop = True)
    df = df.astype(float)

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
    df_good = df[df['hypostome_likelihood'] > threshold]
    df_good = df[df['armpit1_likelihood'] > threshold]
    df_good = df[df['armpit2_likelihood'] > threshold]
    df_good = df[df['peduncle_likelihood'] > threshold]
    inds_good = df_good.index.values

    inds_bad = np.setdiff1d(inds, inds_good)
    df.iloc[inds_bad] = None

    return inds_bad

def locate_point(marker, contour):
    '''
    Locate the corresponding index of the marker on contour
    :return: int; index of marker
    '''
    index = 0
    mindist = np.inf
    for j in range(len(contour)):
        dist = (marker[0] - contour[j][0])**2 + (marker[1] - contour[j][1])**2
        if dist < mindist:
            mindist = dist
            index = j

    return index

def divide_contour(markers, contour):
    '''
    Divide contour based on the relative positions of markers
    :return: (list, list); two segments from the contour
    '''
    # Find the corresponding positions of the tracked points on contour
    ind_arm1 = locate_point((markers['armpit1_x'], markers['armpit1_y']), contour)
    ind_arm2 = locate_point((markers['armpit2_x'], markers['armpit2_y']), contour)
    ind_ped = locate_point((markers['peduncle_x'], markers['peduncle_y']), contour)

    # Divide the contour into two segments joint at ind_ped
    marker_hypostome = (markers['hypostome_x'], markers['hypostome_y'])
    if ind_arm1 < ind_arm2 < ind_ped:
        seg1 = [marker_hypostome] + contour[ind_arm2:ind_ped+1]
        seg2 = contour[ind_ped:] + contour[0:ind_arm1+1] + [marker_hypostome]
        seg2.reverse()
    elif ind_arm1 < ind_ped < ind_arm2:
        seg1 = contour[ind_ped:ind_arm2+1] + [marker_hypostome]
        seg1.reverse()
        seg2 = [marker_hypostome] + contour[ind_arm1:ind_ped+1]
    elif ind_arm2 < ind_arm1 < ind_ped:
        seg2 = [marker_hypostome] + contour[ind_arm1:ind_ped+1]
        seg1 = contour[ind_ped:] + contour[0:ind_arm2+1] + [marker_hypostome]
        seg1.reverse()
    elif ind_arm2 < ind_ped < ind_arm1:
        seg1 = [marker_hypostome] + contour[ind_arm2:ind_ped+1]
        seg2 = contour[ind_ped:ind_arm1+1] + [marker_hypostome]
        seg2.reverse()
    elif ind_ped < ind_arm1 < ind_arm2:
        seg2 = contour[ind_ped:ind_arm1+1] + [marker_hypostome]
        seg2.reverse()
        seg1 = [marker_hypostome] + contour[ind_arm2:] + contour[0:ind_ped+1]
    else:
        seg1 = contour[ind_ped:ind_arm2+1] + [marker_hypostome]
        seg1.reverse()
        seg2 = [marker_hypostome] + contour[ind_arm1:] + contour[0:ind_ped+1]

    # print(seg2)

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

# make sure to adjust scale (it is the ratio of the
# resolution of the video to the resolution used on icy to extract contour)
def pedunc_trace(file_icy, file_dlc, max_depth , video, scale = (2.0, 2.0)):
    '''
    :return: an array of the integration of fluorescence for each frame of the video
    :rtype: list
    '''
    # Load data
    contours, df, _ = load_data(file_icy, file_dlc)

    # Presettings
    lengths = []
    num_frames = len(contours)
    intensities_ = []
    cap = cv2.VideoCapture(video)

    firstf = True
    iframe = 0

    while True:

        if iframe >= num_frames:
            break
        # Capture frame-by-frame
        ret, frame = cap.read()

        markers = df.iloc[iframe]
        contour = contours[iframe]

        # Pass dropped frames
        if np.isnan(markers[0]):
            iframe += 1
            continue

        # Get midpoints
        seg1, seg2 = divide_contour(markers, contour)
        midpoints, sidepoints = find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [])

        # Sort midpoints based on the distances with the peduncle point
        ped_point = (markers['peduncle_x'], markers['peduncle_y'])
        hyp_point = (markers['hypostome_x'], markers['hypostome_y'])
        dist_with_ped = [length_segment([p, ped_point]) for p in midpoints]
        indexs = np.argsort(dist_with_ped)
        midpoints = np.array(midpoints)[indexs]
        np.append(midpoints, hyp_point)
        sidepoints = np.array(sidepoints)[indexs]
        # print(sidepoints.shape)
        # Append length of midline
        lengths.append(length_segment(midpoints))

        # Extract coordinates lists
        contour_x = [p[0] for p in contour]
        contour_y = [p[1] for p in contour]
        mid_x = [p[0] for p in midpoints]
        mid_y = [p[1] for p in midpoints]

        # obtain the sidepoints used to calculate the midpoint
        side1_x = [p[0][0] for p in sidepoints]
        side1_y = [p[0][1] for p in sidepoints]

        side2_x = [p[1][0] for p in sidepoints]
        side2_y = [p[1][1] for p in sidepoints]
        # reverse one side so the vertices are in cyclic order
        side2_x.reverse()
        side2_y.reverse()

        # extract the percentage of the midline
        # to be defined as peduncle
        polypoints_x = side1_x + side2_x + [ped_point[0]]
        polypoints_y = side1_y + side2_y + [ped_point[1]]
        length_points = len(polypoints_x)
        polypoints_x = polypoints_x[:int(length_points*0.2)] + polypoints_x[-int(length_points*0.2):]
        polypoints_y = polypoints_y[:int(length_points*0.2)] + polypoints_y[-int(length_points*0.2):]

        # append to a single list as tuples (x, y)
        polypoints = []
        # rescale the polygon vertices to the video size
        for x,y in zip(polypoints_x, polypoints_y):
            polypoints.append((x*scale[0],y*scale[1]))

        #  create polygon with Polygon function (easy to plot)
        # display the video with the polygon superimposed to
        # check that the peduncle is captured correnctly
        poly = Polygon(polypoints)
        plt.clf()
        plt.imshow(frame)
        plt.plot(*poly.exterior.xy)

        # plt.xlim(left = 0, right = 640)
        # plt.ylim(bottom = 0, top = 360)
        plt.pause(0.001)

        if ret:
            if firstf:
                size = len(frame) * len(frame[0])
                # get the resolution of the video
                r2,r1, _ignore = np.shape(frame)
                firstf = False

            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intensity = 0
            # convert the vertices list to numpy array
            polypoints = np.array( [polypoints], dtype = np.int32)
            # create an empty frame with same resolution as video
            cframe = np.zeros_like(frame)
            # fill the polygon in the empty frame to create a mask
            cv2.fillPoly( cframe, polypoints, 255 )
            # compare mask to video to extract sum of the intensity of
            # pixels inside the polygon
            pts = np.where(cframe == 255)
            intensity = np.sum(frame[pts])

            # normalize the intensity
            intensities_.append(intensity/size)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        print(iframe)
        iframe += 1


    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return intensities_



if __name__ == '__main__':

    # lengths = main('../data/hy78clip1_R2.xml', '../data/hy78clip1DeepCut_resnet50_clip1Mar24shuffle1_124000.csv', max_depth = 5, 'path to video file')
    lengths = pedunc_trace(sys.argv[1], sys.argv[2], max_depth = int(sys.argv[3]), video=sys.argv[6], scale=(int(sys.argv[4]), int(sys.argv[5])) )
    fig = plt.figure()
    plt.plot(lengths)
    plt.show()
    identifier = sys.argv[1].split('/')[-1].strip('.xml')
    try:
        fig.savefig('output/pedunc_fluo_' + identifier + '.png')
    except FileNotFoundError:
        os.makedirs('output/')
        fig.savefig('output/pedunc_fluo_' + identifier + '.png')
    df = pd.DataFrame(lengths)
    df.to_csv('output/pedunc_fluo_' + identifier + '.csv', index=False)
