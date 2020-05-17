import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import xml.etree.ElementTree as ET
from cv2 import cv2
from tqdm import tqdm
import os

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

def cut_contour(contours, markers):
    "Cut the tentacles of the contour with the markers"
    for iframe in range(len(contours)):
        contour = contours[iframe]
        marker_mat = markers[iframe]
        marker = defaultdict(tuple)
        marker['hypostome'] = (marker_mat[0], marker_mat[1])
        marker['armpit1'] = (marker_mat[3], marker_mat[4])
        marker['armpit2'] = (marker_mat[6], marker_mat[7])
        marker['peduncle'] = (marker_mat[9], marker_mat[10])
        ind_ped = locate_point(marker['peduncle'], contour)
        contour = contour[ind_ped:] + contour[:ind_ped]
        ind_ped = 0
        ind_arp1 = locate_point(marker['armpit1'], contour)
        ind_arp2 = locate_point(marker['armpit2'], contour)
        ind_hyp = locate_point(marker['hypostome'], contour)
        ind_arp1, ind_arp2 = min(ind_arp1, ind_arp2), max(ind_arp1, ind_arp2)
        contour_half_1 = contour[:ind_arp1]
        contour_half_2 = contour[ind_arp2:]
        contour = contour_half_1 + [contour[ind_hyp]] + contour_half_2
        contours[iframe] = contour
    return contours
        


def fill_contour(contour, ny, nx):
    "Fill the contour with white points"

    img = np.zeros((ny, nx, 3), dtype=np.uint8)
    cv2.fillPoly(img, [contour, ], (255, 255, 255))

    return img

def fill(file_contour, file_marker, outname):
    "Fill contours"

    # Load contours
    contours = load_contour(file_contour)
    markers = load_marker(file_marker).values
    contours = cut_contour(contours, markers)

    # Get frame size
    cap = cv2.VideoCapture("/home/hengji/Documents/hydrafiles/videos/EGCaMP/Control-EGCaMP_exp1_a1_30x10fps.avi")
    _, frame = cap.read()
    ny, nx, _ = frame.shape
    cv2.destroyAllWindows()

    DPI = 100

    # plt.figure()
    plt.figure(figsize=(nx/DPI, ny/DPI), dpi=DPI)
    for iframe in tqdm(range(len(contours))):
        # Reformat to numpy array
        contour = np.array(contours[iframe], dtype=np.int32)

        plt.clf()
        img = fill_contour(contour, ny, nx)
        plt.imshow(img, cmap='binary', aspect='equal')
        plt.axis('off')

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        plt.xlim(0, nx)
        plt.ylim(0, ny)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        
        plt.savefig('./results/'+ outname +'/frames/img' + str(iframe) + '.jpg', dpi=DPI) # , orientation='landscape')
        plt.pause(0.001)

def save_video(outname):
    "Count the number of frames"

    # Directory name
    DIR = './results/'+ outname +'/frames/'

    # Number of frames
    nframes = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

    # Frame size
    frame_size = cv2.imread(DIR + 'img0.jpg').shape
    nx, ny = frame_size[0], frame_size[1]

    # Convert the frames into video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('./results/' + outname + '/video/movie.avi', fourcc, 40, (ny, nx))

    for iframe in tqdm(range(nframes)):
        frame = cv2.imread(DIR + 'img' + str(iframe) + '.jpg')
        videoWriter.write(frame)
    videoWriter.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    fill("../data/contour/Control-EGCaMP_exp1_a1_30x10fps_5%.xml", 
         "../data/marker/Control-EGCaMP_exp1_a1_30x10fps_5%_001DLC_resnet50_EGCaMPFeb14shuffle1_576000.csv", 
         "Control-EGCaMP_exp1_a1_30x10fps")
    save_video("Control-EGCaMP_exp1_a1_30x10fps")
