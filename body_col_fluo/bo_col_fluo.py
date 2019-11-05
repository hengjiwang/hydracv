import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os
from shapely.geometry import Polygon, Point, MultiPoint
# from hydra_data_analysis.find_midline.midline import *

sys.path.append(os.path.join( os.path.dirname(os.path.realpath(__file__)), '../find_midline/'))

from midline import *
import cv2

def bo_col_trace(file_icy, file_dlc, max_depth , video, scale = (2.0, 2.0)):
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

        if firstf and ret:
            size = len(frame) * len(frame[0])
            r2,r1, _ignore = np.shape(frame)
            print(r2,r1)
            firstf = False

        markers = df.iloc[iframe]
        contour = contours[iframe]

        # Pass dropped frames
        if np.isnan(markers[0]):
            # print('drop', iframe)
            iframe += 1
            continue

        # Get midpoints
        seg1, seg2 = divide_contour(markers, contour)
        midpoints, sidepoints = find_midline(seg1, seg2, max_depth, midpoints = [], sidepoints = [])

        # seg1 = np.array(seg1)
        # seg2 = np.array(seg2)
        # Sort midpoints based on the distances with the peduncle point
        ped_point = (markers['peduncle_x'], markers['peduncle_y'])
        hyp_point = ((markers['armpit1_x']+markers['armpit2_x'])/2, (markers['armpit1_y']+markers['armpit2_y'])/2)
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

        side1_x = [p[0][0] for p in sidepoints]
        side1_y = [p[0][1] for p in sidepoints]

        side2_x = [p[1][0] for p in sidepoints]
        side2_y = [p[1][1] for p in sidepoints]

        # get_seg_point()

        side2_x.reverse()
        side2_y.reverse()

        # print(ped_point[0] in side2_x)

        polypoints_x = side1_x + side2_x + [ped_point[0]]
        polypoints_y = side1_y + side2_y + [ped_point[1]]
        length_points = len(polypoints_x)
        # print(length_points)
        # polypoints_x = polypoints_x[int(length_points*0.2) : len(side1_x)] + polypoints_x[-len(side2_x)-1 : -int(length_points*0.2)-1]
        # polypoints_y = polypoints_y[int(length_points*0.2) : len(side1_y)] + polypoints_y[-len(side2_y)-1 : -int(length_points*0.2)-1]

        chk = np.where( np.array(seg1)[:,0] == polypoints_x[int(length_points*0.2)] )
        chk_ = np.where( np.array(seg1)[:,0] == polypoints_x[len(side1_x)-1] )

        chk1 = np.where( np.array(seg2)[:,0] == polypoints_x[-int(length_points*0.2)-2] )
        chk1_ = np.where( np.array(seg2)[:,0] == polypoints_x[-len(side2_x)-1] )

        seg1ind = int(chk[0][0])
        seg1inde = int(chk_[0][0])
        seg2ind = int(chk1[0][0])
        seg2inde = int(chk1_[0][0])

        # # print(type(np.array(seg1)[seg1ind:,0]))
        polypoints_x = np.concatenate( ( np.array(seg1)[seg1inde:seg1ind+1, 0] , np.flip(np.array(seg2)[seg2inde:seg2ind+1, 0] ) ), axis = None )
        polypoints_y = np.concatenate( ( np.array(seg1)[seg1inde:seg1ind+1, 1] , np.flip(np.array(seg2)[seg2inde:seg2ind+1, 1] ) ), axis = None )

        polypoints = []
        for x,y in zip(polypoints_x, polypoints_y):
            polypoints.append( (x*scale[0], y*scale[1]) )

        # print(len(polypoints))
        # poly = Polygon(polypoints)
        plt.clf()
        rframe = cv2.resize(frame, ( int(r1/scale[0]) , int(r2/scale[1]) ))
        plt.imshow(rframe)


        # plt.plot([polypoints[3][0],polypoints[4][0]],[polypoints[3][1],polypoints[4][1]], 'b-')
        # plt.plot(*poly.exterior.xy, )
        plt.fill(polypoints_x, polypoints_y, alpha = 0.5)

        plt.scatter(np.array(seg1)[:,0], np.array(seg1)[:,1], color = '', marker = 'o', edgecolors= 'g')
        plt.scatter(np.array(seg2)[:,0], np.array(seg2)[:,1], color = '', marker = 'o', edgecolors= 'g')
        for i in range(len(side1_x)):
            plt.plot([side1_x[i], side2_x[-i-1]], [side1_y[i], side2_y[-i-1]], color='burlywood')
            # (side2_x[i], side2_y[i], color='maroon', marker ='o')
        plt.plot(mid_x, mid_y, 'r.-')
        plt.plot([markers['armpit1_x'],hyp_point[0]], [markers['armpit1_y'],hyp_point[1]], 'go-')
        plt.plot([markers['armpit2_x'],hyp_point[0]], [markers['armpit2_y'],hyp_point[1]], 'go-')
        plt.plot([hyp_point[0], mid_x[-1]], [hyp_point[1], mid_y[-1]], 'r-')
        plt.plot([ped_point[0], mid_x[0]], [ped_point[1], mid_y[0]], 'r-')
        plt.plot(markers['armpit1_x'], markers['armpit1_y'], 'bo')
        plt.plot(markers['armpit2_x'], markers['armpit2_y'], 'bo')
        plt.plot(hyp_point[0],hyp_point[1], color='orange', marker='o')
        plt.plot(ped_point[0],ped_point[1], color= 'purple', marker = 'o')
        plt.xlim(left = 0, right = r1/scale[0])
        plt.ylim(bottom = 0, top = r2/scale[1])

        # plt.scatter(np.array(seg2)[70:,0], np.array(seg2)[70:,1], marker = 'o', color='midnightblue')
        # plt.scatter(np.array(seg1)[35:,0], np.array(seg1)[35:,1], marker = 'o', color='midnightblue')
        # input('press enter')
        plt.pause(0.001)
        # plt.savefig('/home/shashank/Downloads/movfig_cont/fig'+str(iframe) +'.png')


        if ret:


            # Our operations on the frame come here
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intensity = 0
            polypoints = np.array( [polypoints], dtype = np.int32)
            # print(polypoints)
            # polypoints*=2
            # print(polypoints)
            cframe = np.zeros_like(frame)
            cv2.fillPoly( cframe, polypoints, 1 )
            cframe = cframe * frame
            intensity = np.sum(cframe)

            intensities_.append(intensity)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # print(iframe)
        iframe += 1


    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return intensities_



if __name__ == '__main__':

    # lengths = main('../data/hy78clip1_R2.xml', '../data/hy78clip1DeepCut_resnet50_clip1Mar24shuffle1_124000.csv', max_depth = 5)
    lengths = bo_col_trace( sys.argv[1], sys.argv[2], max_depth = int(sys.argv[3]), video=sys.argv[4], scale= ( int(sys.argv[5]), int(sys.argv[6]) ) )
    fig = plt.figure()
    plt.plot(lengths)
    plt.show()
    identifier = sys.argv[1].split('/')[-1].strip('.xml')
    try:
        fig.savefig('output_con/body_col_fluo_' + identifier + '.png')
    except FileNotFoundError:
        os.makedirs('output_con/')
        fig.savefig('output_con/body_col_fluo_' + identifier + '.png')
    df = pd.DataFrame(lengths)
    df.to_csv('output_con/body_col_fluo_' + identifier + '.csv', index=False)
