import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xml.etree.ElementTree as ET
import sys, csv, os, re, cv2
sys.path.append('/home/shashank/Downloads/hydra_data_analysis/find_midline')
from midline_ped import *
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d



def find_inter(tr_pt, midpoints, seg1, seg2):

    if(length_segment([tr_pt, seg1[0]]) > length_segment( [tr_pt, seg1[-1]] )):
        seg1.reverse()

    if(length_segment([tr_pt, seg2[0]]) > length_segment( [tr_pt, seg2[-1]] )):
        seg2.reverse()

    print(type(seg1))
    print(type(seg2))

    md1 = np.inf
    md2 = np.inf
    for pt in midpoints:
        d = length_segment([pt, tr_pt])
        if(d < md1 and d < md2):
            md1 = d
            cp1 = pt
        else:
            if(d < md2):
                md2 = d
                cp2 = pt

    sl = slope(cp1, cp2)

    psl = -(1/sl)
    # print(psl)

    npt = np.array([0,0], 'float')

    npt[0] = (tr_pt[0] + sl*sl*cp1[0] + sl*tr_pt[1] - sl*cp1[1] ) / (sl*sl + 1)
    npt[1] = sl*(npt[0] - cp1[0]) + cp1[1]

    # print(npt)
    s = np.sign( (seg1[0][0]-npt[0])*(tr_pt[1]-npt[1]) -  (seg1[0][1]-npt[1])*(tr_pt[0]-npt[0]) )

    cpt1seg1 = seg1[0]
    # print(s)
    for pt in seg1:
        cs = np.sign( (pt[0]-npt[0])*(tr_pt[1]-npt[1]) -  (pt[1]-npt[1])*(tr_pt[0]-npt[0]) )
        if( cs != s ):
            cpt2seg1 = pt
            # print(csl, psl)
            break
        cpt1seg1 = pt


    s = np.sign( (seg2[0][0]-npt[0])*(tr_pt[1]-npt[1]) -  (seg2[0][1]-npt[1])*(tr_pt[0]-npt[0]) )

    cpt1seg2 = seg2[0]

    for pt in seg2:
        cs = np.sign( (pt[0]-npt[0])*(tr_pt[1]-npt[1]) -  (pt[1]-npt[1])*(tr_pt[0]-npt[0]) )
        if( cs != s ):
            cpt2seg2 = pt
            break
        cpt1seg2 = pt

    ncpt1 = np.array([0,0], 'float')
    ncpt2 = np.array([0,0], 'float')

    sl = slope(cpt1seg1, cpt2seg1)
    # print(npt)
    ncpt1[1] = ( sl*psl*(cpt1seg1[0] - npt[0]) - psl*cpt1seg1[1] + sl*npt[1] )/(sl - psl)
    ncpt1[0] = cpt1seg1[0] - (cpt1seg1[1] - ncpt1[1])/sl

    sl = slope(cpt1seg2, cpt2seg2)
    ncpt2[1] = (sl*psl*(cpt1seg2[0] - npt[0]) - psl*cpt1seg2[1] + sl*npt[1])/(sl - psl)
    ncpt2[0] = cpt1seg2[0] - (cpt1seg2[1] - ncpt2[1])/sl


    return ncpt1, ncpt2


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
        plt.scatter(seg1_points[j][0], seg1_points[j][1], color="orange")
        plt.scatter(seg2_points[j][0], seg2_points[j][1], color="orange")
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

    cname = df.columns.values
    for nm in cname:
        if( re.search('likelihood', nm) or re.search('bodyparts', nm)):
          continue
        x = df[nm].values
        # print(x)
        # print(np.shape(np.reshape(x, -1)))
        xfit = gaussian_filter1d(x, 20)
        # print(len(xfit))
        df = df.assign(nm = xfit)

    # Presettings
    lengths = []
    num_frames = len(contours)
    plt.figure(figsize=(20,20))
    cap = cv2.VideoCapture(videopath)

    contour_points = {}

    # Loop over all frames
    for iframe in tqdm(range(num_frames)):

        markers = df.iloc[iframe]
        contour = contours[iframe]
        ret, frame = cap.read()
        # contour = interpolate(contour, interpolate_contour_num)

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
        # midpoints = interpolate(midpoints, interpolate_midline_num)

        # Find mapped points of spots on midline
        map_points = []
        seg1_points = []
        seg2_points = []

        spn = 0
        for spot in spots:
            c1, c2 = find_inter(spot, midpoints, seg1, seg2)
            # print(c1,c2)
            seg1_points.append(c1)
            seg2_points.append(c2)
            try:
                contour_points['seg1_x_'+str(spn)].append(c1[0])
                contour_points['seg1_y_'+str(spn)].append(c1[1])
                contour_points['seg2_x_'+str(spn)].append(c2[0])
                contour_points['seg2_y_'+str(spn)].append(c2[1])
            except:
                contour_points['seg1_x_'+str(spn)] = []
                contour_points['seg1_y_'+str(spn)] = []
                contour_points['seg2_x_'+str(spn)] = []
                contour_points['seg2_y_'+str(spn)] = []
                contour_points['seg1_x_'+str(spn)].append(c1[0])
                contour_points['seg1_y_'+str(spn)].append(c1[1])
                contour_points['seg2_x_'+str(spn)].append(c2[0])
                contour_points['seg2_y_'+str(spn)].append(c2[1])
            spn += 1



        # for spot in spots:
        #
        #     map_index = locate_point(spot, midpoints)
        #     map_point = midpoints[map_index]
        #     map_points.append(map_point)
        #     sl = slope(spot, map_point)
        #     min_p1 = None
        #     min_p2 = None
        #     min_diff1 = np.Infinity
        #     min_diff2 = np.Infinity
        #     for p in seg1:
        #         sl2 = slope(p, map_point)
        #         diff = np.abs(sl - sl2)
        #         if diff < min_diff1 and length_segment([p, map_point]) < 60:
        #             min_diff1 = diff
        #             min_p1 = p
        #     seg1_points.append(min_p1)
        #     for p in seg2:
        #         sl2 = slope(p, map_point)
        #         diff = np.abs(sl - sl2)
        #         if diff < min_diff2 and length_segment([p, map_point]) < 60:
        #             min_diff2 = diff
        #             min_p2 = p
        #     seg2_points.append(min_p2)
            # print(min_p1, min_p2, spot)

        # Draw
        # input('-------------------------------')
        draw(contour, midpoints, hyp_point, ped_point, spots, map_points, seg1_points, seg2_points, frame, scale, project_name, iframe)

    print(contour_points.keys())
    input('Press Enter')

    df_cpt = pd.DataFrame.from_dict(contour_points, orient="index")
    df_cpt.to_csv('/home/shashank/Downloads/save/animations/cnt_pt.csv', encoding='utf-8', index=False)
    return lengths

if __name__ == "__main__":
    df = pd.read_json('/home/shashank/Downloads/hydra_data_analysis/segment_body/config.json')
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
