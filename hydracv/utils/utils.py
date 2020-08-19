import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.ndimage

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

    return list(np.array(clusters)[indices_to_keep])

def reformat_periods(periods, unit='sec'):
    "Reformat periods extracted from manual labeling data"
    for j in range(len(periods)):
        
        time_str_pair = periods[j].split('-')
        
        if len(time_str_pair) < 2:
            continue
        
        periods[j] = (reformat_time(time_str_pair[0], unit), reformat_time(time_str_pair[1], unit))
        
    return periods

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
    