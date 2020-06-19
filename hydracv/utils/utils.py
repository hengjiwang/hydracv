import numpy as np

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