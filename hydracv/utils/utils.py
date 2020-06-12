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