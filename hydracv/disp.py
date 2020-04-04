import numpy as np

def add_fluorescence(ax, time_axis, fluo):
    """Add a plot of fluo to ax"""
    ax.plot(time_axis, fluo, 'g-')
    ax.set_xlabel("time(s)")
    ax.set_ylabel("fluorescence(a.u.)")

def add_peaks(ax, pks, fluo, fps):
    """Add pks on fluo"""
    ax.plot([x/fps for x in pks], np.array(fluo)[pks], 'xb')

def add_spike_trains(ax, spk_trains, fps):
    """Add a scatter plot of spk_trains to ax"""

    if type(fps) == int:
        for j in range(len(spk_trains)):
            ax.vlines(np.array(spk_trains[j])/fps, j, j+1)

    elif type(fps) == list:
        for j in range(len(spk_trains)):
            ax.vlines(np.array(spk_trains[j])/fps[j], j, j+1)

    else:
        raise TypeError("fps must be list or int!")
    
    ax.set_xlabel("time(s)")
    ax.set_ylabel("cluster #")

def add_isi_curves(ax, isi):
    """Add ISI curves to ax"""
    for trail in isi:
        ax.plot(trail)
        ax.set_xlabel("Interval #")
        ax.set_ylabel("ISI(s)")

def add_boxplot(ax, isi_t):
    """Add boxplot for isi_t (transposed ISI)"""
    ax.boxplot(isi_t)
    ax.set_xlabel("Interval #")
    ax.set_ylabel("ISI(s)")