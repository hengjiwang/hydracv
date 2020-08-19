import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
import numpy.linalg as LA
from tqdm import tqdm

def load_midpoints(filename):
    "Load midline data"
    midpoints = []
    data = pd.read_csv(filename).values
    nframes = len(data)
    npoints = len(data[0]) // 2

    for i in range(nframes):
        midpoints.append([])
        for j in range(npoints):
            midpoints[-1].append((data[i][2*j], data[i][2*j+1]))

    return np.array(midpoints)

def curvature(x, y):
    "Returns the curvature of three points, referring to https://github.com/Pjer-zhang/PJCurvature/blob/master/src/py/PJcurvature.py"
    t_a = LA.norm([x[1]-x[0], y[1]-y[0]])
    t_b = LA.norm([x[2]-x[1], y[2]-y[1]])
    
    M = np.array([
        [1, -t_a, t_a**2],
        [1, 0, 0],
        [1, t_b, t_b**2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    kappa = 2*(a[2]*b[1]-b[2]*a[1])/(a[1]**2.+b[1]**2.)**(1.5)
    return kappa, [b[1], -a[1]]/np.sqrt(a[1]**2.+b[1]**2.)

def curvatures_of_line(x_arr, y_arr):
    "Returns the curvatures of a curve"
    ka = []
    no = []

    for j in range(1, len(x_arr)-2):
        x = x_arr[j:j+3]
        y = y_arr[j:j+3]
        kappa, norm = curvature(x, y)
        ka.append(kappa)
        no.append(norm)

    return np.array(ka), np.array(no)


def fit(filename, display=True):
    "Do fitting on midpoints data in filename"
    midpoints = load_midpoints(filename)
    splines = []
    curvatures = []
    norms = []
    maxcurvs = []
    meancurvs = []

    if display:
        fig = plt.figure(figsize=(6, 6))


    for iframe in tqdm(range(len(midpoints))):

        line = midpoints[iframe]

        x = line[:, 0]
        y = line[:, 1]
        
        # Delete the duplicates
        okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
        x = x[okay]
        y = y[okay]
        
        # Fit the splines
        tck, u = interpolate.splprep([x, y])
        xnew, ynew = interpolate.splev(np.linspace(0, 1, 200), tck)
        splines.append([xnew, ynew])

        # Calculate the curvatures and norms
        ka, no = curvatures_of_line(xnew, ynew)
        curvatures.append(ka)
        norms.append(no)

        maxcur = np.max(np.abs(ka))
        meancur = np.mean(np.abs(ka))

        maxcurvs.append(maxcur)
        meancurvs.append(meancur)

        # Plot
        if display:
            plt.clf()
            plt.plot(x, y, 'r.')
            plt.plot(xnew, ynew, 'b')
            gap = 2
            plt.quiver(xnew[1:len(xnew)-2:gap], ynew[1:len(ynew)-2:gap], (ka*no[:, 0])[::gap], (ka*no[:, 1])[::gap], width=0.002, headwidth=3, headlength=5, scale=1.5)
            plt.text(280, 380, "Max Abs. Curv. = " + str(round(maxcur, 4)), color='k', fontsize=10)
            plt.text(280, 360, "Avg Abs. Curv. = " + str(round(meancur, 4)), color='k', fontsize=10)
            plt.xlim([100, 400])
            plt.ylim([100, 400])
            plt.savefig('./results/'+ 'Control-EGCaMP_exp1_a1_30x10fps' +'/frames/img' + str(iframe) + '.jpg', dpi=200) # , orientation='landscape')
            plt.pause(0.00001)

    # Return results
    return (np.array(maxcurvs), np.array(meancurvs))

if __name__ == "__main__":
    maxcurvs, meancurvs = fit("./results/Control-EGCaMP_exp1_a1_30x10fps/midpoints/midpoints_bisection_corrected.csv", display=True)

    # df = pd.DataFrame(splines)
    # df.to_csv('./results/Control-EGCaMP_exp1_a1_30x10fps/splines/splines.csv')
    df = pd.DataFrame(maxcurvs)
    df.to_csv('./results/Control-EGCaMP_exp1_a1_30x10fps/splines/maxcurvs.csv')
    df = pd.DataFrame(meancurvs)
    df.to_csv('./results/Control-EGCaMP_exp1_a1_30x10fps/splines/meancurvs.csv')
