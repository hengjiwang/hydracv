import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))

from cv2 import cv2
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from hydracv.scripts.find_midline import midline_new
from hydracv.scripts.peduncle_fluo import pedunc_fluo_contour
from hydracv.scripts.contour import contour_find

import hydracv.disp as disp
from analyzer import Analyzer

def _find_midline(self, name, scale=(1,1), max_depth=3, plot = False, display=False):
    """Obtain the midline length of the video with the given name.

    Obtain the midline length from the video with the given name, and
    set it as the  _midline_len. The midpoints as the _midpoints_pos (array of lists,
    each list contains tuples(x, y) equal to the number of midpoints,
    with each frame appended into the array as a seperate list)

    Args:
        name: A list of names of target videos. Default as None. If set as
            None, this function traces all added videos.
        scale: Tuple of ratios of resolution of the contour extracted and the video
        max_depth: Parameter deciding the number of midpoints to be produced (2^(max_depth+1)-1) midpoints generated
        plot:  Whether to plot the midline lengths of each frame? Default value is False.
        display: Whether to display conturs ? Default as False.
    """

    video = self._video(name)
    name, ext = name.split('.')

    contour_path = video._path + name + '.xml'
    if not os.path.isfile(contour_path):
        contour_path = video._path + name + '.pkl'
        if not os.path.isfile(contour_path):
            print('.pkl or .xml contour file not found')
            raise FileNotFoundError("Icy contour file not found: " + contour_path)

    dlc_path = video._path + name + '.csv'
    if not os.path.isfile(dlc_path):
        raise FileNotFoundError("DeepLabCut data file not found: " + dlc_path)
    lengths, dropped_frames, midpoints_all = midline_new.main(contour_path,
                dlc_path, max_depth, scale, display)

    video._midline_len = lengths
    video._midpoints_pos = midpoints_all

    if(plot):
        self.plot_midline_len(name+'.'+ext)

def find_midline(self, name=None, scale=None, max_depth=3, plot = False, display=False):
    """Obtain the midline lengths of the videos in name.

    Args:
        name: A list of names of target videos. Default as None. If set as
            None, this function traces all added videos.
        scale: Tuple of ratios of resolution of the contour extracted and the video
        max_depth: Parameter deciding the number of midpoints to be produced (2^(max_depth+1)-1) midpoints generated
        plot:  Whether to plot the trace? Default value is False.
        display: Whether to display conturs ? Default as False.
    Raises:
        TypeError: An error occurred if name is neither a list nor a string.
    """
    if name is None:
        name = self._video_names()

    if scale is None:
        scale = (1,1)

    if type(name) != list:
        if type(name) == str:
            self._find_midline(name, scale, max_depth, plot, display)
        else:
            raise(TypeError("name can only be list or string!"))
    else:
        for nm in name:
            self._find_midline(nm, scale, max_depth, plot, display)
