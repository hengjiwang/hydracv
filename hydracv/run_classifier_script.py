import NN_classifier
import sys

"""
This is a short script used for running NN_classifier.py.
Specify the arguments you'd like to pass into the classifier here.
"""

VIDEO_NAME = "71420_ngcamp_hy6_peduncle72hr" # TO MODIFY
BODY_PART = "/Peduncle/" # TO MODIFY
SAVE_DIR = '/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6' + BODY_PART #TO MODIFY

FILE_FLUO = SAVE_DIR + 'total_fluo_' + VIDEO_NAME + '.csv'

FILE_LENGTH = SAVE_DIR + "lengths_" + VIDEO_NAME + ".csv"

FILE_VIDEO = SAVE_DIR + VIDEO_NAME + ".avi"

FILE_FLUO_PEAKS = SAVE_DIR + "fluo_peaks_" + VIDEO_NAME + ".csv"

FRAMES_DIR = SAVE_DIR + VIDEO_NAME + "/"

SAVE_FILE = SAVE_DIR + "classified_peaks_" + VIDEO_NAME + ".csv"

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    args = Namespace(
        video_path = FILE_VIDEO,
        fluo_path = FILE_FLUO,
        midlens_path = FILE_LENGTH,
        peaks_path = FILE_FLUO_PEAKS,
        frames_dir = FRAMES_DIR,
        save_file = SAVE_FILE,
        start_index = 0,
        extract = True
    )
    NN_classifier.initialize_GUI(args)

if __name__ == '__main__':
    sys.exit(main())