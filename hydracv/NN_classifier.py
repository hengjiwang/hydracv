from typing import Container
import wx
import sys
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from utils import utils
import cv2
import scipy
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

class Classifier_GUI(wx.Frame):  
    def __init__(self):
        self.video_path = "/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6/71420_ngcamp_hy6_prebisect.avi"
        self.fluo_path = "/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6/total_fluo_71420_ngcamp_hy6_prebisect.csv"
        self.midlens_path = "/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6/lengths_71420_ngcamp_hy6_prebisect.csv"
        self.peaks_path = "/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6/fluo_peaks_71420_ngcamp_hy6_prebisect.csv"
        self.save_dir = "/Users/katyabrooun/Fairhall_Lab/hydra/bisection-data-analysis/exp6/71420_ngcamp_hy6_prebisect/"
        self.generate_frames = False

        self.peaks = pd.read_csv(self.peaks_path).values.reshape(-1)
        self.curr_pk_ind = 0

        self.pk_network = []

        # pre-process video
        if self.generate_frames:
            self.convert_video_to_frames()

        # Create GUI window
        super().__init__(parent=None, title='Hydra Neural Network Classifier', size=(1000, 400))
        
        # Create panel within window that contains rest of GUI components
        container_panel = ContainerPanel(self)   
        container_sizer = wx.BoxSizer(wx.VERTICAL)

        # Informational text
        vid_name = wx.StaticText(container_panel)
        vid_name.SetLabel("Currently analyzing: " + self.video_path.split('/')[-1])
        container_sizer.Add(vid_name)

        self.frame_num = wx.StaticText(container_panel)
        self.frame_num.SetLabel("Current frame number: " + str(self.peaks[self.curr_pk_ind]))
        container_sizer.Add(self.frame_num)

        currFrame = self.peaks[self.curr_pk_ind]

        # Video frames panel
        self.video_frames = VideoFramesPanel(parent=container_panel, currFrame=currFrame)
        container_sizer.Add(self.video_frames, 0, wx.EXPAND)
        
        # Matplot panel
        self.fluo_len_plot = MatplotPanel(parent=container_panel, currFrame=currFrame)
        container_sizer.Add(self.fluo_len_plot, 0, wx.EXPAND)

        button_box = wx.BoxSizer(wx.HORIZONTAL)

        # Classification buttons
        cb_btn = wx.Button(container_panel, label='CB')
        cb_btn.name = "CB"
        cb_btn.Bind(wx.EVT_BUTTON, self.on_press)

        rp_btn = wx.Button(container_panel, label='RP')
        rp_btn.name = "RP"
        rp_btn.Bind(wx.EVT_BUTTON, self.on_press)

        button_box.Add(cb_btn, 0, wx.EXPAND)
        button_box.Add(rp_btn, 2)

        container_sizer.Add(button_box, 0, wx.ALL | wx.CENTER, 5) 

        container_panel.SetSizer(container_sizer)        
        self.Show()

    def on_press(self, event):
        btn_name = event.GetEventObject().name
        self.pk_network.append(btn_name)
        self.curr_pk_ind += 1

        currFrame = self.peaks[self.curr_pk_ind]
        # update current frame information
        self.frame_num.SetLabel("Current frame number: " + str(currFrame))
        # move frame marker to next peak in plot
        self.fluo_len_plot.draw_graph(currFrame)
        # update current video frame(s) being displayed
        self.video_frames.update_frames(currFrame)
    
    def convert_video_to_frames(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(self.save_dir + "frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
    
class ContainerPanel(wx.Panel):
    def __init__(self, parent):
        self.video_path = parent.video_path
        self.fluo_path = parent.fluo_path
        self.midlens_path = parent.midlens_path
        self.save_dir = parent.save_dir

        wx.Panel.__init__(self, parent)

class VideoFramesPanel(wx.Panel):
    def __init__(self, parent, currFrame):
        self.save_dir = parent.save_dir
        self.currFrame = currFrame
        
        # Gathering info to determine panel size
        image_path = self.save_dir + "frame%d.jpg" % currFrame
        image = wx.Image(image_path, wx.BITMAP_TYPE_JPEG)
        size = image.GetSize()

        panel_width = int(size[0]*3/2)
        panel_height = int(size[1]/2)

        wx.Panel.__init__(self, parent,size=(panel_width,panel_height))

        # render initial image
        self.update_frames(currFrame)

    def update_frames(self, currFrame):
        image_path = self.save_dir + "frame%d.jpg" % currFrame
        image = wx.Image(image_path, wx.BITMAP_TYPE_JPEG)
        size = image.GetSize()
        image = image.Scale(int(size[0]/2), int(size[1]/2), wx.IMAGE_QUALITY_HIGH)
        temp = image.ConvertToBitmap()
        wx.StaticBitmap(self, -1, temp, (5, 5))
        
class MatplotPanel(wx.Panel):
    def __init__(self, parent, currFrame):

        wx.Panel.__init__(self, parent)
        self.currFrame = currFrame

        # Smooth and normalize midlens and fluorescence timeseries
        midlens = pd.read_csv(parent.midlens_path).values.reshape(-1)
        midlens = scipy.signal.savgol_filter(midlens, 15, 3)
        
        midlens = utils.normalize(midlens)

        fluo = pd.read_csv(parent.fluo_path).values.reshape(-1)
        fluo = scipy.signal.savgol_filter(fluo, 15, 3)
        fluo = utils.remove_trend_fluo(fluo, display=False)
        fluo = utils.normalize(fluo)

        # Make sure series are same length and define x-axis
        min_length = min(len(fluo), len(midlens))
        self.fluo = fluo[:min_length]
        self.midlens = midlens[:min_length]
        self.x_frame = np.arange(0, min_length,1)

        self.draw_graph(self.currFrame)
        
    def draw_graph(self, currFrame):
        self.figure = Figure(figsize=(10,2))
        self.ax = self.figure.add_subplot(111)
        utils.plot_fluo_len(self.ax,self.fluo,self.midlens,self.x_frame)
        # Plot position of current frame
        self.ax.axvline(currFrame, color='r')
        self.canvas = FigureCanvas(self, -1, self.figure)

def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help="Path for .avi Hydra movie you'd like to analyze.")
    args = parser.parse_args(args=argv)

    app = wx.App()
    gui = Classifier_GUI()
    app.MainLoop()

if __name__ == '__main__':
    sys.exit(main())

