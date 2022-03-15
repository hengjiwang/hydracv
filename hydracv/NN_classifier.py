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

"""
A GUI for manually classifying Hydra neural activity into "CB", "RP", or "Neither."

Required:
* GCaMP video of behaving Hydra
* Midline lengths for each video frame
* Total fluorescence intensity for each frame of video
* Indices of peaks in fluorescence activity

You can run this script from the command line using:

`pythonw NN_classifier.py [arg1] [arg2] ... [arg_n]`

Since there are quite a few file path arguments, you can use `run_classifier_script.py`
to more conveniently define these arguments. Then you can simply run:

`pythonw run_classifier_script.py`

"""

IMG_SCALE = 0.75

class Classifier_GUI(wx.Frame):
    def __init__(self, args):
        self.video_path = args.video_path
        self.fluo_path = args.fluo_path
        self.midlens_path = args.midlens_path
        self.peaks_path = args.peaks_path
        self.frames_dir = args.frames_dir
        self.save_file = args.save_file
        self.curr_pk_ind = args.start_index
        self.generate_frames = args.extract

        self.peaks = pd.read_csv(self.peaks_path).values.reshape(-1)

        # 0 = neither, 1 = CB, 2 = RP
        self.pk_network = np.zeros(len(self.peaks))

        default_font = wx.Font(18,wx.MODERN, wx.NORMAL, wx.NORMAL)

        # pre-process video
        if self.generate_frames:
            self.convert_video_to_frames()

        # Create GUI window
        super().__init__(parent=None, title='Hydra Neural Network Classifier', size=(1200, 850))

        # Create panel within window that contains rest of GUI components
        container_panel = ContainerPanel(self)   
        container_sizer = wx.BoxSizer(wx.VERTICAL)

        # Informational text
        self.info = wx.StaticText(container_panel)
        self.info.SetLabel(self.getInfoText(self.peaks[self.curr_pk_ind]))
        self.info.SetFont(default_font)
        self.info.SetBackgroundColour(wx.Colour(255,255,255))
        container_sizer.Add(self.info, 0, wx.CENTER|wx.TOP, border=10)

        self.currFrame = self.peaks[self.curr_pk_ind]

        top_button_box = wx.BoxSizer(wx.HORIZONTAL)

        # Option to hide video frames to improve perf
        toggle_video_frames = wx.ToggleButton(container_panel,-1,"Hide video frames", size=(150,50))
        toggle_video_frames.Bind(wx.EVT_TOGGLEBUTTON,self.OnToggle) 
        top_button_box.Add(toggle_video_frames,0)
       
        # Add an undo button
        undo_btn = wx.Button(container_panel,-1,"Undo", size=(150,50))
        undo_btn.Bind(wx.EVT_BUTTON, self.Undo)
        top_button_box.Add(undo_btn, 0)

        # Add save progess button
        save_btn = wx.Button(container_panel,-1,"Save progress",size=(150,50))
        save_btn.Bind(wx.EVT_BUTTON, self.SaveProgress)
        top_button_box.Add(save_btn, 0)

        container_sizer.Add(top_button_box, 0, wx.CENTER) 

        # Video frames panel
        self.video_frames = VideoFramesPanel(parent=container_panel, currFrame=self.currFrame)
        container_sizer.Add(self.video_frames, 0, wx.CENTER|wx.ALL, border=5)
        
        # Matplot panel
        self.fluo_len_plot = MatplotPanel(parent=container_panel, currFrame=self.currFrame)
        container_sizer.Add(self.fluo_len_plot, 0, wx.CENTER|wx.ALL, border=5)

        bottom_button_box = wx.BoxSizer(wx.HORIZONTAL)

        # Classification buttons
        cb_btn = wx.Button(container_panel, label='CB', size=(90,50))
        cb_btn.SetFont(default_font)
        cb_btn.name = "CB"
        cb_btn.Bind(wx.EVT_BUTTON, self.on_press)

        neither_btn = wx.Button(container_panel, label='Neither', size=(90,50))
        neither_btn.SetFont(default_font)
        neither_btn.name = "Neither"
        neither_btn.Bind(wx.EVT_BUTTON, self.on_press)

        rp_btn = wx.Button(container_panel, label='RP', size=(90,50))
        rp_btn.SetFont(default_font)
        rp_btn.name = "RP"
        rp_btn.Bind(wx.EVT_BUTTON, self.on_press)

        bottom_button_box.Add(cb_btn, 0)
        bottom_button_box.Add(neither_btn, 0)
        bottom_button_box.Add(rp_btn, 0)

        container_sizer.Add(bottom_button_box, 0, wx.ALL | wx.CENTER, 10) 

        container_panel.SetSizer(container_sizer)        
        self.Show()
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def OnToggle(self, event):
        state = event.GetEventObject().GetValue() 
        if state == True: 
            self.video_frames.Hide()
            self.Refresh()
            event.GetEventObject().SetLabel("Display video frames") 
        else: 
            self.video_frames.Show()
            event.GetEventObject().SetLabel("Hide video frames")
    
    def Undo(self, event):
        if self.curr_pk_ind > 0:
            self.curr_pk_ind -= 1
            # change previous element to "Neither"
            self.pk_network[self.curr_pk_ind] = 0
            print("Undid classification for previous frame.")
            self.updateForNewFrame()
    
    def SaveProgress(self,event):
        self.saveClassifications()
        print(f"Your classifications for the first {self.curr_pk_ind} peak(s) have been saved in {self.save_file}.")
    
    def saveClassifications(self):
        # convert np array of integers to list of strings
        network_labels = []
        for id in self.pk_network:
            if id == 1:
                network_labels.append("CB")
            elif id == 2:
                network_labels.append("RP")
            else:
                network_labels.append("Neither")
        df = pd.DataFrame(network_labels)
        df.to_csv(self.save_file,index=False)

    def onClose(self,event):
        if self.curr_pk_ind < len(self.peaks):
            print("App closed before classification was completed. Your work has not been saved.")
        self.Destroy()
    
    def getInfoText(self,frame_num):
        return "Currently analyzing: " + self.video_path.split('/')[-1] + "\n"\
            "Current frame number: " + str(frame_num)
    
    def on_press(self, event):
        btn_name = event.GetEventObject().name

        if btn_name == "CB":
            self.pk_network[self.curr_pk_ind] = 1
        elif btn_name == "RP":
            self.pk_network[self.curr_pk_ind] = 2
        print(f'Frame {self.peaks[self.curr_pk_ind]} classified as {btn_name}.')
        self.curr_pk_ind += 1

        # Stopping condition
        if self.curr_pk_ind >= len(self.peaks):
            self.saveClassifications()
            print(f"Your classifications have been saved in {self.save_file}.")
            # Close app window
            self.Close()
        
        else:
            self.updateForNewFrame()
    
    def updateForNewFrame(self):
        currFrame = self.peaks[self.curr_pk_ind]
        self.info.SetLabel(self.getInfoText(currFrame))
        self.fluo_len_plot.update_frame_marker(currFrame)
        self.video_frames.update_frames(currFrame)

    def convert_video_to_frames(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success,image = vidcap.read()
        count = 0
        while success:
            frame_name = "frame%d.jpg" % count
            writeStatus = cv2.imwrite(self.frames_dir + frame_name, image)     # save frame as JPEG file      
            if writeStatus is True:
                print(f"Wrote {frame_name}")
            else:
                raise Exception("Failed to write image. Make sure the directory exists.")
            success,image = vidcap.read()
            count += 1
    
class ContainerPanel(wx.Panel):
    def __init__(self, parent):
        self.video_path = parent.video_path
        self.fluo_path = parent.fluo_path
        self.midlens_path = parent.midlens_path
        self.frames_dir = parent.frames_dir

        wx.Panel.__init__(self, parent)

class ImagePanel(wx.Panel):
    """ create a panel with a wx.StaticBitmap """
    def __init__(self, parent, bmp, label):
        self.bmp = bmp
        bmp_size = self.bmp.GetSize()
        # add space for caption
        self.caption_size = 20
        wx.Panel.__init__(self, parent, size=(bmp_size[0],bmp_size[1] + self.caption_size))
        self.SetBackgroundColour("#87ceeb")
        self.caption = wx.StaticText(self)
        self.caption.SetLabel(label)

        # set up a paint event on a PaintDC canvas
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event=None):
        # create the paint canvas
        dc = wx.PaintDC(self)
        # clear the canvas
        dc.Clear()
        # draw the image
        dc.DrawBitmap(self.bmp, 0, self.caption_size, True)

class VideoFramesPanel(wx.Panel):
    def __init__(self, parent, currFrame):
        self.frames_dir = parent.frames_dir
        self.currFrame = currFrame
        
        # Gathering info to determine panel size
        image_path = self.frames_dir + "frame%d.jpg" % currFrame
        image = wx.Image(image_path, wx.BITMAP_TYPE_JPEG)
        size = image.GetSize()

        panel_width = int(size[0]*3*IMG_SCALE)
        panel_height = int(size[1]*IMG_SCALE)

        wx.Panel.__init__(self, parent,size=(panel_width,panel_height))
        self.frame_box = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.frame_box)

        self.curr = ImagePanel(self, self.get_bitmap(currFrame), "Curr frame")
        self.prev = ImagePanel(self, self.get_bitmap(currFrame - 1), "Prev frame")
        self.next = ImagePanel(self, self.get_bitmap(currFrame + 1), "Next frame")
        self.update_frames(currFrame)

        self.frame_box.Add(self.prev,1, wx.ALL, 5)
        self.frame_box.Add(self.curr,1, wx.ALL,5)
        self.frame_box.Add(self.next,1, wx.ALL,5)
    
    def get_bitmap(self, frame_num):
        image_path = self.frames_dir + "frame%d.jpg" % frame_num
        image = wx.Image(image_path, wx.BITMAP_TYPE_JPEG)
        size = image.GetSize()
        image = image.Scale(int(size[0]*IMG_SCALE), int(size[1]*IMG_SCALE), wx.IMAGE_QUALITY_HIGH)
        bmp = image.ConvertToBitmap()
        return bmp
    
    def update_frames(self, currFrame):
        self.curr.bmp = self.get_bitmap(currFrame)
        self.curr.OnPaint()

        self.prev.bmp = self.get_bitmap(currFrame -1)
        self.prev.OnPaint()

        self.next.bmp = self.get_bitmap(currFrame +1)
        self.next.OnPaint()

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

        self.figure = Figure(figsize=(15,3))
        self.ax = self.figure.add_subplot(111)
        utils.plot_fluo_len(self.ax,self.fluo,self.midlens,self.x_frame)

        self.ax.axvline(currFrame, color='magenta')
        self.canvas = FigureCanvas(self, -1, self.figure)
        
    def update_frame_marker(self, currFrame):
        # Only update the frame marker line, leave the fluo/len plot as is
        self.ax.lines.pop(-1)
        # Plot position of current frame
        self.ax.axvline(currFrame, color='magenta')
        self.canvas = FigureCanvas(self, -1, self.figure)

def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', help=".avi Hydra movie you'd like to analyze")
    parser.add_argument('fluo_path', help=".csv file containing total fluorescence intensity of each video frame")
    parser.add_argument('midlens_path',help=".csv file containing the length of Hydra's midline in each video frame")
    parser.add_argument('peaks_path',help=".csv file containing a series of frame indicies at which there are spikes in fluorescence")
    parser.add_argument('frames_dir', help="directory where video frames are saved")
    parser.add_argument('save_file', help="file name where you'd like to save classification results")
    parser.add_argument('start_index', help="Use this if you have some classification progress already saved.")
    parser.add_argument('--extract', help="extract frames of videos so they can be displayed in GUI")
    
    args = parser.parse_args(args=argv)
    initialize_GUI(args)
    

def initialize_GUI(args):
    app = wx.App()
    gui = Classifier_GUI(args)
    app.MainLoop()

if __name__ == '__main__':
    sys.exit(main())

