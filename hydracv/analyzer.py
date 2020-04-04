import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath(''), '..')))

from cv2 import cv2
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import hydracv.disp as disp

class Analyzer:
    """A tool for analyzing calcium imaging videos (of Hydra)"""

    class _Video:
        """A data structure for storing related information of a video"""
        def __init__(self, name, path, fps, numframes):
            """Constructor"""
            if type(name) != str:
                raise TypeError("Name must be a string!")
            if type(path) != str:
                raise TypeError("Path must be a string!")
            if type(fps) != int:
                raise TypeError("FPS must be an int!")
            if type(numframes) != int:
                raise TypeError("Frames number must be an int!")

            self._name = name
            self._path = path
            self._fps = fps
            self._numframes = numframes
            self._fluo_trace = []
            self._peak_clusters = []
            self._peaks = []

        def name(self):
            """Get the name of the video"""
            if self._name:  
                return self._name
            else:
                raise ValueError("The video is not initialized with a name!")

        def path(self):
            """Get the path of the video"""
            if self._path:  
                return self._path
            else:
                raise ValueError("The video is not initialized with a path!")
        
        def fps(self):
            """Get the FPS of the video"""
            if self._fps:  
                return self._fps
            else:
                raise ValueError("The video is not initialized with an FPS!")

        def numframes(self):
            """Get the frames number of the video"""
            if self._numframes:  
                return self._numframes
            else:
                raise ValueError("The frames number of the video has not been calculated yet!")

        def fluo_trace(self):
            """Get the fluorescence trace of the video"""
            if self._fluo_trace:
                return self._fluo_trace
            else:
                raise ValueError("The fluorescence trace has not been obtained yet!")

        def set_fluo_trace(self, trace):
            """Set the video's fluorescence trace"""
            self._fluo_trace = trace

        def peaks(self):
            """Get the fluorescence peaks of the video"""
            if self._peaks:
                return self._peaks
            else:
                raise ValueError("The fluorescence peaks have not been obtained yet!")

        def set_peaks(self, pks):
            """Set the video's fluorescence peaks"""
            self._peaks = pks

        def peak_clusters(self):
            """Get the fluorescence peak clusters of the video"""
            if self._peak_clusters:
                return self._peak_clusters
            else:
                raise ValueError("The fluorescence peaks have not been obtained yet!")

        def set_peak_clusters(self, pk_clstrs):
            """Set the video's fluorescence peak clusters"""
            self._peak_clusters = pk_clstrs


    def __init__(self):
        """Constructor"""
        self._videos = defaultdict()
        self._spike_trains = None
        self._fps = None
        self._isi = None

    def _add_video(self, pn, fps=None):
        """Add a video to the analyzer.
        
        Args:
            pn: The path to the video file. 
            fps: The FPS of the video.

        Raises:
            FileNotFoundError: An error occurred if the video file is not found
        """

        if not os.path.isfile(pn):
            raise FileNotFoundError("File not found: " + pn)
        
        path, name = pn.rsplit('/', 1)
        path += '/'

        # Judege whether the video has already been added
        if name in self._videos:
            return

        # Count the number of frames of the video
        cap = cv2.VideoCapture(pn)
        numframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Add the video to the Hash Map
        video = self._Video(name, path, fps, numframes)
        self._videos[name] = video

    def add_videos(self, pathname, fps=None):
        """Add a list of videos to the analyzer.
        
        Args:
            pathname: A list of paths to the video files. 
            fps: A list of corresponding FPS of the videos.

        Raises:
            ValueError: An error occurred if the length of pathname is 
                different with that of fps
            FileNotFoundError: An error occurred if the video file is not found
        """
        if len(pathname) != len(fps):
            raise ValueError("The lengths of pathname and fps must be the same!")

        for j in range(len(pathname)):
            self._add_video(pathname[j], fps[j])

    def _video_names(self):
        """Returns the names of added videos"""
        return list(self._videos.keys())

    def videos(self):
        """Print the names of added videos."""
        for key in self._video_names():
            print(key)


    def _video(self, name):
        """
        Get a video with the given name.

        Args:
            name: Name of the target video.

        Returns:
            An Video object with the given name.
        
        Raises:
            NameError: An error occurred if no video has the given name.
        """

        return self._videos[name]

    def _trace_fluo(self, name, normalized=True, display=False):
        """Trace the fluorescence of the video with the given name.

        Obtain the fluorescence trace from the video with the given name, and 
        set it as the _fluo_trace attribute of the video.

        Args:
            name: Name of the target video.
            display: Whether to display frames during tracing? Default as False.
        """

        if len(self._videos[name]._fluo_trace) > 0:
            return

        fluo = []
        # Get the full path of video
        video = self._video(name)
        cap = cv2.VideoCapture(video.path() + video.name())

        firstf = True
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                if firstf:
                    size = len(frame) * len(frame[0])
                    firstf = False
                # Our operations on the frame come here
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                intensity = np.sum(frame)

                fluo.append(intensity)

                # Display the current frame
                if display:
                    cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Find OpenCV version
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        # Normalize the fluorescence if normalized==True
        fluo = Analyzer.normalize_fluo(fluo)

        # Set the fluorescence trace to video
        video.set_fluo_trace(fluo)

    @staticmethod
    def normalize_fluo(fluo):
        """Normalize fluo"""
        maxfluo = max(fluo)
        minfluo = min(fluo)
        return [(x - minfluo) / (maxfluo - minfluo) for x in fluo]

    def trace_fluos(self, name=None, normalized=True, display=False, plot=False):
        """Trace the fluorescences of the videos in name.

        Args:
            name: A list of names of target videos. Default as None. If set as 
                None, this function traces all added videos.
            normalized: Whether to normalize the fluorescence? Default as True.
            display: Whether to display frames during tracing? Default as False.
            plot:  Whether to plot the trace? Default value is False.
        Raises:
            TypeError: An error occurred if name is neither a list nor a string.
        """

        if name is None:
            name = self._video_names()

        if type(name) != list:
            if type(name) == str:
                self._trace_fluo(name, normalized, display)
            else:
                raise(TypeError("name can only be list or string!"))
        else:
            for nm in name:
                self._trace_fluo(nm, normalized, display)

        # Plot the trace if plot is True
        if plot:
            self.plot_fluos(name)

    def plot_fluos(self, name=None):
        """Plot the fluorescence trace of the video with given name.
        
        Args:
            name: A list of names of target videos. Default as None. If set as 
                None, this function plots the fluorescence of all added videos.
        Raises:
            TypeError: An error occurred if name is neither a list nor a string.
        """

        # video = self._video(name)

        if name is None:
            name = self._video_names()

        if type(name) != list:
            if type(name) == str:
                fig = plt.figure(figsize=(20,5))       
                video = self._videos[name]
                ax = fig.add_subplot(1, 1, 1)
                time_axis = np.linspace(0, video.numframes()/video.fps(), video.numframes())
                disp.add_fluorescence(ax, time_axis, video.fluo_trace())
            else:
                raise(TypeError("name can only be list or string!"))
        
        else:
            nvideos = len(name)

            fig = plt.figure(figsize=(20,5*nvideos))
            for j in range(nvideos):
                video = self._videos[name[j]]
                ax = fig.add_subplot(nvideos, 1, j+1)
                time_axis = np.linspace(0, video.numframes()/video.fps(), video.numframes())
                disp.add_fluorescence(ax, time_axis, video.fluo_trace())
        plt.show()
        
    def _find_peaks_for_single(self, name, plot=True, height=0.1, wlen=100,
                   prominence=0.025, min_cb_interval=10):
        """Find the fluorescence peaks of the video with given name.

        Save the found peaks as the _peaks attribute of the video. And save the 
        peak clusters as _save_clusters attribute of the video.
        
        Args:
            name: Name of the target video.
            plot: Whether to plot the marked peaks? Default as True.
            height: Required height of peaks.
            wlen: A window length in samples that optionally
                limits the evaluated area for each peak to a
                subset of x. The peak is always placed in the
                middle of the window therefore the given length
                is rounded up to the next odd integer.
            prominence: Required prominence of peaks.
            min_cb_interval: Minimum interval(in real seconds) 
                between CBs, used in clustering peaks.
        """

        video = self._video(name)
        time_axis = np.linspace(0, video.numframes()/video.fps(), video.numframes())
        fluo = video.fluo_trace()
        peaks, _ = scipy.signal.find_peaks(fluo, height=height, wlen=wlen, prominence=prominence)
        video.set_peaks(peaks)

        # Cluster peaks into different CB
        peak_clusters = Analyzer.cluster_peaks(peaks, min_cb_interval=min_cb_interval*video.fps())
        video.set_peak_clusters(peak_clusters)

        if plot:
        
            fig = plt.figure(figsize=(20, 10))

            ax1 = fig.add_subplot(2, 1, 1)
            disp.add_fluorescence(ax1, time_axis, fluo)
            disp.add_peaks(ax1, peaks, fluo, video.fps())

            ax2 = fig.add_subplot(2, 1, 2)
            disp.add_spike_trains(ax2, peak_clusters, video.fps())

        plt.show()

    def find_peaks(self, name=None, plot=False, height=0.1, wlen=100, 
                   prominence=0.025, min_cb_interval=10):
        """Find the fluorescence peaks of a list of videos.

        Save the found peaks as the _peaks attribute of the videos. And save the 
        peak clusters as _save_clusters attribute of the videos.
        
        Args:
            name: A list of names of the target videos. Default as None.
                If its value is None, it will be set as the list of all 
                added videos
            plot: Whether to plot the marked peaks? Default as False.
            height: Required height of peaks.
            wlen: A window length in samples that optionally
                limits the evaluated area for each peak to a
                subset of x. The peak is always placed in the
                middle of the window therefore the given length
                is rounded up to the next odd integer.
            prominence: Required prominence of peaks.
            min_cb_interval: Minimum interval(in real seconds) 
                between CBs, used in clustering peaks.
        """
        if name is None:
            name = self._video_names()

        if type(name) != list:
            if type(name) == str:
                self._find_peaks_for_single(name, plot, height, wlen, 
                                            prominence, min_cb_interval)
            else:
                raise(TypeError("name can only be list or string!"))
        
        else:
            nvideos = len(name)
            for j in range(nvideos):
                self._find_peaks_for_single(name[j], plot, height, wlen, 
                                            prominence, min_cb_interval)


    @staticmethod
    def cluster_peaks(peaks, min_cb_interval):
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
            offset = cluster[0]
            for j in range(len(cluster)):
                cluster[j] -= offset

        return list(np.array(clusters)[indices_to_keep])

    def _update_spike_trains_and_fps(self):
        """Update the _spike_trains and _fps of the Analyzer object."""

        spike_trains = []
        fps_list = []

        for name in self._video_names():
            video = self._videos[name]
            peak_clusters = video.peak_clusters()
            spike_trains.extend(peak_clusters)
            fps_list.extend([video.fps()]*len(peak_clusters))

        self._spike_trains = spike_trains
        self._fps = fps_list

    def _update_isi(self):
        """Update the _isi (interspike interval) of the Analyzer object"""

        self._isi = []
        self._update_spike_trains_and_fps()
        for i, trial in enumerate(self._spike_trains):
            self._isi.append([(trial[j+1] - trial[j])/self._fps[i] for j in range(len(trial)-1)])


    def plot_spike_trains(self):
        """Plot the spike trains of all added videos"""
        
        self._update_spike_trains_and_fps()
        
        fig = plt.figure(figsize=(20, 0.1*len(self._spike_trains)))
        ax = fig.add_subplot(1, 1, 1)

        disp.add_spike_trains(ax, self._spike_trains, self._fps)
        
        plt.show()

    def stat_isi(self):
        """Plot the statistics of ISI"""

        # Update ISI
        self._update_isi()
        fig = plt.figure(figsize=(20,5))

        # Plot the curves
        ax1 = fig.add_subplot(1, 2, 1)
        disp.add_isi_curves(ax1, self._isi)

        # Plot the box plots
        ax2 = fig.add_subplot(1, 2, 2)
        isi_t = Analyzer.transpose_2d_list(self._isi)
        disp.add_boxplot(ax2, isi_t)

        plt.show()

    @staticmethod
    def transpose_2d_list(list2d):
        """Returns a transposed 2d list"""

        maxlen = max([len(l) for l in list2d])
        for j in range(len(list2d)):
            list2d[j].extend([None]*(maxlen-len(list2d[j])))
        
        list2d = np.array(list2d).T.tolist()

        for j in range(len(list2d)):
            list2d[j] = [x for x in list2d[j] if x]

        return list2d

    def delete_video(self, name):
        """Delete a video with name from the Analyzer object"""

        if name not in self._videos:
            raise NameError(name + " is not added!")

        del self._videos[name]
        


        






        




        
