# Data Analysis Scripts for *Hydra* Videos

- [Data Analysis Scripts for *Hydra* Videos](#data-analysis-tools-for-hydra-videos)
  - [0. Dependencies](#0-dependencies)
  - [1. Find Midline](#1-find-midline)
    - [1.1 Full Body](#11-full-body)
    - [1.2 Peduncle Half](#12-peduncle-half)
  - [2. Trace Fluorescence](#2-trace-fluorescence)
    - [2.1 Entire Frame](#21-entire-frame)
    - [2.2 Peduncle Part](#22-peduncle-part)
    - [2.3 Body Column Part](#23-body-column-part)
  - [3. Segmentation](#3-segmentation)

## 0. Dependencies
numpy, matplotlib, opencv-python, pandas, tqdm

## 1. Find Midline

### 1.1 Full Body

Track the midline of *hydra* based on the contours data from [ICY](http://icy.bioimageanalysis.org/) and coordinates of some tracked points from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut). Measure the length of the midline for each frame and save it as a csv file.

**Steps:**

- Clone this repo
- Enter the directory find_midline/
- Modify config.json
- Run `python midline.py`

**Parameters in config.json:**

- _IcyFilePath_: Path of the .xml file from ICY

- _DeeplabcutFilePath_: Path of the .csv file from DeepLabCut
  
- _MaxDepth_: The depth of recursion when finding midline, based on which the program will generate 2<sup>max\_of\_depth+1</sup>-1 midpoints for drawing the midline.

- _ScaleX_: the compression ratio by ICY in x-direction

- _ScaleY_: is the compression ratio by ICY in y-direction

**_Note_:** The tracked parts of Hydra must be in the order of *hypostome, armpit1, armpit2 and peduncle*

### 1.2 Peduncle Half

The same as [Full Body](#full-body) except for the last step should be replaced by Run `python midline_ped.py`

## 2. Trace Fluorescence

### 2.1 Entire Frame

Integrate the fluorescence of all pixels in each frame of a video and plot the trace.

**Steps:**

- Clone the repo
- Enter the directory trace_fluorescence
- Run `python trace_fluo.py path/of/video/file fps`

where fps is the fps(frames per second) of the video.

### 2.2 Peduncle Part

Using the Midline integrate the fluorescence in the peduncle of the Hydra.

**Steps:**

- Clone the repo
- Enter the directory peduncle_fluo/
- Run `python pedunc_fluo_contour.py path/of/icy/file path/of/DeepLabCut/file max_depth /path/of/video scale_of_x scale_of_y`
- The arguments to the script are the same as midline.py, the path to the video being used is added

**_Note_:** This code requires the midline.py script in the same relative path as present in the repo.

### 2.3 Body Column Part

Using the Midline find the fluorescence in the body column of the Hydra.

- Clone the repo
- Enter the directory body_col\_fluo/
- Run `python bo_col_fluo.py path/of/icy/file path/of/DeepLabCut/file max_depth /path/of/video scale_of_x scale_of_y`
- The arguments to the script are the same as the ones used in Peduncle Fluorescence

**_Note_:** This code requires the midline.py script in the same relative path as present in the repo

## 3. Segmentation

Segment Hydra body into different parts based on some tracked spots and the midline.

**Steps:**

- Clone this repo
- Enter the directory segment_body/
- Modify config.json
- Run `python segment.py`

**Parameters in config.json:**

- _IcyFilePath_: Path of the .xml file from ICY

- _DeeplabcutFilePath_: Path of the .csv file from DeepLabCut
  
- _MaxDepth_: The depth of recursion when finding midline, based on which the program will generate 2<sup>max\_of\_depth+1</sup>-1 midpoints for drawing the midline.

- _ScaleX_: The compression ratio by ICY in x-direction

- _ScaleY_: The compression ratio by ICY in y-direction

- _VideoPath_: The path of the original video file
