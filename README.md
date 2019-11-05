# Data Analysis Tools for *Hydra* Videos

- [Data Analysis Tools for *Hydra* Videos](#data-analysis-tools-for-hydra-videos)
  - [Dependencies](#dependencies)
  - [Find Midline](#find-midline)
  - [Trace Fluorescence](#trace-fluorescence)
  - [Peduncle Fluorescence](#peduncle-fluo)
  - [Body Column Fluorescence](#body-col-fluo)

## Dependencies
numpy, matplotlib, opencv-python, pandas

## Find Midline

Track the midline of *hydra* based on the contours data from [ICY](http://icy.bioimageanalysis.org/) and coordinates of some tracked points from [DeepLabCut](https://github.com/AlexEMG/DeepLabCut). Measure the length of the midline for each frame and save it as a csv file.

- Clone this repo
- Enter the directory find_midline/
- Run `python midline.py path/of/icy/file path/of/deeplabcut/file max_of_depth scale_of_x scale_of_y`

where the file from ICY should be .xml file, the file from DeepLabCut should be .csv file.

- max_of_depth is the depth of recursion when finding midline, based on which the program will generate 2<sup>max\_of\_depth+1</sup>-1 midpoints for drawing the midline.

- scale_of_x is the compression ratio by ICY in x-direction

- scale_of_y is the compression ratio by ICY in y-direction

**Note:** The tracked parts of Hydra must be in the order of *hypostome, armpit1, armpit2 and peduncle*

## Trace Fluorescence

Integrate the fluorescence in each frame of a video and plot the trace.

- Clone the repo
- Enter the directory trace_fluorescence
- Run `python trace_fluo.py path/of/video/file fps`

where fps is the fps(frames per second) of the video.

## Peduncle Fluorescence

Using the Midline integrate the fluorescence in the peduncle of the Hydra.

- Clone the repo
- Enter the directory peduncle_fluo/
- Run `python pedunc_fluo_contour.py path/of/icy/file path/of/DeepLabCut/file max_depth /path/of/video scale_of_x scale_of_y`
- The arguments to the script are the same as midline.py, the path to the video being used is added

**Note:** This code requires the midline.py script in the same relative path as present in the repo.

## Body Column Fluorescence

Using the Midline find the fluorescence in the body column of the Hydra.

- Clone the repo
- Enter the directory peduncle_fluo/
- Run `python bo_col_fluo.py path/of/icy/file path/of/DeepLabCut/file max_depth /path/of/video scale_of_x scale_of_y`
- The arguments to the script are the same as the ones used in Peduncle Fluorescence

**Note:** This code requires the midline.py script in the same relative path as present in the repo
