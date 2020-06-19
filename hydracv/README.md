# Data analysis testbeds

## Notebooks

### `midline_and_fluo.ipynb`

-- _Midline and fluorescence extraction_

Requirements:
- Origininal video
- Contour
- Tracked points by DeepLabCut

### `fluo_analysis`

-- _Fluorescence analysis (find peaks, cluster peaks, spike trains, ISI statistics)_

### `classify_behaviors.ipynb`

-- _Classify behaviors based on midline_

Requirements:
- Midpoints
- Manual labels (optional)

## Data

Create a directory named as **data** under **hydracv/hydracv**, with the following architecture as well, for storing required and generated data.

data

├─behaviors (results of classified behaviors)

├─contour (contour data)

├─fluorescence (extracted fluorescence)

├─length (extracted midline lengths)

├─manual_labels (manual labels of behaviors)

├─marker (files of DeepLabCut results)

├─midpoints (extracted midpoints)

├─objects (analyzer objects)

└─videos (original videos)
