%% Load Video

close all; clear all; clc;

v = VideoReader("/home/hengji/Documents/hydracv/hydracv/midline/results/Control-EGCaMP_exp1_a1_30x10fps/video/fillpoly.avi");
nframes = floor(v.Duration * v.FrameRate);

%% Loop over frames and skeletonize

video = VideoWriter('skeleton_matlab.avi', 'Archival');
video.FrameRate = 40;
open(video);

for iframe = 1:nframes
    
    disp(iframe)
    
    frame = read(v, iframe);

    frame_bin = im2bw(frame, 0.5);

    sklt = bwskel(frame_bin);
    sklt = im2uint8(sklt);
    
    
    writeVideo(video, sklt);

    imshow(sklt);

end

close(video);