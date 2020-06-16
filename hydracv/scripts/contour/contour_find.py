import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sys



class Contours:
	def __init__(self, contours, hierarchy = None):
		self.contours = contours
		#If hierarchy specified then we create an iterator to iterate over it...
		self.hierarchy = hierarchy
		self.nC = len(contours)

	def traverse(self):
		levels = np.zeros((self.nC, 1)).tolist()
		for idx,ct in enumerate(self.contours):
			if self.hierarchy is not None:
				parent = self.hierarchy[0,idx,3]
				if parent == -1:
					level = 0
				else:
					level = levels[parent]+1
				levels[idx] = level
			else:
				level = None
			yield (ct, level)

	def remove(self, idx):
		#Remove a contour and all of its children
		toremove = []
		#Do recursively
		self._remove(toremove, idx, 1)
		#We have the nodes to delete, now we remove these from contours list, and
		#reindex the hierarchy variable
		nR = len(toremove)
		minidx = min(toremove)
		maxidx = max(toremove)
		contours = []
		hierarchy = []
		for idx in range(self.nC):
			if idx not in toremove:
				contours.append(self.contours[idx])
				hierarchy.append(self.hierarchy[0,idx,:].tolist())
		self.contours = contours
		hierarchy = np.array(hierarchy, dtype = 'int32')
		hierarchy[np.where(np.logical_and(hierarchy >= minidx, hierarchy <= maxidx))] = -1
		hierarchy[np.where(hierarchy > maxidx)] = hierarchy[np.where(hierarchy > maxidx)] - nR
		self.hierarchy = hierarchy[np.newaxis, :,:]
		self.nC = len(contours)

	def _remove(self, toremove, idx, top):
		#If I have children, remove these first
		if self.hierarchy[0,idx,2] != -1:
			toremove.append(self._remove(toremove, self.hierarchy[0,idx,2], 0))
		#If I have siblings and I am a subtree, delete my siblings too
		if self.hierarchy[0,idx,0] != -1 and top == 0:
			toremove.append(self._remove(toremove, self.hierarchy[0,idx,0], 0))
		#Finally remove myself
		if top == 0:
			return idx
		else:
			toremove.append(idx)
			return toremove

def findObjectThreshold(img, contours_, threshold = 7):
	"""Find object within image using simple thresholding

	Input:
		-img: input image object
		-threshold: threshold intensity to apply (can be chosen from looking at
			histograms)

	Output:
		-mask: mask containing object
		-contours: contours outlining mask
		-hierarchy: hierarchy of contours
	"""
	#Test code:
	#img = cv2.imread('./video/testcontours.jpg')

	#Just try simple thresholding instead: (quick!, seems to work fine)
	if len(img.shape) == 3:
		frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		frame_gray = img

	frame_gray = cv2.blur(frame_gray, (5,5))
	frame_gray = cv2.blur(frame_gray, (5,5))
	frame_gray = cv2.blur(frame_gray, (5,5))

	#Global threshold
	ret1, mask = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_TOZERO)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')


	# print(threshold)
	# plt.imshow(mask2, cmap='gray')
	# plt.pause(0.001)
	# input('Press Enter')
	#Find contours of mask

	c, h = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
	ctrs = Contours(c, h)

	# img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

	#Remove contours that are smaller than 40 square pixels, or that are above
	#level one
	changed = True
	maxarea = 0
	for (ct, level) in ctrs.traverse():
		maxarea = max(maxarea, cv2.contourArea(ct))
	while changed:
		changed = False
		#print 'length: %d' % len(ctrs.contours)
		for idx, (ct, level) in enumerate(ctrs.traverse()):
			area = cv2.contourArea(ct)
			if area < maxarea :
				ctrs.remove(idx)
				changed = True
				break
			if area < 40:
				ctrs.remove(idx)
				changed = True
				break
			if level > 1:
				ctrs.remove(idx)
				changed = True
				break

	contours_.append(ctrs.contours)

def contour_call(video, threshold):
	# '/home/shashank/Downloads/Control-EGCaMP_exp1_a1_30x10fps_5%_cnt.avi'

	try:
		cap = cv2.VideoCapture(video)
	except:
		raise FileNotFoundError("Video file not found: " + video)
	firstf = True
	iframe = 0
	contours_ = []
	number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for iframe in tqdm(range(number_frames)):
	    # Capture frame-by-frame
		ret, frame = cap.read()
		if ret:
			findObjectThreshold(frame, contours_, threshold)
		else:
			break
	vname = video.split('/')[-1]
	vname = vname.split('.')[0]
	try:
		with open('./contours/'+vname+'.pkl', 'wb') as f:
			pickle.dump(contours_, f)
	except:
		os.makedirs('./contours')

if __name__ = '__main__':
	video = sys.argv[1]
	try:
		threshold = sys.argv[2]
	except:
		threshold = 0
	contour_call(video, threshold)
