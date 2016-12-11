import cv2
import numpy as np
import os

directory = '../data/sample/'
all_files = []
for subdir, dirs, files in os.walk(directory):
	for filename in files:
		if filename.endswith(".jpeg"):
			all_files.append(filename)

Metadata_file = open('../data/sample_metadata.txt', 'w')
Metadata_file.truncate()

threshold = 15

for image_name in all_files:
	img = cv2.imread(directory + image_name) #BGR color space
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Mask of non-black pixels (assuming image has a single channel).
	mask = gray > threshold

	# Coordinates of non-black pixels.
	coords = np.argwhere(mask)

	# Bounding box of non-black pixels.
	x0, y0 = coords.min(axis=0)
	x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

	Metadata_file.write("%s,%d,%d,%d,%d\n" %(image_name, x0, y0, x1, y1))

Metadata_file.close()