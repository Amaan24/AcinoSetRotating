# Importing all necessary libraries
import os
import csv
from tempfile import template
import cv2
import re
import pickle
import numpy as np

cwd = 'C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\'
dir1 = os.path.join(cwd, 'Left')
dir2 = os.path.join(cwd, 'Right')


f1, f2 = [], []
for (dirpath, dirnames, filenames) in os.walk(dir1):
    for file in filenames:
        f1.append(file)
    break

for (dirpath, dirnames, filenames) in os.walk(dir2):
    for file in filenames:
        f2.append(file)
    break

out_path1 = os.path.join(cwd, "GPLeftSynced.avi")
out_path2 = os.path.join(cwd, "GPRightSynced.avi")
fps = 60

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(out_path1, fourcc, fps, (1920, 1090))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (1920, 1090))

for f in f1:
    out1.write(cv2.imread(os.path.join(dir1, f)))
    print(f)
for f in f2:
    out2.write(cv2.imread(os.path.join(dir2, f)))
    print(f)

# Release all space and windows once done
out1.release()
out2.release()

cv2.destroyAllWindows()
