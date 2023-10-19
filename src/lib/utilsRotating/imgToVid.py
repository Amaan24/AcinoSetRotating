import os
from tempfile import template
import cv2
import re
import numpy as np
import pickle

cwd = 'C:\\Users\\user-pc\\Desktop\\18May2023'
date = '2023_05_18_06_55'

dir1 = os.path.join(cwd, '1')
dir2 = os.path.join(cwd, '2')

encFile1 = os.path.join(cwd, ('1_encoder_vals-' + date + '.txt'))
encFile2 = os.path.join(cwd, ('2_encoder_vals-' + date + '.txt'))

out_path1 = os.path.join(cwd, "1_unsynced.mp4")
out_path2 = os.path.join(cwd, "2_unsynced.mp4")
fps = 60


fourcc = cv2.VideoWriter_fourcc(*'MP4V')#VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(out_path1, fourcc, fps, (1280, 720))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (1280, 720))

# Collect filenames (timestamps) into lists
print("Collecting image filenames into list")
f1, f2 = [], []
for (dirpath, dirnames, filenames) in os.walk(dir1):
    for file in filenames:
        file.split('.')[0]
        f1.append(int(file.split('.')[0][0:13]))
    break

for (dirpath, dirnames, filenames) in os.walk(dir2):
    for file in filenames:
        f2.append(int(file.split('.')[0][0:13]))
    break

# Collect encoder value and timestamps into lists
print("Collecting encoder timestamps")
with open(encFile1) as tsF:
    file_content = tsF.read()
content_list = re.split('\n', file_content)
tsF.close()

encVals1, encTS1 = [], []
for item in content_list:
    if len(item) > 2:
        val = re.split(',', item)
        encVals1.append(int(val[0]))
        encTS1.append(int(val[1].replace('.', '').replace(' ', '')[0:13]))

with open(encFile2) as tsF:
    file_content = tsF.read()
content_list = re.split('\n', file_content)
tsF.close()

encVals2, encTS2 = [], []
for item in content_list:
    if len(item) > 2:
        val = re.split(',', item)
        encVals2.append(int(val[0]))
        encTS2.append(int(val[1].replace('.', '').replace(' ', '')[0:13]))

print("Matching encoder timestamps...")
corr_timestamps1 = list(map(lambda y: min(encTS1, key=lambda x: abs(x - y)), f1))
corr_timestamps2 = list(map(lambda y: min(encTS2, key=lambda x: abs(x - y)), f2)) 

matches1 = []
matches2 = []

for el1, el2 in zip(f1, corr_timestamps1):
    matches1.append([el1, el2, encVals1[encTS1.index(el2)], abs(el1 - el2)])

for el1, el2 in zip(f2, corr_timestamps2):
    matches2.append([el1, el2, encVals2[encTS2.index(el2)], abs(el1 - el2)])

matches1 = np.array(matches1)
matches2 = np.array(matches2)

syncedData_dict = dict()
syncedData_dict["enc1tick"] = matches1[:, 2]
syncedData_dict["enc2tick"] = matches2[:, 2]

print("Saving synced data...")
with open(os.path.join(cwd, 'encoder_ticks_unsynced.pkl'), 'wb') as f:
    pickle.dump(syncedData_dict, f)
print("Synced data saved.")

print("Writing Videos...")
print("Writing Video 1...")
for (dirpath, dirnames, filenames) in os.walk(dir1):
    for file in filenames:
        out1.write(cv2.imread(os.path.join(dirpath, file)))
    break
print("Video 1 saved successfully.")

print("Writing Video 2...")
for (dirpath, dirnames, filenames) in os.walk(dir2):
    for file in filenames:
        out2.write(cv2.imread(os.path.join(dirpath, file)))
    break
print("Video 2 saved successfully.")

# Release all space and windows once done
out1.release()
out2.release()

cv2.destroyAllWindows()