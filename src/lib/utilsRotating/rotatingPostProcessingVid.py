# Importing all necessary libraries
import os
import csv
from tempfile import template
import cv2
import re
import pickle

# Specify working directory and relevant file names
import numpy as np

cwd = 'C:\\Users\\user-pc\\Desktop\\26Mar2023'
date = '2023_03_26_10_27'

vid1 = os.path.join(cwd, ('1_video-' + date + '.mp4'))
vid2 = os.path.join(cwd, ('2_video-' + date + '.mp4'))

timestamp1 = os.path.join(cwd, ('1_timestamps-' + date + '.txt'))
timestamp2 = os.path.join(cwd, ('2_timestamps-' + date + '.txt'))

logFile1 = os.path.join(cwd, ('1_terminal_logs-' + date + '.txt'))
logFile2 = os.path.join(cwd, ('2_terminal_logs-' + date + '.txt'))

encFile1 = os.path.join(cwd, ('1_encoder_vals-' + date + '.txt'))
encFile2 = os.path.join(cwd, ('2_encoder_vals-' + date + '.txt'))

# Determine skipped frames in each stream
skipped1, skipped2 = [], []

if not skipped1 or not skipped2:
    with open(logFile1) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'skipping' in line:
                line = f.readline()
                skipped1.append(line.split(' ')[-1].replace('\n', ''))

    with open(logFile2) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'skipping' in line:
                line = f.readline()
                skipped2.append(line.split(' ')[-1].replace('\n', ''))

print(skipped1)
print(skipped2)
skipped1 = ['2', '3', '4', '5', '6', '7']
skipped2 = ['4', '5', '6', '7', '8', '9', '10', '11', '12']
# Try creating folders for stream 1 and 2
dir1 = os.path.join(cwd, '1')
dir2 = os.path.join(cwd, '2')

try:
    if not os.path.exists(dir1) and not os.path.exists(dir2):
        os.makedirs(dir1)
        os.makedirs(dir2)
        os.makedirs(os.path.join(cwd, '1', 'dupes'))
        os.makedirs(os.path.join(cwd, '2', 'dupes'))
except OSError:
    print('Error: Creating directory for streams 1 or 2. Ensure there are no folders named "1" or "2" in your working '
          'directory.')

# Read the first video from specified path and create images from stream
cam1 = cv2.VideoCapture(vid1)

count = 0

with open(timestamp1) as f:
    while True:
        line = f.readline().replace('\n', '').split(',')
        if not line:
            break
        if len(line) < 2:
            break
        count += 1

        frame = line[0]
        timestamp = line[1].replace('.', '')

        if len(timestamp) < 19:
            temp1 = timestamp[:10]
            temp2 = timestamp[10:]
            zeroes = '0'*(19-len(timestamp))
            timestamp = temp1 + zeroes + temp2

        if frame not in skipped1:
            ret, frame = cam1.read()
            if ret:
                # print(str(count) + ": " + str(row))
                frameName = timestamp[:13] + '.jpg'
                #print(frameName)
                framePath = os.path.join(dir1, frameName)
                # print(framePath)
                if not os.path.exists(framePath):
                    cv2.imwrite(framePath, frame)
                    count = count
                else:
                    frameName = str(int(line[1]) + 1) + '.jpg'
                    framePath = os.path.join(dir1, 'dupes', frameName)
                    cv2.imwrite(framePath, frame)
            else:
                print("End of video file.")
                break
print(count)
cam1.release()

# Read the second video from specified path and create images from stream
cam2 = cv2.VideoCapture(vid2)

count = 0
with open(timestamp2) as f:
    while True:
        line = f.readline().replace('\n', '').split('-')
        if not line:
            break
        if len(line) < 2:
            break
        count += 1

        frame = line[0]
        timestamp = line[1].replace('.', '')

        if len(timestamp) < 19:
            temp1 = timestamp[:10]
            temp2 = timestamp[10:]
            zeroes = '0'*(19-len(timestamp))
            timestamp = temp1 + zeroes + temp2

        if frame not in skipped2:
            ret, frame = cam2.read()
            if ret:
                # print(str(count) + ": " + str(row))
                frameName = timestamp[:13] + '.jpg'
                framePath = os.path.join(dir2, frameName)
                # print(framePath)
                if not os.path.exists(framePath):
                    cv2.imwrite(framePath, frame)
                    count = count
                else:
                    frameName = str(int(line[1]) + 1) + '.jpg'
                    framePath = os.path.join(dir2, 'dupes', frameName)
                    cv2.imwrite(framePath, frame)
            else:
                print("End of video file.")
                break
cam2.release()

dir1 = os.path.join(cwd, '1')
dir2 = os.path.join(cwd, '2')

# Collect filenames (timestamps) into lists
f1, f2 = [], []
for (dirpath, dirnames, filenames) in os.walk(dir1):
    for file in filenames:
        file.split('.')[0]
        f1.append(int(file.split('.')[0]))
    break

for (dirpath, dirnames, filenames) in os.walk(dir2):
    for file in filenames:
        f2.append(int(file.split('.')[0]))
    break

# Find closest matches in f2 to each element in f1
corr_frames = list(map(lambda y: min(f2, key=lambda x: abs(x - y)), f1))

# Find closest encoder timestamps (matched to video 1 (left))
with open(encFile1) as tsF:
    file_content = tsF.read()
content_list = re.split('\n', file_content)
tsF.close()

encVals1, encTS1 = [], []
for item in content_list:
    if len(item) > 2:
        val = re.split(',', item)
        encVals1.append(int(val[0]))
        encTS1.append(int(val[1].replace('.', '').replace(' ', '')[:13]))

with open(encFile2) as tsF:
    file_content = tsF.read()
content_list = re.split('\n', file_content)
tsF.close()

encVals2, encTS2 = [], []
for item in content_list:
    if len(item) > 2:
        val = re.split(',', item)
        encVals2.append(int(val[0]))
        encTS2.append(int(val[1].replace('.', '').replace(' ', '')[:13]))

corr_timestamps1 = list(map(lambda y: min(encTS1, key=lambda x: abs(x - y)), f1))
corr_timestamps2 = list(map(lambda y: min(encTS2, key=lambda x: abs(x - y)), f1))

# Create new list with [f1, closest Match in f2, abs(offset)]
matches = []

for el1, el2, el3, el4 in zip(f1, corr_frames, corr_timestamps1, corr_timestamps2):
    matches.append([el1, el2, abs(el1 - el2), el3, encVals1[encTS1.index(el3)], abs(el1 - el3), el4,
                    encVals2[encTS2.index(el4)], abs(el1 - el4)])
print(matches)

# for el1, el2 in zip(f1, corr_frames):
#    matches.append([el1, el2, abs(el1-el2)])
# print(matches)

matches = np.array(matches)

syncedData_dict = dict()
syncedData_dict["c1ts"] = matches[:, 0]
syncedData_dict["c2ts"] = matches[:, 1]
syncedData_dict["c1c2offset"] = matches[:, 2]
syncedData_dict["encts"] = matches[:, 3]
syncedData_dict["enc1tick"] = matches[:, 4]
syncedData_dict["c1enc1offset"] = matches[:, 5]
syncedData_dict["enc2ts"] = matches[:, 6]
syncedData_dict["enc2tick"] = matches[:, 7]
syncedData_dict["c1enc2offset"] = matches[:, 8]

with open(os.path.join(cwd, 'synced_data.pkl'), 'wb') as f:
    pickle.dump(syncedData_dict, f)


# Loops through closest matches and creates new video with synced frames.
# Also writes encoder values to file for new video.
out_path1 = os.path.join(cwd, "1.avi")
out_path2 = os.path.join(cwd, "2.avi")
fps = 100

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter(out_path1, fourcc, fps, (1280, 720))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (1280, 720))

for row in matches:
    out1.write(cv2.imread(os.path.join(dir1, (str(row[0]) + '.jpg'))))
    out2.write(cv2.imread(os.path.join(dir2, (str(row[1]) + '.jpg'))))

# Release all space and windows once done
out1.release()
out2.release()

cv2.destroyAllWindows()
