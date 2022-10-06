# Importing all necessary libraries
import os
import cv2

# Specify working directory and relevant file names
cwd = 'C:\\Users\\user-pc\\Desktop\\22Sep2022'
date = '2022_09_22_05_46'

logFile1 = os.path.join(cwd, ('1_terminal_logs-' + date + '.txt'))
logFile2 = os.path.join(cwd, ('2_terminal_logs-' + date + '.txt'))

vid1 = os.path.join(cwd, ('1_video-' + date + '.mp4'))
vid2 = os.path.join(cwd, ('2_video-' + date + '.mp4'))

# Determine skipped frames in each stream
skipped1 = []
skipped2 = []

if not skipped1 or not skipped2:
    with open(os.path.join(cwd, logFile1)) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'skipping' in line:
                line = f.readline()
                skipped1.append(line.split(' ')[-1].removesuffix('\n'))

    with open(os.path.join(cwd, logFile2)) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if 'skipping' in line:
                line = f.readline()
                skipped2.append(line.split(' ')[-1].removesuffix('\n'))

cap = cv2.VideoCapture(vid1)
length1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap = cv2.VideoCapture(vid2)
length2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(length1)
print(len(skipped1))
print(skipped1)

print(length2)
print(len(skipped2))
print(skipped2)

cap.release()