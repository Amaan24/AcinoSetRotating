import collections
import csv
import os
from time import time


cwd = 'C:\\Users\\user-pc\\Desktop\\22Sep2022'
date = '2022_09_22_05_46'

logFile1 = os.path.join(cwd, ('1_terminal_logs-' + date + '.txt'))
logFile2 = os.path.join(cwd, ('2_terminal_logs-' + date + '.txt'))

filtered1 = os.path.join(cwd, ('1_terminal_logs_filtered-' + date + '.txt'))
filtered2 = os.path.join(cwd, ('2_terminal_logs_filtered-' + date + '.txt'))

timestamps = []
frames = []

with open(filtered1, 'w', newline='') as c:
    with open(logFile1) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if 'skipping' in line:
                    line = f.readline()
                elif ('[ INFO]' in line and '/video_source' in line):
                    #print(line.strip('\n').split(' '))

                    frame = line.strip('\n').split(' ')[5]
                    timestamp = line.strip('\n').split(' ')[2].replace('.', '').replace('[','')[:14]

                    c.write(frame + ',' + timestamp + '\n')

                    frames.append(frame)
                    timestamps.append(timestamp)
print(len(frames))    
print(len(timestamps))
print('Duplicate Timestamps in Cam1: '+ str(set([x for x in timestamps if timestamps.count(x) > 1])))

timestamps = []
frames = []

with open(filtered2, 'w', newline='') as c:
    with open(logFile2) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if 'skipping' in line:
                    line = f.readline()
                elif ('[ INFO]' in line and '/video_source' in line):
                    #print(line.strip('\n').split(' '))

                    frame = line.strip('\n').split(' ')[5]
                    timestamp = line.strip('\n').split(' ')[2].replace('.', '').replace('[','')[:14]

                    c.write(frame + ',' + timestamp + '\n')

                    frames.append(frame)
                    timestamps.append(timestamp)

print(len(frames))    
print(len(timestamps))
print('Duplicate Timestamps in Cam2: '+ str(set([x for x in timestamps if timestamps.count(x) > 1])))