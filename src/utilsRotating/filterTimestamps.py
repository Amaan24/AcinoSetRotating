import collections
import csv
import os
from time import time


cwd = 'C:\\Users\\user-pc\\Desktop\\22Sep2022'
date = '2022_09_22_05_46'

timestamp1 = os.path.join(cwd, ('1_timestamps-' + date + '.txt'))
timestamp2 = os.path.join(cwd, ('2_timestamps-' + date + '.txt'))

filtered1 = os.path.join(cwd, ('1_timestamps_filtered-' + date + '.txt'))
filtered2 = os.path.join(cwd, ('2_timestamps_filtered-' + date + '.txt'))

timestamps = []
frames = []

with open(filtered1, 'w', newline='') as c:
    with open(timestamp1) as f:
            while True:
                line = f.readline().replace('\n', '').split(',')
        
                if not line:
                    break
                
                if len(line) < 2:
                    break

                frame = line[0]
                timestamp = line[1]

                if len(timestamp) < 19:
                    temp1 = timestamp[:10]
                    temp2 = timestamp[10:]
                    zeroes = '0'*(19-len(timestamp))
                    timestamp = temp1 + zeroes + temp2

                c.write(frame + ',' + timestamp + '\n')

                frames.append(frame)
                timestamps.append(timestamp)
print(len(frames))    
print(len(timestamps))
print('Duplicate Timestamps in Cam1: '+ str(set([x for x in timestamps if timestamps.count(x) > 1])))


timestamps = []
frames = []

with open(filtered2, 'w', newline='') as c:
    with open(timestamp2) as f:
            while True:
                line = f.readline().replace('\n', '').split(',')
        
                if not line:
                    break
                
                if len(line) < 2:
                    break

                frame = line[0]
                timestamp = line[1]

                if len(timestamp) < 19:
                    temp1 = timestamp[:10]
                    temp2 = timestamp[10:]
                    zeroes = '0'*(19-len(timestamp))
                    timestamp = temp1 + zeroes + temp2

                c.write(frame + ',' + timestamp + '\n')

                frames.append(frame)
                timestamps.append(timestamp)
print(len(frames))    
print(len(timestamps))
print('Duplicate Timestamps in Cam2: '+ str(set([x for x in timestamps if timestamps.count(x) > 1])))