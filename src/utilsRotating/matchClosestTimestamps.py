import os
import re
import pickle

cwd = 'C:\\Users\\user-pc\\Desktop\\22Sep2022'
date = '2022_09_22_05_46'

encFile1 = os.path.join(cwd, ('1_encoder_vals-' + date + '.txt'))
encFile2 = os.path.join(cwd, ('2_encoder_vals-' + date + '.txt'))

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
        encTS1.append(int(val[1].replace('.', '').replace(' ', '')))

with open(encFile2) as tsF:
    file_content = tsF.read()
content_list = re.split('\n', file_content)
tsF.close()

encVals2, encTS2 = [], []
for item in content_list:
    if len(item) > 2:
        val = re.split(',', item)
        encVals2.append(int(val[0]))
        encTS2.append(int(val[1].replace('.', '').replace(' ', '')))

corr_timestamps1 = list(map(lambda y: min(encTS1, key=lambda x: abs(x - y)), f1))
corr_timestamps2 = list(map(lambda y: min(encTS2, key=lambda x: abs(x - y)), f1))

# Create new list with [f1, closest Match in f2, abs(offset)]
matches = []

for el1, el2, el3, el4 in zip(f1, corr_frames, corr_timestamps1, corr_timestamps2):
    matches.append(
        [el1, el2, abs(el1 - el2), el3, encVals1[encTS1.index(el3)], abs(el1 - el3), el4, encVals2[encTS2.index(el4)],
         abs(el1 - el4)])
print(matches)
print(len(matches))
#with open(os.path.join(cwd, 'synced_data.pkl'), 'wb') as f:
    #pickle.dump(matches, f)
