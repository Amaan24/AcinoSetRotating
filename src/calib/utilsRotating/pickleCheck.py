import pickle
import numpy as np
from typing import Dict
import os

# cwd = 'C:\\Users\\user-pc\\Desktop\\19Aug2022'
file = "C:\\Users\\user-pc\\Desktop\\11Oct2022S\\synced_data_trimmed.pkl"


# file = "C:\\Users\\user-pc\\Desktop\\AcinoSet\\skeletons\\new_human.pickle"
# file = 'C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\19Aug2022\\results\\traj_results.pickle'


def load_pickle(pickle_file) -> Dict:
    """
    Loads a .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return data


contents = load_pickle(file)

#contents = np.array(contents)

print(len(contents['enc1tick']))
#print(contents['enc2tick'])
#for i in range(0, len(contents['enc1tick'])):
for i in range(0, 500):
    print(i, contents['enc1tick'][i], contents['enc2tick'][i])


# print(max(contents['enc1tick']))
# print(min(contents['enc1tick']))
# print(max(contents['enc2tick']))
# print(min(contents['enc2tick']))

'''
c1ts = contents['c1ts'][9000:]
c2ts = contents['c2ts'][9000:]
c1c2offset = contents['c1c2offset'][9000:]
encts = contents['encts'][9000:]
enc1tick = contents['enc1tick'][9000:]
c1enc1offset = contents['c1enc1offset'][9000:]
enc2ts = contents['enc2ts'][9000:]
enc2tick = contents['enc2tick'][9000:]
c1enc2offset = contents['c1enc2offset'][9000:]


synced_dict = {'c1ts': c1ts,
               'c2ts': c2ts,
               'c1c2offset': c1c2offset,
               'encts': encts,
               'enc1tick': enc1tick,
               'c1enc1offset': c1enc1offset,
               'enc2ts': enc2ts,
               'enc2tick': enc2tick,
               'c1enc2offset': c1enc2offset
               }

with open('C:\\Users\\user-pc\\Desktop\\11Oct2022S\\synced_data_trimmed.pkl', 'wb') as f:
    pickle.dump(synced_dict, f)
'''