import pickle
import numpy as np
from typing import Dict
import os

file = "C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\Rig\\synced_data.pkl"


def load_pickle(pickle_file) -> Dict:
    """
    Loads a .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)

    return data

start_frame = 1420
contents = load_pickle(file)


c1ts = contents['c1ts'][start_frame:]
c2ts = contents['c2ts'][start_frame:]
c1c2offset = contents['c1c2offset'][start_frame:]
encts = contents['encts'][start_frame:]
enc1tick = contents['enc1tick'][start_frame:]
c1enc1offset = contents['c1enc1offset'][start_frame:]
enc2ts = contents['enc2ts'][start_frame:]
enc2tick = contents['enc2tick'][start_frame:]
c1enc2offset = contents['c1enc2offset'][start_frame:]


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

#with open('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\Rig\\synced_data_trimmed.pkl', 'wb') as f:
#    pickle.dump(synced_dict, f)
