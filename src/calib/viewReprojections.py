from math import atan
import os
import pickle
from turtle import position
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from utils import load_scene
import cv2

def load_pickle(pickle_file) -> Dict:
    """
    Loads a .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def pt3d_to_2d(x, y, z, K, D, R, t):
    x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
    y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
    z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
    # project onto camera plane
    a = x_2d / z_2d
    b = y_2d / z_2d
    # fisheye params
    r = (a ** 2 + b ** 2 + 1e-12) ** 0.5
    th = atan(r)
    # distortion
    th_D = th * (1 + D[0] * th ** 2 + D[1] * th ** 4 + D[2] * th ** 6 + D[3] * th ** 8)
    x_P = a * th_D / r
    y_P = b * th_D / r
    u = K[0, 0] * x_P + K[0, 2]
    v = K[1, 1] * y_P + K[1, 2]
    return u, v

def np_rot_y(y):
    c = np.cos(y)
    s = np.sin(y)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

cwd = "C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\11Oct2022S\\"
vid_dir = "C:\\Users\\user-pc\\Desktop\\11Oct2022S"
vid_path1 = os.path.join(vid_dir, "1_trimmed.avi")
vid_path2 = os.path.join(vid_dir, "2_trimmed.avi")
results = os.path.join(cwd, 'results\\traj_results.pickle')
scene_path = os.path.join(cwd, "extrinsic_calib", "2_cam_scene_sba.json")
encoder_path = os.path.join(cwd, "synced_data.pkl")

#Load 3D Points from trajectory optimization
opt_results = load_pickle(results)

positions =  np.array([[
    [0.365857,  6.890354, -0.822666],
    [0.062193,  6.857276, -0.792951],
    [0.263661,  6.627421,  0.654366],
    [0.616712,  7.046775,  0.341119],
    [-0.066443,  6.639662,  0.359051],
    [0.26898,  6.636344,  0.816441],
    [0.356712,  6.70816, -0.06091],
    [0.098365,  6.704401, -0.051397],
    [0.373637,  6.857199, -0.484482],
    [0.06255,  6.695557, -0.460909],
    [0.261187,  6.572296,  0.573736],
    [0.457773,  6.759311,  0.576362],
    [0.078325,  6.570917,  0.585779],
    [0.472888,  6.699009,  0.12238],
    [0.001705,  6.733185,  0.149866]
]])
positions = opt_results['positions']

print(positions.shape)

#Open encoder files for R and t changes
with open(encoder_path, 'rb') as handle:
    synced_data = pickle.load(handle)

enc1 = np.reshape(synced_data['enc1tick'], (-1, 1))
enc1 = np.reshape(synced_data['enc1tick'][:7357], (-1, 1))
enc2 = np.reshape(synced_data['enc2tick'], (-1, 1))
encoder_arr = np.hstack((enc1, enc2))

#Load Camera intrinsics and initial extrinsics
K_arr, D_arr, R_arr, t_arr, _ = load_scene(scene_path)
D_arr = D_arr.reshape((-1, 4))

#
frame_num = 0
cap1 = cv2.VideoCapture(vid_path1)
cap2 = cv2.VideoCapture(vid_path2)
for i,frame in enumerate(positions):
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame1 = cap1.read()

    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame2 = cap2.read()

    K1, D1, R1, t1 = K_arr[0], D_arr[0], R_arr[0], t_arr[0]
    R1 =  np_rot_y(encoder_arr[i, 0]).T @ R1
    t1 =  np_rot_y(encoder_arr[i, 0]).T @ t1

    K2, D2, R2, t2 = K_arr[1], D_arr[1], R_arr[1], t_arr[1]
    R2 =  np_rot_y(encoder_arr[i, 1]).T @ R2
    t2 =  np_rot_y(encoder_arr[i, 1]).T @ t2

    for point in frame:
        u1, v1 = pt3d_to_2d(point[0], point[1], point[2], K1, D1, R1, t1)
        frame1 = cv2.circle(frame1, (int(u1),int(v1)), radius=5, color=(0, 0, 255), thickness=-1)

        u2, v2 = pt3d_to_2d(point[0], point[1], point[2], K2, D2, R2, t2)
        frame2 = cv2.circle(frame2, (int(u2),int(v2)), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    
    key = cv2.waitKey()
    if key == ord('q'):
        break

    print(i)


