from math import atan
import os
import pickle
from turtle import position
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from utils import load_scene
import cv2
import scipy.io as sio

def load_pickle(pickle_file) -> Dict:
    """
    Loads a .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)
    return data


def pt3d_to_2d_fisheye(x, y, z, K, D, R, t):
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

def pt3d_to_2d(x, y, z, K, D, R, t):
    x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
    y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
    z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
    # project onto camera plane
    a = x_2d / z_2d
    b = y_2d / z_2d

    # distortion
    r = (a ** 2 + b ** 2) ** 0.5
    thD = (1 + D[0] * r**2 + D[1] * r**4)
    
    x_P = a * thD
    y_P = b * thD
    u = K[0, 0] * x_P + K[0, 2]
    v = K[1, 1] * y_P + K[1, 2]
    return u, v

def pt3d_to_2d_rotating(x, y, z, K, D):    
    # project onto camera plane
    a = x / z
    b = y / z

    # distortion
    r = (a ** 2 + b ** 2 + 1e-12) ** 0.5
    th_D = (1 + D[0] * r**2 + D[1] * r**4)
    
    x_P = a * th_D
    y_P = b * th_D
    
    u = K[0, 0] * x_P + K[0, 2]
    v = K[1, 1] * y_P + K[1, 2]
    return u, v

def np_rot_x(x):
    c = np.cos(x)
    s = np.sin(x)
    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])

def np_rot_y(y):
    c = np.cos(y)
    s = np.sin(y)
    return np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])

def np_rot_z(z):
    c = np.cos(z)
    s = np.sin(z)
    return np.array([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])

def count_to_rad(enc_count):
    """
    Returns the given encoder count as its equivalent angle in radians
    """
    ang = enc_count * 2 * np.pi / 102000
    return ang

cwd = "C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\FinalHumanRig\\"
cwd = "C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\FinalHumanGoPro\\"
#vid_dir = "C:\\Users\\user-pc\\Desktop\\24Nov2022"
vid_dir = "C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\Rig"
vid_dir = "C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro"
vid_path1 = os.path.join(vid_dir, "1_Synced.avi")
vid_path2 = os.path.join(vid_dir, "2_Synced.avi")
vid_path1 = os.path.join(vid_dir, "GPLSynced.avi")
vid_path2 = os.path.join(vid_dir, "GPRSynced.avi")
results = os.path.join(cwd, 'results\\traj_results.pickle')
scene_path = os.path.join(cwd, "extrinsic_calib", "2_cam_scene_sba.json")
#encoder_path = os.path.join(cwd, "synced_data.pkl")

#Load 3D Points from trajectory optimization
opt_results = load_pickle(results)
#positions = [[[0, , 0]]]*5000
positions = opt_results['positions']
print(positions)
#Open encoder files for R and t changes
#with open(encoder_path, 'rb') as handle:
#    synced_data = pickle.load(handle)

#enc1 = np.reshape(synced_data['enc1tick'], (-1, 1))
#enc2 = np.reshape(synced_data['enc2tick'], (-1, 1))
#encoder_arr = np.hstack((enc1, enc2))
encoder_arr = np.zeros((25000,2))
estEnc = np.reshape(opt_results['x'][:, 42:44], (-1,2))

#Load Camera intrinsics and initial extrinsics
#K_arr, D_arr, R_arr, t_arr, _ = load_scene(scene_path)

#Load Camera intrinsics and initial extrinsics from matlab mat file
mat_contents = sio.loadmat('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\extrinsicsRigLRigR.mat')
mat_contents = sio.loadmat('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\extrinsicsGPLGPR.mat')

K_arr = np.array([mat_contents['k1'], mat_contents['k2']])
D_arr = np.array([mat_contents['d1'][0][0:4], mat_contents['d2'][0][0:4]])
R_arr = np.array([mat_contents['r1'], mat_contents['r2']])
t_arr = np.array([mat_contents['t1'][0], mat_contents['t2'][0]])
print(t_arr)

#start_frame = 11350
start_frame = 11475 
frame_num = start_frame
cap1 = cv2.VideoCapture(vid_path1)
cap2 = cv2.VideoCapture(vid_path2)
count = 0
for frame in positions:
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame1 = cap1.read()

    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame2 = cap2.read()

    K1, D1, R1, t1 = K_arr[0], D_arr[0], R_arr[0], t_arr[0]
    R1 =  np_rot_y(count_to_rad(encoder_arr[frame_num, 0])).T @ R1 
    t1 =  np_rot_y(count_to_rad(encoder_arr[frame_num, 0])).T @ t1
    print('Alpha:' + str(count_to_rad(encoder_arr[frame_num, 0])))

    R1_slack = np.array(opt_results['rot_slack'][count][0]).reshape((3,3))
    t1_slack = np.array(opt_results['trans_slack'][count][0]).reshape((1,3))

    #R1 = R1 + R1_slack
    #t1 =  t1 + t1_slack

    #print('Estimated Alpha: ' + str(estEnc[frame_num-start_frame,0]))

    K2, D2, R2, t2 = K_arr[1], D_arr[1], R_arr[1], t_arr[1]
    R2 =   np_rot_y(count_to_rad(encoder_arr[frame_num, 1])).T @ R2 
    t2 =   np_rot_y(count_to_rad(encoder_arr[frame_num, 1])).T @ t2
    print('Beta:' + str(count_to_rad(encoder_arr[frame_num, 1])))
    #print('Estimated Beta: ' + str(estEnc[frame_num-start_frame,1]))
    
    R2_slack = np.array(opt_results['rot_slack'][count][1]).reshape((3,3))
    t2_slack = np.array(opt_results['trans_slack'][count][1]).reshape((1,3))

    #R2 = R2 + R2_slack
    #t2 =  t2 + t2_slack

    count += 1

    for point in frame:
        u1, v1 = pt3d_to_2d_fisheye(point[0], point[1], point[2], K1, D1, R1, t1)
        frame1 = cv2.circle(frame1, (int(u1),int(v1)), radius=5, color=(0, 0, 255), thickness=-1)

        u2, v2 = pt3d_to_2d_fisheye(point[0], point[1], point[2], K2, D2, R2, t2)
        frame2 = cv2.circle(frame2, (int(u2),int(v2)), radius=5, color=(0, 0, 255), thickness=-1)

    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    
    key = cv2.waitKey()
    if key == ord('q'):
        break

    frame_num += 1

    print(frame_num)


