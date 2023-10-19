from math import atan
import os
import pickle
from turtle import position
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy.io as sio

import tkinter as tk
from tkinter import filedialog, simpledialog

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


root = tk.Tk()
root.withdraw()  # Hide the main window

results = filedialog.askopenfilename(
    title="Select a Pickle File",
    filetypes=[("Pickle files", "*.pickle")]
)
if not results:
    print("No file chosen. Exiting...")
    exit()

cwd = os.path.dirname(results)
cwd = cwd.replace('results','').replace('/', '\\')

start_frame = simpledialog.askinteger("Start Frame", "Enter the start frame:")
if not start_frame:
    start_frame = 0
fps = 10

vid_path1 = os.path.join(cwd, "1.avi")
vid_path2 = os.path.join(cwd, "2.avi")

encoder_path = os.path.join(cwd, "synced_data.pkl")
camera_params_path = os.path.join(cwd, 'extrinsics.mat') 
motor_camera_params_path = os.path.join(cwd, "cam_motor_transformations.mat")

out_path = os.path.join(cwd, f"results\\reprojections-{start_frame}.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(out_path, fourcc, fps, (1920, 540))

#Load Camera intrinsics and initial extrinsics from matlab mat file
mat_contents = sio.loadmat(camera_params_path)

#Load 3D Points from trajectory optimization
opt_results = load_pickle(results)
positions = opt_results['positions']# positions =[[[ 0.5,    -0.35,    5]]] * 500
print(positions)

#Open encoder files for R and t changes
with open(encoder_path, 'rb') as handle:
    synced_data = pickle.load(handle)

enc1 = np.reshape(synced_data['enc1tick'], (-1, 1))
enc2 = np.reshape(synced_data['enc2tick'], (-1, 1))
encoder_arr = np.hstack((enc1, enc2))

estEnc = np.reshape(opt_results['x'][:, 42:44], (-1,2))

#Load Camera intrinsics and initial extrinsics
K_arr = np.array([mat_contents['k1'], mat_contents['k2']])
D_arr = np.array([mat_contents['d1'][0][0:4], mat_contents['d2'][0][0:4]])
R_arr = np.array([mat_contents['r1'], mat_contents['r2']])
t_arr = np.array([mat_contents['t1'][0], mat_contents['t2'][0]])

# Camera<->Motor Transformations
#mat_contents = sio.loadmat(motor_camera_params_path)

#R_CM_arr = np.array([mat_contents['R_CM1'], mat_contents['R_CM2']])
#R_MC_arr = np.array([mat_contents['R_MC1'], mat_contents['R_MC2']])
#t_CM_arr = np.array([mat_contents['t_CM1'][0], mat_contents['t_CM2'][0]])
#t_MC_arr = np.array([mat_contents['t_MC1'][0], mat_contents['t_MC2'][0]])

R_CM_arr = np.array([np_rot_y(0) @ np_rot_x(0*0.00174533) @ np_rot_z(0*0.00174533), 
                         np_rot_y(0) @ np_rot_x(0) @ np_rot_z(0)])     
c_CM_arr = np. array([[ 0.01,  0.00, 0.00,],
                        [0.00,  0.00,  0.00,]])
t_CM_arr = np.array([-R_CM_arr[0] @ c_CM_arr[0],
                    -R_CM_arr[1] @ c_CM_arr[1]])

R_MC_arr = np.array([R_CM_arr[0].T, 
                        R_CM_arr[1].T])
c_MC_arr = np.array([-R_CM_arr[0] @ c_CM_arr[0], 
                    -R_CM_arr[1] @ c_CM_arr[1]])
t_MC_arr = np.array([-R_MC_arr[0] @ c_MC_arr[0],
                    -R_MC_arr[1] @ c_MC_arr[1]])

#X, Z
frame_num = start_frame
cap1 = cv2.VideoCapture(vid_path1)
cap2 = cv2.VideoCapture(vid_path2)
count = 0

for i, frame in enumerate(positions):
    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame1 = cap1.read()

    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame2 = cap2.read()

    K1, D1, R1, t1, R_CM1, t_CM1, R_MC1, t_MC1 = K_arr[0], D_arr[0], R_arr[0], t_arr[0], R_CM_arr[0], t_CM_arr[0], R_MC_arr[0], t_MC_arr[0]
    
    #Convert Coordinate Transformations to homogenous coordinates
    t1 = t1[:, np.newaxis]
    R_i_homog1 = np.vstack([np.hstack([R1, t1]), [0, 0, 0, 1]])

    t_CM1 = t_CM1[:, np.newaxis]
    R_CM_homog1 = np.vstack([np.hstack([R_CM1, t_CM1]), [0, 0, 0, 1]])

    R_MM1 = np_rot_y(estEnc[i, 0]).T
    t_MM1 = np.array([0,0,0])
    t_MM1 = t_MM1[:, np.newaxis]
    R_MM_homog1 = np.vstack([np.hstack([R_MM1, t_MM1]), [0, 0, 0, 1]])

    t_MC1 = t_MC1[:, np.newaxis]
    R_MC_homog1 = np.vstack([np.hstack([R_MC1, t_MC1]), [0, 0, 0, 1]])

    R_homog1 = R_MC_homog1 @ R_MM_homog1 @ R_CM_homog1 @ R_i_homog1
    #R_homog1 = R_MM_homog1 @ R_i_homog1
    
    R1 = R_homog1[:3, :3]
    t1 = R_homog1[:3, 3]

    print('Alpha:' + str(count_to_rad(encoder_arr[frame_num, 0])))
    print('Estimated Alpha: ' + str(estEnc[frame_num-start_frame,0]))

    K2, D2, R2, t2, R_CM2, t_CM2, R_MC2, t_MC2 = K_arr[1], D_arr[1], R_arr[1], t_arr[1], R_CM_arr[1], t_CM_arr[1], R_MC_arr[1], t_MC_arr[1]
    
    t2 = t2[:, np.newaxis]
    R_i_homog2 = np.vstack([np.hstack([R2, t2]), [0, 0, 0, 1]])

    t_CM2 = t_CM2[:, np.newaxis]
    R_CM_homog2 = np.vstack([np.hstack([R_CM2, t_CM2]), [0, 0, 0, 1]])

    R_MM2 = np_rot_y(estEnc[i, 1]).T
    t_MM2 = np.array([0,0,0])
    t_MM2 = t_MM2[:, np.newaxis]
    R_MM_homog2 = np.vstack([np.hstack([R_MM2, t_MM2]), [0, 0, 0, 1]])

    t_MC2 = t_MC2[:, np.newaxis]
    R_MC_homog2 = np.vstack([np.hstack([R_MC2, t_MC2]), [0, 0, 0, 1]])

    #R_homog2 = R_MC_homog2 @ R_MM_homog2 @ R_CM_homog2 @ R_i_homog2
    R_homog2 = R_MM_homog2 @ R_i_homog2
    
    R2 = R_homog2[:3, :3]
    t2 = R_homog2[:3, 3]

    print('Beta:' + str(count_to_rad(encoder_arr[frame_num, 1])))
    print('Estimated Beta: ' + str(estEnc[frame_num-start_frame,1]))
    
    print(f"Frame: {frame_num}")
    count += 1

    for point in frame:
        u1, v1 = pt3d_to_2d(point[0], point[1], point[2], K1, D1, R1, t1)
        frame1 = cv2.circle(frame1, (int(u1),int(v1)), radius=5, color=(0, 0, 255), thickness=-1)

        u2, v2 = pt3d_to_2d(point[0], point[1], point[2], K2, D2, R2, t2)
        frame2 = cv2.circle(frame2, (int(u2),int(v2)), radius=5, color=(0, 0, 255), thickness=-1)

    #cv2.imshow('Frame 1', frame1)
    #cv2.imshow('Frame 2', frame2)
    
    resize_scale = 0.75
    frame1_rs = cv2.resize(frame1, (0, 0), None, resize_scale, resize_scale)
    frame2_rs = cv2.resize(frame2, (0, 0), None, resize_scale, resize_scale)

    frames = np.hstack((frame1_rs, frame2_rs))
    cv2.imshow('Reprojections', frames)
    out.write(frames)

    key = cv2.waitKey()
    if key == ord('q'):
        out.release()
        break

    frame_num += 1

    print(frame_num)

out.release()