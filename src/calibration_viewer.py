import numpy as np
import json
import lib.plotting as plotting
import lib.app as app
import lib.utils as utils

scene_path = 'C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\11Oct2022S\\extrinsic_calib\\2_cam_scene_sba.json'

K_arr, D_arr, R_arr, t_arr, _ = utils.load_scene(scene_path)
D_arr = D_arr.reshape((-1, 4))

print("Cam 1:") 
print("R: " + str(R_arr[0]))
print("t: " + str(t_arr[0]))

# t = -RC so -R't = C
C1 = -R_arr[0].T @ t_arr[0]
print("C: " + str(C1))

#print("D: " + str(D_arr[0]))
#print("K: " + str(K_arr[0]))

print("Cam 2:")
print("R: " + str(R_arr[1]))
print("t: " + str(t_arr[1]))
#print("D: " + str(D_arr[1]))
#print("K: " + str(K_arr[1]))

# t = -RC so -R't = C
C2 = -R_arr[1].T @ t_arr[1]
print("C: " + str(C2))

