import numpy as np
import json
import calib.plotting as plotting
import calib.app as app
import calib.utils as utils

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

R = np.array([[1,  0, -0],
    [ 0,  1,  0],
    [ 0,  0,  1]]
    )
t = np.array([[0.00000000e+00],
 [-4.83642408e-06],
 [0.00000000e+00]])

print(R @ t)