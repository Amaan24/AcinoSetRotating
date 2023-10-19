from cmath import pi
from heapq import merge
from operator import concat
from sqlite3 import dbapi2
from this import d
from typing import Dict
import pickle
from pyomo.core.base.constraint import Constraint, ConstraintList
import sympy as sp
import numpy as np
import os
import glob
from lib import utils, calib, plotting, app, extract
from scipy import stats
from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.PyomoModel import ConcreteModel
from argparse import ArgumentParser
import lib.utilsRotating.pan_compensation as pc
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime

pose_to_3d = []


def build_model(project_dir) -> ConcreteModel:
    """
    Builds a pyomo human model 
    """  
    L = 13  # number of joints in the human model

    markers = ["forehead", "chin", "neck", "shoulder1", "elbow1", "wrist1", "shoulder2", "elbow2",
                "wrist2", "pelvis", "hip1", "hip2", "knee1","ankle1", "knee2", "ankle2"]

    # defines arrays of angles, velocities and accelerations
    phi = [sp.symbols(f"\\phi_{{{l}}}") for l in range(L)]
    theta = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
    psi = [sp.symbols(f"\\psi_{{{l}}}") for l in range(L)]


    ## Rotations
    RI_0 = rot_z(psi[0]) @ rot_x(phi[0]) @ rot_y(theta[0]) # forehead to inertial
    R0_I = RI_0.T

    RI_1 = rot_z(psi[1]) @ rot_x(phi[1]) @ rot_y(theta[1]) @ RI_0 # neck to forehead
    R1_I = RI_1.T

    RI_2 = rot_z(psi[2]) @ rot_x(phi[2]) @ RI_1 # L Shoulder to inertial
    R2_I = RI_2.T
    RI_3 = rot_z(psi[3]) @ rot_x(phi[3]) @ rot_y(theta[3]) @ RI_2 # L Elbow to inertial
    R3_I = RI_3.T
    RI_4 = rot_y(theta[4]) @ RI_3 # L wrist to inertial
    R4_I = RI_4.T

    RI_5 = rot_z(psi[5]) @ rot_x(phi[5]) @ RI_1 # R Shoulder to inertial
    R5_I = RI_5.T
    RI_6 = rot_z(psi[6]) @ rot_x(phi[6]) @ rot_y(theta[6]) @ RI_5 # L Elbow to inertial
    R6_I = RI_6.T
    RI_7 = rot_y(theta[7]) @ RI_6 # R wrist to inertial
    R7_I = RI_7.T

    RI_8 = rot_z(psi[8]) @ RI_1 # Pelvis to inertial
    R8_I = RI_8.T

    RI_9 = rot_z(psi[9]) @ rot_x(phi[9]) @ rot_y(theta[9]) @ RI_8 # L Knee to inertial
    R9_I = RI_9.T

    RI_10 = rot_y(theta[10]) @ RI_9 # L Ankle to inertial
    R10_I = RI_10.T

    RI_11 = rot_z(psi[11]) @ rot_x(phi[11]) @ rot_y(theta[11]) @ RI_8 # R Knee to inertial
    R11_I = RI_11.T

    RI_12 = rot_y(theta[12]) @ RI_11 # R Ankle to inertial
    R12_I = RI_12.T

    # defines the position, velocities and accelerations in the inertial frame
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")
    
    # SYMBOLIC CHEETAH POSE POSITIONS
    p_head         = sp.Matrix([x, y, z])
    p_chin         = p_head         + R0_I  @ sp.Matrix([0, 0, -0.22])
    p_neck         = p_head         + R0_I  @ sp.Matrix([-0.1, 0, -0.25])
    p_shoulder1    = p_neck         + R2_I  @ sp.Matrix([0, 0.18, 0])
    p_elbow1       = p_shoulder1    + R3_I  @ sp.Matrix([0, 0, -0.28])
    p_wrist1       = p_elbow1       + R4_I  @ sp.Matrix([0, 0, -0.25])
    p_shoulder2    = p_neck         + R5_I  @ sp.Matrix([0, -0.18, 0])
    p_elbow2       = p_shoulder2    + R6_I  @ sp.Matrix([0, 0, -0.28])
    p_wrist2       = p_elbow2       + R7_I  @ sp.Matrix([0, 0, -0.25])
    p_pelvis       = p_neck         + R8_I  @ sp.Matrix([0, 0, -0.5])
    p_hip1         = p_pelvis       + R8_I  @ sp.Matrix([0, 0.10, 0])
    p_knee1        = p_hip1         + R9_I  @ sp.Matrix([0, 0, -0.44])
    p_ankle1       = p_knee1        + R10_I @ sp.Matrix([0, 0, -0.42])
    p_hip2         = p_pelvis       + R8_I  @ sp.Matrix([0, -0.10, 0])
    p_knee2        = p_hip2         + R11_I @ sp.Matrix([0, 0, -0.44])
    p_ankle2       = p_knee2        + R12_I @ sp.Matrix([0, 0, -0.42])

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    positions = sp.Matrix([p_head.T, p_chin.T, p_neck.T, p_shoulder1.T, p_elbow1.T, p_wrist1.T, 
                           p_shoulder2.T, p_elbow2.T, p_wrist2.T, p_pelvis.T, p_hip1.T, p_hip2.T,
                           p_knee1.T, p_ankle1.T, p_knee2.T, p_ankle2.T])

    func_map = {"sin": sin, "cos": cos, "ImmutableDenseMatrix": np.array}
    sym_list = [x, y, z, *phi, *theta, *psi]
    pose_to_3d = sp.lambdify(sym_list, positions, modules=[func_map])
    pos_funcs = []

    for i in range(positions.shape[0]):
        lamb = sp.lambdify(sym_list, positions[i, :], modules=[func_map])
        pos_funcs.append(lamb)

    encoder_path = os.path.join(project_dir, "data", args.project, "synced_data.pkl")
    with open(encoder_path, 'rb') as handle:
        synced_data = pickle.load(handle)

    enc1 = np.reshape(synced_data['enc1tick'], (-1, 1))
    enc2 = np.reshape(synced_data['enc2tick'], (-1, 1))
    encoder_arr = np.hstack((enc1, enc2))

    #Load Camera intrinsics and initial extrinsics from matlab mat file
    camera_params_path = os.path.join(project_dir, "data", args.project, "extrinsics.mat")
    mat_contents = sio.loadmat(camera_params_path)

    K_arr = np.array([mat_contents['k1'], mat_contents['k2']])
    D_arr = np.array([mat_contents['d1'][0][0:4], mat_contents['d2'][0][0:4]])
    R_arr = np.array([mat_contents['r1'], mat_contents['r2']])
    t_arr = np.array([mat_contents['t1'][0], mat_contents['t2'][0]])

    R_CM_arr = np.array([np_rot_y(0*0.00174533) @ np_rot_x(-0*0.00174533) @ np_rot_z(0*0.00174533), 
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
    

    print(f"\n\n\nLoading data")

    df_paths = sorted(glob.glob(os.path.join(project_dir, "data", args.project, '*.h5')))
    #points_2d_df = utils.create_dlc_points_2d_file(df_paths)
    points_2d_df = utils.create_dlc_points_2d_file_hand_labelled(df_paths)
    print("2d df points:")
    print(points_2d_df)

    def get_meas_from_df(n, c, l, d):
        n_mask = points_2d_df["frame"] == n - 1
        l_mask = points_2d_df["marker"] == markers[l - 1]
        c_mask = points_2d_df["camera"] == c - 1
        d_idx = {1: "x", 2: "y"}
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val[d_idx[d]].values[0]

    def get_enc_meas(n, c):
        return encoder_arr[n - 1, c]

    def get_likelihood_from_df(n, c, l):
        n_mask = points_2d_df["frame"] == n - 1
        l_mask = points_2d_df["marker"]== markers[l-1]
        c_mask = points_2d_df["camera"] == c - 1
        val = points_2d_df[n_mask & l_mask & c_mask]
        return val["likelihood"].values[0]

    h = 1 / 60  # timestep: 1/framerate
    start_frame = args.start_frame  
    N = args.end_frame - args.start_frame
    C = len(K_arr)
    P = 3 + len(phi) + len(theta) + len(psi)
    Pc = P + C
    L = len(pos_funcs)
    D2 = 2 
    D3 = 3 

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]

    triangulate_func = calib.triangulate_points_fisheye_rotating #Fix!!
    points_2d_filtered_df = points_2d_df[points_2d_df['likelihood'] > 0.4]

    #points_3d_df = calib.get_pairwise_3d_points_from_df_rotating(points_2d_filtered_df, K_arr, D_arr, R_arr, t_arr, encoder_arr,
    #                                                    triangulate_func) #Fix!!
    #print("3d points")
    #print(points_3d_df)

    # estimate initial points
    # Sets the initial points uding triangulation                                                        
    #forehead_points = points_3d_df[points_3d_df["marker"] == "forehead"][["x", "y", "z", "frame" ]].values
    frame_est = np.arange(N)

    #x_est = np.array([forehead_points[start_frame + i][0]  for i in range(len(frame_est))])
    #y_est = np.array([forehead_points[start_frame + i][1]  for i in range(len(frame_est))])
    #z_est = np.array([forehead_points[start_frame + i][2]  for i in range(len(frame_est))])

    x_est = np.array([0  for i in range(len(frame_est))])
    y_est = np.array([6  for i in range(len(frame_est))])
    z_est = np.array([0.7  for i in range(len(frame_est))])

    alpha_est = np.array([pc.count_to_rad(encoder_arr[start_frame + i][0]) for i in range (len(frame_est))])
    beta_est = np.array([pc.count_to_rad(encoder_arr[start_frame + i][1]) for i in range (len(frame_est))])

    print("Started Optimisation")
    m = ConcreteModel(name="Skeleton")

    # ===== SETS =====
    m.N = RangeSet(N)  # number of timesteps in trajectory
    m.Pc = RangeSet(Pc)  # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n) incl cameras
    m.P = RangeSet(P) ## number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n) excl cameras
    m.L = RangeSet(L)  # number of labels
    m.C = RangeSet(C)  # number of cameras
    m.D2 = RangeSet(D2)  # dimensionality of measurements
    m.D3 = RangeSet(D3)  # dimensionality of measurements

    R = 0.5  # measurement standard deviation (hand labelled data)
    sigma_model_xyz = 0.01
    sigmal_model_angles = 22.36 #Largest in acinoset. COuld use refinement

    #CHeck DLC Covariance
    #Try with 10 handlabelled frame    
    def init_meas_weights(model, n, c, l):
        likelihood = get_likelihood_from_df(n + start_frame, c, l)
        if likelihood > 0.4:
            return 1 / R**2
        else:
            return 0
    m.meas_err_weight = Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True,
                              within=Any)  
    def init_model_weights(m, p):
        if p < 4: #XYZ
            return 1/sigma_model_xyz**2
        else:
            #return 0.002
            return 1/sigmal_model_angles**2 
    m.model_err_weight = Param(m.P, initialize=init_model_weights, within=Any)

    m.h = h

    def init_measurements_df(m, n, c, l, d2):
            return get_meas_from_df(n + start_frame, c, l, d2)
    m.meas = Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df, within=Any)
    
    def init_encoder_measurements(m, n, c):
            return pc.count_to_rad(get_enc_meas(n + start_frame, c - 1)) 
    m.meas_enc = Param(m.N, m.C, initialize=init_encoder_measurements, within=Any)

    # ===== VARIABLES =====
    m.x = Var(m.N, m.Pc)  # position # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n)
    m.dx = Var(m.N, m.P)  # velocity
    m.ddx = Var(m.N, m.P)  # acceleration

    m.poses = Var(m.N, m.L, m.D3)
    m.slack_model = Var(m.N, m.P, initialize=0.0)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0) #Update

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, Pc))
    init_x[:, 0] = x_est  # x
    init_x[:, 1] = y_est  # 
    init_x[:, 2] = z_est  # z

    init_x[:, -2] = alpha_est
    init_x[:, -1] = beta_est

    init_dx = np.zeros((N, P))
    init_ddx = np.zeros((N, P))

    for n in range(1, N + 1):
        for p in range(1, P + 1):
            if n < len(init_x):  # init using known values
                m.x[n, p].value = init_x[n - 1, p - 1]
                m.dx[n, p].value = init_dx[n - 1, p - 1]
                m.ddx[n, p].value = init_ddx[n - 1, p - 1]
            else:  # init using last known value
                m.x[n, p].value = init_x[-1, p - 1]
                m.dx[n, p].value = init_dx[-1, p - 1]
                m.ddx[n, p].value = init_ddx[-1, p - 1]
        for p in range(P + 1, Pc + 1):
            m.x[n, p].value = init_x[n - 1, p - 1]

        # init pose
        var_list = [m.x[n, p].value for p in range(1, P + 1)]####*********
        for l in range(1, L + 1):
            [pos] = pos_funcs[l - 1](*var_list)
            for d3 in range(1, D3 + 1):
                m.poses[n, l, d3].value = pos[d3 - 1]

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m, n, l, d3):
        # get 3d points
        var_list = [m.x[n, p] for p in range(1, P + 1)]
        [pos] = pos_funcs[l - 1](*var_list)
        return pos[d3 - 1] == m.poses[n, l, d3]
    m.pose_constraint = Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    def backwards_euler_pos(m, n, p):  # position
        if n > 1:
            if p < 4:
                return m.x[n, p] == m.x[n - 1, p] + m.h * m.dx[n, p]
            else:
                return m.x[n, p] == m.x[n - 1, p] + m.h * m.dx[n, p]
        else:
            return Constraint.Skip
    m.integrate_p = Constraint(m.N, m.P, rule=backwards_euler_pos)

    def backwards_euler_vel(m, n, p):  # velocity
        if n > 1:
            return m.dx[n, p] == m.dx[n - 1, p] + m.h * m.ddx[n, p]
        else:
            return Constraint.Skip
    m.integrate_v = Constraint(m.N, m.P, rule=backwards_euler_vel)

    # MODEL
    def constant_acc(m, n, p): #Subject
        if n > 1:
            return m.ddx[n, p] == m.ddx[n - 1, p] + m.slack_model[n, p]
        else:
            return Constraint.Skip
    m.constant_acc = Constraint(m.N, m.P, rule=constant_acc)


    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2):
        # Get camera parameters for current camera
        K, D, R, t = K_arr[c - 1], D_arr[c - 1], R_arr[c - 1], t_arr[c - 1]

        R_CM = R_CM_arr[c - 1]
        t_CM = t_CM_arr[c - 1]
        R_MC = R_MC_arr[c - 1]
        t_MC = t_CM_arr[c - 1]

        p = P + c 

        #Convert Coordinate Transformations to homogenous coordinates
        t = t[:, np.newaxis]
        R_i_homog = np.vstack([np.hstack([R, t]), [0, 0, 0, 1]]) 

        t_CM = t_CM[:, np.newaxis]
        R_CM_homog = np.vstack([np.hstack([R_CM, t_CM]), [0, 0, 0, 1]])

        R_MM = np_rot_y(m.x[n, p].value).T
        t_MM = np.array([0,0,0])
        t_MM = t_MM[:, np.newaxis]
        R_MM_homog = np.vstack([np.hstack([R_MM, t_MM]), [0, 0, 0, 1]])
 
        t_MC = t_MC[:, np.newaxis]
        R_MC_homog = np.vstack([np.hstack([R_MC, t_MC]), [0, 0, 0, 1]])

        #Calculate final Transformation
        R_homog = R_MC_homog @ R_MM_homog @ R_CM_homog @ R_i_homog
        #R_homog = R_MM_homog @ R_i_homog

        R = R_homog[:3, :3]
        t = R_homog[:3, 3]

        x, y, z = m.poses[n, l, 1], m.poses[n, l, 2], m.poses[n, l, 3]

        return proj_funcs[d2 - 1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0 #+m.meas = dlc points
    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule=measurement_constraints)

    def enc_measurement_constraints(m, n, p):
        if (p <= P):
            return Constraint.Skip
        elif (p == 43):
            return  m.x[n, p] - m.meas_enc[n, 1] == 0
        elif (p == 44):
            return  m.x[n, p] - m.meas_enc[n, 2] == 0
    m.enc_measurement = Constraint(m.N, m.Pc, rule=enc_measurement_constraints)

    
    #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
    ##SEPARATE abs values to seperate constraints
    #Forehead
    def head_phi_0(m,n):                                #X
        return m.x[n,4] <= np.pi
    m.head_phi_0 = Constraint(m.N, rule=head_phi_0)
    def head_phi_1(m,n):                                
        return m.x[n,4] >= -np.pi
    m.head_phi_1 = Constraint(m.N, rule=head_phi_1)

    def head_theta_0(m,n):                              #Y
        return m.x[n,17] <= np.pi
    m.head_theta_0 = Constraint(m.N, rule=head_theta_0)
    def head_theta_1(m,n):                              
        return m.x[n,17] >= -np.pi
    m.head_theta_1 = Constraint(m.N, rule=head_theta_1)

    def head_psi_0(m,n):                                #Z
        return m.x[n,30] <= np.pi
    m.head_psi_0 = Constraint(m.N, rule=head_psi_0)
    def head_psi_1(m,n):                                
        return m.x[n,30] >= -np.pi
    m.head_psi_1 = Constraint(m.N, rule=head_psi_1)

    #Neck
    def neck_phi_0(m,n):
       return m.x[n,5] <= np.pi/2
    m.neck_phi_0 = Constraint(m.N, rule=neck_phi_0)
    def neck_phi_1(m,n):
       return m.x[n,5] >= -np.pi/2
    m.neck_phi_1 = Constraint(m.N, rule=neck_phi_1)

    def neck_theta_0(m,n):
        return m.x[n,18] <= np.pi/2
    m.neck_theta_0 = Constraint(m.N, rule=neck_theta_0)
    def neck_theta_1(m,n):
        return m.x[n,18] >= -np.pi/2
    m.neck_theta_1 = Constraint(m.N, rule=neck_theta_1)

    def neck_psi_0(m,n):
        return m.x[n,31] <= np.pi/2
    m.neck_psi_0 = Constraint(m.N, rule=neck_psi_0)
    def neck_psi_1(m,n):
        return m.x[n,31] >= -np.pi/2
    m.neck_psi_1 = Constraint(m.N, rule=neck_psi_1)
    
    #Left Shoulder
    def l_shoulder_phi_0(m,n):
       return m.x[n,6] <= np.pi/4
    m.l_shoulder_phi_0 = Constraint(m.N, rule=l_shoulder_phi_0)
    def l_shoulder_phi_1(m,n):
       return m.x[n,6] >= 0
    m.l_shoulder_phi_1 = Constraint(m.N, rule=l_shoulder_phi_1)

    def l_shoulder_theta_0(m,n):
        return m.x[n,19] <= 0
    m.l_shoulder_theta_0 = Constraint(m.N, rule=l_shoulder_theta_0)
    def l_shoulder_theta_1(m,n):
        return m.x[n,19] >= 0
    m.l_shoulder_theta_1 = Constraint(m.N, rule=l_shoulder_theta_1)
    
    def l_shoulder_psi_0(m,n):
        return m.x[n,32] <= np.pi/4
    m.l_shoulder_psi_0 = Constraint(m.N, rule=l_shoulder_psi_0)
    def l_shoulder_psi_1(m,n):
        return m.x[n,32] >= -np.pi/4
    m.l_shoulder_psi_1 = Constraint(m.N, rule=l_shoulder_psi_1)

    #Left Elbow
    def l_elbow_phi_0(m,n):
       return m.x[n,7] <= np.pi
    m.l_elbow_phi_0 = Constraint(m.N, rule=l_elbow_phi_0)
    def l_elbow_phi_1(m,n):
       return m.x[n,7] >= -np.pi
    m.l_elbow_phi_1 = Constraint(m.N, rule=l_elbow_phi_1)

    def l_elbow_theta_0(m,n):
        return m.x[n,20] <= np.pi
    m.l_elbow_theta_0 = Constraint(m.N, rule=l_elbow_theta_0)
    def l_elbow_theta_1(m,n):
        return m.x[n,20] >= -np.pi
    m.l_elbow_theta_1 = Constraint(m.N, rule=l_elbow_theta_1)    

    def l_elbow_psi_0(m,n):
        return m.x[n,33] <= np.pi/2
    m.l_elbow_psi_0 = Constraint(m.N, rule=l_elbow_psi_0)
    def l_elbow_psi_1(m,n):
        return m.x[n,33] >= -np.pi/2
    m.l_elbow_psi_1 = Constraint(m.N, rule=l_elbow_psi_1)
    
    #Left wrist
    def l_wrist_phi_0(m,n):
       return m.x[n,8] <= np.pi
    m.l_wrist_phi_0 = Constraint(m.N, rule=l_wrist_phi_0)
    def l_wrist_phi_1(m,n):
       return m.x[n,8] >= -np.pi
    m.l_wrist_phi_1 = Constraint(m.N, rule=l_wrist_phi_1)

    def l_wrist_theta_0(m,n):
        return m.x[n,21] <= 0
    m.l_wrist_theta_0 = Constraint(m.N, rule=l_wrist_theta_0)
    def l_wrist_theta_1(m,n):
        return m.x[n,21] >= -np.pi
    m.l_wrist_theta_1 = Constraint(m.N, rule=l_wrist_theta_1)

    def l_wrist_psi_0(m,n):
        return m.x[n,34] <= 0
    m.l_wrist_psi_0 = Constraint(m.N, rule=l_wrist_psi_0)
    def l_wrist_psi_1(m,n):
        return m.x[n,34] >= 0
    m.l_wrist_psi_1 = Constraint(m.N, rule=l_wrist_psi_1)

    #Right Shoulder
    def r_shoulder_phi_0(m,n):
       return m.x[n,9] <= 0
    m.r_shoulder_phi_0 = Constraint(m.N, rule=r_shoulder_phi_0)
    def r_shoulder_phi_1(m,n):
       return m.x[n,9] >= -np.pi/4
    m.r_shoulder_phi_1 = Constraint(m.N, rule=r_shoulder_phi_1)

    def r_shoulder_theta_0(m,n):
        return m.x[n,22] <= 0
    m.r_shoulder_theta_0 = Constraint(m.N, rule=r_shoulder_theta_0)
    def r_shoulder_theta_1(m,n):
        return m.x[n,22] >= 0
    m.r_shoulder_theta_1 = Constraint(m.N, rule=r_shoulder_theta_1)

    def r_shoulder_psi_0(m,n):
        return m.x[n,35] <= np.pi/4
    m.r_shoulder_psi_0 = Constraint(m.N, rule=r_shoulder_psi_0)
    def r_shoulder_psi_1(m,n):
        return m.x[n,35] >= -np.pi/4
    m.r_shoulder_psi_1 = Constraint(m.N, rule=r_shoulder_psi_1)

    #Right Elbow
    def r_elbow_phi_0(m,n):
       return m.x[n,10] <= np.pi
    m.r_elbow_phi_0 = Constraint(m.N, rule=r_elbow_phi_0)
    def r_elbow_phi_1(m,n):
       return m.x[n,10] >= -np.pi
    m.r_elbow_phi_1 = Constraint(m.N, rule=r_elbow_phi_1)

    def r_elbow_theta_0(m,n):
        return m.x[n,23] <= np.pi
    m.r_elbow_theta_0 = Constraint(m.N, rule=r_elbow_theta_0)
    def r_elbow_theta_1(m,n):
        return m.x[n,23] >= -np.pi
    m.r_elbow_theta_1 = Constraint(m.N, rule=r_elbow_theta_1)
    
    def r_elbow_psi_0(m,n):
        return m.x[n,36] <= np.pi/2
    m.r_elbow_psi_0 = Constraint(m.N, rule=r_elbow_psi_0)
    def r_elbow_psi_1(m,n):
        return m.x[n,36] >= np.pi/2
    m.r_elbow_psi_1 = Constraint(m.N, rule=r_elbow_psi_1)

    #Right wrist
    def r_wrist_phi_0(m,n):
       return m.x[n,11] <= np.pi
    m.r_wrist_phi_0 = Constraint(m.N, rule=r_wrist_phi_0)
    def r_wrist_phi_1(m,n):
       return m.x[n,11] >= -np.pi
    m.r_wrist_phi_1 = Constraint(m.N, rule=r_wrist_phi_1)

    def r_wrist_theta_0(m,n):
        return m.x[n,24] <= 0
    m.r_wrist_theta_0 = Constraint(m.N, rule=r_wrist_theta_0)
    def r_wrist_theta_1(m,n):
        return m.x[n,24] >= -np.pi
    m.r_wrist_theta_1 = Constraint(m.N, rule=r_wrist_theta_1)
    
    def r_wrist_psi_0(m,n):
        return m.x[n,37] <= 0
    m.r_wrist_psi_0 = Constraint(m.N, rule=r_wrist_psi_0)
    def r_wrist_psi_1(m,n):
        return m.x[n,37] >= 0
    m.r_wrist_psi_1 = Constraint(m.N, rule=r_wrist_psi_1)

    #Pelvis
    def pelvis_phi_0(m,n):
       return m.x[n,12] <= 0
    m.pelvis_phi_0 = Constraint(m.N, rule=pelvis_phi_0)
    def pelvis_phi_1(m,n):
       return m.x[n,12] >= 0
    m.pelvis_phi_1 = Constraint(m.N, rule=pelvis_phi_1)

    def pelvis_theta_0(m,n):
        return m.x[n,25] <= 0
    m.pelvis_theta_0 = Constraint(m.N, rule=pelvis_theta_0)
    def pelvis_theta_1(m,n):
        return m.x[n,25] >= 0
    m.pelvis_theta_1 = Constraint(m.N, rule=pelvis_theta_1)
    
    def pelvis_psi_0(m,n):
        return m.x[n,38] <= np.pi/4
    m.pelvis_psi_0 = Constraint(m.N, rule=pelvis_psi_0)
    def pelvis_psi_1(m,n):
        return m.x[n,38] >= -np.pi/4
    m.pelvis_psi_1 = Constraint(m.N, rule=pelvis_psi_1)

    #Left knee    
    def l_knee_phi_0(m,n):
       return m.x[n,13] <= np.pi
    m.l_knee_phi_0 = Constraint(m.N, rule=l_knee_phi_0)
    def l_knee_phi_1(m,n):
       return m.x[n,13] >= -np.pi/2
    m.l_knee_phi_1 = Constraint(m.N, rule=l_knee_phi_1)

    def l_knee_theta_0(m,n):
        return m.x[n,26] <= np.pi/4
    m.l_knee_theta_0 = Constraint(m.N, rule=l_knee_theta_0)
    def l_knee_theta_1(m,n):
        return m.x[n,26] >= -np.pi
    m.l_knee_theta_1 = Constraint(m.N, rule=l_knee_theta_1)
    
    def l_knee_psi_0(m,n):
        return m.x[n,39] <= np.pi/2
    m.l_knee_psi_0 = Constraint(m.N, rule=l_knee_psi_0)
    def l_knee_psi_1(m,n):
        return m.x[n,39] >= -np.pi/2
    m.l_knee_psi_1 = Constraint(m.N, rule=l_knee_psi_1)

    #Left ankle
    def l_ankle_phi_0(m,n):
       return m.x[n,14] <= 0
    m.l_ankle_phi_0 = Constraint(m.N, rule=l_ankle_phi_0)
    def l_ankle_phi_1(m,n):
       return m.x[n,14] >= 0
    m.l_ankle_phi_1 = Constraint(m.N, rule=l_ankle_phi_1)

    def l_ankle_theta_0(m,n):
        return m.x[n,27] <= np.pi
    m.l_ankle_theta_0 = Constraint(m.N, rule=l_ankle_theta_0)
    def l_ankle_theta_1(m,n):
        return m.x[n,27] >= 0
    m.l_ankle_theta_1 = Constraint(m.N, rule=l_ankle_theta_1)    
    
    def l_ankle_psi_0(m,n):
        return m.x[n,40] <= 0
    m.l_ankle_psi_0 = Constraint(m.N, rule=l_ankle_psi_0)
    def l_ankle_psi_1(m,n):
        return m.x[n,40] >= 0
    m.l_ankle_psi_1 = Constraint(m.N, rule=l_ankle_psi_1)

    #Right knee 
    def r_knee_phi_0(m,n):
       return m.x[n,15] <= np.pi/2
    m.r_knee_phi_0 = Constraint(m.N, rule=r_knee_phi_0)
    def r_knee_phi_1(m,n):
       return m.x[n,15] >= -np.pi
    m.r_knee_phi_1 = Constraint(m.N, rule=r_knee_phi_1)

    def r_knee_theta_0(m,n):
        return m.x[n,28] <= np.pi/4
    m.r_knee_theta_0 = Constraint(m.N, rule=r_knee_theta_0)
    def r_knee_theta_1(m,n):
        return m.x[n,28] >= -np.pi
    m.r_knee_theta_1 = Constraint(m.N, rule=r_knee_theta_1)    
    
    def r_knee_psi_0(m,n):
        return m.x[n,41] <= np.pi/2
    m.r_knee_psi_0 = Constraint(m.N, rule=r_knee_psi_0)
    def r_knee_psi_1(m,n):
        return m.x[n,41] >= -np.pi/2
    m.r_knee_psi_1 = Constraint(m.N, rule=r_knee_psi_1)

    #Right ankle
    def r_ankle_phi_0(m,n):
       return m.x[n,16] <= 0
    m.r_ankle_phi_0 = Constraint(m.N, rule=r_ankle_phi_0)
    def r_ankle_phi_1(m,n):
       return m.x[n,16] >= 0
    m.r_ankle_phi_1 = Constraint(m.N, rule=r_ankle_phi_1)

    def r_ankle_theta_0(m,n):
        return m.x[n,29] <= np.pi
    m.r_ankle_theta_0 = Constraint(m.N, rule=r_ankle_theta_0)
    def r_ankle_theta_1(m,n):
        return m.x[n,29] >= 0
    m.r_ankle_theta_1 = Constraint(m.N, rule=r_ankle_theta_1)    
    
    def r_ankle_psi_0(m,n):
        return m.x[n,42] <= 0
    m.r_ankle_psi_0 = Constraint(m.N, rule=r_ankle_psi_0)
    def r_ankle_psi_1(m,n):
        return m.x[n,42] >= 0
    m.r_ankle_psi_1 = Constraint(m.N, rule=r_ankle_psi_1)

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0

        for n in m.N: #Frame
            # Model Error
            for p in m.P:
                slack_model_err += m.model_err_weight[p] * (m.slack_model[n, p] ** 2)
            # Measurement Error
            for l in m.L: #labels
                for c in m.C: #cameras
                    for d2 in m.D2: #Dimension of measurements
                        #slack_meas_err += redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2]**2, 3, 5, 15)
                        slack_meas_err += m.meas_err_weight[n, c, l] * (m.slack_meas[n, c, l, d2] ** 2)

        return slack_meas_err + slack_model_err \

    m.obj = Objective(rule=obj)

    return (m, pose_to_3d)


def solve_optimisation(model, exe_path, project_dir, poses) -> None:
    """
    Solves a given trajectory optimisation problem given a model and solver
    """
    opt = SolverFactory(
        'ipopt',
        executable=exe_path
    )

    # solver options
    opt.options["print_level"] = 5
    opt.options["max_iter"] = 150000
    opt.options["max_cpu_time"] = 30000
    opt.options["tol"] = 1e-1
    opt.options["OF_print_timing_statistics"] = "yes"
    opt.options["OF_print_frequency_iter"] = 10
    opt.options["OF_hessian_approximation"] = "limited-memory"
    # opt.options["linear_solver"] = "ma86"

    LOG_DIR = os.path.join(project_dir, "logs")

    # --- This step may take a while! ---
    results = opt.solve(
        model, tee=True,
        keepfiles=True,
        logfile=os.path.join(LOG_DIR, "solver.log")
    )

    result_dir = os.path.join(project_dir, "data", args.project, "results")

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f'traj_results_{current_datetime}.pickle'

    save_data(model, file_path=os.path.join(result_dir, file_name), poses=poses)


def convert_to_dict(m, poses) -> Dict:
    x_optimised = np.array([[value(m.x[n, p]) for p in m.Pc] for n in m.N])
    dx_optimised = np.array([[value(m.dx[n, p]) for p in m.P] for n in m.N])
    ddx_optimised = np.array([[value(m.ddx[n, p]) for p in m.P] for n in m.N])
    # Isolate the last two elements for each frame in x_optimised
    x_cam_optimised = x_optimised[:, -2:]


    file_data = {
        'x': np.array([[value(m.x[n, p]) for p in m.Pc] for n in m.N]),
        'dx': np.array([[value(m.dx[n, p]) for p in m.P] for n in m.N]),
        'ddx': np.array([[value(m.ddx[n, p]) for p in m.P] for n in m.N]),
        'x_cam': x_cam_optimised,
        'positions': np.array([poses(*states) for states in x_optimised[:, :42]]),
        'model_slack': np.array([[value(m.slack_model[n, p]) for p in m.P] for n in m.N]),
        'slack_meas': np.array([[[[value(m.slack_meas[n, c, l, d]) for d in m.D2] for l in m.L] for c in m.C] for n in m.N]),
    }
    print(file_data['x'])
    print(file_data['x_cam'])

    return file_data

def save_data(file_data, file_path, poses, dict=True) -> None:
    if dict:
        file_data = convert_to_dict(file_data, poses)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(file_data, f)

    print(f'save {file_path}')


# --- OUTLIER REJECTING COST FUNCTION (REDESCENDING LOSS) ---

def func_step(start, x):
    return 1 / (1 + np.e ** (-1 * (x - start)))


def func_piece(start, end, x):
    return func_step(start, x) - func_step(end, x)


def redescending_loss(err, a, b, c):
    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e)) / 2 * e ** 2
    cost += func_piece(a, b, e) * (a * e - (a ** 2) / 2)
    cost += func_piece(b, c, e) * (a * b - (a ** 2) / 2 + (a * (c - b) / 2) * (1 - ((c - e) / (c - b)) ** 2))
    cost += func_step(c, e) * (a * b - (a ** 2) / 2 + (a * (c - b) / 2))
    return cost


# --- Rotation matrices for x, y, and z axes ---

def rot_x(x):
    c = sp.cos(x)
    s = sp.sin(x)
    return sp.Matrix([
        [1, 0, 0],
        [0, c, s],
        [0, -s, c]
    ])


def rot_y(y):
    c = sp.cos(y)
    s = sp.sin(y)
    return sp.Matrix([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])


def rot_z(z):
    c = sp.cos(z)
    s = sp.sin(z)
    return sp.Matrix([
        [c, s, 0],
        [-s, c, 0],
        [0, 0, 1]
    ])


# --- Numpy equivalent rotation matrices ---

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


# --- Reprojection functions ---

def pt3d_to_2d_fisheye(x, y, z, K, D, R, t):
    x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
    y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
    z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
    # project onto camera plane
    a = x_2d / z_2d
    b = y_2d / z_2d
    # fisheye params
    #r = (a ** 2 + b ** 2 + 1e-12) ** 0.5
    r = (a ** 2 + b ** 2 + 1e-12) ** 0.5
    #th = atan(r)
    # distortion
    #th_D = th * (1 + D[0] * th ** 2 + D[1] * th ** 4 + D[2] * th ** 6 + D[3] * th ** 8)
    th_D = (1 + D[0] * r**2 + D[1] * r**4)
    #x_P = a * th_D / r
    x_P = a * th_D
    #y_P = b * th_D / r
    y_P = b * th_D
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
    r = (a ** 2 + b ** 2 + 1e-12) ** 0.5
    th_D = (1 + D[0] * r**2 + D[1] * r**4)
    
    x_P = a * th_D
    y_P = b * th_D
    
    u = K[0, 0] * x_P + K[0, 2]
    v = K[1, 1] * y_P + K[1, 2]
    return u, v


def pt3d_to_x2d(x, y, z, K, D, R, t):
    u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
    return u


def pt3d_to_y2d(x, y, z, K, D, R, t):
    v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
    return v

def pt3d_to_x2d_fisheye(x, y, z, K, D, R, t):
    u = pt3d_to_2d_fisheye(x, y, z, K, D, R, t)[0]
    return u

def pt3d_to_y2d_fisheye(x, y, z, K, D, R, t):
    v = pt3d_to_2d_fisheye(x, y, z, K, D, R, t)[1]
    return v


if __name__ == "__main__":
    parser = ArgumentParser(description="Build and Optimise")
    parser.add_argument('--top_dir', type=str, help='The path to Acinoset')
    parser.add_argument('--project', type=str, help='The project to be optimized')
    parser.add_argument('--start_frame', type=int, default=1,
                        help='The frame at which the optimized reconstruction will start at')
    parser.add_argument('--end_frame', type=int, default=-100,
                        help='The frame at which the optimized reconstruction will end at')
    parser.add_argument('--dlc_thresh', type=float, default=0.5,
                        help='The likelihood of the dlc points below which will be excluded from the optimization')
    args = parser.parse_args()

    #exit()
    data_path = os.path.join(args.top_dir, "data", args.project)
    model1, pose3d = build_model(args.top_dir)
    ipopt_path = "C:\\Users\\user-pc\\anaconda3\\pkgs\\Ipopt-3.14.1-win64-msvs2019-md\\bin\\ipopt.exe"
    solve_optimisation(model1, exe_path=ipopt_path, project_dir=args.top_dir, poses=pose3d)