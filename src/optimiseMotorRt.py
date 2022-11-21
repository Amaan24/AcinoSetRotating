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
import pan_compensation as pc
import scipy.io as sio

pose_to_3d = []


def build_model(project_dir) -> ConcreteModel:
    """
    Builds a pyomo human model 
    """  

    # defines arrays of angles, velocities and accelerations
    phi = sp.symbols("phi")
    theta = sp.symbols("theta")
    psi = sp.symbols("psi")

    RI_0 = rot_z(psi) @ rot_x(phi) @ rot_y(theta) # origin to inertial
    R0_I = RI_0.T

    # defines the position, velocities and accelerations in the inertial frame
    x,   y,   z   = sp.symbols("x y z")
    dx,  dy,  dz  = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
    ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")
    
    square_size = 0.09

    # SYMBOLIC CHECKERBOARD POSE POSITIONS
    p_origin        = sp.Matrix([x, y, z])
    p_r0_c1         = p_origin + R0_I @ sp.Matrix([0, square_size, 0])
    p_r1_c1         = p_r0_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r2_c1         = p_r1_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r3_c1         = p_r2_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r4_c1         = p_r3_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r5_c1         = p_r4_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r6_c1         = p_r5_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r7_c1         = p_r6_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r8_c1         = p_r7_c1 + R0_I @ sp.Matrix([0, square_size, 0])
    p_r9_c1         = p_r8_c1 + R0_I @ sp.Matrix([0, square_size, 0])

    # ========= LAMBDIFY SYMBOLIC FUNCTIONS ========
    positions = sp.Matrix([p_origin.T, p_r0_c1.T, p_r1_c1.T])

    func_map = {"sin": sin, "cos": cos, "ImmutableDenseMatrix": np.array}
    sym_list = [x, y, z, phi, theta, psi]
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

    corner_mat_file_path = os.path.join(project_dir, "data", args.project, "checkerboard_corners.mat")
    mat_contents = sio.loadmat(corner_mat_file_path)

    K_arr = np.array([[[ 1901.2, 0,  585.8],
            [ 0, 1895.0, 438.5],
            [ 0,  0, 1]],

            [[ 1923.3,  0,  538.2],
            [ 0,  1917.7, 403.3],
            [0,  0,  1]]])
    D_arr = np.array([[[-0.5403],
            [0.3943],
            [0],
            [0]],

            [[-0.5662],
            [ 0.4610],
            [0],
            [0]]])
    R_arr = np.array([[[ 1, 0,  0],
            [ 0, 1, 0],
            [ 0,  0, 1]],

            [[0.9990,  0.0023,  -0.04373],
            [0.0001,  0.9984, 0.0561],
            [0.0437,  -0.0561,  0.9975]]])
    t_arr = np.array([[[ 0.],
                [ 0.],
                [ 0.]],

                [[-0.3684],
                [-0.0091],
                [0.1081]]])

    D_arr = D_arr.reshape((-1, 4))

    print(f"\n\n\nLoading data")

    def get_meas_from_df(n, c, l, d):
        return mat_contents['cornerPoints'][n-1][c-1][l-1][d-1]

    def get_enc_meas(n, c):
        return encoder_arr[n, c]

    h = 1 / 60  # timestep: 1/framerate
    start_frame = args.start_frame  
    N = args.end_frame - args.start_frame
    P = 6 
    L = len(pos_funcs)
    C = len(K_arr)
    D2 = 2 
    D3 = 3 

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]
    proj_funcs_rotating = [pt3d_to_x2d_rotating, pt3d_to_y2d_rotating]

    R = 0.5  # measurement standard deviation

    # estimate initial points
    frame_est = np.arange(N)

    x_est = np.array([1  for i in range(len(frame_est))])
    y_est = np.array([6  for i in range(len(frame_est))])
    z_est = np.array([0.5  for i in range(len(frame_est))])

    print("Started Optimisation")
    m = ConcreteModel(name="Skeleton")

    # ===== SETS =====
    m.N = RangeSet(N)  # number of timesteps in trajectory
    m.P = RangeSet(P)  # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n)
    m.L = RangeSet(L)  # number of labels
    m.C = RangeSet(C)  # number of cameras
    m.D2 = RangeSet(D2)  # dimensionality of measurements
    m.D3 = RangeSet(D3)  # dimensionality of measurements

    m.MAT = RangeSet(9) # number of elements in rotation matrix
    m.VEC = RangeSet(3) # number of elements in translation vector

    def init_meas_weights(model, n, c, l):
            return 1/R
    m.meas_err_weight = Param(m.N, m.C, m.L, initialize=init_meas_weights, mutable=True,
                              within=Any)  

    m.h = h

    def init_measurements_df(m, n, c, l, d2):
            return get_meas_from_df(n + start_frame, c, l, d2)
    m.meas = Param(m.N, m.C, m.L, m.D2, initialize=init_measurements_df, within=Any)
    
    ## For rotating c1 and c2
    def init_encoder_err_weight(m):
        return ((102000*(2*np.pi)))
    m.enc_err_weight = Param(initialize=init_encoder_err_weight, mutable=True, within=Any)

    def init_enc_model_weights(m):
        return 1/(2*np.pi*102000)**2
    m.enc_model_err_weight = Param(initialize=init_enc_model_weights, within=Any)

    def init_encoder_measurements(m, n, c):
            return pc.count_to_rad(get_enc_meas(n + start_frame - 1, c - 1)) 
    m.meas_enc = Param(m.N, m.C, initialize=init_encoder_measurements, within=Any)

    # ===== VARIABLES =====
    m.x = Var(m.N, m.P)  # position # number of pose parameters (x, y, z, phi_1..n, theta_1..n, psi_1..n)

    m.poses = Var(m.N, m.L, m.D3)
    m.slack_meas = Var(m.N, m.C, m.L, m.D2, initialize=0.0) #Update

    m.x_cam = Var(m.N, m.C, initialize=0.0) # Encoder position
    m.dx_cam = Var(m.N, m.C, initialize=0.0) # Encoder position
    m.ddx_cam = Var(m.N, m.C, initialize=0.0) # Encoder position

    m.enc_slack_meas = Var(m.N, m.C, initialize=0.0) #Update
    m.enc_slack_model = Var(m.N, m.C, initialize=0.0)

    m.rot_ct0_mt0 = Var(m.MAT, m.C)
    m.tran_cam_motor = Var(m.VEC, m.C) 

    # ===== VARIABLES INITIALIZATION =====
    init_x = np.zeros((N, P))
    init_x[:, 0] = x_est  # x
    init_x[:, 1] = y_est  # y
    init_x[:, 2] = z_est  # z
    
    for n in range(1, N + 1):
        for p in range(1, P + 1):
            if n < len(init_x):  # init using known values
                m.x[n, p].value = init_x[n - 1, p - 1]
            else:  # init using last known value
                m.x[n, p].value = init_x[-1, p - 1]

        # init pose
        var_list = [m.x[n, p].value for p in range(1, P + 1)]
        for l in range(1, L + 1):
            [pos] = pos_funcs[l - 1](*var_list)
            print(pos)
            for d3 in range(1, D3 + 1):
                m.poses[n, l, d3].value = pos[d3 - 1]

    for n in range(1, 10):
        if (n in (1, 5, 9)):
            m.rot_ct0_mt0[n, 1] = 1
            m.rot_ct0_mt0[n, 2] = 1
        else: 
            m.rot_ct0_mt0[n, 1] = 0
            m.rot_ct0_mt0[n, 2] = 0

    for n in range(1, 10):
        print(m.rot_ct0_mt0[n, 1].value)
    
    for n in range(1, 4):
        m.tran_cam_motor[n, 1] = 0
        m.tran_cam_motor[n, 2] = 0

    # ===== CONSTRAINTS =====
    # 3D POSE
    def pose_constraint(m, n, l, d3):
        # get 3d points
        var_list = [m.x[n, p] for p in range(1, P + 1)]
        [pos] = pos_funcs[l - 1](*var_list)
        return pos[d3 - 1] == m.poses[n, l, d3]
    m.pose_constraint = Constraint(m.N, m.L, m.D3, rule=pose_constraint)

    # CHECKERBOARD MODEL
    def constant_pos(m, n, p):
        if n > 1:
            return m.x[n, p] == m.x[n - 1, p]
        else:
            return Constraint.Skip
    m.constant_pos = Constraint(m.N, m.P, rule=constant_pos)

    # ENCODER/MOTOR MODEL
    def backwards_euler_pos(m, n, c):  # position
        if n > 1:
            return m.x_cam[n, c] == m.x_cam[n - 1, c] + m.h * m.dx_cam[n, c]
        else:
            return Constraint.Skip
    m.integrate_p = Constraint(m.N, m.C, rule=backwards_euler_pos)

    def backwards_euler_vel(m, n, c):  # velocity
        if n > 1:
            return m.dx_cam[n, c] == m.dx_cam[n - 1, c] + m.h * m.ddx_cam[n, c]
        else:
            return Constraint.Skip
    m.integrate_v = Constraint(m.N, m.C, rule=backwards_euler_vel)

    def constant_acc(m, n, c):
        if n > 1:
            return m.ddx_cam[n, c] == m.ddx_cam[n - 1, c] + m.enc_slack_model[n, c]
        else:
            return Constraint.Skip
    m.constant_acc = Constraint(m.N, m.C, rule=constant_acc)

    # MEASUREMENT
    def measurement_constraints(m, n, c, l, d2):
        # project
        K, D, R, t = K_arr[c - 1], D_arr[c - 1], R_arr[c - 1], t_arr[c - 1]

        RA_O = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]])

        RO_Ct0 = np.array(R)
        Cc = np.array(-1*RO_Ct0 @ t)    
        
        RCt0_Mt0 = np.array([[m.rot_ct0_mt0[1, c], m.rot_ct0_mt0[2, c], m.rot_ct0_mt0[3, c]],
                       [m.rot_ct0_mt0[4, c], m.rot_ct0_mt0[5, c], m.rot_ct0_mt0[6, c]],
                       [m.rot_ct0_mt0[7, c], m.rot_ct0_mt0[8, c], m.rot_ct0_mt0[9, c]]])
        #RCt0_Mt0 = np.array([[1, 0, 0],
        #                     [0, 1, 0],
        #                     [0, 0, 1]])
        #Cm = np.array([[m.tran_cam_motor[1, c], m.tran_cam_motor[2, c], m.tran_cam_motor[3, c]]]).T
        Cm = np.array([[0, 0, 0]]).T 

        RMt0_Mt1 = np_rot_y(-1*m.x_cam[n, c].value)

        P_world = np.array([[m.poses[n, l, 1]],
                           [m.poses[n, l, 2]],
                           [m.poses[n, l, 3]]])

        P_cam =  RCt0_Mt0.T @ (RMt0_Mt1 @ RCt0_Mt0 @ (RO_Ct0 @ (RA_O @ P_world - Cc) - Cm) + Cm) 
        #P_cam = RMt0_Mt1 @ RO_Ct0 @ RA_O @ P_world + RMt0_Mt1 @ t

        x = P_cam[0]
        y = P_cam[1]
        z = P_cam[2]


        #R =  np_rot_y(m.x_cam[n, c].value).T @ R @ RA_O
        #t =  np_rot_y(m.x_cam[n, c].value).T @ t 
        #x, y, z = m.poses[n, l, 1], m.poses[n, l, 2], m.poses[n, l, 3]
        #return proj_funcs[d2 - 1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0 #+m.meas = dlc points
        
        return proj_funcs_rotating[d2 - 1](x, y, z, K, D) - m.meas[n, c, l, d2] - m.slack_meas[n, c, l, d2] == 0 #+m.meas = dlc points

    m.measurement = Constraint(m.N, m.C, m.L, m.D2, rule=measurement_constraints)

    def enc_measurement_constraints(m, n, c):
                return  m.x_cam[n, c] - m.meas_enc[n, c] == 0 #- m.enc_slack_meas[n, c]== 0
    m.enc_measurement = Constraint(m.N, m.C, rule=enc_measurement_constraints)

    def motor_rotation_constraints(m, mat, c):
        return abs(m.rot_ct0_mt0[mat, c]) <= 1
    m.motor_rot_estimate = Constraint(m.MAT, m.C, rule=motor_rotation_constraints)

    #def motor_translation_constraints(m, vec, c):
    #    return abs(m.tran_cam_motor[vec, c]) <= 0.10
    #m.motor_trans_estimate = Constraint(m.VEC, m.C, rule=motor_translation_constraints)

    # ======= OBJECTIVE FUNCTION =======
    def obj(m):
        slack_model_err = 0.0
        slack_meas_err = 0.0
        enc_model_err = 0.0
        enc_meas_err = 0.0

        for n in range(1, N + 1): #Frame
            # Model Error
            for p in range(1, P - 1):
                slack_model_err += 0 #We know the checkerboard is stationary!
            # Measurement Error
            for l in range(1, L + 1): #labels
                for c in range(1, C + 1): #cameras
                    for d2 in range(1, D2 + 1): #Dimension on measurements
                        #slack_meas_err += redescending_loss(m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2], 3, 5, 15)
                        slack_meas_err += m.meas_err_weight[n, c, l] * m.slack_meas[n, c, l, d2] **2
            # Encoder Error
            for c in range(1, C + 1): # Encoder/Cameras
                # Encoder Model Error
                enc_model_err += m.enc_model_err_weight * m.enc_slack_model[n, c] ** 2
                # Encoder Measurement Error
                enc_meas_err += m.enc_err_weight * m.enc_slack_meas[n, c]**2 #Removed - Assuming zero encoder meas error!

        return slack_meas_err + slack_model_err + enc_meas_err + enc_model_err

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
    save_data(model, file_path=os.path.join(result_dir, 'traj_results.pickle'), poses=poses)

def convert_to_dict(m, poses) -> Dict:
    x_optimised = []
    x_cam_optimised = []
    dx_cam_optimised = []
    ddx_cam_optimised = []
    x_cam_model_slack_optimised = []
    x_slack_meas_optimised = []

    for n in m.N:
        x_optimised.append([value(m.x[n, p]) for p in m.P])
        x_cam_optimised.append([value(m.x_cam[n, c]) for c in m.C])
        dx_cam_optimised.append([value(m.dx_cam[n, c]) for c in m.C])
        ddx_cam_optimised.append([value(m.ddx_cam[n, c]) for c in m.C])
        
        temp = []
        for c in m.C:
            for l in m.L:
                temp.append([value(m.slack_meas[n, c, l, d]) for d in m.D2])
        x_slack_meas_optimised.append(temp)

    motor_rot = []
    motor_trans = []

    for c in m.C:
        motor_rot.append([value(m.rot_ct0_mt0[i, c]) for i in range(1, 10)])
        motor_trans.append([value(m.tran_cam_motor[i, c]) for i in m.VEC])

    x_optimised = np.array(x_optimised)
    x_cam_optimised = np.array(x_cam_optimised)
    dx_cam_optimised = np.array(dx_cam_optimised)
    ddx_cam_optimised = np.array(ddx_cam_optimised)
    x_cam_model_slack_optimised = np.array(x_cam_model_slack_optimised)
    x_slack_meas_optimised = np.array(x_slack_meas_optimised)

    motor_rot_optimised = np.array(motor_rot)
    motor_trans_optimised = np.array(motor_trans)

    print(x_optimised)
    print(x_cam_optimised)

    print(motor_rot_optimised)
    print(motor_trans_optimised)

    positions = np.array([poses(*states) for states in x_optimised[:, :45]])
    file_data = dict(
        positions=positions,
        x=x_optimised,
        x_cam=x_cam_optimised,
        dx_cam=dx_cam_optimised,
        ddx_cam=ddx_cam_optimised,
        x_cam_model_slack=x_cam_model_slack_optimised,
        slack_meas=x_slack_meas_optimised,

        motor_rot=motor_rot_optimised,
        motor_trans=motor_trans_optimised
        )

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

def pt3d_to_x2d(x, y, z, K, D, R, t):
    u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
    return u


def pt3d_to_y2d(x, y, z, K, D, R, t):
    v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
    return v

def pt3d_to_x2d_rotating(x, y, z, K, D):
    u = pt3d_to_2d_rotating(x, y, z, K, D)[0]
    return u

def pt3d_to_y2d_rotating(x, y, z, K, D):
    v = pt3d_to_2d_rotating(x, y, z, K, D)[1]
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

    model1, pose3d = build_model(args.top_dir)
    ipopt_path = "C:\\Users\\user-pc\\anaconda3\\pkgs\\Ipopt-3.14.1-win64-msvs2019-md\\bin\\ipopt.exe"
    solve_optimisation(model1, exe_path=ipopt_path, project_dir=args.top_dir, poses=pose3d)