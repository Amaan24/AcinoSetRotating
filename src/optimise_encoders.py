from cmath import pi
from sqlite3 import dbapi2
from typing import Dict
import pickle
from pyomo.core.base.constraint import Constraint, ConstraintList
import sympy as sp
import numpy as np
import os
import glob
from calib import utils, calib, plotting, app, extract
from scipy import stats
from pprint import pprint
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.core.base.PyomoModel import ConcreteModel
from argparse import ArgumentParser
import pan_compensation as pc

pose_to_3d = []


def load_skeleton(skel_file) -> Dict:
    """
    Loads a skeleton dictionary from a saved skeleton .pickle file
    """
    with open(skel_file, 'rb') as handle:
        skel_dict = pickle.load(handle)

    return skel_dict


def build_model(skel_dict, project_dir) -> ConcreteModel:
    """
    Builds a pyomo model from a given saved skeleton dictionary
    """
    encoder_path = os.path.join(project_dir, "data", args.project, "extrinsic_calib", "encoder_data.pkl")
    with open(encoder_path, 'rb') as handle:
        encoder_arr = pickle.load(handle)

    encoder_arr = np.zeros((51, 2)) 
    print(encoder_arr)

    def get_enc_meas(n, c):
        return encoder_arr[n, c]

    h = 1 / 200  # timestep: 1/(2 x framerate)?
    start_frame = args.start_frame  # 50
    N = args.end_frame - args.start_frame
    P = 2
    R = 3  # measurement standard deviation

    print("Started Optimisation")
    m = ConcreteModel(name="Encoders")

    # ===== SETS =====
    m.N = RangeSet(N)  # number of timesteps in trajectory
    m.P = RangeSet(P)  # number of pose parameters alpha and beta


    ## For rotating c1 and c2
    def init_encoder_err_weight(m):
        return ((2*np.pi*102000))**2

    m.enc_err_weight = Param(initialize=init_encoder_err_weight, mutable=True, within=Any)

    def init_enc_model_weights(m):
        return 1/(2*np.pi*102000)**2

    m.enc_model_err_weight = Param(initialize=init_enc_model_weights, within=Any)

    m.h = h

    def init_encoder_measurements(m, n,c):
        return pc.count_to_rad(get_enc_meas(n-1,c-1))
    
    m.meas_enc = Param(m.N, m.C, initialize=init_encoder_measurements, within=Any)

    # ===== VARIABLES =====
    m.x_cam = Var(m.N, m.P2, initialize=0) #Cam position   
    m.dx_cam = Var(m.N, m.P2, initialize=0) #Cam velocity
    m.ddx_cam = Var(m.N, m.P2, initialize=0) #Cam acceleration

    # ===== CONSTRAINTS =====
    # 3D POSE
    def enc_backwards_euler_pos(m, n, p):  # position
        if n > 1:
            #             return m.x[n,p] == m.x[n-1,p] + m.h*m.dx[n-1,p] + m.h**2 * m.ddx[n-1,p]/2
            return m.x_cam[n, p] == m.x_cam[n - 1, p] + m.h * m.dx_cam[n, p]

        else:
            return Constraint.Skip

    m.integrate_enc_p = Constraint(m.N, m.P2, rule=enc_backwards_euler_pos)

    def enc_backwards_euler_vel(m, n, p):  # velocity
        if n > 1:
            return m.dx_cam[n, p] == m.dx_cam[n - 1, p] + m.h * m.ddx_cam[n, p]
        else:
            return Constraint.Skip

    m.integrate_enc_v = Constraint(m.N, m.P2, rule=enc_backwards_euler_vel)

    # MODEL
    def enc_constant_acc(m, n, p):
        if n > 1:
            return m.ddx_cam[n, p] == m.ddx_cam[n - 1, p] #Add slack?
        else:
            return Constraint.Skip

    m.enc_constant_acc = Constraint(m.N, m.P2, rule=enc_constant_acc)

    # MEASUREMENT

    def obj(m):
        enc_model_err = 0.0
        enc_meas_err = 0.0

        for n in range(1, N + 1):
            # Encoder Model Error
            for p in range(1, P2 + 1):
               enc_model_err += m.enc_model_err_weight ** 2
            # Encoder Measurement Error
            for  p in range(1, P2 + 1):
                if p == 1:
                    enc_meas_err += abs(m.enc_err_weight * (m.x_cam[n, p] - m.meas_enc[n,1]))
                else:
                    enc_meas_err += abs(m.enc_err_weight * (m.x_cam[n, p] - m.meas_enc[n,2]))

        return enc_meas_err + enc_model_err

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
    opt.options["max_iter"] = 1000
    opt.options["max_cpu_time"] = 3600
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
    save_data_rotating(model, file_path=os.path.join(result_dir, 'traj_results.pickle'), poses=poses)


# def save_data(file_data, filepath, dict=False):
# if dict:
# file_data = convert_to_dict(file_data)

# with open(filepath, 'wb') as f:
# pickle.dump(file_data, f)

def convert_to_dict(m, poses) -> Dict:
    x_optimised = []
    dx_optimised = []
    ddx_optimised = []
    for n in m.N:
        x_optimised.append([value(m.x[n, p]) for p in m.P])
        dx_optimised.append([value(m.dx[n, p]) for p in m.P])
        ddx_optimised.append([value(m.ddx[n, p]) for p in m.P])
    x_optimised = np.array(x_optimised)
    dx_optimised = np.array(dx_optimised)
    ddx_optimised = np.array(ddx_optimised)

    print(poses)
    print(x_optimised)

    positions = np.array([poses(*states) for states in x_optimised])
    file_data = dict(
        positions=positions,
        x=x_optimised,
        dx=dx_optimised,
        ddx=ddx_optimised,
    )
    return file_data

def convert_to_dict_rotating(m, poses) -> Dict:
    x_optimised = []
    dx_optimised = []
    ddx_optimised = []

    x_cam_optimised = []
    dx_cam_optimised = []
    ddx_cam_optimised = []

    for n in m.N:
        x_optimised.append([value(m.x[n, p]) for p in m.P])
        dx_optimised.append([value(m.dx[n, p]) for p in m.P])
        ddx_optimised.append([value(m.ddx[n, p]) for p in m.P])

        x_cam_optimised.append([value(m.x_cam[n, p]) for p in m.P2])
        dx_cam_optimised.append([value(m.dx_cam[n, p]) for p in m.P2])
        ddx_cam_optimised.append([value(m.ddx_cam[n, p]) for p in m.P2])

    x_optimised = np.array(x_optimised)
    dx_optimised = np.array(dx_optimised)
    ddx_optimised = np.array(ddx_optimised)

    x_cam_optimised = np.array(x_cam_optimised)
    dx_cam_optimised = np.array(dx_cam_optimised)
    ddx_cam_optimised = np.array(ddx_cam_optimised)

    print(poses)
    print(x_optimised)
    print(x_cam_optimised)
    print(dx_cam_optimised)
    print(ddx_cam_optimised)

    positions = np.array([poses(*states) for states in x_optimised])
    file_data = dict(
        positions=positions,
        x=x_optimised,
        dx=dx_optimised,
        ddx=ddx_optimised,
    )
    return file_data

def save_data(file_data, file_path, poses, dict=True) -> None:
    if dict:
        file_data = convert_to_dict(file_data, poses)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(file_data, f)

    print(f'save {file_path}')

def save_data_rotating(file_data, file_path, poses, dict=True) -> None:
    if dict:
        file_data = convert_to_dict_rotating(file_data, poses)

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


def pt3d_to_x2d(x, y, z, K, D, R, t):
    u = pt3d_to_2d(x, y, z, K, D, R, t)[0]
    return u


def pt3d_to_y2d(x, y, z, K, D, R, t):
    v = pt3d_to_2d(x, y, z, K, D, R, t)[1]
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

    skeleton_path = os.path.join(args.top_dir, "skeletons", "new_human.pickle")
    skelly = load_skeleton(skeleton_path)
    print(skelly)
    data_path = os.path.join(args.top_dir, "data", args.project)
    model1, pose3d = build_model(skelly, args.top_dir)
    ipopt_path = "C:\\Users\\user-pc\\anaconda3\\pkgs\\Ipopt-3.14.1-win64-msvs2019-md\\bin\\ipopt.exe"
    solve_optimisation(model1, exe_path=ipopt_path, project_dir=args.top_dir, poses=pose3d)