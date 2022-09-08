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


def build_model(project_dir) -> ConcreteModel:
    """
    Builds a pyomo model from a given saved skeleton dictionary
    """
    encoder_path = os.path.join(project_dir, "data", args.project, "extrinsic_calib", "encoder_data.pkl")
    with open(encoder_path, 'rb') as handle:
        encoder_arr = pickle.load(handle)

    encoder_arr = np.ones((5001, 2)) #Encoder Count (102000 CPT)
    for i in range(0, len(encoder_arr)):
        encoder_arr[i, 0] = i
        encoder_arr[i, 1] = i
    #encoder_arr = np.zeros((5001, 2))    
    print(encoder_arr)

    def get_enc_meas(n, c):
        return encoder_arr[n, c]

    h = 1/200  # timestep: 1/(2 x framerate)?
    N = 100 # No. of timesteps
    C = 2 # No. of cameras

    print("Started Optimisation")
    m = ConcreteModel(name="Encoders")

    # ===== SETS =====
    m.N = RangeSet(N)  # number of timesteps in trajectory
    m.C = RangeSet(C)  # number of pose parameters/cameras (alpha and beta)

    def init_encoder_err_weight(m): # Encoder error weight = (2*pi*CPT)^2
        return ((2*np.pi*102000))**2

    m.enc_err_weight = Param(initialize=init_encoder_err_weight, mutable=True, within=Any)

    def init_enc_model_weights(m):
        return 1/(2*np.pi*102000)**2 # Model error weight = 1/(2*pi*CPT)^2

    m.enc_model_err_weight = Param(initialize=init_enc_model_weights, within=Any)

    m.h = h

    def init_encoder_measurements(m, n, c):
        #return pc.count_to_rad(get_enc_meas(n-1, c-1))
        return get_enc_meas(n-1, c-1)
    
    m.meas_enc = Param(m.N, m.C, initialize=init_encoder_measurements, within=Any)

    for n in range(1, N + 1):
        for c in range(1, C + 1):
            print(m.meas_enc[n , c])

    # ===== VARIABLES =====
    m.x_cam = Var(m.N, m.C, initialize=10.0) #Cam position   
    m.dx_cam = Var(m.N, m.C, initialize=0.0) #Cam velocity
    m.ddx_cam = Var(m.N, m.C, initialize=0.0) #Cam acceleration

    m.slack_model = Var(m.N, m.C)
    m.slack_meas = Var(m.N, m.C, initialize=0.0) #Update

    for n in range(1, N + 1):
        for c in range(1, C + 1):
            if n == 1:
                m.x_cam[n, c] = 0.0
            else:
                m.x_cam[n, c] = 0.0

    # ===== CONSTRAINTS =====
    def enc_backwards_euler_pos(m, n, c):  # position
        if n > 1:
            #             return m.x[n,c] == m.x[n-1,c] + m.h*m.dx[n-1,c] + m.h**2 * m.ddx[n-1,c]/2
            return m.x_cam[n, c] == m.x_cam[n - 1, c] + m.h * m.dx_cam[n, c]

        else:
            return Constraint.Skip

    m.integrate_enc_p = Constraint(m.N, m.C, rule=enc_backwards_euler_pos)

    def enc_backwards_euler_vel(m, n, c):  # velocity
        if n > 1:
            return m.dx_cam[n, c] == m.dx_cam[n - 1, c] + m.h * m.ddx_cam[n, c]
        else:
            return Constraint.Skip

    m.integrate_enc_v = Constraint(m.N, m.C, rule=enc_backwards_euler_vel)

    # MODEL
    def enc_constant_acc(m, n, c):
        if n > 1:
            return m.ddx_cam[n, c] == m.ddx_cam[n - 1, c] + m.slack_model[n, c]
        else:
            return Constraint.Skip

    m.enc_constant_acc = Constraint(m.N, m.C, rule=enc_constant_acc)

    # MEASUREMENT

    def measurement_constraints(m, n, c):
        return  m.x_cam[n, c] - m.meas_enc[n, c] - m.slack_meas[n, c] == 0
    m.measurement = Constraint(m.N, m.C, rule=measurement_constraints)

    def obj(m):
        enc_model_err = 0.0
        enc_meas_err = 0.0

        for n in range(1, N + 1): # Frame
            # Encoder Model Error
            for c in range(1, C + 1): # Encoder
               enc_model_err += m.enc_model_err_weight * m.slack_model[n, c] ** 2
            # Encoder Measurement Error
            for  c in range(1, C + 1): #Encoder
                    enc_meas_err += m.enc_err_weight * m.slack_meas[n ,c] ** 2


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
    opt.options["max_iter"] = 15000
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

def convert_to_dict_rotating(m, poses) -> Dict:
    #x_optimised = []
    #dx_optimised = []
    #ddx_optimised = []

    x_cam_optimised = []
    dx_cam_optimised = []
    ddx_cam_optimised = []

    for n in m.N:
        x_cam_optimised.append([value(m.x_cam[n, c]) for c in m.C])
        dx_cam_optimised.append([value(m.dx_cam[n, c]) for c in m.C])
        ddx_cam_optimised.append([value(m.ddx_cam[n, c]) for c in m.C])

    x_cam_optimised = np.array(x_cam_optimised)
    dx_cam_optimised = np.array(dx_cam_optimised)
    ddx_cam_optimised = np.array(ddx_cam_optimised)

    print(x_cam_optimised)
    print(dx_cam_optimised)
    print(ddx_cam_optimised)

    enc_meas_err = 0.0
    for n in range(1, 101): # Frame
        # Encoder Model Error
        #for c in range(1, C + 1): # Encoder
        #   enc_model_err += m.enc_model_err_weight ** 2
        # Encoder Measurement Error
        for  c in range(1, 2): #Encoder
                enc_meas_err += (value(m.x_cam[n, c]) - value(m.meas_enc[n, c]))**2
    print("Total Error = " + str(enc_meas_err))

    file_data = dict(
        x=x_cam_optimised,
        dx=dx_cam_optimised,
        ddx=ddx_cam_optimised,
    )
    return file_data


def save_data_rotating(file_data, file_path, poses, dict=True) -> None:
    if dict:
        file_data = convert_to_dict_rotating(file_data, poses)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(file_data, f)

    print(f'save {file_path}')

if __name__ == "__main__":
    parser = ArgumentParser(description="Build and Optimise")
    parser.add_argument('--top_dir', type=str, help='The path to Acinoset')
    parser.add_argument('--project', type=str, help='The project to be optimized')
    args = parser.parse_args()

    data_path = os.path.join(args.top_dir, "data", args.project)
    model1, pose3d = build_model(args.top_dir)
    ipopt_path = "C:\\Users\\user-pc\\anaconda3\\pkgs\\Ipopt-3.14.1-win64-msvs2019-md\\bin\\ipopt.exe"
    solve_optimisation(model1, exe_path=ipopt_path, project_dir=args.top_dir, poses=pose3d)