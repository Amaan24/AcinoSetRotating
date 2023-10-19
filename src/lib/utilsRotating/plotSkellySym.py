import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Add this import
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import numpy as np

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


# Define symbolic variables for the rotation angles
L = 13  # Number of segments in the skeleton
phi = [sp.symbols(f"\\phi_{{{l}}}") for l in range(L)]
theta = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
psi = [sp.symbols(f"\\psi_{{{l}}}") for l in range(L)]

# ... (code for defining rotation matrices and position expressions)
phi = [sp.symbols(f"\\phi_{{{l}}}") for l in range(L)]
theta = [sp.symbols(f"\\theta_{{{l}}}") for l in range(L)]
psi = [sp.symbols(f"\\psi_{{{l}}}") for l in range(L)]

# Rotations
RI_0 = rot_z(psi[0]) @ rot_x(phi[0]) @ rot_y(theta[0])  # forehead to inertial
R0_I = RI_0.T

RI_1 = rot_z(psi[1]) @ rot_x(phi[1]) @ rot_y(theta[1]) @ RI_0  # neck to forehead
R1_I = RI_1.T

RI_2 = rot_z(psi[2]) @ rot_x(phi[2]) @ RI_1  # L Shoulder to inertial
R2_I = RI_2.T
RI_3 = rot_z(psi[3]) @ rot_x(phi[3]) @ rot_y(theta[3]) @ RI_2  # L Elbow to inertial
R3_I = RI_3.T
RI_4 = rot_y(theta[4]) @ RI_3  # L wrist to inertial
R4_I = RI_4.T

RI_5 = rot_z(psi[5]) @ rot_x(phi[5]) @ RI_1  # R Shoulder to inertial
R5_I = RI_5.T
RI_6 = rot_z(psi[6]) @ rot_x(phi[6]) @ rot_y(theta[6]) @ RI_5  # L Elbow to inertial
R6_I = RI_6.T
RI_7 = rot_y(theta[7]) @ RI_6  # R wrist to inertial
R7_I = RI_7.T

RI_8 = rot_z(psi[8]) @ RI_1  # Pelvis to inertial
R8_I = RI_8.T

RI_9 = rot_z(psi[9]) @ rot_x(phi[9]) @ rot_y(theta[9]
                                             ) @ RI_8  # L Knee to inertial
R9_I = RI_9.T

RI_10 = rot_y(theta[10]) @ RI_9  # L Ankle to inertial
R10_I = RI_10.T

RI_11 = rot_z(psi[11]) @ rot_x(phi[11]) @ rot_y(theta[11]
                                                ) @ RI_8  # R Knee to inertial
R11_I = RI_11.T

RI_12 = rot_y(theta[12]) @ RI_11  # R Ankle to inertial
R12_I = RI_12.T

# defines the position, velocities and accelerations in the inertial frame
x,   y,   z = sp.symbols("x y z")
dx,  dy,  dz = sp.symbols("\\dot{x} \\dot{y} \\dot{z}")
ddx, ddy, ddz = sp.symbols("\\ddot{x} \\ddot{y} \\ddot{z}")

# SYMBOLIC CHEETAH POSE POSITIONS
p_head = sp.Matrix([x, y, z])
p_chin = p_head + R0_I  @ sp.Matrix([0, 0, -0.22])
p_neck = p_head + R0_I  @ sp.Matrix([-0.1, 0, -0.27])
p_shoulder1 = p_neck + R2_I  @ sp.Matrix([0, 0.18, 0])
p_elbow1 = p_shoulder1 + R3_I  @ sp.Matrix([0, 0, -0.28])
p_wrist1 = p_elbow1 + R4_I  @ sp.Matrix([0, 0, -0.25])
p_shoulder2 = p_neck + R5_I  @ sp.Matrix([0, -0.18, 0])
p_elbow2 = p_shoulder2 + R6_I  @ sp.Matrix([0, 0, -0.28])
p_wrist2 = p_elbow2 + R7_I  @ sp.Matrix([0, 0, -0.25])
p_pelvis = p_neck + R8_I  @ sp.Matrix([0, 0, -0.5])
p_hip1 = p_pelvis + R8_I  @ sp.Matrix([0, 0.10, 0])
p_knee1 = p_hip1 + R9_I  @ sp.Matrix([0, 0, -0.44])
p_ankle1 = p_knee1 + R10_I @ sp.Matrix([0, 0, -0.42])
p_hip2 = p_pelvis + R8_I  @ sp.Matrix([0, -0.10, 0])
p_knee2 = p_hip2 + R11_I @ sp.Matrix([0, 0, -0.44])
p_ankle2 = p_knee2 + R12_I @ sp.Matrix([0, 0, -0.42])

# Evaluation function to obtain numerical values for positions
evaluate_positions = sp.lambdify((phi, theta, psi, x, y, z), [p_head, p_chin, p_neck, p_shoulder1, p_elbow1,
                                                              p_wrist1, p_shoulder2, p_elbow2, p_wrist2, p_pelvis,
                                                              p_hip1, p_knee1, p_ankle1, p_hip2, p_knee2, p_ankle2])


# Numerical values for the rotation angles (in radians)
# inertial,neck, l_neck,l_shoulder,l_elbow,r_neck,r_shoulder,r_elbow,pelvis,l_hip,l_knee,r_hip,r_knee
rotation_values_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rotation_values_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
rotation_values_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# Define numerical values for x, y, and z coordinates (in meters)
x_val, y_val, z_val = 0.0, 0.0, 0.75

# Evaluate the positions of body segments
positions = evaluate_positions(
    rotation_values_x, rotation_values_y, rotation_values_z, x_val, y_val, z_val)

# Extract x, y, z coordinates for each body segment
x_coords, y_coords, z_coords = zip(
    *[(float(p[0]), float(p[1]), float(p[2])) for p in positions])

# Define joint names and their corresponding indices
joint_names = {
    'head': 0,
    'chin': 1,
    'neck': 2,
    'l_shoulder': 3,
    'l_elbow': 4,
    'l_wrist': 5,
    'r_shoulder': 6,
    'r_elbow': 7,
    'r_wrist': 8,
    'pelvis': 9,
    'l_hip': 10,
    'l_knee': 11,
    'l_ankle': 12,
    'r_hip': 13,
    'r_knee': 14,
    'r_ankle': 15
}

# Create an array of links between joints (pairs of joint indices)
links = [
    (joint_names['chin'], joint_names['head']),         # Link between chin and head
    (joint_names['neck'], joint_names['head']),         # Link between neck and head
    (joint_names['neck'], joint_names['l_shoulder']),   # Link between neck and left shoulder
    (joint_names['neck'], joint_names['r_shoulder']),   # Link between neck and right shoulder
    (joint_names['neck'], joint_names['pelvis']),       # Link between neck and pelvis
    (joint_names['l_shoulder'], joint_names['l_elbow']), # Link between left shoulder and left elbow
    (joint_names['l_elbow'], joint_names['l_wrist']),   # Link between left elbow and left wrist
    (joint_names['r_shoulder'], joint_names['r_elbow']), # Link between right shoulder and right elbow
    (joint_names['r_elbow'], joint_names['r_wrist']),   # Link between right elbow and right wrist
    (joint_names['pelvis'], joint_names['l_hip']),      # Link between pelvis and left hip
    (joint_names['pelvis'], joint_names['r_hip']),      # Link between pelvis and right hip
    (joint_names['l_hip'], joint_names['l_knee']),      # Link between left hip and left knee
    (joint_names['l_knee'], joint_names['l_ankle']),    # Link between left knee and left ankle
    (joint_names['r_hip'], joint_names['r_knee']),      # Link between right hip and right knee
    (joint_names['r_knee'], joint_names['r_ankle']),    # Link between right knee and right ankle
]

# Plot the skeleton
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define joint markers
joint_labels = ['Head', 'Chin', 'Neck', 'Left Shoulder', 'Left Elbow', 'Left Wrist',
                'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Pelvis', 'Left Hip',
                'Left Knee', 'Left Ankle', 'Right Hip', 'Right Knee', 'Right Ankle']

# Plot each body segment and label the joints
for i in range(len(x_coords)):
    ax.scatter(x_coords[i], y_coords[i], z_coords[i], color='black', s=10)
    ax.text(x_coords[i], y_coords[i], z_coords[i],
            joint_labels[i], fontsize=8, ha='right', va='bottom')

# Draw lines between linked joints
for link in links:
    i, j = link
    ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]],
            [z_coords[i], z_coords[j]], color='blue', linestyle='-')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits (adjust these based on your data)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)

# Show the plot
#plt.show()

# Create a Tkinter GUI
class SkeletonGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # Set up the GUI elements
        self.title("Skeleton Pose Manipulation")
        self.geometry("1000x600")
        self.create_widgets()

    def create_widgets(self):
        # Create sliders for each joint
        self.sliders = []
        joint_names = ["inertial", "neck", "l_neck", "l_shoulder", "l_elbow", 'r_neck', 'r_shoulder', "r_elbow", "pelvis", "l_hip", "l_knee", "r_hip", "r_knee"]
        limits = {"inertial"    :   {"x": [-np.pi, np.pi], "y": [-np.pi, np.pi], "z": [-np.pi, np.pi]},
                  "neck"        :   {"x": [-np.pi/2, np.pi/2], "y": [-np.pi/2, np.pi/2], "z": [-np.pi/2, np.pi/2]},
                  "l_neck"      :   {"x": [0, np.pi/4], "y": [0, 0], "z": [-np.pi/4, np.pi/4]},
                  "l_shoulder"  :   {"x": [-np.pi, np.pi], "y": [-np.pi, np.pi], "z": [-np.pi/2, np.pi/2]}, 
                  "l_elbow"     :   {"x": [-np.pi, np.pi], "y": [-np.pi, 0], "z": [0, 0]},
                  "r_neck"      :   {"x": [-np.pi/4, 0], "y": [0, 0], "z": [-np.pi/4, np.pi/4]},
                  "r_shoulder"  :   {"x": [-np.pi, np.pi], "y": [-np.pi, np.pi], "z": [-np.pi/2, np.pi/2]}, 
                  "r_elbow"     :   {"x": [-np.pi, np.pi], "y": [-np.pi, 0], "z": [0, 0]},
                  "pelvis"      :   {"x": [0, 0], "y": [0, 0], "z": [-np.pi/4, np.pi/4]},
                  "l_hip"       :   {"x": [-np.pi/2, np.pi], "y": [-np.pi, np.pi/4], "z": [-np.pi/2, np.pi/2]},
                  "l_knee"      :   {"x": [0, 0], "y": [0, np.pi], "z": [0, 0]},
                  "r_hip"       :   {"x": [-np.pi, np.pi/2], "y": [-np.pi, np.pi/4], "z": [-np.pi/2, np.pi/2]},
                  "r_knee"      :   {"x": [0, 0], "y": [0, np.pi], "z": [0, 0]}
                  }

        #X Axis Sliders
        for row, joint_name in enumerate(joint_names):
            joint_label = ttk.Label(self, text=joint_name)
            joint_label.grid(row=row, column=0)

            joint_slider = ttk.Scale(self, from_=limits[joint_name]["x"][0], to=limits[joint_name]["x"][1], command=self.update_skeleton_pose)
            joint_slider.grid(row=row, column=1)
            self.sliders.append(joint_slider)

        #Y Axis Sliders
        for row, joint_name in enumerate(joint_names):
            joint_slider = ttk.Scale(self, from_=limits[joint_name]["y"][0], to=limits[joint_name]["y"][1], command=self.update_skeleton_pose)
            joint_slider.grid(row=row, column=2)
            self.sliders.append(joint_slider)

        #Z Axis Sliders
        for row, joint_name in enumerate(joint_names):
            joint_slider = ttk.Scale(self, from_=limits[joint_name]["z"][0], to=limits[joint_name]["z"][1], command=self.update_skeleton_pose)
            joint_slider.grid(row=row, column=3)
            self.sliders.append(joint_slider)

    def update_skeleton_pose(self, event):
        # Get the values from the sliders
        rotation_values = [s.get() for s in self.sliders]
        rotation_values_x = rotation_values[:13]
        rotation_values_y = rotation_values[13:26]
        rotation_values_z = rotation_values[26:]

        # Evaluate the positions of body segments
        positions = evaluate_positions(rotation_values_x, rotation_values_y, rotation_values_z, x_val, y_val, z_val)

        # Extract x, y, z coordinates for each body segment
        x_coords, y_coords, z_coords = zip(*[(float(p[0]), float(p[1]), float(p[2])) for p in positions])

        # Update the plot
        self.ax.clear()
        for i in range(len(x_coords)):
            self.ax.scatter(x_coords[i], y_coords[i], z_coords[i], color='black', s=10)
            #self.ax.text(x_coords[i], y_coords[i], z_coords[i], joint_labels[i], fontsize=8, ha='right', va='bottom')

        for link in links:
            i, j = link
            self.ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], [z_coords[i], z_coords[j]], color='blue', linestyle='-')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        self.canvas.draw()

    def plot_skeleton(self):
        # Plot the skeleton initially
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=4, rowspan=len(joint_names), padx=10, pady=10)

        self.update_skeleton_pose(None)


if __name__ == "__main__":
    # Create the GUI and run the main loop
    root = SkeletonGUI()
    root.plot_skeleton()
    root.mainloop()