import pickle
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

file = "C:\\Users\\user-pc\\Desktop\\AcinoSetRotating\\data\\11Oct2022S\\results\\traj_results.pickle"

def load_pickle(pickle_file) -> Dict:
    """
    Loads a .pickle file
    """
    with open(pickle_file, 'rb') as handle:
        data = pickle.load(handle)
    return data


opt_results = load_pickle(file)
print(opt_results.keys())

model_slack = opt_results['x_model_slack']
x_cam = opt_results['x_cam']
dx_cam = opt_results['dx_cam']
ddx_cam = opt_results['ddx_cam']
enc_slack = opt_results['x_cam_model_slack']

slack_meas = opt_results['slack_meas']
#slack_meas.reshape(20, 28, 2)

print(slack_meas.shape)

meas_slack = []
for frame in range (0, len(slack_meas)):
    temp = 0
    for label in range(0, len(slack_meas[0])):
        for point in range(0, 1):
            temp+=abs(slack_meas[frame, label, point])
    meas_slack.append(temp)

model_slack_m = []
model_slack_rad = []
for frame in range(0, len(model_slack)):
    m_sum = 0
    rad_sum = 0
    for joint in range (0, 2):
        m_sum += model_slack[frame][joint]
    for joint in range (3, len(model_slack[frame])):
        rad_sum += model_slack[frame][joint]
    model_slack_m.append(m_sum)
    model_slack_rad.append(rad_sum)

frames = np.arange(0, len(x_cam))

fig, axs = plt.subplots(2, 3)
axs[0, 0].plot(frames, x_cam[:, 0], "C0", label='Alpha')
axs[0, 0].plot(frames, dx_cam[:, 0], "C1", label='dAlpha')
axs[0, 0].plot(frames, ddx_cam[:, 0], "C2", label='ddAlpha')
axs[0, 0].set_title("Alpha, dAlpha, ddAlpha")
axs[0, 0].set_xlabel('Frame (N)')
axs[0, 0].set_ylabel('Angle rad rad/s rad/s^2')
axs[0, 0].legend()

axs[1, 0].plot(frames, x_cam[:, 1], "C1", label='Beta')
axs[1, 0].plot(frames, dx_cam[:, 1], "C2", label='dBeta')
axs[1, 0].plot(frames, ddx_cam[:, 1], "C3", label='ddBeta')
axs[1, 0].set_title("Beta, dBeta, ddBeta")
axs[1, 0].set_xlabel('frame (N)')
axs[1, 0].set_ylabel('Angle rad rad/s rad/s^2')
axs[1, 0].legend()

axs[0, 1].set_title("Model Slack")
axs[0, 1].plot(frames, model_slack_m, "C0", label='Slack m')
axs[0, 1].plot(frames, model_slack_rad, "C1", label='Slack rad')
axs[0, 1].set_xlabel('frame (N)')
axs[0, 1].set_ylabel('Total Slack m/s^2 rad/s^2')
axs[0, 1].legend()

axs[1, 1].set_title("Encoder Slack")
axs[1, 1].plot(frames, enc_slack[:, 0], "C1", label='Enc1')
axs[1, 1].plot(frames, enc_slack[:, 1], "C2", label='Enc2')
axs[1, 1].set_xlabel('frame (N)')
axs[1, 1].set_ylabel('Slack (rad/s^2)')
axs[1, 1].legend()

axs[0, 2].set_title("Meas Slack Per Frame")
axs[0, 2].plot(frames, meas_slack, "C0", label='Meas Slack')
axs[0, 2].set_xlabel('frame (N)')
axs[0, 2].set_ylabel('Slack (pixels)')
axs[0, 2].legend()

fig.tight_layout()
plt.show()