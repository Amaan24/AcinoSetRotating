import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

import tkinter as tk
from tkinter import filedialog

# Parameters
model_slack_sigma = [0.01, 22.36] 
slack_meas_sigma = 0.5
#enc_slack_sigma = 1000*0.0174533
model_slack_weight = [1/model_slack_sigma[0]**2, 1/model_slack_sigma[1]**2]   
slack_meas_err_weight = 1/slack_meas_sigma**2
#enc_slack_weight = 1/enc_slack_sigma**2

print(f"Model Slack Weight: {model_slack_weight}")
print(f"Measurement Slack Weight: {slack_meas_err_weight}")
#print(f"Encoder Measurement Slack Weight: {enc_slack_weight}")


# Load the data from the pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        file_data = pickle.load(f)
    return file_data


# Create a GUI window for file selection
root = tk.Tk()
root.withdraw()  # Hide the main window

# Prompt the user to select a file
results_file_path = filedialog.askopenfilename(
    title="Select a Pickle File",
    filetypes=[("Pickle files", "*.pickle")]
)

if not results_file_path:
    print("No file selected. Exiting.")
    exit()
else:
    # Load the data
    data = load_data(results_file_path)

    # Extract data
    model_slack = data['model_slack']
    slack_meas = data['slack_meas']
    #enc_slack_meas = data['enc_slack_meas']

    # Create a figure for model_slack
    plt.figure(figsize=(10, 5))
    plt.title('Model Slack')
    plt.xlabel('N')
    plt.ylabel('Scaled and Squared Value')

    # Calculate and plot model_slack for each P
    for p in range(model_slack.shape[1]):
        if p < 3:
            scaled_squared = (model_slack[:, p] ** 2) * model_slack_weight[0]
        else:
            scaled_squared = (model_slack[:, p] ** 2) * model_slack_weight[1]
        plt.plot(scaled_squared, label=f'P={p}')

    grandtotal = 0
    total_sum = 0
    # Calculate and plot the total sum
    for p in range(model_slack.shape[1]):
        for n in range(model_slack.shape[0]):
            if p < 3:
                total_sum += model_slack_weight[0] * (model_slack[n, p] ** 2) 
            else:
                total_sum += model_slack_weight[1] * (model_slack[n, p] ** 2) 
    print(f"Model Slack Error Total Sum: {total_sum}")
    grandtotal = total_sum
    plt.legend()
    plt.grid(True)

    # Camera 1 Slack Measurement
    # Create a figure for C1 Residual
    plt.figure(figsize=(10, 5))
    plt.title('Camera 1 Slack Meas Per Labelled Point')
    plt.xlabel('N')
    plt.ylabel('slack weight * (X_slack**2 + Y_slack**2)')

    # Calculate and plot Residuals for C1
    c = 0
    for l in range(slack_meas.shape[2]):
        squared_values = (slack_meas[:, c, l, 0] ** 2 + slack_meas[:, c, l, 1] ** 2) * slack_meas_err_weight
        plt.plot(squared_values, label=f'L={l}')

    plt.legend()
    plt.grid(True)

    total_sum = 0
    for n in range(slack_meas.shape[0]):
        for l in range(slack_meas.shape[2]):
            for d in range(slack_meas.shape[3]):
                total_sum += slack_meas_err_weight * (slack_meas[n][0][l][d]**2)
    print(f"Camera 1 Measurement Slack Error: {total_sum}")
    grandtotal += total_sum

    # Camera 2 Slack Measurement
    # Create a figure for slack_meas
    plt.figure(figsize=(10, 5))
    plt.title('Camera 2 Slack Meas Per Labelled Point')
    plt.xlabel('N')
    plt.ylabel('slack weight * (X_slack**2 + Y_slack**2)')

    # Calculate and plot slack_meas for each C
    c = 1
    for l in range(slack_meas.shape[2]):
        squared_values = (slack_meas[:, c, l, 0] ** 2 + slack_meas[:, c, l, 1] ** 2) * slack_meas_err_weight
        plt.plot(squared_values, label=f'L={l}')

    plt.legend()
    plt.grid(True)

    total_sum = 0
    for n in range(slack_meas.shape[0]):
        for l in range(slack_meas.shape[2]):
            for d in range(slack_meas.shape[3]):
                total_sum += slack_meas_err_weight * (slack_meas[n][1][l][d]**2)
    print(f"Camera 2 Measurement Slack Error: {total_sum}")
    grandtotal += total_sum
    print(f"Total Error: {grandtotal}")
    # Camera 2 Slack Measurement
    # Create a figure for encoder slack_meas
    # plt.figure(figsize=(10, 5))
    # plt.title('Encoder Slack Measument')
    # plt.xlabel('N')
    # plt.ylabel('slack weight * slack**2)')

    # # Calculate and plot encoder slack measurement 
    # for c in range(enc_slack_meas.shape[1]):
    #     squared_values = (enc_slack_meas[:, c] ** 2 + enc_slack_meas[:, c] ** 2) * enc_slack_weight
    #     plt.plot(squared_values, label=f'C={c}')

    # plt.legend()
    # plt.grid(True)

    # total_sum = 0
    # for n in range(enc_slack_meas.shape[0]):
    #     for c in range(enc_slack_meas.shape[1]):
    #         total_sum += enc_slack_weight * (enc_slack_meas[n][c]**2)
    # print(f"Encoder Slack Error: {total_sum}")

    plt.show()
