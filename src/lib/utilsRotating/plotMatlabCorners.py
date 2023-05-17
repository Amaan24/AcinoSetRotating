import scipy.io as sio
import cv2
import numpy as np
import os


cwd = 'C:\\Users\\user-pc\\Desktop\\25Apr2023' #14000 to 20000 only

vid_path1 = os.path.join(cwd, "1.avi")
vid_path2 = os.path.join(cwd, "2.avi")

mat_contents = sio.loadmat(os.path.join(cwd, "checkerboard_corners.mat"))

frame_num = 1000
step = 1

while 1:
    cap = cv2.VideoCapture(vid_path1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame1 = cap.read()

    for i in range(20):
        try:
            u1 = int(mat_contents['cornerPoints'][frame_num][0][i][0])
            v1 = int(mat_contents['cornerPoints'][frame_num][0][i][1])
            frame1 = cv2.circle(frame1, (u1,v1), radius=5, color=(0, 0, 255), thickness=-1)
        except:
            print(f"An exception occurred in frame {frame_num}")

    cap = cv2.VideoCapture(vid_path2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    res, frame2 = cap.read()

    for i in range(20):
        try:
            u2 = int(mat_contents['cornerPoints'][frame_num][1][i][0])
            v2 = int(mat_contents['cornerPoints'][frame_num][1][i][1])
            frame2 = cv2.circle(frame2, (u2,v2), radius=5, color=(0, 0, 255), thickness=-1)
        except:
            print(f"An exception occurred in frame {frame_num}")

    frame1_rs = cv2.resize(frame1, (0, 0), None, .75, .75)
    frame2_rs = cv2.resize(frame2, (0, 0), None, .75, .75)

    frames = np.hstack((frame1_rs, frame2_rs))

    font_size = 0.3
    colour = (0, 0, 0)
    cv2.putText(frames, ('Step: ' + str(step)), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
    cv2.putText(frames, 'W/S - Change Step', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
    cv2.putText(frames, 'A/D - Prev/Next', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
    cv2.putText(frames, '1/2/3 - Save 1/2/Both', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
    cv2.putText(frames, 'Q - Quit', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, font_size, colour, 1, cv2.LINE_AA)
    cv2.imshow('Synced Frames', frames)
    cv2.setWindowTitle('Synced Frames', ('Frame ' + str(frame_num)))

    key = cv2.waitKey(33)
    if key == ord('d'):
        frame_num += 1*step
    elif key == ord('a'):
        frame_num -= 1*step
    elif key == ord('s'):
        step -= 1
        if step < 1:
            step = 1
    elif key == ord('w'):
        step += 1
    elif key == ord('q'):
        break

