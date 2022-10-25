import cv2
import numpy as np
import os

cwd = 'C:\\Users\\user-pc\\Desktop\\11Oct2022S'

vid_path1 = os.path.join(cwd, "1.avi")
vid_path2 = os.path.join(cwd, "2.avi")

#vid_path1 = os.path.join(cwd, "1_trimmed.avi")
#vid_path2 = os.path.join(cwd, "2_trimmed.avi")

#vid_path1 = os.path.join(cwd, "DLC", "1DLC_resnet101_human_pretrainedFeb18shuffle1_103000_labeled.mp4")
#vid_path2 = os.path.join(cwd, "DLC", "2DLC_resnet101_human_pretrainedFeb18shuffle1_103000_labeled.mp4")

frame_num = 7000
step = 1
show_corners = 0

while 1:
    cap = cv2.VideoCapture(vid_path1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    res, frame1 = cap.read()

    cap = cv2.VideoCapture(vid_path2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
    res, frame2 = cap.read()

    if show_corners:
        ret, corners = cv2.findChessboardCorners(frame1, (9, 6), None)
        frame1corners = frame1
        if ret:
            cv2.drawChessboardCorners(frame1corners, (9, 6), corners, ret)

        frame2corners = frame2
        ret, corners = cv2.findChessboardCorners(frame2, (9, 6), None)
        if ret:
            cv2.drawChessboardCorners(frame2corners, (9, 6), corners, ret)

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
    elif key == ord('c'):
        if show_corners == 0:
            show_corners = 1
        else:
            show_corners = 0
    elif key == ord('1'):
        try:
            if not os.path.exists(os.path.join(cwd, 'in_calib1')):
                os.makedirs(os.path.join(cwd, 'in_calib1'))
        except OSError:
            continue
        cv2.imwrite(os.path.join(cwd, 'in_calib1', (str(frame_num) + '.jpg')), frame1)
    elif key == ord('2'):
        try:
            if not os.path.exists(os.path.join(cwd, 'in_calib2')):
                os.makedirs(os.path.join(cwd, 'in_calib2'))
        except OSError:
            continue
        cv2.imwrite(os.path.join(cwd, 'in_calib2', (str(frame_num) + '.jpg')), frame2)
    elif key == ord('3'):
        try:
            if not os.path.exists(os.path.join(cwd, 'ex_calib1')) and not os.path.exists(
                    os.path.join(cwd, 'ex_calib2')):
                os.makedirs(os.path.join(cwd, 'ex_calib1'))
                os.makedirs(os.path.join(cwd, 'ex_calib2'))
        except OSError:
            continue
        cv2.imwrite(os.path.join(cwd, 'ex_calib1', (str(frame_num) + '.jpg')), frame1)
        cv2.imwrite(os.path.join(cwd, 'ex_calib2', (str(frame_num) + '.jpg')), frame2)
    elif key == ord('q'):
        break

