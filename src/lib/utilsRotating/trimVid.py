import cv2
import numpy as np
import os

cwd = 'C:\\Users\\user-pc\\Desktop\\15Nov2022'

vid_path1 = os.path.join(cwd, "1.avi")
vid_path2 = os.path.join(cwd, "2.avi")

out_path1 = os.path.join(cwd, "1_trimmed.avi")
out_path2 = os.path.join(cwd, "2_trimmed.avi")

frame = 5000

cap1 = cv2.VideoCapture(vid_path1)
cap1.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)

cap2 = cv2.VideoCapture(vid_path2)
cap2.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)

numFrames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

fps = 60
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter(out_path1, fourcc, fps, (1280, 720))
out2 = cv2.VideoWriter(out_path2, fourcc, fps, (1280, 720))

curr_frame = frame - 1

while curr_frame < (frame - 1 + numFrames):
    res, frame1 = cap1.read()
    res, frame2 = cap2.read()

    #alpha = 1.5
    #beta = 60
    #frame1 = cv2.convertScaleAbs(frame1, alpha=alpha, beta=beta)

    out1.write(frame1)
    out2.write(frame2)

    curr_frame += 1

out1.release()
out2.release()
cap1.release()
cap2.release()
