import os
import cv2

cwd = 'C:\\Users\\user-pc\\Desktop\\22Sep2022'
date = '2022_09_22_05_46'

vid1 = os.path.join(cwd, ('1_video-' + date + '.mp4'))
vid2 = os.path.join(cwd, ('2_video-' + date + '.mp4'))

cwd = 'C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\Rig'

vid1 = os.path.join(cwd, "1_Synced.avi")
vid2 = os.path.join(cwd, "2_Synced.avi")

#vid1 = os.path.join(cwd, ('1.avi'))
#vid2 = os.path.join(cwd, ('2.avi'))

cap = cv2.VideoCapture(vid1)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(length)

cap = cv2.VideoCapture(vid2)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(length)
