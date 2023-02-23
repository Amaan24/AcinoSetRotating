import cv2
import numpy as np

cap = cv2.VideoCapture('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPLCalib.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPLCalib.avi',fourcc, 60, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap = cv2.VideoCapture('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPRCalib.mp4')
out = cv2.VideoWriter('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPRCalib.avi',fourcc, 60, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap = cv2.VideoCapture('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPLSynced.mp4')
out = cv2.VideoWriter('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPLSynced.avi',fourcc, 60, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap = cv2.VideoCapture('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPRSynced.mp4')
out = cv2.VideoWriter('C:\\Users\\user-pc\\Desktop\\FinalHuman\\Recon\\GoPro\\GPRSynced.avi',fourcc, 60, (1280,720))

while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        out.write(b)
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()