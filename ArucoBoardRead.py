"""
using pyplot interactive mode to stop figure hang after show
"""
import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from collections import deque

fig = plt.figure()
cap = cv2.VideoCapture(0)    # video capture(id of camera)
#cap = cv2.VideoCapture('http://192.168.1.109:8080/video') # video capture object (#ofcamera)
#cap = cv2.VideoCapture('http://10.0.0.43:8080/video') # video capture object (#ofcamera)

arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Specify the dictionary
parameters = aruco.DetectorParameters_create()       # Specify detection parameters
board = aruco.GridBoard_create(3, 4, 0.3, 0.3, arucoDict)

img = aruco.drawPlanarBoard(board, (1200, 1500))
cv2.imshow('Aruco Marker', img)
# multiple creations of moving average.
avx = deque([0, 0, 0])
avy = deque([0, 0, 0])
avz = deque([0, 0, 0])
avg = len(avx)
# load previously saved calibration data
with np.load('camCalibration.npz') as X:
    cameraMatrix, distCoeffs = [X[i] for i in ('cameraMatrix', 'distCoeffs')]
#cv2.namedWindow('Video Out', cv2.WINDOW_NORMAL)     # enable resize
while True:
    # capture frame by frame
    ret, image = cap.read()

    # Operations on the frame come here
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect for aruco marker
    corners, ids, rejectedImgPoints = aruco.detectMarkers(image, arucoDict, parameters=parameters)
    # If at least one marker detected, process aruco marker
    if len(corners) > 0:
        # estimate the pose of each marker
        retval, rvecs, tvecs = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)

        camR = np.transpose(rvecs)
        R, jac = cv2.Rodrigues(camR)  # From Rotation Vector to Rotation matrix
        camR = np.transpose(R)  # Camera Rotation Matrix
        #T = np.transpose(tvecs)  # Prepare aruco translation for MatMul
        camT = np.transpose(np.matmul(-camR, tvecs))  # Camera Translation Vector

        X = camT[0][0]
        Y = camT[0][1]
        Z = camT[0][2]

        print(camT)
        plt.scatter(X, Y, c='r')
        plt.scatter(9.5, Z, c='b')
        plt.axis([-5, 10, -5, 30])
        plt.pause(0.000001)
        plt.cla()
        # draw marker and id
        image = aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 255, 0))
        # draw the pose axis of marker
        image = aruco.drawAxis(image, cameraMatrix, distCoeffs, rvecs, tvecs, 1)

    # display the resulting frame
    cv2.imshow('Video Out', image)                      # show image
    #cv2.resizeWindow('Video Out', 2048, 1840)             # image resize
    if cv2.waitKey(1) % 0xFF == ord('q'):               # quit at "q"
        break

# when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
