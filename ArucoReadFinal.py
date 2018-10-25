"""
Load camera data, then infinite loop
recognize points, find the minimum distance while camera is still the origin
send the id of the minimum distance aruco to point of view converter then adjust based on the id position
output the closest one to the flight controller.




try peel back the extra brackets and if rodriquez and others down stream still work
"""
import numpy as np
import cv2
import cv2.aruco as aruco
import math

#cap = cv2.VideoCapture(0)    # video capture(id of camera)
#cap = cv2.VideoCapture('http://10.186.38.197:8080/video') # video capture object (#ofcamera)
cap = cv2.VideoCapture('http://10.0.0.43:8080/video') # video capture object (#ofcamera)


arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Specify the dictionary
parameters = aruco.DetectorParameters_create()       # Specify detection parameters

#load previously saved calibration data
with np.load('PhoneCalibration.npz') as X:
    cameraMatrix, distCoeffs = [X[i] for i in ('cameraMatrix', 'distCoeffs')]

cv2.namedWindow('Video Out', cv2.WINDOW_NORMAL)     # enable resize

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
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)
        l = []
        for x in range(len(ids)):
            vec = tvecs[x][0]
            magvec = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
            l.append(magvec)

        C = np.argmin(l)                    # closest marker to the camera
        R, jac = cv2.Rodrigues(rvecs[C])    # From Rotation Vector to Rotation matrix
        camR = np.transpose(R)                          # Camera Rotation Matrix
        T = np.transpose(tvecs[C])          # Prepare aruco translation for MatMul
        camT = np.transpose(np.matmul(-camR, T))        # Camera Translation Vector

        # here create a switch case for different ID for their relation to origin and Adjust camT
        if (ids[C] == 6):
            camT[0][0] = camT[0][0] + 6

        print(ids[C])
        print(camT)
        # draw marker and id
        image = aruco.drawDetectedMarkers(image, corners, ids, borderColor=(0, 255, 0))
        # draw the pose axis of marker
        image = aruco.drawAxis(image, cameraMatrix, distCoeffs, rvecs[C], tvecs[C], 1)

    # display the resulting frame
    cv2.imshow('Video Out', image)                      # show image
    cv2.resizeWindow('Video Out', 2048, 1840)             # image resize
    #out.write(image)
    if cv2.waitKey(1) % 0xFF == ord('q'):               # quit at "q"
        break

# when everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
