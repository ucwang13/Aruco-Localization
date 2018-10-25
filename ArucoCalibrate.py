
"""
I can also save these images that pass

I should pause so the thing doesn't take repeated images


program starts, press start detection
when image is detected, image save, image pause until space is pressed
then reposition, then press another key to resume detection

maybe 1 full time video window, one other window close after short delay
"""

import numpy as np
import cv2

sidelength = 2.5
# termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
objp = objp*sidelength

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
img_count = 0
delay = 0

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://10.0.0.43:8080/video') # video capture object (#ofcamera)

while img_count < 30:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret1, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret1 is True and delay > 30:
        img_count += 1
        img_name = "calibration/opencv_frame_{}.jpg".format(img_count)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (7, 6), corners2, ret1)
        cv2.imshow('snap', frame)
        cv2.waitKey(1)
        delay = 0

    cv2.imshow('img', frame)
    delay += 1
    print(delay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cap = cv2.VideoCapture('http://10.0.0.43:8080/video')  # video capture object (#ofcamera)

cap.release()
cv2.destroyAllWindows()
#save data
ret, cameraMatrix, distCoeffs, rvec, tvec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('camCalibration.npz', cameraMatrix=cameraMatrix, distCoeffs=distCoeffs, rvec=rvec, tvec=tvec)


