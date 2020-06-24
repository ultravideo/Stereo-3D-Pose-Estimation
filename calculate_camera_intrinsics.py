'''
    How to use:
        open the included checkerboard image on the screen.
        Run the file.
        Point the camera at the checkerboard and press 'R' every time you 
            want to add another sample (5-10 times should be enough).
        Press 'Q' to quit.
        The intrinsics matrix is printed to the terminal.
'''

import numpy as np
import cv2

checkerboard_images = []

cam = cv2.VideoCapture(4)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)

while True:
    img = cam.read()[1]
    img = cv2.resize(img, (360, 270))

    cv2.imshow("img", img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('a'):
        checkerboard_images.append(img)

    
print("checkerboard images captured")

CHECKERBOARD = (6, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpoints = []
imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

for img in checkerboard_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow("img", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
h, w, _ = img.shape

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
