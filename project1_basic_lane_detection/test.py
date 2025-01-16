import cv2
import numpy as np

# Load the image
image_path = 'project1_lane_detection/camera_cal/calibration1.jpg'
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect corners using goodFeaturesToTrack
corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=10)

# Define the chessboard size (internal corners in the grid)
pattern_size = (7, 4)

# Prepare object points (3D points in real world space)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane

# If corners were detected, process further
if corners is not None:
    corners = np.int64(corners)

    # Sort and filter the detected corners to match the pattern
    corners = sorted(corners, key=lambda x: (x.ravel()[1], x.ravel()[0]))
    if len(corners) >= len(objp):
        imgpoints.append(np.array(corners[:len(objp)], dtype=np.float32).reshape(-1, 2))
        objpoints.append(objp)

        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Undistort the image
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        # Show results
        cv2.imshow("Original Image", img)
        cv2.imshow("Undistorted Image", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save the undistorted image
        # cv2.imwrite("undistorted_image_goodFeatures.jpg", undistorted)

        # Print camera matrix and distortion coefficients
        # print("Camera Matrix:\n", mtx)
        # print("Distortion Coefficients:\n", dist)
    else:
        print("Not enough corners detected to perform calibration.")
else:
    print("No corners were detected.")
