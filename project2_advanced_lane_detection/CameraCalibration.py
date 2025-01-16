import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class CameraCalibration():
    def __init__(self, image_dir, nx, ny, debug=False):
        fnames = glob.glob("{}/*".format(image_dir))
        objpoints = []
        imgpoints = []
        
        # 체스판 모서리의 좌표
        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        
        # 체스판 이미지 확인
        for f in fnames:
            img = mpimg.imread(f)
            
            # grayscale 이미지로 변경
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 체스판 모서리 찾기
            ret, corners = cv2.findChessboardCorners(img, (nx,ny))
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
        
        shape = (img.shape[1], img.shape[0])
        # 카메라 캘리브레이션 - 렌즈 왜곡으로 인해 이미지가 휘거나 왜곡되는 현상을 보정
        ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        
        if not ret:
            raise Exception("Unable to calibrate camera")

    def undistort(self, img):
        # 카메라 왜곡 펴기
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)