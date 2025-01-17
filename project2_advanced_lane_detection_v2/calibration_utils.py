import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

def lazy_calibration(func):
    """
    재계산 방지를 위한 calibrate_camera 데코레이터
    calibrate_camera() 함수 호출 시 lazy_calibration 데코레이션의 wrapper 실행 후 calibrate_camera 함수 실행
    """
    calibration_cache = 'project2_advanced_lane_detection_v2/camera_cal/calibration_data.pickle'
    
    def wrapper(*args, **kwargs):
        if os.path.exists(calibration_cache):
            print('Loading cached camera calibration...', end=' ')
            with open(calibration_cache, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            print('Computing camera calibration...', end=' ')
            calibration = func(*args, **kwargs)
            with open(calibration_cache, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)
        print('Done.')
        return calibration

    return wrapper


@lazy_calibration # 데코레이션
def calibrate_camera(calib_images_dir, verbose=False):
    assert os.path.exists(calib_images_dir), '"{}" must exist and contain calibration images.'.format(calib_images_dir)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # 사물 점(포인트) 초기화
    # np.mgrid: 격자형 좌표 배열 생성기
    objp = np.zeros((6*9,3), np.float32) # (54, 3)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # (6x9=54,2) 텐서로 변환 후 3열전까지 업데이트
    
    objpoints = []
    imgpoints = []
    
    images = glob.glob(os.path.join(calib_images_dir, 'calibration*.jpg'))
    
    # 체크보드에 코너 찾기
    for filename in images:
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        pattern_found, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        if pattern_found is True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            if verbose:
                # 코너 그리기
                img = cv2.drawChessboardCorners(img, (9,6), corners, pattern_found)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    
    if verbose:
        cv2.destroyAllWindows()
    
    # shape[::-1] => 원래 차원 순서를 뒤집어 반환. (height, width) => (width, height)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return ret, mtx, dist, rvecs, tvecs


def undistort(frame, mtx, dist, verbose=False):
    # 카메라 왜곡 펴기
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)
    
    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()
    
    return frame_undistorted


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='project2_advanced_lane_detection_v2/camera_cal')
    
    img = cv2.imread('project2_advanced_lane_detection_v2/test_images/test2.jpg')
    
    img_undistorted = undistort(img, mtx, dist)
    
    cv2.imshow('test_calibration_before', img)
    cv2.imshow('test_calibration_after', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()