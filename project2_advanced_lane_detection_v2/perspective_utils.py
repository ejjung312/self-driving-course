import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from project2_advanced_lane_detection_v2.calibration_utils import calibrate_camera, undistort
from project2_advanced_lane_detection_v2.binarization_utils import binarize

"""
프레임에 원근 변환을 적용하여 상공에서 보는 관점을 얻음
"""
def birdeye(img, verbose=False):
    h,w = img.shape[:2]
    
    src = np.float32([[w, h-10],  # br
                    [0, h-10],    # bl
                    [546, 460],   # tl
                    [732, 460]])  # tr
    dst = np.float32([[w, h],       # br
                    [0, h],       # bl
                    [0, 0],       # tl
                    [w, 0]])      # tr

    # 이미지를 일부분을 펼치거나 좁힘 => 기하학적 변환
    # src, dst의 관계를 계산하여 변환행렬을 반환
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # 변환행렬으로 이미지를 기하학적으로 변환시킴
    img_birdeye = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    
    if verbose:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('Before perspective transform')
        axarray[0].imshow(img, cmap='gray')
        
        for point in src:
            axarray[0].plot(*point, '.')
            
        axarray[1].set_title('After perspective transform')
        axarray[1].imshow(img_birdeye, cmap='gray')
        
        for point in dst:
            axarray[1].plot(*point, '.')
        
        for axis in axarray:
            axis.set_axis_off()
        
        plt.show()

    return img_birdeye, M, Minv


if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    for test_img in glob.glob('project2_advanced_lane_detection_v2/test_images/*.jpg'):
        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary = binarize(img_undistorted, verbose=False)

        img_birdeye, M, Minv = birdeye(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB), verbose=True)