import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img) # 이미지 최소 픽셀 값
    vmax = np.max(img) # 이미지 최대 픽셀 값
    
    # 밝기값의 하한선(vlo): 최소값에서 전체범위(vmax - vmin)의 lo비율만큼 더함
    # 밝기값의 상한선(vhi): 최소값에서 전체범위(vmax - vmin)의 hi비율만큼 더함
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    
    # img에서 특정 밝기 범위(vlo와 vhi)에 해당하는 픽셀 선택
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    def __init__(self):
        pass
    
    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS) # HLS로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) # HSV로 변환
        h_channel = hls[:,:,0] # Hue 채널
        l_channel = hls[:,:,1] # Lightness 채널
        s_channel = hls[:,:,2] # Saturation 채널
        v_channel = hsv[:,:,2] # Value 채널
        
        # 오른쪽 차선 감지
        right_lane = threshold_rel(l_channel, 0.8, 1.0) # 0.8~1.0 사이의 이진화된 이미지
        right_lane[:,:750] = 0 # 열 ~749까지 0으로 만듬
        
        # 왼쪽 차선 감지
        left_lane = threshold_abs(h_channel, 20, 30) # 20~30 사이의 이진화된 이미지
        left_lane &= threshold_rel(v_channel, 0.7, 1.0) # 0.7~1.0 사이의 이진화된 이미지
        left_lane[:,550:] = 0 # 열 550부터 0으로 만듬
        
        img2 = left_lane | right_lane # 이미지를 하나로 합침(차선 결합)
        
        return img2