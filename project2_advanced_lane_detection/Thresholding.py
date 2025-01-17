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


def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


class Thresholding:
    def __init__(self):
        pass
    
    """
    TODO: 둘 다 흰색 차선일 경우?
    """
    def forward(self, img):
        img = adjust_gamma(img, gamma=0.5)
        
        # hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HLS로 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # HSV로 변환
        # h_channel = hls[:,:,0] # Hue 채널
        # l_channel = hls[:,:,1] # Lightness 채널
        # s_channel = hls[:,:,2] # Saturation 채널
        # v_channel = hsv[:,:,2] # Value 채널
        
        # 오른쪽 차선 감지
        # right_lane = threshold_rel(l_channel, 0.8, 1.0) # 0.8~1.0 사이의 이진화된 이미지
        # right_lane[:,:750] = 0 # 열 ~749까지 0으로 만듬
        
        # 왼쪽 차선 감지
        # left_lane = threshold_abs(h_channel, 20, 30) # 20~30 사이의 이진화된 이미지
        # left_lane &= threshold_rel(v_channel, 0.7, 1.0) # 0.7~1.0 사이의 이진화된 이미지
        # left_lane[:,550:] = 0 # 열 550부터 0으로 만듬
        
        # img2 = left_lane | right_lane # 이미지를 하나로 합침(차선 결합)
        
        # 노란색 범위 정의 (범위는 조정 필요)
        lower_yellow = np.array([18, 80, 80])  # 예: 노란색 범위의 하한값
        upper_yellow = np.array([30, 255, 255])  # 예: 노란색 범위의 상한값

        # 노란색 차선 필터링
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 흰색 차선 필터링 (기존 범위 조정 가능)
        """
        Hue (H): 색상 (흰색은 모든 색조에서 나타날 수 있음, 범위는 0~180).
        Saturation (S): 색의 채도 (흰색은 채도가 낮음, 일반적으로 0~50).
        Value (V): 밝기 (흰색은 밝기가 높음, 일반적으로 200 이상).
        """
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        img2 = yellow_mask | white_mask
        # img2 = white_mask
        
        # cv2.imshow("ori", img)
        # cv2.imshow("forward", img2)
        
        return img2 # 흑백사진으로 리턴

        # 흰색과 노란색 마스크 결합
        # combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        # filtered_image = cv2.bitwise_and(img, img, mask=combined_mask)
        
        # cv2.imshow("img", img)
        # cv2.imshow("hls", hls)
        # cv2.imshow("hsv", hsv)
        # cv2.imshow("white_mask", white_mask)
        # cv2.imshow("yellow_mask", yellow_mask)
        # cv2.imshow("filtered_image", filtered_image)
        
        # return filtered_image