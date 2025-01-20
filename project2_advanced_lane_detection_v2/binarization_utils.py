import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# 노란색 차선 임계값 선택
# H: 노란색은 0~50 (Hue 기준, 보정된 범위) 사이에 존재
yellow_HSV_th_min = np.array([0,70,70]) # 노란색 범위의 최소 HSV 값
yellow_HSV_th_max = np.array([50,255,255]) # 노란색 범위의 최대 HSV 값


def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    min_th_ok = np.all(HSV > min_values, axis=2) # 이미지의 세번째 축(h,w,3)을 따라 연산
    max_th_ok = np.all(HSV < max_values, axis=2) 
    
    out = np.logical_and(min_th_ok, max_th_ok)

    if verbose:
        plt.imshow(out, cmap='gray')
        plt.show()
    
    return out


def thresh_frame_sobel(frame, kernel_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 에지 검출
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size) # x축 방향 경계 계산. 수평 경계에서의 기울기 배열
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size) # y축 방향 경계 계산. 수직 경계에서의 기울기 배열
    
    # 경계강도(gradient magnitude)
    # - 기울기 벡터의 크기. 기울기는 이미지에서 픽셀 값의 변화율을 측정하며, 변화율이 크다는 것은 픽셀 값이 급격히 변화하는 경계를 의미
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2) # 경계 강도 계산 = 루트(sobel_x^2 + sobel_y^2)
    # 모든 값을 경계강도의 최대값으로 나누어 0~1사이로 스케일링 후 8비트(0~255)로 변환 ==> 그레이스케일 이미지
    sobel_mag = np.uint8(sobel_mag/np.max(sobel_mag) * 255)
    
    # 경계강도 이진화
    # 경계강도, 임계값, True로 설정할 겂, 경계강도가 50이상이면 1/아니면 0 ==> 흑백이미지
    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)
    
    return sobel_mag.astype(bool)


"""
히스토그램 평활화 적용하고 임계값 설정 후 이진결과 반환
"""
def get_binary_from_equalized_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 히스토그램 평활화 - 히스토그램이 평평하게 되도록 영상을 조작해 영상의 명암 대비를 높이는 기법. 물체를 더 잘 식별해짐
    eq_global = cv2.equalizeHist(gray)
    
    # cv2.threshold는 흑백 이미지 또는 그레이스케일 이미지 생성
    # cv2.THRESH_BINARY: 픽셀 값이 thresh보다 크면 maxval 할당. 그렇지 않으면 0
    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
    
    return th

"""
차선 강조한 이미지 변환
"""
def binarize(img, verbose=False):
    h,w = img.shape[:2]
    
    binary = np.zeros(shape=(h,w), dtype=np.uint8)
    
    # 노란차선 검출
    HSV_yellow_mask = thresh_frame_in_HSV(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
    binary = np.logical_or(binary, HSV_yellow_mask)
    
    # 흰차선 검출
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask)
    
    # 차선의 경계 검출
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)
    
    # 모폴로지 연산(침식,팽창)
    # - 흑백 이미지에 적용되어 노이즈 제거, 객체 크기 조정, 구멍 메우기 등의 작업에 사용
    kernel = np.ones((5,5), np.uint8)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel) # 팽창 후 침식(closing)
    
    if verbose:        
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title('input_frame')
        ax[0, 0].set_axis_off()
        ax[0, 0].set_facecolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(HSV_yellow_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        plt.show()

    return closing


if __name__ == '__main__':
    test_images = glob.glob('project2_advanced_lane_detection_v2/test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        
        binarize(img=img, verbose=True)