import cv2
import numpy as np

# 캐니 엣지 검출
def canny(img):
    # 1. 이미지를 gray scale로 변경
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 이미지 blur 처리
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    
    # 캐니 엣지 검출
    canny = cv2.Canny(blur, 50, 150)
    
    return canny

# ROI - 관심영역 검출
def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    mask = np.zeros_like(img)
    triangle = np.array([[(200, height), (800, 350), (1200,height),]], np.int32) # 삼각형 도로의 지점
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

# 허프변환P(확률적) - 확률적으로 임의의 점만을 허프 변환시켜 선 검출 시간을 줄여줌
def houghLines(img):
    houghLines = cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    return houghLines


def display_lines(img, lines):
    line_image = np.zeros_like(img)
    
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4) # 길이가 4인 1차원 배열로 변환
            # cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
            cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 10)
    
    return img
            

cap = cv2.VideoCapture('project1_lane_detection/test1.mp4')
ret, frame = cap.read()

canny_output = canny(frame)
masked_output = region_of_interest(canny_output) # 노이즈가 남아있음
lines = houghLines(masked_output)
# line_output = display_lines(masked_output, lines)
line_output = display_lines(frame, lines)

cv2.imshow("Image", line_output)
cv2.waitKey(0)