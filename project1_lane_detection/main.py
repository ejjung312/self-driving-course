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
            for x1,y1,x2,y2 in line:
                # x1,y1,x2,y2 = line.reshape(4) # 길이가 4인 1차원 배열로 변환
                # cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
                cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 10)
            
    return img


def make_points(img, line_slope_intercept):
    # https://miro.medium.com/v2/resize:fit:1400/1*zT7ed-RhcuWbcFej2WRgOw.png
    slope, intercept = line_slope_intercept
    height = img.shape[0]
    y1 = int(height)
    y2 = int(y1*3.0/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    return [[x1,y1,x2,y2]]


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            # x좌표, y좌표끼리 전달해야 함
            fit = np.polyfit((x1,x2), (y1,y2), 1) # 두 점을 통과하는 1차원 직선의 기울기와 절편을 구함
            slope = fit[0] # 기울기
            intercept = fit[1] # 절편
            if slope < 0:
                # left_fit.append((slope, intercept))
                right_fit.append((slope, intercept))
            else:
                # right_fit.append((slope, intercept))
                left_fit.append((slope, intercept))
    
    # 왼/오로 분리된 선들의 평균 기울기,절편 계산
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    
    # 평균 기울기,절편의 점 계산
    left_line = make_points(img, left_fit_average)
    right_line = make_points(img, right_fit_average)
    average_lines = [left_line, right_line]
    
    return average_lines


cap = cv2.VideoCapture('project1_lane_detection/test1.mp4')
ret, frame = cap.read()

canny_output = canny(frame)
masked_output = region_of_interest(canny_output) # 노이즈가 남아있음
lines = houghLines(masked_output)
average_lines = average_slope_intercept(frame, lines)
# line_output = display_lines(masked_output, lines)
line_output = display_lines(frame, average_lines)

cv2.imshow("Image", line_output)
cv2.waitKey(0)