import cv2
import numpy as np

def nothing(x):
    pass

# 트랙바 생성
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Lower", "Trackbars", 0, 180, nothing)
cv2.createTrackbar("H Upper", "Trackbars", 180, 180, nothing)
cv2.createTrackbar("S Lower", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("S Upper", "Trackbars", 30, 255, nothing)
cv2.createTrackbar("V Lower", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("V Upper", "Trackbars", 255, 255, nothing)

while True:
    image = cv2.imread("project2_advanced_lane_detection/test3.jpg")
    
    # HSV 범위 트랙바 값 가져오기
    h_lower = cv2.getTrackbarPos("H Lower", "Trackbars")
    h_upper = cv2.getTrackbarPos("H Upper", "Trackbars")
    s_lower = cv2.getTrackbarPos("S Lower", "Trackbars")
    s_upper = cv2.getTrackbarPos("S Upper", "Trackbars")
    v_lower = cv2.getTrackbarPos("V Lower", "Trackbars")
    v_upper = cv2.getTrackbarPos("V Upper", "Trackbars")

    lower_white = np.array([h_lower, s_lower, v_lower])
    upper_white = np.array([h_upper, s_upper, v_upper])

    # HSV 변환 및 필터링
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 결과 시각화
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()