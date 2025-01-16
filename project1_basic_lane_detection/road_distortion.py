import cv2
import numpy as np

src = np.float32([(550,460),
                (150,720), 
                (1200,720), 
                (770,460)])

dst = np.float32([(100,0),
                (100,720),
                (1100,720),
                (1100,0)])

M = cv2.getPerspectiveTransform(src, dst) # 투시 변환 행렬
M_inv = cv2.getPerspectiveTransform(dst, src)

size = (1280,720)

img = cv2.imread("project1_lane_detection/road.jpg")
output = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR) # 투시 변환을 적용하여 이미지를 변환
output2 = cv2.warpPerspective(output, M_inv, size, flags=cv2.INTER_LINEAR)

cv2.imshow("output", output)
cv2.imshow("output2", output2)
cv2.waitKey(0)