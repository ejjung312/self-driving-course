import pprint
import cv2
import numpy as np

birdeye_binary = cv2.imread('project2_advanced_lane_detection_v2/test2.jpg')

birdeye_binary = cv2.cvtColor(birdeye_binary, cv2.COLOR_BGR2GRAY)

# height, width = birdeye_binary.shape
#
# histogram = np.sum(birdeye_binary[height // 2:-30, :], axis=0)
#
# print(histogram)

print(birdeye_binary.nonzero())