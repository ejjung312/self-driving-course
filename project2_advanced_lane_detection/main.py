import numpy as np
import matplotlib.image as mpimg
import cv2
# from moviepy.video.io.VideoFileClip import VideoFileClip

from project2_advanced_lane_detection.CameraCalibration import CameraCalibration
from project2_advanced_lane_detection.Thresholding import *
from project2_advanced_lane_detection.PerspectiveTransformation import *
from project2_advanced_lane_detection.LaneLines import *

class FindLaneLines:
    def __init__(self):
        self.calibration = CameraCalibration('project2_advanced_lane_detection/camera_cal', 9, 6)
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()


    def forward(self, img):
        out_img = np.copy(img)        
        img = self.calibration.undistort(img) # 카메라 왜곡 펴기
        img = self.transform.forward(img) # 투시 변환을 적용하여 이미지를 변환
        img = self.thresholding.forward(img) # 차선감지
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0) # 이미지 합성
        out_img = self.lanelines.plot(out_img)
        
        return out_img
        # return img


    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)


    def process_video(self, input_path):
        cap = cv2.VideoCapture(input_path)
        
        # success, img = cap.read()
        # img = self.forward(img)
        # cv2.imshow("Image", img)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cap.release()
        #     cv2.destroyAllWindows()
        
        while cap.isOpened():
            success, img = cap.read()
            
            if success:
                img = self.forward(img)
                
                cv2.imshow("Image", img)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


def main():
    input = 'project2_advanced_lane_detection/test_videos/project_video.mp4'
    # input = 'project2_advanced_lane_detection/test_videos/project_video2.mp4'
    # input = 'project2_advanced_lane_detection/test_videos/challenge_video.mp4'

    findLaneLines = FindLaneLines()
    findLaneLines.process_video(input)
    # args = docopt(__doc__)
    # input = args['INPUT_PATH']
    # output = args['OUTPUT_PATH']

    # findLaneLines = FindLaneLines()
    # if args['--video']:
    #     findLaneLines.process_video(input, output)
    # else:
    #     findLaneLines.process_image(input, output)


if __name__ == "__main__":
    main()