import numpy as np
import matplotlib.image as mpimg
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

from project2_advanced_lane_detection.CameraCalibration import CameraCalibration
from project2_advanced_lane_detection.Thresholding import *
from project2_advanced_lane_detection.PerspectiveTransformation import *
from project2_advanced_lane_detection.LaneLines import *

# from CameraCalibration import CameraCalibration
# from Thresholding import *
# from PerspectiveTransformation import *
# from LaneLines import *

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
        img = self.thresholding.forward(img)
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)
        
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        out_img = self.lanelines.plot(out_img)
        
        return out_img
    
    def process_image(self, input_path, output_path):
        img = mpimg.imread(input_path)
        out_img = self.forward(img)
        mpimg.imsave(output_path, out_img)
    
    def process_video(self, input_path, output_path):
        # clip = VideoFileClip(input_path)
        # out_clip = clip.fl(self.forward)
        # out_clip.write_videofile(output_path, audio=False)
        
        # with VideoFileClip(input_path) as clip:
        #     self.forward()
        cap = cv2.VideoCapture(input_path)
        
        current_frame = 0
        total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        while cap.isOpened():
            success, img = cap.read()
            
            if success:
                
                # current_frame += 1
                # print(current_frame)
                
                img = self.forward(img)
                
                cv2.imshow("Image", img)
            
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                
                # cv2.waitKey(25)
                # if current_frame == total_frame:
                #     break
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    findLaneLines = FindLaneLines()
    
    input = 'project2_advanced_lane_detection/test_videos/project_video.mp4'
    output = 'project2_advanced_lane_detection/output_videos/result.mp4'
    
    findLaneLines.process_video(input, output)
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