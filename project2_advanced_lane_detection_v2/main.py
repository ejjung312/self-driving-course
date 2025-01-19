import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from project2_advanced_lane_detection_v2.globals import xm_per_pix, time_window
from project2_advanced_lane_detection_v2.calibration_utils import calibrate_camera, undistort
from project2_advanced_lane_detection_v2.binarization_utils import binarize
from project2_advanced_lane_detection_v2.perspective_utils import birdeye
from project2_advanced_lane_detection_v2.line_utils import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits


processed_frames = 0 # 프레임 개수
line_lt = Line(buffer_len=time_window) # 왼쪽 차선
line_rt = Line(buffer_len=time_window) # 오른쪽 차선

"""
Prepare the final pretty pretty output blend, given all intermediate pipeline images
"""
def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h,w = blend_on_road.shape[:2]
    
    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio*h), int(thumb_ratio*w)
    
    off_x, off_y = 20, 15
    
    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0,0), pt2=(w, thumb_h+2*off_y), color=(0,0,0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)
    
    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary
    
    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye
    
    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.
    """
    if line_lt.detected and line_rt.detected:
        # 탐지된 y 좌표들 중 최댓값, 즉 이미지에서 차선의 가장 아래쪽(하단)을 나타냄 => 버드아이뷰에서는 보통 y값이 클수록 아래쪽을 의미
        # 가장 큰 값의 95% 이상인 y 좌표들을 선택해서 x값을 구한 후 평균을 구함
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > (0.95 * line_lt.all_y.max())])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > (0.95 * line_rt.all_y.max())])
        # 차선의 너비
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        # (왼쪽 차선의 위치 + 왼쪽 차선과 오른쪽 차선의 중앙 지점) - 차량의 중앙 x 좌표 ==> 차량의 중심이 도로 중앙에서 얼마나 벗어나 있는지(픽셀)
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)

        # 차량의 중심이 도로 중앙에서 얼마나 벗어나 있는지 미터로 환산
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1
    
    return offset_meter


def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames
    
    # 카메라 왜곡 펴기
    img_undistorted = undistort(frame, mtx, dist, verbose=False)
    
    # 프레임 이진화 후 차선 라인 강조
    img_binary = binarize(img_undistorted, verbose=False)

    # 원근 변환하여 상공에서 보는 관점을 얻음
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)
    
    # TODO - 둘의 차이가 뭔가
    # 2차원 다항식 곡선 그리기
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])
    
    # 이미지에 차선과 차선영역 그리기
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    
    # 썸네일(이진화, 상공뷰, 상공뷰 차선라인)
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)
    
    processed_frames += 1
    
    return blend_output
    # return blend_on_road


if __name__ == '__main__':
    # 카메라 왜곡 펴기
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')
    
    # mode = 'images'
    mode = 'video'
    
    if mode == 'video':
        input_path = 'project2_advanced_lane_detection_v2/test_videos/project_video.mp4'
        # input_path = 'project2_advanced_lane_detection_v2/test_videos/challenge_video.mp4'
        # input_path = 'project2_advanced_lane_detection_v2/test_videos/harder_challenge_video.mp4'
        
        cap = cv2.VideoCapture(input_path)
        success, img = cap.read()
        img = process_pipeline(img, keep_state=False)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
        # cap = cv2.VideoCapture(input_path)
        # # 초당 프레임 수 및 프레임 크기 가져오기
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_size = (frame_width, frame_height)
        
        # # 비디오 작성 객체 생성 (코덱: MJPG)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('project2_advanced_lane_detection_v2/out_video3.mp4', fourcc, fps, frame_size)
        
        # while cap.isOpened():
        #     success, img = cap.read()
        #
        #     # 영상 끝나면 자동 종료
        #     if not success:
        #         break
        #
        #     # img = process_pipeline(img, keep_state=False)
        #     img = process_pipeline(img, keep_state=True)
        #
        #     cv2.imshow("Image", img)
        #
        #     # 프레임 저장
        #     # out.write(img)
        #
        #     # if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        #     if cv2.waitKey(25) & 0xFF == ord('q'):
        #         break
        #
        # cap.release()
        # # out.release()
        # cv2.destroyAllWindows()
        
    else:
        test_img_dir = 'project2_advanced_lane_detection_v2/test_images'
        for test_img in os.listdir(test_img_dir):
            frame = cv2.imread(os.path.join(test_img_dir, test_img))
            blend = process_pipeline(frame, keep_state=False)
            
            plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
            plt.show()