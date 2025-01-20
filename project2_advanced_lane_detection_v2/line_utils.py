import cv2
import numpy as np
import glob
import collections
import matplotlib.pyplot as plt

from project2_advanced_lane_detection_v2.globals import ym_per_pix, xm_per_pix

class Line:
    def __init__(self, buffer_len=10):
        self.detected = False
        
        # 반복문 마지막의 다항식 계수
        self.last_fit_pixel = None
        self.last_fit_meter = None
        
        # 반복문 거친 다항식 계수 리스트
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2*buffer_len)
        
        self.radius_of_curvature = None
        
        # 라인으로 감지된 모든 x,y좌표
        self.all_x = None
        self.all_y = None
    
    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.
        """
        
        self.detected = detected
        
        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter
        
        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    def draw(self, mask, color=(255,0,0), line_width=50, average=False):
        """
        차선 그리기
        """
        h,w,c = mask.shape

        # 0~(h-1)까지 h 만큼 구간 생성
        plot_y = np.linspace(0, h-1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        # 2차원 함수에 대한 결과값 반환 (Ax^2 + Bx + C)
        # 차선 가운데 값을 구해 왼쪽/오른쪽 차선을 구함
        line_center = coeffs[0]*plot_y**2 + coeffs[1]*plot_y + coeffs[2]
        line_left_side = line_center - line_width//2
        line_right_side = line_center + line_width//2

        # 차선의 (x,y)값 반환
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        # 위아래로 뒤집어 순서를 반전(flipud) => 오른쪽 차선의 점들을 거꾸로 배열해 차선 영역을 닫을 수 있도록 만듬
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        # 수직으로 행렬을 합침(vstack)
        pts = np.vstack([pts_left, pts_right])

        return cv2.fillPoly(mask, [np.int32(pts)], color)

    @property
    # average of polynomial coefficients of the last N iterations
    def average_fit(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # radius of curvature of the line (averaged)
    def curvature(self):
        y_eval = 0
        coeffs = self.average_fit
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line (averaged)
    def curvature_meter(self):
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


"""
이진화 이미지에서 감지한 차선에 대한 다각형 계수 가져오기
"""
def get_fits_by_sliding_windows(birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):
    height, width = birdeye_binary.shape

    # np.dstack: 2D 배열을 3D 배열로 확장해 RGB 이미지로 만들 때 사용
    # 255를 곱해 [0,255] 범위로 확장
    out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

    # 픽셀이 0이 아닌 행렬의 위치 값을 반환 (y값, x값 따로)
    # (array([0, 0, 1, 1, 2]), array([0, 2, 0, 1, 0]))
    # => ((0,0), (0, 2), (1, 0), (1, 1), (2, 0) 에 위치한 요소들이 0 이 아닌 값
    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # 이미지의 높이/2-30의 크기에서 y축으로 값 더하기(axis=0)
    # 픽셀의 합계를 계산하여 차선의 위치 판단. 픽셀 합계가 큰 곳은 차선이 있을 확률이 높음
    histogram = np.sum(birdeye_binary[height // 2:-30, :], axis=0)

    # 히스토그램의 가운데를 기준으로 왼쪽/오른쪽 차선을 찾음. 픽셀의 값이 큰 곳을 차선 위치로 판단
    midpoint = len(histogram) // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int64(height / n_windows)
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100  # width of the windows +/- margin
    minpix = 50   # minimum number of pixels found to recenter window

    # 차선 픽셀 인덱스 저장
    left_lane_inds = []
    right_lane_inds = []

    for window in range(n_windows):
        # 슬라이딩 윈도우
        # height- 형태이기 때문에 맨 아래부터 시작하여 점점 위로 올라감
        win_y_low = height - (window + 1) * window_height # 윈도우의 아래쪽 y 좌표
        win_y_high = height - window * window_height # 윈도우의 위쪽 y 좌표
        win_xleft_low = leftx_current - margin # 왼쪽 차선의 창 왼쪽 끝 x 좌표
        win_xleft_high = leftx_current + margin # 왼쪽 차선의 창 오른쪽 끝 x 좌표
        win_xright_low = rightx_current - margin # 오른쪽 차선의 창 왼쪽 끝 x 좌표
        win_xright_high = rightx_current + margin # 오른쪽 차선의 창 오른쪽 끝 x 좌표

        # Draw the windows on the visualization image
        # 없어도 되잖아
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        # 윈도우 안에서 0이 아닌 픽셀들의 y축 인덱스 값 보관
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                        & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                        & (nonzero_x < win_xright_high)).nonzero()[0]

        # 차선 픽셀 인덱스 저장
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 최소 minpix 이상 차선의 픽셀을 찾았다면, 차선 중심 위치를 평균값으로 계산해 다음 슬라이딩 윈도우의 x좌표로 사용
        if len(good_left_inds) > minpix:
            leftx_current = np.int64(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int64(np.mean(nonzero_x[good_right_inds]))

    # y축 기준으로 리스트 합침
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 왼쪽/오른쪽 차선의 x,y 분리
    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        # 감지된 차선이 없다면, 마지막으로 감지된 차선의 데이터 반환
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        # 감지된 차선이 있다면, x,y 값으로 2차원 식의 계수 반환
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

    if not list(line_rt.all_x) or not list(line_rt.all_y):
        # 감지된 차선이 없다면, 마지막으로 감지된 차선의 데이터 반환
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        # 감지된 차선이 있다면, x,y 값으로 2차원 식의 계수 반환
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

    # 감지된 차선의 픽셀 업데이트
    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    # 그림 그리기용으로 없어도 됨
    # 0~(height-1)까지 height개 만큼 구간 생성
    ploty = np.linspace(0, height - 1, height)
    # 2차원 함수에 대한 결과값 반환 (Ax^2 + Bx + C)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # 0이 아닌 픽셀값을 칠함. 왼쪽차선은 빨간색, 오른쪽차선은 파란색
    out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
    out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

    if verbose:
        f, ax = plt.subplots(1, 2)
        f.set_facecolor('white')
        ax[0].imshow(birdeye_binary, cmap='gray')
        ax[1].imshow(out_img)
        ax[1].plot(left_fitx, ploty, color='yellow')
        ax[1].plot(right_fitx, ploty, color='yellow')
        ax[1].set_xlim(0, 1280)
        ax[1].set_ylim(720, 0)

        plt.show()

    return line_lt, line_rt, out_img


"""
이진화 이미지에서 감지한 차선에 대한 다각형 계수 가져오기
"""
def get_fits_by_previous_fits(birdeye_binary, line_lt, line_rt, verbose=False):
    height, width = birdeye_binary.shape
    
    left_fit_pixel = line_lt.last_fit_pixel
    right_fit_pixel = line_rt.last_fit_pixel
    
    nonzero = birdeye_binary.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzero_x > (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] - margin)) & 
                        (nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
                        (nonzero_x < (left_fit_pixel[0] * (nonzero_y ** 2) + left_fit_pixel[1] * nonzero_y + left_fit_pixel[2] + margin)))
    right_lane_inds = ((nonzero_x > (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] - margin)) & 
                        (nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))
                        (nonzero_x < (right_fit_pixel[0] * (nonzero_y ** 2) + right_fit_pixel[1] * nonzero_y + right_fit_pixel[2] + margin)))
    
    # 차선이 있는 위치의 픽셀 추출
    line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
    line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]
    
    detected = True
    if not list(line_lt.all_x) or not list(line_lt.all_y):
        left_fit_pixel = line_lt.last_fit_pixel
        left_fit_meter = line_lt.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
        left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)
        
    if not list(line_rt.all_x) or not list(line_rt.all_y):
        right_fit_pixel = line_rt.last_fit_pixel
        right_fit_meter = line_rt.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
        right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)
        
    line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)
    
    # x,y 생성
    ploty = np.linspace(0, height-1, height)
    left_fitx = left_fit_pixel[0]*ploty**2 + left_fit_pixel[1]*ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0]*ploty** 2 + right_fit_pixel[1]*ploty + right_fit_pixel[2]
    
    # 그림 그릴 이미지 생성
    img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
    window_img = np.zeros_like(img_fit)
    
    # 차선 색칠
    img_fit[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255,0,0]
    img_fit[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0,0,255]
    
    # 다각형 생성
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
    # 빈 이미지에 차선 그리기
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)
    
    if verbose:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return line_lt, line_rt, img_fit


def draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state):
    """
    이미지에 차선과 차선영역 그리기
    """
    height, width, _ = img_undistorted.shape

    left_fit = line_lt.average_fit if keep_state else line_lt.last_fit_pixel
    right_fit = line_rt.average_fit if keep_state else line_rt.last_fit_pixel

    # 0~(height-1)까지 height개 만큼 구간 생성
    ploty = np.linspace(0, height - 1, height)
    # 2차원 함수에 대한 결과값 반환 (Ax^2 + Bx + C)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 차선영역 색칠
    # 왼쪽과 오른쪽 차선 픽셀을 연결하여 도로 영역을 나타내는 다각형 좌표를 생성
    road_warp = np.zeros_like(img_undistorted, dtype=np.uint8) # img_undistorted와 같은 크기의 검정이미지 생성
    # 수직으로 행렬을 합친 후(vstack), 전치(transpose)하여 (x,y) 형식으로 만듬
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # 위아래로 뒤집어 순서를 반전(flipud) => 오른쪽 차선의 점들을 거꾸로 배열해 차선 영역을 닫을 수 있도록 만듬
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # 가로로 행 결합 => 왼쪽 차선과 오른쪽 차선을 연결하여 하나의 다각형 생성
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 255))
    # 변환행렬으로 변환시킨 이미지 원래대로 다시 복원시킴
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (width, height))
    
    # 원본 이미지와 차선영역 합성
    blend_onto_road = cv2.addWeighted(img_undistorted, 1., road_dewarped, 0.3, 0)
    
    # now separately draw solid lines to highlight them
    # 차선 색칠 후 변환행렬으로 이미지 복원
    line_warp = np.zeros_like(img_undistorted)
    line_warp = line_lt.draw(line_warp, color=(255, 0, 0), average=keep_state)
    line_warp = line_rt.draw(line_warp, color=(0, 0, 255), average=keep_state)
    line_dewarped = cv2.warpPerspective(line_warp, Minv, (width, height))

    lines_mask = blend_onto_road.copy()
    idx = np.any([line_dewarped != 0][0], axis=2) # 차선이 있는 픽셀의 위치를 나타내는 불리언 배열
    lines_mask[idx] = line_dewarped[idx] # 차선 부분만 결합

    blend_onto_road = cv2.addWeighted(src1=lines_mask, alpha=0.8, src2=blend_onto_road, beta=0.5, gamma=0.)

    return blend_onto_road