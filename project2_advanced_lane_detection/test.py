from moviepy import VideoFileClip

def process_frame(frame):
    # 여기서 frame은 변환할 단일 프레임입니다.
    # 예를 들어, frame을 그레이스케일로 변환할 수 있습니다.
    return frame  # 변환된 프레임 반환

# VideoFileClip 객체 생성
clip = VideoFileClip("project2_advanced_lane_detection/test_videos/project_video.mp4")

# fl 메서드로 각 프레임 처리
processed_clip = clip.fl(lambda gf, t: process_frame(gf(t)))

# 결과 비디오 저장
processed_clip.write_videofile("output_video.mp4")