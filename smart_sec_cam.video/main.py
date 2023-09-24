import os
import time
import cv2
from detection import MotionDetector
#from smart_sec_cam.redis import RedisImageReceiver


CHANNEL_LIST_INTERVAL = 10
SLEEP_TIME = 0.01


def main(video_dir: str, motion_threshold: int):
    
    # Create and start MotionDetection instance for each channel
    motion_detector = MotionDetector('left_cam', motion_threshold=motion_threshold, video_dir=video_dir)    
    motion_detector.run_in_background()
    cap = cv2.VideoCapture('input\\output1024_crop.mp4')
    while True:
        
        # read each frame from video file 'input/output1024_crop.mp4'
        # and add it to the MotionDetection instance        
        ret, frame = cap.read()
        if not ret:
            print("Can't read video file")
            exit()
        motion_detector.add_frame(frame)
        print(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # check if frame is the last frame of the video file
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            motion_detector.run()
            break   
        #motion_detector.stop()
        #print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-dir', help='Directory in which video files are stored', type=str,
                        default="output")
    args = parser.parse_args()

    motion_threshold = 50000

    main(args.video_dir, motion_threshold)
    