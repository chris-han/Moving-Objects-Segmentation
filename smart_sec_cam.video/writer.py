import datetime
import os
import time
from typing import Tuple

import cv2


class VideoWriter:
    FILENAME_DELIM = "__"

    def __init__(self, channel: str, path="output",
                 resolution: Tuple[int, int] = (640, 480)):
        self.channel = channel
        self.video_dir = path
        self.full_filepath = None
        self._generate_file_name()
        self._make_target_dir(path)
        self.resolution = resolution
        self.frame_buffer = []
        self.first_frame_time = time.monotonic()

    @staticmethod
    def _make_target_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def add_frame(self, frame):
        resized_frame = cv2.resize(frame, self.resolution)
        if not self.frame_buffer:
            self.first_frame_time = time.monotonic()
        self.frame_buffer.append(resized_frame)

    def write(self):
        self._generate_file_name()
        print("Writing video to: " + self.full_filepath + " ...")
        fps = self._calculate_fps()
        
        '''
        # Write to .webm
        webm_file = self.full_filepath + ".webm"        
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        writer = cv2.VideoWriter(webm_file, fourcc, fps, self.resolution)
        for frame in self.frame_buffer:
            writer.write(frame)
        writer.release()
        del writer
        '''
        # Write to .mp4
        mp4_file = self.full_filepath + ".mp4"
        # if mp4_file exists, ignor it
        if os.path.exists(mp4_file):
            print("File already exists")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(mp4_file, fourcc, fps, self.resolution)
        for frame in self.frame_buffer:
            writer.write(frame)
        writer.release()
        del writer
        self._clear_frame_buffer()

    def reset(self):
        self._clear_frame_buffer()
        self.first_frame_time = time.monotonic()

    def _clear_frame_buffer(self):
        self.frame_buffer = []

    def _generate_file_name(self):
        date = datetime.datetime.now()
        filename = self.channel + self.FILENAME_DELIM + date.strftime("%Y-%m-%d_%H_%M_%S")
        self.full_filepath = os.path.join(self.video_dir, filename)

    def _calculate_fps(self) -> int:
        elapsed_time = time.monotonic() - self.first_frame_time
        return int(len(self.frame_buffer) / elapsed_time)
