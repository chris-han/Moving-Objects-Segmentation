import cv2
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)



# Define the number of frames to use for the moving average filter
num_frames = 5
# Parameters
block_size = 32
threshold = 50

# Open the video file
cap = cv2.VideoCapture('input\\output1024_crop.mp4')
# print the dimensions of the video
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# Create a buffer to store the previous frames
frame_buffer = []

# create a np array smooth_frame_buffer to store the smoothed frames
smooth_frame_buffer = []

# Iterate over each frame in the video
while cap.isOpened():
    # Read the next frame from the video
    ret, frame = cap.read()

    # If the frame was successfully read, apply the filter
    if ret:
        # Convert the frame to grayscale
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the frame to HSV format
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Add the current frame to the buffer
        frame_buffer.append(hsv_frame)
        # Split the image into its component channels
        hsv_channels = cv2.split(hsv_frame)
        #cv2.imshow('Hue Channel', hsv_channels[0])
        #cv2.imshow('Saturation Channel', hsv_channels[1])
        #cv2.imshow('Value Channel', hsv_channels[2])

        hue_channel = hsv_channels[0]
        hue_buffer = [hue_channel]
        sat_channel = hsv_channels[1]
        sat_buffer = [sat_channel]
        cv2.imshow('Hue Channel', hue_channel)
        cv2.imshow('Saturation Channel', sat_channel)

        if len(frame_buffer) == num_frames:
            for i in range(num_frames - 1):
                hue_buffer.append(frame_buffer[i][:, :, 0])
                sat_buffer.append(frame_buffer[i][:, :, 1])
            avg_hue = np.mean(hue_buffer, axis=0).astype(np.uint8)
            avg_sat = np.mean(sat_buffer, axis=0).astype(np.uint8)
            hsv_channels = list(hsv_channels)
            hsv_channels[0] = avg_hue
            hsv_channels[1] = avg_sat
            hsv_channels = tuple(hsv_channels)

            # Merge the channels back into an HSV frame
            avg_frame = cv2.merge(hsv_channels)

            # Convert the output frame back to BGR format for display
            out_frame = cv2.cvtColor(avg_frame, cv2.COLOR_HSV2BGR)
            # add out_frame to smooth_frame_buffer
            smooth_frame_buffer.append(out_frame)

            # Display the smoothed output frame
            cv2.imshow('Smoothed Frame', out_frame)

            # Remove the oldest frame from the buffer
            frame_buffer.pop(0)

        # Wait for a key press and exit if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break



# Read the first frame from smooth_frame_buffer
previous_frame = smooth_frame_buffer[0]

# Convert the frame to grayscale
current_frame_g = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
# Get the size of the frame
rows, cols = current_frame_g.shape

# Convert the frame to the HSV color space
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
#previous_frame = denoise_wavelet(previous_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)

# interage over each frame in smooth_frame_buffer
for f in range(1, len(smooth_frame_buffer)):

    # Read the next frame
    current_frame_origi = smooth_frame_buffer[f]

    # Convert the frame to the HSV color space
    current_frame = current_frame_origi# cv2.cvtColor(current_frame_origi, cv2.COLOR_BGR2HSV)
    #current_frame = denoise_wavelet(current_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
    # Iterate over each block
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Get the current block in the current and previous frames based on color not grayscale
      
            current_block = current_frame[i:i+block_size, j:j+block_size, 1]
            previous_block = previous_frame[i:i+block_size, j:j+block_size, 1]
            

            # Compute the mean squared error between the blocks on color not grayscale
            mse = np.mean((previous_block - current_block) ** 2)  
            
            print(mse)
            # If the mse is above a threshold, mark it as a changed block
            if mse > threshold:
                # draw a red rectangles around blocks that have changed in HSV color space
                cv2.rectangle(current_frame_origi, (j, i), (j+block_size, i+block_size), (0, 255, 0), 1)

                #cv2.rectangle(current_frame, (j, i), (j+block_size, i+block_size), (0, 255, 0), 1)

    # Display the result
    #current_frame = cv2.cvtColor(current_frame, cv2.COLOR_HSV2BGR)
    cv2.imshow('Block Matching', current_frame_origi)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # The current frame becomes the previous frame for the next iteration
    previous_frame = current_frame.copy()


# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
