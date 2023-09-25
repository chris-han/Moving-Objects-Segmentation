import cv2,math,time
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from sklearn.cluster import DBSCAN

# motion_blocks = np.array([[110, 120], [129, 130], [140, 150], [1260, 270], [1280, 270]])
# clustering = DBSCAN(eps=30, min_samples=2).fit(motion_blocks)
# labels = clustering.labels_
# print("Labels:", labels)
# # show the data under labels
# # foreach unique label
# for label in np.unique(labels):
#     print(motion_blocks[labels == label])



# Define the number of frames to use for the moving average filter
num_frames = 5
# Parameters
block_size = 32
threshold = 50

# Open the video file
cap = cv2.VideoCapture('input\\output1024_crop.mp4')
# print the dimensions of the video
#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# Create a buffer to store the previous frames
frame_buffer = []

# create a np array smooth_frame_buffer to store the smoothed frames
smooth_frame_buffer = []

# create a np array to store a 2-dimonsional tuple which contains the coordinate of (x,y)
motion_blocks = np.empty((0, 2), int)
# bottom_rights = np.empty((0, 2), int)

# append tuple (110,120) to the np array
# upper_lefts = np.append(upper_lefts, [(110, 120)], axis=0)
# upper_lefts = np.append(upper_lefts, [(129, 130)], axis=0)
# print("Max of x:", np.max(upper_lefts[:, 0]))
# print("Max of y:", np.max(upper_lefts[:, 1]))

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
        #hsv_channels = cv2.split(hsv_frame)
        #cv2.imshow('Hue Channel', hsv_channels[0])
        #cv2.imshow('Saturation Channel', hsv_channels[1])
        #cv2.imshow('Value Channel', hsv_channels[2])

        
        # If the buffer is full, apply the moving average filter
        if len(frame_buffer) == num_frames:
            # Compute the average of the frames in the buffer
            avg_frame = np.mean(frame_buffer, axis=0).astype(np.uint8)

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
current_frame_g = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
# Get the size of the frame
rows, cols = current_frame_g.shape[:2]

# Convert the frame to the HSV color space
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
#previous_frame = denoise_wavelet(previous_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
overall_upper_left = (0,0)
overall_bottom_right = (cols,rows)
# interage over each frame in smooth_frame_buffer
for f in range(1, len(smooth_frame_buffer)):
    # Read the next frame
    current_frame_origi = smooth_frame_buffer[f]

    # Convert the frame to the HSV color space
    current_frame = current_frame_origi# cv2.cvtColor(current_frame_origi, cv2.COLOR_BGR2HSV)
    #current_frame = denoise_wavelet(current_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
    # Iterate over each block
    # Create an array to store the block coordinates
    #block_coords = np.zeros((rows, cols, 4), dtype=np.int32)

    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):            

            current_block = current_frame[i:i+block_size, j:j+block_size, :]
            previous_block = previous_frame[i:i+block_size, j:j+block_size, :]
            
            # Compute the mean squared error between the blocks on color not grayscale
            mse = np.mean((previous_block - current_block) ** 2)  
            #print(mse)
            # Compute the mean squared error between the block and the original image
            mse_hue = np.mean((previous_block[:, :, 0] - current_block[:, :, 0]) ** 2)
            mse_sat = np.mean((previous_block[:, :, 1] - current_block[:, :, 1]) ** 2)
            mse_val = np.mean((previous_block[:, :, 2] - current_block[:, :, 2]) ** 2)
            # Print the mean squared error for each channel            

            

            # If the mse is above a threshold, mark it as a changed block
            if mse > threshold:
                mse_ratio = mse_val/mse_hue
                mse_ratio2 = mse_val/mse_sat
                if mse_ratio < 0.7:
                    # draw a red rectangles around blocks that have changed in HSV color space
                    j2=j+block_size
                    i2=i+block_size
                    cv2.rectangle(current_frame_origi, (j, i), (j2,i2), (0, 255, 0), 5)
                    # draw mse_hue, mse_sat, mse_val on the frame
                    #cv2.putText(current_frame_origi, 'MSE (H) = {:.2f}'.format(mse_hue), (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #cv2.putText(current_frame_origi, 'MSE (S) = {:.2f}'.format(mse_sat), (j, i+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #cv2.putText(current_frame_origi, 'MSE (V) = {:.2f}'.format(mse_val), (j, i+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(current_frame_origi, '{:.2f}'.format(mse_ratio), (j, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # print('Ratio = {:.2f},Ratio2 = {:.2f}'.format(mse_ratio,mse_ratio2))
                    
                    motion_blocks = np.append(motion_blocks, [(j, i)], axis=0)
                    clustering = DBSCAN(eps=100, min_samples=3).fit(motion_blocks)
                    labels = clustering.labels_
                    # only get the labels that are not outliers
                    # labels = labels[labels >= 0]
                    for label in np.unique(labels):
                        # ignore the outliers which are labeled as -1
                        if label == -1:
                            continue
                        hot_area =motion_blocks[labels == label]
                        print("label:", label)
                        # print(hot_area)
                        # draw a red rectangle around the overall changed area
                        center_x = int(np.mean(hot_area[:, 0]))
                        center_y = int(np.mean(hot_area[:, 1]))

                        # keep the center of the overall changed area within the frame
                        if center_x < 320: 
                            upper_left_x = 0 
                        elif center_x>cols-320:
                            center_x = cols-320 
                        else: 
                            upper_left_x = center_x-320

                        if center_y < 320:
                            upper_left_y = 0
                        elif center_y>rows-320:
                            center_y = rows-320
                        else:
                            upper_left_y = center_y-320

                        upper_left_x = center_x-320 if center_x>=320 else 0
                        upper_left_y = center_y-320 if center_y>=320 else 0
                        bottom_right_x = 640+upper_left_x
                        bottom_right_y = 640+upper_left_y

                        # upper_left_x = np.min(hot_area[:, 0])
                        # upper_left_y = np.min(hot_area[:, 1])
                        # bottom_right_x = np.max(hot_area[:, 0])+block_size
                        # bottom_right_y = np.max(hot_area[:, 1])+block_size
                        
                        overall_upper_left = (upper_left_x, upper_left_y)
                        overall_bottom_right = (bottom_right_x, bottom_right_y)
                        # draw a red rectangle around the overall changed area
                        # cv2.rectangle(current_frame_origi, overall_upper_left, overall_bottom_right, (0, 0, 255), 5)  
    cv2.rectangle(current_frame_origi, overall_upper_left, overall_bottom_right, (0, 0, 255), 5)                     



                    
                    

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
