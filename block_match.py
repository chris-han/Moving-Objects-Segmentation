import cv2
import numpy as np
from skimage.restoration import (denoise_wavelet, estimate_sigma)




# Parameters
block_size = 32
threshold = 100

# Open the video file
cap = cv2.VideoCapture('input\\output1024_crop.mp4')
# print the dimensions of the video
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))



# Read the first frame
ret, previous_frame = cap.read()
if not ret:
    print("Can't read video file")
    exit()

# Convert the frame to grayscale
current_frame_g = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
# Get the size of the frame
rows, cols = current_frame_g.shape

# Convert the frame to the HSV color space
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2HSV)
#previous_frame = denoise_wavelet(previous_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)


while True:
    # Read the next frame
    ret, current_frame_origi = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    current_frame = cv2.cvtColor(current_frame_origi, cv2.COLOR_BGR2HSV)
    #current_frame = denoise_wavelet(current_frame, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
   

    

    # Iterate over each block
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Get the current block in the current and previous frames based on color not grayscale
            
            current_block = current_frame[i:i+block_size, j:j+block_size, 1]
            previous_block = previous_frame[i:i+block_size, j:j+block_size, 1]
            

            # Compute the mean squared error between the blocks
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
