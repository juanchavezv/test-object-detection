""" stds-sample-code-for-object-detection.py
    
    Author: Andres Hernandez Gutierrez
    Organisation: Universidad de Monterrey
    Contact: andres.hernandezg@udem.edu

    USAGE: 
    $ python stds-sample-code-for-object-detection.py --video_file football-field-cropped-video.mp4 --frame_resize_percentage 30
"""

# Import standard libraries 
import cv2 
import argparse
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Union, List

window_params = {'capture_window_name':'Input video',
                 'detection_window_name':'Detected object'}

def parse_cli_data()->argparse:
    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', 
                        type=int, 
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    return args


def initialise_camera(args:argparse)->cv2.VideoCapture:
    
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap

def rescale_frame(frame:NDArray, percentage:np.intc=20)->NDArray:
    
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def segment_object(cap:cv2.VideoCapture, args:argparse)->None:

    # Main loop
    while cap.isOpened(): 

        # Read current frame
        ret, frame = cap.read()

        # Check if the image was correctly captured
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        # Resize current frame
        frame = rescale_frame(frame, args.frame_resize_percentage)
        
        # Apply the median filter
        frame = cv2.medianBlur(frame,5)

        # Convert the current frame from BGR to HSV
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range for black
        lower_black = np.array([70, 0, 0])
        upper_black = np.array([180, 255, 70])

        # Define the range for blue
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Apply a threshold to the HSV image
        mask_black = cv2.inRange(frame_HSV,lower_black,upper_black) 
        
        mask_blue = cv2.inRange(frame_HSV,lower_blue,upper_blue) 

        # Filter out the grassy region from current frame, but keep the moving object 
        mask_combined = cv2.bitwise_or(mask_black, mask_blue)
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=mask_combined)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Assume the largest contour is the person
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Draw a red rectangle around the person
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Visualise both the input video and the object detection windows
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break


def close_windows(cap:cv2.VideoCapture)->None:
    
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()


def run_pipeline(args:argparse)->None:

    # Initialise video capture
    cap = initialise_camera(args)

    # Process video
    segment_object(cap, args)

    # Close all open windows
    close_windows(cap)



if __name__=='__main__':

    # Get data from CLI
    args = parse_cli_data()

    # Run pipeline
    run_pipeline(args)