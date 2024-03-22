""" HW6 - Object Detection Module
    This scripts is a Python module that implements the following functions:
    parse_cli_data: Parses command-line arguments for video file and frame resize percentage.
    initialise_camera: Initializes video capture from a camera or a video file.
    rescale_frame: Rescales a video frame based on a specified percentage.
    segment_object: Processes video frames to detect and segment objects.
    close_windows: Closes all OpenCV visualization windows and releases video capture resources.

    Authors: Juan Carlos Chávez Villarreal & Jorge Rodrigo Gomez Mayo
    Contact: juan.chavezv@udem.edu & jorger.gomez@udem.edu
    Organization: Universidad de Monterrey
"""
import argparse
import cv2
import numpy as np
from argparse import Namespace
from cv2 import VideoCapture
from numpy.typing import NDArray
from typing import Tuple

window_params = {'capture_window_name':'Input video',
                'detection_window_name':'Detected object'}

def parse_cli_data() -> Namespace:
    """
    Parses command line interface data for object detection settings.

    Sets up and processes command line arguments for specifying a video file
    and frame resize percentage. The video file can be a path to a file or 'camera'
    to use the default camera. The frame resize percentage scales the video frames
    for processing.

    Returns:
        Namespace: Parsed command line arguments with video file path or camera indicator
        and frame resize percentage.
    """
    parser = argparse.ArgumentParser(prog='HW6 - Object Detector',
                                    description='Detect a person moving around in the football field',
                                    epilog='JRGM & JCCV - 2024')
    parser.add_argument('-vf',
                        '--video_file', 
                        type=str, 
                        default='camera', 
                        help='Video file used for the object detection process')
    parser.add_argument('-frp',
                        '--frame_resize_percentage', 
                        type=int, 
                        default=20,
                        help='Rescale the video frames, e.g., 20 if scaled to 20%')
    args = parser.parse_args()

    if args.video_file == "camera":
        args.video_file = 0

    return args


def initialise_camera(args: Namespace) -> VideoCapture:
    """
    Initializes a video capture object with the specified video file or camera.

    Args:
        args (Namespace): Parsed command line arguments including the video file or camera index.

    Returns:
        VideoCapture: OpenCV video capture object initialized with the specified video source.
    """
    # Create a video capture object
    cap = cv2.VideoCapture(args.video_file)
    
    return cap

def rescale_frame(frame: NDArray, percentage: int = 20) -> NDArray:
    """
    Rescales a video frame to a specified percentage of its original size.

    Args:
        frame (ndarray): The original video frame to be rescaled.
        percentage (int, optional): The percentage of the original size to which
                                    the frame should be resized. Defaults to 20.

    Returns:
        NDArray: The rescaled video frame.
    """
    # Resize current frame
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def segment_object(cap: VideoCapture, args: Namespace) -> None:
    """
    Processes video from a capture device to segment and highlight moving objects.

    Continuously reads frames from the video capture, rescales them, applies median filtering,
    and uses color segmentation to identify and draw rectangles around detected objects.

    Args:
        cap (VideoCapture): The video capture object from which frames are read.
        args (Namespace): Parsed command line arguments including frame resize percentage.

    Returns:
        None
    """
    # Rectangle size
    FIXED_RECTANGLE_SIZE = (args.frame_resize_percentage, args.frame_resize_percentage)
    
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
        filtered_frame = cv2.medianBlur(frame,5)

        # Convert the current frame from BGR to HSV
        frame_HSV = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

        # Define the range for black
        lower_black = np.array([70, 0, 0])
        upper_black = np.array([180, 255, 80])

        # Define the range for blue
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Apply a threshold to the HSV image
        mask_black = cv2.inRange(frame_HSV,lower_black,upper_black) 
        mask_blue = cv2.inRange(frame_HSV,lower_blue,upper_blue) 

        # Combine masks to isolate the object
        mask_obj = cv2.bitwise_or(mask_black, mask_blue)

        # Create a copy of frame to make modifications 
        output_frame = frame.copy()

        # Find contours in the original object mask
        contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Assume the largest contour is the object to be negated 
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the largest contour
            center_x, center_y = x + w // 2, y + h // 2

            # Adjust the fixed-size rectangle around the center
            fixed_x = max(center_x - FIXED_RECTANGLE_SIZE[0] // 2, 0)
            fixed_y = max(center_y - FIXED_RECTANGLE_SIZE[1] // 2, 0)
            cv2.rectangle(output_frame, 
                            (fixed_x, fixed_y), 
                            (fixed_x + FIXED_RECTANGLE_SIZE[0], fixed_y + FIXED_RECTANGLE_SIZE[1]), 
                            (0, 0, 255), 2)


        # Visualise both the input video and the modified detection window
        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], output_frame)

        # The program finishes if the key 'q' is pressed
        key = cv2.waitKey(5)
        if key == ord('q') or key == 27:
            print("Program finished!")
            break


def close_windows(cap: VideoCapture) -> None:
    """
    Closes all OpenCV windows and releases the video capture object.

    Args:
        cap (VideoCapture): The video capture object to be released.

    Returns:
        None
    """
    # Destroy all visualisation windows
    cv2.destroyAllWindows()

    # Destroy 'VideoCapture' object
    cap.release()