""" HW6 - Object segmentation using the HSV image colour space
    This script is a vision-based object detection algorithm able to detect a person moving around in the football field.
    
    Authors: Juan Carlos Chávez Villarreal & Jorge Rodrigo Gomez Mayo
    Contact: juan.chavezv@udem.edu & jorger.gomez@udem.edu
    Organization: Universidad de Monterrey

    ** This script was made based on stds-sample-code-for-object-detection.py from Andres Hernandez Gutierrez **
    USAGE: 
    py .\team-3-test-object-detection.py -vf .\football-field-cropped-video.mp4 -frp 25
"""
#Import developed libraries
import modules.od as od

def run_pipeline()->None:
    # Get data from CLI
    args = od.parse_cli_data()

    # Initialise video capture
    cap = od.initialise_camera(args)

    # Process video
    od.segment_object(cap, args)

    # Close all open windows
    od.close_windows(cap)

def main():
    run_pipeline()

if __name__ == "__main__":
    main()
