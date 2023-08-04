from threading import Thread, Event
from time import sleep
import numpy as np
from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil
import video_maker
import depth_control

from april_tag import *
from pid_from_frame import *

print("Main started!")

# Line Detection Parameter
angle_tol = 5
forward_tol = 100
x_intercept_tolerance=25
lanes_x_tolerance=300
lanes_y_tolerance=100
lanes_darkness_threshold=10

# Create the video object
video = Video()
FPS = 5
RC_SLEEP = 0.0

FOUND_APRIL_TAG = False

# Create the PID object
PIDHorizontal = PID(35, 0.05, -5, 100) # this will recieve values between -1 and 1.
PIDLongitudinal = PID(50, 0, 0, 100) # this will recieve values between -1 and 1.
PIDYaw = PID(20, 0, -5, 100) # this will recieve values between +- pi/2 radians
PIDVertical = PID(40, 0.00, -10, 100) # this will recieve error in meters
# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
master_id = mav_comn.mode_mapping()["MANUAL"] # The ID for MANUAL mode control. Some of the robots are weird.
# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)
# where to write frames to. If empty string, no photos are written!
output_path = "frames/frame"  # {n}.jpg

window_frame_count = 11  # the number of center lines to store for calculating median
max_misses = 6  # the number of frames without receiving a center line before the robot starts to spin

frame = None
frame_available = Event()
frame_available.set()

longitudinal_power = 0
lateral_power = 0
yaw_power = 0
vertical_power = 0
crop_x = slice(20, -1)
crop_y = slice(20, -1)

def _get_frame():
    global frame
    global yaw_power
    global lateral_power
    global longitudinal_power
    global vertical_power
    global FOUND_APRIL_TAG
    count = 0
    center_lines = []

    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)

    try:
        while True:
            if video.frame_available():
                frame = video.frame()
                # print("frame found")
                # April Tag Processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                tags = get_tags(gray)
                if len(tags) > 0:
                    FOUND_APRIL_TAG = True
                else:
                    FOUND_APRIL_TAG = False

                cv2.imwrite("camera_stream.jpg", frame)
                if not FOUND_APRIL_TAG:
                    try:
                        center_line = line_from_frame(
                            frame[crop_x, crop_y],
                            x_intercept_tolerance=x_intercept_tolerance,
                            lanes_x_tolerance=lanes_x_tolerance,
                            lanes_y_tolerance=lanes_y_tolerance,
                            lanes_darkness_threshold=lanes_darkness_threshold,
                        )
                        # cv2.imwrite("testing/center.jpg", draw_lines(frame, center_lines, offset=True))
                        center_lines.append(center_line)
                        if len(center_lines) > window_frame_count:
                            center_lines.pop(0)

                        if center_lines.count(None) > 5:
                            print("No lines found in last 5 images. ")
                            yaw_power = 10  # %
                            lateral_power = 0 # %
                            longitudinal_power = 0 # %
                        good_lines = list(
                            filter(lambda line: line is not None, center_lines)
                        )
                        if len(good_lines) > 0:
                            good_lines.sort(key=lambda x: x.slope)
                            middle_line = good_lines[len(good_lines) // 2]
                            (
                                longitudinal_power,
                                lateral_power,
                                yaw_power,
                            ) = pid_from_line(
                                middle_line,
                                PIDHorizontal,
                                PIDLongitudinal,
                                PIDYaw,
                                frame.shape[1],
                                angle_tol,
                                forward_tol
                            )
                            print("Found a line!")

                    except Exception as e:
                        print(f"caught: {e}")
                        yaw_power = 0
                        lateral_power = 0
                        longitudinal_power = 0
                        
                else:
                    try:
                        print("Found a tag!")
                        vertical_power, lateral_power, longitudinal_power, yaw_power = pid_from_frame(
                                gray, PIDVertical=PIDVertical, PIDHorizontal=PIDHorizontal, PIDLongitudinal=PIDLongitudinal, PIDYaw=PIDYaw
                        )
                    except Exception as e:
                        print(f"caught while looking for tag: {e}")
                        yaw_power = 0
                        lateral_power = 0
                        longitudinal_power = 0
                        vertical_power = 0

                print(f"{yaw_power = }")
                print(f"{longitudinal_power = }")
                print(f"{lateral_power = }")
                print(f"{vertical_power = }")
                count += 1

                sleep(1 / FPS)
    except KeyboardInterrupt:
        return


def _send_rc():
    # on first startup, set everything to neutral
    # bluerov.set_rc_channels_to_neutral()
    bluerov.set_lateral_power(0)
    bluerov.set_vertical_power(0)
    bluerov.set_yaw_rate_power(0)
    bluerov.set_longitudinal_power(0)

    while True:
        bluerov.arm()

        mav_comn.set_mode(master_id)
        bluerov.set_longitudinal_power(int(longitudinal_power))
        bluerov.set_lateral_power(int(lateral_power))
        bluerov.set_yaw_rate_power(int(yaw_power))
        bluerov.set_vertical_power(int(vertical_power))
        sleep(RC_SLEEP)

def _depth_control():
    global vertical_power
    while True:
        if not FOUND_APRIL_TAG:
            depth_error = depth_control.get_depth_error(mav_comn, desired_depth=0.5)
            vertical_power = PIDVertical.update(depth_error) 

def main():
    # Start the video thread
    video_thread = Thread(target=_get_frame)
    video_thread.start()

    # # Start the RC thread
    # rc_thread = Thread(target=_send_rc)
    # rc_thread.start()

    # Start the Depth thread
    depth_thread = Thread(target=_depth_control)
    depth_thread.start()

    # Main loop
    try:
        while True:
            mav_comn.wait_heartbeat()
    except KeyboardInterrupt:
        video_thread.join()
        # rc_thread.join()
        depth_thread.start()
        bluerov.set_lights(False)
        bluerov.disarm()
        print("Exiting...")


if __name__ == "__main__":
    main()
