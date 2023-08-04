from threading import Thread, Event
from time import sleep
import numpy as np
from pid import PID
from video import Video
from bluerov_interface import BlueROV
from pymavlink import mavutil

# TODO: import your processing functions
from april_tag import *

print("Main started!")

# Create the video object
video = Video()
FPS = 30
rc_sleep = 0
# Create the PID objects
PIDVertical = PID(50, 0, -6, 100, 20, 3) # values range from [-1, 1]
PIDHorizontal = PID(45, 0.05, 0, 100) # values range from [-1, 1]
PIDLongitudinal = PID(30, 0, -3, 100) # values range from [-1, 1]
PIDYaw = PID(10, 0.0, 0, 100) # values range from [-pi, pi]
# PIDVertical = PID(70, 0, -6, 100)
# PIDHorizontal = PID(0, 0.00, 0, 100)
# PIDLongitudinal = PID(0, 0, 0, 100)
# PIDYaw = PID(0, 0.0, 0, 0)

# Create the mavlink connection
mav_comn = mavutil.mavlink_connection("udpin:0.0.0.0:14550")
master_id = mav_comn.mode_mapping()["MANUAL"] # The ID for MANUAL mode control. Some of robots are weird.

# Create the BlueROV object
bluerov = BlueROV(mav_connection=mav_comn)

frame = None
frame_available = Event()
frame_available.set()

vertical_power = 0
lateral_power = 0
longitudinal_power = 0
yaw_power = 0


def _get_frame():
    global frame
    global vertical_power
    global lateral_power
    global longitudinal_power
    global yaw_power

    while not video.frame_available():
        print("Waiting for frame...")
        sleep(0.01)

    try:
        while True:
            if video.frame_available():
                frame = video.frame()
                #cv2.imwrite("camera_stream.jpg", frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if type(frame) == np.ndarray:
                    try:
                        vertical_power, lateral_power, longitudinal_power, yaw_power = pid_from_frame(
                            frame, PIDVertical=PIDVertical, PIDHorizontal=PIDHorizontal, PIDLongitudinal=PIDLongitudinal, PIDYaw=PIDYaw
                        )
                        # vertical_power = 0
                        # lateral_power = 0
                        # longitudinal_power = 0
                        # yaw_power = 10
                        # vertical_power *= -1
                        print(f"{longitudinal_power = }")
                        print(f"{yaw_power = }")
                        print(f"{vertical_power = }")
                        print(f"{lateral_power = }")

                    except Exception as e:
                        print(f"caught: {e}")
                        vertical_power = 0
                        lateral_power = 0
                        longitudinal_power = 0
                        yaw_power = 0

                # TODO: set vertical_power and lateral_power here
                # print(frame.shape)
                sleep(1 / FPS)
    except KeyboardInterrupt:
        return


def _send_rc():
    # on first startup, set everything 0
    bluerov.set_vertical_power(0)
    bluerov.set_lateral_power(0)
    bluerov.set_yaw_rate_power(0)
    bluerov.set_longitudinal_power(0)

    while True:
        bluerov.arm()
        mav_comn.set_mode(master_id) # make sure the robot is in the right mode
        bluerov.set_vertical_power(int(vertical_power))
        bluerov.set_lateral_power(int(lateral_power))
        bluerov.set_longitudinal_power(int(longitudinal_power))
        bluerov.set_yaw_rate_power(int(yaw_power))
        sleep(rc_sleep)


def main():
    # Start the video thread
    video_thread = Thread(target=_get_frame)
    video_thread.start()

    # # Start the RC thread
    rc_thread = Thread(target=_send_rc)
    rc_thread.start()

    # Main loop
    try:
        while True:
            mav_comn.wait_heartbeat()
    except KeyboardInterrupt:
        video_thread.join()
        rc_thread.join()
        bluerov.set_lights(False)
        bluerov.disarm()
        print("Exiting...")


if __name__ == "__main__":
    main()