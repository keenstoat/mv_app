import cv2
import time
import subprocess

# Formats
"""
[0]: 'MJPG' (Motion-JPEG, compressed)
        Size: Discrete 3840x2160
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 2560x1440
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 1920x1080
                Interval: Discrete 0.017s (60.000 fps)
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 1280x960
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 1280x720
                Interval: Discrete 0.017s (60.000 fps)
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 1024x576
                Interval: Discrete 0.017s (60.000 fps)
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 960x720
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 800x600
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 640x480
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 640x360
                Interval: Discrete 0.017s (60.000 fps)
                Interval: Discrete 0.033s (30.000 fps)
[1]: 'YUYV' (YUYV 4:2:2)
        Size: Discrete 640x480
                Interval: Discrete 0.033s (30.000 fps)
        Size: Discrete 640x360
                Interval: Discrete 0.033s (30.000 fps)
"""

# Controls

"""
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                       contrast 0x00980901 (int)    : min=0 max=100 step=1 default=57 value=57
                     saturation 0x00980902 (int)    : min=0 max=128 step=1 default=80 value=80
                            hue 0x00980903 (int)    : min=-40 max=40 step=1 default=0 value=0
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                          gamma 0x00980910 (int)    : min=72 max=255 step=1 default=214 value=214
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=1 value=2 (60 Hz)
      white_balance_temperature 0x0098091a (int)    : min=2300 max=6500 step=1 default=5000 value=5000 flags=inactive
                      sharpness 0x0098091b (int)    : min=1 max=64 step=1 default=32 value=32
         backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=0 value=0

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=300 value=300 flags=inactive
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=192 value=192 flags=inactive
     focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
                  zoom_absolute 0x009a090d (int)    : min=0 max=100 step=1 default=0 value=0
                zoom_continuous 0x009a090f (int)    : min=0 max=0 step=0 default=0 value=0
"""

def set_hardware_ctrl(ctrl_name, value, device="/dev/video0"):
    
    command = [
        "v4l2-ctl", 
        f"--device={device}", 
        f"--set-ctrl={ctrl_name}={value}"
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Driver Error setting {ctrl_name}: {e.stderr.strip()}")

try:

    # print("Disabling focus_automatic_continuous to unlock manual zoom...")
    # set_hardware_ctrl("focus_automatic_continuous", 1)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    sizes = {
        0: (1920, 1080, 60),
        1: (1280, 720, 60),
        2: (640, 480, 30),
        3: (640, 360, 60)

    }
    size_idx = 3

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, sizes[size_idx][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sizes[size_idx][1])
    cap.set(cv2.CAP_PROP_FPS, sizes[size_idx][2])
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)

    zoom  = 0
    cap.set(cv2.CAP_PROP_ZOOM, zoom)
    w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)

    time_0 = time.time()
    loop_fps = 0
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab a frame.")
            break

        cv2.imshow('webcam', frame)
        cv2.setWindowTitle("webcam", f"webcam z: {zoom}, {int(w)}x{int(h)}, fps: {fps} - {loop_fps:.2f}")

        key = cv2.waitKey(1) & 0xFF
        if  key == ord('q'):
            break
            
        if key == ord('x'):
            zoom = min(zoom + 5, 100)
            cap.set(cv2.CAP_PROP_ZOOM, zoom)
            print("zoom: ", zoom)
        
        elif key == ord('z'):
            zoom = max(zoom - 5, 0)
            cap.set(cv2.CAP_PROP_ZOOM, zoom)
            print("zoom: ", zoom)

        elif key == ord('s'):
            cv2.imwrite("shot.jpg", frame)

        loop_fps = 1 / (time.time() - time_0)
        time_0 = time.time()
finally:
        
        cap.release()
        cv2.destroyAllWindows()













