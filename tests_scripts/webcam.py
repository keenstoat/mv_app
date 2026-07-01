import cv2
import time
import subprocess

"""
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













