from linuxpy.video.device import Device, PixelFormat
from fractions import Fraction
from pathlib import Path


def get_connected_uvc_cams():
    
    v4l_path = Path("/sys/class/video4linux")

    video_devices = {}
    for video_dir in v4l_path.iterdir():
        if not video_dir.is_dir():
            continue

        with open(video_dir / "index") as video_index_file:
            if int(video_index_file.read().strip()) != 0:
                continue
        
        with open(video_dir / "name") as video_name_file:
            device_path = f"/dev/{video_dir.name}"
            video_devices[device_path] = video_name_file.read().strip()

    return video_devices

def get_frame_types(uvc_path):
    
    with Device(uvc_path) as cam:

        # FrameType(type=<Frmivaltypes.DISCRETE: 1>, pixel_format=<PixelFormat.MJPEG: 1196444237>, width=3840, height=2160, min_fps=Fraction(30, 1), max_fps=Fraction(30, 1), step_fps=Fraction(30, 1)),
        frame_types = []
        for frame_type in cam.info.frame_types:

            if frame_type.pixel_format != PixelFormat.MJPEG:
                continue
            frame_types.append({
                "width": frame_type.width,
                "height": frame_type.height,
                "fps": int(frame_type.max_fps)
            })
        frame_types.sort(key=lambda x: x["height"])
        return frame_types

def get_focus(uvc_path):
    with Device(uvc_path) as cam:
        return {
            "min": cam.controls.focus_absolute.minimum,
            "max": cam.controls.focus_absolute.maximum,
            "val": cam.controls.focus_absolute.value
        }

def set_focus(uvc_path, value):

    with Device(uvc_path) as cam:
        cam.controls.focus_automatic_continuous.value = False
        focus_min = cam.controls.focus_absolute.minimum
        focus_max = cam.controls.focus_absolute.maximum
        cam.controls.focus_absolute.value = max(focus_min, min(value, focus_max))

def get_zoom(uvc_path):
    with Device(uvc_path) as cam:
        return {
            "min": cam.controls.zoom_absolute.minimum,
            "max": cam.controls.zoom_absolute.maximum,
            "val": cam.controls.zoom_absolute.value
        }

def set_zoom(uvc_path, value):

    with Device(uvc_path) as cam:
        cam.controls.focus_automatic_continuous.value = False
        zoom_min = cam.controls.zoom_absolute.minimum
        zoom_max = cam.controls.zoom_absolute.maximum
        cam.controls.zoom_absolute.value = max(zoom_min, min(value, zoom_max))

def reset(uvc_path):
    with Device(uvc_path) as cam:
        cam.controls.focus_automatic_continuous.value = True
        cam.controls.zoom_absolute.value = cam.controls.zoom_absolute.minimum

if __name__ == "__main__":

    import sys

    args = sys.argv.copy()
    args.pop(0)

    uvc_path = args.pop(0).strip()
    control = args.pop(0).strip()
    if control == 'zoom':
        value = int(args.pop(0).strip())
        set_zoom(uvc_path, value)

    elif control == 'focus':
        value = int(args.pop(0).strip())
        set_focus(uvc_path, value)

    elif control == 'reset':
        reset(uvc_path)

    elif control == 'format':
        print(get_frame_types(uvc_path))