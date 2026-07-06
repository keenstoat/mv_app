
# UVC camera capabilities

Query controls
```bash
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext --device /dev/video0
v4l2-ctl --list-ctrls --device /dev/video0
```

Set controls

```bash
v4l2-ctl --device=/dev/video0 --set-ctrl=focus_automatic_continuous=0
```

# Emulate USB WebCams

```bash

# mount the virtual webcam devices under /dev/video5 and /dev/video6
sudo modprobe v4l2loopback devices=2 video_nr=5,6 card_label="Virtual Cam 1","Virtual Cam 2" exclusive_caps=1,1

# take real webcam's video feed and split it onto the virtual cams
ffmpeg -f v4l2 -i /dev/video0 \
  -vf format=yuv420p -f v4l2 /dev/video5 \
  -vf format=yuv420p -f v4l2 /dev/video6

# unmount the virtual webcam devices
sudo rmmod v4l2loopback

# list camera devices
v4l2-ctl --list-devices

# get info for camera devices

## index=0 is video feed, index=1 is metadata
cat /sys/class/video4linux/video5/index

## get device name
cat /sys/class/video4linux/video5/name
```

# 3D SBS videos

https://www.youtube.com/playlist?list=PLFd8mVbj7VrA2D93PVURmZc3LfQfzap7Z




# Study

How Field of view changes depth estimation in stereo vision?

https://erget.wordpress.com/2014/02/01/calibrating-a-stereo-camera-with-opencv/

https://robotics.stackexchange.com/questions/22033/use-2-cameras-as-1-stereo-camera

https://nerdcave.com/tailwind-cheat-sheet