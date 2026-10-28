

def _get_pipeline_str(self, appsink_name):

    w = self._frame_width
    h = self._frame_height
    fps = self._fps
    devices = self._device_paths
    mem = "(memory:NVMM)" if self._image_as_cuda_tensor else ""

    if len(devices) == 1:
        #"nvjpegdec" should be faster than "nvv4l2decoder mjpeg=true"  but cant make the former work
        return (
            f"v4l2src device={devices[0]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            "nvv4l2decoder mjpeg=true ! "
            f"nvvideoconvert compute-hw=GPU ! video/x-raw{mem}, format=RGB ! "
            f"appsink name={appsink_name} emit-signals=true drop=true max-buffers=1 sync=true "
        )
        
    if len(devices) == 2:
        sync =  "sync-inputs=true align-inputs=true" if self._gst_sync_inputs else  ""
        return (
            f"nvstreammux name=mux width={w} height={h} batch-size=2 batched-push-timeout={int(1e6 / (fps * 2))} live-source=true max-latency={int(1e6 / (fps * 2.5))} {sync} ! "
            f"nvmultistreamtiler rows=1 columns=2 width={w * 2} height={h} ! "
            f"nvvideoconvert compute-hw=GPU ! video/x-raw{mem}, format=RGB ! "
            f"appsink name={appsink_name} emit-signals=true drop=true max-buffers=1 sync=false "
            
            f"v4l2src device={devices[0]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            "nvv4l2decoder mjpeg=true ! queue max-size-buffers=1 leaky=downstream ! mux.sink_0 "
            
            f"v4l2src device={devices[1]} do-timestamp=true io-mode=dmabuf ! image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
            "nvv4l2decoder mjpeg=true ! queue max-size-buffers=1 leaky=downstream ! mux.sink_1"
        )


def on_new_sample(sink):

    sample = sink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR
        
    buff = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    
    total_width = structure.get_value("width")
    height = structure.get_value("height")

    success, map_info = buff.map(Gst.MapFlags.READ)
    if not success:
        return Gst.FlowReturn.OK

    try:
        image = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, total_width, 3))
        try:
            self._buffer.put(image, timeout=buffer_put_timeout)
        except Full:
            try: self._buffer.get_nowait()
            except: pass

            try: self._buffer.put_nowait(image)
            except: pass
        except:
            pass
        
    finally:
        buff.unmap(map_info)
    
    return Gst.FlowReturn.OK

buffer_put_timeout = 1 / self._fps
sink_name = "imagesink"
pipeline_string = self._get_pipeline_str(sink_name)
log.debug(f"gst pipeline: {pipeline_string}")
pipeline = Gst.parse_launch(pipeline_string)
appsink = pipeline.get_by_name(sink_name)
appsink.connect("new-sample", on_new_sample)
pipeline.set_state(Gst.State.PLAYING)

self._gst_loop = GLib.MainLoop()
try:
    log.info(f"Gst connected to video devices: {self._device_paths}")
    self._gst_loop.run()

except Exception as ex:
    log.error(f"Gst loop error: {ex}")

finally:
    pipeline.set_state(Gst.State.NULL)
    self._buffer = None
    log.info(f"Gst disconnected from video devices: {self._device_paths}")