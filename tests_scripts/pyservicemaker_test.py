
import torch
import cv2
from pyservicemaker import Pipeline, BufferRetriever, Receiver
from queue import Queue, Full, Empty
import threading
import time

import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request

FRAME_BUFFER = Queue(maxsize=1)


class GstVideoSrc:

    def __init__(self, device_paths) -> None:
        self.pipe:Pipeline = None

        self._device_paths = device_paths
        self._fps = 30
        self._frame_width = 640
        self._frame_height = 480

        self._gst_sync_inputs = False

        self._buffer = FRAME_BUFFER

    def get_pipeline(self, appsink_name:str):

        pipeline = Pipeline("stereo-webcams")

        pipeline.add("nvstreammux", "mux", { # sink and src only produce video/x-raw(memory:NVMM)
            "width": self._frame_width,
            "height": self._frame_height,
            "batch-size": len(self._device_paths),
            "batched-push-timeout": int(1e6 / (self._fps)), # time in microsecs to wait for both buffers
            "live-source": True,
            "max-latency": int(1e9 / (self._fps)), # in live mode, extra delay in nanosecs to allow buffers to create images
            # "align-inputs": True, # added
            # "sync-inputs": True, # added
            "sort-batch": True, # required bc we used sink_%u and pads are auto-assigned
        }) 
    
        pipeline.add("nvvideoconvert", "convert", {
            "copy-hw": 1, # (1): GPU (def), (2): VIC 
            "compute-hw": 1,  # (1): GPU, (2): VIC 
            "nvbuf-memory-type": 4 # (4): nvbuf-mem-surface-array, (2): nvbuf-mem-cuda-device
        })

        pipeline.add("capsfilter", "output_caps", {
            "caps": "video/x-raw(memory:NVMM),format=RGB",
        })
    
        pipeline.add("appsink", appsink_name, {
            "emit-signals": True,
            "drop": True,
            "max-buffers": 1,
            "sync": False, #Set sync=True for real-time extraction
        })
        pipeline.link("mux", "convert", "output_caps", appsink_name)
        
        
        for idx, device in enumerate(self._device_paths):

            cam_source_name = f"camera_{idx}"
            pipeline.add("v4l2src", cam_source_name, {
                "device": device,
                "do-timestamp": True,
                "io-mode": 4,  # io-mode=4=dmabuf, 2=mmap would work too
            })

            cam_caps_name = f"input_caps_{idx}"
            pipeline.add("capsfilter", cam_caps_name, {
                "caps": f"image/jpeg,width={self._frame_width},height={self._frame_height},framerate={self._fps}/1",
            })
            pipeline.link(cam_source_name, cam_caps_name)

            decoder_name = f"decoder_{idx}"
            pipeline.add("nvv4l2decoder", decoder_name, { # only src cap is video/x-raw(memory:NVMM)
                "mjpeg": True,
            })
            pipeline.link(cam_caps_name, decoder_name)

            queue_name = f"queue_{idx}"
            pipeline.add("queue", queue_name, {
                "max-size-buffers": 1,
                "leaky": 2,  # leaky=downstream, discards old frames when full
            })
            pipeline.link(decoder_name, queue_name)
    
            # the sink_%u notation is required. it does work with sink_0, but it is not recommended
            # https://github.com/NVIDIA/skills/blob/main/skills/deepstream-dev/references/service_maker_api.md
            # Link the queue src pad to a mux sink pad
            pipeline.link((queue_name, "mux"), ("", "sink_%u"))
    
        return pipeline
    
    def grabber(self):
        
        class FrameRetriever(BufferRetriever):
            
            def __init__(self, expected_frames:int, buffer_put_timeout:float, buffer:Queue):
                super().__init__()
        
                self.expected_frames = expected_frames
                self.buffer_put_timeout = buffer_put_timeout
                self.buffer = buffer
                self.stop_looping = False
        
            def consume(self, buffer_retriever):

                start = time.perf_counter()
                print("ENTER -----------------------------------")
                    
                if buffer_retriever.batch_size != self.expected_frames:
                    print(f"Expected {self.expected_frames} frames, got {buffer_retriever.batch_size}")
                
                tensor_list:list = [None] * self.expected_frames
                for frame_meta in buffer_retriever.batch_meta.frame_items:

                    print("Frame: ", frame_meta.frame_number, "Slice: ", frame_meta.source_id)
                    
                    tensor = buffer_retriever.extract(frame_meta.source_id).clone()

                    image_tensor:torch.Tensor = torch.utils.dlpack.from_dlpack(tensor) # pyright: ignore
                    tensor_list[frame_meta.source_id] = image_tensor
                    
                if tensor_list[0] is None:
                    print("Tensor list 0 is missing")
                    
                if tensor_list[1] is None:
                    print("Tensor list 1 is missing")
                    
                try:
                    if all(t is not None for t in tensor_list):
                        self.buffer.put_nowait(tensor_list)
                except Full:
                    try: self.buffer.get_nowait()
                    except: pass

                    try: self.buffer.put_nowait(tensor_list)
                    except: pass
                except: pass
                    
                end = time.perf_counter()
                print(f"EXIT time={(end - start) * 1000:.2f} ms ==========================\n")
                return 1

        
        appsink_name = "imagesink"
        self.pipe = self.get_pipeline(appsink_name)

        buffer_put_timeout = 1 / self._fps
        retriever = FrameRetriever(len(self._device_paths), buffer_put_timeout, self._buffer)
        self.pipe.attach(
            appsink_name,
            Receiver("receiver", retriever),
            tips="new-sample"
        )
        
        try:
            self.pipe.start()
            self.pipe.wait()
        except KeyboardInterrupt:
            pass
        except Exception as ex:
            print("ERROR: ", ex)
        finally:
            retriever.stop_looping = True
            self.pipe.stop()

# ==========================================================================================

app = FastAPI()
@app.get('/mjpeg-stream')
async def mjpeg_stream(request:Request):

    async def generate_mjpeg_frames(request:Request):
        
        while not await request.is_disconnected():
            if FRAME_BUFFER is None:
                break
                
            try:
                # tensor_list = await asyncio.to_thread(
                #     FRAME_BUFFER.get,
                #     True,
                #     0.1
                # )
                tensor_list = FRAME_BUFFER.get(timeout=1 / 30)
            except Empty:
                continue
            
            tensor_image = torch.hstack(tensor_list).cpu().numpy()
            tensor_image = tensor_image[:, :, [2, 1, 0]]
            jpeg_bytes = cv2.imencode('.jpg', tensor_image)[1].tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " +
                str(len(jpeg_bytes)).encode() +
                b"\r\n\r\n" +
                jpeg_bytes +
                b"\r\n"
            )
            
            # await asyncio.sleep(1 / 30)
    
    return StreamingResponse(
        generate_mjpeg_frames(request),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


# ==========================================================================================


if __name__ == "__main__":

    
    _gst = GstVideoSrc(["/dev/video0", "/dev/video2"])
    _grab_thread = threading.Thread(target=_gst.grabber)
    _grab_thread.daemon = True
    _grab_thread.start()

    try:
        uvicorn.run(
            app, 
            host="0.0.0.0",
            port=8000, 
            reload=False
        )

    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        
    finally:
        if _gst.pipe:
            _gst.pipe.stop()


