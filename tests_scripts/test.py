
import asyncio
import json
import struct

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pathlib import Path
from fastapi.responses import FileResponse



app = FastAPI()

@app.get("/")
async def index():
    HTML_FILE = Path(__file__).resolve().parent / "index.html"
    return FileResponse(HTML_FILE, media_type="text/html")
    

def bundle_frame(image, metadata, quality=85):
    """Bundle image and metadata into a binary packet."""

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed")

    metadata_bytes = json.dumps(metadata).encode("utf-8")

    return (
        struct.pack("!I", len(metadata_bytes))
        + metadata_bytes
        + encoded.tobytes()
    )

@app.websocket("/ws/frames")
async def stream_frames(websocket: WebSocket):
    await websocket.accept()

    video_cap = cv2.VideoCapture(0)
    video_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    video_cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_id = 0

    try:
        while True:
            success, image = await asyncio.to_thread(video_cap.read)

            if not success:
                break

            height, width = image.shape[:2]

            # Replace with actual detection results.
            metadata = {
                "frame_id": frame_id,
                "width": width,
                "height": height,
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.94,
                        "bbox": [100, 80, 180, 250],
                        "depth": frame_id,
                    }
                ],
            }

            packet = bundle_frame(image, metadata)

            await websocket.send_bytes(packet)

            frame_id += 1

    except WebSocketDisconnect:
        pass
    finally:
        video_cap.release()
        try:
            await websocket.close()
        except RuntimeError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,
    )
    