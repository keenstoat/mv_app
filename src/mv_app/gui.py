from __future__ import annotations  # must be the first line of code
from collections.abc import Callable
import asyncio
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import VideoFrame
import fractions

from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from aiortc import RTCPeerConnection, RTCSessionDescription

from nicegui import Client, ui, app
from .ui_ux import UiUx

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .lib.frame import Frame

import logging
log = logging.getLogger(__name__)

_client_data:dict[str, UiUx] = {}

class VideoStreamTrack(MediaStreamTrack):

    kind = "video"

    def __init__(self, get_frame_func:Callable[[],Frame]):
        super().__init__()

        self._get_frame_func = get_frame_func
        
        self._pts = 0
        self._webrtc_clk_freq = 90000 # 90kHz clock standard for WebRTC video = 90K ticks / s
        self._time_base = fractions.Fraction(1, self._webrtc_clk_freq) # time between ticks = s / ticks

        self._fps = 30.0
        self._period = 1.0 / self._fps
        self._base_pts = int(self._period * self._webrtc_clk_freq) # time between frames in ticks
        
    def update_stream_playback(self, fps:float):

        if fps < 1 or fps == self._fps:
            return
        self._fps = fps
        self._period = 1.0 / self._fps
        self._base_pts = int(self._period * self._webrtc_clk_freq) # time between frames in ticks

    async def recv(self):

        # Pace the stream so it doesn't run at infinite speed
        await asyncio.sleep(self._period)
                
        frame = self._get_frame_func()
        if frame is None:
            self.stop()
            raise MediaStreamError
        self.update_stream_playback(frame.fps)

        # Images are assumed to be in RGB pixel format - because this is what the inference expects
        av_frame = VideoFrame.from_ndarray(frame.annotated_image, format="rgb24")
        av_frame.pts = self._pts
        av_frame.time_base = self._time_base 
        self._pts += self._base_pts
        
        return av_frame

@app.post("/webrtc-stream/{client_id}")
async def webrtc_stream(client_id:str, request:Request):

    if client_id not in _client_data:
        log.warning(f"client id {client_id} not in client data - it got disconnected")
        return
        
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    peer_conn = RTCPeerConnection()

    @peer_conn.on("connectionstatechange")
    async def on_connectionstatechange():
        if peer_conn.connectionState in ("failed", "closed", "disconnected"):
            await peer_conn.close()
            log.info(f"webRTC streaming closed for client {client_id}")

    uiux:UiUx = _client_data[client_id]
    def get_frame():
        return uiux.get_display_frame()
    
    track = VideoStreamTrack(get_frame) 
    peer_conn.addTrack(track)

    @track.on("ended")
    async def on_track_ended():
        log.info(f"webRTC video track ended for client {client_id}")
        await peer_conn.close()

    # Handle the SDP Negotiation handshake
    await peer_conn.setRemoteDescription(offer)
    answer = await peer_conn.createAnswer()
    await peer_conn.setLocalDescription(answer)

    # SERVER-SIDE ICE GATHERING: Wait until complete before replying
    while peer_conn.iceGatheringState != "complete":
        await asyncio.sleep(0.02)

    log.info(f"webRTC streaming started for client {client_id}")
    return JSONResponse({
        "sdp": peer_conn.localDescription.sdp,
        "type": peer_conn.localDescription.type,
    })

@app.get('/mjpeg-stream/{client_id}')
async def mjpeg_stream(client_id:str, request:Request):

    async def generate_mjpeg_frames(client_id:str, request:Request):

        uiux:UiUx = _client_data[client_id]
        while True:

            if await request.is_disconnected():
                break

            frame = uiux.get_display_frame()
            jpeg_bytes = frame.annotated_image_to_bytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            
            await asyncio.sleep(1 / frame.fps)
    
    return StreamingResponse(
        generate_mjpeg_frames(client_id, request),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.websocket("/ws-stream/{client_id}")
async def websocket_stream(client_id:str, websocket:WebSocket):

    await websocket.accept()
    
    try:
        if client_id not in _client_data:
            log.warning(f"client id {client_id} not in client data - it got disconnected")
            return
            
        log.info(f"ws-stream streaming started for client {client_id}")
        uiux:UiUx = _client_data[client_id]
        while websocket.client_state == WebSocketState.CONNECTED:

            frame = uiux.get_display_frame()
            if frame is None:
                await websocket.close(code=1000, reason="No more frames")
                return

            packet, delay_secs = await asyncio.to_thread(frame.bundle_frame) #frame.bundle_frame()
            await websocket.send_bytes(packet)

            # normally we wait 1/fps period until sending the next frame, but bundling the frame
            # takes some time, so we wait the remaining time, if any
            # this aims to keep the streaming in real-time
            delay_secs = 1/30 if frame.is_blank else 1/frame.fps #- delay_secs
            if delay_secs > 0:
                await asyncio.sleep(delay_secs)

    except WebSocketDisconnect:
        pass
    finally:
        log.info(f"ws-stream streaming ended for client {client_id}")
        try:
            await websocket.close()
        except RuntimeError:
            pass

# @ui.page("/")
def page(client:Client):

    uiux = UiUx(client)
    _client_data[client.id] = uiux

    log.info(f"client connected: {client.id}")

    @app.on_disconnect
    def disconnect_client(client:Client):
        if client.id not in _client_data:
            return
        _client_data[client.id].terminate_all()
        del _client_data[client.id]
        log.info(f"client disconnected: {client.id}")
    
    client.content.classes('p-0') 

    ui.add_css(
        """
        .q-field--outlined.q-field--readonly .q-field__control:before {
            border-style: solid;
        }
        .focus-glow {
            box-shadow: 0 0 5px 3px #005F92FF !important;
        }
        @font-face {
            font-family: UbuntuMono;
            src: url('/fonts/UbuntuMono.ttf') format('truetype');
        }
        """
    )
    ui.colors(
        primary="#005F92FF",
        positive="#177F2F",
        info="#009FBF",
        warning="#b6800d",
        negative="#940000"
    )
    ui.page_title("mv app")
    
    # MASTER WINDOW CONTAINER (Fixed to exact screen size, no global scrolling)
    with ui.column().classes('w-full h-screen gap-0 flex-nowrap'):

        # MAIN WORKSPACE ROW ====================================================================================
        
        workspace = ui.row().classes('w-full flex-1 min-h-0 overflow-hidden')
        splitter = ui.splitter(limits=(10,70), value=20).classes('w-full h-full min-h-0')
        with workspace, splitter:
            
            # left panel --------------------------------------------------------------------------------
            # with ui.column().classes('w-full h-full min-h-0 overflow-auto')
            with splitter.before, ui.column().classes('w-full h-full min-h-0 no-wrap shrink-0 p-2 gap-2'):
                with ui.tabs().props('inline-label align="justify"').classes('w-full') as tabs:
                    uiux.config_tabs = tabs
                    tab_explorer = ui.tab('Explorer')
                    tab_config = ui.tab('Config')
                    tab_runtime = ui.tab('Runtime')
                
                with ui.tab_panels(tabs, value=tab_explorer).classes("w-full flex-1") as panels:
                    with ui.tab_panel(tab_explorer).classes('w-full h-full p-2 gap-2'):
                        uiux.create_ui_video_source_tree()
                        ui.separator()
                        uiux.create_ui_vision_models_tree()

                    with ui.tab_panel(tab_config).classes('w-full p-2 gap-2'):
                        uiux.create_ui_config_panel()
                        
                    with ui.tab_panel(tab_runtime).classes('w-full p-2 gap-2'):
                        uiux.create_ui_runtime_panel()
                        
                    
                
            # right panel -------------------------------------------------------------------------------
            with splitter.after, ui.column().classes('w-full h-full min-h-0 p-2 gap-2'):
                
                # play buttons -------------------------------------------------------------------------
                with ui.row().classes('w-full gap-2 shrink-0'):
                    uiux.create_ui_buttons_run_pipeline()
                    uiux.create_ui_button_take_snapshot()

                with ui.row().classes('w-full gap-2 no-wrap min-h-0'):
                    uiux.create_ui_inputs_frame_source_url()
                    
                with ui.row().classes('w-full gap-2 no-wrap min-h-0'):
                    uiux.create_playback_controls()

                with ui.row().classes('w-full flex-1 min-h-0 items-center justify-center overflow-hidden bg-black') as display_row:
                    uiux.display_container = display_row
                    uiux.create_ui_image_viewport()

                
        # FOOTER ROW ================================================================================================
        with ui.row().classes('w-full h-10 bg-[#222] text-s items-center px-4 gap-2 shrink-0'):
            uiux.create_ui_status_bar()

