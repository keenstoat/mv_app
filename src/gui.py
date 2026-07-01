from __future__ import annotations  # must be the first line of code
from typing import Callable
import asyncio
import uuid
from aiortc import MediaStreamTrack
from av import VideoFrame
import fractions

from fastapi import Request
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from aiortc import RTCPeerConnection, RTCSessionDescription

from nicegui import Client, ui, app
import logging as log

from ui_ux import UiUx

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lib.frame import Frame

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
        self.update_stream_playback(frame.fps)

        # Images are assumed to be in RGB pixel format - because this is what the inference expects
        frame = VideoFrame.from_ndarray(frame.annotated_image, format="rgb24")
        frame.pts = self._pts
        frame.time_base = self._time_base 
        self._pts += self._base_pts
        
        return frame


_client_data:dict[str, UiUx] = dict()

@app.get('/displayframe/{client_id}')
async def response_display_frame(client_id:str):
    """
    Updating the ui.interactive_image used to show the display image was done by updating the source attribute with the image as a base64 string, which can be intensive when the image is large, and may result in the app crashing and auto-refreshing.
    To avoid this, the image update is done by updating its source attribute. This will make the frontend component request the /displayframe/client_id endpoint which is served by this function. This is the least resource-intensive way to update the image because it is also asynchronous and does not block the main GUI thread.

    """

    uiux:UiUx = _client_data[client_id]
    jpeg_bytes = uiux.get_display_frame().annotated_image_to_bytes()
    return Response(content=jpeg_bytes, media_type='image/jpeg')

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

__active_sessions = {}
@app.post("/webrtc-stream/{client_id}")
async def webrtc_stream(client_id:str, request:Request):

    params = await request.json()

    if params['action'] == "close":
        
        session_id = params.get("sessionId")
        if session_id in __active_sessions:
            peer_conn = __active_sessions.pop(session_id)
            await peer_conn.close()
            log.info(f"webRTC streaming closed for {session_id}")
            return JSONResponse({"status": "closed"})

        return JSONResponse({"status": "not_found"}, status_code=404)

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    peer_conn = RTCPeerConnection()
    session_id = str(uuid.uuid4())
    __active_sessions[session_id] = peer_conn

    def get_frame():
        if client_id in _client_data:
            return _client_data[client_id].get_display_frame()
        return UiUx.BLANK_FRAME
    
    track = VideoStreamTrack(get_frame) 
    peer_conn.addTrack(track)

    # Handle the SDP Negotiation handshake
    await peer_conn.setRemoteDescription(offer)
    answer = await peer_conn.createAnswer()
    await peer_conn.setLocalDescription(answer)

    # SERVER-SIDE ICE GATHERING: Wait until complete before replying
    while peer_conn.iceGatheringState != "complete":
        await asyncio.sleep(0.02)

    log.info(f"webRTC streaming started for {session_id}")
    return JSONResponse({
        "sdp": peer_conn.localDescription.sdp,
        "type": peer_conn.localDescription.type,
        "sessionId": session_id
    })


@ui.page("/")
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
    with ui.column().classes('w-full h-screen gap-0 overflow-hidden'):
        
        # HEADER ROW =============================================================================================
        with ui.row().classes('w-full h-10 bg-[#222] items-center px-4 shrink-0'):
            ui.label('MV App').classes('text-md font-bold tracking-wide')
            ui.space()
            with ui.button(icon='menu').props('flat color=white').classes("text-sm"):
                with ui.menu():
                    ui.menu_item('menu option 1')
                    ui.menu_item('menu option 2')
                    ui.separator()
                    ui.menu_item('About')
            
        # MAIN WORKSPACE ROW ====================================================================================
        with ui.row().classes('w-full flex-1 no-wrap gap-0 min-h-0'):
            
            # left panel --------------------------------------------------------------------------------
            with ui.column().classes('w-1/5 h-full no-wrap shrink-0 p-2 gap-2'):
                with ui.tabs().props('inline-label align="justify"').classes('w-full') as tabs:
                    tab_explorer = ui.tab('Explorer')
                    tab_models = ui.tab('Models')
                    tab_config = ui.tab('Config')
                
                with ui.tab_panels(tabs, value=tab_explorer).classes("w-full flex-1"):
                
                    with ui.tab_panel(tab_explorer).classes('w-full p-2 gap-2'):
                        uiux.create_ui_video_source_tree()

                    with ui.tab_panel(tab_models).classes('w-full p-2 gap-2'):
                        uiux.create_ui_vision_models_tree()
                
                    with ui.tab_panel(tab_config).classes('w-full p-2 gap-2'):
                        uiux.create_ui_config_panel()
                
            # right panel -------------------------------------------------------------------------------
            with ui.column().classes('h-full flex-1 p-2 gap-2 no-wrap min-h-0'):
                
                # play buttons -------------------------------------------------------------------------
                with ui.row().classes('w-full gap-2 shrink-0'):
                    uiux.create_ui_run_pipeline_buttons()

                with ui.row().classes('w-full gap-2 no-wrap min-h-0'):
                    
                    # left and right images -----------------------------------------------------
                    with ui.column().classes('w-1/2 h-full p-0 gap-1 overflow-hidden'):
                        uiux.create_ui_input_frame_source_url(0)
                    
                    with ui.column().classes('w-1/2 h-full p-0 gap-1 overflow-hidden'):
                        uiux.create_ui_input_frame_source_url(1)
                
                with ui.row().classes('w-full gap-2 no-wrap min-h-0'):
                    uiux.create_playback_controls()
                    
                with ui.row().classes('w-full flex-1 gap-2 no-wrap min-h-0') as display_row:
                    uiux.create_ui_image_viewport(display_row)

                with ui.row().classes('w-full gap-2 no-wrap min-h-0'):
                    uiux.create_ui_log_process_stats()
                            
        # FOOTER ROW ================================================================================================
        with ui.row().classes('w-full h-5 bg-[#222] text-xs items-center px-4 gap-2 shrink-0'):
            uiux.create_ui_status_bar()

